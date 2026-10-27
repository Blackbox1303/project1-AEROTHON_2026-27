"""Find a corridor entrance from its banner and line the drone up with the corridor.

The banner (green: entry corridor, orange: return corridor) only says WHICH corridor and roughly
WHERE: its size and mounting are unknown. Its position comes from triangulating banner_detector's
rays; the corridor's axis comes from the walls the D435 sees. Then corridor_nav takes over.
  1. SEARCH     at the current altitude (mission: no altitude change while searching): look for the
                banner, turning in yaw_step steps for a full turn; then a second turn offset by half a
                step (covers the gaps between the first turn's headings); else FAILED. climb_retry > 0
                climbs that much before the second turn (off by default)
  2. LOCATE     face the banner and strafe sideways (strafe_dist each way as needed) while collecting
                rays, until >= min_rays spanning >= min_spread_deg; least-squares ray intersection.
                There are two banners per corridor on the day (across the top + on the ground;
                organisers, 2026-10-19) and either will do, but the rays must all be to the SAME one:
                each frame's ray is the detector candidate within track_deg of the previous ray
  3. STANDOFF   fly to standoff metres from the banner on our side, facing it, at <= align_altitude
                (return lap: comes down from 10 m here, in the open)
  4. AXIS       sweep the heading over +-sweep_deg around the banner (the banner may hang beside the
                corridor, not over it) collecting the D435 cloud (/vio/depth_points); points
                wall_band metres above ground within axis_radius of the banner, on the ground plane:
                RANSAC pairs of parallel wall lines ~corridor_width apart (both corridors may be in
                view); take the well-supported pair whose centre line passes nearest the banner ->
                centre line, direction into the corridor (away from us), entrance where the walls
                start. No wall pair: step down axis_step metres and sweep again, down to the corridor
                altitude (still outside the corridor; at 5 m the D435 sees the wall band only ~3.6-6 m
                away, at 3 m everywhere). Still none: FAILED - or, if allow_banner_only, entrance =
                banner x, y with our bearing to it as the axis (wrong for a banner beside the
                corridor: 2026-10-19 sim, side mount -> 3.7 m and 40 deg off).
  5. HAND OVER  corridor_nav entry_x/y/yaw <- the entrance; /nav/corridor/run (if run_corridor)
mode geofence (the exit fallback when no orange banner is found, set by the mission supervisor from the entrance
geofence): no SEARCH / LOCATE - the predicted mouth predict_x/y with direction predict_yaw stands in for the banner.
STANDOFF is standoff metres before it on the axis, facing along predict_yaw; AXIS keeps only wall pairs within
geofence_max_angle_deg of predict_yaw whose centre line is more than exclude_radius from the excluded corridor's
(exclude_x/y/yaw: the entry corridor, which lies parallel right beside the return one), takes the one nearest the
predicted point (<= geofence_max_lateral beside it - which side the return corridor is on needn't be known) and
orients it along predict_yaw.
Service /nav/entry/align (std_srvs/Trigger), /nav/entry/cancel; state on /nav/entry/state (String);
result on /nav/entry/pose (PoseStamped, odom) and /nav/entry/result (String JSON).
"""
import json
import math
import threading
import time

import numpy as np
import rclpy
import rclpy.time
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import SetBool, Trigger
from tf2_ros import Buffer, TransformListener

from aerothon_nav.banner_detector import lookup_recent, quat_to_rot
from aerothon_nav.direct_flight import DirectFlight, yaw_of


def pval(v):
    if isinstance(v, str):
        return ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=v)
    return ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))


def triangulate(origins, rays):
    """Least-squares point closest to all rays; returns (point, smallest eigenvalue of the normal
    matrix / n - near 0 when the rays are close to parallel)."""
    A, b = np.zeros((3, 3)), np.zeros(3)
    for o, d in zip(origins, rays):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    return np.linalg.solve(A, b), float(np.linalg.eigvalsh(A)[0]) / len(rays)


def wall_pairs(pts, width, width_tol, rng, iters=500, tol=0.08, min_inliers=25, keep=6):
    """RANSAC on ground-plane points (N, 2) for pairs of parallel lines width +- width_tol apart
    (two corridors side by side give several). Returns up to `keep` distinct pairs, best supported
    first: (support, u direction, n normal, (c1, c2) offsets along n with c1 < c2, inliers1, inliers2);
    lines are n . p = c."""
    if len(pts) < 2 * min_inliers:
        return []
    found = []
    for _ in range(iters):
        i, j = rng.choice(len(pts), 2, replace=False)
        d = pts[j] - pts[i]
        L = np.linalg.norm(d)
        if L < 0.3:
            continue
        u = d / L
        n = np.array([-u[1], u[0]])
        off = pts @ n - n @ pts[i]
        inl = np.abs(off) < tol
        if inl.sum() < min_inliers:
            continue
        for sign in (1, -1):                         # the other wall: same direction, width away
            band = np.abs(off - sign * width) < width_tol
            if band.sum() < min_inliers:
                continue
            inl2 = np.abs(off - float(np.median(off[band]))) < tol
            if inl2.sum() >= min_inliers:
                found.append((int(min(inl.sum(), inl2.sum())), inl, inl2))
    pairs = []
    claimed = np.zeros(len(pts), bool)               # points of accepted pairs can't support another
    for _, inl1, inl2 in sorted(found, key=lambda f: -f[0]):
        inl1, inl2 = inl1 & ~claimed, inl2 & ~claimed
        if min(inl1.sum(), inl2.sum()) < min_inliers:
            continue
        # least-squares common direction on both inlier sets, normal in one fixed orientation (so the
        # same pair found twice compares equal), then inliers again for the refined lines
        both = np.vstack([pts[inl1] - pts[inl1].mean(0), pts[inl2] - pts[inl2].mean(0)])
        u = np.linalg.svd(both, full_matrices=False)[2][0]
        n = np.array([-u[1], u[0]])
        if n[1] < 0 or (abs(n[1]) < 1e-9 and n[0] < 0):
            n, u = -n, -u
        c1, c2 = sorted((float(np.median(pts[inl1] @ n)), float(np.median(pts[inl2] @ n))))
        off = pts @ n
        in1, in2 = (np.abs(off - c1) < tol) & ~claimed, (np.abs(off - c2) < tol) & ~claimed
        span = min(np.ptp(pts[in1] @ u) if in1.any() else 0, np.ptp(pts[in2] @ u) if in2.any() else 0)
        support = int(min(in1.sum(), in2.sum()))
        if support < min_inliers or span < 1.0 or abs((c2 - c1) - width) > width_tol:
            continue
        centre = (c1 + c2) / 2
        if any(abs(u @ q[1]) > 0.996 and abs(centre - sum(q[3]) / 2) < 0.4 for q in pairs):
            continue                                 # same pair again
        pairs.append((support, u, n, (c1, c2), in1, in2))
        claimed |= in1 | in2
        if len(pairs) >= keep:
            break
    pairs.sort(key=lambda q: -q[0])
    return pairs


class EntryAlign(Node):
    def __init__(self):
        super().__init__('entry_align')
        p = self.declare_parameter
        p('colour', 'green')
        p('align_altitude', 2.3)     # m: max altitude for the wall fit (the supervisor passes its own)
        p('corridor_altitude', 1.5)  # m: where the step-down stops (= the tunnel flight height, v3 < 8 ft)
        p('yaw_step_deg', 45.0)
        p('look_time', 1.5)          # s per search heading
        p('climb_retry', 0.0)        # m before the second turn; 0 = never change altitude (mission rule)
        p('strafe_dist', 1.0)        # m per LOCATE step, all to one side (strafe_steps of them at most)
        p('strafe_steps', 4)
        p('min_rays', 10)
        p('min_spread_deg', 8.0)
        p('banner_z_range', [-0.5, 4.0])  # m: a triangulated banner outside this (or missing its rays by more than
        p('banner_max_miss', 0.4)    # this median) is a bad fix - search again. One at 9.7 m (the drone's own height,
        #                              0.77 m miss) put the standoff 9 m from the corridor, out of depth range
        p('banner_tries', 3)
        p('ground_banner_max_z', 1.65)  # m (v3: ground banner z 0, top banner ~3.3 m)  # m: a triangulated fix below this is the GROUND banner -> re-fix as the rays'
        #                              median ground (z 0) intersection. From 10 m the rays are steep and the free 3D
        #                              intersection slid along them: z 2.03, 1.1 m sideways, the corridor line ran
        #                              beside the entry corridor and the orange fit failed 3 x (2026-10-23 sim)
        p('axis_yaw', float('nan'))  # rad, banner mode: only wall pairs parallel to this (either direction) within
        p('axis_yaw_tol_deg', 25.0)  # this. The supervisor gives the entry corridor's yaw for the return lap: a
        #                              crosswise pair (window face + corridor ends, 1.8 m apart) fit at 98 deg
        #                              (2026-10-21 sim). NaN: no constraint
        p('track_deg', 8.0)          # max ray change between frames on one banner (strafing: ~2 deg)
        p('standoff', 3.0)           # m, horizontal, banner -> drone for the wall fit (4.0 from above; at 2.5 m height
        #                              the walls must be close enough to be in the depth cloud)
        p('close_standoff', 1.5)     # m: last try from this close on the corridor line. From 4.7 m out a full-height
        #                              window panel 2.1 m inside the return corridor left each inner wall ~3 depth
        #                              columns (sim D435 320 px) per heading: < min_inliers, no pair (2026-10-23 sim)
        p('sweep_deg', 20.0)         # +-: 3 headings, all into OUR tunnel (+-43 deg D435 view). 60 turned the drone to
        #                              look at the neighbouring (blue) tunnel and, with refine_low, 2 x ~70 s of
        #                              sweeping before the entry (user, 2026-10-24: go in as soon as the green is found)
        p('sweep_step_deg', 30.0)
        p('look_time_axis', 0.8)     # s per sweep heading
        p('axis_step', 0.7)          # m down per retry
        p('allow_banner_only', False)
        p('axis_radius', 9.0)
        p('wall_band', [0.5, 2.4])   # m above ground: the walls below the v3 roof (3.05 m)
        p('corridor_width', 3.353)   # m: course v3, 11 ft (v2: 1.675)
        p('width_tol', 0.35)
        p('max_banner_offset', 3.0)  # m: entrance found from the walls must be this close to the banner (it stands
        #                              ~1.5 m before the mouth). 6.0 passed an entrance 5.96 m away, 4.5 m inside the
        #                              return corridor, facing out of it (2026-10-21 sim)
        p('max_banner_lateral', 1.2) # m: and the corridor's centre line must pass this close beside it.
        #   Course v2: both banners are centred on their corridor; the neighbour's centre line is ~2.6 m away.
        #   A stand beside the corridor is ~3.4 m off (sim side mount); the NEIGHBOUR corridor ~4.8 m - with
        #   only its walls in view, it was taken for the banner's corridor (2026-10-19, seed 1)
        p('run_corridor', True)
        p('move_min_span', 3.0)      # m: a rejected pair's walls must be this long to trust its direction
        p('move_max_turn_deg', 60.0) # deg: and run within this of the bearing to the banner: a crosswise pair
        #                              (85 deg, the standoff faced 167) sent the standoff beside the corridors
        p('move_min_span_aligned', 1.5)  # m: shorter walls will do when they run within move_aligned_deg of the
        p('move_aligned_deg', 25.0)  # bearing: from 2.6 m (10 ft rule) the neighbour's walls looked < 3 m long and
        #                              the standoff, in line with our own north wall, never moved (2026-10-21)
        p('refine_low', False)       # fit the walls again at corridor_altitude after a fit from above (v3: align at 2.3 m
        #                              sees the 3 m walls fine; the second sweep only cost time)
        p('tf_max_age', 1.0)         # s: a depth cloud newer than the latest pose by up to this uses it
        p('mode', 'banner')          # banner | geofence (see the docstring)
        p('predict_x', 0.0)          # geofence mode: predicted corridor mouth (odom) and direction into it
        p('predict_y', 0.0)
        p('predict_yaw', 0.0)
        p('exclude_x', 0.0)          # geofence mode: a point on the corridor to ignore and its axis
        p('exclude_y', 0.0)
        p('exclude_yaw', 0.0)
        p('exclude_radius', 0.9)     # m: centre lines this close to the excluded one are that corridor
        p('banner_exclude', False)   # banner mode too (return lap): drop pairs within exclude_radius of the excluded
        #                              corridor before choosing - the gap BETWEEN the corridors fit as the exit, its
        #                              centre line 1.72 m from the entry's (2026-10-21 sim)
        p('geofence_max_lateral', 5.0)
        p('geofence_max_angle_deg', 12.0)  # 20 accepted the gap between the corridors at 18 deg (2026-10-21)
        p('geofence_max_offset', 7.0)
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731

        self.odom = None
        self.dets = []               # (wall time, detection dict)
        self.clouds = None           # list while collecting
        self.last_dir = None         # direction (into the corridors) of the last wall pair found
        self.collect_from = None     # sim time from which clouds count (None: between sweep headings)
        self.state = 'IDLE'
        self.running = self.cancel = False
        self.rng = np.random.default_rng(0)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.flight = DirectFlight(self, self.pose, lambda: self.cancel)
        self.state_pub = self.create_publisher(String, '/nav/entry/state', 10)
        self.pose_pub = self.create_publisher(PoseStamped, '/nav/entry/pose', 10)
        self.result_pub = self.create_publisher(String, '/nav/entry/result', 10)
        self.create_subscription(Odometry, '/nav/odom', lambda m: setattr(self, 'odom', m), 10)
        self.create_subscription(String, '/nav/banner/detection', self.on_det, 10)
        self.create_subscription(PointCloud2, '/vio/depth_points', self.on_cloud, qos_profile_sensor_data)
        self.offboard = self.create_client(SetBool, '/nav/offboard')
        self.detector_params = self.create_client(SetParameters, '/banner_detector/set_parameters')
        self.corridor_params = self.create_client(SetParameters, '/corridor_nav/set_parameters')
        self.corridor_run = self.create_client(Trigger, '/nav/corridor/run')
        self.create_service(Trigger, '/nav/entry/align', self.on_run)
        self.create_service(Trigger, '/nav/entry/cancel', self.on_cancel)
        self.create_timer(1.0, lambda: self.state_pub.publish(String(data=self.state)))

    # ---- inputs
    def pose(self):
        p = self.odom.pose.pose
        return p.position.x, p.position.y, p.position.z, yaw_of(p.orientation)

    def on_det(self, m):
        d = json.loads(m.data)
        if d['colour'] == self.g('colour'):
            self.dets.append((time.monotonic(), d))
            self.dets = self.dets[-400:]

    def recent(self, since):
        return [d for t, d in self.dets if t >= since]

    def track(self, dets, ref):
        """Per detection, the candidate ray within track_deg of `ref` (updated as it goes):
        [(origin, ray)]. ref None: start on the best candidate of the latest detection."""
        if not dets:
            return []
        if ref is None:
            ref = np.array(dets[-1]['ray'])
        cos_max, out = math.cos(math.radians(self.g('track_deg'))), []
        for d in dets:
            rays = [np.array(c['ray']) for c in d.get('candidates') or [d]]
            r = max(rays, key=lambda r: r @ ref)
            if r @ ref >= cos_max:
                out.append((np.array(d['origin']), r))
                ref = r
        return out

    def on_cloud(self, m):
        # only while collecting, and only clouds taken after the drone settled on the sweep heading: with the pose
        # lagging (tf_max_age fallback), a cloud taken mid-turn got a pre-turn heading and the smeared walls gave a
        # corridor axis 45 deg off (2026-10-20 sim)
        # local copies: the run thread resets both between the checks (a TypeError killed the node, 2026-10-21)
        clouds, collect_from = self.clouds, self.collect_from
        if clouds is None or collect_from is None:
            return
        if rclpy.time.Time.from_msg(m.header.stamp) < collect_from:
            return
        try:
            t = lookup_recent(self.tf, 'odom', m.header.frame_id, m.header.stamp, self.g('tf_max_age'))
        except Exception:
            return
        pts = np.frombuffer(m.data, np.float32).reshape(-1, 3).astype(np.float64)
        R = quat_to_rot(t.transform.rotation)
        o = t.transform.translation
        clouds.append(pts @ R.T + [o.x, o.y, o.z])

    # ---- helpers
    def call(self, client, req, timeout=5.0):
        if not client.wait_for_service(timeout_sec=timeout):
            raise RuntimeError(f'{client.srv_name} not available')
        fut = client.call_async(req)
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.02)
        if not fut.done():
            raise RuntimeError(f'{client.srv_name} timed out')
        return fut.result()

    def set_params(self, client, **kv):
        self.call(client, SetParameters.Request(parameters=[Parameter(name=k, value=pval(v)) for k, v in kv.items()]))

    def set_detector_colour(self, colour, tries=3):
        """banner_detector's colour; not fatal - on_det keeps only `colour` detections anyway, so a detector left on
        another colour just finds no banner (and the search fails as such). On the loaded sim the call timed out after
        5 s and failed the exit align although the colour (green both ways in v3) was already right (2026-10-24)."""
        for k in range(1, tries + 1):
            try:
                self.call(self.detector_params, SetParameters.Request(
                    parameters=[Parameter(name='colour', value=pval(colour))]), timeout=15.0)
                return
            except RuntimeError as e:
                self.get_logger().warn(f'banner_detector colour {colour}: {e} (try {k})')
        self.get_logger().warn(f'banner_detector colour not confirmed: searching for {colour} anyway')

    def set_state(self, s):
        if s.split(' ')[0] != self.state.split(' ')[0] or s.startswith(('DONE', 'FAILED')):
            self.get_logger().info(s)
        self.state = s
        self.state_pub.publish(String(data=s))

    def on_run(self, req, res):
        if self.running:
            res.success, res.message = False, f'already running ({self.state})'
            return res
        self.running, self.cancel = True, False
        # this run's state at once: the 1 Hz timer kept publishing the LAST run's 'DONE ...', which the supervisor took
        # for this run's result and aborted ('DONE but no /nav/entry/result', 2026-10-24 sim)
        self.set_state('STARTING')
        threading.Thread(target=self.run, daemon=True).start()
        res.success, res.message = True, 'entry alignment started'
        return res

    def on_cancel(self, req, res):
        self.cancel = True
        res.success, res.message = True, 'cancel requested'
        return res

    # ---- steps
    def search(self):
        _, _, z0, yaw0 = self.pose()
        step = math.radians(self.g('yaw_step_deg'))
        for turn, alt in enumerate((z0, z0 + self.g('climb_retry'))):
            for k in range(int(round(2 * math.pi / step)) + 1):
                x, y, _, _ = self.pose()
                yaw = yaw0 + k * step + turn * step / 2
                self.set_state(f'SEARCH heading {math.degrees(yaw) % 360:.0f} alt {alt:.1f}')
                self.flight.fly(x, y, alt, yaw, tol=0.5, yaw_tol=0.1, timeout=15)
                t0 = time.monotonic()
                self.flight.hold(self.g('look_time'), alt)
                seen = self.track(self.recent(t0), None)
                if len(seen) >= 2:
                    return seen
                if self.cancel:
                    raise RuntimeError('cancelled')
        raise RuntimeError(f'no {self.g("colour")} banner seen in two full turns at {z0:.1f} m')

    def locate(self, first):
        """Strafe across the line of sight, collecting rays, until they span enough angle."""
        alt = self.pose()[2]
        origins = [o for o, _ in first]
        rays = [r for _, r in first]
        seen = set()

        def collect():
            new = [d for d in self.recent(time.monotonic() - 0.5) if d['stamp'] not in seen]
            seen.update(d['stamp'] for d in new)
            for o, r in self.track(new, rays[-1]):
                if np.linalg.norm(o - origins[-1]) > 0.05:
                    origins.append(o)
                    rays.append(r)

        def face_banner(px, py):                     # aim at the latest ray's horizontal direction
            r = rays[-1]
            return math.atan2(r[1], r[0])

        # ONE side, strafe_dist steps, never back (user, 2026-10-24: the old right / left / twice-as-wide legs swept
        # the drone across both tunnels). The side: away from the excluded corridor if there is one (exit lap: the
        # entry corridor), else left - from the start area that is towards the green tunnel. 1 m at ~3.5 m from the
        # banner adds ~16 deg of spread (8 needed)
        x0, y0, _, _ = self.pose()
        az0 = face_banner(x0, y0)
        left = np.array([-math.sin(az0), math.cos(az0)])
        sign = 1.0
        if self.g('banner_exclude') or self.g('mode') == 'geofence':
            away = np.array([x0 - self.g('exclude_x'), y0 - self.g('exclude_y')])
            sign = 1.0 if away @ left >= 0 else -1.0
        legs = (sign,) * int(self.g('strafe_steps'))
        for leg in legs:
            spread = max(math.degrees(math.acos(np.clip(a @ b, -1, 1))) for a in rays for b in rays)
            if len(rays) >= self.g('min_rays') and spread >= self.g('min_spread_deg'):
                break
            self.set_state(f'LOCATE rays {len(rays)} spread {spread:.1f} deg')
            x, y, _, _ = self.pose()
            az = face_banner(x, y)
            side = np.array([-math.sin(az), math.cos(az)]) * leg * self.g('strafe_dist')
            self.flight.fly(x + side[0], y + side[1], alt, face_banner, vmax=0.8, tol=0.3, yaw_tol=0.3,
                            timeout=20, each=collect)
            self.flight.hold(0.5, alt, each=collect)
        spread = max(math.degrees(math.acos(np.clip(a @ b, -1, 1))) for a in rays for b in rays)
        if len(rays) < self.g('min_rays') or spread < self.g('min_spread_deg'):
            raise RuntimeError(f'banner rays too few / too parallel ({len(rays)}, {spread:.1f} deg)')
        p, cond = triangulate(np.array(origins), np.array(rays))
        O, D = np.array(origins), np.array(rays)
        if p[2] < self.g('ground_banner_max_z') and (D[:, 2] < -0.2).all():
            g = O[:, :2] + D[:, :2] * (-O[:, 2] / D[:, 2])[:, None]        # each ray's ground (z 0) point
            pg = np.array([*np.median(g, axis=0), 0.0])
            self.get_logger().info(f'ground banner: 3D fix ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) -> ground '
                                   f'({pg[0]:.2f}, {pg[1]:.2f}), ground spread {np.median(np.linalg.norm(g - pg[:2], axis=1)):.2f} m')
            p = pg
        resid = np.median([np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o)) for o, d in zip(origins, rays)])
        self.get_logger().info(f'banner at ({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) from {len(rays)} rays, '
                               f'spread {spread:.1f} deg, median miss {resid:.2f} m')
        return p, {'rays': len(rays), 'spread_deg': round(spread, 1), 'median_miss_m': round(float(resid), 3)}

    def wall_axis(self, banner, face):
        """Corridor centre line from the D435 cloud, collected over a heading sweep around `face`:
        ((entrance xy, yaw into the corridor, info), None) or (None, reason)."""
        x, y, z, _ = self.pose()
        sweep = math.radians(self.g('sweep_deg'))
        step = math.radians(self.g('sweep_step_deg'))
        self.clouds = []
        for a in np.arange(-sweep, sweep + 1e-6, step):
            self.flight.fly(x, y, z, face + a, tol=0.5, yaw_tol=0.1, timeout=10)
            self.flight.hold(self.g('tf_max_age'), z)          # settle: the (lagging) pose catches up
            self.collect_from = self.get_clock().now()
            self.flight.hold(self.g('look_time_axis'), z)
            self.collect_from = None
        self.flight.fly(x, y, z, face, tol=0.5, yaw_tol=0.1, timeout=10)
        clouds, self.clouds = self.clouds, None
        t0 = time.monotonic()
        if not clouds:
            return None, 'no depth clouds'
        pts = np.vstack(clouds)
        lo, hi = self.g('wall_band')
        keep = (pts[:, 2] > lo) & (pts[:, 2] < hi) & \
            (np.hypot(pts[:, 0] - banner[0], pts[:, 1] - banner[1]) < self.g('axis_radius'))
        xy = pts[keep, :2]
        if len(xy) < 50:
            return None, f'only {len(xy)} wall-band points'
        xy = np.unique(np.round(xy / 0.05) * 0.05, axis=0)            # 5 cm grid: even density
        pairs = wall_pairs(xy, self.g('corridor_width'), self.g('width_tol'), self.rng)
        self.get_logger().info(f'wall fit: {len(clouds)} clouds, {len(pts)} points, {len(xy)} in the wall band, '
                               f'{len(pairs)} pairs, {time.monotonic() - t0:.1f} s')
        if not pairs:
            return None, f'no wall pair in {len(xy)} points'
        lateral = lambda q: abs(q[2] @ banner[:2] - sum(q[3]) / 2)  # noqa: E731
        geofence = self.g('mode') == 'geofence'
        if geofence:
            # parallel to the predicted axis, not the excluded (entry) corridor; any support
            up = np.array([math.cos(self.g('predict_yaw')), math.sin(self.g('predict_yaw'))])
            ex = np.array([self.g('exclude_x'), self.g('exclude_y')])
            cos_max = math.cos(math.radians(self.g('geofence_max_angle_deg')))
            strong = [q for q in pairs if abs(q[1] @ up) >= cos_max and
                      abs(q[2] @ ex - sum(q[3]) / 2) > self.g('exclude_radius')]
            if not strong:
                return None, f'none of {len(pairs)} wall pairs is parallel and not the excluded corridor'
            max_lateral = self.g('geofence_max_lateral')
        else:
            # the corridor the banner marks: among well-supported pairs, the centre line nearest the banner
            if math.isfinite(self.g('axis_yaw')):
                ay = np.array([math.cos(self.g('axis_yaw')), math.sin(self.g('axis_yaw'))])
                cos_min = math.cos(math.radians(self.g('axis_yaw_tol_deg')))
                pairs = [q for q in pairs if abs(q[1] @ ay) >= cos_min]
                if self.g('banner_exclude') and pairs:
                    ex = np.array([self.g('exclude_x'), self.g('exclude_y')])
                    kept = [q for q in pairs if abs(q[2] @ ex - sum(q[3]) / 2) > self.g('exclude_radius')]
                    if not kept:
                        # only the excluded corridor / the gap: parallel, so their direction still puts the standoff
                        # on the line through the banner (the move in run())
                        u = max(pairs, key=lambda q: q[0])[1]
                        x, y, _, _ = self.pose()
                        self.last_dir = u if (banner[:2] - np.array([x, y])) @ u >= 0 else -u
                        return None, f'only the excluded corridor in view: wall pairs beside the banner'
                    pairs = kept
                if not pairs:
                    return None, (f'no wall pair within {self.g("axis_yaw_tol_deg"):.0f} deg of the expected '
                                  f'{math.degrees(self.g("axis_yaw")):.0f} deg')
            strong = [q for q in pairs if q[0] >= 0.4 * pairs[0][0]]
            max_lateral = self.g('max_banner_lateral')
        support, u, n, (c1, c2), in1, in2 = min(strong, key=lateral)
        x, y, _, _ = self.pose()
        # corridors are parallel: a neighbour's walls give the direction - if they are long side walls. A short
        # pair (wall ends seen across) once sent the standoff to face -107 deg, across the corridors (2026-10-21)
        span = min(np.ptp(xy[in1] @ u), np.ptp(xy[in2] @ u))
        to_banner = (banner[:2] - np.array([x, y])) / max(np.linalg.norm(banner[:2] - np.array([x, y])), 1e-6)
        along = abs(to_banner @ u)
        # with axis_yaw (return lap) every pair here already runs parallel to the entry corridor: its direction is
        # right whatever its length (1.8 m walls 28 deg off the bearing blocked the move and the fit, 2026-10-21)
        trusted = (span >= self.g('move_min_span') and along >= math.cos(math.radians(self.g('move_max_turn_deg')))) \
            or (span >= self.g('move_min_span_aligned') and along >= math.cos(math.radians(self.g('move_aligned_deg')))) \
            or (not geofence and math.isfinite(self.g('axis_yaw')) and span >= 1.0)
        self.last_dir = (u if to_banner @ u >= 0 else -u) if trusted else None
        if not trusted:
            self.get_logger().info(f'wall pair direction not trusted for a standoff move (walls {span:.1f} m, '
                                   f'{math.degrees(math.acos(min(1.0, along))):.0f} deg off the bearing)')
        if lateral((support, u, n, (c1, c2))) > max_lateral:
            return None, (f'nearest wall pair passes {lateral((support, u, n, (c1, c2))):.1f} m beside the '
                          f'{"predicted mouth" if geofence else "banner (another corridor?)"}')
        x, y, _, _ = self.pose()
        here = np.array([x, y])
        centre_c = (c1 + c2) / 2
        if geofence:
            if u @ up < 0:                                             # along the predicted direction
                u = -u
        elif (banner[:2] - here) @ u < 0:                              # into the corridor = away from us
            u = -u
        walls = np.vstack([xy[in1], xy[in2]])
        t_start = float(np.percentile(walls @ u, 2))                    # nearest wall ends along the axis
        point_on_centre = n * centre_c
        entrance = point_on_centre + u * (t_start - point_on_centre @ u)
        width = c2 - c1
        max_offset = self.g('geofence_max_offset') if geofence else self.g('max_banner_offset')
        if np.linalg.norm(entrance - banner[:2]) > max_offset:
            return None, f'wall pair {np.linalg.norm(entrance - banner[:2]):.1f} m from the banner'
        info = {'width_m': round(width, 2), 'wall_points': [int(in1.sum()), int(in2.sum())],
                'pairs_found': len(pairs)}
        return (entrance, math.atan2(u[1], u[0]), info), None

    def run(self):
        g = self.g
        try:
            if self.odom is None:
                raise RuntimeError('no /nav/odom')
            res = self.call(self.offboard, SetBool.Request(data=True))
            if not res.success:
                raise RuntimeError(f'offboard refused: {res.message}')
            x, y, z, _ = self.pose()
            if g('mode') == 'geofence':
                # the predicted mouth stands in for the banner; standoff before it on the predicted axis
                banner = np.array([g('predict_x'), g('predict_y'), 0.0])
                loc = {'mode': 'geofence'}
                face = g('predict_yaw')
                away = -np.array([math.cos(face), math.sin(face)])
                self.get_logger().info(f'geofence: predicted mouth ({banner[0]:.2f}, {banner[1]:.2f}) '
                                       f'yaw {math.degrees(face):.0f}')
            else:
                self.set_detector_colour(g('colour'))
                self.dets = []
                for attempt in range(1, g('banner_tries') + 1):
                    try:
                        banner, loc = self.locate(self.search())
                    except RuntimeError as e:
                        if 'banner rays' not in str(e) or attempt == g('banner_tries') or self.cancel:
                            raise
                        self.get_logger().warn(f'banner fix {attempt}: {e}: searching again')
                        self.dets = []
                        continue
                    zlo, zhi = g('banner_z_range')
                    if zlo <= banner[2] <= zhi and loc['median_miss_m'] <= g('banner_max_miss'):
                        break
                    self.get_logger().warn(f'banner fix {attempt} rejected (z {banner[2]:.2f} m, miss '
                                           f'{loc["median_miss_m"]:.2f} m): searching again')
                    self.dets = []
                else:
                    raise RuntimeError(f'no plausible banner fix in {g("banner_tries")} tries')
                # standoff on our side of the banner, facing it, no higher than align_altitude
                x, y, z, _ = self.pose()
                away = np.array([x, y]) - banner[:2]
                away /= max(np.linalg.norm(away), 1e-6)
                face = math.atan2(-away[1], -away[0])
            sx, sy = banner[:2] + away * g('standoff')
            alt = min(z, g('align_altitude'))
            self.set_state(f'STANDOFF ({sx:.1f}, {sy:.1f}) alt {alt:.1f}')
            self.flight.fly(sx, sy, max(z, alt), face, timeout=40)
            if not self.flight.fly(sx, sy, alt, face, timeout=40):
                raise RuntimeError('could not reach the standoff point')

            axis, why, k, moves, closer = None, None, 0, 0, False
            while not self.cancel:
                k += 1
                self.set_state(f'AXIS try {k} alt {self.pose()[2]:.1f}')
                self.last_dir = None
                axis, why = self.wall_axis(banner, face)
                if not axis:
                    self.get_logger().info(f'AXIS try {k}: {why}')
                if not axis and self.last_dir is not None and 'beside the' in why and moves < 2:
                    # only the NEIGHBOUR corridor's walls were in view (a standoff off to one side faced the banner
                    # diagonally: 2026-10-20 sim). Its direction is ours: stand off on the line through the banner
                    # along it, facing in, and look again at this height
                    moves += 1
                    u = self.last_dir
                    nx, ny = banner[:2] - u * g('standoff')
                    if math.hypot(nx - sx, ny - sy) > 0.5 or moves == 1:   # the same spot again sees the same
                        sx, sy = nx, ny
                        face = math.atan2(u[1], u[0])
                        self.get_logger().info(f'AXIS: move the standoff onto the corridor line ({sx:.1f}, {sy:.1f}),'
                                               f' facing {math.degrees(face):.0f} deg')
                        self.flight.fly(sx, sy, self.pose()[2], face, timeout=40)
                        continue
                if axis:
                    break
                if self.pose()[2] <= g('corridor_altitude') + 0.15:
                    if closer:
                        break
                    # last try: close in along the line (or the bearing to the banner) - the walls fill more of
                    # the depth image
                    closer = True
                    u = self.last_dir if self.last_dir is not None else np.array([math.cos(face), math.sin(face)])
                    sx, sy = banner[:2] - u * g('close_standoff')
                    face = math.atan2(u[1], u[0])
                    self.get_logger().info(f'AXIS: closer, {g("close_standoff"):.1f} m from the banner '
                                           f'({sx:.1f}, {sy:.1f}), facing {math.degrees(face):.0f} deg')
                    self.flight.fly(sx, sy, self.pose()[2], face, timeout=40)
                    continue
                lower = max(self.pose()[2] - g('axis_step'), g('corridor_altitude'))
                self.flight.fly(sx, sy, lower, face, timeout=20)
            if axis and g('refine_low') and self.pose()[2] > g('corridor_altitude') + 0.15 and not self.cancel:
                # a fit from above sees the walls obliquely and sparsely (287 points: 7 deg off, the drone hit the
                # wall 2 m in, 2026-10-20 sim); at corridor height the D435 looks straight at both: fit again there
                self.flight.fly(sx, sy, g('corridor_altitude'), face, timeout=30)
                self.set_state(f'AXIS refine alt {self.pose()[2]:.1f}')
                low, why_low = self.wall_axis(banner, face)
                if low:
                    self.get_logger().info(f'AXIS refined at corridor height: yaw {math.degrees(axis[1]):.1f} -> '
                                           f'{math.degrees(low[1]):.1f} deg, entrance ({axis[0][0]:.2f}, '
                                           f'{axis[0][1]:.2f}) -> ({low[0][0]:.2f}, {low[0][1]:.2f})')
                    axis = low
                else:
                    self.get_logger().warn(f'AXIS refine failed ({why_low}): keeping the fit from above')
            if axis:
                (ex, ey), yaw, info = axis
                confidence = 'walls'
            elif g('allow_banner_only'):
                ex, ey = banner[:2]
                yaw, info, confidence = face, {'reason': why}, 'banner only'
                self.get_logger().warn(f'no wall axis ({why}): using the banner bearing')
            else:
                raise RuntimeError(f'no corridor walls found near the banner ({why})')

            out = PoseStamped()
            out.header.frame_id = 'odom'
            out.header.stamp = self.get_clock().now().to_msg()
            out.pose.position.x, out.pose.position.y = float(ex), float(ey)
            out.pose.orientation.z, out.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            self.pose_pub.publish(out)
            result = {'colour': g('colour'), 'mode': g('mode'), 'entrance': [round(float(ex), 3), round(float(ey), 3)],
                      'yaw_deg': round(math.degrees(yaw), 2), 'confidence': confidence,
                      'banner': [round(float(v), 3) for v in banner], **loc, **info}
            self.result_pub.publish(String(data=json.dumps(result)))
            self.set_state(f'DONE entrance ({ex:.2f}, {ey:.2f}) yaw {math.degrees(yaw):.1f} ({confidence})')
            if g('run_corridor'):
                params = dict(entry_x=float(ex), entry_y=float(ey), entry_yaw=float(yaw))
                if 'width_m' in info:                     # the measured corridor, not the default
                    params['half_width'] = float(info['width_m']) / 2
                self.set_params(self.corridor_params, **params)
                r = self.call(self.corridor_run, Trigger.Request())
                if not r.success:
                    self.get_logger().error(f'corridor_nav refused: {r.message}')
        except Exception as e:
            self.flight.stop()
            self.set_state(f'FAILED: {e}')
        finally:
            self.clouds = self.collect_from = None
            self.running = False


def main():
    rclpy.init()
    node = EntryAlign()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:  # RCLError "context is not valid" when shutdown races the wait set (see spin.py)
        if rclpy.ok():
            raise
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
