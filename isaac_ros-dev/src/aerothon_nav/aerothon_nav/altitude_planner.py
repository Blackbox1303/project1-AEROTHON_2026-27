"""Variable-altitude corridor flight (course v2, Branch C, 2.5D): choose the flight height so the drone passes
over / under obstacles; sideways avoidance stays with Nav2 (its scan band follows the drone's height).

Corridor frame from corridor_nav (entry_x/y/yaw, half_width): s along the axis from the entrance, l across
(left +), z up from the take-off ground (PX4 altitude; real drone: needs a downward rangefinder).
Per /vio/depth_points cloud (D435, body frame, 10 Hz) -> odom (TF) -> corridor frame; points inside the walls
(|l| < half_width - wall_skip), above the floor (z > floor_skip), within the map, count as hits in an
(s, z, l) grid of `res` cells. The grid is KEPT for the whole run: flying over a box, the D435 (+-29 deg
vertically) no longer sees it. A cell is occupied after `min_hits`.
  blocked(s, z) = lateral_mode path (default; corridor_nav nav_mode direct flies the centre line, nothing steers
                  round a partial block): any occupied cell in the band the drone sweeps from its lateral position
                  to the centre line, + drone_width / 2 + path_margin each side. After window 2 (opening 0.33 m
                  right of centre) 'any free run' let it hold 9.6 m beside a hanging box (bottom 9.72 m) while
                  corridor_nav steered back to the centre - under the box: it hit it (2026-10-23 sim). Window slabs
                  keep the 'any' rule (corridor_nav's window pass threads them off-centre).
                  lateral_mode any: no free lateral run at (s, z) as wide as drone_width + 2 clear_lat (a partial
                  block leaves room: not blocked, Nav2 steers around it) - OR never seen AND below the drone (more than unknown_below
                  under its height), from unknown_behind behind it to unknown_ahead ahead. Each cloud ray marks the
                  (s, z) cells it crosses and ends in as seen. Unseen used to count as free: under a window the
                  wall's lower part was below the D435's view, the plan dived to 0.6 m to 'pass under' and the drone
                  hit the solid wall (2026-10-20 sim). Only BELOW: rays up through a window opening hit nothing (no
                  depth return), so the opening itself is never 'seen' - blocking unseen space above kept the drone
                  under the opening's lower edge (offline test). Further ahead unseen stays free.
  feasible(s, zc) = no blocked cell within zc +- (drone_height / 2 + clear_z) and s +- (drone_depth / 2 + clear)
  plan: shortest path over s from the drone to lookahead metres ahead, height bins within [z_min, z_max],
        at most max_slope bins up/down per s bin, cost = climb_weight x |dz| + nominal_weight x (z - nominal)^2,
        starting from the current height. Target = of the planned heights within `lead` metres ahead, the one
        furthest from the current height (start climbs early). No path (impassable) -> hold, no_path in state.
  windows: a slab across the corridor (within window_lookahead) whose cells in the flight band are at least
           window_frac occupied, with a region free through the whole slab that the drone fits (drone_width + 2
           clear_lat wide, drone_height + 2 clear_z_min tall, in one contiguous height band): a wall with an opening.
           Published (corridor frame: s, l, z = opening centre; z_lo / z_hi, l_lo / l_hi = its free extent) on
           /nav/altitude/windows (String JSON list, 2 Hz) for corridor_nav's window pass - Nav2 (2D, square footprint)
           could not thread the 1.06 x 0.94 m openings (2026-10-21). Hanging boxes block far less than window_frac.
  override: /nav/altitude/override (Float32): fly this height instead of the plan (corridor_nav's window pass);
           NaN releases it.
  lateral: (lateral_plan, course v3 - standing obstacles in the exit tunnel, no room over them) at the drone's
           height band, occupied (s, l) cells grown by drone_width / 2 + lateral_margin across (the grid edges = the
           walls) and drone_depth / 2 + clear along; shortest path over s from the drone's l, at most lateral_slope
           bins across per s bin, cost lateral_weight x |dl| + centre_weight x l^2 (keeps to the centre line).
           /nav/lateral (Float32, corridor frame l, left +; NaN = no path / inactive): the path's l lateral_lead
           ahead - corridor_nav (nav_mode direct) steers to it. The height plan's band (lateral_mode path) then
           follows this path instead of the line to the centre.
  out: /nav/altitude (Float32, 10 Hz while active); /speed_limit (nav2_msgs/SpeedLimit, %): slow_percent while
       the drone is more than slow_error away from the height it should have now, else 100;
       /nav/altitude/state (String JSON: target, blocked ahead, occupied cells).
  srv: /nav/altitude/start, /nav/altitude/stop (std_srvs/Trigger); start clears the map and reads the corridor
       parameters (set them first).
"""
import json
import math
import os

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav2_msgs.msg import SpeedLimit
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from aerothon_nav.banner_detector import quat_to_rot
from vio_bringup.spin import spin_node


class AltitudePlanner(Node):
    def __init__(self):
        super().__init__('altitude_planner')
        p = self.declare_parameter
        p('entry_x', 0.0)
        p('entry_y', 0.0)
        p('entry_yaw', 0.0)
        p('half_width', 0.8)
        p('length', 25.0)            # m of corridor mapped (s from -1)
        p('res', 0.1)
        # course v3 (2026-10-24): roofed tunnels 10 ft high, the drone stays below 8 ft (2.44 m) inside
        p('z_top', 3.0)              # m: map height (the roof is at 3.05 m)
        p('z_min', 0.6)              # m: flight band
        p('z_max', 2.0)              # m: 8 ft rule with margin (offboard_bridge caps 2.3 in tunnels) AND below what clears
        #                              a 6 ft standing block (2.13): those are passed beside, never over - at 2.25 the
        #                              drone crept up to 2.2-2.28 m and flew over all three (2026-10-24 sim)
        p('nominal', 1.5)            # m: preferred height (corridor_nav passes its altitude)
        p('drone_width', 0.85)       # m: 847 mm prop tip to tip
        p('lateral_mode', 'path')    # path | any (see blocked in the docstring)
        p('path_margin', 0.2)        # m each side of the swept band: lateral tracking (+-0.15 m seen) + the centre
        #                              correction corridor_nav applies and this frame doesn't
        p('drone_depth', 0.85)       # m: 847 mm prop tip to tip
        p('drone_height', 0.3)
        p('clear', 0.15)             # m along the corridor
        p('clear_lat', 0.05)         # m each side for "does the drone fit at all" - Nav2's footprint +
        #                              inflation keep the real lateral clearance (0.9 m needed of 0.95-1.1 m
        #                              windows on a 0.1 m grid: edge cells made every window "no path")
        p('clear_z', 0.15)           # m above / below. drone_height / 2 + clear_z rounds UP to whole cells: 0.25 needed
        #                              9 free cells and a window's 0.94 m opening maps as 8 (cells, edge hits) -> no
        #                              path, the drone held its height and hit the edge (2026-10-21). Keep obstacle_scan's
        #                              band (sim 0.2) inside the gap this leaves, or Nav2's 2D scan sees the window's
        #                              solid part across the corridor (band 0.3: MPPI found no path at a window).
        #                              PX4's height bias (sim baro ~0.15 m) cancels: obstacles are measured relative
        #                              to the drone and placed with the same height
        p('window_frac', 0.35)       # occupied share of a slab's flight-band cells: a wall across the corridor
        p('window_lookahead', 4.0)   # m ahead of the drone searched for windows
        p('clear_z_min', 0.05)       # m: fallback clearance when nothing fits with clear_z (see plan)
        p('wall_skip', 0.08)         # m inside the wall face not mapped (wall points)
        p('floor_skip', 0.12)
        p('min_hits', 5)             # 3 let stray edge points mark cells inside window openings
        p('stride', 2)               # use every n-th cloud point
        p('lookahead', 4.0)
        p('response_dist', 0.25)     # m: a height must be reached this far BEFORE it is needed (the drone
        #                              follows ~0.85 s late); with 0.5 a high window 1.75 m before a low one
        #                              left no room to change height (forward speed is cut meanwhile instead)
        p('max_slope', 2)            # height bins per s bin (2:1, ~63 deg): steep is fine, forward speed
        #                              drops to slow_percent while the drone is off its target height
        p('lead', 0.3)               # m: target = the planned height this far ahead of the drone. (Picking
        #                              an extreme within 1.5 m pulled a later descent forward, before a high
        #                              window, 2026-10-19.)
        p('climb_weight', 4.0)
        p('nominal_weight', 0.5)
        p('slow_error', 0.1)         # m off the current planned height -> slow down
        p('slow_percent', 30.0)
        p('odom_frame', 'odom')
        p('unknown_ahead', 1.5)      # m ahead of the drone where never-seen cells count as blocked
        p('unknown_behind', 0.3)
        p('unknown_below', 0.6)      # m: unseen cells this far below the drone's height count as blocked. 0.2 at 10 m:
        #                              the D435 never sees the floor (9.5 m down, clouds clipped at 9 m), so nothing
        #                              below the drone is ever 'seen' - the plan could not dip the 0.3 m under a hanging
        #                              box and the drone hit it twice (2026-10-23 sim; offline: -1.12 m gap at 0.2,
        #                              +0.22 m at 0.6). Windows are threaded by corridor_nav's override, not this rule
        p('seen_stride', 4)          # every n-th kept cloud point traced for the seen grid
        p('lateral_plan', True)      # plan the lateral offset too (see lateral in the docstring)
        p('vertical_avoid', False)   # False (v3, user 2026-10-24: 2D SLAM - round obstacles LEFT / RIGHT, not up / down):
        #                              the height stays `nominal`, no height search; no way round -> hold (the stall
        #                              -> FAILED -> the failsafe retraces out). A dive under a hanging block into a
        #                              mis-detected 'window' flipped the drone (2026-10-24 sim)
        p('lateral_width', 0.72)     # m across the corridor: 847 mm is the prop-tip DIAGONAL; nose along the axis the
        #                              X-quad spans ~0.72 m sideways (motor diagonal ~0.50 + props ~0.35)
        p('lateral_margin', 0.03)    # m beyond the drone's half width (prop tips) from obstacle faces / walls; the 0.1 m
        #                              cells add ~0.05. 0.25 closed the 1.23 m channel two wall-side v3 blocks leave
        #                              for the 0.85 m drone (offline 2026-10-24)
        p('lateral_slope', 3)        # l bins per s bin (72 deg; the drone strafes). 1 found no way between blocks
        #                              alternating 2 m apart 0.4 m off the walls (offline 2026-10-24)
        p('lateral_clear', 0.2)      # m along the corridor beyond the drone's half depth. Offline (2026-10-24): 0.05 -
        #                              4 cm into the block corners (steps checked at cell centres), 0.2 + slope 3:
        #                              v3 wall-side blocks +0.16 m, a middle block +0.05 m; blocks alternating 0.4 m off
        #                              the walls 2 m apart = no path -> hold (lateral_block_hold)
        p('lateral_weight', 1.0)
        p('centre_weight', 0.3)
        p('lateral_block_hold', True)  # no lateral path round what is mapped ahead: speed limit 0 % (corridor_nav holds,
        #                              its stall timer -> FAILED -> the failsafe retraces out) - not the centre line
        p('lateral_min_hits', 40)    # hits for a cell of the lateral map (live map 2026-10-24: block faces ~1000, edge noise
        #                              ~20: at 40 the edge sits within 0.06 m of the true face). Edge noise at 5 widened the blocks enough to
        #                              close the 1.22 m channel two v3 wall-side blocks leave (30 s of holding, 2026-10-24)
        p('lateral_heights', 0.2)    # m steps: no way round at this height -> the nearest height in [z_min, nominal +
        #                              lateral_up] with one (under a hanging block when there is no room beside it)
        p('lateral_up', 0.1)         # m: how far ABOVE the cruise height that search may go
        p('lateral_lead', 0.4)       # m: target = the planned l this far ahead (= corridor_nav lateral_lookahead: its
        #                              steer point IS the path there). 0.8-1.0 m: a straight chord of the path's bends -
        #                              it cut the blocks' corners (offline 2026-10-24)
        p('dump_file', os.path.expanduser('~/.ros/aerothon_altitude_map.npz'))   # save_run.sh copies it to the run           # debug: save the map (npz) the first time there is no path, and at stop
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.active = False
        self.odom = None
        self.hits = None
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.alt_pub = self.create_publisher(Float32, '/nav/altitude', 10)
        self.lat_pub = self.create_publisher(Float32, '/nav/lateral', 10)
        self.speed_pub = self.create_publisher(SpeedLimit, '/speed_limit', 10)
        self.state_pub = self.create_publisher(String, '/nav/altitude/state', 10)
        self.create_subscription(Odometry, '/nav/odom', lambda m: setattr(self, 'odom', m), 10)
        self.create_subscription(PointCloud2, '/vio/depth_points', self.on_cloud, qos_profile_sensor_data)
        self.override = None
        self.create_subscription(Float32, '/nav/altitude/override',
                                 lambda m: setattr(self, 'override', None if math.isnan(m.data) else float(m.data)), 10)
        self.win_pub = self.create_publisher(String, '/nav/altitude/windows', 10)
        self.create_timer(0.5, self.publish_windows)
        self.create_service(Trigger, '/nav/altitude/start', self.on_start)
        self.create_service(Trigger, '/nav/altitude/stop', self.on_stop)
        self.create_timer(0.1, self.tick)
        self.target = None
        self.last_speed = None
        self.clouds = self.tf_fail = 0
        self.dumped = False

    # ---- corridor frame + map
    def frame(self):
        yaw = self.g('entry_yaw')
        return (np.array([self.g('entry_x'), self.g('entry_y')]), np.array([math.cos(yaw), math.sin(yaw)]),
                np.array([-math.sin(yaw), math.cos(yaw)]))

    def on_start(self, req, res):
        r, hw = self.g('res'), self.g('half_width')
        self.ns, self.nz = int(round((self.g('length') + 1.0) / r)), int(round(self.g('z_top') / r))
        self.nl = max(1, int(round(2 * hw / r)))
        self.hits = np.zeros((self.ns, self.nz, self.nl), np.uint16)
        self.seen = np.zeros((self.ns, self.nz), bool)
        self.origin, self.u, self.n = self.frame()
        self.active, self.target = True, None
        self.clouds = self.tf_fail = 0
        self.dumped = False
        res.success, res.message = True, f'mapping {self.ns} x {self.nz} x {self.nl} cells, half width {hw:.2f} m'
        self.get_logger().info(res.message)
        return res

    def dump(self, why, s_now=None, z_now=None):
        if not self.g('dump_file') or self.hits is None:
            return
        self.dumped = True
        np.savez_compressed(self.g('dump_file'), hits=self.hits, seen=self.seen, why=why, s=s_now, z=z_now, res=self.g('res'),
                            half_width=self.g('half_width'))
        self.get_logger().info(f'map saved ({why}) to {self.g("dump_file")}')

    def on_stop(self, req, res):
        if self.active and not self.dumped:
            self.dump('stop')
        self.active = False
        self.publish_speed(100.0)
        self.lat_pub.publish(Float32(data=float('nan')))
        res.success, res.message = True, 'stopped'
        return res

    def on_cloud(self, m):
        if not self.active:
            return
        try:
            t = self.tf.lookup_transform(self.g('odom_frame'), m.header.frame_id, m.header.stamp,
                                         timeout=Duration(seconds=0.05))
        except Exception:
            try:                                    # cloud newer than the pose: latest pose (drone is slow)
                t = self.tf.lookup_transform(self.g('odom_frame'), m.header.frame_id, rclpy.time.Time())
            except Exception as e:
                self.tf_fail += 1
                self.get_logger().warn(f'cloud dropped, no TF: {e}', throttle_duration_sec=5.0)
                return
        self.clouds += 1
        pts = np.frombuffer(m.data, np.float32).reshape(-1, 3)[::self.g('stride')].astype(np.float64)
        R = quat_to_rot(t.transform.rotation)
        o = t.transform.translation
        pts = pts @ R.T + [o.x, o.y, o.z]
        d = pts[:, :2] - self.origin
        s, l, z = d @ self.u, d @ self.n, pts[:, 2]
        r, hw = self.g('res'), self.g('half_width')
        keep = (np.abs(l) < hw - self.g('wall_skip')) & (z > self.g('floor_skip')) & (z < self.g('z_top'))
        si = np.floor((s[keep] + 1.0) / r).astype(int)
        zi = np.floor(z[keep] / r).astype(int)
        li = np.floor((l[keep] + hw) / r).astype(int)
        ok = (si >= 0) & (si < self.ns) & (zi >= 0) & (zi < self.nz) & (li >= 0) & (li < self.nl)
        np.add.at(self.hits, (si[ok], zi[ok], li[ok]), 1)
        np.minimum(self.hits, 1000, out=self.hits)
        self.mark_seen(np.array([o.x, o.y, o.z]), pts)

    def mark_seen(self, cam, pts):
        """Cells the rays from the camera to the cloud points cross or end in (inside the walls): seen."""
        r, hw = self.g('res'), self.g('half_width')
        pts = pts[::self.g('seen_stride')]
        if len(pts) == 0:
            return
        L = np.linalg.norm(pts - cam, axis=1)
        steps = int(min(np.ceil(L.max() / r), 120)) + 1
        t = np.linspace(0.0, 1.0, steps)
        samp = cam + (pts - cam)[:, None, :] * t[None, :, None]          # (N, steps, 3)
        samp = samp.reshape(-1, 3)
        d = samp[:, :2] - self.origin
        s, l, z = d @ self.u, d @ self.n, samp[:, 2]
        keep = (np.abs(l) < hw) & (z >= 0.0) & (z < self.g('z_top'))
        si = np.floor((s[keep] + 1.0) / r).astype(int)
        zi = np.floor(z[keep] / r).astype(int)
        ok = (si >= 0) & (si < self.ns) & (zi >= 0) & (zi < self.nz)
        self.seen[si[ok], zi[ok]] = True

    # ---- planning
    def blocked(self, s_now=None, z_now=None, l_now=None, l_path=None):
        """(ns, nz) bool: the drone's lateral path (lateral_mode path) or every lateral run (any) is occupied, or
        never seen close ahead."""
        occ = self.hits >= self.g('min_hits')
        r, hw = self.g('res'), self.g('half_width')
        need = int(math.ceil((self.g('drone_width') + 2 * self.g('clear_lat')) / r - 1e-6))
        if need > self.nl:
            return occ.any(axis=2) | True       # corridor narrower than the drone + clearance: nothing fits
        free = ~occ
        # longest run of free lateral cells: for each start, are `need` consecutive cells free?
        win = np.lib.stride_tricks.sliding_window_view(free, need, axis=2).all(axis=3)
        b = ~win.any(axis=2)
        if self.g('lateral_mode') == 'path' and s_now is not None:
            l_now = 0.0 if l_now is None else l_now
            half = self.g('drone_width') / 2 + self.g('path_margin')
            j0 = int(np.clip(math.floor((min(l_now, 0.0) - half + hw) / r), 0, self.nl))
            j1 = int(np.clip(math.ceil((max(l_now, 0.0) + half + hw) / r), j0 + 1, self.nl))
            path = occ[:, :, j0:j1].any(axis=2)
            if l_path is not None:                  # the lateral plan's own path, s by s, with ITS margin (the wider
                half = self.g('drone_width') / 2 + self.g('lateral_margin')   # path_margin saw the blocks beside
                for si, lp in l_path.items():         # the path and climbed to 2.1 m, offline 2026-10-24)
                    a = int(np.clip(math.floor((lp - half + hw) / r), 0, self.nl))
                    e = int(np.clip(math.ceil((lp + half + hw) / r), a + 1, self.nl))
                    path[si] = occ[si, :, a:e].any(axis=1)
            for w in self.find_windows(s_now):  # thread windows off-centre (corridor_nav's pass): the 'any' rule
                a = int(np.clip(math.floor((w['s'] - w['thickness'] / 2 + 1.0) / r) - 1, 0, self.ns))
                e = int(np.clip(math.ceil((w['s'] + w['thickness'] / 2 + 1.0) / r) + 1, 0, self.ns))
                path[a:e] = b[a:e]
            b = path
        if s_now is not None:
            a = int(np.clip(math.floor((s_now - self.g('unknown_behind') + 1.0) / r), 0, self.ns))
            e = int(np.clip(math.floor((s_now + self.g('unknown_ahead') + 1.0) / r), 0, self.ns))
            zc = self.nz if z_now is None else int(np.clip(math.floor((z_now - self.g('unknown_below')) / r), 0, self.nz))
            b[a:e, :zc] |= ~self.seen[a:e, :zc]
        return b

    def feasible(self, blocked, clear_z=None):
        """(ns, nz) bool: the drone centred at (s, z) keeps `clear` from every blocked cell."""
        r = self.g('res')
        clear_z = self.g('clear_z') if clear_z is None else clear_z
        hz = int(math.ceil((self.g('drone_height') / 2 + clear_z) / r - 1e-6))
        hs_back = int(math.ceil((self.g('drone_depth') / 2 + self.g('clear')) / r - 1e-6))
        hs_ahead = int(math.ceil((self.g('drone_depth') / 2 + self.g('clear') + self.g('response_dist')) / r - 1e-6))
        b = blocked.copy()
        for k in range(1, hz + 1):              # dilate in z
            b[:, k:] |= blocked[:, :-k]
            b[:, :-k] |= blocked[:, k:]
        b2 = b.copy()
        for k in range(1, hs_back + 1):         # in s: behind an obstacle cell (the drone's tail) ...
            b2[k:] |= b[:-k]
        for k in range(1, hs_ahead + 1):        # ... and in front of it, + response_dist to get there in time
            b2[:-k] |= b[k:]
        return ~b2

    def plan(self, s_now, z_now, l_now=None, l_path=None):
        r = self.g('res')
        blocked = self.blocked(s_now, z_now, l_now, l_path)
        feas = self.feasible(blocked)
        target, stuck = self.solve(feas, s_now, z_now, skip=0)
        if stuck and self.g('clear_z_min') < self.g('clear_z'):
            # a window opening maps smaller than it is (edge hits: 0.94 m showed as 0.6-0.7 m, 2026-10-21) and with
            # full clearance nothing fit: the drone HELD its height and hit the edge. The map's own shrink is a
            # margin, so retry with the minimum clearance (the cost still centres the drone in the gap)
            feas_min = self.feasible(blocked, self.g('clear_z_min'))
            target_min, stuck_min = self.solve(feas_min, s_now, z_now, skip=0)
            if not stuck_min:
                return target_min, False
        if stuck:                                   # already too close to change height in time: plan from
            hs = int(math.ceil((self.g('drone_depth') / 2 + self.g('clear') + self.g('response_dist')) / r - 1e-6))
            target, retry_stuck = self.solve(feas, s_now, z_now, skip=hs)
            if not retry_stuck:
                return target, False
        return target, stuck

    def solve(self, feas, s_now, z_now, skip):
        r = self.g('res')
        s0 = int(np.clip(math.floor((s_now + 1.0) / r), 0, self.ns - 1))
        s1 = min(self.ns, s0 + int(self.g('lookahead') / r) + 1)
        z_lo, z_hi = int(math.ceil(self.g('z_min') / r)), int(math.floor(self.g('z_max') / r))
        zs = np.arange(z_lo, z_hi + 1)
        z_start = int(np.clip(round(z_now / r), z_lo, z_hi))
        cw, nw, nom = self.g('climb_weight'), self.g('nominal_weight'), self.g('nominal') / r
        inf = np.inf
        cost = np.full(len(zs), inf)
        cost[z_start - z_lo] = 0.0
        back = []
        for si in range(s0 + 1, s1):
            new = np.full(len(zs), inf)
            arg = np.zeros(len(zs), int)
            m = int(self.g('max_slope'))
            for dz in range(-m, m + 1):
                prev = np.roll(cost, dz)            # from zs[j - dz] to zs[j]
                if dz > 0:
                    prev[:dz] = inf
                elif dz < 0:
                    prev[dz:] = inf
                c = prev + cw * abs(dz) * r
                better = c < new
                new[better], arg[better] = c[better], (np.arange(len(zs)) - dz)[better]
            new += nw * ((zs - nom) * r) ** 2 * r
            if si - s0 > skip:
                new[~feas[si, z_lo:z_hi + 1]] = inf
            cost = new
            back.append(arg)
        if not back or not np.isfinite(cost).any():
            return None, True
        j = int(np.argmin(cost))
        path = [j]
        for arg in reversed(back):
            j = int(arg[j])
            path.append(j)
        path.reverse()                              # height bins from s0 to s1 - 1
        k = min(len(path) - 1, int(round(self.g('lead') / r)))
        target = zs[path[k]]
        # but never outside the heights that are safe where the drone IS: the planned height `lead` ahead may
        # already be past an obstacle the drone's tail is still in (descending out of a window too early)
        here = feas[s0, z_lo:z_hi + 1]
        j0 = z_start - z_lo
        if here[j0]:
            lo = j0
            while lo > 0 and here[lo - 1]:
                lo -= 1
            hi = j0
            while hi < len(here) - 1 and here[hi + 1]:
                hi += 1
            target = int(np.clip(target, zs[lo], zs[hi]))
        return float(target * r), False

    def plan_lateral(self, s_now, z_now, l_now, z_target=None):
        """(target l, {s index: l} path) round what fills the drone's height band, or (None, None) if no way."""
        r, hw = self.g('res'), self.g('half_width')
        occ = self.hits >= self.g('lateral_min_hits')
        zt = z_now if z_target is None else z_target
        h = self.g('drone_height') / 2 + self.g('clear_z')
        z0 = int(np.clip(math.floor((min(z_now, zt) - h) / r), 0, self.nz - 1))
        z1 = int(np.clip(math.ceil((max(z_now, zt) + h) / r), z0 + 1, self.nz))
        # across at half the map's cells: rounding the drone's half width up to 0.1 m cells closed the 1.22 m
        # channel two wall-side v3 blocks leave (0.19 m each side for 0.85 m) once the axis was 1.9 deg off (2026-10-24)
        fl = 2
        rl, nlf = r / fl, self.nl * fl
        occ2 = np.repeat(occ[:, z0:z1, :].any(axis=1), fl, axis=1).astype(np.uint8)   # (s, l / fl)
        kl = int(math.ceil((self.g('lateral_width') / 2 + self.g('lateral_margin')) / rl - 1e-6))
        ks = int(math.ceil((self.g('drone_depth') / 2 + self.g('lateral_clear')) / r - 1e-6))
        grown = occ2.astype(bool)
        for k in range(1, kl + 1):                  # across
            grown[:, k:] |= occ2[:, :-k].astype(bool)
            grown[:, :-k] |= occ2[:, k:].astype(bool)
        g2 = grown.copy()
        for k in range(1, ks + 1):                  # along
            g2[k:] |= grown[:-k]
            g2[:-k] |= grown[k:]
        free = ~g2
        edge = max(0, kl - int(round(self.g('wall_skip') / rl)))    # the walls at the grid edges
        if edge:
            free[:, :edge] = False
            free[:, nlf - edge:] = False
        s0 = int(np.clip(math.floor((s_now + 1.0) / r), 0, self.ns - 1))
        s1 = min(self.ns, s0 + int(self.g('lookahead') / r) + 1)
        ls = (np.arange(nlf) + 0.5) * rl - hw
        j_start = int(np.clip(math.floor((l_now + hw) / rl), 0, nlf - 1))
        cost = np.full(nlf, np.inf)
        cost[j_start] = 0.0
        back, m = [], int(self.g('lateral_slope')) * fl
        lw, cw = self.g('lateral_weight'), self.g('centre_weight')
        for si in range(s0 + 1, s1):
            new = np.full(nlf, np.inf)
            arg = np.zeros(nlf, int)
            for dj in range(-m, m + 1):
                prev = np.roll(cost, dj)
                if dj > 0:
                    prev[:dj] = np.inf
                elif dj < 0:
                    prev[dj:] = np.inf
                c = prev + lw * abs(dj) * rl
                better = c < new
                new[better], arg[better] = c[better], (np.arange(nlf) - dj)[better]
            new += cw * ls ** 2 * r
            new[~free[si]] = np.inf                 # every row ahead (only the drone's own row s0 is exempt: 'si - s0 >
            #                                         ks' let the path run through a block just ahead, offline 2026-10-24)
            cost = new
            back.append(arg)
        if not back or not np.isfinite(cost).any():
            return None, None
        j = int(np.argmin(cost))
        idx = [j]
        for arg in reversed(back):
            j = int(arg[j])
            idx.append(j)
        idx.reverse()                               # l bins for s0 .. s1 - 1
        k = min(len(idx) - 1, int(round(self.g('lateral_lead') / r)))
        return float(ls[idx[k]]), {s0 + i: float(ls[j]) for i, j in enumerate(idx)}

    def obstacle_ahead(self, s_now, z_now):
        """Anything mapped in the drone's height band within lookahead ahead (inside the walls)."""
        r = self.g('res')
        h = self.g('drone_height') / 2 + self.g('clear_z')
        s0 = int(np.clip(math.floor((s_now + 1.0) / r), 0, self.ns - 1))
        s1 = min(self.ns, s0 + int(self.g('lookahead') / r) + 1)
        z0 = int(np.clip(math.floor((z_now - h) / r), 0, self.nz - 1))
        z1 = int(np.clip(math.ceil((z_now + h) / r), z0 + 1, self.nz))
        return bool((self.hits[s0:s1, z0:z1, :] >= self.g('min_hits')).any())

    def find_windows(self, s_now):
        """Walls across the corridor with an opening the drone fits, within window_lookahead ahead (see docstring)."""
        r, hw = self.g('res'), self.g('half_width')
        occ = self.hits >= self.g('min_hits')
        z0, z1 = int(math.ceil(self.g('z_min') / r)), int(math.floor(self.g('z_max') / r)) + 1
        s0 = int(np.clip(math.floor((s_now - 0.5 + 1.0) / r), 0, self.ns))
        s1 = int(np.clip(math.floor((s_now + self.g('window_lookahead') + 1.0) / r), 0, self.ns))
        if s1 <= s0:
            return []
        frac = occ[s0:s1, z0:z1, :].mean(axis=(1, 2))
        need_l = int(math.ceil((self.g('drone_width') + 2 * self.g('clear_lat')) / r - 1e-6))
        need_z = int(math.ceil((self.g('drone_height') + 2 * self.g('clear_z_min')) / r - 1e-6))
        out, i = [], 0
        while i < len(frac):
            if frac[i] < self.g('window_frac'):
                i += 1
                continue
            a = i                                    # the slab: neighbours down to half the threshold
            while a > 0 and frac[a - 1] >= self.g('window_frac') / 2:
                a -= 1
            b = i
            while b < len(frac) - 1 and frac[b + 1] >= self.g('window_frac') / 2:
                b += 1
            i = b + 1
            free = ~occ[s0 + a:s0 + b + 1, z0:z1, :].any(axis=0)          # (z, l) free through the whole slab
            runs = []                                                     # per height: longest lateral free run
            for row in free:
                best, cur, start, bs = 0, 0, 0, 0
                for j, f in enumerate(row):
                    cur = cur + 1 if f else 0
                    if cur == 1:
                        start = j
                    if cur > best:
                        best, bs = cur, start
                runs.append((best, bs))
            ok = [k for k, (n, _) in enumerate(runs) if n >= need_l]
            bands, cur = [], []
            for k in ok:                             # contiguous height bands of fitting rows
                if cur and k != cur[-1] + 1:
                    bands.append(cur)
                    cur = []
                cur.append(k)
            if cur:
                bands.append(cur)
            fit = None
            for band in sorted(bands, key=len, reverse=True):
                lo = max(runs[k][1] for k in band)
                hi = min(runs[k][1] + runs[k][0] for k in band)
                if len(band) >= need_z and hi - lo >= need_l:
                    fit = (band, lo, hi)
                    break
            if fit is None:
                continue                              # a closed wall: nothing to thread (the plan holds / goes over)
            band, lo, hi = fit
            zl, zh = (band[0] + z0) * r, (band[-1] + z0 + 1) * r
            ll, lh = lo * r - hw, hi * r - hw
            out.append({'s': round(((s0 + a + s0 + b + 1) / 2) * r - 1.0, 2), 'thickness': round((b - a + 1) * r, 2),
                        'l': round((ll + lh) / 2, 2), 'l_lo': round(ll, 2), 'l_hi': round(lh, 2),
                        'z': round((zl + zh) / 2, 2), 'z_lo': round(zl, 2), 'z_hi': round(zh, 2),
                        'blocked_share': round(float(frac[a:b + 1].max()), 2)})
        return out

    def publish_windows(self):
        if not self.active or self.odom is None:
            return
        pp = self.odom.pose.pose.position
        s_now = float((np.array([pp.x, pp.y]) - self.origin) @ self.u)
        self.win_pub.publish(String(data=json.dumps(self.find_windows(s_now))))

    def publish_speed(self, pct):
        if self.last_speed != pct:
            msg = SpeedLimit()
            msg.percentage, msg.speed_limit = True, float(pct)
            self.speed_pub.publish(msg)
            self.last_speed = pct

    def tick(self):
        if not self.active or self.odom is None:
            return
        pp = self.odom.pose.pose.position
        d = np.array([pp.x, pp.y]) - self.origin
        s_now, z_now, l_now = float(d @ self.u), float(pp.z), float(d @ self.n)
        l_target, l_path = (self.plan_lateral(s_now, z_now, l_now, self.target) if self.g('lateral_plan')
                            else (None, None))
        z_lateral = None                             # a height the lateral plan found a way at, when not this one
        if l_target is None and self.g('lateral_plan') and self.g('vertical_avoid') and self.obstacle_ahead(s_now, z_now):
            # DOWN from the cruise height only: standing blocks are passed beside, never over - searching up found a
            # 'way' at 2.0 m, 2 cm over a 1.83 m block whose sparsely-seen top was under lateral_min_hits (2026-10-24)
            step = self.g('lateral_heights')
            top = min(self.g('z_max'), self.g('nominal') + self.g('lateral_up'))
            heights = sorted(np.arange(self.g('z_min'), top + 1e-6, step), key=lambda z: abs(z - min(z_now, top)))
            for zh in heights:
                lt, lp = self.plan_lateral(s_now, float(zh), l_now)
                if lt is not None:
                    l_target, l_path, z_lateral = lt, lp, float(zh)
                    self.get_logger().info(f'no way round at {z_now:.2f} m: at {zh:.2f} m there is', throttle_duration_sec=5.0)
                    break
        lat_blocked = (l_target is None and self.g('lateral_plan') and self.g('lateral_block_hold')
                       and self.obstacle_ahead(s_now, z_now))
        changing = z_lateral is not None and abs(z_now - z_lateral) > self.g('slow_error')
        if changing:
            l_target = l_now                         # first the height (in place), then along the way found there
            self.get_logger().warn(f'no lateral path at s={s_now:.1f}: holding', throttle_duration_sec=5.0)
            l_target = l_now                         # stay put across; speed 0 below
        self.lat_pub.publish(Float32(data=float('nan') if l_target is None else l_target))
        target, stuck = self.plan(s_now, z_now, l_now, l_path)
        if not self.g('vertical_avoid'):
            target = self.g('nominal')               # left / right only: the height never changes for obstacles
        if z_lateral is not None:
            target = z_lateral                       # the height with a way round (speed limited until there, below)
        if lat_blocked and target is not None:
            # holding for a way round: at most the cruise height - capping at the CURRENT height let overshoot ratchet
            # the drone up to 2.28 m, over the blocks (2026-10-24 sim); standing obstacles go round, not over
            target = min(target, self.g('nominal'))
        if stuck and not self.dumped:
            self.dump('no_path', s_now, z_now)
        if target is None:
            target = self.target if self.target is not None else z_now
        if self.override is not None:               # corridor_nav's window pass
            target = self.override
        self.target = target
        self.alt_pub.publish(Float32(data=float(target)))
        self.publish_speed(0.0 if lat_blocked or changing else
                           self.g('slow_percent') if abs(z_now - target) > self.g('slow_error') else 100.0)
        self.state_pub.publish(String(data=json.dumps({
            's': round(s_now, 2), 'z': round(z_now, 2), 'target': round(target, 2), 'no_path': stuck,
            'l': round(l_now, 2), 'l_target': None if l_target is None else round(l_target, 2),
            'lateral_blocked': bool(lat_blocked),
            'occupied_cells': int((self.hits >= self.g('min_hits')).sum()), 'clouds': self.clouds,
            'tf_fail': self.tf_fail})))


def main():
    spin_node(AltitudePlanner)


if __name__ == '__main__':
    main()
