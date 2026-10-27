"""Fly through a corridor with Nav2, from a start pose in front of its entrance to the open beyond.

Given the entrance (entry_x, entry_y in odom, entry_yaw = direction INTO the corridor, rad):
  1. STAGE      fly (direct velocity control, current altitude) to stage_dist before the entrance,
                facing into the corridor
  2. DESCEND    to `altitude` (v3: 1.5 m, < 8 ft) and settle; then CLEAR both costmaps and let the D435 re-map
                (remap_time): the scan is a level slice at flight height, so while the drone was above
                the walls (10 m on the return lap) every empty bearing cleared the 2D map - walls
                included - and plans then ran through the walls beside the drone, which the +-45 deg
                D435 can't re-see (2026-10-19). Cleared = unknown, which NavFn avoids (cost 253 vs 50).
     Then VIO: reset cuVSLAM (/vio/reset) here - low, facing the walls - and wait for /vio/health OK
                (vio_ready: require = FAILED if not OK within vio_wait, warn = log and go on, off). GPS is
                available outside the corridors but not counted on inside, and outside VIO loses
                tracking on any turn (2026-10-19 sim: sky, bare ground, walls beyond the D435's
                stereo range) and diverges - so it is re-armed at every entrance.
     Then the altitude planner (altitude_mode planner, course v2): corridor_nav hands it the corridor
                (entrance, axis, half_width) and starts it; inside, IT sets /nav/altitude (over / under
                obstacles) and slows Nav2 while changing height; stopped at the end of the run.
     Virtual walls: while inside, lines at +-(half_width + virtual_offset) from virtual_from metres before the
                entrance to virtual_ahead metres past the current goal (/nav/corridor/virtual_walls, a marking
                source of both costmaps): NavFn may plan through unmapped space and went ROUND the narrow
                corridor's wall outside it (course v2, 2026-10-19). Exit detection looks inside them.
  3. CORRIDOR   rolling NavigateToPose goals goal_ahead metres down the corridor axis, re-sent
                every goal_period: Nav2 (NavFn A* + MPPI omni controller, costmaps from the D435
                scan) handles obstacles, and the corridor length needn't be known. A goal that
                falls on an obstacle is slid sideways to the nearest free cell inside the corridor.
  4. EXIT       once past min_length, the corridor has ended when the wall map shows no wall
                beside the drone (+-wall_window along) on either side for exit_clear metres of travel (the walls were
                mapped while they were still ahead - the D435 sees +-45 deg only); or at max_length.
                One last goal beyond_exit further, then hold. Wall map: nav_mode direct = slam_toolbox's /map (the
                corridor Nav2 isn't running, or is paused: a paused costmap stops publishing and its last grid -
                from before the corridor - showed no walls at all); nav2 = the corridor Nav2's global costmap.
Window pass: altitude_planner reports walls across the corridor with an opening (/nav/altitude/windows, corridor
frame of the raw axis). Nav2 (2D map, square footprint) could not thread a 1.06 x 0.94 m opening - it kept failing
to find a path and backed out (2026-10-21). window_approach before one: Nav2 is cancelled, the planner's height is
overridden to the opening's centre (/nav/altitude/override), the drone lines up window_standoff in front of the
opening's centre facing along the axis, waits for the height, flies straight through at window_speed to
window_standoff past it (direct control), releases the override and Nav2 resumes.
Centre-line correction: entry_align's axis is only as good as the heading the depth clouds were placed with - in
sim PX4's yaw sat 3-6 deg off while its GPS position was right, so the walls (and the axis fitted to them) came out
rotated and the goals ran 1.4 m sideways by the exit: the drone hit the wall end twice (2026-10-20). Inside, the
scan's rays to each wall give its perpendicular distance r sin(angle to the axis); the angles are relative to the
drone, so a heading error cancels. Each frame with both walls (left + right ~ the corridor width) is a sample of the
true centre; a line through them (offset + slope, refitted as the drone flies) shifts axis() and along(), so goals,
virtual walls, exit detection and the exit goal all follow the real corridor.
No progress for stall_time -> cancel, hold position (offboard_bridge does that once /cmd_vel stops),
FAILED. In the real mission the entrance pose comes from the banner alignment; until then it is a
parameter (sim tests take it from the world's ground truth).

  service: /nav/corridor/run (std_srvs/Trigger) - starts a run with the current parameters
           /nav/corridor/cancel (std_srvs/Trigger)
  topic:   /nav/corridor/state (std_msgs/String): IDLE, STAGE, DESCEND, CORRIDOR s=<m>, EXIT,
           DONE length=<m>, FAILED: <reason>
"""
import json
import math
import threading
import time

import numpy as np
import rclpy
import rclpy.time
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan, PointCloud2, PointField
from nav2_msgs.msg import SpeedLimit
from nav2_msgs.srv import ClearEntireCostmap
from nav2_simple_commander.robot_navigator import BasicNavigator
from nav_msgs.msg import OccupancyGrid, Odometry
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool, Trigger
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from tf2_ros import Buffer, TransformListener

from aerothon_nav.direct_flight import DirectFlight, wrap, yaw_of

WALL = 99            # OccupancyGrid value of lethal (100) / inscribed (99) cells


class CorridorNav(Node):
    def __init__(self):
        super().__init__('corridor_nav')
        p = self.declare_parameter
        p('entry_x', 0.0)
        p('entry_y', 0.0)
        p('entry_yaw', 0.0)
        p('altitude', 1.5)           # m: cruise height entering (v3 tunnels: < 8 ft; the supervisor passes tunnel_altitude;
        #                                the altitude planner changes it inside when altitude_mode = planner)
        p('stage_dist', 3.0)         # m before the entrance (entry_align's estimate can be ~1 m inside)
        # NO A* IN THE CORRIDORS (user, 2026-10-22): nav_mode direct = follow the centre-corrected corridor line with
        # direct velocity commands, the pose from the 2D SLAM map (slam_toolbox map -> odom applied, pose_frame
        # map). Nav2 (NavFn A* + MPPI) planned round the entry corridor's wall end through SLAM-mapped free space
        # outside it (2026-10-22 sim). nav2 = the old Nav2 rolling goals
        p('nav_mode', 'direct')
        p('pose_frame', 'map')       # map: SLAM-corrected pose; odom: PX4's
        p('slam_max_jump', 1.0)      # m: a larger change of the SLAM correction during a corridor = SLAM lost: odom
        p('corridor_speed', 0.5)     # m/s along the line (direct)
        p('lookahead', 1.0)          # m: the point on the line steered to (direct)
        p('lateral_lookahead', 0.4)  # m: ... while following the altitude_planner's lateral plan (= its lateral_lead)
        p('sidestep_lookahead', 0.25)  # m: ... while more than 0.08 m off the altitude_planner's lateral target
        p('goal_ahead', 4.0)
        p('goal_period', 1.0)        # s between goal updates
        p('min_length', 6.0)         # m before exit detection starts
        p('max_length', 15.0)        # m (v3 tunnels: 6.7 m)
        p('half_width', 1.675)       # m; course v3 tunnels 11 ft (entry_align passes the measured one)
        p('exit_clear', 2.0)         # m of travel with no wall on either side
        p('wall_window', 0.75)       # m along the axis searched for wall cells beside the drone
        p('beyond_exit', 2.0)
        p('stall_time', 10.0)        # s without 0.3 m of progress
        p('max_speed_direct', 1.0)   # m/s for the STAGE leg (direct control)
        p('stage_timeout', 60.0)     # s (wall clock) for the STAGE leg
        p('stage_low_max', 4.0)      # m: a stage point farther than this is reached at transit_altitude, not at
        p('transit_altitude', 2.3)   # the current height: a low straight 8 m leg to a wrong stage point hit the
        #                              return corridor's window wall from outside (2026-10-21 sim)
        p('remap_time', 2.0)         # s after clearing the costmaps at corridor altitude
        p('virtual_offset', 0.7)     # m outside the wall face. Exit detection looks to half_width + 0.35; at 0.45
        #                              the costmap's inscribed padding (0.3 m) of the virtual walls fell inside that
        #                              band, the walls never 'ended' and the lap ran to max_length (2026-10-19). At 0.8
        #                              the gap outside a 0.15 m wall fitted the 0.6 m drone: it slipped round the wall
        #                              end at the mouth and flew alongside the corridor (2026-10-21)
        p('virtual_from', 3.5)       # m before the entrance: from the stage point (from 0.5 m the drone left
        #                              sideways ~1 m before the entrance and flew outside the wall, 2026-10-19)
        p('virtual_ahead', 2.0)      # m past the current goal
        p('altitude_mode', 'planner') # planner: altitude_planner sets the height inside; fixed: `altitude`
        p('vio_ready', 'warn')       # require | warn | off: VIO healthy before entering
        p('vio_wait', 10.0)          # s
        p('window_pass', False)      # thread wall openings under direct control (course v2's window panels). v3: off -
        #                              round obstacles sideways only; a hanging block's gap read as a 'window' (2026-10-24)
        p('window_approach', 1.4)    # m before a window (its slab centre) where the pass starts
        p('window_standoff', 0.9)    # m before the window's centre: line up
        p('window_exit', 1.2)        # m past the window's centre: aim point, capped window_standoff before the next
        p('window_clear', 0.75)      # m past the centre: done (0.6 left the tail under a 0.39 m lintel). Not the aim
        #                              point itself: the obstacle-scan clamp stops the drone ~0.85 m short of the
        #                              next wall, and a box 1.75 m before a window kept it 0.35 m short (2026-10-21)
        p('window_timeout', 600.0)   # s (wall clock) per leg: 90 ran out 0.4 m short at ~0.1x real time (the
        #                              corridor stall timer still catches a real stall)
        p('window_speed', 0.25)      # m/s through
        p('window_height_tol', 0.08) # m: height reached before going through
        p('centre_correction', True) # follow the centre line measured from the walls (see the docstring)
        p('centre_min_samples', 6)
        p('centre_from', 2.5)        # m before the entrance where centre samples start (the walls ahead are in view;
        #                              from s >= 0 only, a 0.3 m entrance error let the drone slip round the wall end)
        p('centre_width_tol', 0.35)  # m: left + right must be within this of 2 x half_width
        p('centre_max_slope_deg', 12.0)

        self.odom = None
        self.grid = None
        self.scan = None
        self.corr = (0.0, 0.0)       # centre line in axis coordinates: lateral = a + b * s
        self.centre_samples = []     # (s, lateral of the measured centre), raw axis coordinates
        self.last_scan_stamp = None
        self.create_subscription(LaserScan, '/vio/obstacle_scan', lambda m: setattr(self, 'scan', m),
                                 qos_profile_sensor_data)
        # altitude_planner's /speed_limit (% while the height is off its plan) - Nav2 obeys it; direct mode scales
        # corridor_speed by it too, or the drone reached a hanging box before the height (2026-10-23 sim)
        self.speed_pct = 100.0
        self.create_subscription(SpeedLimit, '/speed_limit', self.on_speed_limit, 10)
        # altitude_planner's lateral plan (course v3: standing obstacles passed sideways), raw corridor frame l
        self.lat_target = None
        self.create_subscription(Float32, '/nav/lateral',
                                 lambda m: setattr(self, 'lat_target', None if math.isnan(m.data) else float(m.data)), 10)
        self.windows = []            # latest altitude_planner report
        self.windows_done = []       # s of windows already passed this run
        self.create_subscription(String, '/nav/altitude/windows', self.on_windows, 10)
        self.alt_override = self.create_publisher(Float32, '/nav/altitude/override', 10)
        self.clamp_suspend = self.create_publisher(Bool, '/nav/clamp_suspend', 10)
        self.state = 'IDLE'
        self.running = False
        self.cancel = False
        self.state_pub = self.create_publisher(String, '/nav/corridor/state', 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.map_corr, self.use_map = None, False   # (x, y, yaw) of map -> odom; applied to the pose during a run
        self.corr0 = None            # map -> odom when the corridor run started
        self.create_timer(0.2, self.update_map_corr)
        self.flight = DirectFlight(self, self.pose, lambda: self.cancel)
        self.alt = self.flight.alt
        self.offboard = self.create_client(SetBool, '/nav/offboard')
        self.clear = [self.create_client(ClearEntireCostmap, f'/{c}_costmap/clear_entirely_{c}_costmap')
                      for c in ('global', 'local')]
        self.create_subscription(Odometry, '/nav/odom', lambda m: setattr(self, 'odom', m), 10)
        self.vio_health = ''
        self.create_subscription(String, '/vio/health', lambda m: setattr(self, 'vio_health', m.data), 10)
        self.vio_reset = self.create_client(Trigger, '/vio/reset')
        self.virtual_pub = self.create_publisher(PointCloud2, '/nav/corridor/virtual_walls', 10)
        self.alt_params = self.create_client(SetParameters, '/altitude_planner/set_parameters')
        self.alt_start = self.create_client(Trigger, '/nav/altitude/start')
        self.alt_stop = self.create_client(Trigger, '/nav/altitude/stop')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/global_costmap/costmap', lambda m: setattr(self, 'grid', m), latched)
        self.slam_grid = None        # (msg, cells as an (h, w) array) of slam_toolbox's /map, frame map
        self.create_subscription(OccupancyGrid, '/map', self.on_slam_map, latched)
        self.direct = True
        self.create_service(Trigger, '/nav/corridor/run', self.on_run)
        self.create_service(Trigger, '/nav/corridor/cancel', self.on_cancel)
        self.create_timer(1.0, lambda: self.state_pub.publish(String(data=self.state)))
        self.nav = BasicNavigator(node_name='corridor_nav_commander')

    # ---- services
    def on_run(self, req, res):
        if self.running:
            res.success, res.message = False, f'already running ({self.state})'
            return res
        self.running, self.cancel = True, False
        threading.Thread(target=self.run, daemon=True).start()
        res.success, res.message = True, 'corridor run started'
        return res

    def on_cancel(self, req, res):
        self.cancel = True
        res.success, res.message = True, 'cancel requested'
        return res

    def set_state(self, s):
        if s.split(' ')[0] != self.state.split(' ')[0] or s.startswith(('DONE', 'FAILED')):
            self.get_logger().info(s)
        self.state = s
        self.state_pub.publish(String(data=s))

    # ---- geometry
    def update_map_corr(self):
        try:
            t = self.tf.lookup_transform('map', 'odom', rclpy.time.Time())
        except Exception:
            return
        q = t.transform.rotation
        self.map_corr = (t.transform.translation.x, t.transform.translation.y, yaw_of(q))

    def to_map(self, x, y, yaw):
        tx, ty, ta = self.map_corr
        return (tx + x * math.cos(ta) - y * math.sin(ta), ty + x * math.sin(ta) + y * math.cos(ta), wrap(yaw + ta))

    def pose(self):
        """odom pose; during a corridor run with the SLAM pose: plus only the SLAM correction gained since the run
        started (corr0). Absolute map coordinates are no use here: in the open delivery zone (no walls to match) the
        map -> odom correction drifted 4.4 m, the return corridor's line went to the altitude planner (which works in
        odom) 4.4 m off, it detected no window and the drone hit window 2 (2026-10-22 sim)."""
        p = self.odom.pose.pose
        x, y, z, yaw = p.position.x, p.position.y, p.position.z, yaw_of(p.orientation)
        if self.use_map and self.map_corr is not None and self.corr0 is not None:
            mx, my, myaw = self.to_map(x, y, yaw)            # SLAM map frame now
            tx, ty, ta = self.corr0                          # back through the correction at the start
            if math.hypot(self.map_corr[0] - tx, self.map_corr[1] - ty) > self.get_parameter('slam_max_jump').value:
                self.get_logger().warn('SLAM correction jumped during the corridor: flying on the odom pose')
                self.use_map = False
                return x, y, z, yaw
            dx, dy = mx - tx, my - ty
            x, y = dx * math.cos(ta) + dy * math.sin(ta), -dx * math.sin(ta) + dy * math.cos(ta)
            yaw = wrap(myaw - ta)
        return x, y, z, yaw

    def on_speed_limit(self, m):
        # 0 % = altitude_planner holding (no lateral path round obstacles ahead); it always sends 0-100 % with
        # percentage set
        self.speed_pct = float(m.speed_limit) if m.percentage else 100.0

    def on_windows(self, m):
        try:
            self.windows = json.loads(m.data)
        except ValueError:
            pass

    def raw_axis(self, s, lateral=0.0):
        """Point s along the uncorrected axis (altitude_planner's corridor frame), lateral to its left."""
        ex, ey, h = self.ex, self.ey, self.h
        return ex + s * math.cos(h) - lateral * math.sin(h), ey + s * math.sin(h) + lateral * math.cos(h)

    def next_window(self, s_raw):
        """The nearest window ahead not yet passed, if its pass should start now."""
        if not self.get_parameter('window_pass').value:
            return None
        ahead = [w for w in self.windows if w['s'] > s_raw - 0.2
                 and all(abs(w['s'] - d) > 0.6 for d in self.windows_done)]
        if not ahead:
            return None
        w = min(ahead, key=lambda w: w['s'])
        return w if s_raw >= w['s'] - self.get_parameter('window_approach').value else None

    def pass_window(self, w):
        """Thread one wall opening under direct control (see the docstring). Raises on failure."""
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.get_logger().info(f'WINDOW at s={w["s"]:.2f}: opening centre l={w["l"]:+.2f} z={w["z"]:.2f} '
                               f'(l {w["l_lo"]:+.2f}..{w["l_hi"]:+.2f}, z {w["z_lo"]:.2f}..{w["z_hi"]:.2f})')
        self.set_state(f'WINDOW s={w["s"]:.1f}')
        self.nav.cancelTask()
        self.flight.stop()
        z = float(w['z'])

        def hold_alt():
            self.alt_override.publish(Float32(data=z))
            self.clamp_suspend.publish(Bool(data=True))   # the gap is mapped: the bridge's clamp would stop us in it
        pre = self.raw_axis(w['s'] - g('window_standoff'), w['l'])
        post_s = w['s'] + g('window_exit')
        later = [o['s'] for o in self.windows if o['s'] > w['s'] + 0.6]
        if later:
            post_s = max(min(post_s, min(later) - g('window_standoff')), w['s'] + g('window_clear'))
        post = self.raw_axis(post_s, w['l'])
        cleared = lambda: self.along_raw(*self.pose()[:2])[0] >= w['s'] + g('window_clear')  # noqa: E731
        try:
            if not self.flight.fly(*pre, None, self.h, vmax=0.3, tol=0.12, yaw_tol=0.06, timeout=g('window_timeout'),
                                   each=hold_alt):
                raise RuntimeError('could not line up with the window opening')
            end = time.monotonic() + g('window_timeout') / 3
            while abs(self.pose()[2] - z) > g('window_height_tol'):
                if time.monotonic() > end:
                    raise RuntimeError(f'window height {z:.2f} not reached (at {self.pose()[2]:.2f})')
                self.flight.hold(0.2, z, each=hold_alt)
            if not self.flight.fly(*post, None, self.h, vmax=g('window_speed'), tol=0.15, yaw_tol=0.1,
                                   timeout=g('window_timeout'), each=hold_alt, until=cleared):
                raise RuntimeError('could not get through the window')
        finally:
            self.alt_override.publish(Float32(data=float('nan')))
            self.clamp_suspend.publish(Bool(data=False))
        self.windows_done.append(w['s'])
        self.get_logger().info(f'WINDOW at s={w["s"]:.2f} passed')

    def corr_at(self, s):
        a, b = self.corr
        return a + b * s

    def axis(self, s, lateral=0.0):
        """Point s along the (centre-corrected) axis, lateral to its left."""
        lateral += self.corr_at(s)
        ex, ey, h = self.ex, self.ey, self.h
        return ex + s * math.cos(h) - lateral * math.sin(h), ey + s * math.sin(h) + lateral * math.cos(h)

    def along_raw(self, x, y):
        dx, dy = x - self.ex, y - self.ey
        return dx * math.cos(self.h) + dy * math.sin(self.h), -dx * math.sin(self.h) + dy * math.cos(self.h)

    def along(self, x, y):
        """(s, lateral from the centre-corrected axis)."""
        s, lat = self.along_raw(x, y)
        return s, lat - self.corr_at(s)

    def centre_sample(self):
        """From the latest scan: the true corridor centre at the drone, (s, lateral) in raw axis coordinates,
        or None. Wall distance per ray = r sin(phi), phi = ray angle to the axis, both relative to the drone."""
        m = self.scan
        if m is None or self.odom is None:
            return None
        stamp = (m.header.stamp.sec, m.header.stamp.nanosec)
        if stamp == self.last_scan_stamp:
            return None
        self.last_scan_stamp = stamp
        px, py, _, yaw = self.pose()
        s, lat = self.along_raw(px, py)
        if s < -self.get_parameter('centre_from').value:   # from before the entrance: the scan sees the walls
            return None                                   # ahead, so the correction is in place AT the mouth
        rel = math.atan2(math.sin(self.h + math.atan(self.corr[1]) - yaw), math.cos(self.h + math.atan(self.corr[1]) - yaw))
        left, right = [], []
        for i, r in enumerate(m.ranges):
            if not (m.range_min < r < m.range_max) or not math.isfinite(r):
                continue
            phi = m.angle_min + i * m.angle_increment - rel
            phi = math.atan2(math.sin(phi), math.cos(phi))
            if r * math.cos(phi) > 4.0:              # local walls only
                continue
            if math.radians(15) <= phi <= math.radians(75):
                left.append(r * math.sin(phi))
            elif -math.radians(75) <= phi <= -math.radians(15):
                right.append(-r * math.sin(phi))
        if len(left) < 5 or len(right) < 5:
            return None
        dl, dr = float(np.median(left)), float(np.median(right))
        hw = self.get_parameter('half_width').value
        if abs(dl + dr - 2 * hw) > self.get_parameter('centre_width_tol').value:
            return None                              # an obstacle, a gap, the exit: not the two walls
        return s, lat - (dr - dl) / 2               # the drone is (dr - dl) / 2 left of the centre

    def update_centre(self):
        """Add a centre sample and refit the centre line (robust, offset only until the samples span 1 m)."""
        if not self.get_parameter('centre_correction').value:
            return
        sample = self.centre_sample()
        if sample is None:
            return
        self.centre_samples.append(sample)
        pts = np.array(self.centre_samples[-200:])
        if len(pts) < self.get_parameter('centre_min_samples').value:
            return
        hw = self.get_parameter('half_width').value
        if np.ptp(pts[:, 0]) < 1.0:
            a, b = float(np.median(pts[:, 1])), 0.0
        else:
            b, a = np.polyfit(pts[:, 0], pts[:, 1], 1)
            keep = np.abs(pts[:, 1] - (a + b * pts[:, 0])) < 0.15
            if keep.sum() >= len(pts) // 2 and keep.sum() >= 3:
                b, a = np.polyfit(pts[keep, 0], pts[keep, 1], 1)
        b = float(np.clip(b, -math.tan(math.radians(self.get_parameter('centre_max_slope_deg').value)),
                          math.tan(math.radians(self.get_parameter('centre_max_slope_deg').value))))
        a = float(np.clip(a, -hw, hw))
        if abs(a - self.corr[0]) > 0.05 or abs(b - self.corr[1]) > 0.005:
            self.get_logger().info(f'centre line: offset {a:+.2f} m, slope {math.degrees(math.atan(b)):+.1f} deg '
                                   f'({len(pts)} samples)', throttle_duration_sec=2.0)
        self.corr = (a, b)

    def on_slam_map(self, m):
        self.slam_grid = (m, np.array(m.data, np.int8).reshape(m.info.height, m.info.width))

    def slam_cost(self, x, y):
        """/map value at the pose-frame point (x, y), the max over +-1 cell: SLAM walls are one 5 cm cell thick and
        wall_beside steps 0.1 m."""
        m, a = self.slam_grid
        tx, ty, ta = self.corr0 if (self.use_map and self.corr0 is not None) else self.map_corr
        mx, my = tx + x * math.cos(ta) - y * math.sin(ta), ty + x * math.sin(ta) + y * math.cos(ta)
        i = int((mx - m.info.origin.position.x) / m.info.resolution)
        j = int((my - m.info.origin.position.y) / m.info.resolution)
        win = a[max(0, j - 1):j + 2, max(0, i - 1):i + 2]
        return int(win.max()) if 0 <= i < m.info.width and 0 <= j < m.info.height and win.size else -1

    def cost(self, x, y):
        if self.direct and self.slam_grid is not None and self.map_corr is not None:
            return self.slam_cost(x, y)
        g = self.grid
        if g is None:
            return -1
        i = int((x - g.info.origin.position.x) / g.info.resolution)
        j = int((y - g.info.origin.position.y) / g.info.resolution)
        if 0 <= i < g.info.width and 0 <= j < g.info.height:
            return g.data[j * g.info.width + i]
        return -1

    def wall_beside(self, s, side):
        """A wall cell within +-wall_window along and half_width +-0.6 m across, on side +1 (left) / -1.
        The window is generous: walls seen at grazing angles are marked sparsely, and a 1.5 m gap in
        the marks read as the corridor's end (exit at 7.0 m of a 10 m corridor, 2026-10-19)."""
        hw = self.get_parameter('half_width').value
        w = self.get_parameter('wall_window').value
        # not the virtual walls nor their 0.3 m inscribed padding in the costmap
        top = min(hw + 0.6, hw + self.get_parameter('virtual_offset').value - 0.45)
        for ds in np.arange(-w, w + 0.01, 0.1):
            for lat in np.arange(hw - 0.6, top + 0.01, 0.1):
                if self.cost(*self.axis(s + ds, side * lat)) >= WALL:
                    return True
        return False

    def free_goal(self, s):
        """Point s down the axis, slid sideways to the nearest free cell inside the corridor."""
        hw = self.get_parameter('half_width').value
        for lat in [0.0] + [k * d for d in np.arange(0.25, hw - 0.35, 0.25) for k in (1, -1)]:
            x, y = self.axis(s, lat)
            c = self.cost(x, y)
            if 0 <= c < 50 or c == -1:      # free or not yet seen (the planner allows unknown)
                return x, y
        return self.axis(s)

    def publish_virtual_walls(self, s_to):
        """Two lines just outside the walls, entrance (- virtual_from) to s_to, in odom."""
        hw = self.get_parameter('half_width').value + self.get_parameter('virtual_offset').value
        s = np.arange(-self.get_parameter('virtual_from').value, s_to, 0.05)
        pts = []
        for side in (1, -1):
            for si in s:
                x, y = self.axis(si, side * hw)
                pts.append((x, y, self.pose()[2]))
        pts = np.asarray(pts, np.float32)
        m = PointCloud2()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = 'odom'
        m.height, m.width = 1, len(pts)
        m.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
        m.is_bigendian, m.is_dense = False, True
        m.point_step, m.row_step = 12, 12 * len(pts)
        m.data = pts.tobytes()
        self.virtual_pub.publish(m)

    def goal_msg(self, x, y):
        g = PoseStamped()
        g.header.frame_id = 'odom'                  # stamp 0: use the latest transform
        g.pose.position.x, g.pose.position.y = x, y
        g.pose.orientation.z, g.pose.orientation.w = math.sin(self.h / 2), math.cos(self.h / 2)
        return g

    def wait_future(self, fut, timeout=5.0):
        end = time.monotonic() + timeout
        while not fut.done() and time.monotonic() < end:
            time.sleep(0.02)
        return fut.result() if fut.done() else None

    def start_altitude_planner(self):
        """Hand the corridor to altitude_planner and start it; False (-> fixed altitude) if it isn't there."""
        if not (self.alt_params.wait_for_service(timeout_sec=2.0) and self.alt_start.wait_for_service(timeout_sec=2.0)):
            self.get_logger().warn('altitude_planner not available: flying the corridor at the fixed altitude')
            return False
        dbl = lambda v: ParameterValue(type=ParameterType.PARAMETER_DOUBLE, double_value=float(v))  # noqa: E731
        hw = self.get_parameter('half_width').value
        params = [Parameter(name=k, value=dbl(v)) for k, v in
                  (('entry_x', self.ex), ('entry_y', self.ey), ('entry_yaw', self.h), ('half_width', hw),
                   ('nominal', self.get_parameter('altitude').value))]
        self.wait_future(self.alt_params.call_async(SetParameters.Request(parameters=params)))
        r = self.wait_future(self.alt_start.call_async(Trigger.Request()))
        if not (r and r.success):
            self.get_logger().warn('altitude_planner did not start: fixed altitude')
            return False
        return True

    def stop_altitude_planner(self):
        if self.alt_stop.service_is_ready():
            self.wait_future(self.alt_stop.call_async(Trigger.Request()), 2.0)

    def arm_vio(self, mode, wait, alt):
        self.set_state('VIO_RESET')
        if not self.vio_health and not self.vio_reset.wait_for_service(timeout_sec=2.0):
            # no VIO in this stack at all (container_all vslam:=false): nothing to wait vio_wait for
            self.get_logger().warn('no VIO running (/vio/reset, /vio/health): entering on the PX4 / SLAM pose')
            return
        if self.vio_reset.service_is_ready():
            self.vio_reset.call_async(Trigger.Request())
            self.flight.hold(1.0, alt)                   # let the reset land before judging health
        else:
            self.get_logger().warn('/vio/reset not available')
        end = time.monotonic() + wait
        while self.vio_health != 'OK' and time.monotonic() < end and not self.cancel:
            self.flight.hold(0.2, alt)
        if self.vio_health == 'OK':
            self.get_logger().info('VIO healthy at the entrance')
        elif mode == 'require':
            raise RuntimeError(f'VIO not healthy at the entrance ({self.vio_health or "no /vio/health"})')
        else:
            self.get_logger().warn(f'entering with VIO not healthy ({self.vio_health or "no /vio/health"})')

    # ---- the run
    def run(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        try:
            self.ex, self.ey, self.h = g('entry_x'), g('entry_y'), g('entry_yaw')
            direct = self.direct = g('nav_mode') == 'direct'
            if direct and self.slam_grid is None:
                self.get_logger().warn('no /map from slam_toolbox yet: exit detection on the Nav2 costmap')
            if g('pose_frame') == 'map' and self.map_corr is not None:
                # the corridor line stays in odom (entry_align, the altitude planner); SLAM corrects the drift from
                # here on (pose())
                self.corr0 = self.map_corr
                self.use_map = True
                self.get_logger().info(f'2D SLAM pose: drift corrected from the corridor start '
                                       f'(map -> odom {self.corr0[0]:+.2f}, {self.corr0[1]:+.2f})')
            elif g('pose_frame') == 'map':
                self.get_logger().warn('no map -> odom from slam_toolbox: flying on the odom pose')
            self.corr, self.centre_samples, self.last_scan_stamp = (0.0, 0.0), [], None
            self.speed_pct, self.lat_target = 100.0, None
            self.windows, self.windows_done = [], []
            alt = g('altitude')
            if self.odom is None:
                raise RuntimeError('no /nav/odom (odom_tf running?)')
            self.set_state('WAIT_NAV2')
            # no AMCL here: wait on another lifecycle node instead (the helper waits for 'localizer')
            if not direct:
                self.nav.waitUntilNav2Active(localizer='controller_server')
            fut = self.offboard.call_async(SetBool.Request(data=True))
            end = time.monotonic() + 5
            while not fut.done() and time.monotonic() < end:
                time.sleep(0.05)
            if not (fut.done() and fut.result().success):
                raise RuntimeError(f'offboard refused: {fut.result().message if fut.done() else "timeout"}')

            self.set_state('STAGE')
            sx, sy = self.axis(-g('stage_dist'))
            vmax = g('max_speed_direct')
            px, py, pz, _ = self.pose()
            if math.hypot(sx - px, sy - py) > g('stage_low_max') and pz < g('transit_altitude'):
                self.get_logger().info(f'stage point {math.hypot(sx - px, sy - py):.1f} m away: '
                                       f'via {g("transit_altitude"):.1f} m')
                if not self.flight.fly(px, py, g('transit_altitude'), self.pose()[3], vmax=vmax,
                                       timeout=g('stage_timeout')):
                    raise RuntimeError('could not climb to the transit altitude')
            if not self.flight.fly(sx, sy, self.pose()[2], self.h, vmax=vmax, timeout=g('stage_timeout')):
                raise RuntimeError('could not reach the stage point')
            self.set_state('DESCEND')
            if not self.flight.fly(sx, sy, alt, self.h, vmax=vmax, timeout=g('stage_timeout')):
                raise RuntimeError(f'could not settle at {alt} m')
            for client in ([] if direct else self.clear):   # forget what the scan 'saw' above the walls
                if not client.wait_for_service(timeout_sec=5.0):
                    raise RuntimeError(f'{client.srv_name} not available')
                fut = client.call_async(ClearEntireCostmap.Request())
                end = time.monotonic() + 5
                while not fut.done() and time.monotonic() < end:
                    time.sleep(0.02)
            if not direct:                       # direct mode: no Nav2 costmaps to re-map (2 s saved per tunnel)
                self.flight.hold(g('remap_time'), alt)  # re-map from flight height before planning
            if g('vio_ready') != 'off':
                self.arm_vio(g('vio_ready'), g('vio_wait'), alt)

            planner = g('altitude_mode') == 'planner' and self.start_altitude_planner()
            self.set_state('CORRIDOR s=0.0')
            # stall timing on the node clock (sim time in sim): a paused / slowed simulator is not a stall
            clock = lambda: self.get_clock().now().nanoseconds * 1e-9  # noqa: E731
            best, best_t = -1e9, clock()
            clear_since = None                       # s where both sides last became wall-free
            exit_s = None
            last_goal = 0.0
            while not self.cancel:
                self.update_centre()
                px, py, pz, _ = self.pose()
                s, lat = self.along(px, py)
                self.publish_virtual_walls(s + g('goal_ahead') + g('virtual_ahead'))
                if not planner:                      # else altitude_planner owns /nav/altitude
                    self.alt.publish(Float32(data=float(alt)))
                if s > best + 0.3:
                    best, best_t = s, clock()
                elif clock() - best_t > g('stall_time'):
                    raise RuntimeError(f'no progress for {g("stall_time"):.0f} s at s={s:.1f} m')
                if exit_s is None:
                    walls = self.wall_beside(s, 1) or self.wall_beside(s, -1)
                    if s >= g('min_length') and not walls:
                        clear_since = s if clear_since is None else clear_since
                        if s - clear_since >= g('exit_clear'):
                            exit_s = s
                    else:
                        clear_since = None
                    if s >= g('max_length'):
                        exit_s = s
                    if exit_s is not None:
                        self.set_state('EXIT')
                        target = s + g('beyond_exit')
                        if not direct:
                            self.nav.goToPose(self.goal_msg(*self.axis(target)))
                        last_goal = time.monotonic()
                if exit_s is not None:
                    if s >= target - 0.4 or (not direct and self.nav.isTaskComplete()):
                        break
                    if direct:
                        self.flight.step(*self.axis(target), None if planner else alt, self.h,
                                         vmax=g('corridor_speed') * min(self.speed_pct, 100.0) / 100)
                elif self.next_window(self.along_raw(px, py)[0]) is not None:
                    self.pass_window(self.next_window(self.along_raw(px, py)[0]))
                    best, best_t = self.along(*self.pose()[:2])[0], clock()   # the pass is progress
                    last_goal = 0.0                                          # Nav2 resumes at once
                elif direct:
                    # straight along the (SLAM-pose, scan-centred) corridor line: no planner
                    ahead = min(s + g('lookahead'), g('max_length') + 1)
                    pct = min(self.speed_pct, 100.0) / 100 if planner else 1.0
                    if planner and self.lat_target is not None:   # round obstacles: the planner's lateral offset
                        s_raw, l_raw = self.along_raw(px, py)
                        look = g('lateral_lookahead')
                        if abs(l_raw - self.lat_target) > 0.08:
                            # sidestep first: steering 1 m ahead moved the drone 2 m on per 1 m across and it
                            # clipped the blocks (offline 2026-10-24); a near point = a strafe
                            look, pct = g('sidestep_lookahead'), min(pct, 0.3)
                        tx, ty = self.raw_axis(s_raw + look, self.lat_target)
                    else:
                        tx, ty = self.axis(ahead)
                    self.flight.step(tx, ty, None if planner else alt, self.h, vmax=g('corridor_speed') * pct)
                    if time.monotonic() - last_goal >= g('goal_period'):
                        last_goal = time.monotonic()
                        self.set_state(f'CORRIDOR s={s:.1f} lateral={lat:+.2f}')
                elif time.monotonic() - last_goal >= g('goal_period'):
                    self.nav.goToPose(self.goal_msg(*self.free_goal(min(s + g('goal_ahead'), g('max_length') + 1))))
                    last_goal = time.monotonic()
                    self.set_state(f'CORRIDOR s={s:.1f} lateral={lat:+.2f}')
                time.sleep(0.1)
            self.nav.cancelTask()
            self.flight.stop()
            if self.cancel:
                self.set_state('FAILED: cancelled')
            else:
                length = clear_since if clear_since is not None else exit_s
                self.set_state(f'DONE length={length:.1f}')
        except Exception as e:                       # hold position and report
            try:
                self.nav.cancelTask()
            except Exception:
                pass
            self.flight.stop()
            self.set_state(f'FAILED: {e}')
        finally:
            self.stop_altitude_planner()             # hand the height back (the bridge keeps the last one)
            self.use_map = False
            self.running = False


def main():
    rclpy.init()
    node = CorridorNav()
    ex = MultiThreadedExecutor()
    ex.add_node(node)                                # BasicNavigator spins its own node
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
