"""Drive the autopilot from ROS velocity commands, through MAVROS: PX4 in OFFBOARD, ArduCopter in GUIDED.

  in:  /cmd_vel, /delivery/cmd_vel  (cmd_topics) geometry_msgs/Twist, BODY frame (x forward, y left, angular.z yaw
                                 rate CCW) - what Nav2's controller and our own controllers publish
       /nav/altitude             std_msgs/Float32, metres above the local origin (take-off point);
                                 held by this node, so planners only steer horizontally
       /mavros/local_position/pose, /mavros/state
  out: /mavros/setpoint_raw/local  PositionTarget (ENU; MAVROS converts to NED), streamed at rate_hz
  srv: /nav/offboard             std_srvs/SetBool: true = switch to OFFBOARD / GUIDED (the drone must be
                                 armed and airborne - take-off is the supervisor's job), false = AUTO.LOITER / BRAKE
       /nav/bridge_state         std_msgs/String, for the supervisor and logs

Safety clamp (independent of planning and localisation): with a fresh /vio/obstacle_scan (D435, +-45 deg
ahead, body yaw, levelled), every bearing with an obstacle closer than slow_dist scales down the part of the
command heading TOWARD it, to zero at stop_dist (ranges from the drone centre: 0.55 m = 0.2 m beyond the 0.35 m
radius); motion past or away from it is untouched. It only covers what the D435 sees - nothing behind or to the
sides. PX4's own Collision Prevention only acts in the manual Position/Acceleration flight tasks
(checked in the PX4 source, flight_mode_manager/tasks), not on offboard setpoints - hence this.

autopilot (default $AEROTHON_AUTOPILOT, else ardupilot - the real drone is ArduCopter 4.7.2): px4 | ardupilot.
PX4: setpoints stream all the time (PX4 refuses OFFBOARD without a stream). ArduCopter: only while offboard is
requested - GUIDED takes any position target at once, so a stream would cut the supervisor's NAV_TAKEOFF short; BRAKE
(not LOITER: in LOITER a low or absent throttle stick sinks the drone) for false. Velocities and the yaw rate
are clamped. If /cmd_vel goes quiet for cmd_timeout, the bridge holds the position where it was
(position setpoint) instead of coasting on the last command. With no altitude command yet, it holds
the altitude it had when offboard was enabled.
"""
import json
import math
import os
import rclpy.duration

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import LaserScan
from mavros_msgs.msg import PositionTarget, State
from mavros_msgs.srv import SetMode
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import SetBool

from vio_bringup.spin import spin_node

PT = PositionTarget
VELOCITY_MASK = (PT.IGNORE_PX | PT.IGNORE_PY | PT.IGNORE_PZ | PT.IGNORE_AFX | PT.IGNORE_AFY |
                 PT.IGNORE_AFZ | PT.IGNORE_YAW)
HOLD_MASK = (PT.IGNORE_VX | PT.IGNORE_VY | PT.IGNORE_VZ | PT.IGNORE_AFX | PT.IGNORE_AFY |
             PT.IGNORE_AFZ | PT.IGNORE_YAW_RATE)


# flight modes (MAVROS custom_mode names) per autopilot: commanded flight, hold position
MODES = {'px4': ('OFFBOARD', 'AUTO.LOITER'), 'ardupilot': ('GUIDED', 'BRAKE')}


def clamp(v, lim):
    return max(-lim, min(lim, v))


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


class OffboardBridge(Node):
    def __init__(self):
        super().__init__('offboard_bridge')
        p = self.declare_parameter
        p('autopilot', os.environ.get('AEROTHON_AUTOPILOT', 'ardupilot'))   # px4 | ardupilot
        p('rate_hz', 20.0)
        p('max_xy', 2.5)            # m/s; rules: 5-10 km/h = 1.4-2.8 m/s
        p('max_vz', 2.0)            # m/s
        # HARD GEOFENCE (user, 2026-10-22: the course's geofence must never be crossed): allowed = union of boxes
        # [x_min, x_max, y_min, y_max] in odom - the start area, each corridor's inside (geofence_margin off its
        # walls: corridors are entered and left only through their mouths), the delivery zone. A command that
        # would leave it within geofence_lookahead is cut (sliding along the edge where possible); outside, the
        # drone is steered back to the nearest allowed point. Boxes from geofence_json (sim world ground truth)
        # and/or geofence_boxes (flat list, the coordinates given on the day)
        p('geofence_json', '')
        p('geofence_boxes', [0.0])  # flat [x0, x1, y0, y1, ...]; [0.0] = none
        p('geofence_start', [-4.0, 5.0, -4.0, 4.0])   # the start area (with the json)
        p('geofence_margin', 0.45)  # m off corridor walls: the 0.85 m drone's half width (prop tips) + 0.03
        p('tunnel_max_altitude', 2.3)   # m: course v3 tunnels are roofed at 10 ft, the drone stays below 8 ft (2.44 m):
        #                                 every height target is capped to this inside a corridor box (+ tunnel_margin
        #                                 along it), and above it the fence keeps the drone out of the corridor boxes
        p('tunnel_margin', 1.0)     # m before / after a tunnel where the cap already / still applies
        p('geofence_lookahead', 0.6)  # s
        p('max_altitude', 11.2)     # m: hard cap on every altitude target (10 m flight + the altitude planner's band),
        #                             whatever a planner or the supervisor asks for (0.33 m overshoot seen, 2026-10-21)
        p('max_yaw_rate', 0.8)      # rad/s
        p('alt_kp', 0.8)            # 1/s, altitude error -> climb rate
        p('alt_ki', 0.3)            # 1/s^2: P alone sat 0.1-0.19 m below the target (course v2 sim, 2026-10-19)
        p('alt_i_max', 0.3)         # m/s, integral contribution limit
        p('cmd_timeout', 0.5)       # s without /cmd_vel -> hold position
        p('cmd_topics', ['/cmd_vel', '/delivery/cmd_vel'])   # corridor Nav2 / own controllers, delivery Nav2
        p('clamp', True)            # safety clamp on /vio/obstacle_scan
        p('stop_dist', 0.65)        # m from the drone centre: no motion toward an obstacle this close
        p('slow_dist', 1.0)         # m: approach speed scaled down linearly from here
        p('scan_timeout', 0.5)      # s: older scan = no clamp (warned)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.autopilot = g('autopilot')
        if self.autopilot not in MODES:
            raise ValueError(f"autopilot must be one of {sorted(MODES)}, not '{self.autopilot}'")
        self.offboard_mode, self.hold_mode = MODES[self.autopilot]
        self.max_xy, self.max_vz, self.max_yaw = g('max_xy'), g('max_vz'), g('max_yaw_rate')
        self.alt_kp, self.cmd_timeout = g('alt_kp'), g('cmd_timeout')
        self.alt_ki, self.alt_i_max = g('alt_ki'), g('alt_i_max')
        self.alt_i = 0.0
        self.last_tick = None
        self.clamp_on, self.stop_dist, self.slow_dist = g('clamp'), g('stop_dist'), g('slow_dist')
        self.scan_timeout = g('scan_timeout')
        self.scan = None
        self.clamped = False

        self.pose = None            # (x, y, z, yaw) ENU
        self.state = State()
        self.cmd = Twist()
        self.cmd_time = None
        self.alt_target = None
        self.hold = None            # (x, y, z, yaw) while holding position
        self.want_offboard = False
        self.status = ''

        self.sp_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)
        self.state_pub = self.create_publisher(String, '/nav/bridge_state', 10)
        for topic in g('cmd_topics'):
            self.create_subscription(Twist, topic, self.on_cmd, 10)
        self.create_subscription(Float32, '/nav/altitude', self.on_alt, 10)
        # /nav/clamp_suspend (Bool true, refreshed): corridor_nav threading a mapped window / box gap - the clamp saw
        # the box face 0.6 m ahead at 39 deg and held the drone beside it for 3 min (2026-10-21 sim). Lapses by
        # itself 1 s after the last true
        self.suspend_until = None
        self.create_subscription(Bool, '/nav/clamp_suspend', self.on_suspend, 10)
        self.tunnels = []            # corridor boxes (geofence_json): the tunnel ceiling applies there
        self.poly = None             # field_polygon of the ground station's geofence json: THE keep-in when given
        self.fence = self.load_fence()
        # the QR AREA LOCK (user, 2026-10-28: once in the QR area the drone must not leave it until the exit-tunnel
        # sequence): /mission/qr_area {polygon, lock} from the supervisor - locked, the polygon is a keep-in too
        self.lock = None
        self.create_subscription(String, '/mission/qr_area', self.on_qr_area, QoSProfile(
            depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL))
        if self.fence:
            self.get_logger().info(f'geofence: {len(self.fence)} boxes ' +
                                   ' '.join(f'[{a:.1f},{b:.1f}]x[{c:.1f},{d:.1f}]' for a, b, c, d in self.fence))
        self.create_subscription(PoseStamped, '/mavros/local_position/pose', self.on_pose, qos_profile_sensor_data)
        self.create_subscription(State, '/mavros/state', self.on_state, 10)
        self.create_subscription(LaserScan, '/vio/obstacle_scan',
                                 lambda m: setattr(self, 'scan', (self.get_clock().now(), m)), qos_profile_sensor_data)
        self.set_mode = self.create_client(SetMode, '/mavros/set_mode')
        self.create_service(SetBool, '/nav/offboard', self.on_offboard)
        self.create_timer(1.0 / g('rate_hz'), self.tick)

    def on_qr_area(self, m):
        d = json.loads(m.data)
        lock = [tuple(map(float, p)) for p in d['polygon']] if d.get('lock') and len(d.get('polygon') or []) >= 3 else None
        if (lock is None) != (self.lock is None):
            self.get_logger().info('QR area lock ' + ('ON: ' + ' '.join(f'({x:.1f},{y:.1f})' for x, y in lock)
                                                      if lock else 'off'))
        self.lock = lock

    def on_cmd(self, m):
        self.cmd, self.cmd_time = m, self.get_clock().now()

    def on_alt(self, m):
        alt = min(float(m.data), self.get_parameter('max_altitude').value)
        if self.alt_target is None or abs(alt - self.alt_target) > 0.05:
            self.get_logger().info(f'altitude target {alt:.2f} m' +
                                   (f' (asked {m.data:.2f}: capped)' if alt < m.data else ''))
        self.alt_target = alt

    def on_pose(self, m):
        q, pp = m.pose.orientation, m.pose.position
        self.pose = (pp.x, pp.y, pp.z, yaw_of(q))

    def on_state(self, m):
        if not m.connected or not m.mode:
            # MAVROS between heartbeats (connection timed out): no mode known, not a mode change. Taken as one, the
            # offboard request was dropped and, on ArduCopter, the setpoints stopped (2026-10-27 sim, ALIGN_GREEN)
            self.state.armed = m.armed if m.connected else self.state.armed
            return
        if self.state.mode != m.mode and self.state.mode:
            self.get_logger().info(f'{self.autopilot} mode {self.state.mode} -> {m.mode}')
            if self.want_offboard and self.state.mode == self.offboard_mode:
                # left offboard without us asking: RC override or failsafe - don't fight it
                self.get_logger().warn(f'{self.autopilot} left {self.offboard_mode} on its own; offboard request dropped')
                self.want_offboard = False
        self.state = m

    def on_offboard(self, req, res):
        if req.data and (self.pose is None or not self.state.armed):
            res.success, res.message = False, 'need a local position and an armed, airborne drone'
            return res
        self.want_offboard = req.data
        if req.data and self.alt_target is None:
            self.alt_target = min(self.pose[2], self.get_parameter('max_altitude').value)
        self.hold = None
        mode = self.offboard_mode if req.data else self.hold_mode
        if not self.set_mode.service_is_ready():
            res.success, res.message = False, '/mavros/set_mode not available'
            return res
        self.set_mode.call_async(SetMode.Request(custom_mode=mode))
        res.success, res.message = True, f'requested {mode}'
        self.get_logger().info(res.message)
        return res

    def load_fence(self):
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        boxes = []
        flat = list(g('geofence_boxes'))
        if len(flat) >= 4:
            boxes += [tuple(flat[i:i + 4]) for i in range(0, len(flat) - 3, 4)]
        if g('geofence_json'):
            w = json.load(open(g('geofence_json')))
            if isinstance(w.get('course'), dict):     # the dashboard's course (sim worlds: a name): the max height inside, 0.14 m below
                cap = min(g('tunnel_max_altitude'), float(w['course']['tunnel_ceiling']) - 0.14)
                self.set_parameters([rclpy.parameter.Parameter('tunnel_max_altitude', value=cap)])
                self.get_logger().info(f'tunnel ceiling from the course: {cap:.2f} m')
            if len(w.get('field_polygon') or []) >= 3:
                # the ground station's field boundary (6-8 GPS corners -> odom). The corridors can't be recorded on the
                # day: the polygon alone is the keep-in; corridor boxes, if the json has any (sim), still cap the height
                self.poly = [tuple(map(float, p)) for p in w['field_polygon']]
                m = g('geofence_margin')
                for c in w.get('corridors', []):
                    half = c['width'] / 2 - m
                    self.tunnels.append((c['x_from'], c['x_to'], c['centre_y'] - half, c['centre_y'] + half))
                self.get_logger().info(f'geofence: field polygon of {len(self.poly)} corners')
                return [(-1e9, 1e9, -1e9, 1e9)]       # placeholder: inside() uses the polygon
            boxes.append(tuple(g('geofence_start')))
            m = g('geofence_margin')
            for c in w['corridors']:
                half = c['width'] / 2 - m
                boxes.append((c['x_from'], c['x_to'], c['centre_y'] - half, c['centre_y'] + half))
                self.tunnels.append(boxes[-1])
            z = w['delivery_zone']
            boxes.append((z['x_from'], z['x_to'], z['y_from'], z['y_to']))
        return boxes

    def in_poly(self, x, y, poly=None):
        poly = poly or self.poly
        inside, n = False, len(poly)
        for i in range(n):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
            if (y1 > y) != (y2 > y) and x < x1 + (y - y1) * (x2 - x1) / (y2 - y1):
                inside = not inside
        return inside

    def nearest_on_poly(self, x, y, poly=None):
        poly = poly or self.poly
        best, bd = None, 1e18
        n = len(poly)
        for i in range(n):
            (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % n]
            dx, dy = x2 - x1, y2 - y1
            t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / max(dx * dx + dy * dy, 1e-9)))
            px, py = x1 + t * dx, y1 + t * dy
            d = (px - x) ** 2 + (py - y) ** 2
            if d < bd:
                best, bd = (px, py), d
        return best

    def inside(self, x, y, z=None):
        """In an allowed box (or the field polygon); a tunnel box only below tunnel_max_altitude (+0.3 m) when z is given."""
        ceiling = self.get_parameter('tunnel_max_altitude').value + 0.3
        if self.lock is not None and not self.in_poly(x, y, self.lock):
            return False
        if not self.fence:
            return True
        if self.poly is not None:
            return self.in_poly(x, y) and (z is None or z <= ceiling or not any(
                a <= x <= b and c <= y <= d for a, b, c, d in self.tunnels))
        return any(a <= x <= b and c <= y <= d and (z is None or z <= ceiling or box not in self.tunnels)
                   for box in self.fence for a, b, c, d in (box,))

    def in_tunnel(self, x, y):
        k = self.get_parameter('tunnel_margin').value
        return any(a - k <= x <= b + k and c <= y <= d for a, b, c, d in self.tunnels)

    def fenced(self, x, y, vx, vy, z=None):
        """World velocity limited to the geofence (tunnels only below their ceiling)."""
        if not self.fence and self.lock is None:
            return vx, vy
        t = self.get_parameter('geofence_lookahead').value
        if self.inside(x, y):
            if self.inside(x + vx * t, y + vy * t, z):
                return vx, vy
            self.get_logger().warn(f'geofence: command stopped at ({x:.2f}, {y:.2f}, {z if z is not None else 0:.2f})',
                                   throttle_duration_sec=2.0)
            if self.inside(x + vx * t, y, z):
                return vx, 0.0
            if self.inside(x, y + vy * t, z):
                return 0.0, vy
            return 0.0, 0.0
        # outside: back to the nearest allowed point
        if self.lock is not None and not self.in_poly(x, y, self.lock):
            best = self.nearest_on_poly(x, y, self.lock)
        elif self.poly is not None:
            best = self.nearest_on_poly(x, y)
        else:
            best = min(((min(max(x, a), b), min(max(y, c), d)) for a, b, c, d in self.fence),
                       key=lambda q: math.hypot(q[0] - x, q[1] - y))
        dx, dy = best[0] - x, best[1] - y
        n = max(math.hypot(dx, dy), 1e-6)
        self.get_logger().error(f'geofence: OUTSIDE at ({x:.2f}, {y:.2f}): back to ({best[0]:.2f}, {best[1]:.2f})',
                                throttle_duration_sec=2.0)
        return 0.4 * dx / n, 0.4 * dy / n

    def on_suspend(self, m):
        now = self.get_clock().now()
        if m.data and self.suspend_until is None:
            self.get_logger().info('safety clamp suspended (window pass)')
        self.suspend_until = now + rclpy.duration.Duration(seconds=1.0) if m.data else None

    def safe(self, bx, by):
        """Body velocity with the components toward close obstacles removed / scaled down."""
        self.clamped = False
        if self.suspend_until is not None:
            if self.get_clock().now() < self.suspend_until:
                return bx, by
            self.suspend_until = None
            self.get_logger().info('safety clamp back on')
        if not self.clamp_on or self.scan is None:
            return bx, by
        age = (self.get_clock().now() - self.scan[0]).nanoseconds * 1e-9
        if age > self.scan_timeout:
            self.get_logger().warn(f'obstacle scan {age:.1f} s old: no safety clamp', throttle_duration_sec=5.0)
            return bx, by
        m = self.scan[1]
        for i, r in enumerate(m.ranges):
            if not (m.range_min <= r < self.slow_dist):
                continue
            a = m.angle_min + i * m.angle_increment
            dx, dy = math.cos(a), math.sin(a)
            toward = bx * dx + by * dy
            if toward <= 0:
                continue
            keep = max(0.0, min(1.0, (r - self.stop_dist) / (self.slow_dist - self.stop_dist)))
            bx -= (1 - keep) * toward * dx
            by -= (1 - keep) * toward * dy
            self.clamped = True
        return bx, by

    def tick(self):
        if self.pose is None:
            return
        x, y, z, yaw = self.pose
        sp = PositionTarget()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.coordinate_frame = PT.FRAME_LOCAL_NED
        fresh = (self.cmd_time is not None and
                 (self.get_clock().now() - self.cmd_time).nanoseconds * 1e-9 < self.cmd_timeout)
        alt = self.alt_target if self.alt_target is not None else z
        if self.in_tunnel(x, y):                     # 8 ft rule: never climb inside / at the mouth of a tunnel
            alt = min(alt, self.get_parameter('tunnel_max_altitude').value)
        if fresh:
            self.hold = None
            bx, by = self.safe(self.cmd.linear.x, self.cmd.linear.y)
            sp.type_mask = VELOCITY_MASK
            vx, vy = bx * math.cos(yaw) - by * math.sin(yaw), bx * math.sin(yaw) + by * math.cos(yaw)
            vx, vy = self.fenced(x, y, vx, vy, z)
            sp.velocity.x, sp.velocity.y = clamp(vx, self.max_xy), clamp(vy, self.max_xy)
            n = math.hypot(sp.velocity.x, sp.velocity.y)
            if n > self.max_xy:
                sp.velocity.x, sp.velocity.y = sp.velocity.x * self.max_xy / n, sp.velocity.y * self.max_xy / n
            now = self.get_clock().now()
            dt = 0.0 if self.last_tick is None else min(0.2, (now - self.last_tick).nanoseconds * 1e-9)
            self.last_tick = now
            self.alt_i = clamp(self.alt_i + self.alt_ki * (alt - z) * dt, self.alt_i_max)
            sp.velocity.z = clamp(self.alt_kp * (alt - z) + self.alt_i, self.max_vz)
            sp.yaw_rate = clamp(self.cmd.angular.z, self.max_yaw)
            status = 'velocity'
        else:
            if self.hold is None or abs(self.hold[2] - alt) > 1e-3:
                self.hold = (x, y, alt, yaw) if self.hold is None else (*self.hold[:2], alt, self.hold[3])
            sp.type_mask = HOLD_MASK
            self.last_tick = None                    # PX4 holds the position itself: restart the integral
            self.alt_i = 0.0
            sp.position.x, sp.position.y, sp.position.z = self.hold[:3]
            sp.yaw = self.hold[3]
            status = 'hold'
        if self.want_offboard or self.autopilot == 'px4':
            self.sp_pub.publish(sp)
        else:
            status = 'idle'                          # ArduCopter: nothing sent outside offboard
        status = (f'{status}; mode {self.state.mode}{" (offboard requested)" if self.want_offboard else ""}; '
                  f'alt target {alt:.2f}{"; clamped" if fresh and self.clamped else ""}')
        if status.split(';')[0:2] != self.status.split(';')[0:2]:
            self.get_logger().info(status)
        self.status = status
        self.state_pub.publish(String(data=status))


def main():
    spin_node(OffboardBridge)


if __name__ == '__main__':
    main()
