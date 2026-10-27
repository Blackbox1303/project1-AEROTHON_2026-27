"""Feed cuVSLAM to PX4's EKF2 as external-vision BODY-FRAME VELOCITY, through MAVROS.

  in:  /visual_slam/tracking/odometry  (twist = body-frame velocity of imu_link, see cuVSLAM
                                        PoseCache::GetVelocity: relative transform of the cache)
       /visual_slam/status             (vo_state)
       /mavros/local_position/velocity_body  (PX4's own estimate, for the consistency check)
  out: /mavros/odometry/out  frames odom -> base_link; MAVROS converts to PX4's FRD/NED.
       /vio/health (std_msgs/String): "OK" or "REJECT: <reason>"

Velocity only, in the body frame: VIO's world frame starts wherever the drone points, not at
north, and body velocity needs no alignment; it doesn't fight GPS position either (EKF2_EV_CTRL
= 3D velocity). imu_link is the Pixhawk body (real: static identity; sim: PX4's IMU frame).

Sanity gate - nothing is sent while any check fails, then `recover_time` of clean data is
needed before sending again. cuVSLAM's vo_state alone is not enough: after a crash in sim it
kept reporting tracking while its pose was ~175 m off.
  not tracking | stale input | position jump between frames (also catches cuVSLAM restarts,
  which reset the pose) | speed above max_speed | disagreement with PX4's velocity |
  displacement: the step between two VIO frames, in the body frame, vs PX4's body velocity x dt. At the
  corridor exit (walls end, open view) cuVSLAM's pose jumped 1.7 m back and 0.7 m sideways while flying 0.4 m/s
  forward; with 1-2 s between VIO frames the max_speed jump bound (4 m/s x dt) let it through (sim, 2026-10-20)
Divergence: when cuVSLAM loses the image it keeps integrating the IMU and runs away (sim, 2026-10-19:
any turn onto a featureless view -> thousands of m/s, and the state stayed corrupt afterwards, so
'recovering' resumed sending bad data). So a speed / jump / disagreement reject also RESETS cuVSLAM
(auto_reset, at most every reset_interval s); after a reset the jump check restarts. /vio/reset
(std_srvs/Trigger) does the same on request - the mission calls it at a corridor entrance, where the
walls give cuVSLAM something to track, before GPS is lost.
Too little to see: cuVSLAM's own status keeps saying 'tracking' over a featureless view (sim, open field:
0-4 observations per frame, odometry hundreds of m off) and the bad velocities that slip through the other
checks pulled the drone off course EVEN WITH GPS (2026-10-19: 38.7 m flown for a 6 m out-and-back). So
nothing is sent while the frame has fewer than min_observations tracked observations
(/visual_slam/vis/observations_cloud; corridor at 3 m: >= 165, typically ~250).
send:=false runs every check (health, gate, resets) but sends nothing to PX4 (sim_vio px4:=false).
autopilot:=ardupilot (ArduCopter, e.g. indoors without GPS: EK3_SRC1_POSXY/VELXY/YAW = 6 ExternalNav, VISO_TYPE 1;
config/ardupilot_vio_params.parm): the FULL pose + velocity go out, with real covariances (ArduPilot's ODOMETRY handler
takes pos / angle error from pose_covariance [0, 6, 11] / [15, 18, 20]), kept continuous across cuVSLAM resets by
re-anchoring the new track at the last pose sent (MAVROS can't pass ODOMETRY.reset_counter: a jump would look like
real motion to the EKF); and with no GPS the bridge sets the EKF origin (SET_GPS_GLOBAL_ORIGIN via MAVROS
global_position/set_gp_origin) until ArduPilot reports one - without an origin it has no local position at all.
Anchor (sim, anchor_child set): TF anchor_parent -> anchor_child (map -> vio_map), at 10 Hz on /tf (static
transforms didn't cross host <-> container): identity at start, then the drone's PX4 pose (position + yaw) at
each cuVSLAM reset - a reset restarts cuVSLAM's track at its own origin, which a fixed identity drew at the old
origin instead of where the drone was (seen in RViz, 2026-10-19). Display only: PX4 gets body-frame velocity.
Stamps: real drone - VIO's capture time (PC clock), which MAVROS maps to PX4 time; sim
(restamp_now) - current wall time, as MAVROS runs on wall time while VIO is on sim time.
Timing (stale / recover / disagree / observation age) runs on the node clock: sim time with use_sim_time, so
the limits mean the same in a sim far slower than real time (wall time: health read 'no VIO odometry' there
with every frame arriving, 2026-10-20 Jetson sim).
"""
import math

from rclpy.clock import Clock, ClockType
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geographic_msgs.msg import GeoPointStamped
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from tf2_ros import TransformBroadcaster
from isaac_ros_visual_slam_interfaces.msg import VisualSlamStatus
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from std_msgs.msg import String
from std_srvs.srv import Trigger
from isaac_ros_visual_slam_interfaces.srv import Reset

from vio_bringup.spin import spin_node


def norm(v):
    return math.sqrt(sum(c * c for c in v))


class VioToPx4(Node):
    def __init__(self):
        super().__init__('vio_to_px4')
        p = self.declare_parameter
        p('rate_hz', 30.0)
        p('max_speed', 4.0)          # m/s; the drone flies <= 3
        p('max_jump', 0.3)           # m allowed beyond max_speed * dt between VIO frames
        p('max_disagree', 1.0)       # m/s vs PX4's body velocity ...
        p('disagree_time', 0.5)      # ... sustained this long
        p('check_px4', True)
        p('stale_after', 0.2)        # s without VIO odometry
        p('recover_time', 1.0)       # s of clean data before sending again
        p('vel_stddev', 0.1)         # m/s, reported in the twist covariance
        p('restamp_now', False)      # sim: wall-clock stamps for MAVROS
        p('auto_reset', True)        # reset cuVSLAM when it diverges
        p('send', True)              # false: checks only, nothing to PX4
        p('anchor_parent', 'map')
        p('anchor_child', '')        # sim: 'vio_map'; empty = no anchor TF
        p('min_observations', 100)   # tracked observations per frame, below = not trustworthy
        p('obs_stale', 0.5)          # s: no observation count this long = unknown = reject
        p('reset_interval', 3.0)     # s, at most one automatic reset per
        p('max_displacement', 0.4)   # m: VIO step vs PX4 velocity x dt, plus ...
        p('displacement_rate', 0.3)  # m/s x dt of allowance (velocity noise over the gap)
        p('displacement_max_dt', 3.0)  # s: frames further apart aren't compared
        p('autopilot', 'px4')        # px4: body velocity only (EKF2_EV_CTRL); ardupilot: full pose + velocity
        p('pos_stddev', 0.05)        # m, ardupilot pose covariance
        p('ang_stddev_deg', 2.0)     # deg, ardupilot pose covariance
        p('origin_lat', 17.5449)     # EKF origin sent to ArduPilot when it has none (no GPS): any fixed point near
        p('origin_lon', 78.5718)     # the flying site keeps the earth-frame maths sane; default BITS Hyderabad
        p('origin_alt', 540.0)       # m AMSL
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.period = 1.0 / g('rate_hz')
        self.max_speed, self.max_jump = g('max_speed'), g('max_jump')
        self.max_disagree, self.disagree_time = g('max_disagree'), g('disagree_time')
        self.check_px4, self.stale_after = g('check_px4'), g('stale_after')
        self.recover_time, self.restamp = g('recover_time'), g('restamp_now')
        self.vel_var = g('vel_stddev') ** 2
        self.max_disp, self.disp_rate, self.disp_max_dt = g('max_displacement'), g('displacement_rate'), \
            g('displacement_max_dt')
        self.wall = Clock(clock_type=ClockType.SYSTEM_TIME)   # MAVROS stamps (restamp_now)

        self.vo_state = None
        self.px4_vel = None          # (node time, body FLU velocity)
        self.prev = None             # (stamp s, position, yaw) of the previous VIO frame
        self.last_rx = 0.0           # node time of the last VIO frame
        self.last_tx = 0.0
        self.disagree_since = None
        self.good_since = None       # node time the data became clean again
        self.reason = 'waiting for VIO'
        self.sent = self.rejected = self.resets = 0
        self.auto_reset, self.reset_interval = g('auto_reset'), g('reset_interval')
        self.min_obs, self.obs_stale = g('min_observations'), g('obs_stale')
        self.send = g('send')
        self.px4_pose = None
        self.create_subscription(PoseStamped, '/mavros/local_position/pose',
                                 lambda m: setattr(self, 'px4_pose', m.pose), qos_profile_sensor_data)
        self.anchor = TransformStamped()
        self.anchor.header.frame_id, self.anchor.child_frame_id = g('anchor_parent'), g('anchor_child')
        self.anchor.transform.rotation.w = 1.0
        if g('anchor_child'):
            self.tf_pub = TransformBroadcaster(self)
            self.create_timer(0.1, self.publish_anchor)
        self.obs = None              # (node time, observations in the latest frame)
        self.create_subscription(PointCloud2, '/visual_slam/vis/observations_cloud',
                                 lambda m: setattr(self, 'obs', (self.t(), m.width * m.height)),
                                 qos_profile_sensor_data)
        self.last_reset = 0.0
        self.reset_cli = self.create_client(Reset, '/visual_slam/reset')
        self.create_service(Trigger, '/vio/reset', self.on_reset_request)

        self.ap = g('autopilot') == 'ardupilot'
        self.pos_var, self.ang_var = g('pos_stddev') ** 2, math.radians(g('ang_stddev_deg')) ** 2
        self.rebase = False          # next frame after a cuVSLAM reset: re-anchor at the last pose sent
        self.anchor_off = (0.0, 0.0, 0.0, 0.0)   # (dx, dy, dz, dyaw) applied to cuVSLAM's pose (ardupilot)
        self.last_out = None         # (x, y, z, yaw) of the last pose sent
        if self.ap:
            self.origin_ok = False
            self.create_subscription(GeoPointStamped, '/mavros/global_position/gp_origin',
                                     lambda m: setattr(self, 'origin_ok', True), qos_profile_sensor_data)
            self.origin_pub = self.create_publisher(GeoPointStamped, '/mavros/global_position/set_gp_origin', 10)
            self.origin = (g('origin_lat'), g('origin_lon'), g('origin_alt'))
            self.create_timer(2.0, self.set_origin)
        self.pub = self.create_publisher(Odometry, '/mavros/odometry/out', 10)
        self.health_pub = self.create_publisher(String, '/vio/health', 10)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.on_odom,
                                 qos_profile_sensor_data)
        self.create_subscription(VisualSlamStatus, '/visual_slam/status',
                                 lambda m: setattr(self, 'vo_state', m.vo_state), qos_profile_sensor_data)
        self.create_subscription(TwistStamped, '/mavros/local_position/velocity_body', self.on_px4_vel,
                                 qos_profile_sensor_data)
        self.create_timer(0.1, self.watchdog)
        self.create_timer(5.0, self.report)

    def t(self):
        """Node clock in s: sim time with use_sim_time, else wall."""
        return self.get_clock().now().nanoseconds * 1e-9

    def on_px4_vel(self, m):
        v = m.twist.linear
        self.px4_vel = (self.t(), (v.x, v.y, v.z))

    def problem(self, m, stamp, pos, vel, yaw):
        """Reason this frame must not reach PX4, or None."""
        if self.vo_state != 1:
            return f'cuVSLAM not tracking (vo_state {self.vo_state})'
        if self.min_obs > 0:
            if self.obs is None or self.t() - self.obs[0] > self.obs_stale:
                return 'no observation count from cuVSLAM (enable_observations_view?)'
            if self.obs[1] < self.min_obs:
                return f'only {self.obs[1]} observations < {self.min_obs} (too little to see)'
        if norm(vel) > self.max_speed:
            return f'speed {norm(vel):.1f} m/s > {self.max_speed}'
        if self.prev is not None:
            dt = max(stamp - self.prev[0], 1e-3)
            jump = math.dist(pos, self.prev[1])
            if jump > self.max_speed * dt + self.max_jump:
                return f'position jump {jump:.2f} m in {dt * 1000:.0f} ms (reset/restart?)'
            px4_fresh = self.px4_vel and self.t() - self.px4_vel[0] < 0.5
            if self.check_px4 and px4_fresh and dt < self.disp_max_dt:
                # the step since the last frame in the previous frame's body axes vs PX4's body velocity x dt
                dx, dy, dz = (a - b for a, b in zip(pos, self.prev[1]))
                c, s = math.cos(self.prev[2]), math.sin(self.prev[2])
                step = (c * dx + s * dy, -s * dx + c * dy, dz)
                miss = math.dist(step, tuple(v * dt for v in self.px4_vel[1]))
                if miss > self.max_disp + self.disp_rate * dt:
                    return f'displacement {miss:.2f} m off PX4 velocity over {dt:.2f} s'
        if self.check_px4 and self.px4_vel and self.t() - self.px4_vel[0] < 0.5:
            diff = math.dist(vel, self.px4_vel[1])
            if diff > self.max_disagree:
                self.disagree_since = self.disagree_since or self.t()
                if self.t() - self.disagree_since > self.disagree_time:
                    return f'disagrees with PX4 velocity by {diff:.1f} m/s'
            else:
                self.disagree_since = None
        return None

    def on_odom(self, m):
        now = self.t()
        self.last_rx = now
        stamp = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        pp, tw, q = m.pose.pose.position, m.twist.twist.linear, m.pose.pose.orientation
        pos, vel = (pp.x, pp.y, pp.z), (tw.x, tw.y, tw.z)
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        reason = self.problem(m, stamp, pos, vel, yaw)
        self.prev = (stamp, pos, yaw)
        if reason:
            self.set_state(reason)
            self.good_since = None
            self.rejected += 1
            if self.auto_reset and reason.startswith(('speed', 'position jump', 'disagrees', 'displacement')) and \
                    now - self.last_reset > self.reset_interval:
                self.reset_vslam(f'diverged: {reason}')
            return
        self.good_since = self.good_since or now
        if now - self.good_since < self.recover_time:
            self.set_state(f'recovering ({now - self.good_since:.1f}/{self.recover_time:.1f} s clean)')
            return
        self.set_state('OK')
        if not self.send or now - self.last_tx < self.period * 0.9:
            return
        self.last_tx = now
        out = Odometry()
        out.header.stamp = self.wall.now().to_msg() if self.restamp else m.header.stamp
        out.header.frame_id, out.child_frame_id = 'odom', 'base_link'
        out.pose.pose = m.pose.pose
        if self.ap:
            self.fill_pose_ardupilot(out, pos, yaw, m.pose.pose.orientation)
        else:
            out.pose.covariance[0] = -1.0      # pose not meant for fusion (EKF2_EV_CTRL: velocity)
        out.twist.twist = m.twist.twist
        for i in (0, 7, 14):
            out.twist.covariance[i] = self.vel_var
        for i in (21, 28, 35):
            out.twist.covariance[i] = 0.1
        self.pub.publish(out)
        self.sent += 1

    def fill_pose_ardupilot(self, out, pos, yaw, q):
        """Full pose for ArduPilot, continuous across cuVSLAM resets: after a reset the new track starts where the
        last pose sent was (position and yaw), so the EKF sees no jump."""
        if self.rebase and self.last_out is not None:
            lx, ly, lz, lyaw = self.last_out
            dyaw = lyaw - yaw
            c, s = math.cos(dyaw), math.sin(dyaw)
            self.anchor_off = (lx - (c * pos[0] - s * pos[1]), ly - (s * pos[0] + c * pos[1]), lz - pos[2], dyaw)
            self.get_logger().info(f'cuVSLAM reset: re-anchored at ({lx:.2f}, {ly:.2f}, {lz:.2f}), yaw '
                                   f'{math.degrees(lyaw):.0f} deg')
        self.rebase = False
        dx, dy, dz, dyaw = self.anchor_off
        c, s = math.cos(dyaw), math.sin(dyaw)
        x, y, z = c * pos[0] - s * pos[1] + dx, s * pos[0] + c * pos[1] + dy, pos[2] + dz
        # yaw offset applied in the world frame: q_out = q_z(dyaw) * q
        hz, hw = math.sin(dyaw / 2), math.cos(dyaw / 2)
        o = out.pose.pose.orientation
        o.w = hw * q.w - hz * q.z
        o.x = hw * q.x - hz * q.y
        o.y = hw * q.y + hz * q.x
        o.z = hw * q.z + hz * q.w
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = x, y, z
        for i in (0, 7, 14):
            out.pose.covariance[i] = self.pos_var
        for i in (21, 28, 35):
            out.pose.covariance[i] = self.ang_var
        self.last_out = (x, y, z, yaw + dyaw)

    def set_origin(self):
        """Without GPS ArduPilot has no EKF origin, hence no local position: give it one until it reports it."""
        if self.origin_ok:
            return
        m = GeoPointStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.position.latitude, m.position.longitude, m.position.altitude = self.origin
        self.origin_pub.publish(m)

    def reset_vslam(self, why):
        if not self.reset_cli.service_is_ready():
            self.get_logger().warn(f'cannot reset cuVSLAM ({why}): /visual_slam/reset not available')
            return False
        self.reset_cli.call_async(Reset.Request())
        self.last_reset = self.t()
        if self.px4_pose is not None:                # cuVSLAM's new origin = where the drone is now
            p, q = self.px4_pose.position, self.px4_pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            t = self.anchor.transform
            t.translation.x, t.translation.y, t.translation.z = p.x, p.y, p.z
            t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w = 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)
        self.prev, self.good_since, self.disagree_since = None, None, None
        self.rebase = True                           # ardupilot: keep the pose sent continuous
        self.resets += 1
        self.set_state(f'cuVSLAM reset ({why})')
        return True

    def publish_anchor(self):
        self.anchor.header.stamp = self.get_clock().now().to_msg()
        self.tf_pub.sendTransform(self.anchor)

    def on_reset_request(self, req, res):
        res.success = self.reset_vslam('requested')
        res.message = 'cuVSLAM reset' if res.success else '/visual_slam/reset not available'
        return res

    def watchdog(self):
        if self.last_rx and self.t() - self.last_rx > self.stale_after:
            self.set_state(f'no VIO odometry for {self.t() - self.last_rx:.1f} s')
            self.good_since = None
        self.health_pub.publish(String(data=self.reason if self.reason.startswith(('OK', 'recovering'))
                                       else 'REJECT: ' + self.reason))

    def set_state(self, reason):
        if reason != self.reason and not (reason.startswith('recovering') and self.reason.startswith('recovering')):
            # separate call sites: rclpy refuses one call site logging at two severities
            if reason == 'OK':
                self.get_logger().info('VIO -> PX4: OK, sending' if self.send else 'VIO OK (send off: nothing to PX4)')
            else:
                self.get_logger().warn(f'VIO -> PX4: {reason}')
        self.reason = reason

    def report(self):
        self.get_logger().info(f'sent {self.sent}, rejected {self.rejected}, resets {self.resets}, state: {self.reason}')


def main():
    spin_node(VioToPx4)


if __name__ == '__main__':
    main()
