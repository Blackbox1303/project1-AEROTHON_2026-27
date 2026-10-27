"""PX4 / VIO / RTAB-Map position estimate as the navigation frame tree (odom -> base_link), plus live
SITL / Hand-Follow Command Arrow (/nav/next_pose, /nav/cmd_arrow, /nav/sitl_command).

  in:  /mavros/local_position/odom   (PX4 EKF2)
       /rtabmap/odom                 (RTAB-Map RGB-D odometry when moving RealSense by hand)
       /visual_slam/tracking/odometry (cuVSLAM stereo VIO)
       /cmd_vel, /nav/altitude, /nav/lateral, /vio/obstacle_scan (for live Hand-Follow SITL Command arrow)
  source (param): auto = PX4 when it publishes, else VIO / RTAB-Map; vio = VIO / RTAB-Map only (PX4 shown on the
                  dashboard for comparison, e.g. walking the drone by hand); px4 = PX4 only
  out: TF odom -> base_link (at 50 Hz so slam_toolbox and RViz2 never drop a scan or cloud)
       /nav/odom (nav_msgs/Odometry, RELIABLE)
       /nav/next_pose (geometry_msgs/PoseStamped, where to move the camera/drone next)
       /nav/cmd_arrow (visualization_msgs/Marker, 3D arrow in RViz2 showing commanded velocity & direction)
       /nav/sitl_command (std_msgs/String JSON, for the dashboard's Hand-Follow HUD & map arrow)
"""
import json
import math
import time

import numpy as np
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point, PoseStamped, TransformStamped, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32, String
from tf2_ros import TransformBroadcaster
from visualization_msgs.msg import Marker

from vio_bringup.spin import spin_node


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


def quat_to_yaw(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class OdomTf(Node):
    def __init__(self):
        super().__init__('odom_tf')
        self.declare_parameter('restamp', False)
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('fallback_rate', 0.0)
        self.declare_parameter('source', 'auto')
        self.source = self.get_parameter('source').value
        self.restamp = self.get_parameter('restamp').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.last_px4 = 0.0
        self.last_vio = 0.0
        self.last_odom_msg = Odometry()
        self.last_odom_msg.pose.pose.position.z = 1.2
        self.last_odom_msg.pose.pose.orientation.w = 1.0

        self.cmd_vel = None
        self.cmd_vel_t = 0.0
        self.target_alt = 1.2
        self.target_lat = 0.0
        self.scan_ranges = None
        self.scan_angle_min = -math.pi / 4
        self.scan_angle_inc = math.radians(1.0)

        self.tf = TransformBroadcaster(self)
        self.pub = self.create_publisher(Odometry, '/nav/odom', 10)
        self.next_pose_pub = self.create_publisher(PoseStamped, '/nav/next_pose', 10)
        self.arrow_pub = self.create_publisher(Marker, '/nav/cmd_arrow', 10)
        self.sitl_pub = self.create_publisher(String, '/nav/sitl_command', 10)

        self.create_subscription(Odometry, '/mavros/local_position/odom', self.on_odom, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/visual_slam/tracking/odometry', self.on_vio, qos_profile_sensor_data)
        self.create_subscription(Odometry, '/rtabmap/odom', self.on_vio, qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(Float32, '/nav/altitude', self.on_alt, 10)
        self.create_subscription(Float32, '/nav/lateral', self.on_lat, 10)
        self.create_subscription(LaserScan, '/vio/obstacle_scan', self.on_scan, qos_profile_sensor_data)

        rate = float(self.get_parameter('fallback_rate').value)
        if rate > 0.0:
            # Broadcast at 50 Hz so TF cache is always ahead of incoming camera/scan frames
            self.create_timer(0.02, self.on_tick_tf)
            self.create_timer(0.10, self.publish_hand_guidance)

    def publish_odom(self, m, use_now=False):
        stamp = self.get_clock().now().to_msg() if (self.restamp or use_now) else m.header.stamp
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.odom_frame, self.base_frame
        p = m.pose.pose.position
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = p.x, p.y, p.z
        t.transform.rotation = m.pose.pose.orientation
        self.tf.sendTransform(t)
        out = Odometry()
        out.header.stamp, out.header.frame_id, out.child_frame_id = stamp, self.odom_frame, self.base_frame
        out.pose = m.pose
        out.twist = m.twist
        self.pub.publish(out)

    def on_odom(self, m):
        if self.source == 'vio':
            return
        self.last_px4 = time.monotonic()
        self.last_odom_msg = m
        self.publish_odom(m)

    def on_vio(self, m):
        now = time.monotonic()
        if self.source == 'px4' or now - self.last_px4 < 1.0:
            return
        p = m.pose.pose.position
        # Ignore invalid NaN poses from visual odometry lost frames
        if not (math.isfinite(p.x) and math.isfinite(p.y) and math.isfinite(p.z)):
            return
        self.last_vio = now
        self.last_odom_msg = m
        self.publish_odom(m, use_now=True)

    def on_tick_tf(self):
        now = time.monotonic()
        if now - self.last_px4 < 0.1:
            return
        # Always keep TF odom -> base_link fresh at 50 Hz using the latest known pose
        self.publish_odom(self.last_odom_msg, use_now=True)

    def on_cmd_vel(self, m):
        self.cmd_vel = m
        self.cmd_vel_t = time.monotonic()

    def on_alt(self, m):
        if math.isfinite(m.data):
            self.target_alt = float(m.data)

    def on_lat(self, m):
        if math.isfinite(m.data):
            self.target_lat = float(m.data)

    def on_scan(self, m):
        self.scan_ranges = np.array(m.ranges, np.float32)
        self.scan_angle_min = m.angle_min
        self.scan_angle_inc = m.angle_increment

    def publish_hand_guidance(self):
        """Compute SITL / Hand-Follow velocity & next-position arrow so the user can follow it by hand."""
        now = time.monotonic()
        p = self.last_odom_msg.pose.pose.position
        q = self.last_odom_msg.pose.pose.orientation
        x, y, z = float(p.x), float(p.y), float(p.z)
        yaw = quat_to_yaw(q)

        # 1. If corridor_nav / entry_align / offboard_bridge is actively publishing /cmd_vel, use it:
        if self.cmd_vel is not None and (now - self.cmd_vel_t) < 0.8:
            bx = float(self.cmd_vel.linear.x)
            by = float(self.cmd_vel.linear.y)
            wz = float(self.cmd_vel.angular.z)
            dz = float(np.clip(self.target_alt - z, -0.5, 0.5))
            source = 'corridor_nav (/cmd_vel)'
        else:
            # 2. Live reactive obstacle-avoidance guidance from the D435 LaserScan (/vio/obstacle_scan)
            source = 'live D435 obstacle avoidance'
            bx, by, wz, dz = 0.50, 0.0, 0.0, 0.0
            if self.scan_ranges is not None and len(self.scan_ranges):
                n = len(self.scan_ranges)
                angles = self.scan_angle_min + np.arange(n) * self.scan_angle_inc
                r = np.where(np.isfinite(self.scan_ranges) & (self.scan_ranges > 0.18), self.scan_ranges, 6.0)
                mid = r[np.abs(angles) < math.radians(18.0)]
                front_min = float(np.min(mid)) if len(mid) else 6.0

                if front_min < 0.55:
                    # Too close ahead -> step back / stop and steer toward the open side
                    left_mean = float(np.mean(r[angles > 0.1])) if np.any(angles > 0.1) else 3.0
                    right_mean = float(np.mean(r[angles < -0.1])) if np.any(angles < -0.1) else 3.0
                    bx = -0.15
                    by = 0.30 if left_mean >= right_mean else -0.30
                    wz = 0.45 if left_mean >= right_mean else -0.45
                elif front_min < 1.8:
                    # Smoothly steer toward the widest gap in the 90 deg FOV
                    kernel = np.ones(9, np.float32) / 9.0
                    smooth_r = np.convolve(r, kernel, mode='same')
                    best_i = int(np.argmax(smooth_r))
                    best_ang = float(angles[best_i])
                    spd = 0.45 * min(1.0, max(0.3, (front_min - 0.4) / 1.4))
                    bx = spd * math.cos(best_ang)
                    by = spd * math.sin(best_ang)
                    wz = float(np.clip(best_ang * 1.2, -0.8, 0.8))
                else:
                    # Center between walls if both left and right walls are visible
                    left_r = float(np.min(r[angles > math.radians(20)])) if np.any(angles > math.radians(20)) else 6.0
                    right_r = float(np.min(r[angles < -math.radians(20)])) if np.any(angles < -math.radians(20)) else 6.0
                    if left_r < 3.0 and right_r < 3.0:
                        by = float(np.clip((left_r - right_r) * 0.25, -0.25, 0.25))

        # Body FLU velocity -> World odom velocity & Next Target Pose (1.2 s lookahead)
        c, s = math.cos(yaw), math.sin(yaw)
        vx_w = c * bx - s * by
        vy_w = s * bx + c * by
        lookahead_t = 1.5
        nx = x + vx_w * lookahead_t
        ny = y + vy_w * lookahead_t
        nz = z + dz
        nyaw = yaw + wz * 1.0
        steer_deg = math.degrees(math.atan2(by, max(abs(bx), 0.05))) if (abs(bx) > 0.02 or abs(by) > 0.02) else 0.0

        if bx < -0.05:
            action = f'🛑 TOO CLOSE — STEP BACK & {"LEFT" if by >= 0 else "RIGHT"}'
        elif abs(by) > 0.12:
            side = 'LEFT ↖' if by > 0 else 'RIGHT ↗'
            action = f'STEER {side} ({abs(steer_deg):.0f}°) · fwd {bx:.2f} m/s, lat {by:+.2f} m/s'
        elif abs(dz) > 0.15:
            action = f'{"⬆ MOVE UP" if dz > 0 else "⬇ MOVE DOWN"} ({dz:+.2f} m) · fwd {bx:.2f} m/s'
        else:
            action = f'⬆ GO STRAIGHT ({bx:.2f} m/s)'

        stamp = self.get_clock().now().to_msg()

        # Publish /nav/next_pose for RViz2
        ps = PoseStamped()
        ps.header.stamp, ps.header.frame_id = stamp, self.odom_frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = nx, ny, nz
        qx, qy, qz, qw = euler_to_quat(0.0, 0.0, nyaw)
        ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w = qx, qy, qz, qw
        self.next_pose_pub.publish(ps)

        # Publish 3D Arrow Marker (/nav/cmd_arrow) in base_link frame for RViz2
        mk = Marker()
        mk.header.stamp, mk.header.frame_id = stamp, self.base_frame
        mk.ns, mk.id = 'hand_follow_cmd', 0
        mk.type = Marker.ARROW
        mk.action = Marker.ADD
        p0 = Point(x=0.0, y=0.0, z=0.0)
        p1 = Point(x=float(bx * 1.6), y=float(by * 1.6), z=float(dz * 1.6))
        if math.hypot(p1.x, p1.y) < 0.15:
            p1.x = 0.25
        mk.points = [p0, p1]
        mk.scale.x = 0.06   # shaft diameter
        mk.scale.y = 0.12   # head diameter
        mk.scale.z = 0.15   # head length
        mk.color.r = 1.0 if bx < 0 else (1.0 if abs(by) > 0.12 else 0.1)
        mk.color.g = 0.2 if bx < 0 else 0.95
        mk.color.b = 0.2 if bx < 0 else 1.0
        mk.color.a = 0.95
        self.arrow_pub.publish(mk)

        self.sitl_pub.publish(String(data=json.dumps({
            'source': source,
            'action': action,
            'bx': round(bx, 2),
            'by': round(by, 2),
            'dz': round(dz, 2),
            'wz': round(wz, 2),
            'steer_deg': round(steer_deg, 1),
            'next_x': round(nx, 2),
            'next_y': round(ny, 2),
            'next_z': round(nz, 2),
            'next_yaw': round(nyaw, 3),
        })))


def main():
    spin_node(OdomTf)


if __name__ == '__main__':
    main()
