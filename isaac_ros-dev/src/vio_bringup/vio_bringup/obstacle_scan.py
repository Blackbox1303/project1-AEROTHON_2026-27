"""Turn the D435 depth image into a horizontal obstacle scan for the tunnel.

Depth pixels are back-projected, rotated into the gravity-aligned body frame
(using the Pixhawk attitude so the scan stays horizontal when the drone
pitches/rolls), and only points within +-`band` of flight height are kept.
For each bearing bin the k-th nearest point is the obstacle range (k > 1
rejects stereo speckle). Published as /vio/obstacle_scan (sensor_msgs/LaserScan,
frame `scan_frame`: body yaw, levelled), the format PX4 Collision Prevention
takes via /mavros/obstacle/send.

Obstacles are deliberately not classified (wall / pillar / anything else):
avoidance only needs free distance per bearing.

For viewing, it also publishes /vio/depth_points: a dense PointCloud2 of the whole depth
image (every `cloud_stride`-th pixel, all heights) in the body frame, at `cloud_rate_hz`.
"""
import math

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CameraInfo, Image, Imu, LaserScan, PointCloud2, PointField
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

from vio_bringup.spin import spin_node


def quat_to_rot(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class ObstacleScan(Node):
    def __init__(self):
        super().__init__('obstacle_scan')
        p = self.declare_parameter
        p('depth_topic', '/camera/depth/image_rect_raw')
        p('info_topic', '/camera/depth/camera_info')
        p('body_frame', 'imu_link')
        p('scan_frame', 'scan_level')
        p('stride', 4)               # use every n-th pixel in u and v
        p('min_range', 0.2)          # m; D435 min-Z at 640x360 is ~0.2 m
        p('max_range', 6.0)          # m
        p('band', 0.25)              # m above/below body origin that counts as "at flight height"
        p('fov_deg', 90.0)
        p('bin_deg', 1.0)
        p('kth', 3)                  # k-th nearest point per bin
        p('min_points', 3)
        p('max_rate_hz', 30.0)       # the planner needs ~20 Hz; skip the rest to save CPU
        p('cloud_stride', 2)         # dense view cloud: every 2nd pixel = ~57k points at 640x360
        p('cloud_rate_hz', 10.0)     # 0 disables the cloud
        p('cloud_max_range', 9.0)    # m: the cloud's own limit (D435 sim clips at 10). entry_align fits the corridor
        #                              walls in it from outside: at max_range (6) only ~0.5 m of wall was in view
        #                              from the standoff at 2.5 m height (10 ft rule) and the fit failed (2026-10-21)
        p('edge_ratio', 0.1)         # cloud: drop pixels whose depth differs from a neighbour by more than this
        #                              fraction ("flying pixels" at depth edges filled window openings with
        #                              phantom points, 2026-10-19); 0 = keep all
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.body_frame, self.scan_frame = g('body_frame'), g('scan_frame')
        self.stride, self.min_r, self.max_r = g('stride'), g('min_range'), g('max_range')
        self.band, self.kth, self.min_pts = g('band'), g('kth'), g('min_points')
        self.fov, self.bin = math.radians(g('fov_deg')), math.radians(g('bin_deg'))
        self.nbins = int(round(self.fov / self.bin))
        self.min_period = 1.0 / g('max_rate_hz')
        self.last_stamp = 0.0
        self.cloud_stride, self.cloud_max_r = g('cloud_stride'), g('cloud_max_range')
        self.cloud_period = 1.0 / g('cloud_rate_hz') if g('cloud_rate_hz') > 0 else None
        self.last_cloud = 0.0
        self.cloud_rays = None
        self.edge_ratio = self.get_parameter('edge_ratio').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_pub = TransformBroadcaster(self)
        self.rays = None
        self.r_body_opt = None
        self.t_body_opt = None
        self.r_level = np.eye(3)     # body -> level frame (roll/pitch removed)
        self.depth_scale = 0.001     # D4xx default: depth in mm

        self.scan_pub = self.create_publisher(LaserScan, '/vio/obstacle_scan', 10)
        self.cloud_pub = self.create_publisher(PointCloud2, '/vio/depth_points', qos_profile_sensor_data)
        self.create_subscription(CameraInfo, g('info_topic'), self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, g('depth_topic'), self.on_depth, qos_profile_sensor_data)
        self.create_subscription(Imu, '/mavros/imu/data', self.on_attitude, qos_profile_sensor_data)

    @staticmethod
    def unit_rays(info, stride):
        """Unit-depth rays in the optical frame (x right, y down, z forward) for every stride-th pixel."""
        fx, fy, cx, cy = info.k[0], info.k[4], info.k[2], info.k[5]
        v, u = np.mgrid[0:info.height:stride, 0:info.width:stride]
        return np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u, float)], -1).reshape(-1, 3)

    def on_info(self, msg):
        if self.rays is not None:
            return
        self.rays = self.unit_rays(msg, self.stride)
        self.cloud_rays = self.unit_rays(msg, self.cloud_stride)
        self.frame_id = msg.header.frame_id

    def on_attitude(self, msg):
        # MAVROS attitude is ENU world -> FLU body; keep roll/pitch only
        q = msg.orientation
        r = quat_to_rot(q.x, q.y, q.z, q.w)
        yaw = math.atan2(r[1, 0], r[0, 0])
        c, s = math.cos(-yaw), math.sin(-yaw)
        self.r_level = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]) @ r

    def lookup_extrinsics(self):
        try:
            t = self.tf_buffer.lookup_transform(self.body_frame, self.frame_id, rclpy.time.Time(),
                                                timeout=Duration(seconds=0.0))
        except Exception:  # TF not up yet
            return False
        q, p = t.transform.rotation, t.transform.translation
        self.r_body_opt = quat_to_rot(q.x, q.y, q.z, q.w)
        self.t_body_opt = np.array([p.x, p.y, p.z])
        return True

    def on_depth(self, msg):
        if self.rays is None or (self.r_body_opt is None and not self.lookup_extrinsics()):
            return
        now_msg = self.get_clock().now().to_msg()
        now_s = now_msg.sec + now_msg.nanosec * 1e-9
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if abs(now_s - t) > 0.3:
            # RealSense HARDWARE_CLOCK domain: restamp to system clock so TF and slam_toolbox align
            msg.header.stamp = now_msg
            t = now_s
        if t - self.last_stamp < self.min_period * 0.9:
            return
        self.last_stamp = t
        if msg.encoding == '32FC1':   # Gazebo: metres as float
            depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
            scale = 1.0
        else:                          # D435 (16UC1 / mono16): millimetres
            depth = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width)
            scale = self.depth_scale
        if self.cloud_period is not None and t - self.last_cloud >= self.cloud_period * 0.9:
            self.last_cloud = t
            self.publish_cloud(depth, scale, msg.header)
        z = depth[::self.stride, ::self.stride].reshape(-1).astype(np.float32) * scale
        z = np.nan_to_num(z, nan=0.0, posinf=0.0)  # gz marks no-return pixels as inf/nan
        ok = (z > self.min_r) & (z < self.max_r)
        pts = self.rays[ok] * z[ok, None]
        # optical -> body -> level (gravity-aligned, yaw of body)
        pts = (pts @ self.r_body_opt.T + self.t_body_opt) @ self.r_level.T
        pts = pts[np.abs(pts[:, 2]) < self.band]

        rng = np.hypot(pts[:, 0], pts[:, 1])
        ang = np.arctan2(pts[:, 1], pts[:, 0])
        b = np.floor((ang + self.fov / 2) / self.bin).astype(int)
        keep = (b >= 0) & (b < self.nbins)
        b, rng = b[keep], rng[keep]

        ranges = np.full(self.nbins, np.inf, np.float32)
        if b.size:
            order = np.lexsort((rng, b))
            b, rng = b[order], rng[order]
            bins, first, counts = np.unique(b, return_index=True, return_counts=True)
            good = counts >= self.min_pts
            k = np.minimum(self.kth - 1, counts - 1)
            ranges[bins[good]] = rng[(first + k)[good]]

        stamp = msg.header.stamp
        self.publish_level_frame(stamp)
        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = stamp, self.scan_frame
        scan.angle_min, scan.angle_max = -self.fov / 2, -self.fov / 2 + self.bin * (self.nbins - 1)
        scan.angle_increment = self.bin
        scan.range_min, scan.range_max = self.min_r, self.max_r
        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)

    def publish_cloud(self, depth, scale, header):
        s = self.cloud_stride
        img = np.nan_to_num(depth[::s, ::s].astype(np.float32) * scale, nan=0.0, posinf=0.0)
        ok2 = (img > self.min_r) & (img < self.cloud_max_r)
        if self.edge_ratio > 0:
            jump = np.zeros(img.shape, bool)             # depth jump to any 4-neighbour (or to no data)
            for a, b in ((img[1:, :], img[:-1, :]), (img[:, 1:], img[:, :-1])):
                j = np.abs(a - b) > self.edge_ratio * np.minimum(a, b).clip(min=1e-3)
                if a.shape[0] < img.shape[0]:
                    jump[1:, :] |= j
                    jump[:-1, :] |= j
                else:
                    jump[:, 1:] |= j
                    jump[:, :-1] |= j
            ok2 &= ~jump
        z, ok = img.reshape(-1), ok2.reshape(-1)
        pts = (self.cloud_rays[ok] * z[ok, None]) @ self.r_body_opt.T + self.t_body_opt
        cloud = PointCloud2()
        cloud.header.stamp, cloud.header.frame_id = header.stamp, self.body_frame
        cloud.height, cloud.width = 1, len(pts)
        cloud.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
                        for i, n in enumerate('xyz')]
        cloud.is_bigendian, cloud.is_dense = False, True
        cloud.point_step, cloud.row_step = 12, 12 * len(pts)
        cloud.data = pts.astype(np.float32).tobytes()
        self.cloud_pub.publish(cloud)

    def publish_level_frame(self, stamp):
        """scan_frame = body origin, yaw of body, roll/pitch removed."""
        q = self.rot_to_quat(self.r_level.T)
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = stamp, self.body_frame, self.scan_frame
        t.transform.rotation.x, t.transform.rotation.y, t.transform.rotation.z, t.transform.rotation.w = q
        self.tf_pub.sendTransform(t)

    @staticmethod
    def rot_to_quat(r):
        w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
        x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
        y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
        z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
        n = math.sqrt(x * x + y * y + z * z + w * w)
        return x / n, y / n, z / n, w / n


def main():
    spin_node(ObstacleScan)


if __name__ == '__main__':
    main()
