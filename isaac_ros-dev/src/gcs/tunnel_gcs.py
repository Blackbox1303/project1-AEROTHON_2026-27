#!/usr/bin/env python3
"""AeroTHON Tunnel + SLAM Dashboard: dedicated web UI for the Intel RealSense D435/D435i tunnel navigation & SLAM stack.

Runs on the Jetson host (not auto-started at boot):
    ~/workspaces/isaac_ros-dev/src/gcs/run_tunnel_gcs.sh        (dashboard only -> http://10.42.0.247:8093)
or launch both the dashboard and the RealSense + SLAM + tunnel stack in one command:
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_slam.launch.py

Open in a browser:
    http://10.42.0.247:8093/   (from the Lenovo laptop)
    http://127.0.0.1:8093/     (on the Jetson localhost)
"""
import argparse
import asyncio
import json
import math
import os
import shutil
import signal
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from aiohttp import web

# cv2 (pip opencv-python) sets QT_QPA_PLATFORM_PLUGIN_PATH to its own incomplete plugins dir, which breaks
# Qt5 GUI apps (rviz2, rtabmap_viz) spawned by this server on the Jetson display (:0). Strip it immediately:
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ.pop('QT_QPA_FONTDIR', None)
os.environ.setdefault('DISPLAY', ':0')

WS = Path.home() / 'workspaces/isaac_ros-dev'
DATA = WS / 'runs/gcs'
HERE = Path(__file__).resolve().parent
LAUNCH_ALL = WS / 'src/launch_all'

CAMERAS = {
    'depth': '/camera/depth/image_rect_raw',
    'color': '/camera/color/image_raw',
    'banner': '/nav/banner/debug',
    'infra1': '/camera/infra1/image_rect_raw',
}


class State:
    def __init__(self):
        self.lock = threading.Lock()
        self.ros_time = {}
        self.map_png = None
        self.map_meta = None
        self.map_ver = 0
        self.raw_grid = None         # (grid int8 HxW, ox, oy, res)
        self.accum_pts = np.zeros((0, 3), np.float32)
        self.start_pose_scan = None
        self.planned_path = []       # [(x, y, z), ...] from sim_saved_map_drone
        self.sim_goal = None         # [x, y]
        self.pose_odom = None        # (x, y, z, yaw) in odom
        self.pose_map = None         # (x, y, z, yaw) in map (SLAM-corrected)
        self.slam_corr = (0.0, 0.0, 0.0)  # (dx, dy, dyaw_deg) map -> odom
        self.track = []
        self.scan = None             # {'angle_min', 'angle_inc', 'range_max', 'ranges', 'min_dist', 'front_dist'}
        self.cloud_pts = 0
        self.corridor_state = 'IDLE'
        self.entry_state = 'IDLE'
        self.entry_result = None
        self.banner_det = None
        self.alt_state = None
        self.alt_target = None
        self.lat_target = None
        self.windows = []
        self.vio_health = ''
        self.vio_pose = None
        # PX4 position estimate (walk-test): raw ENU local pose, the VIO -> PX4 input, and PX4 drawn in the odom
        # frame. The two frames don't share an origin or heading (PX4: north + EKF origin, VIO: where the camera
        # started), so the first sample with both (or Align) pins PX4's pose + yaw onto the odom pose.
        self.px4_pose = None         # (x, y, z, yaw) PX4 local ENU
        self.px4_vel = None          # (vx, vy, vz) PX4 local ENU
        self.px4_align = None        # (px4 x, y, yaw, odom x, y, yaw) at alignment
        self.px4_odom = None         # (x, y, z, yaw) PX4 in the odom frame
        self.px4_track = []          # [(x, y), ...] PX4 in the map frame
        self.ev_times = []           # monotonic stamps of /mavros/odometry/out (VIO -> PX4 EKF2)
        self.ev_vel = None           # last velocity sent to PX4 (body FLU)
        self.px4_params = {}
        self.grids = {}              # topic -> (grid int8 HxW, ox, oy, res, monotonic stamp)
        self.mavros = {'connected': False, 'armed': False, 'mode': ''}
        self.corridor_params = {
            'entry_x': 0.0, 'entry_y': 0.0, 'entry_yaw': 0.0,
            'altitude': 1.5, 'corridor_speed': 0.5, 'half_width': 1.675, 'max_length': 15.0,
        }


S = State()


class Ros(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.ready = threading.Event()
        self.camera_name = 'depth'
        self.jpeg = None
        self.jpeg_time = 0.0
        self.scan_band = 0.20
        self.last_accum = 0.0

    def run(self):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
        from mavros_msgs.msg import State as MavState
        from mavros_msgs.srv import CommandBool, CommandLong, SetMode
        from geometry_msgs.msg import PoseStamped, TwistStamped
        from nav_msgs.msg import OccupancyGrid, Odometry
        from rcl_interfaces.srv import GetParameters
        from std_srvs.srv import Empty
        from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
        from rcl_interfaces.srv import SetParameters
        from sensor_msgs.msg import Image, LaserScan, PointCloud2
        from std_msgs.msg import Float32, String
        from std_srvs.srv import Trigger
        from tf2_ros import Buffer, TransformListener

        from rclpy.signals import SignalHandlerOptions
        # rclpy's own SIGINT handler would only stop ROS and leave the web server running (Ctrl-C / launch stop
        # hung on this process): let aiohttp's run_app get SIGINT and exit the whole dashboard
        rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        n = self.node = Node('tunnel_slam_dashboard')
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.Image = Image
        self.String = String
        self.Trigger = Trigger
        self.SetParameters = SetParameters
        self.GetParameters = GetParameters
        self.Empty = Empty
        self.CommandLong = CommandLong
        self.SetMode = SetMode
        self.CommandBool = CommandBool
        self.Parameter = Parameter
        self.ParameterType = ParameterType
        self.ParameterValue = ParameterValue
        self.qos_img = qos_profile_sensor_data
        self.tf = Buffer()
        TransformListener(self.tf, n)

        def stamp(k):
            S.ros_time[k] = time.monotonic()

        def on_odom(m):
            p = m.pose.pose.position
            q = m.pose.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            pose_o = (round(p.x, 3), round(p.y, 3), round(p.z, 3), round(yaw, 4))
            pose_m = pose_o
            corr = S.slam_corr
            try:
                t = self.tf.lookup_transform('map', 'odom', rclpy.time.Time())
                tx, ty = t.transform.translation.x, t.transform.translation.y
                tq = t.transform.rotation
                tyaw = math.atan2(2 * (tq.w * tq.z + tq.x * tq.y), 1 - 2 * (tq.y * tq.y + tq.z * tq.z))
                c, s = math.cos(tyaw), math.sin(tyaw)
                mx = c * p.x - s * p.y + tx
                my = s * p.x + c * p.y + ty
                pose_m = (round(mx, 3), round(my, 3), round(p.z, 3), round(yaw + tyaw, 4))
                corr = (round(tx, 3), round(ty, 3), round(math.degrees(tyaw), 2))
                stamp('tf_map')
            except Exception:
                pass
            with S.lock:
                stamp('odom')
                S.pose_odom = pose_o
                S.pose_map = pose_m
                S.slam_corr = corr
                tr = S.track
                if not tr or math.dist(tr[-1], (pose_m[0], pose_m[1])) > 0.10:
                    S.track = (tr + [(round(pose_m[0], 2), round(pose_m[1], 2))])[-2000:]

        def on_scan(m):
            rngs = [round(r, 2) if (math.isfinite(r) and r >= m.range_min) else None for r in m.ranges]
            valid = [r for r in rngs if r is not None]
            mid = len(rngs) // 2
            front = [r for r in rngs[max(0, mid - 10):min(len(rngs), mid + 11)] if r is not None]
            with S.lock:
                stamp('scan')
                S.scan = {
                    'angle_min': round(m.angle_min, 4),
                    'angle_inc': round(m.angle_increment, 4),
                    'range_max': round(m.range_max, 2),
                    'ranges': rngs,
                    'min_dist': round(min(valid), 2) if valid else None,
                    'front_dist': round(min(front), 2) if front else None,
                }

        def on_cloud(m):
            with S.lock:
                stamp('cloud')
                S.cloud_pts = m.width * m.height
            # Accumulate voxel-downsampled 3D obstacle points in map frame for saving; at most 2 Hz: the voxel
            # dedupe over up to 35k points on every cloud was a large part of this server's CPU (2026-10-27)
            now = time.monotonic()
            if now - self.last_accum < 0.5:
                return
            self.last_accum = now
            if m.point_step >= 12 and m.width * m.height > 0:
                try:
                    raw = np.frombuffer(m.data, np.float32).reshape(-1, m.point_step // 4)[:, :3]
                    pts = raw[::8]
                    pm = S.pose_map or S.pose_odom
                    if pm is not None and len(pts):
                        px, py, pz, pyaw = pm
                        c, s = math.cos(pyaw), math.sin(pyaw)
                        wx = c * pts[:, 0] - s * pts[:, 1] + px
                        wy = s * pts[:, 0] + c * pts[:, 1] + py
                        wz = pts[:, 2] + pz
                        wpts = np.column_stack([wx, wy, wz]).astype(np.float32)
                        wpts = wpts[(wpts[:, 2] > 0.12) & (wpts[:, 2] < 2.6)]
                        if len(wpts):
                            with S.lock:
                                merged = np.vstack([S.accum_pts, wpts]) if len(S.accum_pts) else wpts
                                vox = np.round(merged / 0.08).astype(np.int32)
                                _, idx = np.unique(vox, axis=0, return_index=True)
                                S.accum_pts = merged[np.sort(idx)][-35000:]
                except Exception:
                    pass

        def on_rtab_cloud(m):
            if m.point_step >= 12 and m.width * m.height > 0:
                try:
                    n_f = m.point_step // 4
                    pts = np.frombuffer(m.data, np.float32).reshape(-1, n_f)[:, :3]
                    pts = pts[np.isfinite(pts).all(axis=1)]
                    if len(pts):
                        with S.lock:
                            stamp('rtab_cloud')
                            S.accum_pts = pts[-35000:].astype(np.float32)
                except Exception:
                    pass

        def on_sim_path(m):
            with S.lock:
                try:
                    d = json.loads(m.data)
                    S.planned_path = d.get('waypoints', [])
                    S.sim_goal = d.get('goal')
                except ValueError:
                    pass

        def on_corridor(m):
            with S.lock:
                stamp('corridor')
                S.corridor_state = m.data

        def on_entry(m):
            with S.lock:
                stamp('entry')
                S.entry_state = m.data

        def on_entry_res(m):
            with S.lock:
                try:
                    S.entry_result = json.loads(m.data)
                    if 'x' in S.entry_result:
                        S.corridor_params['entry_x'] = round(float(S.entry_result['x']), 2)
                        S.corridor_params['entry_y'] = round(float(S.entry_result['y']), 2)
                        S.corridor_params['entry_yaw'] = round(float(S.entry_result.get('yaw', 0.0)), 3)
                    if 'half_width' in S.entry_result:
                        S.corridor_params['half_width'] = round(float(S.entry_result['half_width']), 2)
                except ValueError:
                    pass

        def on_banner(m):
            with S.lock:
                stamp('banner')
                try:
                    S.banner_det = json.loads(m.data)
                except ValueError:
                    pass

        def on_alt_state(m):
            with S.lock:
                stamp('altitude')
                try:
                    S.alt_state = json.loads(m.data)
                except ValueError:
                    S.alt_state = {'raw': m.data}

        def on_alt_target(m):
            with S.lock:
                S.alt_target = round(float(m.data), 2) if math.isfinite(m.data) else None

        def on_lat_target(m):
            with S.lock:
                S.lat_target = round(float(m.data), 2) if math.isfinite(m.data) else None

        def on_windows(m):
            with S.lock:
                try:
                    S.windows = json.loads(m.data)
                except ValueError:
                    pass

        def on_vio_health(m):
            with S.lock:
                stamp('vio')
                S.vio_health = m.data

        def on_vio_odom(m):
            p = m.pose.pose.position
            with S.lock:
                stamp('vio_odom')
                S.vio_pose = (round(p.x, 2), round(p.y, 2), round(p.z, 2))

        def on_px4_pose(m):
            p, q = m.pose.position, m.pose.orientation
            yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            with S.lock:
                stamp('px4_pose')
                S.px4_pose = (round(p.x, 3), round(p.y, 3), round(p.z, 3), round(yaw, 4))
                if S.px4_align is None and S.pose_odom is not None:
                    o = S.pose_odom
                    S.px4_align = (p.x, p.y, yaw, o[0], o[1], o[3])
                if S.px4_align is not None:
                    ax, ay, ayaw, ox, oy, oyaw = S.px4_align
                    d = oyaw - ayaw
                    c, s = math.cos(d), math.sin(d)
                    dx, dy = p.x - ax, p.y - ay
                    S.px4_odom = (round(ox + c * dx - s * dy, 3), round(oy + s * dx + c * dy, 3), round(p.z, 3),
                                  round(yaw + d, 4))
                    tx, ty, tyaw = S.slam_corr[0], S.slam_corr[1], math.radians(S.slam_corr[2])
                    c, s = math.cos(tyaw), math.sin(tyaw)
                    mx = c * S.px4_odom[0] - s * S.px4_odom[1] + tx
                    my = s * S.px4_odom[0] + c * S.px4_odom[1] + ty
                    tr = S.px4_track
                    if not tr or math.dist(tr[-1], (mx, my)) > 0.10:
                        S.px4_track = (tr + [(round(mx, 2), round(my, 2))])[-2000:]

        def on_px4_vel(m):
            v = m.twist.linear
            with S.lock:
                S.px4_vel = (round(v.x, 2), round(v.y, 2), round(v.z, 2))

        def on_ev_out(m):
            v = m.twist.twist.linear
            now = time.monotonic()
            with S.lock:
                stamp('ev_out')
                S.ev_times = [t for t in S.ev_times[-200:] if now - t < 2.0] + [now]
                S.ev_vel = (round(v.x, 2), round(v.y, 2), round(v.z, 2))

        def read_px4_params():
            # PX4: EKF2_EV_CTRL; ArduCopter: VISO_TYPE + the indoor source set (the other names read back empty)
            vals = self.get_params('mavros/param', ['EKF2_EV_CTRL', 'VISO_TYPE', 'EK3_SRC2_POSXY', 'EK3_SRC1_POSXY'])
            with S.lock:
                S.px4_params = vals or {}
                S.px4_params_reading = False

        def on_mavros(m):
            with S.lock:
                stamp('mavros')
                S.mavros = {'connected': bool(m.connected), 'armed': bool(m.armed), 'mode': m.mode}
                # EKF2_EV_CTRL tells whether PX4 fuses the VIO at all: read it once per connection
                start = m.connected and not S.px4_params and not getattr(S, 'px4_params_reading', False)
                if start:
                    S.px4_params_reading = True
                elif not m.connected:
                    S.px4_params = {}
            if start:
                threading.Thread(target=read_px4_params, daemon=True).start()

        n.create_subscription(OccupancyGrid, '/map', lambda m: self.on_map(m, '/map'), latched)
        n.create_subscription(OccupancyGrid, '/map', lambda m: self.on_map(m, '/map'), 10)
        n.create_subscription(OccupancyGrid, '/rtabmap/grid_map', lambda m: self.on_map(m, '/rtabmap/grid_map'), 10)
        n.create_subscription(PointCloud2, '/rtabmap/cloud_map', on_rtab_cloud, 10)
        n.create_subscription(String, '/sim_map/path_json', on_sim_path, latched)
        n.create_subscription(String, '/sim_map/path_json', on_sim_path, 10)
        self.pub_sim_goal = n.create_publisher(String, '/sim_map/set_goal', 10)
        n.create_subscription(Odometry, '/nav/odom', on_odom, 10)
        n.create_subscription(LaserScan, '/vio/obstacle_scan', on_scan, qos_profile_sensor_data)
        n.create_subscription(PointCloud2, '/vio/depth_points', on_cloud, qos_profile_sensor_data)
        n.create_subscription(String, '/nav/corridor/state', on_corridor, 10)
        n.create_subscription(String, '/nav/entry/state', on_entry, 10)
        n.create_subscription(String, '/nav/entry/result', on_entry_res, 10)
        n.create_subscription(String, '/nav/banner/detection', on_banner, 10)
        n.create_subscription(String, '/nav/altitude/state', on_alt_state, 10)
        n.create_subscription(Float32, '/nav/altitude', on_alt_target, 10)
        n.create_subscription(Float32, '/nav/lateral', on_lat_target, 10)
        n.create_subscription(String, '/nav/altitude/windows', on_windows, 10)
        n.create_subscription(String, '/vio/health', on_vio_health, 10)
        n.create_subscription(Odometry, '/visual_slam/tracking/odometry', on_vio_odom, qos_profile_sensor_data)
        n.create_subscription(MavState, '/mavros/state', on_mavros, 10)
        n.create_subscription(PoseStamped, '/mavros/local_position/pose', on_px4_pose, qos_profile_sensor_data)
        n.create_subscription(TwistStamped, '/mavros/local_position/velocity_local', on_px4_vel,
                              qos_profile_sensor_data)
        n.create_subscription(Odometry, '/mavros/odometry/out', on_ev_out, 10)

        self.img_sub = None
        self.set_camera('depth')
        ex = MultiThreadedExecutor()
        ex.add_node(n)
        self.ready.set()
        ex.spin()

    def on_map(self, m, topic='/map'):
        if m.info.width == 0 or m.info.height == 0:
            return
        a = np.array(m.data, np.int16).reshape(m.info.height, m.info.width)
        with S.lock:
            S.grids[topic] = (a.astype(np.int8), float(m.info.origin.position.x), float(m.info.origin.position.y),
                              float(m.info.resolution), time.monotonic())
            slam = S.grids.get('/map')
        if topic != '/map' and slam is not None and time.monotonic() - slam[4] < 15.0:
            return
        img = np.zeros((m.info.height, m.info.width, 4), np.uint8)
        img[a < 0] = (30, 38, 48, 120)                # unknown
        img[a == 0] = (45, 115, 80, 140)              # free tunnel interior
        img[(a > 0) & (a < 50)] = (80, 160, 120, 160)
        img[a >= 50] = (50, 210, 255, 255)            # SLAM tunnel walls / obstacles (bright cyan/amber in RGBA)
        ok, png = cv2.imencode('.png', cv2.cvtColor(img[::-1], cv2.COLOR_RGBA2BGRA))
        with S.lock:
            if S.start_pose_scan is None and (S.pose_map or S.pose_odom):
                S.start_pose_scan = list(S.pose_map or S.pose_odom)
            S.raw_grid = (a.astype(np.int8), float(m.info.origin.position.x),
                          float(m.info.origin.position.y), float(m.info.resolution))
            S.map_png = png.tobytes() if ok else None
            S.map_meta = {
                'x0': round(m.info.origin.position.x, 3),
                'y0': round(m.info.origin.position.y, 3),
                'w': round(m.info.width * m.info.resolution, 3),
                'h': round(m.info.height * m.info.resolution, 3),
                'res': round(m.info.resolution, 3),
                'cells': f'{m.info.width}x{m.info.height}',
            }
            S.map_ver += 1
            S.ros_time['map'] = time.monotonic()

    def set_camera(self, name):
        if self.img_sub is not None:
            self.node.destroy_subscription(self.img_sub)
            self.img_sub = None
        self.camera_name = name
        topic = CAMERAS.get(name, '/camera/depth/image_rect_raw')
        self.img_sub = self.node.create_subscription(self.Image, topic, self.on_image, self.qos_img)

    def on_image(self, m):
        now = time.monotonic()
        S.ros_time[f'cam_{self.camera_name}'] = now
        S.ros_time['camera'] = now
        if now - self.jpeg_time < 0.12:               # ~8 Hz to browser
            return
        try:
            if m.encoding in ('16UC1', 'mono16'):
                d = np.frombuffer(m.data, np.uint16).reshape(m.height, m.width).astype(np.float32) * 0.001
                img = self.colorize_depth(d)
            elif m.encoding == '32FC1':
                d = np.frombuffer(m.data, np.float32).reshape(m.height, m.width)
                img = self.colorize_depth(d)
            elif m.encoding in ('mono8', '8UC1'):
                g = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width)
                img = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
            else:
                img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)[..., :3].copy()
                if m.encoding == 'rgb8':
                    img = img[..., ::-1]
            scale = 640 / max(img.shape[1], 1)
            if scale < 1.0:
                img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            ok, jpg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                self.jpeg, self.jpeg_time = jpg.tobytes(), now
        except Exception:
            pass

    @staticmethod
    def colorize_depth(d):
        valid = np.isfinite(d) & (d > 0.15) & (d < 6.0)
        norm = np.clip((6.0 - np.where(valid, d, 6.0)) / 5.85 * 255.0, 0, 255).astype(np.uint8)
        col = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
        col[~valid] = (18, 22, 28)
        h, w = d.shape
        # Draw horizontal obstacle scan band guide lines
        y1, y2 = int(h * 0.42), int(h * 0.58)
        cv2.line(col, (0, y1), (w, y1), (200, 200, 200), 1)
        cv2.line(col, (0, y2), (w, y2), (200, 200, 200), 1)
        mid = d[y1:y2, w // 3:2 * w // 3]
        mv = mid[np.isfinite(mid) & (mid > 0.18)]
        if mv.size:
            dmin = float(np.percentile(mv, 5))
            cv2.putText(col, f'ahead: {dmin:.2f} m', (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return col

    def trigger(self, srv_name, timeout=8.0):
        cli = self.node.create_client(self.Trigger, srv_name)
        try:
            if not cli.wait_for_service(timeout_sec=2.5):
                return False, f'{srv_name} not available (tunnel stack not running?)'
            fut = cli.call_async(self.Trigger.Request())
            end = time.monotonic() + timeout
            while not fut.done() and time.monotonic() < end:
                time.sleep(0.05)
            if not fut.done():
                return False, f'{srv_name} timed out'
            r = fut.result()
            return getattr(r, 'success', True), getattr(r, 'message', 'ok')
        finally:
            self.node.destroy_client(cli)

    def get_params(self, node_name, names):
        cli = self.node.create_client(self.GetParameters, f'/{node_name}/get_parameters')
        try:
            if not cli.wait_for_service(timeout_sec=2.5):
                return None
            fut = cli.call_async(self.GetParameters.Request(names=list(names)))
            end = time.monotonic() + 5.0
            while not fut.done() and time.monotonic() < end:
                time.sleep(0.05)
            if not fut.done():
                return None
            out = {}
            for k, v in zip(names, fut.result().values):
                out[k] = {1: v.bool_value, 2: v.integer_value, 3: v.double_value, 4: v.string_value}.get(v.type)
            return out
        finally:
            self.node.destroy_client(cli)

    def call(self, srv_type, srv_name, req, timeout=10.0):
        """Call a service; returns the response or None."""
        cli = self.node.create_client(srv_type, srv_name)
        try:
            if not cli.wait_for_service(timeout_sec=2.0):
                return None
            fut = cli.call_async(req)
            end = time.monotonic() + timeout
            while not fut.done() and time.monotonic() < end:
                time.sleep(0.05)
            return fut.result() if fut.done() else None
        finally:
            self.node.destroy_client(cli)

    def set_params(self, node_name, kv):
        cli = self.node.create_client(self.SetParameters, f'/{node_name}/set_parameters')
        try:
            if not cli.wait_for_service(timeout_sec=2.5):
                return False, f'/{node_name}/set_parameters not available'
            params = []
            for k, v in kv.items():
                if isinstance(v, bool):
                    pv = self.ParameterValue(type=self.ParameterType.PARAMETER_BOOL, bool_value=v)
                elif isinstance(v, int) and not isinstance(v, bool):
                    pv = self.ParameterValue(type=self.ParameterType.PARAMETER_INTEGER, integer_value=v)
                elif isinstance(v, float):
                    pv = self.ParameterValue(type=self.ParameterType.PARAMETER_DOUBLE, double_value=float(v))
                else:
                    pv = self.ParameterValue(type=self.ParameterType.PARAMETER_STRING, string_value=str(v))
                params.append(self.Parameter(name=k, value=pv))
            fut = cli.call_async(self.SetParameters.Request(parameters=params))
            end = time.monotonic() + 5.0
            while not fut.done() and time.monotonic() < end:
                time.sleep(0.05)
            return (True, 'parameters updated') if fut.done() else (False, 'set_parameters timed out')
        finally:
            self.node.destroy_client(cli)

    def px4_action(self, act):
        if act == 'land':
            cli = self.node.create_client(self.SetMode, '/mavros/set_mode')
            try:
                if not cli.wait_for_service(timeout_sec=2.0):
                    return False, 'MAVROS not running'
                req = self.SetMode.Request(custom_mode='AUTO.LAND')
                fut = cli.call_async(req)
                time.sleep(0.3)
                return True, 'AUTO.LAND sent to PX4'
            finally:
                self.node.destroy_client(cli)
        if act == 'rtl':
            cli = self.node.create_client(self.SetMode, '/mavros/set_mode')
            try:
                if not cli.wait_for_service(timeout_sec=2.0):
                    return False, 'MAVROS not running'
                req = self.SetMode.Request(custom_mode='AUTO.RTL')
                fut = cli.call_async(req)
                time.sleep(0.3)
                return True, 'AUTO.RTL sent to PX4'
            finally:
                self.node.destroy_client(cli)
        if act == 'kill':
            cli = self.node.create_client(self.CommandLong, '/mavros/cmd/command')
            try:
                if not cli.wait_for_service(timeout_sec=2.0):
                    return False, 'MAVROS not running'
                req = self.CommandLong.Request(command=400, param1=0.0, param2=21196.0)
                cli.call_async(req)
                time.sleep(0.3)
                return True, 'KILL sent to PX4'
            finally:
                self.node.destroy_client(cli)
        return False, 'unknown action'


class TunnelStack:
    def __init__(self):
        self.proc = None
        self.run = None
        self.started = None
        self.link = '/dev/ttyACM0:921600'
        self.vslam = True

    @staticmethod
    def color_running():
        return subprocess.run(['pgrep', '-f', 'cuvslam\\.launch\\.py.*color:=true'], capture_output=True).returncode == 0

    @staticmethod
    def cuvslam_running():
        return subprocess.run(['pgrep', '-f', 'cuvslam\\.launch\\.py'], capture_output=True).returncode == 0

    @staticmethod
    def launches():
        r = subprocess.run(
            ['pgrep', '-f', r'(tunnel_slam|tunnel_real|tunnel_sim_saved)\.launch\.py|container_tunnel\.sh|sim_saved_map_drone\.py'],
            capture_output=True, text=True)
        return [int(p) for p in r.stdout.split()]

    @staticmethod
    def sim_saved_running():
        return subprocess.run(['pgrep', '-f', 'sim_saved_map_drone\\.py'], capture_output=True).returncode == 0

    def running(self):
        if self.proc is not None and self.proc.poll() is None:
            return True
        return bool(self.launches())

    def start(self, link='/dev/ttyTHS1:921600', vslam=True, course='v2', scan_band=0.20, gcs_ip='', rviz=False,
              rtabmap=False, nav=False):
        if self.running():
            return False, 'tunnel stack is already running'
        self.link = link
        self.vslam = bool(vslam)
        self.run = f'tunnel_slam_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        (WS / 'runs').mkdir(parents=True, exist_ok=True)
        log = open(WS / 'runs' / f'{self.run}_console.log', 'w')
        env = dict(os.environ, FASTRTPS_DEFAULT_PROFILES_FILE=str(WS / 'fastdds_udp.xml'))
        cmd = [
            'ros2', 'launch', str(LAUNCH_ALL / 'tunnel_slam.launch.py'),
            f'run:={self.run}', f'link:={link}', f'vio:={"cuvslam" if vslam else "rgbd"}',
            f'course:={course}', f'scan_band:={scan_band}', 'dashboard:=false',
            f'rviz:={"true" if rviz else "false"}', f'rtabmap:={"true" if rtabmap else "false"}',
            # colour on even without RTAB-Map at launch: the RTAB-Map button starts it later on colour + aligned depth
            'color:=true', f'nav:={"true" if nav else "false"}',
        ]
        if gcs_ip:
            cmd.append(f'gcs_ip:={gcs_ip}')
        self.proc = subprocess.Popen(
            ['bash', '-c', 'source /opt/ros/humble/setup.bash && exec "$@"', '_', *cmd],
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.started = time.time()
        with S.lock:
            S.track = []
            S.px4_track = []
            S.px4_align = None
        return True, f'Tunnel + SLAM stack starting (run {self.run}, {"cuVSLAM" if vslam else "RGB-D"})'

    def stop(self):
        """Stop the camera / SLAM stack. RViz and RTAB-Map stay open: they close only with their own buttons or
        when you close their windows. Ctrl-C the launch (its own shutdown stops its nodes cleanly), SIGTERM it if it
        is still up after 10 s, then the name-based sweep for anything left."""
        pids = self.launches()
        for sig, wait in ((signal.SIGINT, 10.0), (signal.SIGTERM, 5.0)):
            for p in pids:
                try:
                    os.kill(p, sig)
                except ProcessLookupError:
                    pass
            end = time.monotonic() + wait
            while time.monotonic() < end and any(Path(f'/proc/{p}').exists() for p in pids):
                time.sleep(0.3)
        subprocess.run(['pkill', '-INT', '-f', 'container_tunnel\\.sh|tunnel_real\\.launch\\.py|sim_saved_map_drone\\.py|'
                        'tunnel_sim_saved\\.launch\\.py'], check=False)
        subprocess.run([str(LAUNCH_ALL / 'scripts/stop_all.sh'), '--tunnel'], timeout=90, check=False)
        self.proc = None
        return True, 'stack stopped (RViz / RTAB-Map stay open)'

    def start_sim_saved(self, rviz=False):
        if self.running():
            self.stop()
            time.sleep(1.0)
        self.run = f'sim_saved_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
        (WS / 'runs').mkdir(parents=True, exist_ok=True)
        log = open(WS / 'runs' / f'{self.run}_console.log', 'w')
        env = dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':0'),
                   FASTRTPS_DEFAULT_PROFILES_FILE=str(WS / 'fastdds_udp.xml'))
        cmd = [
            'ros2', 'launch', str(LAUNCH_ALL / 'tunnel_sim_saved.launch.py'),
            f'rviz:={"true" if rviz else "false"}', 'autonav:=true', 'dashboard:=false',
        ]
        self.proc = subprocess.Popen(
            ['bash', '-c', 'source /opt/ros/humble/setup.bash && exec "$@"', '_', *cmd],
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        self.started = time.time()
        with S.lock:
            S.track = []
        return True, 'Sim Drone launched on Saved RealSense Map — navigating obstacles!'

    @staticmethod
    def rviz_running():
        # anchored: any shell whose command line mentions rviz2 (a grep, a tail) matched before
        return subprocess.run(['pgrep', '-f', '^/opt/ros/humble/lib/rviz2/rviz2 '], capture_output=True).returncode == 0

    @staticmethod
    def rtabmap_running():
        return subprocess.run(['pgrep', '-f', '^/opt/ros/humble/lib/rtabmap_slam/rtabmap'],    # the mapper; GUI is optional
                              capture_output=True).returncode == 0

    @staticmethod
    def stop_procs(pattern, grace=3.0):
        """Ctrl-C, then SIGTERM, then SIGKILL: RViz2 often ignores SIGINT and stayed open ('stuck', 2026-10-27)."""
        for sig in ('-INT', '-TERM', '-KILL'):
            subprocess.run(['pkill', sig, '-f', pattern], check=False)
            end = time.monotonic() + grace
            while time.monotonic() < end:
                if subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode != 0:
                    return
                time.sleep(0.2)

    def toggle_rviz(self, start=True):
        if not start:
            self.stop_procs('^/opt/ros/humble/lib/rviz2/rviz2 ')
            return True, 'RViz2 stopped'
        if self.rviz_running():
            return True, 'RViz2 already running on Jetson (:0)'
        env = dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':0'),
                   FASTRTPS_DEFAULT_PROFILES_FILE=str(WS / 'fastdds_udp.xml'))
        env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        env.pop('QT_QPA_FONTDIR', None)
        log = open(WS / 'runs/gcs/rviz_tunnel.log', 'w')
        subprocess.Popen(
            ['bash', '-c', 'source /opt/ros/humble/setup.bash && exec /opt/ros/humble/lib/rviz2/rviz2 -d '
             f'{WS / "src/vio_bringup/rviz/tunnel_slam.rviz"} --ros-args -r __node:=rviz2_tunnel'],
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        return True, 'RViz2 launched on Jetson display (:0)'

    def toggle_rtabmap(self, start=True):
        if not start:
            # anchored: a bare 'rtabmap' pattern also matched unrelated shells whose command line mentions it
            self.stop_procs('^/opt/ros/humble/lib/rtabmap_(slam|viz|odom)/|^/usr/bin/python3 .*tunnel_viz\\.launch')
            return True, 'RTAB-Map stopped'
        if self.rtabmap_running():
            return True, 'RTAB-Map already running on Jetson (:0)'
        env = dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':0'),
                   FASTRTPS_DEFAULT_PROFILES_FILE=str(WS / 'fastdds_udp.xml'))
        env.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
        env.pop('QT_QPA_FONTDIR', None)
        log = open(WS / 'runs/gcs/rtabmap_tunnel.log', 'w')
        subprocess.Popen(
            ['bash', '-c', 'source /opt/ros/humble/setup.bash && exec ros2 launch '
             f'{LAUNCH_ALL / "tunnel_viz.launch.py"} rviz:=false rtabmap:=true rtabmap_viz:=true '
             + ('source:=color' if not self.cuvslam_running() or self.color_running() else 'source:=infra')
             + (' rgbd_odom:=false' if self.cuvslam_running() else '')],
            stdout=log, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        return True, 'RTAB-Map started with its window on the Jetson display (:0)'


STACK = TunnelStack()


class ProcState(threading.Thread):
    """Which parts run (pgrep), polled every 2 s. snapshot() used to run 5-6 pgreps per websocket frame per browser
    inside the event loop: with the stack up the frame rate fell to <1 Hz and every check read stale (2026-10-27)."""
    def __init__(self):
        super().__init__(daemon=True)
        self.v = {'running': False, 'rviz': False, 'rtabmap': False, 'sim_saved': False, 'cuvslam': False}

    def run(self):
        while True:
            try:
                self.v = {'running': STACK.running(), 'rviz': STACK.rviz_running(),
                          'rtabmap': STACK.rtabmap_running(), 'sim_saved': STACK.sim_saved_running(),
                          'cuvslam': STACK.cuvslam_running()}
            except Exception:
                pass
            time.sleep(2.0)


PROCS = ProcState()


def fresh(key, age=4.0):
    t = S.ros_time.get(key)
    return t is not None and (time.monotonic() - t) < age


def stack_phase():
    if not PROCS.v['running'] and not fresh('scan', 4.0):
        return 'stopped'
    if not fresh('scan', 4.0):
        return 'starting: RealSense D435 + obstacle scan'
    if not fresh('map', 8.0) and not fresh('corridor', 5.0):
        return 'starting: 2D SLAM + corridor nodes'
    return 'ready'


def diagnostics():
    checks = [
        ('RealSense Depth (/vio/obstacle_scan)', fresh('scan', 3.0),
         f'min {S.scan["min_dist"]} m, ahead {S.scan["front_dist"]} m' if S.scan else 'waiting for D435 depth'),
        ('RealSense 3D Cloud (/vio/depth_points)', fresh('cloud', 3.0), f'{S.cloud_pts} pts'),
        ('Camera Stream (' + ROS.camera_name + ')', fresh('camera', 4.0), CAMERAS.get(ROS.camera_name, '')),
        ('2D SLAM Map (/map, slam_toolbox)', fresh('map', 10.0) or S.map_meta is not None,
         f'{S.map_meta["cells"]} @ {S.map_meta["res"]}m' if S.map_meta else 'waiting for first scan'),
        ('SLAM TF (map -> odom -> base_link)', fresh('odom', 3.0),
         f'corr dx={S.slam_corr[0]:.2f} dy={S.slam_corr[1]:.2f} dyaw={S.slam_corr[2]:.1f}°'),
        ('Pixhawk / MAVROS (optional on bench)', S.mavros.get('connected', False),
         f'mode {S.mavros.get("mode") or "-"}, armed={S.mavros.get("armed")}'),
    ]
    if fresh('corridor', 10.0) or fresh('entry', 10.0) or fresh('altitude', 10.0):      # nav:=true runs only
        checks += [('Corridor Navigator (corridor_nav)', fresh('corridor', 5.0), S.corridor_state),
                   ('Tunnel Entrance Aligner (entry_align)', fresh('entry', 5.0), S.entry_state),
                   ('3D Altitude & Window Planner', fresh('altitude', 5.0), f'target z={S.alt_target} m')]
    if PROCS.v['cuvslam'] or fresh('vio_odom', 5.0):
        checks.append(('cuVSLAM Stereo VIO (/visual_slam/tracking/odometry)', fresh('vio_odom', 2.0),
                       f'pose {S.vio_pose}' if S.vio_pose else 'starting'))
    if fresh('vio', 5.0) or S.mavros.get('connected'):
        checks.append(('VIO -> PX4 EKF2 (/mavros/odometry/out)', fresh('ev_out', 2.0),
                       f'{len(S.ev_times) / 2.0:.0f} Hz, {S.vio_health or "vio_to_px4 not running"}'))
        checks.append(('PX4 local position (/mavros/local_position/pose)', fresh('px4_pose', 2.0),
                       f'{S.px4_pose[:3]}' if S.px4_pose else 'none: EKF2 has no position (EV fusion off?)'))
    return [{'name': k, 'ok': bool(ok), 'detail': d} for k, ok, d in checks]


def snapshot():
    with S.lock:
        return {
            'time': time.time(),
            'stack': {
                'running': PROCS.v['running'] or fresh('scan', 4.0),
                'phase': stack_phase(),
                'run': STACK.run,
                'link': STACK.link,
                'vslam': STACK.vslam,
                'cuvslam': PROCS.v['cuvslam'],
                'rviz': PROCS.v['rviz'],
                'rtabmap': PROCS.v['rtabmap'],
                'sim_saved': PROCS.v['sim_saved'],
                'saved_map_exists': (WS / 'runs/saved_tunnel_map/saved_map.npz').is_file(),
                'accum_pts': len(S.accum_pts),
            },
            'planned_path': S.planned_path,
            'sim_goal': S.sim_goal,
            'pose_map': S.pose_map,
            'pose_odom': S.pose_odom,
            'slam_corr': S.slam_corr,
            'track': S.track[-1000:],
            'map': {'ver': S.map_ver, **(S.map_meta or {})} if S.map_meta else None,
            'scan': S.scan,
            'cloud_pts': S.cloud_pts,
            'corridor_state': S.corridor_state,
            'corridor_params': S.corridor_params,
            'entry_state': S.entry_state,
            'entry_result': S.entry_result,
            'banner': S.banner_det,
            'alt_state': S.alt_state,
            'alt_target': S.alt_target,
            'lat_target': S.lat_target,
            'windows': S.windows,
            'vio_health': S.vio_health,
            'vio_pose': S.vio_pose,
            'px4': {
                'pose': S.px4_pose if fresh('px4_pose', 2.0) else None,
                'vel': S.px4_vel,
                'odom': S.px4_odom if fresh('px4_pose', 2.0) else None,
                'track': S.px4_track[-1000:],
                'aligned': S.px4_align is not None,
                'ev_hz': round(len([t for t in S.ev_times if time.monotonic() - t < 2.0]) / 2.0, 1),
                'ev_vel': S.ev_vel if fresh('ev_out', 2.0) else None,
                'err': round(math.dist(S.px4_odom[:2], S.pose_odom[:2]), 2)
                if S.px4_odom and S.pose_odom and fresh('px4_pose', 2.0) else None,
                'params': S.px4_params,
            },
            'mavros': S.mavros,
            'camera': ROS.camera_name,
            'checks': diagnostics(),
        }


async def ws_handler(request):
    ws = web.WebSocketResponse(heartbeat=10)
    await ws.prepare(request)
    try:
        while not ws.closed:
            await ws.send_str(json.dumps(snapshot(), default=float))
            await asyncio.sleep(0.15)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return ws


async def map_png(request):
    png = S.map_png
    if png is None:
        raise web.HTTPNotFound()
    return web.Response(body=png, content_type='image/png', headers={'Cache-Control': 'no-store'})


async def mjpeg(request):
    resp = web.StreamResponse(headers={'Content-Type': 'multipart/x-mixed-replace; boundary=frame',
                                       'Cache-Control': 'no-store'})
    await resp.prepare(request)
    last = None
    try:
        while True:
            j = ROS.jpeg
            if j is not None and j is not last:
                last = j
                await resp.write(b'--frame\r\nContent-Type: image/jpeg\r\nContent-Length: ' + str(len(j)).encode() +
                                 b'\r\n\r\n' + j + b'\r\n')
            await asyncio.sleep(0.08)
    except (ConnectionResetError, asyncio.CancelledError):
        pass
    return resp


def write_costmap(grid, ox, oy, res, path):
    """nav2 map_server format: <path>.pgm (trinary: 0 occupied, 254 free, 205 unknown) + <path>.yaml."""
    img = np.full(grid.shape, 205, np.uint8)
    img[(grid >= 0) & (grid <= 25)] = 254
    img[grid >= 65] = 0
    img = img[::-1]                                  # pgm row 0 = top = highest y
    h, w = img.shape
    path.with_suffix('.pgm').write_bytes(f'P5\n{w} {h}\n255\n'.encode() + img.tobytes())
    path.with_suffix('.yaml').write_text(
        f'image: {path.with_suffix(".pgm").name}\nmode: trinary\nresolution: {res}\n'
        f'origin: [{ox}, {oy}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')


def save_costmaps(dirs):
    """Save every received occupancy grid (slam_toolbox /map, RTAB-Map /rtabmap/grid_map) as pgm + yaml into each
    of dirs, plus the slam_toolbox pose graph and a RTAB-Map database backup. Returns what got saved."""
    saved = []
    with S.lock:
        grids = dict(S.grids)
    for topic, (grid, ox, oy, res, _) in grids.items():
        name = 'slam_map' if topic == '/map' else 'rtabmap_grid'
        for d in dirs:
            write_costmap(grid, ox, oy, res, d / name)
        saved.append(f'{name}.pgm/.yaml')
    try:
        from slam_toolbox.srv import SerializePoseGraph
        r = ROS.call(SerializePoseGraph, '/slam_toolbox/serialize_map',
                     SerializePoseGraph.Request(filename=str(dirs[-1] / 'slam_posegraph')))
        if r is not None and (dirs[-1] / 'slam_posegraph.posegraph').is_file():
            saved.append('slam_posegraph')
    except ImportError:
        pass
    # RTAB-Map keeps part of the map in memory: backup flushes it to <db>.back
    db = Path.home() / '.ros/rtabmap.db'
    if ROS.call(ROS.Empty, '/rtabmap/rtabmap/backup', ROS.Empty.Request(), timeout=30.0) is not None:
        db = db.with_name('rtabmap.db.back') if db.with_name('rtabmap.db.back').is_file() else db
    if db.is_file():
        for d in dirs:
            shutil.copy(db, d / 'rtabmap.db')
        saved.append('rtabmap.db')
    return saved


def ok(success, message, **extra):
    return web.json_response({'ok': bool(success), 'message': message, **extra})


async def api(request):
    what = request.match_info['what']
    body = await request.json() if request.can_read_body else {}
    loop = asyncio.get_running_loop()

    if what == 'stack_start':
        link = body.get('link', '/dev/ttyTHS1:921600')
        vslam = bool(body.get('vslam', True))
        course = body.get('course', 'v2')
        band = float(body.get('scan_band', 0.20))
        ip = request.remote or ''
        gcs_ip = '' if ip.startswith('127.') or ip == '::1' else ip
        return ok(*STACK.start(link=link, vslam=vslam, course=course, scan_band=band, gcs_ip=gcs_ip,
                               rviz=bool(body.get('rviz', False)), rtabmap=bool(body.get('rtabmap', False))))

    if what == 'stack_stop':
        return ok(*(await loop.run_in_executor(None, STACK.stop)))

    if what == 'rviz':
        return ok(*(await loop.run_in_executor(None, lambda: STACK.toggle_rviz(body.get('start', True)))))

    if what == 'rtabmap':
        return ok(*(await loop.run_in_executor(None, lambda: STACK.toggle_rtabmap(body.get('start', True)))))

    if what == 'map_realsense_start':
        # Step 1: Launch Real RealSense + 2D SLAM + RTAB-Map 3D SLAM + RViz2 in one click
        if STACK.running():
            await loop.run_in_executor(None, STACK.stop)
        with S.lock:
            S.accum_pts = np.zeros((0, 3), np.float32)
            S.start_pose_scan = None
            S.planned_path = []
            S.track = []
        s1, m1 = await loop.run_in_executor(None, lambda: STACK.start(
            link=body.get('link', 'none'), vslam=bool(body.get('vslam', True)), rtabmap=True,
            rviz=bool(body.get('rviz', True))))
        return ok(s1, f'{m1} + RTAB-Map + RViz2. Walk the drone through, then click Save Map.')

    if what == 'save_map':
        # Step 2: Save the scanned 2D SLAM map + 3D point cloud to runs/saved_tunnel_map/saved_map.npz
        out_dir = WS / 'runs/saved_tunnel_map'
        out_dir.mkdir(parents=True, exist_ok=True)
        with S.lock:
            rg = S.raw_grid
            pts = S.accum_pts.copy()
            sp = S.start_pose_scan or [0.0, 0.0, 1.2, 0.0]
            ep = list(S.pose_map or S.pose_odom or [4.0, 0.0, 1.2, 0.0])
            png = S.map_png
        if rg is None and len(pts) == 0:
            return ok(False, 'No SLAM map or 3D point cloud received yet — launch RealSense first!')
        if rg is not None:
            grid, ox, oy, res = rg
        else:
            # Build 2D occupancy grid from accumulated 3D points
            res = 0.05
            ox, oy = float(pts[:, 0].min() - 1.5), float(pts[:, 1].min() - 1.5)
            w = max(40, int((pts[:, 0].max() + 1.5 - ox) / res))
            h = max(40, int((pts[:, 1].max() + 1.5 - oy) / res))
            grid = np.zeros((h, w), np.int8)
            ii = np.clip(((pts[:, 0] - ox) / res).astype(int), 0, w - 1)
            jj = np.clip(((pts[:, 1] - oy) / res).astype(int), 0, h - 1)
            grid[jj, ii] = 100
        npz_path = out_dir / 'saved_map.npz'
        np.savez_compressed(str(npz_path), grid=grid, origin=np.array([ox, oy, res]),
                            points=pts, start_pose=np.array(sp), end_pose=np.array(ep))
        if png:
            (out_dir / 'saved_map.png').write_bytes(png)
        stamp_dir = WS / 'runs/maps' / time.strftime('%Y-%m-%d_%H-%M-%S')
        stamp_dir.mkdir(parents=True, exist_ok=True)
        saved = await loop.run_in_executor(None, lambda: save_costmaps([out_dir, stamp_dir]))
        for f in ('saved_map.npz', 'saved_map.png'):
            if (out_dir / f).is_file():
                shutil.copy(out_dir / f, stamp_dir / f)
        return ok(True, f'Saved map ({grid.shape[1]}x{grid.shape[0]} cells, {len(pts)} 3D pts) + '
                        f'{", ".join(saved) or "no costmaps"} -> {stamp_dir}')

    if what == 'px4_align':
        with S.lock:
            S.px4_align = None
            S.px4_track = []
        return ok(True, 'PX4 track re-aligned to the current VIO pose on the next PX4 sample')

    if what == 'px4_params':
        names = ['EKF2_EV_CTRL', 'EKF2_HGT_REF', 'EKF2_GPS_CTRL', 'EKF2_EV_DELAY', 'EKF2_EVV_NOISE']
        vals = await loop.run_in_executor(None, lambda: ROS.get_params('mavros/param', names))
        if vals is None:
            return ok(False, 'MAVROS param plugin not available (Pixhawk link off?)')
        with S.lock:
            S.px4_params = vals
        ev = vals.get('EKF2_EV_CTRL')
        hint = '' if ev is not None and int(ev) & 4 else \
            ' - EKF2_EV_CTRL lacks bit 2 (velocity): PX4 ignores the VIO; see vio_bringup/config/px4_6c_vio_params.txt'
        return ok(True, ', '.join(f'{k}={v}' for k, v in vals.items()) + hint)

    if what == 'sim_saved_start':
        # Step 3: Launch Simulated Drone on the Saved RealSense Map
        return ok(*(await loop.run_in_executor(None, lambda: STACK.start_sim_saved(rviz=body.get('rviz', False)))))

    if what == 'sim_navigate':
        if 'goal_x' in body and 'goal_y' in body:
            ROS.pub_sim_goal.publish(ROS.String(data=json.dumps(body)))
            return ok(True, f'Sim drone navigating around obstacles to ({body["goal_x"]:.2f}, {body["goal_y"]:.2f})')
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/sim_map/navigate'))))

    if what == 'sim_reset':
        with S.lock:
            S.track = []
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/sim_map/reset'))))

    if what == 'camera':
        name = body.get('name', 'depth')
        if name not in CAMERAS:
            return ok(False, f'unknown camera {name}')
        ROS.set_camera(name)
        return ok(True, f'camera switched to {name}')

    if what == 'use_current_pose':
        p = S.pose_odom or S.pose_map or (0.0, 0.0, 1.5, 0.0)
        with S.lock:
            S.corridor_params['entry_x'] = round(p[0], 2)
            S.corridor_params['entry_y'] = round(p[1], 2)
            S.corridor_params['entry_yaw'] = round(p[3], 3)
        kv = {'entry_x': float(p[0]), 'entry_y': float(p[1]), 'entry_yaw': float(p[3])}
        await loop.run_in_executor(None, lambda: ROS.set_params('corridor_nav', kv))
        await loop.run_in_executor(None, lambda: ROS.set_params('altitude_planner', kv))
        return ok(True, f'entrance set to current pose ({p[0]:.2f}, {p[1]:.2f}, yaw {math.degrees(p[3]):.1f}°)')

    if what == 'corridor_params':
        kv = {}
        for k in ('entry_x', 'entry_y', 'entry_yaw', 'altitude', 'corridor_speed', 'half_width', 'max_length'):
            if k in body and body[k] is not None:
                kv[k] = float(body[k])
                with S.lock:
                    S.corridor_params[k] = round(kv[k], 3)
        s1, m1 = await loop.run_in_executor(None, lambda: ROS.set_params('corridor_nav', kv))
        akv = {k: kv[k] for k in ('entry_x', 'entry_y', 'entry_yaw', 'half_width') if k in kv}
        if 'altitude' in kv:
            akv['nominal'] = kv['altitude']
        if akv:
            await loop.run_in_executor(None, lambda: ROS.set_params('altitude_planner', akv))
        return ok(s1, m1)

    if what == 'corridor_run':
        if body:
            kv = {k: float(body[k]) for k in ('entry_x', 'entry_y', 'entry_yaw', 'altitude',
                                              'corridor_speed', 'half_width', 'max_length')
                  if k in body and body[k] is not None}
            if kv:
                with S.lock:
                    S.corridor_params.update(kv)
                await loop.run_in_executor(None, lambda: ROS.set_params('corridor_nav', kv))
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/nav/corridor/run'))))

    if what == 'corridor_cancel':
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/nav/corridor/cancel'))))

    if what == 'entry_align':
        colour = body.get('colour', 'green')
        run_corr = bool(body.get('run_corridor', False))
        await loop.run_in_executor(None, lambda: ROS.set_params('banner_detector', {'colour': colour}))
        await loop.run_in_executor(None, lambda: ROS.set_params(
            'entry_align', {'colour': colour, 'mode': 'banner', 'run_corridor': run_corr}))
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/nav/entry/align'))))

    if what == 'entry_cancel':
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/nav/entry/cancel'))))

    if what == 'altitude_start':
        with S.lock:
            akv = {
                'entry_x': float(S.corridor_params['entry_x']),
                'entry_y': float(S.corridor_params['entry_y']),
                'entry_yaw': float(S.corridor_params['entry_yaw']),
                'half_width': float(S.corridor_params['half_width']),
                'nominal': float(S.corridor_params['altitude']),
            }
        await loop.run_in_executor(None, lambda: ROS.set_params('altitude_planner', akv))
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/nav/altitude/start'))))

    if what == 'altitude_stop':
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/nav/altitude/stop'))))

    if what == 'vio_reset':
        with S.lock:
            S.track = []
        return ok(*(await loop.run_in_executor(None, lambda: ROS.trigger('/vio/reset'))))

    if what == 'clear_track':
        with S.lock:
            S.track = []
            S.map_png = None
            S.map_meta = None
        return ok(True, 'track cleared')

    if what in ('land', 'rtl', 'kill'):
        return ok(*(await loop.run_in_executor(None, lambda: ROS.px4_action(what))))

    raise web.HTTPNotFound()


async def index(request):
    return web.FileResponse(HERE / 'static/tunnel.html')


def main():
    global ROS
    ap = argparse.ArgumentParser()
    # 8093: 8091 is the mission stack's QR HUD (qr_locator hud_port), which runs at the same time
    ap.add_argument('--port', type=int, default=8093)
    args = ap.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('FASTRTPS_DEFAULT_PROFILES_FILE', str(WS / 'fastdds_udp.xml'))
    ROS = Ros()
    ROS.start()
    PROCS.start()
    ROS.ready.wait(15)
    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/ws', ws_handler)
    app.router.add_get('/map.png', map_png)
    app.router.add_get('/camera.mjpg', mjpeg)
    app.router.add_post('/api/{what}', api)
    app.router.add_static('/static/', HERE / 'static')
    print(f'Tunnel + SLAM dashboard: http://10.42.0.247:{args.port}/  (localhost: http://127.0.0.1:{args.port}/)',
          flush=True)
    # Exit hard once shutdown starts: with browsers attached the cleanup took 5-8 s, then a C++ thread
    # (rclpy executor / OpenCV / serial) aborted the interpreter at exit ('terminate called without an
    # active exception', apport crash dump), so a stop + relaunch found the old process still on the port
    # (2026-10-27). Bounded to 2 s after SIGINT / SIGTERM.
    async def exit_soon(_app):
        threading.Timer(2.0, os._exit, args=(0,)).start()
    app.on_shutdown.append(exit_soon)
    web.run_app(app, host='0.0.0.0', port=args.port, print=None, shutdown_timeout=2.0)
    print('stopped', flush=True)
    os._exit(0)


if __name__ == '__main__':
    main()
