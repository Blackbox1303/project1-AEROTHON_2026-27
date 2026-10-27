#!/usr/bin/env python3
"""Simulate the drone flying through a saved RealSense SLAM map + 3D point cloud (runs/saved_tunnel_map/saved_map.npz).

Publishes:
  - /map (nav_msgs/OccupancyGrid, latched): the saved 2D SLAM map from the real Intel RealSense scan
  - /rtabmap/cloud_map (sensor_msgs/PointCloud2): the saved 3D obstacle point cloud in `map` frame (for RViz2)
  - /vio/obstacle_scan (sensor_msgs/LaserScan): live 90 deg FOV raycast of the saved map/3D cloud from the sim drone
  - /vio/depth_points (sensor_msgs/PointCloud2): live 3D obstacle points in the sim drone's camera frustum
  - /camera/depth/image_rect_raw (sensor_msgs/Image): synthetic depth view of the saved 3D obstacles from the sim drone
  - /nav/odom (nav_msgs/Odometry) + TF map -> odom -> base_link -> imu_link -> scan_level
  - /nav/planned_path (nav_msgs/Path): A* obstacle-avoidance path through the saved map (for RViz2 + Dashboard)
  - /nav/corridor/state (std_msgs/String) & /nav/altitude (std_msgs/Float32)

Accepts velocity commands on /cmd_vel + /nav/altitude (from corridor_nav / altitude_planner), OR runs its own
built-in A* + 3D obstacle avoidance navigator via service /sim_map/navigate (std_srvs/Trigger) or /nav/corridor/run.
"""
import heapq
import json
import math
import os
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped, TransformStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path as NavPath
from sensor_msgs.msg import Image, LaserScan, PointCloud2, PointField
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger
from tf2_ros import TransformBroadcaster

WS = Path.home() / 'workspaces/isaac_ros-dev'
DEFAULT_MAP = WS / 'runs/saved_tunnel_map/saved_map.npz'


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def euler_to_quat(roll, pitch, yaw):
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class SimSavedMapDrone(Node):
    def __init__(self):
        super().__init__('sim_saved_map_drone')
        self.declare_parameter('map_file', str(DEFAULT_MAP))
        self.declare_parameter('clearance', 0.35)      # m obstacle inflation radius for A*
        self.declare_parameter('speed', 0.65)          # m/s cruise speed
        self.declare_parameter('nominal_z', 1.2)       # m nominal flight height

        self.map_file = Path(self.get_parameter('map_file').value)
        self.clearance = float(self.get_parameter('clearance').value)
        self.speed = float(self.get_parameter('speed').value)
        self.nominal_z = float(self.get_parameter('nominal_z').value)

        self.load_saved_map()

        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', latched)
        self.cloud_map_pub = self.create_publisher(PointCloud2, '/rtabmap/cloud_map', latched)
        self.scan_pub = self.create_publisher(LaserScan, '/vio/obstacle_scan', qos_profile_sensor_data)
        self.depth_pts_pub = self.create_publisher(PointCloud2, '/vio/depth_points', qos_profile_sensor_data)
        self.depth_img_pub = self.create_publisher(Image, '/camera/depth/image_rect_raw', qos_profile_sensor_data)
        self.odom_pub = self.create_publisher(Odometry, '/nav/odom', 10)
        self.path_pub = self.create_publisher(NavPath, '/nav/planned_path', latched)
        self.path_json_pub = self.create_publisher(String, '/sim_map/path_json', latched)
        self.state_pub = self.create_publisher(String, '/nav/corridor/state', 10)
        self.alt_pub = self.create_publisher(Float32, '/nav/altitude', 10)
        self.alt_state_pub = self.create_publisher(String, '/nav/altitude/state', 10)
        self.tf_pub = TransformBroadcaster(self)

        # External control inputs (if corridor_nav is driving)
        self.ext_cmd = None
        self.ext_cmd_t = 0.0
        self.ext_alt = None
        self.create_subscription(Twist, '/cmd_vel', self.on_cmd_vel, 10)
        self.create_subscription(Float32, '/nav/altitude', self.on_ext_alt, 10)
        self.create_subscription(String, '/sim_map/set_goal', self.on_set_goal, 10)

        self.create_service(Trigger, '/sim_map/navigate', self.srv_navigate)
        self.create_service(Trigger, '/sim_map/stop', self.srv_stop)
        self.create_service(Trigger, '/sim_map/reset', self.srv_reset)

        self.nav_active = False
        self.waypoints = []          # [(x, y, z), ...]
        self.wp_idx = 0
        self.state_str = 'READY (Saved RealSense Map loaded)'

        # Publish initial latched map + 3D cloud
        self.publish_static_maps()
        self.create_timer(1.5, self.publish_static_maps)
        self.create_timer(1.0 / 25.0, self.step_sim)
        self.create_timer(1.0 / 12.0, self.step_sensors)

    def load_saved_map(self):
        if self.map_file.is_file():
            data = np.load(str(self.map_file), allow_pickle=True)
            self.grid = data['grid'].astype(np.int8)
            origin = data['origin']
            self.ox, self.oy, self.res = float(origin[0]), float(origin[1]), float(origin[2])
            self.points = data['points'].astype(np.float32) if 'points' in data and len(data['points']) else np.zeros((0, 3), np.float32)
            sp = data['start_pose'] if 'start_pose' in data else [0.0, 0.0, self.nominal_z, 0.0]
            ep = data['end_pose'] if 'end_pose' in data else [4.0, 0.0, self.nominal_z, 0.0]
            self.start_pose = [float(sp[0]), float(sp[1]), float(sp[2]) if len(sp) > 2 else self.nominal_z,
                               float(sp[3]) if len(sp) > 3 else 0.0]
            self.goal_pose = [float(ep[0]), float(ep[1]), float(ep[2]) if len(ep) > 2 else self.nominal_z,
                              float(ep[3]) if len(ep) > 3 else 0.0]
            if math.hypot(self.goal_pose[0] - self.start_pose[0], self.goal_pose[1] - self.start_pose[1]) < 0.8:
                # Pick furthest free point in the map as default goal
                free_j, free_i = np.nonzero(self.grid == 0)
                if len(free_i):
                    wx = self.ox + (free_i + 0.5) * self.res
                    wy = self.oy + (free_j + 0.5) * self.res
                    d = np.hypot(wx - self.start_pose[0], wy - self.start_pose[1])
                    idx = int(np.argmax(d))
                    self.goal_pose[0], self.goal_pose[1] = float(wx[idx]), float(wy[idx])
            self.get_logger().info(
                f'Loaded saved RealSense map {self.map_file}: grid {self.grid.shape[1]}x{self.grid.shape[0]} '
                f'@ {self.res}m, {len(self.points)} 3D pts')
        else:
            # Synthesize a fallback tunnel with obstacles if no file saved yet
            self.res = 0.05
            self.ox, self.oy = -1.5, -2.5
            w, h = int(12.0 / self.res), int(5.0 / self.res)
            self.grid = np.zeros((h, w), np.int8)
            # Tunnel walls at y = +-1.6 m and two staggered obstacles
            j_top = int((1.6 - self.oy) / self.res)
            j_bot = int((-1.6 - self.oy) / self.res)
            self.grid[j_top:min(h, j_top + 3), :] = 100
            self.grid[max(0, j_bot - 3):j_bot, :] = 100
            # Obstacle 1 at x=3.0, y=0.4
            self.grid[int((0.1 - self.oy) / self.res):int((0.9 - self.oy) / self.res),
                      int((2.7 - self.ox) / self.res):int((3.3 - self.ox) / self.res)] = 100
            # Obstacle 2 at x=6.0, y=-0.45
            self.grid[int((-0.9 - self.oy) / self.res):int((-0.1 - self.oy) / self.res),
                      int((5.7 - self.ox) / self.res):int((6.3 - self.ox) / self.res)] = 100
            obs_j, obs_i = np.nonzero(self.grid >= 50)
            wx = self.ox + (obs_i + 0.5) * self.res
            wy = self.oy + (obs_j + 0.5) * self.res
            zs = np.linspace(0.1, 2.0, 8)
            pts = np.stack([np.repeat(wx, len(zs)), np.repeat(wy, len(zs)), np.tile(zs, len(wx))], axis=-1)
            self.points = pts.astype(np.float32)
            self.start_pose = [0.0, 0.0, self.nominal_z, 0.0]
            self.goal_pose = [8.5, 0.0, self.nominal_z, 0.0]

        # Build 3D points from 2D occupied cells if 3D point cloud is sparse
        if len(self.points) < 200:
            obs_j, obs_i = np.nonzero(self.grid >= 50)
            if len(obs_i):
                wx = self.ox + (obs_i + 0.5) * self.res
                wy = self.oy + (obs_j + 0.5) * self.res
                zs = np.linspace(0.15, 2.0, 6)
                pts = np.stack([np.repeat(wx, len(zs)), np.repeat(wy, len(zs)), np.tile(zs, len(wx))], axis=-1)
                self.points = np.vstack([self.points, pts.astype(np.float32)]) if len(self.points) else pts.astype(np.float32)

        self.x, self.y, self.z, self.yaw = self.start_pose
        self.vx, self.vy, self.vz, self.vyaw = 0.0, 0.0, 0.0, 0.0

    def publish_static_maps(self):
        stamp = self.get_clock().now().to_msg()
        m = OccupancyGrid()
        m.header.stamp = stamp
        m.header.frame_id = 'map'
        m.info.resolution = self.res
        m.info.width = self.grid.shape[1]
        m.info.height = self.grid.shape[0]
        m.info.origin.position.x = self.ox
        m.info.origin.position.y = self.oy
        m.info.origin.orientation.w = 1.0
        m.data = self.grid.flatten().tolist()
        self.map_pub.publish(m)

        if len(self.points):
            pc = PointCloud2()
            pc.header.stamp = stamp
            pc.header.frame_id = 'map'
            pts = self.points[:40000]
            pc.height, pc.width = 1, len(pts)
            # Pack x, y, z + rgb (height-colored)
            z_norm = np.clip((pts[:, 2] - 0.1) / 2.2, 0.0, 1.0)
            r = (50 + 205 * z_norm).astype(np.uint32)
            g = (220 - 100 * np.abs(z_norm - 0.5)).astype(np.uint32)
            b = (255 - 180 * z_norm).astype(np.uint32)
            rgb_uint = (r << 16) | (g << 8) | b
            rgb_float = rgb_uint.view(np.float32)
            packed = np.column_stack([pts[:, :3], rgb_float]).astype(np.float32)
            pc.fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            pc.is_bigendian, pc.is_dense = False, True
            pc.point_step, pc.row_step = 16, 16 * len(pts)
            pc.data = packed.tobytes()
            self.cloud_map_pub.publish(pc)

    def plan_path(self, start_xy, goal_xy):
        """A* path on the saved 2D SLAM map + 3D height planning over low obstacles."""
        h, w = self.grid.shape
        lethal = (self.grid >= 50).astype(np.uint8)
        if lethal.any():
            dist_m = cv2.distanceTransform(1 - lethal, cv2.DIST_L2, 3) * self.res
        else:
            dist_m = np.full((h, w), 10.0, np.float32)

        # Try full clearance first, then relax if narrow tunnel
        for clr in (self.clearance, 0.22, 0.12, 0.05):
            blocked = dist_m < clr
            wps = self._astar(blocked, dist_m, start_xy, goal_xy)
            if wps:
                break
        else:
            wps = [start_xy, goal_xy]

        # Assign 3D height z to each waypoint (climb over low 3D obstacles if any lie near the path)
        wps_3d = []
        for wx, wy in wps:
            z_target = self.nominal_z
            if len(self.points):
                near = self.points[np.hypot(self.points[:, 0] - wx, self.points[:, 1] - wy) < 0.45]
                if len(near):
                    max_obs_z = float(np.percentile(near[:, 2], 90))
                    if 0.25 < max_obs_z < 1.65:
                        z_target = min(2.15, max_obs_z + 0.40)
            wps_3d.append((round(wx, 3), round(wy, 3), round(z_target, 2)))
        return wps_3d

    def _astar(self, blocked, dist_m, start_xy, goal_xy):
        h, w = blocked.shape
        cell = lambda p: (min(h - 1, max(0, int((p[1] - self.oy) / self.res))),
                          min(w - 1, max(0, int((p[0] - self.ox) / self.res))))
        s, g = cell(start_xy), cell(goal_xy)

        # Snap start/goal to nearest unblocked cell if needed
        for name, c in (('s', s), ('g', g)):
            if blocked[c]:
                js, iis = np.nonzero(~blocked)
                if not len(js):
                    return None
                idx = int(np.argmin((js - c[0]) ** 2 + (iis - c[1]) ** 2))
                if name == 's':
                    s = (int(js[idx]), int(iis[idx]))
                else:
                    g = (int(js[idx]), int(iis[idx]))

        moves = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
                 (-1, -1, 1.414), (-1, 1, 1.414), (1, -1, 1.414), (1, 1, 1.414)]
        best = {s: 0.0}
        came = {}
        q = [(math.hypot(s[0] - g[0], s[1] - g[1]), 0.0, s)]
        while q:
            _, cost, c = heapq.heappop(q)
            if c == g:
                break
            if cost > best.get(c, math.inf):
                continue
            for dj, di, dc in moves:
                nj, ni = c[0] + dj, c[1] + di
                if not (0 <= nj < h and 0 <= ni < w) or blocked[nj, ni]:
                    continue
                # Extra penalty near walls so the drone stays centred in the tunnel
                wall_pen = max(0.0, (0.8 - float(dist_m[nj, ni]))) * 1.8
                nc = cost + dc * (1.0 + wall_pen)
                nxt = (nj, ni)
                if nc < best.get(nxt, math.inf):
                    best[nxt] = nc
                    came[nxt] = c
                    heapq.heappush(q, (nc + math.hypot(nj - g[0], ni - g[1]), nc, nxt))
        else:
            return None

        cells = [g]
        while cells[-1] != s and cells[-1] in came:
            cells.append(came[cells[-1]])
        cells.reverse()
        pts = [(self.ox + (c[1] + 0.5) * self.res, self.oy + (c[0] + 0.5) * self.res) for c in cells]
        # Subsample every ~0.35 m for smooth trajectory tracking
        out = [pts[0]]
        for p in pts[1:-1]:
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= 0.35:
                out.append(p)
        if len(pts) > 1:
            out.append(pts[-1])
        return out

    def publish_planned_path(self):
        stamp = self.get_clock().now().to_msg()
        np_msg = NavPath()
        np_msg.header.stamp = stamp
        np_msg.header.frame_id = 'map'
        for wx, wy, wz in self.waypoints:
            ps = PoseStamped()
            ps.header = np_msg.header
            ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = float(wx), float(wy), float(wz)
            ps.pose.orientation.w = 1.0
            np_msg.poses.append(ps)
        self.path_pub.publish(np_msg)
        self.path_json_pub.publish(String(data=json.dumps({
            'waypoints': self.waypoints,
            'start': self.start_pose[:2],
            'goal': self.goal_pose[:2],
        })))

    def on_set_goal(self, m):
        try:
            d = json.loads(m.data)
            if 'start_x' in d and 'start_y' in d:
                self.start_pose[0], self.start_pose[1] = float(d['start_x']), float(d['start_y'])
                self.x, self.y = self.start_pose[0], self.start_pose[1]
            if 'goal_x' in d and 'goal_y' in d:
                self.goal_pose[0], self.goal_pose[1] = float(d['goal_x']), float(d['goal_y'])
            if d.get('navigate', True):
                self.start_navigation()
        except Exception as e:
            self.get_logger().warning(f'on_set_goal error: {e}')

    def start_navigation(self):
        self.waypoints = self.plan_path((self.x, self.y), (self.goal_pose[0], self.goal_pose[1]))
        self.wp_idx = 0
        self.nav_active = True
        self.state_str = f'NAVIGATING SAVED MAP (0/{len(self.waypoints)} wps)'
        self.publish_planned_path()

    def srv_navigate(self, req, res):
        self.start_navigation()
        res.success = True
        res.message = f'Sim drone navigating saved map ({len(self.waypoints)} waypoints to goal ({self.goal_pose[0]:.2f}, {self.goal_pose[1]:.2f}))'
        return res

    def srv_stop(self, req, res):
        self.nav_active = False
        self.vx = self.vy = self.vz = self.vyaw = 0.0
        self.state_str = 'HOLD (stopped)'
        res.success, res.message = True, 'Sim drone stopped'
        return res

    def srv_reset(self, req, res):
        self.nav_active = False
        self.load_saved_map()
        self.waypoints = self.plan_path((self.x, self.y), (self.goal_pose[0], self.goal_pose[1]))
        self.publish_static_maps()
        self.publish_planned_path()
        self.state_str = 'READY (reset to start)'
        res.success, res.message = True, 'Sim drone reset to start of saved map'
        return res

    def on_cmd_vel(self, m):
        self.ext_cmd = m
        self.ext_cmd_t = self.get_clock().now().nanoseconds * 1e-9

    def on_ext_alt(self, m):
        if math.isfinite(m.data):
            self.ext_alt = float(m.data)

    def step_sim(self):
        dt = 1.0 / 25.0
        now_s = self.get_clock().now().nanoseconds * 1e-9

        if self.nav_active and self.waypoints:
            tx, ty, tz = self.waypoints[min(self.wp_idx, len(self.waypoints) - 1)]
            dx, dy = tx - self.x, ty - self.y
            dist = math.hypot(dx, dy)
            if dist < 0.25 and self.wp_idx < len(self.waypoints) - 1:
                self.wp_idx += 1
                tx, ty, tz = self.waypoints[self.wp_idx]
                dx, dy = tx - self.x, ty - self.y
                dist = math.hypot(dx, dy)
            elif dist < 0.20 and self.wp_idx >= len(self.waypoints) - 1:
                self.nav_active = False
                self.vx = self.vy = self.vz = self.vyaw = 0.0
                self.state_str = f'DONE (arrived at ({self.x:.2f}, {self.y:.2f}))'

            if self.nav_active:
                des_yaw = math.atan2(dy, dx) if dist > 0.15 else self.yaw
                spd = min(self.speed, max(0.18, dist * 1.2))
                vx_w = (dx / max(dist, 1e-3)) * spd
                vy_w = (dy / max(dist, 1e-3)) * spd

                # Reactive repulsion from nearby occupied cells in the saved map
                if len(self.points):
                    near = self.points[(np.abs(self.points[:, 2] - self.z) < 0.35) &
                                       (np.hypot(self.points[:, 0] - self.x, self.points[:, 1] - self.y) < 0.75)]
                    if len(near):
                        rx = np.mean(self.x - near[:, 0])
                        ry = np.mean(self.y - near[:, 1])
                        rn = math.hypot(rx, ry)
                        if rn > 1e-3:
                            vx_w += (rx / rn) * 0.25
                            vy_w += (ry / rn) * 0.25

                self.x += vx_w * dt
                self.y += vy_w * dt
                self.z += np.clip((tz - self.z) * 1.8, -0.8, 0.8) * dt
                self.yaw = wrap(self.yaw + np.clip(wrap(des_yaw - self.yaw) * 2.5, -1.2, 1.2) * dt)
                self.alt_pub.publish(Float32(data=float(tz)))
                self.state_str = f'NAVIGATING SAVED MAP wp {self.wp_idx + 1}/{len(self.waypoints)} (z={self.z:.2f}m)'

        elif self.ext_cmd is not None and (now_s - self.ext_cmd_t) < 0.5:
            # Driven externally by corridor_nav (/cmd_vel in body FLU frame)
            c, s = math.cos(self.yaw), math.sin(self.yaw)
            bx, by = float(self.ext_cmd.linear.x), float(self.ext_cmd.linear.y)
            vx_w = c * bx - s * by
            vy_w = s * bx + c * by
            self.x += vx_w * dt
            self.y += vy_w * dt
            tz = self.ext_alt if self.ext_alt is not None else self.nominal_z
            self.z += np.clip((tz - self.z) * 1.5, -0.8, 0.8) * dt
            self.yaw = wrap(self.yaw + float(self.ext_cmd.angular.z) * dt)

        # Broadcast TF map -> odom -> base_link -> imu_link -> scan_level and /nav/odom
        stamp = self.get_clock().now().to_msg()
        qx, qy, qz, qw = euler_to_quat(0.0, 0.0, self.yaw)

        t_mo = TransformStamped()
        t_mo.header.stamp, t_mo.header.frame_id, t_mo.child_frame_id = stamp, 'map', 'odom'
        t_mo.transform.rotation.w = 1.0

        t_ob = TransformStamped()
        t_ob.header.stamp, t_ob.header.frame_id, t_ob.child_frame_id = stamp, 'odom', 'base_link'
        t_ob.transform.translation.x = float(self.x)
        t_ob.transform.translation.y = float(self.y)
        t_ob.transform.translation.z = float(self.z)
        t_ob.transform.rotation.x, t_ob.transform.rotation.y, t_ob.transform.rotation.z, t_ob.transform.rotation.w = qx, qy, qz, qw

        t_bi = TransformStamped()
        t_bi.header.stamp, t_bi.header.frame_id, t_bi.child_frame_id = stamp, 'base_link', 'imu_link'
        t_bi.transform.rotation.w = 1.0

        t_is = TransformStamped()
        t_is.header.stamp, t_is.header.frame_id, t_is.child_frame_id = stamp, 'imu_link', 'scan_level'
        t_is.transform.rotation.w = 1.0

        self.tf_pub.sendTransform([t_mo, t_ob, t_bi, t_is])

        od = Odometry()
        od.header.stamp, od.header.frame_id, od.child_frame_id = stamp, 'odom', 'base_link'
        od.pose.pose.position.x, od.pose.pose.position.y, od.pose.pose.position.z = float(self.x), float(self.y), float(self.z)
        od.pose.pose.orientation.x, od.pose.pose.orientation.y, od.pose.pose.orientation.z, od.pose.pose.orientation.w = qx, qy, qz, qw
        self.odom_pub.publish(od)
        self.state_pub.publish(String(data=self.state_str))

    def step_sensors(self):
        """Raycast the saved RealSense 3D point cloud & 2D grid from the sim drone's pose."""
        stamp = self.get_clock().now().to_msg()
        fov = math.radians(90.0)
        nbins = 90
        ang_inc = fov / nbins
        ang_min = -fov / 2.0
        ranges = np.full(nbins, np.inf, np.float32)

        c, s = math.cos(-self.yaw), math.sin(-self.yaw)
        body_pts = np.zeros((0, 3), np.float32)

        if len(self.points):
            dx = self.points[:, 0] - self.x
            dy = self.points[:, 1] - self.y
            dz = self.points[:, 2] - self.z
            bx = c * dx - s * dy
            by = s * dx + c * dy
            bz = dz
            in_frustum = (bx > 0.18) & (bx < 6.0) & (np.abs(by) < bx * 1.05) & (np.abs(bz) < bx * 0.65)
            body_pts = np.column_stack([bx[in_frustum], by[in_frustum], bz[in_frustum]]).astype(np.float32)

            # Slice within +-0.25m of drone height for 2D LaserScan
            band_pts = body_pts[np.abs(body_pts[:, 2]) < 0.25]
            if len(band_pts):
                rng = np.hypot(band_pts[:, 0], band_pts[:, 1])
                ang = np.arctan2(band_pts[:, 1], band_pts[:, 0])
                bins = np.floor((ang - ang_min) / ang_inc).astype(int)
                valid = (bins >= 0) & (bins < nbins) & (rng >= 0.2) & (rng <= 6.0)
                for b_i, r_i in zip(bins[valid], rng[valid]):
                    if r_i < ranges[b_i]:
                        ranges[b_i] = r_i

        scan = LaserScan()
        scan.header.stamp, scan.header.frame_id = stamp, 'scan_level'
        scan.angle_min = ang_min
        scan.angle_max = ang_min + ang_inc * (nbins - 1)
        scan.angle_increment = ang_inc
        scan.range_min, scan.range_max = 0.2, 6.0
        scan.ranges = ranges.tolist()
        self.scan_pub.publish(scan)

        # Publish /vio/depth_points in imu_link frame
        pc = PointCloud2()
        pc.header.stamp, pc.header.frame_id = stamp, 'imu_link'
        pts_sub = body_pts[:15000]
        pc.height, pc.width = 1, len(pts_sub)
        pc.fields = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate('xyz')]
        pc.is_bigendian, pc.is_dense = False, True
        pc.point_step, pc.row_step = 12, 12 * len(pts_sub)
        pc.data = pts_sub.tobytes()
        self.depth_pts_pub.publish(pc)

        # Render synthetic 16UC1 depth image from body_pts so the dashboard camera view shows the sim drone's view
        w_img, h_img = 320, 180
        depth_mm = np.zeros((h_img, w_img), np.uint16)
        if len(pts_sub):
            fx, fy, cx, cy = 160.0, 160.0, 160.0, 90.0
            u = np.round(cx - pts_sub[:, 1] / pts_sub[:, 0] * fx).astype(int)
            v = np.round(cy - pts_sub[:, 2] / pts_sub[:, 0] * fy).astype(int)
            ok = (u >= 0) & (u < w_img) & (v >= 0) & (v < h_img)
            u, v, z_mm = u[ok], v[ok], (pts_sub[ok, 0] * 1000.0).clip(200, 6000).astype(np.uint16)
            order = np.argsort(-z_mm)   # nearer overwrites further
            for ui, vi, zi in zip(u[order], v[order], z_mm[order]):
                depth_mm[max(0, vi - 2):min(h_img, vi + 3), max(0, ui - 2):min(w_img, ui + 3)] = zi
        img_msg = Image()
        img_msg.header.stamp, img_msg.header.frame_id = stamp, 'camera_depth_optical_frame'
        img_msg.height, img_msg.width = h_img, w_img
        img_msg.encoding = '16UC1'
        img_msg.step = w_img * 2
        img_msg.data = depth_mm.tobytes()
        self.depth_img_pub.publish(img_msg)

        self.alt_state_pub.publish(String(data=json.dumps({
            'mode': 'sim_saved_map',
            'z': round(self.z, 2),
            'pts_in_view': int(len(pts_sub)),
        })))


def main():
    rclpy.init()
    node = SimSavedMapDrone()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
