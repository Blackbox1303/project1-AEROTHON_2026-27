"""Costmap of the delivery zone from the Arducam: what has been searched, where the red zones are.

Grid over the zone rectangle (`zone` = [x_min, x_max, y_min, y_max] in odom, given on the day; the sim takes it
from the world file) plus a `pad` around it, `res` cells. Per Arducam frame (at most rate_hz): a lattice of
pixels -> fisheye rays (aerothon_nav/fisheye.py, the real lens model - a pinhole projection is wrong for a 175 deg
lens) -> odom (TF at the image stamp) -> the ground plane z = ground_z.
  explored  cells hit by rays within read_angle of straight down: where a QR on the ground can be read
            (25 deg = ~4.7 m around the drone at 10 m)
  red       HSV red pixels (two hue bands, as in the aeolus red_zone_detection) within red_angle of straight down
            vote for their cell; a cell is red after min_red_votes with red share >= red_share; red cells are
            grown by red_margin (GPS error + drone size: flying over a red zone is a violation)
  boundary  a ring of no-fly cells on the zone edge (the drone starts inside the zone)
  rejected  ground bounding boxes of QRs that don't match the entrance QR (qr_locator /delivery/qrs status
            `rejected`), grown by reject_margin, cost reject_cost: A* keeps off them and the frontier search doesn't
            go back. Below 100 (lethal) on purpose: the drone reads a QR from right above it, and Nav2 can't plan
            out of a lethal start cell (needs trinary_costmap: false, config/nav2_delivery.yaml)
Memory: red votes and rejected boxes survive /delivery/map/reset (only `explored` is forgotten) and are saved to
memory_file every save_period s (+ memory_file's .pgm/.yaml: the costmap in map_server format, for viewing). A
restarted node loads memory_file if its zone and resolution match; /delivery/memory/clear forgets it.
out: /delivery/map (nav_msgs/OccupancyGrid, odom, latched, 1 Hz): -1 unexplored, 0 explored, 100 red / boundary,
     reject_cost rejected QR;
     /delivery/red (OccupancyGrid, same grid: 100 = red zone itself, no margin - for "is this QR in a red zone");
     /delivery/coverage (std_msgs/String JSON: explored share of the zone, red cells, frames).
srv: /delivery/map/reset (std_srvs/Trigger): forget what was explored (start of the search), keep the memory;
     /delivery/memory/clear (std_srvs/Trigger): forget red zones and rejected QRs too, delete memory_file.
"""
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
import rclpy.parameter
import rclpy.time
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformListener

from aerothon_nav import qr_area
from aerothon_nav.banner_detector import quat_to_rot
from aerothon_nav.fisheye import Fisheye
from vio_bringup.spin import spin_node


class DeliveryMapper(Node):
    def __init__(self):
        super().__init__('delivery_mapper')
        p = self.declare_parameter
        p('zone', [15.0, 55.0, -15.0, 15.0])   # x_min, x_max, y_min, y_max in odom (given on the day)
        p('world_json', '')          # sim: take the zone from make_competition_world's ground truth
        p('pad', 2.0)                # m of free margin around the zone (outside the boundary ring)
        p('res', 0.25)
        p('rate_hz', 3.0)
        p('stride', 8)               # px between lattice points (1640 x 1232 -> ~31k rays)
        p('read_angle', 25.0)        # deg from straight down: explored = a QR there WILL be read. Must match the
        #                              camera that actually decodes: sim = the nadir_hd stand-in (+-20 deg) -> 18;
        #                              at 25 the target's cell was marked explored without being read (2026-10-19).
        #                              Real Arducam: measure on the hardware
        p('red_angle', 60.0)         # deg: red zones are big, look further
        p('ground_z', 0.0)
        p('min_red_votes', 3)
        p('red_share', 0.5)
        p('red_margin', 1.5)         # m
        p('red_h1', [0, 10])         # OpenCV hue bands of red, S and V minimums
        p('red_h2', [170, 180])
        p('red_s_min', 100)
        p('red_v_min', 70)
        p('odom_frame', 'odom')
        p('reject_margin', 0.5)      # m around a rejected QR's box
        p('reject_cost', 90)         # 1-99: expensive, not lethal (see the docstring)
        p('memory_file', os.path.expanduser('~/.ros/aerothon_delivery_memory.npz'))
        p('load_memory', True)       # False: start clean (e.g. a new world in sim)
        p('save_period', 5.0)        # s
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.poly = None             # the QR area polygon (ground station / /mission/qr_area); None: the zone rectangle
        if self.g('world_json'):
            w = json.load(open(self.g('world_json')))
            z = w['delivery_zone']
            self.set_parameters([rclpy.parameter.Parameter('zone', value=[float(z['x_from']), float(z['x_to']),
                                                                           float(z['y_from']), float(z['y_to'])])])
            if len(w.get('delivery_polygon') or []) >= 3:
                self.poly = [tuple(map(float, p)) for p in w['delivery_polygon']]
        self.cam = None
        self.last = 0.0
        self.frames = 0
        self.tf = Buffer()
        TransformListener(self.tf, self)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.map_pub = self.create_publisher(OccupancyGrid, '/delivery/map', latched)
        self.red_pub = self.create_publisher(OccupancyGrid, '/delivery/red', latched)
        self.cov_pub = self.create_publisher(String, '/delivery/coverage', 10)
        self.create_subscription(CameraInfo, '/arducam/camera_info', self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, '/arducam/image_raw', self.on_image, 1)   # RELIABLE: big frames
        self.create_subscription(String, '/delivery/qrs', self.on_qrs, 10)
        self.create_service(Trigger, '/delivery/map/reset', self.on_reset)
        self.create_service(Trigger, '/delivery/memory/clear', self.on_clear)
        self.create_subscription(String, qr_area.TOPIC, self.on_qr_area, qr_area.QOS)
        self.reset(memory=True)
        if self.g('load_memory'):
            self.load_memory()
        self.last_save = time.monotonic()
        self.create_timer(1.0, self.publish)

    # ---- grid
    def reset(self, memory=False):
        """Forget what was explored; memory=True also the remembered red votes and rejected QRs."""
        x0, x1, y0, y1 = self.g('zone')
        pad, r = self.g('pad'), self.g('res')
        self.ox, self.oy = x0 - pad, y0 - pad
        self.nx, self.ny = int(math.ceil((x1 - x0 + 2 * pad) / r)), int(math.ceil((y1 - y0 + 2 * pad) / r))
        self.explored = np.zeros((self.ny, self.nx), bool)
        if memory:
            self.red_votes = np.zeros((self.ny, self.nx), np.uint16)
            self.obs_votes = np.zeros((self.ny, self.nx), np.uint16)
            self.rejected = {}           # (payload, x, y rounded) -> box [x_min, y_min, x_max, y_max]
        xs = self.ox + (np.arange(self.nx) + 0.5) * r
        ys = self.oy + (np.arange(self.ny) + 0.5) * r
        inside = ((xs[None, :] >= x0) & (xs[None, :] <= x1) & (ys[:, None] >= y0) & (ys[:, None] <= y1))
        if self.poly:                                # a turned / cut QR area: its polygon, not the bounding box
            mask = np.zeros((self.ny, self.nx), np.uint8)
            cv2.fillPoly(mask, [np.round([((px - self.ox) / r - 0.5, (py - self.oy) / r - 0.5)
                                          for px, py in self.poly]).astype(np.int32)], 1)
            inside &= mask.astype(bool)
        ring = inside & ~cv2.erode(inside.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        self.inside, self.ring = inside, ring
        self.frames = 0

    def on_qr_area(self, m):
        """The supervisor's QR area (cut at the entry tunnel's exit in flight): a new grid; the red votes and rejected
        QRs are kept when the grid keeps its size (else forgotten: they were seen outside the QR area anyway)."""
        poly = [tuple(map(float, p)) for p in json.loads(m.data).get('polygon') or []]
        if len(poly) < 3 or poly == self.poly:
            return
        shape = (self.ny, self.nx)
        self.poly = poly
        self.set_parameters([rclpy.parameter.Parameter('zone', value=[float(v) for v in qr_area.bbox(poly)])])
        self.reset()
        if (self.ny, self.nx) != shape:
            self.reset(memory=True)
        self.get_logger().info(f'QR area: {len(poly)} corners, grid {self.nx} x {self.ny}')

    def on_reset(self, req, res):
        self.reset()
        res.success = True
        res.message = (f'map {self.nx} x {self.ny} cells reset; kept {len(self.rejected)} rejected QRs, '
                       f'{int(self.red_mask().sum())} red cells')
        return res

    def on_clear(self, req, res):
        self.reset(memory=True)
        Path(self.g('memory_file')).unlink(missing_ok=True)
        res.success, res.message = True, 'memory cleared'
        return res

    # ---- rejected QRs (qr_locator)
    def on_qrs(self, m):
        try:
            qrs = json.loads(m.data)['qrs']
        except (ValueError, KeyError):
            return
        for q in qrs:
            if q.get('status') != 'rejected' or len(q.get('box') or []) != 4:
                continue
            # one entry per QR: replace one of the same payload whose centre is within 2 m (the box is refined)
            key = next((k for k in self.rejected if k[0] == q['payload'] and math.dist(k[1:], (q['x'], q['y'])) < 2.0),
                       None)
            if key is None:
                key = (q['payload'], round(q['x'], 1), round(q['y'], 1))
                self.get_logger().info(f'rejected QR {q["payload"]!r} at ({q["x"]:.1f}, {q["y"]:.1f}): in the costmap')
            self.rejected[key] = [float(v) for v in q['box']]

    def rejected_mask(self):
        mask = np.zeros((self.ny, self.nx), np.uint8)
        r, k = self.g('res'), self.g('reject_margin')
        for x0, y0, x1, y1 in self.rejected.values():
            i0, i1 = int(math.floor((x0 - k - self.ox) / r)), int(math.floor((x1 + k - self.ox) / r))
            j0, j1 = int(math.floor((y0 - k - self.oy) / r)), int(math.floor((y1 + k - self.oy) / r))
            cv2.rectangle(mask, (i0, j0), (i1, j1), 1, -1)   # clipped to the grid
        return mask.astype(bool)

    # ---- memory
    def save_memory(self, data):
        f = Path(self.g('memory_file'))
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            tmp = f.with_name(f.stem + '.tmp.npz')
            np.savez(tmp, zone=np.array(self.g('zone'), float), res=float(self.g('res')),
                     red_votes=self.red_votes, obs_votes=self.obs_votes,
                     rejected=json.dumps([[*k, v] for k, v in self.rejected.items()]))
            tmp.replace(f)
            # map_server format (row 0 = top = max y): free 254, occupied 0, unknown 205
            img = np.full(data.shape, 205, np.uint8)
            img[data == 0], img[data > 0] = 254, 0
            cv2.imwrite(str(f.with_suffix('.pgm')), img[::-1])
            f.with_suffix('.yaml').write_text(
                f'image: {f.with_suffix(".pgm").name}\nresolution: {self.g("res")}\n'
                f'origin: [{self.ox}, {self.oy}, 0.0]\nnegate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')
        except OSError as e:
            self.get_logger().error(f'could not save the memory: {e}', throttle_duration_sec=30.0)

    def load_memory(self):
        f = Path(self.g('memory_file'))
        if not f.exists():
            return
        try:
            d = np.load(f)
            if (not np.allclose(d['zone'], self.g('zone')) or float(d['res']) != float(self.g('res'))
                    or d['red_votes'].shape != (self.ny, self.nx)):
                self.get_logger().warn(f'{f}: another zone / resolution, not loaded')
                return
            self.red_votes, self.obs_votes = d['red_votes'].astype(np.uint16), d['obs_votes'].astype(np.uint16)
            self.rejected = {(p, x, y): box for p, x, y, box in json.loads(str(d['rejected']))}
            self.get_logger().info(f'memory from {f}: {len(self.rejected)} rejected QRs, '
                                   f'{int(self.red_mask().sum())} red cells')
        except Exception as e:
            self.get_logger().error(f'could not load {f}: {e}')

    def cells(self, pts):
        ix = np.floor((pts[:, 0] - self.ox) / self.g('res')).astype(int)
        iy = np.floor((pts[:, 1] - self.oy) / self.g('res')).astype(int)
        ok = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        return iy[ok], ix[ok], ok

    # ---- camera
    def on_info(self, m):
        if self.cam is not None and self.cam.size == (m.width, m.height):
            return
        self.cam = Fisheye(m)
        s = self.g('stride')
        u, v = np.meshgrid(np.arange(s // 2, m.width, s), np.arange(s // 2, m.height, s))
        self.lat_px = np.column_stack([u.ravel(), v.ravel()])
        self.lat_rays = self.cam.rays(self.lat_px)
        self.get_logger().info(f'camera {m.width}x{m.height}: {len(self.lat_px)} lattice rays')

    def on_image(self, m):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.cam is None or now - self.last < 1.0 / self.g('rate_hz'):
            return
        self.last = now
        try:
            t = self.tf.lookup_transform(self.g('odom_frame'), m.header.frame_id, m.header.stamp,
                                         timeout=Duration(seconds=0.1))
        except Exception:
            try:
                t = self.tf.lookup_transform(self.g('odom_frame'), m.header.frame_id, rclpy.time.Time())
            except Exception as e:
                self.get_logger().warn(f'no TF for the Arducam: {e}', throttle_duration_sec=5.0)
                return
        R = quat_to_rot(t.transform.rotation)
        o = t.transform.translation
        rays = self.lat_rays @ R.T
        down = -rays[:, 2]                               # cos of the angle from straight down
        use = down > math.cos(math.radians(self.g('red_angle')))
        if not use.any():
            return
        tt = (o.z - self.g('ground_z')) / down[use]
        pts = np.array([o.x, o.y]) + rays[use, :2] * tt[:, None]
        # red pixels at the lattice points
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        px = self.lat_px[use]
        bgr = img[px[:, 1], px[:, 0], :3].reshape(-1, 1, 3)
        if m.encoding == 'rgb8':
            bgr = bgr[..., ::-1]
        hsv = cv2.cvtColor(np.ascontiguousarray(bgr), cv2.COLOR_BGR2HSV).reshape(-1, 3)
        h, s_, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
        (a1, b1), (a2, b2) = self.g('red_h1'), self.g('red_h2')
        red = (((h >= a1) & (h <= b1)) | ((h >= a2) & (h <= b2))) & (s_ >= self.g('red_s_min')) & (v >= self.g('red_v_min'))
        iy, ix, ok = self.cells(pts)
        np.add.at(self.obs_votes, (iy, ix), 1)
        np.add.at(self.red_votes, (iy, ix), red[ok].astype(np.uint16))
        near = down[use][ok] > math.cos(math.radians(self.g('read_angle')))
        self.explored[iy[near], ix[near]] = True
        self.frames += 1

    # ---- output
    def red_mask(self):
        return (self.red_votes >= self.g('min_red_votes')) & \
               (self.red_votes >= self.g('red_share') * np.maximum(self.obs_votes, 1))

    def grid(self):
        red = self.red_mask()
        k = int(math.ceil(self.g('red_margin') / self.g('res')))
        if k > 0 and red.any():
            disk = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
            red_grown = cv2.dilate(red.astype(np.uint8), disk).astype(bool)
        else:
            red_grown = red
        data = np.full((self.ny, self.nx), -1, np.int8)
        data[self.explored] = 0
        data[~self.inside & ~self.ring] = 0               # the pad: known, free (never a frontier)
        data[self.rejected_mask()] = self.g('reject_cost')
        data[red_grown | self.ring] = 100
        return data, red

    def publish(self):
        data, red = self.grid()
        m = OccupancyGrid()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.g('odom_frame')
        m.info.resolution = float(self.g('res'))
        m.info.width, m.info.height = self.nx, self.ny
        m.info.origin.position.x, m.info.origin.position.y = float(self.ox), float(self.oy)
        m.info.origin.orientation.w = 1.0
        m.data = data.ravel().tolist()
        self.map_pub.publish(m)
        m.data = np.where(red, 100, 0).astype(np.int8).ravel().tolist()
        self.red_pub.publish(m)
        zone = self.inside & ~self.ring
        self.cov_pub.publish(String(data=json.dumps({
            'explored': round(float((self.explored & zone).sum()) / max(1, zone.sum()), 3),
            'red_cells': int(red.sum()), 'rejected_qrs': len(self.rejected), 'frames': self.frames})))
        if time.monotonic() - self.last_save > self.g('save_period'):
            self.last_save = time.monotonic()
            self.save_memory(data)


def main():
    spin_node(DeliveryMapper)


if __name__ == '__main__':
    main()
