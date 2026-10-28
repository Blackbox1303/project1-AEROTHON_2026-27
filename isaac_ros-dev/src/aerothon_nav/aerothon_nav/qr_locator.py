"""Read every QR in the delivery zone, find where it lies on the ground, and name the delivery target.

Upgraded with the complete `drone_qr_detector` pipeline:
  1. Circular fisheye active-area masking (`circle_mask` at 82 deg) to eliminate corner vignette false positives.
  2. Multi-scale `QrNet` bounding-box search (160, 224, 384 px tiles) + dark-square contour refinement (`refine_qr_box`).
  3. Virtual camera adjustment ("searching QR bounding box, then adjusting camera to scan"):
     - Steers a virtual gnomonic pinhole patch (`rectify_patch`, fov=22 deg, zoom=2.5x) directly along each
       detected QR bounding box's 3D ray to remove fisheye barrel distortion before decoding.
     - Stacks and aligns consecutive crops (`aligned_mean`) over a short window to average out sensor noise,
       and triggers `/arducam/qr_boost` if a boxed QR remains unread.
  4. Short-window temporal averaging (`avg_window_s`, `avg_window_samples`) with `cos^4(theta)` confidence
     weighting and median outlier rejection for both QR ground coordinates `(x, y)` and pairwise inter-QR distances.
  5. Strict start-QR payload matching: `/delivery/target` (which triggers the `DELIVER` -> `DROP` winch mechanism
     in `mission_supervisor`) is published ONLY when a confirmed QR inside the delivery zone (and outside red zones)
     has `payload == start_qr`.
  ArUco mission (parameter `marker` 'aruco', default the AEROTHON_MARKER environment variable): the boards carry
     ArUco markers 3x3 .. 6x6 instead; they're found on undistorted gnomonic views of the fisheye
     (aruco_tools.detect_fisheye, straight on the sim's pinhole nadir camera) and the payload is e.g. '4X4#23' -
     everything after the read (ground position, averaging, zone / red checks, the start match) is the same.
  6. Live HTTP QR HUD on port `8091` (`/`, `/annotated.jpg`, `/frame.jpg`, `/status.json`), integrated into the
     main mission dashboard at port `8080`.
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
import rclpy.executors
import rclpy.time
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener

from aerothon_nav.banner_detector import quat_to_rot
from aerothon_nav.fisheye import Fisheye
from aerothon_nav.qr_reader import to_gray
from aerothon_nav import qr_tools


HUD_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Drone QR Detector HUD (:8091)</title>
<style>
  body { margin:0; background:#0f1318; color:#dfe6ee; font:13px/1.4 system-ui, sans-serif; padding:12px; }
  .top { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-bottom:10px; }
  .pill { padding:4px 10px; border-radius:12px; background:#1e2631; font-weight:600; }
  .ok { background:#163323; color:#3fbf6f; } .warn { background:#3a2a12; color:#f1a33b; } .bad { background:#3a1517; color:#e5484d; }
  .grid { display:grid; grid-template-columns: minmax(0, 1fr) 380px; gap:12px; }
  .card { background:#171d24; border:1px solid #2a333d; border-radius:8px; padding:10px; }
  .card h2 { font-size:12px; text-transform:uppercase; color:#8b98a6; margin:0 0 8px; }
  img { width:100%; border-radius:6px; background:#000; display:block; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th, td { text-align:left; padding:4px 6px; border-bottom:1px solid #232b35; }
  a { color:#4aa3ff; text-decoration:none; }
</style></head>
<body>
  <div class="top">
    <b style="font-size:16px">Drone QR Detection & Camera-Adjusted Scan HUD (:8091)</b>
    <span class="pill" id="st">state: -</span>
    <span class="pill" id="sqr">start QR: -</span>
    <span class="pill" id="boost">boost: off</span>
    <a href="http://localhost:8080" target="_blank" class="pill">Open Main Mission Dashboard (:8080) ↗</a>
  </div>
  <div class="grid">
    <div class="card">
      <h2>Live Annotated Camera + Gnomonic Adjusted Zoom Patches</h2>
      <img id="feed" src="/annotated.jpg" alt="QR HUD">
    </div>
    <div>
      <div class="card">
        <h2>Detected QRs (Short-Window Averaged)</h2>
        <table id="qtable"><tr><th>Payload</th><th>Status</th><th>Pos (x, y) m</th><th>Hits</th></tr></table>
      </div>
      <div class="card" style="margin-top:10px">
        <h2>Stabilized Inter-QR Distances</h2>
        <table id="dtable"><tr><th>Pair</th><th>Avg Dist (m)</th><th>Samples</th></tr></table>
      </div>
      <div class="card" style="margin-top:10px">
        <h2>Boxed Candidates Being Scanned</h2>
        <table id="ctable"><tr><th>Ground (x, y) m</th><th>Conf</th><th>Hits</th></tr></table>
      </div>
    </div>
  </div>
<script>
setInterval(async () => {
  try {
    const im = document.getElementById('feed');
    im.src = '/annotated.jpg?t=' + Date.now();
    const s = await (await fetch('/status.json')).json();
    const st = document.getElementById('st');
    st.textContent = 'state: ' + s.state;
    st.className = 'pill ' + (s.state === 'TARGET_FOUND' ? 'ok' : s.state === 'SEARCHING' ? 'warn' : '');
    const sqr = document.getElementById('sqr');
    sqr.textContent = 'start QR: ' + (s.start_qr || '(waiting for 5m takeoff scan)');
    sqr.className = 'pill ' + (s.start_qr ? 'ok' : 'warn');
    const b = document.getElementById('boost');
    b.textContent = 'exposure boost: ' + (s.boost ? 'ON' : 'off');
    b.className = 'pill ' + (s.boost ? 'warn' : '');
    document.getElementById('qtable').innerHTML = '<tr><th>Payload</th><th>Match</th><th>Pos (x, y) m</th><th>Hits</th></tr>' +
      (s.qrs || []).map(q => `<tr><td><b>${q.payload}</b></td><td class="${q.status==='target'?'ok':q.status==='rejected'?'bad':'warn'}">${q.status.toUpperCase()}</td><td>(${q.x.toFixed(2)}, ${q.y.toFixed(2)})</td><td>${q.n}</td></tr>`).join('');
    document.getElementById('dtable').innerHTML = '<tr><th>Pair</th><th>Avg Dist (m)</th><th>Window</th></tr>' +
      (s.pairwise_distances_m || []).map(d => `<tr><td>${d.qr_1} ↔ ${d.qr_2}</td><td><b>${d.distance_m.toFixed(3)} m</b></td><td>${d.samples}</td></tr>`).join('');
    document.getElementById('ctable').innerHTML = '<tr><th>Ground (x, y) m</th><th>Conf</th><th>Hits</th></tr>' +
      (s.candidates || []).map(c => `<tr><td>(${c.x.toFixed(2)}, ${c.y.toFixed(2)})</td><td>${c.conf ?? '-'}</td><td>${c.n}</td></tr>`).join('');
  } catch (e) {}
}, 400);
</script></body></html>
"""


class QrLocator(Node):
    def __init__(self):
        super().__init__('qr_locator')
        p = self.declare_parameter
        p('topic', '/arducam/image_raw')
        p('nadir_topic', '')         # sim: /arducam/nadir_hd/image_raw
        p('hd_topic', '')            # real: /arducam/qr/image_raw
        p('cand_edge', 1.0)
        p('cand_sat_max', 60.0)
        p('cand_red_slack', 1.5)
        p('nadir_hfov', 1.0472)
        p('rate_hz', 3.5)
        p('scales', [1.0, 1.5, 0.75])
        p('ground_z', 0.0)
        p('cluster_radius', 4.0)
        p('confirm_hits', 3)
        p('odom_frame', 'odom')
        p('zone', [15.0, 55.0, -15.0, 15.0])
        p('world_json', '')
        p('pylib', '')
        p('start_qr_file', os.path.expanduser('~/.ros/aerothon_start_qr.json'))
        p('cand_rate', 1.5)
        p('cand_std', 45.0)
        p('cand_std_min', 6.0)
        p('cand_levels', [1.0, 1.6, 2.5, 4.0])
        p('cand_min_transitions', 3.0)
        p('cand_win', 5)
        p('cand_size', [0.5, 2.0])
        p('cand_min_hits', 2)
        p('wechat_dir', '')
        p('net', True)
        p('net_conf', 0.60)          # refined with dark-square contour + gnomonic patch verification
        p('net_max_px', 260.0)
        p('net_full_period', 3.5)
        p('chase_avg', 5)            # short-window aligned crop stack for unread boxed QRs
        p('boost_after', 2.5)
        p('boost_hold', 5.0)
        p('fine_height', 6.5)
        p('avg_window_s', 4.0)       # short time window (s) for stabilizing QR positions & inter-QR distances
        p('avg_window_samples', 15)  # max samples in the short sliding window
        p('hud_port', 8091)          # live HTTP QR HUD port (integrated with :8080 dashboard)
        p('marker', os.environ.get('AEROTHON_MARKER', 'qr'))   # qr | aruco (the ArUco delivery mission, 8084)

        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        if g('world_json') and os.path.isfile(g('world_json')):
            z = json.load(open(g('world_json')))['delivery_zone']
            self.zone = (z['x_from'], z['x_to'], z['y_from'], z['y_to'])
        else:
            self.zone = tuple(g('zone'))
        self.aruco = None
        if g('marker') == 'aruco':
            from aerothon_nav import aruco_tools
            self.aruco_tools = aruco_tools
            self.aruco = aruco_tools.Detector(raw=False)   # a 4x4-6x6 grid off the dictionaries is a misread
            self.zbar = self.wc = self.qnet = None
            self.get_logger().info('ArUco delivery: markers 3x3 .. 6x6 on the undistorted image')
        else:
            self.init_qr()
        self.blob_px = (0.0, [], (1, 1))
        self.net_full_t = 0.0
        self.chase = {}              # box key -> [crops]: boxed QRs not read yet (aligned & averaged)
        self.last_zoom_patches = []  # [(label, patch_img)] for HUD inset visualization
        self.last_boxes_px = []      # [(conf, poly, label)] in latest frame
        self.unread_t, self.boost, self.boost_t = 0.0, False, 0.0
        self.boost_pub = self.create_publisher(Bool, '/arducam/qr_boost', 10)
        self.n_hd = 0
        self.cam = None
        self.cam_z = None
        self.last_ray_cos4 = 1.0
        self.last = {}
        self.qrs = []                # {payload, x, y, box, n, source, first, samples: [(t, x, y, box, w)]}
        self.pair_dist_hist = {}     # (qr1, qr2) -> [(t, dist)] short-window inter-QR distance buffer
        self.cands = []              # {x, y, n, samples: [(t, x, y)]}
        self.cand_last = 0.0
        self.start_qr = None
        try:
            self.start_qr = json.loads(Path(g('start_qr_file')).read_text())['payload']
        except (OSError, ValueError, KeyError):
            pass
        self.target_sent = False
        self.red = None
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.proc_lock = threading.Lock()
        cb = MutuallyExclusiveCallbackGroup()
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.qrs_pub = self.create_publisher(String, '/delivery/qrs', 10)
        self.target_pub = self.create_publisher(PoseStamped, '/delivery/target', latched)
        self.fine = []
        self.fine_pub = self.create_publisher(String, '/delivery/target_fine', latched)
        self.marker_pub = self.create_publisher(MarkerArray, '/delivery/qr_markers', 10)
        self.create_subscription(CameraInfo, '/arducam/camera_info', self.on_info, qos_profile_sensor_data)
        self.image_cb, self.image_subs = cb, []
        self.create_subscription(String, '/mission/start_qr', self.on_start_qr, latched)
        self.create_subscription(OccupancyGrid, '/delivery/red', lambda m: setattr(self, 'red', m), latched)
        self.state = 'IDLE'
        self.hud_lock = threading.Lock()
        self.raw_jpeg = None
        self.annotated_jpeg = None
        self.status_json = {'state': 'IDLE', 'start_qr': self.start_qr, 'qrs': [], 'candidates': [],
                            'pairwise_distances_m': [], 'boost': False}
        self.create_service(Trigger, '/delivery/search/start', self.on_search_start)
        self.create_service(Trigger, '/delivery/search/stop', self.on_search_stop)
        self.create_timer(1.0, self.publish)
        # Always subscribe to the main fisheye feed so the port 8091 HUD is live during takeoff/corridor too
        self.cameras(True)
        if int(g('hud_port')) > 0:
            self._start_hud_server(int(g('hud_port')))

    def _start_hud_server(self, port):
        locator = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path.startswith('/annotated.jpg'):
                    with locator.hud_lock:
                        jpg = locator.annotated_jpeg or locator.raw_jpeg
                    if not jpg:
                        self.send_response(503)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(jpg)
                elif self.path.startswith('/frame.jpg'):
                    with locator.hud_lock:
                        jpg = locator.raw_jpeg
                    if not jpg:
                        self.send_response(503)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(jpg)
                elif self.path.startswith('/status.json'):
                    with locator.hud_lock:
                        data = json.dumps(locator.status_json).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    body = HUD_HTML.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(body)

            def log_message(self, format, *args):
                return

        def serve():
            try:
                srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
                locator.get_logger().info(f'QR Detection HUD server listening on http://0.0.0.0:{port}')
                srv.serve_forever()
            except OSError as e:
                locator.get_logger().warn(f'Port {port} already in use ({e}); HUD server skipped in qr_locator')

        threading.Thread(target=serve, daemon=True).start()

    def init_qr(self):
        g = self.g
        if g('pylib'):
            sys.path.insert(0, g('pylib'))
        try:
            from pyzbar.pyzbar import ZBarSymbol, decode
            self.zbar = lambda img: decode(img, symbols=[ZBarSymbol.QRCODE])
        except ImportError:
            self.zbar = None
            self.get_logger().warn('pyzbar not available: WeChat QR detector only')
        self.wc = qr_tools.make_wechat(g('wechat_dir'))
        self.qnet = qr_tools.QrNet(g('wechat_dir')) if g('net') else None
        if self.qnet is not None and self.qnet.net is None:
            self.qnet = None
        self.get_logger().info(f'decoders: {"WeChat + " if self.wc else ""}{"zbar" if self.zbar else "none"}; '
                               f'QR detection network {"on" if self.qnet else "off"}')

    def cameras(self, on):
        """Subscribe to the cameras."""
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        if on and not self.image_subs:
            self.image_subs.append(self.create_subscription(
                Image, g('topic'), lambda m: self.on_image(m, 'fisheye'), 1, callback_group=self.image_cb))
            if g('nadir_topic'):
                self.image_subs.append(self.create_subscription(
                    Image, g('nadir_topic'), lambda m: self.on_image(m, 'nadir_hd'), 1, callback_group=self.image_cb))
            if g('hd_topic'):
                self.image_subs.append(self.create_subscription(
                    Image, g('hd_topic'), lambda m: self.on_image(m, 'fisheye_hd'), 1, callback_group=self.image_cb))
        elif not on:
            for sub in self.image_subs:
                self.destroy_subscription(sub)
            self.image_subs = []

    def on_search_start(self, req, res):
        with self.proc_lock:
            self.cameras(True)
            self.qrs, self.target_sent, self.state, self.fine, self.cands = [], False, 'SEARCHING', [], []
            self.pair_dist_hist.clear()
            self.chase.clear()
        res.success, res.message = True, f'searching for {self.start_qr!r}'
        self.get_logger().info(res.message)
        return res

    def on_search_stop(self, req, res):
        with self.proc_lock:
            self.state = 'IDLE'
            if self.boost:
                self.boost = False
                self.boost_pub.publish(Bool(data=False))
        res.success, res.message = True, 'idle'
        return res

    def on_info(self, m):
        if self.cam is None or self.cam.size != (m.width, m.height):
            self.cam = Fisheye(m)

    def on_start_qr(self, m):
        try:
            self.start_qr = json.loads(m.data)['payload']
        except (ValueError, KeyError):
            self.start_qr = m.data.strip() or None
        self.get_logger().info(f'delivery target payload: {self.start_qr!r}')

    # ---- decoding with WeChat CNN + quiet-zone padding + CLAHE + zbar
    def decode(self, gray):
        if gray is None or gray.size == 0:
            return {}
        mean = float(gray.mean())
        work = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray) if (mean < 80 or mean > 200) else gray
        out = {}
        if self.wc is not None and work.shape[1] <= 2000:
            out = qr_tools.wechat_decode(self.wc, work)
            if not out and work is not gray:
                out = qr_tools.wechat_decode(self.wc, gray)
            if out:
                return out
        blur = cv2.GaussianBlur(work, (3, 3), 0)
        if self.zbar:
            for scale in [sc for sc in self.g('scales') if sc <= 1.0 or blur.shape[1] <= 2000]:
                img = blur if scale == 1.0 else cv2.resize(blur, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                img = np.ascontiguousarray(img)
                try:
                    dec_list = self.zbar(img)
                except Exception:
                    dec_list = []
                for d in dec_list:
                    payload = d.data.decode('utf-8', errors='replace').strip()
                    r = d.rect
                    pts = [(pt.x, pt.y) for pt in d.polygon] or \
                          [(r.left, r.top), (r.left + r.width, r.top), (r.left + r.width, r.top + r.height),
                           (r.left, r.top + r.height)]
                    if payload and payload not in out:
                        out[payload] = np.array(pts, float) / scale
                if out:
                    return out
        return out

    # ---- pixels -> ground
    def ground_points(self, px, m, source):
        """Pixels (N x 2) -> ground points (N x 2) in odom, or None if any ray misses the ground."""
        try:
            frame = m.header.frame_id if source in ('fisheye', 'fisheye_hd') else 'base_link'
            t = self.tf.lookup_transform(self.g('odom_frame'), frame, m.header.stamp, timeout=Duration(seconds=0.3))
        except Exception:
            return None
        if source in ('fisheye', 'fisheye_hd'):
            if self.cam is None:
                return None
            rays_cam = self.cam.rays(px, scale=m.width / self.cam.size[0])
        else:
            f = (m.width / 2) / math.tan(self.g('nadir_hfov') / 2)
            rays_cam = np.column_stack([-(px[:, 1] - m.height / 2) / f, -(px[:, 0] - m.width / 2) / f,
                                        -np.ones(len(px))])
            rays_cam /= np.linalg.norm(rays_cam, axis=1, keepdims=True)
        rays = rays_cam @ quat_to_rot(t.transform.rotation).T
        o = t.transform.translation
        self.cam_z = o.z
        if (rays[:, 2] > -0.05).any():
            return None
        cos_nadir = float(np.clip(-rays[0, 2], 0.15, 1.0))
        self.last_ray_cos4 = cos_nadir ** 4
        k = (o.z - self.g('ground_z')) / -rays[:, 2]
        return np.array([o.x, o.y]) + rays[:, :2] * k[:, None]

    def near_decoded(self, x, y):
        return any(math.dist((q['x'], q['y']), (x, y)) < self.g('cluster_radius') for q in self.qrs)

    def detect_candidates(self, m):
        """Boards seen on the fisheye frame m, not decoded -> self.cands (masked to active fisheye circle)."""
        gray_full = to_gray(m)
        g = cv2.resize(gray_full, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
        sat = None
        if m.encoding in ('rgb8', 'bgr8'):
            rgb = cv2.resize(np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, 3), (g.shape[1], g.shape[0]),
                             interpolation=cv2.INTER_AREA)
            sat = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV if m.encoding == 'rgb8' else cv2.COLOR_BGR2HSV)[..., 1]
        g0 = g
        g = cv2.GaussianBlur(g, (0, 0), 1.2)
        k = int(self.g('cand_win'))
        mu = cv2.boxFilter(g, -1, (k, k))
        sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (k, k)) - mu * mu, 0))
        # Mask out the black outer fisheye vignette ring so the circular border never triggers false blobs
        if self.cam is not None:
            fmask = self.cam.circle_mask(g.shape[1], g.shape[0], max_theta_deg=82.0)
            sd = sd * cv2.erode(fmask.astype(np.uint8), np.ones((7, 7), np.uint8))
        thr = qr_tools.contrast_threshold(sd, floor=self.g('cand_std_min'))
        blobs = []
        for level in self.g('cand_levels'):
            mask = (sd > thr * level).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                bx, by, bw, bh, area = stats[i]
                if area < 6 or area > 20000:
                    continue
                if max(bw, bh) / max(min(bw, bh), 1) > 2.6:
                    continue
                if any(math.dist(cent[i], c) < 0.5 * max(bw, bh) for _, c, _ in blobs):
                    continue
                blobs.append((stats[i], cent[i], lab == i))
        self.blob_px = (time.monotonic(), [c * 2.0 for _, c, _ in blobs], (m.width, m.height))
        lo, hi = self.g('cand_size')
        x0, x1, y0, y1 = self.zone
        for (bx, by, bw, bh, area), cen, blob in blobs:
            if min(bw, bh) >= 8:
                tr = qr_tools.transitions(g0[by:by + bh, bx:bx + bw])
                if min(tr) < self.g('cand_min_transitions'):
                    continue
            px = np.array([cen, (bx, by), (bx + bw, by), (bx + bw, by + bh), (bx, by + bh)], float) * 2.0
            pts = self.ground_points(px, m, 'fisheye')
            if pts is None:
                continue
            ext = pts[1:].max(axis=0) - pts[1:].min(axis=0)
            size = float(max(ext))
            cx, cy = float(pts[0][0]), float(pts[0][1])
            e = self.g('cand_edge')
            if not lo <= size <= hi or not (x0 + e <= cx <= x1 - e and y0 + e <= cy <= y1 - e):
                continue
            if sat is not None and float(sat[blob].mean()) > self.g('cand_sat_max'):
                continue
            if self.deep_red(cx, cy) or self.near_decoded(cx, cy):
                continue
            self.add_cand(cx, cy)

    def add_cand(self, cx, cy, conf=None):
        """A candidate board at (cx, cy) (odom): short-window averaged within cluster_radius."""
        now = time.monotonic()
        win_s = float(self.g('avg_window_s'))
        max_n = int(self.g('avg_window_samples'))
        for c in self.cands:
            if math.dist((c['x'], c['y']), (cx, cy)) < self.g('cluster_radius'):
                c['n'] += 1 if conf is None else self.g('cand_min_hits')
                samples = [s for s in c.get('samples', []) if now - s[0] <= win_s]
                samples.append((now, cx, cy))
                c['samples'] = samples[-max_n:]
                xs = [s[1] for s in c['samples']]
                ys = [s[2] for s in c['samples']]
                c['x'] = float(np.median(xs))
                c['y'] = float(np.median(ys))
                if conf is not None:
                    c['conf'] = max(c.get('conf', 0.0), conf)
                return
        c = {'x': cx, 'y': cy, 'n': 1 if conf is None else self.g('cand_min_hits'), 'samples': [(now, cx, cy)]}
        if conf is not None:
            c['conf'] = conf
        self.cands.append(c)

    def on_image(self, m, source):
        if not self.proc_lock.acquire(blocking=False):
            return
        try:
            now = time.monotonic()
            rate = self.g('rate_hz') if self.state == 'SEARCHING' else 1.0
            if source == 'fisheye' and self.state == 'SEARCHING' and now - self.cand_last >= 1.0 / self.g('cand_rate'):
                self.cand_last = now
                try:
                    self.detect_candidates(m)
                except Exception as e:
                    self.get_logger().warn(f'candidate detection: {e}', throttle_duration_sec=10.0)
            if now - self.last.get(source, 0.0) < 1.0 / max(0.2, rate):
                return
            self.last[source] = now
            gray = to_gray(m)
            if self.aruco is not None:
                found = self.read_aruco(gray, source)
            elif self.qnet is not None and (source in ('fisheye_hd', 'nadir_hd') or not self.g('hd_topic')):
                found = self.on_hd(gray, m, source, now)
            else:
                found = self.decode(gray)
            self.use_reads(found, m, source)
            self._update_hud_image(gray, found)
        finally:
            self.proc_lock.release()

    def read_aruco(self, gray, source):
        """{payload: corners in this image's pixels}: undistorted views on the fisheye, straight on the nadir."""
        if source in ('fisheye', 'fisheye_hd'):
            if self.cam is None:
                return {}
            return self.aruco_tools.detect_fisheye(self.aruco, gray, self.cam)
        return dict(self.aruco.detect(gray))

    def on_hd(self, gray, m, source, now):
        """Multi-scale QrNet bounding-box search + virtual camera adjustment (`rectify_patch`) + phase-correlated
        frame averaging (`aligned_mean`) + exposure boost."""
        conf = self.g('net_conf')
        t, centres, (bw_, bh_) = self.blob_px
        s = m.width / max(bw_, 1)
        if not centres and self.cam is not None:
            fmask = self.cam.circle_mask(m.width, m.height, max_theta_deg=82.0)
            cands_local = qr_tools.find_qr_candidates(gray, fisheye_mask=fmask, max_cands=8)
            centres = [c['center'] for c in cands_local]
            s = 1.0
            t = now

        full = now - self.net_full_t > self.g('net_full_period') or source == 'nadir_hd'
        if full:
            self.net_full_t = now
            small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) if m.width > 2000 else gray
            k = m.width / small.shape[1]
            boxes = [(c, qr_tools.refine_qr_box(gray, p * k))
                     for c, p in self.qnet.scan(small, conf, multi_scale=False)]
        else:
            boxes = []
        if now - t < 2.0 and centres:
            near = self.qnet.scan(gray, conf, centres=[(cx * s, cy * s) for cx, cy in centres[:10]], multi_scale=True)
            boxes += [b for b in near if all(qr_tools.iou(b[1], q) < 0.3 for _, q in boxes)]
        boxes = [(c, p) for c, p in boxes if 20.0 <= max(np.ptp(p[:, 0]), np.ptp(p[:, 1])) <= self.g('net_max_px')]

        found = {}
        unread = False
        zoom_patches = []
        boxes_viz = []
        scale_cam = m.width / max(self.cam.size[0], 1) if self.cam is not None else 1.0

        for c, poly in boxes:
            (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)
            pad = max(36.0, 0.45 * max(x1 - x0, y1 - y0))
            X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            X1, Y1 = int(min(m.width, x1 + pad)), int(min(m.height, y1 + pad))
            crop = gray[Y0:Y1, X0:X1]

            # Step A: Direct padded WeChat decode on the raw crop
            got = qr_tools.wechat_decode(self.wc, crop, x0=X0, y0=Y0) if self.wc is not None else {}

            # Step B: Adjust virtual gnomonic camera (`rectify_patch`) along the bounding box's 3D ray to un-distort & zoom
            patch_rect = None
            p_meta = None
            if not got and self.cam is not None and source in ('fisheye', 'fisheye_hd'):
                center_px = poly.mean(axis=0, keepdims=True)
                ray_d = self.cam.rays(center_px, scale=scale_cam)[0]
                patch_rect, p_meta = self.cam.rectify_patch(gray, ray_d=ray_d, fov_deg=22.0, zoom_scale=2.5, max_size=400)
                hit_rect = (qr_tools.wechat_decode(self.wc, patch_rect) if self.wc is not None else {}) or self.decode(patch_rect)
                if hit_rect:
                    got = {pl: self.cam.patch_px_to_fisheye_px(pp, p_meta) for pl, pp in hit_rect.items()}

            # Step C: Short-window multi-frame phase-correlated stacking (`aligned_mean`) for faint/distant boxes
            key = (int((x0 + x1) / 2) // 96, int((y0 + y1) / 2) // 96)
            work_patch = patch_rect if patch_rect is not None else crop
            if not got and work_patch.size > 0:
                stack = self.chase.setdefault(key, [])
                stack.append(cv2.resize(work_patch, (256, 256), interpolation=cv2.INTER_AREA))
                del stack[:-self.g('chase_avg')]
                if len(stack) >= 2:
                    avg, n_used = qr_tools.aligned_mean(stack)
                    clean = cv2.fastNlMeansDenoising(avg, None, 9, 5, 7)
                    for img_try in (clean, avg):
                        hit = (qr_tools.wechat_decode(self.wc, img_try) if self.wc is not None else {}) or self.decode(img_try)
                        if hit:
                            if p_meta is not None:
                                sx, sy = patch_rect.shape[1] / 256.0, patch_rect.shape[0] / 256.0
                                got = {pl: self.cam.patch_px_to_fisheye_px(pp * [sx, sy], p_meta) for pl, pp in hit.items()}
                            else:
                                sx, sy = (X1 - X0) / 256.0, (Y1 - Y0) / 256.0
                                got = {pl: pp * [sx, sy] + [X0, Y0] for pl, pp in hit.items()}
                            self.get_logger().info(f'chased QR read after camera-adjusted averaging ({n_used} frames): {list(got)}')
                            break
                    work_patch = clean

            if len(zoom_patches) < 3 and work_patch is not None and work_patch.size > 0:
                lbl = next(iter(got.keys()), f'scan {c:.2f}')
                zoom_patches.append((lbl, cv2.resize(work_patch, (160, 160), interpolation=cv2.INTER_AREA)))

            if got:
                self.chase.pop(key, None)
                found.update(got)
                boxes_viz.append((c, poly, next(iter(got.keys()))))
                continue

            unread = True
            boxes_viz.append((c, poly, None))
            pts = self.ground_points(np.vstack([poly.mean(axis=0), poly]), m, source)
            if pts is not None:
                cx, cy = float(pts[0][0]), float(pts[0][1])
                zx0, zx1, zy0, zy1 = self.zone
                if zx0 <= cx <= zx1 and zy0 <= cy <= zy1 and not self.deep_red(cx, cy) and not self.near_decoded(cx, cy):
                    self.add_cand(cx, cy, conf=round(c, 2))

        self.last_zoom_patches = zoom_patches
        self.last_boxes_px = boxes_viz
        if len(self.chase) > 20:
            self.chase = dict(list(self.chase.items())[-20:])
        self.n_hd += 1
        if not found and self.n_hd % 3 == 0:
            found = self.decode(gray)
        # exposure boost while a boxed QR stays unread
        if unread:
            self.unread_t = self.unread_t or now
        else:
            self.unread_t = 0.0
        want = bool(self.unread_t and now - self.unread_t > self.g('boost_after'))
        if want:
            self.boost_t = now
        if want != self.boost and (want or now - self.boost_t > self.g('boost_hold')):
            self.boost = want
            self.boost_pub.publish(Bool(data=want))
            self.get_logger().info(f'QR exposure boost {"on: a boxed QR stays unread" if want else "off"}')
        return found

    def use_reads(self, found, m, source):
        for payload, poly in found.items():
            pts = self.ground_points(np.vstack([poly.mean(axis=0), poly]), m, source)
            if pts is not None and payload == self.start_qr and self.cam_z is not None \
                    and self.cam_z < self.g('fine_height'):
                self.fine = (self.fine + [tuple(pts[0])])[-15:]
                fx, fy = np.median(np.array(self.fine), axis=0)
                self.fine_pub.publish(String(data=json.dumps(
                    {'x': round(float(fx), 3), 'y': round(float(fy), 3), 'n': len(self.fine), 'source': source})))
            if pts is not None:
                corners = pts[1:]
                box = [*corners.min(axis=0), *corners.max(axis=0)]
                self.add(payload, pts[0], box, source, weight=self.last_ray_cos4)

    def add(self, payload, xy, box, source, weight=1.0):
        """Short-window temporal averaging with cos^4(theta) weighting + median outlier rejection."""
        now = time.monotonic()
        win_s = float(self.g('avg_window_s'))
        max_n = int(self.g('avg_window_samples'))
        w = max(0.05, float(weight))
        for q in self.qrs:
            if q['payload'] == payload and math.dist((q['x'], q['y']), xy) < self.g('cluster_radius'):
                q['n'] += 1
                samples = [s for s in q.get('samples', []) if now - s[0] <= win_s]
                samples.append((now, float(xy[0]), float(xy[1]), [float(v) for v in box], w))
                q['samples'] = samples[-max_n:]
                if len(q['samples']) >= 3:
                    arr_xy = np.array([[s[1], s[2]] for s in q['samples']], dtype=np.float64)
                    ws = np.array([s[4] for s in q['samples']], dtype=np.float64)
                    med = np.median(arr_xy, axis=0)
                    dists = np.linalg.norm(arr_xy - med, axis=1)
                    mad = float(np.median(dists)) + 0.15
                    keep = dists <= 2.5 * mad
                    arr_xy, ws = arr_xy[keep], ws[keep]
                    ws /= max(ws.sum(), 1e-6)
                    q['x'] = float((arr_xy[:, 0] * ws).sum())
                    q['y'] = float((arr_xy[:, 1] * ws).sum())
                    boxes_k = np.array([s[3] for s, k_ in zip(q['samples'], keep) if k_], dtype=np.float64)
                    q['box'] = [float((boxes_k[:, i] * ws).sum()) for i in range(4)]
                else:
                    ws = np.array([s[4] for s in q['samples']], dtype=np.float64)
                    ws /= max(ws.sum(), 1e-6)
                    q['x'] = float(sum(s[1] * wi for s, wi in zip(q['samples'], ws)))
                    q['y'] = float(sum(s[2] * wi for s, wi in zip(q['samples'], ws)))
                    q['box'] = [float(sum(s[3][i] * wi for s, wi in zip(q['samples'], ws))) for i in range(4)]
                return
        self.qrs.append({
            'payload': payload,
            'x': float(xy[0]),
            'y': float(xy[1]),
            'box': [float(v) for v in box],
            'n': 1,
            'source': source,
            'first': time.time(),
            'samples': [(now, float(xy[0]), float(xy[1]), [float(v) for v in box], w)],
        })
        self.get_logger().info(f'QR {payload!r} at ({xy[0]:.2f}, {xy[1]:.2f}) [{source}]')

    def compute_stabilized_pairwise_distances(self):
        """Compute short-window averaged inter-QR distances so distances between QRs stay steady."""
        now = time.monotonic()
        win_s = max(float(self.g('avg_window_s')), 6.0)
        max_n = int(self.g('avg_window_samples'))
        qs = sorted(self.qrs, key=lambda q: q['payload'])
        out = []
        for i in range(len(qs)):
            for j in range(i + 1, len(qs)):
                a, b = qs[i], qs[j]
                key = (a['payload'], b['payload'])
                d_inst = math.dist((a['x'], a['y']), (b['x'], b['y']))
                hist = [h for h in self.pair_dist_hist.get(key, []) if now - h[0] <= win_s]
                hist.append((now, d_inst))
                hist = hist[-max_n:]
                self.pair_dist_hist[key] = hist
                vals = np.array([h[1] for h in hist], dtype=np.float64)
                if len(vals) >= 3:
                    med = float(np.median(vals))
                    mad = float(np.median(np.abs(vals - med))) + 0.05
                    vals = vals[np.abs(vals - med) <= 2.5 * mad]
                d_avg = float(np.mean(vals))
                out.append({
                    'qr_1': a['payload'],
                    'qr_2': b['payload'],
                    'distance_m': round(d_avg, 3),
                    'samples': len(hist),
                })
        return out

    def _update_hud_image(self, gray, found):
        try:
            scale = 960.0 / max(gray.shape[1], 1)
            vis_g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
            scale = vis_g.shape[1] / max(gray.shape[1], 1)
            ok_raw, raw_jpg = cv2.imencode('.jpg', vis_g, [cv2.IMWRITE_JPEG_QUALITY, 75])
            vis = cv2.cvtColor(vis_g, cv2.COLOR_GRAY2BGR)

            # Draw active fisheye circle boundary
            if self.cam is not None:
                fx, fy, cx, cy = self.cam._scaled_intrinsics(vis.shape[1] / max(self.cam.size[0], 1))
                r = int(0.5 * (fx + fy) * math.radians(82.0))
                cv2.circle(vis, (int(cx), int(cy)), r, (70, 70, 70), 1)

            # Draw contrast blob centres in yellow
            _, centres, (bw_, bh_) = self.blob_px
            s_b = vis.shape[1] / max(bw_, 1)
            for cx, cy in centres[:10]:
                cv2.circle(vis, (int(cx * s_b), int(cy * s_b)), 14, (0, 210, 255), 1)

            # Draw QrNet bounding boxes in green/cyan
            for conf, poly, lbl in self.last_boxes_px:
                pts = (poly * scale).astype(np.int32)
                col = (50, 230, 80) if lbl else (255, 190, 0)
                cv2.polylines(vis, [pts], True, col, 2)
                p0 = pts.min(axis=0)
                tag = f'{lbl} ({conf:.2f})' if lbl else f'QR box {conf:.2f} (adjusting scan)'
                cv2.putText(vis, tag, (int(p0[0]), max(16, int(p0[1]) - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

            # Draw decoded polygons from full frame
            for payload, poly in found.items():
                pts = (poly * scale).astype(np.int32)
                match = (self.start_qr and payload == self.start_qr)
                col = (40, 255, 40) if match else (40, 180, 255)
                cv2.polylines(vis, [pts], True, col, 2)
                p0 = pts.min(axis=0)
                tag = f'TARGET: {payload}' if match else f'QR: {payload}'
                cv2.putText(vis, tag, (int(p0[0]), max(18, int(p0[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

            # Draw inset gnomonic-rectified zoom patches ("Camera Adjusted Scan") in top-right corner
            x_off = vis.shape[1] - 110
            for lbl, patch in self.last_zoom_patches[:3]:
                thumb = cv2.cvtColor(cv2.resize(patch, (100, 100)), cv2.COLOR_GRAY2BGR)
                if x_off >= 10 and vis.shape[0] >= 125:
                    vis[10:110, x_off:x_off + 100] = thumb
                    cv2.rectangle(vis, (x_off, 10), (x_off + 100, 110), (74, 163, 255), 1)
                    cv2.putText(vis, str(lbl)[:14], (x_off + 2, 122),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 230, 240), 1, cv2.LINE_AA)
                    x_off -= 110

            # Status banner at top-left
            hdr = f'State: {self.state} | Start QR: {self.start_qr or "none"} | Tracked: {len(self.qrs)}'
            cv2.putText(vis, hdr, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2, cv2.LINE_AA)
            ok_ann, ann_jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 78])
            with self.hud_lock:
                if ok_raw:
                    self.raw_jpeg = raw_jpg.tobytes()
                if ok_ann:
                    self.annotated_jpeg = ann_jpg.tobytes()
        except Exception:
            pass

    def in_red(self, x, y):
        g = self.red
        if g is None:
            return None
        i = int((x - g.info.origin.position.x) / g.info.resolution)
        j = int((y - g.info.origin.position.y) / g.info.resolution)
        if 0 <= i < g.info.width and 0 <= j < g.info.height:
            return g.data[j * g.info.width + i] >= 100
        return False

    def deep_red(self, x, y):
        if not self.in_red(x, y):
            return False
        r = self.g('cand_red_slack')
        return all(self.in_red(x + r * math.cos(a), y + r * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8,
                                                                                              endpoint=False))

    def publish(self):
        out = []
        for q in self.qrs:
            q['confirmed'] = q['n'] >= self.g('confirm_hits')
            q['in_red_zone'] = self.in_red(q['x'], q['y'])
            x0, x1, y0, y1 = self.zone
            q['in_zone'] = bool(x0 <= q['x'] <= x1 and y0 <= q['y'] <= y1)
            if self.start_qr and q['payload'] == self.start_qr:
                q['status'] = 'target'
            elif self.start_qr and q['confirmed'] and q['in_zone']:
                q['status'] = 'rejected'
            else:
                q['status'] = 'pending'
            out.append({k: (round(v, 2) if isinstance(v, float) else [round(b, 2) for b in v] if k == 'box' else v)
                        for k, v in q.items() if k not in ('first', 'samples')})
            if (self.state == 'SEARCHING' and q['confirmed'] and q['in_zone'] and self.start_qr and
                    q['payload'] == self.start_qr):
                if q['in_red_zone']:
                    self.get_logger().warn(f'target payload found at ({q["x"]:.1f}, {q["y"]:.1f}) but in a red zone',
                                           throttle_duration_sec=10.0)
                    continue
                t = PoseStamped()
                t.header.frame_id = self.g('odom_frame')
                t.header.stamp = self.get_clock().now().to_msg()
                t.pose.position.x, t.pose.position.y = float(q['x']), float(q['y'])
                t.pose.orientation.w = 1.0
                self.target_pub.publish(t)
                self.target_sent, self.state = True, 'TARGET_FOUND'
                self.get_logger().info(f'TARGET MATCHED START QR {q["payload"]!r} at ({q["x"]:.2f}, {q["y"]:.2f}) '
                                       f'after {q["n"]} reads -> triggering delivery & winch sequence!')
        cands = [{'x': round(c['x'], 2), 'y': round(c['y'], 2), 'n': c['n'],
                  **({'conf': c['conf']} if 'conf' in c else {})} for c in self.cands
                 if c['n'] >= self.g('cand_min_hits') and not self.near_decoded(c['x'], c['y'])
                 and not self.deep_red(c['x'], c['y'])]
        pairs = self.compute_stabilized_pairwise_distances()
        msg_dict = {
            'state': self.state,
            'start_qr': self.start_qr,
            'qrs': out,
            'candidates': cands,
            'pairwise_distances_m': pairs,
            'boost': self.boost,
        }
        with self.hud_lock:
            self.status_json = msg_dict
        self.qrs_pub.publish(String(data=json.dumps(msg_dict)))
        self.publish_markers()

    def publish_markers(self):
        colour = {'target': (0.1, 0.9, 0.1), 'rejected': (0.9, 0.1, 0.1), 'pending': (0.9, 0.8, 0.1)}
        arr = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        for i, q in enumerate(self.qrs):
            mk = Marker(type=Marker.LINE_STRIP, ns='qr_boxes', id=i)
            mk.header.frame_id = self.g('odom_frame')
            mk.header.stamp = self.get_clock().now().to_msg()
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.1
            mk.color.r, mk.color.g, mk.color.b = colour[q['status']]
            mk.color.a = 1.0
            x0, y0, x1, y1 = q['box']
            z = self.g('ground_z') + 0.05
            mk.points = [Point(x=x, y=y, z=z) for x, y in ((x0, y0), (x1, y0), (x1, y1), (x0, y1), (x0, y0))]
            arr.markers.append(mk)
        self.marker_pub.publish(arr)


def main():
    rclpy.init()
    node = QrLocator()
    ex = rclpy.executors.MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:
        if rclpy.ok():
            raise
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
