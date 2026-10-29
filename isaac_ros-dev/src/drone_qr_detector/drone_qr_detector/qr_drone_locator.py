"""3D Drone-Frame QR Detector & Localizer (`qr_drone_locator`).

Designed for:
  1. Drone flying at 10 m altitude detecting 1 m x 1 m QR codes on the ground (`target_distance_m:=10.0`, `qr_size_m:=1.0`).
  2. Bench / Wall calibration & verification with QR codes at 3 m distance (`target_distance_m:=3.0`, `qr_size_m:=<size>`).

Key improvements over aerothon_nav/qr_locator.py:
  - Decoupled from global `odom`, mission state machines, and delivery-zone bounds: runs immediately on startup and
    reports every detected QR code in the DRONE FRAME (`base_link`) and CAMERA OPTICAL FRAME (`arducam_optical_frame`).
  - Gnomonic (pinhole) rectification before decoding:
      * Rectifies the central FOV (default 75 deg) to remove fisheye barrel distortion.
      * Detects candidate QR blobs + QrNet boxes across the entire active fisheye circle and rectifies a local
        high-resolution pinhole patch aimed directly along each candidate's 3D ray (`rot_to_ray`).
      * Maps decoded polygon corners from any rectified patch back to exact unit 3D rays in the camera optical frame.
  - Phase-correlated multi-frame crop averaging (`aligned_mean`) for low-px/module QR codes (~2.8-3.3 px/module at 10 m).
  - Confidence-weighted (`cos^4(theta)`) filtering and automatic computation of pairwise 3D distances between all
    detected QR codes (for verifying known inter-QR lengths on the 3 m wall).
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Point, Pose, PoseArray
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray
from tf2_ros import Buffer, TransformListener

from drone_qr_detector.fisheye_model import (
    FisheyeModel,
    default_base_from_optical,
    quat_to_rot,
)
from drone_qr_detector import qr_tools


def image_to_gray(msg):
    a = np.frombuffer(msg.data, dtype=np.uint8)
    if msg.encoding in ('mono8', '8UC1'):
        return a.reshape(msg.height, msg.width)
    code = cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(a.reshape(msg.height, msg.width, 3), code)


class QrDroneLocator(Node):
    def __init__(self):
        super().__init__('qr_drone_locator')
        p = self.declare_parameter
        p('image_topic', '/drone_qr/image_raw')
        p('info_topic', '/drone_qr/camera_info')
        p('calibration_file', '/workspaces/isaac_ros-dev/calibration/arducam_fisheye_1640x1232.yaml')
        p('focal_scale', 1.0)
        p('drone_frame', 'base_link')
        p('camera_frame', 'arducam_optical_frame')
        p('mount_xyz', [0.13, 0.0, -0.03])            # base_link -> arducam_link translation (m)
        p('mount_rpy', [0.0, 1.570796, 0.0])          # base_link -> arducam_link roll, pitch, yaw (rad)
        p('qr_size_m', 1.0)                           # Physical QR side length (m): 1.0 m in flight, or wall QR size
        p('qr_modules', 21)                           # 21 (v1) or 25 (v2) modules for px/module diagnostics
        p('target_distance_m', 10.0)                  # Perpendicular distance (m) to QR plane: 10.0 m altitude or 3.0 m wall
        p('rate_hz', 4.0)                             # Decode rate (Hz)
        p('center_fov_deg', 75.0)                     # Central gnomonic rectified view FOV (deg)
        p('patch_fov_deg', 22.0)                      # Local gnomonic patch FOV (deg) around off-axis candidates
        p('patch_zoom', 2.0)                          # Zoom factor for off-axis candidate patches
        p('max_candidates', 8)                        # Max off-axis candidates rectified per frame
        p('chase_avg_frames', 4)                      # Phase-correlated frame stack depth for faint 10m QRs
        p('cluster_radius_m', 0.35)                   # Merge detections of the same payload within this 3D distance
        p('track_timeout_s', 30.0)                    # Keep tracked QRs in memory for pairwise distance comparison
        p('avg_window_s', 4.0)                        # Short sliding time window (s) to stabilize positions & distances
        p('avg_window_samples', 15)                   # Max samples in the sliding window
        p('wechat_dir', '')
        p('use_net', True)
        p('net_conf', 0.55)                           # Lower threshold safe because patches are rectified!
        p('hud_port', 8091)                           # Live HTTP QR HUD port (shared with :8080 dashboard)

        self.g = lambda name: self.get_parameter(name).value
        self.cam = FisheyeModel.from_yaml(
            self.g('calibration_file'),
            focal_scale=float(self.g('focal_scale')),
        )
        self.wc = qr_tools.make_wechat(self.g('wechat_dir'))
        self.zbar = qr_tools.make_zbar()
        self.cv_qr = cv2.QRCodeDetector()
        self.qnet = qr_tools.QrNet(self.g('wechat_dir')) if self.g('use_net') else None
        if self.qnet is not None and self.qnet.net is None:
            self.qnet = None

        self.get_logger().info(
            f'Drone QR Locator ready: decoders={"WeChat+" if self.wc else ""}{"zbar+" if self.zbar else ""}OpenCV, '
            f'QrNet={"on" if self.qnet else "off"}, qr_size={self.g("qr_size_m")}m, '
            f'target_distance={self.g("target_distance_m")}m'
        )

        self.tf_buffer = Buffer()
        TransformListener(self.tf_buffer, self)

        self.tracks = {}       # payload -> tracked state dict
        self.chase_stacks = {} # candidate ray key -> list of 256x256 crops
        self.last_decode_t = 0.0
        self.last_boxes_px = []
        self.last_zoom_patches = []
        self.hud_lock = threading.Lock()
        self.raw_jpeg = None
        self.annotated_jpeg = None
        self.status_json = {'state': 'STANDALONE_3D', 'start_qr': None, 'qrs': [], 'candidates': [],
                            'pairwise_distances_m': [], 'boost': False}

        cb = ReentrantCallbackGroup()
        self.det_pub = self.create_publisher(String, '/drone_qr/detections', 10)
        self.pose_pub = self.create_publisher(PoseArray, '/drone_qr/poses_drone', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/drone_qr/markers', 10)

        self.create_subscription(CameraInfo, self.g('info_topic'), self.on_info, qos_profile_sensor_data)
        self.create_subscription(Image, self.g('image_topic'), self.on_image, 1, callback_group=cb)
        if int(self.g('hud_port')) > 0:
            self._start_hud_server(int(self.g('hud_port')))

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
                    html = (
                        '<!doctype html><html><head><meta charset="utf-8"><title>Drone QR HUD (:8091)</title>'
                        '<style>body{margin:0;background:#0f1318;color:#dfe6ee;font:13px system-ui;padding:12px}'
                        'img{max-width:100%;border-radius:6px;border:1px solid #2a333d}</style></head><body>'
                        '<h3>Drone QR Detector Live HUD (:8091) — <a style="color:#4aa3ff" href="http://localhost:8080">Main Dashboard (:8080)</a></h3>'
                        '<img id="f" src="/annotated.jpg"><pre id="s"></pre>'
                        '<script>setInterval(async()=>{document.getElementById("f").src="/annotated.jpg?t="+Date.now();'
                        'document.getElementById("s").textContent=JSON.stringify(await(await fetch("/status.json")).json(),null,2)},400)</script>'
                        '</body></html>'
                    ).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.end_headers()
                    self.wfile.write(html)

            def log_message(self, format, *args):
                return

        def serve():
            try:
                srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
                locator.get_logger().info(f'Drone QR HUD listening on http://0.0.0.0:{port}')
                srv.serve_forever()
            except OSError:
                pass

        threading.Thread(target=serve, daemon=True).start()

    def on_info(self, msg):
        if self.cam is None or self.cam.size != (msg.width, msg.height):
            self.cam = FisheyeModel.from_camera_info(msg, focal_scale=float(self.g('focal_scale')))

    def get_base_from_optical(self, stamp):
        """Lookup TF `base_link <- arducam_optical_frame`, falling back to configured static mount extrinsics."""
        drone_frame = str(self.g('drone_frame'))
        cam_frame = str(self.g('camera_frame'))
        try:
            t = self.tf_buffer.lookup_transform(drone_frame, cam_frame, stamp, timeout=Duration(seconds=0.05))
            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = quat_to_rot(t.transform.rotation)
            T[:3, 3] = [
                float(t.transform.translation.x),
                float(t.transform.translation.y),
                float(t.transform.translation.z),
            ]
            return T
        except Exception:
            return default_base_from_optical(
                mount_xyz=self.g('mount_xyz'),
                mount_rpy=self.g('mount_rpy'),
            )

    def decode_frame_multi_stage(self, gray):
        """Return `{payload: poly_fisheye_px (4x2)}` across central rectified view + candidate ray patches."""
        h, w = gray.shape[:2]
        model = self.cam.scaled(w, h)
        found_fisheye = {}

        # Stage 1: Central gnomonic rectified view (eliminates fisheye distortion for nadir / front codes)
        center_fov = float(self.g('center_fov_deg'))
        if center_fov > 0:
            c_view, c_meta = model.rectify_patch(gray, ray_d=None, fov_deg=center_fov, zoom_scale=1.0, max_size=1600)
            c_hits = qr_tools.decode_gray(c_view, wc=self.wc, zbar=self.zbar, cv_qr=self.cv_qr, upscale_small=False)
            for payload, poly_patch in c_hits.items():
                found_fisheye[payload] = model.patch_px_to_fisheye_px(poly_patch, c_meta)

        # Stage 2: Contrast candidate blobs + multi-scale QrNet boxes across the active fisheye circle
        fmask = model.circle_mask(w, h, max_theta_deg=82.0)
        cands = qr_tools.find_qr_candidates(gray, fisheye_mask=fmask, max_cands=int(self.g('max_candidates')))
        cand_centres = [c['center'] for c in cands]
        boxed_polys = []

        if self.qnet is not None:
            for tile in (160, 224):
                if cand_centres:
                    for conf, box_poly in self.qnet.scan(
                        gray, conf=float(self.g('net_conf')), centres=cand_centres, tile=tile, overlap=tile // 3
                    ):
                        wb, hb = box_poly.ptp(axis=0)
                        if 24 <= max(wb, hb) <= 160 and 0.55 <= wb / max(hb, 1.0) <= 1.75:
                            boxed_polys.append((conf, box_poly))
                            c_pt = box_poly.mean(axis=0)
                            if all(math.dist(c_pt, cc) > 40.0 for cc in cand_centres):
                                cand_centres.append((float(c_pt[0]), float(c_pt[1])))

        # Filter out candidate centres that are already inside a decoded QR polygon from Stage 1
        unread_centres = []
        for cx, cy in cand_centres:
            if any(math.dist((cx, cy), poly.mean(axis=0)) < 60.0 for poly in found_fisheye.values()):
                continue
            unread_centres.append((cx, cy))

        # Rectify a focused gnomonic pinhole patch looking directly along each unread candidate's 3D ray
        if unread_centres:
            rays = model.rays(np.array(unread_centres, dtype=np.float64))
            patch_fov = float(self.g('patch_fov_deg'))
            patch_zoom = float(self.g('patch_zoom'))
            for ray_d in rays[:int(self.g('max_candidates'))]:
                patch, p_meta = model.rectify_patch(
                    gray, ray_d=ray_d, fov_deg=patch_fov, zoom_scale=patch_zoom, max_size=512
                )
                hits = qr_tools.decode_gray(patch, wc=self.wc, zbar=self.zbar, cv_qr=self.cv_qr, upscale_small=True)
                rkey = tuple(np.round(ray_d, 2))
                if not hits and int(self.g('chase_avg_frames')) >= 2:
                    # Temporal phase-correlated frame stacking for low-SNR / 10m altitude codes
                    stack = self.chase_stacks.setdefault(rkey, [])
                    stack.append(cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA))
                    del stack[:-int(self.g('chase_avg_frames'))]
                    if len(stack) >= 2:
                        avg, _ = qr_tools.aligned_mean(stack)
                        clean = cv2.fastNlMeansDenoising(avg, None, 8, 5, 7)
                        sx = patch.shape[1] / 256.0
                        sy = patch.shape[0] / 256.0
                        for img_try in (clean, avg):
                            h_try = qr_tools.decode_gray(
                                img_try, wc=self.wc, zbar=self.zbar, cv_qr=self.cv_qr, upscale_small=True
                            )
                            if h_try:
                                hits = {pl: poly * [sx, sy] for pl, poly in h_try.items()}
                                break
                if hits:
                    self.chase_stacks.pop(rkey, None)
                    for payload, poly_patch in hits.items():
                        if payload not in found_fisheye:
                            found_fisheye[payload] = model.patch_px_to_fisheye_px(poly_patch, p_meta)

        if len(self.chase_stacks) > 24:
            self.chase_stacks = dict(list(self.chase_stacks.items())[-24:])

        # Stage 3: Full-frame direct decode fallback if nothing found yet
        if not found_fisheye:
            found_fisheye.update(
                qr_tools.decode_gray(gray, wc=self.wc, zbar=self.zbar, cv_qr=self.cv_qr, upscale_small=False)
            )

        return found_fisheye

    def update_tracks(self, detections_3d, now):
        """Update short-window confidence-weighted 3D estimates in the drone and camera frames."""
        timeout = float(self.g('track_timeout_s'))
        win_s = float(self.g('avg_window_s'))
        max_n = int(self.g('avg_window_samples'))
        stale = [k for k, v in self.tracks.items() if now - v['last_seen'] > timeout]
        for k in stale:
            del self.tracks[k]

        for payload, loc in detections_3d.items():
            w = max(0.05, float(loc['confidence_weight']))
            p_drone = np.array(loc['pos_drone'], dtype=np.float64)
            p_cam = np.array(loc['pos_camera'], dtype=np.float64)
            p_pnp = np.array(loc['pos_camera_pnp'], dtype=np.float64)
            if payload in self.tracks:
                tr = self.tracks[payload]
                samples = [s for s in tr.get('samples', []) if now - s[0] <= win_s]
                samples.append((now, p_drone, p_cam, p_pnp, w))
                tr['samples'] = samples[-max_n:]
                if len(tr['samples']) >= 3:
                    arr_d = np.array([s[1] for s in tr['samples']], dtype=np.float64)
                    arr_c = np.array([s[2] for s in tr['samples']], dtype=np.float64)
                    arr_p = np.array([s[3] for s in tr['samples']], dtype=np.float64)
                    ws = np.array([s[4] for s in tr['samples']], dtype=np.float64)
                    med = np.median(arr_d, axis=0)
                    dists = np.linalg.norm(arr_d - med, axis=1)
                    mad = float(np.median(dists)) + 0.05
                    keep = dists <= 2.5 * mad
                    ws_k = ws[keep]
                    ws_k /= max(ws_k.sum(), 1e-6)
                    tr['pos_drone'] = (arr_d[keep] * ws_k[:, None]).sum(axis=0)
                    tr['pos_camera'] = (arr_c[keep] * ws_k[:, None]).sum(axis=0)
                    tr['pos_camera_pnp'] = (arr_p[keep] * ws_k[:, None]).sum(axis=0)
                else:
                    ws = np.array([s[4] for s in tr['samples']], dtype=np.float64)
                    ws /= max(ws.sum(), 1e-6)
                    tr['pos_drone'] = sum(s[1] * wi for s, wi in zip(tr['samples'], ws))
                    tr['pos_camera'] = sum(s[2] * wi for s, wi in zip(tr['samples'], ws))
                    tr['pos_camera_pnp'] = sum(s[3] * wi for s, wi in zip(tr['samples'], ws))
                tr['weight_sum'] += w
                tr['hits'] += 1
                tr['last_seen'] = now
                tr['latest'] = loc
            else:
                self.tracks[payload] = {
                    'payload': payload,
                    'pos_drone': p_drone,
                    'pos_camera': p_cam,
                    'pos_camera_pnp': p_pnp,
                    'weight_sum': w,
                    'hits': 1,
                    'last_seen': now,
                    'latest': loc,
                    'samples': [(now, p_drone, p_cam, p_pnp, w)],
                }

    def compute_pairwise_distances(self):
        """Compute 3D distances (m) between all pairs of currently tracked QR codes."""
        keys = sorted(self.tracks.keys())
        pairs = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = self.tracks[keys[i]], self.tracks[keys[j]]
                d_plane = float(np.linalg.norm(a['pos_drone'] - b['pos_drone']))
                d_pnp = float(np.linalg.norm(a['pos_camera_pnp'] - b['pos_camera_pnp']))
                pairs.append({
                    'qr_1': keys[i],
                    'qr_2': keys[j],
                    'distance_m': round(d_plane, 4),
                    'distance_pnp_m': round(d_pnp, 4),
                })
        return pairs

    def on_image(self, msg):
        now = time.monotonic()
        if now - self.last_decode_t < 1.0 / max(0.2, float(self.g('rate_hz'))):
            return
        self.last_decode_t = now

        gray = image_to_gray(msg)
        found_polys = self.decode_frame_multi_stage(gray)
        T_base_opt = self.get_base_from_optical(msg.header.stamp)

        detections_3d = {}
        for payload, poly_px in found_polys.items():
            loc = self.cam.localize_qr_3d(
                poly_fisheye_px=poly_px,
                image_size=(msg.width, msg.height),
                qr_size_m=float(self.g('qr_size_m')),
                plane_distance_m=float(self.g('target_distance_m')),
                T_base_from_opt=T_base_opt,
                modules=int(self.g('qr_modules')),
            )
            detections_3d[payload] = loc
            pd = loc['pos_drone']
            pc = loc['pos_camera']
            self.get_logger().info(
                f'QR {payload!r} | drone_frame({self.g("drone_frame")}): '
                f'x={pd[0]:+.3f}m y={pd[1]:+.3f}m z={pd[2]:+.3f}m | '
                f'cam_frame: ({pc[0]:+.3f}, {pc[1]:+.3f}, {pc[2]:+.3f})m | '
                f'angle={loc["off_axis_deg"]:.1f}deg | {loc["px_per_module"]:.1f} px/mod'
            )

        self.update_tracks(detections_3d, now)
        self._update_hud(gray, found_polys)
        self.publish_results(msg.header.stamp, list(detections_3d.keys()))

    def _update_hud(self, gray, found_polys):
        try:
            scale = 960.0 / max(gray.shape[1], 1)
            vis_g = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1.0 else gray
            scale = vis_g.shape[1] / max(gray.shape[1], 1)
            ok_raw, raw_jpg = cv2.imencode('.jpg', vis_g, [cv2.IMWRITE_JPEG_QUALITY, 75])
            vis = cv2.cvtColor(vis_g, cv2.COLOR_GRAY2BGR)
            for payload, poly in found_polys.items():
                pts = (poly * scale).astype(np.int32)
                cv2.polylines(vis, [pts], True, (40, 255, 60), 2)
                p0 = pts.min(axis=0)
                cv2.putText(vis, f'QR: {payload}', (int(p0[0]), max(18, int(p0[1]) - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 255, 60), 2, cv2.LINE_AA)
            hdr = f'Drone 3D QR Detector (:8091) | Tracked QRs: {len(self.tracks)}'
            cv2.putText(vis, hdr, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (80, 240, 120), 2, cv2.LINE_AA)
            ok_ann, ann_jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 78])
            with self.hud_lock:
                if ok_raw:
                    self.raw_jpeg = raw_jpg.tobytes()
                if ok_ann:
                    self.annotated_jpeg = ann_jpg.tobytes()
        except Exception:
            pass

    def publish_results(self, stamp, current_frame_payloads):
        drone_frame = str(self.g('drone_frame'))
        pose_arr = PoseArray()
        pose_arr.header.stamp = stamp
        pose_arr.header.frame_id = drone_frame

        markers = MarkerArray(markers=[Marker(action=Marker.DELETEALL)])
        qrs_out = []

        for idx, (payload, tr) in enumerate(sorted(self.tracks.items())):
            pd = [round(float(v), 4) for v in tr['pos_drone']]
            pc = [round(float(v), 4) for v in tr['pos_camera']]
            p_pnp = [round(float(v), 4) for v in tr['pos_camera_pnp']]
            lat = tr['latest']

            qrs_out.append({
                'payload': payload,
                'in_current_frame': payload in current_frame_payloads,
                'hits': tr['hits'],
                'frame_id': drone_frame,
                'position_drone_m': {'x': pd[0], 'y': pd[1], 'z': pd[2]},
                'position_camera_m': {'x': pc[0], 'y': pc[1], 'z': pc[2]},
                'position_camera_pnp_m': {'x': p_pnp[0], 'y': p_pnp[1], 'z': p_pnp[2]},
                'distance_m': round(float(np.linalg.norm(tr['pos_drone'])), 4),
                'off_axis_deg': lat['off_axis_deg'],
                'px_per_module': lat['px_per_module'],
                'side_px': lat['side_px'],
                'center_px': lat['center_px'],
            })

            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = pd[0], pd[1], pd[2]
            pose.orientation.w = 1.0
            pose_arr.poses.append(pose)

            # 3D Polygon outline marker in drone_frame
            mk = Marker(type=Marker.LINE_STRIP, ns='drone_qr_boxes', id=idx * 2)
            mk.header.stamp = stamp
            mk.header.frame_id = drone_frame
            mk.pose.orientation.w = 1.0
            mk.scale.x = 0.05
            mk.color.r, mk.color.g, mk.color.b, mk.color.a = (0.1, 1.0, 0.2, 1.0)
            corners = lat['corners_drone']
            mk.points = [Point(x=c[0], y=c[1], z=c[2]) for c in corners + [corners[0]]]
            markers.markers.append(mk)

            # Text label marker in drone_frame
            txt = Marker(type=Marker.TEXT_VIEW_FACING, ns='drone_qr_labels', id=idx * 2 + 1)
            txt.header.stamp = stamp
            txt.header.frame_id = drone_frame
            txt.pose.position.x, txt.pose.position.y, txt.pose.position.z = pd[0], pd[1], pd[2] + 0.25
            txt.pose.orientation.w = 1.0
            txt.scale.z = 0.25
            txt.color.r, txt.color.g, txt.color.b, txt.color.a = (1.0, 1.0, 0.2, 1.0)
            txt.text = f'{payload}\n({pd[0]:+.2f}, {pd[1]:+.2f}, {pd[2]:+.2f})m'
            markers.markers.append(txt)

        pairs = self.compute_pairwise_distances()
        summary = {
            'drone_frame': drone_frame,
            'camera_frame': str(self.g('camera_frame')),
            'target_distance_m': float(self.g('target_distance_m')),
            'qr_size_m': float(self.g('qr_size_m')),
            'detected_qrs': qrs_out,
            'pairwise_distances_m': pairs,
            'timestamp': time.time(),
        }
        with self.hud_lock:
            self.status_json = summary
        self.det_pub.publish(String(data=json.dumps(summary)))
        self.pose_pub.publish(pose_arr)
        self.marker_pub.publish(markers)


def main():
    rclpy.init()
    node = QrDroneLocator()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
