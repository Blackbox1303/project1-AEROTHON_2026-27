#!/usr/bin/env python3
"""FIELD TEST, the Jetson's eyes (dev container, http://<jetson>:8097): test the real sensors and the mission's
decisions with no course material and no flying - PROPS OFF, the drone carried by hand.

Started by field_test.sh (it stops the calibration / red / ArUco pages and the ROS Arducam first: one process holds
the camera) together with the RealSense node (/camera/color/image_raw, /camera/depth/image_rect_raw, as the mission)
and field_link.py on the host (the Pixhawk: GPS / heading / altitude in, the motor cues out).

  views      /cam/arducam.jpg (straight down, boards / red overlaid), /cam/color.jpg (D435 forward, banners),
             /cam/depth.jpg (D435 depth, the obstacle grid)
  one-shot   detect_qr (boards in view), scan_qr (decode + the deep read: the decoder settings on 5 frames, then the
             image vs the start QR's), start_qr (the target: payload + image), red (the delivery_mapper rule in its 60
             deg cone), green / blue banner (banner_detector's rule on the D435 colour), obstacle (D435 depth grid)
  walk_qr    the QR area search walked by hand: lanes over the QR area (4 corners: captured here from the Pixhawk's
             GPS, or the geofence's QR slots), a board in view -> walk over it -> read it -> MATCH: "height down to 5 m,
             lower the payload" / another board: next / unsure: "height down 0.5 m" and again, never below 5.5 m (5 m
             is reached ONLY to lower the payload, user 2026-10-30) - the height is VIRTUAL (it can't be changed by
             hand): it is what the drone would fly, shown and cued; the camera's ground projection uses the carry
             height instead. Red under the drone: step away from it.
  walk_obstacle  D435 depth ahead: walk on / sidestep / height up (over) / height down (under) / back up.
  guidance   {say, yaw, go, dist, height, motor cue}: the page, the radio (via mav_agent) and field_link (motors).
API: GET /api/state, POST /api/cmd {name|code, arg, text}, POST /api/fc (field_link: Pixhawk state in, motor cue out),
GET /api/radio?since=N (mav_agent: a compact status + the events after N).
"""
import argparse
import json
import math
import os
import signal
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent.parent
WS = SRC.parent
for p in (HERE, SRC / 'aerothon_nav', SRC / 'vio_bringup/calibration'):
    sys.path.insert(0, str(p))
import field_proto as fp                                  # noqa: E402
from aerothon_nav import aruco_match, aruco_tools, qr_match, qr_tools   # noqa: E402
from aerothon_nav.fisheye import Fisheye                  # noqa: E402

DATA = WS / 'runs/field_test'
EARTH = 6378137.0
SEARCH_ALT, DROP_ALT, SCAN_FLOOR, STEP = 10.0, 5.0, 5.5, 0.5   # m: the mission's QR area heights
CLIMB = 0.5                                                   # m/s the virtual height changes at
RED = {'h1': (0, 10), 'h2': (170, 180), 's': 100, 'v': 70, 'ang': 60.0}   # delivery_mapper red_*
BANNER = {'green': ((68, 92), 100, 30), 'blue': ((100, 125), 140, 30)}   # (h, s_min, v_min): the real green banner
#                                                                            is H 81-86 (teal), grass H 19-46, the sky
#                                                                            H ~106 S ~100 (blue s_min 140); the real camera's
#                                                                            colours are paler than the sim's (s 160)


def wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def side_words(rel):
    """A bearing relative to the nose (deg, + right) -> words."""
    a = abs(rel)
    if a < 20:
        return 'forward'
    if a > 160:
        return 'backward'
    lr = 'right' if rel > 0 else 'left'
    if a < 60:
        return f'forward-{lr}'
    if a <= 120:
        return lr
    return f'back-{lr}'


# ------------------------------------------------------------------ cameras
class Arducam:
    """CSI CAM0 straight down (aruco_test.Camera: GStreamer nvarguscamerasrc), read in its own thread."""

    def __init__(self, a):
        from aruco_test import Camera
        self.cam = Camera(SimpleNamespace(source=a.arducam, sensor_id=a.sensor_id, width=a.width, height=a.height,
                                          fps=a.fps, exposure_ms=a.exposure_ms))
        self.frame, self.seq, self.t, self.err = None, 0, 0.0, 'starting'
        threading.Thread(target=self.loop, daemon=True).start()

    def loop(self):
        while True:
            try:
                f = self.cam.read()
            except Exception as e:
                self.err = f'Arducam: {e}'
                time.sleep(2)
                continue
            if f is None:
                self.err = 'Arducam: no frame'
                time.sleep(0.3)
                continue
            self.frame, self.seq, self.t, self.err = f, self.seq + 1, time.time(), ''

    def latest(self, max_age=2.0):
        return self.frame if self.frame is not None and time.time() - self.t < max_age else None


class D435:
    """The RealSense node's colour + depth (realsense-ros has no Python bindings here). Mounted UPSIDE DOWN
    (vio_bringup/config/rig.yaml): both images turned 180 deg (--d435-flip) so up is up and left is left."""

    def __init__(self, a):
        self.flip = a.d435_flip
        self.color = self.depth = None
        self.t_color = self.t_depth = 0.0
        self.k_color = self.k_depth = None            # (fx, fy, cx, cy) after the flip
        self.err = ''
        try:
            import rclpy
            from rclpy.qos import qos_profile_sensor_data
            from sensor_msgs.msg import CameraInfo, Image
        except ImportError as e:
            self.err = f'D435: no ROS 2 here ({e})'
            return
        rclpy.init()
        n = self.node = rclpy.create_node('field_vision')
        n.create_subscription(Image, '/camera/color/image_raw', self.on_color, qos_profile_sensor_data)
        n.create_subscription(Image, '/camera/depth/image_rect_raw', self.on_depth, qos_profile_sensor_data)
        n.create_subscription(CameraInfo, '/camera/color/camera_info', lambda m: self.on_info(m, 'k_color'),
                              qos_profile_sensor_data)
        n.create_subscription(CameraInfo, '/camera/depth/camera_info', lambda m: self.on_info(m, 'k_depth'),
                              qos_profile_sensor_data)
        threading.Thread(target=rclpy.spin, args=(n,), daemon=True).start()

    def on_info(self, m, attr):
        fx, fy, cx, cy = m.k[0], m.k[4], m.k[2], m.k[5]
        if self.flip:
            cx, cy = m.width - 1 - cx, m.height - 1 - cy
        setattr(self, attr, (fx, fy, cx, cy))

    def turn(self, img):
        return cv2.rotate(img, cv2.ROTATE_180) if self.flip else img

    def on_color(self, m):
        ch = 3 if m.encoding in ('rgb8', 'bgr8') else 1
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.step)[:, :m.width * ch].reshape(m.height, m.width, ch)
        if m.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.color, self.t_color = self.turn(np.ascontiguousarray(img)), time.time()

    def on_depth(self, m):
        img = np.frombuffer(m.data, np.uint16).reshape(m.height, m.step // 2)[:, :m.width]
        self.depth, self.t_depth = self.turn(np.ascontiguousarray(img)), time.time()

    def latest_color(self):
        return self.color if self.color is not None and time.time() - self.t_color < 2.0 else None

    def latest_depth(self):
        return self.depth if self.depth is not None and time.time() - self.t_depth < 2.0 else None

    def status(self):
        if self.err:
            return self.err
        bad = [n for n, t in (('colour', self.t_color), ('depth', self.t_depth)) if time.time() - t > 2.0]
        return f'D435: no {" / ".join(bad)} (RealSense node up? USB 3?)' if bad else ''


def load_lens(path, w, h):
    """The Arducam's calibration, else the nominal 140 deg equidistant lens (arducam_real.launch.py)."""
    try:
        import yaml
        with open(path) as f:
            c = yaml.safe_load(f)
        info = SimpleNamespace(distortion_model=c['distortion_model'], k=c['camera_matrix']['data'],
                               d=c['distortion_coefficients']['data'], width=c['image_width'],
                               height=c['image_height'], header=SimpleNamespace(frame_id='arducam'))
        return Fisheye(info), f'calibration {Path(path).name}'
    except Exception:
        f = (w / 2) / math.radians(70.0)
        info = SimpleNamespace(distortion_model='equidistant', k=[f, 0, w / 2, 0, f, h / 2, 0, 0, 1], d=[0, 0, 0, 0],
                               width=w, height=h, header=SimpleNamespace(frame_id='arducam'))
        return Fisheye(info), 'NO calibration: nominal 140 deg lens (distances rough)'


# ------------------------------------------------------------------ detectors
class Eyes:
    """The detectors, each on the latest frame. Body frame of a ray from the down-looking Arducam (its optical frame:
    x right, y = towards the tail, z down - arducam_real.launch.py MOUNT): forward = -y, right = x."""

    def __init__(self, a):
        self.a = a
        self.lens, self.lens_note = load_lens(a.calib, a.width, a.height)
        self.wc = qr_tools.make_wechat()
        self.net = qr_tools.QrNet()
        try:
            from pyzbar.pyzbar import ZBarSymbol, decode
            self.zbar = lambda g: [r.data.decode('utf-8', 'replace') for r in decode(g, symbols=[ZBarSymbol.QRCODE])]
        except Exception:
            self.zbar = None
        self.cone_cache = {}
        self.lock = threading.Lock()                  # one detector run at a time (the DNN is shared)

    # geometry
    def ray(self, px, shape):
        return self.lens.rays(np.asarray(px, float).reshape(-1, 2), scale=shape[1] / self.lens.size[0])

    def body(self, ray, carry_h):
        """Unit ray (Arducam optical) -> (bearing deg from the nose, + right; off-axis deg; ground metres away)."""
        fwd, right, down = -ray[1], ray[0], ray[2]
        off = math.degrees(math.acos(max(-1.0, min(1.0, down))))
        dist = carry_h * math.tan(math.radians(min(off, 80.0)))
        return math.degrees(math.atan2(right, fwd)), off, dist

    # QR boards / ArUco markers
    def boards(self, img, conf=0.45):
        """[{poly, conf, c}]: WeChat's detector on the whole frame (boards carried over: big) + 1/2-scale tiles;
        --marker aruco: the marker-like squares on undistorted views (aruco_match, as qr_locator: nothing read)."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if self.a.marker == 'aruco':
            keep = []
            for ray in aruco_tools.view_rays(55.0, 5):
                view, meta = self.lens.rectify_patch(gray, ray_d=ray, fov_deg=100.0 if ray is None else 90.0,
                                                     zoom_scale=1.0, max_size=1100)
                for q in aruco_match.marker_quads(view):
                    px = self.lens.patch_px_to_fisheye_px(q, meta)
                    c = px.mean(axis=0)
                    if np.isfinite(px).all() and all(np.linalg.norm(k['c'] - c) > 20 for k in keep):
                        keep.append({'poly': px, 'conf': round(aruco_match.sharpness(view, q), 2), 'c': c})
            return gray, keep
        found = [(c, p) for c, p in self.net.boxes(gray, conf=conf)]
        half = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        found += [(c, p * 2.0) for c, p in self.net.scan(half, conf=conf, multi_scale=False)]
        keep = []
        for c, p in sorted(found, key=lambda t: -t[0]):
            if all(qr_tools.iou(p, q['poly']) < 0.3 for q in keep):
                keep.append({'poly': p, 'conf': round(c, 2), 'c': p.mean(axis=0)})
        return gray, keep

    def patch(self, gray, poly):
        """The board straightened: a virtual pinhole along its centre ray, its field ~1.6x the board."""
        c = poly.mean(axis=0)
        rc = self.ray(c, gray.shape)[0]
        corners = self.ray(poly, gray.shape)
        ang = max(math.degrees(math.acos(max(-1.0, min(1.0, float(r @ rc))))) for r in corners)
        fov = min(120.0, max(12.0, 3.2 * ang))
        p, _ = self.lens.rectify_patch(gray, ray_d=rc, fov_deg=fov, zoom_scale=1.0, max_size=512)
        return p

    def decode_board(self, gray, poly):
        """-> (payload, how) or (None, ''): WeChat, then the decoder settings (qr_match.variants), then zbar. ArUco:
        never read (matched by image)."""
        if self.a.marker == 'aruco':
            return None, ''
        p = self.patch(gray, poly)
        for name, img in qr_match.variants(p):
            hit = qr_tools.wechat_decode(self.wc, img)
            if hit:
                return next(iter(hit)), name
        if self.zbar:
            for name, img in (('zbar', p), ('zbar-raw', self.crop(gray, poly))):
                r = self.zbar(img)
                if r:
                    return r[0], name
        return None, ''

    @staticmethod
    def crop(gray, poly, margin=0.3):
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        h, w = gray.shape
        return gray[int(max(0, y0 - my)):int(min(h, y1 + my)), int(max(0, x0 - mx)):int(min(w, x1 + mx))]

    def signature(self, gray, poly):
        p = self.patch(gray, poly)
        if self.a.marker != 'aruco':
            return qr_match.signature(p)
        qs = aruco_match.marker_quads(p)              # the marker in the straightened patch, nearest its centre
        if not qs:
            return None
        h, w = p.shape[:2]
        return aruco_match.signature(p, min(qs, key=lambda q: np.hypot(*(q.mean(axis=0) - [w / 2, h / 2]))))

    # red zone
    def cone(self, w, h, ang):
        key = (w, h, ang)
        if key not in self.cone_cache:
            s = 4
            us, vs = np.meshgrid(np.arange(0, w, s) + s / 2, np.arange(0, h, s) + s / 2)
            r = self.ray(np.column_stack([us.ravel(), vs.ravel()]), (h, w))
            m = (r[:, 2] > math.cos(math.radians(ang))).reshape(us.shape).astype(np.uint8)
            self.cone_cache[key] = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
        return self.cone_cache[key]

    def red(self, img, carry_h):
        """The delivery_mapper rule in its cone (on a 1/2-scale frame) -> dict + the mask."""
        small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        h, w = small.shape[:2]
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        red = ((((H >= RED['h1'][0]) & (H <= RED['h1'][1])) | ((H >= RED['h2'][0]) & (H <= RED['h2'][1])))
               & (S >= RED['s']) & (V >= RED['v']))
        cone = self.cone(w, h, RED['ang'])
        red_in = red & cone
        share = float(red_in.sum()) / max(1, int(cone.sum()))
        out = {'share': round(100 * share, 1), 'found': False}
        nb, _, stats, cents = cv2.connectedComponentsWithStats(red_in.astype(np.uint8), connectivity=8)
        if nb > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = stats[i, cv2.CC_STAT_AREA] / float(w * h)
            if area >= 0.002:
                ray = self.ray(cents[i] * 2.0, img.shape)[0]
                b, off, d = self.body(ray, carry_h)
                under = bool(red_in[h // 2 - 2:h // 2 + 3, w // 2 - 2:w // 2 + 3].any())
                out.update(found=True, bearing=round(b), off=round(off), dist=round(d, 1), under=under,
                           blob=round(100 * area, 2))
        return out, red_in

    # banners (D435 colour)
    def banners(self, img, colour, k):
        """The course banners (2026-10-30: plain green entry / blue exit, a thin white border) on the D435 colour: a
        solid colour blob, elongated; the white border round it adds confidence (evidence only: thin or far, it's
        missing), letters inside are allowed (the sim's lettered ones) but not needed."""
        (h_lo, h_hi), s_min, v_min = BANNER[colour]
        ds = 2
        small = cv2.resize(img, (img.shape[1] // ds, img.shape[0] // ds), interpolation=cv2.INTER_AREA)
        mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), (h_lo, s_min, v_min), (h_hi, 255, 255))
        joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        found = []
        blobs = list(cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])
        for c in list(blobs):                     # a banner joined to a pole of its colour (the green poles at the
            if cv2.contourArea(c) < 400:          # mouth touch the doormat): its pieces once the thin part is opened away
                continue
            blob = np.zeros_like(joined)
            cv2.drawContours(blob, [c], -1, 255, -1)
            for kk in (9, 15):
                blobs += [p for p in cv2.findContours(cv2.morphologyEx(
                    cv2.bitwise_and(blob, joined), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] if cv2.contourArea(p) >= 40]
        for c in blobs:
            area = cv2.contourArea(c)
            if area < 40:
                continue
            hull = cv2.convexHull(c)
            x, y, w, h = cv2.boundingRect(hull * ds)
            crop = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2HSV)
            hm = np.zeros(crop.shape[:2], np.uint8)
            cv2.fillConvexPoly(hm, hull * ds - [x, y], 255)
            n_hull = cv2.countNonZero(hm)
            col = cv2.bitwise_and(cv2.inRange(crop, (h_lo, s_min, v_min), (h_hi, 255, 255)), hm)
            n_col = cv2.countNonZero(col)
            if not n_hull or not n_col:
                continue
            v_col = float(np.median(crop[..., 2][col > 0]))
            white = cv2.inRange(crop, (0, 0, int(0.85 * v_col)), (180, 100, 255))
            n_white = cv2.countNonZero(cv2.bitwise_and(white, hm))
            solidity, wf = (n_col + n_white) / n_hull, n_white / n_hull
            (_, _), (rw, rh), ang = cv2.minAreaRect(hull)
            aspect = max(rw, rh) / max(1e-6, min(rw, rh))
            tilt = abs(((ang if rw >= rh else ang + 90.0) + 90.0) % 180.0 - 90.0)   # long side from level: a pole ~90
            ring = cv2.subtract(cv2.dilate(hm, np.ones((7, 7), np.uint8)), hm)    # just outside the hull
            rim = cv2.countNonZero(cv2.bitwise_and(white, ring)) / max(1, cv2.countNonZero(ring))
            ok = solidity >= 0.75 and wf <= 0.5 and aspect >= 4.0 and tilt <= 35.0   # the real ones: 15 : 1 face-on
            m = cv2.moments(hull)
            cx, cy = m['m10'] / m['m00'] * ds, m['m01'] / m['m00'] * ds
            fx, fy, kx, ky = k or (img.shape[1] * 0.9, img.shape[1] * 0.9, img.shape[1] / 2, img.shape[0] / 2)
            share = 100.0 * area * ds * ds / (img.shape[0] * img.shape[1])
            found.append({'ok': ok, 'az': round(math.degrees(math.atan2(cx - kx, fx))),
                          'el': round(-math.degrees(math.atan2(cy - ky, fy))), 'area': round(share, 2),
                          'white': round(wf, 2), 'solid': round(solidity, 2), 'aspect': round(aspect, 1),
                          'rim': round(rim, 2), 'score': area * solidity * (0.9 + 0.25 * min(1.0, rim / 0.35)),
                          'hull': (hull * ds).tolist()})
        found.sort(key=lambda d: (not d['ok'], -d['score']))
        return found

    # obstacles (D435 depth)
    @staticmethod
    def obstacles(depth, k):
        """5 columns x 3 rows of the depth image -> the 5th-percentile range in each (m, None = no data)."""
        h, w = depth.shape
        z = depth.astype(np.float32) / 1000.0
        z[(z < 0.15) | (z > 8.0)] = np.nan
        grid = []
        for r in range(3):
            row = []
            for c in range(5):
                cell = z[r * h // 3:(r + 1) * h // 3, c * w // 5:(c + 1) * w // 5]
                v = cell[np.isfinite(cell)]
                row.append(round(float(np.percentile(v, 5)), 2) if v.size > 0.03 * cell.size else None)
            grid.append(row)
        fx = (k or (w * 0.6,))[0]
        cols = [round(math.degrees(math.atan2((c + 0.5) * w / 5 - w / 2, fx))) for c in range(5)]
        return {'grid': grid, 'col_deg': cols}


# ------------------------------------------------------------------ the app
class App:
    def __init__(self, a):
        self.a = a
        DATA.mkdir(parents=True, exist_ok=True)
        self.ard = Arducam(a)
        self.d435 = D435(a)
        self.eyes = Eyes(a)
        self.lock = threading.Lock()
        self.events, self.n_ev = [], 0
        self.fc, self.fc_t = {}, 0.0
        self.fc_req = None                            # a mode / winch test for field_link (see command)
        self.motors_on = False
        self.spin_until = 0.0                         # spin_test: the cw chase on all four until then
        self.levels = {'low': a.spin_low, 'mid': a.spin_mid, 'high': a.spin_high}
        self.mode = 'idle'                            # idle | walk_qr | walk_obstacle
        self.guide = self.idle_guide()
        self.overlay = {'boards': [], 'red': None, 'banners': [], 'obst': None}
        self.last = {}                                # name -> the last one-shot result
        self.carry_h, self.lane_spacing = a.carry_height, a.lane_spacing
        self.search_alt = SEARCH_ALT
        self.valt = self.valt_goal = SEARCH_ALT       # the virtual height, and where it is going
        self.start = self.load_start()
        self.corners = self.load_json('corners.json', {}).get('gps', [None] * 4)
        self.reset_walk()
        threading.Thread(target=self.walk_loop, daemon=True).start()
        self.event(f'field test up: Arducam {a.arducam} ({self.eyes.lens_note}), D435 via ROS, start QR '
                   f'{self.start_text()}')

    # ---- bookkeeping
    def event(self, text):
        with self.lock:
            self.n_ev += 1
            self.events.append({'n': self.n_ev, 't': time.strftime('%H:%M:%S'), 'text': text})
            self.events = self.events[-200:]
        print(f'{time.strftime("%H:%M:%S")} {text}', flush=True)
        with open(DATA / 'events.log', 'a') as f:
            f.write(f'{time.strftime("%F %T")} {text}\n')

    def load_json(self, name, default):
        try:
            return json.loads((DATA / name).read_text())
        except Exception:
            return default

    def load_start(self):
        d = self.load_json('start_qr.json', {})
        sig = np.array(d['sig'], np.float32) if d.get('sig') else None
        return {'payload': d.get('payload'), 'sig': sig}

    def save_start(self):
        s = self.start
        (DATA / 'start_qr.json').write_text(json.dumps({'payload': s['payload'],
                                                        'sig': s['sig'].tolist() if s['sig'] is not None else None,
                                                        'saved': time.ctime()}))

    def start_text(self):
        p, sig = self.start['payload'], self.start['sig'] is not None
        if p is None and not sig:
            return 'not set'
        return (repr(p) if p else '(no data: image only)') + (' + image' if sig and p else '')

    # ---- the Pixhawk (field_link)
    def pose(self):
        """(x east, y north, heading deg) in the QR area's frame from the Pixhawk's GPS, or None."""
        f = self.fc
        if time.time() - self.fc_t > 3.0 or not f.get('fix') or f.get('fix', 0) < 3 or f.get('lat') is None:
            return None
        o = self.origin()
        if o is None:
            return None
        x, y = self.local(f['lat'], f['lon'], o)
        return x, y, float(f.get('hdg') or 0.0)

    def origin(self):
        pts = [c for c in self.corners if c]
        return pts[0] if pts else None

    @staticmethod
    def local(lat, lon, o):
        return (math.radians(lon - o[1]) * EARTH * math.cos(math.radians(o[0])), math.radians(lat - o[0]) * EARTH)

    def area(self):
        """The QR area's 4 corners in the local frame, counter-clockwise, or None."""
        if not all(self.corners):
            return None
        o = self.origin()
        pts = [self.local(c[0], c[1], o) for c in self.corners]
        cx, cy = sum(p[0] for p in pts) / 4, sum(p[1] for p in pts) / 4
        return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    # ---- commands
    def command(self, name, arg=0.0, text=None):
        n = name
        if n in ('detect_qr', 'scan_qr', 'start_qr', 'red', 'green', 'blue', 'obstacle'):
            return getattr(self, 'do_' + n)(text) if n == 'start_qr' else getattr(self, 'do_' + n)()
        if n in ('walk_qr', 'walk_obstacle'):
            self.reset_walk()
            self.mode = n
            what = 'QR area search' if n == 'walk_qr' else 'obstacle avoidance'
            msg = f'{what} walk-through started - carry the drone level, nose ahead, {self.carry_h:.1f} m up'
            if n == 'walk_qr':
                msg += (f'; area {"set" if self.area() else "NOT set (camera-only guidance)"}, start QR '
                        f'{self.start_text()}, virtual height {self.valt:.1f} m')
            self.event(msg)
            return True, msg
        if n == 'walk_stop':
            self.spin_until = 0.0
            self.mode = 'idle'
            self.guide = self.idle_guide()
            self.event('walk-through stopped')
            return True, 'stopped'
        if n == 'motors':
            self.motors_on = bool(float(arg))
            if not self.motors_on:
                self.spin_until = 0.0
            self.event(f'motor cues {"ON - props must be OFF" if self.motors_on else "off"}')
            return True, f'motor cues {"on" if self.motors_on else "off"}'
        if n == 'spin_test':
            secs = min(20.0, float(arg) or 6.0)
            if secs < 0:
                self.spin_until = 0.0
                return True, 'spin test stopped'
            self.spin_until = time.time() + secs
            msg = (f'spin test: all four motors in turn FR RR RL FL for {secs:.0f} s at {self.levels["mid"]:.0f} % '
                   f'- PROPS OFF')
            self.event(msg)
            return True, msg
        if n == 'virtual_alt':
            v = float(arg)
            if not SCAN_FLOOR <= v <= 30.0:
                return False, f'virtual height {v:g} m: {SCAN_FLOOR:g}-30 m (5 m is for the drop only)'
            self.search_alt = self.valt = self.valt_goal = v
            self.event(f'virtual search height {v:.1f} m')
            return True, f'virtual height {v:.1f} m'
        if n == 'corner':
            k = int(float(arg))
            if k == 0:
                self.corners = [None] * 4
                self.save_corners()
                self.event('QR area corners cleared')
                return True, 'corners cleared'
            if not 1 <= k <= 4:
                return False, 'corner slot 1-4 (0 clears)'
            f = self.fc
            if time.time() - self.fc_t > 3.0 or not f.get('lat') or (f.get('fix') or 0) < 3:
                return False, f'no GPS fix from the Pixhawk (fix {f.get("fix")}, {f.get("sats")} sats)'
            self.corners[k - 1] = [f['lat'], f['lon']]
            self.save_corners()
            msg = f'corner {k}: {f["lat"]:.7f}, {f["lon"]:.7f} ({f.get("sats")} sats)'
            if self.area():
                a = self.area()
                msg += f' - area {self.poly_area(a):.0f} m2, {len(self.lanes()) // 2} lanes'
            self.event(msg)
            return True, msg
        if n == 'area_geofence':
            return self.area_from_geofence()
        if n == 'next':
            if self.mode == 'walk_qr' and self.board is not None:
                self.finish_board('skipped by hand')
                return True, 'board skipped'
            return False, 'no board in hand'
        if n in ('carry_height', 'lane_spacing'):
            v = float(arg)
            lim = (0.3, 3.0) if n == 'carry_height' else (1.0, 15.0)
            if not lim[0] <= v <= lim[1]:
                return False, f'{n}: {lim[0]}-{lim[1]} m'
            if n == 'carry_height':
                self.carry_h = v
            else:
                self.lane_spacing = v
                self.lane_i = 0
            self.event(f'{n.replace("_", " ")} {v:.1f} m')
            return True, f'{n} {v:.1f} m'
        if n in ('set_mode', 'winch'):                # the autopilot tests: field_link does them, its answer back
            self.fc_req_n = getattr(self, 'fc_req_n', 0) + 1
            rid = self.fc_req_n
            self.fc_req = {'id': rid, 'do': n, 'arg': float(arg)}
            t0 = time.time()
            while time.time() - t0 < 8.0:
                done = (self.fc or {}).get('req_done') or {}
                if done.get('id') == rid:
                    self.fc_req = None
                    self.event(f'{"MODE" if n == "set_mode" else "WINCH"} {done.get("msg", "")}')
                    return bool(done.get('ok')), str(done.get('msg', ''))
                time.sleep(0.1)
            self.fc_req = None
            return False, 'no answer from field_link (the Pixhawk link)'
        if n == 'status':
            return True, self.guide.get('say', '')
        return False, f'unknown command {name}'

    def save_corners(self):
        (DATA / 'corners.json').write_text(json.dumps({'gps': self.corners}))
        self.lane_i = 0

    def area_from_geofence(self):
        try:
            gf = json.loads((WS / 'runs/gcs/geofence.json').read_text())
        except Exception as e:
            return False, f'no geofence: {e}'
        q = gf.get('qr') or gf.get('delivery') or []
        if len(q) != 4:
            return False, f'the geofence has {len(q)} QR-area corners, needs 4'
        self.corners = [[float(p[0]), float(p[1])] for p in q]
        self.save_corners()
        msg = f'QR area from the geofence: {self.poly_area(self.area()):.0f} m2, {len(self.lanes()) // 2} lanes'
        self.event(msg)
        return True, msg

    @staticmethod
    def poly_area(p):
        return abs(sum(p[i][0] * p[i - 1][1] - p[i - 1][0] * p[i][1] for i in range(len(p)))) / 2

    # ---- one-shot detections (each also leaves its overlay on the views)
    def frame_or_fail(self):
        img = self.ard.latest()
        if img is None:
            raise RuntimeError(self.ard.err or 'no Arducam frame')
        return img

    def describe_board(self, b, shape):
        ray = self.eyes.ray(b['c'], shape)[0]
        bear, off, dist = self.eyes.body(ray, self.carry_h)
        return bear, off, dist

    def do_detect_qr(self):
        try:
            img = self.frame_or_fail()
        except RuntimeError as e:
            return False, str(e)
        with self.eyes.lock:
            _, bs = self.eyes.boards(img)
        self.overlay['boards'] = [{'poly': b['poly'].tolist(), 'label': f'{b["conf"]:.2f}'} for b in bs]
        if not bs:
            msg = 'detect QR: no board in view'
        else:
            parts = []
            for b in bs[:4]:
                bear, off, dist = self.describe_board(b, img.shape)
                parts.append(f'{side_words(bear)} {dist:.1f} m ({off:.0f} deg off, conf {b["conf"]:.2f})')
            msg = f'detect QR: {len(bs)} board(s): ' + '; '.join(parts)
        self.last['detect_qr'] = msg
        self.event(msg)
        return True, msg

    def read_boards(self, frames=5):
        """Decode every board in view over up to `frames` frames -> [{payload, how, bear, dist, sigs}]."""
        out, seq = [], -1
        for _ in range(frames):
            t0 = time.time()
            while self.ard.seq == seq and time.time() - t0 < 1.0:
                time.sleep(0.02)
            seq = self.ard.seq
            img = self.ard.latest()
            if img is None:
                break
            with self.eyes.lock:
                gray, bs = self.eyes.boards(img)
                for b in bs:
                    bear, off, dist = self.describe_board(b, img.shape)
                    slot = next((o for o in out if abs(wrap(o['bear'] - bear)) < 25 and abs(o['dist'] - dist) < 0.6),
                                None)
                    if slot is None:
                        slot = {'payload': None, 'how': '', 'bear': bear, 'dist': dist, 'sigs': [], 'poly': b['poly']}
                        out.append(slot)
                    slot['poly'] = b['poly']
                    if slot['payload'] is None:
                        slot['payload'], slot['how'] = self.eyes.decode_board(gray, b['poly'])
                    s = self.eyes.signature(gray, b['poly'])
                    if s is not None:
                        slot['sigs'].append(s)
            if out and all(o['payload'] is not None for o in out):
                break
        return out

    def judge(self, o):
        """A board read -> (verdict, text): match | different | unsure | read (no start QR to compare)."""
        st = self.start
        if o['payload'] is not None:
            if st['payload']:
                return ('match', 'MATCH') if o['payload'] == st['payload'] else ('different', 'not the target')
            if st['sig'] is not None and o['sigs']:
                s = max(qr_match.similarity(st['sig'], x) or -1 for x in o['sigs'])
                if s >= self.a.match_hi:
                    return 'match', f'MATCH by image ({s:.2f})'
            return 'read', 'no start QR to compare'
        aruco = self.a.marker == 'aruco'              # matched by image only, the mission's thresholds
        sim, hi, lo = ((aruco_match.similarity, self.a.aruco_hi, self.a.aruco_lo) if aruco else
                       (qr_match.similarity, self.a.match_hi, self.a.match_lo))
        pre, other = ('', 'another marker') if aruco else ('unread, ', 'another board')
        if st['sig'] is None:
            return 'unsure', f'{pre}no start {"marker" if aruco else "QR"} image to compare'
        sc = sorted(sim(st['sig'], x) or -1.0 for x in o['sigs'])
        if len(sc) < 3:
            return 'unsure', f'{pre}image seen {len(sc)}x (needs 3)'
        s = float(np.median(sc[-3:]))
        if s >= hi:
            return 'match', f'{pre}image MATCH ({s:.2f})'
        if s <= lo:
            return 'different', f'{pre}image of {other} ({s:.2f})'
        return 'unsure', f'{pre}image unsure ({s:.2f})'

    def do_scan_qr(self):
        if self.ard.latest() is None:
            return False, self.ard.err or 'no Arducam frame'
        res = self.read_boards()
        self.overlay['boards'] = [{'poly': np.asarray(o['poly']).tolist(), 'label': (o['payload'] or '?')[:24]}
                                  for o in res]
        if not res:
            msg = 'scan QR: no board in view'
        else:
            parts = []
            for o in res:
                verdict, why = self.judge(o)
                read = f'read {o["payload"]!r} ({o["how"]})' if o['payload'] is not None else \
                    'marker' if self.a.marker == 'aruco' else 'NOT decoded'
                tip = (f' -> height down to {DROP_ALT:g} m and lower the payload' if verdict == 'match' else
                       f' -> height down {STEP:g} m and read again (not below {SCAN_FLOOR:g} m)'
                       if verdict == 'unsure' else '')
                parts.append(f'{side_words(o["bear"])} {o["dist"]:.1f} m: {read}: {why}{tip}')
            msg = 'scan QR: ' + ' | '.join(parts)
        self.last['scan_qr'] = msg
        self.event(msg)
        return True, msg

    def do_start_qr(self, text=None):
        if text is not None and str(text).strip():
            self.start = {'payload': str(text).strip(), 'sig': None}
            self.save_start()
            msg = f'start QR typed: {self.start["payload"]!r} (no image)'
            self.event(msg)
            return True, msg
        if self.ard.latest() is None:
            return False, self.ard.err or 'no Arducam frame'
        res = self.read_boards()
        if not res:
            return False, 'start QR: no board in view - hold the drone over it'
        o = min(res, key=lambda o: o['dist'])         # the one under the drone
        sig = o['sigs'][-1] if o['sigs'] else None
        if o['payload'] is None and sig is None:
            return False, 'start QR: board seen but neither decoded nor imaged - closer / steadier'
        self.start = {'payload': o['payload'], 'sig': sig}
        self.save_start()
        msg = f'start QR set: {self.start_text()} ({o["how"] or "image"})'
        self.event(msg)
        return True, msg

    def do_red(self):
        try:
            img = self.frame_or_fail()
        except RuntimeError as e:
            return False, str(e)
        r, mask = self.eyes.red(img, self.carry_h)
        self.overlay['red'] = mask
        if not r['found']:
            msg = f'red zone: none ({r["share"]} % of the {RED["ang"]:.0f} deg cone red)'
        else:
            where = 'UNDER the drone' if r['under'] else f'{side_words(r["bearing"])} {r["dist"]} m'
            msg = f'red zone {where}: {r["share"]} % of the cone - the mission avoids it (no-fly + 1.5 m margin)'
        self.last['red'] = msg
        self.event(msg)
        return True, msg

    def banner(self, colour):
        img = self.d435.latest_color()
        if img is None:
            return False, self.d435.status() or 'no D435 colour'
        bs = self.eyes.banners(img, colour, self.d435.k_color)
        self.overlay['banners'] = [dict(b, colour=colour) for b in bs[:4]]
        good = [b for b in bs if b['ok']]
        if good:
            b = good[0]
            turn = 'ahead' if abs(b['az']) < 5 else f'{abs(b["az"])} deg {"right" if b["az"] > 0 else "left"}'
            msg = (f'{colour} banner FOUND {turn}, {b["el"]:+d} deg up/down, {b["area"]} % of the view (white border '
                   f'{"seen" if b["rim"] >= 0.2 else "not seen"} {b["rim"]}, solid {b["solid"]}, aspect {b["aspect"]})')
        elif bs:
            b = bs[0]
            msg = (f'{colour}: colour seen but not a banner (solid {b["solid"]} needs >= 0.75, aspect {b["aspect"]} '
                   f'>= 4 and level, white inside {b["white"]} <= 0.5)')
        else:
            msg = f'{colour} banner: none in the D435 view'
        self.last[colour] = msg
        self.event(msg)
        return True, msg

    def do_green(self):
        return self.banner('green')

    def do_blue(self):
        return self.banner('blue')

    def do_obstacle(self):
        d = self.d435.latest_depth()
        if d is None:
            return False, self.d435.status() or 'no D435 depth'
        o = self.eyes.obstacles(d, self.d435.k_depth)
        self.overlay['obst'] = o
        g = self.obstacle_guide(o)
        msg = f'obstacles: {g["say"]} (nearest ahead {self.fmt(o["grid"][1][2])})'
        self.last['obstacle'] = msg
        self.event(msg)
        return True, msg

    @staticmethod
    def fmt(v):
        return f'{v:.1f} m' if v is not None else 'none'

    # ---- guidance
    @staticmethod
    def idle_guide():
        return {'mode': 'idle', 'state': '-', 'say': 'no walk-through running', 'yaw': None, 'go': None, 'dist': None,
                'height': 0, 'motor': {'pattern': 'off', 'deg': 0, 'level': 'mid'}}

    def cue(self, say, state, yaw=None, go=None, dist=None, extra=None, h=None):
        """Guidance + its motor cue. yaw: turn this much (deg, + right) first; go: walk this way (deg from the nose);
        h: the height cue (-1 / 0 / 1) - by default where the virtual height is going."""
        hc = h if h is not None else (0 if abs(self.valt_goal - self.valt) < 0.05 else
                                      (1 if self.valt_goal > self.valt else -1))
        level = {1: 'high', 0: 'mid', -1: 'low'}[hc]
        if yaw is not None and abs(yaw) >= self.a.yaw_tol:
            pat, deg = ('cw' if yaw > 0 else 'ccw'), 0
        elif go is not None:
            pat, deg = 'dir', go
        elif hc:
            pat, deg = 'all', 0
        else:
            pat, deg = 'off', 0
        if h is not None:                             # the obstacle walk: up / down, no QR-area heights
            hs = f'height {"UP" if hc > 0 else "DOWN" if hc < 0 else "keep"}'
        else:
            hs = (f'height {"UP" if hc > 0 else "DOWN"} to {self.valt_goal:.1f} m (now {self.valt:.1f})' if hc else
                  f'height keep {self.valt:.1f} m')
        g = {'mode': self.mode, 'state': state, 'say': say, 'yaw': None if yaw is None else round(yaw),
             'go': None if go is None else round(go), 'dist': None if dist is None else round(dist, 1),
             'height': hc, 'height_say': hs, 'valt': round(self.valt, 2), 'valt_goal': round(self.valt_goal, 2),
             'motor': {'pattern': pat, 'deg': round(deg), 'level': level}}
        g.update(extra or {})
        return g

    def toward(self, tx, ty, pose, what):
        """Guidance to a point of the local frame."""
        x, y, hdg = pose
        dx, dy = tx - x, ty - y
        dist = math.hypot(dx, dy)
        rel = wrap(math.degrees(math.atan2(dx, dy)) - hdg)
        if abs(rel) >= self.a.yaw_tol:
            say = f'YAW {"RIGHT" if rel > 0 else "LEFT"} {abs(rel):.0f} deg, then walk {dist:.1f} m to {what}'
            return say, rel, None, dist
        return f'WALK {side_words(rel)} {dist:.1f} m to {what}', rel, rel, dist

    # the QR area walk
    def reset_walk(self):
        self.lane_i = 0
        self.board = None            # the board in hand: {bear, dist, t_seen, sigs, payload, tries, state, t}
        self.done = []               # boards settled: (x, y) local or None, payload, verdict
        self.walk_state = 'SEARCH'
        self.delivered = False
        self.target_text = ''
        self.t_state = time.time()
        self.valt = self.valt_goal = self.search_alt

    def lanes(self):
        """Lawn-mower waypoints over the area (local frame): lanes along its longest edge, lane_spacing apart."""
        a = self.area()
        if a is None:
            return []
        e = max(range(4), key=lambda i: math.dist(a[i], a[(i + 1) % 4]))
        p0, p1 = a[e], a[(e + 1) % 4]
        L = math.dist(p0, p1)
        u = ((p1[0] - p0[0]) / L, (p1[1] - p0[1]) / L)
        v = (-u[1], u[0])
        uv = [((p[0] - p0[0]) * u[0] + (p[1] - p0[1]) * u[1], (p[0] - p0[0]) * v[0] + (p[1] - p0[1]) * v[1]) for p in a]
        vmin, vmax = min(q[1] for q in uv), max(q[1] for q in uv)
        m, sp = 1.0, self.lane_spacing
        pts, k = [], 0
        c = vmin + min(sp / 2, (vmax - vmin) / 2)
        while c <= vmax - min(sp / 2, (vmax - vmin) / 2) + 1e-6:
            xs = []
            for i in range(4):
                (a0, b0), (a1, b1) = uv[i], uv[(i + 1) % 4]
                if (b0 - c) * (b1 - c) <= 0 and b0 != b1:
                    xs.append(a0 + (c - b0) / (b1 - b0) * (a1 - a0))
            if len(xs) >= 2 and max(xs) - min(xs) > 2 * m:
                s0, s1 = min(xs) + m, max(xs) - m
                ends = [(s0, c), (s1, c)] if k % 2 == 0 else [(s1, c), (s0, c)]
                pts += [(p0[0] + q[0] * u[0] + q[1] * v[0], p0[1] + q[0] * u[1] + q[1] * v[1]) for q in ends]
                k += 1
            c += sp
        return pts

    def board_world(self, bear, dist, pose):
        if pose is None:
            return None
        x, y, hdg = pose
        b = math.radians(hdg + bear)
        return x + dist * math.sin(b), y + dist * math.cos(b)

    def settled(self, bear, dist, pose):
        """Was this board in view already settled? (by its place; without GPS by the last few seconds)"""
        w = self.board_world(bear, dist, pose)
        for d in self.done:
            if w is not None and d['xy'] is not None and math.dist(w, d['xy']) < 1.5:
                return True
            if (w is None or d['xy'] is None) and time.time() - d['t'] < 8.0:
                return True
        return False

    def finish_board(self, verdict, payload=None):
        b, pose = self.board, self.pose()
        xy = self.board_world(b['bear'], b['dist'], pose) if b else None
        self.done.append({'xy': xy, 'payload': payload, 'verdict': verdict, 't': time.time()})
        self.board = None
        self.walk_state = 'SEARCH'
        self.valt_goal = self.search_alt              # the deep read's steps down are undone
        self.t_state = time.time()

    def walk_qr_step(self, img):
        pose = self.pose()
        now = time.time()
        ws = self.walk_state
        # the drop: down to 5 m, lower, back up, done
        if ws in ('DESCEND_DROP', 'LOWERING', 'CLIMB_OUT'):
            if ws == 'DESCEND_DROP':
                if abs(self.valt - DROP_ALT) < 0.05:
                    self.walk_state, self.t_state = 'LOWERING', now
                    self.event(f'at {DROP_ALT:g} m over the target: lowering the payload on the winch')
                return self.cue(f'MATCH {self.target_text}: HEIGHT DOWN to {DROP_ALT:g} m and lower the payload - '
                                f'hold position over the board', 'DESCEND_DROP')
            if ws == 'LOWERING':
                if now - self.t_state > self.a.lower_s:
                    self.walk_state, self.valt_goal, self.t_state = 'CLIMB_OUT', self.search_alt, now
                    self.event(f'payload released: height UP to {self.search_alt:g} m, then the exit')
                return self.cue(f'LOWERING the payload at {DROP_ALT:g} m ({self.a.lower_s - (now - self.t_state):.0f} s) '
                                '- hold still', 'LOWERING')
            if abs(self.valt - self.search_alt) < 0.05:
                self.walk_state, self.delivered = 'DONE', True
                self.event('delivered: the mission now leaves the QR area by the exit tunnel')
            return self.cue(f'HEIGHT UP to {self.search_alt:g} m after the drop - hold position', 'CLIMB_OUT')
        if ws == 'DONE':
            return self.cue('DONE: target delivered - the mission heads for the exit tunnel. Stop or start again.',
                            'DONE')
        with self.eyes.lock:
            gray, bs = self.eyes.boards(img)
            red, mask = self.eyes.red(img, self.carry_h)
        self.overlay['boards'] = [{'poly': b['poly'].tolist(), 'label': f'{b["conf"]:.2f}'} for b in bs]
        self.overlay['red'] = mask
        boards = []
        for b in bs:
            bear, off, dist = self.describe_board(b, img.shape)
            boards.append(dict(b, bear=bear, off=off, dist=dist))
        # red under / close: step away first (the mission's no-fly + 1.5 m margin)
        if red['found'] and (red['under'] or red['dist'] < 1.5) and ws in ('SEARCH', 'COMPLETE', 'APPROACH'):
            away = wrap(red['bearing'] + 180)
            return self.cue(f'RED ZONE {"UNDER YOU" if red["under"] else side_words(red["bearing"])}: walk '
                            f'{side_words(away)} out of it (no-fly)', 'AVOID_RED', go=away, dist=1.5)
        if ws in ('SEARCH', 'COMPLETE'):
            fresh = [b for b in boards if b['off'] < 70 and not self.settled(b['bear'], b['dist'], pose)]
            if fresh:
                b = min(fresh, key=lambda b: b['dist'])
                self.board = {'bear': b['bear'], 'dist': b['dist'], 'off': b['off'], 'c': b['c'], 'poly': b['poly'],
                              't_seen': now, 'sigs': [], 'payload': None, 'how': '', 'frames': 0}
                self.walk_state, self.t_state = 'APPROACH', now
                self.event(f'board seen {side_words(b["bear"])} {b["dist"]:.1f} m: going over it')
                ws = 'APPROACH'
            else:
                return self.search_cue(pose, red)
        b = self.board
        if b is None:
            self.walk_state = 'SEARCH'
            return self.search_cue(pose, red)
        # the board in hand: track it (nearest in the image to where it was)
        if boards:
            cur = min(boards, key=lambda q: float(np.linalg.norm(q['c'] - b['c'])))
            if float(np.linalg.norm(cur['c'] - b['c'])) < 0.35 * img.shape[1]:
                b.update(bear=cur['bear'], dist=cur['dist'], off=cur['off'], c=cur['c'], poly=cur['poly'], t_seen=now)
        lost = now - b['t_seen']
        if lost > 4.0 and ws in ('APPROACH', 'READ'):
            self.event('board lost from view: back to the search')
            self.board, self.walk_state = None, 'SEARCH'
            return self.search_cue(pose, red)
        if ws == 'APPROACH':
            if b.get('off', 90) <= self.a.centre_deg:
                self.walk_state, self.t_state = 'READ', now
                b['frames'], b['sigs'] = 0, []
            else:
                return self.cue(f'BOARD {side_words(b["bear"])}: walk {b["dist"]:.1f} m to be right over it',
                                'APPROACH', go=b['bear'], dist=b['dist'])
        if self.walk_state == 'DESCEND_READ':
            if abs(self.valt - self.valt_goal) < 0.05 and now - self.t_state > 1.5:
                self.walk_state, self.t_state = 'READ', now
                b['frames'], b['sigs'] = 0, []
            return self.cue(f'unsure: HEIGHT DOWN {STEP:g} m to {self.valt_goal:.1f} m and read again - hold '
                            'position', 'DESCEND_READ')
        # READ: the decoder settings on read_frames frames, then the image vs the start QR's
        if lost < 0.5 and b.get('poly') is not None and b['payload'] is None:
            b['frames'] += 1
            with self.eyes.lock:
                b['payload'], b['how'] = self.eyes.decode_board(gray, b['poly'])
                s = self.eyes.signature(gray, b['poly'])
            if s is not None:
                b['sigs'].append(s)
        if b['payload'] is None and b['frames'] < self.a.read_frames:
            return self.cue(f'READING the board under you (frame {b["frames"]}/{self.a.read_frames}, '
                            f'{"decoded" if b["payload"] else "not decoded yet"}) - hold still', 'READ')
        verdict, why = self.judge({'payload': b['payload'], 'sigs': b['sigs']})
        p = b['payload']
        read = repr(p) if p is not None else 'not decoded'
        if verdict == 'match':
            self.target_text = read
            self.event(f'QR {read} ({b["how"] or "image"}): {why} - HEIGHT DOWN to {DROP_ALT:g} m and lower the payload')
            self.done.append({'xy': self.board_world(b['bear'], b['dist'], pose), 'payload': p, 'verdict': 'match',
                              't': now})
            self.walk_state, self.valt_goal, self.t_state = 'DESCEND_DROP', DROP_ALT, now
            return self.cue(f'MATCH {read}: HEIGHT DOWN to {DROP_ALT:g} m and lower the payload', 'DESCEND_DROP')
        if verdict in ('different', 'read'):
            self.event(f'QR {read} ({b["how"] or "image"}): {why} - next')
            self.finish_board(verdict, p)
            return self.cue(f'board {read}: {why} - on with the search', 'SEARCH')
        # unsure: 0.5 m lower and again, never below 5.5 m
        if self.valt_goal - STEP >= SCAN_FLOOR - 1e-6:
            self.valt_goal = round(self.valt_goal - STEP, 2)
            self.walk_state, self.t_state = 'DESCEND_READ', now
            self.event(f'board {read}: {why} - height down {STEP:g} m to {self.valt_goal:.1f} m and again')
            return self.cue(f'unsure: HEIGHT DOWN {STEP:g} m to {self.valt_goal:.1f} m and read again', 'DESCEND_READ')
        self.event(f'board {read}: {why} at {self.valt:.1f} m (the lowest for scanning) - giving up on it, next')
        self.finish_board('gave up', p)
        return self.cue(f'gave up on that board at {SCAN_FLOOR:g} m - HEIGHT UP to {self.search_alt:g} m, on with the '
                        'search', 'SEARCH')

    def search_cue(self, pose, red):
        warn = (f' (red zone {side_words(red["bearing"])} {red["dist"]} m: keep clear)' if red['found'] else '')
        if pose is None:
            why = ('no GPS fix from the Pixhawk' if self.area() else 'no QR area (capture 4 corners or use the '
                                                                        'geofence)')
            return self.cue(f'SEARCH, camera only ({why}): walk slow lanes yourself, I call the boards I see{warn}',
                            'SEARCH')
        pts = self.lanes()
        if not pts:
            return self.cue(f'SEARCH: the area is too small for {self.lane_spacing:g} m lanes{warn}', 'SEARCH')
        x, y, _ = pose
        while self.lane_i < len(pts) and math.dist(pts[self.lane_i], (x, y)) < self.a.reach:
            self.lane_i += 1
            if self.lane_i < len(pts):
                self.event(f'waypoint reached: next lane {self.lane_i // 2 + 1}/{len(pts) // 2}')
        if self.lane_i >= len(pts):
            self.walk_state = 'COMPLETE'
            return self.cue('SEARCH COMPLETE: every lane walked, no target matched - the mission would leave by '
                            'the exit tunnel', 'COMPLETE')
        tx, ty = pts[self.lane_i]
        what = f'lane {self.lane_i // 2 + 1}/{len(pts) // 2} {"start" if self.lane_i % 2 == 0 else "end"}'
        say, yaw, go, dist = self.toward(tx, ty, pose, what)
        return self.cue(say + warn, 'SEARCH', yaw=yaw, go=go, dist=dist,
                        extra={'lane': self.lane_i // 2 + 1, 'lanes': len(pts) // 2})

    # the obstacle walk
    def obstacle_guide(self, o):
        g, clear, close = o['grid'], self.a.clear_m, 0.6
        top, mid, bot = g
        free = lambda v: v is None or v >= clear        # noqa: E731  (no return = nothing within 8 m)
        centre = [top[2], mid[2], bot[2]]
        near = [v for v in centre + [mid[1], mid[3]] if v is not None]
        if near and min(near) < close:
            return {'say': f'STOP - obstacle {min(near):.1f} m ahead: back up', 'go': 180}
        if all(free(v) for v in (mid[1], mid[2], mid[3])) and free(top[2]) and free(bot[2]):
            l, r = mid[0], mid[4]
            if l is not None and r is not None and l < 2.5 and r < 2.5 and abs(l - r) > 0.4:
                side = 'right' if l < r else 'left'
                return {'say': f'path clear: walk forward, drift {side} to the middle (walls {l:.1f} / {r:.1f} m)',
                        'go': 20 if side == 'right' else -20}
            return {'say': 'path clear: walk forward', 'go': 0}
        if free(top[2]) and not free(bot[2]):
            return {'say': f'low obstacle {self.fmt(bot[2])} ahead, clear above: HEIGHT UP over it', 'go': None,
                    'h': 1}
        if free(bot[2]) and free(mid[2]) and not free(top[2]):
            return {'say': f'overhead obstacle {self.fmt(top[2])} ahead, clear below: HEIGHT DOWN under it',
                    'go': None, 'h': -1}
        cols = [(i, mid[i]) for i in (0, 1, 3, 4)]
        best = max(cols, key=lambda t: 99 if t[1] is None else t[1])
        if best[1] is not None and best[1] < clear:
            return {'say': f'blocked ahead and to the sides (best {best[1]:.1f} m): back up', 'go': 180}
        deg = o['col_deg'][best[0]]
        return {'say': f'obstacle {self.fmt(mid[2])} ahead: sidestep {"right" if deg > 0 else "left"} '
                       f'({self.fmt(best[1])} free at {deg:+d} deg)', 'go': 90 if deg > 0 else -90}

    def walk_obstacle_step(self):
        d = self.d435.latest_depth()
        if d is None:
            return self.cue(self.d435.status() or 'no D435 depth', 'NO_DEPTH')
        o = self.eyes.obstacles(d, self.d435.k_depth)
        self.overlay['obst'] = o
        r = self.obstacle_guide(o)
        return self.cue(r['say'], 'OBSTACLE', go=r.get('go'), extra={'grid': o['grid']}, h=r.get('h', 0))

    def walk_loop(self):
        t_last = time.time()
        while True:
            now = time.time()
            dt, t_last = now - t_last, now
            # the virtual height moves at CLIMB m/s towards its goal
            d = self.valt_goal - self.valt
            self.valt += max(-CLIMB * dt, min(CLIMB * dt, d))
            try:
                if self.mode == 'walk_qr':
                    img = self.ard.latest()
                    self.guide = (self.walk_qr_step(img) if img is not None else
                                  self.cue(self.ard.err or 'no Arducam frame', 'NO_CAMERA'))
                elif self.mode == 'walk_obstacle':
                    self.guide = self.walk_obstacle_step()
            except Exception as e:                    # keep guiding: one bad frame isn't the end of the test
                self.guide = self.cue(f'error: {e}', 'ERROR')
                print(f'walk error: {e!r}', flush=True)
            time.sleep(max(0.05, self.a.period - (time.time() - now)))

    # ---- state out
    def motor_cue(self):
        g = self.guide
        fresh = time.time() - self.fc_t < 3.0
        if time.time() < self.spin_until:            # spin_test: on by itself, for its seconds only
            return {'pattern': 'cw', 'deg': 0, 'level': 'mid', 'throttle': self.levels['mid'], 'on': fresh}
        m = dict(g['motor']) if self.motors_on and self.mode != 'idle' else {'pattern': 'off', 'deg': 0, 'level': 'mid'}
        m['throttle'] = self.levels[m['level']]
        m['on'] = self.motors_on and fresh
        return m

    def state(self):
        f = self.fc
        with self.lock:
            ev = list(self.events[-40:])
        return {'t': time.time(), 'mode': self.mode, 'guide': self.guide, 'motors_on': self.motors_on,
                'motor': self.motor_cue(), 'levels': self.levels,
                'fc': dict(f, age=round(time.time() - self.fc_t, 1) if self.fc_t else None),
                'cams': {'arducam': self.ard.err or f'ok ({self.eyes.lens_note})', 'd435': self.d435.status() or 'ok'},
                'start_qr': self.start_text(), 'corners': self.corners, 'area': self.area(),
                'lanes': self.lanes(), 'lane_i': self.lane_i, 'pose': self.pose(),
                'done': [{'xy': d['xy'], 'payload': d['payload'], 'verdict': d['verdict']} for d in self.done],
                'settings': {'carry_height': self.carry_h, 'lane_spacing': self.lane_spacing,
                             'search_alt': self.search_alt, 'valt': round(self.valt, 2), 'floor': SCAN_FLOOR,
                             'drop': DROP_ALT},
                'last': self.last, 'events': ev}

    def radio(self, since):
        """mav_agent's poll: the compact status (TUNNEL, <= 128 B) + the events after `since`."""
        g = self.guide
        with self.lock:
            ev = [e for e in self.events if e['n'] > since]
        st = {'m': {'idle': 'i', 'walk_qr': 'q', 'walk_obstacle': 'o'}[self.mode], 's': g.get('state', '-')[:12],
              'y': g.get('yaw'), 'g': g.get('go'), 'd': g.get('dist'), 'h': g.get('height', 0),
              'va': round(self.valt, 1), 'mc': fp.PATTERN_CODE[g['motor']['pattern']] + g['motor']['level'][0],
              'mo': int(self.motors_on), 'n': len(self.done)}
        return {'status': st, 'say': g.get('say', ''), 'events': ev, 'n': self.n_ev}

    # ---- views
    def jpeg(self, which):
        if which == 'arducam':
            img = self.ard.latest()
            if img is None:
                return None
            sc = 820.0 / img.shape[1]
            out = cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            red = self.overlay.get('red')
            if red is not None and red.shape[:2] != out.shape[:2]:
                red = cv2.resize(red.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
            if red is not None:
                out[red] = (0.5 * out[red] + [127, 0, 127]).astype(np.uint8)
            cv2.circle(out, (out.shape[1] // 2, out.shape[0] // 2), 6, (0, 255, 255), 2)
            for b in self.overlay.get('boards', []):
                p = (np.asarray(b['poly']) * sc).astype(np.int32)
                cv2.polylines(out, [p], True, (0, 255, 0), 2)
                cv2.putText(out, str(b['label']), tuple(int(v) for v in p[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 1)
            cv2.putText(out, 'NOSE ^', (out.shape[1] // 2 - 30, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        elif which == 'color':
            img = self.d435.latest_color()
            if img is None:
                return None
            out = img.copy()
            for b in self.overlay.get('banners', []):
                c = (0, 255, 0) if b['ok'] else (0, 165, 255)
                cv2.polylines(out, [np.asarray(b['hull'], np.int32).reshape(-1, 1, 2)], True, c, 2)
        elif which == 'depth':
            d = self.d435.latest_depth()
            if d is None:
                return None
            v = np.clip(d.astype(np.float32) / (self.a.clear_m * 2 * 1000.0), 0, 1)
            out = cv2.applyColorMap((255 - v * 255).astype(np.uint8), cv2.COLORMAP_JET)
            out[d == 0] = 0
            o = self.overlay.get('obst')
            if o:
                h, w = d.shape
                for r in range(3):
                    for c in range(5):
                        val = o['grid'][r][c]
                        x0, y0 = c * w // 5, r * h // 3
                        cv2.rectangle(out, (x0, y0), (x0 + w // 5, y0 + h // 3), (255, 255, 255), 1)
                        cv2.putText(out, '-' if val is None else f'{val:.1f}', (x0 + 6, y0 + 22),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            return None
        return cv2.imencode('.jpg', out, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()


def make_handler(app):
    page = HERE / 'field_test.html'

    class H(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *_):
            pass

        def send(self, body, ctype='application/json', code=200):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(body)

        def js(self, obj, code=200):
            self.send(json.dumps(obj, default=lambda o: o.tolist() if hasattr(o, 'tolist') else str(o)).encode(),
                      code=code)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            if u.path in ('/', '/index.html'):
                return self.send(page.read_bytes(), 'text/html; charset=utf-8')
            if u.path == '/api/state':
                return self.js(app.state())
            if u.path == '/api/radio':
                return self.js(app.radio(int(q.get('since', 0))))
            if u.path == '/api/info':
                return self.js({'side': 'jetson', 'link': 'cable', 'codes': fp.CODES})
            if u.path.startswith('/cam/') and u.path.endswith('.jpg'):
                j = app.jpeg(u.path[5:-4])
                return self.send(j, 'image/jpeg') if j else self.send(b'no image', 'text/plain', 503)
            self.send(b'not found', 'text/plain', 404)

        def do_POST(self):
            u = urlparse(self.path)
            n = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(n) or b'{}')
            except ValueError:
                return self.js({'ok': False, 'message': 'bad JSON'}, 400)
            if u.path == '/api/fc':                   # field_link: the Pixhawk's state in, the motor cue out
                app.fc, app.fc_t = body, time.time()
                return self.js(dict(app.motor_cue(), req=app.fc_req))   # + a mode / winch test to do
            if u.path == '/api/cmd':
                name = body.get('name') or fp.CODES.get(int(body.get('code', 0)), ('?',))[0]
                try:
                    ok, msg = app.command(name, float(body.get('arg') or 0.0), body.get('text'))
                except Exception as e:
                    ok, msg = False, f'{name}: {e}'
                return self.js({'ok': ok, 'message': msg})
            if u.path == '/api/levels':               # motor test throttle % per level
                for k in ('low', 'mid', 'high'):
                    if k in body:
                        app.levels[k] = max(0.0, min(30.0, float(body[k])))
                return self.js({'ok': True, 'message': f'spin levels {app.levels}'})
            self.send(b'not found', 'text/plain', 404)

    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=fp.PORT_VISION)
    ap.add_argument('--arducam', default='csi', help='csi | http://.../frame.jpg | /dev/video index | file')
    ap.add_argument('--sensor-id', type=int, default=0)
    ap.add_argument('--width', type=int, default=1640)
    ap.add_argument('--height', type=int, default=1232)
    ap.add_argument('--fps', type=int, default=30)
    ap.add_argument('--exposure-ms', type=float, default=8.0, help='Arducam exposure cap: short = sharp while walking')
    ap.add_argument('--calib', default=str(WS / 'calibration/arducam_fisheye_1640x1232.yaml'))
    ap.add_argument('--d435-flip', type=int, default=1, help='1: the D435 is mounted upside down (rig.yaml)')
    ap.add_argument('--carry-height', type=float, default=1.2, help='m: how high the drone is carried')
    ap.add_argument('--lane-spacing', type=float, default=3.0, help='m between search lanes (walking: the camera '
                    'sees ~2 m round from 1.2 m; the mission flies 5.5 m lanes from 10 m)')
    ap.add_argument('--reach', type=float, default=1.0, help='m: a lane waypoint this close is reached')
    ap.add_argument('--yaw-tol', type=float, default=35.0, help='deg: turn first when the way is further off the nose')
    ap.add_argument('--centre-deg', type=float, default=12.0, help='deg off straight down: over the board')
    ap.add_argument('--read-frames', type=int, default=5, help='frames of the deep read (the mission: 5)')
    ap.add_argument('--match-hi', type=float, default=0.80, help='image match >= this: the start QR (qr_locator)')
    ap.add_argument('--match-lo', type=float, default=0.50, help='image match <= this: another board')
    ap.add_argument('--marker', default=os.environ.get('AEROTHON_MARKER', 'aruco'), choices=('aruco', 'qr'),
                    help='the boards: ArUco markers matched by image (the course since 2026-10-30), or QR codes')
    ap.add_argument('--aruco-hi', type=float, default=0.80, help='ArUco image match >= this: the start marker')
    ap.add_argument('--aruco-lo', type=float, default=0.65, help='ArUco image match <= this: another marker')
    ap.add_argument('--lower-s', type=float, default=8.0, help='s the winch takes (the walk-through only)')
    ap.add_argument('--clear-m', type=float, default=1.5, help='m: an obstacle further than this is no obstacle')
    ap.add_argument('--period', type=float, default=0.3, help='s per guidance step')
    ap.add_argument('--spin-low', type=float, default=7.0, help='motor test %% for "height down" (slow)')
    ap.add_argument('--spin-mid', type=float, default=11.0, help='motor test %% for "keep the height"')
    ap.add_argument('--spin-high', type=float, default=17.0, help='motor test %% for "height up" (fast)')
    a = ap.parse_args()
    os.environ.pop('DISPLAY', None)                   # Argus fails with a DISPLAY it can't use
    app = App(a)
    # after rclpy.init: its handlers only shut ROS down, the page kept running (field_test.sh stop did nothing)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: os._exit(0))  # Argus is freed by the exit; don't wait on a blocked camera read
    srv = ThreadingHTTPServer(('0.0.0.0', a.port), make_handler(app))
    srv.daemon_threads = True
    print(f'field test (vision): http://0.0.0.0:{a.port}', flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    os._exit(0)                                       # don't wait on a blocked camera read; Argus is freed by the exit


if __name__ == '__main__':
    main()
