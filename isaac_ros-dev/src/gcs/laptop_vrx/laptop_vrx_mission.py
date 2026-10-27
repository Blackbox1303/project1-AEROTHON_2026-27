#!/usr/bin/env python3
"""Laptop VRX + Ground Telemetry Radio — Standalone Delivery-Only Mission & QR Detection System.

Runs directly on the ground laptop with:
  1. Video input from the USB VRX (Video Receiver, e.g. /dev/video2 or /dev/video0) attached to the laptop.
  2. Telemetry & offboard control over the Ground Telemetry Radio (e.g. /dev/ttyUSB0:57600) attached to the laptop,
     with automatic bidirectional UDP relay to QGroundControl (127.0.0.1:14550).
  3. Complete QR Code Detection & Localization on the laptop:
     - Multi-scale WeChat QrNet Caffe SSD bounding-box detector (160, 224, 384 px tiles) + local contrast & Otsu
       transition candidate finder (`find_qr_candidates`) + dark-square contour refinement (`refine_qr_box`).
     - Virtual camera adjustment: crops, perspective/gnomonic-rectifies, and zooms each detected QR bounding box,
       stacks consecutive VRX frames with phase-correlation alignment (`aligned_mean`) to cancel analog VRX snow/noise,
       and decodes with quiet-zone border-padded WeChat CNN super-resolution + CLAHE + pyzbar.
     - Projects pixel rays through the drone's synchronized MAVLink attitude (roll, pitch, yaw) and local ENU pose
       onto the ground plane, and averages QR positions and pairwise inter-QR distances over a short sliding time
       window (`avg_window_s = 4.0 s`, up to 15 samples) with cos^4(theta) weighting and median outlier rejection.
  4. Autonomous Delivery-Only Mission FSM (`TAKEOFF` -> `READ_START_QR` -> `DIRECT_TO_ZONE` -> `SEARCH_DELIVERY` ->
     `DELIVER` -> `DROP` winch/gripper -> `RETURN_HOME` -> `LAND` -> `DONE`):
     - Takes off to 5.0 m, moves 1.0 m forward to scan the Start QR (or uses the Start QR payload entered on the UI).
     - Climbs to 10.0 m (configurable) and flies a lawnmower search pattern + visual candidate investigation across
       the delivery zone.
     - ONLY when a confirmed QR inside the delivery zone has `payload == start_qr` does it transition to `DELIVER`,
       descend to 5.0 m over the target, fine-center on the live VRX feed, and trigger the winch/drop mechanism
       (`MAV_CMD_DO_GRIPPER = 211` + optional `MAV_CMD_DO_SET_SERVO = 183`).

Usage on the laptop:
  python3 laptop_vrx_mission.py --vrx /dev/video2 --radio /dev/ttyUSB0:57600 --port 8082 --hud-port 8091
"""
import argparse
import asyncio
import glob
import json
import math
import os
import socket
import threading
import time
from collections import Counter
from pathlib import Path

from aiohttp import web
import cv2
import numpy as np

os.environ['MAVLINK20'] = '1'
from pymavlink import mavutil  # noqa: E402

HERE = Path(__file__).resolve().parent
EARTH_R = 6378137.0
PX4_MAIN = {1: 'MANUAL', 2: 'ALTCTL', 3: 'POSCTL', 4: 'AUTO', 5: 'ACRO', 6: 'OFFBOARD', 7: 'STABILIZED'}
PX4_AUTO = {1: 'READY', 2: 'TAKEOFF', 3: 'LOITER', 4: 'MISSION', 5: 'RTL', 6: 'LAND', 8: 'FOLLOW', 9: 'PRECLAND'}
# ArduCopter custom_mode numbers (the heartbeat's custom_mode, and DO_SET_MODE param2)
COPTER_MODES = {0: 'STABILIZE', 1: 'ACRO', 2: 'ALT_HOLD', 3: 'AUTO', 4: 'GUIDED', 5: 'LOITER', 6: 'RTL', 7: 'CIRCLE',
                9: 'LAND', 11: 'DRIFT', 13: 'SPORT', 14: 'FLIP', 15: 'AUTOTUNE', 16: 'POSHOLD', 17: 'BRAKE',
                18: 'THROW', 19: 'AVOID_ADSB', 20: 'GUIDED_NOGPS', 21: 'SMART_RTL'}
# the generic (PX4-style) names the missions use -> ArduCopter modes. Hold = BRAKE (stops and holds; LOITER would also
# take RC stick input). RTL climbs to RTL_ALT (15 m default) first: not for the 1 m flights.
COPTER_FOR = {'OFFBOARD': 4, 'AUTO.TAKEOFF': 4, 'AUTO.LOITER': 17, 'AUTO.RTL': 6, 'AUTO.LAND': 9}
FSM_DELIVERY = ['TAKEOFF', 'READ_START_QR', 'DIRECT_TO_ZONE', 'SEARCH_DELIVERY', 'DELIVER', 'DROP', 'RETURN_HOME', 'LAND', 'DONE']
MODEL_FILES = ('detect.prototxt', 'detect.caffemodel', 'sr.prototxt', 'sr.caffemodel')


def gps_to_local(lat, lon, lat0, lon0):
    return (
        math.radians(lon - lon0) * EARTH_R * math.cos(math.radians(lat0)),
        math.radians(lat - lat0) * EARTH_R,
    )


def rpy_to_rot_enu(roll_ned, pitch_ned, yaw_ned):
    """Convert PX4 NED Euler angles (roll, pitch, yaw in rad) to ENU body->world rotation matrix (FLU -> ENU)."""
    # First build body(FRD) -> NED rotation matrix
    cr, sr = math.cos(roll_ned), math.sin(roll_ned)
    cp, sp = math.cos(pitch_ned), math.sin(pitch_ned)
    cy, sy = math.cos(yaw_ned), math.sin(yaw_ned)
    R_frd_to_ned = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)
    # Transform NED -> ENU: x_enu = E = y_ned, y_enu = N = x_ned, z_enu = U = -z_ned
    T_ned_to_enu = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], dtype=np.float64)
    # Transform FLU -> FRD: x_frd = F = x_flu, y_frd = R = -y_flu, z_frd = D = -z_flu
    T_flu_to_frd = np.diag([1.0, -1.0, -1.0])
    return T_ned_to_enu @ R_frd_to_ned @ T_flu_to_frd


# ---------------------------------------------------------------- QR Detection & Camera-Adjusted Scan Engine
def find_model_dir(custom=''):
    candidates = [
        custom,
        str(HERE / 'wechat_qrcode'),
        str(Path.home() / '.local/share/vrx_delivery/wechat_qrcode'),
        '/workspaces/isaac_ros-dev/calibration/wechat_qrcode',
        str(Path.home() / 'workspaces/isaac_ros-dev/calibration/wechat_qrcode'),
    ]
    for d in candidates:
        if d and all((Path(d) / f).exists() for f in MODEL_FILES):
            return Path(d)
    return None


def make_wechat(custom=''):
    d = find_model_dir(custom)
    if d is None or not hasattr(cv2, 'wechat_qrcode_WeChatQRCode'):
        return None
    try:
        return cv2.wechat_qrcode_WeChatQRCode(*(str(d / f) for f in MODEL_FILES))
    except cv2.error:
        return None


class QrNet:
    def __init__(self, custom=''):
        d = find_model_dir(custom)
        self.net = cv2.dnn.readNetFromCaffe(str(d / 'detect.prototxt'), str(d / 'detect.caffemodel')) if d else None

    def boxes(self, crop, x0=0, y0=0, conf=0.55):
        if self.net is None or crop is None or crop.size == 0:
            return []
        self.net.setInput(cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (384, 384), 0, False, False))
        h, w = crop.shape[:2]
        out = []
        for d in self.net.forward().reshape(-1, 7):
            if d[2] >= conf:
                x1, y1, x2, y2 = d[3] * w + x0, d[4] * h + y0, d[5] * w + x0, d[6] * h + y0
                if x2 > x1 + 6 and y2 > y1 + 6:
                    out.append((float(d[2]), np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)))
        return out

    def scan(self, gray, conf=0.55, centres=None, tile=384, overlap=96):
        h, w = gray.shape[:2]
        found = []
        if centres is None:
            t_sz = min(tile, w, h)
            step = max(64, t_sz - overlap)
            spots = [(max(0, min(x, w - t_sz)), max(0, min(y, h - t_sz)), t_sz)
                     for y in range(0, max(1, h - overlap), step)
                     for x in range(0, max(1, w - overlap), step)]
        else:
            spots = []
            for t_sz in (160, 224, min(tile, w, h)):
                t_sz = min(t_sz, w, h)
                for cx, cy in centres:
                    spots.append((
                        int(max(0, min(cx - t_sz / 2, w - t_sz))),
                        int(max(0, min(cy - t_sz / 2, h - t_sz))),
                        t_sz,
                    ))
        for x0, y0, t_sz in dict.fromkeys(spots):
            found += self.boxes(gray[y0:y0 + t_sz, x0:x0 + t_sz], x0, y0, conf)
        keep = []
        for c, p in sorted(found, key=lambda t: -t[0]):
            p_ref = refine_qr_box(gray, p)
            if all(poly_iou(p_ref, q) < 0.3 for _, q in keep):
                keep.append((c, p_ref))
        return keep


def poly_iou(a, b):
    ax0, ay0 = a.min(axis=0)
    ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0)
    bx1, by1 = b.max(axis=0)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    return inter / max(1e-6, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def refine_qr_box(gray, poly):
    h, w = gray.shape[:2]
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    bw, bh = x1 - x0, y1 - y0
    if bw < 16 or bh < 16 or bw > 280 or bh > 280:
        return poly
    X0, Y0 = max(0, int(x0)), max(0, int(y0))
    X1, Y1 = min(w, int(x1)), min(h, int(y1))
    roi = gray[Y0:Y1, X0:X1]
    if roi.size < 64:
        return poly
    med = float(np.median(roi))
    _, dark = cv2.threshold(roi, int(max(30, min(190, med * 0.88))), 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ys, xs = np.where(dark > 0)
    if len(xs) < 20:
        return poly
    rx0, rx1 = float(np.percentile(xs, 3)) + X0, float(np.percentile(xs, 97)) + X0
    ry0, ry1 = float(np.percentile(ys, 3)) + Y0, float(np.percentile(ys, 97)) + Y0
    rw, rh = rx1 - rx0, ry1 - ry0
    if 18 <= max(rw, rh) <= 220 and 0.60 <= rw / max(rh, 1.0) <= 1.65:
        return np.array([[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]], dtype=np.float64)
    return poly


def transitions(crop):
    if crop.size < 16:
        return 0.0, 0.0
    c = crop if crop.dtype == np.uint8 else np.clip(crop, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(c, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bw.astype(np.int8)
    return (
        float(np.abs(np.diff(bw, axis=1)).sum(axis=1).mean()),
        float(np.abs(np.diff(bw, axis=0)).sum(axis=0).mean()),
    )


def find_qr_candidates(gray, max_cands=8):
    h, w = gray.shape[:2]
    scale = 0.5 if w > 960 else 1.0
    g_s = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else gray
    g_f = cv2.GaussianBlur(g_s.astype(np.float32), (0, 0), 1.2)
    mu = cv2.boxFilter(g_f, -1, (5, 5))
    sd = np.sqrt(np.maximum(cv2.boxFilter(g_f * g_f, -1, (5, 5)) - mu * mu, 0.0))
    # Border margin mask so analog VRX OSD / frame edges don't trigger false candidates
    border = np.zeros_like(sd, dtype=np.uint8)
    mx, my = int(0.06 * g_s.shape[1]), int(0.06 * g_s.shape[0])
    border[my:g_s.shape[0] - my, mx:g_s.shape[1] - mx] = 1
    sd = sd * border
    valid = sd[sd > 0.5]
    med = float(np.median(valid)) if valid.size else 6.0
    thr = max(6.0, med + 4.0 * 1.4826 * (float(np.median(np.abs(valid - med))) + 1e-3 if valid.size else 1.0))
    blobs = []
    for level in (1.0, 1.6, 2.5):
        mask = (sd > thr * level).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area < 8 or area > 12000:
                continue
            if max(bw, bh) / max(min(bw, bh), 1) > 2.4:
                continue
            if min(bw, bh) >= 8 and min(transitions(g_s[by:by + bh, bx:bx + bw])) < 3.0:
                continue
            score = float(sd[by:by + bh, bx:bx + bw].mean())
            if any(math.dist(cent[i], c) < 0.5 * max(bw, bh) for _, c, _ in blobs):
                continue
            blobs.append((stats[i], cent[i], score))
    blobs.sort(key=lambda t: -t[2])
    inv = 1.0 / scale
    out = []
    for (bx, by, bw, bh, _), _, score in blobs[:max_cands]:
        poly = np.array([
            [bx * inv, by * inv],
            [(bx + bw) * inv, by * inv],
            [(bx + bw) * inv, (by + bh) * inv],
            [bx * inv, (by + bh) * inv],
        ], dtype=np.float64)
        poly = refine_qr_box(gray, poly)
        out.append({'center': tuple(poly.mean(axis=0)), 'poly': poly, 'score': score})
    return out


def aligned_mean(crops):
    ref = crops[0].astype(np.float32)
    acc, n = ref.copy(), 1
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    for c in crops[1:]:
        f = c.astype(np.float32)
        if f.shape != ref.shape:
            continue
        (dx, dy), resp = cv2.phaseCorrelate(ref, f, win)
        if resp < 0.04 or abs(dx) > ref.shape[1] / 4 or abs(dy) > ref.shape[0] / 4:
            continue
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        acc += cv2.warpAffine(f, m, ref.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
        n += 1
    return np.clip(acc / n, 0, 255).astype(np.uint8), n


def wechat_decode(wc, gray, x0=0, y0=0):
    if wc is None or gray is None or gray.size == 0:
        return {}
    out = {}
    h, w = gray.shape[:2]
    pad = 32 if max(h, w) <= 360 else 0
    work = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else gray
    try:
        texts, pts = wc.detectAndDecode(work)
        for t, p in zip(texts, pts):
            t = t.strip()
            if t and t not in out:
                out[t] = np.asarray(p, dtype=np.float64).reshape(-1, 2) - [pad - x0, pad - y0]
    except cv2.error:
        pass
    return out


# ---------------------------------------------------------------- Main VRX + Radio Delivery Controller
class VrxDeliverySystem:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()

        # Telemetry & Pose state
        self.tel = {}
        self.tel_time = {}
        self.origin = None       # (lat0, lon0, alt0)
        self.pose = None         # (x_enu, y_enu, z_enu, yaw_enu)
        self.rpy_ned = (0.0, 0.0, 0.0)
        self.track = []
        self.statustext = []
        self.acks = {}
        self.radio_err = 'connecting...'
        self.mav = None
        self.autopilot = None    # 'px4' | 'ardupilot', from the autopilot's heartbeat

        # QGC UDP relay socket
        self.qgc_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.qgc_sock.bind(('127.0.0.1', 0))
        self.qgc_sock.setblocking(False)
        q_host, q_port = args.qgc.split(':')
        self.qgc_addr = (q_host, int(q_port))

        # Offboard setpoint state (sent at 15 Hz while offboard_active is True)
        self.offboard_active = False
        self.setpoint = (0.0, 0.0, 2.5, 0.0)  # (x_enu, y_enu, z_enu, yaw_enu)

        # Geofence & Delivery Zone in ENU metres (default 20m x 20m box in front of Home if not set from GPS)
        self.gf_file = Path.home() / '.local/share/vrx_delivery/geofence.json'
        self.gf_file.parent.mkdir(parents=True, exist_ok=True)
        self.zone_enu = (-10.0, 25.0, -12.0, 12.0)  # x_min, x_max, y_min, y_max in ENU odom

        # QR Decoders
        self.wc = make_wechat(args.wechat_dir)
        self.qnet = QrNet(args.wechat_dir)
        try:
            from pyzbar.pyzbar import ZBarSymbol, decode
            self.zbar = lambda img: decode(img, symbols=[ZBarSymbol.QRCODE])
        except ImportError:
            self.zbar = None
        print(f'[VRX-QR] Decoders ready: WeChat={"YES" if self.wc else "NO"}, '
              f'QrNet={"YES" if self.qnet and self.qnet.net else "NO"}, zbar={"YES" if self.zbar else "NO"}')

        # Camera & QR Detection state
        self.vrx_device = args.vrx
        self.fov_deg = float(args.fov_deg)
        self.raw_jpeg = None
        self.annotated_jpeg = None
        self.latest_gray = None
        self.latest_frame_pose = None
        self.chase_stacks = {}
        self.last_zoom_patches = []
        self.last_boxes_px = []
        self.qrs = []            # tracked QRs with short-window averaging
        self.cands = []          # unread boxed QR candidates on the ground
        self.pair_dist_hist = {}
        self.start_qr = args.start_qr or None
        self.target = None       # (x, y) of matched target QR
        self.target_fine = None  # (x, y, n) fine fix from 5m drop height
        self.searching = True    # Always run QR detection on VRX so operator sees live detections immediately

        # Mission Supervisor state
        self.state = 'IDLE'
        self.mission_running = False
        self.abort_reason = None
        self.mission_t0 = None
        self.zone_t0 = None
        self.log_entries = []
        self.takeoff_alt = 5.0
        self.search_alt = 10.0
        self.deliver_alt = 5.0

    def log_mission(self, state, **info):
        with self.lock:
            self.state = state
            entry = {'time': time.time(), 'state': state, **info}
            self.log_entries = (self.log_entries + [entry])[-200:]
        print(f'[MISSION] {state} {json.dumps(info)}', flush=True)

    # ---- MAVLink Ground Telemetry Radio Connection & Stream Requests
    def open_radio(self):
        dev = self.args.radio
        try:
            if dev.startswith(('udpin:', 'udpout:', 'tcp:')):
                self.mav = mavutil.mavlink_connection(dev, source_system=255, source_component=190)
            else:
                path, _, baud = dev.partition(':')
                if not Path(path).exists():
                    raise OSError(f'{path} not found (plug in Ground Telemetry Unit)')
                self.mav = mavutil.mavlink_connection(path, baud=int(baud or 57600), source_system=255,
                                                      source_component=190)
            self.radio_err = None
            print(f'[RADIO] Connected to {dev}', flush=True)
        except Exception as e:
            self.mav = None
            self.radio_err = str(e)

    def request_streams(self):
        """Request LOCAL_POSITION_NED (32), ATTITUDE (30), GLOBAL_POSITION_INT (33), GPS_RAW_INT (24) over SiK radio."""
        if self.mav is None or not self.mav.target_system:
            return
        rates = [(30, 15), (32, 10), (33, 5), (24, 2), (1, 2)]  # (msg_id, rate_hz)
        for msg_id, hz in rates:
            try:
                self.mav.mav.command_long_send(
                    self.mav.target_system or 1, 1,
                    mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                    msg_id, int(1e6 / hz), 0, 0, 0, 0, 0,
                )
            except Exception:
                pass
        if self.autopilot == 'ardupilot':
            # EXTENDED_STATUS (SYS_STATUS, GPS_RAW_INT), POSITION (LOCAL/GLOBAL_POSITION), EXTRA1 (ATTITUDE)
            for stream, hz in ((2, 2), (6, 10), (10, 15)):
                try:
                    self.mav.mav.request_data_stream_send(self.mav.target_system or 1, 1, stream, hz, 1)
                except Exception:
                    pass
        for msg_id in (49, 242):  # GPS_GLOBAL_ORIGIN, HOME_POSITION
            try:
                self.mav.mav.command_long_send(
                    self.mav.target_system or 1, 1,
                    mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE, 0,
                    msg_id, 0, 0, 0, 0, 0, 0,
                )
            except Exception:
                pass

    def mav_rx_loop(self):
        last_req = 0.0
        while True:
            if self.mav is None:
                self.open_radio()
                time.sleep(2.0)
                continue
            try:
                msg = self.mav.recv_match(blocking=True, timeout=0.5)
            except Exception as e:
                self.radio_err = str(e)
                self.mav = None
                continue
            now = time.monotonic()
            if now - last_req > 6.0 and self.mav.target_system:
                last_req = now
                self.request_streams()
            if msg is None:
                continue
            try:
                self.qgc_sock.sendto(msg.get_msgbuf(), self.qgc_addr)
            except OSError:
                pass
            t = msg.get_type()
            if t == 'BAD_DATA':
                continue
            with self.lock:
                self.tel_time[t] = now
                if t == 'HEARTBEAT' and msg.get_srcComponent() == 1 and msg.autopilot in (
                        mavutil.mavlink.MAV_AUTOPILOT_PX4, mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA):
                    self.tel_time['HEARTBEAT'] = now
                    base, cm = msg.base_mode, msg.custom_mode
                    if msg.autopilot == mavutil.mavlink.MAV_AUTOPILOT_ARDUPILOTMEGA:
                        self.autopilot = 'ardupilot'
                        mode = COPTER_MODES.get(cm, str(cm))
                    else:
                        self.autopilot = 'px4'
                        main_m, sub_m = (cm >> 16) & 0xFF, (cm >> 24) & 0xFF
                        mode = PX4_MAIN.get(main_m, str(main_m))
                        if main_m == 4:
                            mode = 'AUTO.' + PX4_AUTO.get(sub_m, str(sub_m))
                    self.tel.update(
                        armed=bool(base & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED),
                        mode=mode,
                    )
                elif t == 'SYS_STATUS':
                    self.tel.update(
                        batt_v=round(msg.voltage_battery / 1000.0, 2) if msg.voltage_battery != 65535 else None,
                        batt_pct=msg.battery_remaining if msg.battery_remaining >= 0 else None,
                    )
                elif t == 'GPS_RAW_INT':
                    self.tel.update(
                        gps_fix=msg.fix_type,
                        sats=msg.satellites_visible,
                        hdop=round(msg.eph / 100.0, 2) if msg.eph != 65535 else None,
                    )
                elif t == 'GLOBAL_POSITION_INT':
                    lat, lon = msg.lat / 1e7, msg.lon / 1e7
                    alt_rel = msg.relative_alt / 1000.0
                    self.tel.update(
                        lat=lat, lon=lon, alt_rel=round(alt_rel, 2),
                        groundspeed=round(math.hypot(msg.vx, msg.vy) / 100.0, 2),
                        climb=round(-msg.vz / 100.0, 2),
                        heading=round(msg.hdg / 100.0) if msg.hdg != 65535 else None,
                    )
                    if self.origin is None and (self.tel.get('gps_fix') or 0) >= 3 and abs(lat) > 1e-4:
                        self.origin = (lat, lon, msg.alt / 1000.0)
                    # Fallback local ENU pose from GPS if LOCAL_POSITION_NED hasn't arrived yet
                    if self.origin is not None and now - self.tel_time.get('LOCAL_POSITION_NED', 0.0) > 2.0:
                        xe, yn = gps_to_local(lat, lon, self.origin[0], self.origin[1])
                        yaw_e = math.radians(90.0 - (self.tel.get('heading') or 0.0))
                        self.pose = (xe, yn, alt_rel, yaw_e)
                elif t in ('GPS_GLOBAL_ORIGIN', 'HOME_POSITION') and self.origin is None:
                    if abs(msg.latitude) > 0:
                        self.origin = (msg.latitude / 1e7, msg.longitude / 1e7, msg.altitude / 1000.0)
                elif t == 'ATTITUDE':
                    self.rpy_ned = (float(msg.roll), float(msg.pitch), float(msg.yaw))
                    self.tel.update(
                        roll=round(math.degrees(msg.roll), 1),
                        pitch=round(math.degrees(msg.pitch), 1),
                        yaw_ned=msg.yaw,
                    )
                elif t == 'LOCAL_POSITION_NED':
                    yaw_ned = self.tel.get('yaw_ned', 0.0)
                    yaw_enu = math.atan2(math.cos(yaw_ned), math.sin(yaw_ned))
                    # NED -> ENU: x_enu = y_ned (East), y_enu = x_ned (North), z_enu = -z_ned (Up)
                    self.pose = (float(msg.y), float(msg.x), float(-msg.z), yaw_enu)
                    self.tel.update(alt_rel=round(float(-msg.z), 2))
                    if not self.track or math.dist(self.track[-1], (msg.y, msg.x)) > 0.3:
                        self.track = (self.track + [(round(msg.y, 2), round(msg.x, 2))])[-2000:]
                elif t == 'STATUSTEXT':
                    self.statustext = (self.statustext + [(time.time(), msg.severity, msg.text.rstrip('\0'))])[-40:]
                elif t == 'COMMAND_ACK':
                    self.acks[msg.command] = (msg.result, now)

    def qgc_tx_loop(self):
        while True:
            try:
                data = self.qgc_sock.recv(4096)
                if self.mav is not None:
                    self.mav.write(data)
            except BlockingIOError:
                time.sleep(0.01)
            except OSError:
                time.sleep(0.2)

    def offboard_tx_loop(self):
        """Send heartbeat (1 Hz) and OFFBOARD position setpoints (15 Hz) in MAV_FRAME_LOCAL_NED."""
        last_hb = 0.0
        while True:
            now = time.monotonic()
            if self.mav is not None:
                if now - last_hb >= 1.0:
                    last_hb = now
                    try:
                        self.mav.mav.heartbeat_send(
                            mavutil.mavlink.MAV_TYPE_GCS if self.autopilot == 'ardupilot' else
                            mavutil.mavlink.MAV_TYPE_ONBOARD_CONTROLLER,
                            mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0,
                            mavutil.mavlink.MAV_STATE_ACTIVE,
                        )
                    except Exception:
                        pass
                if self.offboard_active:
                    with self.lock:
                        xe, yn, zu, yaw_enu = self.setpoint
                    # Convert ENU setpoint -> NED for PX4 SET_POSITION_TARGET_LOCAL_NED
                    x_ned = float(yn)
                    y_ned = float(xe)
                    z_ned = float(-zu)
                    yaw_ned = float(math.atan2(math.cos(yaw_enu), math.sin(yaw_enu)))
                    type_mask = 0b0000101111111000  # use x, y, z, yaw only
                    try:
                        self.mav.mav.set_position_target_local_ned_send(
                            int((now * 1000) % 0xFFFFFFFF),
                            self.mav.target_system or 1, 1,
                            mavutil.mavlink.MAV_FRAME_LOCAL_NED,
                            type_mask,
                            x_ned, y_ned, z_ned,
                            0.0, 0.0, 0.0,
                            0.0, 0.0, 0.0,
                            yaw_ned, 0.0,
                        )
                    except Exception:
                        pass
            time.sleep(1.0 / 15.0)

    def mav_cmd(self, cmd, p1=0.0, p2=0.0, p3=0.0, p4=0.0, p5=0.0, p6=0.0, p7=0.0, timeout=3.0):
        if self.mav is None:
            return False
        t0 = time.monotonic()
        try:
            self.mav.mav.command_long_send(
                self.mav.target_system or 1, 1, cmd, 0,
                float(p1), float(p2), float(p3), float(p4), float(p5), float(p6), float(p7),
            )
        except Exception:
            return False
        end = t0 + timeout
        while time.monotonic() < end:
            with self.lock:
                ack = self.acks.get(cmd)
            if ack and ack[1] >= t0:
                return ack[0] in (0, mavutil.mavlink.MAV_RESULT_ACCEPTED, mavutil.mavlink.MAV_RESULT_IN_PROGRESS)
            time.sleep(0.05)
        return False

    @property
    def offboard_name(self):
        """The mode external position setpoints fly in: PX4 OFFBOARD, ArduCopter GUIDED."""
        return 'GUIDED' if self.autopilot == 'ardupilot' else 'OFFBOARD'

    def set_px4_mode(self, mode_name):
        """Set the mode by its PX4 name (OFFBOARD, AUTO.TAKEOFF, AUTO.LAND, AUTO.RTL, AUTO.LOITER); on ArduCopter the
        matching mode (COPTER_FOR: GUIDED, LAND, RTL, BRAKE)."""
        if self.mav is None:
            return False
        if self.autopilot == 'ardupilot':
            return self.mav_cmd(mavutil.mavlink.MAV_CMD_DO_SET_MODE,
                                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, COPTER_FOR.get(mode_name, 17))
        custom_map = {
            'OFFBOARD': (6, 0),
            'AUTO.TAKEOFF': (4, 2),
            'AUTO.LOITER': (4, 3),
            'AUTO.RTL': (4, 5),
            'AUTO.LAND': (4, 6),
        }
        main_m, sub_m = custom_map.get(mode_name, (4, 3))
        base_mode = mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        if self.tel.get('armed'):
            base_mode |= mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
        # PX4's DO_SET_MODE takes the main mode in param2 and the sub mode in param3 as plain numbers. The
        # heartbeat's packed custom_mode (main << 16 | sub << 24) here was truncated to main mode 0: PX4 ACKed it and
        # changed nothing - no OFFBOARD / LAND / RTL ever switched (found against the SIH SITL, 2026-10-27)
        return self.mav_cmd(mavutil.mavlink.MAV_CMD_DO_SET_MODE, base_mode, main_m, sub_m)

    # ---- VRX Video Capture & QR Detection Loop on Laptop
    def open_vrx(self):
        devs = [self.vrx_device] + sorted(glob.glob('/dev/video*'))
        for d in dict.fromkeys(devs):
            if not d or not Path(d).exists():
                continue
            cap = cv2.VideoCapture(d, cv2.CAP_V4L2)
            if not cap.isOpened():
                cap = cv2.VideoCapture(d)
            if cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None and frame.size > 0:
                    print(f'[VRX] Opened VRX video capture on {d} ({frame.shape[1]}x{frame.shape[0]})', flush=True)
                    self.vrx_device = d
                    return cap
                cap.release()
        return None

    def _decode_patch(self, img):
        if img is None or img.size == 0:
            return {}
        mean = float(img.mean())
        work = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(img) if (mean < 85 or mean > 195) else img
        out = {}
        if self.wc is not None:
            out = wechat_decode(self.wc, work)
            if not out and work is not img:
                out = wechat_decode(self.wc, img)
            if out:
                return out
        if self.zbar is not None:
            blur = cv2.GaussianBlur(work, (3, 3), 0)
            for sc in (1.0, 1.5):
                im = blur if sc == 1.0 else cv2.resize(blur, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
                for d in self.zbar(im):
                    pl = d.data.decode('utf-8', errors='replace').strip()
                    r = d.rect
                    pts = [(pt.x, pt.y) for pt in d.polygon] or [
                        (r.left, r.top), (r.left + r.width, r.top),
                        (r.left + r.width, r.top + r.height), (r.left, r.top + r.height),
                    ]
                    if pl and pl not in out:
                        out[pl] = np.array(pts, dtype=np.float64) / sc
                if out:
                    return out
        return out

    def pixel_to_ground_enu(self, px_pts, frame_w, frame_h, pose_enu, rpy_ned):
        """Project pixel coordinates (N x 2) from the downward VRX camera onto ground z = 0 in ENU metres."""
        if pose_enu is None:
            return None, 1.0
        xe, yn, zu, _ = pose_enu
        if zu < 0.4:
            zu = float(self.search_alt)  # bench test fallback when drone is on the ground
        pts = np.asarray(px_pts, dtype=np.float64).reshape(-1, 2)
        f = (frame_w / 2.0) / math.tan(math.radians(max(20.0, min(150.0, self.fov_deg))) / 2.0)
        cx, cy = (frame_w - 1) / 2.0, (frame_h - 1) / 2.0
        # Camera optical frame: +x right, +y down, +z out of lens.
        # Mounted straight down under FLU body: x_flu = -y_cam, y_flu = -x_cam, z_flu = -z_cam
        u = (pts[:, 0] - cx) / f
        v = (pts[:, 1] - cy) / f
        rays_flu = np.column_stack([-v, -u, -np.ones(len(pts), dtype=np.float64)])
        rays_flu /= np.linalg.norm(rays_flu, axis=1, keepdims=True)
        R_body_to_enu = rpy_to_rot_enu(*rpy_ned)
        rays_enu = rays_flu @ R_body_to_enu.T
        if (rays_enu[:, 2] > -0.08).any():
            return None, 0.1
        cos_nadir = float(np.clip(-rays_enu[0, 2], 0.15, 1.0))
        k = zu / (-rays_enu[:, 2])
        ground_xy = np.array([xe, yn]) + rays_enu[:, :2] * k[:, None]
        return ground_xy, cos_nadir ** 4

    def vrx_loop(self):
        cap = None
        last_det = 0.0
        while True:
            if cap is None or not cap.isOpened():
                cap = self.open_vrx()
                if cap is None:
                    time.sleep(1.5)
                    continue
            ok, frame = cap.read()
            if not ok or frame is None:
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

            now = time.monotonic()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
            with self.lock:
                pose_now = self.pose
                rpy_now = self.rpy_ned
                self.latest_gray = gray
                self.latest_frame_pose = (pose_now, rpy_now)
                S_ros_time_cam = now

            if now - last_det >= 0.20:  # 5 Hz QR detection & camera-adjusted scan on laptop
                last_det = now
                found, boxes_viz, zoom_patches = self.process_vrx_frame(gray, pose_now, rpy_now, now)
                self.render_hud_frame(frame, gray, found, boxes_viz, zoom_patches)

    def process_vrx_frame(self, gray, pose_now, rpy_now, now):
        h, w = gray.shape[:2]
        found = {}
        boxes_viz = []
        zoom_patches = []

        # 1. Contrast candidates + Multi-scale QrNet bounding-box search
        cands = find_qr_candidates(gray, max_cands=8)
        centres = [c['center'] for c in cands]
        boxes = []
        if self.qnet is not None and self.qnet.net is not None:
            boxes = self.qnet.scan(gray, conf=0.52, centres=centres if centres else None)
        for c in cands:
            if all(poly_iou(c['poly'], bp) < 0.3 for _, bp in boxes):
                boxes.append((0.48, c['poly']))

        # 2. For each bounding box: adjust/zoom virtual patch + multi-frame phase-correlated stacking (`aligned_mean`)
        for conf, poly in boxes[:8]:
            (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)
            bw, bh = x1 - x0, y1 - y0
            pad = max(28.0, 0.45 * max(bw, bh))
            X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
            X1, Y1 = int(min(w, x1 + pad)), int(min(h, y1 + pad))
            crop = gray[Y0:Y1, X0:X1]
            if crop.size == 0:
                continue
            # Zoomed virtual patch (upscaled 2x with Lanczos4 + CLAHE to sharpen VRX scanlines)
            zoom_crop = cv2.resize(crop, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            sx, sy = (X1 - X0) / 256.0, (Y1 - Y0) / 256.0
            got = self._decode_patch(zoom_crop)
            key = (int((x0 + x1) / 2) // 64, int((y0 + y1) / 2) // 64)
            work_img = zoom_crop
            if not got:
                stack = self.chase_stacks.setdefault(key, [])
                stack.append(zoom_crop)
                del stack[:-5]
                if len(stack) >= 2:
                    avg, _ = aligned_mean(stack)
                    clean = cv2.fastNlMeansDenoising(avg, None, 9, 5, 7)
                    work_img = clean
                    got = self._decode_patch(clean) or self._decode_patch(avg)
            if got:
                self.chase_stacks.pop(key, None)
                mapped = {pl: pp * [sx, sy] + [X0, Y0] for pl, pp in got.items()}
                found.update(mapped)
                lbl = next(iter(mapped.keys()))
                boxes_viz.append((conf, poly, lbl))
                if len(zoom_patches) < 3:
                    zoom_patches.append((lbl, work_img))
            else:
                boxes_viz.append((conf, poly, None))
                if len(zoom_patches) < 3:
                    zoom_patches.append((f'scan {conf:.2f}', work_img))
                # Project unread boxed QR to ground as a candidate for the drone to inspect
                g_pts, _ = self.pixel_to_ground_enu(np.vstack([poly.mean(axis=0), poly]), w, h, pose_now, rpy_now)
                if g_pts is not None:
                    self.add_candidate(float(g_pts[0, 0]), float(g_pts[0, 1]), conf=round(conf, 2), now=now)

        # 3. Full-frame decode pass for large/centered QR codes
        if not found:
            found.update(self._decode_patch(gray))

        # 4. Update short-window averaged QR ground positions & target match
        for payload, poly in found.items():
            g_pts, weight = self.pixel_to_ground_enu(np.vstack([poly.mean(axis=0), poly]), w, h, pose_now, rpy_now)
            if g_pts is not None:
                center_xy = (float(g_pts[0, 0]), float(g_pts[0, 1]))
                corners = g_pts[1:]
                box = [*corners.min(axis=0), *corners.max(axis=0)]
                self.add_qr_detection(payload, center_xy, box, weight, now, pose_now)

        return found, boxes_viz, zoom_patches

    def add_candidate(self, cx, cy, conf, now):
        with self.lock:
            if any(math.dist((q['x'], q['y']), (cx, cy)) < 3.5 for q in self.qrs):
                return
            for c in self.cands:
                if math.dist((c['x'], c['y']), (cx, cy)) < 3.5:
                    c['n'] += 1
                    samples = [s for s in c.get('samples', []) if now - s[0] <= 4.0] + [(now, cx, cy)]
                    c['samples'] = samples[-15:]
                    c['x'] = float(np.median([s[1] for s in c['samples']]))
                    c['y'] = float(np.median([s[2] for s in c['samples']]))
                    c['conf'] = max(c.get('conf', 0.0), conf)
                    return
            self.cands.append({'x': cx, 'y': cy, 'n': 1, 'conf': conf, 'samples': [(now, cx, cy)]})

    def add_qr_detection(self, payload, xy, box, weight, now, pose_now):
        with self.lock:
            # Fine target fix when below 6.5m over the matched start QR
            if self.start_qr and payload == self.start_qr and pose_now is not None and pose_now[2] < 6.5:
                prev = self.target_fine['pts'] if self.target_fine else []
                pts = (prev + [xy])[-15:]
                med = np.median(np.array(pts), axis=0)
                self.target_fine = {'x': float(med[0]), 'y': float(med[1]), 'n': len(pts), 'pts': pts}

            for q in self.qrs:
                if q['payload'] == payload and math.dist((q['x'], q['y']), xy) < 4.0:
                    q['n'] += 1
                    samples = [s for s in q.get('samples', []) if now - s[0] <= 4.0]
                    samples.append((now, xy[0], xy[1], box, max(0.05, weight)))
                    q['samples'] = samples[-15:]
                    arr = np.array([[s[1], s[2]] for s in q['samples']], dtype=np.float64)
                    ws = np.array([s[4] for s in q['samples']], dtype=np.float64)
                    if len(arr) >= 3:
                        med = np.median(arr, axis=0)
                        dists = np.linalg.norm(arr - med, axis=1)
                        keep = dists <= 2.5 * (float(np.median(dists)) + 0.12)
                        arr, ws = arr[keep], ws[keep]
                    ws /= max(ws.sum(), 1e-6)
                    q['x'] = float((arr[:, 0] * ws).sum())
                    q['y'] = float((arr[:, 1] * ws).sum())
                    self._update_qr_status(q)
                    return
            q = {
                'payload': payload, 'x': xy[0], 'y': xy[1], 'box': [float(v) for v in box],
                'n': 1, 'source': 'vrx', 'samples': [(now, xy[0], xy[1], box, max(0.05, weight))],
            }
            self._update_qr_status(q)
            self.qrs.append(q)
            print(f'[VRX-QR] Detected QR {payload!r} at ({xy[0]:.2f}, {xy[1]:.2f}) m -> status={q["status"]}', flush=True)

    def _update_qr_status(self, q):
        q['confirmed'] = q['n'] >= 2
        x0, x1, y0, y1 = self.zone_enu
        q['in_zone'] = bool(x0 <= q['x'] <= x1 and y0 <= q['y'] <= y1)
        if self.start_qr and q['payload'] == self.start_qr:
            q['status'] = 'target'
            if q['confirmed'] and self.target is None:
                self.target = (q['x'], q['y'])
                print(f'[VRX-QR] ★ TARGET MATCHED START QR ({q["payload"]!r}) at ({q["x"]:.2f}, {q["y"]:.2f}) m! '
                      f'Winch/Delivery unlocked!', flush=True)
        elif self.start_qr and q['confirmed']:
            q['status'] = 'rejected'
        else:
            q['status'] = 'pending'

    def compute_pairwise_distances(self):
        now = time.monotonic()
        with self.lock:
            qs = sorted(self.qrs, key=lambda q: q['payload'])
            out = []
            for i in range(len(qs)):
                for j in range(i + 1, len(qs)):
                    a, b = qs[i], qs[j]
                    key = (a['payload'], b['payload'])
                    d_inst = math.dist((a['x'], a['y']), (b['x'], b['y']))
                    hist = [h for h in self.pair_dist_hist.get(key, []) if now - h[0] <= 5.0] + [(now, d_inst)]
                    self.pair_dist_hist[key] = hist[-15:]
                    vals = np.array([h[1] for h in self.pair_dist_hist[key]], dtype=np.float64)
                    if len(vals) >= 3:
                        med = float(np.median(vals))
                        vals = vals[np.abs(vals - med) <= 2.5 * (float(np.median(np.abs(vals - med))) + 0.05)]
                    out.append({'qr_1': a['payload'], 'qr_2': b['payload'], 'distance_m': round(float(vals.mean()), 3),
                                'samples': len(hist)})
            return out

    def render_hud_frame(self, frame, gray, found, boxes_viz, zoom_patches):
        vis = frame.copy() if frame.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        ok_raw, raw_jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 75])
        for conf, poly, lbl in boxes_viz:
            pts = poly.astype(np.int32)
            col = (50, 230, 80) if lbl else (0, 200, 255)
            cv2.polylines(vis, [pts], True, col, 2)
            p0 = pts.min(axis=0)
            tag = f'{lbl} ({conf:.2f})' if lbl else f'QR box {conf:.2f} (adjusting scan)'
            cv2.putText(vis, tag, (int(p0[0]), max(16, int(p0[1]) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        for payload, poly in found.items():
            pts = poly.astype(np.int32)
            match = bool(self.start_qr and payload == self.start_qr)
            col = (40, 255, 40) if match else (40, 170, 255)
            cv2.polylines(vis, [pts], True, col, 2)
            p0 = pts.min(axis=0)
            tag = f'★ TARGET MATCH: {payload}' if match else f'QR: {payload}'
            cv2.putText(vis, tag, (int(p0[0]), max(18, int(p0[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, col, 2, cv2.LINE_AA)
        x_off = vis.shape[1] - 106
        for lbl, patch in zoom_patches[:3]:
            thumb = cv2.cvtColor(cv2.resize(patch, (96, 96)), cv2.COLOR_GRAY2BGR)
            if x_off >= 8 and vis.shape[0] >= 120:
                vis[8:104, x_off:x_off + 96] = thumb
                cv2.rectangle(vis, (x_off, 8), (x_off + 96, 104), (74, 163, 255), 1)
                cv2.putText(vis, str(lbl)[:13], (x_off + 2, 116),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 230, 240), 1, cv2.LINE_AA)
                x_off -= 104
        hdr = f'VRX ({self.vrx_device}) | State: {self.state} | Start QR: {self.start_qr or "none"}'
        cv2.putText(vis, hdr, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 240, 120), 2, cv2.LINE_AA)
        ok_ann, ann_jpg = cv2.imencode('.jpg', vis, [cv2.IMWRITE_JPEG_QUALITY, 78])
        with self.lock:
            if ok_raw:
                self.raw_jpeg = raw_jpg.tobytes()
            if ok_ann:
                self.annotated_jpeg = ann_jpg.tobytes()

    # ---- Autonomous Delivery-Only Mission Execution (Over Ground Telemetry Radio)
    def fly_to(self, tx, ty, tz, yaw=None, vmax=1.5, tol=0.45, timeout=90.0):
        """Smoothly step OFFBOARD position setpoint to (tx, ty, tz) at `vmax` m/s so PX4 never jerks."""
        t0 = time.monotonic()
        with self.lock:
            p0 = self.pose or (0.0, 0.0, tz, 0.0)
        cx, cy, cz, cyaw = p0
        if yaw is None:
            yaw = math.atan2(ty - cy, tx - cx) if math.hypot(tx - cx, ty - cy) > 1.0 else cyaw
        dt = 0.1
        while time.monotonic() - t0 < timeout:
            if self.abort_reason:
                raise RuntimeError(self.abort_reason)
            with self.lock:
                p = self.pose or (cx, cy, cz, yaw)
            px, py, pz, _ = p
            if math.hypot(tx - px, ty - py) <= tol and abs(tz - pz) <= 0.45:
                with self.lock:
                    self.setpoint = (tx, ty, tz, yaw)
                return True
            dx, dy, dz = tx - cx, ty - cy, tz - cz
            dist = math.hypot(dx, dy)
            step = vmax * dt
            if dist > step:
                cx += dx / dist * step
                cy += dy / dist * step
            else:
                cx, cy = tx, ty
            if abs(dz) > 0.8 * dt:
                cz += math.copysign(0.8 * dt, dz)
            else:
                cz = tz
            with self.lock:
                self.setpoint = (cx, cy, cz, yaw)
            time.sleep(dt)
        raise RuntimeError(f'fly_to ({tx:.1f}, {ty:.1f}, {tz:.1f}) timed out')

    def arm_takeoff(self, x, y, alt, yaw, timeout=45.0):
        """Arm and climb to `alt` over (x, y), ending in the setpoint mode with setpoints streaming.
        PX4: stream the setpoint, OFFBOARD, arm, fly up. ArduCopter: GUIDED, arm, MAV_CMD_NAV_TAKEOFF (it ignores
        position targets until it has taken off), then stream the setpoint."""
        def wait(cond, secs, what):
            end = time.monotonic() + secs
            while time.monotonic() < end:
                if self.abort_reason:
                    raise RuntimeError(self.abort_reason)
                if cond():
                    return
                time.sleep(0.2)
            raise RuntimeError(f'{what} (mode {self.tel.get("mode")}, armed {self.tel.get("armed")}): see the '
                               'autopilot messages in the log')
        with self.lock:
            self.setpoint = (x, y, alt, yaw)
        if self.autopilot == 'ardupilot':
            self.offboard_active = False
            for _ in range(5):
                if self.tel.get('mode') == 'GUIDED':
                    break
                self.set_px4_mode('OFFBOARD')
                time.sleep(1.0)
            wait(lambda: self.tel.get('mode') == 'GUIDED', 3.0, 'ArduCopter did not switch to GUIDED')
            for _ in range(5):
                if self.tel.get('armed'):
                    break
                self.mav_cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, 0)
                time.sleep(1.5)
            wait(lambda: self.tel.get('armed'), 3.0, 'ArduCopter did not arm (pre-arm checks)')
            self.mav_cmd(mavutil.mavlink.MAV_CMD_NAV_TAKEOFF, 0, 0, 0, 0, 0, 0, alt)
            # a band, not just a floor: a broken height estimate read 144 m as 'reached' and flew on (SITL, 2026-10-27)
            wait(lambda: self.pose is not None and alt * 0.85 <= self.pose[2] <= alt + 1.0, timeout,
                 'takeoff did not settle at the target height')
            self.offboard_active = True
            return
        self.offboard_active = True
        time.sleep(0.5)
        for _ in range(10):
            if self.abort_reason:
                raise RuntimeError(self.abort_reason)
            if self.tel.get('armed') and self.tel.get('mode') == 'OFFBOARD':
                break
            self.set_px4_mode('OFFBOARD')
            self.mav_cmd(mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 1, 0)
            time.sleep(1.5)
        wait(lambda: self.tel.get('armed') and self.tel.get('mode') == 'OFFBOARD', 1.0, 'PX4 did not arm in OFFBOARD')
        self.fly_to(x, y, alt, yaw=yaw, vmax=min(1.2, max(0.5, alt / 4)), tol=0.3, timeout=timeout)

    def trigger_winch_drop(self):
        """Trigger payload release via MAV_CMD_DO_GRIPPER (211) + servo fallback."""
        self.log_mission('DROP', action='releasing winch / gripper')
        ok1 = self.mav_cmd(211, 1, 0)  # MAV_CMD_DO_GRIPPER: GRIPPER_ACTION_RELEASE = 1
        time.sleep(1.5)
        self.mav_cmd(211, 0, 0)        # GRIPPER_ACTION_GRAB = 0 (reset after drop_wait)
        return ok1

    def scan_start_qr_here(self, duration=5.0):
        """Scan VRX frames for `duration` seconds and latch the most frequent QR payload as `start_qr`."""
        votes = Counter()
        t_end = time.monotonic() + duration
        while time.monotonic() < t_end:
            if self.abort_reason:
                raise RuntimeError(self.abort_reason)
            with self.lock:
                gray = self.latest_gray
            if gray is not None:
                for pl in self._decode_patch(gray):
                    votes[pl] += 1
            if any(c >= 2 for c in votes.values()):
                break
            time.sleep(0.15)
        if not votes:
            return None
        winner = votes.most_common(1)[0][0]
        with self.lock:
            self.start_qr = winner
            for q in self.qrs:
                self._update_qr_status(q)
        return winner

    def run_delivery_mission(self, read_start_qr=True):
        self.mission_running = True
        self.abort_reason = None
        self.mission_t0 = time.time()
        with self.lock:
            self.target = None
            self.target_fine = None
            self.qrs = []
            self.cands = []
            home_pose = self.pose or (0.0, 0.0, 0.0, 0.0)
        home = (home_pose[0], home_pose[1])
        yaw0 = home_pose[3]
        try:
            # 1. TAKEOFF to 5.0 m
            self.log_mission('TAKEOFF', altitude_m=self.takeoff_alt)
            self.arm_takeoff(home[0], home[1], self.takeoff_alt, yaw0)

            # 2. READ_START_QR: Move 1.0 m forward at 5.0 m altitude and scan Start QR on VRX
            if read_start_qr or not self.start_qr:
                self.log_mission('READ_START_QR', step='1m forward at 5m')
                for k in range(8):
                    yaw = yaw0 + k * (math.pi / 4.0)
                    self.fly_to(home[0] + math.cos(yaw), home[1] + math.sin(yaw), self.takeoff_alt, yaw=yaw)
                    pl = self.scan_start_qr_here(duration=4.5)
                    if pl:
                        self.log_mission('READ_START_QR', payload=pl)
                        break
                if not self.start_qr:
                    raise RuntimeError('Start QR not decoded from VRX feed')
            else:
                self.log_mission('READ_START_QR', using_preset=self.start_qr)

            # 3. DIRECT_TO_ZONE & SEARCH_DELIVERY at 10.0 m altitude
            self.zone_t0 = time.time()
            self.log_mission('DIRECT_TO_ZONE', search_altitude_m=self.search_alt)
            with self.lock:
                cur = self.pose or (home[0], home[1], self.takeoff_alt, yaw0)
            self.fly_to(cur[0], cur[1], self.search_alt, yaw=yaw0)

            x0, x1, y0, y1 = self.zone_enu
            lane_spacing = 5.0
            ys = np.arange(y0 + 2.5, y1 - 2.0, lane_spacing)
            if len(ys) == 0:
                ys = [0.0]
            wps = []
            for idx, y_lane in enumerate(ys):
                xs = (x0 + 2.5, x1 - 2.5) if idx % 2 == 0 else (x1 - 2.5, x0 + 2.5)
                wps.extend([(xs[0], float(y_lane)), (xs[1], float(y_lane))])

            self.log_mission('SEARCH_DELIVERY', start_qr=self.start_qr, waypoints=len(wps))
            for wx, wy in wps:
                if self.target is not None:
                    break
                # If a boxed QR candidate is seen on the ground, detour over it first to scan it!
                with self.lock:
                    cand = next((c for c in self.cands if not c.get('visited')), None)
                if cand is not None:
                    cand['visited'] = True
                    self.log_mission('SEARCH_DELIVERY', investigating_candidate=[round(cand['x'], 2), round(cand['y'], 2)])
                    self.fly_to(cand['x'], cand['y'], self.search_alt, vmax=1.5)
                    time.sleep(2.0)
                    if self.target is not None:
                        break
                self.fly_to(wx, wy, self.search_alt, vmax=1.5)

            # 4. DELIVER & DROP (Strictly when target QR payload == start_qr!)
            if self.target is not None:
                tx, ty = self.target
                self.log_mission('DELIVER', target=[round(tx, 2), round(ty, 2)], payload=self.start_qr)
                self.fly_to(tx, ty, self.search_alt, vmax=1.5, tol=0.20)
                with self.lock:
                    self.target_fine = None
                self.fly_to(tx, ty, self.deliver_alt, vmax=0.6, tol=0.15)
                # Wait up to 5s at 5m drop height for fine centering over the QR on the VRX feed
                t_fine_end = time.monotonic() + 5.0
                while time.monotonic() < t_fine_end:
                    with self.lock:
                        f = self.target_fine
                    if f and f['n'] >= 3:
                        if math.hypot(f['x'] - tx, f['y'] - ty) <= 2.0:
                            tx, ty = f['x'], f['y']
                        break
                    time.sleep(0.2)
                self.fly_to(tx, ty, self.deliver_alt, vmax=0.35, tol=0.10)
                time.sleep(1.5)  # settle hold
                self.trigger_winch_drop()
                self.fly_to(tx, ty, self.search_alt, vmax=1.0)
            else:
                self.log_mission('SEARCH_DELIVERY', result='target matching start_qr not found in zone')

            # 5. RETURN_HOME & LAND
            self.log_mission('RETURN_HOME', home=[round(home[0], 2), round(home[1], 2)])
            self.fly_to(home[0], home[1], self.search_alt, vmax=1.5)
            self.log_mission('LAND')
            self.offboard_active = False
            self.set_px4_mode('AUTO.LAND')
            t_land = time.monotonic() + 60.0
            while time.monotonic() < t_land and self.tel.get('armed'):
                time.sleep(0.5)
            self.log_mission('DONE', total_s=round(time.time() - self.mission_t0, 1))
        except Exception as e:
            self.log_mission('FAILSAFE', reason=str(e))
            self.offboard_active = False
            self.set_px4_mode('AUTO.LAND')
        finally:
            self.mission_running = False

    def snapshot(self):
        now = time.monotonic()
        pairs = self.compute_pairwise_distances()
        with self.lock:
            hb_age = round(now - self.tel_time['HEARTBEAT'], 1) if 'HEARTBEAT' in self.tel_time else None
            qrs_clean = [{k: v for k, v in q.items() if k != 'samples'} for q in self.qrs]
            cands_clean = [{k: v for k, v in c.items() if k != 'samples'} for c in self.cands]
            return {
                'time': time.time(),
                'tel': dict(self.tel),
                'link_age': hb_age,
                'radio': self.args.radio,
                'radio_err': self.radio_err,
                'vrx': self.vrx_device,
                'pose': self.pose,
                'track': self.track[-1500:],
                'zone': self.zone_enu,
                'state': self.state,
                'fsm': FSM_DELIVERY,
                'start_qr': self.start_qr,
                'target': self.target,
                'qrs': qrs_clean,
                'candidates': cands_clean,
                'pairwise_distances_m': pairs,
                'log': self.log_entries[-60:],
                'statustext': self.statustext[-15:],
                'mission_s': (time.time() - self.mission_t0) if self.mission_t0 else None,
            }


DASHBOARD_HTML = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Laptop VRX + Telemetry Delivery GCS</title>
<style>
  :root { --bg:#0f1318; --panel:#171d24; --line:#2a333d; --text:#dfe6ee; --dim:#8b98a6; --ok:#3fbf6f; --bad:#e5484d; --warn:#f1a33b; --acc:#4aa3ff; }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:13px/1.4 system-ui, sans-serif; }
  header { display:flex; flex-wrap:wrap; gap:10px; align-items:center; padding:10px 14px; background:var(--panel); border-bottom:1px solid var(--line); }
  .pill { padding:4px 10px; border-radius:12px; background:#222b35; color:var(--dim); font-weight:600; }
  .pill.ok { background:#163323; color:var(--ok); } .pill.bad { background:#3a1517; color:var(--bad); } .pill.warn { background:#3a2a12; color:var(--warn); }
  main { display:grid; grid-template-columns: 1.15fr 1fr; gap:10px; padding:10px; }
  .card { background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:10px; }
  .card h2 { font-size:12px; text-transform:uppercase; color:var(--dim); margin:0 0 8px; }
  img#cam { width:100%; border-radius:6px; background:#000; min-height:260px; display:block; }
  canvas#map { width:100%; height:300px; background:#0b0f13; border-radius:6px; display:block; }
  .tiles { display:grid; grid-template-columns: repeat(3, 1fr); gap:6px; }
  .tile { background:#11161c; border-radius:6px; padding:6px 8px; }
  .tile b { display:block; font-size:16px; } .tile span { color:var(--dim); font-size:11px; }
  .fsm { display:flex; flex-wrap:wrap; gap:4px; margin-bottom:8px; }
  .fsm span { padding:3px 8px; border-radius:4px; background:#222b35; color:var(--dim); font-size:11px; }
  .fsm span.now { background:var(--acc); color:#fff; font-weight:700; }
  button { background:#26313d; color:var(--text); border:1px solid var(--line); border-radius:5px; padding:7px 12px; cursor:pointer; font:inherit; font-weight:600; }
  button.go { background:#1d5c36; border-color:#2b8a50; } button.stop { background:#6b1f22; border-color:#a3343a; }
  input { background:#11161c; color:var(--text); border:1px solid var(--line); border-radius:4px; padding:6px; font:inherit; }
  .row { display:flex; flex-wrap:wrap; gap:6px; align-items:center; margin:6px 0; }
  .ok { color:var(--ok); } .bad { color:var(--bad); } .warn { color:var(--warn); }
  #log { height:160px; overflow:auto; font:12px ui-monospace, monospace; background:#0b0f13; padding:6px; border-radius:6px; }
</style></head>
<body>
<header>
  <b>Laptop VRX + Telemetry Delivery Mission GCS</b>
  <span class="pill" id="p_radio">radio: -</span>
  <span class="pill" id="p_vrx">VRX: -</span>
  <span class="pill" id="p_state">state: -</span>
  <span class="pill" id="p_sqr">Start QR: -</span>
</header>
<main>
  <div style="display:flex;flex-direction:column;gap:10px">
    <div class="card">
      <h2>Live VRX Video + Camera-Adjusted QR Detection HUD</h2>
      <div class="row">
        <input id="vrxdev" placeholder="/dev/video2" size="14">
        <button onclick="post('set_vrx', {device: document.getElementById('vrxdev').value})">Switch VRX Device</button>
        <button onclick="post('scan_start_qr')">Scan Start QR Now (from VRX)</button>
      </div>
      <img id="cam" src="/annotated.jpg" alt="VRX Feed">
      <div id="qrbox" style="margin-top:8px"></div>
    </div>
    <div class="card">
      <h2>Delivery Zone Track & Detected QRs (ENU metres from Home)</h2>
      <canvas id="map"></canvas>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:10px">
    <div class="card">
      <h2>Telemetry (via Ground Telemetry Unit)</h2>
      <div class="tiles" id="tiles"></div>
    </div>
    <div class="card">
      <h2>Delivery-Only Mission Control</h2>
      <div class="fsm" id="fsm"></div>
      <div class="row">
        <input id="payload" placeholder="Start QR payload (or leave blank to scan at 5m + 1m fwd)" style="flex:1">
        <button onclick="post('set_start_qr', {payload: document.getElementById('payload').value})">Set Start QR</button>
      </div>
      <div class="row">
        <label>Zone X (m): <input id="zx0" value="-10" size="4"> to <input id="zx1" value="25" size="4"></label>
        <label>Y (m): <input id="zy0" value="-12" size="4"> to <input id="zy1" value="12" size="4"></label>
        <button onclick="post('set_zone', {zone: [+val('zx0'), +val('zx1'), +val('zy0'), +val('zy1')]})">Apply Zone</button>
      </div>
      <div class="row">
        <label><input type="checkbox" id="readqr" checked> Scan Start QR at 5m takeoff (+1m fwd)</label>
      </div>
      <div class="row">
        <button class="go" onclick="post('start_delivery', {read_qr: document.getElementById('readqr').checked, payload: val('payload')})">▶ Start Delivery-Only Mission</button>
        <button class="stop" onclick="post('abort')">■ Abort</button>
        <button onclick="post('drop')">Trigger Winch / Drop</button>
      </div>
      <div class="row">
        <button class="stop" onclick="post('land')">LAND</button>
        <button class="stop" onclick="post('rtl')">RTL</button>
        <button class="stop" onclick="if(confirm('KILL MOTORS?')) post('kill')">KILL</button>
      </div>
      <div id="msg" style="color:var(--dim);margin-top:4px"></div>
    </div>
    <div class="card">
      <h2>Mission Log · PX4 Status</h2>
      <div id="log"></div>
    </div>
  </div>
</main>
<script>
const $ = id => document.getElementById(id), val = id => $(id).value;
async function post(what, body={}) {
  const r = await fetch('/api/' + what, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
  const j = await r.json();
  $('msg').textContent = (j.ok ? '✓ ' : '✗ ') + (j.message || '');
}
setInterval(() => { $('cam').src = '/annotated.jpg?t=' + Date.now(); }, 220);
function drawMap(s) {
  const cv = $('map'), ctx = cv.getContext('2d');
  const W = cv.clientWidth, H = cv.clientHeight; cv.width = W; cv.height = H;
  ctx.clearRect(0, 0, W, H);
  const [zx0, zx1, zy0, zy1] = s.zone || [-10, 25, -12, 12];
  const pad = 4, x0 = Math.min(-5, zx0) - pad, x1 = Math.max(5, zx1) + pad;
  const y0 = Math.min(-5, zy0) - pad, y1 = Math.max(5, zy1) + pad;
  const sc = Math.min(W / (x1 - x0), H / (y1 - y0));
  const X = x => (x - x0) * sc, Y = y => H - (y - y0) * sc;
  ctx.strokeStyle = '#e8c547'; ctx.setLineDash([5, 4]);
  ctx.strokeRect(X(zx0), Y(zy1), (zx1 - zx0) * sc, (zy1 - zy0) * sc); ctx.setLineDash([]);
  (s.qrs || []).forEach(q => {
    ctx.fillStyle = q.status === 'target' ? '#3fbf6f' : q.status === 'rejected' ? '#e5484d' : '#f1a33b';
    ctx.fillRect(X(q.x) - 5, Y(q.y) - 5, 10, 10);
    ctx.fillText(q.payload, X(q.x) + 7, Y(q.y) + 4);
  });
  if (s.pose) {
    ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(X(s.pose[0]), Y(s.pose[1]), 5, 0, 7); ctx.fill();
  }
}
function render(s) {
  const t = s.tel || {};
  $('p_radio').textContent = s.radio_err ? `radio: ${s.radio_err}` : `radio ${s.radio} (${s.link_age ?? '-'}s)`;
  $('p_radio').className = 'pill ' + (!s.radio_err && s.link_age != null && s.link_age < 3 ? 'ok' : 'bad');
  $('p_vrx').textContent = `VRX: ${s.vrx}`; $('p_vrx').className = 'pill ok';
  $('p_state').textContent = `state: ${s.state}`; $('p_state').className = 'pill ok';
  $('p_sqr').textContent = `Start QR: ${s.start_qr || 'awaiting scan'}`;
  $('p_sqr').className = 'pill ' + (s.start_qr ? 'ok' : 'warn');
  $('fsm').innerHTML = (s.fsm || []).map(n => `<span class="${n === s.state ? 'now' : ''}">${n}</span>`).join('');
  const tiles = [
    ['Mode', t.mode], ['Armed', t.armed ? 'ARMED' : 'disarmed'], ['Alt (rel)', (t.alt_rel ?? '-') + ' m'],
    ['Speed', (t.groundspeed ?? '-') + ' m/s'], ['Battery', (t.batt_pct ?? '-') + ' %'], ['GPS', `fix ${t.gps_fix ?? '-'} · ${t.sats ?? '-'} sat`],
    ['Roll/Pitch', `${t.roll ?? '-'}° / ${t.pitch ?? '-'}°`], ['Local X/Y', s.pose ? `${s.pose[0].toFixed(1)} / ${s.pose[1].toFixed(1)} m` : '-'],
    ['Target', s.target ? `(${s.target[0].toFixed(1)}, ${s.target[1].toFixed(1)})` : 'searching'],
  ];
  $('tiles').innerHTML = tiles.map(([k, v]) => `<div class="tile"><span>${k}</span><b>${v ?? '-'}</b></div>`).join('');
  const qrs = (s.qrs || []).map(q => `<div class="${q.status==='target'?'ok':q.status==='rejected'?'bad':'warn'}">` +
    `${q.status==='target'?'★ TARGET MATCH (Winch Ready)':q.status.toUpperCase()}: <b>${q.payload}</b> at (${q.x.toFixed(2)}, ${q.y.toFixed(2)}) m [${q.n} hits]</div>`).join('');
  const dists = (s.pairwise_distances_m || []).map(d => `<div>Dist ${d.qr_1} ↔ ${d.qr_2}: <b>${d.distance_m.toFixed(3)} m</b> (avg)</div>`).join('');
  $('qrbox').innerHTML = qrs + dists;
  $('log').innerHTML = (s.log || []).map(l => `<div><b>${l.state}</b> ${JSON.stringify(l)}</div>`).join('');
  drawMap(s);
}
function connect() {
  const ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onmessage = e => render(JSON.parse(e.data));
  ws.onclose = () => setTimeout(connect, 1500);
}
connect();
</script></body></html>
"""


def make_web_app(sys_ctrl):
    async def index(_):
        return web.Response(text=DASHBOARD_HTML, content_type='text/html')

    async def annotated(_):
        with sys_ctrl.lock:
            jpg = sys_ctrl.annotated_jpeg or sys_ctrl.raw_jpeg
        if not jpg:
            raise web.HTTPServiceUnavailable()
        return web.Response(body=jpg, content_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    async def frame(_):
        with sys_ctrl.lock:
            jpg = sys_ctrl.raw_jpeg
        if not jpg:
            raise web.HTTPServiceUnavailable()
        return web.Response(body=jpg, content_type='image/jpeg', headers={'Cache-Control': 'no-store'})

    async def status(_):
        return web.json_response(sys_ctrl.snapshot())

    async def ws_handler(request):
        ws = web.WebSocketResponse(heartbeat=10)
        await ws.prepare(request)
        try:
            while not ws.closed:
                await ws.send_str(json.dumps(sys_ctrl.snapshot()))
                await asyncio.sleep(0.25)
        except Exception:
            pass
        return ws

    async def api(request):
        what = request.match_info['what']
        body = await request.json() if request.can_read_body else {}
        if what == 'set_vrx':
            sys_ctrl.vrx_device = body.get('device', '/dev/video0').strip()
            return web.json_response({'ok': True, 'message': f'VRX set to {sys_ctrl.vrx_device}'})
        if what == 'set_start_qr':
            pl = (body.get('payload') or '').strip()
            with sys_ctrl.lock:
                sys_ctrl.start_qr = pl or None
                for q in sys_ctrl.qrs:
                    sys_ctrl._update_qr_status(q)
            return web.json_response({'ok': True, 'message': f'Start QR set to {sys_ctrl.start_qr!r}'})
        if what == 'scan_start_qr':
            loop = asyncio.get_running_loop()
            pl = await loop.run_in_executor(None, lambda: sys_ctrl.scan_start_qr_here(3.5))
            return web.json_response({'ok': bool(pl), 'message': f'Scanned Start QR: {pl!r}' if pl else 'No QR decoded on VRX'})
        if what == 'set_zone':
            z = body.get('zone', [-10, 25, -12, 12])
            if len(z) == 4:
                sys_ctrl.zone_enu = tuple(float(v) for v in z)
            return web.json_response({'ok': True, 'message': f'Delivery zone set to {sys_ctrl.zone_enu}'})
        if what == 'start_delivery':
            if sys_ctrl.mission_running:
                return web.json_response({'ok': False, 'message': 'Mission already running'})
            pl = (body.get('payload') or '').strip()
            if pl:
                sys_ctrl.start_qr = pl
            read_qr = bool(body.get('read_qr', True))
            threading.Thread(target=sys_ctrl.run_delivery_mission, args=(read_qr,), daemon=True).start()
            return web.json_response({'ok': True, 'message': 'Delivery-Only mission started'})
        if what == 'abort':
            sys_ctrl.abort_reason = 'Operator abort'
            sys_ctrl.offboard_active = False
            sys_ctrl.set_px4_mode('AUTO.LOITER')
            return web.json_response({'ok': True, 'message': 'Mission aborted -> hold position'})
        if what == 'drop':
            loop = asyncio.get_running_loop()
            ok = await loop.run_in_executor(None, sys_ctrl.trigger_winch_drop)
            return web.json_response({'ok': ok, 'message': 'Winch / Gripper release sent'})
        if what == 'land':
            sys_ctrl.abort_reason = 'Operator LAND'
            sys_ctrl.offboard_active = False
            sys_ctrl.set_px4_mode('AUTO.LAND')
            return web.json_response({'ok': True, 'message': 'LAND sent to PX4'})
        if what == 'rtl':
            sys_ctrl.abort_reason = 'Operator RTL'
            sys_ctrl.offboard_active = False
            sys_ctrl.set_px4_mode('AUTO.RTL')
            return web.json_response({'ok': True, 'message': 'RTL sent to PX4'})
        if what == 'kill':
            sys_ctrl.abort_reason = 'Operator KILL'
            sys_ctrl.offboard_active = False
            sys_ctrl.mav_cmd(400, 0, 21196)
            return web.json_response({'ok': True, 'message': 'KILL sent to PX4'})
        raise web.HTTPNotFound()

    app = web.Application()
    app.router.add_get('/', index)
    app.router.add_get('/annotated.jpg', annotated)
    app.router.add_get('/frame.jpg', frame)
    app.router.add_get('/status.json', status)
    app.router.add_get('/ws', ws_handler)
    app.router.add_post('/api/{what}', api)
    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--vrx', default='/dev/video2', help='USB VRX capture device on laptop (e.g. /dev/video2 or /dev/video0)')
    ap.add_argument('--radio', default='/dev/ttyUSB0:57600', help='Ground telemetry radio (e.g. /dev/ttyUSB0:57600)')
    ap.add_argument('--qgc', default='127.0.0.1:14550', help='QGroundControl UDP relay address')
    ap.add_argument('--fov-deg', type=float, default=90.0, help='VRX camera horizontal FOV in degrees')
    ap.add_argument('--start-qr', default='', help='Optional preset Start QR payload')
    ap.add_argument('--wechat-dir', default='', help='Directory containing WeChat QR Caffe models')
    ap.add_argument('--port', type=int, default=8082, help='Dashboard HTTP port on the laptop (default 8082)')
    args = ap.parse_args()

    sys_ctrl = VrxDeliverySystem(args)
    for fn in (sys_ctrl.mav_rx_loop, sys_ctrl.qgc_tx_loop, sys_ctrl.offboard_tx_loop, sys_ctrl.vrx_loop):
        threading.Thread(target=fn, daemon=True).start()
    print(f'[VRX-DELIVERY] Dashboard + QR HUD live at http://127.0.0.1:{args.port} '
          f'(VRX={args.vrx}, Radio={args.radio}, QGC relay={args.qgc})', flush=True)
    app = make_web_app(sys_ctrl)
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
