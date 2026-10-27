#!/usr/bin/env python3
"""Calibrate the real Arducam IMX219 fisheye and test QR reading - a web page on the Jetson, no ROS needed.

    python3 arducam_calib.py [--port 8090] [--board 9x6] [--sensor-id 0]       (in the dev container)
    then open http://10.42.0.247:8090 on the Lenovo

The camera is opened directly (GStreamer nvarguscamerasrc, 1640x1232 = the mode the drone flies, 20 ms exposure cap),
so stop the ROS camera first (scripts/arducam_down.sh): only one process can hold it. The sim stack doesn't use it.
Modes (buttons on the page):
  calib   checkerboard of --board inner corners (cols x rows; default 9x6 = 10 x 7 squares) - the page serves one to
          show full screen on a laptop / tablet (/board.png) or to print. Frames are captured by themselves when the
          board is found AND the view is new (board centre / size / tilt differ from every capture so far); cover the
          whole image, the EDGES especially (a fisheye is mostly edge), near and far, tilted. 20-40 captures, then
          Calibrate: cv2.fisheye (equidistant, k1..k4; frames that break the fit are dropped one at a time) -> YAML
          for vio_bringup fisheye_camera_info (calibration_file) and the RMS reprojection error (< 1 px is good).
          The square size only scales the board poses, not the lens: any screen size works.
  undist  the live image undistorted with the calibration (straight lines must look straight)
  qr      decode QRs live at full resolution (3280 x 2464, --qr-max-exposure-ms): on a STRAIGHTENED view of the
          middle of the lens (FOV buttons: 60 / 75 / 90 / 120 deg, or the whole fisheye) - the calibration's lens, else
          an estimate from the 175 deg image circle - with OpenCV's WeChat decoder (a CNN detector + super-resolution:
          every 3 px/module code of a noisy test, zbar 12-17 of 20, 2026-10-26) on 768 px tiles, pyzbar every 3rd frame
          as a backup: boxes, payloads, px/module, decode rate, angle off the optical axis. /qr.png = a test code
          (calibration/test_qr.png: AEROTHON26-TEST-0001).
Output: <out> (default /workspaces/isaac_ros-dev/calibration/arducam_fisheye_1640x1232.yaml), captures in
<out dir>/arducam_calib_frames/ (so a calibration can be recomputed: --recompute).
"""
import argparse
import json
import math
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

PAGE = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Arducam calibration</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{padding:8px 12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;background:#222}
button{padding:8px 12px;border:0;border-radius:6px;background:#3a3f4b;color:#eee;font-size:14px;cursor:pointer}
button.on{background:#2e7d32}button.warn{background:#8a5a00}
#st{padding:6px 12px;font-family:ui-monospace,monospace;font-size:13px;white-space:pre-wrap}
img#v{width:100%;max-width:1640px;display:block}a{color:#8cf}</style></head><body>
<header><b>Arducam</b>
<button id="m_view" onclick="mode('view')">View</button>
<button id="m_calib" onclick="mode('calib')">Calibrate: capture</button>
<button onclick="post('capture')">Capture now</button>
<button class="warn" onclick="post('calibrate')">Compute calibration</button>
<button onclick="if(confirm('Forget all captures?'))post('reset')">Clear captures</button>
<button id="m_undist" onclick="mode('undist')">Undistorted</button>
<button id="m_detect" onclick="mode('detect')">Detect QRs</button>
<button id="m_qr" onclick="mode('qr')">QR test (read)</button>
<span style="color:#aaa">QR view:</span>
<button id="f_0" onclick="fov(0)">fisheye</button><button id="f_120" onclick="fov(120)">120°</button>
<button id="f_90" onclick="fov(90)">90°</button><button id="f_75" onclick="fov(75)">75°</button><button id="f_60" onclick="fov(60)">60°</button>
<button id="az" onclick="post('autozoom')">auto-zoom</button>
<span style="color:#aaa">exposure:</span>
<button id="e_3" onclick="expo(3)">3 ms</button><button id="e_5" onclick="expo(5)">5 ms</button>
<button id="e_8" onclick="expo(8)">8 ms</button><button id="e_12" onclick="expo(12)">12 ms</button>
<span style="color:#aaa">QR size m:</span><input id="qs" size="4" title="side of the dark code square (not the paper), for the distances; empty = in code widths" onchange="post('qrsize','?v='+(this.value||0))">
<span style="color:#aaa">options:</span>
<button id="btn_gpu" onclick="post('toggle_gpu')">GPU: OFF</button>
<button id="btn_enhance" onclick="post('toggle_enhance')">Enhance: OFF</button>
<button id="btn_hq" onclick="post('toggle_hq')">HQ: 70%</button>
<a href="/board.png" target="_blank">checkerboard</a> <a href="/qr.png" target="_blank">test QR</a></header>
<details id="help" style="padding:4px 12px;background:#1a1d22">
<summary style="cursor:pointer;color:#8cf">How to calibrate the camera (click to open)</summary>
<ol style="line-height:1.6;max-width:1100px">
<li><b>Show the checkerboard.</b> Click the <i>checkerboard</i> link (top right) on a laptop, tablet or phone and make it
full screen - or print it and tape it flat to cardboard. Any size works (9 x 6 inside corners). Good, even light; no glare.</li>
<li><b>Click "Calibrate: capture".</b> The page shows the raw fisheye (calibration needs it).</li>
<li><b>Hold the board 30-80 cm from the lens</b>, the whole board in view, and keep it still for a second. When its corners
are found, coloured dots appear on them. Each NEW position is captured by itself: the frame flashes green and an orange
outline of the board stays on the image. The status line counts the captures.</li>
<li><b>Move it everywhere.</b> Centre first, then each of the four corners and the four edges of the image - the fisheye's
edge matters most - near and a little farther, and tilted about 30-45 degrees left, right, up and down. Aim for
<b>25-40 captures</b> with the orange outlines covering the whole image circle. "Capture now" forces one; a blurred
or wrong capture: "Clear captures" and start again.</li>
<li><b>Click "Compute calibration"</b> (10-60 s). It reports the RMS error: <b>under 1 px is good</b> (0.3-0.6 is
typical); over 1 px - recapture with sharper, more varied views. It is saved by itself
(calibration/arducam_fisheye_1640x1232.yaml) and the captures are kept: add more and compute again any time.</li>
<li><b>Click "Undistorted"</b> (the FOV buttons set how wide; "fisheye" = 120 deg) to check: straight lines (door frames, ceiling edges, shelves) must look straight right
to the edges. From now on the QR view and the zoom use the true lens, and the real mission stack loads it by itself.</li>
</ol>
<p style="max-width:1100px;color:#aaa"><b>Detect QRs</b>: boxes only, no reading - <span style="color:#3cff3c">green</span> =
a QR found by the trained QR detection network (its confidence), <span style="color:#ff8c00">orange</span> = found by its
three corner squares, <span style="color:#ff40ff">magenta</span> = the mission's contrast detector's guess (how QR-like).
Boxes are numbered #1, #2 ... left to right; the cyan lines give the centre-to-centre distance between neighbouring codes (every pair: the status line), from the
calibrated lens: each code's range from its apparent size, all codes put on one plane (a wall / the floor). Type the side of
the dark code square in <i>QR size m</i> for metres, else they are in code widths (~5 % - check one with a tape).
<b>QR test (read)</b>: reads codes (yellow); as soon as a box is found but can't be read the camera switches to SCAN
settings (full resolution, 10 fps, exposure up to 80 ms: far less sensor grain - hold the camera still) and the view
zooms onto the box: 30 deg x2 (6 frames aligned + averaged) -> 15 deg x3 (10 frames), back to wide once read or gone.</p>
</details>
<div id="st">...</div><img id="v" alt="camera">
<script>
async function post(p,q=''){const r=await fetch('/'+p+q,{method:'POST'});const j=await r.json();if(j.message)alert(j.message)}
function mode(m){post('mode','?m='+m)}
function fov(v){post('fov','?v='+v)}
function expo(v){post('exposure','?v='+v)}
// one small JPEG at a time, swapped in only when fully loaded: a long MJPEG stream went black when it stalled
let busy=false;
function frame(){ if(busy) return; busy=true; const im=new Image();
  im.onload=()=>{document.getElementById('v').src=im.src; busy=false; setTimeout(frame,80)};
  im.onerror=()=>{busy=false; setTimeout(frame,500)};
  im.src='/view.jpg?t='+Date.now(); setTimeout(()=>{ if(busy){busy=false; frame()} }, 3000); }
frame();
async function tick(){try{const s=await (await fetch('/status')).json();
for(const m of ['view','calib','undist','detect','qr'])document.getElementById('m_'+m).className=s.mode==m?'on':'';
for(const v of [0,120,90,75,60])document.getElementById('f_'+v).className=s.fov==v?'on':'';
for(const v of [3,5,8,12])document.getElementById('e_'+v).className=s.exposure==v?'on':'';
document.getElementById('az').className=s.auto_zoom?'on':'';
const bgpu=document.getElementById('btn_gpu');if(bgpu){bgpu.className=s.use_gpu?'on':'';bgpu.textContent=s.use_gpu?'GPU: ON (Orin)':'GPU: OFF (CPU)';}
const benh=document.getElementById('btn_enhance');if(benh){benh.className=s.enhance?'on':'';benh.textContent=s.enhance?'Enhance: ON':'Enhance: OFF';}
const bhq=document.getElementById('btn_hq');if(bhq){bhq.className=s.hq?'on':'';bhq.textContent=s.hq?'HQ: 95%':'HQ: 70%';}
const qs=document.getElementById('qs');if(document.activeElement!==qs)qs.value=s.qr_size?s.qr_size:'';
document.getElementById('st').textContent=s.text}catch(e){} setTimeout(tick,500)} tick()
</script></body></html>"""


# the chase on a QR box that can't be read: (fov deg, scale x the lens's own detail, frames averaged) per step; the
# whole chase runs on the camera's SCAN settings (--scan-exposure-ms at --scan-fps: the camera is held still to read)
CHASE = [(30, 2.0, 6), (15, 3.0, 10)]


class Tool:
    def __init__(self, a):
        self.a = a
        self.cols, self.rows = (int(v) for v in a.board.lower().split('x'))
        self.out = Path(a.out)
        self.frames_dir = self.out.parent / 'arducam_calib_frames'
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.frame = None            # latest BGR frame
        self.frame_n, self.cam_open_t = 0, 0.0   # frames so far, when the camera was last (re)opened
        self.scan_ok = True              # the scan settings opened (off for the session once they gave no frame)
        self.qr_pairs = (None, [])       # (view cfg, [(id, id, distance in code widths)]) of the boxes of qr_seen
        self.size = None
        self.view = None             # latest annotated JPEG
        self.mode = 'view'
        self.caps = []               # [(corners (N,1,2) float32, descriptor)]
        self.prev_c, self.prev_t = None, 0.0   # the board's corners in the previous calib frame + time (stillness check)
        self.calib = self.load_calib()
        self.maps = None
        self.msg = ''
        self.qr_log = []             # (time, payloads)
        self.qr_n, self.qr_last, self.qr_last_t = 0, ('', ''), 0.0   # frames, (last payload, its clock time)
        self.qr_found, self.qr_found_t = {}, 0.0                     # latest decode: payload -> polygon
        self.fps = 0.0
        self.fov = a.qr_fov              # QR view: degrees of the straightened middle view, 0 = the whole fisheye
        self.rect = {}                   # (frame shape, direction, fov, scale) -> (remap maps, focal, R)
        self.auto_zoom = a.auto_zoom     # zoom the view onto a QR as soon as one is read
        self.zoom = None                 # {d: ray of the code (camera frame), t: last read}
        self.qr_cfg = None               # the view config of qr_found
        self.zstack = []                 # [(view cfg, zoomed grey view float32)] - the frames averaged
        self.qr_seen = (None, [], 0.0)   # (view cfg, detector boxes [{poly, kind, score}], time)
        self.load_frames()
        self.wc = None
        wd = Path(a.wechat_dir)
        try:
            if all((wd / n).exists() for n in ('detect.prototxt', 'detect.caffemodel', 'sr.prototxt', 'sr.caffemodel')):
                self.wc = cv2.wechat_qrcode_WeChatQRCode(str(wd / 'detect.prototxt'), str(wd / 'detect.caffemodel'),
                                                         str(wd / 'sr.prototxt'), str(wd / 'sr.caffemodel'))
        except (AttributeError, cv2.error) as e:
            print(f'WeChat QR decoder unavailable: {e}')
        print(f'WeChat QR decoder: {"on" if self.wc else "off (zbar only)"}')
        self.net = None                  # WeChat's QR DETECTION network alone (an SSD, 384 x 384 grey in, QR boxes out)
        try:
            if (wd / 'detect.caffemodel').exists():
                self.net = cv2.dnn.readNetFromCaffe(str(wd / 'detect.prototxt'), str(wd / 'detect.caffemodel'))
        except cv2.error as e:
            print(f'QR detection network unavailable: {e}')
        self.n_detect = 0
        self.last_full = ([], 0.0)       # (boxes of the last full network scan, time)
        from pyzbar.pyzbar import ZBarSymbol, decode
        self.zbar = lambda img: decode(img, symbols=[ZBarSymbol.QRCODE])
        self.cv_qr = cv2.QRCodeDetector()
        self.use_gpu = getattr(a, 'gpu', False)
        self.enhance = getattr(a, 'enhance', False)
        self.hq = getattr(a, 'hq', False)
        self.cuda_available = False
        try:
            import torch
            self.cuda_available = torch.cuda.is_available()
            if self.cuda_available:
                print('GPU Acceleration: PyTorch CUDA ready on Orin')
        except Exception:
            pass

    # ---- camera
    def cam_mode(self):
        """(width, height, max exposure ms): QR test = full resolution + short exposure (as the mission's QR scan,
        arducam_csi QR mode); calibration / view / undistort = --width x --height (the mode the calibration is for)."""
        a = self.a
        if self.mode == 'qr' and self.zoom is not None and self.scan_ok:
            # a code found but not read: the camera is held still, so trade fps for exposure - at 8 ms indoors the
            # gain was maxed and the grain hid 2.5 px/module codes (2026-10-26)
            return a.qr_width, a.qr_height, a.scan_exposure_ms, a.scan_fps
        if self.mode in ('qr', 'detect'):
            return a.qr_width, a.qr_height, a.qr_max_exposure_ms, 21
        return a.width, a.height, a.max_exposure_ms, 30

    def pipeline(self, w, h, exp, fps):
        a = self.a
        fps = min(fps, 21 if h > 2000 else 30)       # the sensor's full-resolution mode tops out at 21 fps
        return (f'nvarguscamerasrc sensor-id={a.sensor_id} exposuretimerange="13000 {int(exp * 1e6)}" '
                f'gainrange="1 10.625" ! video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,'
                f'format=NV12 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! '
                f'appsink drop=true max-buffers=1 sync=false')

    def capture_loop(self):
        os.environ.pop('DISPLAY', None)     # Argus' EGL stream fails with a DISPLAY it can't use
        while not STOP.is_set():
            cfg = self.cam_mode()
            cap = cv2.VideoCapture(self.pipeline(*cfg), cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                self.msg = 'camera did not open: is the ROS camera running (scripts/arducam_down.sh)?'
                time.sleep(3)
                continue
            CAM_RELEASED.clear()
            n, t0 = 0, time.monotonic()
            self.cam_open_t = t0
            got = False
            while self.cam_mode() == cfg and not STOP.is_set():   # a mode change (QR test on / off) reopens it
                ok, f = cap.read()
                if not ok:
                    if not got and cfg[3] == self.a.scan_fps and cfg[2] == self.a.scan_exposure_ms:
                        self.scan_ok = False
                        self.msg = 'the scan settings gave no frame: off for this session (normal QR settings)'
                    break
                got = True
                with self.lock:
                    self.frame, self.size = f, (f.shape[1], f.shape[0])
                    self.frame_n += 1
                n += 1
                if time.monotonic() - t0 > 2:
                    self.fps, n, t0 = n / (time.monotonic() - t0), 0, time.monotonic()
            cap.release()
            CAM_RELEASED.set()
            with self.lock:
                self.frame = None                         # no stale frame of the other size
            # Argus needs the old session gone before the next one: reopening after 0.5 s for the scan settings
            # wedged nvargus-daemon (InvalidState, then every open timed out until it was restarted, 2026-10-26)
            time.sleep(1.5)

    # ---- calibration
    def objp(self):
        o = np.zeros((1, self.cols * self.rows, 3), np.float64)
        o[0, :, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2) * self.a.square
        return o

    def find_board(self, gray):
        """Corners (N,1,2) float32 or None. SB on a half-size image (fast, robust to the fisheye's blur), refined
        on the full one."""
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        ok, c = cv2.findChessboardCornersSB(small, (self.cols, self.rows),
                                            cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
        if not ok:
            return None
        c = (c * 2.0).astype(np.float32)
        cv2.cornerSubPix(gray, c, (5, 5), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
        return c

    @staticmethod
    def describe(c, size):
        """(centre x, centre y, area, tilt) of the board in the image, normalised - to tell new views from old."""
        p = c.reshape(-1, 2)
        hull = cv2.convexHull(p.astype(np.float32))
        w, h = size
        (_, _), (bw, bh), _ = cv2.minAreaRect(hull)
        return np.array([p[:, 0].mean() / w, p[:, 1].mean() / h, math.sqrt(cv2.contourArea(hull) / (w * h)),
                         min(bw, bh) / max(bw, bh, 1e-6)])

    def is_new(self, d):
        return all(np.abs(d - e).max() > 0.06 for _, e in self.caps)

    def add_capture(self, c, frame, d):
        self.caps.append((c, d))
        k = len(self.caps)
        cv2.imwrite(str(self.frames_dir / f'cap_{k:03d}.png'), frame)

    def load_frames(self):
        """Re-detect the saved captures (after a restart / --recompute)."""
        for f in sorted(self.frames_dir.glob('cap_*.png')):
            img = cv2.imread(str(f))
            if img is None:
                continue
            c = self.find_board(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            if c is not None:
                self.caps.append((c, self.describe(c, (img.shape[1], img.shape[0]))))
        if self.caps:
            print(f'{len(self.caps)} saved captures loaded from {self.frames_dir}')

    def calibrate(self):
        if len(self.caps) < 10:
            return False, f'only {len(self.caps)} captures: take at least 10 (20-40 is better), edges included'
        w, h = self.a.width, self.a.height                # the captures' size (taken in calib mode), not the QR mode's
        objp = self.objp()
        keep = list(range(len(self.caps)))
        flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv2.fisheye.CALIB_CHECK_COND | cv2.fisheye.CALIB_FIX_SKEW
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-7)
        dropped = []
        while len(keep) >= 8:
            K, D = np.zeros((3, 3)), np.zeros((4, 1))
            try:
                rms, K, D, _, _ = cv2.fisheye.calibrate([objp] * len(keep), [self.caps[i][0].reshape(1, -1, 2).astype(np.float64)
                                                                             for i in keep],
                                                        (w, h), K, D, flags=flags, criteria=crit)
                break
            except cv2.error as e:
                # CHECK_COND names the ill-conditioned view: "... input array N" - drop it and fit again
                import re
                m = re.search(r'input array (\d+)', str(e))
                if not m:
                    return False, f'calibration failed: {str(e).splitlines()[-1]}'
                dropped.append(keep.pop(int(m.group(1))) + 1)
        else:
            return False, 'calibration failed: too many ill-conditioned views - recapture'
        hfov = 2 * self.max_angle(K, D, w / 2) if K is not None else 0
        data = {'image_width': w, 'image_height': h, 'camera_name': 'arducam_imx219_fisheye',
                'distortion_model': 'equidistant',
                'camera_matrix': {'rows': 3, 'cols': 3, 'data': [float(v) for v in K.ravel()]},
                'distortion_coefficients': {'rows': 1, 'cols': 4, 'data': [float(v) for v in D.ravel()]},
                'rms_px': round(float(rms), 4), 'views': len(keep), 'dropped_views': dropped,
                'board': f'{self.cols}x{self.rows}', 'date': time.strftime('%Y-%m-%d %H:%M')}
        import yaml
        self.out.write_text(yaml.safe_dump(data, sort_keys=False))
        self.calib, self.maps = data, None
        return True, (f'saved {self.out}\nRMS {rms:.3f} px over {len(keep)} views (dropped {dropped or "none"})\n'
                      f'fx {K[0, 0]:.1f} fy {K[1, 1]:.1f} cx {K[0, 2]:.1f} cy {K[1, 2]:.1f}  D {np.round(D.ravel(), 4)}\n'
                      f'model angle at the image side edges ~{math.degrees(hfov) / 2:.0f} deg off axis (a circular fisheye: past the image circle)' +
                      ('\nRMS > 1 px: recapture with sharper, more varied views' if rms > 1.0 else ''))

    @staticmethod
    def max_angle(K, D, r_px):
        """Angle (rad) off the axis of a pixel r_px from the centre (inverse of theta_d = theta(1 + k1 t^2 ...))."""
        k = np.asarray(D).ravel()
        td = r_px / K[0, 0]
        t = td
        for _ in range(20):
            t2 = t * t
            f = t * (1 + k[0] * t2 + k[1] * t2 ** 2 + k[2] * t2 ** 3 + k[3] * t2 ** 4) - td
            t -= f / (1 + 3 * k[0] * t2 + 5 * k[1] * t2 ** 2 + 7 * k[2] * t2 ** 3 + 9 * k[3] * t2 ** 4)
        return float(t)

    def load_calib(self):
        if not self.out.exists():
            return None
        import yaml
        try:
            return yaml.safe_load(self.out.read_text())
        except Exception:
            return None

    def KD(self):
        c = self.calib
        return (np.array(c['camera_matrix']['data'], float).reshape(3, 3),
                np.array(c['distortion_coefficients']['data'], float).reshape(4, 1))

    # ---- straightened (pinhole) view of the middle of the lens
    def lens(self, w, h):
        """(K, D) of the fisheye at a w x h frame: the calibration scaled, else the estimate from the image circle
        (a 175 deg equidistant lens whose circle is ~0.975 of the frame height: f = 0.319 h)."""
        if self.calib:
            K, D = self.KD()
            K = K.copy()
            K[:2] *= w / self.calib['image_width']
            return K, D
        f = 0.319 * h
        return np.array([[f, 0, (w - 1) / 2], [0, f, (h - 1) / 2], [0, 0, 1]]), np.zeros((4, 1))

    def view_cfg(self):
        """(direction or None = the optical axis, fov deg (0 = the raw fisheye), scale): the chase's view of a QR box
        (its current step, following the box) or the wide view."""
        z = self.zoom
        if z is not None:
            fov, scale, _ = CHASE[min(z['level'], len(CHASE) - 1)]
            return tuple(np.round(z['d'], 3)), fov, scale
        return None, self.fov, 1.0

    @staticmethod
    def rot_to(d):
        """R (camera -> view) of a view looking along the unit ray d (camera frame), image up ~ the camera's."""
        z = np.asarray(d, float) / np.linalg.norm(d)
        x = np.cross([0.0, 1.0, 0.0], z)
        x = x / np.linalg.norm(x) if np.linalg.norm(x) > 1e-6 else np.array([1.0, 0.0, 0.0])
        y = np.cross(z, x)
        return np.vstack([x, y, z])

    def rectify(self, img, d=None, fov=None, scale=1.0):
        """(view, focal px of the view, R camera -> view): a pinhole view along d (default the optical axis), fov wide,
        at scale x the lens's own detail in the middle; the image itself (None, None) when fov is 0."""
        fov = self.fov if fov is None else fov
        if not fov:
            return img, None, None
        h, w = img.shape[:2]
        key = (h, w, d, fov, scale, bool(self.calib))
        if key not in self.rect:
            K, D = self.lens(w, h)
            f = float(K[0, 0]) * scale
            n = int(min(2400, 2 * f * math.tan(math.radians(fov) / 2)))
            newK = np.array([[f, 0, n / 2], [0, f, n / 2], [0, 0, 1]])
            R = np.eye(3) if d is None else self.rot_to(d)
            maps = cv2.fisheye.initUndistortRectifyMap(K, D, R, newK, (n, n), cv2.CV_16SC2)
            if len(self.rect) > 6:
                self.rect.clear()
            self.rect[key] = (maps, f, R)
        maps, f, R = self.rect[key]
        return cv2.remap(img, *maps, interpolation=cv2.INTER_LINEAR), f, R

    def ray_of(self, c, shape, fv, R, frame_shape):
        """Unit ray (camera frame) of view pixel c: through the view's pinhole, or the fisheye model (fov 0)."""
        if fv:
            v = np.array([(c[0] - shape[1] / 2) / fv, (c[1] - shape[0] / 2) / fv, 1.0])
            v = R.T @ v
            return v / np.linalg.norm(v)
        K, D = self.lens(frame_shape[1], frame_shape[0])
        dx, dy = c[0] - K[0, 2], c[1] - K[1, 2]
        th, ph = self.max_angle(K, D, math.hypot(dx, dy)), math.atan2(dy, dx)
        return np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])

    # ---- QR
    def wechat(self, gray, tile=768, overlap=160):
        """WeChat over tiles: its CNN detector shrinks its input to 384 px, so a 75 px code in a 3280 px frame is lost
        whole (2026-10-26 benchmark: 768 px tiles found 9 of 12 codes at 2.5-3.5 px/module)."""
        out = {}
        h, w = gray.shape[:2]
        step = tile - overlap
        for y in range(0, max(1, h - overlap), step):
            for x in range(0, max(1, w - overlap), step):
                x0, y0 = max(0, min(x, w - tile)), max(0, min(y, h - tile))
                texts, pts = self.wc.detectAndDecode(gray[y0:y0 + tile, x0:x0 + tile])
                for t, p in zip(texts, pts):
                    if t and t.strip() not in out:
                        out[t.strip()] = np.asarray(p, float).reshape(-1, 2) + [x0, y0]
        return out

    def seen_qrs(self, gray):
        """QR codes FOUND but not necessarily readable: OpenCV's detector (the three finder squares), no decoding."""
        try:
            ok, pts = self.cv_qr.detectMulti(gray)
        except cv2.error:
            return []
        return [np.asarray(p, float).reshape(-1, 2) for p in pts] if ok and pts is not None else []

    def contrast_blobs(self, gray):
        """Candidate boards, as the mission's qr_locator.detect_candidates (local contrast: grey-level std over
        cand_win px at half resolution > cand_std -> blobs), in image space instead of on the ground: kept if
        square-ish (aspect 0.6-1.6), 30-900 px across and densely textured (a QR is contrast all over) - largest 6."""
        a = self.a
        g = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA).astype(np.float32)
        g = cv2.GaussianBlur(g, (0, 0), 1.2)              # sensor grain at high gain is contrast everywhere
        k = a.cand_win
        mu = cv2.boxFilter(g, -1, (k, k))
        sd = np.sqrt(np.maximum(cv2.boxFilter(g * g, -1, (k, k)) - mu * mu, 0))
        # threshold from this frame's own contrast (median + 4 robust sigma): a fixed 45 caught the grain, not the
        # boards (2026-10-26); --cand-std is the floor
        med = float(np.median(sd))
        thr = max(a.cand_std_min, med + 4.0 * 1.4826 * (float(np.median(np.abs(sd - med))) + 1e-3))
        # several thresholds: at the first one the clutter of a room merges into one huge blob that swallows the
        # board (2026-10-26); a QR's black/white contrast stays a compact blob at the higher ones
        out = []
        for level in (1.0, 1.6, 2.5, 4.0):
            mask = (sd > thr * level).astype(np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
            for i in range(1, n):
                bx, by, bw, bh, area = stats[i]
                side = max(bw, bh) * 2
                # aspect 0.4-2.5: a board seen at an angle / curled is no square in the image (2026-10-26)
                if not 36 <= side <= 900 or not 0.4 <= bw / max(bh, 1) <= 2.5 or area < 0.4 * bw * bh:
                    continue
                c = np.array([bx + bw / 2, by + bh / 2]) * 2
                if any(np.linalg.norm(c - q.mean(axis=0)) < 0.5 * max(bw, bh) * 2 for _, q in out):
                    continue                          # the same blob at another threshold
                out.append((area * level, np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]],
                                                   float) * 2))
        return [p for _, p in sorted(out, key=lambda t: -t[0])[:10]]

    def net_boxes(self, crop, x0, y0, conf):
        """The detection network on one crop (resized to its 384 x 384 input; pixels 0-1: fed 0-255 it returned junk
        boxes at 0.3 confidence, 2026-10-26) -> [(conf, poly)] in the full image."""
        if self.use_gpu and self.cuda_available:
            try:
                import torch
                t = torch.from_numpy(crop).cuda().float()
                if t.shape != (384, 384):
                    t = torch.nn.functional.interpolate(
                        t.unsqueeze(0).unsqueeze(0), size=(384, 384), mode='bilinear', align_corners=False
                    ).squeeze()
                blob = (t * (1.0 / 255.0)).cpu().numpy().reshape(1, 1, 384, 384)
            except Exception:
                blob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (384, 384), 0, False, False)
        else:
            blob = cv2.dnn.blobFromImage(crop, 1.0 / 255, (384, 384), 0, False, False)
        self.net.setInput(blob)
        out = []
        h, w = crop.shape[:2]
        for d in self.net.forward().reshape(-1, 7):
            if d[2] >= conf:
                x1, y1, x2, y2 = d[3] * w + x0, d[4] * h + y0, d[5] * w + x0, d[6] * h + y0
                out.append((float(d[2]), np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)))
        return out

    def net_scan(self, gray, conf, centres=None, tile=384, overlap=96):
        """Tiles of 384 px at full detail (a code must be >= ~40 px at the network's input: a 60 px code in a 768 px
        tile, halved, fell to 0.45), either all over (~0.9 s for a 1572 px view) or one tile round each centre."""
        h, w = gray.shape[:2]
        if centres is None:
            step = tile - overlap
            spots = [(max(0, min(x, w - tile)), max(0, min(y, h - tile)))
                     for y in range(0, max(1, h - overlap), step) for x in range(0, max(1, w - overlap), step)]
        else:
            spots = [(int(max(0, min(cx - tile / 2, w - tile))), int(max(0, min(cy - tile / 2, h - tile))))
                     for cx, cy in centres]
        out = []
        for x0, y0 in dict.fromkeys(spots):
            out += self.net_boxes(gray[y0:y0 + tile, x0:x0 + tile], x0, y0, conf)
        keep = []                                     # overlapping boxes of one code (tiles overlap): the best one
        for c, p in sorted(out, key=lambda t: -t[0]):
            if all(self.iou(p, q) < 0.3 for _, q in keep):
                keep.append((c, p))
        return keep

    @staticmethod
    def iou(a, b):
        ax0, ay0 = a.min(axis=0)
        ax1, ay1 = a.max(axis=0)
        bx0, by0 = b.min(axis=0)
        bx1, by1 = b.max(axis=0)
        iw, ih = max(0.0, min(ax1, bx1) - max(ax0, bx0)), max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = iw * ih
        return inter / max(1e-6, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)

    def detect_boxes(self, gray):
        """Bounding boxes of QR codes, read or not: [{poly, kind, score}], best first.
          model     WeChat's trained QR detection network (detect.caffemodel), on 384 px crops round the contrast
                    candidates every frame and over the whole view every --scan-every frames (score = confidence;
                    the board held 3-4 m away: 1.00, the scenes without a code: nothing, 2026-10-26)
          finder    OpenCV's QR detector found the three corner squares
          contrast  the mission's candidate detector (qr_locator.detect_candidates in image space), scored - only
                    where the network sees nothing (or without the network)"""
        cands = self.contrast_blobs(gray)
        out = []
        if self.net is not None:
            self.n_detect += 1
            full = self.n_detect % self.a.scan_every == 0
            for c, p in self.net_scan(gray, self.a.net_conf, None if full else [q.mean(axis=0) for q in cands]):
                out.append({'poly': p, 'kind': 'model', 'score': round(c, 2)})
            if full:
                self.last_full = ([b for b in out], time.monotonic())
            elif time.monotonic() - self.last_full[1] < 1.5:   # what the last full scan found, while it is fresh
                out += [b for b in self.last_full[0] if all(self.iou(b['poly'], o['poly']) < 0.3 for o in out)]
        for p in self.seen_qrs(gray):
            if all(self.iou(p, b['poly']) < 0.3 for b in out):
                out.append({'poly': p, 'kind': 'finder', 'score': 1.0})
        for b in self.contrast_scored(gray, cands):
            if all(self.iou(b['poly'], o['poly']) < 0.1 for o in out):
                out.append(b)
        return sorted(out, key=lambda b: (b['kind'] == 'contrast', -b['score']))

    def contrast_scored(self, gray, cands):
        """Boxes that look like a QR, no reading: [{poly, kind, score}], best first.
          finder    OpenCV's QR detector found the three corner squares (score 1.0)
          contrast  the mission's candidate detector (contrast_blobs), each box scored 0-1 on how QR-like it is:
                    squareness, how densely it is textured, and how two-tone it is (Otsu between-class share of the
                    grey variance: a QR is black and white, a shelf or a face is not); the crop, enlarged 2x, is also
                    searched for corner squares (-> finder). Shown from min_score."""
        """The contrast candidates scored 0-1 on how QR-like they are: squareness, texture density, two-tone (Otsu
        between-class share of the grey variance) and black/white changes per row and column."""
        out = []
        for p in cands:
            x0, y0 = np.maximum(p.min(axis=0).astype(int) - 8, 0)
            x1, y1 = p.max(axis=0).astype(int) + 8
            crop = gray[y0:y1, x0:x1]
            if crop.size < 400 or any(np.linalg.norm(p.mean(axis=0) - b['poly'].mean(axis=0)) < 40 for b in out):
                continue
            h, w = crop.shape
            square = min(w, h) / max(w, h)
            sd = cv2.boxFilter(crop.astype(np.float32) ** 2, -1, (5, 5)) - cv2.boxFilter(crop.astype(np.float32), -1, (5, 5)) ** 2
            texture = float((sd > 150).mean())
            t, _ = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            lo, hi = crop[crop <= t], crop[crop > t]
            var = float(crop.var()) + 1e-6
            two_tone = (lo.size * hi.size / crop.size ** 2 * (float(hi.mean()) - float(lo.mean())) ** 2 / var
                        if lo.size and hi.size else 0.0)
            # black/white changes per row and per column: a QR has many (21+ modules), an edge one - a single
            # straight edge scored 0.76 on two-tone alone (2026-10-26)
            bw_ = (crop > t).astype(np.int8)
            tr_rows = float(np.abs(np.diff(bw_, axis=1)).sum(axis=1).mean())
            tr_cols = float(np.abs(np.diff(bw_, axis=0)).sum(axis=0).mean())
            if min(tr_rows, tr_cols) < self.a.min_transitions:
                continue
            trans = min(1.0, min(tr_rows, tr_cols) / 10.0)
            score = round(0.2 * square + 0.2 * min(1.0, texture / 0.6) + 0.3 * min(1.0, two_tone) + 0.3 * trans, 2)
            if score >= self.a.min_score:
                out.append({'poly': p, 'kind': 'contrast', 'score': score})
        return sorted(out, key=lambda b: -b['score'])

    def measure(self, view, boxes, fv, R, frame_shape):
        """Number the boxes left to right (b['id']) and place each code with the calibrated lens: its dark square
        (Otsu inside the box) -> its angular width -> range in code widths; with 3+ codes all are put on one fitted
        plane (a wall / the floor: the ranges alone are ~5 % noisy). -> [(id, id, centre distance in code widths)]"""
        if not boxes:
            return []
        X = []
        for k, b in enumerate(sorted(boxes, key=lambda b: b['poly'][:, 0].mean())):
            b['id'] = k + 1
            p = b['poly']
            m = 0.15 * np.ptp(p, axis=0)
            x0, y0 = np.maximum(p.min(axis=0) - m, 0).astype(int)
            x1, y1 = np.minimum(p.max(axis=0) + m, view.shape[::-1]).astype(int)
            sq = p
            crop = view[y0:y1, x0:x1]
            if crop.size > 400:
                g = cv2.GaussianBlur(crop, (0, 0), 2)
                _, mk = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
                cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                ctr = ((x1 - x0) / 2, (y1 - y0) / 2)
                cs = [c for c in cs if cv2.pointPolygonTest(c, ctr, False) >= 0]
                if cs:
                    rect = cv2.minAreaRect(cs[0])
                    if rect[1][0] * rect[1][1] < 0.8 * crop.shape[0] * crop.shape[1]:   # not the whole background
                        sq = cv2.boxPoints(rect).astype(float) + [x0, y0]
            b['square'] = sq
            r = [self.ray_of(q, view.shape, fv, R, frame_shape) for q in sq]
            ang = lambda u, v: math.acos(max(-1.0, min(1.0, float(u @ v))))
            # the less foreshortened of the two directions = the code's true width
            w = max(ang(r[0], r[1]) + ang(r[2], r[3]), ang(r[1], r[2]) + ang(r[3], r[0])) / 2
            c = self.ray_of(sq.mean(axis=0), view.shape, fv, R, frame_shape)
            X.append(c / (2 * math.tan(max(w, 1e-4) / 2)))
        X = np.array(X)
        if len(X) >= 3:
            cen = X.mean(axis=0)
            nrm = np.linalg.svd(X - cen)[2][2]
            u = X / np.linalg.norm(X, axis=1, keepdims=True)
            if np.all(np.abs(u @ nrm) > 0.2):         # the plane faces the camera: every ray meets it
                X = u * ((cen @ nrm) / (u @ nrm))[:, None]
        # one frame's distances jump ~10 % (a 65 px square's edges in sensor grain): the median of the last 15 frames
        # with the same number of codes (ids are left-to-right, so stable while the camera is)
        hist = getattr(self, 'pair_hist', [])
        hist = (hist if hist and len(hist[-1]) == len(X) else []) + [X]
        self.pair_hist = hist[-15:]
        D = np.median([[np.linalg.norm(Y[i] - Y[j]) for i in range(len(Y)) for j in range(i + 1, len(Y))]
                       for Y in self.pair_hist], axis=0)
        ij = [(i + 1, j + 1) for i in range(len(X)) for j in range(i + 1, len(X))]
        return [(i, j, float(d)) for (i, j), d in zip(ij, D)]

    def dist_text(self, d):
        return f'{d * self.a.qr_size:.2f} m' if self.a.qr_size > 0 else f'{d:.2f} w'

    def decode(self, gray, slow=True):
        """slow=False: native size only (the live test: 2x upscaling and OpenCV's detector every third frame)."""
        mean = float(gray.mean())
        if mean < 80 or mean > 200:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        out = self.wechat(gray) if self.wc is not None else {}
        if out or (self.wc is not None and not slow):
            return out
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        # 2x upscaling only below full resolution (a 6560 px image takes seconds)
        for scale in ((1.0, 2.0) if slow and gray.shape[1] <= 2000 else (1.0,)):
            img = gray if scale == 1.0 else cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            for d in self.zbar(img):
                p = d.data.decode('utf-8', errors='replace').strip()
                pts = [(pt.x, pt.y) for pt in d.polygon]
                if p and p not in out and pts:
                    out[p] = np.array(pts, float) / scale
            if out:
                return out
        if not slow:
            return out
        try:
            ok, pl, pts, _ = self.cv_qr.detectAndDecodeMulti(gray)
            if ok and pts is not None:
                for p, poly in zip(pl, pts):
                    if p:
                        out[p.strip()] = np.asarray(poly, float).reshape(-1, 2)
        except cv2.error:
            pass
        return out

    # ---- worker: annotate the latest frame for the stream
    def work_loop(self):
        while True:
            with self.lock:
                f = None if self.frame is None else self.frame.copy()
            if f is None:
                time.sleep(0.2)
                continue
            vis = f
            if self.mode in ('calib', 'capture'):
                gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                c = self.find_board(gray)
                vis = f.copy()
                for cc, _ in self.caps:                   # coverage so far: every captured board's outline
                    hull = cv2.convexHull(cc.reshape(-1, 2).astype(np.float32)).astype(np.int32)
                    cv2.polylines(vis, [hull], True, (255, 160, 0), 2)
                # moving board = motion blur + rolling shutter skew (~50 ms readout): auto-capture only below
                # --still-px-s (RMS 4.5 px from captures taken on the move, 2026-10-26); a speed, so a hand-held camera's
                # slow drift passes
                now = time.monotonic()
                moved = (np.abs(c - self.prev_c).max() * 1640 / f.shape[1] / max(now - self.prev_t, 1e-3)
                         if c is not None and self.prev_c is not None else None)
                self.prev_c, self.prev_t = c, now
                if c is not None:
                    cv2.drawChessboardCorners(vis, (self.cols, self.rows), c, True)
                    d = self.describe(c, self.size)
                    still = moved is not None and moved < self.a.still_px_s
                    if self.mode == 'calib' and not still and self.is_new(d):
                        self.msg = (f'new view - steady it: moving {moved:.0f} px/s, captures below {self.a.still_px_s:g}'
                                    if moved is not None else 'new view - steady it')
                    if self.mode == 'capture' or (still and self.is_new(d)):
                        self.msg = ''
                        self.add_capture(c, f, d)
                        self.mode = 'calib'
                        cv2.rectangle(vis, (0, 0), (vis.shape[1] - 1, vis.shape[0] - 1), (0, 255, 0), 16)
                elif self.mode == 'capture':
                    self.msg = 'capture: no board in view'
                    self.mode = 'calib'
            elif self.mode == 'undist' and self.calib:
                # the FOV buttons' straightened middle (120 deg for "fisheye"): a flat view of the whole 180+ deg
                # lens smears everything past ~60 deg off axis and looks broken even with a good calibration
                vis, _, _ = self.rectify(f, None, self.fov or 120, 1.0)
            elif self.mode == 'detect':
                cfg = (None, self.fov, 1.0)
                vis, fv, _ = self.rectify(f, *cfg)
                vis = vis.copy()
                self.draw_boxes(vis, cfg, fv, {})
            elif self.mode == 'qr':
                # the decode runs in qr_loop at its own pace; every streamed frame gets its latest boxes (< 1 s old)
                cfg = self.view_cfg()
                vis, fv, R = self.rectify(f, *cfg)
                vis = vis.copy()
                # boxes only from a decode of this same view (a zoom change moves everything)
                found = self.qr_found if time.monotonic() - self.qr_found_t < 1.0 and self.qr_cfg == cfg else {}
                if cfg[0] is not None:
                    z = self.zoom
                    step = '' if z is None else (f'step {z["level"] + 1}/{len(CHASE)}  SCAN: exposure <= '
                                                 f'{self.a.scan_exposure_ms:g} ms, {self.a.scan_fps} fps, '
                                                 f'{len(self.zstack)} frames averaged - hold still')
                    cv2.putText(vis, f'CHASING the QR: {cfg[1]} deg x{cfg[2]:g}  {step}', (12, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 80), 2)
                self.draw_boxes(vis, cfg, fv, found)
                for p, poly in found.items():
                    pts = poly.astype(np.int32)
                    cv2.polylines(vis, [pts], True, (0, 255, 255), 4)
                    c = poly.mean(axis=0)
                    ray = self.ray_of(c, vis.shape, fv, R, f.shape)
                    ang = f'  {math.degrees(math.acos(max(-1.0, min(1.0, ray[2])))):.0f} deg off axis'
                    side = min(np.linalg.norm(poly[i] - poly[(i + 1) % 4]) for i in range(4)) / (cfg[2] if fv else 1)
                    # px per module of the SENSOR (the zoom's enlargement taken out), 21-module (version 1) code
                    cv2.putText(vis, f'{p[:48]}  {side:.0f}px = {side / 21:.1f} px/module (v1){ang}', (int(pts[:, 0].min()), int(pts[:, 1].min()) - 12),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)
            if self.enhance:
                try:
                    lab = cv2.cvtColor(vis, cv2.COLOR_BGR2LAB)
                    l, a, b = cv2.split(lab)
                    cl = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(l)
                    blur = cv2.GaussianBlur(cl, (0, 0), 1.2)
                    cl = cv2.addWeighted(cl, 1.3, blur, -0.3, 0)
                    vis = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)
                except Exception:
                    pass
            # the view at up to 1600 px wide (max resolution for now, user 2026-10-26)
            k = min(1.0, 1600.0 / vis.shape[1])
            small = cv2.resize(vis, None, fx=k, fy=k, interpolation=cv2.INTER_AREA) if k < 1 else vis
            q = 95 if (self.hq or self.enhance) else 70
            ok, jpg = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, q])
            if ok:
                self.view = jpg.tobytes()
            time.sleep(0.08)

    def draw_boxes(self, vis, cfg, fv, found):
        """The detector's boxes (from a pass over this same view, < 1.5 s old) unless the code was read."""
        scfg, boxes, st = self.qr_seen
        if scfg != cfg or time.monotonic() - st > 1.5 or found:
            return
        pcfg, pairs = self.qr_pairs
        if pcfg == cfg:
            ctr = {b['id']: b.get('square', b['poly']).mean(axis=0) for b in boxes if 'id' in b}
            # lines only between neighbours (the minimum spanning tree), all pairs are listed in the status
            tree, done = [], {1}
            while len(done) < len(ctr):
                e = min((e for e in pairs if (e[0] in done) != (e[1] in done)), key=lambda e: e[2], default=None)
                if e is None:
                    break
                tree.append(e)
                done |= {e[0], e[1]}
            for i, j, d in tree:
                if i in ctr and j in ctr:
                    a, c = ctr[i].astype(int), ctr[j].astype(int)
                    cv2.line(vis, tuple(a), tuple(c), (255, 220, 0), 2)
                    m = (a + c) // 2
                    for col, th in (((0, 0, 0), 6), ((255, 220, 0), 2)):
                        cv2.putText(vis, self.dist_text(d), (int(m[0]) - 40, int(m[1]) - 8), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9, col, th)
        for b in boxes:
            pts = b['poly'].astype(np.int32)
            side = min(np.linalg.norm(b['poly'][i] - b['poly'][(i + 1) % 4]) for i in range(4)) / (cfg[2] if fv else 1)
            if b['kind'] == 'model':
                col, label = (60, 255, 60), f'QR {b["score"]:.2f}'
            elif b['kind'] == 'finder':
                col, label = (0, 140, 255), 'QR corners'
            else:
                col, label = (255, 64, 255), f'QR? {b["score"]:.2f}'
            cv2.polylines(vis, [pts], True, col, 3 if b['kind'] == 'contrast' else 4)
            cv2.putText(vis, f'#{b.get("id", "?")} {label} {side:.0f}px', (int(pts[:, 0].min()),
                        max(24, int(pts[:, 1].min()) - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)

    def qr_loop(self):
        """Decode the newest frame whenever the last decode is done (QR mode only) - the stream doesn't wait for it."""
        while True:
            if self.mode not in ('qr', 'detect') or self.frame is None:
                time.sleep(0.1)
                continue
            with self.lock:
                f, fn = self.frame, self.frame_n
            if fn == getattr(self, 'qr_fn', -1):
                time.sleep(0.02)                      # the same frame again (10 fps scan): don't average it twice
                continue
            self.qr_fn = fn
            self.qr_n += 1
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            if self.enhance:
                try:
                    cl = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
                    blur = cv2.GaussianBlur(cl, (0, 0), 1.2)
                    gray = cv2.addWeighted(cl, 1.35, blur, -0.35, 0)
                except Exception:
                    pass
            if self.mode == 'detect':                 # boxes only, the wide view, no reading, no chase
                self.zoom = None
                cfg = (None, self.fov, 1.0)
                view, fv, R = self.rectify(gray, *cfg)
                boxes = self.detect_boxes(view)
                self.qr_pairs = (cfg, self.measure(view, boxes, fv, R, f.shape))
                self.qr_seen = (cfg, boxes, time.monotonic())
                continue
            cfg = self.view_cfg()
            view, fv, R = self.rectify(gray, *cfg)
            if cfg[0] is not None:
                # zoomed: average the last views of the code (the step's count) (sensor grain at high gain / short exposure
                # was what lost a 3 px/module code, 2026-10-26), denoise the small view, decode that, then the
                # single frame
                if time.monotonic() - self.cam_open_t < 0.8:
                    continue                          # the scan settings' first frames: exposure still settling
                n_avg = CHASE[min(self.zoom['level'], len(CHASE) - 1)][2]
                v32 = view.astype(np.float32)
                if self.zstack and self.zstack[0][0] == cfg and self.zstack[0][1].shape == view.shape:
                    # a hand-held camera drifts a few px between frames: shift each onto the first before averaging
                    (dx, dy), _ = cv2.phaseCorrelate(self.zstack[0][1], v32)
                    if abs(dx) < view.shape[1] / 8 and abs(dy) < view.shape[0] / 8:
                        v32 = cv2.warpAffine(v32, np.float32([[1, 0, -dx], [0, 1, -dy]]), view.shape[::-1],
                                             borderMode=cv2.BORDER_REPLICATE)
                    self.zstack = (self.zstack + [(cfg, v32)])[-n_avg:]
                else:
                    self.zstack = [(cfg, v32)]
                avg = np.clip(sum(v for _, v in self.zstack) / len(self.zstack), 0, 255).astype(np.uint8)
                clean = cv2.fastNlMeansDenoising(avg, None, self.a.denoise, 5, 7) if self.a.denoise > 0 else avg   # ~100 ms
                found = self.decode(clean, slow=True) or self.decode(view, slow=self.qr_n % 3 == 0)
                if not found:
                    # the zoom missed: try the wide view in the same pass instead of losing the code
                    wcfg = (None, self.fov, 1.0)
                    wview, wfv, wR = self.rectify(gray, *wcfg)
                    wfound = self.decode(wview, slow=False)
                    if wfound:
                        cfg, view, fv, R, found = wcfg, wview, wfv, wR, wfound
            else:
                self.zstack = []
                found = self.decode(view, slow=self.qr_n % 3 == 0)
            now = time.monotonic()
            boxes = [] if found else self.detect_boxes(view)        # in frame but not read
            if cfg[0] is None:
                self.qr_pairs = (cfg, self.measure(view, boxes, fv, R, f.shape))
            self.qr_seen = (cfg, boxes, now)
            strong = [b for b in boxes if b['kind'] != 'contrast' or b['score'] >= 0.75]
            target = next(iter(found.values())) if found else (strong[0]['poly'] if strong else None)
            z = self.zoom
            if target is not None and self.auto_zoom:
                d = self.ray_of(target.mean(axis=0), view.shape, fv, R, f.shape)
                if z is None:
                    self.zoom = {'d': d, 't': now, 'level': 0, 't_level': now}             # start the chase
                else:
                    z['d'], z['t'] = d, now                                                # follow it
                    if (not found and now - max(z['t_level'], self.cam_open_t + 0.8) > self.a.chase_step
                            and z['level'] < len(CHASE) - 1):
                        z['level'], z['t_level'] = z['level'] + 1, now                     # still unread: step up
                        self.zstack = []
                    elif found:
                        z['t_level'] = now                                                 # read: stay at this step
            elif z is not None and now - max(z['t'], self.cam_open_t + 0.8) > self.a.zoom_hold:   # (not the reopen's gap)
                self.zoom = None                                                           # gone: back to wide
            for p in found:
                if p != self.qr_last[0] or now - self.qr_last_t > 5.0:
                    with open(self.out.parent / 'qr_reads.log', 'a') as log:
                        log.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")}  {p}\n')
                self.qr_last, self.qr_last_t = (p, time.strftime('%H:%M:%S')), now
            self.qr_log = [e for e in self.qr_log if now - e[0] < 5.0] + [(now, list(found))]
            if found:
                self.qr_found, self.qr_found_t, self.qr_cfg = found, now, cfg

    def status(self):
        lines = [f'mode {self.mode}   camera {self.fps:.0f} fps {self.size}   captures {len(self.caps)}   '
                 f'QR view {str(self.fov) + " deg straightened" if self.fov else "fisheye"}'
                 f'{" -> CHASING a QR (scan settings), step " + str(self.zoom["level"] + 1) if self.zoom else ""}'
                 f'   auto-zoom {"on" if self.auto_zoom else "off"}   QR exposure <= {self.a.qr_max_exposure_ms:g} ms'
                 f'{"" if self.calib else " (estimated lens: calibrate for a true one)"}   '
                 f'decoder {"WeChat + zbar" if self.wc else "zbar"}   '
                 f'GPU {"ON (Orin CUDA)" if self.use_gpu else "OFF (CPU)"}   '
                 f'Enhance {"ON" if self.enhance else "OFF"}   HQ {"95%" if (self.hq or self.enhance) else "70%"}']
        if self.calib:
            c = self.calib
            lines.append(f'calibration {self.out.name}: RMS {c.get("rms_px")} px, {c.get("views")} views, '
                         f'{c.get("date", "")}')
        else:
            lines.append('no calibration yet')
        if self.mode in ('calib', 'capture'):
            lines.append('hold the board still; captures happen by themselves for each NEW view once it stops moving (orange = taken). '
                         'Cover the corners and edges of the image, near and far, tilted.')
        if self.mode == 'qr':
            n = len(self.qr_log)
            hits = sum(1 for _, p in self.qr_log if p)
            seen = sorted({p for _, ps in self.qr_log for p in ps})
            lines.append(f'QR: decoded in {hits}/{n} frames (last 5 s, {n / 5:.1f} decodes/s): {", ".join(seen) or "-"}')
        if self.mode in ('qr', 'detect'):
            _, boxes, st = self.qr_seen
            if time.monotonic() - st < 2.0:
                nm = [b for b in boxes if b['kind'] == 'model']
                nf = sum(b['kind'] == 'finder' for b in boxes)
                nc = sum(b['kind'] == 'contrast' for b in boxes)
                pcfg, pairs = self.qr_pairs
                if pairs and pcfg == self.qr_seen[0]:
                    lines.append('distances (centre to centre' + ('' if self.a.qr_size > 0 else
                                 ', w = code widths: type the QR size for metres') + '):  ' +
                                 '   '.join(f'#{i}-#{j} {self.dist_text(d)}' for i, j, d in pairs))
                lines.append(f'detector: {len(nm)} QR(s) by the model (green{", best " + str(nm[0]["score"]) if nm else ""}), '
                             f'{nf} by their corner squares (orange), {nc} contrast guess(es) (magenta)'
                             + ('' if self.mode == 'detect' else ' - not read'))
            if self.qr_last[0]:
                lines.append(f'last read {self.qr_last[1]}:  {self.qr_last[0]}   (all reads: calibration/qr_reads.log; '
                             f'payloads are shown, never opened)')
        if self.msg:
            lines.append(self.msg)
        return '\n'.join(lines)


def make_handler(tool):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def send(self, code, body, ctype='application/json'):
            self.send_response(code)
            self.send_header('Content-Type', ctype)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass                                  # the page gave up on a slow frame (its 3 s retry)

        def do_GET(self):
            u = urlparse(self.path)
            if u.path == '/':
                return self.send(200, PAGE.encode(), 'text/html; charset=utf-8')
            if u.path == '/status':
                return self.send(200, json.dumps({'mode': tool.mode, 'fov': tool.fov, 'auto_zoom': tool.auto_zoom,
                                                  'exposure': tool.a.qr_max_exposure_ms, 'qr_size': tool.a.qr_size,
                                                  'use_gpu': tool.use_gpu, 'enhance': tool.enhance, 'hq': tool.hq,
                                                  'text': tool.status()}).encode())
            if u.path == '/view.jpg':                  # the latest annotated view (the page polls it)
                if not tool.view:
                    return self.send(503, b'{}')
                return self.send(200, tool.view, 'image/jpeg')
            if u.path == '/frame.jpg':                 # the latest raw frame, full resolution
                with tool.lock:
                    f = tool.frame
                if f is None:
                    return self.send(503, b'{}')
                return self.send(200, cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes(), 'image/jpeg')
            if u.path == '/board.png':
                sq = 120
                img = np.full(((tool.rows + 3) * sq, (tool.cols + 3) * sq), 255, np.uint8)
                for r in range(tool.rows + 1):
                    for c in range(tool.cols + 1):
                        if (r + c) % 2 == 0:
                            img[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
                return self.send(200, cv2.imencode('.png', img)[1].tobytes(), 'image/png')
            if u.path == '/qr.png':                    # made on the host (segno isn't in the container)
                p = Path(tool.a.out).parent / 'test_qr.png'
                if p.exists():
                    return self.send(200, p.read_bytes(), 'image/png')
                return self.send(404, b'{"message": "no calibration/test_qr.png"}')
            if u.path == '/stream.mjpg':
                self.send_response(200)
                self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=f')
                self.end_headers()
                try:
                    while True:
                        jpg = tool.view
                        if jpg:
                            self.wfile.write(b'--f\r\nContent-Type: image/jpeg\r\nContent-Length: ' +
                                             str(len(jpg)).encode() + b'\r\n\r\n' + jpg + b'\r\n')
                        time.sleep(0.07)              # ~14 fps
                except (BrokenPipeError, ConnectionResetError):
                    return
            self.send(404, b'{}')

        def do_POST(self):
            u = urlparse(self.path)
            q = parse_qs(u.query)
            if u.path == '/mode':
                m = q.get('m', ['view'])[0]
                if m == 'undist' and not tool.calib:
                    return self.send(200, json.dumps({'message': 'no calibration yet'}).encode())
                tool.mode, tool.msg = m, ''
                return self.send(200, b'{}')
            if u.path == '/fov':
                tool.fov, tool.zoom = int(float(q.get('v', ['0'])[0])), None
                return self.send(200, b'{}')
            if u.path == '/autozoom':
                tool.auto_zoom, tool.zoom = not tool.auto_zoom, None
                return self.send(200, b'{}')
            if u.path == '/exposure':                  # QR mode exposure cap: the camera reopens with it
                tool.a.qr_max_exposure_ms = float(q.get('v', ['5'])[0])
                return self.send(200, b'{}')
            if u.path == '/toggle_gpu':
                tool.use_gpu = not tool.use_gpu
                return self.send(200, json.dumps({'use_gpu': tool.use_gpu}).encode())
            if u.path == '/toggle_enhance':
                tool.enhance = not tool.enhance
                return self.send(200, json.dumps({'enhance': tool.enhance}).encode())
            if u.path == '/toggle_hq':
                tool.hq = not tool.hq
                return self.send(200, json.dumps({'hq': tool.hq}).encode())
            if u.path == '/qrsize':                    # the dark code square's side, m (0 = distances in code widths)
                tool.a.qr_size = max(0.0, float(q.get('v', ['0'])[0] or 0))
                return self.send(200, b'{}')
            if u.path == '/capture':
                tool.mode = 'capture'
                return self.send(200, b'{}')
            if u.path == '/reset':
                tool.caps = []
                for f in tool.frames_dir.glob('cap_*.png'):
                    f.unlink()
                return self.send(200, json.dumps({'message': 'captures cleared'}).encode())
            if u.path == '/calibrate':
                ok, msg = tool.calibrate()
                tool.msg = msg
                return self.send(200, json.dumps({'ok': ok, 'message': msg}).encode())
            self.send(404, b'{}')
    return H


STOP = threading.Event()             # set on SIGINT / SIGTERM: the capture loop releases the camera and stops
CAM_RELEASED = threading.Event()     # clear while an Argus session is open
CAM_RELEASED.set()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8090)
    ap.add_argument('--board', default='9x6', help='inner corners, cols x rows')
    ap.add_argument('--square', type=float, default=0.025, help='m (only scales the board poses)')
    ap.add_argument('--sensor-id', type=int, default=0)
    ap.add_argument('--width', type=int, default=1640)
    ap.add_argument('--height', type=int, default=1232)
    ap.add_argument('--max-exposure-ms', type=float, default=20.0)
    ap.add_argument('--still-px-s', type=float, default=15.0,
                    help='calib: auto-capture only when the corners move slower than this (px/s at 1640 wide)')
    ap.add_argument('--qr-width', type=int, default=3280, help='QR test: full resolution')
    ap.add_argument('--qr-height', type=int, default=2464)
    ap.add_argument('--qr-max-exposure-ms', type=float, default=5.0, help='QR test: short exposure (less blur)')
    ap.add_argument('--qr-fov', type=int, default=75, help='QR test view: straightened middle of the lens, deg (0 = fisheye)')
    ap.add_argument('--auto-zoom', type=int, default=1, help='1: zoom the view onto a QR once one is read')
    ap.add_argument('--zoom-hold', type=float, default=4.0, help='s without a read or a box before going back to wide')
    ap.add_argument('--chase-step', type=float, default=4.0, help='s at one chase step without a read before the next')
    ap.add_argument('--scan-exposure-ms', type=float, default=80.0,
                    help='QR test: exposure cap once a code is found but not read (the camera held still)')
    ap.add_argument('--scan-fps', type=int, default=10, help='QR test: frame rate of the scan settings (<= 1000 / exposure)')
    ap.add_argument('--qr-size', type=float, default=0.0, help='side of the dark code square, m, for the distances (0: code widths)')
    ap.add_argument('--min-score', type=float, default=0.55, help='"looks like a QR" boxes from this score')
    ap.add_argument('--net-conf', type=float, default=0.5, help='QR detection network: boxes from this confidence')
    ap.add_argument('--scan-every', type=int, default=3, help='a full network scan of the view every N detections')
    ap.add_argument('--min-transitions', type=float, default=4.0,
                    help='"looks like a QR": black/white changes per row AND per column (an edge has 1)')
    ap.add_argument('--cand-std-min', type=float, default=6.0, help='contrast candidates: lowest grey std threshold')
    ap.add_argument('--cand-win', type=int, default=5, help='contrast candidates: window px at half resolution')
    ap.add_argument('--denoise', type=float, default=10.0, help='non-local-means strength on the zoomed view (0 = off)')
    ap.add_argument('--wechat-dir', default='/workspaces/isaac_ros-dev/calibration/wechat_qrcode',
                    help="OpenCV WeChat QR model files (detect / sr .prototxt + .caffemodel)")
    ap.add_argument('--gpu', action='store_true', help='enable GPU tensor acceleration (PyTorch CUDA on Orin)')
    ap.add_argument('--enhance', action='store_true', help='enable image quality enhancement (CLAHE + unsharp mask)')
    ap.add_argument('--hq', action='store_true', help='enable 95% JPEG quality stream')
    ap.add_argument('--out', default='/workspaces/isaac_ros-dev/calibration/arducam_fisheye_1640x1232.yaml')
    ap.add_argument('--recompute', action='store_true', help='calibrate from the saved captures and exit')
    a = ap.parse_args()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tool = Tool(a)
    if a.recompute:
        tool.size = (a.width, a.height)
        print(tool.calibrate()[1])
        return
    cam = threading.Thread(target=tool.capture_loop, daemon=True)
    cam.start()

    def stop(signum, _frame):
        # Release the Argus session before exiting: a process killed with the pipeline open can wedge
        # nvargus-daemon, and the next start then can't open the camera (2026-10-26/27). docker exec doesn't
        # forward signals, so the launcher (arducam_calib.sh stop / the service) signals this process directly.
        STOP.set()
        CAM_RELEASED.wait(3.0)
        print(f'arducam_calib: stopped on signal {signum}, camera released={CAM_RELEASED.is_set()}', flush=True)
        os._exit(0)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    threading.Thread(target=tool.work_loop, daemon=True).start()
    threading.Thread(target=tool.qr_loop, daemon=True).start()
    print(f'Arducam calibration page: http://10.42.0.247:{a.port}')
    ThreadingHTTPServer(('0.0.0.0', a.port), make_handler(tool)).serve_forever()


if __name__ == '__main__':
    main()
