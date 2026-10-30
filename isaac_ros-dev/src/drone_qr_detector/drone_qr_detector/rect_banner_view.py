#!/usr/bin/env python3
"""Live rectilinear (de-fisheyed) view of the Arducam IMX219 + ~185 deg circular fisheye, with the green (entry) and
blue (exit) / red (the wrong tunnel, course of 2026-10-28) banners detected on it.

    python3 rect_banner_view.py                         # camera; window pops up, banners are outlined when in view
    python3 rect_banner_view.py --hfov 100 --vfov 75    # narrower view, less edge stretch
    python3 rect_banner_view.py --source some_frame.jpg # or a video: offline test
    ros2 run drone_qr_detector rect_banner_view         # same, once the package is built

Camera (--source auto): if arducam_calib.py holds the CSI camera, its shared http://127.0.0.1:8090/frame.jpg is used
(the same non-intrusive sharing as csi_camera_node / arducam_csi); otherwise nvarguscamerasrc directly at the full
3280x2464 @ 21 fps (--mode full) or the 2x2-binned 1640x1232 @ 30 fps (--mode half). Argus does debayer / ISP.

Dewarp: the lens is equidistant (r = f * theta, calibration/arducam_fisheye_1640x1232.yaml - despite the name it is
the 3280x2464 calibration; FisheyeModel scales it to the half mode). The output is a pinhole (rectilinear) view:
straight lines stay straight. A pinhole view cannot hold 180 deg (r = f tan theta -> infinity at 90 deg), so it is a
window of --hfov x --vfov degrees (default 120 x 90, 16:9). Its focal length equals the lens's (--zoom 1), so the image
centre is sampled 1:1 from the sensor - no resolution is thrown away there; towards the edges the fisheye squeezed the
scene and the pinhole view stretches it back (1/cos^2 theta: x4 at 60 deg), which no remap can sharpen. 3280x2464 in
-> 2448x1416 out. The remap runs on the GPU (VPI, Catmull-Rom) when available, else cv2.remap on the CPU (~4x slower,
bilinear by default there: --interp cubic to force it). It runs on its own thread, overlapped with the display.
Detection runs on its own thread on the newest frame, so the video is not held back by it.

Banner detection, on the rectilinear image (a banner is a true quadrilateral there, not a curved blob):
  1. white balance  grey-world gains from the near-neutral pixels (the IMX219 here has a strong magenta cast),
                    smoothed over frames; applied to the shown feed too (key w)
  2. masking        Gaussian blur (sensor noise) -> HSV -> per-colour inRange, AND the valid (inside the image circle)
                    mask -> open (speckle) -> close (joins the colour around the white letters)
  3. shape          external contours -> quadrilateral fit (approxPolyDP on the hull, minAreaRect fallback);
                    keep: area, rectangularity (blob / quad area), elongation (long / short side)
  4. surface        on a full-resolution crop: the banner colour must fill the quad (--min-solid; the real banners
                    are plain). With --with-letters (the sim's lettered banners) instead: white-letter fraction in
                    range, and the density of Harris corners of the letters adds confidence
  5. corners        edge lines fitted at full resolution, each corner confirmed by a Harris peak of the banner mask
                    and refined to sub-pixel (cornerSubPix)
  6. true shape     the banner is --banner-size (its green field 15.8 : 1, from the 2026-10-30 photo; was 7.7 : 1): the real aspect of the rectangle the 4
                    corners are the perspective image of (known camera matrix, so whatever the viewing angle) must match
                    - this is what rejects bags, boxes and doors with colour + white print. The known size also gives
                    the distance (solvePnP), shown on the label
  7. tracking       per colour: matched frame to frame, smoothed, shown once seen twice, held 0.5 s when lost
HSV ranges default to the real-camera-friendly values below (looser than config/banner.yaml's sim values); tune live
with key t (trackbars) and pass the result as --green-h / --blue-h / --red-h / --s-min / --v-min. Red wraps round
hue 0: --red-h LO HI with LO > HI is H >= LO or H <= HI.

Keys: q / Esc quit, s save (raw fisheye, rectilinear and annotated PNGs, full resolution), m colour-mask overlay,
w white balance on/off, t tuning trackbars, h HUD on/off.
"""
import argparse
import contextlib
import functools
import http.client
import math
import os
import signal
import sys
import threading
import time
from pathlib import Path

# The system OpenCV has GStreamer (nvarguscamerasrc) and matches the system VPI / numpy; ~/.local's pip opencv doesn't
if os.path.isdir('/usr/lib/python3/dist-packages'):
    sys.path.insert(0, '/usr/lib/python3/dist-packages')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # run as a plain script: drone_qr_detector importable

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from drone_qr_detector import white_frame  # noqa: E402
from drone_qr_detector.fisheye_model import FisheyeModel  # noqa: E402


@contextlib.contextmanager
def no_display():
    """$DISPLAY hidden for the duration. Argus and VPI's CUDA backend bind EGL to $DISPLAY when they start; on an X
    server without NVIDIA's EGL (ssh -X, Xvfb, VNC) that fails (EGL_BAD_DISPLAY) - without one they use the GPU
    directly. The window, opened by GTK before, keeps working (as arducam_csi / csi_camera_node drop it too)."""
    display = os.environ.pop('DISPLAY', None)
    try:
        yield
    finally:
        if display is not None:
            os.environ['DISPLAY'] = display


def _vpi_search_paths():
    """Where JetPack puts VPI's Python bindings when they are not already importable (VPI 2: /opt/nvidia/vpi2/lib64/
    python, VPI 3: /opt/nvidia/vpi3/lib/aarch64-linux-gnu/python, plus the versioned system dist-packages)."""
    import glob
    pats = ('/opt/nvidia/vpi*/lib*/python', '/opt/nvidia/vpi*/lib/*/python',
            f'/usr/lib/python{sys.version_info.major}.{sys.version_info.minor}/dist-packages')
    return [d for p in pats for d in sorted(glob.glob(p)) if os.path.isdir(d) and d not in sys.path]


VPI_ERROR = None
try:
    with no_display():                                        # VPI binds EGL to $DISPLAY on import
        import vpi  # noqa: E402
except ImportError as e:
    vpi, VPI_ERROR = None, e
    for _d in _vpi_search_paths():
        sys.path.append(_d)
        try:
            with no_display():
                import vpi  # noqa: E402,F811
            VPI_ERROR = None
            break
        except ImportError as e2:
            sys.path.remove(_d)
            vpi, VPI_ERROR = None, e2

WORKSPACE = Path(__file__).resolve().parents[3]
CALIB_NAME = 'calibration/arducam_fisheye_1640x1232.yaml'
DEFAULT_CALIB = next((str(p) for p in (WORKSPACE / CALIB_NAME,                       # source tree
                                       Path('/workspaces/isaac_ros-dev') / CALIB_NAME,  # container
                                       Path.home() / 'workspaces/isaac_ros-dev' / CALIB_NAME)
                      if p.exists()), '')
MODES = {'full': (3280, 2464, 21), 'half': (1640, 1232, 30)}
COLOURS = {'green': (0, 220, 0), 'blue': (255, 120, 0), 'red': (0, 0, 255)}      # BGR drawing colours
TEXT_SCALE = 80                                              # px, banner short side when counting letter corners
WINDOW = 'Arducam rectilinear - banners'
TUNING = 'banner tuning'


# ------------------------------------------------------------------ camera
class FrameSource:
    """Newest frame from the camera (shared HTTP stream or CSI), a video or a still, read on a background thread."""

    def __init__(self, args):
        self.args = args
        self.frame, self.seq, self.cond = None, 0, threading.Condition()
        self.running, self.error, self.kind = True, None, None
        self.n_grab, self.t0 = 0, time.monotonic()
        src = args.source
        if src == 'auto':
            src = 'http' if self.http_frame(None)[1] is not None else 'csi'
        self.kind = src
        if src == 'http':
            target = self.http_loop
        elif src == 'csi':
            w, h, fps = MODES[args.mode]
            head = (f'nvarguscamerasrc sensor-id={args.sensor_id} '
                    f'exposuretimerange="13000 {int(args.max_exposure_ms * 1e6)}" gainrange="1 10.625" ! '
                    f'video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! '
                    f'nvvidconv flip-method={args.flip} ! video/x-raw,format=BGRx ! ')
            sink = 'appsink drop=true max-buffers=1 sync=false'
            # BGRx straight into the app: the Rectifier drops the 4th byte on the GPU, instead of videoconvert doing
            # it for all 8 MP on one CPU core (that capped the camera below 21 fps); older OpenCV: the BGR pipeline
            for pipe in (head + sink, head + 'videoconvert ! video/x-raw,format=BGR ! ' + sink):
                with no_display():
                    self.cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    break
            if not self.cap.isOpened():
                raise SystemExit(f'cannot open the CSI camera:\n  {pipe}\nnvargus-daemon running? Another process '
                                 'holding the camera (arducam_csi, csi_camera_node, a gst-launch)? Use --source http '
                                 'if arducam_calib.py is sharing it on :8090.')
            target = self.cap_loop
        elif Path(src).suffix.lower() in ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'):
            img = cv2.imread(src)
            if img is None:
                raise SystemExit(f'cannot read {src}')
            self.kind, target = 'image', lambda: self.still_loop(img)
        else:
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise SystemExit(f'cannot open {src}')
            self.kind, target = 'video', self.cap_loop
        self.thread = threading.Thread(target=target, daemon=True)
        self.thread.start()

    def publish(self, img):
        with self.cond:
            self.frame, self.seq = img, self.seq + 1
            self.n_grab += 1
            self.cond.notify_all()

    def wait(self, last_seq, timeout=2.0):
        """(frame, seq) newer than last_seq, or (None, last_seq) on timeout."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout)
            return (self.frame, self.seq) if self.seq != last_seq else (None, last_seq)

    def http_frame(self, conn):
        try:
            if conn is None:
                conn = http.client.HTTPConnection(self.args.http_host, self.args.http_port, timeout=1.5)
            conn.request('GET', '/frame.jpg')
            resp = conn.getresponse()
            body = resp.read()
            if resp.status == 200 and len(body) > 1000:
                return conn, cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            conn = None
        return conn, None

    def http_loop(self):
        conn, misses = None, 0
        while self.running:
            conn, img = self.http_frame(conn)
            if img is None:
                misses += 1
                if misses == 10:
                    print('WARN: shared stream :%d not answering (arducam_calib.py stopped?) - restart with '
                          '--source csi' % self.args.http_port)
                time.sleep(0.2)
                continue
            misses = 0
            self.publish(img)

    def cap_loop(self):
        period = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30) if self.kind == 'video' else 0.0
        while self.running:
            t = time.monotonic()
            ok, img = self.cap.read()
            if not ok:
                if self.kind == 'video':
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)     # loop the clip
                    continue
                time.sleep(0.01)
                continue
            self.publish(img)
            if period:
                time.sleep(max(0.0, period - (time.monotonic() - t)))

    def still_loop(self, img):
        while self.running:
            self.publish(img)
            time.sleep(0.1)

    def fps(self):
        now = time.monotonic()
        f = self.n_grab / max(1e-3, now - self.t0)
        if now - self.t0 > 2.0:
            self.n_grab, self.t0 = 0, now
        return f

    def stop(self):
        """Reader thread off, then the device released: releasing during a read in flight segfaults (as in
        fpv_camera)."""
        self.running = False
        with self.cond:
            self.cond.notify_all()
        self.thread.join(timeout=3.0)
        cap = getattr(self, 'cap', None)
        if cap is not None and not self.thread.is_alive():
            cap.release()


# ------------------------------------------------------------------ dewarp
class Rectifier:
    """Fisheye -> pinhole remap with fixed maps (built once per input size), on the GPU if VPI is there."""

    def __init__(self, calib, in_size, hfov, vfov, zoom, backend, interp, model=None):
        w, h = in_size
        # model: a ready FisheyeModel (banner_node: from the topic's CameraInfo) instead of the calibration YAML
        self.model = model if model is not None else \
            FisheyeModel.from_yaml(calib, target_width=w, target_height=h) if calib else FisheyeModel(w, h).scaled(w, h)
        K, D = self.model.K, self.model.D
        f = 0.5 * (K[0, 0] + K[1, 1]) * zoom
        # multiples of 24: the working copy is then an exact 1/2, 1/3, 1/4 ... (OpenCV's fast INTER_AREA path)
        W = int(round(2 * f * math.tan(math.radians(hfov) / 2) / 24)) * 24
        H = int(round(2 * f * math.tan(math.radians(vfov) / 2) / 24)) * 24
        self.size, self.f = (W, H), f
        self.P = np.array([[f, 0, (W - 1) / 2], [0, f, (H - 1) / 2], [0, 0, 1]], np.float64)
        mx, my = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), self.P, (W, H), cv2.CV_32FC1)
        # valid = maps to the inside of the image circle (lens barrel / black border excluded) and the sensor
        circle = self.model.circle_mask(w, h, max_theta_deg=88.0)
        self.valid = cv2.remap(circle, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        self.valid_small = {}
        self.backend = 'vpi' if backend in ('auto', 'vpi') and vpi is not None else 'cpu'
        if backend in ('auto', 'vpi') and vpi is None:
            print(f'WARN: VPI not importable ({VPI_ERROR}) - remapping on the CPU, several times slower. Install '
                  "JetPack's VPI Python bindings (or check numpy matches them) to remap on the GPU")
        # 'auto': Catmull-Rom on the GPU (free there), bilinear on the CPU (cubic costs ~2.5x as much there)
        self.interp_name = 'cubic' if interp == 'cubic' or (interp == 'auto' and self.backend == 'vpi') else 'linear'
        if self.backend == 'vpi':
            self.warp = vpi.WarpMap(vpi.WarpGrid((W, H)))
            a = np.asarray(self.warp)                         # rows padded up to the grid's block size
            a[...] = -1.0
            a[:H, :W, 0], a[:H, :W, 1] = mx, my
            self.interp = vpi.Interp.CATMULL_ROM if self.interp_name == 'cubic' else vpi.Interp.LINEAR
            try:
                with no_display():
                    self(np.zeros((h, w, 3), np.uint8))       # first CUDA use: fail here, not mid-video
            except Exception as e:
                print(f'WARN: VPI remap failed ({e}) - remapping on the CPU')
                self.backend = 'cpu'
                if interp == 'auto':
                    self.interp_name = 'linear'
        if self.backend == 'cpu':
            self.map1, self.map2 = cv2.convertMaps(mx, my, cv2.CV_16SC2)
            self.interp = cv2.INTER_CUBIC if self.interp_name == 'cubic' else cv2.INTER_LINEAR
        corner_deg = math.degrees(math.atan(math.hypot(W / 2, H / 2) / f))
        print(f'INFO: {w}x{h} fisheye (f {K[0, 0]:.1f} px, centre {K[0, 2]:.1f},{K[1, 2]:.1f}) -> {W}x{H} '
              f'rectilinear {hfov:.0f} x {vfov:.0f} deg (corners {corner_deg:.0f} deg off axis, edge stretch '
              f'x{1 / math.cos(math.radians(hfov / 2)) ** 2:.1f}), {self.backend} remap, {self.interp_name}')

    def downscale(self, view, width):
        """(view, valid mask) shrunk by the whole factor closest to `width` px wide; the mask (eroded off the
        lens-barrel rim) is cached."""
        k = max(1, int(round(view.shape[1] / width)))
        size = (view.shape[1] // k, view.shape[0] // k)
        if size not in self.valid_small:
            v = cv2.resize(self.valid, size, interpolation=cv2.INTER_NEAREST)
            self.valid_small[size] = cv2.erode(v, np.ones((5, 5), np.uint8))
        return cv2.resize(view, size, interpolation=cv2.INTER_AREA), self.valid_small[size]

    def __call__(self, img):
        """BGR, or BGRx (the CSI pipeline's) -> rectilinear BGR."""
        bgrx = img.ndim == 3 and img.shape[2] == 4
        if self.backend == 'vpi':
            with vpi.Backend.CUDA:
                out = vpi.asimage(img, vpi.Format.BGRA8 if bgrx else vpi.Format.BGR8).remap(
                    self.warp, self.size, interp=self.interp, border=vpi.Border.ZERO)
                if bgrx:
                    out = out.convert(vpi.Format.BGR8)        # on the GPU, on 3.5 MP instead of the CPU on 8 MP
            return out.cpu()                                  # an array of its own (VPI copies out)
        if bgrx:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return cv2.remap(img, self.map1, self.map2, self.interp, borderMode=cv2.BORDER_CONSTANT)


# ------------------------------------------------------------------ white balance
class WhiteBalance:
    """Grey-world on near-neutral pixels, gains smoothed over frames and clamped (a big green banner must not be
    'balanced' away)."""

    def __init__(self, alpha=0.15):
        self.gains, self.alpha = np.ones(3), alpha

    def update(self, small, valid_small):
        k = max(1, small.shape[1] // 160)                     # statistics only: 160 px wide is plenty
        tiny, vt = small[::k, ::k], valid_small[::k, ::k]
        hsv = cv2.cvtColor(tiny, cv2.COLOR_BGR2HSV)
        sel = cv2.bitwise_and(cv2.inRange(hsv, (0, 0, 41), (180, 89, 244)), vt)
        if cv2.countNonZero(sel) < 0.05 * max(1, cv2.countNonZero(vt)):
            return self.gains
        mean = np.array(cv2.mean(tiny, mask=sel)[:3])
        g = np.clip(mean.mean() / np.maximum(mean, 1.0), 0.6, 1.8)
        self.gains = (1 - self.alpha) * self.gains + self.alpha * g
        return self.gains

    def lut(self):
        x = np.arange(256, dtype=np.float64)
        return np.dstack([np.clip(x * g, 0, 255) for g in self.gains]).astype(np.uint8).reshape(1, 256, 3)


# ------------------------------------------------------------------ detection
def order_quad(q):
    """4 points -> TL, TR, BR, BL (image coordinates)."""
    q = np.asarray(q, np.float32).reshape(4, 2)
    c = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]    # clockwise from the left in image coordinates
    start = int(np.argmin(q.sum(axis=1)))                               # TL: smallest x + y
    return np.roll(q, -start, axis=0)


def true_aspect(quad, K):
    """Real width / height of the rectangle whose perspective image is `quad` (TL, TR, BR, BL), for a pinhole
    camera K - independent of the viewing angle (Zhang & He, "Whiteboard scanning", 2007). None if the quad can't be
    the image of a rectangle (vanishing geometry inconsistent)."""
    m1, m2, m4, m3 = (np.array([x, y, 1.0]) for x, y in quad)
    den2, den3 = np.cross(m2, m4) @ m3, np.cross(m3, m4) @ m2
    if abs(den2) < 1e-9 or abs(den3) < 1e-9:
        return None
    k2, k3 = (np.cross(m1, m4) @ m3) / den2, (np.cross(m1, m4) @ m2) / den3
    if k2 <= 0 or k3 <= 0:
        return None
    n2, n3 = k2 * m2 - m1, k3 * m3 - m1
    Ki = np.linalg.inv(K)
    A = Ki.T @ Ki
    q2, q3 = n2 @ A @ n2, n3 @ A @ n3
    return math.sqrt(q2 / q3) if q2 > 0 and q3 > 0 else None


def banner_distance(quad, K, size_m, wide):
    """Camera -> banner centre (m) from the known banner size (solvePnP, square-ish planar target)."""
    w, h = size_m if wide else size_m[::-1]
    obj = np.array([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]], np.float64)
    ok, _, t = cv2.solvePnP(obj, quad.astype(np.float64), K, None, flags=cv2.SOLVEPNP_IPPE)
    return float(np.linalg.norm(t)) if ok else None


@functools.lru_cache(maxsize=4096)
def ransac_samples(n, iters):
    """The point pairs ransac_line tries for n points: seeded, so the same for every call with that n - drawn once
    (seeding a Generator costs more than the whole fit)."""
    i, j = np.random.default_rng(0).integers(0, n, (2, iters))
    i.flags.writeable = j.flags.writeable = False
    return i, j


def ransac_line(pts, tol=1.5, iters=40):
    """(point, unit direction) of the straight edge most of `pts` lie on: bumps of noise on either side of it are
    outliers, not a pull on the fit."""
    i, j = ransac_samples(len(pts), iters)                     # all hypotheses scored at once
    a, d = pts[i], pts[j] - pts[i]
    n = np.hypot(d[:, 0], d[:, 1])
    ok = n > 1e-6
    a, d, n = a[ok], d[ok], n[ok]
    if not len(n):
        return ransac_line_fallback(pts)
    dist = np.abs((pts[None, :, 0] - a[:, None, 0]) * d[:, None, 1] -
                  (pts[None, :, 1] - a[:, None, 1]) * d[:, None, 0]) / n[:, None]
    inl = dist < tol
    best = inl[np.argmax(inl.sum(axis=1))]
    sel = pts[best] if best.sum() >= 2 else pts
    return ransac_line_fallback(sel)


def ransac_line_fallback(pts):
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).ravel()
    return np.array([x0, y0]), np.array([vx, vy])


def fit_quad(contour, hull):
    """Blob -> 4-corner polygon. Two rough fits - approxPolyDP on the hull (growing tolerance) and minAreaRect - each
    refined by edge lines (refine_by_lines); the one overlapping the hull best wins: a noise spike on the hull can
    make approxPolyDP take its tip for a corner, minAreaRect is pulled by it far less (and the other way round for
    a strongly foreshortened banner, which isn't a rectangle in the image)."""
    peri = cv2.arcLength(hull, True)
    rough = [(order_quad(cv2.boxPoints(cv2.minAreaRect(hull))), 'rect')]
    for eps in (0.01, 0.02, 0.03, 0.045, 0.06, 0.08):
        a = cv2.approxPolyDP(hull, eps * peri, True)
        if len(a) == 4 and cv2.isContourConvex(a):
            rough.append((order_quad(a), 'poly'))
            break
        if len(a) < 4:
            break
    pts = contour.reshape(-1, 2).astype(np.float32)
    hull_f = hull.reshape(-1, 2).astype(np.float32)
    hull_area = max(cv2.contourArea(hull_f), 1e-6)
    best, best_iou = None, -1.0
    for quad, how in rough:
        q, how = refine_by_lines(quad, how, pts, hull_f)
        q_area = cv2.contourArea(q.astype(np.float32))
        if q_area <= 0:
            continue
        inter = cv2.intersectConvexConvex(q.astype(np.float32), hull_f)[0]
        iou = inter / (q_area + hull_area - inter)
        if iou > best_iou:
            best, best_iou = (q, how), iou
    return best if best is not None else rough[0]


def refine_by_lines(quad, how, pts, hull):
    """Each side of `quad` re-fitted as a RANSAC line through the outline points `pts` along its middle 70 %, the
    corners re-made as the intersections of neighbouring sides: a corner chewed off (letters running into the
    banner's edge, a chamfer on the hull) comes back where the edges meet. An intersection is taken only near the
    blob's `hull` (the real corner is on it; nearly parallel neighbours shoot off far outside)."""
    lines, sides = [], []
    for i in range(4):
        p, q = quad[i], quad[(i + 1) % 4]
        d = q - p
        sides.append(np.linalg.norm(d))
        L = float(sides[-1])
        if L < 4:
            return quad, how
        d /= L
        tol = max(3.0, 0.06 * L)
        # only the points in the side's bounding box (+ tol) can be near it: the exact test below on those alone
        lo, hi = np.minimum(p, q) - (tol + 1), np.maximum(p, q) + (tol + 1)
        near = pts[(pts[:, 0] > lo[0]) & (pts[:, 0] < hi[0]) & (pts[:, 1] > lo[1]) & (pts[:, 1] < hi[1])]
        rel = near - p
        t = rel[:, 0] * d[0] + rel[:, 1] * d[1]
        off = np.abs(rel[:, 0] * d[1] - rel[:, 1] * d[0])
        sel = near[(t > 0.15 * L) & (t < 0.85 * L) & (off < tol)]
        if len(sel) < 5:
            return quad, how
        lines.append(ransac_line(sel))
    out = quad.copy()
    for i in range(4):
        (p1, d1), (p2, d2) = lines[i - 1], lines[i]                  # sides ending / starting at corner i
        den = d1[0] * d2[1] - d1[1] * d2[0]
        if abs(den) < 1e-3:
            continue
        s = ((p2[0] - p1[0]) * d2[1] - (p2[1] - p1[1]) * d2[0]) / den
        c = p1 + s * d1
        side = min(sides[i - 1], sides[i])                           # the sides meeting at corner i
        if cv2.pointPolygonTest(hull, (float(c[0]), float(c[1])), True) > -max(3.0, 0.15 * side):
            out[i] = c
    return out, how + '+lines'


class BannerDetector:
    def __init__(self, args):
        self.a = args
        self.hsv = {'green': [*args.green_h, args.s_min, args.v_min],
                    'blue': [*args.blue_h, max(args.s_min, args.blue_s_min), args.v_min],   # the sky: S ~100
                    'red': [*args.red_h, args.s_min, args.v_min]}
        self.last_masks = {}

    def colour_mask(self, hsv, colour):
        lo, hi, s_min, v_min = self.hsv[colour]
        if lo > hi:                                           # red: round hue 0
            return cv2.inRange(hsv, (lo, s_min, v_min), (180, 255, 255)) | cv2.inRange(hsv, (0, s_min, v_min),
                                                                                      (hi, 255, 255))
        return cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))

    def suppress_ground(self, hsv, m, valid):
        """Looking down on GRASS (user, 2026-10-29: the banners lie / stand on grass): when the green mask covers more
        than --ground-frac of the view it is mostly grass - keep only pixels clearly unlike its median (more saturated,
        brighter, another hue). (A 'locally solid' test let 90 % of blurred grass through - dropped.)"""
        a = self.a
        on = m > 0
        if on.sum() < a.ground_frac * max(int((valid > 0).sum()), 1):
            return m
        h, sat, v = (hsv[..., i][on] for i in range(3))
        hm, sm, vm = float(np.median(h)), float(np.median(sat)), float(np.median(v))
        hh, ss, vv = (hsv[..., i].astype(np.float32) for i in range(3))
        distinct = (ss >= sm + a.ground_ds) | (vv >= vm + a.ground_dv) | (np.abs(hh - hm) >= a.ground_dh)
        return np.where(on & distinct, 255, 0).astype(np.uint8)

    def detect(self, img, valid, small, vsmall, lut=None, K=None):
        """img: rectilinear BGR (full resolution), valid: its valid-pixel mask; small / vsmall: the same at
        --proc-width, already white-balanced; lut: the white balance, applied to the full-resolution crops (not to the
        whole frame: only the crops are looked at); K: the rectilinear view's camera matrix (true aspect, distance)
        -> detections, best first."""
        self.lut, self.K = lut, K
        a = self.a
        s = img.shape[1] / float(small.shape[1])
        hsv = cv2.cvtColor(cv2.GaussianBlur(small, (5, 5), 0), cv2.COLOR_BGR2HSV)
        framed, white_px = white_frame.frames(small)
        k_close = max(5, (a.proc_width // 90) | 1)
        min_area = a.min_area_frac * small.shape[0] * small.shape[1]
        found = []
        for colour in COLOURS:
            m = cv2.bitwise_and(self.colour_mask(hsv, colour), vsmall)
            if colour == 'green' and a.ground_suppress:
                m = self.suppress_ground(hsv, m, vsmall)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close)))
            if colour in framed:                              # the white frame first; the colour blob for far banners
                m = cv2.bitwise_and(white_frame.cut_by_frame(m, framed[colour], white_px), vsmall)
            self.last_masks[colour] = m
            # second look with thin bridges cut: a banner touching same-coloured clutter (a stalk to a dark print on
            # the wall) gets a corner pulled; the cut costs far banners whose letters already thin the colour - so
            # both, only for blobs the cut changed, and duplicates of one banner merged below
            cut = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            seen = set()
            for mask in (m, cut):
                for c in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:  # all pixels: line fits
                    box = cv2.boundingRect(c)
                    if cv2.contourArea(c) < min_area or box in seen:
                        continue
                    seen.add(box)
                    d = self.judge(c, mask, img, valid, s, colour)
                    if d is not None:
                        found.append(d)
                    elif cv2.contourArea(c) > 4 * min_area:
                        found += self.split(c, mask, img, valid, s, colour, min_area)
        found.sort(key=lambda d: -d['score'])
        kept = []                                             # one detection per banner: the best scoring
        for d in found:
            if not any(k['colour'] == d['colour'] and
                       cv2.pointPolygonTest(k['quad'].astype(np.float32), tuple(map(float, d['centre'])), False) >= 0
                       for k in kept):
                kept.append(d)
        return kept

    def split(self, contour, mask, img, valid, s, colour, min_area):
        """A blob that is no banner may be a banner joined to something thinner of its colour - the green POLES at the
        tunnel mouth touch the floor banner (2026-10-30 photo: an L of pole + banner). Opened with growing kernels the
        thin part goes first; the pieces are judged again (the first size that gives a banner wins)."""
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        blob = cv2.bitwise_and(blob, mask)
        unit = max(1.0, mask.shape[1] / 640.0)
        for k in (5, 9, 15, 23, 33):
            kk = int(k * unit) | 1
            opened = cv2.morphologyEx(blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
            got = []
            for c in cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                if cv2.contourArea(c) >= min_area:
                    d = self.judge(c, opened, img, valid, s, colour)
                    if d is not None:
                        got.append(d)
            if got:
                return got
        return []

    def judge(self, contour, mask_small, img, valid, s, colour):
        a = self.a
        hull = cv2.convexHull(contour)
        quad_s, how = fit_quad(contour, hull)
        quad_area_s = cv2.contourArea(quad_s)
        if quad_area_s <= 0:
            return None
        # rectangularity: how much of the fitted quad the (closed) colour blob fills, and how much of the blob the quad
        # covers - a banner fills its quad; grass, trees, a curved tarp edge don't
        x, y, w, h = cv2.boundingRect(quad_s.astype(np.int32))
        x0, y0 = max(0, x - 1), max(0, y - 1)
        blob = mask_small[y0:y + h + 1, x0:x + w + 1]
        qm = np.zeros(blob.shape, np.uint8)
        cv2.fillConvexPoly(qm, np.round(quad_s - [x0, y0]).astype(np.int32), 255)
        inter = cv2.countNonZero(cv2.bitwise_and(blob, qm))
        fill = inter / max(1.0, cv2.countNonZero(qm))
        cover = inter / max(1.0, cv2.contourArea(contour))
        if fill < a.min_fill or cover < 0.8:
            return None
        sides = np.linalg.norm(quad_s - np.roll(quad_s, -1, axis=0), axis=1)
        pair = sides[0] + sides[2], sides[1] + sides[3]
        long_, short = max(pair) / 2, min(pair) / 2
        aspect = long_ / max(short, 1e-6)
        if not a.min_aspect <= aspect <= a.max_aspect:
            return None

        # full-resolution crop: text evidence, Harris corners
        quad = quad_s * s
        pad = int(max(8, 0.15 * short * s))
        X0, Y0 = np.maximum(np.floor(quad.min(axis=0)).astype(int) - pad, 0)
        X1, Y1 = np.minimum(np.ceil(quad.max(axis=0)).astype(int) + pad, [img.shape[1], img.shape[0]])
        crop = img[Y0:Y1, X0:X1] if self.lut is None else cv2.LUT(img[Y0:Y1, X0:X1], self.lut)
        if crop.size == 0:
            return None
        local = (quad - [X0, Y0]).astype(np.float32)
        inside = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillConvexPoly(inside, local.astype(np.int32), 255)
        # cut off by the image circle or the frame edge: shown, flagged, scored lower
        partial = cv2.countNonZero(cv2.bitwise_and(inside, cv2.bitwise_not(valid[Y0:Y1, X0:X1]))) > \
            0.02 * cv2.countNonZero(inside) or quad.min() < 3 or \
            (quad[:, 0] > img.shape[1] - 4).any() or (quad[:, 1] > img.shape[0] - 4).any()
        crop_hsv = cv2.cvtColor(cv2.GaussianBlur(crop, (5, 5), 0), cv2.COLOR_BGR2HSV)  # sensor noise
        col = cv2.bitwise_and(self.colour_mask(crop_hsv, colour), inside)
        n_in, n_col = cv2.countNonZero(inside), cv2.countNonZero(col)
        solid = n_col / max(1, n_in)
        if solid < 0.3:
            return None
        white, white_frac, corner_density = None, 0.0, 0.0
        v_col = float(np.median(crop_hsv[..., 2][col > 0]))
        # the thin WHITE BORDER round the banner (course 2026-10-30): white in a band just outside the quad - evidence
        # only (a far banner's border is under a pixel, a cut-off one is half missing), never a reason to reject
        band = max(2, int(round(0.12 * short * s)))
        ring = cv2.subtract(cv2.dilate(inside, np.ones((2 * band + 1, 2 * band + 1), np.uint8)), inside)
        rim = cv2.bitwise_and(cv2.inRange(crop_hsv, (0, 0, int(max(a.v_min, 0.9 * v_col))),
                                          (180, min(a.white_s_max, white_frame.PALE_S), 255)), ring)   # the sky is S ~100
        white_rim = cv2.countNonZero(rim) / max(1, cv2.countNonZero(ring))
        if short * s >= a.rim_min_px and white_rim < a.rim_min and not partial:
            return None                                       # close enough to see a frame, and there is none: not a banner
        if not a.with_letters:
            # plain banner: its colour fills it (the fitted quad's rim may catch a little background)
            if solid < a.min_solid:
                return None
        else:
            # white letters: unsaturated and not darker than text_v_ratio x the banner colour (in shade they go grey)
            white = cv2.inRange(crop_hsv, (0, 0, int(a.text_v_ratio * v_col)), (180, a.white_s_max, 255))
            white = cv2.bitwise_and(white, inside)
            white_frac = cv2.countNonZero(white) / n_in
            corner_density, _ = self.text_corners(crop, inside, short * s)
            # the letters must be there; their Harris corners only add confidence. A far banner's letters blur into
            # its colour: below --min-text-px tall (and fully in view) too little white is excused, too much never
            resolved = short * s >= a.min_text_px or partial
            if white_frac > a.white_max or (resolved and white_frac < a.white_min):
                return None

        # outline mask: the banner's colour at >= half its own median saturation (noise speckle in a greyish
        # background passes --s-min but not that, and would bulge the edges) + its letters, if it has any
        lo, hi, s_min, v_min = self.hsv[colour]
        s_ban = max(s_min, int(0.5 * np.median(crop_hsv[..., 1][col > 0])))
        outline = cv2.inRange(crop_hsv, (lo, s_ban, v_min), (hi, 255, 255))
        if white is not None:
            outline = cv2.bitwise_or(outline, white)
        quad, n_harris = self.refine_corners(local, outline, X0, Y0, short * s)
        # shape: the banner is a W x H rectangle. Its true aspect (whatever the angle it's seen at) must match -
        # a bag, a box, a blue door are colour + rectangle (+ maybe some print) too, but not 7.7 : 1. Cut off by the
        # frame edge, only the visible part is measured: then it only has to be clearly elongated
        ratio, dist = None, None
        nominal = max(a.banner_size) / min(a.banner_size)
        if self.K is not None:
            r = true_aspect(quad, self.K)
            if r is None:
                return None
            ratio = max(r, 1 / r)
            lo_r, hi_r = (a.min_partial_aspect if partial else nominal / a.aspect_tol), nominal * a.aspect_tol
            if not lo_r <= ratio <= hi_r:
                return None
            if not partial:
                dist = banner_distance(quad, self.K, [x * 0.3048 for x in a.banner_size], r >= 1)
        if a.with_letters:
            surface = 0.5 + 0.5 * (min(1.0, white_frac / 0.08) * 0.5 +
                                   min(1.0, corner_density / (2 * a.min_corner_density)) * 0.5)
        else:
            surface = min(1.0, solid / 0.95)
        shape = 1.0 if ratio is None or partial else \
            max(0.0, 1 - abs(math.log(ratio / nominal)) / math.log(a.aspect_tol))
        conf = fill * surface * (0.8 + 0.05 * n_harris) * (0.7 + 0.3 * shape) * (0.7 if partial else 1.0) * \
            (0.9 + 0.25 * min(1.0, white_rim / 0.35))         # a white border seen: up to +12 %
        area = cv2.contourArea(quad)
        return {'colour': colour, 'quad': quad, 'centre': quad.mean(axis=0), 'area': area, 'aspect': aspect,
                'fill': fill, 'solid': solid, 'white_frac': white_frac, 'corners_per_kpx': corner_density,
                'partial': partial, 'harris_corners': n_harris, 'true_aspect': ratio, 'distance_m': dist,
                'white_rim': round(white_rim, 3),
                'conf': float(np.clip(conf, 0, 1)), 'score': float(np.clip(conf, 0, 1)) * math.sqrt(area), 'fit': how}

    def text_corners(self, crop, inside, short):
        """Harris corners per 1000 px^2 inside the banner, measured at a fixed scale (short side TEXT_SCALE px) on the
        contrast-stretched grey crop: at full resolution sensor noise in a dark plain patch out-scores real letter
        corners; shrunk, the noise averages out and the letters stay. Returns (density, contrast 0-1)."""
        s = min(1.0, TEXT_SCALE / max(short, 1.0))
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        if s < 1.0:
            gray = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
            inside = cv2.resize(inside, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        er = max(3, int(0.08 * short * s))
        interior = cv2.erode(inside, np.ones((er, er), np.uint8))   # the banner's own outline doesn't count
        n = cv2.countNonZero(interior)
        if n < 50:
            return 0.0, 0.0
        lo, hi = np.percentile(gray[interior > 0], (2, 98))
        contrast = (hi - lo) / 255.0
        if contrast < self.a.min_text_contrast:                       # plain patch: nothing to stretch but noise
            return 0.0, contrast
        g = np.clip((gray.astype(np.float32) - lo) / (hi - lo), 0, 1)
        R = cv2.cornerHarris(g, 3, 3, 0.04)
        peaks = (R > self.a.harris_thresh) & (R >= cv2.dilate(R, np.ones((5, 5), np.uint8))) & (interior > 0)
        return float(peaks.sum()) / (n / 1000.0), contrast

    def refine_corners(self, local, banner, X0, Y0, short):
        """Full-resolution corners -> (quad in image pixels, number of Harris-confirmed corners).
        `banner` (crop mask) = colour + the letters inside the rough quad, so text running into an edge makes no
        corners of its own. Opened (sensor-noise speckle touching the banner), closed, outline re-fitted as 4 edge
        lines (fit_quad) - the rough quad came from a 1/2.5 image. Then each corner is checked against the Harris
        response of the mask: a peak within a few px confirms it and is refined to sub-pixel; no peak there (a
        corner hidden, cut off or rounded) keeps the edge-line intersection."""
        k = max(5, int(0.05 * short) | 1)                       # rounds the corners: the edge lines restore them
        m = cv2.morphologyEx(banner, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        k = max(5, int(0.06 * short) | 1)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        lines = local.copy()
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        centre = tuple(float(v) for v in local.mean(axis=0))
        mine = [c for c in contours if cv2.pointPolygonTest(c, centre, False) >= 0]
        if mine:
            c = max(mine, key=cv2.contourArea)
            # edge lines through the blob's outline AND its hull's (traced densely): where letters notch an edge the
            # hull spans the gap, where noise bulges out of it the blob doesn't - the RANSAC line both agree on wins
            hull = cv2.convexHull(c)
            filled = np.zeros_like(m)
            cv2.fillConvexPoly(filled, hull, 255)
            dense = max(cv2.findContours(filled, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=cv2.contourArea)
            q, _ = fit_quad(np.vstack([c, dense]), hull)
            ok = np.linalg.norm(q - local, axis=1) < 0.25 * short          # same banner, not a merged neighbour
            lines[ok] = q[ok]
        mf = cv2.GaussianBlur(m, (5, 5), 0).astype(np.float32) / 255.0
        R = cv2.cornerHarris(mf, 5, 3, 0.04)
        r = int(np.clip(0.04 * short, 3, 12))
        floor = 0.1 * max(float(R.max()), 1e-9)
        out, confirmed = lines.copy(), np.zeros(4, bool)
        for i, (u, v) in enumerate(lines):
            u0, v0 = int(round(u)), int(round(v))
            ya, yb, xa, xb = max(0, v0 - r), min(R.shape[0], v0 + r + 1), max(0, u0 - r), min(R.shape[1], u0 + r + 1)
            win = R[ya:yb, xa:xb]
            if win.size and win.max() > floor:
                j = np.unravel_index(np.argmax(win), win.shape)
                out[i], confirmed[i] = (xa + j[1], ya + j[0]), True
        if confirmed.any():
            pts = out[confirmed].reshape(-1, 1, 2).astype(np.float32)
            cv2.cornerSubPix(mf, pts, (r, r), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01))
            out[confirmed] = pts.reshape(-1, 2)
            drift = np.linalg.norm(out - lines, axis=1) > 2 * r              # wandered off: trust the edge lines
            out[drift] = lines[drift]
            confirmed &= ~drift
        return order_quad(out + [X0, Y0]), int(confirmed.sum())


class Tracker:
    """Per-colour tracks: matched by centre distance, corners smoothed, confirmed after `confirm` hits, held for
    `hold_s` after the last hit (time, not frames: the detection rate depends on the load). Detections cut off by
    the frame edge only extend existing tracks."""

    def __init__(self, confirm=2, hold_s=0.5, alpha=0.5):
        self.tracks, self.confirm, self.hold_s, self.alpha, self.next_id = [], confirm, hold_s, alpha, 1

    def update(self, dets, now=None):
        now = time.monotonic() if now is None else now
        for t in self.tracks:
            t['matched'] = False
        for d in dets:
            best, best_dist = None, None
            for t in self.tracks:
                if t['matched'] or t['colour'] != d['colour']:
                    continue
                dist = np.linalg.norm(t['centre'] - d['centre'])
                gate = 0.6 * math.sqrt(max(t['det']['area'], d['area']))
                # or its centre on the track's banner: a long thin banner moves a lot relative to sqrt(area), and
                # the visible part of one leaving the frame has its centre shifted towards the inside
                on_it = cv2.pointPolygonTest(t['quad'].astype(np.float32), tuple(map(float, d['centre'])), False) >= 0
                if (dist < gate or on_it) and (best is None or dist < best_dist):
                    best, best_dist = t, dist
            if best is None:
                if d['partial']:
                    # cut off by the frame edge its true shape is unknown - a poster's visible part can look like a
                    # banner's: it may only continue a banner already seen whole, never start one
                    continue
                best = {'id': self.next_id, 'colour': d['colour'], 'hits': 0, 'miss': 0, 'quad': d['quad'].copy()}
                self.next_id += 1
                self.tracks.append(best)
            else:
                best['quad'] = (1 - self.alpha) * best['quad'] + self.alpha * d['quad']
            best.update(det=d, centre=best['quad'].mean(axis=0), matched=True, miss=0, hits=best['hits'] + 1,
                        seen=now)
        for t in self.tracks:
            if not t['matched']:
                t['miss'] += 1
        self.tracks = [t for t in self.tracks if now - t['seen'] <= self.hold_s]
        return [t for t in self.tracks if t['hits'] >= self.confirm]


class DetectWorker:
    """Banner detection on a thread of its own, always on the newest frame offered (frames offered while it is busy
    are skipped): the video is shown at remap speed and the outlines refresh as fast as detection runs. OpenCV / VPI
    release the GIL, so the two really run in parallel."""

    def __init__(self, det):
        self.det, self.tracker, self.wb = det, Tracker(), WhiteBalance()
        self.job, self.cond, self.running = None, threading.Condition(), True
        self.tracks, self.masks, self.lut, self.ms, self.n = [], {}, None, 0.0, 0
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def offer(self, view, rect, use_wb):
        with self.cond:
            self.job = (view, rect, use_wb)                   # replaces a job not yet started: newest wins
            self.cond.notify()

    def wait_idle(self, timeout=5.0):
        """Block until the offered frame is processed (headless / tests: every frame detected)."""
        with self.cond:
            self.cond.wait_for(lambda: self.job is None and not self.busy, timeout)

    busy = False

    def loop(self):
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.job is not None or not self.running)
                if not self.running:
                    return
                (view, rect, use_wb), self.job, self.busy = self.job, None, True
            t0 = time.monotonic()
            try:
                small, vsmall = rect.downscale(view, self.det.a.proc_width)
                self.wb.update(small, vsmall)
                lut = self.wb.lut()
                dets = self.det.detect(view, rect.valid, cv2.LUT(small, lut) if use_wb else small, vsmall,
                                       lut if use_wb else None, rect.P)
                tracks = self.tracker.update(dets)
            except Exception:                                 # keep the video going; say what broke
                import traceback
                traceback.print_exc()
                tracks, lut = self.tracks, self.lut
            with self.cond:
                self.tracks, self.masks, self.lut = tracks, dict(self.det.last_masks), lut
                self.ms, self.n, self.busy = 1e3 * (time.monotonic() - t0), self.n + 1, False
                self.cond.notify_all()

    def stop(self):
        with self.cond:
            self.running = False
            self.cond.notify_all()
        self.thread.join(timeout=3.0)


class RemapWorker:
    """Fisheye -> rectilinear on a thread of its own: the remap of frame N+1 runs while the main thread draws and shows
    frame N, so the display is not held back by the remap on top of everything else. The Rectifier is (re)built here
    too, on the thread that then uses it (VPI / CUDA)."""

    def __init__(self, src, args):
        self.src, self.args = src, args
        self.cond, self.running = threading.Condition(), True
        self.out, self.seq, self.rect = None, 0, None          # out = (frame, view, rect, remap seconds)
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def loop(self):
        a, src_seq = self.args, 0
        while self.running:
            frame, src_seq = self.src.wait(src_seq, timeout=0.5)
            if frame is None:
                continue
            try:
                rect = self.rect
                if rect is None or rect.model.size != (frame.shape[1], frame.shape[0]):
                    rect = Rectifier(a.calibration, (frame.shape[1], frame.shape[0]), a.hfov, a.vfov, a.zoom,
                                     a.backend, a.interp)
                    self.rect = rect
                t0 = time.monotonic()
                view = rect(frame)
                out = (frame, view, rect, time.monotonic() - t0)
            except Exception:                                 # say what broke, keep going on the next frame
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                continue
            with self.cond:
                self.out, self.seq = out, self.seq + 1
                self.cond.notify_all()

    def wait(self, last_seq, timeout=2.0):
        """(frame, view, rect, remap s), seq - newer than last_seq; (None, last_seq) on timeout."""
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq or not self.running, timeout)
            return (self.out, self.seq) if self.seq != last_seq else (None, last_seq)

    def stop(self):
        with self.cond:
            self.running = False
            self.cond.notify_all()
        self.thread.join(timeout=3.0)


# ------------------------------------------------------------------ drawing
def draw(view, tracks, masks, show_masks, hud_lines, hud):
    H, W = view.shape[:2]
    th = max(2, W // 600)
    fs = max(0.6, W / 1600)
    if show_masks:
        for colour, m in masks.items():
            big = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0
            view[big] = (0.5 * view[big] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    for t in tracks:
        d, q = t['det'], t['quad']
        c = COLOURS[t['colour']]
        stale = t['miss'] > 0
        cv2.polylines(view, [q.astype(np.int32)], True, (0, 255, 255) if stale else c, th + 1, cv2.LINE_AA)
        for p in q:
            cv2.circle(view, tuple(int(v) for v in p), th * 3, (0, 0, 255), -1, cv2.LINE_AA)
        cx, cy = (int(v) for v in t['centre'])
        cv2.drawMarker(view, (cx, cy), (0, 0, 255), cv2.MARKER_CROSS, th * 10, th)
        label = f'{t["colour"].upper()} BANNER {d["conf"]:.2f}' + (' (edge)' if d['partial'] else '') + \
            (f'  {d["distance_m"]:.1f} m' if d.get('distance_m') else '')
        org = (int(q[:, 0].min()), max(int(fs * 30), int(q[:, 1].min()) - th * 4))
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        cv2.rectangle(view, (org[0], org[1] - tht - th * 3), (org[0] + tw + th * 2, org[1] + th * 2), (0, 0, 0), -1)
        cv2.putText(view, label, (org[0] + th, org[1]), cv2.FONT_HERSHEY_SIMPLEX, fs, c, th, cv2.LINE_AA)
    if hud:
        for i, line in enumerate(hud_lines):
            y = int(fs * 34 * (i + 1))
            cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (0, 0, 0), th + 3, cv2.LINE_AA)
            cv2.putText(view, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, fs * 0.8, (255, 255, 0), th, cv2.LINE_AA)
    return view


def fit_to_window(img):
    """Scale (area-averaged, crisp) to the window's current size; imshow's own scaling is slow on 3 MP frames."""
    try:
        _, _, ww, wh = cv2.getWindowImageRect(WINDOW)
    except cv2.error:
        ww = wh = -1
    if ww <= 0 or wh <= 0:
        return img
    s = min(ww / img.shape[1], wh / img.shape[0])
    if s >= 1.0:
        return img
    return cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)),
                      interpolation=cv2.INTER_AREA if s < 0.5 else cv2.INTER_LINEAR)


def make_tuning(det):
    cv2.namedWindow(TUNING, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(TUNING, 520, 330)
    for colour in COLOURS:
        for i, name in enumerate(('H lo', 'H hi', 'S min', 'V min')):
            cv2.createTrackbar(f'{colour} {name}', TUNING, det.hsv[colour][i], 180 if i < 2 else 255, lambda _: None)


def read_tuning(det):
    try:
        for colour in COLOURS:
            det.hsv[colour] = [cv2.getTrackbarPos(f'{colour} {n}', TUNING) for n in ('H lo', 'H hi', 'S min', 'V min')]
    except cv2.error:
        return False                                          # tuning window closed
    return True


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group('camera')
    g.add_argument('--source', default='auto', help='auto | http | csi | <image or video file> (default auto)')
    g.add_argument('--mode', choices=MODES, default='full', help='CSI mode: full 3280x2464@21, half 1640x1232@30')
    g.add_argument('--sensor-id', type=int, default=0)
    g.add_argument('--flip', type=int, default=0, help='nvvidconv flip-method (2 = rotate 180)')
    g.add_argument('--max-exposure-ms', type=float, default=20.0, help='exposure cap (motion blur vs noise)')
    g.add_argument('--http-host', default='127.0.0.1')
    g.add_argument('--http-port', type=int, default=8090, help="arducam_calib.py's shared /frame.jpg")
    g = p.add_argument_group('dewarp')
    g.add_argument('--calibration', default=DEFAULT_CALIB, help='equidistant lens YAML (default: %(default)s)')
    g.add_argument('--hfov', type=float, default=120.0, help='output horizontal field of view, deg (< 180)')
    g.add_argument('--vfov', type=float, default=90.0, help='output vertical field of view, deg (< 180)')
    g.add_argument('--zoom', type=float, default=1.0,
                   help='output focal / lens focal: 1 = sensor-native sampling at the centre; >1 only upsamples')
    g.add_argument('--backend', choices=('auto', 'vpi', 'cpu'), default='auto')
    g.add_argument('--interp', choices=('auto', 'cubic', 'linear'), default='auto',
                   help='remap interpolation; auto = cubic on the GPU (VPI), linear on the CPU')
    g = p.add_argument_group('detection')
    g.add_argument('--green-h', type=int, nargs=2, default=[45, 99], metavar=('LO', 'HI'),
                   help='OpenCV hue 0-180 (the real banner H 81-86 teal green, grass H 19-46: 2026-10-30 photo)')
    g.add_argument('--blue-h', type=int, nargs=2, default=[95, 135], metavar=('LO', 'HI'))
    g.add_argument('--rim-min', type=float, default=0.35, help='white share of the band round a banner (white_frame.py)')
    g.add_argument('--rim-min-px', type=float, default=12.0, help='banner height from which that frame must be seen')
    g.add_argument('--blue-s-min', type=int, default=140,
                   help='blue only: the sky is H ~106, S ~96-113 (2026-10-30 photo) - a printed banner is far more saturated')
    g.add_argument('--red-h', type=int, nargs=2, default=[170, 8], metavar=('LO', 'HI'),
                   help='LO > HI: round hue 0 (H >= LO or H <= HI)')
    g.add_argument('--ground-suppress', action='store_true',
                   help='green over grass: drop the dominant (ground) green, keep what differs from it or is solid')
    g.add_argument('--ground-frac', type=float, default=0.15, help='green share of the view that means grass')
    g.add_argument('--ground-ds', type=float, default=40.0, help='saturation above the grass median that is banner')
    g.add_argument('--ground-dv', type=float, default=45.0, help='brightness above the grass median that is banner')
    g.add_argument('--ground-dh', type=float, default=12.0, help='hue away from the grass median that is banner')
    g.add_argument('--s-min', type=int, default=80, help='minimum saturation of the banner colour')
    g.add_argument('--v-min', type=int, default=35, help='minimum brightness of the banner colour')
    g.add_argument('--proc-width', type=int, default=1224,
                   help='width of the working copy for masking / contours (1/2 scale: banners ~20 m away still ~7 px)')
    g.add_argument('--min-area-frac', type=float, default=0.0002, help='smallest banner, fraction of the image area')
    g.add_argument('--min-fill', type=float, default=0.72, help='blob area / fitted quadrilateral area')
    g.add_argument('--banner-size', type=float, nargs=2, default=[11.5, 0.73], metavar=('W', 'H'),
                   help='ft: the banner coloured field (the white border not counted). The real one (photo 2026-10-30, '
                        'face-on): 15.8 : 1 - the old 11.5 x 1.5 ft (7.7 : 1) rejected it')
    g.add_argument('--aspect-tol', type=float, default=1.4,
                   help='true aspect may be off by this factor either way (corner noise on small / far banners)')
    g.add_argument('--min-partial-aspect', type=float, default=3.5,
                   help='true aspect of the visible part of a banner cut off by the frame edge')
    g.add_argument('--min-aspect', type=float, default=1.5,
                   help='quick pre-filter on the apparent long / short side, before the true-aspect check')
    g.add_argument('--max-aspect', type=float, default=40.0)
    g.add_argument('--min-solid', type=float, default=0.8,
                   help='plain banner: fraction of it in the banner colour (a banner is one solid colour)')
    g.add_argument('--with-letters', action='store_true',
                   help='banners carry white letters (the sim\'s "AeroTHON 2026"): require them; options below')
    g.add_argument('--white-s-max', type=int, default=100, help='letters: saturation below this ...')
    g.add_argument('--text-v-ratio', type=float, default=0.85, help="... and V >= this x the banner colour's V")
    g.add_argument('--white-min', type=float, default=0.03, help='white-letter fraction of the banner, min ...')
    g.add_argument('--white-max', type=float, default=0.45, help='... and max')
    g.add_argument('--harris-thresh', type=float, default=0.008,
                   help=f'Harris response of a letter corner (contrast-stretched grey 0-1, short side {TEXT_SCALE} px)')
    g.add_argument('--min-corner-density', type=float, default=1.0,
                   help='Harris corners per 1000 px^2 inside for full text confidence (not a pass / fail test)')
    g.add_argument('--min-text-contrast', type=float, default=0.12, help='grey p98 - p2 inside the banner (0-1)')
    g.add_argument('--min-text-px', type=float, default=30,
                   help='banners less tall than this (px) are too far for their letters to be checked')
    g = p.add_argument_group('output')
    g.add_argument('--no-wb', dest='wb', action='store_false', help='start with white balance off (key w)')
    g.add_argument('--save-dir', default=str(Path.cwd() / 'banner_view_saves'))
    g.add_argument('--headless', action='store_true', help='no window: log detections (e.g. over ssh)')
    g.add_argument('--max-frames', type=int, default=0, help='stop after this many processed frames (0 = never)')
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.headless and not os.environ.get('DISPLAY') and not os.environ.get('WAYLAND_DISPLAY'):
        raise SystemExit('no display: run it on the Jetson desktop (or ssh -X / export DISPLAY=:0), or use --headless')
    if not args.calibration:
        print('WARN: no lens calibration found - nominal 185 deg lens, straight lines will bend. Pass --calibration.')
    for name, v in (('hfov', args.hfov), ('vfov', args.vfov)):
        if not 10 <= v <= 170:
            raise SystemExit(f'--{name} {v}: a rectilinear view needs 10-170 deg (180 would be infinitely wide)')

    def on_term(*_):
        raise KeyboardInterrupt                           # SIGTERM too: release the camera cleanly (Argus)
    signal.signal(signal.SIGTERM, on_term)

    src = FrameSource(args)
    shared = f' (http://{args.http_host}:{args.http_port}/frame.jpg)' if src.kind == 'http' else ''
    print(f'INFO: source {src.kind}{shared}')
    det = BannerDetector(args)
    worker = DetectWorker(det)
    remap = RemapWorker(src, args)
    rect, seq, n, t_last_log = None, 0, 0, 0.0
    show_masks, use_wb, hud, tuning, can_close = False, args.wb, True, False, None
    fps_show, fps_det, n_det, t_prev, t_rate = 0.0, 0.0, 0, time.monotonic(), time.monotonic()
    if not args.headless:
        cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    try:
        while True:
            out, seq = remap.wait(seq)
            if out is None:
                print('WARN: no frame for 2 s')
                if not args.headless and (cv2.waitKey(1) & 0xFF) in (ord('q'), 27):
                    break
                continue
            frame, view, new_rect, t_remap = out
            if new_rect is not rect:
                rect = new_rect
                if not args.headless:
                    W, H = rect.size
                    s = min(1.0, 1600 / W, 900 / H)
                    cv2.resizeWindow(WINDOW, int(W * s), int(H * s))
            if tuning and not read_tuning(det):
                tuning = False
            worker.offer(view, rect, use_wb)
            if args.headless:
                worker.wait_idle()                            # nothing to show: detect every frame
            tracks, masks, lut, det_ms = worker.tracks, worker.masks, worker.lut if use_wb else None, worker.ms
            now = time.monotonic()
            fps_show = 0.9 * fps_show + 0.1 / max(1e-3, now - t_prev) if fps_show else 1 / max(1e-3, now - t_prev)
            t_prev = now
            if now - t_rate > 1.0:
                fps_det, n_det, t_rate = (worker.n - n_det) / (now - t_rate), worker.n, now
            n += 1

            if now - t_last_log > 1.0 and (tracks or args.headless):
                t_last_log = now
                summary = ', '.join(f'{t["colour"]} #{t["id"]} conf {t["det"]["conf"]:.2f} centre '
                                    f'({t["centre"][0]:.0f},{t["centre"][1]:.0f})'
                                    f'{"" if t["miss"] == 0 else " (held)"}' for t in tracks) or 'no banner'
                print(f'{fps_show:4.1f} fps (detect {det_ms:.0f} ms) | {summary}', flush=True)
            if not args.headless:
                hud_lines = [f'{rect.size[0]}x{rect.size[1]} rectilinear {args.hfov:.0f}x{args.vfov:.0f} deg | '
                             f'camera {src.fps():.1f} fps, shown {fps_show:.1f} fps, detection {fps_det:.1f} fps | '
                             f'remap {1e3 * t_remap:.0f} ms ({rect.backend} {rect.interp_name}), detect {det_ms:.0f} ms',
                             'banners: ' + (', '.join(f'{t["colour"]} {t["det"]["conf"]:.2f}' for t in tracks)
                                            or 'none') + f' | WB {"on" if use_wb else "off"} | '
                             'q quit  s save  m mask  w WB  t tune  h HUD']
                shown = fit_to_window(view)                   # draw at screen size: crisp lines, cheap
                if lut is not None:
                    shown = cv2.LUT(shown, lut)
                elif shown is view:
                    shown = view.copy()                       # `view` stays clean for saving
                scale = shown.shape[1] / view.shape[1]
                scaled_tracks = [{**t, 'quad': t['quad'] * scale, 'centre': t['centre'] * scale} for t in tracks]
                cv2.imshow(WINDOW, draw(shown, scaled_tracks, masks, show_masks, hud_lines, hud))
                key = cv2.waitKey(1) & 0xFF
                if can_close is None:                         # GTK builds report -1 even while open: then q / Esc
                    can_close = cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 0
                if key in (ord('q'), 27) or (can_close and cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1):
                    break
                if key == ord('s'):
                    out = Path(args.save_dir)
                    out.mkdir(parents=True, exist_ok=True)
                    stamp = time.strftime('%Y%m%d_%H%M%S')
                    cv2.imwrite(str(out / f'{stamp}_fisheye.png'),
                                cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR) if frame.shape[2] == 4 else frame)
                    full = view.copy() if lut is None else cv2.LUT(view, lut)
                    cv2.imwrite(str(out / f'{stamp}_rect.png'), full)
                    cv2.imwrite(str(out / f'{stamp}_annotated.png'),
                                draw(full, tracks, masks, show_masks, hud_lines, True))
                    print(f'INFO: saved {out}/{stamp}_*.png')
                elif key == ord('m'):
                    show_masks = not show_masks
                elif key == ord('w'):
                    use_wb = not use_wb
                elif key == ord('h'):
                    hud = not hud
                elif key == ord('t'):
                    tuning = not tuning
                    if tuning:
                        make_tuning(det)
                    else:
                        cv2.destroyWindow(TUNING)
                        print('INFO: tuned: --green-h {} {} --blue-h {} {}  (S/V min green {} {}, blue {} {})'.format(
                            *det.hsv['green'][:2], *det.hsv['blue'][:2], *det.hsv['green'][2:], *det.hsv['blue'][2:]))
            if args.max_frames and n >= args.max_frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        remap.stop()
        worker.stop()
        src.stop()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
