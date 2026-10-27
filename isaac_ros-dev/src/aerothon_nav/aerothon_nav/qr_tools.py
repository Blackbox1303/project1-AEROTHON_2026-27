"""QR detection and decoding helpers shared by qr_locator and qr_reader:
  WeChat     OpenCV's WeChat QR reader (opencv contrib, in the dev container's cv2 4.5.4): a CNN detector +
             super-resolution in front of the decode. Includes quiet-zone border replication for tight crops.
  QrNet      the same detector network on its own (detect.caffemodel through cv2.dnn): QR bounding boxes with a
             confidence, supporting multi-scale tile scanning (160, 224, 384 px) for 55-80 px QR codes at 10 m
             altitude, plus dark-square contour refinement (`refine_qr_box`) to exclude paper borders.
  transitions(crop)   black/white changes per row and per column after Otsu: a QR has many, an edge one
  aligned_mean(crops) frames of one spot averaged after phase-correlation alignment
"""
import math
from pathlib import Path
import threading

import cv2
import numpy as np

DEFAULT_DIR = '/workspaces/isaac_ros-dev/calibration/wechat_qrcode'
FILES = ('detect.prototxt', 'detect.caffemodel', 'sr.prototxt', 'sr.caffemodel')
_dnn_lock = threading.Lock()


def model_dir(path=''):
    """The model directory: path, else the container's, else the host's copy of it; None if the files aren't there."""
    for d in (path, DEFAULT_DIR, str(Path.home() / 'workspaces/isaac_ros-dev/calibration/wechat_qrcode')):
        if d and all((Path(d) / f).exists() for f in FILES):
            return Path(d)
    return None


def make_wechat(path=''):
    """cv2.wechat_qrcode_WeChatQRCode, or None (no contrib module / no model files)."""
    d = model_dir(path)
    if d is None or not hasattr(cv2, 'wechat_qrcode_WeChatQRCode'):
        return None
    try:
        return cv2.wechat_qrcode_WeChatQRCode(*(str(d / f) for f in FILES))
    except cv2.error:
        return None


def wechat_decode(wc, gray, tile=768, overlap=160, x0=0, y0=0):
    """{payload: polygon (4 x 2, in the full image)} from WeChat over tiles of gray (with border padding for small crops)."""
    if wc is None or gray is None or gray.size == 0:
        return {}
    with _dnn_lock:
        out = {}
        h, w = gray.shape[:2]
        if max(h, w) <= 320:
            pad = 32
            padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
            try:
                texts, pts = wc.detectAndDecode(padded)
                for t, p in zip(texts, pts):
                    t = t.strip()
                    if t and t not in out and len(p) >= 4:
                        out[t] = np.asarray(p, float).reshape(-1, 2) - [pad - x0, pad - y0]
                if out:
                    return out
            except Exception:
                pass
        step = max(64, tile - overlap)
        for y in range(0, max(1, h - overlap), step):
            for x in range(0, max(1, w - overlap), step):
                tx, ty = max(0, min(x, w - tile)), max(0, min(y, h - tile))
                try:
                    texts, pts = wc.detectAndDecode(gray[ty:ty + tile, tx:tx + tile])
                except Exception:
                    continue
                for t, p in zip(texts, pts):
                    t = t.strip()
                    if t and t not in out and len(p) >= 4:
                        out[t] = np.asarray(p, float).reshape(-1, 2) + [tx + x0, ty + y0]
        return out


def refine_qr_box(gray, poly):
    """Refine a candidate/QrNet bounding box around the inner high-contrast dark QR pattern (excluding white paper margin)."""
    h, w = gray.shape[:2]
    x0, y0 = poly.min(axis=0)
    x1, y1 = poly.max(axis=0)
    bw, bh = x1 - x0, y1 - y0
    if bw < 20 or bh < 20 or bw > 240 or bh > 240:
        return poly
    X0, Y0 = max(0, int(x0)), max(0, int(y0))
    X1, Y1 = min(w, int(x1)), min(h, int(y1))
    roi = gray[Y0:Y1, X0:X1]
    if roi.size < 100:
        return poly
    med = float(np.median(roi))
    _, dark = cv2.threshold(roi, int(max(30, min(190, med * 0.88))), 255, cv2.THRESH_BINARY_INV)
    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    ys, xs = np.where(dark > 0)
    if len(xs) < 25:
        return poly
    rx0, rx1 = float(np.percentile(xs, 3)) + X0, float(np.percentile(xs, 97)) + X0
    ry0, ry1 = float(np.percentile(ys, 3)) + Y0, float(np.percentile(ys, 97)) + Y0
    rw, rh = rx1 - rx0, ry1 - ry0
    if 24 <= max(rw, rh) <= 180 and 0.65 <= rw / max(rh, 1.0) <= 1.50:
        return np.array([[rx0, ry0], [rx1, ry0], [rx1, ry1], [rx0, ry1]], dtype=np.float64)
    return poly


class QrNet:
    """WeChat's QR detection network alone: boxes + confidence."""

    def __init__(self, path=''):
        d = model_dir(path)
        self.net = cv2.dnn.readNetFromCaffe(str(d / 'detect.prototxt'), str(d / 'detect.caffemodel')) if d else None

    def boxes(self, crop, x0=0, y0=0, conf=0.5):
        """[(confidence, polygon 4 x 2 in the full image)] on one crop (resized to the 384 x 384 input)."""
        if self.net is None or crop.size == 0:
            return []
        with _dnn_lock:
            try:
                self.net.setInput(cv2.dnn.blobFromImage(crop, 1.0 / 255, (384, 384), 0, False, False))
                h, w = crop.shape[:2]
                out = []
                res = self.net.forward()
                for d in res.reshape(-1, 7):
                    if d[2] >= conf:
                        x1, y1, x2, y2 = d[3] * w + x0, d[4] * h + y0, d[5] * w + x0, d[6] * h + y0
                        if x2 > x1 + 6 and y2 > y1 + 6:
                            out.append((float(d[2]), np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)))
                return out
            except Exception:
                return []

    def scan(self, gray, conf=0.5, centres=None, tile=384, overlap=96, multi_scale=True):
        """Scan tiles across `gray` or around `centres` (with multi-scale fine tiles 160/224/384 px when centres given)."""
        h, w = gray.shape[:2]
        found = []
        if centres is None:
            step = max(64, tile - overlap)
            spots = [(max(0, min(x, w - tile)), max(0, min(y, h - tile)), tile)
                     for y in range(0, max(1, h - overlap), step) for x in range(0, max(1, w - overlap), step)]
        else:
            tiles = (160, 224, tile) if multi_scale else (tile,)
            spots = []
            for t_sz in dict.fromkeys(tiles):
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
            if all(iou(p_ref, q) < 0.3 for _, q in keep):
                keep.append((c, p_ref))
        return keep


def iou(a, b):
    ax0, ay0 = a.min(axis=0)
    ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0)
    bx1, by1 = b.max(axis=0)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    return inter / max(1e-6, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def transitions(crop):
    """(changes per row, changes per column) of the Otsu-thresholded crop: a QR has many, a straight edge one."""
    if crop.size < 16:
        return 0.0, 0.0
    c = crop if crop.dtype == np.uint8 else np.clip(crop, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(c, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bw.astype(np.int8)
    return (float(np.abs(np.diff(bw, axis=1)).sum(axis=1).mean()),
            float(np.abs(np.diff(bw, axis=0)).sum(axis=0).mean()))


def contrast_threshold(sd, floor=6.0, k=4.0):
    """A local-contrast threshold from the frame itself: median + k robust sigma (MAD) of the std map, >= floor."""
    valid = sd[sd > 0.5]
    if valid.size == 0:
        return float(floor)
    med = float(np.median(valid))
    return max(floor, med + k * 1.4826 * (float(np.median(np.abs(valid - med))) + 1e-3))


def find_qr_candidates(gray, fisheye_mask=None, min_transitions=3.0, max_cands=12):
    """Find high-contrast square-ish QR candidate regions inside the active fisheye circle."""
    h, w = gray.shape[:2]
    scale = 0.5 if w > 1200 else 1.0
    g_small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else gray
    g_f = cv2.GaussianBlur(g_small.astype(np.float32), (0, 0), 1.2)
    mu = cv2.boxFilter(g_f, -1, (5, 5))
    sd = np.sqrt(np.maximum(cv2.boxFilter(g_f * g_f, -1, (5, 5)) - mu * mu, 0.0))

    if fisheye_mask is not None:
        m_small = (
            cv2.resize(fisheye_mask.astype(np.uint8), (g_small.shape[1], g_small.shape[0]), interpolation=cv2.INTER_NEAREST)
            if scale != 1.0 else fisheye_mask.astype(np.uint8)
        )
        m_eroded = cv2.erode(m_small, np.ones((9, 9), np.uint8))
        sd = sd * m_eroded

    thr = contrast_threshold(sd, floor=6.0)
    blobs = []
    for level in (1.0, 1.6, 2.5, 4.0):
        mask = (sd > thr * level).astype(np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            if area < 8 or area > 16000:
                continue
            aspect = max(bw, bh) / max(min(bw, bh), 1)
            if aspect > 2.5:
                continue
            if min(bw, bh) >= 8:
                tr = transitions(g_small[by:by + bh, bx:bx + bw])
                if min(tr) < min_transitions:
                    continue
            score = float(sd[by:by + bh, bx:bx + bw].mean())
            if any(math.dist(cent[i], c) < 0.5 * max(bw, bh) for _, c, _ in blobs):
                continue
            blobs.append((stats[i], cent[i], score))

    blobs.sort(key=lambda t: -t[2])
    inv = 1.0 / scale
    out = []
    for (bx, by, bw, bh, _), (cx, cy), score in blobs[:max_cands]:
        poly = np.array([
            [bx * inv, by * inv],
            [(bx + bw) * inv, by * inv],
            [(bx + bw) * inv, (by + bh) * inv],
            [bx * inv, (by + bh) * inv],
        ], dtype=np.float64)
        poly = refine_qr_box(gray, poly)
        x0, y0 = poly.min(axis=0)
        x1, y1 = poly.max(axis=0)
        out.append({
            'center': (float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)),
            'bbox': (float(x0), float(y0), float(x1 - x0), float(y1 - y0)),
            'poly': poly,
            'score': score,
        })
    return out


def aligned_mean(crops):
    """Mean of same-size grey crops after aligning each to the first (phase correlation, whole pixels)."""
    ref = crops[0].astype(np.float32)
    acc, n = ref.copy(), 1
    win = cv2.createHanningWindow(ref.shape[::-1], cv2.CV_32F)
    for c in crops[1:]:
        f = c.astype(np.float32)
        if f.shape != ref.shape:
            continue
        (dx, dy), resp = cv2.phaseCorrelate(ref, f, win)
        if resp < 0.05 or abs(dx) > ref.shape[1] / 4 or abs(dy) > ref.shape[0] / 4:
            continue
        m = np.float32([[1, 0, -dx], [0, 1, -dy]])
        acc += cv2.warpAffine(f, m, ref.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
        n += 1
    return np.clip(acc / n, 0, 255).astype(np.uint8), n
