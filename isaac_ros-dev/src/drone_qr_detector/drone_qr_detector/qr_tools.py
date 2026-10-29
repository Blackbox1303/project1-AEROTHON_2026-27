"""QR detection and decoding toolkit for full-resolution (3280x2464) fisheye and gnomonic-rectified patches.

Enhancements over aerothon_nav/qr_tools.py:
- Circular-fisheye mask in contrast candidate detection (prevents false candidates on the lens's black vignette ring).
- Combined multi-decoder (`decode_gray`) running WeChat CNN + super-resolution, pyzbar (1x and 2x upscaled on patches),
  and OpenCV's QR detector, returning exact 4-corner polygons.
- `QrNet` SSD detector + Otsu transition / two-tone scoring to find distant (10 m) or oblique QR boards before they
  can be decoded on the wide view, so the localizer can rectify a high-resolution gnomonic patch directly along each
  candidate's 3D ray.
"""
from pathlib import Path
import cv2
import numpy as np

DEFAULT_DIR = '/workspaces/isaac_ros-dev/calibration/wechat_qrcode'
FILES = ('detect.prototxt', 'detect.caffemodel', 'sr.prototxt', 'sr.caffemodel')


def model_dir(path=''):
    """Locate the WeChat QR Caffe model directory inside the container or on the host."""
    candidates = [
        path,
        DEFAULT_DIR,
        str(Path.home() / 'workspaces/isaac_ros-dev/calibration/wechat_qrcode'),
        '/home/szyfrowac/workspaces/isaac_ros-dev/calibration/wechat_qrcode',
    ]
    for d in candidates:
        if d and all((Path(d) / f).exists() for f in FILES):
            return Path(d)
    return None


def make_wechat(path=''):
    """Return cv2.wechat_qrcode_WeChatQRCode instance, or None if unavailable."""
    d = model_dir(path)
    if d is None or not hasattr(cv2, 'wechat_qrcode_WeChatQRCode'):
        return None
    try:
        return cv2.wechat_qrcode_WeChatQRCode(*(str(d / f) for f in FILES))
    except cv2.error:
        return None


def make_zbar():
    """Return a callable `zbar(gray)` or None if pyzbar is not installed."""
    try:
        from pyzbar.pyzbar import ZBarSymbol, decode
        return lambda img: decode(img, symbols=[ZBarSymbol.QRCODE])
    except ImportError:
        return None


def wechat_decode(wc, gray, tile=768, overlap=160, x0=0, y0=0):
    """Run WeChat QR detector + super-resolution over overlapping tiles (or a single crop if <= tile)."""
    out = {}
    if wc is None or gray is None or gray.size == 0:
        return out
    h, w = gray.shape[:2]
    if w <= tile and h <= tile:
        try:
            # Pad tight crops so WeChat's internal 384x384 SSD detector has quiet-zone context
            pad = 32 if max(w, h) < 280 else 0
            work_img = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_REPLICATE) if pad else gray
            texts, pts = wc.detectAndDecode(work_img)
        except cv2.error:
            return out
        for t, p in zip(texts, pts):
            t = t.strip()
            if t and t not in out:
                out[t] = np.asarray(p, dtype=np.float64).reshape(-1, 2) + [x0 - pad, y0 - pad]
        return out

    step = max(64, tile - overlap)
    ys = list(range(0, max(1, h - overlap), step))
    xs = list(range(0, max(1, w - overlap), step))
    for y in ys:
        for x in xs:
            tx = max(0, min(x, w - tile))
            ty = max(0, min(y, h - tile))
            try:
                texts, pts = wc.detectAndDecode(gray[ty:ty + tile, tx:tx + tile])
            except cv2.error:
                continue
            for t, p in zip(texts, pts):
                t = t.strip()
                if t and t not in out:
                    out[t] = np.asarray(p, dtype=np.float64).reshape(-1, 2) + [tx + x0, ty + y0]
    return out


def decode_gray(gray, wc=None, zbar=None, cv_qr=None, upscale_small=True):
    """Decode all QR codes in a grayscale image/patch -> {payload: polygon (4x2 float64)}."""
    if gray is None or gray.size == 0:
        return {}
    work = gray
    mean_val = float(work.mean())
    if mean_val < 75.0 or mean_val > 205.0:
        work = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(work)

    out = {}
    if wc is not None:
        out.update(wechat_decode(wc, work))

    # Run zbar as well (on multiple QR codes on a wall, WeChat + zbar together maximize recall)
    if zbar is not None:
        blurred = cv2.GaussianBlur(work, (3, 3), 0)
        scales = [1.0]
        if upscale_small and max(work.shape[:2]) <= 1200:
            scales.append(2.0)
        for sc in scales:
            img_s = blurred if sc == 1.0 else cv2.resize(blurred, None, fx=sc, fy=sc, interpolation=cv2.INTER_CUBIC)
            for d in zbar(img_s):
                payload = d.data.decode('utf-8', errors='replace').strip()
                if not payload or payload in out:
                    continue
                r = d.rect
                pts = [(pt.x, pt.y) for pt in d.polygon]
                if len(pts) != 4:
                    pts = [
                        (r.left, r.top),
                        (r.left + r.width, r.top),
                        (r.left + r.width, r.top + r.height),
                        (r.left, r.top + r.height),
                    ]
                out[payload] = np.asarray(pts, dtype=np.float64) / sc

    return out


class QrNet:
    """WeChat's Caffe SSD QR detection network: returns candidate QR bounding boxes + confidence."""

    def __init__(self, path=''):
        d = model_dir(path)
        self.net = cv2.dnn.readNetFromCaffe(str(d / 'detect.prototxt'), str(d / 'detect.caffemodel')) if d else None

    def boxes(self, crop, x0=0, y0=0, conf=0.5):
        if self.net is None or crop is None or crop.size == 0:
            return []
        self.net.setInput(cv2.dnn.blobFromImage(crop, 1.0 / 255.0, (384, 384), 0, False, False))
        h, w = crop.shape[:2]
        out = []
        for d in self.net.forward().reshape(-1, 7):
            if d[2] >= conf:
                x1, y1, x2, y2 = d[3] * w + x0, d[4] * h + y0, d[5] * w + x0, d[6] * h + y0
                out.append((float(d[2]), np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)))
        return out

    def scan(self, gray, conf=0.5, centres=None, tile=384, overlap=96):
        if self.net is None or gray is None or gray.size == 0:
            return []
        h, w = gray.shape[:2]
        if w <= tile and h <= tile:
            spots = [(0, 0)]
        elif centres is None:
            step = max(64, tile - overlap)
            spots = [
                (max(0, min(x, w - tile)), max(0, min(y, h - tile)))
                for y in range(0, max(1, h - overlap), step)
                for x in range(0, max(1, w - overlap), step)
            ]
        else:
            spots = [
                (int(max(0, min(cx - tile / 2, w - tile))), int(max(0, min(cy - tile / 2, h - tile))))
                for cx, cy in centres
            ]
        found = []
        for x0, y0 in dict.fromkeys(spots):
            found += self.boxes(gray[y0:y0 + tile, x0:x0 + tile], x0, y0, conf)
        keep = []
        for c, p in sorted(found, key=lambda t: -t[0]):
            if all(iou(p, q) < 0.3 for _, q in keep):
                keep.append((c, p))
        return keep


def iou(a, b):
    ax0, ay0 = a.min(axis=0)
    ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0)
    bx1, by1 = b.max(axis=0)
    inter = max(0.0, min(ax1, bx1) - max(ax0, bx0)) * max(0.0, min(ay1, by1) - max(ay0, by0))
    return inter / max(1e-6, (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter)


def transitions(crop):
    """Mean black/white transitions per row and per column after Otsu thresholding."""
    if crop is None or crop.size < 16:
        return 0.0, 0.0
    c = crop if crop.dtype == np.uint8 else np.clip(crop, 0, 255).astype(np.uint8)
    _, bw = cv2.threshold(c, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw = bw.astype(np.int8)
    return (
        float(np.abs(np.diff(bw, axis=1)).sum(axis=1).mean()),
        float(np.abs(np.diff(bw, axis=0)).sum(axis=0).mean()),
    )


def contrast_threshold(sd, valid_mask=None, floor=6.0, k=4.0):
    """Adaptive local-contrast threshold (median + k * MAD) computed only inside the valid fisheye circle."""
    vals = sd[valid_mask > 0] if valid_mask is not None and np.any(valid_mask > 0) else sd.ravel()
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return max(floor, med + k * 1.4826 * (mad + 1e-3))


def find_qr_candidates(gray, fisheye_mask=None, win=5, std_min=6.0, min_transitions=3.2, max_cands=12):
    """Detect high-contrast QR-like candidate boxes in `gray` (returns list of dicts with 'poly', 'center', 'score')."""
    h, w = gray.shape[:2]
    scale = 0.5 if w > 1640 else 1.0
    small_u8 = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else gray
    small = cv2.GaussianBlur(small_u8.astype(np.float32), (0, 0), 1.2)
    mask_s = None
    if fisheye_mask is not None:
        mask_s = cv2.resize(fisheye_mask, (small.shape[1], small.shape[0]), interpolation=cv2.INTER_NEAREST)
        mask_s = cv2.erode(mask_s, np.ones((9, 9), np.uint8))

    mu = cv2.boxFilter(small, -1, (win, win))
    sd = np.sqrt(np.maximum(cv2.boxFilter(small * small, -1, (win, win)) - mu * mu, 0.0))
    if mask_s is not None:
        sd = np.where(mask_s > 0, sd, 0.0)

    thr = contrast_threshold(sd, valid_mask=mask_s, floor=std_min)
    raw_blobs = []
    for level in (1.0, 1.6, 2.5, 4.0):
        m = (sd > thr * level).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, _, stats, cents = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            bx, by, bw, bh, area = stats[i]
            side_full = max(bw, bh) / scale
            aspect = bw / max(bh, 1.0)
            if not (24 <= side_full <= 700) or not (0.38 <= aspect <= 2.6) or area < 0.35 * bw * bh:
                continue
            cx, cy = cents[i] / scale
            if any(np.hypot(cx - c['center'][0], cy - c['center'][1]) < 0.5 * side_full for c in raw_blobs):
                continue
            crop = small_u8[by:by + bh, bx:bx + bw]
            tr_r, tr_c = transitions(crop)
            if min(tr_r, tr_c) < min_transitions:
                continue
            poly = np.array([[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]], dtype=np.float64) / scale
            score = min(1.0, 0.5 * (min(tr_r, tr_c) / 8.0) + 0.5 * (min(bw, bh) / max(bw, bh)))
            raw_blobs.append({'poly': poly, 'center': (float(cx), float(cy)), 'score': round(float(score), 2)})

    raw_blobs.sort(key=lambda b: -b['score'])
    return raw_blobs[:max_cands]


def aligned_mean(crops):
    """Align same-size grayscale crops via phase correlation and return (averaged_uint8, n_used)."""
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
        m = np.float32([[1.0, 0.0, -dx], [0.0, 1.0, -dy]])
        acc += cv2.warpAffine(f, m, ref.shape[::-1], borderMode=cv2.BORDER_REPLICATE)
        n += 1
    return np.clip(acc / n, 0, 255).astype(np.uint8), n
