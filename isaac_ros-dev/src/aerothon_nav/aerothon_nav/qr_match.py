"""Is this board the start QR? - by IMAGE, for boards that don't decode (user, 2026-10-28: the QRs carry no useful
data - random text or nothing - and the zone boards are seen from 10 m, the start QR from 5 m).

signature(patch): the board in an undistorted patch (fisheye.rectify_patch along its ray, or a pinhole crop) -> the
dark-module square found (Otsu + closing, the largest square-ish blob near the patch centre), warped upright to
SIZE x SIZE, blurred to a common coarse resolution (a 10 m view has ~2.5 px per module) and normalised.
similarity(a, b): normalised correlation over the board WITHOUT its four corners (the three finder patterns are the
same on every QR and would make any two boards look alike), best of the four 90 deg turns: ~1 same board,
~0 another board. Thresholds in qr_locator (match_hi / match_lo; between them: unsure -> step down 0.5 m).
"""
import cv2
import numpy as np

SIZE = 48                      # px of the upright board (21-29 modules: ~2 px per module, both views blurred to it)
CORNER = 0.34                  # share of the side masked in each corner (finder pattern + separator, version 1)
IMAGE_ONLY = '<image>'         # start "payload" when the start QR has no data: the zone boards are matched by image


TIMING = (0.21, 0.31)          # share of the side: the timing row / column (module 6 of 21-29) - also the same on all


def _mask():
    m = np.ones((SIZE, SIZE), bool)
    k = int(round(CORNER * SIZE))
    for ys in (slice(0, k), slice(SIZE - k, SIZE)):
        for xs in (slice(0, k), slice(SIZE - k, SIZE)):
            m[ys, xs] = False
    a, b = int(TIMING[0] * SIZE), int(round(TIMING[1] * SIZE))
    for band in (slice(a, b), slice(SIZE - b, SIZE - a)):       # all four sides: the board may be turned
        m[band, :] = False
        m[:, band] = False
    return m


MASK = _mask()


def find_square(gray):
    """The board's 4 corners (float32 4 x 2) in a patch where it is roughly in the middle, or None."""
    if gray is None or gray.size < 400:
        return None
    h, w = gray.shape[:2]
    f = gray.astype(np.float32)
    best, best_score = None, 0.0
    # the board = the high-contrast textured square (black + white modules) on plainer ground: local contrast at a
    # few window sizes (the module size in px is unknown: 2-3 px at 10 m, more in a zoomed patch)
    for k in sorted({max(3, (min(h, w) // d) | 1) for d in (40, 24, 14)}):
        mean = cv2.blur(f, (k, k))
        std = np.sqrt(np.maximum(cv2.blur(f * f, (k, k)) - mean * mean, 0.0))
        _, tex = cv2.threshold(np.clip(std, 0, 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        tex = cv2.morphologyEx(tex, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        for c in cv2.findContours(tex, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(c)
            if area < 0.002 * h * w:
                continue
            (cx, cy), (rw, rh), ang = cv2.minAreaRect(c)
            if min(rw, rh) < 8 or not 0.75 <= rw / max(rh, 1e-6) <= 1.33:
                continue
            fill = area / max(rw * rh, 1e-6)           # a QR fills its square; a line / L-shape doesn't
            off = np.hypot(cx - w / 2, cy - h / 2) / (0.5 * np.hypot(w, h))
            score = area * fill * (1.0 - 0.7 * off)
            if fill > 0.75 and score > best_score:
                # the window spreads the texture ~k/2 past the board's edge: shrink back
                rect = ((cx, cy), (max(4.0, rw - (k - 1)), max(4.0, rh - (k - 1))), ang)
                best, best_score = cv2.boxPoints(rect).astype(np.float32), score
    return best


def signature(gray, quad=None):
    """SIZE x SIZE float32 (zero mean, unit norm) of the board in gray, or None."""
    quad = find_square(gray) if quad is None else np.asarray(quad, np.float32).reshape(4, 2)
    if quad is None:
        return None
    c = quad.mean(axis=0)
    quad = quad[np.argsort(np.arctan2(quad[:, 1] - c[1], quad[:, 0] - c[0]))]   # a consistent winding
    dst = np.float32([[0, 0], [SIZE - 1, 0], [SIZE - 1, SIZE - 1], [0, SIZE - 1]])
    up = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad, dst), (SIZE, SIZE), flags=cv2.INTER_AREA)
    up = cv2.GaussianBlur(up.astype(np.float32), (5, 5), 1.0)    # both views down to the same coarse detail
    v = up - up[MASK].mean()
    n = np.linalg.norm(v[MASK])
    return (v / n).astype(np.float32) if n > 1e-3 else None


def similarity(a, b):
    """Masked normalised correlation, best of the 4 turns of b (-1 .. 1); None if either is missing."""
    if a is None or b is None:
        return None
    return float(max(np.sum(a[MASK] * np.rot90(b, k)[MASK]) for k in range(4)))


def variants(patch):
    """The filter settings tried on one undistorted patch before giving up on a decode (user, 2026-10-28: run the
    trial-and-error settings ~5 times): as is, CLAHE, unsharp, darker / brighter gamma, median, 2x upscale, local
    threshold."""
    yield 'raw', patch
    yield 'clahe', cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(patch)
    blur = cv2.GaussianBlur(patch, (0, 0), 2.0)
    yield 'sharpen', cv2.addWeighted(patch, 1.8, blur, -0.8, 0)
    for gname, gm in (('gamma0.6', 0.6), ('gamma1.6', 1.6)):
        lut = np.clip(((np.arange(256) / 255.0) ** gm) * 255.0, 0, 255).astype(np.uint8)
        yield gname, cv2.LUT(patch, lut)
    yield 'median', cv2.medianBlur(patch, 3)
    if max(patch.shape[:2]) <= 400:
        yield 'up2', cv2.resize(patch, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    yield 'adaptive', cv2.adaptiveThreshold(patch, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                           max(11, (min(patch.shape[:2]) // 8) | 1), 2)
