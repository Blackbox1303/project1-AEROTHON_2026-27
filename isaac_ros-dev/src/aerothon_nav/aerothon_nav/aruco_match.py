"""Is this marker the start ArUco? - by IMAGE only, never by reading its id (user, 2026-10-30: the course has ArUco
markers only; the target is the zone marker that LOOKS like the start marker - no decoding).

marker_quads(gray): the marker-like squares in an undistorted view - OpenCV's ArUco square finder (accepted and
    rejected candidates: nothing is looked up) kept when the square has a dark border ring on a lighter margin and a
    textured inside; nested squares (a marker's inner edge) dropped.
signature(gray, quad): the square warped upright to SIZE x SIZE, blurred to a common coarse level (a 10 m view has
    ~5 px per cell, the start marker from 5 m twice that), normalised over the INSIDE only - the black border ring is
    the same on every marker and would make any two alike.
similarity(a, b): masked normalised correlation, best of the four 90 deg turns: ~1 the same marker, ~0 another.
sharpness(gray, quad): how crisp the cell edges are (0 .. 1) - below BLURRY a score can't be trusted.
SETTINGS: the deep read's trial grid (focal length = virtual camera zoom, FOV, output resolution) and filters
    (exposure / gamma, CLAHE, sharpen, denoise) run on a marker that is too blurry to match.
"""
import cv2
import numpy as np

SIZE = 64                      # px of the upright marker (4x4 .. 6x6 + border: 6 .. 8 cells, 8 .. 10 px a cell)
BORDER = 0.14                  # share of the side masked all round: the black border ring (1/6 .. 1/8 of the side)
BLUR = 1.6                     # Gaussian sigma (px at SIZE) both views are brought down to
BLURRY = 0.18                  # sharpness below this: the view says little - try other settings / come lower
IMAGE_ONLY = '<image>'         # the start "payload": the target is matched by image (= qr_match.IMAGE_ONLY)
MIN_SIDE = 30.0                # px (in a 1:1 view) a marker must have before 'another marker' is believed: the course's
#                                6x6 markers (2026-10-30 scans) at 22 px a side - 2.75 px a cell - scored the target as
#                                low as 0.39; >= 40 px: same >= 0.91, others <= 0.39


def classify(score, sharp, side, hi=0.80, lo=0.65):
    """One view of a marker -> 'match' | 'different' | 'unsure'. A match needs a crisp view; 'different' also needs
    the marker big enough (MIN_SIDE) - a small target can look like another marker, never the other way round."""
    if score is None or sharp < BLURRY:
        return 'unsure'
    if score >= hi:
        return 'match'
    if score <= lo and side >= MIN_SIDE:
        return 'different'
    return 'unsure'


def side_px(quad):
    q = np.asarray(quad, float).reshape(4, 2)
    return float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))

# the deep read's camera settings: (name, FOV deg, zoom = focal length x the lens's, max output px = resolution)
SETTINGS = [('f2.5', 22.0, 2.5, 400), ('f4', 14.0, 4.0, 400), ('f1.8', 32.0, 1.8, 400), ('f6', 9.0, 6.0, 400),
            ('f2.5hi', 22.0, 2.5, 640), ('f4lo', 14.0, 4.0, 256)]


def _mask():
    m = np.zeros((SIZE, SIZE), bool)
    k = int(round(BORDER * SIZE))
    m[k:SIZE - k, k:SIZE - k] = True
    return m


MASK = _mask()


def _finder():
    from aerothon_nav import aruco_tools                       # its square finder only: no dictionary lookups
    return aruco_tools.Detector(dicts=('4X4',), raw=False)


_FINDER = None


def candidates(gray):
    """OpenCV's square candidates (accepted + rejected) as float64 4 x 2."""
    global _FINDER
    if _FINDER is None:
        _FINDER = _finder()
    return [r.reshape(4, 2).astype(np.float64) for r in _FINDER._candidates(gray)]


def _upright(gray, quad, size, margin=0):
    """The quad warped to size x size (+ margin px of what's round it), consistent winding."""
    q = np.asarray(quad, np.float32).reshape(4, 2)
    c = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]
    a, b = margin, margin + size - 1
    dst = np.float32([[a, a], [b, a], [b, b], [a, b]])
    return cv2.warpPerspective(gray, cv2.getPerspectiveTransform(q, dst), (size + 2 * margin, size + 2 * margin),
                               flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def is_marker(gray, quad):
    """A dark border ring on a lighter margin round it, and a textured inside (no bits read)."""
    m = 8
    w = _upright(gray, quad, 48, margin=m).astype(np.float32)
    inner = w[m:-m, m:-m]
    k = max(2, int(round(BORDER * 48)))
    ring = np.concatenate([inner[:k].ravel(), inner[-k:].ravel(), inner[k:-k, :k].ravel(), inner[k:-k, -k:].ravel()])
    # the white margin right round the border: the band 0 .. 8 % of the side out (the real boards' white edge is ~9 %
    # of the marker's side - further out is grass, 2026-10-30 scans)
    out = np.concatenate([w[m // 2:m, m:-m].ravel(), w[-m:-(m // 2), m:-m].ravel(), w[m:-m, m // 2:m].ravel(),
                          w[m:-m, -m:-(m // 2)].ravel()])
    core = inner[k:-k, k:-k]
    lo, hi = float(np.percentile(core, 10)), float(np.percentile(core, 90))
    return (float(np.median(out)) - float(np.median(ring)) > 25.0 and float(np.median(ring)) < lo + 0.5 * (hi - lo)
            and hi - lo > 30.0)


def marker_quads(gray, min_px=10.0):
    """Marker-like squares in gray, largest first, nested ones dropped."""
    out = []
    for q in sorted(candidates(gray), key=lambda q: -cv2.contourArea(q.astype(np.float32))):
        side = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))
        if side < min_px:
            continue
        c = tuple(float(v) for v in q.mean(axis=0))
        if any(cv2.pointPolygonTest(p.astype(np.float32), c, False) >= 0 for p in out):
            continue                                            # inside a kept one: its own border's inner edge
        if is_marker(gray, q):
            out.append(q)
    return out


def signature(gray, quad):
    """SIZE x SIZE float32 (zero mean, unit norm over MASK) of the marker inside quad, or None."""
    if quad is None:
        return None
    up = cv2.GaussianBlur(_upright(gray, quad, SIZE).astype(np.float32), (0, 0), BLUR)
    v = up - up[MASK].mean()
    n = np.linalg.norm(v[MASK])
    return (v / n).astype(np.float32) if n > 1e-3 else None


def similarity(a, b):
    """Masked normalised correlation, best of the 4 turns of b (-1 .. 1); None if either is missing."""
    if a is None or b is None:
        return None
    return float(max(np.sum(a[MASK] * np.rot90(b, k)[MASK]) for k in range(4)))


def sharpness(gray, quad):
    """0 .. 1: the strongest cell-edge gradient inside the marker vs its contrast (a crisp edge steps in ~1 px)."""
    up = _upright(gray, quad, SIZE).astype(np.float32)
    core = up[MASK.nonzero()[0].min():MASK.nonzero()[0].max() + 1, MASK.nonzero()[1].min():MASK.nonzero()[1].max() + 1]
    rng = float(np.percentile(core, 95) - np.percentile(core, 5))
    if rng < 10:
        return 0.0
    gx, gy = cv2.Sobel(core, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(core, cv2.CV_32F, 0, 1, ksize=3)
    g = float(np.percentile(np.hypot(gx, gy), 98)) / 4.0       # Sobel 3x3: a 1 px step of rng gives ~4 x rng ... /4
    return float(np.clip(g / rng, 0.0, 1.0))


def filters(patch):
    """The trial filters on one undistorted patch: as is, exposure (gamma darker / brighter, CLAHE), sharpen, denoise."""
    yield 'raw', patch
    for name, gm in (('gamma0.6', 0.6), ('gamma1.6', 1.6)):
        lut = np.clip(((np.arange(256) / 255.0) ** gm) * 255.0, 0, 255).astype(np.uint8)
        yield name, cv2.LUT(patch, lut)
    yield 'clahe', cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(patch)
    blur = cv2.GaussianBlur(patch, (0, 0), 2.0)
    yield 'sharpen', cv2.addWeighted(patch, 1.8, blur, -0.8, 0)
    yield 'median', cv2.medianBlur(patch, 3)


def best_in_patch(patch, tmpl, min_px=10.0):
    """The marker in an undistorted patch nearest its centre, matched against tmpl over the filters ->
    (score, sharpness, how) or None when no marker is in it."""
    h, w = patch.shape[:2]
    best = None
    for fname, img in filters(patch):
        qs = marker_quads(img, min_px=min_px)
        if not qs:
            continue
        q = min(qs, key=lambda q: np.hypot(*(q.mean(axis=0) - [w / 2, h / 2])))
        s = similarity(tmpl, signature(img, q))
        if s is None:
            continue
        if best is None or s > best[0]:
            best = (s, sharpness(img, q), fname)
    return best
