"""ArUco markers 3x3 .. 6x6 for the ArUco version of the delivery mission (shared by qr_reader, qr_locator,
vrx_explore and the 8084 test page).

4x4 / 5x5 / 6x6 are matched against the OpenCV _1000 dictionaries (the _50 / _100 / _250 sets are their first
ids). OpenCV has no 3x3 dictionary, so square candidates that no dictionary took are read bit by bit (black
border, clean cells, a lighter margin round it) and named by the smallest of the code's four rotations.
Payloads are plain strings: '4X4#23', '3X3#87' (raw grid), 'raw5x5#...' (a 5x5 grid in no dictionary).

On a fisheye the image is first undistorted: detect_fisheye() looks through gnomonic views (Fisheye.rectify_patch)
- one straight down and `ring` views tilted round it - and maps the corners back to fisheye pixels, so the
callers' pixel -> ground code is unchanged. Works with OpenCV 4.5 (container) and 4.7+ (host) ArUco APIs.
"""
import math

import cv2
import numpy as np

AR = cv2.aruco
DICTS = {'4X4': AR.DICT_4X4_1000, '5X5': AR.DICT_5X5_1000, '6X6': AR.DICT_6X6_1000}


def _params():
    p = AR.DetectorParameters_create() if hasattr(AR, 'DetectorParameters_create') else AR.DetectorParameters()
    p.cornerRefinementMethod = AR.CORNER_REFINE_SUBPIX
    p.minMarkerPerimeterRate = 0.02
    # keep nested squares: a board's outline 1/2 cell outside the marker would otherwise hide the marker itself
    p.minMarkerDistanceRate = 0.001
    return p


def _table(dict_id, n):
    """Every marker of a dictionary as rows of n*n bits, all four rotations: (bits [4*N, n*n], ids [4*N])."""
    d = AR.getPredefinedDictionary(dict_id)
    draw = AR.generateImageMarker if hasattr(AR, 'generateImageMarker') else AR.drawMarker
    count = (d.bytesList.shape[0] if hasattr(d, 'bytesList') else 1000)
    rows, ids = [], []
    for i in range(count):
        b = draw(d, i, n + 2)[1:-1, 1:-1] > 127
        for r in range(4):
            rows.append(np.rot90(b, r).ravel())
            ids.append(i)
    return np.array(rows, bool), np.array(ids)


class Detector:
    """One candidate pass per image (OpenCV's square finder), then each candidate's bit grid is read once and
    looked up in the 4x4 / 5x5 / 6x6 tables (all rotations, max_err bits off); a 3x3 grid gets its raw code."""

    def __init__(self, dicts=('4X4', '5X5', '6X6'), raw=True, min_px=12.0, max_err=None):
        self.params = _params()
        self.raw, self.min_px = raw, float(min_px)
        self.max_err = max_err or {4: 0, 5: 1, 6: 2}
        self.tables = {int(k[0]): (k, *_table(DICTS[k], int(k[0]))) for k in dicts}
        d = AR.getPredefinedDictionary(DICTS['4X4'])
        self.finder = AR.ArucoDetector(d, self.params) if hasattr(AR, 'ArucoDetector') else d

    def _candidates(self, gray):
        if hasattr(AR, 'ArucoDetector'):
            corners, ids, rej = self.finder.detectMarkers(gray)
        else:
            corners, ids, rej = AR.detectMarkers(gray, self.finder, parameters=self.params)
        return list(corners or []) + list(rej or [])

    def detect(self, gray):
        """[(payload, 4x2 corners)] in gray's pixels."""
        found = []
        quads = [r.reshape(4, 2).astype(np.float64) for r in self._candidates(gray)]
        # largest first: a marker's outline before the inner edge of its own border, which is skipped below
        for q in sorted(quads, key=lambda q: -cv2.contourArea(q.astype(np.float32))):
            side = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))
            c = tuple(float(v) for v in q.mean(axis=0))
            if side < self.min_px or any(
                    cv2.pointPolygonTest(f[1].astype(np.float32), c, True) > -side / 2 for f in found):
                continue
            reads = [r for r in (read_bits(gray, q, n, score=True) for n in (3, 4, 5, 6)) if r is not None]
            if not reads:
                continue
            # the grid whose cells read most cleanly (a wrong grid size straddles cell edges); a dictionary hit
            # wins over a raw grid that is cleaner by less than 0.03
            reads.sort(key=lambda r: r[1])
            names = [(self.lookup(b), sc) for b, sc in reads]
            best = names[0]
            known = [nm for nm in names if nm[0] and not nm[0].startswith('raw') and nm[1] < best[1] + 0.03]
            name = (known[0] if known else best)[0]
            if name:
                found.append((name, q))
        return found

    def lookup(self, bits):
        n = bits.shape[0]
        if n in self.tables:
            k, rows, ids = self.tables[n]
            err = (rows != bits.ravel()).sum(axis=1)
            j = int(err.argmin())
            if err[j] <= self.max_err.get(n, 0):
                return f'{k}#{int(ids[j])}'
        if n == 3:
            return f'3X3#{canon(bits)[0]}'
        return f'raw{n}x{n}#{canon(bits)[0]}' if self.raw else None


def read_bits(gray, quad, n, score=False):
    """Inner n x n bits of a quad with a one-cell black border and a lighter margin round it, else None.
    score: (bits, how far the cells are from pure black / white, 0 .. 0.5) instead."""
    cell = 8
    s = (n + 4) * cell                      # one extra cell each side = the margin outside the black border
    a, b = cell, s - 1 - cell
    dst = np.array([[a, a], [b, a], [b, b], [a, b]], np.float32)
    w = cv2.warpPerspective(gray, cv2.getPerspectiveTransform(quad.astype(np.float32), dst), (s, s))
    lvl, t = cv2.threshold(w[cell:-cell, cell:-cell], 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    hc = cell // 2
    margin = np.concatenate([w[:hc].ravel(), w[-hc:].ravel(), w[:, :hc].ravel(), w[:, -hc:].ravel()])
    if (margin > lvl).mean() < 0.6:
        return None
    mean = t.reshape(n + 2, cell, n + 2, cell)[:, 2:-2, :, 2:-2].mean(axis=(1, 3))
    if ((mean > 0.3) & (mean < 0.7)).sum() > 1:   # cells half black, half white: wrong grid size
        return None
    m = mean > 0.5
    if np.concatenate([m[0], m[-1], m[1:-1, 0], m[1:-1, -1]]).any():
        return None
    inner = m[1:-1, 1:-1]
    if inner.all() or not inner.any():
        return None
    return (inner, float(np.minimum(mean, 1 - mean).mean())) if score else inner


def canon(bits):
    """Smallest integer of the four rotations, and how many 90 deg turns gave it."""
    best = None
    for r in range(4):
        v = int(''.join('1' if x else '0' for x in np.rot90(bits, r).ravel()), 2)
        if best is None or v < best[0]:
            best = (v, r)
    return best


def view_rays(ring_tilt_deg=50.0, ring=4):
    """The gnomonic views' axes: straight down the optical axis, and `ring` tilted ring_tilt_deg round it."""
    out = [None]
    t = math.radians(ring_tilt_deg)
    for k in range(ring):
        a = 2 * math.pi * k / ring
        out.append(np.array([math.sin(t) * math.cos(a), math.sin(t) * math.sin(a), math.cos(t)]))
    return out


def detect_fisheye(det, gray, cam, centre_fov=100.0, ring_fov=90.0, ring_tilt=55.0, ring=5, max_size=1100,
                   views_out=None):
    """{payload: corners in fisheye pixels} from undistorted gnomonic views of the fisheye frame gray. The default
    views (100 deg straight down + 5 x 90 deg tilted 55 deg) leave no gap out to 70 deg off axis.
    views_out (a list) gets [(view image, meta, [(payload, corners in view px)])] for display."""
    found = {}
    for ray in view_rays(ring_tilt, ring if ring_fov > 0 else 0):
        fov = centre_fov if ray is None else ring_fov
        view, meta = cam.rectify_patch(gray, ray_d=ray, fov_deg=fov, zoom_scale=1.0, max_size=max_size)
        hits = det.detect(view)
        if views_out is not None:
            views_out.append((view, meta, hits))
        for payload, q in hits:
            px = cam.patch_px_to_fisheye_px(q, meta)
            if payload not in found and np.isfinite(px).all():
                found[payload] = px
    return found
