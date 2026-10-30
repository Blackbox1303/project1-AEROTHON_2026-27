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


GLARE_MAX = 0.05               # share of the black border ring that may be bright before 'different' is not believed


def classify(score, sharp, side, hi=0.80, lo=0.65, glare=0.0):
    """One view of a marker -> 'match' | 'different' | 'unsure'. A match needs a crisp view; 'different' also needs
    the marker big enough (MIN_SIDE) and no glare on it (glare_share <= GLARE_MAX): a small or glared target can look
    like another marker (2026-10-30 tests: 5 of 8 glared targets), never the other way round."""
    if score is None or sharp < BLURRY:
        return 'unsure'
    if score >= hi:
        return 'match'
    if score <= lo and side >= MIN_SIDE and glare <= GLARE_MAX:
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


def _order(q):
    q = np.asarray(q, np.float64).reshape(4, 2)
    c = q.mean(axis=0)
    return q[np.argsort(np.arctan2(q[:, 1] - c[1], q[:, 0] - c[0]))]


def marker_in_board(gray, board):
    """The dark marker inside a white board quad: its extent in the board straightened (so it works whatever the white
    margin), mapped back -> 4 x 2 quad or None. Glare / blur on the marker's own border don't matter here."""
    S = 96
    q = _order(board)
    dst = np.float32([[0, 0], [S - 1, 0], [S - 1, S - 1], [0, S - 1]])
    M = cv2.getPerspectiveTransform(q.astype(np.float32), dst)
    w = cv2.warpPerspective(gray, M, (S, S), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)
    t, dark = cv2.threshold(cv2.GaussianBlur(w, (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if not 0.15 < (dark > 0).mean() < 0.85:
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(dark)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))         # the marker: the biggest dark part
    x, y, bw, bh = stats[k, :4]
    if min(bw, bh) < 0.45 * S or max(bw, bh) > 1.4 * min(bw, bh):
        return None
    if np.median(w[dark == 0]) - np.median(w[lab == k]) < 30:   # dark on white, clearly
        return None
    box = np.float32([[x, y], [x + bw, y], [x + bw, y + bh], [x, y + bh]]).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(box, np.linalg.inv(M)).reshape(4, 2).astype(np.float64)


def board_quads(gray, min_px=10.0):
    """Marker quads found through their WHITE BOARD on the darker ground (a strong outline that blur and the sun's
    glare don't break - the marker's own thin black border they do: 2026-10-30 tests)."""
    g = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = g.shape[:2]
    out = []
    _, otsu = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    block = max(31, (min(h, w) // 6) | 1)
    adapt = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, -10)
    for bw in (otsu, adapt):
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # a board = a bright blob WITH the dark marker inside: fill its holes (RETR_EXTERNAL) and take its outline
        for c in cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
            area = cv2.contourArea(c)
            if area < (1.15 * min_px) ** 2 or area > 0.6 * h * w:
                continue
            hull = cv2.convexHull(c)
            peri = cv2.arcLength(hull, True)
            ap = cv2.approxPolyDP(hull, 0.04 * peri, True)
            q = ap.reshape(-1, 2).astype(np.float64) if len(ap) == 4 else \
                cv2.boxPoints(cv2.minAreaRect(hull)).astype(np.float64)
            if cv2.contourArea(q.astype(np.float32)) < 0.8 * cv2.contourArea(hull) and len(ap) != 4:
                continue
            sides = np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)
            if sides.min() < 1.15 * min_px or sides.max() > 2.5 * sides.min():
                continue
            x, y, bw_, bh_ = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + bw_ >= w - 2 or y + bh_ >= h - 2:
                continue                                        # the board cut off by the view's edge
            mq = marker_in_board(gray, q)
            if mq is not None and side_px(mq) >= min_px:
                out.append(mq)
    return out


def inside_view(gray, q, margin=3.0):
    """Every corner margin px inside the image: a marker cut off by the view's edge is never judged (a cut board gave a
    wrong outline that scored -0.02 against its own start marker, 2026-10-30 test)."""
    h, w = gray.shape[:2]
    q = np.asarray(q).reshape(4, 2)
    return bool((q[:, 0] >= margin).all() and (q[:, 1] >= margin).all() and (q[:, 0] <= w - 1 - margin).all()
                and (q[:, 1] <= h - 1 - margin).all())


def marker_quads(gray, min_px=10.0):
    """Marker-like squares in gray, largest first, nested ones dropped: OpenCV's square finder on the marker's own
    border, plus the markers found through their white board (board_quads) where that finder lost them. Only those
    wholly inside the view."""
    out = []
    for q in sorted(candidates(gray), key=lambda q: -cv2.contourArea(q.astype(np.float32))):
        side = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))
        if side < min_px:
            continue
        c = tuple(float(v) for v in q.mean(axis=0))
        if any(cv2.pointPolygonTest(p.astype(np.float32), c, False) >= 0 for p in out):
            continue                                            # inside a kept one: its own border's inner edge
        if inside_view(gray, q) and is_marker(gray, q):
            out.append(q)
    for q in board_quads(gray, min_px):
        if not inside_view(gray, q, margin=6.0):
            continue
        c = tuple(float(v) for v in q.mean(axis=0))
        if not any(cv2.pointPolygonTest(p.astype(np.float32), c, False) >= 0 for p in out):
            out.append(q)
    return out


GLARE = 245                    # grey level: a pixel this bright is burnt out (sun glare on the board) - not compared
MIN_VALID = 0.5                # share of the inside that must be glare-free to compare at all


def _norm(up, ok):
    if ok.sum() < MIN_VALID * MASK.sum():
        return None
    v = up - up[ok].mean()
    n = np.linalg.norm(v[ok])
    if n <= 1e-3:
        return None
    v = (v / n).astype(np.float32)
    v[~ok] = np.nan
    return v


def signature(gray, quad):
    """2 x SIZE x SIZE float32 of the marker inside quad (or None): [0] every pixel, [1] the burnt-out ones (>= GLARE:
    the sun's reflection on a laminated board - or, in bright sun, the white cells themselves) left out as NaN.
    similarity() takes the better of the two: glare can't spoil the match, nor can bright whites empty it."""
    if quad is None:
        return None
    raw = _upright(gray, quad, SIZE)
    up = cv2.GaussianBlur(raw.astype(np.float32), (0, 0), BLUR)
    burnt = cv2.dilate((raw >= GLARE).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    full = _norm(up, MASK.copy())
    if full is None:
        return None
    masked = _norm(up, MASK & ~burnt) if burnt[MASK].any() else full
    return np.stack([full, full if masked is None else masked])


def similarity(a, b):
    """Masked normalised correlation, best of the 4 turns of b (-1 .. 1) and of all pixels / glare left out (see
    signature); None if either is missing."""
    if a is None or b is None:
        return None
    a, b = np.asarray(a), np.asarray(b)
    if a.ndim == 3 or b.ndim == 3:
        aa, bb = (a if a.ndim == 3 else np.stack([a, a])), (b if b.ndim == 3 else np.stack([b, b]))
        got = [v for v in (_sim(aa[0], bb[0]), _sim(aa[1], bb[1]), _sim(aa[0], bb[1]), _sim(aa[1], bb[0]))
               if v is not None]
        return max(got) if got else None
    return _sim(a, b)


def _sim(a, b):
    best = None
    for k in range(4):
        r = np.rot90(b, k)
        m = MASK & np.isfinite(a) & np.isfinite(r)
        if m.sum() < MIN_VALID * MASK.sum():
            continue
        x, y = a[m] - a[m].mean(), r[m] - r[m].mean()
        d = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
        if d > 1e-9:
            c = float(np.sum(x * y) / d)
            best = c if best is None else max(best, c)
    return best


def glare_share(gray, quad):
    """0 .. 1: how much of the marker's BLACK BORDER RING (black on every marker) is bright - the sun's glare. Sampled in
    the middle of the border cell (blur bleeds only its edges); bright = above 75 % of the marker's own range."""
    up = _upright(gray, quad, SIZE).astype(np.float32)
    a, b = int(round(0.035 * SIZE)), int(round(0.095 * SIZE))
    ring = np.concatenate([up[a:b, a:SIZE - a].ravel(), up[SIZE - b:SIZE - a, a:SIZE - a].ravel(),
                           up[b:SIZE - b, a:b].ravel(), up[b:SIZE - b, SIZE - b:SIZE - a].ravel()])
    lo_, hi_ = float(np.percentile(up, 3)), float(np.percentile(up, 99))
    if hi_ - lo_ < 20:
        return 0.0
    return float((ring > lo_ + 0.75 * (hi_ - lo_)).mean())


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
    (score, sharpness, how, glare share) or None when no marker is in it."""
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
            best = (s, sharpness(img, q), fname, glare_share(img, q))
    return best
