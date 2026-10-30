"""Banner segmentation by its WHITE FRAME (user, 2026-10-30): the exact hue changes with the sun and the camera, the frame
does not. The white pixels of the frame are found first (pale, among the brightest pale pixels of THIS image, so the
threshold follows the exposure); whatever elongated and solidly coloured area they enclose is the banner. Its hue only
says which banner it is, by wide sectors. The frame also keeps the banner apart from grass of a similar colour.
Used by aerothon_nav's banner_detector (D435) and rect_banner_view / banner_node (Arducam).
"""
import cv2
import numpy as np

SECTORS = {'green': (35, 99), 'blue': (100, 135)}    # OpenCV H 0-180: wide on purpose (the real banner: 81-86, the sim's: 66)
PALE_S, FILL_S = 70, 60                              # white: saturation up to this; inside the frame: from this
MIN_ASPECT, MIN_FILL, MIN_COLOURED = 2.5, 0.8, 0.55
MIN_SHORT = 3                                        # px: thinner than this nothing is left of the inside


def white_mask(hsv):
    pale = hsv[..., 1] <= PALE_S
    if not pale.any():
        return np.zeros(hsv.shape[:2], np.uint8)
    v_ref = float(np.percentile(hsv[..., 2][pale], 90))
    white = (pale & (hsv[..., 2] >= max(60.0, 0.72 * v_ref))).astype(np.uint8) * 255
    return cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))


def frames(bgr, min_area=40):
    """-> ({colour: mask of the areas framed in white}, the white mask). Letters inside (the sim's banners) count."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (3, 3), 0), cv2.COLOR_BGR2HSV)
    white = white_mask(hsv)
    out = {k: np.zeros(white.shape, np.uint8) for k in SECTORS}
    contours, hier = cv2.findContours(white, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return out, white
    for c, h in zip(contours, hier[0]):
        area = cv2.contourArea(c)
        if h[3] < 0 or area < min_area:              # h[3] >= 0: a hole in the white
            continue
        (_, _), (w, ht), _ = cv2.minAreaRect(c)
        if min(w, ht) < MIN_SHORT or max(w, ht) / min(w, ht) < MIN_ASPECT or area / (w * ht) < MIN_FILL:
            continue
        m = np.zeros(white.shape, np.uint8)
        cv2.drawContours(m, [c], -1, 255, -1)
        inside = cv2.erode(m, np.ones((3, 3), np.uint8)) > 0
        if not inside.any():
            continue
        sat = hsv[..., 1][inside] >= FILL_S
        if sat.mean() < MIN_COLOURED:
            continue
        hue = float(np.median(hsv[..., 0][inside][sat]))
        colour = next((k for k, (lo, hi) in SECTORS.items() if lo <= hue <= hi), None)
        if colour:
            out[colour] = cv2.bitwise_or(out[colour], m)
    return out, white


def cut_by_frame(blobs, framed, white):
    """A framed area in view: only that, a floor or a plank of the colour has no say. None: the colour mask with the white
    cut out - it parts a far banner from grass of its colour where a morphological close had bridged the frame."""
    return framed if framed.any() else cv2.bitwise_and(blobs, cv2.bitwise_not(white))
