"""Find the corridor-entrance banner (green entry / blue exit since 2026-10-30, plain with a thin white border; the
sim's: "AeroTHON 2026" lettered, v2 orange exit) in the camera image and publish the RAY to it in odom.

The banner's size and mounting are unknown, so nothing here assumes them: a banner is a strongly
coloured, solid, elongated patch with white text inside. Per processed frame (at most rate_hz, on
a 1/downscale copy):
  colour mask (HSV, config/banner.yaml) -> close (joins the colour around the letters) -> blobs ->
  keep blobs with: area >= min_area, solidity (colour + text inside the hull / hull) >= min_solidity,
  text fraction inside the hull in [white_min, white_max] (rejects grass, corridor floors, red zones:
  no text), elongation >= min_aspect measured on the rays (undistorted, around the blob centre).
  Text = low-saturation pixels inside the hull that are not darker than text_v_ratio x the banner
  colour's brightness: in shade the white letters render as grey as dark as the colour itself (sim,
  orange in shade: letters S ~20-70 V ~100 on S ~244 V ~96), so saturation, not brightness, separates
  them; the brightness floor keeps shadows and gaps out.
  Blobs are found on the small image, but colour / text are counted on a FULL-RESOLUTION crop of each
  candidate: a banner 10 m away is ~65 x 15 px at 1/4 scale, where downscaling blends the thin
  letters into the colour (text fraction 0.004 instead of ~0.2).
  -> best by area x solidity. Every banner-like blob is published too (`candidates`): on the day
  there are two banners per corridor (across the top + on the ground), and the aligner has to stay
  on one of them while triangulating.
  in:  image_topic + info_topic (equidistant -> Fisheye, else Pinhole), TF odom <- camera: the forward D435 colour
       camera since the Arducam looks straight down (2026-10-25), the 45 deg Arducam before
  out: /nav/banner/detection  std_msgs/String JSON {colour, stamp, origin (camera in odom), ray (unit,
       odom), azimuth_deg, elevation_deg, area_px, white_frac, solidity, aspect, score,
       candidates: [{colour, ray, centre_px, score}, best first]}; nothing when no banner is seen
       /nav/banner/debug  sensor_msgs/Image (bgr8, downscaled) with the mask and the pick - only
       while someone subscribes; use it to tune the HSV ranges on the day
Params: colour (green | orange | both), set by entry_align for each lap.
"""
import json
import math

import cv2
import numpy as np
import rclpy.time
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from aerothon_nav.fisheye import camera_model
from vio_bringup.spin import spin_node


def quat_to_rot(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])



def lookup_recent(tf, target, source, stamp, max_age, wait=0.3):
    """TF target <- source at `stamp`; if the pose hasn't arrived for that time yet, the latest one, provided the
    image is at most max_age s newer. The pose (PX4 via MAVROS) can lag the camera: on the loaded Jetson sim by
    0.07-0.4 s, and every banner sighting / depth cloud was dropped with 'extrapolation into the future' (2026-10-20)."""
    try:
        return tf.lookup_transform(target, source, stamp, timeout=Duration(seconds=wait))
    except Exception:
        t = tf.lookup_transform(target, source, rclpy.time.Time())
        age = (rclpy.time.Time.from_msg(stamp) - rclpy.time.Time.from_msg(t.header.stamp)).nanoseconds * 1e-9
        if age > max_age:
            raise RuntimeError(f'latest pose {age:.2f} s older than the image')
        return t

class BannerDetector(Node):
    def __init__(self):
        super().__init__('banner_detector')
        p = self.declare_parameter
        p('image_topic', '/arducam/image_raw')     # the D435 colour (/camera/color/image_raw) since the Arducam looks
        p('info_topic', '/arducam/camera_info')    # straight down (2026-10-25): it can't see a banner 5 m ahead
        p('colour', 'green')                    # green | orange | blue | both (v3: green = fly in, blue = the other)
        p('rate_hz', 5.0)
        p('downscale', 4)
        p('odom_frame', 'odom')
        p('tf_max_age', 1.0)         # s: an image newer than the latest pose by up to this uses that pose
        # OpenCV HSV: H 0-180. Measured in sim (banner H ~66 / ~13, S ~240); retune on the day.
        p('green.h', [68, 92])                  # the real banner: H 81-86 (teal green), grass H 19-46 (2026-10-30)
        p('green.s_min', 110)
        p('green.v_min', 30)
        p('orange.h', [7, 24])
        p('orange.s_min', 160)
        p('orange.v_min', 50)
        p('blue.h', [100, 125])                 # v3 banners (sim texture RGB 25, 60, 200 -> H ~112)
        p('blue.s_min', 160)
        p('blue.v_min', 30)
        p('white.s_max', 100)                   # letters: saturation below this ...
        p('text_v_ratio', 0.85)                 # ... and V >= this x the banner colour's median V
        p('min_area', 60)                       # px in the downscaled image
        p('min_solidity', 0.75)
        p('max_tilt_deg', 35.0)      # deg: the banner's long side from level in the view (an upright pole: ~90)
        p('white_min', 0.0)          # 2026-10-30: plain banners (no letters), a thin white border; the sim's: letters
        p('white_max', 0.5)
        p('min_aspect', 1.5)
        self.g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.cam = None
        self.last = 0.0
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub = self.create_publisher(String, '/nav/banner/detection', 10)
        self.debug_pub = self.create_publisher(Image, '/nav/banner/debug', 1)
        self.create_subscription(CameraInfo, self.g('info_topic'), self.on_info, qos_profile_sensor_data)
        # RELIABLE depth 1: best effort loses most fragments of these multi-MB frames
        self.create_subscription(Image, self.g('image_topic'), self.on_image, 1)

    def on_info(self, m):
        if self.cam is None or self.cam.size != (m.width, m.height):
            self.cam = camera_model(m)
            self.get_logger().info(f'camera {m.width}x{m.height}, fx {m.k[0]:.1f}')

    def on_image(self, m):
        now = self.get_clock().now().nanoseconds * 1e-9
        if self.cam is None or now - self.last < 1.0 / self.g('rate_hz'):
            return
        self.last = now
        ds = self.g('downscale')
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        small = cv2.resize(img, (m.width // ds, m.height // ds), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        colours = ('green', 'orange', 'blue') if self.g('colour') == 'both' else (self.g('colour'),)
        found, debug_masks = [], []
        for colour in colours:
            mask = self.colour_mask(hsv, colour)
            debug_masks.append(mask)
            joined = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            contours, _ = cv2.findContours(joined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                d = self.judge(c, img, ds, colour)
                if d:
                    found.append(d)
                elif cv2.contourArea(c) > 4 * self.g('min_area'):
                    found += self.split(c, joined, img, ds, colour)
        found.sort(key=lambda d: -d['score'])
        best = found[0] if found else None
        if found:
            self.publish(found, m)
        if self.debug_pub.get_subscription_count():
            self.publish_debug(small, debug_masks, best, ds, m.header)

    def colour_mask(self, hsv, colour):
        lo, hi = self.g(f'{colour}.h')
        return cv2.inRange(hsv, (lo, self.g(f'{colour}.s_min'), self.g(f'{colour}.v_min')), (hi, 255, 255))

    def split(self, contour, mask, img, ds, colour):
        """A blob joined to something thinner of its colour (the green POLES at the mouth touch the floor banner, 2026-10-30
        photo): opened with growing kernels the thin part goes first; the pieces are judged again."""
        blob = np.zeros_like(mask)
        cv2.drawContours(blob, [contour], -1, 255, -1)
        blob = cv2.bitwise_and(blob, mask)
        for k in (3, 5, 9, 13, 19):
            opened = cv2.morphologyEx(blob, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
            got = [d for d in (self.judge(c, img, ds, colour) for c in
                                cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]) if d]
            if got:
                return got
        return []

    def judge(self, contour, img, ds, colour):
        area = cv2.contourArea(contour)
        if area < self.g('min_area'):
            return None
        hull = cv2.convexHull(contour)
        # colour and text on a full-resolution crop around the hull
        x, y, w, h = cv2.boundingRect(hull * ds)
        pad = ds + 6                                   # room for the white border round it
        x0, y0 = max(0, x - pad), max(0, y - pad)
        crop = cv2.cvtColor(img[y0:y + h + pad, x0:x + w + pad], cv2.COLOR_BGR2HSV)
        hull_mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillConvexPoly(hull_mask, hull * ds - [x0, y0], 255)
        n_hull = cv2.countNonZero(hull_mask)
        col = cv2.bitwise_and(self.colour_mask(crop, colour), hull_mask)
        n_col = cv2.countNonZero(col)
        if n_hull == 0 or n_col == 0:
            return None
        v_col = float(np.median(crop[..., 2][col > 0]))
        white = cv2.inRange(crop, (0, 0, int(self.g('text_v_ratio') * v_col)), (180, self.g('white.s_max'), 255))
        n_white = cv2.countNonZero(cv2.bitwise_and(white, hull_mask))
        solidity = (n_col + n_white) / n_hull
        white_frac = n_white / n_hull
        if solidity < self.g('min_solidity') or not self.g('white_min') <= white_frac <= self.g('white_max'):
            return None
        # the thin white border round the banner (course 2026-10-30): evidence only, a far one is under a pixel
        ring = cv2.subtract(cv2.dilate(hull_mask, np.ones((7, 7), np.uint8)), hull_mask)
        white_rim = cv2.countNonZero(cv2.bitwise_and(white, ring)) / max(1, cv2.countNonZero(ring))
        # elongation on the rays: gnomonic projection around the blob's centre ray, so the fisheye's
        # bending near the image edge doesn't distort the shape
        pts = hull.reshape(-1, 2).astype(np.float64) * ds
        rays = self.cam.rays(pts)
        mom = cv2.moments(hull)
        centre_px = np.array([[mom['m10'] / mom['m00'] * ds, mom['m01'] / mom['m00'] * ds]])
        c = self.cam.rays(centre_px)[0]
        u = np.cross(c, [0.0, 1.0, 0.0] if abs(c[1]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = np.cross(c, u)
        dots = rays @ c
        if np.any(dots < 0.2):                         # blob spans > ~78 deg: not a banner
            return None
        plane = np.column_stack([rays @ u / dots, rays @ v / dots]).astype(np.float32)
        (_, _), (w, h), _ = cv2.minAreaRect(plane)
        aspect = max(w, h) / max(min(w, h), 1e-6)
        if aspect < self.g('min_aspect'):
            return None
        # lying ACROSS the view: both banners run horizontally (top and doormat); the green POLES beside the mouth are
        # just as long and solid but upright (2026-10-30 photo). plane x = u (level), y = v (camera down)
        ev, evec = np.linalg.eigh(np.cov(plane.T.astype(np.float64)))
        long_axis = evec[:, int(np.argmax(ev))]
        if abs(math.degrees(math.atan2(abs(long_axis[1]), abs(long_axis[0])))) > self.g('max_tilt_deg'):
            return None
        return {'colour': colour, 'centre_px': centre_px[0].tolist(), 'ray_cam': c, 'area_px': area * ds * ds,
                'white_frac': round(white_frac, 3), 'solidity': round(solidity, 3), 'aspect': round(aspect, 2),
                'white_rim': round(white_rim, 3), 'score': area * solidity * (0.9 + 0.25 * min(1.0, white_rim / 0.35)),
                'hull_small': hull}

    def publish(self, found, m):
        d = found[0]
        try:
            t = lookup_recent(self.tf, self.g('odom_frame'), m.header.frame_id, m.header.stamp, self.g('tf_max_age'))
        except Exception as e:
            self.get_logger().warn(f'no TF {self.g("odom_frame")} <- {m.header.frame_id}: {e}',
                                   throttle_duration_sec=5.0)
            return
        R = quat_to_rot(t.transform.rotation)
        o = t.transform.translation
        ray = R @ d['ray_cam']
        cands = [{'colour': e['colour'], 'ray': (R @ e['ray_cam']).tolist(), 'centre_px': e['centre_px'],
                  'score': e['score']} for e in found[:4]]
        out = {'colour': d['colour'], 'stamp': m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
               'origin': [o.x, o.y, o.z], 'ray': ray.tolist(),
               'azimuth_deg': math.degrees(math.atan2(ray[1], ray[0])),
               'elevation_deg': math.degrees(math.asin(max(-1.0, min(1.0, ray[2])))),
               **{k: d[k] for k in ('centre_px', 'area_px', 'white_frac', 'solidity', 'aspect', 'score')},
               'candidates': cands}
        self.pub.publish(String(data=json.dumps(out)))

    def publish_debug(self, small, masks, best, ds, header):
        view = small.copy()
        for mask in masks:
            view[mask > 0] = (view[mask > 0] * 0.4 + np.array([255, 0, 255]) * 0.6).astype(np.uint8)
        if best is not None:
            cv2.polylines(view, [best['hull_small']], True, (0, 255, 255), 2)
            cv2.putText(view, f'{best["colour"]} w{best["white_frac"]:.2f} s{best["solidity"]:.2f} '
                        f'a{best["aspect"]:.1f}', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
        msg = Image(header=header, height=view.shape[0], width=view.shape[1], encoding='bgr8',
                    step=view.shape[1] * 3, data=view.tobytes())
        self.debug_pub.publish(msg)


def main():
    spin_node(BannerDetector)


if __name__ == '__main__':
    main()
