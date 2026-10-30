"""Read and store the start QR (mission start: take off to 5 m, move ~1 m forward, read it).

The node doesn't fly: the caller hovers over the QR and calls /mission/read_qr. For scan_time seconds every
new Arducam frame is decoded using:
  1. Central gnomonic fisheye rectification (75 deg FOV) + WeChat CNN / zbar.
  2. Multi-scale QrNet bounding-box search + circular-fisheye-masked contrast candidates -> virtual camera adjustment
     (`rectify_patch` aimed along each candidate's 3D ray at 2.5x zoom) + phase-correlated multi-frame stacking
     (`aligned_mean`) + border-padded WeChat super-resolution / zbar.
  3. Direct full-frame WeChat + zbar fallback.
The payload decoded most often wins. Payloads are plain strings - never opened or followed.
The board's IMAGE is kept too (qr_match.signature of the central view, the frame with the biggest QrNet box):
<store_file>.sig.npy + .png - qr_locator compares the zone boards that don't decode with it (user, 2026-10-28: the
QRs may hold random text or nothing). A QR seen (box + signature in min_hits frames) that never decodes - no data -
is a read too: payload qr_match.IMAGE_ONLY, the zone boards are then matched by image only.
marker 'aruco' (default: the AEROTHON_MARKER environment variable, else 'qr'): the start board is an ArUco marker, and
it is NOT read (user, 2026-10-30: match by image only). Every frame: the marker-like square nearest straight down in the
undistorted view (aruco_match.marker_quads) -> its image signature; the crispest x biggest one is stored (.sig.npy +
.png) and the payload is aruco_match.IMAGE_ONLY - qr_locator matches the zone markers against that image.
  service: /mission/read_qr (std_srvs/Trigger) -> success + payload in `message`
  out:     /mission/start_qr (std_msgs/String JSON {payload, source, hits, frames, also_seen, time}),
           latched; the same JSON in store_file, so a restarted node (or the supervisor) still has it
"""
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from std_srvs.srv import Trigger

from aerothon_nav import qr_match
from aerothon_nav.fisheye import Fisheye


def to_gray(m):
    a = np.frombuffer(m.data, np.uint8)
    if m.encoding in ('mono8', '8UC1'):
        return a.reshape(m.height, m.width)
    code = cv2.COLOR_RGB2GRAY if m.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(a.reshape(m.height, m.width, 3), code)


class QrReader(Node):
    def __init__(self):
        super().__init__('qr_reader')
        p = self.declare_parameter
        p('topic', '/arducam/image_raw')
        p('info_topic', '/arducam/camera_info')
        p('fallback_topic', '')
        p('scan_time', 5.0)          # s of decoding per request
        p('min_hits', 2)             # decodes of the winning payload needed
        p('store_file', os.path.expanduser('~/.ros/aerothon_start_qr.json'))
        p('pylib', '')
        p('wechat_dir', '')          # WeChat model files (qr_tools.model_dir) - '' = the default place
        p('net_conf', 0.55)
        p('chase_avg', 4)
        p('marker', os.environ.get('AEROTHON_MARKER', 'qr'))   # qr | aruco
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.cam = None
        self.templates, self.reading = [], False
        self.frames = {}
        self.chase = {}
        self.aruco = None
        if g('marker') == 'aruco':
            from aerothon_nav import aruco_match
            self.aruco = aruco_match
            self.get_logger().info('start marker: ArUco, its IMAGE (no id read) on the undistorted view')
        else:
            self.init_qr()
        self.init_ros()

    def init_qr(self):
        g = self.g
        if g('pylib'):
            sys.path.insert(0, g('pylib'))
        from pyzbar.pyzbar import ZBarSymbol, decode
        self.decode = lambda img: {d.data.decode(errors='replace').strip()
                                   for d in decode(img, symbols=[ZBarSymbol.QRCODE])
                                   if d.data.decode(errors='replace').strip()}
        from aerothon_nav import qr_tools
        self.qr_tools = qr_tools
        self.wc = qr_tools.make_wechat(g('wechat_dir'))
        self.qnet = qr_tools.QrNet(g('wechat_dir'))
        if self.qnet is not None and self.qnet.net is None:
            self.qnet = None
        self.get_logger().info(f'start QR decoders: {"WeChat + " if self.wc else ""}zbar, '
                               f'QrNet={"on" if self.qnet else "off"}')

    def init_ros(self):
        g = self.g
        cb = ReentrantCallbackGroup()
        self.create_subscription(CameraInfo, g('info_topic'), self.on_info, qos_profile_sensor_data)
        # RELIABLE depth 1: best effort loses most fragments of these large frames
        for key in ('topic', 'fallback_topic'):
            if g(key):
                self.create_subscription(Image, g(key), lambda m, k=key: self.frames.__setitem__(k, m), 1,
                                         callback_group=cb)
        latched = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                             durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.pub = self.create_publisher(String, '/mission/start_qr', latched)
        self.create_service(Trigger, '/mission/read_qr', self.on_read, callback_group=cb)
        stored = Path(g('store_file'))
        if stored.exists():                          # re-announce what an earlier run read
            self.pub.publish(String(data=stored.read_text()))
            self.get_logger().info(f'stored start QR: {stored.read_text()}')

    def on_info(self, m):
        if self.cam is None or self.cam.size != (m.width, m.height):
            try:
                self.cam = Fisheye(m)
            except ValueError:
                pass

    def _decode_crop(self, img):
        if img is None or img.size == 0:
            return set()
        mean = float(img.mean())
        work = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(img) if (mean < 80 or mean > 200) else img
        if self.wc is not None:
            found = set(self.qr_tools.wechat_decode(self.wc, work))
            if not found and work is not img:
                found = set(self.qr_tools.wechat_decode(self.wc, img))
            if found:
                return found
        imgs = [work] + ([cv2.resize(work, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)] if work.shape[1] <= 1200 else [])
        for im in imgs:
            found = self.decode(im)
            if found:
                return found
        return set()

    def read_aruco(self, gray, fisheye=True):
        """{IMAGE_ONLY} when a marker is under the drone (nearest straight down in the undistorted view), its image
        kept while reading; else {}. No id is read."""
        am = self.aruco
        view = gray
        if fisheye and self.cam is not None:
            view, _ = self.cam.rectify_patch(gray, ray_d=None, fov_deg=100.0, zoom_scale=1.0, max_size=1100)
        qs = am.marker_quads(view)
        if not qs:
            return set()
        h, w = view.shape[:2]
        q = min(qs, key=lambda q: np.hypot(*(q.mean(axis=0) - [w / 2, h / 2])))
        sig = am.signature(view, q)
        if sig is None:
            return set()
        if self.reading:
            side = float(np.mean(np.linalg.norm(q - np.roll(q, 1, axis=0), axis=1)))
            crop = am._upright(view, q, 160, margin=20)
            self.templates.append((sig, am.sharpness(view, q) * side, crop))   # save_template keeps the best
        return {am.IMAGE_ONLY}

    def template(self, view):
        """The board's image signature in an undistorted view: around the best QrNet box (else the whole view) ->
        (signature, box size px, crop) or None."""
        crop, size = view, 0.0
        if self.qnet is not None:
            boxes = self.qnet.scan(view, conf=float(self.g('net_conf')), multi_scale=True)
            if boxes:
                _, poly = max(boxes, key=lambda b: b[0])
                (x0, y0), (x1, y1) = poly.min(axis=0), poly.max(axis=0)
                size = float(max(x1 - x0, y1 - y0))
                pad = 0.35 * size
                X0, Y0 = int(max(0, x0 - pad)), int(max(0, y0 - pad))
                crop = view[Y0:int(min(view.shape[0], y1 + pad)), X0:int(min(view.shape[1], x1 + pad))]
        sig = qr_match.signature(crop)
        return None if sig is None else (sig, size, crop)

    def read(self, gray):
        """Multi-stage start QR decode: central gnomonic rectification -> candidate/QrNet bounding box search +
        virtual camera steering + aligned multi-frame averaging -> direct full-frame fallback."""
        h, w = gray.shape[:2]
        # 1. Central gnomonic rectified view if fisheye camera_info is available
        if self.cam is not None:
            try:
                c_view, _ = self.cam.rectify_patch(gray, ray_d=None, fov_deg=75.0, zoom_scale=1.0, max_size=1400)
                if self.reading:
                    t = self.template(c_view)
                    if t is not None:
                        self.templates.append(t)
                found = self._decode_crop(c_view)
                if found:
                    return found
            except Exception:
                pass

            # 2. Search for QR bounding boxes (masked contrast candidates + multi-scale QrNet) and adjust virtual camera
            try:
                fmask = self.cam.circle_mask(w, h, max_theta_deg=82.0)
                cands = self.qr_tools.find_qr_candidates(gray, fisheye_mask=fmask, max_cands=8)
                centres = [c['center'] for c in cands]
                if self.qnet is not None and centres:
                    for _, poly in self.qnet.scan(gray, conf=float(self.g('net_conf')), centres=centres, tile=256):
                        pt = poly.mean(axis=0)
                        if all(math.dist(pt, cc) > 40.0 for cc in centres):
                            centres.insert(0, (float(pt[0]), float(pt[1])))
                if centres:
                    scale = w / max(self.cam.size[0], 1)
                    rays = self.cam.rays(np.array(centres[:6], dtype=np.float64), scale=scale)
                    for ray_d in rays:
                        patch, _ = self.cam.rectify_patch(gray, ray_d=ray_d, fov_deg=22.0, zoom_scale=2.5, max_size=480)
                        found = self._decode_crop(patch)
                        rkey = tuple(np.round(ray_d, 2))
                        if not found:
                            stack = self.chase.setdefault(rkey, [])
                            stack.append(cv2.resize(patch, (256, 256), interpolation=cv2.INTER_AREA))
                            del stack[:-int(self.g('chase_avg'))]
                            if len(stack) >= 2:
                                avg, _ = self.qr_tools.aligned_mean(stack)
                                clean = cv2.fastNlMeansDenoising(avg, None, 8, 5, 7)
                                found = self._decode_crop(clean) or self._decode_crop(avg)
                        if found:
                            self.chase.pop(rkey, None)
                            return found
            except Exception:
                pass

        # 3. Direct full-frame WeChat + zbar fallback
        return self._decode_crop(gray)

    def save_template(self):
        """The start board's image (the biggest box seen) -> <store_file>.sig.npy / .png; False if none."""
        if not self.templates:
            return False
        sig, size, crop = max(self.templates, key=lambda t: t[1])
        base = Path(self.g('store_file'))
        try:
            base.parent.mkdir(parents=True, exist_ok=True)
            np.save(base.with_suffix('.sig.npy'), sig)
            cv2.imwrite(str(base.with_suffix('.png')), crop)
        except OSError as e:
            self.get_logger().error(f'could not store the start QR image: {e}')
            return False
        self.get_logger().info(f'start QR image stored ({len(self.templates)} views, box {size:.0f} px)')
        return True

    def on_read(self, req, res):
        self.chase.clear()
        self.templates, self.reading = [], True
        try:
            return self.scan(res)
        finally:
            self.reading = False

    def scan(self, res):
        votes = {'topic': Counter(), 'fallback_topic': Counter()}
        seen, last = Counter(), {}
        end = time.monotonic() + self.g('scan_time')
        while time.monotonic() < end:
            for key in ('topic', 'fallback_topic'):
                m = self.frames.get(key)
                if m is None or last.get(key) is m:
                    continue
                last[key] = m
                seen[key] += 1
                if key == 'fallback_topic' and votes['topic']:
                    continue                         # only when the real camera's view fails
                gray = to_gray(m)
                for payload in (self.read_aruco(gray, key == 'topic') if self.aruco else self.read(gray)):
                    votes[key][payload] += 1
            if any(c >= self.g('min_hits') + 1 for c in votes['topic'].values()):
                break                                # early exit once confidently decoded multiple times
            time.sleep(0.05)
        source = 'topic' if votes['topic'] else 'fallback_topic' if votes['fallback_topic'] else None
        counts = votes[source] if source else Counter()
        payload, hits = counts.most_common(1)[0] if counts else (None, 0)
        stored = self.save_template()
        if payload is None and stored and len(self.templates) >= self.g('min_hits'):
            # a board in view, never decoded: no data on it - match the zone boards by image
            payload, hits, source = qr_match.IMAGE_ONLY, len(self.templates), 'topic'
            self.get_logger().warn(f'start QR seen in {hits} frames but no data decoded: matching by image only')
        if payload is None or hits < self.g('min_hits'):
            res.success = False
            res.message = (f'no QR read ({hits} hits; frames {dict(seen)})' if not seen or payload is None
                           else f'only {hits} decode(s) of {payload!r}')
            self.get_logger().warn(res.message)
            return res
        out = json.dumps({'payload': payload, 'source': self.g(source), 'hits': hits, 'frames': dict(seen),
                          'also_seen': sorted(set(counts) - {payload}), 'time': time.time()})
        self.pub.publish(String(data=out))
        try:
            Path(self.g('store_file')).parent.mkdir(parents=True, exist_ok=True)
            Path(self.g('store_file')).write_text(out)
        except OSError as e:
            self.get_logger().error(f'could not store the QR: {e}')
        self.get_logger().info(f'start QR {payload!r} ({hits} decodes, {self.g(source)})')
        res.success, res.message = True, payload
        return res


def main():
    rclpy.init()
    node = QrReader()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    except Exception:  # RCLError "context is not valid" when shutdown races the wait set (see vio_bringup spin.py)
        if rclpy.ok():
            raise
    finally:
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
