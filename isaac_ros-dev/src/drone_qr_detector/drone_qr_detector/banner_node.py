#!/usr/bin/env python3
"""rect_banner_view's banner detector as a ROS node for the corridor stack: the Arducam fisheye -> rectilinear view ->
green (entry) / blue (exit) / red (the wrong tunnel, 2026-10-28) banners -> the RAY to each in odom, as aerothon_nav's banner_detector publishes it, so
entry_align and corridor_survey use it unchanged. Runs as /banner_detector (the 8080 dashboard's "Corridor nav:
AeroTHON" choice, aerothon_nav corridor_nav.launch.py corridor_impl:=aerothon).
    ros2 run drone_qr_detector banner_node --ros-args -p image_topic:=/arducam/image_raw

Per frame (at most rate_hz, newest frame, on a thread of its own): Rectifier (fisheye -> pinhole, VPI when available),
WhiteBalance, BannerDetector.detect (colour mask, quadrilateral fit, solid colour / letters, true 7.7 : 1 aspect),
Tracker (confirmed after 2 hits, held 0.5 s) - all from rect_banner_view.py. The lens model is the calibration YAML
if `calibration` is set, else the topic's CameraInfo (equidistant; the sim's comes from sim_arducam.launch.py).
The rectilinear view has the fisheye's axes (rotation identity), so a banner centre (u, v) is the optical-frame ray
((u - cx) / f, (v - cy) / f, 1); TF odom <- the image's frame at its stamp turns it into odom.
  in:  image_topic (bgr8 / rgb8), info_topic, TF odom <- camera
  out: /nav/banner/detection  std_msgs/String JSON, one message per colour in view: {colour, stamp, origin (camera in
       odom), ray (unit, odom), azimuth_deg, elevation_deg, centre_px, area_px, aspect, score, distance_m,
       candidates: [{colour, ray, centre_px, score}, best first], corner_rays: the best one's 4 quad corners (unit,
       odom) - the supervisor's overhead banner fix intersects them with the ground for the tunnel axis}
A second instance `overhead_banner` (out_topic /nav/banner/overhead, active false until the supervisor's 5 m banner
search sets it) serves the overhead fix in either corridor implementation.
Params `colour` / `exit_colour` exist only so entry_align's set_parameters (colour both) succeeds: both colours are
always published and the consumers keep their own.
"""
import json
import math
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener

from drone_qr_detector.fisheye_model import FisheyeModel, quat_to_rot
from drone_qr_detector.rect_banner_view import BannerDetector, Rectifier, Tracker, WhiteBalance, parse_args


class BannerNode(Node):
    def __init__(self):
        super().__init__('banner_detector')
        p = self.declare_parameter
        p('image_topic', '/arducam/image_raw')
        p('info_topic', '/arducam/camera_info')
        p('calibration', '')            # lens YAML; '' = the CameraInfo
        p('odom_frame', 'odom')
        p('rate_hz', 5.0)
        p('hfov', 120.0)                # rectilinear window, deg (rect_banner_view's defaults)
        p('vfov', 90.0)
        p('with_letters', False)        # the sim's lettered banners (rect_banner_view --with-letters)
        p('white_balance', True)
        p('colour', 'both')
        p('exit_colour', 'blue')
        p('out_topic', '/nav/banner/detection')
        p('active', True)               # false: frames dropped (the overhead instance idles outside its search)
        p('ground_suppress', False)     # green over grass (rect_banner_view --ground-suppress): the overhead view
        p('zoom', 1.0)                  # rectilinear view scale (overhead: 0.5 - 2328 x 1344 cost ~0.4 s a frame)
        p('heartbeat', False)           # true: {colour: 'none', frame: n} per processed frame (the overhead fix counts
        #                                 frames, not seconds: the loaded sim processed one every ~8 s)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        argv = ['--headless'] + (['--with-letters'] if g('with_letters') else []) + \
            (['--ground-suppress'] if g('ground_suppress') else [])
        self.args = parse_args(argv)
        self.det, self.tracker, self.wb = BannerDetector(self.args), Tracker(), WhiteBalance()
        self.rect, self.info, self.frame = None, None, None
        self.cond = threading.Condition()
        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.pub = self.create_publisher(String, g('out_topic'), 10)
        self.create_subscription(CameraInfo, g('info_topic'), lambda m: setattr(self, 'info', m), 1)
        self.create_subscription(Image, g('image_topic'), self.on_image, qos_profile_sensor_data)
        self.n, self.ms, self.sent = 0, 0.0, 0
        self.create_timer(10.0, self.report)
        threading.Thread(target=self.loop, daemon=True).start()

    def on_image(self, m):
        if not self.g('active'):
            return
        with self.cond:
            self.frame = m                                    # newest wins
            self.cond.notify()

    def rectifier(self, w, h):
        if self.rect is None or self.rect.model.size != (w, h):
            calib = self.g('calibration')
            model = None
            if not calib:
                model = FisheyeModel.from_camera_info(self.info)
                if model.size != (w, h):
                    model = model.scaled(w, h)
            self.rect = Rectifier(calib or None, (w, h), self.g('hfov'), self.g('vfov'), self.g('zoom'), 'auto', 'auto',
                                  model=model)
            self.get_logger().info(f'{w}x{h} fisheye -> {self.rect.size[0]}x{self.rect.size[1]} rectilinear, '
                                   f'{self.rect.backend} remap, lens from {calib or "CameraInfo"}')
        return self.rect

    def loop(self):
        period = 1.0 / self.g('rate_hz')
        while rclpy.ok():
            with self.cond:
                self.cond.wait_for(lambda: self.frame is not None, timeout=1.0)
                m, self.frame = self.frame, None
            if m is None or (self.info is None and not self.g('calibration')):
                continue
            t0 = time.monotonic()
            try:
                self.process(m)
            except Exception as e:                            # keep running; say what broke
                self.get_logger().error(f'banner detection: {e!r}', throttle_duration_sec=5.0)
            self.n, self.ms = self.n + 1, 1e3 * (time.monotonic() - t0)
            time.sleep(max(0.0, period - (time.monotonic() - t0)))

    def process(self, m):
        img = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        if m.encoding == 'rgb8':
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        rect = self.rectifier(m.width, m.height)
        view = rect(np.ascontiguousarray(img[:, :, :3]))
        small, vsmall = rect.downscale(view, self.args.proc_width)
        self.wb.update(small, vsmall)
        lut = self.wb.lut() if self.g('white_balance') else None
        dets = self.det.detect(view, rect.valid, cv2.LUT(small, lut) if lut is not None else small, vsmall, lut, rect.P)
        tracks = self.tracker.update(dets)
        if self.g('heartbeat'):
            self.pub.publish(String(data=json.dumps({'colour': 'none', 'frame': self.n})))
        if not tracks:
            return
        try:
            t = self.tf.lookup_transform(self.g('odom_frame'), m.header.frame_id, m.header.stamp,
                                         timeout=Duration(seconds=0.1))
        except Exception as e:
            self.get_logger().warn(f'no TF {self.g("odom_frame")} <- {m.header.frame_id}: {e}',
                                   throttle_duration_sec=5.0)
            return
        R, o = quat_to_rot(t.transform.rotation), t.transform.translation
        P = rect.P
        for colour in ('green', 'blue', 'red'):
            mine = sorted((tr for tr in tracks if tr['colour'] == colour), key=lambda tr: -tr['det']['score'])
            if not mine:
                continue
            rays = []
            for tr in mine:
                u, v = tr['centre']
                r = np.array([(u - P[0, 2]) / P[0, 0], (v - P[1, 2]) / P[1, 1], 1.0])
                rays.append(R @ (r / np.linalg.norm(r)))
            d, ray = mine[0]['det'], rays[0]
            out = {'colour': colour, 'stamp': m.header.stamp.sec + m.header.stamp.nanosec * 1e-9,
                   'origin': [o.x, o.y, o.z], 'ray': ray.tolist(),
                   'azimuth_deg': math.degrees(math.atan2(ray[1], ray[0])),
                   'elevation_deg': math.degrees(math.asin(max(-1.0, min(1.0, ray[2])))),
                   'centre_px': [float(c) for c in mine[0]['centre']], 'area_px': float(d['area']),
                   'aspect': float(d['aspect']), 'score': float(d['score']), 'distance_m': d.get('distance_m'),
                   'candidates': [{'colour': colour, 'ray': r.tolist(), 'centre_px': [float(c) for c in tr['centre']],
                                   'score': float(tr['det']['score'])} for tr, r in list(zip(mine, rays))[:4]]}
            quad = d.get('quad')
            if quad is not None:
                cr = []
                for u, v in np.asarray(quad, float).reshape(-1, 2)[:4]:
                    r = np.array([(u - P[0, 2]) / P[0, 0], (v - P[1, 2]) / P[1, 1], 1.0])
                    cr.append((R @ (r / np.linalg.norm(r))).tolist())
                out['corner_rays'] = cr
            self.pub.publish(String(data=json.dumps(out)))
            self.sent += 1

    def report(self):
        self.get_logger().info(f'{self.n} frames, last {self.ms:.0f} ms, {self.sent} detections published')


def main():
    rclpy.init()
    node = BannerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
