"""The real Arducam IMX219 (CSI) -> /arducam/image_raw, the topic the sim's Gazebo camera publishes.

GStreamer through OpenCV (the container's cv2 has GStreamer and the nvarguscamerasrc plugin): Argus does the
debayer / ISP on the Jetson, nvvidconv copies out of NVMM, appsink keeps only the newest frame (drop=true), so a
slow subscriber never builds a backlog. Needs the IMX219 device-tree overlay (/dev/video0) and the host's
nvargus-daemon (its socket in /tmp, bind-mounted into the container).
If `arducam_calib.py` is running on `http://127.0.0.1:8090/frame.jpg` and holding the Argus stream, `arducam_csi`
seamlessly shares frames from `:8090` so bench testing and the mission stack never conflict.
"""
import array
import os
import threading
import time
import urllib.request

import cv2
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

from vio_bringup.spin import spin_node


class ArducamCsi(Node):
    def __init__(self):
        super().__init__('arducam_csi')
        p = self.declare_parameter
        p('sensor_id', 0)
        p('width', 1640)
        p('height', 1232)
        p('fps', 30)
        p('rate', 15.0)
        p('max_exposure_ms', 20.0)
        p('flip_method', 0)
        p('image_topic', '/arducam/image_raw')
        p('frame_id', 'arducam_optical_frame')
        p('qr_width', 3280)
        p('qr_height', 2464)
        p('qr_fps', 21)
        p('qr_max_exposure_ms', 5.0)
        p('qr_rate', 4.0)
        p('qr_topic', '/arducam/qr/image_raw')
        p('qr_states', ['READ_START_QR', 'SEARCH_DELIVERY', 'DELIVER'])
        p('state_topic', '/mission/state')
        p('force_qr', False)
        p('qr_boost_exposure_ms', 12.0)
        p('boost_topic', '/arducam/qr_boost')
        p('shared_http_url', 'http://127.0.0.1:8090/frame.jpg')
        p('v4l2_device', '')         # optional USB VRX capture device (e.g. /dev/video2)
        g = lambda n: self.get_parameter(n).value  # noqa: E731
        self.g = g
        self.pub = self.create_publisher(Image, g('image_topic'), 2)
        self.qr_pub = self.create_publisher(Image, g('qr_topic'), 2)
        self.want_qr = bool(g('force_qr'))
        self.create_subscription(String, g('state_topic'), self.on_state, 10)
        self.boost = False
        self.create_subscription(Bool, g('boost_topic'), self.on_boost, 10)
        self.frames, self.t_log = 0, time.monotonic()
        self.running = True
        threading.Thread(target=self.loop, daemon=True).start()

    def on_state(self, m):
        st = m.data.split()[0] if m.data else ''
        want = self.g('force_qr') or st in self.g('qr_states') or m.data in self.g('qr_states')
        if want != self.want_qr:
            self.get_logger().info(f'mission {m.data}: {"QR mode (full resolution, short exposure)" if want else "normal mode"}')
        self.want_qr = want

    def on_boost(self, m):
        if m.data != self.boost:
            self.get_logger().info(f'QR exposure boost {"on" if m.data else "off"}')
        self.boost = m.data

    def pipeline(self, qr, boost=False):
        g = self.g
        w, h, fps, exp = ((g('qr_width'), g('qr_height'), g('qr_fps'),
                           g('qr_boost_exposure_ms') if boost else g('qr_max_exposure_ms')) if qr else
                          (g('width'), g('height'), g('fps'), g('max_exposure_ms')))
        return (f'nvarguscamerasrc sensor-id={g("sensor_id")} '
                f'exposuretimerange="13000 {int(exp * 1e6)}" gainrange="1 10.625" ! '
                f'video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! '
                f'nvvidconv flip-method={g("flip_method")} ! video/x-raw,format=BGRx ! videoconvert ! '
                f'video/x-raw,format=BGR ! appsink drop=true max-buffers=1 sync=false')

    def image(self, frame, stamp, encoding):
        m = Image()
        m.header.stamp = stamp
        m.header.frame_id = self.g('frame_id')
        m.height, m.width = frame.shape[:2]
        m.encoding = encoding
        m.step = frame.shape[1] * (3 if encoding == 'bgr8' else 1)
        m.data = array.array('B', frame.tobytes())
        return m

    def _try_http_shared(self):
        url = str(self.g('shared_http_url'))
        if not url:
            return None
        try:
            with urllib.request.urlopen(url, timeout=0.6) as resp:
                if resp.status == 200:
                    buf = np.frombuffer(resp.read(), dtype=np.uint8)
                    return cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception:
            pass
        return None

    def loop(self):
        os.environ.pop('DISPLAY', None)
        while self.running:
            # First check if arducam_calib.py (:8090) is already holding the CSI camera
            shared_frame = self._try_http_shared()
            if shared_frame is not None:
                self.get_logger().info(f'Sharing CSI camera stream from {self.g("shared_http_url")} ({shared_frame.shape[1]}x{shared_frame.shape[0]})')
                period = 1.0 / max(0.1, self.g('rate'))
                qr_period = 1.0 / max(0.1, self.g('qr_rate'))
                size = (self.g('width'), self.g('height'))
                last = last_qr = 0.0
                while self.running:
                    frame = self._try_http_shared()
                    if frame is None:
                        self.get_logger().info('Shared HTTP stream closed; switching to direct GStreamer CSI capture')
                        break
                    now = time.monotonic()
                    stamp = self.get_clock().now().to_msg()
                    if now - last_qr >= qr_period * 0.95 and self.qr_pub.get_subscription_count():
                        last_qr = now
                        self.qr_pub.publish(self.image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), stamp, 'mono8'))
                    if now - last >= period * 0.95:
                        last = now
                        small = frame if (frame.shape[1], frame.shape[0]) == size else \
                            cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                        self.pub.publish(self.image(small, stamp, 'bgr8'))
                        self.frames += 1
                    time.sleep(0.04)
                continue

            qr = self.want_qr
            boost = qr and self.boost
            v4l2_dev = str(self.g('v4l2_device')).strip()
            if v4l2_dev and os.path.exists(v4l2_dev):
                self.get_logger().info(f'opening V4L2 VRX device {v4l2_dev}')
                cap = cv2.VideoCapture(v4l2_dev, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap = cv2.VideoCapture(v4l2_dev)
            else:
                pipe = self.pipeline(qr, boost)
                self.get_logger().info(f'opening ({"QR" if qr else "normal"} mode) {pipe}')
                cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
            if not cap.isOpened():
                self.get_logger().error('camera did not open (overlay enabled? /dev/video0? nvargus-daemon running '
                                        'on the host?): retrying in 3 s')
                time.sleep(3.0)
                continue
            period = 1.0 / max(0.1, self.g('rate'))
            qr_period = 1.0 / max(0.1, self.g('qr_rate'))
            size = (self.g('width'), self.g('height'))
            last = last_qr = 0.0
            while self.running and self.want_qr == qr and (qr and self.boost) == boost:
                ok, frame = cap.read()
                if not ok:
                    self.get_logger().warn('no frame: reopening the camera')
                    break
                now = time.monotonic()
                stamp = self.get_clock().now().to_msg()
                if qr and now - last_qr >= qr_period * 0.95 and self.qr_pub.get_subscription_count():
                    last_qr = now
                    self.qr_pub.publish(self.image(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), stamp, 'mono8'))
                if now - last < period * 0.95:
                    continue
                last = now
                small = frame if (frame.shape[1], frame.shape[0]) == size else \
                    cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
                self.pub.publish(self.image(small, stamp, 'bgr8'))
                self.frames += 1
                if now - self.t_log > 10.0:
                    self.get_logger().info(f'{self.frames / (now - self.t_log):.1f} Hz {size[0]}x{size[1]}'
                                           f'{" (QR mode: " + str(frame.shape[1]) + " px grey on " + self.g("qr_topic") + ")" if qr else ""}')
                    self.frames, self.t_log = 0, now
            cap.release()
            time.sleep(0.5 if self.running and (self.want_qr != qr or (qr and self.boost) != boost) else 1.0)

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main():
    spin_node(ArducamCsi)


if __name__ == '__main__':
    main()
