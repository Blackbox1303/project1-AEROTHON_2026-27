"""Shared / Non-Intrusive IMX219 CSI Camera + Equidistant CameraInfo Node for `drone_qr_detector`.

Solves the single-process CSI hardware lock (`nvarguscamerasrc`) when another user/process is accessing the camera:
  1. `http_shared` (auto-detected when `arducam_calib.py` is running on 127.0.0.1:8090):
     Pulls raw full-resolution (3280x2464) frames from `http://127.0.0.1:8090/frame.jpg` over a keep-alive HTTP
     connection without stopping or disturbing the other user's session.
  2. `ros_topic` (auto-detected when `vio_bringup/arducam_csi` is already publishing `/arducam/qr/image_raw` or
     `/arducam/image_raw`):
     Subscribes to the existing ROS 2 camera topic so both stacks receive frames simultaneously.
  3. `csi` (when the CSI hardware is free):
     Opens `nvarguscamerasrc` directly at full sensor resolution (3280x2464 @ 21 fps, capped exposure for zero motion
     blur) using hardware `nvvidconv` directly to `GRAY8` (`mono8`) to eliminate 24 MB BGR CPU conversion overhead,
     and never drops the pipeline mid-flight. Also serves a lightweight read-only `/frame.jpg` on port 8091 so other
     tools can share the stream too.
"""
import array
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from drone_qr_detector.fisheye_model import FisheyeModel


class CsiCameraNode(Node):
    def __init__(self):
        super().__init__('csi_camera_node')
        p = self.declare_parameter
        p('source_mode', 'auto')          # 'auto' | 'http_shared' | 'ros_topic' | 'csi'
        p('shared_http_host', '127.0.0.1')
        p('shared_http_port', 8090)       # port of vio_bringup/calibration/arducam_calib.py
        p('share_server_port', 8092)      # read-only /frame.jpg server when this node owns direct CSI (8091 is QR HUD)
        p('existing_ros_qr_topic', '/arducam/qr/image_raw')
        p('existing_ros_image_topic', '/arducam/image_raw')
        p('sensor_id', 0)
        p('width', 3280)                  # Full IMX219 sensor resolution required for 1m QR at 10m altitude (~70 px)
        p('height', 2464)
        p('fps', 21)
        p('publish_rate', 6.0)            # Hz published on /drone_qr/image_raw
        p('exposure_ms', 6.0)             # Capped exposure to prevent motion blur on the drone
        p('flip_method', 0)
        p('frame_id', 'arducam_optical_frame')
        p('image_topic', '/drone_qr/image_raw')
        p('info_topic', '/drone_qr/camera_info')
        p('calibration_file', '/workspaces/isaac_ros-dev/calibration/arducam_fisheye_1640x1232.yaml')
        p('focal_scale', 1.0)
        p('camera_latency_ms', 45.0)      # Argus ISP + rolling shutter midpoint offset subtracted from header.stamp

        self.g = lambda name: self.get_parameter(name).value
        self.calib_model = FisheyeModel.from_yaml(
            self.g('calibration_file'),
            focal_scale=float(self.g('focal_scale')),
        )
        self.info_cache = {}
        self.latest_jpeg = None
        self.lock = threading.Lock()

        self.img_pub = self.create_publisher(Image, self.g('image_topic'), 2)
        self.info_pub = self.create_publisher(CameraInfo, self.g('info_topic'), 10)

        self.ros_frame = None
        self.ros_sub_qr = None
        self.ros_sub_rgb = None
        self._setup_ros_relay_subs()

        port_out = int(self.g('share_server_port'))
        if port_out > 0:
            self._start_share_server(port_out)

        self.running = True
        self.active_backend = 'initializing'
        self.frames_pub = 0
        self.t_log = time.monotonic()
        threading.Thread(target=self.loop, daemon=True).start()

    def _setup_ros_relay_subs(self):
        qr_top = self.g('existing_ros_qr_topic')
        im_top = self.g('existing_ros_image_topic')
        if qr_top:
            self.ros_sub_qr = self.create_subscription(Image, qr_top, self._on_existing_ros_img, 1)
        if im_top:
            self.ros_sub_rgb = self.create_subscription(Image, im_top, self._on_existing_ros_img, 1)

    def _on_existing_ros_img(self, msg):
        with self.lock:
            self.ros_frame = (time.monotonic(), msg)

    def _start_share_server(self, port):
        node = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                if self.path.startswith('/frame.jpg'):
                    with node.lock:
                        jpg = node.latest_jpeg
                    if jpg is None:
                        self.send_response(503)
                        self.end_headers()
                        return
                    self.send_response(200)
                    self.send_header('Content-Type', 'image/jpeg')
                    self.send_header('Content-Length', str(len(jpg)))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    try:
                        self.wfile.write(jpg)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                self.send_response(404)
                self.end_headers()

        def serve():
            try:
                srv = ThreadingHTTPServer(('0.0.0.0', port), Handler)
                srv.serve_forever()
            except OSError:
                pass

        threading.Thread(target=serve, daemon=True).start()

    def build_camera_info(self, width, height, stamp):
        key = (int(width), int(height))
        if key not in self.info_cache:
            m = self.calib_model.scaled(width, height)
            d = m.to_camera_info_dict()
            info = CameraInfo()
            info.width = d['width']
            info.height = d['height']
            info.distortion_model = d['distortion_model']
            info.d = d['d']
            info.k = d['k']
            info.r = d['r']
            info.p = d['p']
            self.info_cache[key] = info
            self.get_logger().info(
                f'CameraInfo ready for {width}x{height}: fx={info.k[0]:.1f} fy={info.k[4]:.1f} '
                f'cx={info.k[2]:.1f} cy={info.k[5]:.1f} D={[round(x, 4) for x in info.d]}'
            )
        info = self.info_cache[key]
        info.header.stamp = stamp
        info.header.frame_id = self.g('frame_id')
        return info

    def publish_gray_frame(self, gray, stamp=None):
        if stamp is None:
            now_ns = self.get_clock().now().nanoseconds - int(float(self.g('camera_latency_ms')) * 1e6)
            stamp = rclpy.time.Time(nanoseconds=max(0, now_ns)).to_msg()
        h, w = gray.shape[:2]
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self.g('frame_id')
        msg.height = int(h)
        msg.width = int(w)
        msg.encoding = 'mono8'
        msg.step = int(w)
        msg.data = array.array('B', gray.tobytes())
        self.img_pub.publish(msg)
        self.info_pub.publish(self.build_camera_info(w, h, stamp))
        self.frames_pub += 1
        now = time.monotonic()
        if now - self.t_log >= 10.0:
            hz = self.frames_pub / (now - self.t_log)
            self.get_logger().info(f'[{self.active_backend}] publishing {w}x{h} mono8 at {hz:.1f} Hz')
            self.frames_pub, self.t_log = 0, now

    def try_http_shared_frame(self, conn):
        """Fetch raw frame from running arducam_calib.py HTTP server (/frame.jpg) without disturbing CSI lock."""
        host = str(self.g('shared_http_host'))
        port = int(self.g('shared_http_port'))
        try:
            if conn is None:
                conn = http.client.HTTPConnection(host, port, timeout=1.5)
            conn.request('GET', '/frame.jpg')
            resp = conn.getresponse()
            body = resp.read()
            if resp.status == 200 and len(body) > 1000:
                with self.lock:
                    self.latest_jpeg = body
                arr = np.frombuffer(body, dtype=np.uint8)
                gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
                if gray is not None:
                    return conn, gray
        except Exception:
            try:
                if conn is not None:
                    conn.close()
            except Exception:
                pass
            conn = None
        return conn, None

    def try_existing_ros_frame(self):
        with self.lock:
            item = self.ros_frame
            self.ros_frame = None
        if item is None:
            return None, None
        t_recv, msg = item
        if time.monotonic() - t_recv > 1.5:
            return None, None
        a = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding in ('mono8', '8UC1'):
            gray = a.reshape(msg.height, msg.width)
        else:
            code = cv2.COLOR_RGB2GRAY if msg.encoding == 'rgb8' else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(a.reshape(msg.height, msg.width, 3), code)
        return gray, msg.header.stamp

    def gstreamer_pipeline(self):
        g = self.g
        w, h, fps = int(g('width')), int(g('height')), int(g('fps'))
        exp_ns = int(float(g('exposure_ms')) * 1e6)
        return (
            f'nvarguscamerasrc sensor-id={int(g("sensor_id"))} '
            f'exposuretimerange="13000 {exp_ns}" gainrange="1 10.625" ! '
            f'video/x-raw(memory:NVMM),width={w},height={h},framerate={fps}/1,format=NV12 ! '
            f'nvvidconv flip-method={int(g("flip_method"))} ! '
            f'video/x-raw,format=GRAY8 ! appsink drop=true max-buffers=1 sync=false'
        )

    def set_backend(self, name):
        if self.active_backend != name:
            self.active_backend = name
            self.get_logger().info(f'CSI camera backend switched to: {name}')

    def loop(self):
        os.environ.pop('DISPLAY', None)
        conn = None
        cap = None
        while self.running:
            mode = str(self.g('source_mode')).lower()
            period = 1.0 / max(0.2, float(self.g('publish_rate')))
            t0 = time.monotonic()

            # 1. Shared HTTP stream from arducam_calib.py (if another user is running it on port 8090)
            if mode in ('auto', 'http_shared'):
                conn, gray = self.try_http_shared_frame(conn)
                if gray is not None:
                    if cap is not None:
                        cap.release()
                        cap = None
                    self.set_backend(f'http_shared({self.g("shared_http_host")}:{self.g("shared_http_port")})')
                    self.publish_gray_frame(gray)
                    dt = time.monotonic() - t0
                    if dt < period:
                        time.sleep(period - dt)
                    continue
                if mode == 'http_shared':
                    time.sleep(0.5)
                    continue

            # 2. Existing ROS 2 topic relay (if vio_bringup/arducam_csi is running)
            if mode in ('auto', 'ros_topic'):
                gray, stamp = self.try_existing_ros_frame()
                if gray is not None:
                    if cap is not None:
                        cap.release()
                        cap = None
                    self.set_backend('ros_topic_relay')
                    self.publish_gray_frame(gray, stamp=stamp)
                    dt = time.monotonic() - t0
                    if dt < period:
                        time.sleep(period - dt)
                    continue
                if mode == 'ros_topic':
                    time.sleep(0.1)
                    continue

            # 3. Direct hardware GStreamer nvarguscamerasrc capture
            if mode in ('auto', 'csi'):
                if cap is None or not cap.isOpened():
                    pipe = self.gstreamer_pipeline()
                    self.get_logger().info(f'Opening direct CSI GStreamer pipeline: {pipe}')
                    cap = cv2.VideoCapture(pipe, cv2.CAP_GSTREAMER)
                    if not cap.isOpened():
                        cap = None
                        self.get_logger().warn(
                            'Direct CSI busy or unavailable; will keep checking shared HTTP (8090) / ROS topics...',
                            throttle_duration_sec=5.0,
                        )
                        time.sleep(1.0)
                        continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.get_logger().warn('Direct CSI frame read failed; releasing camera and re-probing sources')
                    cap.release()
                    cap = None
                    time.sleep(0.5)
                    continue
                gray = frame if frame.ndim == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                self.set_backend('direct_csi_nvargus')
                self.publish_gray_frame(gray)
                if int(self.g('share_server_port')) > 0:
                    ok_j, jpg = cv2.imencode('.jpg', gray, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    if ok_j:
                        with self.lock:
                            self.latest_jpeg = jpg.tobytes()
                dt = time.monotonic() - t0
                if dt < period:
                    time.sleep(period - dt)

        if cap is not None:
            cap.release()

    def destroy_node(self):
        self.running = False
        super().destroy_node()


def main():
    rclpy.init()
    node = CsiCameraNode()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
