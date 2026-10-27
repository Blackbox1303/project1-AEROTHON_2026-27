"""CameraInfo for the Arducam fisheye, one per image, same stamp and frame.

Two sources for the lens, both the equidistant model of image_geometry / cv2.fisheye:
  calibration_file  a YAML from calibration/arducam_calib.py (real camera). If the image size
                    differs, the lens is scaled by the HEIGHT ratio (same pixel pitch both ways) and
                    the width difference treated as a centred crop: the IMX219's 3264x2464 mode is
                    the 3280-wide sensor less 8 columns each side, 1640x1232 is all 3280 binned 2x2.
                    Calibrating in the mode you fly avoids relying on that.
  otherwise         the lens Gazebo's wide-angle camera renders: r = f * theta, D = 0, with
                    f = gz_focal_scale * (width / 2) / (hfov / 2). gz renders f 0.51 % longer than
                    that formula: measured 675.0 px at 1640 wide from markers at 20/40/60 deg off
                    axis (2026-10-19), against 671.6 from the formula.
Gazebo's own camera_info is a pinhole model from the 140 deg FOV, useless for unprojection.
  in:  image_topic (sensor_msgs/Image)
  out: info_topic  (sensor_msgs/CameraInfo)
Subscribes RELIABLE: a 6 MB frame is hundreds of UDP fragments, and best effort lost most of
them (7 of 21 Hz arrived). Subscribers of /arducam/image_raw should do the same.
"""
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image

from vio_bringup.spin import spin_node


class FisheyeCameraInfo(Node):
    def __init__(self):
        super().__init__('fisheye_camera_info')
        self.declare_parameter('image_topic', '/arducam/image_raw')
        self.declare_parameter('info_topic', '/arducam/camera_info')
        self.declare_parameter('calibration_file', '')
        self.declare_parameter('hfov', 2.44203)          # rad, as in the model's <horizontal_fov>
        self.declare_parameter('gz_focal_scale', 1.0051)
        self.calib = None
        path = self.get_parameter('calibration_file').value
        if path:
            with open(path) as f:
                c = yaml.safe_load(f)
            if c.get('distortion_model') != 'equidistant':
                raise ValueError(f'{path}: distortion_model {c.get("distortion_model")!r}, expected equidistant')
            self.calib = (c['image_width'], c['image_height'], c['camera_matrix']['data'],
                          c['distortion_coefficients']['data'])
            self.get_logger().info(f'lens from {path} ({c["image_width"]}x{c["image_height"]})')
        self.hfov = self.get_parameter('hfov').value
        self.focal_scale = self.get_parameter('gz_focal_scale').value
        self.size = None
        self.info = CameraInfo()
        self.pub = self.create_publisher(CameraInfo, self.get_parameter('info_topic').value, 10)
        self.create_subscription(Image, self.get_parameter('image_topic').value, self.on_image, 2)

    def lens(self, w, h):
        """fx, fy, cx, cy, D for a w x h image."""
        if self.calib is None:
            f = self.focal_scale * w / self.hfov      # (w / 2) / (hfov / 2)
            return f, f, (w - 1) / 2, (h - 1) / 2, [0.0] * 4
        cw, ch, K, D = self.calib
        s = h / ch                                   # binning factor (rows are never cropped)
        crop = (cw * s - w) / 2                      # columns cut off each side, in the new pixels
        return K[0] * s, K[4] * s, (K[2] + 0.5) * s - 0.5 - crop, (K[5] + 0.5) * s - 0.5, list(D)

    def on_image(self, img):
        if self.size != (img.width, img.height):
            self.size = (img.width, img.height)
            fx, fy, cx, cy, d = self.lens(img.width, img.height)
            self.info.width, self.info.height = img.width, img.height
            self.info.distortion_model = 'equidistant'
            self.info.d = [float(x) for x in d]
            self.info.k = [fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0]
            self.info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
            self.info.p = [fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0]
            self.get_logger().info(f'{img.width}x{img.height}, equidistant fx {fx:.1f} fy {fy:.1f} px, '
                                   f'D {", ".join(f"{x:+.4f}" for x in d)}')
        self.info.header = img.header
        self.pub.publish(self.info)


def main():
    spin_node(FisheyeCameraInfo)


if __name__ == '__main__':
    main()
