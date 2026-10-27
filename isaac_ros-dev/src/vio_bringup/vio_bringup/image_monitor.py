"""Report D435 stereo stream health: per-camera rate, drops, and L/R sync.

cuVSLAM needs the left/right IR frames to carry identical stamps; the D435
captures both from one sensor module, so any offset here indicates a problem.
"""
import statistics

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from vio_bringup.spin import spin_node


def to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


class ImageMonitor(Node):
    def __init__(self):
        super().__init__('image_monitor')
        self.declare_parameter('left', '/camera/infra1/image_rect_raw')
        self.declare_parameter('right', '/camera/infra2/image_rect_raw')
        self.stamps = {'left': [], 'right': []}
        self.info = {}
        self.encoding = {}
        for side in ('left', 'right'):
            topic = self.get_parameter(side).value
            self.create_subscription(
                Image, topic, lambda m, s=side: self.on_image(s, m), qos_profile_sensor_data)
            info_topic = topic.rsplit('/', 1)[0] + '/camera_info'
            self.create_subscription(
                CameraInfo, info_topic, lambda m, s=side: self.info.__setitem__(s, m),
                qos_profile_sensor_data)
        self.create_timer(2.0, self.report)

    def on_image(self, side, msg):
        self.stamps[side].append(to_sec(msg.header.stamp))
        self.encoding[side] = f'{msg.width}x{msg.height} {msg.encoding}'

    def report(self):
        parts = []
        for side, st in self.stamps.items():
            if len(st) < 3:
                parts.append(f'{side}: NO DATA')
                continue
            dts = [b - a for a, b in zip(st, st[1:])]
            mean_dt = statistics.mean(dts)
            drops = sum(d > 1.5 * mean_dt for d in dts)
            parts.append(f'{side}: {1.0 / mean_dt:5.1f} Hz drops {drops} [{self.encoding[side]}]')
        common = set(self.stamps['left']) & set(self.stamps['right'])
        n = max(len(self.stamps['left']), 1)
        parts.append(f'L/R identical stamps {len(common)}/{n}')
        if 'right' in self.info:
            p = self.info['right'].p
            if p[0] > 0:
                parts.append(f'baseline {-p[3] / p[0] * 1000:.1f} mm')
        if 'left' in self.info:
            k = self.info['left'].k
            parts.append(f'fx {k[0]:.1f} cx {k[2]:.1f} cy {k[5]:.1f}')
        self.get_logger().info(' | '.join(parts))
        self.stamps = {'left': [], 'right': []}


def main():
    spin_node(ImageMonitor)


if __name__ == '__main__':
    main()
