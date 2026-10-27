"""Report IMU stream health: rate, header-stamp jitter, and gravity sanity.

Header stamps come from the FCU clock mapped through MAVROS timesync, so
their jitter is what a VIO backend will actually see.
"""
import math
import statistics

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu

from vio_bringup.spin import spin_node


class ImuMonitor(Node):
    def __init__(self):
        super().__init__('imu_monitor')
        self.declare_parameter('topic', '/mavros/imu/data_raw')
        self.declare_parameter('report_period', 2.0)
        topic = self.get_parameter('topic').value

        self.stamps = []
        self.recv = []
        self.acc_norms = []
        self.gyro_norms = []
        self.create_subscription(Imu, topic, self.on_imu, qos_profile_sensor_data)
        self.create_timer(self.get_parameter('report_period').value, self.report)
        self.get_logger().info(f'Monitoring {topic}')

    def on_imu(self, msg):
        self.stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
        self.recv.append(self.get_clock().now().nanoseconds * 1e-9)
        a, w = msg.linear_acceleration, msg.angular_velocity
        self.acc_norms.append(math.sqrt(a.x**2 + a.y**2 + a.z**2))
        self.gyro_norms.append(math.sqrt(w.x**2 + w.y**2 + w.z**2))
        self.last = msg

    def report(self):
        if len(self.stamps) < 3:
            self.get_logger().warn('No / too few IMU messages')
            return
        dts = [b - a for a, b in zip(self.stamps, self.stamps[1:])]
        span = self.recv[-1] - self.recv[0]
        rate = (len(self.recv) - 1) / span if span > 0 else float('nan')
        mean_dt = statistics.mean(dts)
        a = self.last.linear_acceleration
        self.get_logger().info(
            f'rate {rate:6.1f} Hz | stamp dt mean {mean_dt*1e3:5.2f} ms '
            f'std {statistics.pstdev(dts)*1e3:5.2f} ms max {max(dts)*1e3:6.2f} ms '
            f'non-increasing {sum(d <= 0 for d in dts)} | '
            f'|acc| {statistics.mean(self.acc_norms):5.2f} m/s^2 '
            f'acc=({a.x:+.2f},{a.y:+.2f},{a.z:+.2f}) | '
            f'|gyro| {statistics.mean(self.gyro_norms):.3f} rad/s | '
            f'frame {self.last.header.frame_id}')
        self.stamps, self.recv, self.acc_norms, self.gyro_norms = [], [], [], []


def main():
    spin_node(ImuMonitor)


if __name__ == '__main__':
    main()
