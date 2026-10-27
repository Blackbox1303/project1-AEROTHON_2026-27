"""Republish the Pixhawk IMU with clean timestamps for VIO.

MAVROS stamps /mavros/imu/data_raw close to PC arrival time, which carries the
USB/serial latency jitter (bench: 0-11 ms spread). PX4's own HIGHRES_IMU
time_usec is on a clean sensor grid, so we decode the raw MAVLink message and
map that FCU time onto the PC clock ourselves:

    stamp = fcu_time + offset_estimate + time_offset

offset_estimate follows the lower envelope of (arrival - fcu_time), i.e. the
least-delayed packets, and creeps upward slowly to track clock drift. Any
constant transport delay left in it is absorbed by `time_offset`, which comes
from the Kalibr camera-IMU calibration (time_offset = -timeshift_cam_imu).

HIGHRES_IMU is body FRD; output is FLU in `frame_id` (rigidly the Pixhawk).

stamp_source:=fcu (simulation): stamp = fcu_time + time_offset, no mapping. PX4 SITL runs on
Gazebo's clock, so its time_usec already is the sim time the simulated camera images carry.
"""
import struct

from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from mavros_msgs.msg import Mavlink
from sensor_msgs.msg import Imu

from vio_bringup.spin import spin_node

HIGHRES_IMU = 105
HIGHRES_IMU_LEN = 63  # MAVLink2 trims trailing zero bytes, so pad back up to this


class ImuRelay(Node):
    def __init__(self):
        super().__init__('imu_relay')
        self.declare_parameter('frame_id', 'imu_link')
        self.declare_parameter('time_offset', 0.0)       # seconds, from Kalibr
        self.declare_parameter('drift_alpha', 0.002)     # upward creep per sample
        self.declare_parameter('accel_noise_density', 0.02)
        self.declare_parameter('gyro_noise_density', 0.002)
        self.declare_parameter('rate_hz', 200.0)
        self.declare_parameter('stamp_source', 'pc')     # 'pc' (real Pixhawk) or 'fcu' (SITL)
        self.frame_id = self.get_parameter('frame_id').value
        self.time_offset = self.get_parameter('time_offset').value
        self.alpha = self.get_parameter('drift_alpha').value
        self.stamp_source = self.get_parameter('stamp_source').value
        if self.stamp_source not in ('pc', 'fcu'):
            raise ValueError(f"stamp_source must be 'pc' or 'fcu', not {self.stamp_source!r}")
        # discrete-time variance = density^2 * rate
        rate = self.get_parameter('rate_hz').value
        self.acc_var = self.get_parameter('accel_noise_density').value ** 2 * rate
        self.gyro_var = self.get_parameter('gyro_noise_density').value ** 2 * rate

        self.offset = None
        self.last_fcu_us = None
        self.dropped = 0
        self.pub = self.create_publisher(Imu, '/vio/imu', qos_profile_sensor_data)
        self.create_subscription(Mavlink, '/uas1/mavlink_source', self.on_mavlink,
                                 qos_profile_sensor_data)
        self.create_timer(10.0, self.report)

    def on_mavlink(self, m):
        if m.msgid != HIGHRES_IMU:
            return
        arrival = self.get_clock().now().nanoseconds * 1e-9
        raw = struct.pack(f'<{len(m.payload64)}Q', *m.payload64)[:m.len]
        raw = raw.ljust(HIGHRES_IMU_LEN, b'\0')
        fcu_us = struct.unpack_from('<Q', raw, 0)[0]
        xacc, yacc, zacc, xgyro, ygyro, zgyro = struct.unpack_from('<6f', raw, 8)

        if self.last_fcu_us is not None and fcu_us <= self.last_fcu_us:
            if self.last_fcu_us - fcu_us > 1_000_000:
                self.get_logger().warn('FCU time jumped back (reboot?), resetting time sync')
                self.offset = None
            else:
                self.dropped += 1
                return
        self.last_fcu_us = fcu_us

        fcu = fcu_us * 1e-6
        if self.stamp_source == 'fcu':
            self.offset = 0.0
        else:
            o = arrival - fcu
            if self.offset is None or o < self.offset:
                self.offset = o
            else:
                self.offset += self.alpha * (o - self.offset)

        msg = Imu()
        msg.header.stamp = Time(seconds=fcu + self.offset + self.time_offset).to_msg()
        msg.header.frame_id = self.frame_id
        # FRD -> FLU
        msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z = (
            float(xacc), float(-yacc), float(-zacc))
        msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = (
            float(xgyro), float(-ygyro), float(-zgyro))
        msg.orientation_covariance[0] = -1.0  # no orientation provided
        for i in (0, 4, 8):
            msg.linear_acceleration_covariance[i] = self.acc_var
            msg.angular_velocity_covariance[i] = self.gyro_var
        self.pub.publish(msg)

    def report(self):
        if self.offset is not None:
            self.get_logger().info(
                f'FCU->PC offset {self.offset:.4f} s, dropped non-increasing {self.dropped}')


def main():
    spin_node(ImuRelay)


if __name__ == '__main__':
    main()
