"""Ask PX4 to stream the IMU-related MAVLink messages at a high rate.

MAVROS builds /mavros/imu/data from ATTITUDE_QUATERNION + HIGHRES_IMU and
/mavros/imu/data_raw from HIGHRES_IMU. PX4's default rates for these on a
USB link are far below what VIO needs, so we request them explicitly once
the FCU is connected, and again after every reconnect: a rebooted PX4 falls
back to its default stream rates.
"""
from rclpy.node import Node
from mavros_msgs.msg import State
from mavros_msgs.srv import MessageInterval

from vio_bringup.spin import spin_node

# MAVLink message ids
HIGHRES_IMU = 105
ATTITUDE_QUATERNION = 31


class ImuRateSetter(Node):
    def __init__(self):
        super().__init__('imu_rate_setter')
        self.declare_parameter('rate_hz', 200.0)
        self.rate_hz = self.get_parameter('rate_hz').value

        self.cli = self.create_client(MessageInterval, '/mavros/set_message_interval')
        self.connected = False
        self.done = False
        self.create_subscription(State, '/mavros/state', self.on_state, 10)
        self.timer = self.create_timer(2.0, self.try_set)

    def on_state(self, msg):
        if self.connected and not msg.connected:
            self.get_logger().warn('FCU disconnected, will re-request IMU rates on reconnect')
            self.done = False
        self.connected = msg.connected

    def try_set(self):
        if self.done:
            return
        if not self.connected or not self.cli.service_is_ready():
            self.get_logger().info('Waiting for FCU connection / set_message_interval service...')
            return
        for msg_id in (HIGHRES_IMU, ATTITUDE_QUATERNION):
            req = MessageInterval.Request(message_id=msg_id, message_rate=float(self.rate_hz))
            self.cli.call_async(req).add_done_callback(
                lambda fut, m=msg_id: self.on_result(fut, m))
        self.done = True

    def on_result(self, fut, msg_id):
        ok = fut.result() is not None and fut.result().success
        self.get_logger().info(
            f'Request msg {msg_id} @ {self.rate_hz:.0f} Hz: {"OK" if ok else "FAILED"}')


def main():
    spin_node(ImuRateSetter)


if __name__ == '__main__':
    main()
