"""Broadcast the Pixhawk attitude as a TF so RViz shows the board rotating.

Only orientation is used: double-integrating the accelerometer for position
drifts by metres within seconds, so translation is left at zero.

The attitude goes on MAVROS' own frames (odom -> base_link) plus a static
map -> odom, which joins MAVROS' static map_ned / odom_ned / base_link_frd
into one tree. Only for the IMU-only view: in the full stack cuVSLAM owns
map -> odom -> imu_link, so this node is not started there.
"""
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from geometry_msgs.msg import TransformStamped
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from vio_bringup.spin import spin_node


class ImuTfBroadcaster(Node):
    def __init__(self):
        super().__init__('imu_tf_broadcaster')
        self.declare_parameter('world_frame', 'odom')
        self.declare_parameter('body_frame', 'base_link')
        self.declare_parameter('map_frame', 'map')
        self.world = self.get_parameter('world_frame').value
        self.body = self.get_parameter('body_frame').value

        self.static_br = StaticTransformBroadcaster(self)
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.get_parameter('map_frame').value
        t.child_frame_id = self.world
        t.transform.rotation.w = 1.0
        self.static_br.sendTransform(t)

        self.br = TransformBroadcaster(self)
        self.create_subscription(Imu, '/mavros/imu/data', self.on_imu, qos_profile_sensor_data)

    def on_imu(self, msg):
        t = TransformStamped()
        t.header.stamp = msg.header.stamp
        t.header.frame_id = self.world
        t.child_frame_id = self.body
        t.transform.rotation = msg.orientation
        self.br.sendTransform(t)


def main():
    spin_node(ImuTfBroadcaster)


if __name__ == '__main__':
    main()
