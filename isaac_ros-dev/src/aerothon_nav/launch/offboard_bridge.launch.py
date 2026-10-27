"""MAVROS offboard bridge: /cmd_vel (body frame) + /nav/altitude -> PX4 setpoints.
Needs MAVROS running (sim: vio_bringup sim_view.launch.py).
    ros2 launch aerothon_nav offboard_bridge.launch.py
    ros2 service call /nav/offboard std_srvs/srv/SetBool "{data: true}"    # armed + airborne first
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('aerothon_nav'), 'config', 'offboard_bridge.yaml')
    return LaunchDescription([
        Node(package='aerothon_nav', executable='offboard_bridge', output='screen', parameters=[cfg]),
    ])
