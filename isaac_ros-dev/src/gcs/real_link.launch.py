"""Real drone link for the ground station: mavlink-router on the flight controller's serial port (router/real.conf:
set Device / Baud) splitting MAVLink to MAVROS, the dashboard (14650) and QGC on the Lenovo; MAVROS behind it.
The drone's own sensor + mission stack runs on the drone; this only makes it visible and commandable.
    ros2 launch ~/workspaces/isaac_ros-dev/src/gcs/real_link.launch.py [conf:=...]
"""
import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = os.path.dirname(os.path.realpath(__file__))
WS = os.path.expanduser('~/workspaces/isaac_ros-dev')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('conf', default_value=os.path.join(HERE, 'router/real.conf')),
        DeclareLaunchArgument('run', default_value='', description='(run name, from the dashboard; unused)'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', os.path.join(WS, 'fastdds_udp.xml')),
        ExecuteProcess(cmd=['mavlink-routerd', '-c', LaunchConfiguration('conf')], output='screen'),
        Node(package='mavros', executable='mavros_node', output='screen',
             parameters=[{'fcu_url': 'udp://:14640@', 'tgt_system': 1, 'tgt_component': 1},
                         os.path.join(WS, 'src/vio_bringup/config/mavros_host_plugins.yaml')]),
    ])
