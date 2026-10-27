"""Step 2: D435 IR stereo pair as cuVSLAM wants it (emitter off), plus monitor.

The D435 has no IMU, so gyro/accel stay disabled; the IMU comes from PX4.
Topics end up on /camera/infra{1,2}/...: realsense-ros 4.51 (Isaac container)
puts them under the namespace, 4.54+ (host apt) under namespace/node name, so
on the host run with camera_ns:=/. Both profile parameter spellings are set.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('vio_bringup')
    return LaunchDescription([
        DeclareLaunchArgument('profile', default_value='640x360x90'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('camera_ns', default_value='camera'),
        # on-camera stereo depth (same IR pair, emitter off) for obstacle detection
        DeclareLaunchArgument('depth', default_value='false'),
        DeclareLaunchArgument('monitor', default_value='true'),
        # color:=true adds the RGB stream + depth aligned to it (/camera/color/image_raw,
        # /camera/aligned_depth_to_color/image_raw) for a coloured RTAB-Map; the IR pair stays as cuVSLAM wants it
        DeclareLaunchArgument('color', default_value='false'),
        # false when the caller already USB-reset the camera (tunnel_slam.launch.py): a second reset re-enumerates it
        # again and costs ~10 s
        DeclareLaunchArgument('initial_reset', default_value='true'),
        DeclareLaunchArgument('color_profile', default_value='640x360x30'),
        Node(
            name='camera', namespace=LaunchConfiguration('camera_ns'),
            package='realsense2_camera', executable='realsense2_camera_node',
            output='screen',
            parameters=[{
                # hardware-reset the D435 on start: it can come up streaming
                # nothing after its previous process was killed mid-stream
                'initial_reset': LaunchConfiguration('initial_reset'),
                'enable_infra1': True,
                'enable_infra2': True,
                'enable_color': LaunchConfiguration('color'),
                'rgb_camera.profile': LaunchConfiguration('color_profile'),
                'rgb_camera.color_profile': LaunchConfiguration('color_profile'),
                'align_depth.enable': LaunchConfiguration('color'),
                'enable_depth': LaunchConfiguration('depth'),
                'enable_gyro': False,
                'enable_accel': False,
                'depth_module.emitter_enabled': 0,
                'depth_module.profile': LaunchConfiguration('profile'),
                'depth_module.infra_profile': LaunchConfiguration('profile'),
                'depth_module.depth_profile': LaunchConfiguration('profile'),
                # stamp frames with capture time in the PC clock, needed to
                # line them up with the Pixhawk IMU
                'depth_module.global_time_enabled': True,
            }]),
        Node(package='vio_bringup', executable='image_monitor', output='screen',
             condition=IfCondition(LaunchConfiguration('monitor'))),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(pkg, 'rviz', 'stereo.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
