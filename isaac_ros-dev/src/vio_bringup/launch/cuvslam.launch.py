"""cuVSLAM on the D435 IR pair with the Pixhawk IMU.

imu:=false runs stereo-only; imu:=true fuses /vio/imu (clean-stamped Pixhawk IMU
from imu_relay). Extrinsics, time offset and IMU noise come from config/rig.yaml.
Pose is reported for imu_link (the Pixhawk body), i.e. the drone.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vio_bringup.rig import imu_to_camera_tf_node, load_rig
from vio_bringup.vslam import visual_slam


def generate_launch_description():
    pkg = get_package_share_directory('vio_bringup')
    rig = load_rig()

    return LaunchDescription([
        DeclareLaunchArgument('imu', default_value='false'),
        DeclareLaunchArgument('mavros', default_value='true'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('camera_ns', default_value='camera'),
        DeclareLaunchArgument('depth', default_value='false'),
        DeclareLaunchArgument('monitor', default_value='true'),
        # 60 Hz: with the camera depth stream on, 90 Hz drops ~1 % of frames (M6-1)
        DeclareLaunchArgument('profile', default_value='640x360x60'),
        # feed VIO to PX4's EKF2 - off by default on the real drone; set the Pixhawk 6C Mini
        # params first (config/px4_6c_vio_params.txt)
        DeclareLaunchArgument('px4', default_value='false'),
        # publish_tf:=false: odometry topics only, no map -> odom -> imu_link TF; base_tf:=false: no
        # imu_link -> base_link either. For the tunnel stack (launch_all/tunnel_slam.launch.py), whose
        # slam_toolbox + odom_tf own map -> odom -> base_link -> imu_link.
        DeclareLaunchArgument('publish_tf', default_value='true'),
        DeclareLaunchArgument('base_tf', default_value='true'),
        DeclareLaunchArgument('color', default_value='false'),  # + RGB and depth aligned to it (RTAB-Map colour)
        DeclareLaunchArgument('initial_reset', default_value='true'),
        # px4: VIO body velocity to EKF2; ardupilot: full pose + velocity for ArduCopter ExternalNav (vio_to_px4)
        DeclareLaunchArgument('autopilot', default_value='px4'),
        # obstacle_scan on the depth stream (default: with depth); the tunnel stack runs its own
        DeclareLaunchArgument('scan', default_value=LaunchConfiguration('depth')),
        # Includes are wrapped in scoped groups: launch arguments otherwise leak back
        # into this file (their rviz:=false used to switch off our own RViz).
        GroupAction([IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(pkg, 'launch', 'realsense_stereo.launch.py')),
            launch_arguments={'rviz': 'false',
                              'camera_ns': LaunchConfiguration('camera_ns'),
                              'depth': LaunchConfiguration('depth'),
                              'monitor': LaunchConfiguration('monitor'),
                              'profile': LaunchConfiguration('profile'),
                              'color': LaunchConfiguration('color'),
                              'initial_reset': LaunchConfiguration('initial_reset')}.items())]),
        GroupAction([IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'mavros_imu.launch.py')),
            launch_arguments={'rviz': 'false', 'tf_viz': 'false'}.items())],
            condition=IfCondition(LaunchConfiguration('mavros'))),
        imu_to_camera_tf_node(rig),
        # MAVROS' body frame is the Pixhawk body: hang base_link (+ its base_link_frd) off imu_link
        Node(package='tf2_ros', executable='static_transform_publisher', name='imu_to_base_link_tf',
             arguments=['--frame-id', 'imu_link', '--child-frame-id', 'base_link'],
             condition=IfCondition(LaunchConfiguration('base_tf'))),
        Node(package='vio_bringup', executable='obstacle_scan', output='screen',
             condition=IfCondition(LaunchConfiguration('scan'))),
        *visual_slam(rig, LaunchConfiguration('imu'), publish_tf=LaunchConfiguration('publish_tf')),
        Node(package='vio_bringup', executable='vio_to_px4', output='screen',
             parameters=[{'autopilot': LaunchConfiguration('autopilot')}],
             condition=IfCondition(LaunchConfiguration('px4'))),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(pkg, 'rviz', 'cuvslam.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
