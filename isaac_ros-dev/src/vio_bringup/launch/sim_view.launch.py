"""RViz view of the simulated drone: MAVROS against PX4 SITL + PX4's position estimate.

Shows /mavros/local_position/odom (PX4 EKF2 estimate, trail of arrows) and the
drone axes via MAVROS' map -> base_link TF. Start the sim first (sim/run_sim.sh).
    ros2 launch vio_bringup sim_view.launch.py
    ros2 launch vio_bringup sim_view.launch.py mavros_tf:=false rviz:=false   # navigation: aerothon_nav
        odom_tf owns base_link (odom -> base_link on sim time)
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, GroupAction,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('vio_bringup')
    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='udp://:14540@127.0.0.1:14557'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('mavros_tf', default_value='true'),
        DeclareLaunchArgument('mavros', default_value='true', description='false: MAVROS runs on the host'),
        DeclareLaunchArgument('map_tf', default_value='true',
                              description='false: something else publishes map -> odom (slam_toolbox)'),
        DeclareLaunchArgument('imu_rate', default_value='250.0', description='Hz: PX4 IMU streams (cuVSLAM)'),
        GroupAction([IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'mavros_imu.launch.py')),
            launch_arguments={'fcu_url': LaunchConfiguration('fcu_url'), 'mavros': LaunchConfiguration('mavros'),
                              'imu_rate': LaunchConfiguration('imu_rate'),
                              'rviz': 'false', 'tf_viz': 'false', 'sim': 'true'}.items())]),
        # MAVROS' map -> base_link TF is off by default; on here so the drone axes move.
        # Retried: a single attempt after a fixed delay raced MAVROS' start-up ("Node not found").
        TimerAction(period=3.0, actions=[ExecuteProcess(
            cmd=['bash', '-c', 'for _ in $(seq 60); do '
                 'ros2 param set /mavros/local_position tf.send true 2>/dev/null | grep -q successful '
                 '&& echo "MAVROS map -> base_link TF on" && exit 0; sleep 2; done; '
                 'echo "could not enable MAVROS map -> base_link TF"; exit 1'],
            output='screen')], condition=IfCondition(LaunchConfiguration('mavros_tf'))),
        # MAVROS publishes odom -> odom_ned on its own; connect odom to map so RViz has one tree
        Node(package='tf2_ros', executable='static_transform_publisher', name='map_to_odom_tf',
             arguments=['--frame-id', 'map', '--child-frame-id', 'odom'],
             condition=IfCondition(LaunchConfiguration('map_tf'))),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(pkg, 'rviz', 'sim.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
