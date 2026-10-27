"""Gazebo -> ROS 2 bridge for the tunnel sim. Run on the HOST (needs ros-humble-ros-gzharmonic).

Publishes the simulated D435 on the real camera's topics (/camera/infra1, /camera/infra2,
/camera/depth), /clock, and the ground truth /sim/ground_truth/odom. map -> vio_map (cuVSLAM's frame in sim)
comes from vio_to_px4 (sim_vio.launch.py), re-anchored at every cuVSLAM reset.
The camera's own fixed frames are published by sim_vio.launch.py in the container:
static transforms published here never reached the container (2026-10-18).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('vio_bringup')
    return LaunchDescription([
        SetEnvironmentVariable('GZ_IP', '127.0.0.1'),  # same as PX4 / sim/run_sim.sh
        Node(package='ros_gz_bridge', executable='parameter_bridge', name='sim_bridge', output='screen',
             parameters=[{'config_file': os.path.join(pkg, 'config', 'sim_bridge.yaml'),
                          'use_sim_time': True}]),
    ])
