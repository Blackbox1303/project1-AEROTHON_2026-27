"""Launch the Simulated Drone on the Saved RealSense Map (runs/saved_tunnel_map/saved_map.npz):
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_sim_saved.launch.py [rviz:=true] [autonav:=true]
Loads the 2D SLAM map + 3D point cloud scanned by your real Intel RealSense camera, spawns a simulated drone,
raycasts the saved obstacles as the drone moves, plans an obstacle-free 2D/3D path, and flies the sim drone
through the obstacles so you can watch it on http://10.42.0.247:8093/ and in RViz2.
"""
import os
import subprocess
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WS = Path.home() / 'workspaces/isaac_ros-dev'
HERE = Path(__file__).resolve().parent


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    map_file = g('map_file')
    use_rviz = g('rviz').lower() in ('true', '1', 'yes')
    autonav = g('autonav').lower() in ('true', '1', 'yes')
    port = g('port')

    actions = []
    dash_running = subprocess.run(['pgrep', '-f', 'tunnel_gcs.py'], capture_output=True).returncode == 0
    if not dash_running and g('dashboard').lower() in ('true', '1', 'yes'):
        actions.append(ExecuteProcess(
            cmd=['python3', str(WS / 'src/gcs/tunnel_gcs.py'), '--port', port],
            output='screen'))

    actions.append(ExecuteProcess(
        cmd=['python3', str(WS / 'src/aerothon_nav/aerothon_nav/sim_saved_map_drone.py'),
             '--ros-args', '-p', f'map_file:={map_file}', '-p', f'speed:={g("speed")}',
             '-p', f'clearance:={g("clearance")}'],
        output='screen'))

    if use_rviz:
        actions.append(Node(
            package='rviz2', executable='rviz2', name='rviz2_tunnel', output='screen',
            arguments=['-d', str(WS / 'src/vio_bringup/rviz/tunnel_slam.rviz')]))

    if autonav:
        actions.append(TimerAction(
            period=2.0,
            actions=[ExecuteProcess(
                cmd=['ros2', 'service', 'call', '/sim_map/navigate', 'std_srvs/srv/Trigger', '{}'],
                output='screen')]))

    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('map_file', default_value=str(WS / 'runs/saved_tunnel_map/saved_map.npz'),
                              description='Saved RealSense map .npz file'),
        DeclareLaunchArgument('speed', default_value='0.65', description='Sim drone cruise speed (m/s)'),
        DeclareLaunchArgument('clearance', default_value='0.35', description='Obstacle clearance radius (m)'),
        DeclareLaunchArgument('rviz', default_value='false', description='Open RViz2 with tunnel_slam.rviz'),
        DeclareLaunchArgument('autonav', default_value='true', description='Automatically start navigating obstacles'),
        DeclareLaunchArgument('dashboard', default_value='true', description='Ensure :8093 dashboard is running'),
        DeclareLaunchArgument('port', default_value='8093', description='Dashboard port'),
        SetEnvironmentVariable('DISPLAY', os.environ.get('DISPLAY', ':0')),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(WS / 'fastdds_udp.xml')),
        OpaqueFunction(function=setup),
    ])
