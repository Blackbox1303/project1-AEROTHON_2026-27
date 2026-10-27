"""Launch the Delivery-Only Mission Stack using a USB VRX video receiver + Ground Telemetry Radio link.

Supports two ways of feeding the VRX video into the ROS 2 mission stack (`qr_reader`, `qr_locator`,
`delivery_mapper`, `frontier_explorer`, `mission_supervisor`):
  1. Direct USB VRX device (`vrx_device:=/dev/video2`), or
  2. Remote HTTP stream from the laptop running `laptop_vrx_mission.py` (`vrx_http_url:=http://10.42.0.1:8082/frame.jpg`).

And connects `mavlink-router` + `MAVROS` to the Ground Telemetry Radio (`link:=/dev/ttyUSB0:57600` or `udp:14555`).

Usage:
  ros2 launch /home/szyfrowac/workspaces/isaac_ros-dev/src/launch_all/vrx_delivery.launch.py \
    link:=/dev/ttyUSB0:57600 \
    vrx_device:=/dev/video2 \
    world_json:=/home/szyfrowac/workspaces/isaac_ros-dev/runs/gcs/mission_geofence.json
"""
import os
from pathlib import Path
import shutil
import subprocess
import time

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    OpaqueFunction,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / 'scripts'
WS = HERE.parent.parent
CONTAINER = 'isaac_ros_dev-aarch64-pyzbar-container'


def router_conf(link, gcs_ip, path):
    if link.startswith('udp:'):
        fcu = f'[UdpEndpoint fcu]\nMode = Server\nAddress = 0.0.0.0\nPort = {int(link[4:])}\n'
    else:
        dev, _, baud = link.partition(':')
        fcu = f'[UartEndpoint fcu]\nDevice = {dev}\nBaud = {int(baud or 57600)}\n'
    path.write_text(
        '[General]\nTcpServerPort = 5760\nReportStats = false\n\n' + fcu +
        '\n[UdpEndpoint mavros]\nMode = Normal\nAddress = 127.0.0.1\nPort = 14640\n'
        '\n[UdpEndpoint gcs_dashboard]\nMode = Normal\nAddress = 127.0.0.1\nPort = 14650\n'
        f'\n[UdpEndpoint qgc]\nMode = Normal\nAddress = {gcs_ip}\nPort = 14550\n'
        '\n[UdpEndpoint qgc_any]\nMode = Server\nAddress = 0.0.0.0\nPort = 14551\n'
    )


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    run = g('run') or f'vrx_delivery_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    run_dir = WS / 'runs' / run
    run_c = f'/workspaces/isaac_ros-dev/runs/{run}'
    (run_dir / 'ros_logs').mkdir(parents=True, exist_ok=True)
    gj = Path(g('world_json'))
    if gj.is_file():
        shutil.copy(gj, run_dir / 'geofence_used.json')
    else:
        # Default delivery geofence around origin for test drone if none applied yet
        default_gf = {
            'frame': 'odom',
            'delivery_zone': {'x_from': -10.0, 'x_to': 25.0, 'y_from': -12.0, 'y_to': 12.0},
            'delivery_polygon': [[-10.0, -12.0], [25.0, -12.0], [25.0, 12.0], [-10.0, 12.0]],
            'field_polygon': [[-15.0, -18.0], [30.0, -18.0], [30.0, 18.0], [-15.0, 18.0]],
            'corridors': [], 'red_zones': [],
        }
        (run_dir / 'geofence_used.json').write_text(json.dumps(default_gf, indent=2))

    conf = run_dir / 'router.conf'
    router_conf(g('link'), g('gcs_ip'), conf)
    subprocess.run(['systemctl', '--user', 'stop', 'arducam-calib'], check=False)

    container_up = ExecuteProcess(cmd=[str(SCRIPTS / 'container_up.sh')], output='screen')
    router = ExecuteProcess(cmd=['mavlink-routerd', '-c', str(conf)], output='screen')
    mavros = Node(
        package='mavros', executable='mavros_node', output='screen',
        parameters=[
            {'fcu_url': 'udp://:14640@', 'tgt_system': 1, 'tgt_component': 1},
            str(WS / 'src/vio_bringup/config/mavros_host_plugins.yaml'),
        ],
    )
    vrx_bridge = Node(
        package='vio_bringup', executable='arducam_csi', name='vrx_camera_bridge', output='screen',
        parameters=[{
            'shared_http_url': g('vrx_http_url'),
            'v4l2_device': g('vrx_device'),
            'force_qr': True,
        }],
    )
    mission = ExecuteProcess(
        cmd=[
            'docker', 'exec', '-u', 'admin', CONTAINER, 'bash',
            '/workspaces/isaac_ros-dev/src/launch_all/scripts/container_real.sh',
            run_c, f'{run_c}/geofence_used.json',
            '/workspaces/isaac_ros-dev/calibration/arducam_fisheye_clean.yaml', 'v2',
        ],
        output='screen',
    )
    return [
        SetEnvironmentVariable('ROS_LOG_DIR', str(run_dir / 'ros_logs')),
        container_up,
        RegisterEventHandler(OnProcessExit(target_action=container_up, on_exit=[
            router, mavros, vrx_bridge,
            TimerAction(period=6.0, actions=[mission]),
        ])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('link', default_value='/dev/ttyUSB0:57600',
                              description='Ground telemetry radio (/dev/ttyUSB0:57600) or UDP stream (udp:14555)'),
        DeclareLaunchArgument('vrx_device', default_value='/dev/video2',
                              description='USB VRX capture device path (e.g. /dev/video2 or /dev/video0)'),
        DeclareLaunchArgument('vrx_http_url', default_value='http://10.42.0.1:8082/frame.jpg',
                              description='Laptop VRX HTTP frame stream URL if VRX is plugged into laptop'),
        DeclareLaunchArgument('gcs_ip', default_value='10.42.0.1'),
        DeclareLaunchArgument('world_json', default_value=str(WS / 'runs/gcs/mission_geofence.json')),
        DeclareLaunchArgument('run', default_value=''),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(WS / 'fastdds_udp.xml')),
        OpaqueFunction(function=setup),
    ])
