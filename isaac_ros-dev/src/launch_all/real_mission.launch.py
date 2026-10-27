"""The REAL drone's whole mission stack on the onboard Jetson in one command - the counterpart of sim_mission.launch.py:
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/real_mission.launch.py link:=/dev/ttyACM0:921600
Normally started from the ground station dashboard (Real mode -> Launch stack); the mission itself is started there
too, after the preflight (no auto-start on the real drone).
Steps: scripts/container_up.sh (the dev container, if not running) -> mavlink-router on the Pixhawk 6C Mini (`link`:
USB /dev/ttyACM0:921600, TELEM2 on the Jetson UART /dev/ttyTHS1:921600, or udp:<port>) splitting MAVLink to MAVROS
(14640), the dashboard (14650) and QGroundControl (gcs_ip:14550) + MAVROS on the host -> after container_delay s
scripts/container_real.sh in the container (aerothon_nav container_real.launch.py: RealSense, Arducam, navigation,
supervisor) + scripts/save_run.sh loop (the container's ~/.ros mission files every 60 s).
runs/<run>/: container.log, mission_log.txt, altitude_state.txt, aerothon_* files, ros_logs/ (host: router, MAVROS),
geofence_used.json, launch_args.txt. PX4's flight log stays on the Pixhawk's SD card.
Needs: the ground station geofence (world_json, default runs/gcs/mission_geofence.json - Save + Apply on the
dashboard), the D435 on USB 3, the Arducam on CSI CAM0 (overlay enabled), and ideally the Arducam calibration.
Ctrl-C (or the dashboard's Stop) stops everything, in the container too (scripts/stop_all.sh).
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler,
                            SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / 'scripts'
WS = Path.home() / 'workspaces/isaac_ros-dev'
CONTAINER = 'isaac_ros_dev-aarch64-pyzbar-container'
CALIB = WS / 'calibration/arducam_fisheye_1640x1232.yaml'


def router_conf(link, gcs_ip, path):
    """mavlink-router config: the flight controller on `link`, out to MAVROS, the dashboard and QGC."""
    if link.startswith('udp:'):
        fcu = f'[UdpEndpoint fcu]\nMode = Server\nAddress = 0.0.0.0\nPort = {int(link[4:])}\n'
    else:
        dev, _, baud = link.partition(':')
        fcu = f'[UartEndpoint fcu]\nDevice = {dev}\nBaud = {int(baud or 921600)}\n'
    path.write_text('[General]\nTcpServerPort = 5760\nReportStats = false\n\n' + fcu +
                    '\n[UdpEndpoint mavros]\nMode = Normal\nAddress = 127.0.0.1\nPort = 14640\n'
                    '\n[UdpEndpoint gcs_dashboard]\nMode = Normal\nAddress = 127.0.0.1\nPort = 14650\n'
                    f'\n[UdpEndpoint qgc]\nMode = Normal\nAddress = {gcs_ip}\nPort = 14550\n'
                    # QGC anywhere on the network: a UDP link to <jetson>:14551 in QGC (Comm Links)
                    '\n[UdpEndpoint qgc_any]\nMode = Server\nAddress = 0.0.0.0\nPort = 14551\n')


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    run = g('run') or f'real_drone_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    run_dir = WS / 'runs' / run
    run_c = f'/workspaces/isaac_ros-dev/runs/{run}'
    (run_dir / 'ros_logs').mkdir(parents=True, exist_ok=True)
    gj = Path(g('world_json'))
    if not gj.is_file():
        raise RuntimeError(f'no geofence {gj}: set it on the dashboard (Geofence -> Save + Apply) first')
    shutil.copy(gj, run_dir / 'geofence_used.json')
    calib = g('calibration') or (str(CALIB) if CALIB.is_file() else '')
    calib_c = calib.replace(str(WS), '/workspaces/isaac_ros-dev') if calib else ''
    conf = run_dir / 'router.conf'
    router_conf(g('link'), g('gcs_ip'), conf)
    (run_dir / 'launch_args.txt').write_text(
        f'link={g("link")}\ngcs_ip={g("gcs_ip")}\nworld_json={gj}\ncalibration={calib or "(none: nominal lens)"}\n'
        f'course={g("course")}\nstarted={time.ctime()}\n')
    print(f'[real_mission] run dir: {run_dir}; link {g("link")}; Arducam calibration {calib or "NONE (nominal lens)"}')
    # the calibration / QR test page (systemd arducam-calib, at boot) holds the camera: the mission's Arducam node needs it
    subprocess.run(['systemctl', '--user', 'stop', 'arducam-calib'], check=False)

    container_up = ExecuteProcess(cmd=[str(SCRIPTS / 'container_up.sh')], output='screen')
    use_onboard = g('router') == 'auto' and subprocess.run(
        ['systemctl', '--user', 'is-active', '--quiet', 'aerothon-router']).returncode == 0
    if use_onboard:     # the always-on router (gcs/onboard) already owns the Pixhawk link
        print('[real_mission] Pixhawk through the onboard aerothon-router (its link: runs/gcs/onboard_link)')
    router = ExecuteProcess(cmd=['mavlink-routerd', '-c', str(conf)], output='screen')
    mavros = Node(package='mavros', executable='mavros_node', output='screen',
                  parameters=[{'fcu_url': 'udp://:14640@', 'tgt_system': 1, 'tgt_component': 1},
                              str(WS / 'src/vio_bringup/config/mavros_host_plugins.yaml')])
    mission = ExecuteProcess(
        cmd=['docker', 'exec', '-u', 'admin', CONTAINER, 'bash',
             '/workspaces/isaac_ros-dev/src/launch_all/scripts/container_real.sh',
             run_c, f'{run_c}/geofence_used.json', calib_c, g('course')],
        output='screen')
    saver = ExecuteProcess(cmd=[str(SCRIPTS / 'save_run.sh'), str(run_dir), 'loop'], output='log')

    def stop_all(_ctx):
        subprocess.run([str(SCRIPTS / 'save_run.sh'), str(run_dir)], timeout=60, check=False)
        subprocess.run([str(SCRIPTS / 'stop_all.sh')], timeout=90, check=False)
        subprocess.run(['systemctl', '--user', 'start', 'arducam-calib'], check=False)   # the camera page back
        print(f'[real_mission] stopped; everything is in {run_dir}')

    return [
        SetEnvironmentVariable('ROS_LOG_DIR', str(run_dir / 'ros_logs')),
        container_up,
        RegisterEventHandler(OnProcessExit(target_action=container_up, on_exit=[
            *([] if use_onboard else [router]), mavros, saver,
            TimerAction(period=float(g('container_delay')), actions=[mission]),
        ])),
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=lambda c: stop_all(c))])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('link', default_value='/dev/ttyACM0:921600',
                              description='Pixhawk: /dev/ttyACM0:921600 (USB), /dev/ttyTHS1:921600 (TELEM2 on the '
                                          'Jetson UART), udp:<port>'),
        DeclareLaunchArgument('gcs_ip', default_value=os.environ.get('GCS_IP', '10.42.0.1'),
                              description='the ground laptop, for QGroundControl on UDP 14550'),
        DeclareLaunchArgument('world_json', default_value=str(WS / 'runs/gcs/mission_geofence.json'),
                              description='the ground station geofence (GPS -> odom)'),
        DeclareLaunchArgument('calibration', default_value='',
                              description=f'Arducam calibration yaml (default {CALIB} if it exists)'),
        DeclareLaunchArgument('course', default_value='v2', description='aerothon_nav config/course_<course>.yaml'),
        DeclareLaunchArgument('run', default_value='', description='run dir name (default real_drone_<date_time>)'),
        DeclareLaunchArgument('container_delay', default_value='8.0', description='s after MAVROS starts'),
        DeclareLaunchArgument('router', default_value='auto',
                              description='auto: the onboard aerothon-router if it runs, else one started here'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(WS / 'fastdds_udp.xml')),
        OpaqueFunction(function=setup),
    ])
