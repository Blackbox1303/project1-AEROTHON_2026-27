"""The whole sim mission on the Jetson in one command - container, Gazebo + autopilot SITL + bridge + MAVROS, the mission stack,
loggers, /mission/start - with everything saved under runs/<run> (on disk: survives reboots, unlike /tmp):
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/sim_mission.launch.py                  # world:=competition
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/sim_mission.launch.py world:=comp_far1 # far target
    ... fcu:=px4      # the old PX4 SITL instead of ArduCopter (default fcu:=ardupilot, as on the real drone, 2026-10-27)
Watch it from the laptop: ~/host_gui.sh there (gz_ip:=10.42.0.247, the default; 127.0.0.1 = Jetson only, no GUI).
Ctrl-C stops everything, in the container too (scripts/stop_all.sh).

Steps: scripts/container_up.sh (the dev container, detached, if not running; its entrypoint hook
install_ignition_deps.sh installs MAVROS into every new container) -> host_all.launch.py (Gazebo headless, PX4 SITL,
ros_gz bridge, MAVROS on one GZ_IP / GZ_PARTITION) -> after container_delay s scripts/container_mission.sh in the
container (container_all + loggers; /mission/start once the delivery Nav2 is up, autostart:=1) + scripts/save_run.sh
loop (PX4 flight log, Gazebo log, the mission's ~/.ros files every 60 s).
runs/<run>/: container.log, mission_log.txt (/mission/log), altitude_state.txt, mission_start.txt, <px4>.ulg,
gz_server.log, aerothon_* (geofence + snapshot, start QR, delivery memory), ros_logs/ (host nodes: MAVROS, bridge,
this launch), <world>.json, launch_args.txt.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, IncludeLaunchDescription, OpaqueFunction,
                            RegisterEventHandler, SetEnvironmentVariable, TimerAction)
from launch.event_handlers import OnProcessExit, OnShutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / 'scripts'
WS = Path.home() / 'workspaces/isaac_ros-dev'
WORLDS = Path.home() / 'workspaces/vio_cuda_ws/sim/worlds'
CONTAINER = 'isaac_ros_dev-aarch64-pyzbar-container'


def onboard_router_active():
    return subprocess.run(['systemctl', '--user', 'is-active', '--quiet', 'aerothon-router']).returncode == 0


def onboard_router(action):
    subprocess.run(['systemctl', '--user', action, 'aerothon-router'], timeout=20, check=False)


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    world = g('world')
    run = g('run') or f'{world}_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    run_dir = WS / 'runs' / run
    run_c = f'/workspaces/isaac_ros-dev/runs/{run}'               # the same dir inside the container
    (run_dir / 'ros_logs').mkdir(parents=True, exist_ok=True)
    # the geofence for the nodes: the ground station's (world_json:=, GPS -> odom) or the sim world's ground truth
    # router auto: a router of the sim's own (gcs/router/sim.conf, same sinks + mav_agent). The always-on onboard
    # router (gcs/onboard, systemd aerothon-router) also owns the real Pixhawk's serial link: with a Pixhawk plugged in,
    # SITL and the Pixhawk were two system-1 autopilots on one link (MAVROS params 1031 / 1377 mixed, 2026-10-27).
    # It is paused for the run (a second router beside it would split SITL's 14540) and resumed on stop.
    router = g('router')
    paused_router = False
    if router == 'auto':
        router = '1'
        if onboard_router_active():
            onboard_router('stop')
            paused_router = True
            print('[sim_mission] onboard aerothon-router paused for the sim (resumed on stop)')
    gj = Path(g('world_json')) if g('world_json') else WORLDS / f'{world}.json'
    if not gj.exists():
        raise RuntimeError(f'no geofence json {gj}')
    shutil.copy(gj, run_dir / 'geofence_used.json')
    gj_c = f'{run_c}/geofence_used.json'                       # the copy, as the container sees it
    if not (WORLDS / f'{world}.sdf').exists():
        raise RuntimeError(f'no world {WORLDS / world}.sdf')
    (run_dir / 'launch_args.txt').write_text(
        f'world={world}\nfcu={g("fcu")}\ngz_ip={g("gz_ip")}\nautostart={g("autostart")}\nworld_json={gj}\nstarted={time.ctime()}\n')
    print(f'[sim_mission] run dir: {run_dir}')

    container_up = ExecuteProcess(cmd=[str(SCRIPTS / 'container_up.sh')], output='screen')
    host = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(str(HERE / 'host_all.launch.py')),
        launch_arguments={'world': str(WORLDS / f'{world}.sdf'), 'gz_ip': g('gz_ip'),
                          'gz_partition': g('gz_partition'), 'headless': '1', 'router': router,
                          'fcu': g('fcu')}.items())
    mission = ExecuteProcess(
        cmd=['docker', 'exec', '-u', 'admin', CONTAINER, 'bash', '/workspaces/isaac_ros-dev/src/launch_all/scripts/'
             'container_mission.sh', run_c, world, g('autostart'), gj_c, g('fcu')],
        output='screen')
    saver = ExecuteProcess(cmd=[str(SCRIPTS / 'save_run.sh'), str(run_dir), 'loop'], output='log')

    def stop_all(_event, _ctx):
        # docker exec doesn't stop what it started in the container when killed: stop it all explicitly, and wait
        subprocess.run([str(SCRIPTS / 'save_run.sh'), str(run_dir)], timeout=60, check=False)
        subprocess.run([str(SCRIPTS / 'stop_all.sh')], timeout=90, check=False)
        if paused_router:
            onboard_router('start')
        print(f'[sim_mission] stopped; everything is in {run_dir}')

    return [
        SetEnvironmentVariable('ROS_LOG_DIR', str(run_dir / 'ros_logs')),
        container_up,
        RegisterEventHandler(OnProcessExit(target_action=container_up, on_exit=[
            host, saver,
            # Gazebo + PX4 up and the bridge running (host_all starts it after 15 s) before the container stack
            TimerAction(period=float(g('container_delay')), actions=[mission]),
        ])),
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=lambda c: stop_all(None, c))])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world', default_value='competition',
                              description='vio_cuda_ws/sim/worlds/<world>.sdf + .json (competition | comp_far1 | ...)'),
        DeclareLaunchArgument('fcu', default_value='ardupilot',
                              description='ardupilot (ArduCopter 4.7.2 SITL, as on the real drone) | px4 (PX4 1.16 SITL)'),
        DeclareLaunchArgument('run', default_value='', description='run dir name (default <world>_<date_time>)'),
        DeclareLaunchArgument('gz_ip', default_value=os.environ.get('GZ_IP', '10.42.0.247'),
                              description="Jetson LAN IP: the laptop's host_gui.sh can watch; 127.0.0.1: Jetson only"),
        DeclareLaunchArgument('gz_partition', default_value=os.environ.get('GZ_PARTITION', 'vio')),
        DeclareLaunchArgument('autostart', default_value='1', description='1: /mission/start when the stack is up'),
        DeclareLaunchArgument('container_delay', default_value='35.0', description='s after the host sim starts'),
        DeclareLaunchArgument('world_json', default_value='',
                              description="geofence json for the nodes (the ground station's); '' = the sim world's"),
        DeclareLaunchArgument('router', default_value='auto',
                              description='auto (= 1, the onboard aerothon-router paused for the run) | 1: start '
                                          'mavlink-router here | 2: the onboard aerothon-router (no Pixhawk plugged in!)'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(WS / 'fastdds_udp.xml')),
        OpaqueFunction(function=setup),
    ])
