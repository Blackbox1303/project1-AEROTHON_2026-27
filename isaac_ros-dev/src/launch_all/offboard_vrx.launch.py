"""OFFBOARD JETSON: the Jetson stays on the ground as the ground station - the 8080 dashboard's "Offboard VRX" mode.
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/offboard_vrx.launch.py link:=/dev/ttyUSB0:57600 segment:=qr_drop
The drone carries only the flight controller (ArduCopter 4.7.2), an analog camera and its VTX: the VRX receiver on a
Jetson USB port is the video, the SiK telemetry radio (or the bench SITL) the command link. No Arducam, no RealSense,
no ROS mission stack: the QR area only - red zones, QR codes, A* + frontier exploration, the drop or not
(gcs/vrx_explore/vrx_explore.py, the 8094 page; its decoder / VRX capture / ground projection are 8082's).
Steps: the always-on onboard router (aerothon-router) paused for the run -> mavlink-router (runs/<run>/router.conf):
the flight controller (`link`) to the 8080 dashboard (14650), the explorer (14552), QGC on the Lenovo (10.42.0.1:14550,
and any QGC on 14551) -> link:=sitl also starts ArduCopter SITL (built-in quad physics, no Gazebo) -> vrx_explore.py.
Ctrl-C (the dashboard's Stop) stops it all and resumes the onboard router.
  link      sitl (bench) | /dev/ttyUSB0:57600 (the SiK radio) | udp:<port> (a MAVLink stream sent to this port)
  segment   qr (search + read, no drop) | qr_drop (the drop on the QR matching the start QR)
  video     the VRX's USB video device ('' = pick it on the 8094 page)
  alt / drop_alt  search / drop height (course rules: 10 m / 5 m)
"""
import json
import os
import subprocess
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration

WS = Path.home() / 'workspaces/isaac_ros-dev'
WORLDS = Path.home() / 'workspaces/vio_cuda_ws/sim/worlds'
SIM = Path.home() / 'workspaces/vio_cuda_ws/sim'
AP = Path.home() / 'ardupilot'
APP = WS / 'src/gcs/vrx_explore/vrx_explore.py'

ROUTER_TAIL = """
[UdpEndpoint gcs_dashboard]
Mode = Normal
Address = 127.0.0.1
Port = 14650

[UdpEndpoint vrx_explore]
Mode = Server
Address = 127.0.0.1
Port = 14552

[UdpEndpoint lenovo_qgc]
Mode = Normal
Address = 10.42.0.1
Port = 14550

[UdpEndpoint qgc_any]
Mode = Server
Address = 0.0.0.0
Port = 14551
"""


def systemctl(*a):
    return subprocess.run(['systemctl', '--user', *a, 'aerothon-router'], capture_output=True, timeout=20).returncode


def zone_from(world_json):
    """x0 x1 y0 y1 (odom) of the delivery zone: the applied ground station geofence, else the sim world's."""
    for f in (world_json, WS / 'runs/gcs/mission_geofence.json', WORLDS / 'competition.json'):
        if f and Path(f).exists():
            z = json.load(open(f))['delivery_zone']
            return [z['x_from'], z['x_to'], z['y_from'], z['y_to']], str(f)
    raise RuntimeError('no delivery zone: apply the geofence on the dashboard')


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    link, seg = g('link'), g('segment')
    if seg not in ('qr', 'qr_drop'):
        raise RuntimeError(f"segment {seg!r}: qr | qr_drop (the offboard Jetson flies the QR area only)")
    run = g('run') or f'offboard_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    run_dir = WS / 'runs' / run
    run_dir.mkdir(parents=True, exist_ok=True)
    zone, zone_src = zone_from(g('world_json'))

    # the onboard router owns UDP 14540 and the Pixhawk's USB port: paused for the run
    paused = systemctl('is-active', '--quiet') == 0
    if paused:
        systemctl('stop')
        print('[offboard_vrx] onboard aerothon-router paused (resumed on stop)')
    if link == 'sitl':
        fcu = '[UdpEndpoint fcu]\nMode = Server\nAddress = 127.0.0.1\nPort = 14540\n'
    elif link.startswith('udp:'):
        fcu = f'[UdpEndpoint fcu]\nMode = Server\nAddress = 0.0.0.0\nPort = {int(link[4:])}\n'
    else:
        dev, _, baud = link.partition(':')
        if not Path(dev).exists():
            raise RuntimeError(f'telemetry radio {dev} not found (the SiK radio: /dev/ttyUSB0)')
        fcu = f'[UartEndpoint fcu]\nDevice = {dev}\nBaud = {int(baud or 57600)}\n'
    conf = run_dir / 'router.conf'
    conf.write_text('# written by offboard_vrx.launch.py\n[General]\nTcpServerPort = 0\nReportStats = false\n\n'
                     + fcu + ROUTER_TAIL)
    (run_dir / 'launch_args.txt').write_text(
        f'link={link}\nsegment={seg}\nvideo={g("video")}\nalt={g("alt")}\ndrop_alt={g("drop_alt")}\n'
        f'zone={zone} ({zone_src})\nstarted={time.ctime()}\n')
    print(f'[offboard_vrx] run dir {run_dir}; zone {zone} from {zone_src}')

    acts = [ExecuteProcess(cmd=['mavlink-routerd', '-c', str(conf)], output='log', name='router')]
    if link == 'sitl':
        sitl_dir = run_dir / 'sitl'
        sitl_dir.mkdir(exist_ok=True)
        # built-in quad physics (no Gazebo, the analog camera has no sim): telemetry + flight only
        acts.append(ExecuteProcess(
            cmd=[str(AP / 'build/sitl/bin/arducopter'), '--model', 'quad', '--speedup', '1', '--wipe', '-I0',
                 '--home', '47.397971,8.546164,488,90',
                 '--defaults', f'{AP}/Tools/autotest/default_params/copter.parm,{SIM}/ardupilot_sim_params.parm',
                 '--serial0', 'udpclient:127.0.0.1:14540'],
            cwd=str(sitl_dir), output='log', name='arducopter_sitl', sigterm_timeout='3', sigkill_timeout='3'))
    app = ['python3', str(APP), '--link', 'udpout:127.0.0.1:14552', '--alt', g('alt'), '--drop-alt', g('drop_alt'),
           '--zone', *[str(v) for v in zone], '--port', g('port'), '--qgc', '127.0.0.1:9']   # QGC: via the router
    if seg == 'qr_drop':
        app.append('--drop')
    if g('video'):
        app += ['--video', g('video')]
    if g('start_qr'):
        app += ['--start-qr', g('start_qr')]
    acts.append(ExecuteProcess(cmd=app, cwd=str(WS), output='screen', name='vrx_explore'))

    def stop(_ctx):
        if paused:
            systemctl('start')
        print(f'[offboard_vrx] stopped; run dir {run_dir}')
    acts.append(RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=stop)])))
    return acts


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('link', default_value='sitl', description='sitl | /dev/ttyUSB0:57600 | udp:<port>'),
        DeclareLaunchArgument('segment', default_value='qr', description='qr | qr_drop'),
        DeclareLaunchArgument('video', default_value='', description="VRX USB video device ('' = pick on the page)"),
        DeclareLaunchArgument('alt', default_value='10.0', description='search height, m'),
        DeclareLaunchArgument('drop_alt', default_value='5.0', description='drop height, m'),
        DeclareLaunchArgument('start_qr', default_value='', description='target payload (else read at the pad)'),
        DeclareLaunchArgument('world_json', default_value='', description='geofence json (delivery_zone)'),
        DeclareLaunchArgument('port', default_value='8094', description='the explorer page'),
        DeclareLaunchArgument('run', default_value=''),
        OpaqueFunction(function=setup),
    ])
