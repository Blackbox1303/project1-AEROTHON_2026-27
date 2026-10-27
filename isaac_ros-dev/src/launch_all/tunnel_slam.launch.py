"""Launch the REAL drone's TUNNEL + SLAM stack (Intel RealSense D435/D435i) and its dedicated web dashboard (:8093)
directly on the Jetson host:
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_slam.launch.py
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_slam.launch.py link:=none rtabmap:=true rviz:=true
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_slam.launch.py link:=/dev/ttyTHS1:921600
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_slam.launch.py link:=/dev/ttyACM0:921600
Walk the drone by hand, cuVSLAM -> PX4 position estimate on the dashboard, RViz + RTAB-Map:
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_slam.launch.py link:=/dev/ttyTHS1:921600 rviz:=true rtabmap:=true
vio:=cuvslam (default): the D435 runs in the Isaac container with cuVSLAM, as vio_bringup/cuvslam.launch.py
  (IR pair, emitter off, + depth for the obstacle scan); with a Pixhawk link vio_to_px4 sends VIO to EKF2.
  RTAB-Map maps left IR + depth on cuVSLAM's odometry. odom_source:=vio (default here) keeps the map on VIO
  and the dashboard shows PX4's estimate next to it.
vio:=rgbd: the older host-only RealSense (color + depth, emitter on), RTAB-Map rgbd_odometry.
Dashboard link:
    http://10.42.0.247:8093/   (from the Lenovo laptop browser)
    http://127.0.0.1:8093/     (on the Jetson localhost)
Does NOT auto-launch on boot. Ctrl-C (or Stop in the dashboard) stops everything cleanly.
"""
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, ExecuteProcess, OpaqueFunction, RegisterEventHandler,
                            SetEnvironmentVariable)
from launch.event_handlers import OnShutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ.pop('QT_QPA_FONTDIR', None)

# Started in the background from a non-interactive shell (ssh '... ros2 launch ... &'), the launch inherits SIGINT and
# SIGQUIT ignored, and the kernel then drops them before ros2 launch sees them (it reacts through signal.set_wakeup_fd,
# which only fires for signals with a Python handler): Ctrl-C and the dashboard's Stop did nothing, and the
# name-based cleanup left a half-stopped stack (2026-10-27). A no-op Python handler lets them through again.
for _sig in (signal.SIGINT, signal.SIGQUIT):
    if signal.getsignal(_sig) == signal.SIG_IGN:
        signal.signal(_sig, lambda *_: None)

HERE = Path(__file__).resolve().parent
SCRIPTS = HERE / 'scripts'
WS = Path.home() / 'workspaces/isaac_ros-dev'
NAV_SRC = WS / 'src/aerothon_nav'
VIO_SRC = WS / 'src/vio_bringup'
CONTAINER = 'isaac_ros_dev-aarch64-pyzbar-container'
RVIZ_PATTERN = r'^/opt/ros/humble/lib/rviz2/rviz2 '
RTAB_PATTERN = r'^/usr/bin/python3 .*tunnel_viz\.launch\.py|^/opt/ros/humble/lib/rtabmap_(slam|viz)/'
VIZ_PATTERN = f'{RVIZ_PATTERN}|{RTAB_PATTERN}'

sys.path.insert(0, str(VIO_SRC))
from vio_bringup.rig import imu_to_camera_tf_node, load_rig  # noqa: E402


def router_conf(link, gcs_ip, path):
    if link.startswith('udp:'):
        fcu = f'[UdpEndpoint fcu]\nMode = Server\nAddress = 0.0.0.0\nPort = {int(link[4:])}\n'
    else:
        dev, _, baud = link.partition(':')
        fcu = f'[UartEndpoint fcu]\nDevice = {dev}\nBaud = {int(baud or 921600)}\n'
    path.write_text('[General]\nTcpServerPort = 5760\nReportStats = false\n\n' + fcu +
                    '\n[UdpEndpoint mavros]\nMode = Normal\nAddress = 127.0.0.1\nPort = 14640\n'
                    '\n[UdpEndpoint gcs_dashboard]\nMode = Normal\nAddress = 127.0.0.1\nPort = 14650\n'
                    f'\n[UdpEndpoint qgc]\nMode = Normal\nAddress = {gcs_ip}\nPort = 14550\n'
                    '\n[UdpEndpoint qgc_any]\nMode = Server\nAddress = 0.0.0.0\nPort = 14551\n')


def process_tree(roots):
    """roots plus all their descendants (by parent PID)."""
    kids = {}
    for line in subprocess.run(['ps', '-eo', 'pid=,ppid='], capture_output=True, text=True).stdout.splitlines():
        pid, ppid = (int(x) for x in line.split())
        kids.setdefault(ppid, []).append(pid)
    tree, todo = set(), list(roots)
    while todo:
        p = todo.pop()
        if p not in tree:
            tree.add(p)
            todo.extend(kids.get(p, []))
    return tree


def stop_previous_launch():
    """Stop any other tunnel_slam launch, and everything it started, before this one starts anything. Its shutdown
    handler pkills by name (slam_toolbox, cuvslam, rtabmap, ...), so a relaunch while the old one was still
    stopping had its own fresh nodes killed (2026-10-27, 00:40); and its dashboard (+ the RViz / RTAB-Map that
    dashboard opened) could outlive it, so the new launch kept serving the old dashboard code (02:36)."""
    me = os.getpid()
    pids = [int(p) for p in subprocess.run(
        ['pgrep', '-f', r'^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch .*tunnel_slam\.launch\.py'],
        capture_output=True, text=True).stdout.split() if int(p) != me]
    # orphans of an earlier launch: the dashboard and what it started (pgrep: not this launch's own processes)
    orphans = [int(p) for p in subprocess.run(['pgrep', '-f', r'^python3 .*tunnel_gcs\.py'],
                                              capture_output=True, text=True).stdout.split()]
    # RViz and RTAB-Map close only when you close them (or with their dashboard buttons), never on a relaunch
    viz = [int(p) for p in subprocess.run(['pgrep', '-f', VIZ_PATTERN], capture_output=True, text=True).stdout.split()]
    ancestors, p = set(), me                         # e.g. the dashboard whose Launch button started this launch
    while p > 1:
        ancestors.add(p)
        try:
            p = int(Path(f'/proc/{p}/stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    tree = process_tree([x for x in pids + orphans if x not in ancestors]) - process_tree([me]) - ancestors - \
        process_tree(viz)
    if not tree:
        return
    print(f'[tunnel_slam] stopping the previous tunnel_slam launch {pids} and {len(tree) - len(pids)} of its '
          'processes first', flush=True)
    for sig, wait in ((signal.SIGINT, 12.0), (signal.SIGTERM, 6.0), (signal.SIGKILL, 2.0)):
        alive = [p for p in tree if Path(f'/proc/{p}').exists()]
        if not alive:
            return
        # the launches first (their own shutdown stops their nodes cleanly), then whatever is left
        for p in (pids if sig == signal.SIGINT else alive):
            try:
                os.kill(p, sig)
            except ProcessLookupError:
                pass
        end = time.monotonic() + wait
        while time.monotonic() < end and any(Path(f'/proc/{p}').exists() for p in pids):
            time.sleep(0.3)
        if sig == signal.SIGINT:                     # the launch is down: give its leftovers a Ctrl-C too
            for p in tree:
                try:
                    os.kill(p, signal.SIGINT)
                except ProcessLookupError:
                    pass
            end = time.monotonic() + 5.0
            while time.monotonic() < end and any(Path(f'/proc/{p}').exists() for p in tree):
                time.sleep(0.3)


def reset_d435():
    """USB-reset the D435 (its /dev/bus/usb node is world-writable: no sudo). After a RealSense process has had it,
    the container's libusb backend can't power it up again ('failed to set power state') until it is reset, and
    initial_reset can't help: it needs the device opened first (2026-10-27)."""
    def dev():
        out = subprocess.run(['lsusb', '-d', '8086:0b07'], capture_output=True, text=True).stdout.split()
        return f'/dev/bus/usb/{out[1]}/{out[3].rstrip(":")}' if len(out) > 3 else None
    old = dev()
    if subprocess.run(['usbreset', '8086:0b07'], capture_output=True, check=False).returncode != 0:
        print('[tunnel_slam] warning: usbreset of the D435 (8086:0b07) failed - replug it if the camera stays down')
        return
    # it re-enumerates under a NEW device number, usually in ~1-3 s but >10 s right after a boot; right after the
    # reset lsusb still lists the old one, so wait for the number to change and udev to create its node. The
    # usb_watch thread keeps syncing afterwards, so a late re-enumeration still reaches the container.
    for _ in range(20):                      # <= 5 s; usb_watch catches a later re-enumeration
        time.sleep(0.25)
        new = dev()
        if new and new != old and Path(new).exists():
            time.sleep(0.5)
            return
    print('[tunnel_slam] D435 still re-enumerating after usbreset: usb_watch will sync it', flush=True)


def usb_nodes():
    return sorted((str(d), d.stat().st_rdev) for b in Path('/dev/bus/usb').iterdir() for d in b.iterdir())


def usb_watch():
    """While the launch runs: re-sync the container's USB nodes whenever the host's change (a D435 that
    re-enumerated late after the reset, or was replugged). One docker exec per change, nothing otherwise."""
    last = usb_nodes()
    while True:
        time.sleep(1.0)
        try:
            now = usb_nodes()
        except OSError:
            continue
        if now != last:
            time.sleep(0.5)                  # let udev finish the node
            sync_container_usb()
            print('[tunnel_slam] USB devices changed: re-synced the container /dev/bus/usb', flush=True)
            last = usb_nodes()


def sync_container_usb():
    """Mirror the host's /dev/bus/usb nodes into the container. Its /dev is a copy made when it started, so a D435
    that re-enumerated since (replug, USB reset) has no node there: librealsense then finds it by serial number but
    fails with 'failed to set power state' (2026-10-27)."""
    script = ['set -e']
    for bus in sorted(Path('/dev/bus/usb').iterdir()):
        script.append(f'mkdir -p {bus}; find {bus} -mindepth 1 -delete')
        for dev in sorted(bus.iterdir()):
            st = dev.stat()
            script.append(f'mknod {dev} c {os.major(st.st_rdev)} {os.minor(st.st_rdev)}; chmod 666 {dev}')
    r = subprocess.run(['docker', 'exec', '-u', 'root', CONTAINER, 'sh', '-c', '; '.join(script)],
                       capture_output=True, text=True, check=False)
    if r.returncode:
        print(f'[tunnel_slam] warning: could not sync USB nodes into {CONTAINER}: {r.stderr.strip()}')


def py_node(script_path, name, params=None):
    cmd = ['python3', str(script_path), '--ros-args', '-r', f'__node:={name}']
    for item in (params or []):
        if isinstance(item, (str, Path)):
            cmd.extend(['--params-file', str(item)])
        elif isinstance(item, dict):
            for k, v in item.items():
                if v is not None and v != '':
                    val_s = str(v).lower() if isinstance(v, bool) else str(v)
                    cmd.extend(['-p', f'{k}:={val_s}'])
    return ExecuteProcess(cmd=cmd, output='screen')


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    run = g('run') or f'tunnel_slam_{time.strftime("%Y-%m-%d_%H-%M-%S")}'
    run_dir = WS / 'runs' / run
    (run_dir / 'ros_logs').mkdir(parents=True, exist_ok=True)

    stop_previous_launch()
    # Stop any leftover realsense / cuVSLAM process (host or container: the container shares the host PID
    # namespace) so this run gets exclusive USB access to the D435
    subprocess.run(['pkill', '-INT', '-f', 'realsense2_camera_node|container_tunnel\\.sh|tunnel_real\\.launch\\.py|'
                    'cuvslam\\.launch\\.py|visual_slam_container'], check=False)
    time.sleep(1.0)

    gj = Path(g('world_json'))
    gj_str = ''
    if gj.is_file():
        shutil.copy(gj, run_dir / 'geofence_used.json')
        gj_str = str(run_dir / 'geofence_used.json')

    link = g('link').strip()
    port = g('port')
    course = g('course')
    scan_band = float(g('scan_band'))
    use_rviz = g('rviz').lower() in ('true', '1', 'yes')
    use_rtabmap = g('rtabmap').lower() in ('true', '1', 'yes')
    vio = g('vio').lower()
    if g('vslam').lower() in ('false', '0', 'no'):
        vio = 'rgbd'
    cuvslam = vio == 'cuvslam'
    nav = g('nav').lower() in ('true', '1', 'yes')
    # colour stream for RTAB-Map (cuVSLAM mode): default on with rtabmap:=true
    color = g('color').lower() in ('true', '1', 'yes') if g('color') != 'auto' else use_rtabmap

    (run_dir / 'launch_args.txt').write_text(
        f'link={link}\nport={port}\ncourse={course}\nscan_band={scan_band}\n'
        f'rviz={use_rviz}\nrtabmap={use_rtabmap}\nvio={vio}\nodom_source={g("odom_source")}\n'
        f'px4={g("px4")}\nstarted={time.ctime()}\n')

    print(f'[tunnel_slam] run dir: {run_dir}')
    print(f'[tunnel_slam] Tunnel + SLAM Dashboard -> http://10.42.0.247:{port}/  (local: http://127.0.0.1:{port}/)')

    rig = load_rig(str(VIO_SRC / 'config/rig.yaml'))
    course_file = NAV_SRC / f'config/course_{course}.yaml'

    actions = []

    # Optional Pixhawk MAVROS link
    use_mavros = link.lower() not in ('', 'none', 'off', 'false')
    use_onboard = subprocess.run(
        ['systemctl', '--user', 'is-active', '--quiet', 'aerothon-router']).returncode == 0
    dev_path = link.partition(':')[0]
    if use_mavros and not use_onboard and not link.startswith('udp:') and not Path(dev_path).exists():
        print(f'[tunnel_slam] note: {dev_path} not plugged in - running RealSense + SLAM without MAVROS (bench mode)')
        use_mavros = False

    if use_mavros:
        if not use_onboard:
            conf = run_dir / 'router.conf'
            router_conf(link, g('gcs_ip'), conf)
            actions.append(ExecuteProcess(cmd=['mavlink-routerd', '-c', str(conf)], output='screen'))
        mavros_running = subprocess.run(['pgrep', '-f', 'mavros_node'], capture_output=True).returncode == 0
        if not mavros_running:
            actions.append(Node(
                package='mavros', executable='mavros_node', output='screen',
                parameters=[{'fcu_url': 'udp://:14640@', 'tgt_system': 1, 'tgt_component': 1},
                            str(VIO_SRC / 'config/mavros_host_plugins.yaml')]))

    # Start the Tunnel + SLAM web dashboard on :8093 if not already running
    # Detached like RViz / RTAB-Map: as a child of this launch the dashboard died with the stack, so its own Stop
    # button killed the page (no camera feed, no Launch button after)
    dash_running = subprocess.run(['pgrep', '-f', r'^python3 .*tunnel_gcs\.py'], capture_output=True).returncode == 0
    if not dash_running and g('dashboard').lower() in ('true', '1', 'yes'):
        subprocess.Popen(['python3', str(WS / 'src/gcs/tunnel_gcs.py'), '--port', port],
                         stdout=open(WS / f'runs/gcs/tunnel_gcs_{port}.log', 'a'), stderr=subprocess.STDOUT,
                         env=dict(os.environ, FASTRTPS_DEFAULT_PROFILES_FILE=str(WS / 'fastdds_udp.xml')),
                         start_new_session=True)

    # Static transforms: base_link -> imu_link (-> camera_link: published by cuvslam.launch.py in cuVSLAM mode)
    actions.append(Node(package='tf2_ros', executable='static_transform_publisher', name='base_to_imu_tf',
                        arguments=['--frame-id', 'base_link', '--child-frame-id', 'imu_link']))
    px4 = g('px4').lower()
    send_px4 = use_mavros if px4 == 'auto' else px4 in ('true', '1', 'yes')
    if cuvslam:
        # The command that works by hand (docker exec ... cuvslam.launch.py imu:=false mavros:=false camera_ns:=/),
        # plus depth for /vio/obstacle_scan, no cuVSLAM TF (slam_toolbox + odom_tf own map -> odom -> base_link)
        # and vio_to_px4 when a Pixhawk is linked. No -t: the launch has no TTY; stop_all signals the container
        # processes directly (shared PID namespace).
        reset_d435()
        sync_container_usb()
        threading.Thread(target=usb_watch, daemon=True).start()
        subprocess.run(['xhost', '+local:'], env=dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':0')),
                       capture_output=True, check=False)
        inner = ('source /opt/ros/humble/setup.bash && source /workspaces/isaac_ros-dev/install/setup.bash && '
                 'exec ros2 launch vio_bringup cuvslam.launch.py imu:=false mavros:=false camera_ns:=/ '
                 f'autopilot:={g("autopilot")} depth:=true scan:=false initial_reset:=false publish_tf:=false base_tf:=false px4:={"true" if send_px4 else "false"} '
                 f'rviz:={g("vio_rviz")} profile:={g("profile")} monitor:={g("monitor")} '
                 f'color:={"true" if color else "false"}')
        actions.append(ExecuteProcess(
            cmd=['docker', 'exec', '-u', 'admin', '-e', f'DISPLAY={os.environ.get("DISPLAY", ":0")}',
                 '-e', 'FASTRTPS_DEFAULT_PROFILES_FILE=/workspaces/isaac_ros-dev/fastdds_udp.xml',
                 '-e', 'ROS_LOG_DIR=/workspaces/isaac_ros-dev/runs/' + run + '/ros_logs',
                 CONTAINER, 'bash', '-c', inner],
            output='screen'))
    else:
        actions.append(imu_to_camera_tf_node(rig))
        # Intel RealSense D435/D435i running directly on the Jetson host (initial_reset False so USB stays open)
        actions.append(Node(
            name='camera', namespace='', package='realsense2_camera', executable='realsense2_camera_node',
            output='screen',
            parameters=[{
                'initial_reset': False,
                'enable_depth': True,
                'enable_color': True,
                'enable_infra1': False,
                'enable_infra2': False,
                'enable_gyro': False,
                'enable_accel': False,
                'depth_module.emitter_enabled': 1,
                'depth_module.profile': '640x360x30',
                'depth_module.depth_profile': '640x360x30',
                'rgb_camera.profile': '640x360x30',
                'rgb_camera.color_profile': '640x360x30',
                'align_depth.enable': True,
                'pointcloud.enable': False,
                'publish_tf': True,
                'depth_module.global_time_enabled': True,
            }]))
    actions.extend([
        # Levelled 2D LaserScan (/vio/obstacle_scan) + 3D PointCloud2 (/vio/depth_points)
        # mapping / walk tests (nav:=false) don't need the planner's 20 Hz: at stride 2 / 30 Hz / 57k-point clouds
        # this node alone took ~75 % of a core and starved the rest of the stack (2026-10-27)
        py_node(VIO_SRC / 'vio_bringup/obstacle_scan.py', 'obstacle_scan',
                [{'use_sim_time': False, 'band': scan_band, 'stride': 2 if nav else 4,
                  'max_rate_hz': 20.0 if nav else 10.0, 'cloud_stride': 4, 'cloud_rate_hz': 5.0}]),
        # odom -> base_link from PX4, RTAB-Map rgbd_odometry (/rtabmap/odom), or 20 Hz bench fallback
        py_node(NAV_SRC / 'aerothon_nav/odom_tf.py', 'odom_tf',
                [{'use_sim_time': False, 'restamp': True, 'fallback_rate': 20.0,
                  'source': g('odom_source') if g('odom_source') != 'default' else ('vio' if cuvslam else 'auto')}]),
        # 2D SLAM (slam_toolbox) on /vio/obstacle_scan -> /map + map -> odom TF
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox', output='screen',
             parameters=[
                 str(NAV_SRC / 'config/slam_toolbox_2d.yaml'),
                 {'use_sim_time': False, 'map_update_interval': 1.0,
                  'minimum_travel_distance': 0.03, 'minimum_travel_heading': 0.03},
             ]),
    ])
    if nav:
        actions.extend([
            # Tunnel navigation + 3D altitude/window planner + banner entrance alignment
            py_node(NAV_SRC / 'aerothon_nav/offboard_bridge.py', 'offboard_bridge',
                    [NAV_SRC / 'config/offboard_bridge.yaml', {'geofence_json': gj_str}]),
            py_node(NAV_SRC / 'aerothon_nav/corridor_nav.py', 'corridor_nav',
                    [course_file, {'use_sim_time': False}]),
            py_node(NAV_SRC / 'aerothon_nav/banner_detector.py', 'banner_detector',
                    [NAV_SRC / 'config/banner.yaml', {'use_sim_time': False}]),
            py_node(NAV_SRC / 'aerothon_nav/entry_align.py', 'entry_align',
                    [course_file, {'use_sim_time': False}]),
            py_node(NAV_SRC / 'aerothon_nav/altitude_planner.py', 'altitude_planner',
                    [{'use_sim_time': False}]),
        ])

    # RViz and RTAB-Map run detached (own session, own log): they stay open across Stop / relaunch and close only
    # when you close them or use their dashboard buttons - same commands as the dashboard's buttons
    env = dict(os.environ, DISPLAY=os.environ.get('DISPLAY', ':0'), FASTRTPS_DEFAULT_PROFILES_FILE=str(WS / 'fastdds_udp.xml'))
    if use_rviz and subprocess.run(['pgrep', '-f', RVIZ_PATTERN], capture_output=True).returncode:
        subprocess.Popen(['/opt/ros/humble/lib/rviz2/rviz2', '-d', str(VIO_SRC / 'rviz/tunnel_slam.rviz'),
                          '--ros-args', '-r', '__node:=rviz2_tunnel'],
                         stdout=open(WS / 'runs/gcs/rviz_tunnel.log', 'w'), stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)
    if use_rtabmap and subprocess.run(['pgrep', '-f', RTAB_PATTERN], capture_output=True).returncode:
        subprocess.Popen(['ros2', 'launch', str(HERE / 'tunnel_viz.launch.py'), 'rviz:=false', 'rtabmap:=true',
                          f'rtabmap_viz:={g("rtabmap_viz")}', f'source:={"infra" if cuvslam and not color else "color"}',
                          f'rgbd_odom:={"false" if cuvslam else "true"}'],
                         stdout=open(WS / 'runs/gcs/rtabmap_tunnel.log', 'w'), stderr=subprocess.STDOUT, env=env,
                         start_new_session=True)

    def stop_all(_ctx):
        subprocess.run([
            'pkill', '-INT', '-f',
            'realsense2_camera_node|obstacle_scan\\.py|odom_tf\\.py|slam_toolbox|offboard_bridge\\.py|'
            'corridor_nav\\.py|banner_detector\\.py|entry_align\\.py|altitude_planner\\.py|'
            'cuvslam\\.launch\\.py|visual_slam_container|vio_to_px4|image_monitor'
        ], check=False)
        print(f'[tunnel_slam] stopped; logs in {run_dir}')

    return [
        SetEnvironmentVariable('ROS_LOG_DIR', str(run_dir / 'ros_logs')),
        SetEnvironmentVariable('PYTHONPATH', f'{VIO_SRC}:{NAV_SRC}:' + os.environ.get('PYTHONPATH', '')),
        SetEnvironmentVariable('DISPLAY', os.environ.get('DISPLAY', ':0')),
        *actions,
        RegisterEventHandler(OnShutdown(on_shutdown=[OpaqueFunction(function=lambda c: stop_all(c))])),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('link', default_value='none',
                              description='Pixhawk: /dev/ttyACM0:921600, /dev/ttyTHS1:921600, or none (camera+SLAM)'),
        DeclareLaunchArgument('port', default_value='8093', description='Tunnel + SLAM dashboard HTTP port'),
        DeclareLaunchArgument('dashboard', default_value='true', description='start the web dashboard on :port'),
        DeclareLaunchArgument('vio', default_value='cuvslam',
                              description='cuvslam: D435 + cuVSLAM in the Isaac container; rgbd: host D435 color+depth'),
        DeclareLaunchArgument('vslam', default_value='true', description='old name: false = vio:=rgbd'),
        DeclareLaunchArgument('autopilot', default_value='ardupilot',
                              description='flight controller firmware for the VIO feed: ardupilot (ArduCopter 4.7) | px4'),
        DeclareLaunchArgument('px4', default_value='auto',
                              description='send cuVSLAM to PX4 EKF2 (vio_to_px4): auto = when a Pixhawk link is up'),
        DeclareLaunchArgument('odom_source', default_value='default',
                              description='odom_tf source: vio | px4 | auto (default: vio with cuVSLAM, else auto)'),
        DeclareLaunchArgument('vio_rviz', default_value='false', description="also open cuVSLAM's own RViz"),
        # 30 Hz: at 60 Hz the whole stack (+ RViz + RTAB-Map GUI) loaded the Jetson to ~25 and starved the nodes
        DeclareLaunchArgument('profile', default_value='640x360x30', description='D435 IR/depth profile (cuVSLAM)'),
        DeclareLaunchArgument('monitor', default_value='false', description='cuVSLAM image_monitor (IR rates)'),
        DeclareLaunchArgument('color', default_value='auto',
                              description='cuVSLAM mode: also stream RGB + aligned depth (coloured RTAB-Map); auto = rtabmap'),
        DeclareLaunchArgument('rtabmap_viz', default_value='true', description="RTAB-Map's own window (~400 MB)"),
        DeclareLaunchArgument('nav', default_value='true',
                              description='false: no corridor/entry/altitude/banner/offboard nodes (mapping, walk tests)'),
        DeclareLaunchArgument('rviz', default_value='false', description='true: open RViz2 on the Jetson (:0)'),
        DeclareLaunchArgument('rtabmap', default_value='false',
                              description='true: open RTAB-Map 3D SLAM + GUI on the Jetson (:0)'),
        DeclareLaunchArgument('course', default_value='v2', description='aerothon_nav config/course_<course>.yaml'),
        DeclareLaunchArgument('scan_band', default_value='0.20', description='m height band for 2D obstacle scan'),
        DeclareLaunchArgument('gcs_ip', default_value=os.environ.get('GCS_IP', '10.42.0.1'),
                              description='ground laptop IP for QGroundControl UDP 14550'),
        DeclareLaunchArgument('world_json', default_value=str(WS / 'runs/gcs/mission_geofence.json'),
                              description='optional geofence json'),
        DeclareLaunchArgument('run', default_value='', description='run dir name (default tunnel_slam_<date_time>)'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(WS / 'fastdds_udp.xml')),
        OpaqueFunction(function=setup),
    ])
