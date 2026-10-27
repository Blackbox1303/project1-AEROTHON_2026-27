"""HOST half of the sim mission (the container half is aerothon_nav container_all.launch.py):
Gazebo + the autopilot SITL (fcu:=ardupilot - ArduCopter 4.7.2, run_competition_ap.sh, the default since 2026-10-27, as on
the real drone; fcu:=px4 - PX4 1.16, run_competition.sh), the Gazebo -> ROS bridge (D435, Arducam, nadir_hd, /clock) and MAVROS.
Gazebo GUI on another machine (vio_cuda_ws/sim/host_gui.sh on the laptop): the Jetson's Gazebo, PX4 and this bridge
must share one gz-transport address and partition -
    ros2 launch host_all.launch.py gz_ip:=10.42.0.247 gz_partition:=vio      (headless:=1 is the default)
A bridge left on 127.0.0.1 / the default partition sees no world: no /clock, no cameras in the container.
The bridge must be the Harmonic build (~/ros_gz_ws): /opt/ros/humble's ros_gz_bridge links Fortress'
ignition-transport11 and sees nothing from Gazebo Harmonic. Without the bridge there is no /clock and no camera
in the container: every sim-time node stalls, qr_reader gets no frames and entry_align finds no walls (2026-10-20).
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, SetEnvironmentVariable, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

WS = os.path.expanduser('~/workspaces/isaac_ros-dev')
GZ_BRIDGE_WS = os.path.expanduser('~/ros_gz_ws/install/setup.bash')
BRIDGE_CONFIG = os.path.join(WS, 'src/vio_bringup/config/sim_bridge.yaml')


def generate_launch_description():
    # Set FastDDS profile
    fastdds_profile = os.path.join(WS, 'fastdds_udp.xml')
    set_fastdds = SetEnvironmentVariable(name='FASTRTPS_DEFAULT_PROFILES_FILE', value=fastdds_profile)

    # Launch the autopilot SITL via its script (runs Gazebo). Both send MAVLink to UDP 14540
    gz_env = {'GZ_IP': LaunchConfiguration('gz_ip'), 'GZ_PARTITION': LaunchConfiguration('gz_partition')}
    run_px4 = ExecuteProcess(
        cmd=[
            PythonExpression(["'/home/szyfrowac/workspaces/vio_cuda_ws/sim/' + ('run_competition.sh' if '",
                              LaunchConfiguration('fcu'), "' == 'px4' else 'run_competition_ap.sh')"]),
            LaunchConfiguration('world')
        ],
        output='screen',
        shell=True,
        # HEADLESS: no local `gz sim -g` (0.9 GB on the Jetson); view it from the laptop instead (host_gui.sh)
        additional_env={**gz_env, 'HEADLESS': LaunchConfiguration('headless')},
    )

    # Gazebo -> ROS bridge (= vio_bringup sim_bridge.launch.py, which isn't installed on the host). Started once
    # Gazebo is up; on the same GZ_IP / GZ_PARTITION as Gazebo and PX4 (run_competition.sh gets them via gz_env)
    gz_bridge = TimerAction(period=15.0, actions=[ExecuteProcess(
        cmd=['bash', '-c',
             f'source /opt/ros/humble/setup.bash && source {GZ_BRIDGE_WS} && '
             f'exec ros2 run ros_gz_bridge parameter_bridge --ros-args -r __node:=sim_bridge '
             f'-p config_file:={BRIDGE_CONFIG} -p use_sim_time:=true'],
        additional_env=gz_env,
        output='screen')])

    # MAVROS node on the host (communicates with PX4 over UDP); the container's mavros_node dies (exit 127), so
    # container_all runs sim_view with mavros:=false. /mavros/imu/data is NOT remapped: obstacle_scan,
    # imu_tf_broadcaster and imu_rate_setter use it under that name
    mavros_node = Node(
        package='mavros',
        executable='mavros_node',
        output='screen',
        parameters=[
            # router:=1 - behind mavlink-router (src/gcs/router/sim.conf): it owns PX4's 14540 and forwards to 14640.
            # router:=2 - behind the always-on onboard router (systemd aerothon-router, gcs/onboard): same port, no
            #             router started here.  router:=0 - straight to PX4 SITL, as before
            {'fcu_url': PythonExpression(["'udp://:14640@' if '", LaunchConfiguration('router'),
                                          "' in ('1', '2') else 'udp://:14540@127.0.0.1:14557'"])},
            {'tgt_system': 1},
            {'tgt_component': 1},
            # a params file, not a `pluginlists_yaml` parameter (mavros_node ignores that: all ~60 plugins loaded)
            os.path.join(WS, 'src/vio_bringup/config/mavros_host_plugins.yaml'),
            # heartbeat timeout for the lock-stepped SITL (its heartbeats slow down with the real-time factor)
            os.path.join(WS, 'src/vio_bringup/config/mavros_sim_sys.yaml'),
        ],
        # ensure simulation time is used
        arguments=['--ros-args', '--param', 'use_sim_time:=true']
    )

    # mavlink-router: PX4's MAVLink split to MAVROS, the ground station dashboard and QGC on the Lenovo
    router = ExecuteProcess(
        cmd=['mavlink-routerd', '-c', LaunchConfiguration('router_conf')], output='screen',
        condition=IfCondition(PythonExpression(["'", LaunchConfiguration('router'), "' == '1'"])))

    return LaunchDescription([
        DeclareLaunchArgument('fcu', default_value='ardupilot', description='ardupilot | px4: the SITL autopilot'),
        DeclareLaunchArgument('router', default_value='1',
                              description='1: PX4 MAVLink through mavlink-router (started here), 2: through the '
                                          'always-on onboard router (aerothon-router), 0: MAVROS straight to PX4'),
        DeclareLaunchArgument('router_conf', default_value=os.path.join(WS, 'src/gcs/router/sim.conf')),
        router,
        # world:=.../worlds/comp_far1.sdf - same course, target QR at the far end past the red zones (red-zone test)
        DeclareLaunchArgument('world', default_value='/home/szyfrowac/workspaces/vio_cuda_ws/sim/worlds/competition.sdf'),
        # gz-transport for Gazebo + PX4 + the bridge: 127.0.0.1 = Jetson only; the Jetson's LAN IP (10.42.0.247) +
        # a fixed partition lets host_gui.sh on the laptop join. Unset GZ_PARTITION means hostname:user - which
        # differs between the machines, so the laptop GUI would see nothing
        DeclareLaunchArgument('gz_ip', default_value=os.environ.get('GZ_IP', '127.0.0.1')),
        DeclareLaunchArgument('gz_partition', default_value=os.environ.get('GZ_PARTITION', 'vio')),
        DeclareLaunchArgument('headless', default_value='1', description='1: no Gazebo GUI on the Jetson'),
        set_fastdds,
        run_px4,
        gz_bridge,
        mavros_node,
    ])
