"""Step 1: PX4 IMU over MAVROS, rate bump, health monitor, and RViz view."""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction, IncludeLaunchDescription, OpaqueFunction
from launch.conditions import IfCondition
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vio_bringup.rig import imu_relay_params, load_rig


def imu_relay(context, pkg):
    """Real Pixhawk: rig.yaml, stamps mapped to the PC clock. sim:=true: rig_sim.yaml and
    PX4's own clock, which in SITL is Gazebo sim time like the simulated camera."""
    sim = LaunchConfiguration('sim').perform(context).lower() in ('true', '1')
    rig = load_rig(os.path.join(pkg, 'config', 'rig_sim.yaml') if sim else None)
    params = {**imu_relay_params(rig), 'stamp_source': 'fcu' if sim else 'pc'}
    return [Node(package='vio_bringup', executable='imu_relay', output='screen', parameters=[params])]


def generate_launch_description():
    pkg = get_package_share_directory('vio_bringup')
    fcu_url = LaunchConfiguration('fcu_url')
    imu_rate = LaunchConfiguration('imu_rate')

    # MAVROS' generic node.launch with PX4's config but our plugin list (px4.launch hard-codes
    # its own). Scoped so its arguments (namespace, ...) don't leak into ours.
    mavros_share = get_package_share_directory('mavros')
    mavros = GroupAction([IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(mavros_share, 'launch', 'node.launch')),
        launch_arguments={
            'fcu_url': fcu_url,
            'gcs_url': '',
            'tgt_system': '1',
            'tgt_component': '1',
            'pluginlists_yaml': os.path.join(pkg, 'config', 'mavros_pluginlists.yaml'),
            'config_yaml': os.path.join(mavros_share, 'launch', 'px4_config.yaml'),
        }.items())], condition=IfCondition(LaunchConfiguration('mavros')))

    return LaunchDescription([
        DeclareLaunchArgument('fcu_url', default_value='/dev/ttyACM0:921600'),
        DeclareLaunchArgument('imu_rate', default_value='250.0'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('monitor_topic', default_value='/vio/imu'),
        # attitude-only TF for the IMU view; off in the full stack (cuVSLAM owns map/odom)
        DeclareLaunchArgument('tf_viz', default_value='true'),
        DeclareLaunchArgument('sim', default_value='false'),
        # false: MAVROS runs elsewhere (sim: on the host - the container's mavros_node dies, exit 127)
        DeclareLaunchArgument('mavros', default_value='true'),
        mavros,
        Node(package='vio_bringup', executable='imu_rate_setter',
             parameters=[{'rate_hz': imu_rate}], output='screen'),
        OpaqueFunction(function=imu_relay, kwargs={'pkg': pkg}),
        Node(package='vio_bringup', executable='imu_monitor', output='screen',
             parameters=[{'topic': LaunchConfiguration('monitor_topic')}]),
        Node(package='vio_bringup', executable='imu_tf_broadcaster', output='screen',
             condition=IfCondition(LaunchConfiguration('tf_viz'))),
        Node(package='rviz2', executable='rviz2', output='log',
             arguments=['-d', os.path.join(pkg, 'rviz', 'imu.rviz')],
             condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
