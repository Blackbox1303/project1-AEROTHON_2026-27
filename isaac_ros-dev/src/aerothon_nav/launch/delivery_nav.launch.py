"""Nav2 for the delivery zone: a second Nav2 instance in namespace `delivery` (config/nav2_delivery.yaml) +
frontier_explorer. The corridor Nav2 (corridor_nav.launch.py) stays as it is.
Unlike nav2_bringup's navigation_launch.py this keeps /tf absolute (that launch remaps /tf -> tf, i.e.
/delivery/tf under a namespace, where nothing publishes). Needs delivery.launch.py (delivery_mapper ->
/delivery/map, qr_locator -> /delivery/target), odom_tf + offboard_bridge (corridor_nav.launch.py).
    ros2 launch aerothon_nav delivery_nav.launch.py sim:=true
    ros2 service call /delivery/explore/start std_srvs/srv/Trigger
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.descriptions import ParameterFile
from nav2_common.launch import RewrittenYaml

NODES = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator', 'velocity_smoother']


def generate_launch_description():
    pkg = get_package_share_directory('aerothon_nav')
    sim = LaunchConfiguration('sim')
    params = ParameterFile(RewrittenYaml(source_file=os.path.join(pkg, 'config', 'nav2_delivery.yaml'),
                                         root_key='delivery', param_rewrites={'use_sim_time': sim},
                                         convert_types=True), allow_substs=True)
    ns = 'delivery'
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false'),
        DeclareLaunchArgument('zone', default_value='[15.0, 55.0, -15.0, 15.0]',
                              description='delivery zone x_min, x_max, y_min, y_max in odom (= delivery.launch.py)'),
        DeclareLaunchArgument('world_json', default_value='', description='sim: zone from the world file'),
        DeclareLaunchArgument('gain_radius', default_value='4.0', description='m: QR-readable radius at 10 m'),
        DeclareLaunchArgument('manager_delay', default_value='20.0', description='s before bringing Nav2 up'),
        DeclareLaunchArgument('goal_timeout', default_value='40.0',
                              description='s per frontier goal (wall time: raise for a slow sim)'),
        Node(package='nav2_controller', executable='controller_server', namespace=ns, output='screen',
             parameters=[params], remappings=[('cmd_vel', 'cmd_vel_nav')]),
        Node(package='nav2_planner', executable='planner_server', namespace=ns, output='screen', parameters=[params]),
        Node(package='nav2_behaviors', executable='behavior_server', namespace=ns, output='screen', parameters=[params]),
        Node(package='nav2_bt_navigator', executable='bt_navigator', namespace=ns, output='screen', parameters=[params]),
        Node(package='nav2_velocity_smoother', executable='velocity_smoother', namespace=ns, output='screen',
             parameters=[params], remappings=[('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel')]),
        # started late: brought up together with the sim and the corridor Nav2, controller_server's reply to
        # 'configure' timed out ("failed to send response ... (timeout)", DDS discovery race) and Humble's
        # lifecycle manager then waited forever (2026-10-19)
        TimerAction(period=LaunchConfiguration('manager_delay'), actions=[
            Node(package='nav2_lifecycle_manager', executable='lifecycle_manager', name='lifecycle_manager_navigation',
                 namespace=ns, output='screen',
                 parameters=[{'use_sim_time': sim, 'autostart': True, 'node_names': NODES}])]),
        Node(package='aerothon_nav', executable='frontier_explorer', output='screen',
             parameters=[{'use_sim_time': sim, 'gain_radius': LaunchConfiguration('gain_radius'),
                          'zone': LaunchConfiguration('zone'), 'world_json': LaunchConfiguration('world_json'),
                          'goal_timeout': LaunchConfiguration('goal_timeout')}]),
    ])
