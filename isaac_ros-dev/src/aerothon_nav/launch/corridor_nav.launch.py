"""Corridor navigation: Nav2 (NavFn A* + MPPI, omnidirectional) in the odom frame on PX4's
estimate, the offboard bridge, corridor_nav, and the entrance finder (banner_detector on the
Arducam + entry_align: banner -> corridor axis from the D435 walls -> corridor_nav). Needs MAVROS (sim: sim_view.launch.py
mavros_tf:=false) and /vio/obstacle_scan (sim: sim_vio.launch.py depth:=true vio_tf:=false
scan_band:=0.4) running.
    ros2 launch aerothon_nav corridor_nav.launch.py sim:=true
    ros2 param set /corridor_nav entry_x 5.0  (entry_y, entry_yaw: entrance in odom, yaw INTO it)
    ros2 service call /nav/corridor/run std_srvs/srv/Trigger
    ros2 topic echo /nav/corridor/state
    or find the entrance from its banner first (then it runs the corridor itself):
    ros2 param set /entry_align colour green      (orange for the return corridor)
    ros2 service call /nav/entry/align std_srvs/srv/Trigger
    ros2 topic echo /nav/entry/state
Needs the Arducam (sim: sim_arducam.launch.py) for the banner and the start QR:
    ros2 service call /mission/read_qr std_srvs/srv/Trigger     (hovering over it; result on /mission/start_qr)
    sim: qr_fallback:=/arducam/nadir_hd/image_raw  pylib:=<ws>/sim/.pylib  (pyzbar for the laptop)
Frames: map -(static, sim_view)-> odom -(odom_tf)-> base_link -> imu_link -> camera_link -> ...
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    pkg = get_package_share_directory('aerothon_nav')
    sim = LaunchConfiguration('sim')
    params = RewrittenYaml(
        source_file=os.path.join(pkg, 'config', 'nav2_corridor.yaml'),
        param_rewrites={'default_nav_to_pose_bt_xml': os.path.join(pkg, 'behavior_trees', 'navigate_corridor.xml')},
        convert_types=True)
    course_file = [os.path.join(pkg, 'config', 'course_'), LaunchConfiguration('course'), '.yaml']
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false', description='sim time + restamped PX4 pose'),
        DeclareLaunchArgument('course', default_value='v2', description='corridor settings: v2 (new) | v1 (first course)'),
        DeclareLaunchArgument('pylib', default_value='', description='extra Python path for pyzbar'),
        DeclareLaunchArgument('altitude_dump', default_value='', description='debug: altitude map npz'),
        DeclareLaunchArgument('qr_fallback', default_value='', description='sim: nadir_hd stand-in topic'),
        DeclareLaunchArgument('world_json', default_value='', description='sim: world ground truth -> geofence'),
        # corridor_nav flies the corridors in nav_mode direct (no A*: the 2D SLAM pose and map, exit from slam's /map),
        # so this Nav2 is unused: off by default - it was only ever paused, and still held ~8 processes' RAM on the
        # 8 GB Jetson. nav2:=true for corridor_nav nav_mode nav2 (and the supervisor's corridor_nav2 true)
        DeclareLaunchArgument('nav2', default_value='false', description='the corridor Nav2 (nav_mode nav2 only)'),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('nav2_bringup'),
                                                       'launch', 'navigation_launch.py')),
            launch_arguments={'use_sim_time': sim, 'params_file': params, 'autostart': 'true'}.items(),
            condition=IfCondition(LaunchConfiguration('nav2'))),
        Node(package='aerothon_nav', executable='odom_tf', output='screen',
             parameters=[{'use_sim_time': sim, 'restamp': sim}]),
        Node(package='aerothon_nav', executable='offboard_bridge', output='screen',
             parameters=[os.path.join(pkg, 'config', 'offboard_bridge.yaml'),
                         {'geofence_json': LaunchConfiguration('world_json')}]),
        Node(package='aerothon_nav', executable='corridor_nav', output='screen',
             parameters=[{'use_sim_time': sim}, course_file]),
        Node(package='aerothon_nav', executable='banner_detector', output='screen',
             parameters=[os.path.join(pkg, 'config', 'banner.yaml'), {'use_sim_time': sim}]),
        Node(package='aerothon_nav', executable='entry_align', output='screen',
             parameters=[{'use_sim_time': sim}, course_file]),
        # 2D SLAM of the corridors on the levelled D435 scan -> /map (corridor Nav2 global costmap static layer)
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox', output='screen',
             parameters=[os.path.join(pkg, 'config', 'slam_toolbox_2d.yaml'), {'use_sim_time': sim}]),
        Node(package='aerothon_nav', executable='altitude_planner', output='screen',
             parameters=[{'use_sim_time': sim, 'dump_file': LaunchConfiguration('altitude_dump')}]),
        Node(package='aerothon_nav', executable='qr_reader', output='screen',
             parameters=[{'use_sim_time': sim, 'pylib': LaunchConfiguration('pylib'),
                          'fallback_topic': LaunchConfiguration('qr_fallback')}]),
    ])
