"""Delivery zone: costmap from the Arducam (delivery_mapper) and QR reading + location (qr_locator).
Needs the Arducam (sim: sim_arducam.launch.py), PX4's pose in TF (odom_tf, corridor_nav.launch.py) and, for the
target, the stored start QR (/mission/start_qr from qr_reader).
    ros2 launch aerothon_nav delivery.launch.py zone:="[15.0, 55.0, -15.0, 15.0]"
    sim:  sim:=true world_json:=<world>.json nadir_topic:=/arducam/nadir_hd/image_raw pylib:=<ws>/sim/.pylib
    ros2 topic echo /delivery/qrs        RViz: /delivery/map (OccupancyGrid)
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    sim = LaunchConfiguration('sim')
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false'),
        DeclareLaunchArgument('zone', default_value='[15.0, 55.0, -15.0, 15.0]',
                              description='delivery zone x_min, x_max, y_min, y_max in odom'),
        DeclareLaunchArgument('world_json', default_value='', description='sim: zone from the world file'),
        DeclareLaunchArgument('nadir_topic', default_value='', description='sim: nadir_hd stand-in'),
        DeclareLaunchArgument('hd_topic', default_value='',
                              description="real: arducam_csi's full-resolution QR-scan stream (/arducam/qr/image_raw)"),
        DeclareLaunchArgument('pylib', default_value='', description='extra Python path for pyzbar'),
        DeclareLaunchArgument('read_angle', default_value='25.0', description='deg: QR-readable footprint'),
        Node(package='aerothon_nav', executable='delivery_mapper', output='screen',
             parameters=[{'use_sim_time': sim, 'zone': LaunchConfiguration('zone'),
                          'world_json': LaunchConfiguration('world_json'),
                          'read_angle': LaunchConfiguration('read_angle')}]),
        # respawn: it died once at start-up in native code (munmap_chunk(): invalid pointer - WeChat / OpenCV decoder
        # heap corruption, 2026-10-27 sim), and without it the delivery search never finds the target
        Node(package='aerothon_nav', executable='qr_locator', output='screen', respawn=True, respawn_delay=2.0,
             parameters=[{'use_sim_time': sim, 'nadir_topic': LaunchConfiguration('nadir_topic'),
                          'hd_topic': LaunchConfiguration('hd_topic'),
                          'pylib': LaunchConfiguration('pylib'), 'zone': LaunchConfiguration('zone'),
                          'world_json': LaunchConfiguration('world_json')}]),
    ])
