"""Mission supervisor (top-level state machine). The rest of the stack runs as for the tests:
corridor_nav.launch.py (Nav2 corridor, offboard_bridge, banner, entry_align, qr_reader, altitude planner),
delivery.launch.py (mapper, QR locator), delivery_nav.launch.py (delivery Nav2 + explorer), MAVROS, cameras.
    ros2 launch aerothon_nav mission.launch.py sim:=true
    ros2 service call /mission/start std_srvs/srv/Trigger       (the drone must be on the ground, disarmed)
    ros2 topic echo /mission/state          ros2 service call /mission/abort std_srvs/srv/Trigger
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('sim', default_value='false'),
        DeclareLaunchArgument('world_json', default_value='', description='the zone (delivery-only mission)'),
        Node(package='aerothon_nav', executable='mission_supervisor', output='screen',
             parameters=[{'use_sim_time': LaunchConfiguration('sim'),
                          'world_json': LaunchConfiguration('world_json')}]),
    ])
