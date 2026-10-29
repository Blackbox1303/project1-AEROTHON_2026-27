"""Launch the standalone `drone_qr_detector` stack (shared/direct CSI camera + static TF + 3D drone-frame QR locator).

Examples:
  1. Drone flying at 10 m altitude detecting 1 m x 1 m QR codes:
     ros2 launch drone_qr_detector drone_qr_detector.launch.py target_distance_m:=10.0 qr_size_m:=1.0

  2. Wall test / verification at 3 m distance (while another user is also accessing CSI on port 8090 or standalone):
     ros2 launch drone_qr_detector drone_qr_detector.launch.py target_distance_m:=3.0 qr_size_m:=0.30
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

MOUNT = ['0.13', '0.0', '-0.03', '0.0', '1.570796', '0.0']
OPTICAL_Q = ['--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5']


def generate_launch_description():
    x, y, z, roll, pitch, yaw = MOUNT
    return LaunchDescription([
        DeclareLaunchArgument('source_mode', default_value='auto',
                              description='auto (shared HTTP 8090 -> ROS relay -> direct CSI) | http_shared | ros_topic | csi'),
        DeclareLaunchArgument('sensor_id', default_value='0'),
        DeclareLaunchArgument('width', default_value='3280'),
        DeclareLaunchArgument('height', default_value='2464'),
        DeclareLaunchArgument('exposure_ms', default_value='6.0'),
        DeclareLaunchArgument('publish_rate', default_value='6.0'),
        DeclareLaunchArgument('calibration_file',
                              default_value='/workspaces/isaac_ros-dev/calibration/arducam_fisheye_clean.yaml'),
        DeclareLaunchArgument('focal_scale', default_value='1.0'),
        DeclareLaunchArgument('qr_size_m', default_value='1.0',
                              description='Physical QR side length (m): 1.0 for drone mission, e.g. 0.20-0.30 for wall test'),
        DeclareLaunchArgument('target_distance_m', default_value='10.0',
                              description='Distance to QR plane (m): 10.0 for flight altitude, 3.0 for 3m wall test'),
        DeclareLaunchArgument('drone_frame', default_value='base_link'),

        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='drone_qr_tf_arducam_link',
            arguments=[
                '--x', x, '--y', y, '--z', z,
                '--roll', roll, '--pitch', pitch, '--yaw', yaw,
                '--frame-id', LaunchConfiguration('drone_frame'),
                '--child-frame-id', 'arducam_link',
            ],
        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='drone_qr_tf_arducam_optical',
            arguments=[
                '--x', '0.018', *OPTICAL_Q,
                '--frame-id', 'arducam_link',
                '--child-frame-id', 'arducam_optical_frame',
            ],
        ),
        Node(
            package='drone_qr_detector',
            executable='csi_camera_node',
            name='csi_camera_node',
            output='screen',
            parameters=[{
                'source_mode': LaunchConfiguration('source_mode'),
                'sensor_id': LaunchConfiguration('sensor_id'),
                'width': LaunchConfiguration('width'),
                'height': LaunchConfiguration('height'),
                'exposure_ms': LaunchConfiguration('exposure_ms'),
                'publish_rate': LaunchConfiguration('publish_rate'),
                'calibration_file': LaunchConfiguration('calibration_file'),
                'focal_scale': LaunchConfiguration('focal_scale'),
            }],
        ),
        Node(
            package='drone_qr_detector',
            executable='qr_drone_locator',
            name='qr_drone_locator',
            output='screen',
            parameters=[{
                'calibration_file': LaunchConfiguration('calibration_file'),
                'focal_scale': LaunchConfiguration('focal_scale'),
                'qr_size_m': LaunchConfiguration('qr_size_m'),
                'target_distance_m': LaunchConfiguration('target_distance_m'),
                'drone_frame': LaunchConfiguration('drone_frame'),
            }],
        ),
    ])
