"""Simulated Arducam fisheye: camera_info, its TF on the drone, and a live view. Run on the HOST
next to sim_bridge.launch.py (which bridges /arducam/image_raw).
    ros2 launch vio_bringup sim_arducam.launch.py               # + rqt_image_view window
    ros2 launch vio_bringup sim_arducam.launch.py view:=false
    ros2 launch vio_bringup sim_arducam.launch.py calibration:=<arducam_WxH.yaml>   # lens from a calibration
Topics: /arducam/image_raw (rgb8, 21 Hz), /arducam/camera_info (equidistant fisheye).
TF: base_link -> arducam_link (mount) -> arducam_optical_frame (z forward, x right, y down).
Gazebo's wide-angle sensor ignores <gz_frame_id>, so images are stamped with its scoped name
GZ_FRAMES; a zero transform makes that an alias of arducam_optical_frame. The real camera
driver stamps arducam_optical_frame itself.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# must match the Arducam <include> pose in sim/models/iris_d435_vio/model.sdf and x500_d435_vio/model.sdf:
# 13 cm forward, 3 cm below base_link (under the body plate), looking straight down - 90 deg (45 deg nose-down until 2026-10-25) (x y z roll pitch yaw)
MOUNT = ['.13', '0', '-.03', '0', '1.570796', '0']
SENSOR_X = '0.018'   # sensor pose in arducam_link (arducam_imx219_fisheye/model.sdf)
# Gazebo's frame_id of the image: <model instance>/arducam_link/arducam - ArduCopter SITL (iris) and PX4 (x500)
GZ_FRAMES = {'iris': 'iris_d435_vio_0/arducam_link/arducam', 'x500': 'x500_d435_vio_0/arducam_link/arducam'}
OPTICAL_Q = ['--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5']


def generate_launch_description():
    x, y, z, roll, pitch, yaw = MOUNT
    return LaunchDescription([
        DeclareLaunchArgument('view', default_value='true'),
        DeclareLaunchArgument('calibration', default_value='',
                              description='YAML from calibration/arducam_calib.py; empty = Gazebo lens'),
        Node(package='vio_bringup', executable='fisheye_camera_info', output='screen',
             parameters=[{'calibration_file': LaunchConfiguration('calibration')}]),
        Node(package='tf2_ros', executable='static_transform_publisher', name='tf_arducam_link',
             arguments=['--x', x, '--y', y, '--z', z, '--roll', roll, '--pitch', pitch, '--yaw', yaw,
                        '--frame-id', 'base_link', '--child-frame-id', 'arducam_link']),
        Node(package='tf2_ros', executable='static_transform_publisher', name='tf_arducam_optical',
             arguments=['--x', SENSOR_X, *OPTICAL_Q,
                        '--frame-id', 'arducam_link', '--child-frame-id', 'arducam_optical_frame']),
        *[Node(package='tf2_ros', executable='static_transform_publisher', name=f'tf_arducam_gz_alias_{k}',
               arguments=['--frame-id', 'arducam_optical_frame', '--child-frame-id', f]) for k, f in GZ_FRAMES.items()],
        Node(package='rqt_image_view', executable='rqt_image_view', output='log',
             arguments=['/arducam/image_raw'], condition=IfCondition(LaunchConfiguration('view'))),
    ])
