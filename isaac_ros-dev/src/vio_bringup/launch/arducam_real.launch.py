"""The real Arducam IMX219 on the Jetson (CSI): /arducam/image_raw + /arducam/camera_info + its TF, the same names
as the sim (sim_arducam.launch.py), so the mission nodes and the ground station work unchanged.
    ros2 launch vio_bringup arducam_real.launch.py [sensor_id:=0] [calibration:=<arducam_calib.py yaml>]
View: the ground station's camera panel (http://10.42.0.247:8080), or rqt_image_view /arducam/image_raw.
Without calibration:= the lens is the sim's nominal 140 deg equidistant model - fine for viewing, calibrate
(calibration/arducam_calib.py) before trusting ground projections.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# straight down under the body plate, as in sim_arducam.launch.py (placeholder until the real mount is measured)
MOUNT = ['.13', '0', '-.03', '0', '1.570796', '0']
OPTICAL_Q = ['--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5']


def generate_launch_description():
    x, y, z, roll, pitch, yaw = MOUNT
    return LaunchDescription([
        DeclareLaunchArgument('sensor_id', default_value='0', description='0 = CAM0, 1 = CAM1'),
        DeclareLaunchArgument('width', default_value='1640'),
        DeclareLaunchArgument('height', default_value='1232'),
        DeclareLaunchArgument('fps', default_value='30'),
        DeclareLaunchArgument('rate', default_value='15.0'),
        DeclareLaunchArgument('flip_method', default_value='0'),
        DeclareLaunchArgument('max_exposure_ms', default_value='20.0'),
        DeclareLaunchArgument('qr_max_exposure_ms', default_value='5.0', description='QR mode exposure cap'),
        DeclareLaunchArgument('force_qr', default_value='false',
                              description='true: QR mode (3280 px + short exposure) without a mission (bench test)'),
        DeclareLaunchArgument('calibration', default_value=''),
        Node(package='vio_bringup', executable='arducam_csi', output='screen',
             parameters=[{'sensor_id': LaunchConfiguration('sensor_id'), 'width': LaunchConfiguration('width'),
                          'height': LaunchConfiguration('height'), 'fps': LaunchConfiguration('fps'),
                          'rate': LaunchConfiguration('rate'), 'flip_method': LaunchConfiguration('flip_method'),
                          'max_exposure_ms': LaunchConfiguration('max_exposure_ms'),
                          'qr_max_exposure_ms': LaunchConfiguration('qr_max_exposure_ms'),
                          'force_qr': LaunchConfiguration('force_qr')}]),
        Node(package='vio_bringup', executable='fisheye_camera_info', output='screen',
             parameters=[{'calibration_file': LaunchConfiguration('calibration')}]),
        Node(package='tf2_ros', executable='static_transform_publisher', name='tf_arducam_link',
             arguments=['--x', x, '--y', y, '--z', z, '--roll', roll, '--pitch', pitch, '--yaw', yaw,
                        '--frame-id', 'base_link', '--child-frame-id', 'arducam_link']),
        Node(package='tf2_ros', executable='static_transform_publisher', name='tf_arducam_optical',
             arguments=['--x', '0.018', *OPTICAL_Q,
                        '--frame-id', 'arducam_link', '--child-frame-id', 'arducam_optical_frame']),
    ])
