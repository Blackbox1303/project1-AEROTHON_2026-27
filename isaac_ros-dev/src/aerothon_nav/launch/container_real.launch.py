"""The REAL drone's mission stack, in the dev container on the onboard Jetson: the same nodes as the sim's
container_all.launch.py, on the real sensors and wall-clock time.
    ros2 launch aerothon_nav container_real.launch.py world_json:=<geofence json> [calibration:=<arducam yaml>]
Normally started by launch_all/real_mission.launch.py (the dashboard's Real mode), which also runs mavlink-router +
MAVROS on the Pixhawk 6C Mini on the host.
  sensors  Intel RealSense D435(i): depth 640x360@30 (/camera/depth/*: obstacle_scan -> /vio/obstacle_scan, the
           2D SLAM and the altitude planner) + colour 640x360@30 (/camera/color/*: banner_detector), emitter on;
           realsense-ros 4.58 in the container names topics <namespace>/<name>/..., so namespace '' + name 'camera'
           gives the sim's /camera/... names. Arducam IMX219 straight down (vio_bringup arducam_real.launch.py:
           /arducam/image_raw + camera_info from calibration:=, the start QR, the QR search, the red zones)
  frames   base_link = imu_link (PX4's estimate, odom_tf) -> camera_link (vio_bringup/config/rig.yaml: MEASURE the
           D435's mount and write it there) -> the RealSense's own optical frames; base_link -> arducam_link
           (arducam_real.launch.py MOUNT: measure it too)
  nav      corridor_nav.launch.py (offboard bridge with the geofence, corridor nav, entry align, banner detector,
           2D SLAM, altitude planner, start-QR reader), delivery.launch.py (red-zone / searched costmap, QR locator),
           delivery_nav.launch.py (explorer; its Nav2 stays paused: A* routes), mission.launch.py (supervisor)
world_json = the ground station's geofence (runs/gcs/mission_geofence.json: the GPS corners -> odom): REQUIRED - the
offboard bridge's keep-in / tunnel ceiling and the supervisor's zone come from it.
corridor_impl:=aerothon: the AeroTHON folder's corridor navigation + VIO (launch_all/aerothon_corridor_stack.py, in its
aerothon-vio container) owns the D435 (IR pair for cuVSLAM, emitter off; depth; colour for its banner detector), the
IMU relay at 250 Hz, base_link -> imu_link -> camera_link (its rig.yaml) and obstacle_scan - none of those start here,
and corridor_nav.launch.py leaves out corridor_nav / entry_align / banner_detector / altitude_planner.
"""
import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vio_bringup.rig import imu_to_camera_tf_node, load_rig


def include(pkg, launch_file, **kwargs):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(pkg), 'launch', launch_file)),
        launch_arguments={k: v if hasattr(v, 'perform') else str(v) for k, v in kwargs.items()}.items())


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    gj = g('world_json')
    if not gj or not os.path.isfile(gj):
        raise RuntimeError(f'world_json {gj!r} missing: the real mission needs the ground station geofence')
    calib = g('calibration')
    if calib and not os.path.isfile(calib):
        raise RuntimeError(f'Arducam calibration {calib!r} not found')
    read_angle = float(g('read_angle'))
    gain = round(float(g('search_altitude')) * math.tan(math.radians(read_angle)), 2)   # read radius on the ground
    rig = load_rig(os.path.join(get_package_share_directory('vio_bringup'), 'config', 'rig.yaml'))
    impl = g('corridor_impl')
    sensors = [] if impl == 'aerothon' else [
        # PX4 IMU (MAVROS runs on the host): rate setter + relay stamped on the PC clock, like the sim's sim_view
        include('vio_bringup', 'mavros_imu.launch.py', mavros='false', sim='false', imu_rate='50.0', rviz='false',
                tf_viz='false'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='base_to_imu_tf',
             arguments=['--frame-id', 'base_link', '--child-frame-id', 'imu_link']),
        imu_to_camera_tf_node(rig),
        Node(name='camera', namespace='', package='realsense2_camera', executable='realsense2_camera_node',
             output='screen',
             parameters=[{
                 'initial_reset': True,              # it can come up streaming nothing after a killed process
                 'enable_depth': True, 'enable_color': True,
                 'enable_infra1': False, 'enable_infra2': False, 'enable_gyro': False, 'enable_accel': False,
                 'depth_module.emitter_enabled': 1,  # the projector: better depth on plain tunnel walls (no VIO)
                 'depth_module.profile': '640x360x30', 'depth_module.depth_profile': '640x360x30',
                 'rgb_camera.profile': '640x360x30', 'rgb_camera.color_profile': '640x360x30',
                 'align_depth.enable': False, 'pointcloud.enable': False, 'publish_tf': True,
                 'depth_module.global_time_enabled': True,
             }]),
        Node(package='vio_bringup', executable='obstacle_scan', output='screen',
             parameters=[{'use_sim_time': False, 'band': 0.15, 'stride': 2, 'cloud_stride': 2}]),
    ]
    return [
        *sensors,
        include('vio_bringup', 'arducam_real.launch.py', calibration=calib),
        # the start QR is read on the full-resolution QR-scan stream too (arducam_csi, while the mission scans QRs)
        include('aerothon_nav', 'corridor_nav.launch.py', sim='false', course=g('course'),
                qr_fallback='/arducam/qr/image_raw',
                nav2='false', world_json=gj, use_sim_time='false', corridor_impl=impl),
        include('aerothon_nav', 'delivery.launch.py', sim='false', world_json=gj, nadir_topic='',
                hd_topic='/arducam/qr/image_raw',
                read_angle=str(read_angle), use_sim_time='false'),
        include('aerothon_nav', 'delivery_nav.launch.py', sim='false', gain_radius=str(gain), world_json=gj,
                goal_timeout='40.0', use_sim_time='false'),
        include('aerothon_nav', 'mission.launch.py', sim='false', use_sim_time='false', world_json=gj),
        Node(package='drone_qr_detector', executable='qr_drone_locator', name='qr_drone_locator', output='screen',
             parameters=[{
                 'image_topic': '/arducam/image_raw',
                 'info_topic': '/arducam/camera_info',
                 'calibration_file': calib or '/workspaces/isaac_ros-dev/calibration/arducam_fisheye_clean.yaml',
                 'focal_scale': 1.0,
                 'qr_size_m': 1.0,
                 'target_distance_m': float(g('search_altitude')),
                 'drone_frame': 'base_link',
             }]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world_json', default_value='/workspaces/isaac_ros-dev/runs/gcs/mission_geofence.json',
                              description='the ground station geofence (GPS corners -> odom)'),
        DeclareLaunchArgument('calibration',
                              default_value='/workspaces/isaac_ros-dev/calibration/arducam_fisheye_clean.yaml',
                              description='Arducam calibration yaml (clean 3m-wall verified)'),
        DeclareLaunchArgument('course', default_value='v2', description='config/course_<course>.yaml (real timings)'),
        DeclareLaunchArgument('read_angle', default_value='15.0',
                              description="deg off nadir a QR is read at (the real fisheye: see arducam_calib's px/module)"),
        DeclareLaunchArgument('search_altitude', default_value='10.0', description='m (for the read radius)'),
        DeclareLaunchArgument('corridor_impl', default_value='current',
                              description='current | aerothon (corridor nav + VIO from ~/AeroTHON_2026, its container)'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', '/workspaces/isaac_ros-dev/fastdds_udp.xml'),
        OpaqueFunction(function=setup),
    ])
