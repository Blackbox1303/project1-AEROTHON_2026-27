"""The REAL drone's TUNNEL + SLAM stack in the dev container on the onboard Jetson (Intel RealSense D435/D435i):
    ros2 launch aerothon_nav tunnel_real.launch.py [vslam:=false] [course:=v2] [world_json:=...]
Normally started by launch_all/tunnel_slam.launch.py or from the Tunnel + SLAM dashboard (:8093).
Runs ONLY the tunnel / corridor + SLAM nodes on the Intel RealSense depth camera:
  - realsense2_camera_node: depth 640x360@30 (/camera/depth/*) + colour 640x360@30 (/camera/color/*);
    emitter on by default for plain tunnel walls (or emitter off + infra1/infra2 on when vslam:=true)
  - vio_bringup obstacle_scan: /vio/obstacle_scan (levelled 2D LaserScan) + /vio/depth_points (3D PointCloud2)
  - slam_toolbox (async_slam_toolbox_node): 2D SLAM of the tunnel -> /map + map->odom TF
  - optional cuVSLAM (vslam:=true): stereo VIO on /camera/infra1+2 -> /visual_slam/tracking/odometry + vio_to_px4
  - aerothon_nav tunnel stack: odom_tf (with 20 Hz fallback so SLAM works even without Pixhawk/GPS on the bench),
    offboard_bridge, banner_detector (on /camera/color/image_raw), entry_align, corridor_nav, altitude_planner.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vio_bringup.rig import imu_to_camera_tf_node, load_rig
from vio_bringup.vslam import visual_slam


def include(pkg, launch_file, **kwargs):
    return IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(get_package_share_directory(pkg), 'launch', launch_file)),
        launch_arguments={k: v if hasattr(v, 'perform') else str(v) for k, v in kwargs.items()}.items())


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context)  # noqa: E731
    nav_pkg = get_package_share_directory('aerothon_nav')
    vio_pkg = get_package_share_directory('vio_bringup')
    rig = load_rig(os.path.join(vio_pkg, 'config', 'rig.yaml'))
    use_vslam = g('vslam').lower() in ('true', '1', 'yes')
    gj = g('world_json')
    if not gj or not os.path.isfile(gj):
        gj = ''
    course_file = os.path.join(nav_pkg, 'config', f'course_{g("course")}.yaml')

    nodes = [
        # PX4 IMU rate setter + relay (works when MAVROS is up on the host; harmless when running camera-only)
        include('vio_bringup', 'mavros_imu.launch.py', mavros='false', sim='false', imu_rate='50.0', rviz='false',
                tf_viz='false'),
        Node(package='tf2_ros', executable='static_transform_publisher', name='base_to_imu_tf',
             arguments=['--frame-id', 'base_link', '--child-frame-id', 'imu_link']),
        imu_to_camera_tf_node(rig),
        # Intel RealSense D435/D435i: depth + RGB (+ IR stereo when vslam:=true)
        Node(name='camera', namespace='', package='realsense2_camera', executable='realsense2_camera_node',
             output='screen',
             parameters=[{
                 'initial_reset': True,
                 'enable_depth': True,
                 'enable_color': True,
                 'enable_infra1': use_vslam,
                 'enable_infra2': use_vslam,
                 'enable_gyro': False,
                 'enable_accel': False,
                 'depth_module.emitter_enabled': 0 if use_vslam else 1,
                 'depth_module.profile': '640x360x30',
                 'depth_module.depth_profile': '640x360x30',
                 'depth_module.infra_profile': '640x360x30',
                 'rgb_camera.profile': '640x360x30',
                 'rgb_camera.color_profile': '640x360x30',
                 'align_depth.enable': True,
                 'pointcloud.enable': False,
                 'publish_tf': True,
                 'depth_module.global_time_enabled': True,
             }]),
        # Levelled 2D LaserScan (/vio/obstacle_scan) + 3D PointCloud2 (/vio/depth_points)
        Node(package='vio_bringup', executable='obstacle_scan', output='screen',
             parameters=[{'use_sim_time': False, 'band': float(g('scan_band')), 'stride': 2, 'cloud_stride': 2}]),
        # odom -> base_link from PX4 (/mavros/local_position/odom), cuVSLAM, or 20 Hz bench fallback
        Node(package='aerothon_nav', executable='odom_tf', output='screen',
             parameters=[{'use_sim_time': False, 'restamp': False, 'fallback_rate': 20.0}]),
        # 2D SLAM of the tunnel on /vio/obstacle_scan -> /map + map -> odom TF
        Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox', output='screen',
             parameters=[
                 os.path.join(nav_pkg, 'config', 'slam_toolbox_2d.yaml'),
                 {'use_sim_time': False, 'map_update_interval': 1.0,
                  'minimum_travel_distance': 0.05, 'minimum_travel_heading': 0.05},
             ]),
        # Tunnel navigation + 3D altitude/window planner + banner entrance alignment
        Node(package='aerothon_nav', executable='offboard_bridge', output='screen',
             parameters=[os.path.join(nav_pkg, 'config', 'offboard_bridge.yaml'), {'geofence_json': gj}]),
        Node(package='aerothon_nav', executable='corridor_nav', output='screen',
             parameters=[{'use_sim_time': False}, course_file]),
        Node(package='aerothon_nav', executable='banner_detector', output='screen',
             parameters=[os.path.join(nav_pkg, 'config', 'banner.yaml'), {'use_sim_time': False}]),
        Node(package='aerothon_nav', executable='entry_align', output='screen',
             parameters=[{'use_sim_time': False}, course_file]),
        Node(package='aerothon_nav', executable='altitude_planner', output='screen',
             parameters=[{'use_sim_time': False, 'dump_file': g('altitude_dump')}]),
    ]
    if use_vslam:
        nodes.extend(visual_slam(rig, LaunchConfiguration('imu'), map_frame='vio_map', odom_frame='vio_odom',
                                 use_sim_time=False, publish_tf=False, image_jitter_ms=45.0))
        nodes.append(Node(package='vio_bringup', executable='vio_to_px4', output='screen',
                          parameters=[{'send': g('px4').lower() in ('true', '1', 'yes'),
                                       'anchor_child': 'vio_map', 'use_sim_time': False}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('world_json', default_value='/workspaces/isaac_ros-dev/runs/gcs/mission_geofence.json',
                              description='optional geofence json (tunnel works with or without it)'),
        DeclareLaunchArgument('course', default_value='v2', description='config/course_<course>.yaml (v2 | v1)'),
        DeclareLaunchArgument('scan_band', default_value='0.20', description='m above/below flight height for 2D scan'),
        DeclareLaunchArgument('vslam', default_value='false',
                              description='true: also run cuVSLAM stereo VIO on RealSense IR pair (emitter off)'),
        DeclareLaunchArgument('imu', default_value='false', description='cuVSLAM IMU fusion when vslam:=true'),
        DeclareLaunchArgument('px4', default_value='false', description='send cuVSLAM velocity to PX4 EKF2'),
        DeclareLaunchArgument('altitude_dump', default_value='', description='debug: altitude map npz'),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', '/workspaces/isaac_ros-dev/fastdds_udp.xml'),
        OpaqueFunction(function=setup),
    ])
