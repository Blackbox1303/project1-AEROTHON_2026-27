"""Launch RViz2 and/or RTAB-Map (3D RGB-D SLAM + rtabmap_viz GUI) alongside the running Tunnel + SLAM stack:
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_viz.launch.py
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_viz.launch.py rviz:=true rtabmap:=true
    ros2 launch ~/workspaces/isaac_ros-dev/src/launch_all/tunnel_viz.launch.py source:=infra rgbd_odom:=false
source:=infra is for cuVSLAM runs (color off, IR pair emitter off): RTAB-Map maps the left IR image + the D435
depth, which the camera already registers to the left IR imager, on /nav/odom (cuVSLAM via odom_tf).
Can also be toggled anytime from the buttons in the Tunnel + SLAM dashboard (http://10.42.0.247:8093/).
"""
import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

WS = Path.home() / 'workspaces/isaac_ros-dev'
RVIZ_CFG = WS / 'src/vio_bringup/rviz/tunnel_slam.rviz'
os.environ.pop('QT_QPA_PLATFORM_PLUGIN_PATH', None)
os.environ.pop('QT_QPA_FONTDIR', None)


def setup(context):
    g = lambda n: LaunchConfiguration(n).perform(context).lower() in ('true', '1', 'yes')  # noqa: E731
    use_rviz = g('rviz')
    use_rtabmap = g('rtabmap')
    use_rtabmap_viz = g('rtabmap_viz')
    use_rgbd_odom = g('rgbd_odom')
    if LaunchConfiguration('source').perform(context) == 'infra':
        img, depth, info = ('/camera/infra1/image_rect_raw', '/camera/depth/image_rect_raw',
                            '/camera/depth/camera_info')
    else:
        img, depth, info = ('/camera/color/image_raw', '/camera/aligned_depth_to_color/image_raw',
                            '/camera/color/camera_info')

    nodes = []
    if use_rtabmap:
        common_params = [{
            'frame_id': 'base_link',
            'odom_frame_id': 'odom',
            'map_frame_id': 'map',
            'publish_tf': False,              # slam_toolbox + odom_tf own map -> odom -> base_link TF
            'subscribe_depth': True,
            'subscribe_rgb': True,
            'subscribe_scan': True,
            'approx_sync': True,
            'queue_size': 20,
            'qos_image': 2,                   # BestEffort (matches realsense2_camera sensor_data QoS)
            'qos_camera_info': 2,
            'qos_scan': 2,
            'qos_odom': 1,
            'Reg/Strategy': '1',              # 1 = Vis + ICP (uses both RealSense RGB-D and /vio/obstacle_scan)
            'RGBD/NeighborLinkRefining': 'true',
            'RGBD/ProximityBySpace': 'true',
            'RGBD/AngularUpdate': '0.05',
            'RGBD/LinearUpdate': '0.05',
            'Grid/FromDepth': 'true',
            'Grid/RangeMax': '6.0',
            'Vis/MinInliers': '12',
        }]
        remaps = [
            ('rgb/image', img),
            ('depth/image', depth),
            ('rgb/camera_info', info),
            ('scan', '/vio/obstacle_scan'),
            ('odom', '/nav/odom'),
        ]
        odom_remaps = [
            ('rgb/image', img),
            ('depth/image', depth),
            ('rgb/camera_info', info),
        ]
        if use_rgbd_odom:
            # RGB-D visual odometry -> /rtabmap/odom (picked up by odom_tf when PX4 odom is absent)
            nodes.append(Node(
                package='rtabmap_odom', executable='rgbd_odometry', name='rgbd_odometry', namespace='rtabmap',
                output='screen',
                parameters=[{
                    'frame_id': 'base_link',
                    'odom_frame_id': 'odom',
                    'publish_tf': False,
                    'approx_sync': True,
                    'approx_sync_max_interval': 0.05,
                    'always_process_most_recent_frame': False,
                    'qos': 2,
                    'Odom/ResetCountdown': '1',
                    'Vis/MinInliers': '8',
                }],
                remappings=odom_remaps))
        nodes.append(Node(
            package='rtabmap_slam', executable='rtabmap', name='rtabmap', namespace='rtabmap',
            output='screen', arguments=['-d'],
            parameters=common_params, remappings=remaps))
        if use_rtabmap_viz:
            nodes.append(Node(
                package='rtabmap_viz', executable='rtabmap_viz', name='rtabmap_viz', namespace='rtabmap',
                output='screen', parameters=common_params, remappings=remaps))

    if use_rviz:
        nodes.append(Node(
            package='rviz2', executable='rviz2', name='rviz2_tunnel', output='screen',
            arguments=['-d', str(RVIZ_CFG)]))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('rviz', default_value='true', description='open RViz2 with tunnel_slam.rviz'),
        DeclareLaunchArgument('rtabmap', default_value='true', description='run RTAB-Map 3D RGB-D SLAM'),
        DeclareLaunchArgument('rtabmap_viz', default_value='true', description='open RTAB-Map GUI window'),
        DeclareLaunchArgument('source', default_value='color',
                              description='color: RGB + aligned depth; infra: left IR + depth (cuVSLAM runs)'),
        DeclareLaunchArgument('rgbd_odom', default_value='true',
                              description='run RTAB-Map rgbd_odometry (/rtabmap/odom) for camera motion tracking'),
        SetEnvironmentVariable('DISPLAY', os.environ.get('DISPLAY', ':0')),
        SetEnvironmentVariable('FASTRTPS_DEFAULT_PROFILES_FILE', str(WS / 'fastdds_udp.xml')),
        OpaqueFunction(function=setup),
    ])
