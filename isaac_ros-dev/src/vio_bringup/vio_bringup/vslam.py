"""The cuVSLAM node as this project runs it, shared by the real and the sim launch files.

cuVSLAM runs as a composable node in its own container (NVIDIA's supported way; its standalone
executable can't find the GXF libraries). The container restarts if it crashes, and the node is
re-loaded into it on every (re)start: launch_ros does not reload composable nodes into a
respawned container by itself. A restart resets the VIO pose to the origin.
(2026-10-18: cuVSLAM crashed with a GPU page fault, Xid 31, the moment the laptop resumed from
suspend; CUDA work doesn't survive s2idle suspend here and the GPU needed a reboot. Don't let
the dev machine sleep during runs.)
"""
from launch.actions import RegisterEventHandler
from launch.event_handlers import OnProcessStart
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch_ros.descriptions import ComposableNode

from vio_bringup.rig import cuvslam_imu_params


def visual_slam(rig, imu, map_frame='map', odom_frame='odom', use_sim_time=False, publish_tf=True,
                image_jitter_ms=34.0):
    """Launch actions for cuVSLAM on /camera/infra1+2 (+ /vio/imu when imu is true).
    Pose is for imu_link (the drone). publish_tf false: odometry topics only, no map -> odom ->
    imu_link TF (when another estimate, e.g. PX4's for navigation, owns the body frame).
    image_jitter_ms: frame gap above which cuVSLAM counts a frame as late (34 = 30 Hz; the sim's 15 Hz D435 needs more).
    Returns a list: use as *visual_slam(...)."""
    node = ComposableNode(
        name='visual_slam_node',
        package='isaac_ros_visual_slam',
        plugin='nvidia::isaac_ros::visual_slam::VisualSlamNode',
        parameters=[{
            'enable_image_denoising': False,
            'image_jitter_threshold_ms': float(image_jitter_ms),
            'rectified_images': True,
            'enable_imu_fusion': imu,
            'imu_frame': 'imu_link',
            'base_frame': 'imu_link',
            'map_frame': map_frame,
            'odom_frame': odom_frame,
            'publish_odom_to_base_tf': publish_tf,
            'publish_map_to_odom_tf': publish_tf,
            'camera_optical_frames': [
                'camera_infra1_optical_frame',
                'camera_infra2_optical_frame',
            ],
            'enable_slam_visualization': True,
            'enable_landmarks_view': True,
            'enable_observations_view': True,
            'use_sim_time': use_sim_time,
            **cuvslam_imu_params(rig),
        }],
        remappings=[
            ('visual_slam/image_0', '/camera/infra1/image_rect_raw'),
            ('visual_slam/camera_info_0', '/camera/infra1/camera_info'),
            ('visual_slam/image_1', '/camera/infra2/image_rect_raw'),
            ('visual_slam/camera_info_1', '/camera/infra2/camera_info'),
            ('visual_slam/imu', '/vio/imu'),
        ])
    container = ComposableNodeContainer(
        name='visual_slam_container', namespace='',
        package='rclcpp_components', executable='component_container_mt',
        composable_node_descriptions=[], output='screen',
        respawn=True, respawn_delay=2.0,
        parameters=[{'use_sim_time': use_sim_time}])
    reload = RegisterEventHandler(OnProcessStart(
        target_action=container,
        on_start=[LoadComposableNodes(target_container=container, composable_node_descriptions=[node])]))
    return [container, reload]
