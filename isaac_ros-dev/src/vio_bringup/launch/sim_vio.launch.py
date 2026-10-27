"""cuVSLAM (and optionally obstacle_scan) on the simulated D435. Run in the CONTAINER, with
sim/run_sim.sh, sim_bridge.launch.py and sim_view.launch.py running on the host.

Same node configuration as the real stack (vio_bringup.vslam), but on sim time, with the sim
rig (config/rig_sim.yaml) and in frames vio_map -> vio_odom -> imu_link, so it doesn't clash
with MAVROS' map -> base_link in the same TF tree.
    ros2 launch vio_bringup sim_vio.launch.py                    # stereo + IMU
    ros2 launch vio_bringup sim_vio.launch.py imu:=false         # stereo only
    ros2 launch vio_bringup sim_vio.launch.py depth:=true        # + obstacle scan / dense cloud
    ros2 launch vio_bringup sim_vio.launch.py px4:=false         # don't feed VIO to PX4 (vio_to_px4 still checks)
    ros2 launch vio_bringup sim_vio.launch.py vslam:=false depth:=true   # no cuVSLAM / vio_to_px4: only the camera
        frames + obstacle scan (navigation on PX4's GPS estimate; a full stack on the Jetson saturated it, 2026-10-20)
    ros2 launch vio_bringup sim_vio.launch.py vio_tf:=false      # navigation: PX4's estimate owns
        the body frame (odom -> base_link from aerothon_nav odom_tf); cuVSLAM publishes no TF and
        imu_link hangs off base_link (the sim IMU sits at the x500's base_link origin)
The IMU is /vio/imu from sim_view.launch.py (imu_relay in sim mode: PX4 = Gazebo clock).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, GroupAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from vio_bringup.rig import imu_to_camera_tf_node, load_rig
from vio_bringup.vslam import visual_slam

# camera_link -> sensor frames of the sim D435 (from `gz model -m x500_d435_vio_0 -l camera_link`),
# rotated into the optical convention (z forward, x right, y down). Published here, next to
# cuVSLAM: static transforms from the host bridge launch never reached the container.
OPTICAL_Q = ['--qx', '-0.5', '--qy', '0.5', '--qz', '-0.5', '--qw', '0.5']
SENSORS = {
    'camera_infra1_optical_frame': ('0.0125', '0.0'),
    'camera_infra2_optical_frame': ('0.0125', '-0.05'),
    'camera_depth_optical_frame': ('0.0125', '0.0'),
    'camera_color_optical_frame': ('0.0125', '0.015'),   # the banner camera (2026-10-25)
}


def camera_frame_tf(child, x, y):
    return Node(package='tf2_ros', executable='static_transform_publisher', name=f'tf_{child}',
                arguments=['--x', x, '--y', y, '--z', '0', *OPTICAL_Q,
                           '--frame-id', 'camera_link', '--child-frame-id', child])


def generate_launch_description():
    pkg = get_package_share_directory('vio_bringup')
    rig = load_rig(os.path.join(pkg, 'config', 'rig_sim.yaml'))
    return LaunchDescription([
        DeclareLaunchArgument('depth', default_value='false'),
        DeclareLaunchArgument('imu', default_value='true'),
        DeclareLaunchArgument('px4', default_value='true'),   # feed VIO to PX4's EKF2
        DeclareLaunchArgument('vio_tf', default_value='true'),
        DeclareLaunchArgument('scan_band', default_value='0.25'),  # m, obstacle_scan height band
        DeclareLaunchArgument('vslam', default_value='true'),      # false: no cuVSLAM, no vio_to_px4
        Node(package='tf2_ros', executable='static_transform_publisher', name='base_to_imu_tf',
             arguments=['--frame-id', 'base_link', '--child-frame-id', 'imu_link'],
             condition=UnlessCondition(LaunchConfiguration('vio_tf'))),
        imu_to_camera_tf_node(rig),
        *[camera_frame_tf(child, x, y) for child, (x, y) in SENSORS.items()],
        GroupAction(visual_slam(rig, LaunchConfiguration('imu'), map_frame='vio_map', odom_frame='vio_odom',
                                use_sim_time=True, publish_tf=LaunchConfiguration('vio_tf'),
                                image_jitter_ms=80.0),   # sim D435 at 15 Hz (sim/models, lowered for the Jetson)
                    condition=IfCondition(LaunchConfiguration('vslam'))),
        Node(package='vio_bringup', executable='obstacle_scan', output='screen',
             # sim depth image is 320x180 (sim/models, lowered for the Jetson: at 640x360 ~1 depth frame per several s
             # reached the container and the corridor costmap stayed empty -> wall strike, 2026-10-20), so strides 2:
             # about the old point density from a quarter of the data
             parameters=[{'use_sim_time': True, 'band': LaunchConfiguration('scan_band'), 'stride': 2,
                          'cloud_stride': 2}],
             condition=IfCondition(LaunchConfiguration('depth'))),
        # wall-clock stamps: MAVROS runs on wall time, VIO on sim time
        # always runs (health, feature gate, resets, RViz anchor of cuVSLAM's frame); sends to PX4 only if px4
        Node(package='vio_bringup', executable='vio_to_px4', output='screen',
             # sim time for its timing (stale / recover / ...); stale_after 2.5: cuVSLAM gives ~2 poses per sim-s on
             # the loaded Jetson sim (real drone: 0.2)
             parameters=[{'restamp_now': True, 'send': LaunchConfiguration('px4'), 'anchor_child': 'vio_map',
                          'use_sim_time': True, 'stale_after': 2.5}],
             condition=IfCondition(LaunchConfiguration('vslam'))),
    ])
