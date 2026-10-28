"""The AeroTHON folder's corridor navigation + VIO (~/AeroTHON_2026/vio_cuda_ws), as the 8080 dashboard's
"Corridor nav: AeroTHON" choice. Runs INSIDE that folder's own container (aerothon-vio, image aerothon_vio:latest, its
install_docker build) - its code is used as it is, nothing in that folder is changed. Fed on stdin by
scripts/aerothon_corridor.sh (that container doesn't mount ~/workspaces):
    docker exec -i aerothon-vio ... python3 - aerothon_corridor_stack mode=sim|real vio_fc=0|1 < this file

What it runs (their drone.launch.py / corridor_nav.launch.py / sim_vio.launch.py, minus what our stack already owns):
  VIO        cuVSLAM on the D435 IR PAIR (infra1 + infra2, emitter off) + the Pixhawk IMU (/vio/imu: HIGHRES_IMU,
             re-stamped by imu_relay) in vio_map -> vio_odom, no TF; vio_to_px4: health gate (vo_state, observations,
             jumps, speed, disagreement with the FC) -> /vio/health, auto-reset of cuVSLAM, and with vio_fc=1 the body
             velocity to ArduPilot EKF3 ExternalNav (/mavros/odometry/out) + /vio/ekf_source (source set 2 in the
             tunnels). The depth stream is NOT used by the VIO: it only feeds obstacle_scan.
  perception obstacle_scan (D435 depth -> /vio/obstacle_scan, /vio/depth_points)
  corridor   their corridor Nav2 (NavFn + MPPI, course v5 rewrites) + lifecycle_bringup, corridor_nav (gap follower,
             yaw lock, VIO reset + EKF source switch at the entrance), entry_align, altitude_planner
  real only  the D435 (IR pair + depth), imu_rate_setter (HIGHRES_IMU + ATTITUDE_QUATERNION at 250 Hz) + imu_relay, base_link -> imu_link ->
             camera_link from THEIR rig.yaml
Ours keeps (container_all / container_real with corridor_impl:=aerothon): the banner detector (drone_qr_detector
banner_node on the Arducam -> /nav/banner/detection; not theirs), odom_tf, offboard_bridge (geofence, tunnel ceiling),
slam_toolbox, qr_reader, delivery, supervisor, MAVROS on the host; sim: the IMU relay (250 Hz) and the sim camera TFs
(sim_view / sim_vio).
vio_fc=0 (default): VIO runs and is checked, nothing goes to the flight controller and corridor_nav never switches the
EKF source - set 2 must be configured on the FC first (EK3_SRC2_VELXY 6, VISO_TYPE 1: sim gets
vio_cuda_ws/sim/ardupilot_vio_sim.parm with vio_fc:=1).
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription, LaunchService
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from nav2_common.launch import RewrittenYaml

from vio_bringup import vslam
from vio_bringup.rig import imu_relay_params, imu_to_camera_tf_node, load_rig

COURSE = 'v5'           # their current course file (corridor_width 3.51 m +- 0.3: our 3.35 m tunnels fit)


def corridor(nav, sim):
    """Their corridor_nav.launch.py without odom_tf / offboard_bridge / qr_reader / banner_detector (ours run)."""
    t = {'use_sim_time': sim}
    course = os.path.join(nav, 'config', f'course_{COURSE}.yaml')
    nav2_params = RewrittenYaml(
        source_file=os.path.join(nav, 'config', 'nav2_corridor.yaml'),
        param_rewrites={'default_nav_to_pose_bt_xml': os.path.join(nav, 'behavior_trees', 'navigate_corridor.xml'),
                        # their launch: v3-v5 values; batch 1000 = their Orin CPU setting
                        'batch_size': '1000', 'cost_scaling_factor': '8.0', 'footprint_padding': '0.05',
                        'vy_max': '0.25', 'inf_is_valid': 'False'},
        convert_types=True)
    # sim (our loaded Jetson runs ~0.2x real time): the slow-sim timings of our course_v2_slowsim
    slow_corridor = {'stall_time': 180.0} if sim else {}
    slow_align = {'look_time': 5.0, 'look_time_axis': 3.0} if sim else {}
    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(os.path.join(get_package_share_directory('nav2_bringup'), 'launch',
                                                       'navigation_launch.py')),
            launch_arguments={'use_sim_time': str(sim), 'params_file': nav2_params, 'autostart': 'false'}.items()),
        Node(package='aerothon_nav', executable='lifecycle_bringup', name='lifecycle_bringup_corridor', output='screen',
             parameters=[{**t, 'node_names': ['controller_server', 'smoother_server', 'planner_server',
                                              'behavior_server', 'bt_navigator', 'waypoint_follower',
                                              'velocity_smoother']}]),
        Node(package='aerothon_nav', executable='corridor_nav', output='screen',
             parameters=[t, course, {'vio_ekf_source': VIO_FC, **slow_corridor}]),
        Node(package='aerothon_nav', executable='entry_align', output='screen', respawn=True, respawn_delay=2.0,
             parameters=[t, course, slow_align]),
        Node(package='aerothon_nav', executable='altitude_planner', output='screen', parameters=[t]),
    ]


def vio(vb, sim):
    if sim:
        # the sim D435 runs at 15 Hz (lowered for the Jetson): frames 66 ms apart are not "late" (our vslam.py's
        # image_jitter_ms 80); their visual_slam() has no argument for it, so it goes in with the IMU parameters
        imu_params = vslam.cuvslam_imu_params
        vslam.cuvslam_imu_params = lambda rig: {**imu_params(rig), 'image_jitter_threshold_ms': 80.0}
        rig = load_rig(os.path.join(vb, 'config', 'rig_sim.yaml'))
        return [
            *vslam.visual_slam(rig, True, map_frame='vio_map', odom_frame='vio_odom', use_sim_time=True,
                               publish_tf=False),
            # their sim_vio.launch.py values; stale_after 2.5: ~2 poses per sim-s on the loaded Jetson
            Node(package='vio_bringup', executable='vio_to_px4', output='screen',
                 parameters=[{'restamp_now': True, 'send': VIO_FC, 'anchor_child': 'vio_map', 'use_sim_time': True,
                              'stale_after': 2.5}]),
            # the sim depth image is 320 x 180: strides 2 (ours does the same)
            Node(package='vio_bringup', executable='obstacle_scan', output='screen',
                 parameters=[{'use_sim_time': True, 'band': 0.3, 'stride': 2, 'cloud_stride': 2}]),
        ]
    rig = load_rig()                                   # THEIR config/rig.yaml (D435 0.14 m ahead of the Pixhawk)
    return [
        # their realsense_stereo.launch.py depth:=true profile 640x360x60
        Node(name='camera', namespace='', package='realsense2_camera', executable='realsense2_camera_node',
             output='screen',
             parameters=[{'initial_reset': True, 'enable_infra1': True, 'enable_infra2': True, 'enable_depth': True,
                          'enable_color': False, 'enable_gyro': False, 'enable_accel': False,
                          'depth_module.emitter_enabled': 0,            # cuVSLAM: no projector dots on the IR pair
                          'depth_module.profile': '640x360x60', 'depth_module.infra_profile': '640x360x60',
                          'depth_module.depth_profile': '640x360x60',
                          'depth_module.global_time_enabled': True}]),
        # their mavros_imu.launch.py without MAVROS (ours runs on the host)
        Node(package='vio_bringup', executable='imu_rate_setter', output='screen', parameters=[{'rate_hz': 250.0}]),
        Node(package='vio_bringup', executable='imu_relay', output='screen',
             parameters=[{**imu_relay_params(rig), 'stamp_source': 'pc'}]),
        Node(package='tf2_ros', executable='static_transform_publisher', name='base_to_imu_tf',
             arguments=['--frame-id', 'base_link', '--child-frame-id', 'imu_link']),
        imu_to_camera_tf_node(rig),
        *vslam.visual_slam(rig, True, map_frame='vio_map', odom_frame='vio_odom', publish_tf=False),
        Node(package='vio_bringup', executable='vio_to_px4', output='screen', parameters=[{'send': VIO_FC}]),
        Node(package='vio_bringup', executable='obstacle_scan', output='screen', parameters=[{'band': 0.3}]),
    ]


def main(argv):
    global VIO_FC
    args = dict(a.split('=', 1) for a in argv if '=' in a)
    sim = args.get('mode', 'sim') == 'sim'
    VIO_FC = args.get('vio_fc', '0') in ('1', 'true')
    nav, vb = get_package_share_directory('aerothon_nav'), get_package_share_directory('vio_bringup')
    print(f'[aerothon_corridor] {"sim" if sim else "real"}: AeroTHON corridor + VIO from {nav}; '
          f'VIO -> flight controller {"ON (EKF3 source set 2 in the tunnels)" if VIO_FC else "OFF (checks only)"}',
          flush=True)
    ld = LaunchDescription([SetEnvironmentVariable('AEROTHON_AUTOPILOT', 'ardupilot'),
                            SetEnvironmentVariable('OPENBLAS_NUM_THREADS', '1'),
                            *vio(vb, sim), *corridor(nav, sim)])
    ls = LaunchService(argv=argv)
    ls.include_launch_description(ld)
    return ls.run()


VIO_FC = False

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
