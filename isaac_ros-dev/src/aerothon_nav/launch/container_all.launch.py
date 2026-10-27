import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    # Fast DDS profile for container
    # UDP-only FastDDS (see the file: without it host <-> container data stops). In the container ~ is /home/admin,
    # so '~/workspaces/...' didn't exist and overrode a correct exported value (2026-10-20)
    fastdds_profile = next((f for f in (os.environ.get('FASTRTPS_DEFAULT_PROFILES_FILE', ''),
                                        '/workspaces/isaac_ros-dev/fastdds_udp.xml',
                                        os.path.expanduser('~/workspaces/isaac_ros-dev/fastdds_udp.xml'))
                            if f and os.path.isfile(f)), '')
    set_fastdds = SetEnvironmentVariable(
        name='FASTRTPS_DEFAULT_PROFILES_FILE',
        value=fastdds_profile,
    )

    def include(pkg, launch_file, **kwargs):
        return IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(
                    get_package_share_directory(pkg),
                    'launch',
                    launch_file,
                )
            ),
            launch_arguments={k: v if hasattr(v, 'perform') else str(v) for k, v in kwargs.items()}.items(),
        )

    # Paths
    # Inside the container: vio_cuda_ws isn't mounted, so this is a copy of vio_cuda_ws/sim/worlds/competition.json
    # (re-copy it after regenerating the world). The old host path made delivery_mapper and qr_locator die at start
    world_json_path = LaunchConfiguration('world_json')

    # The same stack as vio_cuda_ws/sim/run_corridor_test.sh. MAVROS and the Gazebo bridge run on the host
    # (launch_all/host_all.launch.py): here only the IMU relay + map -> odom from sim_view
    # vslam:=false (reduced stack for the Jetson): no cuVSLAM, and PX4's IMU streams at 50 Hz instead of 250
    # (only cuVSLAM needs the rate; obstacle_scan uses the attitude)
    vslam = LaunchConfiguration('vslam')
    sim_view = include('vio_bringup', 'sim_view.launch.py',
                       mavros='false', mavros_tf='false', rviz='false', use_sim_time='true',
                       # slam_toolbox (corridor_nav.launch.py) owns map -> odom: a second, static identity one made
                       # the SLAM correction corridor_nav reads flip between the two
                       map_tf='false',
                       imu_rate=PythonExpression(["'250.0' if '", vslam, "'.lower() == 'true' else '50.0'"]))

    # cuVSLAM + /vio/obstacle_scan (the corridor and delivery costmaps; entry_align's walls). PX4's estimate
    # owns base_link (odom_tf), so no VIO TF
    vio = include('vio_bringup', 'sim_vio.launch.py',
                  # scan_band 0.15: Nav2's 2D scan band must fit inside the gap the altitude planner flies a window
                  # opening through (0.6 m as mapped, drone 0.3 m: at 0.3 MPPI saw the window's solid part, 2026-10-21)
                  depth='true', vio_tf='false', scan_band='0.15', vslam=vslam)

    # Arducam camera_info + TF (the image itself comes over the host bridge)
    arducam = include('vio_bringup', 'sim_arducam.launch.py',
                       view='false', use_sim_time='true')

    # Corridor navigation. qr_fallback: Gazebo's fisheye render is marginal for the start QR, read it on the
    # nadir_hd stand-in too
    corridor = include('aerothon_nav', 'corridor_nav.launch.py',
                       sim='true', course=LaunchConfiguration('course'), qr_fallback='/arducam/nadir_hd/image_raw',
                       nav2=LaunchConfiguration('corridor_nav2'),
                       world_json=world_json_path,   # geofence (offboard_bridge)
                       use_sim_time='true')

    # 4️⃣ Delivery mapper & QR locator
    delivery = include('aerothon_nav', 'delivery.launch.py',
                       sim='true',
                       world_json=world_json_path,
                       nadir_topic='/arducam/nadir_hd/image_raw',
                       # 22 deg = 4.0 m round the drone at the 10 m search height: the sim nadir_hd (60 deg, 960 px)
                       # holds a whole 1.5 m QR to ~26 deg off nadir. (At 2.4 m, 10 ft: 45 with the fisheye; at 10 m
                       # with the old 40 deg / 640 px nadir_hd: 14, 18 missed qr_3 at the edge, 2026-10-20)
                       read_angle='18.0',   # v3: 1 m borderless QRs need >= ~70 px/m (offline test 2026-10-24)
                       use_sim_time='true')

    delivery_nav = include('aerothon_nav', 'delivery_nav.launch.py',
                          sim='true', gain_radius='3.25', use_sim_time='true',  # = read_angle footprint at 10 m
                          world_json=world_json_path,                          # the zone, as delivery.launch.py
                          # wall-time limit per frontier goal: 40 s covered only ~12 m at ~0.2x real time; 300 on the
                          # slow sim (a loaded Jetson ran ~0.02-0.2x)
                          goal_timeout=PythonExpression(["'300.0' if '", LaunchConfiguration('course'),
                                                         "' == 'v2_slowsim' else '40.0'"]))

    # 5️⃣ Mission supervisor
    mission = include('aerothon_nav', 'mission.launch.py', sim='true', use_sim_time='true', world_json=world_json_path)

    return LaunchDescription([
        DeclareLaunchArgument('world_json', default_value='/workspaces/isaac_ros-dev/sim/worlds/competition.json',
                              description="copy of the host world's .json (e.g. .../comp_far1.json)"),
        DeclareLaunchArgument('course', default_value='v2',
                              description='config/course_<course>.yaml: v2 | v2_slowsim (slow Jetson sim timings)'),
        # default false: navigation runs on PX4's estimate + 2D SLAM; cuVSLAM + vio_to_px4 only cost RAM / GPU here
        DeclareLaunchArgument('vslam', default_value='false',
                              description='true: cuVSLAM + vio_to_px4 (heavy on the 8 GB Jetson)'),
        DeclareLaunchArgument('corridor_nav2', default_value='false',
                              description='true: launch the corridor Nav2 (only for corridor_nav nav_mode nav2)'),
        set_fastdds,
        sim_view,
        vio,
        arducam,
        corridor,
        delivery,
        delivery_nav,
        mission,
    ])
