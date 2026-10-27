"""Load config/rig.yaml (camera-IMU extrinsics, time offset, IMU noise) for launch files."""
import os

import yaml
from ament_index_python.packages import get_package_share_directory
from launch_ros.actions import Node


def default_rig_path():
    return os.path.join(get_package_share_directory('vio_bringup'), 'config', 'rig.yaml')


def load_rig(path=None):
    with open(path or default_rig_path()) as f:
        return yaml.safe_load(f)


def imu_to_camera_tf_node(rig):
    """Static TF imu_link -> camera_link from the rig file."""
    t = rig['imu_to_camera']['translation']
    q = rig['imu_to_camera']['rotation_xyzw']
    return Node(
        package='tf2_ros', executable='static_transform_publisher',
        name='imu_to_camera_tf',
        arguments=['--x', str(t[0]), '--y', str(t[1]), '--z', str(t[2]),
                   '--qx', str(q[0]), '--qy', str(q[1]), '--qz', str(q[2]), '--qw', str(q[3]),
                   '--frame-id', 'imu_link', '--child-frame-id', 'camera_link'])


def imu_relay_params(rig):
    n = rig['imu_noise']
    return {
        'time_offset': float(rig['time_offset']),
        'accel_noise_density': float(n['accel_noise_density']),
        'gyro_noise_density': float(n['gyro_noise_density']),
        'rate_hz': float(n['rate_hz']),
    }


def cuvslam_imu_params(rig):
    n = rig['imu_noise']
    return {
        'gyro_noise_density': float(n['gyro_noise_density']),
        'gyro_random_walk': float(n['gyro_random_walk']),
        'accel_noise_density': float(n['accel_noise_density']),
        'accel_random_walk': float(n['accel_random_walk']),
        'calibration_frequency': float(n['rate_hz']),
    }
