from glob import glob

from setuptools import setup

package_name = 'vio_bringup'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/rviz', glob('rviz/*.rviz')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'imu_rate_setter = vio_bringup.imu_rate_setter:main',
            'imu_monitor = vio_bringup.imu_monitor:main',
            'imu_tf_broadcaster = vio_bringup.imu_tf_broadcaster:main',
            'image_monitor = vio_bringup.image_monitor:main',
            'imu_relay = vio_bringup.imu_relay:main',
            'guided_test = vio_bringup.guided_test:main',
            'obstacle_scan = vio_bringup.obstacle_scan:main',
            'sim_teleop = vio_bringup.sim_teleop:main',
            'vio_to_px4 = vio_bringup.vio_to_px4:main',
            'fisheye_camera_info = vio_bringup.fisheye_camera_info:main',
            'arducam_csi = vio_bringup.arducam_csi:main',
        ],
    },
)
