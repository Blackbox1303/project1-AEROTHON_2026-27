from glob import glob
from setuptools import setup

package_name = 'drone_qr_detector'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'CALIBRATION_AND_USAGE_GUIDE.md']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'csi_camera_node = drone_qr_detector.csi_camera_node:main',
            'qr_drone_locator = drone_qr_detector.qr_drone_locator:main',
            'calib_and_wall_eval = drone_qr_detector.calib_and_wall_eval:main',
            'rect_banner_view = drone_qr_detector.rect_banner_view:main',
            'banner_node = drone_qr_detector.banner_node:main',
        ],
    },
)
