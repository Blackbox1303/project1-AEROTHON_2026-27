from glob import glob

from setuptools import setup

package_name = 'aerothon_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/behavior_trees', glob('behavior_trees/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'offboard_bridge = aerothon_nav.offboard_bridge:main',
            'odom_tf = aerothon_nav.odom_tf:main',
            'corridor_nav = aerothon_nav.corridor_nav:main',
            'banner_detector = aerothon_nav.banner_detector:main',
            'entry_align = aerothon_nav.entry_align:main',
            'qr_reader = aerothon_nav.qr_reader:main',
            'altitude_planner = aerothon_nav.altitude_planner:main',
            'delivery_mapper = aerothon_nav.delivery_mapper:main',
            'qr_locator = aerothon_nav.qr_locator:main',
            'frontier_explorer = aerothon_nav.frontier_explorer:main',
            'mission_supervisor = aerothon_nav.mission_supervisor:main',
        ],
    },
)
