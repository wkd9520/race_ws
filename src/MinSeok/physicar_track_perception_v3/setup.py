from glob import glob
import os

from setuptools import find_packages, setup


package_name = 'physicar_track_perception_v3'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'docs'), glob('docs/*.md')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='MinSeok',
    maintainer_email='maintainer@todo.invalid',
    description='PhysiCar V3 direct-center metric-BEV perception.',
    license='Apache-2.0',
    entry_points={'console_scripts': [
        'bev_frontend_node = physicar_track_perception_v3.bev_frontend_node:main',
    ]},
)
