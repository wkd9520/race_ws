import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'physicar_race'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ANSL',
    maintainer_email='wkd9520@gmail.com',
    description='2026 AMET 2차선 코스용 인지/판단 스택',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'perception_v3_follow_node = physicar_race.perception_v3_follow_node:main',
            'cone_bev_node = physicar_race.cone_bev_node:main',
            'race_overlay_node = physicar_race.race_overlay_node:main',
            'hsv_tuner_node = physicar_race.hsv_tuner_node:main',
        ],
    },
)
