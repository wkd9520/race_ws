from setuptools import find_packages, setup


package_name = 'physicar_camera_tf_correction'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='PhysiCar Participant',
    maintainer_email='maintainer@example.com',
    description='Publishes a parallel corrected camera tilt TF branch for PhysiCar.',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_corrected_tf_broadcaster = '
            'physicar_camera_tf_correction.corrected_tf_broadcaster:main',
        ],
    },
)
