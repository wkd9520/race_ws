import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    original_launch = os.path.join(
        get_package_share_directory('physicar_track_perception_v3'),
        'launch',
        'real_vehicle_bev.launch.py',
    )
    camera_height_correction_z = LaunchConfiguration(
        'camera_height_correction_z')

    return LaunchDescription([
        DeclareLaunchArgument(
            'camera_height_correction_z',
            default_value='0.0',
            description=(
                'Vehicle-Z camera-origin correction in metres. This launch '
                'assumes no camera-origin height error.'
            ),
        ),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(original_launch),
            launch_arguments={
                # PROPOSED ASSUMPTION: no camera-origin height error exists.
                # All other arguments and nodes come from the original launch.
                'camera_height_correction_z': camera_height_correction_z,
            }.items(),
        ),
    ])
