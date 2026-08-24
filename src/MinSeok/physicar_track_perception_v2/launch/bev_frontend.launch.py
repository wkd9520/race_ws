import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    share = get_package_share_directory('physicar_track_perception_v2')
    params = os.path.join(share, 'config', 'bev_frontend.yaml')
    return LaunchDescription([
        Node(
            package='physicar_track_perception_v2',
            executable='bev_frontend_node',
            name='physicar_track_perception_v2_bev',
            output='screen',
            parameters=[params, {'use_sim_time': True}],
        ),
    ])
