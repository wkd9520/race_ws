import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    default_params = os.path.join(
        get_package_share_directory('physicar_track_perception_v3'),
        'config',
        'perception_v3.yaml',
    )

    params_file = LaunchConfiguration('params_file')
    use_sim_time = LaunchConfiguration('use_sim_time')
    camera_topic = LaunchConfiguration('camera_topic')
    joint_states_topic = LaunchConfiguration('joint_states_topic')
    scan_topic = LaunchConfiguration('scan_topic')
    tilt_topic = LaunchConfiguration('tilt_topic')
    tilt_degrees = LaunchConfiguration('tilt_degrees')
    tilt_publish_rate_hz = LaunchConfiguration('tilt_publish_rate_hz')

    return LaunchDescription([
        DeclareLaunchArgument('params_file', default_value=default_params),
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('joint_states_topic', default_value='/joint_states'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),
        DeclareLaunchArgument('tilt_topic', default_value='/camera/tilt'),
        DeclareLaunchArgument('tilt_degrees', default_value='-30.0'),
        DeclareLaunchArgument('tilt_publish_rate_hz', default_value='10.0'),
        Node(
            package='physicar_track_perception_v3',
            executable='camera_tilt_publisher',
            name='perception_v3_camera_tilt_publisher',
            output='screen',
            parameters=[{
                'tilt_degrees': ParameterValue(
                    tilt_degrees, value_type=float),
                'publish_rate_hz': ParameterValue(
                    tilt_publish_rate_hz, value_type=float),
            }],
            remappings=[('/camera/tilt', tilt_topic)],
        ),
        Node(
            package='physicar_camera_tf_correction',
            executable='camera_corrected_tf_broadcaster',
            name='camera_corrected_tf_broadcaster',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time}],
            remappings=[('/joint_states', joint_states_topic)],
        ),
        Node(
            package='physicar_track_perception_v3',
            executable='bev_frontend_node',
            name='physicar_track_perception_v3',
            output='screen',
            parameters=[
                params_file,
                {
                    'use_sim_time': use_sim_time,
                    'lidar.scan_topic': scan_topic,
                },
            ],
            remappings=[
                ('/camera/image_raw', camera_topic),
                ('/joint_states', joint_states_topic),
            ],
        ),
    ])
