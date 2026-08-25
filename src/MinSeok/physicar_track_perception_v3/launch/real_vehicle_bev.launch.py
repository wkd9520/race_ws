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
    camera_topic = LaunchConfiguration('camera_topic')
    joint_states_topic = LaunchConfiguration('joint_states_topic')
    scan_topic = LaunchConfiguration('scan_topic')
    tilt_topic = LaunchConfiguration('tilt_topic')
    bev_topic = LaunchConfiguration('bev_topic')
    tilt_degrees = LaunchConfiguration('tilt_degrees')
    tilt_publish_rate_hz = LaunchConfiguration('tilt_publish_rate_hz')
    pitch_offset_deg = LaunchConfiguration('pitch_offset_deg')
    camera_height_correction_z = LaunchConfiguration(
        'camera_height_correction_z')

    return LaunchDescription([
        DeclareLaunchArgument(
            'params_file',
            default_value=default_params,
            description='Perception V3 parameter file.',
        ),
        DeclareLaunchArgument(
            'camera_topic',
            default_value='/camera/image_raw',
            description='sensor_msgs/msg/Image input.',
        ),
        DeclareLaunchArgument(
            'joint_states_topic',
            default_value='/joint_states',
            description='JointState input used for pan/tilt state and corrected TF.',
        ),
        DeclareLaunchArgument(
            'scan_topic',
            default_value='/scan',
            description='LaserScan input used by the existing V3 pipeline.',
        ),
        DeclareLaunchArgument(
            'tilt_topic',
            default_value='/camera/tilt',
            description='Float64 camera tilt command output.',
        ),
        DeclareLaunchArgument(
            'bev_topic',
            default_value='/perception_v3/debug/bev',
            description='sensor_msgs/msg/Image metric-BEV output.',
        ),
        DeclareLaunchArgument(
            'tilt_degrees',
            default_value='-30.0',
            description='Physical camera tilt hold command in degrees.',
        ),
        DeclareLaunchArgument(
            'tilt_publish_rate_hz',
            default_value='10.0',
            description='Camera tilt hold command rate.',
        ),
        DeclareLaunchArgument(
            'pitch_offset_deg',
            default_value='2.8',
            description=(
                'Additional BEV projection pitch correction in degrees; '
                'this does not change the physical camera tilt command.'
            ),
        ),
        DeclareLaunchArgument(
            'camera_height_correction_z',
            default_value='-0.018',
            description=(
                'Vehicle-Z camera-origin correction in metres. The existing '
                'validation launch preserves the configured -0.018 m value.'
            ),
        ),
        Node(
            package='physicar_track_perception_v3',
            executable='camera_tilt_publisher',
            name='perception_v3_camera_tilt_publisher',
            output='screen',
            parameters=[{
                'use_sim_time': False,
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
            parameters=[{'use_sim_time': False}],
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
                    'use_sim_time': False,
                    'projection.pitch_offset_deg': ParameterValue(
                        pitch_offset_deg, value_type=float),
                    'sim_geometry.camera_height_correction_z': ParameterValue(
                        camera_height_correction_z, value_type=float),
                    'lidar.scan_topic': scan_topic,
                },
            ],
            remappings=[
                ('/camera/image_raw', camera_topic),
                ('/joint_states', joint_states_topic),
                ('/perception_v3/debug/bev', bev_topic),
            ],
        ),
    ])
