"""MinSeok 님의 perception_v3 인지 + 우리 순수추종 컨트롤러.

`physicar_track_perception_v3/launch/perception_v3.launch.py` 를 **수정 없이
그대로** include 하고, 그 뒤에 `perception_v3_follow_node` 하나만 붙인다.
인지 쪽(TF correction, V2 metric-BEV, V3 경로 추출)은 원본 그대로다 --
"완전히 동일하게" 적용하는 것이 이 launch 의 목적이다.

    ros2 launch physicar_race perception_v3_race_launch.py

다른 주행 스택(race_launch.py, bev_launch.py, los_launch.py)과 **동시에
띄우면 안 된다.** 전부 /speed 를 발행한다.

━━━ 먼저 확인할 것 ━━━

이 인지 스택은 특정 TF 트리와 토픽을 전제한다(INSTALL_KO.md 1절):
`/camera/image_raw`, `/joint_states`(camera_tilt_joint 포함), `/scan`,
`/clock`, 그리고 `odom -> base_footprint`, `base_footprint <->
camera_optical_frame_corrected` TF. 우리 기존 노드들(centroid_follow_node,
los_drive_node)은 TF를 전혀 안 썼으므로, 이 시뮬레이터가 그 트리를 실제로
주는지 **띄우기 전에 반드시 확인**해야 한다.

    ros2 run rclpy 대신 --
    bash scripts/preflight_runtime.sh

전부 [PASS] 가 아니면(특히 TF 두 줄) 이 launch 를 시도하기 전에 원인부터
잡을 것 -- 인지 자체가 조용히 아무것도 못 만든다.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

PKG = 'physicar_race'


def _f(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description():
    args = [
        DeclareLaunchArgument('use_sim_time', default_value='true'),
        DeclareLaunchArgument('camera_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('joint_states_topic', default_value='/joint_states'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),

        # --- 우리 컨트롤러: los_drive_node 와 같은 물리, 같은 튜닝 이름 ---
        DeclareLaunchArgument('control_hz', default_value='30.0'),
        DeclareLaunchArgument('ld_min_m', default_value='0.35'),
        DeclareLaunchArgument('ld_max_m', default_value='1.30'),
        DeclareLaunchArgument('ld_k', default_value='0.90'),
        DeclareLaunchArgument('steer_sign', default_value='1.0'),
        DeclareLaunchArgument('v_max', default_value='1.20'),
        DeclareLaunchArgument('v_min', default_value='0.45'),
        DeclareLaunchArgument('a_lat_max', default_value='3.0'),
    ]

    perception_v3 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('physicar_track_perception_v3'),
            'launch', 'perception_v3.launch.py')),
        launch_arguments={
            'use_sim_time': LaunchConfiguration('use_sim_time'),
            'camera_topic': LaunchConfiguration('camera_topic'),
            'joint_states_topic': LaunchConfiguration('joint_states_topic'),
            'scan_topic': LaunchConfiguration('scan_topic'),
        }.items(),
    )

    follow = Node(
        package=PKG, executable='perception_v3_follow_node',
        name='perception_v3_follow_node', output='screen',
        parameters=[{
            'control_hz': _f('control_hz'),
            'ld_min_m': _f('ld_min_m'),
            'ld_max_m': _f('ld_max_m'),
            'ld_k': _f('ld_k'),
            'steer_sign': _f('steer_sign'),
            'v_max': _f('v_max'),
            'v_min': _f('v_min'),
            'a_lat_max': _f('a_lat_max'),
        }],
    )

    return LaunchDescription(args + [perception_v3, follow])
