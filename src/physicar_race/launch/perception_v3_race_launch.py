"""MinSeok 님의 perception_v3 인지 + 우리 순수추종 컨트롤러.

`physicar_track_perception_v3/launch/perception_v3.launch.py` 를 **수정 없이
그대로** include 하고, 그 뒤에 `perception_v3_follow_node` 하나만 붙인다.
인지 쪽(TF correction, V2 metric-BEV, V3 경로 추출)은 원본 그대로다 --
"완전히 동일하게" 적용하는 것이 이 launch 의 목적이다.

    ros2 launch physicar_race perception_v3_race_launch.py

같이 뜨는 것 (전부 인자로 끌 수 있다):

    카메라 틸트 -0.5236 rad 를 10Hz 로 계속 발행   publish_tilt:=false
    rqt_image_view 로 /race/debug/path_overlay     open_rqt:=false

헤드리스 환경이면 open_rqt:=false 로 꺼야 한다. 틸트를 시뮬레이터나 다른
노드가 이미 잡고 있으면 publish_tilt:=false 로 끄고 그쪽에 맡길 것 --
둘이 동시에 보내면 값이 번갈아 들어간다.

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
from launch.actions import (DeclareLaunchArgument, ExecuteProcess,
                            IncludeLaunchDescription, TimerAction)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory
import os

PKG = 'physicar_race'


def _b(name):
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _f(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _i(name):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


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

        # --- 초록 고깔 회피 ---
        DeclareLaunchArgument('avoid_enabled', default_value='true'),
        DeclareLaunchArgument('green_h_min', default_value='40'),
        DeclareLaunchArgument('green_h_max', default_value='85'),
        DeclareLaunchArgument('green_s_min', default_value='80'),
        DeclareLaunchArgument('green_v_min', default_value='60'),
        DeclareLaunchArgument('cone_margin_m', default_value='0.12'),
        DeclareLaunchArgument('wall_margin_m', default_value='0.10'),
        DeclareLaunchArgument('max_offset_m', default_value='0.45'),

        # --- 카메라 틸트 고정 ---
        # V2 요구사항에 tilt -0.5236 rad (-30도) 가 required 로 명시돼 있다.
        # 이 값이 아니면 IPM 이 그만큼 틀어져 BEV 가 왜곡된다.
        # 시뮬레이터가 값을 물고 있지 않을 수 있어 주기적으로 계속 보낸다.
        DeclareLaunchArgument('publish_tilt', default_value='true'),
        DeclareLaunchArgument('camera_tilt', default_value='-0.5235987756'),
        DeclareLaunchArgument('tilt_rate', default_value='10'),

        # --- 디버그 화면 ---
        DeclareLaunchArgument('open_rqt', default_value='true'),
        DeclareLaunchArgument('rqt_topic',
                              default_value='/race/debug/path_overlay'),
        DeclareLaunchArgument('rqt_delay', default_value='5.0'),
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

    cones = Node(
        package=PKG, executable='cone_bev_node', name='cone_bev_node',
        output='screen',
        parameters=[{
            # perception_v3.yaml 의 bev.* 와 반드시 같아야 한다.
            # 다르면 고깔 좌표가 통째로 틀어진다 -- 노드가 이미지 크기로
            # 교차 검증해서 다르면 에러를 찍는다.
            'bev_x_min': 0.10, 'bev_x_max': 2.00,
            'bev_y_min': -0.75, 'bev_y_max': 0.75,
            'bev_resolution': 0.01,
            'green_h_min': _i('green_h_min'),
            'green_h_max': _i('green_h_max'),
            'green_s_min': _i('green_s_min'),
            'green_v_min': _i('green_v_min'),
        }],
    )

    overlay = Node(
        package=PKG, executable='race_overlay_node', name='race_overlay_node',
        output='screen',
        parameters=[{
            'bev_x_min': 0.10, 'bev_x_max': 2.00,
            'bev_y_min': -0.75, 'bev_y_max': 0.75,
            'bev_resolution': 0.01,
        }],
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
            'avoid_enabled': _b('avoid_enabled'),
            'cone_margin_m': _f('cone_margin_m'),
            'wall_margin_m': _f('wall_margin_m'),
            'max_offset_m': _f('max_offset_m'),
        }],
    )

    tilt = ExecuteProcess(
        cmd=['ros2', 'topic', 'pub',
             '-r', LaunchConfiguration('tilt_rate'),
             '/camera/tilt', 'std_msgs/msg/Float64',
             ['{data: ', LaunchConfiguration('camera_tilt'), '}']],
        output='screen',
        condition=IfCondition(LaunchConfiguration('publish_tilt')),
    )

    # rqt 는 늦게 띄운다. 토픽이 생기기 전에 열면 목록이 비어 있어서
    # 매번 새로고침을 눌러야 한다.
    rqt = TimerAction(
        period=LaunchConfiguration('rqt_delay'),
        actions=[Node(
            package='rqt_image_view', executable='rqt_image_view',
            name='rqt_image_view', output='screen',
            arguments=[LaunchConfiguration('rqt_topic')],
            condition=IfCondition(LaunchConfiguration('open_rqt')),
        )],
    )

    return LaunchDescription(
        args + [perception_v3, cones, follow, overlay, tilt, rqt])
