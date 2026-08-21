"""IPM(bird's eye) 기반 주행 스택.

표준 파이프라인이다: 원근 변환 -> 색 임계 -> 슬라이딩 윈도우 -> 다항식 피팅.
화면 좌표에서 밴드 피크만 보던 기존 스택(race_launch.py)과 별개 경로이므로
동시에 띄우면 /speed 가 충돌한다. 하나만 쓸 것.

    ros2 launch physicar_race bev_launch.py
    ros2 launch physicar_race bev_launch.py publish_debug:=true

사다리꼴은 직선 구간에서 흰선을 보고 자동으로 잡는다(auto_calibrate).
20프레임 모아 확정하므로, 출발 직후 직선을 잠깐 보여주면 된다.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = 'physicar_race'


def _b(name):
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _f(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def generate_launch_description():
    args = [
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('publish_debug', default_value='false'),
        DeclareLaunchArgument('auto_calibrate', default_value='true'),

        # 중앙선을 BEV 폭 대비 어디에 둘지
        DeclareLaunchArgument('target_offset_frac', default_value='0.25'),

        # 제어 게인. 반대로 꺾이면 steer_sign 을 -1.0 으로.
        DeclareLaunchArgument('k_lat', default_value='0.7'),
        DeclareLaunchArgument('k_head', default_value='0.5'),
        DeclareLaunchArgument('steer_sign', default_value='1.0'),
        DeclareLaunchArgument('v_max', default_value='0.8'),
    ]

    lane = Node(
        package=PKG, executable='bev_lane_node', name='bev_lane_node',
        output='screen',
        parameters=[{
            'publish_debug': _b('publish_debug'),
            'auto_calibrate': _b('auto_calibrate'),
            'target_offset_frac': _f('target_offset_frac'),
        }],
        remappings=[('image_raw', LaunchConfiguration('image_topic'))],
    )

    drive = Node(
        package=PKG, executable='bev_drive_node', name='bev_drive_node',
        output='screen',
        parameters=[{
            'k_lat': _f('k_lat'),
            'k_head': _f('k_head'),
            'steer_sign': _f('steer_sign'),
            'v_max': _f('v_max'),
        }],
    )

    return LaunchDescription(args + [lane, drive])
