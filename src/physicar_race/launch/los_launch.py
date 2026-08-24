"""IPM(버드아이) + LOS 가이던스 주행.

노드 하나만 띄운다. 인지와 제어가 한 노드에 있는 이유는, 둘 사이에 오갈
값이 '중심선 한 줄'뿐이라 토픽으로 쪼갤 이득이 없기 때문이다.

    ros2 launch physicar_race los_launch.py
    ros2 launch physicar_race los_launch.py publish_debug:=true

`ros2 run` 으로 직접 띄우면 image_raw 리맵을 손으로 줘야 한다. 안 주면
아무 데이터도 안 들어오는데 에러도 안 난다. 이 launch 는 그걸 막아준다.

다른 주행 스택(race_launch.py, bev_launch.py)과 **동시에 띄우면 안 된다.**
셋 다 /speed 를 발행하므로 명령이 충돌한다.

━━━ 처음 붙일 때 순서 ━━━

1. publish_debug:=true 로 띄우고 los/debug_image 를 본다.
   직선 구간에서 **두 흰선이 평행한 세로선**이 되도록 src_* 를 맞춘다.
   이게 이 노드에서 유일하게 실측이 필요한 값이다.

       ros2 run rqt_image_view rqt_image_view

   빨강 = 흰선으로 잡힌 곳, 초록 점 = 통로 중심선, 보라 = LOS 점.

2. 차가 반대로 꺾이면 steer_sign:=-1.0

3. track_width_m 을 실제 트랙 폭(m)으로. 한쪽 경계가 화면 밖일 때
   중심을 어디로 잡을지가 이 값에 달려 있다.

4. 코너가 약하면 ld_k 를 올린다(멀리 봄). 흔들리면 내린다.
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

        # --- 사다리꼴: 실측으로 맞출 것 (위 주석 1번) ---
        DeclareLaunchArgument('src_top_y', default_value='0.58'),
        DeclareLaunchArgument('src_top_half', default_value='0.16'),
        DeclareLaunchArgument('src_bot_y', default_value='1.00'),
        DeclareLaunchArgument('src_bot_half', default_value='0.70'),
        DeclareLaunchArgument('src_center', default_value='0.50'),

        # --- BEV 가 덮는 실제 크기 ---
        DeclareLaunchArgument('bev_near_m', default_value='0.30'),
        DeclareLaunchArgument('bev_range_m', default_value='2.00'),
        DeclareLaunchArgument('bev_width_m', default_value='1.80'),
        DeclareLaunchArgument('track_width_m', default_value='1.00'),

        # --- 전방주시거리: 코너 성능을 좌우하는 값 ---
        DeclareLaunchArgument('ld_min_m', default_value='0.55'),
        DeclareLaunchArgument('ld_max_m', default_value='1.30'),
        DeclareLaunchArgument('ld_k', default_value='0.90'),
        DeclareLaunchArgument('steer_sign', default_value='1.0'),

        # --- 속도 ---
        DeclareLaunchArgument('v_max', default_value='1.60'),
        DeclareLaunchArgument('v_min', default_value='0.45'),
        DeclareLaunchArgument('a_lat_max', default_value='3.0'),

        # --- 흰선 임계 (hsv_tuner_node 로 맞출 것) ---
        DeclareLaunchArgument('white_s_max', default_value='60'),
        DeclareLaunchArgument('white_v_min', default_value='180'),
    ]

    drive = Node(
        package=PKG, executable='los_drive_node', name='los_drive_node',
        output='screen',
        parameters=[{
            'publish_debug': _b('publish_debug'),
            'src_top_y': _f('src_top_y'),
            'src_top_half': _f('src_top_half'),
            'src_bot_y': _f('src_bot_y'),
            'src_bot_half': _f('src_bot_half'),
            'src_center': _f('src_center'),
            'bev_near_m': _f('bev_near_m'),
            'bev_range_m': _f('bev_range_m'),
            'bev_width_m': _f('bev_width_m'),
            'track_width_m': _f('track_width_m'),
            'ld_min_m': _f('ld_min_m'),
            'ld_max_m': _f('ld_max_m'),
            'ld_k': _f('ld_k'),
            'steer_sign': _f('steer_sign'),
            'v_max': _f('v_max'),
            'v_min': _f('v_min'),
            'a_lat_max': _f('a_lat_max'),
            'white_s_max': ParameterValue(
                LaunchConfiguration('white_s_max'), value_type=int),
            'white_v_min': ParameterValue(
                LaunchConfiguration('white_v_min'), value_type=int),
        }],
        remappings=[('image_raw', LaunchConfiguration('image_topic'))],
    )

    return LaunchDescription(args + [drive])
