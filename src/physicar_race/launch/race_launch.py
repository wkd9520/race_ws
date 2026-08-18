"""2026-08-18 코스 스펙용 레이스 스택 (인지 + 판단만).

센서 드라이버(카메라/라이다)와 차량 드라이버 노드는 시뮬레이터/실차가
시스템 서비스로 이미 띄운다. 여기서는 인지/판단 노드만 올린다.
센서를 직접 띄우면 이미 돌고 있는 것과 충돌한다.

실행:
    ros2 launch physicar_race race_launch.py
    ros2 launch physicar_race race_launch.py require_green:=false v_max:=2.0

토픽 이름이 다르면 (ros2 topic list -t 로 먼저 확인):
    ros2 launch physicar_race race_launch.py \\
        image_topic:=/image_raw scan_topic:=/scan
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PKG = 'physicar_race'


def _bool(name):
    """launch 인자는 문자열로 도착한다. bool 파라미터에 그대로 넘기면 타입 에러."""
    return ParameterValue(LaunchConfiguration(name), value_type=bool)


def _float(name):
    return ParameterValue(LaunchConfiguration(name), value_type=float)


def _int(name):
    return ParameterValue(LaunchConfiguration(name), value_type=int)


def generate_launch_description():
    args = [
        # 실차/시뮬레이터는 보통 /camera/image_raw, usb_cam 연습 환경은 /image_raw.
        DeclareLaunchArgument('image_topic', default_value='/camera/image_raw'),
        DeclareLaunchArgument('scan_topic', default_value='/scan'),

        # 신호등이 없는 맵이면 false. 안 그러면 초록을 못 봐서 영원히 출발하지 않는다.
        DeclareLaunchArgument('require_green', default_value='true'),

        # 8/18 공개된 코스 규격의 실제 차선 폭으로 교체할 것. 기본값은 추정치다.
        DeclareLaunchArgument('lane_width_m', default_value='0.50'),

        # 라이다 정면 기준각. 시뮬레이터는 보통 0, 연습 섀시 실측값은 180.
        DeclareLaunchArgument('front_offset_deg', default_value='0.0'),

        # 드라이버 상한은 3.0. 차선 유지가 안정되면 단계적으로 올린다.
        DeclareLaunchArgument('v_max', default_value='1.5'),

        # HSV 튜닝 중에는 켜두면 훨씬 빠르다. 기록 측정 시에는 꺼서 부하를 줄인다.
        DeclareLaunchArgument('publish_debug', default_value='false'),

        # --- 차선 검출 튜닝 ---
        # 기본값은 placeholder다. 안 잡히면 debug_probe:=true 로 실측 H/S/V 를
        # 로그에 찍어보고 그 값으로 채운다.
        DeclareLaunchArgument('lane_roi_top_frac', default_value='0.55'),
        DeclareLaunchArgument('lane_white_s_max', default_value='60'),
        DeclareLaunchArgument('lane_white_v_min', default_value='180'),
        DeclareLaunchArgument('lane_yellow_h_min', default_value='18'),
        DeclareLaunchArgument('lane_yellow_h_max', default_value='38'),
        DeclareLaunchArgument('lane_yellow_s_min', default_value='90'),
        DeclareLaunchArgument('lane_yellow_v_min', default_value='90'),
        DeclareLaunchArgument('lane_min_peak_px', default_value='8'),

        # --- 신호등 검출 튜닝 ---
        # 기본값은 placeholder다. 검출이 안 되면 debug_probe:=true 로 실제 H/S/V 를
        # 로그에 찍어보고 그 값으로 아래를 채우면 된다.
        DeclareLaunchArgument('debug_probe', default_value='false'),
        # probe 가 후보를 찾을 때 쓰는 느슨한 기준. 화면이 어두우면 낮춘다.
        DeclareLaunchArgument('probe_s_min', default_value='40'),
        DeclareLaunchArgument('probe_v_min_relaxed', default_value='60'),
        DeclareLaunchArgument('probe_top_n', default_value='5'),
        # 신호등이 화면에서 차지하는 세로 구간. 1.0 이면 화면 전체를 본다.
        DeclareLaunchArgument('tl_roi_top_frac', default_value='0.0'),
        DeclareLaunchArgument('tl_roi_bottom_frac', default_value='0.55'),
        # LED 는 가운데가 하얗게 떠서 채도가 낮게 나오는 일이 잦다.
        DeclareLaunchArgument('tl_sat_min', default_value='120'),
        DeclareLaunchArgument('tl_val_min', default_value='120'),
        DeclareLaunchArgument('tl_min_blob_px', default_value='60'),
        DeclareLaunchArgument('tl_green_h_min', default_value='40'),
        DeclareLaunchArgument('tl_green_h_max', default_value='90'),
        DeclareLaunchArgument('tl_red_h_lo_max', default_value='10'),
        DeclareLaunchArgument('tl_red_h_hi_min', default_value='170'),
    ]

    image_topic = LaunchConfiguration('image_topic')
    scan_topic = LaunchConfiguration('scan_topic')

    lane_detect = Node(
        package=PKG, executable='lane_detect_node', name='lane_detect_node',
        output='screen',
        parameters=[{
            'publish_debug': _bool('publish_debug'),
            'debug_probe': _bool('debug_probe'),
            'roi_top_frac': _float('lane_roi_top_frac'),
            'white_s_max': _int('lane_white_s_max'),
            'white_v_min': _int('lane_white_v_min'),
            'yellow_h_min': _int('lane_yellow_h_min'),
            'yellow_h_max': _int('lane_yellow_h_max'),
            'yellow_s_min': _int('lane_yellow_s_min'),
            'yellow_v_min': _int('lane_yellow_v_min'),
            'min_peak_px': _int('lane_min_peak_px'),
        }],
        remappings=[('image_raw', image_topic)],
    )

    traffic_light = Node(
        package=PKG, executable='traffic_light_node', name='traffic_light_node',
        output='screen',
        parameters=[{
            'publish_debug': _bool('publish_debug'),
            'debug_probe': _bool('debug_probe'),
            'probe_s_min': _int('probe_s_min'),
            'probe_v_min_relaxed': _int('probe_v_min_relaxed'),
            'probe_top_n': _int('probe_top_n'),
            'roi_top_frac': _float('tl_roi_top_frac'),
            'roi_bottom_frac': _float('tl_roi_bottom_frac'),
            'sat_min': _int('tl_sat_min'),
            'val_min': _int('tl_val_min'),
            'min_blob_px': _int('tl_min_blob_px'),
            'green_h_min': _int('tl_green_h_min'),
            'green_h_max': _int('tl_green_h_max'),
            'red_h_lo_max': _int('tl_red_h_lo_max'),
            'red_h_hi_min': _int('tl_red_h_hi_min'),
        }],
        remappings=[('image_raw', image_topic)],
    )

    obstacle = Node(
        package=PKG, executable='lane_obstacle_node', name='lane_obstacle_node',
        output='screen',
        parameters=[{
            'lane_width_m': _float('lane_width_m'),
            'front_offset_deg': _float('front_offset_deg'),
        }],
        remappings=[('scan', scan_topic)],
    )

    judgment = Node(
        package=PKG, executable='race_judgment_node', name='race_judgment_node',
        output='screen',
        parameters=[{
            'require_green': _bool('require_green'),
            'v_max': _float('v_max'),
        }],
    )

    return LaunchDescription(args + [lane_detect, traffic_light, obstacle, judgment])
