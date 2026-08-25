"""MinSeok 님의 perception_v3 인지 + 우리 순수추종 컨트롤러.

`physicar_track_perception_v3/launch/perception_v3.launch.py` 를 **수정 없이
그대로** include 하고, 그 뒤에 `perception_v3_follow_node` 하나만 붙인다.
인지 쪽(TF correction, V2 metric-BEV, V3 경로 추출)은 원본 그대로다 --
"완전히 동일하게" 적용하는 것이 이 launch 의 목적이다.

    ros2 launch physicar_race perception_v3_race_launch.py

디버그 시각화는 기본이 꺼져 있다. 켜려면:

    debug_view:=true

그러면 race_overlay_node(주행선 합성)와 cone_bev_node 의 고깔 화면이
발행되고, 5초 뒤 rqt_image_view 가 /race/debug/path_overlay 를 띄운다.

실차에서는 꺼두는 게 맞다 -- 이미지를 만들고 발행하는 CPU 가 아깝고
헤드리스면 볼 수도 없다.

    참고: MinSeok 님 노드는 디버그 이미지를 항상 발행한다(끄는 스위치가
    없다). 그건 우리가 못 끈다 -- 그의 코드는 안 건드리기 때문.

BEV 격자와 투영 보정은 launch 인자다. 눈으로 보며 맞출 값들이라 yaml 을
고치지 않고 바로 바꿀 수 있게 뺐다. 세 노드가 같은 인자에서 받으므로
한 곳만 고치면 전부 따라간다:

    ros2 launch physicar_race perception_v3_race_launch.py       bev_x_max:=1.4 pitch_offset_deg:=1.5

━━━ 실차 설정을 쓴다 ━━━

perception_v3_real.yaml 을 읽는다. 시뮬용과 카메라 내부 파라미터가 다르다:

    시뮬  fx=fy=201.4  cx=240  cy=180   D 있음   (화각 100도)
    실차  fx=fy=260.9  cx=231  cy=169   D 전부 0 (드라이버가 이미 보정)

시뮬 값으로 실차 영상을 펴면 BEV 가 통째로 틀어진다. 실제로 겪었다.

━━━ LiDAR 회피는 꺼둔다 ━━━

새 버전에는 LiDAR 원형 장애물 회피가 들어 있다. 우리는 초록 고깔을
카메라로 보고 피하므로(cone_bev_node) lidar_avoidance:=false 로 끈다.
스위치 다섯 개가 이 인자 하나로 같이 움직인다.

center_hybrid / center_history 는 중앙선 경로 품질을 올리는 것이라
회피와 무관하다 -- 켠 채로 둔다.

━━━ 틸트 ━━━

서보가 처지면 /joint_states 는 명령값을 보고하는데 실제 각도는 달라진다.
그러면 TF 가 거짓이 되고 BEV 가 틀어진다. 실차 라이브에서는:

    hold_tilt:=true

로스백 재생 때는 필요 없다(각도가 이미 기록돼 있다).

/speed 를 발행하는 노드는 항상 하나여야 한다. 이전 launch 가 안 죽었으면
명령이 번갈아 들어가 주행이 망가진다:

    ros2 topic info /speed --verbose | grep -c "Node name"   # 2 이상이면 충돌

━━━ 먼저 확인할 것 ━━━

이 인지 스택은 특정 TF 트리와 토픽을 전제한다(INSTALL_KO.md 1절):
`/camera/image_raw`, `/joint_states`(camera_tilt_joint 포함), `/scan`,
`/clock`, 그리고 `odom -> base_footprint`, `base_footprint <->
camera_optical_frame_corrected` TF. 이 저장소의 이전 스택들은 TF를 전혀 안
썼으므로, 이 시뮬레이터가 그 트리를 실제로 주는지 **띄우기 전에 반드시
확인**해야 한다.

    ros2 run rclpy 대신 --
    bash scripts/preflight_runtime.sh

전부 [PASS] 가 아니면(특히 TF 두 줄) 이 launch 를 시도하기 전에 원인부터
잡을 것 -- 인지 자체가 조용히 아무것도 못 만든다.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import PythonExpression
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from typing import List
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

        # --- BEV 격자: 세 노드가 이 값을 같이 쓴다 ---
        # 라즈베리파이 5 에서 인지가 카메라를 못 따라가서 줄였다. 인지
        # 비용은 BEV 픽셀 수에 거의 비례한다(연결요소마다 픽셀 BFS).
        #
        #   실차 yaml  0.01, x 0.1~2.0, y ±0.75 -> 150 x 190 = 28,500
        #   지금       0.01, x 0.2~1.2, y ±0.70 -> 100 x 140 = 14,000
        #
        # 절반이다. **범위만 줄이고 해상도(0.01)는 손대지 않는다.**
        #
        # 0.02 로 올렸다가 조향이 통째로 죽었다. 이유는 마스킹 순서다 --
        # bev_frontend_node.py:1452 에서 색 마스킹이 BEV 를 만든 *뒤에*
        # 돈다. 흰선 폭이 2~3 cm 라 0.02 m/px 면 BEV 에서 1~1.5 픽셀이고,
        # 투영 보간에서 아스팔트와 섞여 흰색 임계값을 못 넘는다. 선이
        # 사라지니 경로도 조향도 없다. los_drive_node 때 똑같이 물렸던
        # 함정이다(939 -> 3117 px). 해상도는 검출 한계이지 표현 정밀도가
        # 아니다. 다시 올리지 말 것.
        #
        # 거리 1.2 m 는 넉넉하다 -- 전방주시점이 최대 1.08 m 다.
        # y ±0.70 인 이유: track_half_m 0.37 + max_offset_m 0.30 = 0.67.
        # 고깔을 피해 붙는 순간 반대편 흰선(넘으면 실격)이 격자 밖으로
        # 나가면 안 된다.
        DeclareLaunchArgument('bev_x_min', default_value='0.20'),
        DeclareLaunchArgument('bev_x_max', default_value='1.20'),
        DeclareLaunchArgument('bev_y_min', default_value='-0.70'),
        DeclareLaunchArgument('bev_y_max', default_value='0.70'),
        DeclareLaunchArgument('bev_resolution', default_value='0.01'),

        # --- 투영 보정 ---
        # 실차 yaml 은 둘 다 0 이다("실물 URDF/TF 를 그대로 믿는다").
        # 그런데 우리 차는 실측 결과 +3.0 이 맞았다 -- 0 / 6 / 12.5 를
        # 비교해서 3.0 에서 BEV 가 가장 곧게 나왔다. 서보가 명령(-30도)보다
        # 약간 아래로 처져 있다는 뜻이다.
        # 카메라를 다시 장착하거나 서보를 바꾸면 재측정할 것.
        DeclareLaunchArgument('pitch_offset_deg', default_value='3.0'),
        DeclareLaunchArgument('camera_height_correction_z',
                              default_value='0.0'),

        # 카메라 내부 파라미터. camera_info 가 껍데기(전부 0)라 yaml 이
        # 유일한 진실이고, 이 값은 MinSeok 님 시뮬레이터 카메라 기준이다.
        #   [fx, 0, cx,  0, fy, cy,  0, 0, 1]
        # fx=201.4 는 수평 화각 100도라는 뜻 -- 실물 렌즈가 좁으면 더 커야 한다.
        # 바닥 체커보드가 BEV 에서 정사각형이 되는 값이 정답이다.
        # 실차 값이다(perception_v3_real.yaml). 시뮬 값(fx=201.4, 화각 100도)과
        # 다르다 -- 드라이버가 640x480 을 찍어 OpenCV 로 왜곡보정한 뒤 480x360
        # 으로 내보내기 때문에, cx/cy 도 정중앙이 아니고 D 는 전부 0 이다.
        DeclareLaunchArgument(
            'camera_k',
            default_value='[260.875, 0.0, 231.31516130651107,'
                           ' 0.0, 260.875, 169.16236121207476,'
                           ' 0.0, 0.0, 1.0]'),
        DeclareLaunchArgument(
            'camera_d', default_value='[0.0, 0.0, 0.0, 0.0, 0.0]'),

        # --- LiDAR 회피: 우리는 안 쓴다 ---
        # MinSeok 님 새 버전에는 LiDAR 원형 장애물 회피가 들어 있다.
        # 우리는 초록 고깔을 카메라로 보고 피하므로(cone_bev_node) 끈다.
        # center_hybrid / center_history 는 중앙선 경로 품질을 올리는
        # 것이라 회피와 무관하다 -- 켠 채로 둔다.
        DeclareLaunchArgument('lidar_avoidance', default_value='false'),

        # --- 카메라 틸트 유지 ---
        # 서보가 중력에 처지면 /joint_states 는 명령값을 보고하는데 실제
        # 각도는 달라진다. 그러면 TF 가 거짓이 되고 BEV 가 통째로 틀어진다.
        # 로스백 재생 때는 필요 없으니 기본은 꺼둔다.
        DeclareLaunchArgument('hold_tilt', default_value='false'),
        DeclareLaunchArgument('tilt_degrees', default_value='-30.0'),

        # --- 우리 컨트롤러 ---
        DeclareLaunchArgument('control_hz', default_value='30.0'),
        DeclareLaunchArgument('ld_min_m', default_value='0.35'),
        DeclareLaunchArgument('ld_max_m', default_value='0.90'),
        DeclareLaunchArgument('ld_k', default_value='0.90'),
        DeclareLaunchArgument('steer_sign', default_value='1.0'),
        DeclareLaunchArgument('v_max', default_value='1.20'),
        DeclareLaunchArgument('v_min', default_value='0.45'),
        DeclareLaunchArgument('a_lat_max', default_value='3.0'),
        DeclareLaunchArgument('k_vis', default_value='1.40'),

        # --- 초록 고깔 회피 ---
        DeclareLaunchArgument('avoid_enabled', default_value='true'),

        # --- 출발 신호등 ---
        # 코스 규정: 정지선 앞 신호등에 초록 원이 들어와야 출발할 수 있다.
        # 이 스위치 하나가 신호등 노드와 follow 노드의 대기를 **같이**
        # 켜고 끈다. 따로 두면 언젠가 한쪽만 켜지고, 그러면 신호등 노드가
        # 없는데 차가 영영 안 움직이거나(대기만 켬) 빨간불에 출발한다
        # (노드만 켬). 둘 다 대회에서 끝장이라 하나로 묶는다.
        DeclareLaunchArgument('traffic_light', default_value='true'),
        # 신호등은 화면 위쪽에 있다. 출발선에 세워놓고 debug_view 로 보면서 맞춘다.
        DeclareLaunchArgument('traffic_roi_bottom', default_value='0.55'),
        DeclareLaunchArgument('traffic_green_h_min', default_value='35'),
        DeclareLaunchArgument('traffic_green_h_max', default_value='95'),
        # 켜진 LED 는 가운데가 포화돼 하얗게 뜬다. 채도 기준이 높으면
        # 초록불인데 아무것도 못 본다. 색은 넉넉히 잡고 모양으로 거른다.
        DeclareLaunchArgument('traffic_sat_min', default_value='70'),
        DeclareLaunchArgument('traffic_val_min', default_value='90'),
        DeclareLaunchArgument('traffic_min_blob_px', default_value='60'),
        # 포화된 흰 중심과 초록 띠가 갈라지면 붙여준다.
        DeclareLaunchArgument('traffic_dilate_px', default_value='2'),
        # 초록 고깔이 초록불로 읽히는 것을 막는 모양 검사. 실측 근거는
        # traffic_light_node 의 주석 표에 있다.
        DeclareLaunchArgument('traffic_require_circle', default_value='true'),
        DeclareLaunchArgument('traffic_min_fill', default_value='0.72'),
        DeclareLaunchArgument('traffic_min_circularity', default_value='0.70'),
        DeclareLaunchArgument('traffic_min_eccentricity', default_value='0.55'),
        # 검출이 안 되면 켠다. 화면의 실측 H/S/V 를 로그로 뽑아준다.
        DeclareLaunchArgument('traffic_probe', default_value='false'),

        # --- 출발 카메라 자세 ---
        # 신호등은 정지선 오른쪽, 트랙은 아래. 카메라 하나로 둘 다 못 봐서
        # 순서대로 본다. 팬은 ROS 규약대로 왼쪽이 양수라 오른쪽은 음수다.
        # 서보가 반대로 돌면 부호만 뒤집으면 된다.
        DeclareLaunchArgument('aim_pan_degrees', default_value='-25.0'),
        DeclareLaunchArgument('aim_tilt_degrees', default_value='0.0'),
        # 카메라가 실제로 다 돌았는지 joint_states 로 확인하고 출발한다.
        # 명령을 보냈다고 카메라가 그 자리에 있는 게 아니다.
        DeclareLaunchArgument('settle_tolerance_deg', default_value='3.0'),
        DeclareLaunchArgument('turn_timeout_s', default_value='3.0'),
        DeclareLaunchArgument('green_h_min', default_value='40'),
        DeclareLaunchArgument('green_h_max', default_value='85'),
        DeclareLaunchArgument('green_s_min', default_value='80'),
        DeclareLaunchArgument('green_v_min', default_value='60'),
        DeclareLaunchArgument('cone_margin_m', default_value='0.12'),
        DeclareLaunchArgument('wall_margin_m', default_value='0.10'),
        DeclareLaunchArgument('max_offset_m', default_value='0.30'),
        DeclareLaunchArgument('track_half_m', default_value='0.37'),

        # --- 디버그 시각화 ---
        # 실차에서는 끈다. 오버레이 이미지를 만들고 발행하는 데 드는 CPU 가
        # 아깝고, 헤드리스라 볼 수도 없다. 기본을 꺼둔 이유가 그것이다.
        # 켜면 세 가지가 같이 켜진다:
        #   race_overlay_node (주행선 합성), cone_bev_node 의 고깔 화면,
        #   rqt_image_view
        DeclareLaunchArgument('debug_view', default_value='false'),
        DeclareLaunchArgument('rqt_topic',
                              default_value='/race/debug/path_overlay'),
        DeclareLaunchArgument('rqt_delay', default_value='5.0'),
    ]

    # MinSeok 님 perception_v3.launch.py 를 include 하지 않고 그의 노드 둘을
    # 직접 띄운다. include 로는 bev.* / projection.* 를 덮어쓸 수 없어서
    # 값을 바꿀 때마다 그의 yaml 을 손으로 고쳐야 하기 때문이다.
    # **코드는 여전히 한 줄도 안 건드린다** -- 실행 인자만 우리가 준다.
    # 노드 구성은 그의 launch 와 동일하게 유지한다(패키지/실행파일/이름/리맵).
    # 실차용 yaml 을 쓴다. 시뮬용(perception_v3.yaml)과 카메라 내부
    # 파라미터가 다르다 -- 그게 BEV 가 틀어지던 원인이었다.
    v3_params = os.path.join(
        get_package_share_directory('physicar_track_perception_v3'),
        'config', 'perception_v3_real.yaml')

    # 서보가 처지지 않게 틸트를 계속 잡아준다. 실차 라이브에서만 켠다.
    tilt_hold = Node(
        package='physicar_track_perception_v3',
        executable='camera_tilt_publisher',
        name='perception_v3_camera_tilt_publisher', output='screen',
        parameters=[{
            'use_sim_time': _b('use_sim_time'),
            'tilt_degrees': _f('tilt_degrees'),
        }],
        # traffic_light 를 켜면 start_sequence_node 가 틸트를 쥔다.
        # 둘이 같이 돌면 /camera/tilt 를 서로 다른 값으로 밀어 카메라가 떤다.
        condition=IfCondition(PythonExpression([
            "'", LaunchConfiguration('hold_tilt'), "' == 'true' and '",
            LaunchConfiguration('traffic_light'), "' != 'true'"])),
    )

    tf_broadcaster = Node(
        package='physicar_camera_tf_correction',
        executable='camera_corrected_tf_broadcaster',
        name='camera_corrected_tf_broadcaster', output='screen',
        parameters=[{'use_sim_time': _b('use_sim_time')}],
        remappings=[('/joint_states',
                     LaunchConfiguration('joint_states_topic'))],
    )

    # yaml 을 먼저 읽고 그 위에 우리 값을 덮는다(뒤에 오는 것이 이긴다).
    perception_v3 = Node(
        package='physicar_track_perception_v3', executable='bev_frontend_node',
        name='physicar_track_perception_v3', output='screen',
        parameters=[v3_params, {
            'use_sim_time': _b('use_sim_time'),
            'lidar.scan_topic': LaunchConfiguration('scan_topic'),
            'bev.x_min': _f('bev_x_min'),
            'bev.x_max': _f('bev_x_max'),
            'bev.y_min': _f('bev_y_min'),
            'bev.y_max': _f('bev_y_max'),
            'bev.resolution': _f('bev_resolution'),
            'projection.pitch_offset_deg': _f('pitch_offset_deg'),
            'sim_geometry.camera_height_correction_z':
                _f('camera_height_correction_z'),
            'camera.K': ParameterValue(LaunchConfiguration('camera_k'),
                                       value_type=List[float]),
            'camera.D': ParameterValue(LaunchConfiguration('camera_d'),
                                       value_type=List[float]),
            # LiDAR 회피 일괄 스위치. 우리는 카메라로 고깔을 피한다.
            'avoidance.shadow_enabled': _b('lidar_avoidance'),
            'avoidance_circle.enabled': _b('lidar_avoidance'),
            'obstacle_track.enabled': _b('lidar_avoidance'),
            'active_lifecycle.enabled': _b('lidar_avoidance'),
            'avoidance_recovery.enabled': _b('lidar_avoidance'),
        }],
        remappings=[('/camera/image_raw', LaunchConfiguration('camera_topic')),
                    ('/joint_states',
                     LaunchConfiguration('joint_states_topic'))],
    )

    # 회피를 안 쓰면 띄우지 않는다. 이미지 두 개를 받아 HSV + 연결요소를
    # 매 프레임 도는 노드라, 결과를 안 쓸 거면 CPU 만 먹는다.
    cones = Node(
        package=PKG, executable='cone_bev_node', name='cone_bev_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('avoid_enabled')),
        parameters=[{
            # perception_v3 와 같은 인자를 쓴다. 손으로 두 벌 적으면
            # 언젠가 어긋나고, 그러면 고깔 좌표가 통째로 틀어진다.
            'bev_x_min': _f('bev_x_min'), 'bev_x_max': _f('bev_x_max'),
            'bev_y_min': _f('bev_y_min'), 'bev_y_max': _f('bev_y_max'),
            'bev_resolution': _f('bev_resolution'),
            'green_h_min': _i('green_h_min'),
            'green_h_max': _i('green_h_max'),
            'green_s_min': _i('green_s_min'),
            'green_v_min': _i('green_v_min'),
            'publish_debug': _b('debug_view'),
        }],
    )

    # 출발 절차. 신호등 자세 -> 초록 -> 트랙 자세 -> 출발 허가.
    #
    # 이 노드가 팬과 틸트를 **둘 다** 쥔다. camera_tilt_publisher 와 같이
    # 돌면 둘이 /camera/tilt 를 서로 다른 값으로 밀어서 카메라가 떨린다.
    # 그래서 아래 tilt_hold 는 traffic_light 가 꺼졌을 때만 뜬다.
    start_sequence = Node(
        package=PKG, executable='start_sequence_node',
        name='start_sequence_node', output='screen',
        condition=IfCondition(LaunchConfiguration('traffic_light')),
        parameters=[{
            'aim_pan_degrees': _f('aim_pan_degrees'),
            'aim_tilt_degrees': _f('aim_tilt_degrees'),
            'drive_tilt_degrees': _f('tilt_degrees'),
            'settle_tolerance_deg': _f('settle_tolerance_deg'),
            'turn_timeout_s': _f('turn_timeout_s'),
        }],
    )

    # 출발 신호등. 지금 프레임에 무엇이 보이는지만 보고한다(무상태).
    # '초록을 봤으니 출발'이라는 래치는 follow 노드가 건다 -- 인지와
    # 상태를 갈라놔야 어느 쪽이 틀렸는지 알 수 있다.
    traffic = Node(
        package=PKG, executable='traffic_light_node', name='traffic_light_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('traffic_light')),
        parameters=[{
            'roi_bottom_frac': _f('traffic_roi_bottom'),
            'green_h_min': _i('traffic_green_h_min'),
            'green_h_max': _i('traffic_green_h_max'),
            'sat_min': _i('traffic_sat_min'),
            'val_min': _i('traffic_val_min'),
            'min_blob_px': _i('traffic_min_blob_px'),
            'dilate_px': _i('traffic_dilate_px'),
            'require_circle': _b('traffic_require_circle'),
            'min_enclosing_fill': _f('traffic_min_fill'),
            'min_circularity': _f('traffic_min_circularity'),
            'min_eccentricity': _f('traffic_min_eccentricity'),
            'debug_probe': _b('traffic_probe'),
            'publish_debug': _b('debug_view'),
        }],
        remappings=[('/camera/image_raw', LaunchConfiguration('camera_topic'))],
    )

    # 그리기만 하는 노드다. 안 볼 거면 띄울 이유가 없다.
    overlay = Node(
        package=PKG, executable='race_overlay_node', name='race_overlay_node',
        output='screen',
        condition=IfCondition(LaunchConfiguration('debug_view')),
        parameters=[{
            'bev_x_min': _f('bev_x_min'), 'bev_x_max': _f('bev_x_max'),
            'bev_y_min': _f('bev_y_min'), 'bev_y_max': _f('bev_y_max'),
            'bev_resolution': _f('bev_resolution'),
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
            'k_vis': _f('k_vis'),
            'avoid_enabled': _b('avoid_enabled'),
            # 신호등 노드와 반드시 같은 값. 위 traffic_light 주석 참고.
            'wait_for_green': _b('traffic_light'),
            'cone_margin_m': _f('cone_margin_m'),
            'wall_margin_m': _f('wall_margin_m'),
            'max_offset_m': _f('max_offset_m'),
            'track_half_m': _f('track_half_m'),
        }],
    )

    # rqt 는 늦게 띄운다. 토픽이 생기기 전에 열면 목록이 비어 있어서
    # 매번 새로고침을 눌러야 한다.
    rqt = TimerAction(
        period=LaunchConfiguration('rqt_delay'),
        actions=[Node(
            package='rqt_image_view', executable='rqt_image_view',
            name='rqt_image_view', output='screen',
            arguments=[LaunchConfiguration('rqt_topic')],
            condition=IfCondition(LaunchConfiguration('debug_view')),
        )],
    )

    return LaunchDescription(
        args + [tilt_hold, tf_broadcaster, perception_v3,
                cones, traffic, start_sequence, follow, overlay, rqt])
