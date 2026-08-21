#!/usr/bin/env python3
"""컨투어 중심 추종 - UCSD MAE/ECE 148 Mjolnir 구조를 그대로 옮긴 것.

참조: https://github.com/ArthurDassier/Mjolnir_kit
      scripts/lane_Detection.py + lane_guidance.py + class_PIDController.py

그쪽은 우리와 문제 정의가 같다(노란 선을 따라가며 두 흰선 안쪽 유지, 1/10 스케일,
ROS + OpenCV + 서보 조향). 실제로 트랙을 도는 코드이고, **내가 만들던 것보다
훨씬 단순하다.**

없는 것: IPM(원근 변환), 슬라이딩 윈도우, 다항식 피팅, 헤딩 추정, 곡률 계산,
밴드 추종, 차선 래치, 선행점.

있는 것 전부:
  1. ROI 잘라내기 (상하좌우 픽셀 단위)
  2. HSV 마스크
  3. findContours
  4. **폭 필터** (min_width < w < max_width)  <- 갓길·차체를 여기서 거른다
  5. 컨투어 중심(centroid) 계산
  6. 2개 이상이면 앞의 둘의 중점, 1개면 그것
  7. PID(centroid, width/2) -> 조향

핵심은 4번이다. 나는 갓길·차체 오검출을 ROI 절단과 탐색창 제한으로 막으려 했는데,
원본은 **컨투어의 폭**으로 거른다. 차선은 폭이 일정한 가늘고 긴 것이고,
갓길이나 차체는 그보다 훨씬 넓다. 색이 같아도 폭으로 갈린다.

목표는 화면 중앙(width/2)이다. 별도 target 개념이 없다.
"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Float64

MAX_STEER = 0.349      # 20도
MIN_SPEED = 0.3        # ESC 데드존


class PIDController:
    """Mjolnir class_PIDController.py 를 그대로 옮긴 것.

    대역 제한 미분기(band-limited differentiator)를 쓴다. 단순 차분은 잡음을
    증폭하는데, 이건 tau 로 저역통과를 걸어 그걸 막는다.
    """

    def __init__(self, kp=0.5, ki=0.0, kd=0.0, tau=0.5, T=0.05,
                 err_norm=1.0 / 400.0):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.tau = tau
        self.T = T
        self.errorNormalize = err_norm
        self.limMin, self.limMax = -1.0, 1.0
        self.integrator = 0.0
        self.prevError = 0.0
        self.differentiator = 0.0
        self.prevMeas = 0.0
        self.out = 0.0

    def tick(self, setpoint, current):
        error = float(setpoint - current) * self.errorNormalize

        prop = self.kp * error
        prop = max(self.limMin, min(self.limMax, prop))

        self.integrator += 0.5 * self.ki * self.T * (error + self.prevError)
        self.integrator = max(self.limMin, min(self.limMax, self.integrator))

        self.differentiator = -(
            2.0 * self.kd * (current - self.prevMeas)
            + (2.0 * self.tau - self.T) * self.differentiator
        ) / (2.0 * self.tau + self.T)

        self.out = prop + self.integrator + self.differentiator
        self.out = max(self.limMin, min(self.limMax, self.out))

        self.prevError = error
        self.prevMeas = current
        return self.out


class CentroidFollowNode(Node):
    def __init__(self):
        super().__init__('centroid_follow_node')

        # --- ROI (화면 비율. 원본은 픽셀이지만 해상도 독립을 위해 비율로) ---
        # 위를 볼수록 먼 곳이라 코너를 미리 본다. donkeycar 는 화면 중간쯤을 본다
        # (SCAN_Y=100 / 480). 너무 위를 보면 직선에서 불안정해진다.
        self.declare_parameter('roi_top', 0.45)
        self.declare_parameter('roi_bottom', 0.80)
        self.declare_parameter('roi_left', 0.0)
        self.declare_parameter('roi_right', 1.0)

        # --- HSV. Mjolnir yellow_filter.yaml 실측값이 기본 ---
        self.declare_parameter('hue_low', 15)
        self.declare_parameter('hue_high', 40)
        self.declare_parameter('sat_low', 100)
        self.declare_parameter('sat_high', 255)
        self.declare_parameter('val_low', 50)
        self.declare_parameter('val_high', 255)

        # --- 폭 필터 (이 파이프라인의 핵심) ---
        # 차선은 가늘고 길다. 갓길·차체는 훨씬 넓다. 색이 같아도 폭으로 갈린다.
        # 원본은 10~30px(800px 폭 기준) = 1.2%~3.8%. 차선은 가늘다는 뜻이다.
        # 상한을 넉넉히 잡으면 갓길이 통과하므로 원본 수준으로 조인다.
        self.declare_parameter('width_min_frac', 0.012)
        self.declare_parameter('width_max_frac', 0.06)

        # --- PID ---
        # 원본 errorNormalize = 1/400 은 800px 폭 카메라 기준이다(화면 반폭).
        # 우리 카메라는 640 또는 320px 이라 그대로 쓰면 같은 '화면 끝'이어도
        # 오차 픽셀 수가 절반 이하가 되고, 조향이 8도(240p 면 4도)까지밖에
        # 안 나온다. 최대 조향을 내려면 800px 오차가 필요한데 불가능하다.
        #
        # 화면 반폭으로 정규화하면 해상도와 무관하게 '화면 끝 = 오차 1.0' 이 된다.
        # kp 도 그에 맞춰 올린다(원본 0.5 는 위 정규화와 짝이었다).
        self.declare_parameter('kp', 1.2)
        self.declare_parameter('ki', 0.0)
        # D 항은 0 이 기본이다. 원본 PID 구현의 미분기는 prevMeas 를
        # current(화면 중앙)로 잡는데, 그건 상수라 변화가 0 이다. 즉 D 를 켜도
        # 잡음만 증폭한다 -- 시뮬레이션에서 실제로 오차가 커졌다(0.217 -> 0.243).
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('steer_sign', 1.0)

        # --- 코너 대응 (donkeycar/parts/line_follower.py) ---
        # 원본은 셋을 함께 쓴다. 서로 보완적이다:
        #   A 감속   : 오차가 크면 코너로 보고 속도를 낮춘다 (물리적 여유)
        #   B 불감대 : 선 근처에서 떨지 않게 한다 (게인을 올릴 수 있게 됨)
        #   C 신뢰도 : 픽셀이 모자라면 조향을 갱신하지 않고 유지한다 (코너에서 버팀)

        # --- 선행 스캔 (90도 코너 대응) ---
        # 감속을 '오차가 커진 뒤'에 시작하면 이미 코너 안이다. 1.2 m/s 로
        # 진입하면 필요 횡가속이 2.91 m/s² 라 물리적으로 못 돈다(최소 반경 0.495m).
        #
        # 그래서 스캔을 둘로 나눈다:
        #   가까운 줄 -> 조향  (지금 어디 있나)
        #   먼 줄     -> 감속  (앞에 뭐가 오나)
        # 먼 줄의 오차가 커지면 아직 직진 중이어도 미리 줄인다.
        self.declare_parameter('lookahead_enable', True)
        # 먼 줄의 위치(ROI 상단부 비율). 0 이면 ROI 맨 위.
        self.declare_parameter('far_band_top', 0.0)
        self.declare_parameter('far_band_height', 0.35)
        # 가까운 줄(조향용)은 ROI 하단부를 쓴다.
        self.declare_parameter('near_band_top', 0.45)
        # 먼 줄 오차가 이 비율을 넘으면 코너가 온다고 본다
        self.declare_parameter('far_threshold_frac', 0.10)
        # 선행 감속 강도 (먼 줄 기준). 조향은 안 건드린다.
        self.declare_parameter('far_brake_step', 0.25)

        # A. 오차가 클 때(코너) 감속, 작을 때(직선) 가속.
        # 원본 실측: MAX 0.3 / MIN 0.15 / STEP 0.05 -- 코너에서 절반까지 떨어진다.
        # 물리 한계: 최소 회전반경 R = 0.18/tan(20°) = 0.495m.
        # 횡가속 a = v²/R 이므로 a_lat 1.5 기준 코너 안전속도는 0.86 m/s.
        # 직선은 반경 제약이 없으므로 max 를 훨씬 높게 잡아도 된다.
        self.declare_parameter('speed_max', 1.2)
        self.declare_parameter('speed_min', 0.45)

        # 가속은 완만하게, 감속은 빠르게. 원본은 둘 다 같은 step 이었는데,
        # 속도를 올리면 그 대칭이 문제가 된다 -- max 1.2 에 step 0.05 면
        # 감속에 0.88m 를 쓰고, 그때는 이미 코너를 지나쳤다.
        self.declare_parameter('speed_step', 0.04)      # 가속
        self.declare_parameter('brake_step', 0.20)      # 감속

        # 감속량을 오차에 비례시킨다. 살짝 벗어나면 조금, 코너에 깊이 들어가면
        # 많이 줄인다. 원본은 고정량이라 완만한 커브에서도 과하게 느려졌다.
        self.declare_parameter('brake_scale', True)

        # 코너에서만 게인을 올린다.
        #
        # 게인을 올리면 코너 추종이 확실히 좋아진다(시뮬레이션: kp 1.2 -> 3.0 에서
        # 평균 이탈 0.217 -> 0.101). 하지만 인지 잡음이 클 때 직선이 떨린다
        # (잡음 0.06 에서 조향 변화 3.0도 -> 11.0도).
        #
        # 그래서 상시로 올리지 않고 코너에서만 올린다. 직선은 낮은 게인으로
        # 안정을 유지하고, 코너에 들어가면 강하게 꺾는다.
        self.declare_parameter('corner_gain_scale', 2.2)
        # 코너로 판정할 연속 프레임 수. 한 프레임 튄 것으로 바꾸면 잡음에 흔들린다.
        self.declare_parameter('corner_enter_frames', 3)

        # B. 이만큼 벗어나야 조향을 바꾼다. 원본 TARGET_THRESHOLD=10px(160px 폭 기준).
        # "선 위나 근처에서 너무 예민하게 떠는 걸 막는다"
        self.declare_parameter('target_threshold_frac', 0.06)

        # --- 선 유실 복구 (90도 코너) ---
        # 참조: nsa31/Line-Lane-Follower-Robot_ROS
        #       white_yellow_lane_follower_sim.py 의 else 분기
        #
        #   else:                          # 선을 못 찾았을 때
        #       linear.x = 0.4             # 평소 0.9 -> 0.4 로 감속
        #       angular.z = -0.7           # 강하게 회전
        #
        # 90도 코너에서는 선이 시야를 완전히 벗어난다. 그때 '마지막 조향 유지'
        # 만으로는 못 따라잡는다 -- 코너 진입 직전 조향은 코너를 다 돌기에
        # 모자란 값이기 때문이다. 감속하면서 마지막 오차 방향으로 더 강하게
        # 꺾어야 한다.
        #
        # 방향은 마지막으로 본 선이 어느 쪽이었는지로 정한다. 웹 자료들이
        # 공통으로 말하는 원리이기도 하다 -- "오른쪽으로 돌다 놓쳤으면 더
        # 오른쪽으로".
        self.declare_parameter('lost_recover', True)
        self.declare_parameter('lost_steer_frac', 1.0)    # 최대조향 대비
        # 원본은 0.9 -> 0.4 로 절반 이하까지 떨어뜨린다. 선을 못 보는 동안은
        # 느릴수록 안전하고, 느려야 최대 조향으로 실제로 돌 수 있다.
        # (반경 0.495m 에서 0.25 m/s 면 횡가속 0.13 -- 여유가 크다)
        self.declare_parameter('lost_speed', 0.25)
        # 이만큼 연속으로 놓쳐야 복구 조향을 건다. 한두 프레임 튄 것과 구분.
        self.declare_parameter('lost_enter_frames', 2)

        # C. 마스크 픽셀이 이 비율 미만이면 조향을 갱신하지 않고 이전 값을 유지한다.
        # 코너에서 선이 잠깐 프레임을 벗어나도 마지막 조향으로 계속 돈다.
        self.declare_parameter('confidence_frac', 0.002)

        self.declare_parameter('speed', 0.5)
        self.declare_parameter('publish_debug', False)
        self.declare_parameter('log_hz', 2.0)

        p = self.get_parameter
        self.roi_top = float(p('roi_top').value)
        self.roi_bottom = float(p('roi_bottom').value)
        self.roi_left = float(p('roi_left').value)
        self.roi_right = float(p('roi_right').value)
        self.lower = np.array([int(p('hue_low').value), int(p('sat_low').value),
                               int(p('val_low').value)])
        self.upper = np.array([int(p('hue_high').value), int(p('sat_high').value),
                               int(p('val_high').value)])
        self.width_min_frac = float(p('width_min_frac').value)
        self.width_max_frac = float(p('width_max_frac').value)
        self.steer_sign = float(p('steer_sign').value)
        self.speed = float(p('speed').value)
        self.speed_max = float(p('speed_max').value)
        self.speed_min = float(p('speed_min').value)
        self.speed_step = float(p('speed_step').value)
        self.brake_step = float(p('brake_step').value)
        self.brake_scale = bool(p('brake_scale').value)
        self.lookahead_enable = bool(p('lookahead_enable').value)
        self.far_band_top = float(p('far_band_top').value)
        self.far_band_height = float(p('far_band_height').value)
        self.near_band_top = float(p('near_band_top').value)
        self.far_threshold_frac = float(p('far_threshold_frac').value)
        self.far_brake_step = float(p('far_brake_step').value)
        self.lost_recover = bool(p('lost_recover').value)
        self.lost_steer_frac = float(p('lost_steer_frac').value)
        self.lost_speed = float(p('lost_speed').value)
        self.lost_enter_frames = int(p('lost_enter_frames').value)
        self._lost_run = 0          # 연속으로 놓친 프레임 수
        self._last_side = 0.0       # 마지막으로 본 선의 방향 (+왼쪽 / -오른쪽)
        self.corner_gain_scale = float(p('corner_gain_scale').value)
        self._kp_base = float(p('kp').value)
        self.corner_enter_frames = int(p('corner_enter_frames').value)
        self._corner_run = 0        # 큰 오차가 연속으로 몇 프레임 이어졌나
        self.target_threshold_frac = float(p('target_threshold_frac').value)
        self.confidence_frac = float(p('confidence_frac').value)
        self._throttle = self.speed_min      # 원본: THROTTLE_INITIAL = THROTTLE_MIN
        self.publish_debug = bool(p('publish_debug').value)
        self.log_period = 1.0 / max(0.1, float(p('log_hz').value))

        # err_norm 은 첫 프레임에서 ROI 폭을 보고 확정한다(아래 on_image).
        self.pid = PIDController(kp=float(p('kp').value),
                                 ki=float(p('ki').value),
                                 kd=float(p('kd').value))
        self._norm_set = False

        self.bridge = CvBridge()
        self._speed_cmd = 0.0
        self._steer_cmd = 0.0
        self._last_log = 0.0

        self.create_subscription(Image, 'image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.pub_speed = self.create_publisher(Float64, '/speed', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering', 10)
        self.pub_dbg = (self.create_publisher(Image, 'centroid/debug_image', 1)
                        if self.publish_debug else None)

        # cmd_timeout 1초 -- 프레임이 끊겨도 계속 내보낸다
        self.create_timer(0.05, self.tick)

        self.get_logger().info('centroid_follow_node 시작 (Mjolnir 구조)')

    def find_centroids(self, roi_bgr):
        """컨투어 중심 목록. Mjolnir lane_Detection.py 의 핵심부.

        폭 필터가 갓길·차체를 거른다 -- 색이 같아도 폭이 다르다.
        """
        w = roi_bgr.shape[1]
        w_min = self.width_min_frac * w
        w_max = self.width_max_frac * w

        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.lower, self.upper)
        res = cv2.bitwise_and(roi_bgr, roi_bgr, mask=mask)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        _, bw = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        out = []
        for c in contours:
            _x, _y, cw, _ch = cv2.boundingRect(c)
            if not (w_min < cw < w_max):
                continue            # 폭 필터 -- 갓길·차체는 여기서 탈락
            m = cv2.moments(c)
            if m['m00'] == 0:
                continue
            out.append((int(m['m10'] / m['m00']), int(m['m01'] / m['m00'])))
        return out, mask

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn('이미지 변환 실패: %s' % e,
                                   throttle_duration_sec=2.0)
            return

        h, w = bgr.shape[:2]
        y0 = int(h * self.roi_top)
        y1 = max(y0 + 4, int(h * self.roi_bottom))
        x0 = int(w * self.roi_left)
        x1 = max(x0 + 4, int(w * self.roi_right))
        roi = bgr[y0:y1, x0:x1]
        roi_w = roi.shape[1]

        # 화면 반폭으로 정규화 -- 해상도가 달라도 '화면 끝 = 오차 1.0'
        if not self._norm_set:
            self.pid.errorNormalize = 1.0 / max(1.0, roi_w * 0.5)
            self._norm_set = True
            self.get_logger().info(
                'PID 정규화 확정: 1/%.0f (ROI 폭 %d) -- 화면 끝에서 최대 조향'
                % (roi_w * 0.5, roi_w))

        # 조향용 가까운 줄 (ROI 하단부)
        rh = roi.shape[0]
        near = roi[int(rh * self.near_band_top):, :]
        cents, mask = self.find_centroids(near)

        # 감속용 먼 줄 (ROI 상단부). 조향에는 절대 안 쓴다 --
        # 먼 곳은 부정확해서 조향에 넣으면 오히려 불안정해진다.
        far_err = None
        if self.lookahead_enable:
            f0 = int(rh * self.far_band_top)
            f1 = max(f0 + 4, int(rh * (self.far_band_top + self.far_band_height)))
            far = roi[f0:f1, :]
            far_cents, _ = self.find_centroids(far)
            if far_cents:
                if len(far_cents) >= 2:
                    fx = 0.5 * (far_cents[0][0] + far_cents[1][0])
                else:
                    fx = far_cents[0][0]
                far_err = abs(fx - roi_w / 2.0)

        # C. 신뢰도 -- 마스크 픽셀이 모자라면 조향을 갱신하지 않고 이전 값을 유지한다.
        # 원본: if confidence >= confidence_threshold: (아니면 아무것도 안 함)
        # 코너에서 선이 잠깐 프레임을 벗어나도 마지막 조향으로 계속 돈다.
        confidence = float((mask > 0).sum()) / max(1, mask.size)
        if len(cents) == 0 or confidence < self.confidence_frac:
            self._lost_run += 1

            # 90도 코너: 선이 시야를 완전히 벗어났다. 마지막 조향을 유지하는
            # 것만으로는 못 따라잡는다 -- 그 값은 코너를 다 돌기에 모자라다.
            # 감속하면서 마지막으로 본 방향으로 더 강하게 꺾는다.
            if (self.lost_recover and self._last_side != 0.0
                    and self._lost_run >= self.lost_enter_frames):
                steer = math.copysign(self.lost_steer_frac * MAX_STEER,
                                      self._last_side)
                self._steer_cmd = self.steer_sign * steer
                self._speed_cmd = max(MIN_SPEED, self.lost_speed)
                self._log('선 유실 %d프레임 - %s으로 강제 조향 %.1f도, 감속 %.2f'
                          % (self._lost_run,
                             '좌' if self._last_side > 0 else '우',
                             np.degrees(self._steer_cmd), self._speed_cmd),
                          warn=True)
            else:
                self._log('선 미검출 (신뢰도 %.4f < %.4f) - 마지막 조향 유지'
                          % (confidence, self.confidence_frac), warn=True)

            if self.pub_dbg is not None:
                self._publish_debug(roi, mask, cents, None, msg.header)
            return

        self._lost_run = 0

        # 원본: 2개 이상이면 앞의 둘의 중점, 1개면 그것
        if len(cents) >= 2:
            centroid = int(0.5 * (cents[0][0] + cents[1][0]))
        else:
            centroid = cents[0][0]

        target = roi_w / 2.0
        err_px = abs(centroid - target)

        # 선을 놓쳤을 때 어느 쪽으로 꺾을지 정하려면 마지막 방향을 알아야 한다.
        # + = 선이 왼쪽에 있었다(좌회전 방향)
        self._last_side = float(target - centroid)

        base_dead = self.target_threshold_frac * roi_w

        # 코너 판정: 큰 오차가 연속으로 이어지면 코너로 본다.
        # 한 프레임 튄 것만으로 바꾸면 잡음에 흔들린다.
        if err_px > base_dead:
            self._corner_run += 1
        else:
            self._corner_run = 0

        # 코너에서만 게인을 올린다. 직선은 낮은 게인으로 떨림을 막고,
        # 코너에 들어가면 강하게 꺾는다.
        dead = base_dead
        in_corner = self._corner_run >= self.corner_enter_frames
        self.pid.kp = (self._kp_base * self.corner_gain_scale if in_corner
                       else self._kp_base)

        # B. 불감대 -- 선 근처에서 떨지 않게 한다.
        # 원본: "prevents algorithm from being too twitchy when it is on the line"
        if err_px > dead:
            # 원본: tickPID(centroid, width/2) -- setpoint 가 centroid 다.
            # error = centroid - 중앙 이므로 선이 오른쪽이면 error 양수.
            # 그때 차는 왼쪽에 있다는 뜻이라 조향은 반대로 나가야 한다.
            out = self.pid.tick(centroid, target)
            steer = -self.steer_sign * out * MAX_STEER
            self._steer_cmd = max(-MAX_STEER, min(MAX_STEER, steer))

            # A. 코너다 -> 감속. 오차가 클수록(깊은 코너) 많이 줄인다.
            # 고정량이면 완만한 커브에서도 과하게 느려지고, 급커브에서는
            # 감속이 모자라 코너를 지나친 뒤에야 느려진다.
            drop = self.brake_step
            if self.brake_scale:
                over = (err_px - dead) / max(1.0, (roi_w * 0.5) - dead)
                drop *= min(1.0, max(0.15, over))
            self._throttle = max(self.speed_min, self._throttle - drop)
        else:
            # 직선이다 -> 가속. 조향은 그대로 둔다(불감대 안).
            self._throttle = min(self.speed_max, self._throttle + self.speed_step)

        # 선행 감속 -- 먼 곳에 코너가 보이면 지금 직진 중이어도 미리 줄인다.
        # 조향은 그대로 두고 속도만 건드리므로 직선 안정성은 유지된다.
        far_slow = False
        if far_err is not None and far_err > self.far_threshold_frac * roi_w:
            over = (far_err - self.far_threshold_frac * roi_w) / max(
                1.0, (roi_w * 0.5) - self.far_threshold_frac * roi_w)
            drop = self.far_brake_step * min(1.0, max(0.2, over))
            self._throttle = max(self.speed_min, self._throttle - drop)
            far_slow = True

        self._speed_cmd = max(MIN_SPEED, self._throttle)
        self._log('centroid=%d/%d  err=%.0f  steer=%+.1f도  spd=%.2f  '
                  'far=%s%s  (컨투어 %d개)'
                  % (centroid, roi_w, err_px, np.degrees(self._steer_cmd),
                     self._speed_cmd,
                     '%.0f' % far_err if far_err is not None else '-',
                     ' [선행감속]' if far_slow else '',
                     len(cents))
                  + ('  [코너 kp=%.1f]' % self.pid.kp if in_corner else ''))

        if self.pub_dbg is not None:
            self._publish_debug(roi, mask, cents, centroid, msg.header)

    def _publish_debug(self, roi, mask, cents, centroid, header):
        dbg = roi.copy()
        dbg[mask > 0] = (0, 255, 255)
        for (cx, cy) in cents:
            cv2.circle(dbg, (cx, cy), 6, (0, 255, 0), -1)
        if centroid is not None:
            cv2.circle(dbg, (centroid, dbg.shape[0] // 2), 8, (255, 0, 0), -1)
        cv2.line(dbg, (dbg.shape[1] // 2, 0),
                 (dbg.shape[1] // 2, dbg.shape[0]), (255, 0, 255), 1)
        out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        out.header = header
        self.pub_dbg.publish(out)

    def _log(self, text, warn=False):
        import time
        now = time.time()
        if now - self._last_log < self.log_period:
            return
        self._last_log = now
        (self.get_logger().warn if warn else self.get_logger().info)(text)

    def tick(self):
        self.pub_speed.publish(Float64(data=float(self._speed_cmd)))
        self.pub_steer.publish(Float64(data=float(self._steer_cmd)))


def main(args=None):
    rclpy.init(args=args)
    node = CentroidFollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
