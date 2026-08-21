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

        # --- PID (원본 기본값) ---
        self.declare_parameter('kp', 0.5)
        self.declare_parameter('ki', 0.0)
        self.declare_parameter('kd', 0.0)
        self.declare_parameter('steer_sign', 1.0)

        # --- 코너 대응 (donkeycar/parts/line_follower.py) ---
        # 원본은 셋을 함께 쓴다. 서로 보완적이다:
        #   A 감속   : 오차가 크면 코너로 보고 속도를 낮춘다 (물리적 여유)
        #   B 불감대 : 선 근처에서 떨지 않게 한다 (게인을 올릴 수 있게 됨)
        #   C 신뢰도 : 픽셀이 모자라면 조향을 갱신하지 않고 유지한다 (코너에서 버팀)

        # A. 오차가 클 때(코너) 감속, 작을 때(직선) 가속.
        # 원본 실측: MAX 0.3 / MIN 0.15 / STEP 0.05 -- 코너에서 절반까지 떨어진다.
        self.declare_parameter('speed_max', 0.6)
        self.declare_parameter('speed_min', 0.3)
        self.declare_parameter('speed_step', 0.05)

        # B. 이만큼 벗어나야 조향을 바꾼다. 원본 TARGET_THRESHOLD=10px(160px 폭 기준).
        # "선 위나 근처에서 너무 예민하게 떠는 걸 막는다"
        self.declare_parameter('target_threshold_frac', 0.06)

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
        self.target_threshold_frac = float(p('target_threshold_frac').value)
        self.confidence_frac = float(p('confidence_frac').value)
        self._throttle = self.speed_min      # 원본: THROTTLE_INITIAL = THROTTLE_MIN
        self.publish_debug = bool(p('publish_debug').value)
        self.log_period = 1.0 / max(0.1, float(p('log_hz').value))

        self.pid = PIDController(kp=float(p('kp').value),
                                 ki=float(p('ki').value),
                                 kd=float(p('kd').value))

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

        cents, mask = self.find_centroids(roi)

        # C. 신뢰도 -- 마스크 픽셀이 모자라면 조향을 갱신하지 않고 이전 값을 유지한다.
        # 원본: if confidence >= confidence_threshold: (아니면 아무것도 안 함)
        # 코너에서 선이 잠깐 프레임을 벗어나도 마지막 조향으로 계속 돈다.
        confidence = float((mask > 0).sum()) / max(1, mask.size)
        if len(cents) == 0 or confidence < self.confidence_frac:
            self._log('선 미검출 (신뢰도 %.4f < %.4f) - 마지막 조향 유지'
                      % (confidence, self.confidence_frac), warn=True)
            if self.pub_dbg is not None:
                self._publish_debug(roi, mask, cents, None, msg.header)
            return

        # 원본: 2개 이상이면 앞의 둘의 중점, 1개면 그것
        if len(cents) >= 2:
            centroid = int(0.5 * (cents[0][0] + cents[1][0]))
        else:
            centroid = cents[0][0]

        target = roi_w / 2.0
        err_px = abs(centroid - target)

        # B. 불감대 -- 선 근처에서 떨지 않게 한다.
        # 원본: "prevents algorithm from being too twitchy when it is on the line"
        if err_px > self.target_threshold_frac * roi_w:
            # 원본: tickPID(centroid, width/2) -- setpoint 가 centroid 다.
            # error = centroid - 중앙 이므로 선이 오른쪽이면 error 양수.
            # 그때 차는 왼쪽에 있다는 뜻이라 조향은 반대로 나가야 한다.
            out = self.pid.tick(centroid, target)
            steer = -self.steer_sign * out * MAX_STEER
            self._steer_cmd = max(-MAX_STEER, min(MAX_STEER, steer))

            # A. 코너다 -> 감속
            self._throttle = max(self.speed_min, self._throttle - self.speed_step)
        else:
            # 직선이다 -> 가속. 조향은 그대로 둔다(불감대 안).
            self._throttle = min(self.speed_max, self._throttle + self.speed_step)

        self._speed_cmd = max(MIN_SPEED, self._throttle)
        self._log('centroid=%d/%d  err=%.0fpx  steer=%+.1f도  spd=%.2f  '
                  'conf=%.4f  (컨투어 %d개)'
                  % (centroid, roi_w, err_px, np.degrees(self._steer_cmd),
                     self._speed_cmd, confidence, len(cents)))

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
