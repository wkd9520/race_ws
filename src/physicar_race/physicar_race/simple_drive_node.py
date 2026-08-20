#!/usr/bin/env python3
"""최소 주행 노드 - 이게 안 되면 나머지는 볼 필요도 없다.

기존 스택은 밴드 추종·래치·선행점·헤딩·코너 검출·유예 조향이 서로 물려 있어서
안 될 때 어디가 원인인지 가릴 수 없다. 그 전부를 걷어내고 최소한만 남긴다.

    한 밴드에서 선 하나 찾기 -> steer = k × (목표 - 위치) -> 고정 저속

없는 것: 밴드 자동추종, 차선 래치, 선행점, 헤딩, 코너 검출, 유예, 차선 변경,
        장애물 회피, 신호등, 속도 프로파일.

먼저 직선을 따라가는지 확인하고, 되면 하나씩 붙이면서 매번 확인한다.

    ros2 run physicar_race simple_drive_node --ros-args \\
        -r image_raw:=/camera/image_raw -p follow:=yellow

follow 로 무엇을 따라갈지 고른다:
    yellow : 노란 중앙선을 화면의 target_frac 위치에 유지
    white  : 흰선 두 개의 중점을 화면 중앙에 유지
"""

import time

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


class SimpleDriveNode(Node):
    def __init__(self):
        super().__init__('simple_drive_node')

        self.declare_parameter('follow', 'yellow')   # yellow | white

        # 볼 영역. 위는 하늘, 아래는 자기 차체라 양쪽을 다 자른다.
        self.declare_parameter('roi_top', 0.55)
        self.declare_parameter('roi_bottom', 0.92)

        # 노란(주황) 중앙선
        self.declare_parameter('y_h_min', 10)
        self.declare_parameter('y_h_max', 30)
        self.declare_parameter('y_s_min', 120)
        self.declare_parameter('y_v_min', 90)

        # 흰 실선
        self.declare_parameter('w_s_max', 60)
        self.declare_parameter('w_v_min', 180)

        # 중앙선을 화면 어디에 둘지 (0.5=화면중앙, 0.35=중앙에서 왼쪽으로 35%)
        self.declare_parameter('target_frac', 0.35)
        self.declare_parameter('steer_gain', 0.8)
        self.declare_parameter('speed', 0.6)

        # 중앙선을 찾을 가로 범위. 가장자리의 갓길·차체를 배제한다.
        self.declare_parameter('search_frac', 0.8)

        self.declare_parameter('log_hz', 2.0)

        p = self.get_parameter
        self.follow = str(p('follow').value)
        self.roi_top = float(p('roi_top').value)
        self.roi_bottom = float(p('roi_bottom').value)
        self.y_lo = (int(p('y_h_min').value), int(p('y_s_min').value),
                     int(p('y_v_min').value))
        self.y_hi = (int(p('y_h_max').value), 255, 255)
        self.w_s_max = int(p('w_s_max').value)
        self.w_v_min = int(p('w_v_min').value)
        self.target_frac = float(p('target_frac').value)
        self.steer_gain = float(p('steer_gain').value)
        self.speed = float(p('speed').value)
        self.search_frac = float(p('search_frac').value)
        self.log_period = 1.0 / max(0.1, float(p('log_hz').value))

        self.bridge = CvBridge()
        self._last_log = 0.0

        self.create_subscription(Image, 'image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.pub_speed = self.create_publisher(Float64, '/speed', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering', 10)

        # cmd_timeout 1초 -- 프레임이 끊겨도 계속 내보내야 한다
        self.create_timer(0.05, self.tick)
        self._speed_cmd = 0.0
        self._steer_cmd = 0.0

        self.get_logger().info('simple_drive_node 시작 (follow=%s)' % self.follow)

    def _peak_x(self, mask, lo, hi):
        """열마다 픽셀을 세어 최대 지점의 x. 없으면 None."""
        prof = (mask > 0).sum(axis=0).astype(np.float32)
        lo, hi = max(0, int(lo)), min(len(prof), int(hi))
        if hi - lo <= 0:
            return None
        seg = prof[lo:hi]
        i = int(np.argmax(seg))
        if seg[i] < 3:
            return None
        return float(lo + i)

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
        roi = bgr[y0:y1, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        k = np.ones((3, 3), np.uint8)

        cx = w * 0.5
        half = w * 0.5
        lo = cx - self.search_frac * half
        hi = cx + self.search_frac * half

        if self.follow == 'white':
            mask = cv2.inRange(hsv, (0, 0, self.w_v_min), (180, self.w_s_max, 255))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            xl = self._peak_x(mask, 0, cx)
            xr = self._peak_x(mask, cx, w)
            if xl is None or xr is None:
                self._stop('흰선 %s 못 찾음' % ('좌' if xl is None else '우'))
                return
            pos = 0.5 * (xl + xr)
            target = cx
        else:
            mask = cv2.inRange(hsv, self.y_lo, self.y_hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
            pos = self._peak_x(mask, lo, hi)
            if pos is None:
                self._stop('중앙선 못 찾음')
                return
            target = cx - self.target_frac * half

        # + = 선이 목표보다 오른쪽 = 차가 왼쪽에 있다 -> 오른쪽으로 꺾어야 한다
        err = (pos - target) / half
        steer = -self.steer_gain * err
        steer = max(-MAX_STEER, min(MAX_STEER, steer))

        self._steer_cmd = steer
        self._speed_cmd = max(MIN_SPEED, self.speed)
        self._log('pos=%.2f target=%.2f err=%+.3f steer=%+.1f도'
                  % (pos / w, target / w, err, np.degrees(steer)))

    def _stop(self, why):
        self._speed_cmd = 0.0
        self._steer_cmd = 0.0
        self._log('정지 - %s' % why, warn=True)

    def _log(self, text, warn=False):
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
    node = SimpleDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
