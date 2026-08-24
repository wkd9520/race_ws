#!/usr/bin/env python3
"""MinSeok 님 path_overlay 위에 '우리가 가려는 길'을 얹어 다시 발행한다.

`/perception_v3/debug/path_overlay` 에는 이미 세 가지가 그려져 있다:
파랑(왼쪽 흰선), 빨강(오른쪽 흰선), 초록/자홍(인지가 만든 경로).

거기에 **회피까지 반영한 실제 주행 곡선**이 없다. 그건 컨트롤러가 정하는
값이라 인지 노드가 알 수 없기 때문이다. 이 노드가 그 자리를 메운다 --
원본 이미지를 받아 우리 결정을 덧그리고 새 토픽으로 낸다.
MinSeok 님 코드는 여전히 건드리지 않는다.

    /perception_v3/debug/path_overlay  (입력, 그가 그린 것)
    /race/avoid_debug                  (입력, 컨트롤러의 결정)
    /cones                             (입력, 고깔과 자유공간 한계)
              |
              v
    /race/debug/path_overlay           (출력, 위 셋을 합친 것)

그리는 것:

    🩵 청록 굵은 곡선   실제로 그릴 호. 조향각에서 나온 원호다.
    🩵 청록 채운 원     회피까지 반영한 목표점
    ⚪ 회색 빈 원       회피 전 목표점 -- 둘의 간격이 곧 회피량
    🟠 주황 원          고깔
    🟠 주황 가로선      그 고깔 행의 자유공간 한계(좌/우 벽)

호를 그리는 이유가 중요하다. 목표점만 찍으면 "저기로 가려 한다"까지만 알 수
있는데, 순수추종은 **직선이 아니라 호**로 간다. 고깔을 실제로 비켜가는지는
호를 봐야 판단된다.
"""

import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

WHEELBASE = 0.18


class RaceOverlayNode(Node):
    def __init__(self):
        super().__init__('race_overlay_node')

        # perception_v3.yaml 의 bev.* 와 같아야 한다 (cone_bev_node 와 동일).
        self.declare_parameter('bev_x_min', 0.10)
        self.declare_parameter('bev_x_max', 2.00)
        self.declare_parameter('bev_y_min', -0.75)
        self.declare_parameter('bev_y_max', 0.75)
        self.declare_parameter('bev_resolution', 0.01)

        self.declare_parameter('stale_s', 0.5)      # 이보다 오래되면 안 그림
        self.declare_parameter('arc_step_m', 0.02)  # 호를 잇는 간격

        p = self.get_parameter
        self.x_min = float(p('bev_x_min').value)
        self.x_max = float(p('bev_x_max').value)
        self.y_min = float(p('bev_y_min').value)
        self.y_max = float(p('bev_y_max').value)
        self.res = float(p('bev_resolution').value)
        self.stale_s = float(p('stale_s').value)
        self.arc_step = float(p('arc_step_m').value)

        self.bridge = CvBridge()
        self._decision = None       # (x_fwd, y_raw, y_used, steer, offset)
        self._decision_time = 0.0
        self._cones = []
        self._cones_time = 0.0

        self.create_subscription(Float32MultiArray, '/race/avoid_debug',
                                 self.on_decision, 10)
        self.create_subscription(Float32MultiArray, '/cones', self.on_cones, 10)
        self.create_subscription(Image, '/perception_v3/debug/path_overlay',
                                 self.on_overlay, 10)
        self.pub = self.create_publisher(Image, '/race/debug/path_overlay', 2)

        self.get_logger().info(
            'race_overlay_node 시작 -> /race/debug/path_overlay')

    # ------------------------------------------------------------ 좌표

    def to_pixel(self, x, y):
        """미터 -> BEV 픽셀. BevGrid.metric_to_pixel 과 같은 식."""
        col = (self.y_max - y) / self.res - 0.5
        row = (self.x_max - x) / self.res - 0.5
        return int(round(col)), int(round(row))

    # ------------------------------------------------------------ 구독

    def on_decision(self, msg):
        d = list(msg.data)
        if len(d) >= 6 and d[5] > 0.5:
            self._decision = tuple(d[:5])
            self._decision_time = time.time()

    def on_cones(self, msg):
        d = list(msg.data)
        self._cones = [tuple(d[i:i + 5]) for i in range(0, len(d) - 4, 5)]
        self._cones_time = time.time()

    # ------------------------------------------------------------ 호

    def arc_points(self, steer, x_target):
        """조향각이 만드는 원호를 미터 좌표 점열로 돌려준다.

        차량은 원점, 전방이 +X. 조향 delta 면 회전반경 R = L/tan(delta) 이고
        순간회전중심이 (0, R) 에 있다. 그 원 위의 점:

            x = R sin(theta),  y = R (1 - cos(theta))

        delta 가 0에 가까우면 직선으로 처리한다(R 이 발산하므로).
        """
        pts = []
        t = math.tan(steer)
        if abs(t) < 1e-4:
            n = max(2, int(x_target / self.arc_step))
            return [(x_target * i / n, 0.0) for i in range(n + 1)]

        radius = WHEELBASE / t          # 부호 유지: 양수면 왼쪽으로 휨
        # theta = s / R 이라 R 이 음수면 theta 도 음수로 진행해야 한다.
        # dtheta 를 abs(R) 로 나누면 우조향에서 x = R sin(theta) 가 음수가 되어
        # 호가 뒤로 그려진다 -- 부호를 살려야 앞으로 간다.
        theta = 0.0
        dtheta = self.arc_step / radius
        while True:
            x = radius * math.sin(theta)
            y = radius * (1.0 - math.cos(theta))
            pts.append((x, y))
            if x >= x_target or abs(theta) > math.pi * 0.9 or len(pts) > 400:
                break
            theta += dtheta
        return pts

    # ------------------------------------------------------------ 그리기

    def on_overlay(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, 'bgr8').copy()
        except Exception as e:                              # noqa: BLE001
            self.get_logger().error('오버레이 변환 실패: %s' % e)
            return

        now = time.time()
        h, w = img.shape[:2]

        # --- 고깔과 그 행의 자유공간 한계 ---
        if (now - self._cones_time) <= self.stale_s:
            for cx, cy, half, wall_l, wall_r in self._cones:
                col, row = self.to_pixel(cx, cy)
                if not (0 <= row < h):
                    continue
                rad = max(2, int(half / self.res))
                cv2.circle(img, (col, row), rad, (0, 140, 255), 2)
                lc, _ = self.to_pixel(cx, wall_l)
                rc, _ = self.to_pixel(cx, wall_r)
                cv2.line(img, (max(0, lc), row), (min(w - 1, rc), row),
                         (0, 140, 255), 1)

        # --- 우리가 가려는 길 ---
        if self._decision is not None and (now - self._decision_time) <= self.stale_s:
            x_fwd, y_raw, y_used, steer, offset = self._decision

            # 호: 이게 실제 궤적이다. 목표점만 보면 고깔을 비켜가는지 모른다.
            pts = self.arc_points(steer, x_fwd)
            poly = [self.to_pixel(x, y) for x, y in pts]
            poly = [(c, r) for c, r in poly if 0 <= c < w and 0 <= r < h]
            if len(poly) >= 2:
                cv2.polylines(img, [np.asarray(poly, np.int32)], False,
                              (255, 255, 0), 2)

            # 회피 전 목표점 (회색 빈 원) -- 둘의 간격이 회피량
            if abs(offset) > 1e-3:
                col, row = self.to_pixel(x_fwd, y_raw)
                if 0 <= col < w and 0 <= row < h:
                    cv2.circle(img, (col, row), 4, (160, 160, 160), 1)

            # 회피까지 반영한 목표점 (청록 채운 원)
            col, row = self.to_pixel(x_fwd, y_used)
            if 0 <= col < w and 0 <= row < h:
                cv2.circle(img, (col, row), 4, (255, 255, 0), -1)

        out = self.bridge.cv2_to_imgmsg(img, 'bgr8')
        out.header = msg.header
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = RaceOverlayNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
