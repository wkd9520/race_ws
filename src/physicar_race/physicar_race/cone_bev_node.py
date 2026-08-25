#!/usr/bin/env python3
"""초록 고깔을 BEV 에서 찾아 미터 좌표로 발행한다.

**MinSeok 님 코드를 한 줄도 안 건드리는 것이 이 노드의 존재 이유다.**

고깔 위치를 미터로 알려면 IPM 이 필요한데, 그건 `physicar_track_perception_v3`
안에 이미 있다. 그걸 복제하는 대신 이미 발행 중인 결과물을 재사용한다:

    /perception_v3/debug/bev          이미 펴진 미터 BEV (150x190, 0.01 m/px)
    /perception_v3/debug/white_mask   같은 격자의 흰선 마스크

두 이미지가 같은 격자 위에 있으므로 픽셀 -> 미터가 단순한 산수다
(`physicar_track_perception_v2.geometry.BevGrid.pixel_to_metric` 과 동일):

    y = y_max - (col + 0.5) * resolution        (+Y 는 왼쪽)
    x = x_max - (row + 0.5) * resolution        (+X 는 전방)

발행하는 것은 고깔 하나당 5개 실수다. 커스텀 메시지를 쓰려면 패키지를
ament_cmake 로 바꾸거나 msg 패키지를 새로 만들어야 해서, 순수 파이썬
패키지에 머물려고 Float32MultiArray 를 쓴다.

    [x, y_cone, cone_half, y_left_wall, y_right_wall] * N

`y_left_wall` / `y_right_wall` 은 그 고깔이 있는 행에서 **자유공간이 끝나는
지점**이다(흰선 또는 BEV 가장자리). 이걸 같이 보내는 이유는, 회피 방향을
고르는 쪽(컨트롤러)이 "어느 쪽이 더 넓은가"를 판단하려면 고깔 위치만으로는
부족하기 때문이다. 흰선을 넘으면 실격이므로 이 값이 안전 울타리가 된다.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray


class ConeBevNode(Node):
    def __init__(self):
        super().__init__('cone_bev_node')

        # --- BEV 격자: perception_v3.yaml 과 반드시 같아야 한다 ---
        # 다르면 좌표가 통째로 틀어진다. 받은 이미지 크기로 교차 검증한다.
        self.declare_parameter('bev_x_min', 0.10)
        self.declare_parameter('bev_x_max', 2.00)
        self.declare_parameter('bev_y_min', -0.75)
        self.declare_parameter('bev_y_max', 0.75)
        self.declare_parameter('bev_resolution', 0.01)

        # --- 초록 고깔 HSV (OpenCV H 는 0~179, 순수 초록이 60) ---
        # 실측 전 추정치다. hsv_tuner_node 로 재서 맞출 것.
        self.declare_parameter('green_h_min', 40)
        self.declare_parameter('green_h_max', 85)
        self.declare_parameter('green_s_min', 80)
        self.declare_parameter('green_v_min', 60)

        self.declare_parameter('min_area_px', 12)      # 이보다 작으면 잡음
        self.declare_parameter('open_kernel', 3)
        self.declare_parameter('wall_row_band', 3)     # 흰선 탐색 시 위아래 여유 행
        self.declare_parameter('log_every', 30)
        # 실차에서는 끈다. 디버그 이미지를 만들고 발행하는 데 드는
        # CPU 가 아깝고, 아무도 안 볼 때가 많다.
        self.declare_parameter('publish_debug', True)

        p = self.get_parameter
        self.x_min = float(p('bev_x_min').value)
        self.x_max = float(p('bev_x_max').value)
        self.y_min = float(p('bev_y_min').value)
        self.y_max = float(p('bev_y_max').value)
        self.res = float(p('bev_resolution').value)
        self.h_min = int(p('green_h_min').value)
        self.h_max = int(p('green_h_max').value)
        self.s_min = int(p('green_s_min').value)
        self.v_min = int(p('green_v_min').value)
        self.min_area = int(p('min_area_px').value)
        self.open_k = int(p('open_kernel').value)
        self.wall_band = int(p('wall_row_band').value)
        self.log_every = int(p('log_every').value)

        self.bridge = CvBridge()
        self._white = None
        self._shape_warned = False
        self._n = 0

        # V3Node 는 두 토픽 모두 기본(RELIABLE) QoS 로 발행한다.
        # sensor QoS(BEST_EFFORT)로 구독하면 한 장도 못 받는다 -- 예전에 겪은 함정.
        self.create_subscription(Image, '/perception_v3/debug/white_mask',
                                 self.on_white, 10)
        self.create_subscription(Image, '/perception_v3/debug/bev',
                                 self.on_bev, 10)
        self.pub_cones = self.create_publisher(Float32MultiArray, '/cones', 10)
        self.pub_dbg = (self.create_publisher(Image, '/cones/debug_image', 1)
                        if bool(p('publish_debug').value) else None)

        self.get_logger().info('cone_bev_node 시작 (초록 고깔 -> 미터 좌표)')

    # ------------------------------------------------------------ 좌표

    def col_to_y(self, col):
        """BEV 열 -> 횡위치(m). 왼쪽이 양수."""
        return self.y_max - (float(col) + 0.5) * self.res

    def row_to_x(self, row):
        """BEV 행 -> 전방 거리(m)."""
        return self.x_max - (float(row) + 0.5) * self.res

    def expected_shape(self):
        """perception_v3 격자에서 나와야 할 이미지 크기."""
        h = int(np.ceil((self.x_max - self.x_min) / self.res))
        w = int(np.ceil((self.y_max - self.y_min) / self.res))
        return h, w

    # ------------------------------------------------------------ 구독

    def on_white(self, msg):
        try:
            self._white = self.bridge.imgmsg_to_cv2(msg, 'mono8')
        except Exception as e:                              # noqa: BLE001
            self.get_logger().error('흰 마스크 변환 실패: %s' % e)

    def green_mask(self, bev_bgr):
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (self.h_min, self.s_min, self.v_min),
                        (self.h_max, 255, 255))
        if self.open_k > 1:
            k = np.ones((self.open_k, self.open_k), np.uint8)
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
        return m

    def free_limits(self, row, col, width):
        """이 행에서 좌/우로 자유공간이 끝나는 y 를 찾는다.

        흰선은 점선이 아니라 실선이지만, BEV 로 편 뒤에는 행마다 끊길 수
        있다. 한 행만 보면 '벽이 없다'고 잘못 읽고 회피 오프셋을 과하게
        키운다 -- 흰선을 넘으면 실격이므로 이쪽 오류가 훨씬 비싸다.
        그래서 위아래 몇 행을 같이 보고 **가장 안쪽에서 발견된 벽**을 쓴다.
        """
        left_wall = self.y_max          # 못 찾으면 BEV 가장자리
        right_wall = self.y_min

        if self._white is None or self._white.shape[1] != width:
            # 흰 마스크가 아직 안 왔거나 격자가 다르면 벽을 모르는 것이다.
            # 가장자리를 돌려주면 컨트롤러가 오프셋을 크게 잡을 수 있으므로,
            # 그쪽에서 max_offset 으로 한 번 더 막는다.
            return left_wall, right_wall

        h = self._white.shape[0]
        r0 = max(0, row - self.wall_band)
        r1 = min(h, row + self.wall_band + 1)
        band = self._white[r0:r1]
        if band.size == 0:
            return left_wall, right_wall

        # 열 방향으로 한 번이라도 흰색이면 그 열은 벽으로 본다
        cols = (band > 0).any(axis=0)

        # 열이 작을수록 y 가 크다(= 왼쪽). 헷갈리기 쉬운 지점이라 못 박아둔다.
        #   col 0   -> y = +0.745  (왼쪽 끝)
        #   col 149 -> y = -0.745  (오른쪽 끝)
        smaller = np.flatnonzero(cols[:col])        # 고깔보다 작은 열 = 왼쪽
        larger = np.flatnonzero(cols[col + 1:])     # 큰 열 = 오른쪽

        if smaller.size:
            # 왼쪽 벽 중 고깔에 가장 가까운 것 = 인덱스가 가장 큰 것
            left_wall = self.col_to_y(int(smaller[-1]))
        if larger.size:
            # 오른쪽 벽 중 가장 가까운 것 = 인덱스가 가장 작은 것
            right_wall = self.col_to_y(int(larger[0]) + col + 1)

        # 못 찾은 쪽은 격자 가장자리로 남는다. 그건 '뚫려 있다'가 아니라
        # '모른다'는 뜻이다 -- 컨트롤러가 track_half_m 로 한 번 더 조인다.
        return left_wall, right_wall

    def on_bev(self, msg):
        try:
            bev = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:                              # noqa: BLE001
            self.get_logger().error('BEV 변환 실패: %s' % e)
            return

        exp_h, exp_w = self.expected_shape()
        if bev.shape[:2] != (exp_h, exp_w) and not self._shape_warned:
            self._shape_warned = True
            self.get_logger().error(
                'BEV 크기가 격자와 다르다: 받은 %dx%d, 기대 %dx%d. '
                'perception_v3.yaml 의 bev.* 와 이 노드의 bev_* 를 맞출 것 '
                '-- 안 맞추면 고깔 좌표가 통째로 틀어진다.'
                % (bev.shape[1], bev.shape[0], exp_w, exp_h))

        mask = self.green_mask(bev)
        cones = self.find_cones(mask)

        flat = []
        for c in cones:
            flat.extend(c)
        self.pub_cones.publish(Float32MultiArray(data=[float(v) for v in flat]))

        self._n += 1
        if self.log_every > 0 and self._n % self.log_every == 0:
            if cones:
                near = min(cones, key=lambda c: c[0])
                self.get_logger().info(
                    '고깔 %d개  가장 가까운 것 %.2fm 앞 %+.2fm 옆 '
                    '(반폭 %.2fm, 좌벽 %+.2f 우벽 %+.2f)'
                    % (len(cones), near[0], near[1], near[2], near[3], near[4]))
            else:
                self.get_logger().info('고깔 없음')

        if self.pub_dbg is not None:
            self._publish_debug(bev, mask, cones, msg.header)

    def find_cones(self, mask):
        """연결요소마다 (x, y, 반폭, 좌벽y, 우벽y) 를 만든다."""
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(
            (mask > 0).astype(np.uint8), connectivity=8)
        out = []
        for i in range(1, count):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            col, row = centroids[i]
            bw = int(stats[i, cv2.CC_STAT_WIDTH])

            # 고깔의 '가장 가까운 면'을 쓴다. 중심을 쓰면 실제보다 멀다고
            # 판단해 회피가 늦어진다.
            near_row = int(stats[i, cv2.CC_STAT_TOP]
                           + stats[i, cv2.CC_STAT_HEIGHT] - 1)
            x = self.row_to_x(near_row)
            y = self.col_to_y(col)
            half = 0.5 * bw * self.res
            lw, rw = self.free_limits(int(round(row)), int(round(col)),
                                      mask.shape[1])
            out.append((x, y, half, lw, rw))
        return out

    def _publish_debug(self, bev, mask, cones, header):
        try:
            dbg = bev.copy()
            dbg[mask > 0] = (0, 255, 0)
            for x, y, half, lw, rw in cones:
                col = int(round((self.y_max - y) / self.res - 0.5))
                row = int(round((self.x_max - x) / self.res - 0.5))
                cv2.circle(dbg, (col, row), 4, (255, 255, 255), 1)
                lc = int(round((self.y_max - lw) / self.res - 0.5))
                rc = int(round((self.y_max - rw) / self.res - 0.5))
                cv2.line(dbg, (lc, row), (rc, row), (255, 255, 0), 1)
            out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
            out.header = header
            self.pub_dbg.publish(out)
        except Exception:                                   # noqa: BLE001
            pass


def main(args=None):
    rclpy.init(args=args)
    node = ConeBevNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
