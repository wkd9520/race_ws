#!/usr/bin/env python3
"""출발 신호등 인지 노드.

코스 스펙 (2026-08-18 확정): 신호등은 출발 시점에만 의미가 있다.
빨강이면 정지, 초록으로 바뀌어야 출발 가능. 주행 중에는 신호등이 없다.

이 노드는 '지금 프레임에 무엇이 보이는가'만 보고한다 (stateless).
'초록을 한 번 봤으니 이제 출발했다'는 래치 판단은 judgment_node의
레이스 상태기계가 담당한다. 인지와 상태를 분리해야 디버깅이 쉽다.

발행 토픽:
  traffic/light_state  String  "RED" | "GREEN" | "NONE"
  traffic/valid        Bool    카메라 살아있음 (프레임 수신+변환 성공)

'카메라 죽음(valid=False)'과 '카메라는 살아있는데 신호등이 안 보임(NONE)'을
반드시 구분해서 발행한다. 전자만 정지 사유이고, 후자는 출발 후 정상 상태다.

주의: HSV 기본값은 placeholder다. 실제 신호등 조명/노출에서 재확인할 것.
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, String

STATE_RED = 'RED'
STATE_GREEN = 'GREEN'
STATE_NONE = 'NONE'


class TrafficLightNode(Node):
    def __init__(self):
        super().__init__('traffic_light_node')

        # 신호등은 화면 위쪽에 있다. 출발선에서 정지한 상태로 보므로 위치가 안정적이다.
        # 실차 인수 후 출발선에 세워놓고 debug_image 보면서 확정할 것.
        self.declare_parameter('roi_top_frac', 0.0)
        self.declare_parameter('roi_bottom_frac', 0.55)

        # 빨강은 HSV 색상환에서 0도를 걸쳐 있어 두 구간으로 나눠 잡는다.
        self.declare_parameter('red_h_lo_max', 10)
        self.declare_parameter('red_h_hi_min', 170)
        self.declare_parameter('green_h_min', 40)
        self.declare_parameter('green_h_max', 90)

        self.declare_parameter('sat_min', 120)
        self.declare_parameter('val_min', 120)

        # 이 픽셀 수 미만이면 노이즈로 보고 NONE 처리
        self.declare_parameter('min_blob_px', 60)

        self.declare_parameter('publish_debug', False)

        p = self.get_parameter
        self.roi_top_frac = float(p('roi_top_frac').value)
        self.roi_bottom_frac = float(p('roi_bottom_frac').value)
        self.red_h_lo_max = int(p('red_h_lo_max').value)
        self.red_h_hi_min = int(p('red_h_hi_min').value)
        self.green_h_min = int(p('green_h_min').value)
        self.green_h_max = int(p('green_h_max').value)
        self.sat_min = int(p('sat_min').value)
        self.val_min = int(p('val_min').value)
        self.min_blob_px = int(p('min_blob_px').value)
        self.publish_debug = bool(p('publish_debug').value)

        self.bridge = CvBridge()

        # 상대 토픽명 'image_raw' + 센서 QoS(BEST_EFFORT). 기본 QoS(RELIABLE)로 두면
        # BEST_EFFORT 퍼블리셔와 매칭이 안 돼 프레임이 조용히 0장 들어온다.
        self.sub = self.create_subscription(
            Image, 'image_raw', self.on_image, qos_profile_sensor_data)
        self.pub_state = self.create_publisher(String, 'traffic/light_state', 10)
        self.pub_valid = self.create_publisher(Bool, 'traffic/valid', 10)
        self.pub_dbg = (
            self.create_publisher(Image, 'traffic/debug_image', 1) if self.publish_debug else None
        )

        self.get_logger().info('traffic_light_node 시작')

    def _largest_blob_px(self, mask):
        """가장 큰 연결 성분의 픽셀 수. 색 번짐/반사에 강하도록 면적 총합이 아닌 최대 blob을 쓴다."""
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return 0
        # 0번은 배경
        return int(stats[1:, cv2.CC_STAT_AREA].max())

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn('이미지 변환 실패: %s' % e, throttle_duration_sec=2.0)
            self.pub_valid.publish(Bool(data=False))
            self.pub_state.publish(String(data=STATE_NONE))
            return

        h = bgr.shape[0]
        y0 = int(h * self.roi_top_frac)
        y1 = max(y0 + 1, int(h * self.roi_bottom_frac))
        roi = bgr[y0:y1, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        s, v = self.sat_min, self.val_min

        red = cv2.bitwise_or(
            cv2.inRange(hsv, (0, s, v), (self.red_h_lo_max, 255, 255)),
            cv2.inRange(hsv, (self.red_h_hi_min, s, v), (180, 255, 255)),
        )
        green = cv2.inRange(hsv, (self.green_h_min, s, v), (self.green_h_max, 255, 255))

        k = np.ones((3, 3), np.uint8)
        red = cv2.morphologyEx(red, cv2.MORPH_OPEN, k)
        green = cv2.morphologyEx(green, cv2.MORPH_OPEN, k)

        red_px = self._largest_blob_px(red)
        green_px = self._largest_blob_px(green)

        if max(red_px, green_px) < self.min_blob_px:
            state = STATE_NONE
        elif red_px >= green_px:
            state = STATE_RED
        else:
            state = STATE_GREEN

        self.pub_valid.publish(Bool(data=True))
        self.pub_state.publish(String(data=state))

        if self.pub_dbg is not None:
            dbg = roi.copy()
            dbg[red > 0] = (0, 0, 255)
            dbg[green > 0] = (0, 255, 0)
            cv2.putText(
                dbg, '%s r=%d g=%d' % (state, red_px, green_px),
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )
            out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
            out.header = msg.header
            self.pub_dbg.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
