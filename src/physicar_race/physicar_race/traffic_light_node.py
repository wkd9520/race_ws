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

import time

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

        # 색 검출이 실패할 때 '실제로 무슨 색이 보이는가'를 찍어주는 진단 모드.
        # 임계값을 눈대중으로 돌리는 대신, 화면에서 밝은 덩어리들의 실측 H/S/V 와
        # 위치를 로그로 뽑아 그 값을 그대로 파라미터에 넣으면 된다.
        # ROI 밖에 있어서 못 보는 경우까지 잡으려고 ROI 가 아닌 전체 화면을 훑는다.
        self.declare_parameter('debug_probe', False)
        self.declare_parameter('probe_v_min', 120)     # '밝은 영역' 목록의 명도 기준
        self.declare_parameter('probe_period_s', 1.0)  # 로그 도배 방지
        self.declare_parameter('probe_top_n', 5)
        # 적/녹 후보를 찾을 때는 일부러 느슨하게 본다. 현재 임계값으로 걸러지는
        # 후보까지 보여야 '무엇 때문에 탈락했는지'를 알 수 있기 때문이다.
        self.declare_parameter('probe_s_min', 40)
        self.declare_parameter('probe_v_min_relaxed', 60)
        self.declare_parameter('probe_green_h_min', 35)
        self.declare_parameter('probe_green_h_max', 95)
        self.declare_parameter('probe_red_h_lo_max', 15)
        self.declare_parameter('probe_red_h_hi_min', 165)

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
        self.debug_probe = bool(p('debug_probe').value)
        self.probe_v_min = int(p('probe_v_min').value)
        self.probe_period_s = float(p('probe_period_s').value)
        self.probe_top_n = int(p('probe_top_n').value)
        self.probe_s_min = int(p('probe_s_min').value)
        self.probe_v_min_relaxed = int(p('probe_v_min_relaxed').value)
        self.probe_green_h_min = int(p('probe_green_h_min').value)
        self.probe_green_h_max = int(p('probe_green_h_max').value)
        self.probe_red_h_lo_max = int(p('probe_red_h_lo_max').value)
        self.probe_red_h_hi_min = int(p('probe_red_h_hi_min').value)
        self._probe_stamp = 0.0

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

    def _hue_blobs(self, hsv, ranges, s_min, v_min):
        """주어진 색상(H) 구간에서 덩어리를 찾아 (면적, H, S, V, 중심) 목록으로 돌려준다.

        면적 순으로만 훑으면 하늘/노면 같은 큰 배경이 상위를 다 차지해서 정작
        작은 램프가 순위 밖으로 밀린다. 게다가 램프가 밝은 배경에 닿아 있으면
        하나의 덩어리로 붙어버려 중앙값이 배경 색으로 나온다.
        그래서 색상 구간을 먼저 좁히고 나서 덩어리를 찾는다.
        """
        mask = None
        for lo, hi in ranges:
            m = cv2.inRange(hsv, (lo, s_min, v_min), (hi, 255, 255))
            mask = m if mask is None else cv2.bitwise_or(mask, m)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return []

        order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        out = []
        for i in order[:self.probe_top_n]:
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 4:
                continue
            m = labels == i
            out.append((
                area,
                int(np.median(hsv[:, :, 0][m])),
                int(np.median(hsv[:, :, 1][m])),
                int(np.median(hsv[:, :, 2][m])),
                cents[i],
            ))
        return out

    def _verdict(self, area, hh, ss, vv, cy, h):
        """이 후보가 현재 기준으로 왜 걸러지는지 한 줄로 판정한다."""
        why = []
        yf = cy / h
        if not (self.roi_top_frac <= yf <= self.roi_bottom_frac):
            why.append('ROI 밖(y %.2f)' % yf)
        if ss < self.sat_min:
            why.append('S %d<%d' % (ss, self.sat_min))
        if vv < self.val_min:
            why.append('V %d<%d' % (vv, self.val_min))
        if area < self.min_blob_px:
            why.append('면적 %d<%d' % (area, self.min_blob_px))
        in_green = self.green_h_min <= hh <= self.green_h_max
        in_red = hh <= self.red_h_lo_max or hh >= self.red_h_hi_min
        if not (in_green or in_red):
            why.append('H %d 가 적/녹 범위 밖' % hh)
        return '통과' if not why else '탈락: ' + ', '.join(why)

    def _probe(self, bgr):
        """검출 실패 원인을 실측값으로 짚어준다.

        세 묶음을 찍는다:
          1. 화면에서 가장 밝은 덩어리들 (하늘/노면이 뭘로 잡히는지 파악용)
          2. 초록 후보 - 색상 구간을 좁히고 채도/명도는 느슨하게
          3. 빨강 후보 - 위와 동일

        2, 3번이 핵심이다. 각 후보마다 현재 기준으로 통과인지 탈락인지,
        탈락이면 어느 조건에서 걸렸는지 같이 찍으므로 고칠 파라미터가 바로 나온다.
        """
        now = time.time()
        if (now - self._probe_stamp) < self.probe_period_s:
            return
        self._probe_stamp = now

        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

        def fmt(blobs, with_verdict):
            rows = []
            for area, hh, ss, vv, (cx, cy) in blobs:
                row = ('  area=%-6d H=%-3d S=%-3d V=%-3d  위치 x=%.2f y=%.2f'
                       % (area, hh, ss, vv, cx / w, cy / h))
                if with_verdict:
                    row += '  -> ' + self._verdict(area, hh, ss, vv, cy, h)
                rows.append(row)
            return rows or ['  (없음)']

        bright = self._hue_blobs(hsv, [(0, 180)], 0, self.probe_v_min)
        green = self._hue_blobs(
            hsv, [(self.probe_green_h_min, self.probe_green_h_max)],
            self.probe_s_min, self.probe_v_min_relaxed)
        red = self._hue_blobs(
            hsv, [(0, self.probe_red_h_lo_max), (self.probe_red_h_hi_min, 180)],
            self.probe_s_min, self.probe_v_min_relaxed)

        self.get_logger().info(
            '[probe] 현재 기준: ROI y %.2f~%.2f, sat_min=%d, val_min=%d, '
            'green H %d~%d, min_blob_px=%d\n'
            '[probe] 밝은 영역 (V>=%d):\n%s\n'
            '[probe] 초록 후보 (H %d~%d, S>=%d, V>=%d):\n%s\n'
            '[probe] 빨강 후보 (H<=%d 또는 >=%d, S>=%d, V>=%d):\n%s'
            % (self.roi_top_frac, self.roi_bottom_frac, self.sat_min, self.val_min,
               self.green_h_min, self.green_h_max, self.min_blob_px,
               self.probe_v_min, '\n'.join(fmt(bright, False)),
               self.probe_green_h_min, self.probe_green_h_max,
               self.probe_s_min, self.probe_v_min_relaxed, '\n'.join(fmt(green, True)),
               self.probe_red_h_lo_max, self.probe_red_h_hi_min,
               self.probe_s_min, self.probe_v_min_relaxed, '\n'.join(fmt(red, True))))

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn('이미지 변환 실패: %s' % e, throttle_duration_sec=2.0)
            self.pub_valid.publish(Bool(data=False))
            self.pub_state.publish(String(data=STATE_NONE))
            return

        if self.debug_probe:
            self._probe(bgr)

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
