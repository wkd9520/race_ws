#!/usr/bin/env python3
"""콘 감지 - 카메라(초록) + 라이다(거리) 융합.

노면 위 형광 초록 콘을 카메라로 먼저 찾고, 그 방위의 라이다 거리로 위치를
확정한다. 둘을 나눠 쓰는 이유:

  카메라: 색이 선명해 **멀리서** 잡힌다. 미리 감속하고 완만히 피할 수 있다.
          대신 거리 추정이 부정확하다(화면 크기로 역산해야 함).
  라이다: 거리는 정확하지만 콘이 작아 가까워야 잡히고, 헤어핀에서 벽·풀숲을
          장애물로 오인하기 쉽다.

그래서 **카메라가 무엇을 볼지 정하고, 라이다가 얼마나 먼지 정한다.**
라이다 단독으로 나온 점은 콘으로 치지 않는다 -- 트랙 밖 지형이 대부분이라
그걸 믿으면 헤어핀마다 헛브레이크가 걸린다.

발행:
  cone/detected     Bool     전방에 콘이 있는가
  cone/distance     Float64  가장 가까운 콘까지 [m] (없으면 +inf)
  cone/bearing      Float64  그 콘의 방위 [rad], + = 좌측
  cone/lateral      Float64  횡방향 오프셋 [m], + = 좌측
  cone/count        Int32    화면에서 검출된 콘 수
"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, LaserScan
from std_msgs.msg import Bool, Float64, Int32


class ConeDetectNode(Node):
    def __init__(self):
        super().__init__('cone_detect_node')

        # --- 초록 콘 HSV. hsv_tuner 로 실측해서 맞출 것 ---
        self.declare_parameter('green_h_min', 40)
        self.declare_parameter('green_h_max', 85)
        self.declare_parameter('green_s_min', 100)
        self.declare_parameter('green_v_min', 60)

        # 노면만 본다. 하늘·먼 배경의 초록(잔디, 나무)을 배제한다.
        self.declare_parameter('roi_top_frac', 0.45)

        self.declare_parameter('min_area_px', 60)
        # 콘과 잔디를 가르는 진짜 판별점은 '크기'다. 잔디는 화면의 큰 영역을
        # 통째로 차지하고 콘은 작고 야무진 덩어리다. 화면 대비 비율로 잡아
        # 해상도가 바뀌어도 유지되게 한다.
        self.declare_parameter('max_area_frac', 0.05)  # 화면 면적 대비
        # 콘은 세로로 길쭉하다. 넓적한 띠 모양을 추가로 걸러낸다.
        self.declare_parameter('min_aspect', 0.8)      # 높이/너비

        # --- 카메라 기하 (거리 역산용) ---
        self.declare_parameter('hfov_deg', 98.0)       # 실차 카메라 FOV
        self.declare_parameter('cone_height_m', 0.15)  # 실제 콘 높이
        # 화면 높이 대비 콘 픽셀높이로 거리를 역산할 때 쓰는 초점거리(px).
        # 0 이면 화면 세로 크기와 FOV 로 추정한다.
        self.declare_parameter('focal_px', 0.0)

        # --- 라이다 융합 ---
        self.declare_parameter('front_offset_deg', 0.0)
        self.declare_parameter('bearing_tol_deg', 8.0)   # 카메라 방위 ±이 안에서 탐색
        self.declare_parameter('scan_timeout_s', 0.5)
        self.declare_parameter('range_min_m', 0.05)
        self.declare_parameter('range_max_m', 8.0)

        self.declare_parameter('publish_debug', False)

        p = self.get_parameter
        self.h_min = int(p('green_h_min').value)
        self.h_max = int(p('green_h_max').value)
        self.s_min = int(p('green_s_min').value)
        self.v_min = int(p('green_v_min').value)
        self.roi_top_frac = float(p('roi_top_frac').value)
        self.min_area_px = int(p('min_area_px').value)
        self.max_area_frac = float(p('max_area_frac').value)
        self.min_aspect = float(p('min_aspect').value)
        self.hfov = math.radians(float(p('hfov_deg').value))
        self.cone_height_m = float(p('cone_height_m').value)
        self.focal_px = float(p('focal_px').value)
        self.front_offset = math.radians(float(p('front_offset_deg').value))
        self.bearing_tol = math.radians(float(p('bearing_tol_deg').value))
        self.scan_timeout = float(p('scan_timeout_s').value)
        self.range_min = float(p('range_min_m').value)
        self.range_max = float(p('range_max_m').value)
        self.publish_debug = bool(p('publish_debug').value)

        self.bridge = CvBridge()
        self._scan = None          # (angles, ranges)
        self._scan_stamp = 0.0

        self.create_subscription(Image, 'image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.create_subscription(LaserScan, 'scan', self.on_scan,
                                 qos_profile_sensor_data)

        self.pub_det = self.create_publisher(Bool, 'cone/detected', 10)
        self.pub_dist = self.create_publisher(Float64, 'cone/distance', 10)
        self.pub_bear = self.create_publisher(Float64, 'cone/bearing', 10)
        self.pub_lat = self.create_publisher(Float64, 'cone/lateral', 10)
        self.pub_cnt = self.create_publisher(Int32, 'cone/count', 10)
        self.pub_dbg = (self.create_publisher(Image, 'cone/debug_image', 1)
                        if self.publish_debug else None)

        self.get_logger().info('cone_detect_node 시작')

    # ------------------------------------------------------------------ 입력

    def on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return
        r = np.asarray(msg.ranges, dtype=np.float32)
        a = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        a = a - self.front_offset
        a = np.arctan2(np.sin(a), np.cos(a))
        ok = np.isfinite(r) & (r > self.range_min) & (r < self.range_max)
        self._scan = (a[ok], r[ok])
        self._scan_stamp = self._now()

    def _now(self):
        return self.get_clock().now().nanoseconds * 1e-9 if hasattr(
            self, 'get_clock') else 0.0

    # ------------------------------------------------------------------ 검출

    def find_cones(self, bgr):
        """초록 콘 후보를 (면적, 중심x, 중심y, 픽셀높이) 목록으로. 순수 함수."""
        h, w = bgr.shape[:2]
        y0 = max(0, min(h - 2, int(h * self.roi_top_frac)))
        roi = bgr[y0:, :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (self.h_min, self.s_min, self.v_min),
                           (self.h_max, 255, 255))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

        n, _labels, stats, cents = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        max_area = self.max_area_frac * (w * h)
        out = []
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area_px or area > max_area:
                continue          # 너무 크면 잔디밭이나 배경이다
            bw = int(stats[i, cv2.CC_STAT_WIDTH])
            bh = int(stats[i, cv2.CC_STAT_HEIGHT])
            if bw <= 0 or (bh / float(bw)) < self.min_aspect:
                continue          # 넓적한 건 잔디밭이지 콘이 아니다
            cx, cy = cents[i]
            out.append((area, float(cx), float(cy) + y0, bh))
        out.sort(key=lambda c: -c[0])
        return out, mask, y0

    def bearing_of(self, cx, width):
        """화면 x 를 방위각으로. + = 좌측 (ROS 관례)."""
        return -(cx / width - 0.5) * self.hfov

    def distance_from_height(self, px_height, img_h):
        """콘의 픽셀 높이로 거리를 역산한다. 라이다가 못 잡을 때의 대비책."""
        if px_height <= 1:
            return float('inf')
        f = self.focal_px
        if f <= 0.0:
            # 수직 화각을 수평 FOV 와 화면비로 근사
            f = (img_h * 0.5) / math.tan(self.hfov * 0.5) if self.hfov > 0 else 0.0
        if f <= 0.0:
            return float('inf')
        return (self.cone_height_m * f) / px_height

    def lidar_range_at(self, bearing):
        """해당 방위 부근의 라이다 최근접 거리. 없으면 None."""
        if self._scan is None:
            return None
        a, r = self._scan
        sel = np.abs(a - bearing) <= self.bearing_tol
        if not np.any(sel):
            return None
        return float(np.min(r[sel]))

    # ------------------------------------------------------------- 메인 콜백

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn('이미지 변환 실패: %s' % e,
                                   throttle_duration_sec=2.0)
            self._publish(False, float('inf'), 0.0, 0.0, 0)
            return

        h, w = bgr.shape[:2]
        cones, mask, y0 = self.find_cones(bgr)

        if not cones:
            self._publish(False, float('inf'), 0.0, 0.0, 0)
            if self.pub_dbg is not None:
                self._publish_debug(bgr, mask, y0, [], msg.header)
            return

        # 가장 가까운(=화면에서 가장 큰) 콘 하나를 대표로 낸다
        area, cx, _cy, px_h = cones[0]
        bearing = self.bearing_of(cx, w)

        # 라이다가 그 방위에서 뭔가 보면 그 거리를 쓴다. 카메라 역산은 부정확해서
        # 대비책일 뿐이다. 반대로 라이다 단독 점은 콘으로 치지 않는다.
        d = self.lidar_range_at(bearing)
        if d is None:
            d = self.distance_from_height(px_h, h)

        lateral = d * math.sin(bearing) if math.isfinite(d) else 0.0
        self._publish(True, d, bearing, lateral, len(cones))

        if self.pub_dbg is not None:
            self._publish_debug(bgr, mask, y0, cones, msg.header)

    def _publish(self, detected, dist, bearing, lateral, count):
        self.pub_det.publish(Bool(data=bool(detected)))
        self.pub_dist.publish(Float64(data=float(dist)))
        self.pub_bear.publish(Float64(data=float(bearing)))
        self.pub_lat.publish(Float64(data=float(lateral)))
        self.pub_cnt.publish(Int32(data=int(count)))

    def _publish_debug(self, bgr, mask, y0, cones, header):
        dbg = bgr.copy()
        dbg[y0:, :][mask > 0] = (0, 255, 0)
        for i, (area, cx, cy, px_h) in enumerate(cones[:5]):
            color = (0, 0, 255) if i == 0 else (255, 0, 0)
            cv2.circle(dbg, (int(cx), int(cy)), 6, color, 2)
        cv2.putText(dbg, 'cones=%d' % len(cones), (8, dbg.shape[0] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        out.header = header
        self.pub_dbg.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = ConeDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
