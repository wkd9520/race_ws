#!/usr/bin/env python3
"""IPM(bird's eye) + 슬라이딩 윈도우 차선 인지 - 표준 파이프라인.

기존 lane_detect_node 는 화면 좌표에서 밴드 한두 개의 피크만 봤다. 그래서:
  - 곡률이 '밴드 두 개의 중심 차이'라는 조잡한 근사였다
  - 90도 코너에서 선이 가로로 누우면 열 히스토그램에 피크가 안 생겨 무너졌다
  - 원근 때문에 멀수록 차선이 좁아 보여 거리 감각이 없었다

원근 변환으로 위에서 본 시점으로 펴면 이 셋이 구조적으로 사라진다:
  - 차선이 어디서나 평행하고 폭이 일정하다
  - 90도 코너도 위에서 보면 그냥 꺾인 선이다
  - 픽셀 거리가 실제 거리에 비례해 곡률을 물리적으로 계산할 수 있다

파이프라인 (automaticaddison / georgesung 등 표준 구성):
  1. ROI 사다리꼴 -> 원근 변환 (bird's eye)
  2. 색 임계 (흰선 / 노란 중앙선 각각)
  3. 아래 절반 열 히스토그램 -> 시작 피크
  4. 슬라이딩 윈도우로 위로 올라가며 픽셀 수집
  5. 2차 다항식 피팅
  6. 차량 위치(맨 아래)와 곡률 계산

발행:
  bev/valid          Bool     인지 신뢰 가능
  bev/offset         Float64  횡오차 [정규화]. + = 차가 목표보다 왼쪽
  bev/heading        Float64  진행 방향 오차 [정규화]. + = 좌커브
  bev/curvature      Float64  곡률 [1/px], 부호 = 방향
  bev/lane_width_px  Float64  검출된 차선 폭 (BEV 픽셀)
  bev/debug_image    Image    publish_debug 시
"""

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64


class BevLaneNode(Node):
    def __init__(self):
        super().__init__('bev_lane_node')

        # --- ROI 사다리꼴 (원본 화면 비율) ---
        # 이게 이 파이프라인의 유일한 캘리브레이션이자 최대 함정이다.
        # 모서리가 어긋나면 이후 계산 전체에 오차가 전파된다.
        # 직선 구간에서 debug_image 를 보며 두 흰선이 '평행한 세로선'이 되도록 맞춘다.
        self.declare_parameter('src_top_y', 0.62)      # 사다리꼴 윗변 높이
        self.declare_parameter('src_top_half', 0.18)   # 윗변 반폭
        self.declare_parameter('src_bot_y', 0.92)      # 아랫변 높이 (차체 위)
        self.declare_parameter('src_bot_half', 0.62)   # 아랫변 반폭
        self.declare_parameter('src_center', 0.50)     # 좌우 중심 (광축 보정)

        # --- BEV 출력 크기 ---
        self.declare_parameter('bev_w', 320)
        self.declare_parameter('bev_h', 320)

        # --- 색 임계 ---
        self.declare_parameter('white_s_max', 60)
        self.declare_parameter('white_v_min', 180)
        self.declare_parameter('yellow_h_min', 10)
        self.declare_parameter('yellow_h_max', 32)
        self.declare_parameter('yellow_s_min', 110)
        self.declare_parameter('yellow_v_min', 90)

        # --- 슬라이딩 윈도우 ---
        self.declare_parameter('n_windows', 10)
        self.declare_parameter('window_margin', 40)    # 창 반폭 (BEV px)
        self.declare_parameter('min_pix', 30)          # 창 재중심 최소 픽셀
        self.declare_parameter('min_total_pix', 200)   # 피팅에 필요한 총 픽셀

        # --- 목표 ---
        # 오른쪽 차선 주행 시 중앙선은 내 왼쪽에 보인다. BEV 폭 대비 목표 위치.
        self.declare_parameter('target_offset_frac', 0.25)

        # 여러 프레임 평활 (표준 구현이 10프레임 이동평균을 쓴다)
        self.declare_parameter('smooth_n', 8)

        # 사다리꼴 자동 맞춤. 직선 구간에서 흰선 두 개를 실제로 찾아 그 모양대로
        # 사다리꼴을 잡는다. 손으로 4점을 추측하는 건 사실상 불가능하고,
        # 어긋나면 이후 계산 전체에 오차가 전파된다(이 파이프라인 최대 함정).
        self.declare_parameter('auto_calibrate', True)
        self.declare_parameter('calib_frames', 20)      # 이만큼 모아서 확정
        self.declare_parameter('calib_margin_px', 12)   # 선 바깥쪽 여유

        self.declare_parameter('publish_debug', False)

        p = self.get_parameter
        self.src_top_y = float(p('src_top_y').value)
        self.src_top_half = float(p('src_top_half').value)
        self.src_bot_y = float(p('src_bot_y').value)
        self.src_bot_half = float(p('src_bot_half').value)
        self.src_center = float(p('src_center').value)
        self.bev_w = int(p('bev_w').value)
        self.bev_h = int(p('bev_h').value)
        self.white_s_max = int(p('white_s_max').value)
        self.white_v_min = int(p('white_v_min').value)
        self.y_h_min = int(p('yellow_h_min').value)
        self.y_h_max = int(p('yellow_h_max').value)
        self.y_s_min = int(p('yellow_s_min').value)
        self.y_v_min = int(p('yellow_v_min').value)
        self.n_windows = int(p('n_windows').value)
        self.window_margin = int(p('window_margin').value)
        self.min_pix = int(p('min_pix').value)
        self.min_total_pix = int(p('min_total_pix').value)
        self.target_offset_frac = float(p('target_offset_frac').value)
        self.smooth_n = max(1, int(p('smooth_n').value))
        self.auto_calibrate = bool(p('auto_calibrate').value)
        self.calib_frames = int(p('calib_frames').value)
        self.calib_margin_px = int(p('calib_margin_px').value)
        self.publish_debug = bool(p('publish_debug').value)
        self._calib = []          # 자동 캘리브레이션 표본
        self._calibrated = not self.auto_calibrate

        self.bridge = CvBridge()
        self._M = None            # 원근 변환 행렬 (첫 프레임에서 계산)
        self._src_shape = None
        self._hist_off = []       # 평활용 이력
        self._hist_head = []

        self.create_subscription(Image, 'image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.pub_valid = self.create_publisher(Bool, 'bev/valid', 10)
        self.pub_off = self.create_publisher(Float64, 'bev/offset', 10)
        self.pub_head = self.create_publisher(Float64, 'bev/heading', 10)
        self.pub_curv = self.create_publisher(Float64, 'bev/curvature', 10)
        self.pub_lw = self.create_publisher(Float64, 'bev/lane_width_px', 10)
        self.pub_dbg = (self.create_publisher(Image, 'bev/debug_image', 1)
                        if self.publish_debug else None)

        self.get_logger().info('bev_lane_node 시작 (IPM + 슬라이딩 윈도우)')

    # ------------------------------------------------------------- 원근 변환

    def _build_M(self, h, w):
        """사다리꼴 -> 직사각형 변환 행렬.

        아래가 넓고 위가 좁은 사다리꼴을 직사각형으로 펴면, 원근 때문에
        멀수록 좁아 보이던 차선이 평행해진다.
        """
        cx = w * self.src_center
        src = np.float32([
            [cx - self.src_top_half * w, h * self.src_top_y],   # 좌상
            [cx + self.src_top_half * w, h * self.src_top_y],   # 우상
            [cx + self.src_bot_half * w, h * self.src_bot_y],   # 우하
            [cx - self.src_bot_half * w, h * self.src_bot_y],   # 좌하
        ])
        dst = np.float32([
            [0, 0], [self.bev_w, 0],
            [self.bev_w, self.bev_h], [0, self.bev_h],
        ])
        return cv2.getPerspectiveTransform(src, dst)

    def _measure_edges(self, bgr, y_frac):
        """원본 화면의 특정 높이에서 좌/우 흰선 x 를 잰다."""
        h, w = bgr.shape[:2]
        y = int(h * y_frac)
        band = bgr[max(0, y - 3):min(h, y + 4), :]
        hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (0, 0, self.white_v_min),
                           (180, self.white_s_max, 255))
        cols = (mask > 0).sum(axis=0)
        cx = w // 2
        left = np.flatnonzero(cols[:cx] > 0)
        right = np.flatnonzero(cols[cx:] > 0)
        if len(left) == 0 or len(right) == 0:
            return None
        return float(left[-1]), float(cx + right[0])

    def _try_calibrate(self, bgr):
        """직선 구간의 흰선 모양에서 사다리꼴을 자동으로 잡는다.

        위쪽(멀리)과 아래쪽(가까이)에서 차선 폭을 재면 원근이 그대로 드러난다.
        그 두 폭을 사다리꼴의 윗변/아랫변으로 쓰면 BEV 에서 평행해진다.
        """
        h, w = bgr.shape[:2]
        top = self._measure_edges(bgr, self.src_top_y)
        bot = self._measure_edges(bgr, self.src_bot_y)
        if top is None or bot is None:
            return
        (tl, tr), (bl, br) = top, bot
        if tr - tl < 8 or br - bl < 8 or (br - bl) <= (tr - tl):
            return          # 원근이면 아래가 더 넓어야 한다
        self._calib.append((tl, tr, bl, br))
        if len(self._calib) < self.calib_frames:
            return

        a = np.array(self._calib, dtype=np.float64)
        tl, tr, bl, br = np.median(a, axis=0)
        m = self.calib_margin_px
        self.src_center = ((tl + tr) * 0.5 + (bl + br) * 0.5) * 0.5 / w
        self.src_top_half = ((tr - tl) * 0.5 + m) / w
        self.src_bot_half = ((br - bl) * 0.5 + m) / w
        self._M = None
        self._calibrated = True
        self.get_logger().info(
            '사다리꼴 자동 확정: center=%.3f top_half=%.3f bot_half=%.3f '
            '(%d프레임 중앙값)'
            % (self.src_center, self.src_top_half, self.src_bot_half,
               self.calib_frames))

    def warp(self, bgr):
        h, w = bgr.shape[:2]
        if self._M is None or self._src_shape != (h, w):
            self._M = self._build_M(h, w)
            self._src_shape = (h, w)
        return cv2.warpPerspective(bgr, self._M, (self.bev_w, self.bev_h),
                                   flags=cv2.INTER_LINEAR)

    # --------------------------------------------------------------- 마스크

    def masks(self, bev_bgr):
        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, self.white_v_min),
                            (180, self.white_s_max, 255))
        yellow = cv2.inRange(hsv, (self.y_h_min, self.y_s_min, self.y_v_min),
                             (self.y_h_max, 255, 255))
        k = np.ones((3, 3), np.uint8)
        return (cv2.morphologyEx(white, cv2.MORPH_OPEN, k),
                cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k))

    # ------------------------------------------------- 슬라이딩 윈도우 + 피팅

    def sliding_fit(self, mask, x_start):
        """아래에서 위로 창을 옮기며 픽셀을 모아 2차 다항식을 맞춘다.

        밴드 하나만 보는 방식과의 차이가 여기서 난다. 창이 선을 따라 올라가므로
        곡선이어도 놓치지 않고, 90도로 꺾여도 그 방향으로 따라간다.
        """
        h, w = mask.shape[:2]
        nz_y, nz_x = mask.nonzero()
        if len(nz_x) == 0:
            return None, [], []

        win_h = h // self.n_windows
        cur_x = int(x_start)
        idxs = []
        centers = []

        for i in range(self.n_windows):
            y_hi = h - i * win_h
            y_lo = y_hi - win_h
            x_lo = cur_x - self.window_margin
            x_hi = cur_x + self.window_margin
            centers.append((cur_x, (y_lo + y_hi) // 2))

            good = ((nz_y >= y_lo) & (nz_y < y_hi) &
                    (nz_x >= x_lo) & (nz_x < x_hi)).nonzero()[0]
            idxs.append(good)
            # 창 안에 충분히 있으면 다음 창을 그 평균으로 옮긴다 -> 선을 따라간다
            if len(good) > self.min_pix:
                cur_x = int(np.mean(nz_x[good]))

        idxs = np.concatenate(idxs) if idxs else np.array([], dtype=int)
        if len(idxs) < self.min_total_pix:
            return None, centers, idxs

        ys, xs = nz_y[idxs], nz_x[idxs]
        if np.ptp(ys) < 10:          # 세로로 안 퍼져 있으면 피팅이 무의미
            return None, centers, idxs
        try:
            fit = np.polyfit(ys, xs, 2)
        except Exception:
            return None, centers, idxs
        return fit, centers, idxs

    def hist_peak(self, mask, lo, hi):
        """아래 절반의 열 히스토그램에서 시작 피크."""
        h = mask.shape[0]
        hist = mask[h // 2:, :].sum(axis=0).astype(np.float32)
        lo, hi = max(0, int(lo)), min(len(hist), int(hi))
        if hi - lo <= 0:
            return None
        seg = hist[lo:hi]
        i = int(np.argmax(seg))
        if seg[i] <= 0:
            return None
        return lo + i

    @staticmethod
    def poly_x(fit, y):
        return fit[0] * y * y + fit[1] * y + fit[2]

    @staticmethod
    def poly_slope(fit, y):
        return 2.0 * fit[0] * y + fit[1]

    def _smooth(self, buf, val):
        buf.append(val)
        if len(buf) > self.smooth_n:
            buf.pop(0)
        return float(np.mean(buf))

    # ------------------------------------------------------------- 메인 콜백

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().warn('이미지 변환 실패: %s' % e,
                                   throttle_duration_sec=2.0)
            self.pub_valid.publish(Bool(data=False))
            return

        if not self._calibrated:
            self._try_calibrate(bgr)

        bev = self.warp(bgr)
        white, yellow = self.masks(bev)
        w, h = self.bev_w, self.bev_h
        mid = w // 2

        # 중앙선(노랑)과 좌우 흰선을 각각 추적한다
        y_start = self.hist_peak(yellow, 0, w)
        fit_y, cen_y, _ = (self.sliding_fit(yellow, y_start)
                           if y_start is not None else (None, [], []))

        wl_start = self.hist_peak(white, 0, mid)
        wr_start = self.hist_peak(white, mid, w)
        fit_wl, cen_wl, _ = (self.sliding_fit(white, wl_start)
                             if wl_start is not None else (None, [], []))
        fit_wr, cen_wr, _ = (self.sliding_fit(white, wr_start)
                             if wr_start is not None else (None, [], []))

        y_bot = float(h - 1)
        car_x = mid       # BEV 에서 차량은 항상 화면 중앙 (사다리꼴을 그렇게 잡았으므로)

        # ---- 기준선 선택 ----
        # 중앙선이 있으면 그걸 쓰고, 없으면 흰선으로 대체한다.
        ref_fit = fit_y
        if ref_fit is None:
            ref_fit = fit_wl if fit_wl is not None else fit_wr
        if ref_fit is None:
            self.pub_valid.publish(Bool(data=False))
            if self.pub_dbg is not None:
                self._publish_debug(bev, white, yellow, [], msg.header)
            return

        # ---- 횡오차 ----
        ref_x = self.poly_x(ref_fit, y_bot)
        target_x = car_x - self.target_offset_frac * w
        if fit_y is None and ref_fit is fit_wr:
            # 우측 흰선을 기준으로 삼았으면 목표는 반대편이다
            target_x = car_x + self.target_offset_frac * w
        offset = (ref_x - target_x) / (w * 0.5)

        # ---- 헤딩 ----
        # BEV 에서 기울기는 곧 진행 방향과 차선 방향의 차이다. 화면 좌표와 달리
        # 원근 왜곡이 없으므로 이 값이 실제 각도에 비례한다.
        slope = self.poly_slope(ref_fit, y_bot)
        heading = float(np.clip(-slope, -2.0, 2.0))

        # ---- 곡률 ----
        # 2차 계수가 곧 곡률에 비례한다. 부호가 방향.
        curvature = float(np.clip(-ref_fit[0] * 1000.0, -1.0, 1.0))

        # ---- 차선 폭 (참고용) ----
        lane_w = 0.0
        if fit_wl is not None and fit_wr is not None:
            lane_w = abs(self.poly_x(fit_wr, y_bot) - self.poly_x(fit_wl, y_bot))

        self.pub_valid.publish(Bool(data=True))
        self.pub_off.publish(Float64(data=self._smooth(self._hist_off, offset)))
        self.pub_head.publish(Float64(data=self._smooth(self._hist_head, heading)))
        self.pub_curv.publish(Float64(data=curvature))
        self.pub_lw.publish(Float64(data=float(lane_w)))

        if self.pub_dbg is not None:
            self._publish_debug(bev, white, yellow,
                                cen_y + cen_wl + cen_wr, msg.header,
                                fits=[f for f in (fit_y, fit_wl, fit_wr)
                                      if f is not None])

    def _publish_debug(self, bev, white, yellow, centers, header, fits=()):
        dbg = bev.copy()
        dbg[white > 0] = (255, 255, 255)
        dbg[yellow > 0] = (0, 255, 255)
        for (cx, cy) in centers:
            cv2.rectangle(dbg,
                          (int(cx - self.window_margin), int(cy - 8)),
                          (int(cx + self.window_margin), int(cy + 8)),
                          (0, 255, 0), 1)
        for f in fits:
            ys = np.linspace(0, self.bev_h - 1, 20)
            for y in ys:
                x = int(self.poly_x(f, y))
                if 0 <= x < self.bev_w:
                    cv2.circle(dbg, (x, int(y)), 2, (0, 0, 255), -1)
        cv2.line(dbg, (self.bev_w // 2, 0), (self.bev_w // 2, self.bev_h),
                 (255, 0, 255), 1)
        out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        out.header = header
        self.pub_dbg.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = BevLaneNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
