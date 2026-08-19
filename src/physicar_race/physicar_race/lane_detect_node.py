#!/usr/bin/env python3
"""2차선 트랙 차선 인지 노드.

코스 스펙 (2026-08-18 확정):
  바깥 경계 : 흰색 실선  -> 넘으면 실격. HARD 제약.
  중앙선    : 노란 점선  -> 차선 구분자. 장애물 회피 시 넘어도 됨.
  구성      : 2차선

기존 lane_follow_node(단일 HSV 임계 + 최대 컨투어 중심)로는 이 두 마킹의
'의미 차이'를 표현할 수 없어 새로 작성한다. 흰색과 노란색을 별개 마스크로
뽑고, 그 기하 관계에서 차선 구조를 복원한다.

발행 토픽 (기존 obstacle/* 와 동일하게 std_msgs 다중 토픽 컨벤션 유지):
  lane/valid         Bool     인지 신뢰 가능 여부
  lane/offset_right  Float64  오른쪽 차선을 따라갈 때의 정규화 횡오차
  lane/offset_left   Float64  왼쪽 차선을 따라갈 때의 정규화 횡오차
  lane/current_lane  Int32    0=RIGHT, 1=LEFT, -1=UNKNOWN
  lane/margin_left   Float64  좌측 흰 실선까지 정규화 거리 (0 = 밟기 직전, 음수 = 이미 넘음)
  lane/margin_right  Float64  우측 흰 실선까지 정규화 거리
  lane/curvature     Float64  정규화 곡률 (+ = 우커브). 속도 프로파일용

흰선 여유를 좌/우 따로 내는 이유: judgment_node가 실격 회피 조향을 걸려면
'얼마나 가까운가'뿐 아니라 '어느 쪽으로 밀어내야 하는가'를 알아야 한다.

정규화 기준은 화면 half-width. offset 부호는 '+ = 차가 차선 중심보다 오른쪽'.
조향 부호 변환(lane_steer_sign)은 여기가 아니라 judgment_node에서 한다.

주의: HSV 기본값은 placeholder다. 반드시 hsv_calibrate_node로 현장 실측 후 교체할 것.
"""

import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64, Int32

LANE_RIGHT = 0
LANE_LEFT = 1
LANE_UNKNOWN = -1


class LaneDetectNode(Node):
    def __init__(self):
        super().__init__('lane_detect_node')

        # --- ROI / 밴드 ---
        # 화면 위쪽은 하늘/관중석이라 버린다. 실차 카메라는 FOV 98도, 480x360이라
        # 연습 카메라(ELP 1280x720)와 화각이 달라 이 값 재조정 필요.
        self.declare_parameter('roi_top_frac', 0.55)
        # 밴드를 흰선이 실제로 있는 세로 위치로 따라가게 한다. 헤어핀에서는 선이
        # 프레임을 드나들어 고정 밴드가 빈 노면이나 차체를 보게 되는데, 그러면
        # 색이 맞아도 valid=False 로 떨어져 차가 선다.
        self.declare_parameter('band_autotrack', True)
        self.declare_parameter('band_track_gain', 0.25)   # 밴드 이동 속도 (0~1)

        # 차선 판정 래치. 매 프레임 재판정하면 헤어핀에서 RIGHT/LEFT/UNKNOWN 이
        # 토글하고, 목표 차선이 바뀌면 조향이 계단식으로 점프해 사행이 생긴다.
        # 중앙선을 '확실히' 넘었을 때만 전환한다.
        self.declare_parameter('lane_switch_hysteresis', 0.08)  # half-width 대비

        self.declare_parameter('near_band_frac', 0.70)   # ROI 내 근거리 밴드 시작점
        self.declare_parameter('far_band_frac', 0.10)    # ROI 내 원거리 밴드 시작점
        self.declare_parameter('band_height_frac', 0.25)

        # --- 흰색(경계선) 임계값: 채도 낮고 명도 높음 ---
        self.declare_parameter('white_s_max', 60)
        self.declare_parameter('white_v_min', 180)

        # --- 노란색(중앙 점선) 임계값 ---
        self.declare_parameter('yellow_h_min', 18)
        self.declare_parameter('yellow_h_max', 38)
        self.declare_parameter('yellow_s_min', 90)
        self.declare_parameter('yellow_v_min', 90)

        # 열 히스토그램 피크로 인정할 최소 픽셀 수
        self.declare_parameter('min_peak_px', 8)

        # 피크 무게중심을 구할 창의 반폭(px). 화면상 선 두께보다 넉넉하게.
        self.declare_parameter('peak_win_px', 12)

        # 노란선을 찾을 때 흰선 안쪽으로 얼마나 들어가서 볼지(px).
        # 흰선 자체의 두께와 그 바깥 갓길이 노란 마스크에 걸리는 걸 배제한다.
        self.declare_parameter('yellow_inset_px', 12)

        # 노란선은 점선이라 대시 사이 공백에서 사라진다. 그동안 마지막 값을 유지한다.
        self.declare_parameter('yellow_hold_s', 0.8)

        # 차선 폭 초기 추정(화면 폭 대비 비율). 한쪽 흰선이 화면 밖일 때 외삽에 쓴다.
        # 흰선+노란선이 동시에 보일 때마다 실측값으로 갱신된다.
        self.declare_parameter('lane_width_frac_init', 0.42)

        # 카메라 광축과 차량 중심선의 픽셀 오프셋. 실차 인수 후 직진 주행으로 실측.
        self.declare_parameter('cam_center_offset_px', 0.0)

        self.declare_parameter('publish_debug', False)

        # 흰선/노란선이 안 잡힐 때 '실제로 무슨 색이 보이는가'를 찍어주는 진단 모드.
        # 후보를 찾을 때는 일부러 느슨한 기준을 쓴다. 현재 임계값으로 걸러지는
        # 것까지 보여야 무엇 때문에 탈락했는지 알 수 있기 때문이다.
        self.declare_parameter('debug_probe', False)
        self.declare_parameter('probe_period_s', 1.0)
        self.declare_parameter('probe_top_n', 4)
        self.declare_parameter('probe_white_s_max', 120)
        self.declare_parameter('probe_white_v_min', 120)
        self.declare_parameter('probe_yellow_h_min', 10)
        self.declare_parameter('probe_yellow_h_max', 45)
        self.declare_parameter('probe_yellow_s_min', 40)
        self.declare_parameter('probe_yellow_v_min', 60)

        p = self.get_parameter
        self.roi_top_frac = float(p('roi_top_frac').value)
        self.band_autotrack = bool(p('band_autotrack').value)
        self.band_track_gain = float(p('band_track_gain').value)
        self.lane_switch_hysteresis = float(p('lane_switch_hysteresis').value)
        self.near_band_frac = float(p('near_band_frac').value)
        self.far_band_frac = float(p('far_band_frac').value)
        self.band_height_frac = float(p('band_height_frac').value)
        self.white_s_max = int(p('white_s_max').value)
        self.white_v_min = int(p('white_v_min').value)
        self.yellow_h_min = int(p('yellow_h_min').value)
        self.yellow_h_max = int(p('yellow_h_max').value)
        self.yellow_s_min = int(p('yellow_s_min').value)
        self.yellow_v_min = int(p('yellow_v_min').value)
        self.min_peak_px = int(p('min_peak_px').value)
        self.peak_win_px = int(p('peak_win_px').value)
        self.yellow_inset_px = int(p('yellow_inset_px').value)
        self.yellow_hold_s = float(p('yellow_hold_s').value)
        self.lane_width_frac = float(p('lane_width_frac_init').value)
        self.cam_center_offset_px = float(p('cam_center_offset_px').value)
        self.publish_debug = bool(p('publish_debug').value)
        self.debug_probe = bool(p('debug_probe').value)
        self.probe_period_s = float(p('probe_period_s').value)
        self.probe_top_n = int(p('probe_top_n').value)
        self.probe_white_s_max = int(p('probe_white_s_max').value)
        self.probe_white_v_min = int(p('probe_white_v_min').value)
        self.probe_yellow_h_min = int(p('probe_yellow_h_min').value)
        self.probe_yellow_h_max = int(p('probe_yellow_h_max').value)
        self.probe_yellow_s_min = int(p('probe_yellow_s_min').value)
        self.probe_yellow_v_min = int(p('probe_yellow_v_min').value)
        self._probe_stamp = 0.0

        self.bridge = CvBridge()

        # 상태: 점선 홀드 + 차선폭 추정
        self._yellow_x = None
        self._yellow_stamp = 0.0
        self._lane_w = None  # px, 첫 프레임에서 초기화
        self._band_frac = None       # 밴드 추종 현재 위치 (ROI 비율). None이면 파라미터값
        self._lane_latch = LANE_UNKNOWN   # 래치된 차선 판정

        # lane_follow_node / traffic_light_node 와 같은 규약: 상대 토픽명 'image_raw' 를
        # launch remapping 으로 흡수하고, 센서 QoS(BEST_EFFORT)로 구독한다.
        # 기본 QoS(RELIABLE)로 두면 BEST_EFFORT 퍼블리셔와 매칭되지 않아
        # 프레임이 한 장도 안 들어온다.
        self.sub = self.create_subscription(
            Image, 'image_raw', self.on_image, qos_profile_sensor_data)

        self.pub_valid = self.create_publisher(Bool, 'lane/valid', 10)
        self.pub_off_r = self.create_publisher(Float64, 'lane/offset_right', 10)
        self.pub_off_l = self.create_publisher(Float64, 'lane/offset_left', 10)
        self.pub_lane = self.create_publisher(Int32, 'lane/current_lane', 10)
        self.pub_margin_l = self.create_publisher(Float64, 'lane/margin_left', 10)
        self.pub_margin_r = self.create_publisher(Float64, 'lane/margin_right', 10)
        self.pub_curv = self.create_publisher(Float64, 'lane/curvature', 10)
        self.pub_dbg = (
            self.create_publisher(Image, 'lane/debug_image', 1) if self.publish_debug else None
        )

        self.get_logger().info('lane_detect_node 시작')

    # ------------------------------------------------------------------ 유틸

    def _masks(self, roi_bgr):
        hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, self.white_v_min), (180, self.white_s_max, 255))
        yellow = cv2.inRange(
            hsv,
            (self.yellow_h_min, self.yellow_s_min, self.yellow_v_min),
            (self.yellow_h_max, 255, 255),
        )
        k = np.ones((3, 3), np.uint8)
        white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k)
        yellow = cv2.morphologyEx(yellow, cv2.MORPH_OPEN, k)
        return white, yellow

    def _profile(self, mask, y0, y1):
        """밴드 구간의 열별 픽셀 수 프로파일."""
        band = mask[y0:y1, :]
        if band.size == 0:
            return np.zeros(mask.shape[1], np.float32)
        return (band > 0).sum(axis=0).astype(np.float32)

    def _peak(self, prof, lo, hi):
        """[lo, hi) 구간 최대 피크의 x. 임계 미달이면 None.

        argmax만 쓰면 굵은 선에서 항상 왼쪽 끝을 집어 횡오차에 편향이 생긴다.
        피크 주변 창의 무게중심으로 보정해 선의 실제 중심을 잡는다.
        """
        lo = max(0, int(lo))
        hi = min(len(prof), int(hi))
        if hi - lo <= 0:
            return None
        seg = prof[lo:hi]
        i = int(np.argmax(seg))
        if seg[i] < self.min_peak_px:
            return None

        half_win = self.peak_win_px
        a = max(0, i - half_win)
        b = min(len(seg), i + half_win + 1)
        win = seg[a:b]
        # 피크 높이의 절반 미만은 배경으로 보고 무게중심에서 제외
        wgt = np.where(win >= 0.5 * seg[i], win, 0.0)
        total = float(wgt.sum())
        if total <= 0.0:
            return float(lo + i)
        centroid = float((wgt * np.arange(a, b)).sum() / total)
        return float(lo + centroid)

    # ------------------------------------------------------------------ 진단

    def _blobs(self, hsv, mask):
        """마스크에서 큰 덩어리들을 (면적, H, S, V, 중심) 목록으로."""
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
        if n <= 1:
            return []
        order = sorted(range(1, n), key=lambda i: stats[i, cv2.CC_STAT_AREA], reverse=True)
        out = []
        for i in order[:self.probe_top_n]:
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < 8:
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

    def _band_report(self, mask, name, ny0, bh, full_h, roi_y0):
        """근거리 밴드에서 실제로 피크를 잡는지 그대로 보고한다.

        색이 맞아도 그 색이 밴드 밖(더 멀리)에 있으면 못 찾는다. 후보 목록만
        봐서는 이걸 구분할 수 없으므로, 판정에 쓰는 바로 그 밴드의 열 프로파일을
        따로 찍는다.
        """
        prof = self._profile(mask, ny0, ny0 + bh)
        w = len(prof)
        cx = w * 0.5 + self.cam_center_offset_px
        y_lo = (roi_y0 + ny0) / full_h
        y_hi = (roi_y0 + ny0 + bh) / full_h

        found = []

        def side(lo, hi, label):
            lo, hi = max(0, int(lo)), min(w, int(hi))
            if hi - lo <= 0:
                found.append(False)
                return '%s: 구간 없음' % label
            seg = prof[lo:hi]
            i = int(np.argmax(seg))
            peak = float(seg[i])
            ok = peak >= self.min_peak_px
            found.append(ok)
            return ('%s: 최대 %d px @ x=%.2f  -> %s'
                    % (label, int(peak), (lo + i) / w,
                       '검출' if ok else '미달(min_peak_px=%d)' % self.min_peak_px))

        text = ('  %s 밴드 y %.2f~%.2f | %s | %s'
                % (name, y_lo, y_hi, side(0, cx, '좌'), side(cx, w, '우')))
        return text, any(found)

    def _advise(self, hsv, strict_white, rh, ny0, bh, band_ok):
        """왜 안 잡히는지 판정하고, 고칠 명령을 그대로 만들어 준다.

        로그를 사람이 읽고 해석해서 파라미터를 고르는 왕복이 느리다.
        흰선이 ROI 안 어디에 있는지는 노드가 이미 알고 있으므로,
        밴드를 거기로 옮기는 값을 직접 계산해서 출력한다.
        """
        if band_ok:
            return '  진단: 판정 밴드에서 흰선 검출됨. 정상.'

        blobs = self._blobs(hsv, strict_white)
        if not blobs:
            return ('  진단: 현재 임계값으로는 ROI 어디에서도 흰선이 안 잡힌다.\n'
                    '        색 문제다. hsv_tuner_launch.py 로 흰선 기준부터 맞출 것.')

        # 현재 임계값을 통과한 덩어리들이 ROI 안에서 차지하는 세로 위치
        ys = sorted(cy for _a, _h, _s, _v, (_cx, cy) in blobs)
        lo_y, hi_y = ys[0], ys[-1]
        span = max(hi_y - lo_y, rh * 0.15)          # 너무 얇으면 최소 폭 확보
        start = max(0.0, (lo_y - rh * 0.05) / rh)
        height = min(1.0 - start, (span + rh * 0.10) / rh)

        return ('  진단: 흰선은 ROI 세로 %.2f~%.2f 위치에 있는데 '
                '판정 밴드는 %.2f~%.2f 를 보고 있다. 색이 아니라 위치 문제다.\n'
                '  권장: lane_near_band_frac:=%.2f lane_band_height_frac:=%.2f'
                % (lo_y / rh, hi_y / rh, ny0 / rh, (ny0 + bh) / rh,
                   round(start, 2), round(height, 2)))

    def _probe(self, roi, full_h, roi_y0):
        """흰선/노란선이 안 잡히는 이유를 실측값으로 짚어준다.

        노면 색(ROI 중앙값)을 같이 찍는 게 중요하다. 흰선 기준은 '노면보다
        밝고 채도가 낮은 것'인데, 노면 자체가 밝으면 둘이 구분되지 않는다.
        """
        now = time.time()
        if (now - self._probe_stamp) < self.probe_period_s:
            return
        self._probe_stamp = now

        rh, rw = roi.shape[:2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        white = cv2.inRange(hsv, (0, 0, self.probe_white_v_min),
                            (180, self.probe_white_s_max, 255))
        yellow = cv2.inRange(hsv,
                             (self.probe_yellow_h_min, self.probe_yellow_s_min,
                              self.probe_yellow_v_min),
                             (self.probe_yellow_h_max, 255, 255))

        def fmt(blobs, kind):
            rows = []
            for area, hh, ss, vv, (cx, cy) in blobs:
                why = []
                if kind == 'white':
                    if ss > self.white_s_max:
                        why.append('S %d>%d' % (ss, self.white_s_max))
                    if vv < self.white_v_min:
                        why.append('V %d<%d' % (vv, self.white_v_min))
                else:
                    if not (self.yellow_h_min <= hh <= self.yellow_h_max):
                        why.append('H %d 가 %d~%d 밖' % (hh, self.yellow_h_min,
                                                        self.yellow_h_max))
                    if ss < self.yellow_s_min:
                        why.append('S %d<%d' % (ss, self.yellow_s_min))
                    if vv < self.yellow_v_min:
                        why.append('V %d<%d' % (vv, self.yellow_v_min))
                rows.append('  area=%-6d H=%-3d S=%-3d V=%-3d  x=%.2f y=%.2f  -> %s'
                            % (area, hh, ss, vv, cx / rw,
                               (roi_y0 + cy) / full_h,
                               '통과' if not why else '탈락: ' + ', '.join(why)))
            return rows or ['  (없음)']

        med = (int(np.median(hsv[:, :, 0])), int(np.median(hsv[:, :, 1])),
               int(np.median(hsv[:, :, 2])))

        # 실제 판정에 쓰는 밴드에서 무엇이 잡히는지. 색은 맞는데 valid=False 면
        # 대개 여기서 갈린다 -- 선이 밴드보다 멀리 있거나, 밴드가 차체를 보고 있다.
        strict_w, strict_y = self._masks(roi)
        bh = max(2, int(rh * self.band_height_frac))
        ny0 = max(0, min(rh - bh, int(rh * self.near_band_frac)))

        band_w_text, band_w_ok = self._band_report(
            strict_w, '흰선', ny0, bh, full_h, roi_y0)
        band_y_text, _ = self._band_report(
            strict_y, '노란선', ny0, bh, full_h, roi_y0)
        advice = self._advise(hsv, strict_w, rh, ny0, bh, band_w_ok)

        self.get_logger().info(
            '[lane probe] 현재 기준: 흰선 S<=%d V>=%d / 노란선 H %d~%d S>=%d V>=%d, '
            'min_peak_px=%d, ROI 위 %.2f 잘라냄\n'
            '[lane probe] ROI 전체 중앙값(=노면 추정): H=%d S=%d V=%d\n'
            '[lane probe] 흰선 후보 (S<=%d, V>=%d):\n%s\n'
            '[lane probe] 노란선 후보 (H %d~%d, S>=%d, V>=%d):\n%s\n'
            '[lane probe] 판정 밴드 (여기서 못 찾으면 valid=False):\n%s\n%s\n%s'
            % (self.white_s_max, self.white_v_min,
               self.yellow_h_min, self.yellow_h_max,
               self.yellow_s_min, self.yellow_v_min,
               self.min_peak_px, self.roi_top_frac,
               med[0], med[1], med[2],
               self.probe_white_s_max, self.probe_white_v_min,
               '\n'.join(fmt(self._blobs(hsv, white), 'white')),
               self.probe_yellow_h_min, self.probe_yellow_h_max,
               self.probe_yellow_s_min, self.probe_yellow_v_min,
               '\n'.join(fmt(self._blobs(hsv, yellow), 'yellow')),
               band_w_text, band_y_text, advice))

    def _track_band(self, white, rh, bh):
        """판정 밴드를 흰선이 실제로 있는 세로 위치로 따라가게 한다.

        고정 밴드는 헤어핀에서 무너진다. 선이 프레임을 드나들면 밴드가 빈 노면이나
        차체를 보게 되고, 색이 맞아도 valid=False 로 떨어져 차가 선다.

        가장 '가까운' 쪽(아래)에서 흰 픽셀이 충분한 행을 찾아 그 바로 위에 밴드를
        놓는다. 가까울수록 조향 기준으로 정확하기 때문이다. 급변을 막으려고
        지수평활로 서서히 옮긴다.
        """
        if self._band_frac is None:
            self._band_frac = self.near_band_frac
        if not self.band_autotrack:
            return int(rh * self.near_band_frac)

        rows = (white > 0).sum(axis=1)
        # 한 행에 이 정도는 있어야 선으로 본다. 열 기준(min_peak_px)과 별개.
        hit = np.flatnonzero(rows >= max(2, self.min_peak_px // 2))
        if hit.size:
            # 가장 아래(가까운) 검출 행을 밴드 하단에 맞춘다
            target = float(hit.max() - bh) / rh
            target = min(max(target, 0.0), 1.0 - bh / rh)
            g = min(max(self.band_track_gain, 0.0), 1.0)
            self._band_frac = (1.0 - g) * self._band_frac + g * target
        return int(rh * self._band_frac)

    # ------------------------------------------------------------- 메인 콜백

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:  # 카메라 죽음 / 포맷 불일치
            self.get_logger().warn('이미지 변환 실패: %s' % e, throttle_duration_sec=2.0)
            self.pub_valid.publish(Bool(data=False))
            return

        h, w = bgr.shape[:2]
        roi = bgr[int(h * self.roi_top_frac):, :]
        rh = roi.shape[0]
        if rh < 4:
            self.pub_valid.publish(Bool(data=False))
            return

        if self._lane_w is None:
            self._lane_w = self.lane_width_frac * w

        if self.debug_probe:
            self._probe(roi, h, int(h * self.roi_top_frac))

        white, yellow = self._masks(roi)

        bh = max(2, int(rh * self.band_height_frac))
        ny0 = max(0, min(rh - bh, self._track_band(white, rh, bh)))
        fy0 = max(0, min(rh - bh, int(rh * self.far_band_frac)))

        near_w = self._profile(white, ny0, ny0 + bh)
        near_y = self._profile(yellow, ny0, ny0 + bh)
        far_w = self._profile(white, fy0, fy0 + bh)

        cx = w * 0.5 + self.cam_center_offset_px
        half = w * 0.5

        # 흰 실선: 차량 중심 기준 좌/우 각각 최대 피크
        x_wl = self._peak(near_w, 0, cx)
        x_wr = self._peak(near_w, cx, w)

        # 노란 점선: 반드시 두 흰선 '사이'에서만 찾는다.
        #
        # 화면 전체에서 최대 피크를 잡으면 안 된다. 흰선 바깥 갓길이 흙색/황토색이면
        # 노란색 임계에 걸리는데, 그 면적이 중앙 점선보다 훨씬 커서 피크를 가져간다.
        # 그러면 중앙선 위치가 갓길로 잡히고 차선 구조가 통째로 어긋난다.
        # HSV 로는 갓길과 주황 점선을 못 가르지만(색상이 겹친다), 기하로는 확실하다 --
        # 중앙선은 정의상 두 경계선 안쪽에만 존재한다.
        y_lo = (x_wl + self.yellow_inset_px) if x_wl is not None else 0
        y_hi = (x_wr - self.yellow_inset_px) if x_wr is not None else w
        x_y = self._peak(near_y, y_lo, y_hi) if y_hi > y_lo else None
        now = time.time()
        if x_y is not None:
            self._yellow_x, self._yellow_stamp = x_y, now
        elif self._yellow_x is not None and (now - self._yellow_stamp) < self.yellow_hold_s:
            x_y = self._yellow_x
        else:
            self._yellow_x = None

        # 흰선이 하나도 안 잡히면 주행 근거가 없다 -> invalid
        if x_wl is None and x_wr is None:
            self.pub_valid.publish(Bool(data=False))
            return

        # 차선폭 추정 갱신: 노란선과 흰선이 함께 보일 때만
        if x_y is not None:
            if x_wr is not None and x_wr > x_y:
                self._lane_w = 0.9 * self._lane_w + 0.1 * (x_wr - x_y)
            elif x_wl is not None and x_y > x_wl:
                self._lane_w = 0.9 * self._lane_w + 0.1 * (x_y - x_wl)

        lw = self._lane_w

        # 노란선을 놓쳤을 때 UNKNOWN 으로 떨어뜨리지 않고, 래치된 차선과 차선폭으로
        # 중앙선 위치를 복원한다. 점선 공백이나 헤어핀에서 차선 개념을 잃으면
        # 목표가 통로 중앙으로 튀면서 조향이 계단식으로 점프한다.
        if x_y is None and self._lane_latch != LANE_UNKNOWN:
            if self._lane_latch == LANE_RIGHT and x_wr is not None:
                x_y = x_wr - lw
            elif self._lane_latch == LANE_LEFT and x_wl is not None:
                x_y = x_wl + lw

        # 화면 밖으로 나간 흰선은 노란선 기준으로 외삽
        if x_y is not None:
            if x_wr is None:
                x_wr = x_y + lw
            if x_wl is None:
                x_wl = x_y - lw

        # ---- 차선 중심 계산 ----
        if x_y is not None:
            center_right = 0.5 * (x_y + x_wr)
            center_left = 0.5 * (x_wl + x_y)

            # 차선 판정 래치. 중앙선을 '확실히' 넘었을 때만 전환한다.
            # 매 프레임 부호만 보고 판정하면 중앙선 근처에서 RIGHT/LEFT 가 떨리고,
            # 목표 차선이 바뀔 때마다 오차가 off_right <-> off_left 로 점프해
            # 조향이 계단식으로 튄다. 이게 직선에서도 사행하는 원인이다.
            d = (cx - x_y) / half
            if d > self.lane_switch_hysteresis:
                self._lane_latch = LANE_RIGHT
            elif d < -self.lane_switch_hysteresis:
                self._lane_latch = LANE_LEFT
            elif self._lane_latch == LANE_UNKNOWN:
                # 아직 한 번도 못 정했으면 불감대 안에서도 일단 정한다
                self._lane_latch = LANE_RIGHT if d >= 0 else LANE_LEFT
            current = self._lane_latch
        else:
            # 노란선 미검출(긴 대시 공백 / 급커브): 두 흰선 사이를 하나의 통로로 본다.
            # 차선 구분은 포기하되 '흰선 안쪽 유지'라는 안전 목표는 지킨다.
            if x_wl is None:
                x_wl = x_wr - 2.0 * lw
            if x_wr is None:
                x_wr = x_wl + 2.0 * lw
            corridor = 0.5 * (x_wl + x_wr)
            center_right = center_left = corridor
            current = LANE_UNKNOWN

        off_r = (cx - center_right) / half
        off_l = (cx - center_left) / half

        # ---- 흰선 여유 ----
        # 부호 있는 값으로 낸다: 양수 = 아직 안쪽, 음수 = 이미 흰선을 넘음(실격 상태).
        # 어느 차선에 있든 가장 가까운 흰선이 곧 바깥 경계이므로, judgment는
        # 두 값 중 작은 쪽을 위험도로, 그 쪽 방향을 밀어낼 방향으로 쓴다.
        margin_l = (cx - x_wl) / half
        margin_r = (x_wr - cx) / half

        # ---- 곡률 ----
        # 근거리/원거리 밴드의 통로 중심 x 차이를 정규화. + 면 우커브.
        fx_l = self._peak(far_w, 0, cx)
        fx_r = self._peak(far_w, cx, w)
        if fx_l is not None and fx_r is not None:
            far_center = 0.5 * (fx_l + fx_r)
            near_center = 0.5 * (x_wl + x_wr)
            curvature = float(np.clip((far_center - near_center) / half, -1.0, 1.0))
        else:
            curvature = 0.0

        self.pub_valid.publish(Bool(data=True))
        self.pub_off_r.publish(Float64(data=float(np.clip(off_r, -2.0, 2.0))))
        self.pub_off_l.publish(Float64(data=float(np.clip(off_l, -2.0, 2.0))))
        self.pub_lane.publish(Int32(data=int(current)))
        self.pub_margin_l.publish(Float64(data=float(np.clip(margin_l, -1.0, 2.0))))
        self.pub_margin_r.publish(Float64(data=float(np.clip(margin_r, -1.0, 2.0))))
        self.pub_curv.publish(Float64(data=curvature))

        if self.pub_dbg is not None:
            self._publish_debug(roi, white, yellow, x_wl, x_wr, x_y, cx, msg.header)

    def _publish_debug(self, roi, white, yellow, x_wl, x_wr, x_y, cx, header):
        dbg = roi.copy()
        dbg[white > 0] = (255, 255, 255)
        dbg[yellow > 0] = (0, 255, 255)
        hh = dbg.shape[0]
        for x, color in ((x_wl, (255, 0, 0)), (x_wr, (255, 0, 0)), (x_y, (0, 165, 255))):
            if x is not None:
                cv2.line(dbg, (int(x), 0), (int(x), hh), color, 2)
        cv2.line(dbg, (int(cx), 0), (int(cx), hh), (0, 0, 255), 1)
        out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
        out.header = header
        self.pub_dbg.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
