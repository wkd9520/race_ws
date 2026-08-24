#!/usr/bin/env python3
"""IPM(버드아이) + LOS 가이던스 주행 노드.

기존 두 파이프라인은 같은 가정 하나를 공유했고, 그 가정 때문에 90도 코너에서
무너졌다.

    centroid_follow : cv2.fitLine 으로 x = f(y)  -> theta -> 90도에서 발산
    bev_lane        : sliding_fit 으로 x = f(y)  -> 창이 옆으로 못 따라감

둘 다 "차선은 대체로 세로다"라고 가정한다. 90도 코너는 정확히 그 가정이
깨지는 지점이다. IPM 자체는 죄가 없었다 -- 가로선도 BEV 에서는 그냥
'일정 거리에 가로로 놓인 선'으로 멀쩡히 표현된다. 무너진 건 그 위에 얹은
세로선 피팅이었다.

그래서 이 노드는 **차선을 피팅하지 않는다.** LOS(Line-Of-Sight) 가이던스가
묻는 건 하나뿐이다:

    "지금부터 d 미터 앞에서, 갈 수 있는 곳은 어디인가?"

이 질문은 차선이 세로든 가로든 대각선이든 똑같이 성립한다. 코너에서는
그 앞의 자유공간이 옆으로 치우쳐 있을 뿐이고, 그러면 그쪽으로 조향한다.

파이프라인:

    원본 -> IPM -> 흰선 마스크 -> 행별 자유공간 추적 -> 중심선(BEV, 미터)
         -> 전방주시점(LOS) -> 순수추종 조향 -> 횡가속 한계로 속도 결정

조향은 애커만 순수추종 공식을 그대로 쓴다:

    delta = atan(2 L sin(alpha) / l_d)

L 은 휠베이스(0.18 m), alpha 는 LOS 점을 향한 각, l_d 는 그 거리다.
픽셀 오차에 kp 를 곱하던 것과 달리 게인이 물리량이라 의미가 있다.
"""

import math

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Bool, Float64

WHEELBASE = 0.18            # m -- 드라이버 계층과 같은 값
MAX_STEER = math.radians(20.0)
MIN_SPEED = 0.3             # ESC 불감대
MAX_SPEED = 3.0


class LosDriveNode(Node):
    def __init__(self):
        super().__init__('los_drive_node')

        # --- IPM 사다리꼴 (원본 화면 비율) ---
        # 이 노드에서 **반드시 실측으로 맞춰야 하는 유일한 값**이다.
        # 직선 구간에서 los/debug_image 를 띄우고 두 흰선이 '평행한 세로선'이
        # 되도록 맞춘다. 어긋나면 이후 거리/횡위치가 전부 틀어진다.
        #
        # 아랫변(src_bot_half)을 무작정 넓히면 안 된다. 넓힐수록 화각은
        # 지키지만 근거리 해상도를 버린다 -- 소스 픽셀을 BEV 폭으로 압축하는
        # 비율이 커져서 얇은 흰선이 통째로 사라진다. 1.6 으로 뒀다가 실제로
        # 근거리 선을 다 잃었다. 화면을 조금 넘는 정도가 상한이다.
        self.declare_parameter('src_top_y', 0.58)
        self.declare_parameter('src_top_half', 0.16)
        self.declare_parameter('src_bot_y', 1.00)
        self.declare_parameter('src_bot_half', 0.70)
        self.declare_parameter('src_center', 0.50)

        # --- BEV 출력과 실제 크기 대응 ---
        self.declare_parameter('bev_w', 200)
        self.declare_parameter('bev_h', 200)
        self.declare_parameter('bev_near_m', 0.30)    # 아랫변까지 거리
        self.declare_parameter('bev_range_m', 2.00)   # 아랫변~윗변 전방 거리
        self.declare_parameter('bev_width_m', 1.80)   # BEV 가로가 덮는 실폭

        # --- 흰선(경계) 임계 ---
        self.declare_parameter('white_s_max', 60)
        self.declare_parameter('white_v_min', 180)

        # --- 자유공간 추적 ---
        self.declare_parameter('min_corridor_px', 8)   # 이보다 좁으면 길 아님
        self.declare_parameter('seed_jump_px', 45)     # 행간 중심 이동 허용치
        self.declare_parameter('row_step', 2)
        self.declare_parameter('max_blind_rows', 10)   # 경계선 못 본 채 버틸 행
        # 한쪽 경계만 보일 때 중심을 잡는 데 쓴다. 정밀할 필요는 없다 --
        # '통로 절반쯤 안쪽'이라는 뜻만 통하면 된다. 코스는 2차선.
        self.declare_parameter('track_width_m', 1.00)

        # --- LOS / 순수추종 ---
        # 전방주시거리 = ld_k * v, [ld_min, ld_max] 로 자름.
        # 이 노드에서 코너 성능을 좌우하는 값이다. 상충이 하나 있다:
        #   짧으면 -> LOS 점이 아직 직선 구간이라 코너를 아예 못 본다
        #   길면   -> 코너는 보지만 atan(2L sin a / l_d) 의 분모가 커져
        #             조향이 오히려 약해진다
        # ld_k 를 0.55 로 뒀더니 v_max(1.6)에서도 0.88m 라 코너를 늦게 봤다.
        # 0.90 이면 v=1.0 에서 0.90m -- 합성 코너에서 조향이 가장 셌던 지점.
        self.declare_parameter('ld_min_m', 0.55)
        self.declare_parameter('ld_max_m', 1.30)
        self.declare_parameter('ld_k', 0.90)
        self.declare_parameter('steer_sign', 1.0)

        # --- 속도 ---
        self.declare_parameter('v_max', 1.60)
        self.declare_parameter('v_min', 0.45)
        self.declare_parameter('a_lat_max', 3.0)       # m/s^2 횡가속 한계
        self.declare_parameter('k_vis', 1.10)          # 보이는 거리 * k 로 제한
        self.declare_parameter('accel_step', 0.05)
        self.declare_parameter('brake_step', 0.25)

        # --- 유실 처리 ---
        self.declare_parameter('lost_hold_frames', 6)
        self.declare_parameter('lost_speed', 0.35)

        self.declare_parameter('publish_debug', True)
        self.declare_parameter('log_every', 10)

        p = self.get_parameter
        self.src_top_y = float(p('src_top_y').value)
        self.src_top_half = float(p('src_top_half').value)
        self.src_bot_y = float(p('src_bot_y').value)
        self.src_bot_half = float(p('src_bot_half').value)
        self.src_center = float(p('src_center').value)
        self.bev_w = int(p('bev_w').value)
        self.bev_h = int(p('bev_h').value)
        self.bev_near_m = float(p('bev_near_m').value)
        self.bev_range_m = float(p('bev_range_m').value)
        self.bev_width_m = float(p('bev_width_m').value)
        self.white_s_max = int(p('white_s_max').value)
        self.white_v_min = int(p('white_v_min').value)
        self.min_corridor_px = int(p('min_corridor_px').value)
        self.seed_jump_px = int(p('seed_jump_px').value)
        self.row_step = max(1, int(p('row_step').value))
        self.max_blind_rows = int(p('max_blind_rows').value)
        self.track_width_m = float(p('track_width_m').value)
        self.ld_min_m = float(p('ld_min_m').value)
        self.ld_max_m = float(p('ld_max_m').value)
        self.ld_k = float(p('ld_k').value)
        self.steer_sign = float(p('steer_sign').value)
        self.v_max = float(p('v_max').value)
        self.v_min = float(p('v_min').value)
        self.a_lat_max = float(p('a_lat_max').value)
        self.k_vis = float(p('k_vis').value)
        self.accel_step = float(p('accel_step').value)
        self.brake_step = float(p('brake_step').value)
        self.lost_hold_frames = int(p('lost_hold_frames').value)
        self.lost_speed = float(p('lost_speed').value)
        self.log_every = int(p('log_every').value)

        self.bridge = CvBridge()
        self._M = None
        self._src_shape = None
        self._observed = None
        self._speed = 0.0
        self._steer = 0.0
        self._lost_run = 0
        self._n = 0

        self.create_subscription(Image, 'image_raw', self.on_image,
                                 qos_profile_sensor_data)
        self.pub_speed = self.create_publisher(Float64, '/speed', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering', 10)
        self.pub_valid = self.create_publisher(Bool, 'los/valid', 10)
        self.pub_dbg = (self.create_publisher(Image, 'los/debug_image', 1)
                        if bool(p('publish_debug').value) else None)

        self.get_logger().info('los_drive_node 시작 (IPM + LOS 가이던스)')

    # ------------------------------------------------------------ 좌표 변환

    def _build_M(self, h, w):
        cx = w * self.src_center
        src = np.float32([
            [cx - self.src_top_half * w, h * self.src_top_y],
            [cx + self.src_top_half * w, h * self.src_top_y],
            [cx + self.src_bot_half * w, h * self.src_bot_y],
            [cx - self.src_bot_half * w, h * self.src_bot_y],
        ])
        dst = np.float32([[0, 0], [self.bev_w, 0],
                          [self.bev_w, self.bev_h], [0, self.bev_h]])
        return cv2.getPerspectiveTransform(src, dst)

    def warp(self, bgr):
        h, w = bgr.shape[:2]
        if self._M is None or self._src_shape != (h, w):
            self._M = self._build_M(h, w)
            self._src_shape = (h, w)
            # 원본 밖은 IPM 이 검게 채운다. 검은 것은 흰선이 아니므로 그냥
            # 두면 '갈 수 있는 곳'으로 세어진다 -- 안 본 곳을 뚫린 길로
            # 착각하는 셈이다. 어디가 진짜 관측된 영역인지 따로 갖고 있는다.
            self._observed = cv2.warpPerspective(
                np.full((h, w), 255, np.uint8), self._M,
                (self.bev_w, self.bev_h), flags=cv2.INTER_NEAREST) > 0
        return cv2.warpPerspective(bgr, self._M, (self.bev_w, self.bev_h),
                                   flags=cv2.INTER_LINEAR)

    def row_to_forward_m(self, y):
        """BEV 행 -> 전방 거리(m). 아래가 가깝고 y=0 이 멀다."""
        f = (self.bev_h - float(y)) / max(1.0, float(self.bev_h))
        return self.bev_near_m + f * self.bev_range_m

    def col_to_lateral_m(self, x):
        """BEV 열 -> 횡위치(m). 왼쪽이 양수(차량 좌표 관례)."""
        scale = self.bev_width_m / max(1.0, float(self.bev_w))
        return -(float(x) - self.bev_w * 0.5) * scale

    def white_mask(self, bgr):
        """흰선 마스크. **원본 해상도에서** 뽑는다 -- 순서가 중요하다.

        처음엔 BEV 로 편 다음 색을 찾았다. 그러면 아래쪽 절반의 흰선이 통째로
        사라진다. IPM 이 근거리에서 소스 폭을 3배 넘게 압축하는데, 8px 짜리
        선이 2px 로 줄고 나면 3x3 열림 연산이 그걸 지워버린다.

        원본에서 먼저 찾고 마스크를 펴면 그런 손실이 없다. 채널도 하나라 싸다.
        """
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = cv2.inRange(hsv, (0, 0, self.white_v_min),
                        (180, self.white_s_max, 255))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        # 펴는 과정은 점 샘플링이라 얇은 선을 통째로 건너뛸 수 있다.
        # 미리 굵혀두면 살아남는다. 경계가 두꺼워지는 쪽이라 안전하기도 하다.
        return cv2.dilate(m, np.ones((5, 5), np.uint8))

    def warp_mask(self, mask):
        """이미 만들어진 마스크를 BEV 로 편다."""
        h, w = mask.shape[:2]
        if self._M is None or self._src_shape != (h, w):
            self.warp(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR))
        return (cv2.warpPerspective(mask, self._M, (self.bev_w, self.bev_h),
                                    flags=cv2.INTER_LINEAR) > 0).astype(
                                        np.uint8) * 255

    # -------------------------------------------------------- 자유공간 추적

    @staticmethod
    def _free_runs(row_free, min_len, row_white=None):
        """한 행에서 연속된 자유 구간들을 (시작, 끝) 으로 뽑는다.

        `row_white` 를 주면 **적어도 한쪽이 흰선에 막힌 구간만** 남긴다.
        경계선이 양쪽 다 안 보이면 그건 '넓은 길'이 아니라 '길을 모르는 것'
        이다. 빈 화면이 뻥 뚫린 도로로 읽히던 문제가 여기서 걸린다.
        """
        idx = np.flatnonzero(row_free)
        if idx.size == 0:
            return []
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.concatenate(([0], breaks + 1))
        ends = np.concatenate((breaks, [idx.size - 1]))
        out = []
        n = row_free.shape[0]
        for s, e in zip(starts, ends):
            a, b = int(idx[s]), int(idx[e])
            if b - a + 1 < min_len:
                continue
            if row_white is not None:
                left_wall = a > 0 and row_white[a - 1] > 0
                right_wall = b < n - 1 and row_white[b + 1] > 0
                if not (left_wall or right_wall):
                    continue
            out.append((a, b))
        return out

    def corridor_path(self, white, observed=None):
        """아래에서 위로 자유공간을 따라가며 중심선을 만든다.

        핵심은 **씨앗을 이어받는 것**이다. 각 행에서 독립적으로 중앙을 찾으면
        코너에서 통로가 옆으로 비켜날 때 놓친다. 이전 행의 중심을 물고
        올라가면 통로가 꺾여도 따라간다 -- 이게 세로선 피팅과의 차이다.

        반환: [(y, x_center)] 아래->위 순서
        """
        free = white == 0
        if observed is not None:
            free = free & observed      # 안 본 곳은 갈 수 있는 곳이 아니다
        h, w = self.bev_h, self.bev_w
        track_px = (self.track_width_m / max(1e-6, self.bev_width_m)) * w
        seed = w * 0.5
        path = []
        blind = 0

        for y in range(h - 1, -1, -self.row_step):
            runs = self._free_runs(free[y], self.min_corridor_px, white[y])
            if not runs:
                # 경계선이 안 보이는 행 하나로 경로를 끊으면 안 된다. 점선
                # 구간이나 멀리 있는 얇은 선에서는 늘 일어나는 일이다.
                # 자유공간 자체가 없을 때만 진짜로 멈춘다.
                if not self._free_runs(free[y], self.min_corridor_px):
                    break
                blind += 1
                if blind > self.max_blind_rows:
                    break       # 경계선을 너무 오래 못 봤다. 여기까지가 시야.
                continue
            blind = 0

            # 씨앗을 품은 구간이 있으면 그것. 없으면 씨앗에 가장 가까운 구간.
            pick = None
            for a, b in runs:
                if a <= seed <= b:
                    pick = (a, b)
                    break
            if pick is None:
                pick = min(runs, key=lambda r: min(abs(r[0] - seed),
                                                   abs(r[1] - seed)))
                near = min(abs(pick[0] - seed), abs(pick[1] - seed))
                if near > self.seed_jump_px:
                    break      # 통로가 끊겼다. 여기까지가 보이는 거리.

            a, b = pick
            # 한쪽 경계만 보이면 평균이 틀린다 -- 반대쪽이 화면 밖이라 통로가
            # 실제보다 넓게 잡히고, 중심이 바깥으로 끌려간다. 90도 코너가
            # 정확히 이 상황이다. 그럴 땐 보이는 벽에서 트랙 반폭만큼
            # 안쪽으로 들어간 곳을 중심으로 삼는다.
            left_wall = a > 0 and white[y][a - 1] > 0
            right_wall = b < w - 1 and white[y][b + 1] > 0
            if (b - a + 1) > track_px * 1.4 and left_wall != right_wall:
                cx = (a + track_px * 0.5) if left_wall else (b - track_px * 0.5)
                cx = min(max(cx, a), b)
            else:
                cx = 0.5 * (a + b)

            path.append((y, cx))
            seed = cx

        return path

    def los_point(self, path, lookahead_m):
        """중심선 위에서 전방주시거리만큼 앞의 점을 고른다.

        그만큼 못 보면 보이는 데까지만 쓴다. 이게 자연스러운 열화다 --
        안 보이면 가까이 본다, 그리고 (속도 항에서) 느려진다.
        """
        if not path:
            return None
        chosen = path[0]
        for y, cx in path:
            chosen = (y, cx)
            if self.row_to_forward_m(y) >= lookahead_m:
                break
        return chosen

    # ------------------------------------------------------------ 제어

    @staticmethod
    def pure_pursuit(x_fwd, y_lat):
        """애커만 순수추종. delta = atan(2 L sin(alpha) / l_d).

        alpha 가 커지면(코너) 조향도 같이 커진다. 픽셀 오차에 게인을 곱하던
        방식과 달리, 코너용 게인 배수 같은 임의 계수가 필요 없다.
        """
        ld = math.hypot(x_fwd, y_lat)
        if ld < 1e-3:
            return 0.0
        alpha = math.atan2(y_lat, x_fwd)
        return math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld)

    def speed_limit(self, steer, visible_m):
        """횡가속 한계와 시야 거리로 속도 상한을 정한다.

        v <= sqrt(a_max * R),  R = L / tan|delta|

        '가속 붙은 상태에서 90도 코너를 못 돈다'는 건 물리적으로 v^2/R 이
        접지 한계를 넘었다는 뜻이다. 그러면 조향을 더 주는 게 아니라 v 를
        낮춰야 한다. 여기서는 그 한계를 직접 계산해서 건다.
        """
        v = self.v_max
        t = abs(math.tan(steer))
        if t > 1e-4:
            radius = WHEELBASE / t
            v = min(v, math.sqrt(max(0.0, self.a_lat_max * radius)))
        # 보이는 만큼만 달린다
        v = min(v, self.k_vis * max(0.0, visible_m))
        return max(self.v_min, min(self.v_max, v))

    # ------------------------------------------------------------ 콜백

    def on_image(self, msg: Image):
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:                              # noqa: BLE001
            self.get_logger().error('이미지 변환 실패: %s' % e)
            return

        white = self.warp_mask(self.white_mask(bgr))
        path = self.corridor_path(white, self._observed)

        if len(path) < 3:
            self._lost_run += 1
            if self._lost_run <= self.lost_hold_frames:
                # 마지막 조향을 유지하며 서행. 코너 한복판에서 통로를 잠깐
                # 놓쳤을 때 조향을 0으로 되돌리면 그대로 밖으로 나간다.
                self._publish(self.lost_speed, self._steer, False)
                self._log('통로 미검출 %d - 마지막 조향 유지 (%.1f도)'
                          % (self._lost_run, math.degrees(self._steer)),
                          warn=True)
            else:
                self._publish(0.0, 0.0, False)
                self._log('통로 미검출 - 정지', warn=True)
            if self.pub_dbg is not None:
                self._publish_debug(self.warp(bgr), white, path, None,
                                    msg.header)
            return

        self._lost_run = 0
        visible_m = self.row_to_forward_m(path[-1][0])

        # 속도에 따라 전방주시거리를 바꾼다. 느리면 가까이 봐서 급하게 꺾고,
        # 빠르면 멀리 봐서 부드럽게 간다. 순수추종의 표준 처방.
        ld = min(self.ld_max_m,
                 max(self.ld_min_m, self.ld_k * max(self._speed, self.v_min)))
        pt = self.los_point(path, ld)
        y_row, x_col = pt
        x_fwd = self.row_to_forward_m(y_row)
        y_lat = self.col_to_lateral_m(x_col)

        steer = self.pure_pursuit(x_fwd, y_lat) * self.steer_sign
        steer = max(-MAX_STEER, min(MAX_STEER, steer))

        v_target = self.speed_limit(steer, visible_m)
        dv = v_target - self._speed
        step = self.accel_step if dv > 0 else self.brake_step
        self._speed = self._speed + max(-step, min(step, dv))
        self._speed = max(MIN_SPEED, min(MAX_SPEED, self._speed))
        self._steer = steer

        self._publish(self._speed, self._steer, True)
        self._log('LOS(%.2fm, %+.2fm) ld=%.2f  보임=%.2fm  '
                  'steer=%+.1f도  v=%.2f (한계 %.2f)'
                  % (x_fwd, y_lat, ld, visible_m,
                     math.degrees(self._steer), self._speed, v_target))

        if self.pub_dbg is not None:
            self._publish_debug(self.warp(bgr), white, path, pt, msg.header)

    def _publish(self, speed, steer, valid):
        if speed == 0.0:
            self._speed = 0.0
        self.pub_speed.publish(Float64(data=float(speed)))
        self.pub_steer.publish(Float64(data=float(steer)))
        self.pub_valid.publish(Bool(data=bool(valid)))

    def _log(self, text, warn=False):
        self._n += 1
        if self.log_every > 0 and self._n % self.log_every:
            return
        (self.get_logger().warn if warn else self.get_logger().info)(text)

    def _publish_debug(self, bev, white, path, pt, header):
        try:
            dbg = bev.copy()
            dbg[white > 0] = (0, 0, 255)
            for y, cx in path:
                cv2.circle(dbg, (int(cx), int(y)), 1, (0, 255, 0), -1)
            if pt is not None:
                cv2.circle(dbg, (int(pt[1]), int(pt[0])), 5, (255, 0, 255), 2)
                cv2.line(dbg, (self.bev_w // 2, self.bev_h - 1),
                         (int(pt[1]), int(pt[0])), (255, 0, 255), 1)
            out = self.bridge.cv2_to_imgmsg(dbg, 'bgr8')
            out.header = header
            self.pub_dbg.publish(out)
        except Exception:                                   # noqa: BLE001
            pass


def main(args=None):
    rclpy.init(args=args)
    node = LosDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
