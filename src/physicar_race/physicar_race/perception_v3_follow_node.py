#!/usr/bin/env python3
"""MinSeok 님의 `/perception_v3/path` 를 따라가는 순수추종 컨트롤러.

`physicar_track_perception_v3` 는 의도적으로 컨트롤러를 뺐다 -- 원본
README 의 "의도적으로 제외한 범위"에 "controller와 closed-loop driving
구성"이 명시되어 있다. 인지(perception_v3, physicar_track_perception_v2,
physicar_camera_tf_correction)는 손대지 않고 그대로 두고, 이 노드가 그 뒤에
붙어 빈 자리(경로 -> /speed, /steering)만 채운다.

`/perception_v3/path` 의 점은 이미 `base_footprint` 미터 좌표계다
(+X 전방, +Y 좌측 -- INSTALL_KO.md "현재 PhysiCar source/interface에서는
base_footprint의 +X가 forward, +Y가 left입니다"). `los_drive_node.py` 가
BEV 픽셀에서 미터로 변환하던 단계가 여기서는 필요 없다 -- 좌표계가 이미
같다. 그래서 조향/속도 물리식(순수추종, 횡가속 한계)만 `los_drive_node.py`
에서 그대로 옮겨왔다. 두 노드가 같은 수식을 쓰는 건 우연이 아니라, 인지
방식이 달라도(가로 자유공간 추적 vs MinSeok 님의 ORANGE 중앙선 추적)
제어가 필요로 하는 입력은 결국 "전방 d 미터의 목표점 하나"로 같기 때문이다.
"""

import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped  # noqa: F401  (Path.poses 항목 타입 문서화용)
from nav_msgs.msg import Path
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, Float64

WHEELBASE = 0.18            # m -- 드라이버 계층과 같은 값 (los_drive_node 와 동일)
MAX_STEER = math.radians(20.0)
MIN_SPEED = 0.3             # ESC 불감대
MAX_SPEED = 3.0


class PerceptionV3FollowNode(Node):
    def __init__(self):
        super().__init__('perception_v3_follow_node')

        self.declare_parameter('control_hz', 30.0)

        # 전방주시거리 = ld_k * v, [ld_min, ld_max] 로 자름.
        # los_drive_node 와 같은 상충: 짧으면 코너를 못 보고, 길면
        # atan(2L sin a / l_d) 의 분모가 커져 조향이 약해진다.
        self.declare_parameter('ld_min_m', 0.35)
        self.declare_parameter('ld_max_m', 1.30)
        self.declare_parameter('ld_k', 0.90)
        self.declare_parameter('steer_sign', 1.0)

        # 속도: 횡가속 한계(v <= sqrt(a_lat_max * R))와 보이는 거리로 정한다.
        self.declare_parameter('v_max', 1.20)
        self.declare_parameter('v_min', 0.45)
        self.declare_parameter('a_lat_max', 3.0)
        self.declare_parameter('k_vis', 1.10)
        self.declare_parameter('accel_step', 0.05)
        self.declare_parameter('brake_step', 0.25)

        # perception_v3/path 는 valid=False 여도 빈 poses 로 계속 발행된다
        # (V3Node.process 참고). 그래서 "메시지가 왔다"와 "쓸만하다"를
        # 따로 본다.
        self.declare_parameter('min_path_points', 2)
        self.declare_parameter('input_timeout_s', 0.5)

        # 인지가 잠깐 끊겨도(예: TF pending, 최대 0.25s) 바로 서지 않는다.
        # 마지막 조향을 유지하며 서행한다.
        self.declare_parameter('grace_s', 1.0)
        self.declare_parameter('grace_speed', 0.35)

        # --- 초록 고깔 회피 (cone_bev_node 가 /cones 로 준다) ---
        # 궤적을 따로 만들지 않고 **전방주시점만 옆으로 민다**. 순수추종이
        # 그 점을 향해 호를 그리므로 회피 궤적은 자동으로 나온다.
        self.declare_parameter('avoid_enabled', True)
        # 목표점 앞뒤로 이만큼 안에 있는 고깔만 본다. ld 가 속도에 따라
        # 변하므로 고정 창을 쓰면 빠를 때 놓친다 -> ld 비율로도 넓힌다.
        self.declare_parameter('cone_window_m', 0.35)
        self.declare_parameter('cone_margin_m', 0.12)   # 고깔 옆 최소 여유
        self.declare_parameter('wall_margin_m', 0.10)   # 흰선 앞 최소 여유
        self.declare_parameter('max_offset_m', 0.30)    # 오프셋 절대 상한
        # 흰선을 못 본 쪽을 'BEV 끝까지 뚫려 있다'로 읽으면 안 된다.
        # cone_bev_node 는 못 찾으면 격자 가장자리(±0.75)를 벽으로 보고하는데,
        # 그러면 **검출에 실패한 쪽이 오히려 넓어 보여** 그쪽으로 꺾는다.
        # 실제로 1차선 고깔을 피하려다 좌측 흰선을 넘는 형태로 나타났다.
        # 그래서 벽 위치를 경로 기준 트랙 반폭으로 한 번 더 조인다.
        # 0.37 은 perception_v3.yaml 의 white.expected_half_width 와 같은 값.
        self.declare_parameter('track_half_m', 0.37)
        # 붙을 땐 빠르게, 풀 땐 천천히. 검출이 깜빡여도 좌우로 안 떨리게.
        self.declare_parameter('offset_engage_rate', 1.20)   # m/s
        self.declare_parameter('offset_release_rate', 0.40)  # m/s
        self.declare_parameter('cones_timeout_s', 0.5)

        p = self.get_parameter
        self.control_hz = float(p('control_hz').value)
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
        self.min_path_points = int(p('min_path_points').value)
        self.input_timeout = float(p('input_timeout_s').value)
        self.grace_s = float(p('grace_s').value)
        self.grace_speed = float(p('grace_speed').value)
        self.avoid_enabled = bool(p('avoid_enabled').value)
        self.cone_window_m = float(p('cone_window_m').value)
        self.cone_margin_m = float(p('cone_margin_m').value)
        self.wall_margin_m = float(p('wall_margin_m').value)
        self.max_offset_m = float(p('max_offset_m').value)
        self.track_half_m = float(p('track_half_m').value)
        self.offset_engage_rate = float(p('offset_engage_rate').value)
        self.offset_release_rate = float(p('offset_release_rate').value)
        self.cones_timeout = float(p('cones_timeout_s').value)

        self._path_points = []      # [(x_fwd, y_lat), ...] 근->원 순서
        self._path_valid = False
        self._last_path_time = 0.0
        self._last_ok_time = 0.0
        self._cones = []            # [(x, y, 반폭, 좌벽y, 우벽y), ...]
        self._last_cones_time = 0.0
        self._offset = 0.0          # 지금 적용 중인 횡 오프셋 (m)
        self._speed = 0.0
        self._steer = 0.0
        self._n = 0

        # 발행 측(V3Node)이 depth=10 기본(RELIABLE) QoS 를 쓴다. 여기서
        # qos_profile_sensor_data(BEST_EFFORT)를 쓰면 예전에 겪은 QoS
        # 불일치로 프레임 0개 수신 문제가 그대로 재현된다 -- 반드시 맞춘다.
        self.create_subscription(Path, '/perception_v3/path', self.on_path, 10)
        self.create_subscription(Bool, '/perception_v3/debug/path_valid',
                                 self.on_valid, 10)
        self.create_subscription(Float32MultiArray, '/cones', self.on_cones, 10)
        self.pub_speed = self.create_publisher(Float64, '/speed', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering', 10)
        # 오버레이 노드가 "우리가 어디로 가려는지"를 그릴 수 있게 결정을 흘린다.
        # [x_fwd, y_raw, y_used, steer, offset, valid]
        self.pub_dbg = self.create_publisher(Float32MultiArray,
                                             '/race/avoid_debug', 10)
        self.create_timer(1.0 / self.control_hz, self.tick)
        self.get_logger().info(
            'perception_v3_follow_node 시작 (MinSeok perception_v3 경로 순수추종)')

    # ------------------------------------------------------------ 구독

    def on_path(self, msg):
        self._path_points = [(pose.pose.position.x, pose.pose.position.y)
                             for pose in msg.poses]
        self._last_path_time = time.time()

    def on_valid(self, msg):
        self._path_valid = bool(msg.data)

    def on_cones(self, msg):
        """[x, y, 반폭, 좌벽y, 우벽y] * N 을 풀어 담는다."""
        data = list(msg.data)
        self._cones = [tuple(data[i:i + 5])
                       for i in range(0, len(data) - 4, 5)]
        self._last_cones_time = time.time()

    # ------------------------------------------------------------ 회피

    def avoid_target_y(self, x_fwd, y_lat, ld):
        """목표점 근처에 고깔이 있으면 옮겨야 할 y 를 돌려준다.

        궤적을 따로 만들지 않는다. 순수추종이 목표점을 향해 호를 그리므로,
        **점 하나만 옮기면 회피 궤적은 저절로 나온다.** 대신 그 점이 실제로
        갈 수 있는 곳이어야 하므로 흰선 여유를 같이 본다(넘으면 실격).

        반환: (목표 y, 판단 근거 문자열)
        """
        if not self.avoid_enabled:
            return y_lat, ''
        if (time.time() - self._last_cones_time) > self.cones_timeout:
            return y_lat, ''

        # 목표점 근처 고깔만 본다. 창은 속도가 붙을수록 넓어져야 한다 --
        # ld 가 길어지면 같은 시간에 더 먼 구간을 지나기 때문.
        window = self.cone_window_m + 0.25 * ld
        near = [c for c in self._cones if abs(c[0] - x_fwd) <= window]
        if not near:
            return y_lat, ''

        # 목표점에 가장 가까운(횡으로) 고깔 하나만 처리한다. 여럿이면
        # 가장 방해되는 것부터 -- 그걸 피하면 대개 나머지도 풀린다.
        cone = min(near, key=lambda c: abs(c[1] - y_lat))
        cx, cy, half, wall_left, wall_right = cone

        # 못 본 벽을 '뚫려 있다'로 읽지 않는다. cone_bev_node 는 흰선을
        # 못 찾으면 격자 가장자리를 돌려주므로, 경로 기준 트랙 반폭으로
        # 조인다. min/max 라 실제로 검출된(더 안쪽인) 벽은 그대로 살아남고,
        # 못 찾아서 벌어진 값만 잘린다.
        wall_left = min(wall_left, y_lat + self.track_half_m)
        wall_right = max(wall_right, y_lat - self.track_half_m)

        blocked_lo = cy - half - self.cone_margin_m     # 고깔의 오른쪽 끝
        blocked_hi = cy + half + self.cone_margin_m     # 고깔의 왼쪽 끝
        if not (blocked_lo < y_lat < blocked_hi):
            return y_lat, ''        # 목표점이 이미 고깔 밖이다

        # 좌/우 빈 공간. +Y 가 왼쪽이므로 왼쪽은 y 가 큰 쪽이다.
        left_lo, left_hi = blocked_hi, wall_left - self.wall_margin_m
        right_lo, right_hi = wall_right + self.wall_margin_m, blocked_lo
        left_gap = left_hi - left_lo
        right_gap = right_hi - right_lo

        if left_gap <= 0.0 and right_gap <= 0.0:
            # 양쪽 다 못 간다. 억지로 밀면 흰선을 넘는다 -- 그냥 둔다.
            return y_lat, '양쪽막힘'

        if left_gap >= right_gap:
            return 0.5 * (left_lo + left_hi), '좌(%.2fm)' % left_gap
        return 0.5 * (right_lo + right_hi), '우(%.2fm)' % right_gap

    def step_offset(self, desired):
        """오프셋을 변화율 제한으로 따라가게 한다.

        붙을 땐 빠르게, 풀 땐 천천히. 검출이 한두 프레임 깜빡여도 조향이
        좌우로 떨리지 않게 하려는 것이다. desired 가 0 이 되면 같은 규칙으로
        서서히 되돌아온다.
        """
        rate = (self.offset_engage_rate if abs(desired) > abs(self._offset)
                else self.offset_release_rate)
        step = rate / max(1.0, self.control_hz)
        delta = desired - self._offset
        self._offset += max(-step, min(step, delta))
        self._offset = max(-self.max_offset_m,
                           min(self.max_offset_m, self._offset))
        return self._offset

    # ------------------------------------------------------------ 물리 (los_drive_node 와 동일)

    @staticmethod
    def pure_pursuit(x_fwd, y_lat):
        """애커만 순수추종. delta = atan(2 L sin(alpha) / l_d)."""
        ld = math.hypot(x_fwd, y_lat)
        if ld < 1e-3:
            return 0.0
        alpha = math.atan2(y_lat, x_fwd)
        return math.atan2(2.0 * WHEELBASE * math.sin(alpha), ld)

    def speed_limit(self, steer, visible_m):
        """횡가속 한계와 시야 거리로 속도 상한을 정한다. v <= sqrt(a R)."""
        v = self.v_max
        t = abs(math.tan(steer))
        if t > 1e-4:
            radius = WHEELBASE / t
            v = min(v, math.sqrt(max(0.0, self.a_lat_max * radius)))
        v = min(v, self.k_vis * max(0.0, visible_m))
        return max(self.v_min, min(self.v_max, v))

    # ------------------------------------------------------------ 경로 기하

    @staticmethod
    def lookahead_point(points, lookahead_m):
        """차량 원점(0,0)에서 누적 거리로 전방주시점을 고른다.

        점들은 이미 base_footprint 미터 좌표라 los_drive_node.los_point()
        처럼 행(row)->거리 변환이 필요 없다. 그만큼 더 단순하다.
        """
        if not points:
            return None
        prev = (0.0, 0.0)
        acc = 0.0
        chosen = points[0]
        for pt in points:
            acc += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            chosen = pt
            prev = pt
            if acc >= lookahead_m:
                break
        return chosen

    @staticmethod
    def path_length(points):
        """차량 원점부터 경로 끝까지의 누적 거리 -- '얼마나 멀리 보이는가'."""
        prev = (0.0, 0.0)
        total = 0.0
        for pt in points:
            total += math.hypot(pt[0] - prev[0], pt[1] - prev[1])
            prev = pt
        return total

    # ------------------------------------------------------------ 제어 루프

    def tick(self):
        now = time.time()
        fresh = (now - self._last_path_time) < self.input_timeout
        usable = (fresh and self._path_valid
                 and len(self._path_points) >= self.min_path_points)

        if usable:
            self._last_ok_time = now
            visible_m = self.path_length(self._path_points)
            ld = min(self.ld_max_m,
                    max(self.ld_min_m, self.ld_k * max(self._speed, self.v_min)))
            x_fwd, y_lat = self.lookahead_point(self._path_points, ld)

            # 고깔이 있으면 목표점을 옆으로 민다. 속도는 안 줄인다 --
            # 미리(ld 앞에서) 피하는 것이 이 설계의 전제다.
            desired_y, why = self.avoid_target_y(x_fwd, y_lat, ld)
            y_used = y_lat + self.step_offset(desired_y - y_lat)

            steer = self.pure_pursuit(x_fwd, y_used) * self.steer_sign
            steer = max(-MAX_STEER, min(MAX_STEER, steer))

            v_target = self.speed_limit(steer, visible_m)
            dv = v_target - self._speed
            step = self.accel_step if dv > 0 else self.brake_step
            self._speed = self._speed + max(-step, min(step, dv))
            self._speed = max(MIN_SPEED, min(MAX_SPEED, self._speed))
            self._steer = steer

            self._publish(self._speed, self._steer)
            self.pub_dbg.publish(Float32MultiArray(data=[
                float(x_fwd), float(y_lat), float(y_used),
                float(self._steer), float(self._offset), 1.0]))
            self._log('LOS(%.2fm, %+.2fm) ld=%.2f  보임=%.2fm  '
                      'steer=%+.1f도  v=%.2f (한계 %.2f)  회피=%+.2fm %s'
                      % (x_fwd, y_used, ld, visible_m,
                         math.degrees(self._steer), self._speed, v_target,
                         self._offset, why or '-'))
            return

        since = now - self._last_ok_time
        if self._last_ok_time > 0.0 and since < self.grace_s:
            self._publish(self.grace_speed, self._steer)
            self.get_logger().warn(
                '경로 유실 %.1fs - 마지막 조향 유지 (%.1f도)'
                % (since, math.degrees(self._steer)),
                throttle_duration_sec=0.5)
            return

        self._publish(0.0, 0.0)
        self.get_logger().warn('경로 없음 - 정지', throttle_duration_sec=1.0)

    def _publish(self, speed, steer):
        if speed == 0.0:
            self._speed = 0.0
        self.pub_speed.publish(Float64(data=float(speed)))
        self.pub_steer.publish(Float64(data=float(steer)))

    def _log(self, text):
        self._n += 1
        if self._n % 10:
            return
        self.get_logger().info(text)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionV3FollowNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
