#!/usr/bin/env python3
"""레이스 판단(중재) 노드 - 최종 /speed + /steering 발행.

이 워크스페이스에서 /speed + /steering 을 내는 유일한 노드다. 다른 판단
노드나 드라이버 노드를 같이 띄우면 명령이 충돌하니 하나만 실행할 것.

코스 스펙 (2026-08-18 확정)에서 도출한 목표 함수:
  1. 흰색 실선 밖으로 나가면 실격      -> 절대 위반 불가 (HARD)
  2. 랜덤 배치 장애물 충돌 회피         -> 차선 변경으로 해결
  3. 출발 신호등: 빨강 정지, 초록 출발  -> 출발 시점 1회성 게이트
  4. 최단 기록                          -> 위 셋을 지키는 선에서 최대 속도

우선순위: 실격 회피(흰선) > 충돌 회피 > 출발 게이트 > 기록 단축.
1번이 2번보다 위에 있는 게 핵심이다. 장애물을 피하려다 흰선을 넘으면
사고가 아니라 실격이므로, 회피 조향은 항상 흰선 여유의 제약을 받는다.

레이스 상태기계:
  WAIT_GREEN --(초록 확인)--> RACING <--> EMERGENCY

출발 게이트를 '래치'로 만든 이유:
기존 judgment_node는 RED가 보이면 언제든 정지했다. 이번 코스는 신호등이
출발 시점에만 존재하므로, 한 번 초록을 보고 출발했으면 그 뒤로는 신호등
인지 결과를 아예 보지 않는다. 그래야 주행 중 빨간색 물체(다른 차, 표지,
관중 옷)를 신호등으로 오인해서 트랙 한복판에 서는 사고가 없다.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, Int32, String

# 실차 드라이버(SDK 계층) 확정 스펙 - 2026-08-16 공식 소스코드로 확인.
# 초과값은 어차피 드라이버가 클램프하지만, 우리 쪽에서 미리 맞춰야
# 적분/미분 항이 포화 구간에서 엉키지 않는다.
MAX_SPEED = 3.0            # m/s
MIN_SPEED = 0.3            # m/s, ESC 데드존. 이보다 작으면 사실상 무반응
MAX_STEER = math.radians(20.0)   # rad

LANE_RIGHT = 0
LANE_LEFT = 1
LANE_UNKNOWN = -1

ST_WAIT_GREEN = 'WAIT_GREEN'
ST_RACING = 'RACING'
ST_EMERGENCY = 'EMERGENCY'


def other_lane(lane):
    if lane == LANE_RIGHT:
        return LANE_LEFT
    if lane == LANE_LEFT:
        return LANE_RIGHT
    return LANE_UNKNOWN


class RaceJudgmentNode(Node):
    def __init__(self):
        super().__init__('race_judgment_node')

        # --- 제어 주기 ---
        # 실차 cmd_timeout 1초. 그 안에 갱신 안 되면 speed가 자동 0이 되므로
        # 최소 1Hz보다 충분히 빠르게 계속 퍼블리시해야 한다.
        self.declare_parameter('control_hz', 30.0)

        # --- 차선 추종 게인 ---
        self.declare_parameter('kp', 0.55)
        self.declare_parameter('kd', 0.12)

        # --- 곡률 피드포워드 ---
        # 현재 횡오차에 대한 P 제어만으로는 일정 곡률 구간에서 정상상태 오차가
        # 남는다. 오차가 생겨야 조향이 나오므로 커브 안쪽을 못 물고 바깥으로
        # 밀린다. 곡률을 보고 '미리' 꺾어주는 항을 더한다.
        self.declare_parameter('k_ff', 0.55)

        # --- 선행 감속 ---
        # 곡률을 원거리에서 읽어 커브 진입 '전에' 줄인다. 지금 곡률만 보고
        # 반응하면 이미 커브 안이라 늦다.
        self.declare_parameter('a_lat_max', 1.5)      # 허용 횡가속 [m/s^2]
        self.declare_parameter('r_min_m', 0.49)       # 조향 한계 회전반경
        self.declare_parameter('brake_rate', 3.0)     # 감속 추종 속도 [m/s per s]
        self.declare_parameter('accel_rate', 1.0)     # 가속 복귀 속도 (급가속 금지)
        # 실차에서 반대로 돌면 -1.0으로 플립 (캘리브레이션 체크리스트 항목)
        self.declare_parameter('lane_steer_sign', 1.0)

        # --- 흰선 실격 방지 ---
        # margin이 이 값 밑으로 내려가면 무조건 안쪽으로 밀어낸다.
        self.declare_parameter('margin_crit', 0.18)
        self.declare_parameter('k_white', 1.6)
        self.declare_parameter('white_speed_factor', 0.55)

        # --- 속도 프로파일 ---
        self.declare_parameter('v_max', 2.2)      # 초반엔 3.0 미만으로 두고 올릴 것
        self.declare_parameter('k_curve', 0.75)   # 곡률에 따른 감속
        self.declare_parameter('k_offset', 0.45)  # 횡오차가 크면 감속
        self.declare_parameter('lane_change_speed_factor', 0.7)
        # 전방 장애물 접근 시 감속: v <= nearest_dist * k_approach
        self.declare_parameter('k_approach', 1.2)

        # --- 차선 변경 ---
        self.declare_parameter('lane_change_cooldown_s', 1.2)
        self.declare_parameter('lane_change_timeout_s', 3.0)

        # --- 출발 게이트 ---
        self.declare_parameter('require_green', True)
        # 단발 오검출로 출발하면 빨간불 출발 페널티. 연속 확인을 요구한다.
        # traffic_light_node 가 이미 confirm_frames_go 로 한 번 걸러주므로
        # 여기서는 최소한만 더 본다. 총 지연 = 그쪽 확인 + 이 값.
        self.declare_parameter('green_confirm_frames', 2)

        # --- 인지 유실 유예 ---
        # 90도 코너에서는 경계선이 화면을 가로로 가로질러 열 히스토그램에 안 잡힌다.
        # 그때 즉시 정지하면 코너 입구에서 서버려 아예 못 돈다. 직전까지 인지가
        # 되고 있었다면 마지막 조향을 유지한 채 느리게 통과시킨다.
        #
        # 안전 근거: 유예 동안은 속도를 크게 낮추므로 흰선을 넘더라도 이동량이
        # 작고, 유예가 끝나면 정지한다. '영원히 눈감고 달리는' 것과는 다르다.
        self.declare_parameter('lane_grace_s', 1.2)
        self.declare_parameter('grace_speed', 0.45)

        # --- 워치독 ---
        self.declare_parameter('lane_timeout_s', 0.5)
        self.declare_parameter('scan_timeout_s', 0.5)

        p = self.get_parameter
        self.control_hz = float(p('control_hz').value)
        self.kp = float(p('kp').value)
        self.kd = float(p('kd').value)
        self.k_ff = float(p('k_ff').value)
        self.a_lat_max = float(p('a_lat_max').value)
        self.r_min_m = float(p('r_min_m').value)
        self.brake_rate = float(p('brake_rate').value)
        self.accel_rate = float(p('accel_rate').value)
        self._v_cmd = 0.0        # 속도 명령 이력 (급가속 억제용)
        self.steer_sign = float(p('lane_steer_sign').value)
        self.margin_crit = float(p('margin_crit').value)
        self.k_white = float(p('k_white').value)
        self.white_speed_factor = float(p('white_speed_factor').value)
        self.v_max = float(p('v_max').value)
        self.k_curve = float(p('k_curve').value)
        self.k_offset = float(p('k_offset').value)
        self.lane_change_speed_factor = float(p('lane_change_speed_factor').value)
        self.k_approach = float(p('k_approach').value)
        self.lane_change_cooldown = float(p('lane_change_cooldown_s').value)
        self.lane_change_timeout = float(p('lane_change_timeout_s').value)
        self.require_green = bool(p('require_green').value)
        self.green_confirm_frames = int(p('green_confirm_frames').value)
        self.lane_grace_s = float(p('lane_grace_s').value)
        self.grace_speed = float(p('grace_speed').value)
        self.lane_timeout = float(p('lane_timeout_s').value)
        self._last_ok_stamp = 0.0     # 마지막으로 인지가 성립한 시각
        self._last_steer = 0.0        # 유예 중 유지할 조향
        self.scan_timeout = float(p('scan_timeout_s').value)

        # --- 인지 입력 상태 ---
        self.lane_valid = False
        self.lane_stamp = 0.0
        self.off_right = 0.0
        self.off_left = 0.0
        self.current_lane = LANE_UNKNOWN
        self.margin_l = 1.0
        self.margin_r = 1.0
        self.curvature = 0.0

        self.traffic_state = 'NONE'
        self.traffic_valid = False

        self.blocked_current = False
        self.blocked_other = False
        self.emergency = False
        self.nearest = float('inf')
        self.scan_stamp = 0.0

        # --- 레이스 상태 ---
        self.state = ST_WAIT_GREEN if self.require_green else ST_RACING
        self.green_count = 0
        self.target_lane = LANE_UNKNOWN
        self.change_start = 0.0
        self.last_change = 0.0
        self.prev_err = 0.0
        self.prev_t = time.time()

        # --- 구독 ---
        self.create_subscription(Bool, 'lane/valid', self._cb_lane_valid, 10)
        self.create_subscription(Float64, 'lane/offset_right', self._cb_off_r, 10)
        self.create_subscription(Float64, 'lane/offset_left', self._cb_off_l, 10)
        self.create_subscription(Int32, 'lane/current_lane', self._cb_lane, 10)
        self.create_subscription(Float64, 'lane/margin_left', self._cb_margin_l, 10)
        self.create_subscription(Float64, 'lane/margin_right', self._cb_margin_r, 10)
        self.create_subscription(Float64, 'lane/curvature', self._cb_curv, 10)

        self.create_subscription(String, 'traffic/light_state', self._cb_light, 10)
        self.create_subscription(Bool, 'traffic/valid', self._cb_traffic_valid, 10)

        self.create_subscription(Bool, 'obstacle/blocked_current', self._cb_blk_cur, 10)
        self.create_subscription(Bool, 'obstacle/blocked_other', self._cb_blk_oth, 10)
        self.create_subscription(Bool, 'obstacle/emergency', self._cb_emg, 10)
        self.create_subscription(Float64, 'obstacle/nearest_dist', self._cb_near, 10)

        # 벤치 테스트/수동 출발용 오버라이드
        self.create_subscription(Bool, 'race/start', self._cb_manual_start, 10)

        # --- 발행 ---
        self.pub_speed = self.create_publisher(Float64, '/speed', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering', 10)
        self.pub_state = self.create_publisher(String, 'race/state', 10)

        self.timer = self.create_timer(1.0 / self.control_hz, self.control_tick)
        self.get_logger().info('race_judgment_node 시작 - 초기 상태 %s' % self.state)

    # ------------------------------------------------------------ 구독 콜백

    def _cb_lane_valid(self, m):
        self.lane_valid = bool(m.data)
        self.lane_stamp = time.time()

    def _cb_off_r(self, m):
        self.off_right = float(m.data)

    def _cb_off_l(self, m):
        self.off_left = float(m.data)

    def _cb_lane(self, m):
        self.current_lane = int(m.data)

    def _cb_margin_l(self, m):
        self.margin_l = float(m.data)

    def _cb_margin_r(self, m):
        self.margin_r = float(m.data)

    def _cb_curv(self, m):
        self.curvature = float(m.data)

    def _cb_light(self, m):
        self.traffic_state = str(m.data)

    def _cb_traffic_valid(self, m):
        self.traffic_valid = bool(m.data)

    def _cb_blk_cur(self, m):
        self.blocked_current = bool(m.data)
        self.scan_stamp = time.time()

    def _cb_blk_oth(self, m):
        self.blocked_other = bool(m.data)

    def _cb_emg(self, m):
        self.emergency = bool(m.data)

    def _cb_near(self, m):
        self.nearest = float(m.data)

    def _cb_manual_start(self, m):
        if bool(m.data) and self.state == ST_WAIT_GREEN:
            self.get_logger().warn('수동 출발 명령 수신 - 신호등 게이트 건너뜀')
            self.state = ST_RACING

    def _lane_stall_reason(self, now):
        """차가 안 가는 이유를 셋으로 갈라 말한다.

        '차선 인지 유실' 한 줄로 뭉뚱그리면 셋을 구분할 수 없다. 특히 런치
        직후에는 아직 첫 메시지가 안 온 것뿐인데 유실로 읽혀 엉뚱한 데를
        뒤지게 된다. 조치가 각각 다르므로 원인을 분리해서 말한다.
        """
        if self.lane_stamp == 0.0:
            return ('차선 인지 입력 대기 중 - lane/valid 를 아직 한 번도 못 받음. '
                    'lane_detect_node 가 떴는지, 카메라 토픽 이름(image_topic)이 '
                    '맞는지 확인할 것')
        age = now - self.lane_stamp
        if age >= self.lane_timeout:
            return ('차선 입력 끊김 - 마지막 수신 %.1f초 전 (허용 %.1f초). '
                    '카메라 프레임이 멈췄거나 노드가 죽었을 수 있다'
                    % (age, self.lane_timeout))
        return ('차선 미검출 - lane/valid=false, 흰선을 못 찾는 중. '
                'debug_probe:=true 로 원인 확인할 것')

    # ------------------------------------------------------------ 제어 루프

    def control_tick(self):
        now = time.time()
        dt = max(1e-3, now - self.prev_t)
        self.prev_t = now

        lane_fresh = self.lane_valid and (now - self.lane_stamp) < self.lane_timeout
        scan_fresh = (now - self.scan_stamp) < self.scan_timeout

        # ---- 1. 출발 게이트 (래치) ----
        if self.state == ST_WAIT_GREEN:
            if self.traffic_valid and self.traffic_state == 'GREEN':
                self.green_count += 1
            else:
                self.green_count = 0

            if self.green_count >= self.green_confirm_frames:
                self.get_logger().info('초록 확인 - 출발')
                self.state = ST_RACING
            else:
                self._publish(0.0, 0.0)
                return

        # 여기 아래로는 이미 출발한 상태. 신호등은 더 이상 보지 않는다.

        # ---- 2. 정지 사유 판정 ----
        if scan_fresh and self.emergency:
            self.state = ST_EMERGENCY
            self._publish(0.0, 0.0)
            return

        if lane_fresh:
            self._last_ok_stamp = now
        else:
            # 90도 코너에서는 경계선이 화면을 가로로 가로질러 열 히스토그램에
            # 안 잡힌다. 즉시 정지하면 코너 입구에서 서버려 아예 못 돈다.
            # 직전까지 되고 있었다면 마지막 조향을 유지한 채 느리게 통과시킨다.
            since_ok = now - self._last_ok_stamp
            if self._last_ok_stamp > 0.0 and since_ok < self.lane_grace_s:
                self.get_logger().warn(
                    '차선 일시 유실 %.1fs -- 마지막 조향 유지하며 서행 (유예 %.1fs)'
                    % (since_ok, self.lane_grace_s), throttle_duration_sec=0.5)
                self._publish(self.grace_speed, self._last_steer)
                return
            # 유예를 넘겼으면 실격 위험을 통제할 수 없다 -> 정지
            self.state = ST_EMERGENCY
            self.get_logger().warn(self._lane_stall_reason(now), throttle_duration_sec=1.0)
            self._publish(0.0, 0.0)
            return

        if scan_fresh and self.blocked_current and self.blocked_other:
            # 양쪽 다 막힘. 갈 곳이 없으므로 감속 정지하고 다음 틱에 재평가.
            self.state = ST_EMERGENCY
            self._publish(0.0, 0.0)
            return

        self.state = ST_RACING

        # ---- 3. 목표 차선 결정 ----
        target = self._decide_target_lane(now, scan_fresh)

        # ---- 4. 조향 ----
        if target == LANE_LEFT:
            off = self.off_left
        else:
            # RIGHT 또는 UNKNOWN. UNKNOWN이면 인지 노드가 두 값을 같게 낸다(통로 중심).
            off = self.off_right

        err = -off                       # + = 왼쪽으로 틀어야 함
        derr = (err - self.prev_err) / dt
        self.prev_err = err

        # 곡률 피드포워드: 오차가 생기기 전에 미리 꺾는다.
        # curvature 는 + 가 우커브인데 조향은 + 가 좌회전이라 부호를 뒤집는다.
        steer = self.kp * err + self.kd * derr - self.k_ff * self.curvature

        # 흰선 실격 방지: 다른 모든 항 위에 얹는 하드 제약
        margin = min(self.margin_l, self.margin_r)
        if margin < self.margin_crit:
            push = (self.margin_crit - margin) / self.margin_crit
            push = max(0.0, min(1.0, push))
            if self.margin_l < self.margin_r:
                steer -= self.k_white * push   # 좌측 흰선이 가까움 -> 우측으로
            else:
                steer += self.k_white * push   # 우측 흰선이 가까움 -> 좌측으로

        steer = self.steer_sign * steer
        steer = max(-MAX_STEER, min(MAX_STEER, steer))

        # ---- 5. 속도 프로파일 ----
        # 곡률에서 안전 속도를 물리로 구한다. 정규화 곡률 1.0 을 최소 회전반경
        # (조향 한계, 0.49m)에 대응시켜 반경을 추정하고 v = sqrt(a_lat * R).
        # 곱셈식 감속 계수는 '얼마나 줄여야 안전한가'와 무관한 임의값이었다.
        curv = min(1.0, abs(self.curvature))
        r_est = self.r_min_m / max(curv, 1e-3)
        v = min(self.v_max, math.sqrt(max(0.0, self.a_lat_max * r_est)))
        v *= max(0.35, 1.0 - self.k_offset * min(1.0, abs(off)))

        if target != self.current_lane and self.current_lane != LANE_UNKNOWN:
            v *= self.lane_change_speed_factor

        if margin < self.margin_crit:
            v *= self.white_speed_factor

        if scan_fresh and self.nearest != float('inf'):
            v = min(v, max(MIN_SPEED, self.k_approach * self.nearest))

        v = max(MIN_SPEED, min(MAX_SPEED, v))

        # 변화율 제한. 감속은 빠르게 허용하고 가속은 천천히 -- 커브 탈출에서
        # 급가속하면 다음 커브 진입 속도가 다시 높아져 같은 문제가 반복된다.
        dv = v - self._v_cmd
        limit = (self.brake_rate if dv < 0.0 else self.accel_rate) * dt
        v = self._v_cmd + max(-limit, min(limit, dv))
        v = max(MIN_SPEED, min(MAX_SPEED, v))
        self._v_cmd = v
        self._last_steer = steer

        self._publish(v, steer)

    def _decide_target_lane(self, now, scan_fresh):
        """차선 변경은 이산 결정이다. 진동을 막기 위해 쿨다운/타임아웃을 건다."""
        cur = self.current_lane

        if cur == LANE_UNKNOWN:
            # 노란선을 못 봄 -> 차선 개념이 없다. 통로 중앙 유지만 한다.
            self.target_lane = LANE_UNKNOWN
            return LANE_UNKNOWN

        # 변경 진행 중인가?
        if self.target_lane not in (LANE_UNKNOWN, cur):
            if (now - self.change_start) < self.lane_change_timeout:
                return self.target_lane
            # 시간 내 못 넘었으면 포기하고 현재 차선 유지
            self.get_logger().warn('차선 변경 타임아웃 - 현재 차선 유지')
            self.target_lane = cur
            return cur

        # 차선 유지가 기본
        self.target_lane = cur

        if not scan_fresh:
            return cur

        if self.blocked_current and not self.blocked_other:
            if (now - self.last_change) >= self.lane_change_cooldown:
                nxt = other_lane(cur)
                self.get_logger().info('현재 차선 막힘 - %s 로 차선 변경'
                                       % ('LEFT' if nxt == LANE_LEFT else 'RIGHT'))
                self.target_lane = nxt
                self.change_start = now
                self.last_change = now
                return nxt

        return cur

    def _publish(self, speed, steer):
        # 정지 경로로 빠졌으면 속도 이력도 0 으로 리셋한다. 안 그러면 재출발할 때
        # 변화율 제한이 정지 전 속도에서 출발한다고 착각한다.
        if speed == 0.0:
            self._v_cmd = 0.0
        self.pub_speed.publish(Float64(data=float(speed)))
        self.pub_steer.publish(Float64(data=float(steer)))
        self.pub_state.publish(String(data=self.state))


def main(args=None):
    rclpy.init(args=args)
    node = RaceJudgmentNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
