#!/usr/bin/env python3
"""차선 단위 장애물 점유 판정 노드.

코스 스펙 (2026-08-18 확정): 2차선 중 랜덤하게 장애물이 배치된다.

기존 avoid_node는 '가까운 장애물 반대쪽으로 조향'하는 반응형 회피였다.
이번 코스에서는 그게 최적이 아니다 - 장애물이 차선 단위로 배치되므로,
회피는 연속적인 스티어 오버라이드가 아니라 **이산적인 차선 변경**이다.
반응형 스웨브는 흰 실선을 밟을 위험(=실격)을 키우고 기록도 손해다.

그래서 이 노드는 조향을 직접 만들지 않고, '어느 차선이 막혔는가'라는
사실만 발행한다. 차선 변경 결정은 judgment_node가 한다.

발행 토픽:
  obstacle/blocked_current  Bool     현재 차선 전방이 막힘
  obstacle/blocked_other    Bool     반대 차선 전방이 막힘
  obstacle/emergency        Bool     즉시 정지 필요 (바로 앞 장애물)
  obstacle/nearest_dist     Float64  전방 최근접 거리 [m] (없으면 +inf)

좌표계는 ROS REP-103 차량 프레임: x 전방, y 좌측.
"""

import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float64, Int32

LANE_RIGHT = 0
LANE_LEFT = 1
LANE_UNKNOWN = -1


class LaneObstacleNode(Node):
    def __init__(self):
        super().__init__('lane_obstacle_node')

        # 라이다 장착각 보정. 연습 섀시 실측값 180도.
        # 실차는 다를 수 있으므로 인수 후 반드시 재측정 (캘리브레이션 체크리스트 참고).
        self.declare_parameter('front_offset_deg', 180.0)

        # 전방 관심 시야각 (이 범위 밖은 무시)
        self.declare_parameter('front_fov_deg', 120.0)

        # 차선 폭 [m]. 8/18 공개된 코스 규격의 실제 값으로 반드시 교체할 것.
        self.declare_parameter('lane_width_m', 0.50)

        # 차선 점유로 볼 횡방향 반폭 비율 (lane_width_m 대비).
        # 1.0으로 하면 차선 경계에 스친 점까지 잡아 과민해진다.
        self.declare_parameter('lane_occupy_frac', 0.8)

        # 이 거리 안의 장애물만 차선 판정에 쓴다. 너무 길면 커브 바깥 벽을 장애물로 오인한다.
        self.declare_parameter('lookahead_m', 2.0)

        # 즉시 정지 판정
        self.declare_parameter('emergency_dist_m', 0.35)
        self.declare_parameter('emergency_half_width_m', 0.12)

        # 라이다 유효 범위
        self.declare_parameter('range_min_m', 0.10)
        self.declare_parameter('range_max_m', 8.0)

        # 노이즈 1점으로 차선이 막혔다고 판단하면 안 된다
        self.declare_parameter('min_points', 3)

        p = self.get_parameter
        self.front_offset = math.radians(float(p('front_offset_deg').value))
        self.front_fov = math.radians(float(p('front_fov_deg').value))
        self.lane_width = float(p('lane_width_m').value)
        self.lane_occupy_frac = float(p('lane_occupy_frac').value)
        self.lookahead = float(p('lookahead_m').value)
        self.emergency_dist = float(p('emergency_dist_m').value)
        self.emergency_half_w = float(p('emergency_half_width_m').value)
        self.range_min = float(p('range_min_m').value)
        self.range_max = float(p('range_max_m').value)
        self.min_points = int(p('min_points').value)

        self.current_lane = LANE_UNKNOWN

        # avoid_node 와 같은 규약: 상대 토픽명 'scan' + 센서 QoS(BEST_EFFORT).
        # 기본 QoS(RELIABLE)면 BEST_EFFORT 퍼블리셔와 매칭이 안 돼 스캔이 안 들어온다.
        self.create_subscription(LaserScan, 'scan', self.on_scan, qos_profile_sensor_data)
        self.create_subscription(Int32, 'lane/current_lane', self.on_lane, 10)

        self.pub_cur = self.create_publisher(Bool, 'obstacle/blocked_current', 10)
        self.pub_oth = self.create_publisher(Bool, 'obstacle/blocked_other', 10)
        self.pub_emg = self.create_publisher(Bool, 'obstacle/emergency', 10)
        self.pub_near = self.create_publisher(Float64, 'obstacle/nearest_dist', 10)

        self.get_logger().info('lane_obstacle_node 시작')

    def on_lane(self, msg: Int32):
        self.current_lane = int(msg.data)

    def on_scan(self, msg: LaserScan):
        n = len(msg.ranges)
        if n == 0:
            return

        r = np.asarray(msg.ranges, dtype=np.float32)
        ang = msg.angle_min + np.arange(n, dtype=np.float32) * msg.angle_increment
        ang = ang - self.front_offset

        # 유효 거리 + 전방 시야만
        ok = np.isfinite(r) & (r > self.range_min) & (r < self.range_max)
        ang_wrapped = np.arctan2(np.sin(ang), np.cos(ang))
        ok &= np.abs(ang_wrapped) < (self.front_fov * 0.5)
        if not np.any(ok):
            self._publish(False, False, False, float('inf'))
            return

        r = r[ok]
        a = ang_wrapped[ok]
        x = r * np.cos(a)
        y = r * np.sin(a)

        ahead = (x > 0.0) & (x < self.lookahead)
        if not np.any(ahead):
            self._publish(False, False, False, float('inf'))
            return

        x, y = x[ahead], y[ahead]

        half = 0.5 * self.lane_width * self.lane_occupy_frac

        # 현재 차선 중심은 y=0 (차가 차선 중앙을 따라간다는 전제).
        # 반대 차선 중심은 좌/우 어느 쪽인지에 따라 부호가 갈린다 (y는 좌측이 +).
        if self.current_lane == LANE_RIGHT:
            other_y = +self.lane_width
        elif self.current_lane == LANE_LEFT:
            other_y = -self.lane_width
        else:
            other_y = None  # 차선을 모르면 반대 차선 판정 불가

        cur_hits = int(np.count_nonzero(np.abs(y) <= half))
        if other_y is None:
            oth_hits = 0
        else:
            oth_hits = int(np.count_nonzero(np.abs(y - other_y) <= half))

        emg_mask = (x < self.emergency_dist) & (np.abs(y) < self.emergency_half_w)
        emergency = int(np.count_nonzero(emg_mask)) >= self.min_points

        nearest = float(np.min(x)) if x.size else float('inf')

        self._publish(
            cur_hits >= self.min_points,
            (other_y is not None) and (oth_hits >= self.min_points),
            emergency,
            nearest,
        )

    def _publish(self, blocked_cur, blocked_oth, emergency, nearest):
        self.pub_cur.publish(Bool(data=bool(blocked_cur)))
        self.pub_oth.publish(Bool(data=bool(blocked_oth)))
        self.pub_emg.publish(Bool(data=bool(emergency)))
        self.pub_near.publish(Float64(data=float(nearest)))


def main(args=None):
    rclpy.init(args=args)
    node = LaneObstacleNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
