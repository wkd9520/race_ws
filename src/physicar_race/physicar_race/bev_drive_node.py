#!/usr/bin/env python3
"""BEV 인지를 받아 /speed + /steering 을 내는 주행 노드.

bev_lane_node 가 원근 왜곡 없는 값을 주므로 제어가 단순해진다:

    steer = k_lat × 횡오차 + k_head × 헤딩오차

이게 Stanley 제어의 형태다. 화면 좌표에서 하던 때와 달리 두 항의 단위가
일관되므로 게인을 물리적으로 해석할 수 있다.

속도는 곡률에 따라 줄인다. BEV 곡률은 실제 곡률에 비례하므로 임의 계수가 아니라
'얼마나 굽었나'를 그대로 반영한다.
"""

import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool, Float64

MAX_STEER = math.radians(20.0)
MIN_SPEED = 0.3
MAX_SPEED = 3.0


class BevDriveNode(Node):
    def __init__(self):
        super().__init__('bev_drive_node')

        self.declare_parameter('control_hz', 30.0)
        self.declare_parameter('k_lat', 0.7)       # 횡오차 게인
        self.declare_parameter('k_head', 0.5)      # 헤딩 게인
        self.declare_parameter('k_damp', 0.1)      # 미분 (진동 억제)
        self.declare_parameter('steer_sign', 1.0)  # 반대로 꺾이면 -1.0

        self.declare_parameter('v_max', 0.8)
        self.declare_parameter('v_min', 0.35)
        self.declare_parameter('k_curve_slow', 1.2)   # 곡률 감속 강도
        self.declare_parameter('accel_rate', 0.8)
        self.declare_parameter('brake_rate', 2.5)

        # 인지가 끊겨도 바로 서지 않는다. 마지막 조향을 유지하며 서행.
        self.declare_parameter('grace_s', 1.0)
        self.declare_parameter('grace_speed', 0.35)
        self.declare_parameter('input_timeout_s', 0.5)

        p = self.get_parameter
        self.control_hz = float(p('control_hz').value)
        self.k_lat = float(p('k_lat').value)
        self.k_head = float(p('k_head').value)
        self.k_damp = float(p('k_damp').value)
        self.steer_sign = float(p('steer_sign').value)
        self.v_max = float(p('v_max').value)
        self.v_min = float(p('v_min').value)
        self.k_curve_slow = float(p('k_curve_slow').value)
        self.accel_rate = float(p('accel_rate').value)
        self.brake_rate = float(p('brake_rate').value)
        self.grace_s = float(p('grace_s').value)
        self.grace_speed = float(p('grace_speed').value)
        self.input_timeout = float(p('input_timeout_s').value)

        self.valid = False
        self.offset = 0.0
        self.heading = 0.0
        self.curvature = 0.0
        self.stamp = 0.0

        self._prev_err = 0.0
        self._prev_t = time.time()
        self._v_cmd = 0.0
        self._last_steer = 0.0
        self._last_ok = 0.0

        self.create_subscription(Bool, 'bev/valid', self._cb_valid, 10)
        self.create_subscription(Float64, 'bev/offset', self._cb_off, 10)
        self.create_subscription(Float64, 'bev/heading', self._cb_head, 10)
        self.create_subscription(Float64, 'bev/curvature', self._cb_curv, 10)

        self.pub_speed = self.create_publisher(Float64, '/speed', 10)
        self.pub_steer = self.create_publisher(Float64, '/steering', 10)

        self.create_timer(1.0 / self.control_hz, self.tick)
        self.get_logger().info('bev_drive_node 시작')

    def _cb_valid(self, m):
        self.valid = bool(m.data)
        self.stamp = time.time()

    def _cb_off(self, m):
        self.offset = float(m.data)

    def _cb_head(self, m):
        self.heading = float(m.data)

    def _cb_curv(self, m):
        self.curvature = float(m.data)

    def tick(self):
        now = time.time()
        dt = max(1e-3, now - self._prev_t)
        self._prev_t = now

        fresh = self.valid and (now - self.stamp) < self.input_timeout

        if fresh:
            self._last_ok = now
        else:
            since = now - self._last_ok
            if self._last_ok > 0.0 and since < self.grace_s:
                # 인지가 잠깐 끊겨도 마지막 조향을 유지하며 서행한다
                self._publish(self.grace_speed, self._last_steer)
                self.get_logger().warn('인지 유실 %.1fs - 서행 유지' % since,
                                       throttle_duration_sec=0.5)
                return
            self._publish(0.0, 0.0)
            self.get_logger().warn('인지 없음 - 정지', throttle_duration_sec=1.0)
            return

        # ---- 조향: 횡오차 + 헤딩 (Stanley 형태) ----
        # offset + = 기준선이 목표보다 오른쪽 = 차가 왼쪽 -> 오른쪽으로 꺾어야
        err = -self.offset
        derr = (err - self._prev_err) / dt
        self._prev_err = err

        steer = (self.k_lat * err
                 + self.k_head * self.heading
                 + self.k_damp * derr)
        steer = self.steer_sign * steer
        steer = max(-MAX_STEER, min(MAX_STEER, steer))

        # ---- 속도: 곡률에 따라 감속 ----
        v = self.v_max / (1.0 + self.k_curve_slow * abs(self.curvature))
        v = max(self.v_min, min(self.v_max, v))

        dv = v - self._v_cmd
        limit = (self.brake_rate if dv < 0 else self.accel_rate) * dt
        v = self._v_cmd + max(-limit, min(limit, dv))
        v = max(MIN_SPEED, min(MAX_SPEED, v))

        self._v_cmd = v
        self._last_steer = steer
        self._publish(v, steer)

    def _publish(self, speed, steer):
        if speed == 0.0:
            self._v_cmd = 0.0
        self.pub_speed.publish(Float64(data=float(speed)))
        self.pub_steer.publish(Float64(data=float(steer)))


def main(args=None):
    rclpy.init(args=args)
    node = BevDriveNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
