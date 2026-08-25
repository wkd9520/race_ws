#!/usr/bin/env python3
"""출발 절차: 신호등을 보다가, 초록이면 트랙을 보고, 그다음 출발한다.

신호등은 정지선 **오른쪽**에 서 있고 트랙은 **아래**에 있다. 카메라 하나로
둘 다 볼 수 없어서 순서대로 본다.

    AIMING   팬 오른쪽 25도, 틸트 수평.  신호등을 본다.
             |  traffic/light_state == GREEN
    TURNING  팬 0도, 틸트 아래 30도.     트랙으로 돌린다.
             |  joint_states 가 목표에 도달  (또는 시간 초과)
    DRIVING  race/go = True.             주행 로직이 여기서부터 돈다.

━━━ TURNING 단계가 왜 따로 있는가 ━━━

초록을 본 그 순간 카메라는 아직 오른쪽을 보고 있다. 바로 출발하면 서보가
내려가는 동안 인지가 엉뚱한 곳의 BEV 를 내고, 차는 그 값으로 조향한다.
출발 첫 1초가 제일 위험한 구간인데 거기서 쓰레기를 먹는 셈이다.

그래서 **joint_states 로 실제 각도가 목표에 닿았는지 확인하고** 넘어간다.
명령을 보냈다고 카메라가 그 자리에 있는 게 아니다. 피드백이 안 오는
로봇도 있어서 시간 초과를 같이 둔다 -- 영영 안 넘어가는 것보다는 낫다.

━━━ 왜 이 노드가 팬·틸트를 다 갖는가 ━━━

camera_tilt_publisher 와 동시에 돌면 둘이 /camera/tilt 를 서로 다른 값으로
밀어서 카메라가 떨린다. 그래서 이 노드가 켜지면 그쪽은 안 띄운다(런치가
처리). 주행 중에도 이 노드가 트랙 자세를 계속 유지 발행한다.

발행:
  /camera/pan   Float64  라디안
  /camera/tilt  Float64  라디안
  race/go       Bool     출발 허가 (follow 노드가 이걸 보고 움직인다)
  race/phase    String   AIMING | TURNING | DRIVING  (디버깅용)
"""

import math
import time

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String

AIMING = 'AIMING'
TURNING = 'TURNING'
DRIVING = 'DRIVING'

PAN_JOINT = 'camera_pan_joint'
TILT_JOINT = 'camera_tilt_joint'


class StartSequenceNode(Node):
    def __init__(self):
        super().__init__('start_sequence_node')

        # 신호등을 볼 자세. 팬은 ROS 규약대로 **왼쪽이 양수**라 오른쪽은
        # 음수다. 로봇 서보가 반대로 돌면 부호만 뒤집으면 된다.
        self.declare_parameter('aim_pan_degrees', -25.0)
        self.declare_parameter('aim_tilt_degrees', 0.0)

        # 트랙을 볼 자세. 기존 주행에서 쓰던 값이다.
        self.declare_parameter('drive_pan_degrees', 0.0)
        self.declare_parameter('drive_tilt_degrees', -30.0)

        # 목표에 닿았다고 볼 오차. 서보는 정확히 안 멈춘다.
        self.declare_parameter('settle_tolerance_deg', 3.0)
        # 오차 안에 이만큼 연속으로 들어와야 인정한다. 지나가는 길에
        # 한 번 스친 것을 도착으로 읽으면 안 된다.
        self.declare_parameter('settle_samples', 5)
        # joint_states 가 안 오거나 목표에 영영 못 닿을 때의 탈출구.
        # 영영 출발 못 하는 것보다는 낫다.
        self.declare_parameter('turn_timeout_s', 3.0)

        self.declare_parameter('publish_rate_hz', 10.0)
        # 신호등을 무시하고 바로 주행 자세로 갈 때 (주행 테스트용)
        self.declare_parameter('skip_light', False)

        p = self.get_parameter
        self.aim = (math.radians(float(p('aim_pan_degrees').value)),
                    math.radians(float(p('aim_tilt_degrees').value)))
        self.drive = (math.radians(float(p('drive_pan_degrees').value)),
                      math.radians(float(p('drive_tilt_degrees').value)))
        self.tolerance = math.radians(float(p('settle_tolerance_deg').value))
        self.settle_samples = int(p('settle_samples').value)
        self.turn_timeout_s = float(p('turn_timeout_s').value)
        rate = float(p('publish_rate_hz').value)
        if not math.isfinite(rate) or rate <= 0.0:
            raise ValueError('publish_rate_hz 는 유한한 양수여야 한다')

        self.phase = DRIVING if bool(p('skip_light').value) else AIMING
        self._turn_started = 0.0
        self._settled = 0
        self._joints = {}
        self._announced = ''

        self.pub_pan = self.create_publisher(Float64, '/camera/pan', 10)
        self.pub_tilt = self.create_publisher(Float64, '/camera/tilt', 10)
        self.pub_go = self.create_publisher(Bool, 'race/go', 10)
        self.pub_phase = self.create_publisher(String, 'race/phase', 10)

        self.create_subscription(String, 'traffic/light_state',
                                 self.on_light, 10)
        self.create_subscription(JointState, '/joint_states',
                                 self.on_joints, qos_profile_sensor_data)
        self.create_timer(1.0 / rate, self.tick)

        self.get_logger().info(
            'start_sequence_node 시작 -- %s\n'
            '  신호등 자세: 팬 %+.1f도 틸트 %+.1f도\n'
            '  주행 자세  : 팬 %+.1f도 틸트 %+.1f도'
            % (self.phase,
               float(p('aim_pan_degrees').value),
               float(p('aim_tilt_degrees').value),
               float(p('drive_pan_degrees').value),
               float(p('drive_tilt_degrees').value)))

    # ------------------------------------------------------------ 입력

    def on_light(self, msg):
        """초록은 AIMING 에서만 의미가 있다.

        일단 돌기 시작하면 신호등은 다시 안 본다. 주행 중에 신호등이
        시야에 들어왔다고 자세를 되돌리면 그 순간 트랙을 잃는다.
        """
        if self.phase != AIMING or msg.data != 'GREEN':
            return
        self.phase = TURNING
        self._turn_started = time.time()
        self._settled = 0
        self.get_logger().info('초록 신호 확인 -- 카메라를 트랙으로 돌린다')

    def on_joints(self, msg):
        for name, position in zip(msg.name, msg.position):
            self._joints[name] = float(position)

    # ------------------------------------------------------------ 상태

    def at_target(self, target):
        """실제 각도가 목표 오차 안에 있는가. 모르면 None."""
        pan = self._joints.get(PAN_JOINT)
        tilt = self._joints.get(TILT_JOINT)
        if pan is None or tilt is None:
            return None
        return (abs(pan - target[0]) <= self.tolerance
                and abs(tilt - target[1]) <= self.tolerance)

    def turn_done(self):
        """트랙 자세로 다 돌았는가. (완료 여부, 근거) 를 돌려준다."""
        reached = self.at_target(self.drive)
        if reached is None:
            # joint_states 에 카메라 조인트가 없다. 피드백이 없는 로봇이다.
            if time.time() - self._turn_started >= self.turn_timeout_s:
                return True, '피드백 없음, %.1f초 대기 후 진행' % self.turn_timeout_s
            return False, ''
        if reached:
            self._settled += 1
            if self._settled >= self.settle_samples:
                return True, '목표 도달'
            return False, ''
        self._settled = 0
        if time.time() - self._turn_started >= self.turn_timeout_s:
            pan = math.degrees(self._joints.get(PAN_JOINT, float('nan')))
            tilt = math.degrees(self._joints.get(TILT_JOINT, float('nan')))
            return True, ('시간 초과 -- 팬 %.1f도 틸트 %.1f도 에서 진행'
                          % (pan, tilt))
        return False, ''

    # ------------------------------------------------------------ 주기 실행

    def tick(self):
        if self.phase == TURNING:
            done, why = self.turn_done()
            if done:
                self.phase = DRIVING
                self.get_logger().info('카메라 정렬 완료 (%s) -- 출발' % why)

        target = self.aim if self.phase == AIMING else self.drive
        self.pub_pan.publish(Float64(data=target[0]))
        self.pub_tilt.publish(Float64(data=target[1]))
        self.pub_go.publish(Bool(data=self.phase == DRIVING))
        self.pub_phase.publish(String(data=self.phase))

        if self.phase != self._announced:
            self._announced = self.phase
        elif self.phase == AIMING:
            self.get_logger().info('신호 대기 중 (신호등을 보고 있다)',
                                   throttle_duration_sec=3.0)


def main(args=None):
    rclpy.init(args=args)
    node = StartSequenceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
