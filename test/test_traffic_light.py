"""출발 절차: 신호등을 보고, 카메라를 돌리고, 그다음 출발한다.

신호등은 정지선 오른쪽, 트랙은 아래. 카메라 하나로 둘 다 못 봐서
순서대로 본다.

    AIMING   팬 오른쪽 25도, 틸트 수평 -- 신호등을 본다
    TURNING  팬 0도, 틸트 아래 30도    -- 트랙으로 돌린다
    DRIVING  race/go = True            -- 주행 시작

여기서 막으려는 실패 넷. **전부 대회에서 끝장이다**:

  1. 빨간불에 출발한다                    -> 실격
  2. 카메라가 다 돌기 전에 출발한다       -> 엉뚱한 BEV 로 조향, 출발 직후 이탈
  3. 초록인데 영영 안 움직인다            -> 완주 못 함
  4. 초록 고깔을 초록불로 읽는다          -> 빨간불에 출발 (1번과 같은 결말)

2번이 이 설계의 핵심이다. 초록을 본 그 순간 카메라는 아직 오른쪽을 보고
있다. 명령을 보냈다고 카메라가 그 자리에 있는 게 아니라서, joint_states
로 실제 각도를 확인하고 나서야 출발 허가를 낸다.

3번은 조용히 온다. 오류도 로그도 없이 차가 그냥 서 있는다.
"""
import ast
import io
import math
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.pardir, 'src', 'physicar_race')
LIGHT = os.path.join(SRC, 'physicar_race', 'traffic_light_node.py')
START = os.path.join(SRC, 'physicar_race', 'start_sequence_node.py')
FOLLOW = os.path.join(SRC, 'physicar_race', 'perception_v3_follow_node.py')
LAUNCH = os.path.join(SRC, 'launch', 'perception_v3_race_launch.py')
SETUP = os.path.join(SRC, 'setup.py')

FAILS = []


def check(label, cond, detail=''):
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % ('PASS' if cond else 'FAIL', label, detail))


def read(path):
    return io.open(path, encoding='utf-8').read()


light_src, start_src = read(LIGHT), read(START)
follow_src, launch_src = read(FOLLOW), read(LAUNCH)


def borrow(source, names, extra=None):
    """소스에서 함수를 그대로 떼어내 실행 가능한 상태로 돌려준다.

    테스트가 로직을 베껴 적으면, 코드가 바뀌어도 테스트는 옛날 로직을
    검사하며 계속 통과한다. 그래서 실제 구현을 가져다 돌린다.
    """
    tree = ast.parse(source)
    picked = [n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = dict(extra or {})
    module = ast.Module(body=picked, type_ignores=[])
    exec(compile(module, '<borrowed>', 'exec'), namespace)
    return namespace


# ============================================================ 초록 원 검출

print('\n[1] 초록 고깔을 초록불로 읽지 않는가 ★')
# 고깔 HSV(40~85)와 신호등 초록(40~90)이 거의 겹친다. 모양으로 가른다.
shape_ns = borrow(light_src, {'_round_enough'})


class Shape:
    def __init__(self, require=True, fill=0.60, lo=0.60, hi=1.70):
        self.require_circle, self.min_fill_ratio = require, fill
        self.min_aspect, self.max_aspect = lo, hi
    _round_enough = shape_ns['_round_enough']


def blob(kind, size=20):
    pad = size // 2
    img = np.zeros((size * 2 + pad, size * 2 + pad), np.uint8)
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    if kind == 'circle':
        cv2.circle(img, (cx, cy), size // 2, 255, -1)
    else:
        cv2.fillPoly(img, [np.array(
            [[cx, cy - size // 2], [cx - size // 3, cy + size // 2],
             [cx + size // 3, cy + size // 2]], np.int32)], 255)
    n, _, stats, _ = cv2.connectedComponentsWithStats(img, connectivity=8)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]))


for kind, want in (('circle', True), ('cone', False)):
    w, h, area = blob(kind)
    ok, why = Shape._round_enough(Shape(), w, h, area)
    check('원은 통과한다' if want else '고깔은 걸러낸다 ★', ok == want,
          '(%dx%d 채움 %.2f%s)'
          % (w, h, area / float(w * h), '' if ok else ' -> ' + why))
w, h, area = blob('cone')
check('require_circle=False 로 되돌릴 수 있다',
      Shape._round_enough(Shape(require=False), w, h, area)[0])


# ============================================================ 출발 상태기계

print('\n[2] 카메라가 다 돌기 전에는 출발하지 않는가 ★★')

class Msg:
    """Float64 / Bool / String 자리를 대신한다. 셋 다 .data 뿐이다."""

    def __init__(self, data):
        self.data = data


seq_ns = borrow(start_src, {'on_light', 'on_joints', 'at_target',
                            'turn_done', 'tick'},
                {'time': time, 'math': math,
                 'AIMING': 'AIMING', 'TURNING': 'TURNING',
                 'DRIVING': 'DRIVING',
                 'PAN_JOINT': 'camera_pan_joint',
                 'TILT_JOINT': 'camera_tilt_joint',
                 'Float64': Msg, 'Bool': Msg, 'String': Msg})


class Log:
    def info(self, *a, **k):
        pass


class Pub:
    def __init__(self, sink, key):
        self.sink, self.key = sink, key

    def publish(self, msg):
        self.sink[self.key] = msg.data


class FakeSeq:
    """진짜 메서드를 그대로 붙인 껍데기. ROS 없이 상태만 돌린다."""

    def __init__(self, feedback=True):
        self.aim = (math.radians(-25.0), 0.0)
        self.drive = (0.0, math.radians(-30.0))
        self.tolerance = math.radians(3.0)
        self.settle_samples = 5
        self.turn_timeout_s = 3.0
        self.phase = 'AIMING'
        self._turn_started = 0.0
        self._settled = 0
        self._joints = {}
        self._announced = ''
        self.sent = {}
        self.pub_pan = Pub(self.sent, 'pan')
        self.pub_tilt = Pub(self.sent, 'tilt')
        self.pub_go = Pub(self.sent, 'go')
        self.pub_phase = Pub(self.sent, 'phase')
        self._feedback = feedback
        if feedback:                       # 처음엔 신호등 자세에 있다
            self.set_angles(*self.aim)

    def get_logger(self):
        return Log()

    def set_angles(self, pan, tilt):
        self._joints = {'camera_pan_joint': pan, 'camera_tilt_joint': tilt}

    on_light = seq_ns['on_light']
    on_joints = seq_ns['on_joints']
    at_target = seq_ns['at_target']
    turn_done = seq_ns['turn_done']
    tick = seq_ns['tick']


seq = FakeSeq()
seq.tick()
check('시작하면 신호등을 본다 (팬 오른쪽 25도)',
      abs(math.degrees(seq.sent['pan']) + 25.0) < 1e-6,
      '(팬 %+.1f도)' % math.degrees(seq.sent['pan']))
check('  이때 출발 허가는 안 나온다 ★', seq.sent['go'] is False)

seq.on_light(Msg('RED'))
seq.tick()
check('빨간불에는 안 움직인다 ★',
      seq.phase == 'AIMING' and seq.sent['go'] is False)

seq.on_light(Msg('NONE'))
seq.tick()
check('신호등이 안 보여도 안 움직인다', seq.sent['go'] is False)

seq.on_light(Msg('GREEN'))
seq.tick()
check('초록이면 카메라를 트랙으로 돌린다',
      seq.phase == 'TURNING'
      and abs(math.degrees(seq.sent['tilt']) + 30.0) < 1e-6,
      '(틸트 %+.1f도)' % math.degrees(seq.sent['tilt']))
check('  돌리는 중에는 아직 출발 안 한다 ★★', seq.sent['go'] is False)

# 서보가 지나가는 길에 목표를 한 번 스치는 경우
seq.set_angles(0.0, math.radians(-30.0))
seq.tick()
check('  한 번 스친 것으로는 출발 안 한다 (연속 확인)',
      seq.sent['go'] is False, '(%d/%d회)' % (seq._settled, seq.settle_samples))

for _ in range(seq.settle_samples):
    seq.tick()
check('목표에 안정적으로 도달하면 출발한다 ★',
      seq.phase == 'DRIVING' and seq.sent['go'] is True)

seq.on_light(Msg('RED'))
seq.tick()
check('출발한 뒤에는 신호등을 다시 안 본다',
      seq.phase == 'DRIVING' and seq.sent['go'] is True)


print('\n[3] 초록인데 영영 안 움직이는 일은 없는가 ★')
# 서보가 목표에 못 닿아도, joint_states 가 아예 안 와도 결국 출발해야 한다.
stuck = FakeSeq()
stuck.on_light(Msg('GREEN'))
stuck.set_angles(math.radians(-25.0), 0.0)      # 서보가 안 움직인다
stuck.tick()
check('서보가 안 움직여도 바로는 출발 안 한다', stuck.sent['go'] is False)
stuck._turn_started -= stuck.turn_timeout_s + 0.1
stuck.tick()
check('시간이 지나면 결국 출발한다 (멈춰 있는 것보단 낫다)',
      stuck.sent['go'] is True)

blind = FakeSeq(feedback=False)                  # joint_states 가 안 온다
blind.on_light(Msg('GREEN'))
blind.tick()
check('피드백이 없어도 바로는 출발 안 한다', blind.sent['go'] is False)
blind._turn_started -= blind.turn_timeout_s + 0.1
blind.tick()
check('피드백이 없으면 시간으로 넘어간다', blind.sent['go'] is True)


print('\n[4] follow 노드는 race/go 만 본다')
# 신호등을 직접 보면, 초록인 그 순간 카메라가 아직 오른쪽이라 출발이 이르다.
check('race/go 를 구독한다', "'race/go'" in follow_src)
check('신호등을 직접 구독하지 않는다 ★',
      "'traffic/light_state'" not in follow_src)
check('래치를 푸는 코드가 없다',
      follow_src.count('self._green_seen = False') == 0)
check('대기 중에는 속도와 조향 둘 다 0',
      'self._publish(0.0, 0.0)' in follow_src)

tick_fn = next(n for n in ast.walk(ast.parse(follow_src))
               if isinstance(n, ast.FunctionDef) and n.name == 'tick')
gate = next((n.lineno for n in ast.walk(tick_fn)
             if isinstance(n, ast.If) and '_green_seen' in ast.unparse(n.test)),
            None)
usable = next((n.lineno for n in ast.walk(tick_fn)
               if isinstance(n, ast.Assign)
               and 'usable' in ast.unparse(n.targets[0])), None)
check('게이트가 경로 판단보다 앞에 있다 ★',
      gate is not None and usable is not None and gate < usable,
      '(게이트 %s행, 경로판단 %s행)' % (gate, usable))


print('\n[5] 배선')
check('노드 셋과 follow 대기가 traffic_light 하나로 묶여 있다 ★',
      launch_src.count(
          "condition=IfCondition(LaunchConfiguration('traffic_light'))") == 2
      and "'wait_for_green': _b('traffic_light')" in launch_src)
# 둘이 같이 돌면 /camera/tilt 를 서로 다른 값으로 밀어 카메라가 떤다.
check('start_sequence 와 camera_tilt_publisher 가 같이 뜨지 않는다 ★',
      "' != 'true'" in launch_src and 'PythonExpression' in launch_src)
check('런치가 두 노드를 실제로 띄운다',
      'traffic, start_sequence, follow' in launch_src)
check('주행 틸트 값을 tilt_degrees 에서 받는다 (두 벌로 안 적는다)',
      "'drive_tilt_degrees': _f('tilt_degrees')" in launch_src)
setup_src = read(SETUP)
for name in ('traffic_light_node', 'start_sequence_node'):
    check('setup.py 에 %s 등록' % name,
          '%s = physicar_race.%s:main' % (name, name) in setup_src)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
