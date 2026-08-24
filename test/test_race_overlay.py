"""race_overlay_node 검증 - 우리가 가려는 길을 path_overlay 위에 그린다.

이 노드는 MinSeok 님 path_overlay 를 받아 회피 결정을 덧그려 다시 낸다.
그림 자체는 눈으로 볼 것이라, 여기서는 **좌표가 맞는가**와 **호가 물리적으로
옳은가**만 본다. 둘 다 손계산이 된다.
"""
import importlib.util
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ros_stubs  # noqa: E402

ros_stubs.install()
from std_msgs.msg import Float32MultiArray  # noqa: E402

SRC = os.path.join(HERE, os.pardir, 'src', 'physicar_race', 'physicar_race',
                   'race_overlay_node.py')
spec = importlib.util.spec_from_file_location('race_overlay_node', SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def node():
    return mod.RaceOverlayNode()


def blank(h=190, w=150):
    return np.full((h, w, 3), 40, np.uint8)


print('\n[1] 미터 -> 픽셀 - cone_bev_node 의 역변환과 맞는가')
n = node()
# cone_bev_node 는 col=60 -> y=0.145 였다. 되돌리면 60 이 나와야 한다.
col, row = n.to_pixel(0.955, 0.145)
check('왕복 변환이 일치 (col 60)', col == 60, '(%d)' % col)
check('  행도 일치 (row 104)', row == 104, '(%d)' % row)
# 폭 150 의 정확한 중앙은 74.5 라 74/75 둘 다 맞다
check('중앙은 화면 중앙', n.to_pixel(1.0, 0.0)[0] in (74, 75),
      '(%d)' % n.to_pixel(1.0, 0.0)[0])
check('왼쪽(+y)이 작은 열', n.to_pixel(1.0, 0.3)[0] < n.to_pixel(1.0, -0.3)[0])
check('먼 곳(+x)이 작은 행', n.to_pixel(1.8, 0.0)[1] < n.to_pixel(0.3, 0.0)[1])


print('\n[2] 호 - 조향각이 만드는 실제 궤적인가 (손계산)')
n2 = node()

straight = n2.arc_points(0.0, 1.0)
check('조향 0 이면 직선', all(abs(y) < 1e-9 for _, y in straight),
      '(%d점)' % len(straight))
check('  목표 x 까지 간다', abs(straight[-1][0] - 1.0) < 1e-9)

# 손계산: delta=20도 -> R = 0.18/tan(20) = 0.4946 m
# theta=90도 지점에서 x = R = 0.4946, y = R = 0.4946
left = n2.arc_points(math.radians(20.0), 0.45)
r_expect = mod.WHEELBASE / math.tan(math.radians(20.0))
check('좌조향(양수)이면 왼쪽(+y)으로 휜다', left[-1][1] > 0,
      '(끝점 %+.3f)' % left[-1][1])
# 원 위의 점은 중심 (0,R) 에서 거리가 R 이어야 한다
worst = max(abs(math.hypot(x - 0.0, y - r_expect) - abs(r_expect))
            for x, y in left)
check('  모든 점이 반경 R=%.3f 원 위' % r_expect, worst < 1e-9,
      '(최대오차 %.2e)' % worst)

right = n2.arc_points(math.radians(-20.0), 0.45)
check('우조향이면 오른쪽으로', right[-1][1] < 0, '(%+.3f)' % right[-1][1])
check('  좌우 대칭', abs(left[-1][1] + right[-1][1]) < 1e-9)

sharp = n2.arc_points(math.radians(20.0), 1.0)
gentle = n2.arc_points(math.radians(5.0), 1.0)
check('많이 꺾을수록 더 휜다',
      abs(sharp[-1][1]) > abs(gentle[-1][1]),
      '(20도 %+.2f vs 5도 %+.2f)' % (sharp[-1][1], gentle[-1][1]))
check('점 개수가 폭주하지 않는다', len(sharp) <= 400, '(%d점)' % len(sharp))


print('\n[3] 그리기 - 원본을 덮어쓰지 않고 얹는가')
n3 = node()
n3.on_decision(Float32MultiArray(data=[1.0, 0.0, 0.30, math.radians(10.0),
                                       0.30, 1.0]))
n3.on_cones(Float32MultiArray(data=[1.0, 0.0, 0.06, 0.55, -0.55]))
n3.on_overlay(ros_stubs.Image(cv=blank()))
out = n3.last('/race/debug/path_overlay')
check('결과를 발행한다', out is not None)
check('  크기가 그대로', out.shape == (190, 150, 3), '(%s)' % (out.shape,))

painted = int((out != 40).any(axis=2).sum())
check('  뭔가 그려졌다', painted > 20, '(%d px)' % painted)

# 청록(255,255,0) = 우리 주행선
cyan = int(((out[:, :, 0] > 200) & (out[:, :, 1] > 200)
            & (out[:, :, 2] < 60)).sum())
check('  청록 주행선이 있다 ★', cyan > 10, '(%d px)' % cyan)
# 주황(0,140,255) = 고깔
orange = int(((out[:, :, 2] > 200) & (out[:, :, 0] < 60)).sum())
check('  주황 고깔 표시가 있다', orange > 5, '(%d px)' % orange)


print('\n[4] 오래된 결정은 안 그린다')
n4 = node()
n4.on_decision(Float32MultiArray(data=[1.0, 0.0, 0.30, 0.2, 0.30, 1.0]))
n4._decision_time = time.time() - 999
n4.on_cones(Float32MultiArray(data=[1.0, 0.0, 0.06, 0.55, -0.55]))
n4._cones_time = time.time() - 999
n4.on_overlay(ros_stubs.Image(cv=blank()))
stale = n4.last('/race/debug/path_overlay')
check('오래되면 아무것도 안 얹는다', int((stale != 40).any(axis=2).sum()) == 0)
check('  그래도 원본은 발행한다', stale is not None)

n5 = node()
n5.on_decision(Float32MultiArray(data=[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]))  # valid=0
check('valid=0 인 결정은 안 받는다', n5._decision is None)


print('\n[5] 화면 밖 좌표에서 안 죽는다')
n6 = node()
n6.on_decision(Float32MultiArray(data=[9.0, 5.0, -5.0, math.radians(19.0),
                                       5.0, 1.0]))
n6.on_cones(Float32MultiArray(data=[9.0, 9.0, 0.06, 9.0, -9.0]))
try:
    n6.on_overlay(ros_stubs.Image(cv=blank()))
    ok = n6.last('/race/debug/path_overlay') is not None
except Exception as e:                                      # noqa: BLE001
    ok = False
    print('       예외:', e)
check('BEV 밖 값이어도 발행된다', ok)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
