"""초록 고깔 회피 검증 - cone_bev_node + perception_v3_follow_node 회피부.

설계 요약(대화에서 확정한 것):
  검출  : 초록 HSV, 새 노드가 /perception_v3/debug/bev 를 재사용(IPM 복제 없음)
  개입  : 궤적을 따로 안 만들고 **전방주시점만 옆으로 민다**
  방향  : 빈 공간이 더 넓은 쪽
  목표  : 그 빈 공간의 중앙
  울타리: /perception_v3/debug/white_mask 로 여유 확인 (흰선 넘으면 실격)
  복귀  : 오프셋을 서서히 되돌림
  속도  : 안 줄인다 (미리 피함)

기하는 손으로 계산되므로 여기서 확실히 잡는다. 픽셀->미터 변환은 격자가
알려진 값이라 마찬가지로 손계산 가능하다.
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
from nav_msgs.msg import Path  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from std_msgs.msg import Bool, Float32MultiArray  # noqa: E402


def load(name):
    src = os.path.join(HERE, os.pardir, 'src', 'physicar_race',
                       'physicar_race', name + '.py')
    spec = importlib.util.spec_from_file_location(name, src)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


cone_mod = load('cone_bev_node')
follow_mod = load('perception_v3_follow_node')

FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


# ══════════════════════════════════════════════ [1] 픽셀 → 미터 (손계산)

print('\n[1] BEV 픽셀 -> 미터 - perception_v3 격자와 같은가')
cn = cone_mod.ConeBevNode()

h, w = cn.expected_shape()
check('격자 크기가 perception_v3 와 같다 (150×190)', (h, w) == (190, 150),
      '(%d×%d)' % (w, h))

# 손계산: y = 0.75 − (col+0.5)×0.01,  x = 2.00 − (row+0.5)×0.01
check('맨 왼쪽 열 -> y 최대', abs(cn.col_to_y(0) - 0.745) < 1e-9,
      '(%.3f)' % cn.col_to_y(0))
check('맨 오른쪽 열 -> y 최소', abs(cn.col_to_y(149) - (-0.745)) < 1e-9,
      '(%.3f)' % cn.col_to_y(149))
check('가운데 열 -> y ~0', abs(cn.col_to_y(74.5)) < 1e-9)
check('맨 아래 행 -> 가장 가까움', abs(cn.row_to_x(189) - 0.105) < 1e-9,
      '(%.3f)' % cn.row_to_x(189))
check('맨 위 행 -> 가장 멂', abs(cn.row_to_x(0) - 1.995) < 1e-9,
      '(%.3f)' % cn.row_to_x(0))
check('  왼쪽이 양수 (차량 좌표 관례)', cn.col_to_y(10) > 0 > cn.col_to_y(140))


print('\n[2] 초록 검출 - BEV 한 장에서 고깔을 미터로 뽑는가')


def bev_with_cone(col, row, size=10, bgr=(0, 200, 0)):
    """BEV 이미지에 초록 사각형 하나."""
    img = np.full((190, 150, 3), 60, np.uint8)
    img[row - size // 2:row + size // 2, col - size // 2:col + size // 2] = bgr
    return img


def white_walls(left_col, right_col):
    """좌우 흰 경계선만 있는 마스크. left_col 이 화면 왼쪽(=y 큰 쪽)."""
    m = np.zeros((190, 150), np.uint8)
    m[:, left_col] = 255
    m[:, right_col] = 255
    return m


n2 = cone_mod.ConeBevNode()
n2._white = white_walls(20, 130)
n2.on_bev(ros_stubs.Image(cv=bev_with_cone(col=60, row=100)))
data = n2.last('/cones')
check('고깔 하나를 찾는다', data is not None and len(data) == 5,
      '(%d개 실수)' % (0 if data is None else len(data)))

if data and len(data) == 5:
    x, y, half, lw, rw = data
    # col=60 -> y = 0.75 − 60.5×0.01 = 0.145
    check('  y 가 손계산과 일치 (0.145)', abs(y - 0.145) < 0.011, '(%.3f)' % y)
    # 사각형 아래 끝 행 = 104 -> x = 2.00 − 104.5×0.01 = 0.955
    check('  x 는 고깔의 가까운 면 (0.955)', abs(x - 0.955) < 0.011, '(%.3f)' % x)
    check('  반폭 ~0.05m', abs(half - 0.05) < 0.011, '(%.3f)' % half)
    # 벽: col 20 -> y=0.545(왼쪽), col 130 -> y=-0.555(오른쪽)
    check('  왼쪽 벽 y (0.545)', abs(lw - 0.545) < 0.011, '(%+.3f)' % lw)
    check('  오른쪽 벽 y (−0.555)', abs(rw - (-0.555)) < 0.011, '(%+.3f)' % rw)

n3 = cone_mod.ConeBevNode()
n3._white = white_walls(20, 130)
n3.on_bev(ros_stubs.Image(cv=np.full((190, 150, 3), 60, np.uint8)))
check('초록이 없으면 빈 배열', n3.last('/cones') == [])

n4 = cone_mod.ConeBevNode()
n4._white = white_walls(20, 130)
n4.on_bev(ros_stubs.Image(cv=bev_with_cone(col=60, row=100, size=2)))
check('잡음 크기는 무시한다 (min_area_px)', n4.last('/cones') == [],
      '(2×2 = 4px < %d)' % n4.min_area)


# ══════════════════════════════════════════ [3] 회피 기하 (컨트롤러, 손계산)

print('\n[3] 회피 방향 - 빈 공간이 더 넓은 쪽 ★')


def follow(**kw):
    f = follow_mod.PerceptionV3FollowNode()
    for k, v in kw.items():
        setattr(f, k, v)
    f._last_cones_time = time.time()
    return f


# 트랙 폭 0.70 (perception_v3.yaml 의 white.track_width) -> 흰선은 ±0.35.
# 고깔 y=+0.10, 반폭 0.05, 고깔여유 0.12 -> 막힌 구간 [-0.07, +0.27]
#   왼쪽  빈칸 [0.27, 0.35-0.10=0.25] = -0.02  (자리 없음)
#   오른쪽 빈칸 [-0.35+0.10=-0.25, -0.07] = 0.18
#   -> 오른쪽, 중앙은 (-0.25 + -0.07)/2 = -0.16
f1 = follow()
f1._cones = [(1.0, 0.10, 0.05, 0.35, -0.35)]
y, why = f1.avoid_target_y(1.0, 0.0, 0.9)
check('넓은 쪽(오른쪽)을 고른다 ★', y < 0, '(%+.3f  %s)' % (y, why))
check('  그 빈 공간의 중앙 (-0.16)', abs(y - (-0.16)) < 0.005, '(%+.3f)' % y)

# 좌우를 뒤집으면 반대가 나와야 한다
f2 = follow()
f2._cones = [(1.0, -0.10, 0.05, 0.35, -0.35)]
y2, why2 = f2.avoid_target_y(1.0, 0.0, 0.9)
check('거울상이면 반대쪽', y2 > 0, '(%+.3f  %s)' % (y2, why2))
check('  크기가 대칭', abs(abs(y2) - abs(y)) < 1e-9)

print('\n[4] 흰선 울타리 - 넘으면 실격이므로 벽이 방향을 뒤집는다')
# 고깔 y=-0.10 -> 막힌 [-0.27, +0.07]. 오른쪽 벽이 -0.20 으로 바짝 붙어 있으면
#   왼쪽  [0.07, 0.25] = 0.18
#   오른쪽 [-0.10, -0.27] = -0.17  (자리 없음)
f3 = follow()
f3._cones = [(1.0, -0.10, 0.05, 0.35, -0.20)]
y3, why3 = f3.avoid_target_y(1.0, 0.0, 0.9)
check('그쪽이 좁으면 반대로 ★', y3 > 0, '(%+.3f  %s)' % (y3, why3))

# 양쪽 다 막히면 억지로 밀지 않는다
f4 = follow()
f4._cones = [(1.0, 0.0, 0.05, 0.20, -0.20)]
y4, why4 = f4.avoid_target_y(1.0, 0.0, 0.9)
check('양쪽 다 막히면 원래 목표 유지', abs(y4 - 0.0) < 1e-9,
      '(%s)' % why4)

print('\n[4b] 흰선을 못 본 쪽으로 새지 않는가 ★ (실차에서 나온 버그)')
# 실차 증상: "1차선에 장애물이 있으면 2차선(오른쪽)으로 가야 하는데
# 좌측 흰선을 넘어서 왼쪽으로 틀어버린다."
#
# 원인: cone_bev_node 는 흰선을 못 찾으면 격자 가장자리(±0.75)를 벽으로
# 보고한다. 그러면 "저쪽은 0.75m 까지 뚫려 있다"가 되어 **검출에 실패한
# 쪽이 오히려 넓어 보인다.** 코너에서 바깥 흰선이 화면 밖으로 나가는 순간
# 정확히 이 조건이 만들어진다.
#
# 왼쪽 차선 고깔(y=+0.10), 오른쪽 흰선은 -0.35 검출, 왼쪽 흰선은 놓쳐서
# +0.75(격자 끝)로 들어온 상황. 막힌 구간은 [-0.07, +0.27]:
#   조임 없음: 왼쪽 [0.27, 0.65] = 0.38  >  오른쪽 0.18  -> 왼쪽 (흰선 넘음)
#   조임 0.37: 왼쪽 [0.27, 0.27] = 0.00  <  오른쪽 0.18  -> 오른쪽 (정상)
LEFT_LOST = (1.0, 0.10, 0.05, 0.75, -0.35)

f_bug = follow(track_half_m=99.0)      # 조임을 꺼서 옛 동작을 재현
f_bug._cones = [LEFT_LOST]
y_bug, why_bug = f_bug.avoid_target_y(1.0, 0.0, 0.9)
check('조임이 없으면 실제로 왼쪽으로 샌다 (버그 재현)', y_bug > 0,
      '(%+.3f  %s)' % (y_bug, why_bug))

f_fix = follow()                        # 기본값 track_half_m = 0.37
f_fix._cones = [LEFT_LOST]
y_fix, why_fix = f_fix.avoid_target_y(1.0, 0.0, 0.9)
check('조이면 오른쪽(2차선)으로 간다 ★', y_fix < 0,
      '(%+.3f  %s)' % (y_fix, why_fix))
check('  목표가 트랙 안에 있다', abs(y_fix) < f_fix.track_half_m,
      '(|%.3f| < %.2f)' % (y_fix, f_fix.track_half_m))

# 거울상: 오른쪽 차선 고깔 + 오른쪽 흰선 유실
f_mir = follow()
f_mir._cones = [(1.0, -0.10, 0.05, 0.35, -0.75)]
y_mir, _ = f_mir.avoid_target_y(1.0, 0.0, 0.9)
check('  거울상도 안쪽으로', y_mir > 0, '(%+.3f)' % y_mir)

# 실제로 검출된 벽은 조임에 안 잘려야 한다 (더 안쪽이므로)
f_keep = follow()
f_keep._cones = [(1.0, 0.10, 0.05, 0.30, -0.35)]   # 왼쪽 벽 0.30 < 0.37
y_keep, _ = f_keep.avoid_target_y(1.0, 0.0, 0.9)
check('  검출된(더 안쪽) 벽은 그대로 쓴다', y_keep < 0, '(%+.3f)' % y_keep)

check('오프셋 상한도 트랙 반폭 안', follow().max_offset_m <= 0.37,
      '(%.2f)' % follow().max_offset_m)


print('\n[5] 언제 반응하지 않는가')
f5 = follow()
f5._cones = [(1.0, 0.60, 0.05, 0.90, -0.55)]
y5, _ = f5.avoid_target_y(1.0, 0.0, 0.9)
check('목표점이 이미 고깔 밖이면 그대로', abs(y5 - 0.0) < 1e-9, '(%+.3f)' % y5)

f6 = follow()
f6._cones = [(1.9, 0.0, 0.05, 0.55, -0.55)]     # 0.9m 앞 목표, 고깔은 1.9m
y6, _ = f6.avoid_target_y(0.9, 0.0, 0.9)
check('창 밖(너무 먼) 고깔은 무시', abs(y6 - 0.0) < 1e-9, '(%+.3f)' % y6)

f7 = follow()
f7._cones = [(1.0, 0.0, 0.05, 0.55, -0.55)]
f7._last_cones_time = time.time() - 999
y7, _ = f7.avoid_target_y(1.0, 0.0, 0.9)
check('오래된 /cones 는 안 쓴다', abs(y7 - 0.0) < 1e-9)

f8 = follow(avoid_enabled=False)
f8._cones = [(1.0, 0.0, 0.05, 0.55, -0.55)]
y8, _ = f8.avoid_target_y(1.0, 0.0, 0.9)
check('avoid_enabled=false 면 아무것도 안 한다', abs(y8 - 0.0) < 1e-9)


print('\n[6] 오프셋 변화율 - 붙을 땐 빠르게, 풀 땐 천천히 ★')
f9 = follow()
steps_in = 0
while abs(f9._offset - 0.30) > 1e-3 and steps_in < 500:
    f9.step_offset(0.30)
    steps_in += 1
steps_out = 0
while abs(f9._offset) > 1e-3 and steps_out < 500:
    f9.step_offset(0.0)
    steps_out += 1
check('복귀가 진입보다 느리다 ★', steps_out > steps_in * 2,
      '(진입 %d틱, 복귀 %d틱)' % (steps_in, steps_out))
check('  한 프레임에 튀지 않는다', steps_in > 3, '(%d틱)' % steps_in)

f10 = follow()
for _ in range(200):
    f10.step_offset(9.9)
check('오프셋 상한을 지킨다', abs(f10._offset - f10.max_offset_m) < 1e-9,
      '(%.2f)' % f10._offset)


print('\n[7] 전 구간 - 고깔이 조향을 실제로 바꾸는가')


def make_path(points):
    msg = Path()
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        msg.poses.append(pose)
    return msg


def run(cones, ticks=60):
    f = follow_mod.PerceptionV3FollowNode()
    f.on_path(make_path([(0.3, 0.0), (0.6, 0.0), (0.9, 0.0),
                         (1.2, 0.0), (1.5, 0.0)]))
    f.on_valid(Bool(data=True))
    flat = []
    for c in cones:
        flat.extend(c)
    for _ in range(ticks):
        f.on_cones(Float32MultiArray(data=flat))
        f._timers[0][1]()
    return f


clear = run([])
check('고깔 없으면 직진', abs(math.degrees(clear._steer)) < 1.0,
      '(%+.1f도)' % math.degrees(clear._steer))

# 경로 위(y=0)에 고깔. 좌우 벽은 대칭이 아니게 두어 방향이 정해지게 한다.
blocked = run([(1.0, 0.0, 0.06, 0.60, -0.30)])
check('경로 위 고깔이면 꺾는다 ★', abs(math.degrees(blocked._steer)) > 2.0,
      '(%+.1f도, 오프셋 %+.2fm)'
      % (math.degrees(blocked._steer), blocked._offset))
check('  넓은 쪽(왼쪽=양수)으로', blocked._offset > 0,
      '(%+.2fm)' % blocked._offset)
check('  조향이 한계 안', abs(blocked._steer) <= follow_mod.MAX_STEER + 1e-9)
check('  속도를 안 줄인다 (설계 선택)',
      abs(blocked._speed - clear._speed) < 1e-9,
      '(고깔 %.2f vs 직진 %.2f)' % (blocked._speed, clear._speed))

mirror = run([(1.0, 0.0, 0.06, 0.30, -0.60)])
check('  거울상이면 반대로', mirror._offset < 0, '(%+.2fm)' % mirror._offset)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
