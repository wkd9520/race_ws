"""los_drive_node 검증 - IPM + LOS 가이던스.

이 테스트는 두 부분으로 나뉜다. **나눈 이유가 중요하다.**

[A] 해석적으로 확인 가능한 것 -- 순수추종 공식, 횡가속 한계, 좌표 변환.
    답을 손으로 계산할 수 있으므로 합성 장면의 사실성과 무관하게 믿을 수 있다.

[B] BEV 마스크를 직접 넣어 확인하는 것 -- 자유공간 추적.
    90도 코너 로직이 사는 곳이다. IPM 을 거치지 않고 BEV 를 직접 만들어
    넣는다. 원근 있는 장면을 내가 상상해서 만들면 또 못 믿을 결과가 나온다.

전 구간(원본->조향)은 부호와 포화만 본다. 파라미터 미세조정은 실차 몫이다.
"""
import importlib.util
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ros_stubs  # noqa: E402

ros_stubs.install()

SRC = os.path.join(HERE, os.pardir, 'src', 'physicar_race', 'physicar_race',
                   'los_drive_node.py')
spec = importlib.util.spec_from_file_location('los_drive_node', SRC)
los = importlib.util.module_from_spec(spec)
spec.loader.exec_module(los)

FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def node():
    return los.LosDriveNode()


# ══════════════════════════════════════════════ [A] 해석적으로 검증 가능

print('\n[1] 순수추종 공식 - delta = atan(2 L sin a / l_d)')
n = node()

check('정면이면 조향 0', abs(n.pure_pursuit(1.0, 0.0)) < 1e-6)

# 손계산: x=1.0, y=+0.5 -> a=26.57도, ld=1.118
#         delta = atan(2*0.18*sin(26.57)/1.118) = atan(0.14403) = 8.20도
d = math.degrees(n.pure_pursuit(1.0, 0.5))
check('손계산과 일치 (x=1.0 y=+0.5 -> 8.20도)', abs(d - 8.20) < 0.05,
      '(%.2f도)' % d)

check('왼쪽 점 -> 양수, 오른쪽 점 -> 음수',
      n.pure_pursuit(1.0, 0.5) > 0 > n.pure_pursuit(1.0, -0.5))
check('좌우 대칭',
      abs(n.pure_pursuit(1.0, 0.5) + n.pure_pursuit(1.0, -0.5)) < 1e-9)

# 같은 각도라도 가까이 보면 더 꺾는다 -- 순수추종의 본질
near = math.degrees(n.pure_pursuit(0.35, 0.35))
far = math.degrees(n.pure_pursuit(1.40, 1.40))
check('같은 각도면 가까울수록 강하게 꺾는다 ★', near > far * 2,
      '(0.35m %.1f도 vs 1.40m %.1f도)' % (near, far))

# 이 차의 물리적 한계: 2L/l_d 가 조향 상한을 정한다
lim = math.degrees(math.atan(2 * los.WHEELBASE / 1.30))
check('  전방주시 1.30m 면 최대 %.1f도밖에 안 나온다' % lim, lim < 20.0,
      '(휠베이스 0.18m 의 결과 -- ld_max 가 조향 상한을 건다)')

print('\n[2] 횡가속 한계 속도 - v = sqrt(a R), R = L / tan|d|')
n2 = node()
n2.k_vis = 99.0          # 시야 제한을 끄고 곡률 항만 본다

# 손계산: d=20도, R = 0.18/tan(20) = 0.4946, v = sqrt(3.0*0.4946) = 1.218
v20 = n2.speed_limit(math.radians(20.0), 10.0)
check('최대조향에서 손계산과 일치 (1.22 m/s)', abs(v20 - 1.218) < 0.01,
      '(%.3f)' % v20)

check('직진이면 v_max', abs(n2.speed_limit(0.0, 10.0) - n2.v_max) < 1e-6,
      '(%.2f)' % n2.speed_limit(0.0, 10.0))
check('많이 꺾을수록 느리다',
      n2.speed_limit(math.radians(5), 10) > n2.speed_limit(math.radians(20), 10))
check('v_min 아래로는 안 내려간다',
      n2.speed_limit(math.radians(20), 10) >= n2.v_min)

n3 = node()
check('안 보이면 느려진다 ★',
      n3.speed_limit(0.0, 0.4) < n3.speed_limit(0.0, 5.0),
      '(0.4m 보임 %.2f < 5m 보임 %.2f)'
      % (n3.speed_limit(0.0, 0.4), n3.speed_limit(0.0, 5.0)))

print('\n[3] 좌표 변환')
n4 = node()
check('아래 행이 가깝다',
      n4.row_to_forward_m(n4.bev_h - 1) < n4.row_to_forward_m(0))
check('맨 아래 = bev_near_m',
      abs(n4.row_to_forward_m(n4.bev_h) - n4.bev_near_m) < 1e-6,
      '(%.2fm)' % n4.row_to_forward_m(n4.bev_h))
check('맨 위 = near + range',
      abs(n4.row_to_forward_m(0) - (n4.bev_near_m + n4.bev_range_m)) < 1e-6,
      '(%.2fm)' % n4.row_to_forward_m(0))
check('중앙 열 = 횡위치 0', abs(n4.col_to_lateral_m(n4.bev_w * 0.5)) < 1e-6)
check('왼쪽 열이 양수 (차량 좌표 관례)',
      n4.col_to_lateral_m(0) > 0 > n4.col_to_lateral_m(n4.bev_w))


# ══════════════════════════════════════════ [B] BEV 마스크를 직접 넣어 검증

def bev_mask(free_fn, n_):
    """free_fn(x, y) -> True 인 곳이 자유공간, 나머지는 흰선(벽)."""
    m = np.full((n_.bev_h, n_.bev_w), 255, np.uint8)
    ys, xs = np.mgrid[0:n_.bev_h, 0:n_.bev_w]
    m[free_fn(xs, ys)] = 0
    return m


print('\n[4] 자유공간 추적 - 직선 통로')
n5 = node()
half = n5.bev_w * 0.5
straight = bev_mask(lambda x, y: np.abs(x - half) < 30, n5)
path = n5.corridor_path(straight)
check('통로를 끝까지 따라간다', len(path) > n5.bev_h / n5.row_step * 0.9,
      '(%d 행)' % len(path))
check('중심이 화면 중앙',
      all(abs(cx - half) < 2 for _, cx in path),
      '(첫 %.0f 끝 %.0f)' % (path[0][1], path[-1][1]))

print('\n[5] 자유공간 추적 - 90도 코너 ★ (기존 두 방식이 무너지던 곳)')
# ㄴ 자 통로: 아래에서 올라오다가 위쪽에서 오른쪽으로 꺾인다.
# 세로선 피팅(x = f(y))으로는 표현 자체가 안 되는 형상이다.
n6 = node()
BEND_Y = int(n6.bev_h * 0.50)


def corner_free(x, y):
    vertical = (np.abs(x - n6.bev_w * 0.5) < 30) & (y >= BEND_Y)
    horizontal = (y >= BEND_Y) & (y <= BEND_Y + 60) & (x >= n6.bev_w * 0.5 - 30)
    return vertical | horizontal


corner = bev_mask(corner_free, n6)
cpath = n6.corridor_path(corner)
check('코너에서도 통로를 찾는다', len(cpath) >= 3, '(%d 행)' % len(cpath))
check('중심선이 오른쪽으로 휜다 ★', cpath[-1][1] > cpath[0][1] + 10,
      '(아래 %.0f -> 위 %.0f)' % (cpath[0][1], cpath[-1][1]))
check('벽 너머로 넘어가지 않는다',
      all(corner[int(y), int(cx)] == 0 for y, cx in cpath),
      '(모든 중심점이 자유공간 안)')

# 이 코너에서 실제로 나오는 조향
ld_test = 0.9
pt = n6.los_point(cpath, ld_test)
xf = n6.row_to_forward_m(pt[0])
yl = n6.col_to_lateral_m(pt[1])
st = math.degrees(n6.pure_pursuit(xf, yl))
check('오른쪽으로 조향한다 ★', st < -1.0,
      '(LOS %.2fm 앞 %+.2fm 옆 -> %+.1f도)' % (xf, yl, st))

# 좌우 대칭 확인 -- 코너를 뒤집으면 조향도 뒤집혀야 한다
n7 = node()
mirrored = np.fliplr(corner)
mpath = n7.corridor_path(mirrored)
mpt = n7.los_point(mpath, ld_test)
mst = math.degrees(n7.pure_pursuit(n7.row_to_forward_m(mpt[0]),
                                   n7.col_to_lateral_m(mpt[1])))
check('  좌코너는 반대로 조향', mst > 1.0, '(%+.1f도)' % mst)
check('  좌우 크기가 비슷', abs(abs(mst) - abs(st)) < 1.5,
      '(우 %+.1f / 좌 %+.1f)' % (st, mst))

print('\n[6] 전방주시거리 - 이 노드의 유일한 핵심 튜닝값')
# 순수추종에는 상충이 하나 있고, ld 가 정확히 그 상충 위에 앉아 있다.
#   너무 짧다 -> LOS 점이 아직 직선 구간이라 코너를 아예 못 본다
#   너무 길다 -> 코너는 보지만 delta = atan(2L sin a / l_d) 의 분모가 커져
#                조향이 오히려 약해진다
# 그래서 "가까이 볼수록 세게 꺾는다"가 아니다. 중간에 최적점이 있다.
n8 = node()
prof = []
for ld in (0.4, 0.6, 0.9, 1.1, 1.3):
    p_ = n8.los_point(cpath, ld)
    s_ = math.degrees(n8.pure_pursuit(n8.row_to_forward_m(p_[0]),
                                      n8.col_to_lateral_m(p_[1])))
    prof.append((ld, n8.row_to_forward_m(p_[0]), s_))
    print('       ld=%.1fm -> %.2fm 앞, %+.1f도' % (ld, prof[-1][1], s_))

best = min(prof, key=lambda r: r[2])          # 가장 세게 우조향한 것
check('코너를 잡아내는 ld 구간이 있다 ★', abs(best[2]) > 5.0,
      '(ld=%.1fm 에서 %+.1f도)' % (best[0], best[2]))
check('  너무 짧으면 코너를 못 본다 (상충의 한쪽)', abs(prof[0][2]) < 1.0,
      '(ld=0.4m -> %+.1f도)' % prof[0][2])
check('  너무 길면 조향이 약해진다 (상충의 반대쪽)',
      abs(prof[-1][2]) < abs(best[2]),
      '(ld=1.3m %+.1f도 < 최적 %+.1f도)' % (prof[-1][2], best[2]))
check('  기본 ld 범위가 그 구간을 덮는다',
      n8.ld_min_m <= best[0] <= n8.ld_max_m,
      '(ld_min %.2f ~ ld_max %.2f)' % (n8.ld_min_m, n8.ld_max_m))

print('\n[7] 막힌 길 / 통로 없음')
n9 = node()
blocked = np.full((n9.bev_h, n9.bev_w), 255, np.uint8)
check('전부 벽이면 경로 없음', len(n9.corridor_path(blocked)) == 0)

# 통로가 도중에 끊기면 거기까지가 '보이는 거리'
n10 = node()
STOP_Y = int(n10.bev_h * 0.6)
partial = bev_mask(
    lambda x, y: (np.abs(x - n10.bev_w * 0.5) < 30) & (y >= STOP_Y), n10)
ppath = n10.corridor_path(partial)
vis = n10.row_to_forward_m(ppath[-1][0])
check('끊긴 지점까지만 본다', len(ppath) > 0 and ppath[-1][0] >= STOP_Y - 2,
      '(보임 %.2fm)' % vis)
check('  그래서 속도가 제한된다', n10.speed_limit(0.0, vis) < n10.v_max,
      '(%.2f < %.2f)' % (n10.speed_limit(0.0, vis), n10.v_max))


# ═════════════════════════════════════════════════ [C] 전 구간 (부호/포화만)

def perspective_scene(bend=0.0):
    """원근 있는 원본 화면. bend > 0 이면 위쪽이 오른쪽으로 휜다."""
    H, W = 480, 640
    img = np.full((H, W, 3), 60, np.uint8)
    for y in range(int(H * 0.55), H):
        t = (y - H * 0.55) / (H * 0.45)          # 0=멀리 1=가까이
        halfw = 40 + 260 * t
        cx = W * 0.5 + bend * (1 - t) * (1 - t) * 90
        for x0 in (cx - halfw, cx + halfw):
            xi = int(x0)
            if 4 <= xi < W - 4:
                img[y, xi - 4:xi + 4] = (255, 255, 255)
    return img


def scene_node():
    """테스트 장면의 기하에 맞춘 사다리꼴.

    출하 기본값을 그대로 쓰지 않는 이유: 사다리꼴은 실제 카메라의 장착
    높이·각도에 맞추는 값이라, 내가 지어낸 장면으로 그 값을 검증하는 건
    의미가 없다. 여기서는 장면에 맞춰 놓고 **부호와 포화만** 본다.
    """
    n_ = node()
    n_.src_top_y, n_.src_top_half = 0.55, 0.30
    n_.src_bot_y, n_.src_bot_half = 1.00, 0.50
    n_.track_width_m = 1.70      # 이 장면의 실제 도로 폭에 맞춘다
    return n_


def settle(n_, img, frames=40):
    """정상상태까지 돌린다.

    한 프레임만 보면 부족하다. 전방주시거리가 속도에 비례하고 속도는 0에서
    출발하므로, 첫 프레임은 늘 코앞만 본다. 실차에서 도는 건 정상상태다.
    """
    msg = ros_stubs.Image(cv=img)
    for _ in range(frames):
        n_.on_image(msg)
    return n_


print('\n[8] 전 구간 (원본 -> 조향) - 부호와 포화만')
n11 = settle(scene_node(), perspective_scene(0.0))
s_straight = n11._steer
check('직선에서 속도가 붙는다 ★', n11._speed > n11.v_min,
      '(%.2f m/s)' % n11._speed)
check('직선에서 거의 직진', abs(math.degrees(s_straight)) < 4.0,
      '(%+.1f도)' % math.degrees(s_straight))
check('  직선에서는 통로를 찾는다 (전제 확인)',
      n11.last('los/valid') is True)

n12 = settle(scene_node(), perspective_scene(+1.0))
n13 = settle(scene_node(), perspective_scene(-1.0))
check('우커브/좌커브 부호가 반대 ★', n12._steer * n13._steer < 0,
      '(우 %+.1f도 / 좌 %+.1f도)'
      % (math.degrees(n12._steer), math.degrees(n13._steer)))

check('조향이 한계 안에 있다',
      all(abs(x._steer) <= los.MAX_STEER + 1e-9 for x in (n11, n12, n13)))
check('속도가 한계 안 또는 정지(0)',
      all(x._speed == 0.0 or los.MIN_SPEED <= x._speed <= los.MAX_SPEED
          for x in (n11, n12, n13)),
      '(%s)' % ['%.2f' % x._speed for x in (n11, n12, n13)])

print('\n[9] 통로 유실 - 마지막 조향을 유지하다 정지')
n14 = settle(scene_node(), perspective_scene(+1.0))
held = n14._steer
blank = np.full((480, 640, 3), 60, np.uint8)
n14.on_image(ros_stubs.Image(cv=blank))
check('유실 직후 마지막 조향 유지 ★',
      abs(n14.last('/steering') - held) < 1e-9,
      '(%+.1f도 유지)' % math.degrees(held))
check('  서행한다', abs(n14.last('/speed') - n14.lost_speed) < 1e-9,
      '(%.2f)' % n14.last('/speed'))
check('  valid=False', n14.last('los/valid') is False)

for _ in range(n14.lost_hold_frames + 2):
    n14.on_image(ros_stubs.Image(cv=blank))
check('계속 유실되면 정지', n14.last('/speed') == 0.0)

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
