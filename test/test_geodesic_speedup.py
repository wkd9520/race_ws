"""`_ordered_geodesic_polyline` 최적화가 결과를 안 바꾸는지 증명한다.

라즈베리파이 5 에서 인지가 카메라를 못 따라갔고, 이 함수가 병목이었다.
연결요소의 픽셀마다 파이썬 반복 8회 + `sorted()` 한 번을 돌기 때문이다.

**알고리즘은 안 건드렸다.** 파이썬 오버헤드만 걷어냈으므로 결과는 원본과
바이트 단위로 같아야 한다. 그걸 여기서 못 박는다 -- 이 함수의 출력이
경로 전체의 기반이라, 조용히 달라지면 주행이 통째로 바뀐다.

아래 `reference_polyline` 은 최적화 전 원본을 그대로 옮긴 것이다.
"""
import importlib.util
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, os.pardir, 'src', 'MinSeok',
                  'physicar_track_perception_v2', 'physicar_track_perception_v2')


def load(name):
    spec = importlib.util.spec_from_file_location(
        'v2_' + name, os.path.join(V2, name + '.py'))
    m = importlib.util.module_from_spec(spec)
    sys.modules['physicar_track_perception_v2.' + name] = m
    spec.loader.exec_module(m)
    return m


geometry = load('geometry')
components = load('components')

FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


# ------------------------------------------------------- 최적화 전 원본

def reference_polyline(ex, rows, cols):
    """커밋 2f10ffd 시점의 구현. 비교 기준으로만 쓴다."""
    pixels = {(int(row), int(col)) for row, col in zip(rows, cols)}
    if len(pixels) < 2:
        return np.empty((0, 2), dtype=np.float64)

    def farthest(start, with_parent=False):
        queue = [start]
        distance = {start: 0}
        parent = {} if with_parent else None
        for current in queue:
            row, col = current
            neighbours = sorted(
                (row + dr, col + dc)
                for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                if (dr or dc) and (row + dr, col + dc) in pixels
            )
            for neighbour in neighbours:
                if neighbour in distance:
                    continue
                distance[neighbour] = distance[current] + 1
                if parent is not None:
                    parent[neighbour] = current
                queue.append(neighbour)
        maximum = max(distance.values())
        endpoint = min(p for p, v in distance.items() if v == maximum)
        return endpoint, parent

    seed = min(pixels)
    first, _ = farthest(seed)
    second, parent = farthest(first, with_parent=True)
    path = [second]
    while path[-1] != first:
        path.append(parent[path[-1]])
    path.reverse()
    path = np.asarray(path, dtype=np.int32)
    x, y = ex.grid.pixel_to_metric(path[:, 1], path[:, 0])
    return ex._orient_near_to_far(np.column_stack((x, y)))


# ------------------------------------------------------------ 장면 만들기

GRID = geometry.BevGrid(0.10, 2.00, -0.75, 0.75, 0.01)   # 150 x 190
EX = components.CanonicalComponentExtractor(GRID)
H, W = GRID.height, GRID.width


def draw(pixel_list):
    """(row, col) 목록을 rows, cols 로. 중복 제거하고 범위 밖은 버린다."""
    seen = {(int(r), int(c)) for r, c in pixel_list
            if 0 <= int(r) < H and 0 <= int(c) < W}
    arr = np.array(sorted(seen), dtype=np.int64)
    return arr[:, 0], arr[:, 1]


def vertical_line(col, thick=2):
    return [(r, col + d) for r in range(5, H - 5) for d in range(thick)]


def diagonal(thick=2):
    return [(r, int(r * 0.6) + d) for r in range(5, H - 5) for d in range(thick)]


def corner_90(thick=3):
    """세로로 올라오다 가로로 꺾인다 -- 이 파이프라인이 무너지던 형상."""
    out = [(r, W // 2 + d) for r in range(H // 2, H - 5) for d in range(thick)]
    out += [(H // 2 + d, c) for c in range(W // 2, W - 5) for d in range(thick)]
    return out


def blob():
    """가는 선이 아닌 덩어리. 이웃 순서에 따라 갈림길이 생긴다."""
    rng = np.random.default_rng(7)
    cx, cy = H // 2, W // 2
    out = []
    for _ in range(400):
        out.append((cx + rng.integers(-12, 13), cy + rng.integers(-12, 13)))
    return out


def dashes():
    """점선 조각 하나."""
    return [(r, W // 3 + d) for r in range(40, 60) for d in range(2)]


def edge_hugging():
    """왼쪽 가장자리에 붙은 선. 열 경계 처리(패딩)를 확인한다."""
    return [(r, d) for r in range(5, H - 5) for d in range(2)]


def edge_hugging_right():
    return [(r, W - 1 - d) for r in range(5, H - 5) for d in range(2)]


SCENES = [
    ('세로선', vertical_line(W // 2)),
    ('대각선', diagonal()),
    ('90도 코너 ★', corner_90()),
    ('덩어리 (갈림길 있음) ★', blob()),
    ('점선 조각', dashes()),
    ('왼쪽 가장자리 ★', edge_hugging()),
    ('오른쪽 가장자리 ★', edge_hugging_right()),
    ('점 두 개', [(10, 10), (10, 11)]),
]


print('\n[1] 최적화 전후 결과가 완전히 같은가 ★')
for name, pts in SCENES:
    rows, cols = draw(pts)
    want = reference_polyline(EX, rows, cols)
    got = EX._ordered_geodesic_polyline(rows, cols)
    same = (want.shape == got.shape) and np.array_equal(want, got)
    check(name, same,
          '(%d점)' % len(got) if same
          else '(원본 %s vs 새것 %s)' % (want.shape, got.shape))


print('\n[2] 원본 sorted() 가 실제로 아무 일도 안 했는가')
# 최적화의 근거. (dr, dc) 오름차순으로 나열하면 (row+dr, col+dc) 튜플도
# 이미 오름차순이라, 픽셀마다 부르던 sorted() 는 낭비였다.
row, col = 50, 60
gen = [(row + dr, col + dc)
       for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr or dc)]
check('이웃 8개가 이미 정렬돼 있다', gen == sorted(gen), '(%s...)' % (gen[:3],))


print('\n[3] 빈 입력 / 픽셀 하나')
for name, pts in (('픽셀 0개', []), ('픽셀 1개', [(10, 10)])):
    rows, cols = (np.array([], np.int64), np.array([], np.int64)) \
        if not pts else draw(pts)
    got = EX._ordered_geodesic_polyline(rows, cols)
    check(name, got.shape == (0, 2), '(%s)' % (got.shape,))


print('\n[4] 얼마나 빨라졌나 (참고용, 판정 아님)')
rows, cols = draw(corner_90())
n = 20
t0 = time.perf_counter()
for _ in range(n):
    reference_polyline(EX, rows, cols)
t_old = (time.perf_counter() - t0) / n
t0 = time.perf_counter()
for _ in range(n):
    EX._ordered_geodesic_polyline(rows, cols)
t_new = (time.perf_counter() - t0) / n
print('       원본 %.1f ms -> 새것 %.1f ms  (%.1f배)'
      % (t_old * 1e3, t_new * 1e3, t_old / max(t_new, 1e-9)))
check('느려지지는 않았다', t_new <= t_old)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
