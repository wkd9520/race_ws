"""안 쓰는 진단 오버레이를 v3 가 그리지 않는지, 그래도 결과가 같은지 본다.

실차 계측:

    total=330.5ms | map=3.6  remap=10.1  seg=99.7  head=114.7

seg 구간이 제일 크다. 그 안에서 seg.process() 는 마지막에 draw_overlay()
를 부르는데, 그 함수는 bev.copy() 를 뜬 다음 컴포넌트마다 픽셀을 최대
300개씩 **파이썬 루프로 하나하나 대입**한다. 컴포넌트가 열 개면 파이썬
반복 3000회다.

그런데 v3 노드는 그 결과를 한 번도 안 읽는다. 매 프레임 그려서 버리고
있었다.

v2 노드는 쓴다(v2/bev_frontend_node.py:654,709). 그래서 기본값은 True 로
두고 v3 만 False 를 넘긴다.

여기서 증명할 것:
  1. 오버레이를 안 그려도 마스크와 컴포넌트가 **완전히 같다**
  2. v2 의 기본 동작은 안 바뀌었다
  3. v3 노드가 실제로 False 를 넘긴다
  4. 계측이 한 프레임 치를 통째로 갈아끼운다 (앞서 tail 이 엉뚱하게
     나온 원인이 프레임 섞임이었다)
"""
import importlib.util
import io
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
V2 = os.path.join(HERE, os.pardir, 'src', 'MinSeok',
                  'physicar_track_perception_v2', 'physicar_track_perception_v2')
NODE = os.path.join(
    HERE, os.pardir, 'src', 'MinSeok', 'physicar_track_perception_v3',
    'physicar_track_perception_v3', 'bev_frontend_node.py')

FAILS = []


def check(label, cond, detail=''):
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % ('PASS' if cond else 'FAIL', label, detail))


PKG = 'physicar_track_perception_v2'
if PKG not in sys.modules:
    # segmentation.py 가 `from .components import ...` 를 쓰므로,
    # 상대 임포트가 풀리도록 패키지를 먼저 등록한다.
    spec = importlib.util.spec_from_file_location(
        PKG, os.path.join(V2, '__init__.py'),
        submodule_search_locations=[os.path.abspath(V2)])
    package = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = package
    spec.loader.exec_module(package)


def load(name):
    return importlib.import_module(PKG + '.' + name)


geometry = load('geometry')
components = load('components')
segmentation = load('segmentation')

GRID = geometry.BevGrid(0.20, 1.20, -0.70, 0.70, 0.01)   # 실차 격자
EXTRACTOR = components.CanonicalComponentExtractor(GRID)
H, W = GRID.height, GRID.width

RANGES = {
    'WHITE': (segmentation.HsvRange(lower=(0, 0, 170), upper=(179, 60, 255)),),
    'ORANGE': (segmentation.HsvRange(lower=(5, 90, 90), upper=(30, 255, 255)),),
}
PIPE = segmentation.ColorComponentPipeline(RANGES, 3, 5, EXTRACTOR)


def scene():
    """흰선 둘 + 주황 점선. 트랙과 닮은 모양을 만든다."""
    bev = np.full((H, W, 3), 40, dtype=np.uint8)          # 어두운 노면
    for col in (int(W * 0.18), int(W * 0.82)):            # 흰 경계선 둘
        bev[5:H - 5, col:col + 3] = (245, 245, 245)
    for start in range(8, H - 10, 22):                    # 주황 점선
        bev[start:start + 12, W // 2 - 1:W // 2 + 2] = (40, 150, 255)
    # 코너처럼 꺾인 흰선 하나 더 -- 컴포넌트를 늘려 오버레이 비용을 키운다
    bev[H // 2:H // 2 + 3, int(W * 0.18):int(W * 0.55)] = (245, 245, 245)
    valid = np.ones((H, W), dtype=bool)
    valid[:4, :] = False
    return bev, valid


BEV, VALID = scene()

print('\n[1] 오버레이를 안 그려도 결과가 같은가 ★')
with_overlay = PIPE.process(BEV, VALID, include_overlay=True)
without = PIPE.process(BEV, VALID, include_overlay=False)

check('흰색 마스크가 같다',
      np.array_equal(with_overlay.white_mask, without.white_mask))
check('주황 마스크가 같다',
      np.array_equal(with_overlay.orange_mask, without.orange_mask))
check('합친 마스크가 같다',
      np.array_equal(with_overlay.combined_mask, without.combined_mask))

a = with_overlay.component_frame.candidates
b = without.component_frame.candidates
check('컴포넌트 개수가 같다', len(a) == len(b), '(%d개)' % len(b))
check('컴포넌트가 실제로 잡혔다 (테스트가 헛돌지 않는다)', len(b) >= 3,
      '(%d개)' % len(b))
same = all(
    x.component_id == y.component_id and x.color == y.color
    and np.array_equal(x.canonical_points, y.canonical_points)
    and np.array_equal(x.raw_ordered_points, y.raw_ordered_points)
    for x, y in zip(a, b))
check('모든 컴포넌트의 좌표가 바이트 단위로 같다 ★', same)


print('\n[2] v2 의 기본 동작은 그대로인가')
default = PIPE.process(BEV, VALID)
check('기본값은 오버레이를 그린다 (v2 가 쓴다)',
      default.overlay.shape == BEV.shape,
      '(%s)' % (default.overlay.shape,))
check('끄면 빈 배열이 온다 (몰래 쓰면 바로 티난다)',
      without.overlay.size == 0, '(%s)' % (without.overlay.shape,))


print('\n[3] v3 노드가 실제로 끄는가')
source = io.open(NODE, encoding='utf-8').read()
check('v3 가 include_overlay=False 를 넘긴다',
      'include_overlay=False' in source)


print('\n[4] 계측이 한 프레임 치를 통째로 갈아끼우는가')
# 앞서 total 만 프레임 끝에서 넣는 바람에, 로그가 이전 프레임의 total
# 에서 이번 프레임의 head 를 빼서 tail 이 215.8 ms 로 튀었다.
check('한 프레임 치를 모았다가 한 번에 넣는다',
      'pending_stage[' in source and 'self.stage=pending_stage' in source)
check('중간에 self.stage 를 갱신하지 않는다',
      'self.stage.update(' not in source)


print('\n[5] 얼마나 빨라지나 (참고용, 판정 아님)')
n = 20
t0 = time.perf_counter()
for _ in range(n):
    PIPE.process(BEV, VALID, include_overlay=True)
t_on = (time.perf_counter() - t0) / n
t0 = time.perf_counter()
for _ in range(n):
    PIPE.process(BEV, VALID, include_overlay=False)
t_off = (time.perf_counter() - t0) / n
print('       그릴 때 %.1f ms -> 안 그릴 때 %.1f ms  (%.0f%% 절감)'
      % (t_on * 1e3, t_off * 1e3, (t_on - t_off) / t_on * 100))
check('느려지지는 않았다', t_off <= t_on)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
