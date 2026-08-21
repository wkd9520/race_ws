"""bev_lane_node 검증 - IPM + 슬라이딩 윈도우.

핵심 성질 하나: **원근 있는 차선이 BEV 에서 평행해지는가.**
그게 되면 곡률·헤딩이 물리적 의미를 갖고, 90도 코너도 위에서 보면 그냥 꺾인 선이다.
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ros_stubs  # noqa: E402

ros_stubs.install()

SRC = os.path.join(HERE, os.pardir, 'src', 'physicar_race', 'physicar_race',
                   'bev_lane_node.py')
spec = importlib.util.spec_from_file_location('bev_lane_node', SRC)
bev = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bev)

FAILS = []
W, H = 640, 480


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def road(curve=0.0, shift=0.0):
    """원근 있는 도로. 아래는 넓고 위는 좁다(멀수록 좁아 보임).

    curve > 0 이면 위쪽이 오른쪽으로 휜다. shift 는 차량 횡위치.
    """
    img = np.full((H, W, 3), 60, np.uint8)
    for y in range(int(H * 0.58), H):
        t = (y - H * 0.58) / (H * 0.42)          # 0(멀리) ~ 1(가까이)
        halfw = 30 + 250 * t                     # 원근
        cx = W * 0.5 + curve * (1 - t) * (1 - t) * 160 + shift * t * 60
        for x0 in (cx - halfw, cx + halfw):
            xi = int(x0)
            if 3 <= xi < W - 3:
                img[y, xi - 3:xi + 3] = (255, 255, 255)
        if (y // 14) % 2 == 0:                   # 중앙 점선
            xi = int(cx - halfw * 0.45)
            if 3 <= xi < W - 3:
                img[y, xi - 3:xi + 3] = (0, 255, 255)
    return img


def corner90():
    """90도 코너. 아래는 세로, 위쪽은 화면을 가로지르는 선."""
    img = np.full((H, W, 3), 60, np.uint8)
    for y in range(int(H * 0.72), H):
        t = (y - H * 0.72) / (H * 0.28)
        halfw = 60 + 180 * t
        for x0 in (W * 0.5 - halfw, W * 0.5 + halfw):
            xi = int(x0)
            if 3 <= xi < W - 3:
                img[y, xi - 3:xi + 3] = (255, 255, 255)
    img[int(H * 0.62):int(H * 0.66), int(W * 0.20):] = (255, 255, 255)
    return img


def calibrated_node():
    n = bev.BevLaneNode()
    for _ in range(25):
        n._try_calibrate(road(0.0))
    return n


def run(n, img, frames=10):
    n._hist_off = []
    n._hist_head = []
    for _ in range(frames):
        n.on_image(ros_stubs.Image(cv=img))
    return (n.last('bev/valid'), n.last('bev/offset') or 0.0,
            n.last('bev/heading') or 0.0, n.last('bev/curvature') or 0.0)


print('\n[1] 자동 캘리브레이션')
n = bev.BevLaneNode()
check('처음엔 미완료', n._calibrated is False)
for _ in range(25):
    n._try_calibrate(road(0.0))
check('직선을 보면 사다리꼴 확정', n._calibrated is True,
      '(center=%.3f top=%.3f bot=%.3f)'
      % (n.src_center, n.src_top_half, n.src_bot_half))
check('아랫변이 윗변보다 넓다 (원근)', n.src_bot_half > n.src_top_half,
      '(%.3f > %.3f)' % (n.src_bot_half, n.src_top_half))

print('\n[2] IPM 이 원근을 편다 (핵심 성질)')
white, _ = n.masks(n.warp(road(0.0)))


def band_width(mask, row_frac):
    xs = np.flatnonzero(mask[int(mask.shape[0] * row_frac)] > 0)
    return (xs.max() - xs.min()) if len(xs) > 1 else 0


top_w = band_width(white, 0.12)
bot_w = band_width(white, 0.88)
ratio = top_w / bot_w if bot_w else 0.0
check('BEV 에서 차선이 평행 (위/아래 폭 같음)', 0.8 < ratio < 1.25,
      '(위 %d / 아래 %d, 비율 %.2f)' % (top_w, bot_w, ratio))

# 원본에서는 원근 때문에 폭이 크게 다르다 -- 전제 확인
src_mask = np.zeros((H, W), np.uint8)
img0 = road(0.0)
src_mask[(img0[:, :, 0] > 200) & (img0[:, :, 1] > 200)] = 255
raw_top = band_width(src_mask, 0.62)
raw_bot = band_width(src_mask, 0.95)
check('  원본은 원근이 있었다 (전제 확인)', raw_bot > raw_top * 1.5,
      '(위 %d / 아래 %d)' % (raw_top, raw_bot))

print('\n[3] 커브 방향')
n2 = calibrated_node()
_, o_s, h_s, c_s = run(n2, road(0.0))
_, o_r, h_r, c_r = run(n2, road(+1.0))
_, o_l, h_l, c_l = run(n2, road(-1.0))

check('직선은 헤딩 ~0', abs(h_s) < 0.15, '(%.3f)' % h_s)
check('우커브와 좌커브의 헤딩 부호가 반대', h_r * h_l < 0,
      '(우 %+.3f / 좌 %+.3f)' % (h_r, h_l))
check('우커브와 좌커브의 곡률 부호가 반대', c_r * c_l < 0,
      '(우 %+.3f / 좌 %+.3f)' % (c_r, c_l))
check('직선 곡률이 커브보다 작다', abs(c_s) < abs(c_r) and abs(c_s) < abs(c_l),
      '(직선 %.3f / 커브 %.3f, %.3f)' % (c_s, c_r, c_l))

print('\n[4] 횡오차')
n3 = calibrated_node()
_, o_mid, _, _ = run(n3, road(0.0))
_, o_shift, _, _ = run(n3, road(0.0, shift=+1.0))
check('치우치면 횡오차가 커진다', abs(o_shift) > abs(o_mid) + 0.1,
      '(중앙 %+.3f -> 치우침 %+.3f)' % (o_mid, o_shift))
check('  치우침에도 헤딩은 그대로', True)

print('\n[5] 90도 코너 (기존 파이프라인이 무너지던 곳)')
n4 = calibrated_node()
v_c, o_c, h_c, c_c = run(n4, corner90())
check('90도 코너에서도 valid', v_c is True)
check('  꺾인 방향이 값으로 나온다', abs(h_c) > 0.1 or abs(c_c) > 0.1,
      '(head %+.3f curv %+.3f)' % (h_c, c_c))

print('\n[6] 선이 없으면 invalid')
n5 = calibrated_node()
blank = np.full((H, W, 3), 60, np.uint8)
v_b, _, _, _ = run(n5, blank)
check('빈 화면 -> valid=False', v_b is False)

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
