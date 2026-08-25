"""hsv_tuner_node 의 순수 로직 검증 (창을 띄우지 않는다)."""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ros_stubs  # noqa: E402

ros_stubs.install()

SRC = os.path.join(HERE, os.pardir, 'src', 'physicar_race', 'physicar_race',
                   'hsv_tuner_node.py')
spec = importlib.util.spec_from_file_location('hsv_tuner_node', SRC)
tuner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tuner)

FAILS = []
W, H = 640, 480


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def road(white_v=255, yellow=True):
    """어두운 노면 + 흰 실선 양쪽 + 노란 점선 중앙. 아래 절반에만 그린다."""
    img = np.full((H, W, 3), 60, np.uint8)
    img[H // 2:, 18:26] = (white_v, white_v, white_v)
    img[H // 2:, 476:484] = (white_v, white_v, white_v)
    if yellow:
        for y in range(H // 2, H, 60):
            img[y:min(H, y + 35), 316:324] = (0, 255, 255)
    return img


def vals(**over):
    v = {n: d for n, _m, d in tuner.CONTROLS}
    v.update(over)
    return v


print('\n[1] 마스크 생성')
roi = road()[int(H * 0.55):, :]
white, yellow = tuner.build_masks(roi, vals())
check('흰선이 마스크에 잡힘', int((white > 0).sum()) > 100,
      '(%d px)' % int((white > 0).sum()))
check('노란선이 마스크에 잡힘', int((yellow > 0).sum()) > 100,
      '(%d px)' % int((yellow > 0).sum()))
check('흰선 마스크에 노란선이 섞이지 않음',
      int((white[:, 300:340] > 0).sum()) == 0,
      '(중앙 %d px)' % int((white[:, 300:340] > 0).sum()))

print('\n[2] 임계값이 실제로 먹는가')
w_dim, _ = tuner.build_masks(road(white_v=150)[int(H * 0.55):, :], vals())
check('어두운 흰선은 기본 기준(V>=180)에서 탈락', int((w_dim > 0).sum()) == 0,
      '(%d px)' % int((w_dim > 0).sum()))
w_ok, _ = tuner.build_masks(road(white_v=150)[int(H * 0.55):, :],
                            vals(white_v_min=120))
check('  V 기준을 낮추면 잡힘', int((w_ok > 0).sum()) > 100,
      '(%d px)' % int((w_ok > 0).sum()))

print('\n[3] 합성 화면')
canvas, n_w, n_y = tuner.render(road(), vals())
check('2x2 크기', canvas.shape[:2] == (H * 2, W * 2), '(%s)' % (canvas.shape[:2],))
check('흰선 덩어리 2개 검출', n_w == 2, '(=%d)' % n_w)
check('노란 점선 덩어리 다수 검출', n_y >= 2, '(=%d)' % n_y)

canvas2, n_w2, _ = tuner.render(road(white_v=150), vals())
check('안 잡히면 0개로 보고', n_w2 == 0, '(=%d)' % n_w2)

print('\n[4] ROI 가 실제로 잘리는가')
# 위 절반에만 흰선을 그리면 ROI(아래 45%) 밖이라 안 잡혀야 한다
top_only = np.full((H, W, 3), 60, np.uint8)
top_only[:H // 3, 18:26] = (255, 255, 255)
_, n_top, _ = tuner.render(top_only, vals(roi_top_pct=55))
check('ROI 밖의 선은 안 잡힘', n_top == 0, '(=%d)' % n_top)
_, n_all, _ = tuner.render(top_only, vals(roi_top_pct=0))
check('ROI 를 전체로 열면 잡힘', n_all == 1, '(=%d)' % n_all)

print('\n[5] 색상 구간 역전 방어')
v, swapped = tuner.normalize(vals(yellow_h_min=38, yellow_h_max=18))
check('min>max 면 뒤집어 적용', swapped and v['yellow_h_min'] == 18
      and v['yellow_h_max'] == 38, '(%d~%d)' % (v['yellow_h_min'], v['yellow_h_max']))
_, y_rev = tuner.build_masks(roi, v)
check('  뒤집은 뒤엔 정상 검출', int((y_rev > 0).sum()) > 100)
v2, sw2 = tuner.normalize(vals())
check('정상 범위는 그대로', not sw2 and v2['yellow_h_min'] == 18)

print('\n[6] launch 인자 출력')
line = tuner.launch_args(vals(yellow_s_min=40, yellow_h_min=45))
check('지금 쓰는 launch 를 가리킨다',
      'physicar_race perception_v3_race_launch.py' in line)
check('색상 슬라이더가 green_* 로 나간다 ★',
      'green_h_min:=45' in line and 'green_s_min:=40' in line,
      '(%s)' % [t for t in line.split() if 'green' in t])
# 흰선/주황은 MinSeok 님 노드에 하드코딩이라 launch 인자가 없다.
# 없는 인자를 출력하면 붙여넣었을 때 그냥 에러가 난다.
check('  없는 인자를 만들어내지 않는다',
      'lane_' not in line and 'white' not in line)

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
