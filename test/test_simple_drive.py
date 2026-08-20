"""simple_drive_node 검증 - 최소 주행 노드.

이 노드가 하는 일은 하나다: 중앙선 하나 찾아서 그 위치로 조향.
그래서 검증할 것도 셋뿐이다 -- 부호가 맞는가, 갓길/차체를 안 잡는가,
못 찾으면 서는가.
"""
import importlib.util
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ros_stubs  # noqa: E402

ros_stubs.install()

SRC = os.path.join(HERE, os.pardir, 'src', 'physicar_race', 'physicar_race',
                   'simple_drive_node.py')
spec = importlib.util.spec_from_file_location('simple_drive_node', SRC)
sd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sd)

FAILS = []
W, H = 640, 480


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def scene(mid_x=None, body=True, shoulder=True):
    """실주행 로그(2026-08-20)의 배치를 재현한다.

    갓길 H=33 S=133, 차체 H=18 S=254 -- 차체는 주황 중앙선과 색이 거의 같아
    색으로는 못 가른다. 위치(화면 맨 아래 가장자리)로 갈라야 한다.
    """
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[:, :] = (106, 113, 73)                       # 노면
    if shoulder:
        hsv[:, :int(W * 0.09)] = (33, 133, 152)
        hsv[:, int(W * 0.92):] = (33, 133, 152)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[int(H * 0.60):, int(W * 0.13):int(W * 0.17)] = (255, 255, 255)
    bgr[int(H * 0.60):, int(W * 0.72):int(W * 0.76)] = (255, 255, 255)
    if body:
        b = cv2.cvtColor(np.uint8([[(18, 254, 216)]]), cv2.COLOR_HSV2BGR)[0][0]
        bgr[int(H * 0.94):, int(W * 0.88):] = b
    if mid_x is not None:
        m = cv2.cvtColor(np.uint8([[(20, 255, 230)]]), cv2.COLOR_HSV2BGR)[0][0]
        for y in range(int(H * 0.58), H, 40):
            bgr[y:y + 22, int(W * mid_x) - 4:int(W * mid_x) + 4] = m
    return bgr


def run(mid_x=None, **kw):
    n = sd.SimpleDriveNode()
    for k, v in kw.items():
        setattr(n, k, v)
    n.on_image(ros_stubs.Image(cv=scene(mid_x)))
    return n._steer_cmd, n._speed_cmd


TARGET = 0.5 - 0.35 * 0.5      # target_frac 0.35 -> 화면 x 0.325

print('\n[1] 조향 부호')
st, sp = run(TARGET)
check('목표와 일치하면 조향 ~0', abs(math.degrees(st)) < 2.0,
      '(%.1f도)' % math.degrees(st))
check('  속도 나감', sp > 0.0, '(%.2f)' % sp)

st_l, _ = run(0.20)
check('중앙선이 왼쪽 -> 좌회전(+)', st_l > math.radians(3),
      '(%.1f도)' % math.degrees(st_l))

st_r, _ = run(0.45)
check('중앙선이 오른쪽 -> 우회전(-)', st_r < -math.radians(3),
      '(%.1f도)' % math.degrees(st_r))

check('오차가 클수록 크게 꺾음', abs(st_l) > abs(math.radians(2)),
      '(좌 %.1f도 / 우 %.1f도)' % (math.degrees(st_l), math.degrees(st_r)))

print('\n[2] 조향 한계')
# 탐색창(search_frac) 안에 있으면서 오차가 큰 위치라야 포화를 실제로 확인한다.
# 창 밖에 두면 '못 찾음 -> 정지'가 되어 한계 검증이 되지 않는다.
st_far, sp_far = run(0.88, search_frac=1.0, steer_gain=3.0)
check('큰 오차에서 실제로 꺾는다 (전제 확인)',
      sp_far > 0.0 and abs(st_far) > 0.1,
      '(%.1f도)' % math.degrees(st_far))
check('한계 안 (±20도)', abs(st_far) <= sd.MAX_STEER + 1e-9,
      '(%.1f도)' % math.degrees(st_far))

print('\n[3] 갓길·차체를 중앙선으로 잡지 않는다')
# 중앙선이 없는 장면. 갓길(H=33)과 차체(H=18 S=254)만 있다.
st, sp = run(None)
check('중앙선 없으면 정지', sp == 0.0, '(speed %.2f)' % sp)
check('  조향도 0', st == 0.0, '(%.1f도)' % math.degrees(st))

print('\n[4] ROI 하단 절단 (차체 배제)')
# roi_bottom 을 1.0 으로 열면 차체가 들어와 오검출된다 -- 절단이 실제로 듣는지 확인
st_open, sp_open = run(None, roi_bottom=1.0, y_h_max=30, y_s_min=120)
st_cut, sp_cut = run(None, roi_bottom=0.92)
check('하단을 열면 차체를 잡아 움직인다 (전제 확인)', sp_open > 0.0,
      '(speed %.2f)' % sp_open)
check('하단을 자르면 안 잡힌다', sp_cut == 0.0, '(speed %.2f)' % sp_cut)

print('\n[5] 흰선 모드')
st_w, sp_w = run(TARGET, follow='white')
check('흰선 두 개 중점 추종', sp_w > 0.0, '(speed %.2f)' % sp_w)
check('  중점이 화면 중앙 근처면 조향 작음', abs(math.degrees(st_w)) < 15.0,
      '(%.1f도)' % math.degrees(st_w))

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
