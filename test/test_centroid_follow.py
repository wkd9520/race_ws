"""centroid_follow_node 검증 - Mjolnir 구조.

참조 원본(github.com/ArthurDassier/Mjolnir_kit)의 핵심은 **폭 필터**다.
나는 갓길·차체 오검출을 ROI 절단과 탐색창 제한으로 막으려 했는데, 원본은
컨투어의 폭으로 거른다. 차선은 가늘고, 갓길·차체는 넓다. 색이 같아도 갈린다.
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
                   'centroid_follow_node.py')
spec = importlib.util.spec_from_file_location('centroid_follow_node', SRC)
cf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cf)

FAILS = []
W, H = 640, 480


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def scene(mid_x=0.35, body=True, shoulder=True):
    """실주행 로그(2026-08-20)의 배치.

    갓길 H=33 폭 57px, 차체 H=18 폭 96px, 중앙 점선 폭 10px.
    셋 다 색이 비슷해서 HSV 로는 못 가른다 -- 폭으로 갈라야 한다.
    """
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[:, :] = (106, 113, 73)
    if shoulder:
        hsv[:, :int(W * 0.09)] = (33, 133, 152)
        hsv[:, int(W * 0.92):] = (33, 133, 152)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if body:
        b = cv2.cvtColor(np.uint8([[(18, 254, 216)]]), cv2.COLOR_HSV2BGR)[0][0]
        bgr[int(H * 0.88):, int(W * 0.85):] = b
    if mid_x:
        m = cv2.cvtColor(np.uint8([[(20, 255, 230)]]), cv2.COLOR_HSV2BGR)[0][0]
        for y in range(int(H * 0.58), H, 40):
            bgr[y:y + 22, int(W * mid_x) - 5:int(W * mid_x) + 5] = m
    return bgr


def run(mid_x=0.35, **kw):
    n = cf.CentroidFollowNode()
    for k, v in kw.items():
        setattr(n, k, v)
    n.on_image(ros_stubs.Image(cv=scene(mid_x)))
    return n._steer_cmd, n._speed_cmd


print('\n[1] 폭 필터가 갓길·차체를 거른다 (이 파이프라인의 핵심)')
n = cf.CentroidFollowNode()
roi = scene()[int(H * n.roi_top):int(H * n.roi_bottom), :]
cents, _ = n.find_centroids(roi)
xs = sorted(c[0] for c in cents)

check('컨투어를 찾았다', len(cents) > 0, '(%d개)' % len(cents))
check('갓길(화면 왼쪽 끝)을 안 잡는다', all(x > W * 0.10 for x in xs),
      '(x=%s)' % xs[:4])
check('차체(화면 오른쪽 끝)를 안 잡는다', all(x < W * 0.80 for x in xs),
      '(x=%s)' % xs[-4:])

# 필터를 풀면 실제로 잡힌다 -- 전제 확인
n2 = cf.CentroidFollowNode()
n2.width_min_frac, n2.width_max_frac = 0.0, 10.0
cents2, _ = n2.find_centroids(roi)
xs2 = sorted(c[0] for c in cents2)
check('  필터를 풀면 잡힌다 (전제 확인)',
      len(cents2) > len(cents) and (xs2[0] < W * 0.10 or xs2[-1] > W * 0.80),
      '(%d개 -> %d개, x=%s)' % (len(cents), len(cents2), xs2))

print('\n[2] 조향 부호')
st_l, _ = run(0.25)
st_c, _ = run(0.50)
st_r, _ = run(0.75)
check('중앙선이 왼쪽 -> 좌회전(+)', st_l > math.radians(1),
      '(%.1f도)' % math.degrees(st_l))
check('중앙 -> 조향 ~0', abs(math.degrees(st_c)) < 1.5,
      '(%.1f도)' % math.degrees(st_c))
check('중앙선이 오른쪽 -> 우회전(-)', st_r < -math.radians(1),
      '(%.1f도)' % math.degrees(st_r))
check('좌우 대칭', abs(abs(st_l) - abs(st_r)) < math.radians(1),
      '(좌 %.1f / 우 %.1f도)' % (math.degrees(st_l), math.degrees(st_r)))

print('\n[3] 조향 한계')
# 실제로 포화시켜야 한계 검증이 된다. PID 는 노드 생성 시 만들어지므로
# 속성만 바꾸면 안 먹는다 -- pid 객체를 직접 교체한다.
nf = cf.CentroidFollowNode()
nf.pid = cf.PIDController(kp=20.0)
nf.on_image(ros_stubs.Image(cv=scene(0.15)))
st_far = nf._steer_cmd
check('큰 오차에서 포화한다 (전제 확인)',
      abs(st_far) > cf.MAX_STEER * 0.9, '(%.1f도)' % math.degrees(st_far))
check('한계 안 (±20도)', abs(st_far) <= cf.MAX_STEER + 1e-9,
      '(%.1f도)' % math.degrees(st_far))

print('\n[4] 선이 없으면 정지')
st, sp = run(None)
check('컨투어 없음 -> 정지', sp == 0.0, '(speed %.2f)' % sp)
check('  조향도 0', st == 0.0, '(%.1f도)' % math.degrees(st))

print('\n[5] PID 는 원본 그대로')
pid = cf.PIDController(kp=0.5)
o1 = pid.tick(300, 200)          # setpoint 가 오른쪽
o2 = cf.PIDController(kp=0.5).tick(100, 200)   # setpoint 가 왼쪽
check('setpoint 가 크면 출력 양수', o1 > 0, '(%.3f)' % o1)
check('setpoint 가 작으면 출력 음수', o2 < 0, '(%.3f)' % o2)
check('출력이 ±1 안', abs(o1) <= 1.0 and abs(o2) <= 1.0)

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
