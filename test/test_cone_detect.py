"""cone_detect_node 검증 (카메라 초록 + 라이다 융합)."""
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
                   'cone_detect_node.py')
spec = importlib.util.spec_from_file_location('cone_detect_node', SRC)
cone = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cone)

FAILS = []
W, H = 640, 480


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def scene(cones=(), grass=None):
    """어두운 노면 + 세로로 길쭉한 초록 콘 + (선택) 초록 지형."""
    img = np.full((H, W, 3), 60, np.uint8)
    if grass:
        x, y, gw, gh = grass
        img[y:y + gh, x:x + gw] = (60, 150, 60)
    for x, y, hh in cones:
        img[y:y + hh, x - 8:x + 8] = (60, 255, 60)
    return img


def scan_at(deg, dist, n=360):
    s = ros_stubs.LaserScan()
    s.angle_min = -math.pi
    s.angle_increment = 2 * math.pi / n
    s.ranges = [float('inf')] * n
    idx = int((math.radians(deg) - s.angle_min) / s.angle_increment) % n
    for k in (-2, -1, 0, 1, 2):
        s.ranges[(idx + k) % n] = dist
    return s


node = cone.ConeDetectNode()

print('\n[1] 콘 검출')
check('콘 1개 검출', len(node.find_cones(scene(cones=[(320, 300, 40)]))[0]) == 1)
check('콘 3개 검출',
      len(node.find_cones(scene(cones=[(160, 300, 40), (320, 310, 36),
                                       (480, 300, 40)]))[0]) == 3)
check('콘 없으면 0', len(node.find_cones(scene())[0]) == 0)

print('\n[2] 초록 지형 배제')
# 잔디는 화면의 큰 영역을 차지한다. 이게 콘과 가르는 진짜 판별점.
check('큰 잔디밭은 콘이 아님',
      len(node.find_cones(scene(grass=(0, 260, 300, 220)))[0]) == 0)
check('넓적한 띠도 콘이 아님',
      len(node.find_cones(scene(grass=(0, 400, 500, 30)))[0]) == 0)
check('잔디 옆의 콘은 그대로 잡힘',
      len(node.find_cones(scene(cones=[(320, 300, 40)],
                                grass=(0, 260, 300, 220)))[0]) == 1)
# ROI 위쪽(하늘/먼 배경)의 초록은 애초에 안 본다
check('ROI 밖 초록은 무시',
      len(node.find_cones(scene(cones=[(320, 20, 40)]))[0]) == 0)

print('\n[3] 방위 계산')
node.on_image(ros_stubs.Image(cv=scene(cones=[(160, 300, 40)])))
b_left = node.last('cone/bearing')
node.on_image(ros_stubs.Image(cv=scene(cones=[(480, 300, 40)])))
b_right = node.last('cone/bearing')
check('좌측 콘 -> 방위 양수 (ROS 관례)', b_left > 0.1, '(%.3f rad)' % b_left)
check('우측 콘 -> 방위 음수', b_right < -0.1, '(%.3f rad)' % b_right)
node.on_image(ros_stubs.Image(cv=scene(cones=[(320, 300, 40)])))
check('정면 콘 -> 방위 ~0', abs(node.last('cone/bearing')) < 0.05,
      '(%.3f rad)' % node.last('cone/bearing'))

print('\n[4] 거리 - 카메라 역산 (라이다 없을 때)')
node2 = cone.ConeDetectNode()
node2.on_image(ros_stubs.Image(cv=scene(cones=[(320, 300, 40)])))
d_far = node2.last('cone/distance')
node2.on_image(ros_stubs.Image(cv=scene(cones=[(320, 260, 80)])))
d_near = node2.last('cone/distance')
check('픽셀 높이가 클수록 가깝게 추정', d_near < d_far,
      '(%.2fm -> %.2fm)' % (d_far, d_near))
check('유한한 값', math.isfinite(d_far))

print('\n[5] 라이다 융합')
node3 = cone.ConeDetectNode()
node3.on_scan(scan_at(0.0, 1.23))
node3.on_image(ros_stubs.Image(cv=scene(cones=[(320, 300, 40)])))
check('라이다가 보면 그 거리를 쓴다', abs(node3.last('cone/distance') - 1.23) < 0.01,
      '(%.3f m)' % node3.last('cone/distance'))

# 콘 방위와 다른 곳의 라이다 점은 쓰지 않는다
node4 = cone.ConeDetectNode()
node4.on_scan(scan_at(60.0, 0.5))          # 콘과 무관한 방위
node4.on_image(ros_stubs.Image(cv=scene(cones=[(320, 300, 40)])))
check('엉뚱한 방위의 라이다 점은 무시',
      abs(node4.last('cone/distance') - 0.5) > 0.05,
      '(%.3f m -> 카메라 역산으로 폴백)' % node4.last('cone/distance'))

print('\n[6] 라이다 단독은 콘이 아니다')
# 트랙 밖 지형이 대부분이라 라이다만 믿으면 헤어핀마다 헛브레이크가 걸린다
node5 = cone.ConeDetectNode()
node5.on_scan(scan_at(0.0, 0.8))
node5.on_image(ros_stubs.Image(cv=scene()))      # 화면에 콘 없음
check('카메라가 못 보면 detected=False', node5.last('cone/detected') is False)
check('  거리도 무한', not math.isfinite(node5.last('cone/distance')))

print('\n[7] 횡방향 오프셋')
node6 = cone.ConeDetectNode()
node6.on_scan(scan_at(0.0, 1.0))
node6.on_image(ros_stubs.Image(cv=scene(cones=[(320, 300, 40)])))
check('정면 콘 -> 횡오프셋 ~0', abs(node6.last('cone/lateral')) < 0.05,
      '(%.3f m)' % node6.last('cone/lateral'))

print('\n[8] 카메라 사망')
node7 = cone.ConeDetectNode()
node7.on_image(ros_stubs.Image(cv=None))
check('변환 실패 -> detected=False', node7.last('cone/detected') is False)

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
