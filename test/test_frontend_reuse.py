"""BevFrontend 를 재사용해도 결과가 같은지 본다.

v3 노드가 프레임마다 BevFrontend 를 새로 만들고 있었다. 그 생성자는
cv2.initUndistortRectifyMap 으로 480x360 왜곡보정 맵을 통째로 계산한다.

그런데 frontend.py:28 에 민석이가 직접 써놨다:

    Camera calibration and undistortion maps are pose invariant.
    The BEV source map is not.

그래서 update_projector() 가 있고 v2 노드는 그걸 쓴다
(v2/bev_frontend_node.py:590). v3 만 빠져 있었다.

여기서 증명할 것은 속도가 아니라 **동등성**이다:

    새로 만든 것        vs   재사용 + update_projector
    ------------------------------------------------
    왜곡보정 맵              같아야 한다 (자세 무관)
    BEV 소스 맵              같아야 한다 (자세를 타지만 다시 만든다)
    실제 이미지 처리 결과    바이트 단위로 같아야 한다

자세를 바꿔가며 확인한다 -- 한 자세로만 보면 캐시가 낡았는지 알 수 없다.
"""
import importlib.util
import io
import os
import sys

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


def load(name):
    spec = importlib.util.spec_from_file_location(
        'v2_' + name, os.path.join(V2, name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules['physicar_track_perception_v2.' + name] = module
    spec.loader.exec_module(module)
    return module


geometry = load('geometry')
frontend_mod = load('frontend')

# 실차 값 (perception_v3_real.yaml)
K = np.array([[260.875, 0.0, 231.31516130651107],
              [0.0, 260.875, 169.16236121207476],
              [0.0, 0.0, 1.0]], dtype=np.float64)
D = np.zeros(5, dtype=np.float64)
CAMERA = geometry.CameraModel(K=K, D=D, width=480, height=360)
GRID = geometry.BevGrid(0.10, 2.00, -0.75, 0.75, 0.01)


def pose(height, pitch_deg):
    """base_footprint -> camera. +X 앞, +Y 왼쪽, 아래로 pitch."""
    a = np.radians(pitch_deg)
    # 광학 좌표계: +Z 앞, +X 오른쪽, +Y 아래
    R = np.array([[0.0, -1.0, 0.0],
                  [np.sin(a), 0.0, -np.cos(a)],
                  [np.cos(a), 0.0, np.sin(a)]], dtype=np.float64)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = (0.05, 0.0, height)
    return T


def projector(height, pitch_deg):
    return geometry.MetricGroundProjector(
        CAMERA, GRID, pose(height, pitch_deg), ground_z=0.0)


rng = np.random.default_rng(3)
IMAGE = rng.integers(0, 256, (360, 480, 3), dtype=np.uint8)

POSES = [(0.148, 3.0), (0.148, 12.5), (0.20, -5.0), (0.148, 3.0)]


print('\n[1] 왜곡보정 맵은 자세와 무관한가 (재사용의 전제) ★')
a = frontend_mod.BevFrontend(CAMERA, projector(*POSES[0]))
b = frontend_mod.BevFrontend(CAMERA, projector(*POSES[1]))
check('자세가 달라도 왜곡보정 맵이 같다',
      np.array_equal(a.undistort_map_x, b.undistort_map_x)
      and np.array_equal(a.undistort_map_y, b.undistort_map_y))
check('BEV 소스 맵은 자세를 탄다 (테스트가 헛돌지 않는다)',
      not np.array_equal(a.bev_map_x, b.bev_map_x))


print('\n[2] 재사용한 것과 새로 만든 것이 같은가 ★')
# 노드가 하는 것과 같은 순서로: 한 번 만들고 자세마다 갱신한다.
reused = frontend_mod.BevFrontend(CAMERA, projector(*POSES[0]))
for height, pitch in POSES:
    fresh = frontend_mod.BevFrontend(CAMERA, projector(height, pitch))
    reused.update_projector(projector(height, pitch))

    label = 'h=%.3f pitch=%.1f' % (height, pitch)
    check('%s : BEV 소스 맵이 같다' % label,
          np.array_equal(fresh.bev_map_x, reused.bev_map_x)
          and np.array_equal(fresh.bev_map_y, reused.bev_map_y))

    out_fresh = fresh.process(IMAGE)
    out_reused = reused.process(IMAGE)
    check('%s : BEV 이미지가 바이트 단위로 같다' % label,
          np.array_equal(out_fresh.bev, out_reused.bev))
    check('%s : 유효맵이 같다' % label,
          np.array_equal(out_fresh.validity_mask, out_reused.validity_mask))


print('\n[3] 노드가 실제로 재사용하는가')
source = io.open(NODE, encoding='utf-8').read()
check('BevFrontend 생성이 self.frontend 캐시로 들어갔다',
      'self.frontend=BevFrontend(self.camera,projector)' in source)
check('갱신은 update_projector 로 한다',
      'self.frontend.update_projector(projector)' in source)
check('프레임마다 새로 만들던 코드가 없어졌다',
      'out=BevFrontend(' not in source)


print('\n[4] 단계별 시간 측정이 들어갔는가 (py-spy 대용)')
for name in ('map', 'remap', 'seg', 'total'):
    check("'%s' 구간을 잰다" % name, "st.get('%s'" % name in source)
check('30프레임마다 찍는다', 'V3 timing' in source)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
