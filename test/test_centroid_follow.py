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
        for y in range(int(H * 0.40), H, 40):
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
# 게인은 매 프레임 _kp_base 에서 다시 정해지므로(코너 부스트) 그쪽을 바꾼다
nf._kp_base = 20.0
nf.on_image(ros_stubs.Image(cv=scene(0.15)))
st_far = nf._steer_cmd
check('큰 오차에서 포화한다 (전제 확인)',
      abs(st_far) > cf.MAX_STEER * 0.9, '(%.1f도)' % math.degrees(st_far))
check('한계 안 (±20도)', abs(st_far) <= cf.MAX_STEER + 1e-9,
      '(%.1f도)' % math.degrees(st_far))

print('\n[4] 처음부터 선이 없으면 안 움직인다')
# C(신뢰도) 도입 후 계약이 바뀌었다. 달리던 중 놓치면 '유지'지만,
# 한 번도 못 봤으면 초기값(0) 그대로라 움직이지 않는다.
st, sp = run(None)
check('한 번도 못 보면 정지 상태', sp == 0.0, '(speed %.2f)' % sp)
check('  조향도 0', st == 0.0, '(%.1f도)' % math.degrees(st))

print('\n[5] PID 는 원본 그대로')
pid = cf.PIDController(kp=0.5)
o1 = pid.tick(300, 200)          # setpoint 가 오른쪽
o2 = cf.PIDController(kp=0.5).tick(100, 200)   # setpoint 가 왼쪽
check('setpoint 가 크면 출력 양수', o1 > 0, '(%.3f)' % o1)
check('setpoint 가 작으면 출력 음수', o2 < 0, '(%.3f)' % o2)
check('출력이 ±1 안', abs(o1) <= 1.0 and abs(o2) <= 1.0)

print('\n[6] 코너 대응 - donkeycar/parts/line_follower.py')

# A. 오차가 크면(코너) 감속, 작으면(직선) 가속
na = cf.CentroidFollowNode()
for _ in range(8):
    na.on_image(ros_stubs.Image(cv=scene(0.50)))     # 직선
v_straight = na._speed_cmd
for _ in range(8):
    na.on_image(ros_stubs.Image(cv=scene(0.15)))     # 큰 오차 = 코너
v_corner = na._speed_cmd
check('A 직선에서 가속', v_straight > na.speed_min, '(%.2f)' % v_straight)
check('A 코너에서 감속', v_corner < v_straight,
      '(%.2f -> %.2f)' % (v_straight, v_corner))
check('  하한을 안 넘는다', v_corner >= na.speed_min - 1e-9, '(%.2f)' % v_corner)

# B. 불감대 -- 선 근처에서 조향을 안 바꾼다
nb = cf.CentroidFollowNode()
nb.on_image(ros_stubs.Image(cv=scene(0.50)))
s_mid = nb._steer_cmd
nb.on_image(ros_stubs.Image(cv=scene(0.52)))         # 아주 조금 벗어남
check('B 불감대 안에서는 조향 고정', abs(nb._steer_cmd - s_mid) < 1e-9,
      '(%.2f -> %.2f도)' % (math.degrees(s_mid), math.degrees(nb._steer_cmd)))
nb.on_image(ros_stubs.Image(cv=scene(0.20)))         # 크게 벗어남
check('B 불감대 밖에서는 반응', abs(nb._steer_cmd) > math.radians(1),
      '(%.1f도)' % math.degrees(nb._steer_cmd))

# C. 신뢰도 -- 달리던 중 선을 놓쳐도 마지막 조향 유지 (예전엔 정지했다)
nc = cf.CentroidFollowNode()
for _ in range(5):
    nc.on_image(ros_stubs.Image(cv=scene(0.20)))
s_before, v_before = nc._steer_cmd, nc._speed_cmd
nc.on_image(ros_stubs.Image(cv=scene(None)))         # 선 사라짐
check('C 선을 놓쳐도 조향 유지', abs(nc._steer_cmd - s_before) < 1e-9,
      '(%.1f -> %.1f도)' % (math.degrees(s_before), math.degrees(nc._steer_cmd)))
check('C 속도도 유지 (정지하지 않는다)', nc._speed_cmd == v_before,
      '(%.2f -> %.2f)' % (v_before, nc._speed_cmd))

print('\n[6b] 속도 프로파일 - 물리 한계 안에서 빠르게')
# 최소 회전반경 R = 0.18/tan(20°) = 0.495m. 횡가속 a = v²/R.
# 미끄러짐 기준을 대략 1.5 m/s² 로 보면 코너 안전속도는 0.86 m/s.
# 직선은 반경 제약이 없으므로 훨씬 높아도 된다.
R_MIN = 0.18 / math.tan(math.radians(20))


def speed_after(mid, straight_ticks=40, corner_ticks=15):
    nn = cf.CentroidFollowNode()
    for _ in range(straight_ticks):
        nn.on_image(ros_stubs.Image(cv=scene(0.50)))     # 직선에서 가속
    if mid is not None:
        for _ in range(corner_ticks):
            nn.on_image(ros_stubs.Image(cv=scene(mid)))  # 코너 진입
    return nn._speed_cmd


v_straight = speed_after(None)
# 이 scene() 은 ROI 전체에 같은 위치로 그리므로 먼 줄도 함께 꺾인 것으로 읽힌다.
# 그러면 선행 감속까지 겹쳐 완만/급 둘 다 하한에 닿아 비교가 안 된다.
# 여기서는 '오차 비례 감속'만 보려는 것이므로 선행 스캔을 끄고 잰다.
def speed_no_lookahead(mid, straight_ticks=40, corner_ticks=15):
    nn = cf.CentroidFollowNode()
    nn.lookahead_enable = False
    for _ in range(straight_ticks):
        nn.on_image(ros_stubs.Image(cv=scene(0.50)))
    for _ in range(corner_ticks):
        nn.on_image(ros_stubs.Image(cv=scene(mid)))
    return nn._speed_cmd


v_gentle = speed_no_lookahead(0.40)
v_sharp = speed_no_lookahead(0.12)

check('직선에서 speed_max 까지 가속', v_straight > 1.0,
      '(%.2f m/s)' % v_straight)
check('급커브가 물리 한계 안', (v_sharp ** 2) / R_MIN < 1.5,
      '(%.2f m/s, 횡가속 %.2f m/s²)' % (v_sharp, v_sharp ** 2 / R_MIN))
check('오차에 비례해 감속 (완만 > 급커브)', v_gentle > v_sharp,
      '(완만 %.2f > 급 %.2f)' % (v_gentle, v_sharp))
check('하한 아래로는 안 떨어짐', v_sharp >= cf.CentroidFollowNode().speed_min - 1e-9,
      '(%.2f)' % v_sharp)

# 감속이 코너 진입 안에 끝나야 한다. 고정 step 이면 max 를 올릴수록 느려진다.
nb2 = cf.CentroidFollowNode()
nb2._throttle = nb2.speed_max
ticks = 0
while nb2._throttle > nb2.speed_min + 0.05 and ticks < 100:
    nb2.on_image(ros_stubs.Image(cv=scene(0.12)))
    ticks += 1
check('급커브 감속이 빠르다 (30Hz 기준 0.5초 안)', ticks < 15,
      '(%d프레임 = %.2f초)' % (ticks, ticks / 30.0))

print('\n[6c] 선행 스캔 - 코너를 미리 보고 감속하는가 (90도 코너 대응)')
# 감속을 '오차가 커진 뒤'에 시작하면 이미 코너 안이다. 1.2 m/s 로 진입하면
# 필요 횡가속이 2.91 m/s² 라 물리적으로 못 돈다(최소 반경 0.495m).
# 먼 줄을 따로 보고, 아직 직진 중이어도 미리 줄여야 한다.


def two_band_scene(near_x=0.5, far_x=None):
    """ROI 안에서 가까운 줄과 먼 줄의 선 위치를 따로 준다."""
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[:, :] = (106, 113, 73)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    m = cv2.cvtColor(np.uint8([[(20, 255, 230)]]), cv2.COLOR_HSV2BGR)[0][0]
    r0, r1 = int(H * 0.45), int(H * 0.80)
    rh = r1 - r0

    def draw(x, y0, y1):
        for y in range(y0, y1, 40):
            xi = int(W * x)
            if 5 <= xi < W - 5:
                bgr[y:y + 22, xi - 5:xi + 5] = m

    draw(far_x if far_x is not None else near_x, r0, r0 + int(rh * 0.35))
    draw(near_x, r0 + int(rh * 0.45), r1)
    return bgr


def drive(near_x, far_x, ticks=25):
    nn = cf.CentroidFollowNode()
    for _ in range(40):                       # 직선에서 가속
        nn.on_image(ros_stubs.Image(cv=two_band_scene(0.5, 0.5)))
    v0 = nn._speed_cmd
    for _ in range(ticks):
        nn.on_image(ros_stubs.Image(cv=two_band_scene(near_x, far_x)))
    return v0, nn._speed_cmd, nn._steer_cmd


v0, v_straight, st_straight = drive(0.5, 0.5)
check('직선에서는 최고속 유지', v_straight >= v0 - 1e-9,
      '(%.2f -> %.2f)' % (v0, v_straight))

# 핵심: 가까운 줄은 아직 직선인데 먼 줄에 코너가 보이는 상황
_, v_ahead, st_ahead = drive(0.5, 0.15)
check('앞에 코너가 보이면 미리 감속 ★', v_ahead < v0 - 0.2,
      '(%.2f -> %.2f)' % (v0, v_ahead))
check('  그때 조향은 아직 직진', abs(math.degrees(st_ahead)) < 2.0,
      '(%.1f도)' % math.degrees(st_ahead))
check('  감속 후 90도 코너가 물리적으로 가능', (v_ahead ** 2) / R_MIN < 1.5,
      '(%.2f m/s, 횡가속 %.2f)' % (v_ahead, v_ahead ** 2 / R_MIN))

# 감속 없이 그 속도로 진입하면 불가능하다는 것 -- 전제 확인
check('  선행 감속이 없었다면 불가능했다 (전제 확인)',
      (v0 ** 2) / R_MIN > 2.0,
      '(%.2f m/s 였다면 횡가속 %.2f)' % (v0, v0 ** 2 / R_MIN))

_, v_in, st_in = drive(0.2, 0.12)
check('코너 진입하면 조향도 반응', abs(math.degrees(st_in)) > 5.0,
      '(%.1f도)' % math.degrees(st_in))

# 먼 줄은 조향에 쓰지 않는다 -- 먼 곳은 부정확해서 넣으면 불안정해진다
_, _, st_far_only = drive(0.5, 0.15)
_, _, st_near_only = drive(0.5, 0.5)
check('먼 줄은 조향에 안 쓴다', abs(st_far_only - st_near_only) < 1e-9,
      '(%.1f도 = %.1f도)'
      % (math.degrees(st_far_only), math.degrees(st_near_only)))

print('\n[6d] 코너 전용 게인 부스트 - 코너만 세게, 직선은 그대로')
# 게인을 올리면 코너 추종이 좋아지지만 인지 잡음이 클 때 직선이 떨린다.
# 그래서 상시로 올리지 않고 코너에서만 올린다.


def track_sim(gain_scale, corner=True, noise=0.0, ticks=70):
    """차량 동역학 근사 - 코너가 차를 밀고, 조향이 되돌린다."""
    nn = cf.CentroidFollowNode()
    nn.corner_gain_scale = gain_scale
    rng = np.random.RandomState(1)
    x = 0.5
    errs = []
    steers = []
    for t in range(ticks):
        far = (0.15 if t >= 5 else 0.5) if corner else 0.5
        xs = min(0.95, max(0.05, x + rng.randn() * noise))
        nn.on_image(ros_stubs.Image(cv=two_band_scene(xs, far)))
        st = nn._steer_cmd
        steers.append(st)
        drift = (-0.020 if t >= 10 else 0.0) if corner else 0.0
        x = min(0.95, max(0.05, x + drift + st * 0.09))
        if t >= 15:
            errs.append(abs(x - 0.5))
    jitter = math.degrees(float(np.std(np.diff(steers[20:]))))
    return sum(errs) / max(1, len(errs)), jitter


err_off, _ = track_sim(1.0, corner=True)
err_on, _ = track_sim(2.2, corner=True)
check('코너 부스트가 이탈을 줄인다', err_on < err_off * 0.7,
      '(부스트 없음 %.3f -> 있음 %.3f)' % (err_off, err_on))

# 핵심: 직선 떨림은 늘지 않아야 한다. 상시로 게인을 올리면 여기서 망가진다.
_, jit_off = track_sim(1.0, corner=False, noise=0.06)
_, jit_on = track_sim(2.2, corner=False, noise=0.06)
check('직선 떨림은 거의 그대로 ★', jit_on < jit_off * 1.5,
      '(%.2f도 -> %.2f도)' % (jit_off, jit_on))

# 부스트가 실제로 코너에서만 켜지는지
nk = cf.CentroidFollowNode()
kp_base = nk._kp_base
for _ in range(6):
    nk.on_image(ros_stubs.Image(cv=two_band_scene(0.5, 0.5)))   # 직선
check('직선에서는 기본 게인', abs(nk.pid.kp - kp_base) < 1e-9,
      '(kp=%.1f)' % nk.pid.kp)
for _ in range(6):
    nk.on_image(ros_stubs.Image(cv=two_band_scene(0.15, 0.15)))  # 코너
check('코너에서는 게인 상승', nk.pid.kp > kp_base * 1.5,
      '(kp=%.1f -> %.1f)' % (kp_base, nk.pid.kp))

# D 항은 0 이어야 한다 -- 원본 미분기는 prevMeas 가 상수라 잡음만 증폭한다
check('D 항은 기본 0 (잡음만 증폭)', cf.CentroidFollowNode().pid.kd == 0.0)

print('\n[6e] 선 유실 복구 - 90도 코너에서 선이 시야를 벗어날 때')
# 참조: nsa31/Line-Lane-Follower-Robot_ROS white_yellow_lane_follower_sim.py
#   else:                    # 선을 못 찾았을 때
#       linear.x = 0.4       # 평소 0.9 -> 절반 이하로 감속
#       angular.z = -0.7     # 강하게 회전
# 90도 코너에서는 선이 시야를 완전히 벗어난다. '마지막 조향 유지'만으로는
# 못 따라잡는다 -- 진입 직전 조향은 코너를 다 돌기에 모자란 값이다.


def corner_exit(lost_recover, approach=(0.5, 0.35, 0.2, 0.1), blind=4):
    """선이 왼쪽으로 밀리다 시야를 벗어나는 90도 코너."""
    nn = cf.CentroidFollowNode()
    nn.lost_recover = lost_recover
    for x in approach:
        nn.on_image(ros_stubs.Image(cv=scene(x)))
    for _ in range(blind):
        nn.on_image(ros_stubs.Image(cv=scene(None)))
    return nn._steer_cmd, nn._speed_cmd


st_off, v_off = corner_exit(False)
st_on, v_on = corner_exit(True)

check('선을 잃으면 감속한다', v_on < v_off - 0.05,
      '(유지 %.2f -> 복구 %.2f)' % (v_off, v_on))
check('  최대 조향은 유지', abs(st_on) > cf.MAX_STEER * 0.9,
      '(%.1f도)' % math.degrees(st_on))

# 느려야 그 조향으로 실제로 돌 수 있다
R_act = 0.18 / math.tan(abs(st_on))
check('  느려진 만큼 횡가속 여유가 생긴다', (v_on ** 2) / R_act < 0.5,
      '(%.2f m/s, 횡가속 %.2f)' % (v_on, v_on ** 2 / R_act))

# 방향: 마지막으로 본 선이 왼쪽이었으면 좌회전
check('마지막으로 본 방향으로 꺾는다 (좌)', st_on > 0,
      '(%.1f도)' % math.degrees(st_on))
st_r, _ = corner_exit(True, approach=(0.5, 0.65, 0.8, 0.9))
check('  반대쪽도 마찬가지 (우)', st_r < 0, '(%.1f도)' % math.degrees(st_r))

# 한두 프레임 튄 것과는 구분해야 한다
nq = cf.CentroidFollowNode()
for x in (0.5, 0.5, 0.5):
    nq.on_image(ros_stubs.Image(cv=scene(x)))
s_before = nq._steer_cmd
nq.on_image(ros_stubs.Image(cv=scene(None)))     # 딱 1프레임만 놓침
check('1프레임 유실은 복구 조향을 안 건다',
      abs(nq._steer_cmd - s_before) < 1e-9,
      '(%.1f도 유지)' % math.degrees(nq._steer_cmd))

print('\n[7] 조향 범위 - 화면 끝에서 최대 조향이 나오는가')
# 원본 errorNormalize=1/400 은 800px 폭 카메라 기준이다. 우리 카메라(640/320)에
# 그대로 쓰면 화면 끝에서도 8도(240p 면 4도)까지밖에 안 나온다 -- 코너에서
# 조향이 부족한 직접 원인이었다. 화면 반폭으로 정규화하면 해결된다.


def steer_at(w, h, mid):
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[:, :] = (106, 113, 73)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if mid is not None:
        m = cv2.cvtColor(np.uint8([[(20, 255, 230)]]), cv2.COLOR_HSV2BGR)[0][0]
        t = max(3, w // 64)
        for y in range(int(h * 0.40), h, 40):
            x = int(w * mid)
            bgr[y:y + 22, max(0, x - t):x + t] = m
    nn = cf.CentroidFollowNode()
    nn.on_image(ros_stubs.Image(cv=bgr))
    return nn._steer_cmd


s_edge = steer_at(640, 480, 0.05)
s_half = steer_at(640, 480, 0.35)
check('화면 끝에서 최대 조향', abs(s_edge) > cf.MAX_STEER * 0.95,
      '(%.1f도)' % math.degrees(s_edge))
check('중간에서는 중간값', math.radians(3) < abs(s_half) < cf.MAX_STEER * 0.7,
      '(%.1f도)' % math.degrees(s_half))

s_edge_240 = steer_at(320, 240, 0.05)
check('해상도가 달라도 같은 조향', abs(s_edge - s_edge_240) < math.radians(0.5),
      '(640p %.1f도 / 240p %.1f도)'
      % (math.degrees(s_edge), math.degrees(s_edge_240)))

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
