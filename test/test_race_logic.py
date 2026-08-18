"""physicar 노드 알고리즘 검증 (ROS 없이 스텁 런타임에서 구동)."""
import importlib.util
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ros_stubs  # noqa: E402

ros_stubs.install()

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PKG = os.path.join(ROOT, 'physicar_race', 'physicar_race')

lane_mod = load(os.path.join(PKG, 'lane_detect_node.py'), 'lane_detect_node')
tl_mod = load(os.path.join(PKG, 'traffic_light_node.py'), 'traffic_light_node')
obs_mod = load(os.path.join(PKG, 'lane_obstacle_node.py'), 'lane_obstacle_node')
jud_mod = load(os.path.join(PKG, 'race_judgment_node.py'), 'race_judgment_node')

FAILS = []
W, H = 640, 480


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def road(white_left=None, white_right=None, yellow=None, thickness=8):
    """어두운 노면 + 흰 실선(양쪽) + 노란 점선(중앙)."""
    img = np.full((H, W, 3), 60, np.uint8)

    def vline(x, color, dashed=False):
        if x is None:
            return
        x0, x1 = max(0, int(x - thickness / 2)), min(W, int(x + thickness / 2))
        if x1 <= x0:
            return
        if dashed:
            for y0 in range(0, H, 60):
                img[y0:min(H, y0 + 35), x0:x1] = color
        else:
            img[:, x0:x1] = color

    vline(white_left, (255, 255, 255))
    vline(white_right, (255, 255, 255))
    vline(yellow, (0, 255, 255), dashed=True)
    return img


def feed(node, img):
    node.clear()
    node.on_image(ros_stubs.Image(cv=img))


# ====================================================================== 1
print('\n[1] lane_detect_node - 2차선 기하')
ln = lane_mod.LaneDetectNode()

# A. 오른쪽 차선 정중앙 (노란선 160, 우측 흰선 480 -> 차선중심 320 = 화면중심)
feed(ln, road(white_left=20, white_right=480, yellow=160))
check('A valid', ln.last('lane/valid') is True)
check('A 현재차선=RIGHT', ln.last('lane/current_lane') == lane_mod.LANE_RIGHT,
      '(=%s)' % ln.last('lane/current_lane'))
check('A offset_right≈0', abs(ln.last('lane/offset_right')) < 0.05,
      '(=%.3f)' % ln.last('lane/offset_right'))
check('A margin_right≈0.5', abs(ln.last('lane/margin_right') - 0.5) < 0.05,
      '(=%.3f)' % ln.last('lane/margin_right'))

# B. 우측 흰선으로 치우침 (우측 흰선 400 -> 차선중심 280, 화면중심 320 -> +0.125)
feed(ln, road(white_left=20, white_right=400, yellow=160))
check('B offset_right>0 (중심보다 우측)', ln.last('lane/offset_right') > 0.08,
      '(=%.3f)' % ln.last('lane/offset_right'))
check('B margin_right≈0.25', abs(ln.last('lane/margin_right') - 0.25) < 0.05,
      '(=%.3f)' % ln.last('lane/margin_right'))

# C. 왼쪽 차선 (노란선 480, 좌측 흰선 160 -> 차선중심 320)
feed(ln, road(white_left=160, white_right=None, yellow=480))
check('C 현재차선=LEFT', ln.last('lane/current_lane') == lane_mod.LANE_LEFT,
      '(=%s)' % ln.last('lane/current_lane'))
check('C offset_left≈0', abs(ln.last('lane/offset_left')) < 0.05,
      '(=%.3f)' % ln.last('lane/offset_left'))

# D. 흰선 밟기 직전 (좌측 흰선이 화면중심 바로 왼쪽)
feed(ln, road(white_left=290, white_right=620, yellow=None))
m_l = ln.last('lane/margin_left')
check('D margin_left 작음', m_l is not None and m_l < 0.15, '(=%.3f)' % (m_l or -9))

# E. 흰선 전무 -> invalid
feed(ln, road())
check('E 흰선 없음 -> valid=False', ln.last('lane/valid') is False)

# F. 노란선 점선 공백 -> 홀드 후 UNKNOWN
ln2 = lane_mod.LaneDetectNode()
feed(ln2, road(white_left=20, white_right=480, yellow=160))
check('F 노란선 보임 -> RIGHT', ln2.last('lane/current_lane') == lane_mod.LANE_RIGHT)
ln2._yellow_stamp -= 10.0  # 홀드 시간 만료 강제
feed(ln2, road(white_left=20, white_right=480, yellow=None))
check('F 홀드 만료 -> UNKNOWN', ln2.last('lane/current_lane') == lane_mod.LANE_UNKNOWN,
      '(=%s)' % ln2.last('lane/current_lane'))
check('F UNKNOWN이어도 valid 유지', ln2.last('lane/valid') is True)

# ====================================================================== 2
print('\n[2] traffic_light_node - 적/녹 판정')
tl = tl_mod.TrafficLightNode()


def light(color):
    """화면 위쪽에 신호등 램프 하나. color=None 이면 신호등 없음."""
    img = np.full((H, W, 3), 30, np.uint8)
    if color is not None:
        img[60:130, 300:370] = color
    return img


for name, bgr, want in (('RED', (0, 0, 255), 'RED'),
                        ('GREEN', (0, 255, 0), 'GREEN'),
                        ('없음', None, 'NONE')):
    tl.clear()
    tl.on_image(ros_stubs.Image(cv=light(bgr)))
    check('신호등 %s' % name, tl.last('traffic/light_state') == want,
          '(=%s)' % tl.last('traffic/light_state'))

# '카메라 사망'과 '신호등이 안 보임'은 반드시 구분돼야 한다.
# 전자만 정지 사유이고 후자는 출발 후 정상 상태다.
tl.clear()
tl.on_image(ros_stubs.Image(cv=None))
check('카메라 사망 -> valid=False', tl.last('traffic/valid') is False)
check('신호등 없음 -> valid=True 유지',
      (lambda: (tl.clear(), tl.on_image(ros_stubs.Image(cv=light(None))),
                tl.last('traffic/valid'))[2])() is True)


def probe_lines(node):
    return [m for lvl, m in node._logger.lines if '[probe]' in m]


# 진단 모드: 실측 H/S/V 를 로그로 뽑아준다
tlp = tl_mod.TrafficLightNode()
tlp.debug_probe = True
tlp._probe_stamp = 0.0
tlp.on_image(ros_stubs.Image(cv=light((0, 255, 0))))       # 순수 초록: H=60 S=255
pl = probe_lines(tlp)
check('probe 로그 출력됨', len(pl) > 0)
check('probe 가 초록 색상(H=60) 보고', bool(pl) and 'H=60' in pl[-1],
      '(%s)' % (pl[-1].split('\n')[-1].strip() if pl else '없음'))

# 실전에서 제일 흔한 실패: LED 가운데가 하얗게 떠서 채도가 낮게 잡히는 경우.
# 검출은 실패(NONE)하지만 probe 는 낮은 S 를 그대로 보고해서 원인을 알려줘야 한다.
tlw = tl_mod.TrafficLightNode()
tlw.debug_probe = True
tlw._probe_stamp = 0.0
tlw.clear()
tlw.on_image(ros_stubs.Image(cv=light((180, 255, 180))))   # 흐린 초록: S 가 낮다
pw = probe_lines(tlw)
check('흐린 초록 -> 검출 실패(NONE)', tlw.last('traffic/light_state') == 'NONE',
      '(=%s)' % tlw.last('traffic/light_state'))
check('probe 가 낮은 채도를 원인으로 노출',
      bool(pw) and any('S=%d' % s in pw[-1] for s in range(60, 100)),
      '(%s)' % (pw[-1].split('\n')[-1].strip() if pw else '없음'))


# 실전에서 실제로 만난 상황: 밝은 하늘이 화면을 덮고 있고 램프가 거기 붙어
# 있으면, 밝기만으로 덩어리를 찾을 때 둘이 하나로 합쳐져 중앙값이 하늘 색으로
# 나온다. 면적 순위로도 램프는 하늘에 밀려 안 보인다.
# 색상 구간을 먼저 좁히면 램프가 분리돼서 잡혀야 한다.
def sky_with_lamp():
    img = np.full((H, W, 3), 40, np.uint8)
    img[0:int(H * 0.45), :] = (235, 180, 120)     # 밝은 하늘 (H~104)
    img[80:110, 300:330] = (0, 255, 0)            # 하늘에 맞닿은 작은 초록 램프
    return img


tlm = tl_mod.TrafficLightNode()
tlm.debug_probe = True
tlm._probe_stamp = 0.0
tlm.on_image(ros_stubs.Image(cv=sky_with_lamp()))
pm = probe_lines(tlm)
msg = pm[-1] if pm else ''
green_sec = msg.split('초록 후보')[-1].split('빨강 후보')[0] if '초록 후보' in msg else ''
bright_sec = msg.split('밝은 영역')[-1].split('초록 후보')[0] if '밝은 영역' in msg else ''

check('밝은 영역은 하늘에 가려 초록을 못 짚음',
      'H=60' not in bright_sec,
      '(하늘 H가 지배: %s)' % bright_sec.strip().split('\n')[0].strip())
check('초록 후보 구간에서는 램프를 분리해서 찾음', 'H=60' in green_sec,
      '(%s)' % green_sec.strip().split('\n')[0].strip())
check('후보마다 통과/탈락 사유를 붙임',
      '통과' in green_sec or '탈락' in green_sec)

# ====================================================================== 3
print('\n[3] lane_obstacle_node - 차선 점유 판정')
ob = obs_mod.LaneObstacleNode()
ob.front_offset = 0.0  # 테스트는 라이다 정면=0도 가정


def scan(points, n=360):
    """points: [(각도deg, 거리m)] -> 나머지는 inf"""
    s = ros_stubs.LaserScan()
    s.angle_min = -math.pi
    s.angle_increment = 2 * math.pi / n
    r = [float('inf')] * n
    for deg, dist in points:
        idx = int((math.radians(deg) - s.angle_min) / s.angle_increment) % n
        for k in (-1, 0, 1):  # min_points=3 충족
            r[(idx + k) % n] = dist
    s.ranges = r
    return s


ob.current_lane = obs_mod.LANE_RIGHT   # 반대 차선은 좌측(y=+0.5)

ob.clear()
ob.on_scan(scan([(0.0, 1.0)]))          # 정면 1m -> 현재 차선
check('정면 장애물 -> blocked_current', ob.last('obstacle/blocked_current') is True)
check('정면 장애물 -> blocked_other 아님', ob.last('obstacle/blocked_other') is False)

ob.clear()
ob.on_scan(scan([(27.0, 1.1)]))         # y=+0.50m, x=0.98m -> 좌측(반대) 차선
check('좌측 장애물 -> blocked_other', ob.last('obstacle/blocked_other') is True,
      '(cur=%s)' % ob.last('obstacle/blocked_current'))

ob.clear()
ob.on_scan(scan([(0.0, 0.25)]))         # 코앞
check('코앞 장애물 -> emergency', ob.last('obstacle/emergency') is True)

ob.clear()
ob.on_scan(scan([(0.0, 5.0)]))          # lookahead(2m) 밖
check('먼 장애물 -> 막힘 아님', ob.last('obstacle/blocked_current') is False)

# ====================================================================== 4
print('\n[4] race_judgment_node - 레이스 상태기계')


def new_judge():
    j = jud_mod.RaceJudgmentNode()
    j._tick = j._timers[0][1]
    return j


def perceive(j, valid=True, off_r=0.0, off_l=0.0, lane=jud_mod.LANE_RIGHT,
             ml=1.0, mr=1.0, curv=0.0):
    j._cb_lane_valid(ros_stubs.Bool(valid))
    j._cb_off_r(ros_stubs.Float64(off_r))
    j._cb_off_l(ros_stubs.Float64(off_l))
    j._cb_lane(ros_stubs.Int32(lane))
    j._cb_margin_l(ros_stubs.Float64(ml))
    j._cb_margin_r(ros_stubs.Float64(mr))
    j._cb_curv(ros_stubs.Float64(curv))


def obstacles(j, cur=False, oth=False, emg=False, near=float('inf')):
    j._cb_blk_cur(ros_stubs.Bool(cur))
    j._cb_blk_oth(ros_stubs.Bool(oth))
    j._cb_emg(ros_stubs.Bool(emg))
    j._cb_near(ros_stubs.Float64(near))


# 4-1 출발 게이트
j = new_judge()
perceive(j)
obstacles(j)
j._cb_traffic_valid(ros_stubs.Bool(True))
j._cb_light(ros_stubs.String('RED'))
j.clear(); j._tick()
check('빨간불 -> 정지', j.last('/speed') == 0.0, '(v=%.2f)' % j.last('/speed'))
check('빨간불 -> 상태 WAIT_GREEN', j.last('race/state') == 'WAIT_GREEN')

j._cb_light(ros_stubs.String('GREEN'))
need = j.green_confirm_frames
for _ in range(need - 1):
    j.clear(); j._tick()
check('초록 %d프레임(확인 %d 미만) -> 아직 정지' % (need - 1, need),
      j.last('/speed') == 0.0)
j.clear(); j._tick()
check('초록 %d프레임 -> 출발' % need, j.last('/speed') > 0.0,
      '(v=%.2f)' % j.last('/speed'))
check('출발 후 상태 RACING', j.last('race/state') == 'RACING')

# 4-2 출발 후 빨간불 무시 (주행 중 오검출 방어)
j._cb_light(ros_stubs.String('RED'))
perceive(j); obstacles(j)
j.clear(); j._tick()
check('출발 후 RED는 무시', j.last('/speed') > 0.0, '(v=%.2f)' % j.last('/speed'))

# 4-3 응급 정지
perceive(j); obstacles(j, emg=True)
j.clear(); j._tick()
check('emergency -> 정지', j.last('/speed') == 0.0)
check('emergency -> 상태', j.last('race/state') == 'EMERGENCY')

# 4-4 차선 인지 유실 -> 정지 (흰선 위치를 모르면 실격 위험 통제 불가)
perceive(j, valid=False); obstacles(j)
j.clear(); j._tick()
check('lane invalid -> 정지', j.last('/speed') == 0.0)

# 4-5 차선 변경
j2 = new_judge()
j2.require_green = False
j2.state = jud_mod.ST_RACING
perceive(j2, lane=jud_mod.LANE_RIGHT)
obstacles(j2, cur=True, oth=False, near=1.5)
j2.last_change = 0.0
j2.clear(); j2._tick()
check('현재차선 막힘+반대 비었음 -> LEFT로 목표 변경',
      j2.target_lane == jud_mod.LANE_LEFT, '(=%s)' % j2.target_lane)

# 4-6 양쪽 다 막힘 -> 정지
j3 = new_judge()
j3.require_green = False
j3.state = jud_mod.ST_RACING
perceive(j3, lane=jud_mod.LANE_RIGHT)
obstacles(j3, cur=True, oth=True, near=1.0)
j3.clear(); j3._tick()
check('양쪽 막힘 -> 정지', j3.last('/speed') == 0.0)

# 4-7 흰선 실격 방지 조향
j4 = new_judge()
j4.require_green = False
j4.state = jud_mod.ST_RACING
# 좌측 흰선에 붙음(ml=0.03) -> 우측으로 밀어내야 함(logical steer 음수)
perceive(j4, lane=jud_mod.LANE_RIGHT, ml=0.03, mr=1.5, off_r=0.0)
obstacles(j4)
j4.clear(); j4._tick()
s_left_danger = j4.last('/steering')
check('좌측 흰선 근접 -> 우측 조향(음수)', s_left_danger < -0.05,
      '(steer=%.3f rad)' % s_left_danger)

perceive(j4, lane=jud_mod.LANE_RIGHT, ml=1.5, mr=0.03, off_r=0.0)
j4.clear(); j4._tick()
s_right_danger = j4.last('/steering')
check('우측 흰선 근접 -> 좌측 조향(양수)', s_right_danger > 0.05,
      '(steer=%.3f rad)' % s_right_danger)

# 4-8 출력 한계 준수
j5 = new_judge()
j5.require_green = False
j5.state = jud_mod.ST_RACING
perceive(j5, lane=jud_mod.LANE_RIGHT, ml=0.0, mr=2.0, off_r=-2.0)
obstacles(j5)
j5.clear(); j5._tick()
sp, st = j5.last('/speed'), j5.last('/steering')
check('speed <= 3.0', sp <= jud_mod.MAX_SPEED, '(=%.2f)' % sp)
check('speed >= 0.3 (ESC 데드존)', sp >= jud_mod.MIN_SPEED or sp == 0.0, '(=%.2f)' % sp)
check('|steer| <= 20deg', abs(st) <= jud_mod.MAX_STEER + 1e-9,
      '(=%.1f deg)' % math.degrees(st))

# 4-9 곡률 감속
j6 = new_judge()
j6.require_green = False
j6.state = jud_mod.ST_RACING
perceive(j6, lane=jud_mod.LANE_RIGHT, curv=0.0)
obstacles(j6)
j6.clear(); j6._tick()
v_straight = j6.last('/speed')
perceive(j6, lane=jud_mod.LANE_RIGHT, curv=0.9)
j6.clear(); j6._tick()
v_curve = j6.last('/speed')
check('급커브에서 감속', v_curve < v_straight,
      '(직선 %.2f -> 커브 %.2f)' % (v_straight, v_curve))

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
