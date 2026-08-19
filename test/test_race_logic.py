"""physicar 노드 알고리즘 검증 (ROS 없이 스텁 런타임에서 구동)."""
import importlib.util
import math
import os
import re
import sys
import time

import cv2
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

# A. 목표는 '중앙선을 화면의 정해진 위치에 두기'다. 오른쪽 차선을 달리면
#    중앙선이 내 왼쪽에 보여야 하므로 목표 x = 중심 - center_target_frac*half.
#    기본 0.35 -> 320 - 112 = 208.
feed(ln, road(white_left=20, white_right=480, yellow=208))
check('A valid', ln.last('lane/valid') is True)
check('A 현재차선=RIGHT', ln.last('lane/current_lane') == lane_mod.LANE_RIGHT,
      '(=%s)' % ln.last('lane/current_lane'))
check('A 중앙선이 목표에 있으면 offset≈0', abs(ln.last('lane/offset_right')) < 0.05,
      '(=%.3f)' % ln.last('lane/offset_right'))
check('A margin_right≈0.5', abs(ln.last('lane/margin_right') - 0.5) < 0.05,
      '(=%.3f)' % ln.last('lane/margin_right'))

# B. 중앙선이 목표(208)보다 오른쪽에 보이면 차가 너무 왼쪽에 있다는 뜻 -> 양수.
lnB = lane_mod.LaneDetectNode()
feed(lnB, road(white_left=20, white_right=480, yellow=270))
check('B 중앙선이 목표보다 우측 -> offset>0 (차가 좌측)',
      lnB.last('lane/offset_right') > 0.08,
      '(=%.3f)' % lnB.last('lane/offset_right'))
lnB2 = lane_mod.LaneDetectNode()
feed(lnB2, road(white_left=20, white_right=480, yellow=150))
check('B 중앙선이 목표보다 좌측 -> offset<0 (차가 우측)',
      lnB2.last('lane/offset_right') < -0.08,
      '(=%.3f)' % lnB2.last('lane/offset_right'))
# margin 은 흰선 위치에서 직접 나온다 (480 -> (480-320)/320 = 0.5)
check('B margin_right≈0.5', abs(lnB.last('lane/margin_right') - 0.5) < 0.05,
      '(=%.3f)' % lnB.last('lane/margin_right'))

# C. 왼쪽 차선 (노란선 480, 좌측 흰선 160 -> 차선중심 320)
feed(ln, road(white_left=160, white_right=None, yellow=480))
check('C 현재차선=LEFT', ln.last('lane/current_lane') == lane_mod.LANE_LEFT,
      '(=%s)' % ln.last('lane/current_lane'))
# 왼쪽 차선이면 중앙선이 내 오른쪽에 보여야 한다 -> 목표 x = 320 + 112 = 432
lnC = lane_mod.LaneDetectNode()
feed(lnC, road(white_left=160, white_right=None, yellow=432))
check('C 왼쪽 차선 목표는 부호 반전', abs(lnC.last('lane/offset_left')) < 0.05,
      '(=%.3f)' % lnC.last('lane/offset_left'))

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
# 홀드가 만료돼도 차선 판정은 래치가 유지한다. 다만 횡오차는 중앙선을
# 못 보므로 0 으로 두고 헤딩(흰선 기울기)만으로 간다 -- 없는 값을 흰선에서
# 지어내면 그 오차가 그대로 조향에 들어간다.
check('F 홀드 만료해도 차선 유지 (래치)',
      ln2.last('lane/current_lane') == lane_mod.LANE_RIGHT,
      '(=%s)' % ln2.last('lane/current_lane'))
check('F valid 유지', ln2.last('lane/valid') is True)
check('F 중앙선 못 보면 횡오차는 0',
      abs(ln2.last('lane/offset_right')) < 1e-6,
      '(=%.3f, 헤딩만으로 주행)' % ln2.last('lane/offset_right'))

# G. 실전에서 만난 '차선 인지 유실'. 흰선이 기준보다 어두우면 마스크에 안 걸려
#    valid=False 가 되고, 판단 노드는 흰선 위치를 모르므로 차를 세운다.
#    probe 는 그 흰선을 찾아내서 V 가 얼마나 모자란지 짚어줘야 한다.
ln3 = lane_mod.LaneDetectNode()
ln3.debug_probe = True
ln3._probe_stamp = 0.0
dim = road(white_left=20, white_right=480, yellow=160)
dim[dim[:, :, 0] == 255] = 150          # 흰선을 회색(V=150)으로 낮춤
feed(ln3, dim)
check('G 어두운 흰선 -> 인지 유실(valid=False)', ln3.last('lane/valid') is False,
      '(=%s)' % ln3.last('lane/valid'))

lp = [m for lvl, m in ln3._logger.lines if 'lane probe' in m]
white_sec = (lp[-1].split('흰선 후보')[-1].split('노란선 후보')[0]) if lp else ''
check('G probe 가 그 흰선을 후보로 찾아냄', 'V=150' in white_sec,
      '(%s)' % white_sec.strip().split('\n')[0].strip())
check('G probe 가 V 부족을 사유로 지목', 'V 150<180' in white_sec)


# H. 시뮬레이터 실측 상황. 흰선 바깥 갓길이 흙색이라 노란색 임계에 걸리는데,
#    그 면적이 중앙 점선보다 훨씬 크다. 화면 전체에서 최대 피크를 잡으면
#    중앙선 위치가 갓길로 잡혀 차선 구조가 통째로 어긋난다.
#    노란선은 두 흰선 '사이'에서만 찾아야 한다.
def road_with_shoulder():
    """HSV 로 직접 만든다 -- 갓길 색(H=30 S=132 V=161)을 실측값에 맞추려고."""
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[:, :] = (110, 90, 70)                 # 어두운 푸른 노면
    hsv[:, :100] = (30, 132, 161)             # 좌측 갓길 (흙색, 면적 큼)
    hsv[:, 540:] = (30, 132, 161)             # 우측 갓길
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr[:, 96:104] = (255, 255, 255)          # 좌 흰선
    bgr[:, 536:544] = (255, 255, 255)         # 우 흰선
    dash = cv2.cvtColor(np.uint8([[(20, 255, 255)]]), cv2.COLOR_HSV2BGR)[0][0]
    for y in range(0, H, 60):                 # 중앙 주황 점선 (면적 작음)
        bgr[y:min(H, y + 30), 276:284] = dash
    return bgr


img_h = road_with_shoulder()
ln4 = lane_mod.LaneDetectNode()
feed(ln4, img_h)

# 갓길이 실제로 노란 마스크를 오염시키는지 먼저 확인 (안 그러면 테스트가 무의미)
roi_h = img_h[int(H * ln4.roi_top_frac):, :]
_, ymask = ln4._masks(roi_h)
rh_h = roi_h.shape[0]
bh_h = max(2, int(rh_h * ln4.band_height_frac))
ny_h = max(0, min(rh_h - bh_h, int(rh_h * ln4.near_band_frac)))
prof_h = ln4._profile(ymask, ny_h, ny_h + bh_h)
naive = ln4._peak(prof_h, 0, W)
check('H 갓길이 노란 마스크를 오염시킴 (전제 확인)',
      naive is not None and (naive < 150 or naive > 490),
      '(전체탐색 피크 x=%s -> 갓길)' % (None if naive is None else round(naive)))

check('H 흰선 사이에서만 찾아 중앙선을 맞춤',
      ln4._yellow_x is not None and abs(ln4._yellow_x - 280) < 20,
      '(x=%s, 기대 280)' % (None if ln4._yellow_x is None
                            else round(ln4._yellow_x)))
check('H 차선 판정 정상 (중심 320 > 중앙선 280 -> RIGHT)',
      ln4.last('lane/current_lane') == lane_mod.LANE_RIGHT,
      '(=%s)' % ln4.last('lane/current_lane'))


# I. 자동 진단이 실제로 문제를 푸는가.
#    "권장: ..." 을 그대로 적용했을 때 valid 가 True 로 바뀌지 않으면
#    조언이 틀린 것이고, 그건 진단이 없느니만 못하다.
far_lines = np.full((H, W, 3), 60, np.uint8)
far_lines[:380, 18:26] = (255, 255, 255)      # 흰선이 밴드보다 위(먼 곳)에만
far_lines[:380, 476:484] = (255, 255, 255)

ln5 = lane_mod.LaneDetectNode()
ln5.debug_probe = True
ln5._probe_stamp = 0.0
feed(ln5, far_lines)
check('I 밴드 밖이면 valid=False', ln5.last('lane/valid') is False)

adv = [m for lvl, m in ln5._logger.lines if '권장' in m]
check('I 위치 문제로 진단', bool(adv) and '위치 문제' in adv[-1])

rec = re.search(r'lane_near_band_frac:=([\d.]+) lane_band_height_frac:=([\d.]+)',
                adv[-1]) if adv else None
check('I 권장값을 명령 형태로 제시', rec is not None,
      '(near=%s height=%s)' % (rec.group(1), rec.group(2)) if rec else '')

if rec:
    ln6 = lane_mod.LaneDetectNode()
    ln6.near_band_frac = float(rec.group(1))
    ln6.band_height_frac = float(rec.group(2))
    feed(ln6, far_lines)
    check('I 권장값 적용하면 실제로 검출됨', ln6.last('lane/valid') is True,
          '(=%s)' % ln6.last('lane/valid'))

# 색 자체가 안 맞는 경우는 위치 조언을 하면 안 된다
dark = np.full((H, W, 3), 60, np.uint8)
dark[:, 18:26] = (150, 150, 150)
dark[:, 476:484] = (150, 150, 150)
ln7 = lane_mod.LaneDetectNode()
ln7.debug_probe = True
ln7._probe_stamp = 0.0
feed(ln7, dark)
adv7 = [m for lvl, m in ln7._logger.lines if '진단' in m]
check('I 색 문제일 땐 튜너를 안내', bool(adv7) and 'hsv_tuner' in adv7[-1])
check('  위치 조언은 하지 않음', bool(adv7) and '권장: lane_near' not in adv7[-1])

# J. 해상도 독립성. 임계값을 640x480 기준 절대 픽셀로 박아두면 240p 카메라에서
#    같은 선인데도 전부 임계 미달로 떨어진다. 실차 카메라가 240p 라 실제로 겪었다.
def road_at(W2, H2):
    img = np.full((H2, W2, 3), 60, np.uint8)
    t = max(2, W2 // 80)
    for x0 in (int(W2 * 0.06), int(W2 * 0.75)):
        img[:, x0 - t:x0 + t] = (255, 255, 255)
    for y in range(0, H2, H2 // 8):
        img[y:y + H2 // 14, int(W2 * 0.32) - t:int(W2 * 0.32) + t] = (0, 255, 255)
    return img


offs = {}
for W2, H2 in ((640, 480), (320, 240), (480, 360)):
    nj = lane_mod.LaneDetectNode()
    nj.on_image(ros_stubs.Image(cv=road_at(W2, H2)))
    check('J %dx%d 검출됨' % (W2, H2), nj.last('lane/valid') is True,
          '(min_peak=%d win=%d inset=%d)' % (nj.min_peak_px, nj.peak_win_px,
                                             nj.yellow_inset_px))
    offs[(W2, H2)] = nj.last('lane/offset_right')

spread = max(offs.values()) - min(offs.values())
check('J 해상도가 달라도 같은 횡오차', spread < 0.02,
      '(편차 %.4f, 값 %s)' % (spread, ['%.3f' % v for v in offs.values()]))

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


# 4-4b 정지 사유를 셋으로 구분해서 말하는가.
# 런치 직후 '아직 안 옴'과 '못 찾는 중'과 '끊김'은 조치가 전혀 다르다.
def stall_msg(setup):
    jj = new_judge()
    jj.require_green = False
    jj.state = jud_mod.ST_RACING
    setup(jj)
    obstacles(jj)
    jj.clear(); jj._tick()
    warns = [m for lvl, m in jj._logger.lines if lvl == 'WARN']
    return warns[-1] if warns else ''


def never_received(jj):
    pass                     # lane_stamp 가 0.0 인 상태 그대로


def not_detected(jj):
    perceive(jj, valid=False)   # 수신은 되는데 valid=false


def went_stale(jj):
    perceive(jj, valid=True)
    jj.lane_stamp -= 10.0       # 받았었지만 오래됨


m1 = stall_msg(never_received)
check('런치 직후 -> "입력 대기 중"으로 안내', '대기 중' in m1, '(%s)' % m1[:40])
check('  토픽 이름 확인을 유도', 'image_topic' in m1)

m2 = stall_msg(not_detected)
check('valid=false -> "미검출"로 안내', '미검출' in m2, '(%s)' % m2[:40])
check('  debug_probe 사용을 유도', 'debug_probe' in m2)

m3 = stall_msg(went_stale)
check('오래됨 -> "끊김"으로 안내', '끊김' in m3, '(%s)' % m3[:40])
check('  경과 시간을 같이 표시', '초 전' in m3)

check('세 사유가 서로 다른 문구', len({m1, m2, m3}) == 3)

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



def settle(node, ticks=60, dt=1.0 / 30.0):
    """속도 변화율 제한이 있으므로 정상상태까지 여러 틱 돌린다.

    스텁에서는 연속 호출 간 실제 경과시간이 0에 가까워 dt 가 무의미해진다.
    실제 30Hz 처럼 보이도록 prev_t 를 뒤로 밀어 준다.
    """
    for _ in range(ticks):
        node.prev_t = time.time() - dt
        node.clear()
        node._tick()
    return node.last('/speed')


perceive(j6, lane=jud_mod.LANE_RIGHT, curv=0.0)
obstacles(j6)
v_straight = settle(j6)
perceive(j6, lane=jud_mod.LANE_RIGHT, curv=0.9)
obstacles(j6)
v_curve = settle(j6)
check('급커브에서 감속', v_curve < v_straight,
      '(직선 %.2f -> 커브 %.2f)' % (v_straight, v_curve))

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
