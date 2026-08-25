"""출발 신호등: 초록 원을 봐야 출발하는가.

코스 규정상 정지선 앞 신호등에 초록 원이 들어와야 출발할 수 있다.

여기서 막으려는 실패는 둘이고, **둘 다 대회에서 끝장이다**:

  1. 빨간불에 출발한다            -> 실격
  2. 초록인데 영영 안 움직인다    -> 완주 못 함

2번이 특히 조용히 온다. 신호등 노드를 안 띄웠는데 follow 노드만 대기를
켜면, 아무 오류 없이 차가 그냥 서 있는다. 그래서 런치가 두 개를
`traffic_light` 인자 하나로 묶는지 여기서 확인한다.

그리고 초록 고깔 문제가 있다. 고깔 HSV(40~85)와 신호등 초록 구간(40~90)이
거의 겹쳐서, 모양을 안 보면 출발선에 놓인 고깔이 초록불로 읽힌다.
신호등은 **원**이고 고깔은 삼각형이라 외접 사각형으로 갈라낸다.
"""
import ast
import io
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.pardir, 'src', 'physicar_race')
NODE = os.path.join(SRC, 'physicar_race', 'traffic_light_node.py')
FOLLOW = os.path.join(SRC, 'physicar_race', 'perception_v3_follow_node.py')
LAUNCH = os.path.join(SRC, 'launch', 'perception_v3_race_launch.py')
SETUP = os.path.join(SRC, 'setup.py')

FAILS = []


def check(label, cond, detail=''):
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % ('PASS' if cond else 'FAIL', label, detail))


node_src = io.open(NODE, encoding='utf-8').read()
follow_src = io.open(FOLLOW, encoding='utf-8').read()
launch_src = io.open(LAUNCH, encoding='utf-8').read()


# ---------------------------------------------------- 모양 검사를 실제로 돌린다

class Shape:
    """_round_enough 를 노드에서 떼어내 그대로 돌린다."""

    def __init__(self, require=True, fill=0.60, lo=0.60, hi=1.70):
        self.require_circle = require
        self.min_fill_ratio = fill
        self.min_aspect = lo
        self.max_aspect = hi

    # 노드의 구현을 그대로 가져다 붙인다 (동작이 갈리면 테스트가 거짓말을 한다)
    _round_enough = None


# 노드 파일에서 _round_enough 함수 본문을 실제로 뽑아 실행한다.
tree = ast.parse(node_src)
fn = next((n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == '_round_enough'), None)
if fn is not None:
    namespace = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), '<node>', 'exec'),
         namespace)
    Shape._round_enough = namespace['_round_enough']


def blob_mask(kind, size=20):
    """원 / 고깔(삼각형) 을 그린 마스크."""
    pad = size // 2
    img = np.zeros((size * 2 + pad, size * 2 + pad), np.uint8)
    cx, cy = img.shape[1] // 2, img.shape[0] // 2
    if kind == 'circle':
        cv2.circle(img, (cx, cy), size // 2, 255, -1)
    else:                                   # 고깔: 위가 뾰족한 삼각형
        pts = np.array([[cx, cy - size // 2],
                        [cx - size // 3, cy + size // 2],
                        [cx + size // 3, cy + size // 2]], np.int32)
        cv2.fillPoly(img, [pts], 255)
    return img


def measure(mask):
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (int(stats[i, cv2.CC_STAT_WIDTH]), int(stats[i, cv2.CC_STAT_HEIGHT]),
            int(stats[i, cv2.CC_STAT_AREA]))


print('\n[1] 원과 고깔을 갈라내는가 ★')
check('_round_enough 를 노드에서 찾았다', Shape._round_enough is not None)
if Shape._round_enough is not None:
    shape = Shape()
    for kind, want in (('circle', True), ('cone', False)):
        w, h, area = measure(blob_mask(kind))
        ok, why = Shape._round_enough(shape, w, h, area)
        label = '원은 통과한다' if want else '고깔은 걸러낸다 ★'
        check(label, ok == want,
              '(%dx%d 면적%d, 채움 %.2f%s)'
              % (w, h, area, area / float(w * h), '' if ok else ' -> ' + why))

    off = Shape(require=False)
    w, h, area = measure(blob_mask('cone'))
    check('require_circle=False 면 모양을 안 본다 (되돌릴 수 있다)',
          Shape._round_enough(off, w, h, area)[0])


print('\n[2] 빨간불에 출발하지 않는가 ★')
check('초록일 때만 래치가 걸린다',
      "msg.data != 'GREEN'" in follow_src)
check('래치는 한 번 걸리면 안 풀린다 (주행 중 신호를 다시 안 본다)',
      'self._green_seen or' in follow_src
      and follow_src.count('self._green_seen = False') == 0)
check('대기 중에는 속도 0 을 낸다',
      'self._publish(0.0, 0.0)' in follow_src)
# 정지 중에 바퀴가 꺾여 있으면 출발 첫 순간에 엉뚱한 데로 튄다.
check('대기 중에는 조향도 0 이다', '_publish(0.0, 0.0)' in follow_src)

# 게이트가 경로 판단보다 **먼저** 와야 한다. 뒤에 있으면 경로가 좋을 때
# 게이트를 지나쳐 출발해버린다.
follow_tree = ast.parse(follow_src)
tick = next(n for n in ast.walk(follow_tree)
            if isinstance(n, ast.FunctionDef) and n.name == 'tick')
gate_line = next((n.lineno for n in ast.walk(tick)
                  if isinstance(n, ast.If)
                  and '_green_seen' in ast.unparse(n.test)), None)
usable_line = next((n.lineno for n in ast.walk(tick)
                    if isinstance(n, ast.Assign)
                    and 'usable' in ast.unparse(n.targets[0])), None)
check('게이트가 경로 판단보다 앞에 있다 ★',
      gate_line is not None and usable_line is not None
      and gate_line < usable_line,
      '(게이트 %s행, 경로판단 %s행)' % (gate_line, usable_line))


print('\n[3] 초록인데 영영 안 움직이는 일은 없는가 ★')
# 신호등 노드가 안 떠 있는데 대기만 켜면 차가 조용히 안 움직인다.
# 오류도 안 나서 제일 찾기 어려운 실패다. 런치가 둘을 하나로 묶어야 한다.
check('노드와 대기가 같은 인자로 묶여 있다',
      "condition=IfCondition(LaunchConfiguration('traffic_light'))" in launch_src
      and "'wait_for_green': _b('traffic_light')" in launch_src)
check('wait_for_green 기본값은 False (노드 단독 실행 시 안 멈춤)',
      "declare_parameter('wait_for_green', False)" in follow_src)
check('traffic_light 기본값은 true (대회 기본이 신호 대기)',
      "DeclareLaunchArgument('traffic_light', default_value='true')" in launch_src)


print('\n[4] 배선')
check('런치가 신호등 노드를 실제로 띄운다', 'cones, traffic, follow' in launch_src)
check('카메라 토픽을 넘겨준다',
      launch_src.count("remappings=[('/camera/image_raw'") >= 2)
check('setup.py 에 실행 파일이 등록돼 있다',
      'traffic_light_node = physicar_race.traffic_light_node:main'
      in io.open(SETUP, encoding='utf-8').read())
check('follow 노드가 traffic/light_state 를 구독한다',
      "'traffic/light_state'" in follow_src)
check('String 을 임포트했다', 'Float64, String' in follow_src)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
