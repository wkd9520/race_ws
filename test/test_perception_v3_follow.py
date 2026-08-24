"""perception_v3_follow_node 검증 - MinSeok 님의 /perception_v3/path 를 따라간다.

`physicar_track_perception_v3` 자체(ORANGE 중앙선 추적, IPM, LiDAR 오버레이)는
이 저장소가 만든 게 아니라 그대로 옮겨온 것이라 여기서 검증하지 않는다
(원본이 이미 비-ROS pytest 22개를 통과한 상태로 옮겨졌다). 여기서 보는 건
그 산출물(`/perception_v3/path`)을 받아 `/speed` + `/steering` 으로 바꾸는
이 저장소 몫의 코드뿐이다.

순수추종/횡가속 공식은 los_drive_node.py 와 동일하다 -- 그쪽에서 이미
손계산으로 검증했으므로 여기서는 좌표계가 이미 미터라는 점(경로->조향
연결부)과 유실/서행/정지 상태기계만 집중해서 본다.
"""
import importlib.util
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ros_stubs  # noqa: E402

ros_stubs.install()
from nav_msgs.msg import Path  # noqa: E402
from geometry_msgs.msg import PoseStamped  # noqa: E402
from std_msgs.msg import Bool  # noqa: E402

SRC = os.path.join(HERE, os.pardir, 'src', 'physicar_race', 'physicar_race',
                   'perception_v3_follow_node.py')
spec = importlib.util.spec_from_file_location('perception_v3_follow_node', SRC)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


def node():
    return mod.PerceptionV3FollowNode()


def make_path(points):
    """[(x,y), ...] -> nav_msgs/Path 스텁."""
    msg = Path()
    for x, y in points:
        pose = PoseStamped()
        pose.pose.position.x = float(x)
        pose.pose.position.y = float(y)
        msg.poses.append(pose)
    return msg


def tick(n_):
    """ros_stubs 는 타이머를 자동 실행하지 않는다 -- 직접 호출한다."""
    n_._timers[0][1]()


print('\n[1] 순수추종/횡가속 - los_drive_node 와 같은 손계산')
n = node()
# los_drive_node 의 [1]에서 검증한 것과 동일한 케이스
d = math.degrees(n.pure_pursuit(1.0, 0.5))
check('손계산과 일치 (x=1.0 y=+0.5 -> 8.20도)', abs(d - 8.20) < 0.05, '(%.2f도)' % d)
check('왼쪽 점 -> 양수, 오른쪽 점 -> 음수',
      n.pure_pursuit(1.0, 0.5) > 0 > n.pure_pursuit(1.0, -0.5))

# v_max/k_vis 로 잘리기 전의 곡률 항만 본다. 이걸 안 풀어두면 v_max 를
# 낮추는 순간 이 검사가 "물리식이 틀렸다"가 아니라 "상한에 걸렸다"로
# 실패한다 -- 실제로 v_max 를 1.2 로 내렸을 때 그렇게 깨졌다.
n.k_vis = 99.0
n.v_max = 99.0
v20 = n.speed_limit(math.radians(20.0), 10.0)
check('횡가속 한계 손계산과 일치 (1.22 m/s)', abs(v20 - 1.218) < 0.01, '(%.3f)' % v20)

# 위 값이 v_max 를 넘는지가 실주행에서 의미가 있다: 넘으면 코너 감속이
# 아예 작동하지 않는다(항상 v_max 가 먼저 걸린다).
n_def = node()
full_lock = math.sqrt(n_def.a_lat_max * (mod.WHEELBASE / math.tan(mod.MAX_STEER)))
print('       최대조향에서 횡가속 한계 %.3f m/s vs v_max %.2f -> 코너 감속 %s'
      % (full_lock, n_def.v_max,
         '작동' if full_lock < n_def.v_max else '무효(v_max 가 항상 먼저 걸림)'))

print('\n[2] 경로 기하 - base_footprint 미터 좌표는 변환이 필요 없다')
n2 = node()
straight = [(0.3, 0.0), (0.6, 0.0), (0.9, 0.0), (1.2, 0.0)]
pt = n2.lookahead_point(straight, 0.9)
check('전방주시점을 그대로 쓴다 (변환 없음)', pt == (0.9, 0.0), '(%s)' % (pt,))
check('경로 길이 = 마지막 점까지 누적 거리',
      abs(n2.path_length(straight) - 1.2) < 1e-9, '(%.3f)' % n2.path_length(straight))

curve = [(0.3, 0.0), (0.6, 0.1), (0.9, 0.3), (1.2, 0.6)]
pt2 = n2.lookahead_point(curve, 0.9)
check('코너에서는 옆으로 치우친 점을 고른다', pt2[1] > 0.2, '(%s)' % (pt2,))

print('\n[3] 유실 처리 - MinSeok 경로가 valid=False 여도 빈 poses 로 계속 온다')
n3 = node()
# process()는 항상 gp 를 발행한다 -- invalid 여도 메시지 자체는 온다.
# 그래서 "메시지 수신"과 "쓸만함"을 따로 봐야 한다: valid=False 를 명시적으로
# 받아야 정지/서행으로 넘어간다.
n3.on_path(make_path([]))
n3.on_valid(Bool(data=False))
tick(n3)
check('빈 경로 + invalid -> 정지 (아직 한 번도 성공 못함)',
      n3._sent['/speed'][-1] == 0.0)

n4 = node()
n4.on_path(make_path([(0.5, 0.0), (1.0, 0.0)]))
n4.on_valid(Bool(data=True))
for _ in range(5):
    tick(n4)
check('정상 수신 -> 속도가 붙는다', n4._sent['/speed'][-1] > n4.v_min,
      '(%.2f)' % n4._sent['/speed'][-1])
held_steer = n4._sent['/steering'][-1]

# 갑자기 valid=False 로 (인지가 놓침) -- 하지만 아직 grace_s 안
n4.on_valid(Bool(data=False))
tick(n4)
check('유실 직후 마지막 조향 유지 ★',
      abs(n4._sent['/steering'][-1] - held_steer) < 1e-9)
check('  서행한다', abs(n4._sent['/speed'][-1] - n4.grace_speed) < 1e-9,
      '(%.2f)' % n4._sent['/speed'][-1])

# grace_s 를 넘기면 정지 -- race_judgment_node 테스트와 같은 방식으로
# _last_ok_time 을 과거로 밀어 넣는다
n4._last_ok_time = time.time() - (n4.grace_s + 0.1)
tick(n4)
check('오래 유실되면 정지', n4._sent['/speed'][-1] == 0.0)

print('\n[4] 입력 신선도 - 메시지가 끊기면 (콜백 자체가 안 옴)')
n5 = node()
n5.on_path(make_path([(0.5, 0.0), (1.0, 0.0)]))
n5.on_valid(Bool(data=True))
n5._last_path_time = time.time() - 999    # 콜백이 옛날에 멈췄다고 가정
tick(n5)
check('오래된 경로는 안 쓴다', n5._sent['/speed'][-1] == 0.0,
      '(%.2f)' % n5._sent['/speed'][-1])

print('\n[5] QoS - 발행 측(V3Node)과 같은 depth 로 구독한다')
n6 = node()
check("'/perception_v3/path' 를 구독한다",
      '/perception_v3/path' in n6._subs)
check("'/perception_v3/debug/path_valid' 를 구독한다",
      '/perception_v3/debug/path_valid' in n6._subs)

print('\n[6] 최소 점 개수 - 점 하나뿐인 경로는 안 쓴다')
n7 = node()
n7.on_path(make_path([(0.5, 0.0)]))
n7.on_valid(Bool(data=True))
tick(n7)
check('min_path_points 미만이면 정지', n7._sent['/speed'][-1] == 0.0,
      '(점 1개, min=%d)' % n7.min_path_points)

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
