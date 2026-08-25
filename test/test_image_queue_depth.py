"""카메라 프레임이 큐에 밀리지 않는지 본다.

실차에서 코너를 못 돌고 라인도 못 따라갔다. 속도가 아니라 **지연**이
의심된다.

    qos_profile_sensor_data 는 깊이 5다.
    카메라 30 Hz, 인지 15 Hz -> 큐에 다섯 장이 밀린다
    5 / 15 Hz = 333 ms,  1.2 m/s 에서 40 cm

차는 40 cm 전에 있던 자리를 기준으로 조향한다. 코너에서 늘 늦게 꺾고
직선에서 넘어섰다 되돌아온다 -- 관찰된 증상 그대로다.

깊이 1이면 밀린 프레임을 붙잡는 대신 버린다. 처리 속도는 그대로고
지연만 한 프레임으로 고정된다.

여기서 지키는 것:
  1. 이미지 구독 깊이가 1이다 (다시 5로 돌아가면 지연이 돌아온다)
  2. BEST_EFFORT 다 (RELIABLE 이면 재전송을 기다리느라 오히려 밀린다)
  3. **이번엔 이것 하나만 바꿨다** -- joint_states/scan 은 그대로.
     한 번에 하나만 바꿔야 안 되면 무엇 때문인지 알 수 있다.
  4. 경로 토픽의 QoS 는 안 건드렸다 (follow 노드와 짝이 맞아야 한다)
"""
import ast
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
NODE = os.path.join(
    HERE, os.pardir, 'src', 'MinSeok', 'physicar_track_perception_v3',
    'physicar_track_perception_v3', 'bev_frontend_node.py')
FOLLOW = os.path.join(
    HERE, os.pardir, 'src', 'physicar_race', 'physicar_race',
    'perception_v3_follow_node.py')

FAILS = []


def check(label, cond, detail=''):
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % ('PASS' if cond else 'FAIL', label, detail))


source = io.open(NODE, encoding='utf-8').read()
tree = ast.parse(source)


def literal(node):
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return None


# --- 구독 지점을 전부 모은다: {토픽: QoS 인자의 소스코드} ---
subs = {}
for node in ast.walk(tree):
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'create_subscription'
            and len(node.args) >= 4):
        topic = literal(node.args[1])
        if topic is None:
            topic = ast.dump(node.args[1])
        subs[topic] = ast.unparse(node.args[3])

# --- NEWEST_IMAGE_QOS 의 실제 값을 읽는다 ---
qos_kwargs = {}
for node in ast.walk(tree):
    if (isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == 'NEWEST_IMAGE_QOS'
                    for t in node.targets)
            and isinstance(node.value, ast.Call)):
        for kw in node.value.keywords:
            qos_kwargs[kw.arg] = ast.unparse(kw.value)


print('\n[1] 카메라는 가장 최신 프레임만 본다 ★')
image_qos = subs.get('/camera/image_raw')
check('이미지 구독이 전용 QoS 를 쓴다', image_qos == 'NEWEST_IMAGE_QOS',
      '(%s)' % image_qos)
check('깊이가 1이다', qos_kwargs.get('depth') == '1',
      '(depth=%s)' % qos_kwargs.get('depth'))
check('BEST_EFFORT 다', 'BEST_EFFORT' in qos_kwargs.get('reliability', ''),
      '(%s)' % qos_kwargs.get('reliability'))
check('KEEP_LAST 다', 'KEEP_LAST' in qos_kwargs.get('history', ''),
      '(%s)' % qos_kwargs.get('history'))

# 지연 계산을 주석이 아니라 여기서 다시 한다.
for rate, depth in ((15.0, 5), (15.0, 1)):
    lag = depth / rate
    print('       깊이 %d, 인지 %.0f Hz -> 지연 %.0f ms, 1.2 m/s 에서 %.0f cm'
          % (depth, rate, lag * 1e3, lag * 1.2 * 100))


print('\n[2] 이번엔 이것 하나만 바꿨는가 ★')
# 한 번에 하나만 바꿔야, 안 되면 무엇 때문인지 알 수 있다.
# 앞서 격자/BFS/디버그를 한꺼번에 바꿨다가 세 번 롤백했다.
for topic in ('/joint_states',):
    check('%s 는 그대로다' % topic,
          subs.get(topic) == 'qos_profile_sensor_data',
          '(%s)' % subs.get(topic))
scan_sub = [q for t, q in subs.items() if 'scan' in str(t).lower()]
check('scan 구독도 그대로다',
      scan_sub == ['qos_profile_sensor_data'] or not scan_sub,
      '(%s)' % scan_sub)


print('\n[3] 경로 토픽 QoS 는 안 건드렸다 (follow 노드와 짝)')
# RELIABLE 구독자 + BEST_EFFORT 발행자는 아예 연결이 안 된다.
# 여기를 잘못 만지면 조용히 0 프레임이 온다.
pubs = {}
for node in ast.walk(tree):
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'create_publisher'
            and len(node.args) >= 3):
        topic = literal(node.args[1])
        if topic is not None:
            pubs[topic] = ast.unparse(node.args[2])

check('/perception_v3/path 는 깊이 10 정수 그대로',
      pubs.get('/perception_v3/path') == '10',
      '(%s)' % pubs.get('/perception_v3/path'))
check('/perception_v3/debug/path_valid 도 그대로',
      pubs.get('/perception_v3/debug/path_valid') == '10',
      '(%s)' % pubs.get('/perception_v3/debug/path_valid'))

follow = ast.parse(io.open(FOLLOW, encoding='utf-8').read())
follow_subs = {}
for node in ast.walk(follow):
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'create_subscription'
            and len(node.args) >= 4):
        topic = literal(node.args[1])
        if topic is not None:
            follow_subs[topic] = ast.unparse(node.args[3])

check('follow 노드도 /perception_v3/path 를 깊이 10 으로 구독',
      follow_subs.get('/perception_v3/path') == '10',
      '(%s)' % follow_subs.get('/perception_v3/path'))


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
