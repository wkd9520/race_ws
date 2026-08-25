"""디버그 출력만 구독자 수로 막았는지, 제어 토픽은 안 막았는지 본다.

라즈베리파이 5 에서 인지가 카메라를 못 따라가서, bev_frontend_node 가
프레임마다 무조건 그리던 그림들을 `debug_wanted()` 로 감쌌다.

여기서 지키려는 건 속도가 아니라 **안전**이다. 실수로 제어가 먹는
토픽까지 감싸면 차가 그냥 선다. 그것도 rqt 를 열었을 때만 멀쩡하고
경기장에서만 죽는, 제일 나쁜 형태로 선다. 그래서 이 관계를 사람 눈이
아니라 구문 트리로 확인한다.

막으면 안 되는 것 (perception_v3_follow_node.py:132-134 가 구독한다):
    geometry_pub -> /perception_v3/path
    valid_pub    -> /perception_v3/debug/path_valid
    source_pub   -> /perception_v3/debug/path_source

`path_valid` 는 이름에 debug 가 들어 있어서 특히 위험하다.
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


def is_gate(test):
    """이 if 조건이 self.debug_wanted(...) 를 부르는가."""
    for node in ast.walk(test):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == 'debug_wanted'):
            return True
    return False


def gated_publishers(node, gated, found):
    """self.X_pub.publish(...) 를 찾아 {이름: 감싸였는가} 로 모은다."""
    if isinstance(node, ast.If) and is_gate(node.test):
        for child in node.body:
            gated_publishers(child, True, found)
        for child in node.orelse:
            gated_publishers(child, gated, found)
        return
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'publish'
            and isinstance(node.func.value, ast.Attribute)):
        found.setdefault(node.func.value.attr, set()).add(gated)
    for child in ast.iter_child_nodes(node):
        gated_publishers(child, gated, found)


found = {}
gated_publishers(tree, False, found)

# 헬퍼 호출을 통해 간접적으로 감싸인 것도 감싸인 것으로 본다.
# publish_lidar_overlay / publish_center_hybrid_debug 가 그 경우다.
INDIRECT = ('publish_lidar_overlay', 'publish_center_hybrid_debug')
indirect_gated = set()
for node in ast.walk(tree):
    if isinstance(node, ast.If) and is_gate(node.test):
        for child in node.body:
            for inner in ast.walk(child):
                if (isinstance(inner, ast.Call)
                        and isinstance(inner.func, ast.Attribute)
                        and inner.func.attr in INDIRECT):
                    indirect_gated.add(inner.func.attr)


print('\n[1] 제어가 먹는 토픽은 절대 막히면 안 된다 ★')
# follow 노드가 실제로 무엇을 구독하는지 코드에서 읽어온다.
follow_src = io.open(FOLLOW, encoding='utf-8').read()
for pub, topic in (('geometry_pub', '/perception_v3/path'),
                   ('valid_pub', '/perception_v3/debug/path_valid')):
    check('follow 노드가 %s 를 구독한다' % topic, topic in follow_src)

for pub in ('geometry_pub', 'valid_pub', 'source_pub'):
    states = found.get(pub)
    check('%s 가 감싸이지 않았다' % pub,
          states == {False},
          '(%s)' % ('감싸임 - 차가 선다!' if states != {False} else '무조건 발행'))


print('\n[2] 그림은 전부 막혔는가')
for pub in ('bev_pub', 'white_pub', 'orange_pub', 'role_pub', 'path_pub'):
    states = found.get(pub)
    check('%s 가 감싸였다' % pub, states == {True},
          '' if states == {True} else '(%s)' % (states,))

for name in INDIRECT:
    check('%s 호출이 감싸였다' % name, name in indirect_gated)


print('\n[3] cv2_to_imgmsg 가 감싸이지 않은 채 남은 곳은 없는가')
# 이미지 변환은 전부 그림이다. 안 감싸인 게 있으면 빠뜨린 것이다.
loose = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        continue
naked = {}


def scan_imgmsg(node, gated):
    if isinstance(node, ast.If) and is_gate(node.test):
        for child in node.body:
            scan_imgmsg(child, True)
        for child in node.orelse:
            scan_imgmsg(child, gated)
        return
    if (isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'cv2_to_imgmsg'):
        naked.setdefault(getattr(node, 'lineno', 0), gated)
    for child in ast.iter_child_nodes(node):
        scan_imgmsg(child, gated)


# 헬퍼 안(publish_lidar_overlay / publish_center_hybrid_debug)의 변환은
# 호출부가 감싸였으므로 예외로 둔다.
helper_lines = set()
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name in INDIRECT:
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and getattr(inner, 'lineno', None):
                helper_lines.add(inner.lineno)

scan_imgmsg(tree, False)
open_lines = sorted(line for line, g in naked.items()
                    if not g and line not in helper_lines)
check('직접 발행하는 변환은 전부 감싸였다', not open_lines,
      '' if not open_lines else '(%d행이 남았다)' % open_lines[0])
check('변환 지점을 실제로 찾았다 (테스트가 헛돌지 않는다)', len(naked) >= 5,
      '(%d곳)' % len(naked))


print('\n[4] 헬퍼가 존재하고 구독자 수를 본다')
check('debug_wanted 가 정의돼 있다',
      any(isinstance(n, ast.FunctionDef) and n.name == 'debug_wanted'
          for n in ast.walk(tree)))
check('get_subscription_count 로 판단한다',
      'get_subscription_count' in source)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
