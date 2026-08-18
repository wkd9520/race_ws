"""race_launch.py 의 조건 로직 검증 (ROS 없이 launch 모듈을 스텁으로 대체).

launch 파일은 빌드해서 실제로 띄워보기 전까지 틀린 걸 알기 어렵다.
특히 '신호등을 끄면 require_green 도 같이 꺼져야 한다' 같은 규칙은
조용히 어긋나면 차가 영원히 안 움직이는 형태로만 드러난다.
"""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH = os.path.join(HERE, os.pardir, 'src', 'physicar_race', 'launch', 'race_launch.py')

FAILS = []


def check(label, cond, detail=''):
    tag = 'PASS' if cond else 'FAIL'
    if not cond:
        FAILS.append(label)
    print('  [%s] %s %s' % (tag, label, detail))


# ------------------------------------------------------------------ 스텁
class LaunchConfiguration:
    def __init__(self, name):
        self.name = name

    def perform(self, context):
        return context[self.name]


class DeclareLaunchArgument:
    def __init__(self, name, default_value=None, **kw):
        self.name = name
        self.default_value = default_value


class LogInfo:
    def __init__(self, msg=''):
        self.msg = msg


class OpaqueFunction:
    def __init__(self, function=None):
        self.function = function


class IfCondition:
    def __init__(self, predicate):
        self.predicate = predicate


class Node:
    def __init__(self, **kw):
        self.kw = kw


class ParameterValue:
    def __init__(self, value, value_type=None):
        self.value = value
        self.value_type = value_type


class LaunchDescription:
    def __init__(self, entities):
        self.entities = entities


def install():
    launch = types.ModuleType('launch')
    launch.LaunchDescription = LaunchDescription

    actions = types.ModuleType('launch.actions')
    actions.DeclareLaunchArgument = DeclareLaunchArgument
    actions.LogInfo = LogInfo
    actions.OpaqueFunction = OpaqueFunction

    conditions = types.ModuleType('launch.conditions')
    conditions.IfCondition = IfCondition

    subs = types.ModuleType('launch.substitutions')
    subs.LaunchConfiguration = LaunchConfiguration

    lr_actions = types.ModuleType('launch_ros.actions')
    lr_actions.Node = Node

    lr_params = types.ModuleType('launch_ros.parameter_descriptions')
    lr_params.ParameterValue = ParameterValue

    launch_ros = types.ModuleType('launch_ros')

    for k, v in {
        'launch': launch, 'launch.actions': actions,
        'launch.conditions': conditions, 'launch.substitutions': subs,
        'launch_ros': launch_ros, 'launch_ros.actions': lr_actions,
        'launch_ros.parameter_descriptions': lr_params,
    }.items():
        sys.modules[k] = v


install()

spec = importlib.util.spec_from_file_location('race_launch', LAUNCH)
rl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rl)

ld = rl.generate_launch_description()
args = {e.name: e.default_value for e in ld.entities if isinstance(e, DeclareLaunchArgument)}
nodes = [e for e in ld.entities if isinstance(e, Node)]
opaques = [e for e in ld.entities if isinstance(e, OpaqueFunction)]


def defaults(**over):
    ctx = {k: (v if v is not None else '') for k, v in args.items()}
    ctx.update(over)
    return ctx


print('\n[launch] 인자 선언')
for need in ('image_topic', 'scan_topic', 'require_green', 'use_traffic_light',
             'lane_width_m', 'v_max', 'debug_probe',
             'lane_near_band_frac', 'tl_roi_bottom_frac'):
    check('인자 %s 선언됨' % need, need in args, '(기본=%s)' % args.get(need))

print('\n[launch] 노드 구성')
execs = [n.kw.get('executable') for n in nodes]
for need in ('lane_detect_node', 'traffic_light_node', 'lane_obstacle_node'):
    check('%s 포함' % need, need in execs)
check('판단 노드는 OpaqueFunction 으로 생성', len(opaques) == 1)

tl_node = next(n for n in nodes if n.kw.get('executable') == 'traffic_light_node')
check('신호등 노드에 IfCondition 걸림',
      isinstance(tl_node.kw.get('condition'), IfCondition))

print('\n[launch] remapping')
lane_node = next(n for n in nodes if n.kw.get('executable') == 'lane_detect_node')
obs_node = next(n for n in nodes if n.kw.get('executable') == 'lane_obstacle_node')
check('차선 노드 image_raw remap',
      any(r[0] == 'image_raw' for r in lane_node.kw.get('remappings', [])))
check('장애물 노드 scan remap',
      any(r[0] == 'scan' for r in obs_node.kw.get('remappings', [])))

print('\n[launch] 출발 게이트 규칙')


def judgment_for(ctx):
    out = opaques[0].function(ctx)
    node = next(o for o in out if isinstance(o, Node))
    logs = [o for o in out if isinstance(o, LogInfo)]
    return node.kw['parameters'][0]['require_green'], logs


rg, logs = judgment_for(defaults(require_green='true', use_traffic_light='true'))
check('신호등 켬 + require_green 켬 -> 게이트 유지', rg is True, '(=%s)' % rg)
check('  경고 없음', not logs)

rg, logs = judgment_for(defaults(require_green='true', use_traffic_light='false'))
check('신호등 끔 -> require_green 강제 해제', rg is False, '(=%s)' % rg)
check('  그 사실을 로그로 알림', len(logs) == 1 and 'require_green' in logs[0].msg)

rg, _ = judgment_for(defaults(require_green='false', use_traffic_light='true'))
check('require_green 끔 -> 그대로 꺼짐', rg is False, '(=%s)' % rg)

rg, _ = judgment_for(defaults(require_green='false', use_traffic_light='false'))
check('둘 다 끔 -> 꺼짐', rg is False, '(=%s)' % rg)

for truthy in ('True', 'TRUE', '1', 'yes', 'on'):
    rg, _ = judgment_for(defaults(require_green=truthy, use_traffic_light='true'))
    check('참 표기 %-5r 인식' % truthy, rg is True)

print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
