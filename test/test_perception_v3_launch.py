"""perception_v3_race_launch.py 검증 (ROS 없이 launch 모듈을 스텁으로 대체).

launch 파일은 실제로 띄우기 전까지 틀린 걸 알기 어렵다. 특히 이 파일은
MinSeok 님 launch 를 include 하고 노드 넷에 외부 프로세스까지 붙이므로,
조용히 빠지면 "왜 안 뜨지" 로만 드러난다. 구조를 여기서 굳혀둔다.
"""
import importlib.util
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCH = os.path.join(HERE, os.pardir, 'src', 'physicar_race', 'launch',
                      'perception_v3_race_launch.py')

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

    def __repr__(self):
        return '<cfg %s>' % self.name


class DeclareLaunchArgument:
    def __init__(self, name, default_value=None, **kw):
        self.name = name
        self.default_value = default_value


class ExecuteProcess:
    def __init__(self, cmd=None, **kw):
        self.cmd = cmd or []
        self.kw = kw


class TimerAction:
    def __init__(self, period=None, actions=None, **kw):
        self.period = period
        self.actions = actions or []


class IncludeLaunchDescription:
    def __init__(self, source=None, launch_arguments=None, **kw):
        self.source = source
        self.launch_arguments = dict(launch_arguments or {})


class PythonLaunchDescriptionSource:
    def __init__(self, path):
        self.path = path


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
    actions.ExecuteProcess = ExecuteProcess
    actions.IncludeLaunchDescription = IncludeLaunchDescription
    actions.TimerAction = TimerAction

    conditions = types.ModuleType('launch.conditions')
    conditions.IfCondition = IfCondition

    subs = types.ModuleType('launch.substitutions')
    subs.LaunchConfiguration = LaunchConfiguration

    sources = types.ModuleType('launch.launch_description_sources')
    sources.PythonLaunchDescriptionSource = PythonLaunchDescriptionSource

    lr_actions = types.ModuleType('launch_ros.actions')
    lr_actions.Node = Node

    lr_params = types.ModuleType('launch_ros.parameter_descriptions')
    lr_params.ParameterValue = ParameterValue

    launch_ros = types.ModuleType('launch_ros')

    aip = types.ModuleType('ament_index_python.packages')
    aip.get_package_share_directory = lambda pkg: os.path.join('/share', pkg)
    ament = types.ModuleType('ament_index_python')
    ament.packages = aip

    for k, v in {
        'launch': launch, 'launch.actions': actions,
        'launch.conditions': conditions, 'launch.substitutions': subs,
        'launch.launch_description_sources': sources,
        'launch_ros': launch_ros, 'launch_ros.actions': lr_actions,
        'launch_ros.parameter_descriptions': lr_params,
        'ament_index_python': ament,
        'ament_index_python.packages': aip,
    }.items():
        sys.modules[k] = v


install()

spec = importlib.util.spec_from_file_location('perception_v3_race_launch',
                                              LAUNCH)
pl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pl)

ld = pl.generate_launch_description()
ents = ld.entities
args = {e.name: e.default_value for e in ents
        if isinstance(e, DeclareLaunchArgument)}
nodes = [e for e in ents if isinstance(e, Node)]
includes = [e for e in ents if isinstance(e, IncludeLaunchDescription)]
procs = [e for e in ents if isinstance(e, ExecuteProcess)]
timers = [e for e in ents if isinstance(e, TimerAction)]


print('\n[1] MinSeok 님 launch 를 수정 없이 include 하는가')
check('include 가 정확히 하나', len(includes) == 1, '(%d개)' % len(includes))
if includes:
    inc = includes[0]
    check('  그의 perception_v3.launch.py 를 부른다',
          inc.source.path.endswith(os.path.join(
              'physicar_track_perception_v3', 'launch',
              'perception_v3.launch.py')),
          '(%s)' % inc.source.path)
    for key in ('use_sim_time', 'camera_topic', 'joint_states_topic',
                'scan_topic'):
        check('  %s 를 그대로 넘긴다' % key, key in inc.launch_arguments)


print('\n[2] 우리 노드 셋이 다 있는가')
names = sorted(n.kw.get('name') for n in nodes)
for want in ('cone_bev_node', 'perception_v3_follow_node',
             'race_overlay_node'):
    check('%s' % want, want in names)


print('\n[3] 격자 값이 세 곳에서 같은가 ★')
# 다르면 고깔/주행선 좌표가 통째로 틀어진다. 조용히 어긋나는 종류의 실수라
# 여기서 잡아둔다.
grids = {}
for n in nodes:
    params = n.kw.get('parameters') or [{}]
    d = params[0] if isinstance(params[0], dict) else {}
    g = {k: v for k, v in d.items() if k.startswith('bev_')}
    if g:
        grids[n.kw.get('name')] = g

check('격자를 쓰는 노드가 둘', len(grids) == 2, '(%s)' % sorted(grids))
if len(grids) == 2:
    vals = list(grids.values())
    check('  두 노드의 격자가 완전히 같다 ★', vals[0] == vals[1],
          '(%s)' % vals[0])
    check('  perception_v3.yaml 과 같다 (x 0.10~2.00, y ±0.75, 0.01)',
          vals[0].get('bev_x_min') == 0.10
          and vals[0].get('bev_x_max') == 2.00
          and vals[0].get('bev_y_min') == -0.75
          and vals[0].get('bev_y_max') == 0.75
          and vals[0].get('bev_resolution') == 0.01)


print('\n[4] 카메라 틸트를 이 launch 가 건드리지 않는가')
# 틸트는 V2 요구사항상 -0.5236 rad 여야 하지만, 시뮬레이터나 다른 노드가
# 이미 잡고 있으면 둘이 동시에 보내 값이 번갈아 들어간다. 그래서 여기서는
# 아예 발행하지 않고 필요할 때 손으로 띄우기로 했다.
check('외부 프로세스를 안 띄운다', len(procs) == 0, '(%d개)' % len(procs))
for gone in ('publish_tilt', 'camera_tilt', 'tilt_rate'):
    check('  %s 인자가 남아 있지 않다' % gone, gone not in args)


print('\n[5] rqt 자동 실행')
check('TimerAction 으로 늦게 띄운다', len(timers) == 1, '(%d개)' % len(timers))
if timers:
    inner = timers[0].actions
    check('  안에 노드가 하나', len(inner) == 1)
    if inner:
        check('  rqt_image_view 다',
              inner[0].kw.get('package') == 'rqt_image_view',
              '(%s)' % inner[0].kw.get('package'))
        check('  토픽을 인자로 준다',
              bool(inner[0].kw.get('arguments')))
        check('  open_rqt 로 끌 수 있다',
              inner[0].kw.get('condition') is not None)
check('기본 토픽이 우리 오버레이',
      args.get('rqt_topic') == '/race/debug/path_overlay',
      '(%s)' % args.get('rqt_topic'))
check('  헤드리스용으로 끌 수 있게 인자가 있다', 'open_rqt' in args)


print('\n[6] 실차에서 자주 바꾸는 값이 인자로 나와 있는가')
for key in ('v_max', 'a_lat_max', 'steer_sign', 'ld_k',
            'avoid_enabled', 'green_h_min', 'green_h_max',
            'max_offset_m', 'camera_topic'):
    check(key, key in args, '(기본 %s)' % args.get(key))


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
