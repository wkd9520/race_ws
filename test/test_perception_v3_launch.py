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


class PythonExpression:
    """조건식을 평가한다. 실제 launch 도 문자열을 이어붙여 eval 한다.

    start_sequence_node 와 camera_tilt_publisher 가 동시에 뜨면 둘이
    /camera/tilt 를 서로 다른 값으로 밀어서 카메라가 떤다. 그 조건이
    실제로 맞는지 보려면 식을 평가할 수 있어야 한다.
    """

    def __init__(self, parts):
        self.parts = parts

    def evaluate(self, values):
        text = ''
        for part in self.parts:
            if isinstance(part, LaunchConfiguration):
                text += str(values.get(part.name, ''))
            else:
                text += str(part)
        return bool(eval(text, {'__builtins__': {}}, {}))

    def __repr__(self):
        return '<expr %s>' % (self.parts,)


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
    subs.PythonExpression = PythonExpression

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


print('\n[1] MinSeok 님 노드 둘을 그대로 띄우는가')
# include 대신 그의 노드를 직접 띄운다. include 로는 bev.* 를 덮어쓸 수
# 없어서, 값 하나 바꿀 때마다 그의 yaml 을 손으로 고쳐야 했기 때문이다.
# 코드는 안 건드리지만 **구성은 그의 launch 와 같아야** 한다.
by_pkg = {n.kw.get('package'): n for n in nodes}

check('camera_corrected_tf_broadcaster 를 띄운다',
      'physicar_camera_tf_correction' in by_pkg)
check('bev_frontend_node 를 띄운다', 'physicar_track_perception_v3' in by_pkg)

v3 = by_pkg.get('physicar_track_perception_v3')
if v3:
    check('  실행파일이 bev_frontend_node',
          v3.kw.get('executable') == 'bev_frontend_node',
          '(%s)' % v3.kw.get('executable'))
    # yaml 최상위 키가 이 이름이라, 다르면 yaml 파라미터가 통째로 안 먹는다
    check('  노드 이름이 yaml 키와 같다 ★',
          v3.kw.get('name') == 'physicar_track_perception_v3',
          '(%s)' % v3.kw.get('name'))
    params = v3.kw.get('parameters') or []
    # 실차용 yaml 이어야 한다. 시뮬용과 camera.K 가 달라서, 잘못 읽으면
    # 다른 카메라로 계산하게 되고 BEV 가 통째로 틀어진다 -- 실제로 겪었다.
    check('  실차용 yaml 을 먼저 읽는다 ★',
          bool(params) and isinstance(params[0], str)
          and params[0].endswith('perception_v3_real.yaml'),
          '(%s)' % (os.path.basename(params[0]) if params else None))
    check('  그 위에 우리 값을 덮는다 (뒤가 이긴다)',
          len(params) >= 2 and isinstance(params[1], dict))
    remaps = dict(v3.kw.get('remappings') or [])
    check('  카메라/조인트 토픽을 리맵한다',
          '/camera/image_raw' in remaps and '/joint_states' in remaps)
check('include 는 이제 안 쓴다', len(includes) == 0, '(%d개)' % len(includes))


print('\n[2] 우리 노드 셋이 다 있는가')
names = sorted(n.kw.get('name') for n in nodes)
for want in ('cone_bev_node', 'perception_v3_follow_node',
             'race_overlay_node'):
    check('%s' % want, want in names)


print('\n[3] 격자를 세 노드가 같은 인자에서 받는가 ★')
# 예전엔 값을 세 곳에 손으로 적어뒀다. 하나만 고치면 조용히 어긋나고,
# 고깔/주행선 좌표가 통째로 틀어지는데 화면상으로는 그럴듯해 보인다.
# 이제는 launch 인자 하나에서 셋이 같이 받아야 한다.
SUFFIX = ('x_min', 'x_max', 'y_min', 'y_max', 'resolution')


def grid_of(node):
    """노드가 받는 격자를 {끝말: 인자이름} 으로 뽑는다.

    perception_v3 는 'bev.x_min', 우리 노드는 'bev_x_min' 이라 이름이
    다르다. 끝말로 맞춰야 셋을 비교할 수 있다.
    """
    for entry in (node.kw.get('parameters') or []):
        if not isinstance(entry, dict):
            continue
        out = {}
        for key, val in entry.items():
            if not (key.startswith('bev_') or key.startswith('bev.')):
                continue
            for suf in SUFFIX:
                if key.endswith(suf):
                    # ParameterValue(LaunchConfiguration(...)) 에서 인자 이름
                    out[suf] = getattr(getattr(val, 'value', val), 'name', val)
        if out:
            return out
    return None


grids = {n.kw.get('name'): grid_of(n) for n in nodes}
grids = {k: v for k, v in grids.items() if v}
check('격자를 쓰는 노드가 셋', len(grids) == 3, '(%s)' % sorted(grids))

if len(grids) == 3:
    vals = list(grids.values())
    check('  셋이 같은 launch 인자를 받는다 ★', vals[0] == vals[1] == vals[2],
          '(%s)' % vals[0])
    check('  리터럴이 아니라 인자다 (한 곳만 고치면 된다) ★',
          all(isinstance(v, str) for v in vals[0].values()),
          '(%s)' % list(vals[0].values())[0])

for key in ('bev_x_min', 'bev_x_max', 'bev_y_min', 'bev_y_max',
            'bev_resolution', 'pitch_offset_deg',
            'camera_height_correction_z', 'camera_k', 'camera_d'):
    check('  %s 인자가 있다' % key, key in args)

# camera_info 가 껍데기(전부 0)라 yaml/인자가 유일한 진실이다.
# 시뮬 값(fx=201.4)을 실차에 쓰면 BEV 가 통째로 틀어진다 -- 겪었다.
k = args.get('camera_k', '')
check('  camera_k 가 실차 값이다 ★ (시뮬 201.4 아님)',
      '260.875' in k and '231.31516130651107' in k
      and '169.16236121207476' in k and '201.389' not in k,
      '(%s)' % k)
# 드라이버가 이미 왜곡보정을 해서 내보내므로 D 는 전부 0 이어야 한다.
# 시뮬 값을 쓰면 왜곡을 두 번 먹인다.
d = args.get('camera_d', '')
check('  camera_d 가 0 이다 (드라이버가 이미 보정)',
      '0.045' not in d and d.count('0.0') >= 5, '(%s)' % d)

# 라즈베리파이 5 가 카메라를 못 따라가서 줄였다. 인지 비용은 BEV 픽셀
# 수에 거의 비례한다(연결요소마다 픽셀 BFS). 28,500 -> 2,500 px.
# 해상도는 0.01 로 못 박는다. 0.02 로 올리면 흰선(2~3 cm)이 BEV 에서
# 1 픽셀이 되고, 마스킹이 투영 *뒤에* 돌기 때문에(bev_frontend_node.py:1452)
# 보간에서 아스팔트와 섞여 임계값을 못 넘는다 -- 선이 통째로 사라진다.
# 실차에서 조향이 완전히 죽었다. 줄이는 건 범위로만 한다.
check('해상도는 0.01 이고 범위만 줄었다 ★',
      args.get('bev_resolution') == '0.01'
      and args.get('bev_x_max') == '1.20'
      and args.get('bev_y_max') == '0.70',
      '(%s m/px, x~%s, y±%s)' % (args.get('bev_resolution'),
                                 args.get('bev_x_max'), args.get('bev_y_max')))
px = ((float(args['bev_x_max']) - float(args['bev_x_min']))
      / float(args['bev_resolution'])
      * (float(args['bev_y_max']) - float(args['bev_y_min']))
      / float(args['bev_resolution']))
check('  BEV 픽셀 수가 실차 yaml(28,500)의 절반 이하', px <= 14500,
      '(%.0f px)' % px)
# 고깔을 피해 max_offset_m 만큼 붙었을 때도 반대편 흰선이 격자 안에
# 남아야 한다. 흰선을 넘으면 실격이라, 회피하는 그 순간 못 보면 안 된다.
far_wall = float(args['track_half_m']) + float(args['max_offset_m'])
check('회피로 붙었을 때도 반대편 흰선이 격자 안에 있다 ★',
      float(args['bev_y_max']) >= far_wall,
      '(먼 벽 %.2f m <= y_max %.2f m)' % (far_wall, float(args['bev_y_max'])))
# 높이 보정은 실차 yaml 그대로 0. 피치는 우리 차에서 실측한 3.0 이다
# (0/6/12.5 비교). 실차 yaml 의 0 과 다른 유일한 값이라 여기 고정해둔다.
check('피치가 실측값 3.0 으로 고정돼 있다 ★',
      args.get('pitch_offset_deg') == '3.0',
      '(%s)' % args.get('pitch_offset_deg'))
check('높이 보정은 실차 yaml 대로 0',
      args.get('camera_height_correction_z') == '0.0',
      '(%s)' % args.get('camera_height_correction_z'))


print('\n[3b] LiDAR 회피를 끄는가 ★ (우리는 카메라로 고깔을 본다)')
SWITCHES = ('avoidance.shadow_enabled', 'avoidance_circle.enabled',
            'obstacle_track.enabled', 'active_lifecycle.enabled',
            'avoidance_recovery.enabled')
v3_over = {}
if v3:
    for entry in (v3.kw.get('parameters') or []):
        if isinstance(entry, dict):
            v3_over.update(entry)
for key in SWITCHES:
    check('  %s 를 인자로 묶었다' % key, key in v3_over)
same = {getattr(getattr(v3_over.get(k2), 'value', None), 'name', None)
        for k2 in SWITCHES if k2 in v3_over}
check('  다섯이 같은 인자 하나로 움직인다', same == {'lidar_avoidance'},
      '(%s)' % same)
check('  기본이 꺼짐', args.get('lidar_avoidance') == 'false')
# 중앙선 품질을 올리는 것들이라 회피와 무관하다. 같이 꺼지면 안 된다.
for key in ('center_hybrid.enabled', 'center_history.enabled'):
    check('  %s 는 안 건드린다' % key, key not in v3_over)

check('  틸트 유지 노드가 있다 (hold_tilt)', 'hold_tilt' in args,
      '(기본 %s)' % args.get('hold_tilt'))
tilt_nodes = [n for n in nodes
              if n.kw.get('executable') == 'camera_tilt_publisher']
check('  camera_tilt_publisher 를 띄운다', len(tilt_nodes) == 1)
if tilt_nodes:
    check('    조건부다 (로스백 재생 땐 불필요)',
          tilt_nodes[0].kw.get('condition') is not None)

# start_sequence_node 도 /camera/tilt 를 민다. 둘이 동시에 뜨면 서로 다른
# 값을 밀어서 카메라가 떤다. 조건식을 실제로 평가해서 확인한다.
seq_node = next((n for n in nodes
                 if n.kw.get('executable') == 'start_sequence_node'), None)
check('  start_sequence_node 를 띄운다', seq_node is not None)


def runs(node, **overrides):
    """이 설정에서 그 노드가 실제로 뜨는가."""
    values = {k: v for k, v in args.items()}
    values.update(overrides)
    cond = node.kw.get('condition')
    if cond is None:
        return True
    predicate = cond.predicate
    if isinstance(predicate, PythonExpression):
        return predicate.evaluate(values)
    return str(values.get(predicate.name, '')).lower() == 'true'


if seq_node is not None and tilt_nodes:
    tilt = tilt_nodes[0]
    both = runs(tilt, traffic_light='true', hold_tilt='true') and         runs(seq_node, traffic_light='true', hold_tilt='true')
    check('  둘이 절대 같이 뜨지 않는다 ★', not both,
          '(/camera/tilt 를 서로 다른 값으로 밀면 카메라가 떤다)')
    check('  신호등을 켜면 start_sequence 가 틸트를 쥔다',
          runs(seq_node, traffic_light='true')
          and not runs(tilt, traffic_light='true', hold_tilt='true'))
    check('  신호등을 끄면 hold_tilt 가 예전처럼 동작한다',
          runs(tilt, traffic_light='false', hold_tilt='true')
          and not runs(tilt, traffic_light='false', hold_tilt='false'))


print('\n[4] 틸트는 MinSeok 님 노드에 맡긴다')
# 한때 ros2 topic pub 을 launch 에서 직접 돌렸다가 뺐다(시뮬레이터와
# 동시에 쏘면 값이 번갈아 들어간다). 새 버전에 camera_tilt_publisher 가
# 들어왔으므로 그걸 조건부로 쓴다 -- 직접 pub 하지 않는다.
check('ros2 topic pub 을 직접 돌리지 않는다', len(procs) == 0,
      '(%d개)' % len(procs))
for gone in ('publish_tilt', 'tilt_rate'):
    check('  %s 인자가 남아 있지 않다' % gone, gone not in args)
check('  대신 tilt_degrees 로 준다', args.get('tilt_degrees') == '-30.0',
      '(%s)' % args.get('tilt_degrees'))


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
        check('  debug_view 로 끌 수 있다',
              inner[0].kw.get('condition') is not None)
check('기본 토픽이 우리 오버레이',
      args.get('rqt_topic') == '/race/debug/path_overlay',
      '(%s)' % args.get('rqt_topic'))


print('\n[5b] 시각화가 기본으로 꺼져 있는가 ★')
# 실차는 헤드리스이고, 오버레이 이미지를 만드는 CPU 가 아깝다.
# 스위치 하나로 세 가지가 같이 움직여야 한다.
check('debug_view 인자가 있고 기본이 꺼짐',
      args.get('debug_view') == 'false', '(%s)' % args.get('debug_view'))
check('  open_rqt 는 없어졌다 (debug_view 로 통합)', 'open_rqt' not in args)

ovl = next((n for n in nodes if n.kw.get('name') == 'race_overlay_node'), None)
check('  race_overlay_node 가 조건부다', ovl is not None
      and ovl.kw.get('condition') is not None)

cone = next((n for n in nodes if n.kw.get('name') == 'cone_bev_node'), None)
cone_dbg = None
if cone:
    for entry in (cone.kw.get('parameters') or []):
        if isinstance(entry, dict) and 'publish_debug' in entry:
            cone_dbg = entry['publish_debug']
check('  cone_bev_node 의 디버그 이미지도 같이 꺼진다 ★',
      getattr(getattr(cone_dbg, 'value', None), 'name', None) == 'debug_view')

# 라즈베리파이 5 에서 인지가 카메라를 못 따라가는 문제를 겪었다.
# 회피를 안 쓸 때 이 노드가 이미지 두 개를 계속 처리하는 건 순수 낭비다.
check('  회피를 끄면 cone_bev_node 자체를 안 띄운다 ★',
      cone is not None and cone.kw.get('condition') is not None)


print('\n[6] 실차에서 자주 바꾸는 값이 인자로 나와 있는가')
for key in ('v_max', 'a_lat_max', 'steer_sign', 'ld_k',
            'avoid_enabled', 'green_h_min', 'green_h_max',
            'max_offset_m', 'camera_topic'):
    check(key, key in args, '(기본 %s)' % args.get(key))



print('\n[9] 리매핑 이름이 노드가 실제로 쓰는 이름과 맞는가 ★')
# 이 실수는 두 번 물렸다.
#   centroid_follow_node -- image_raw 를 remap 안 해서 차가 안 움직였다
#   traffic_light_node   -- 노드는 'image_raw', 런치는 '/camera/image_raw'
#                           를 remap 해서 프레임을 0장 받았다
#
# 짝이 안 맞으면 **오류가 안 난다.** 노드는 있지도 않은 토픽을 조용히
# 기다리고, 우리는 엉뚱한 곳(HSV, 임계값)을 파게 된다. 그래서 여기서
# 소스를 열어 그 이름을 실제로 쓰는지 확인한다.
PKG_DIR = os.path.join(HERE, os.pardir, 'src', 'physicar_race',
                       'physicar_race')
checked = 0
for node in nodes:
    if node.kw.get('package') != 'physicar_race':
        continue
    source_path = os.path.join(PKG_DIR, '%s.py' % node.kw.get('executable'))
    if not os.path.exists(source_path):
        check('%s 의 소스 파일이 있다' % node.kw.get('executable'), False)
        continue
    node_source = open(source_path, encoding='utf-8').read()
    for source_name, _ in (node.kw.get('remappings') or []):
        checked += 1
        check("%s 가 '%s' 를 실제로 쓴다"
              % (node.kw.get('name'), source_name),
              "'%s'" % source_name in node_source,
              '' if "'%s'" % source_name in node_source
              else '(짝이 안 맞으면 조용히 0장 받는다)')
check('리매핑을 실제로 검사했다 (테스트가 헛돌지 않는다)', checked >= 1,
      '(%d개)' % checked)


print('\n' + '=' * 58)
if FAILS:
    print('실패 %d건: %s' % (len(FAILS), ', '.join(FAILS)))
    sys.exit(1)
print('전부 통과')
