"""rclpy/cv_bridge/std_msgs/sensor_msgs 최소 스텁.

Windows에는 ROS2가 없으므로, 노드의 순수 알고리즘만 떼어 검증하기 위한 가짜 런타임.
발행된 메시지는 node._sent[topic] 리스트에 쌓인다.
"""
import sys
import types


# ---------------------------------------------------------------- std_msgs
def _mk_msg(name):
    def __init__(self, data=None):
        self.data = data
    return type(name, (), {'__init__': __init__})


Bool = _mk_msg('Bool')
Float64 = _mk_msg('Float64')
Int32 = _mk_msg('Int32')
String = _mk_msg('String')


class Header:
    def __init__(self):
        self.stamp = None
        self.frame_id = ''


class Image:
    def __init__(self, cv=None):
        self._cv = cv
        self.header = Header()


class LaserScan:
    def __init__(self):
        self.angle_min = 0.0
        self.angle_increment = 0.0
        self.ranges = []
        self.header = Header()


# --------------------------------------------------------- nav/geometry_msgs
class Point:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z


class Quaternion:
    def __init__(self, x=0.0, y=0.0, z=0.0, w=1.0):
        self.x, self.y, self.z, self.w = x, y, z, w


class Pose:
    def __init__(self):
        self.position = Point()
        self.orientation = Quaternion()


class PoseStamped:
    def __init__(self):
        self.header = Header()
        self.pose = Pose()


class Path:
    def __init__(self):
        self.header = Header()
        self.poses = []


# ---------------------------------------------------------------- cv_bridge
class CvBridge:
    def imgmsg_to_cv2(self, msg, encoding='bgr8'):
        if getattr(msg, '_cv', None) is None:
            raise ValueError('빈 이미지')
        return msg._cv

    def cv2_to_imgmsg(self, arr, encoding='bgr8'):
        return Image(cv=arr)


# ---------------------------------------------------------------- rclpy
class _Param:
    def __init__(self, value):
        self.value = value


class _Logger:
    def __init__(self, name):
        self.name = name
        self.lines = []

    def _log(self, lvl, msg):
        self.lines.append((lvl, msg))

    def info(self, msg, **kw):
        self._log('INFO', msg)

    def warn(self, msg, **kw):
        self._log('WARN', msg)

    def error(self, msg, **kw):
        self._log('ERROR', msg)


class _Pub:
    def __init__(self, node, topic):
        self.node, self.topic = node, topic

    def publish(self, msg):
        self.node._sent.setdefault(self.topic, []).append(msg.data)


class Node:
    def __init__(self, name):
        self._name = name
        self._params = {}
        self._logger = _Logger(name)
        self._sent = {}
        self._timers = []
        self._subs = {}

    def declare_parameter(self, name, value):
        self._params.setdefault(name, _Param(value))

    def get_parameter(self, name):
        return self._params[name]

    def set_param(self, name, value):
        """테스트 편의: 선언된 파라미터 값을 바꾼다."""
        self._params[name] = _Param(value)

    def create_publisher(self, msg_type, topic, qos):
        return _Pub(self, topic)

    def create_subscription(self, msg_type, topic, cb, qos):
        self._subs[topic] = cb
        return object()

    def create_timer(self, period, cb):
        self._timers.append((period, cb))
        return object()

    def get_logger(self):
        return self._logger

    def destroy_node(self):
        pass

    # 테스트 헬퍼
    def last(self, topic, default=None):
        v = self._sent.get(topic)
        return v[-1] if v else default

    def clear(self):
        self._sent.clear()


def init(args=None):
    pass


def shutdown():
    pass


def spin(node):
    raise RuntimeError('스텁에서는 spin 사용 안 함')


def install():
    """sys.modules에 가짜 패키지들을 등록한다."""
    rclpy = types.ModuleType('rclpy')
    rclpy.init, rclpy.shutdown, rclpy.spin = init, shutdown, spin
    rclpy_node = types.ModuleType('rclpy.node')
    rclpy_node.Node = Node
    rclpy.node = rclpy_node

    # 노드들이 센서 QoS를 import 한다. 스텁에서는 값만 있으면 되고 의미는 없다.
    rclpy_qos = types.ModuleType('rclpy.qos')
    rclpy_qos.qos_profile_sensor_data = object()
    rclpy.qos = rclpy_qos

    std = types.ModuleType('std_msgs')
    std_msg = types.ModuleType('std_msgs.msg')
    std_msg.Bool, std_msg.Float64 = Bool, Float64
    std_msg.Int32, std_msg.String = Int32, String
    std.msg = std_msg

    sen = types.ModuleType('sensor_msgs')
    sen_msg = types.ModuleType('sensor_msgs.msg')
    sen_msg.Image, sen_msg.LaserScan = Image, LaserScan
    sen.msg = sen_msg

    nav = types.ModuleType('nav_msgs')
    nav_msg = types.ModuleType('nav_msgs.msg')
    nav_msg.Path = Path
    nav.msg = nav_msg

    geo = types.ModuleType('geometry_msgs')
    geo_msg = types.ModuleType('geometry_msgs.msg')
    geo_msg.PoseStamped, geo_msg.Pose = PoseStamped, Pose
    geo_msg.Point, geo_msg.Quaternion = Point, Quaternion
    geo.msg = geo_msg

    cvb = types.ModuleType('cv_bridge')
    cvb.CvBridge = CvBridge

    for k, v in {
        'rclpy': rclpy, 'rclpy.node': rclpy_node, 'rclpy.qos': rclpy_qos,
        'std_msgs': std, 'std_msgs.msg': std_msg,
        'sensor_msgs': sen, 'sensor_msgs.msg': sen_msg,
        'nav_msgs': nav, 'nav_msgs.msg': nav_msg,
        'geometry_msgs': geo, 'geometry_msgs.msg': geo_msg,
        'cv_bridge': cvb,
    }.items():
        sys.modules[k] = v
