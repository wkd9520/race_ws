"""Continuously command the fixed camera tilt required by perception V3."""

import math

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64


DEFAULT_TILT_DEGREES = -30.0
DEFAULT_PUBLISH_RATE_HZ = 10.0


def degrees_to_radians(degrees):
    """Convert a finite tilt command from degrees to radians."""
    value = float(degrees)
    if not math.isfinite(value):
        raise ValueError('camera tilt must be finite')
    return math.radians(value)


class CameraTiltPublisher(Node):
    def __init__(self):
        super().__init__('perception_v3_camera_tilt_publisher')
        self.declare_parameter('tilt_degrees', DEFAULT_TILT_DEGREES)
        self.declare_parameter('publish_rate_hz', DEFAULT_PUBLISH_RATE_HZ)

        tilt_degrees = float(self.get_parameter('tilt_degrees').value)
        publish_rate_hz = float(self.get_parameter('publish_rate_hz').value)
        if not math.isfinite(publish_rate_hz) or publish_rate_hz <= 0.0:
            raise ValueError('camera tilt publish rate must be finite and positive')

        self._tilt_radians = degrees_to_radians(tilt_degrees)
        self._publisher = self.create_publisher(Float64, '/camera/tilt', 10)
        self._timer = self.create_timer(1.0 / publish_rate_hz, self._publish)
        self._publish()
        self.get_logger().info(
            'camera tilt hold enabled: %.3f deg (%.6f rad), rate=%.1f Hz'
            % (tilt_degrees, self._tilt_radians, publish_rate_hz)
        )

    def _publish(self):
        self._publisher.publish(Float64(data=self._tilt_radians))


def main(args=None):
    rclpy.init(args=args)
    node = CameraTiltPublisher()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
