"""Publish a parallel corrected PhysiCar camera TF branch."""

from __future__ import annotations

import threading
import time
from typing import Dict

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.clock import JumpThreshold
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import JointState
from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster

from physicar_camera_tf_correction.core import CorrectedTfCore
from physicar_camera_tf_correction.core import EXPECTED_TILT_LIMIT_RAD
from physicar_camera_tf_correction.core import TransformSpec
from physicar_camera_tf_correction.core import corrected_static_transforms


JOINT_STATE_TOPIC = '/joint_states'
WARNING_THROTTLE_SECONDS = 5.0


class CameraCorrectedTfBroadcaster(Node):
    """Convert validated camera tilt JointState samples to corrected TF."""

    def __init__(self) -> None:
        super().__init__('camera_corrected_tf_broadcaster')

        self._state_lock = threading.RLock()
        self._core = CorrectedTfCore()
        self._last_warning_time: Dict[str, float] = {}
        self._first_sample_logged = False

        self._dynamic_broadcaster = TransformBroadcaster(self)
        self._static_broadcaster = StaticTransformBroadcaster(self)
        self._callback_group = MutuallyExclusiveCallbackGroup()

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=20,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._joint_state_subscription = self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self._joint_state_callback,
            qos,
            callback_group=self._callback_group,
        )

        jump_threshold = JumpThreshold(
            min_forward=None,
            min_backward=Duration(nanoseconds=-1),
            on_clock_change=True,
        )
        self._clock_jump_handler = self.get_clock().create_jump_callback(
            jump_threshold,
            post_callback=self._clock_jump_callback,
        )

        self._publish_static_transforms_once()
        self.get_logger().info(
            'corrected TF broadcaster initialized: '
            'camera_pan_link -> camera_tilt_link_corrected; '
            'topic=/joint_states; '
            f'use_sim_time={self.get_parameter("use_sim_time").value}'
        )

    @property
    def counters(self):
        """Expose counters for diagnostics and ROS-side tests."""
        return self._core.counters.copy()

    def _publish_static_transforms_once(self) -> None:
        transforms = [
            self._to_transform_stamped(spec)
            for spec in corrected_static_transforms()
        ]
        self._static_broadcaster.sendTransform(transforms)

    def _joint_state_callback(self, message: JointState) -> None:
        with self._state_lock:
            outcome = self._core.validate(
                names=message.name,
                positions=message.position,
                stamp_sec=message.header.stamp.sec,
                stamp_nanosec=message.header.stamp.nanosec,
            )
            if not outcome.accepted:
                self._warn_rejection(outcome.rejection_reason or 'unknown_rejection')
                return

            sample = outcome.sample
            if sample is None:
                return

            if abs(sample.q_joint_states) > EXPECTED_TILT_LIMIT_RAD:
                self._warn_throttled(
                    'tilt_outside_expected_limit',
                    'camera_tilt_joint position exceeds expected +/-0.5236 rad; '
                    'value is not clamped',
                )

            transform = self._to_transform_stamped(
                sample.transform,
                stamp_sec=sample.stamp[0],
                stamp_nanosec=sample.stamp[1],
            )
            try:
                self._dynamic_broadcaster.sendTransform(transform)
            except Exception as error:  # Keep the node alive and do not commit the stamp.
                self._core.record_send_error()
                self._warn_throttled(
                    'send_error',
                    f'failed to publish corrected dynamic TF: {error!r}',
                )
                return

            self._core.commit(sample)
            if not self._first_sample_logged:
                self.get_logger().info(
                    'first corrected TF: '
                    f'q={sample.q_joint_states:.9f}, '
                    f'q_corrected={sample.q_corrected:.9f}, '
                    f'stamp={sample.stamp[0]}.{sample.stamp[1]:09d}'
                )
                self._first_sample_logged = True

    def _clock_jump_callback(self, _time_jump) -> None:
        with self._state_lock:
            self._core.reset_clock_epoch()
            self._first_sample_logged = False
        self._warn_throttled(
            'clock_reset',
            'ROS clock changed or jumped backward; accepted JointState epoch reset',
        )

    def _warn_rejection(self, reason: str) -> None:
        messages = {
            'invalid_stamp': (
                'JointState has zero stamp; message rejected without now() substitution'
            ),
            'missing_joint': 'JointState does not contain camera_tilt_joint',
            'duplicate_joint': 'JointState contains camera_tilt_joint more than once',
            'malformed_joint_state': (
                'JointState position array does not cover camera_tilt_joint index'
            ),
            'invalid_position': 'camera_tilt_joint position is not finite',
            'out_of_order_or_duplicate': (
                'JointState stamp is duplicate or older in the current clock epoch'
            ),
        }
        self._warn_throttled(reason, messages.get(reason, reason))

    def _warn_throttled(self, key: str, message: str) -> None:
        now = time.monotonic()
        last = self._last_warning_time.get(key)
        if last is None or now - last >= WARNING_THROTTLE_SECONDS:
            self.get_logger().warning(message)
            self._last_warning_time[key] = now

    @staticmethod
    def _to_transform_stamped(
        spec: TransformSpec,
        stamp_sec: int = 0,
        stamp_nanosec: int = 0,
    ) -> TransformStamped:
        transform = TransformStamped()
        transform.header.stamp.sec = int(stamp_sec)
        transform.header.stamp.nanosec = int(stamp_nanosec)
        transform.header.frame_id = spec.parent_frame
        transform.child_frame_id = spec.child_frame

        translation = spec.translation
        transform.transform.translation.x = translation[0]
        transform.transform.translation.y = translation[1]
        transform.transform.translation.z = translation[2]

        quaternion = spec.quaternion_xyzw
        transform.transform.rotation.x = quaternion[0]
        transform.transform.rotation.y = quaternion[1]
        transform.transform.rotation.z = quaternion[2]
        transform.transform.rotation.w = quaternion[3]
        return transform


def main(args=None) -> None:
    """Run the broadcaster in the required single-threaded executor."""
    rclpy.init(args=args)
    node = CameraCorrectedTfBroadcaster()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.remove_node(node)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
