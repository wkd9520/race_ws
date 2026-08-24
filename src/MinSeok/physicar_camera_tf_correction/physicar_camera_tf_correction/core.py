"""ROS-independent validation and transform math for corrected camera TF."""

from collections import Counter
from dataclasses import dataclass
import math
from typing import Optional, Sequence, Tuple


TILT_JOINT_NAME = 'camera_tilt_joint'

DYNAMIC_PARENT_FRAME = 'camera_pan_link'
DYNAMIC_CHILD_FRAME = 'camera_tilt_link_corrected'
STATIC_CAMERA_CHILD_FRAME = 'camera_link_corrected'
STATIC_OPTICAL_CHILD_FRAME = 'camera_optical_frame_corrected'

DYNAMIC_TRANSLATION = (0.025, 0.0, 0.013)
CAMERA_TRANSLATION = (0.030, 0.0, 0.014)
OPTICAL_TRANSLATION = (0.0, 0.0, 0.0)

IDENTITY_QUATERNION = (0.0, 0.0, 0.0, 1.0)
OPTICAL_QUATERNION = (-0.5, 0.5, -0.5, 0.5)
EXPECTED_TILT_LIMIT_RAD = 0.5236

Stamp = Tuple[int, int]
Vector3 = Tuple[float, float, float]
Quaternion = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TransformSpec:
    """A transport-neutral transform description."""

    parent_frame: str
    child_frame: str
    translation: Vector3
    quaternion_xyzw: Quaternion


@dataclass(frozen=True)
class AcceptedSample:
    """A validated JointState tilt sample awaiting TF transmission."""

    stamp: Stamp
    q_joint_states: float
    q_corrected: float
    transform: TransformSpec


@dataclass(frozen=True)
class ValidationOutcome:
    """Result of validating one JointState message."""

    sample: Optional[AcceptedSample]
    rejection_reason: Optional[str]

    @property
    def accepted(self) -> bool:
        """Return whether the input is eligible for TF transmission."""
        return self.sample is not None


def corrected_dynamic_transform(q_joint_states: float) -> TransformSpec:
    """Build camera_pan_link -> corrected tilt from q_corrected = -q."""
    q_corrected = -q_joint_states
    half_angle = 0.5 * q_corrected
    quaternion = (0.0, math.sin(half_angle), 0.0, math.cos(half_angle))
    return TransformSpec(
        parent_frame=DYNAMIC_PARENT_FRAME,
        child_frame=DYNAMIC_CHILD_FRAME,
        translation=DYNAMIC_TRANSLATION,
        quaternion_xyzw=quaternion,
    )


def corrected_static_transforms() -> Tuple[TransformSpec, TransformSpec]:
    """Return the two immutable transforms below the corrected tilt link."""
    camera = TransformSpec(
        parent_frame=DYNAMIC_CHILD_FRAME,
        child_frame=STATIC_CAMERA_CHILD_FRAME,
        translation=CAMERA_TRANSLATION,
        quaternion_xyzw=IDENTITY_QUATERNION,
    )
    optical = TransformSpec(
        parent_frame=STATIC_CAMERA_CHILD_FRAME,
        child_frame=STATIC_OPTICAL_CHILD_FRAME,
        translation=OPTICAL_TRANSLATION,
        quaternion_xyzw=OPTICAL_QUATERNION,
    )
    return camera, optical


class CorrectedTfCore:
    """Validate JointState data and commit only successfully sent samples."""

    def __init__(self) -> None:
        self.last_accepted_stamp: Optional[Stamp] = None
        self.last_q: Optional[float] = None
        self.clock_epoch = 0
        self.counters: Counter[str] = Counter()

    def validate(
        self,
        names: Sequence[str],
        positions: Sequence[float],
        stamp_sec: int,
        stamp_nanosec: int,
    ) -> ValidationOutcome:
        """Validate one JointState without mutating accepted-sample state."""
        stamp = (int(stamp_sec), int(stamp_nanosec))
        if stamp == (0, 0):
            return self._reject('invalid_stamp')

        indices = [i for i, name in enumerate(names) if name == TILT_JOINT_NAME]
        if not indices:
            return self._reject('missing_joint')
        if len(indices) != 1:
            return self._reject('duplicate_joint')

        index = indices[0]
        if index >= len(positions):
            return self._reject('malformed_joint_state')

        try:
            q_joint_states = float(positions[index])
        except (TypeError, ValueError, OverflowError):
            return self._reject('invalid_position')
        if not math.isfinite(q_joint_states):
            return self._reject('invalid_position')

        if self.last_accepted_stamp is not None and stamp <= self.last_accepted_stamp:
            return self._reject('out_of_order_or_duplicate')

        sample = AcceptedSample(
            stamp=stamp,
            q_joint_states=q_joint_states,
            q_corrected=-q_joint_states,
            transform=corrected_dynamic_transform(q_joint_states),
        )
        return ValidationOutcome(sample=sample, rejection_reason=None)

    def commit(self, sample: AcceptedSample) -> None:
        """Record a sample only after TransformBroadcaster accepts it."""
        self.last_accepted_stamp = sample.stamp
        self.last_q = sample.q_joint_states
        self.counters['accepted'] += 1

    def record_send_error(self) -> None:
        """Count a failed send without advancing the accepted stamp."""
        self.counters['send_error'] += 1

    def reset_clock_epoch(self) -> None:
        """Allow a new stamp sequence after a detected backward clock jump."""
        self.clock_epoch += 1
        self.last_accepted_stamp = None
        self.last_q = None
        self.counters['clock_reset'] += 1

    def _reject(self, reason: str) -> ValidationOutcome:
        self.counters[reason] += 1
        return ValidationOutcome(sample=None, rejection_reason=reason)
