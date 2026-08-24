"""Unit tests for the fixed corrected-camera TF contract."""

import math
import unittest

from physicar_camera_tf_correction.core import CAMERA_TRANSLATION
from physicar_camera_tf_correction.core import CorrectedTfCore
from physicar_camera_tf_correction.core import DYNAMIC_CHILD_FRAME
from physicar_camera_tf_correction.core import DYNAMIC_PARENT_FRAME
from physicar_camera_tf_correction.core import DYNAMIC_TRANSLATION
from physicar_camera_tf_correction.core import OPTICAL_QUATERNION
from physicar_camera_tf_correction.core import STATIC_CAMERA_CHILD_FRAME
from physicar_camera_tf_correction.core import STATIC_OPTICAL_CHILD_FRAME
from physicar_camera_tf_correction.core import corrected_static_transforms


def validate(core, q, sec=10, nanosec=20, names=None, positions=None):
    """Build the minimum JointState-like input accepted by the pure core."""
    if names is None:
        names = ['camera_pan_joint', 'camera_tilt_joint']
    if positions is None:
        positions = [0.1, q]
    return core.validate(names, positions, sec, nanosec)


class TestCorrectedTfCore(unittest.TestCase):
    """Cover the broadcaster cases fixed in design section 11.1."""

    def assert_quaternion(self, actual, expected):
        """Compare an xyzw quaternion component-wise."""
        for actual_value, expected_value in zip(actual, expected):
            self.assertAlmostEqual(actual_value, expected_value, places=6)

    def test_zero_tilt_and_exact_stamp(self):
        core = CorrectedTfCore()
        outcome = validate(core, 0.0, sec=42, nanosec=123456789)

        self.assertTrue(outcome.accepted)
        sample = outcome.sample
        self.assertIsNotNone(sample)
        self.assertEqual(sample.stamp, (42, 123456789))
        self.assertEqual(sample.q_corrected, 0.0)
        self.assertEqual(sample.transform.translation, DYNAMIC_TRANSLATION)
        self.assert_quaternion(
            sample.transform.quaternion_xyzw,
            (0.0, 0.0, 0.0, 1.0),
        )

    def test_negative_tilt_uses_negated_joint_state_position(self):
        core = CorrectedTfCore()
        sample = validate(core, -0.5224).sample

        self.assertIsNotNone(sample)
        self.assertAlmostEqual(sample.q_corrected, 0.5224)
        self.assert_quaternion(
            sample.transform.quaternion_xyzw,
            (0.0, 0.258240, 0.0, 0.966081),
        )

    def test_positive_tilt_uses_negative_y_rotation(self):
        core = CorrectedTfCore()
        sample = validate(core, math.pi / 6.0).sample

        self.assertIsNotNone(sample)
        self.assertAlmostEqual(sample.q_corrected, -math.pi / 6.0)
        self.assert_quaternion(
            sample.transform.quaternion_xyzw,
            (0.0, -0.258819, 0.0, 0.965926),
        )

    def test_joint_name_order_selects_matching_position(self):
        core = CorrectedTfCore()
        outcome = core.validate(
            ['camera_tilt_joint', 'camera_pan_joint'],
            [-0.25, 0.4],
            1,
            2,
        )

        sample = outcome.sample
        self.assertIsNotNone(sample)
        self.assertEqual(sample.q_joint_states, -0.25)
        self.assertEqual(sample.q_corrected, 0.25)

    def test_joint_array_validation(self):
        cases = [
            (['camera_pan_joint'], [0.0], 'missing_joint'),
            (
                ['camera_tilt_joint', 'camera_tilt_joint'],
                [0.0, 0.1],
                'duplicate_joint',
            ),
            (
                ['camera_pan_joint', 'camera_tilt_joint'],
                [0.0],
                'malformed_joint_state',
            ),
        ]
        for names, positions, reason in cases:
            with self.subTest(reason=reason):
                core = CorrectedTfCore()
                outcome = core.validate(names, positions, 1, 2)

                self.assertFalse(outcome.accepted)
                self.assertEqual(outcome.rejection_reason, reason)
                self.assertEqual(core.counters[reason], 1)
                self.assertIsNone(core.last_accepted_stamp)

    def test_non_finite_position_is_rejected(self):
        for q in (math.nan, math.inf, -math.inf):
            with self.subTest(q=q):
                core = CorrectedTfCore()
                outcome = validate(core, q)

                self.assertFalse(outcome.accepted)
                self.assertEqual(outcome.rejection_reason, 'invalid_position')
                self.assertEqual(core.counters['invalid_position'], 1)

    def test_zero_stamp_is_rejected_without_substitution(self):
        core = CorrectedTfCore()
        outcome = validate(core, 0.0, sec=0, nanosec=0)

        self.assertFalse(outcome.accepted)
        self.assertEqual(outcome.rejection_reason, 'invalid_stamp')
        self.assertIsNone(core.last_accepted_stamp)

    def test_duplicate_and_older_stamps_are_rejected_after_commit(self):
        core = CorrectedTfCore()
        first = validate(core, -0.1, sec=5, nanosec=10)
        self.assertIsNotNone(first.sample)
        core.commit(first.sample)

        duplicate = validate(core, -0.2, sec=5, nanosec=10)
        older = validate(core, -0.3, sec=5, nanosec=9)

        self.assertEqual(
            duplicate.rejection_reason,
            'out_of_order_or_duplicate',
        )
        self.assertEqual(older.rejection_reason, 'out_of_order_or_duplicate')
        self.assertEqual(core.last_accepted_stamp, (5, 10))
        self.assertEqual(core.last_q, -0.1)
        self.assertEqual(core.counters['accepted'], 1)

    def test_validation_does_not_commit_before_send_success(self):
        core = CorrectedTfCore()
        outcome = validate(core, -0.1, sec=5, nanosec=10)

        self.assertTrue(outcome.accepted)
        self.assertIsNone(core.last_accepted_stamp)
        self.assertEqual(core.counters['accepted'], 0)

        core.record_send_error()
        self.assertIsNone(core.last_accepted_stamp)
        self.assertEqual(core.counters['send_error'], 1)

    def test_clock_reset_starts_a_new_stamp_epoch(self):
        core = CorrectedTfCore()
        first = validate(core, -0.1, sec=100, nanosec=0)
        self.assertIsNotNone(first.sample)
        core.commit(first.sample)

        rejected = validate(core, -0.2, sec=1, nanosec=0)
        self.assertEqual(
            rejected.rejection_reason,
            'out_of_order_or_duplicate',
        )
        core.reset_clock_epoch()
        after_reset = validate(core, -0.2, sec=1, nanosec=0)

        self.assertTrue(after_reset.accepted)
        self.assertEqual(core.clock_epoch, 1)
        self.assertEqual(core.counters['clock_reset'], 1)

    def test_static_transform_contract(self):
        camera, optical = corrected_static_transforms()

        self.assertEqual(camera.parent_frame, DYNAMIC_CHILD_FRAME)
        self.assertEqual(camera.child_frame, STATIC_CAMERA_CHILD_FRAME)
        self.assertEqual(camera.translation, CAMERA_TRANSLATION)
        self.assert_quaternion(
            camera.quaternion_xyzw,
            (0.0, 0.0, 0.0, 1.0),
        )

        self.assertEqual(optical.parent_frame, STATIC_CAMERA_CHILD_FRAME)
        self.assertEqual(optical.child_frame, STATIC_OPTICAL_CHILD_FRAME)
        self.assertEqual(optical.translation, (0.0, 0.0, 0.0))
        self.assert_quaternion(optical.quaternion_xyzw, OPTICAL_QUATERNION)

    def test_dynamic_parent_and_child_names_are_fixed(self):
        core = CorrectedTfCore()
        sample = validate(core, 0.0).sample

        self.assertIsNotNone(sample)
        self.assertEqual(sample.transform.parent_frame, DYNAMIC_PARENT_FRAME)
        self.assertEqual(sample.transform.child_frame, DYNAMIC_CHILD_FRAME)


if __name__ == '__main__':
    unittest.main()
