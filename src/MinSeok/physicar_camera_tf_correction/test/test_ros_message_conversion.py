"""ROS-dependent tests for TransformStamped field conversion."""

from importlib.util import find_spec
import unittest


ROS_MESSAGES_AVAILABLE = find_spec('rclpy') is not None

if ROS_MESSAGES_AVAILABLE:
    from physicar_camera_tf_correction.core import corrected_dynamic_transform
    from physicar_camera_tf_correction.core import corrected_static_transforms
    from physicar_camera_tf_correction.corrected_tf_broadcaster import (
        CameraCorrectedTfBroadcaster,
    )


@unittest.skipUnless(ROS_MESSAGES_AVAILABLE, 'requires sourced ROS 2 Python packages')
class TestRosMessageConversion(unittest.TestCase):
    """Check exact frame and timestamp fields in generated ROS messages."""

    def test_dynamic_message_copies_stamp_exactly(self):
        spec = corrected_dynamic_transform(-0.5224)
        message = CameraCorrectedTfBroadcaster._to_transform_stamped(
            spec,
            stamp_sec=42,
            stamp_nanosec=123456789,
        )

        self.assertEqual(message.header.stamp.sec, 42)
        self.assertEqual(message.header.stamp.nanosec, 123456789)
        self.assertEqual(message.header.frame_id, 'camera_pan_link')
        self.assertEqual(message.child_frame_id, 'camera_tilt_link_corrected')
        self.assertAlmostEqual(message.transform.rotation.y, 0.258240, places=6)
        self.assertAlmostEqual(message.transform.rotation.w, 0.966081, places=6)

    def test_static_messages_keep_zero_stamp_and_fixed_frames(self):
        camera_spec, optical_spec = corrected_static_transforms()
        camera = CameraCorrectedTfBroadcaster._to_transform_stamped(camera_spec)
        optical = CameraCorrectedTfBroadcaster._to_transform_stamped(optical_spec)

        self.assertEqual((camera.header.stamp.sec, camera.header.stamp.nanosec), (0, 0))
        self.assertEqual(camera.header.frame_id, 'camera_tilt_link_corrected')
        self.assertEqual(camera.child_frame_id, 'camera_link_corrected')
        self.assertEqual((optical.header.stamp.sec, optical.header.stamp.nanosec), (0, 0))
        self.assertEqual(optical.header.frame_id, 'camera_link_corrected')
        self.assertEqual(optical.child_frame_id, 'camera_optical_frame_corrected')


if __name__ == '__main__':
    unittest.main()
