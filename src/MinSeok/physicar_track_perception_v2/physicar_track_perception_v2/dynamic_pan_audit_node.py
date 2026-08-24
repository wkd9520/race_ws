"""P0 diagnostic-only exact-TF dynamic-pan BEV capture node."""

from pathlib import Path
import json
import math

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String
import tf2_ros

from .components import CanonicalComponentExtractor, ComponentExtractionConfig
from .frontend import BevFrontend
from .geometry import (BevGrid, CameraModel, MetricGroundProjector,
                       apply_projection_corrections)
from .segmentation import ColorComponentPipeline, HsvRange


class DynamicPanAuditNode(Node):
    """Captures a counterfactual dynamic projector; never publishes commands."""

    def __init__(self):
        super().__init__('physicar_v2_dynamic_pan_audit')
        self.declare_parameter('directory', '')
        self.declare_parameter('stride', 5)
        self.directory = Path(self.get_parameter('directory').value)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, int(self.get_parameter('stride').value))
        self.bridge = CvBridge()
        self.camera = CameraModel(
            np.array([[201.38988018035889, 0., 240.],
                      [0., 201.38988733291626, 180.], [0., 0., 1.]]),
            np.array([-0.045, -0.0001, -0.0003, -0.0001, 0.001]),
            480, 360)
        self.grid = BevGrid(.10, 2.00, -.75, .75, .01)
        extractor = CanonicalComponentExtractor(
            self.grid, ComponentExtractionConfig())
        self.segmentation = ColorComponentPipeline({
            'WHITE': (HsvRange((0, 0, 170), (179, 90, 255)),),
            'ORANGE': (HsvRange((5, 100, 100), (30, 255, 255)),),
        }, 3, 5, extractor)
        self.tf_buffer = tf2_ros.Buffer(
            cache_time=Duration(seconds=10.), node=self)
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True)
        self.latest_joint = None
        self.marker = 'UNMARKED'
        self.production_ready = False
        self.frame = self.success = self.failure = 0
        self.timeout = 0
        self.index = self.directory/'pan_audit_index.jsonl'
        self.create_subscription(JointState, '/joint_states', self._joint, 10)
        self.create_subscription(Image, '/camera/image_raw', self._image,
                                 qos_profile_sensor_data)
        self.create_subscription(String, '/perception_v2/debug/pan_audit_marker',
                                 self._marker, 10)
        self.create_subscription(Bool, '/perception_v2/debug/bev_ready',
                                 self._ready, 10)
        self.get_logger().info(
            f'P0 audit directory={self.directory} exact_image_tf=True '
            'dynamic_projector_shadow_only=True command_publisher=False')

    def _joint(self, message):
        values = dict(zip(message.name, message.position))
        if 'camera_pan_joint' in values and 'camera_tilt_joint' in values:
            self.latest_joint = (message.header.stamp,
                                 float(values['camera_pan_joint']),
                                 float(values['camera_tilt_joint']))

    def _marker(self, message):
        self.marker = message.data.strip().upper() or 'UNMARKED'
        self.get_logger().info(f'P0 marker={self.marker}')

    def _ready(self, message):
        self.production_ready = bool(message.data)

    @staticmethod
    def _matrix(transform):
        t, q = transform.transform.translation, transform.transform.rotation
        n = math.sqrt(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w)
        x, y, z, w = q.x/n, q.y/n, q.z/n, q.w/n
        value = np.eye(4)
        value[:3, :3] = [
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ]
        value[:3, 3] = [t.x, t.y, t.z]
        return value

    @staticmethod
    def _seconds(stamp):
        return float(stamp.sec)+1e-9*float(stamp.nanosec)

    def _image(self, message):
        self.frame += 1
        if self.frame % self.stride:
            return
        stamp = Time.from_msg(message.header.stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                'base_footprint', 'camera_optical_frame_corrected', stamp,
                timeout=Duration(seconds=0.0))
        except Exception:
            self.failure += 1
            self.timeout += 1
            return
        self.success += 1
        raw_matrix = self._matrix(transform)
        corrected = apply_projection_corrections(raw_matrix)
        frontend = BevFrontend(
            self.camera, MetricGroundProjector(
                self.camera, self.grid, corrected, ground_z=0.0))
        image = self.bridge.imgmsg_to_cv2(message, 'bgr8')
        output = frontend.process(image)
        segmented = self.segmentation.process(
            output.bev, frontend.bev_valid_map)
        pan = tilt = joint_stamp = float('nan')
        if self.latest_joint is not None:
            js, pan, tilt = self.latest_joint
            joint_stamp = self._seconds(js)
        image_stamp = self._seconds(message.header.stamp)
        candidates = []
        for value in segmented.component_frame.candidates:
            candidates.append({
                'id': int(value.component_id), 'color': value.color,
                'support': float(value.support_length),
                'points': int(value.canonical_point_count),
                'near': value.near_endpoint.tolist(),
                'far': value.far_endpoint.tolist(),
                'extent_x': float(np.ptp(value.canonical_points[:, 0])),
                'extent_y': float(np.ptp(value.canonical_points[:, 1])),
            })
        record = {
            'image_stamp': image_stamp, 'joint_stamp': joint_stamp,
            'image_minus_joint_s': image_stamp-joint_stamp,
            'marker': self.marker, 'pan': pan, 'tilt': tilt,
            'translation': raw_matrix[:3, 3].tolist(),
            'quaternion_xyzw': [transform.transform.rotation.x,
                                transform.transform.rotation.y,
                                transform.transform.rotation.z,
                                transform.transform.rotation.w],
            'optical_z_base': raw_matrix[:3, 2].tolist(),
            'valid_fraction': float(np.mean(frontend.bev_valid_map)),
            'production_bev_ready_latest': self.production_ready,
            'exact_tf_success': self.success, 'exact_tf_failure': self.failure,
            'candidates': candidates,
        }
        stem = f'{message.header.stamp.sec:010d}_{message.header.stamp.nanosec:09d}'
        np.savez_compressed(
            self.directory/f'{stem}.npz', source=image,
            dynamic_bev=output.bev, validity=output.validity_mask,
            white_mask=segmented.white_mask,
            orange_mask=segmented.orange_mask,
            component_overlay=segmented.overlay,
            metadata_json=np.asarray(json.dumps(record)))
        with self.index.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True)+'\n')


def main(args=None):
    rclpy.init(args=args)
    node = DynamicPanAuditNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
