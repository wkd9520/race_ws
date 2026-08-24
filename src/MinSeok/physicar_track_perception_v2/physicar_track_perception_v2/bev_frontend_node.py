"""ROS 2 BEV-only node with verified legacy Stage 3 projection semantics."""

import math
from pathlib import Path
import time
from types import SimpleNamespace

import cv2
from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32, String, UInt32
import tf2_ros

from .frontend import BevFrontend
from .both_geometry import BothGeometryConfig, FrameLocalBothGeometry
from .components import CanonicalComponentExtractor, ComponentExtractionConfig
from .geometry import (
    BevGrid,
    CameraModel,
    MetricGroundProjector,
    apply_projection_corrections,
)
from .segmentation import ColorComponentPipeline, HsvRange
from .trusted_identity import IdentityConfig, TrustedBoundaryIdentity
from .single_reconstruction import TrustedSingleReconstruction, ValidatedWidth, WidthConfig
from .runtime_geometry_capture import RuntimeGeometryCapture
from .arc_shadow import ArcShadowConfig, ArcShadowTracker
from .arc_shadow_capture import ArcShadowCapture
from .arc_prior import ArcPriorConfig, OdomArcPrior
from .white_continuity_capture import WhiteContinuityCapture
from .arc_support_capture import ArcSupportCapture
from .dynamic_bev import BoundedPendingFrames, DynamicPanGuard
from .pan_association_characterization import PanAssociationCharacterizer
from .side_gate_characterization import SideGateCharacterizer


VEHICLE_FRAME = 'base_footprint'
CAMERA_FRAME = 'camera_optical_frame_corrected'
IMAGE_TOPIC = '/camera/image_raw'
JOINT_STATE_TOPIC = '/joint_states'
PAN_JOINT = 'camera_pan_joint'
TILT_JOINT = 'camera_tilt_joint'


class BevFrontendNode(Node):
    def __init__(self):
        super().__init__('physicar_track_perception_v2_bev')
        self._declare_parameters()
        self.bridge = CvBridge()
        self.camera = CameraModel(
            np.asarray(self.get_parameter('camera.K').value).reshape(3, 3),
            np.asarray(self.get_parameter('camera.D').value),
            int(self.get_parameter('camera.width').value),
            int(self.get_parameter('camera.height').value),
        )
        self.grid = BevGrid(
            float(self.get_parameter('bev.x_min').value),
            float(self.get_parameter('bev.x_max').value),
            float(self.get_parameter('bev.y_min').value),
            float(self.get_parameter('bev.y_max').value),
            float(self.get_parameter('bev.resolution').value),
        )
        self.frontend = None
        self.segmentation = None
        self.both_geometry = None
        self.identity = None
        self.single_reconstruction = None
        self.fixed_pose_ready = False
        self.current_pan = float('nan')
        self.current_tilt = float('nan')
        self.pending = BoundedPendingFrames(2)
        self.arc_prior_pending = []
        self.camera_tf_immediate = 0
        self.camera_tf_eventual = 0
        self.camera_tf_timeout = 0
        self.camera_tf_waits = []
        self.dynamic_map_times = []
        self.last_tf_reason = 'not checked'
        self.tf_buffer = tf2_ros.Buffer(cache_time=Duration(seconds=10.0), node=self)
        self.tf_listener = tf2_ros.TransformListener(
            self.tf_buffer, self, spin_thread=True
        )
        self.create_subscription(
            Image, IMAGE_TOPIC, self._image_callback, qos_profile_sensor_data
        )
        self.create_subscription(
            JointState, JOINT_STATE_TOPIC, self._joint_callback,
            qos_profile_sensor_data,
        )
        self.create_timer(
            float(self.get_parameter('tf_wait.timer_period').value),
            self._process_pending,
        )
        self.undistorted_pub = self.create_publisher(
            Image, '/perception_v2/debug/undistorted', 2
        )
        self.bev_pub = self.create_publisher(
            Image, '/perception_v2/debug/bev', 2
        )
        self.validity_pub = self.create_publisher(
            Image, '/perception_v2/debug/bev_validity', 2
        )
        self.ready_pub = self.create_publisher(
            Bool, '/perception_v2/debug/bev_ready', 10
        )
        self.valid_fraction_pub = self.create_publisher(
            Float32, '/perception_v2/debug/bev_valid_fraction', 10
        )
        self.current_pan_pub = self.create_publisher(
            Float32, '/perception_v2/debug/current_pan', 10
        )
        self.dynamic_ready_pub = self.create_publisher(
            Bool, '/perception_v2/debug/dynamic_bev_ready', 10
        )
        self.white_mask_pub = self.create_publisher(
            Image, '/perception_v2/debug/white_mask', 2
        )
        self.orange_mask_pub = self.create_publisher(
            Image, '/perception_v2/debug/orange_mask', 2
        )
        self.component_overlay_pub = self.create_publisher(
            Image, '/perception_v2/debug/component_overlay', 2
        )
        self.candidate_count_pub = self.create_publisher(
            UInt32, '/perception_v2/debug/candidate_count', 10
        )
        self.both_overlay_pub = self.create_publisher(
            Image, '/perception_v2/debug/both_center_overlay', 2
        )
        self.both_valid_pub = self.create_publisher(
            Bool, '/perception_v2/debug/both_center_valid', 10
        )
        self.usable_count_pub = self.create_publisher(
            UInt32, '/perception_v2/debug/usable_candidate_count', 10
        )
        self.center_point_count_pub = self.create_publisher(
            UInt32, '/perception_v2/debug/center_point_count', 10
        )
        self.observed_width_pub = self.create_publisher(
            Float32, '/perception_v2/debug/observed_width', 10
        )
        self.identity_overlay_pub = self.create_publisher(
            Image, '/perception_v2/debug/identity_overlay', 2
        )
        self.identity_initialized_pub = self.create_publisher(
            Bool, '/perception_v2/debug/identity_initialized', 10
        )
        self.initialization_streak_pub = self.create_publisher(
            UInt32, '/perception_v2/debug/initialization_streak', 10
        )
        self.single_overlay_pub = self.create_publisher(
            Image, '/perception_v2/debug/single_reconstruction_overlay', 2
        )
        self.observation_mode_pub = self.create_publisher(
            String, '/perception_v2/debug/observation_mode', 10
        )
        self.provenance_pub = self.create_publisher(
            String, '/perception_v2/debug/provenance', 10
        )
        self.trusted_width_pub = self.create_publisher(
            Float32, '/perception_v2/debug/trusted_width', 10
        )
        self.width_update_allowed_pub = self.create_publisher(
            Bool, '/perception_v2/debug/width_update_allowed', 10
        )
        self.white_raw_pub = self.create_publisher(
            Image, '/perception_v2/debug/white_raw_mask', 2)
        self.white_post_validity_pub = self.create_publisher(
            Image, '/perception_v2/debug/white_post_validity_diagnostic', 2)
        self.white_after_open_pub = self.create_publisher(
            Image, '/perception_v2/debug/white_after_open', 2)
        self.white_after_close_pub = self.create_publisher(
            Image, '/perception_v2/debug/white_after_close', 2)
        self.white_continuity_capture = None
        if bool(self.get_parameter('white_continuity_capture.enabled').value):
            self.white_continuity_capture = WhiteContinuityCapture(
                Path(self.get_parameter(
                    'white_continuity_capture.directory').value),
                int(self.get_parameter(
                    'white_continuity_capture.stride').value))
            self.get_logger().info(
                '[BEV V2 WHITE CONTINUITY] diagnostics_only=True '
                f'directory={self.white_continuity_capture.directory} '
                f'stride={self.white_continuity_capture.stride} '
                'production_mask_unchanged=True')
        self.arc_support_capture = None
        if bool(self.get_parameter('arc_support_capture.enabled').value):
            self.arc_support_capture = ArcSupportCapture(
                Path(self.get_parameter('arc_support_capture.directory').value),
                int(self.get_parameter('arc_support_capture.stride').value))
            self.create_subscription(
                String, '/perception_v2/debug/arc_support_scene',
                self._arc_support_scene_callback, 10)
            self.get_logger().info(
                '[BEV V2 ARC SUPPORT] diagnostics_only=True '
                f'directory={self.arc_support_capture.directory} '
                'production_connected=False')
        self.geometry_capture = None
        if bool(self.get_parameter('capture.enabled').value):
            self.geometry_capture = RuntimeGeometryCapture(Path(
                self.get_parameter('capture.directory').value))
            self.create_subscription(
                String, '/perception_v2/debug/ransac_capture_scene',
                self._capture_scene_callback, 10)
            self.get_logger().info(
                f'[BEV V2 CAPTURE] directory={self.geometry_capture.directory} '
                'scene_topic=/perception_v2/debug/ransac_capture_scene state=IDLE')
        self.arc_shadow = None
        self.arc_shadow_capture = None
        self.arc_shadow_pending = []
        if bool(self.get_parameter('arc_shadow.enabled').value):
            self.arc_shadow = ArcShadowTracker(ArcShadowConfig())
            self.arc_shadow_capture = ArcShadowCapture(Path(
                self.get_parameter('arc_shadow.directory').value))
            self.create_subscription(
                String, '/perception_v2/debug/arc_shadow_scene',
                self._arc_shadow_scene_callback, 10)
            self.get_logger().info(
                f'[BEV V2 ARC SHADOW] directory={self.arc_shadow_capture.directory} '
                'fixed_frame=odom vehicle_frame=base_footprint exact_stamp=True '
                'production_connected=False')
        self.arc_prior = (OdomArcPrior(ArcPriorConfig())
                          if bool(self.get_parameter('arc_prior.enabled').value)
                          else None)
        self.pan_association_characterizer = None
        if bool(self.get_parameter(
                'pan_association_characterization.enabled').value):
                self.pan_association_characterizer = PanAssociationCharacterizer(
                Path(self.get_parameter(
                    'pan_association_characterization.directory').value))
        self.side_gate_characterizer = None
        if bool(self.get_parameter(
                'side_gate_characterization.enabled').value):
            self.side_gate_characterizer = SideGateCharacterizer(str(
                self.get_parameter(
                    'side_gate_characterization.directory').value))
            self.get_logger().info(
                '[BEV V2 PAN ASSOCIATION CHARACTERIZATION] '
                'diagnostics_only=True production_connected=False')
        self.frame_count = 0
        self.get_logger().info(
            f'[BEV V2 READY] image={IMAGE_TOPIC} message=sensor_msgs/msg/Image '
            f'tf={VEHICLE_FRAME}<-{CAMERA_FRAME} exact_image_stamp=True '
            f'grid={self.grid.width}x{self.grid.height}'
        )

    def _declare_parameters(self):
        self.declare_parameter('camera.width', 480)
        self.declare_parameter('camera.height', 360)
        self.declare_parameter('camera.K', [
            201.38988018035889, 0.0, 240.0,
            0.0, 201.38988733291626, 180.0,
            0.0, 0.0, 1.0,
        ])
        self.declare_parameter(
            'camera.D', [-0.045, -0.0001, -0.0003, -0.0001, 0.001]
        )
        self.declare_parameter('bev.x_min', 0.10)
        self.declare_parameter('bev.x_max', 2.00)
        self.declare_parameter('bev.y_min', -0.75)
        self.declare_parameter('bev.y_max', 0.75)
        self.declare_parameter('bev.resolution', 0.01)
        self.declare_parameter('ground_z', 0.0)
        self.declare_parameter('sim_geometry.camera_height_correction_z', -0.018)
        self.declare_parameter('projection.pitch_offset_deg', 2.7)
        self.declare_parameter('projection.pitch_correction_frame', 'pan_local_y')
        self.declare_parameter('fixed_pose.expected_pan', 0.0)
        self.declare_parameter('fixed_pose.expected_tilt', -0.5236)
        self.declare_parameter('fixed_pose.tolerance', 0.01)
        self.declare_parameter('fixed_pose.require_expected', True)
        self.declare_parameter('dynamic_pan.min', -0.5236)
        self.declare_parameter('dynamic_pan.max', 0.5236)
        self.declare_parameter('dynamic_pan.limit_tolerance', 0.001)
        self.declare_parameter('tf_wait.timer_period', 0.02)
        self.declare_parameter('tf_wait.max_pending_age', 0.25)
        self.declare_parameter('hsv.white.lower', [0, 0, 170])
        self.declare_parameter('hsv.white.upper', [179, 90, 255])
        self.declare_parameter('hsv.orange.lower', [5, 100, 100])
        self.declare_parameter('hsv.orange.upper', [30, 255, 255])
        self.declare_parameter('morphology.open_kernel', 3)
        self.declare_parameter('morphology.close_kernel', 5)
        self.declare_parameter('component.min_area', 8)
        self.declare_parameter('component.min_valid_pixels', 3)
        self.declare_parameter('component.min_valid_overlap', 0.70)
        self.declare_parameter('canonical.spacing', 0.05)
        self.declare_parameter('canonical.duplicate_tolerance', 1e-9)
        self.declare_parameter('both.usable_min_support', 0.20)
        self.declare_parameter('both.usable_min_points', 5)
        self.declare_parameter('both.min_width', 0.60)
        self.declare_parameter('both.max_width', 0.95)
        self.declare_parameter('both.min_correspondences', 4)
        self.declare_parameter('both.min_overlap_support', 0.15)
        self.declare_parameter('both.max_tangent_angle', 0.45)
        self.declare_parameter('both.min_side_consistency', 0.80)
        self.declare_parameter('both.max_width_spread', 0.15)
        self.declare_parameter('both.ambiguity_score_margin', 0.05)
        self.declare_parameter('both.center_spacing', 0.05)
        self.declare_parameter('identity.initialization_frames', 3)
        self.declare_parameter('identity.initialization_min_overlap_fraction', 0.50)
        self.declare_parameter('identity.distance_gate', 0.12)
        self.declare_parameter('identity.tangent_gate', 0.45)
        self.declare_parameter('identity.min_overlap_support', 0.15)
        self.declare_parameter('identity.min_accepted_support', 0.20)
        self.declare_parameter('identity.max_continuation', 0.15)
        self.declare_parameter('identity.max_gap', 0.075)
        self.declare_parameter('identity.conflict_min_side_consistency', 0.80)
        self.declare_parameter('identity.conflict_max_lateral_residual', 0.12)
        self.declare_parameter('width.initialization_frames', 3)
        self.declare_parameter('width.ema_alpha', 0.20)
        self.declare_parameter('width.update_gate', 0.12)
        self.declare_parameter('single.curvature_support', 0.10)
        self.declare_parameter('single.persistence_span', 0.15)
        self.declare_parameter('single.min_curvature_samples', 3)
        self.declare_parameter('capture.enabled', False)
        self.declare_parameter('capture.directory', '')
        self.declare_parameter('arc_shadow.enabled', False)
        self.declare_parameter('arc_shadow.directory', '')
        self.declare_parameter('arc_prior.enabled', True)
        # OBSERVED runtime TF lag was about 0.7 s. This bounded exact-stamp
        # queue never falls back to latest TF and releases production on expiry.
        self.declare_parameter('arc_prior.max_pending_wall_age', 1.25)
        self.declare_parameter('white_continuity_capture.enabled', False)
        self.declare_parameter('white_continuity_capture.directory', '')
        self.declare_parameter('white_continuity_capture.stride', 5)
        self.declare_parameter('arc_support_capture.enabled', False)
        self.declare_parameter('arc_support_capture.directory', '')
        self.declare_parameter('arc_support_capture.stride', 1)
        self.declare_parameter(
            'pan_association_characterization.enabled', False)
        self.declare_parameter(
            'pan_association_characterization.directory', '')
        self.declare_parameter('side_gate_characterization.enabled', False)
        self.declare_parameter('side_gate_characterization.directory', '')

    def _capture_scene_callback(self, message):
        if self.geometry_capture is None:
            return
        self.geometry_capture.set_scene(message.data, time.time())
        self.get_logger().info(
            f'[BEV V2 CAPTURE MARKER] scene={self.geometry_capture.scene} '
            f'active={self.geometry_capture.active}')

    def _arc_shadow_scene_callback(self, message):
        if self.arc_shadow_capture is None:
            return
        self.arc_shadow_capture.set_scene(message.data, time.time())
        self.get_logger().info(
            f'[BEV V2 ARC SHADOW MARKER] scene={self.arc_shadow_capture.scene}')

    def _arc_support_scene_callback(self, message):
        if self.arc_support_capture is None:
            return
        self.arc_support_capture.set_scene(message.data)
        self.get_logger().info(
            f'[BEV V2 ARC SUPPORT MARKER] scene={self.arc_support_capture.scene}')

    def _joint_callback(self, message):
        if len(message.name) != len(message.position):
            self.fixed_pose_ready = False
            return
        values = dict(zip(message.name, message.position))
        if PAN_JOINT not in values or TILT_JOINT not in values:
            self.fixed_pose_ready = False
            return
        pan, tilt = float(values[PAN_JOINT]), float(values[TILT_JOINT])
        self.current_pan, self.current_tilt = pan, tilt
        if not math.isfinite(pan) or not math.isfinite(tilt):
            self.fixed_pose_ready = False
            return
        if not bool(self.get_parameter('fixed_pose.require_expected').value):
            self.fixed_pose_ready = True
            return
        tolerance = float(self.get_parameter('fixed_pose.tolerance').value)
        self.fixed_pose_ready = DynamicPanGuard(
            pan_min=float(self.get_parameter('dynamic_pan.min').value),
            pan_max=float(self.get_parameter('dynamic_pan.max').value),
            pan_tolerance=float(self.get_parameter(
                'dynamic_pan.limit_tolerance').value),
            expected_tilt=float(self.get_parameter(
                'fixed_pose.expected_tilt').value),
            tilt_tolerance=tolerance).accepts(pan, tilt)
        self.current_pan_pub.publish(Float32(data=pan))

    def _image_callback(self, message):
        if message.header.stamp.sec == 0 and message.header.stamp.nanosec == 0:
            self.get_logger().warning('zero image stamp rejected; latest TF fallback forbidden')
            self.ready_pub.publish(Bool(data=False))
            return
        if not self.fixed_pose_ready:
            self.ready_pub.publish(Bool(data=False))
            return
        entry = (message, time.monotonic())
        self.pending.append(*entry)
        self._process_camera_pending(from_callback=True)

    def _process_pending(self):
        self._process_camera_pending(from_callback=False)
        self._process_arc_prior_pending()

    def _process_camera_pending(self, *, from_callback):
        if not self.fixed_pose_ready:
            return
        max_age = float(self.get_parameter('tf_wait.max_pending_age').value)
        now = time.monotonic()
        while len(self.pending):
            self.camera_tf_timeout += self.pending.expire(now, max_age)
            if not len(self.pending):
                self.dynamic_ready_pub.publish(Bool(data=False))
                break
            message, queued = self.pending.peek()
            age = now-queued
            transform = self._lookup_transform(message)
            if transform is None:
                break
            self.pending.pop()
            self.camera_tf_waits.append(age)
            if from_callback and age < 0.005:
                self.camera_tf_immediate += 1
            else:
                self.camera_tf_eventual += 1
            if self.frontend is None:
                self._initialize_frontend(transform)
            if self.arc_prior is None:
                self._process_image(message, transform, None)
            else:
                self.arc_prior_pending.append(
                    (message, transform, time.monotonic()))

    def _process_arc_prior_pending(self):
        if self.frontend is None or not self.arc_prior_pending:
            return
        now = time.monotonic()
        max_age = float(self.get_parameter(
            'arc_prior.max_pending_wall_age').value)
        while self.arc_prior_pending:
            message, camera_transform, queued = self.arc_prior_pending[0]
            transform = self._lookup_odom_transform(message.header.stamp)
            if transform is None and now-queued <= max_age:
                break
            self.arc_prior_pending.pop(0)
            if transform is None:
                self.get_logger().warning(
                    '[BEV V2 ARC PRIOR] exact_transform_timeout '
                    f'stamp={message.header.stamp.sec}.'
                    f'{message.header.stamp.nanosec:09d} '
                    'production_fallback=existing_association')
            self._process_image(message, camera_transform, transform)

    def _lookup_transform(self, message):
        stamp = Time.from_msg(message.header.stamp)
        try:
            ready, debug = self.tf_buffer.can_transform(
                VEHICLE_FRAME, CAMERA_FRAME, stamp,
                timeout=Duration(seconds=0.0), return_debug_tuple=True,
            )
            if not ready:
                self.last_tf_reason = str(debug)
                return None
            return self.tf_buffer.lookup_transform(
                VEHICLE_FRAME, CAMERA_FRAME, stamp,
                timeout=Duration(seconds=0.0),
            )
        except Exception as error:
            self.last_tf_reason = str(error)
            return None

    @staticmethod
    def _transform_matrix(transform):
        t, q = transform.transform.translation, transform.transform.rotation
        norm = math.sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w)
        if norm <= 0.0:
            raise ValueError('zero-norm TF quaternion')
        x, y, z, w = q.x/norm, q.y/norm, q.z/norm, q.w/norm
        result = np.eye(4)
        result[:3, :3] = [
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ]
        result[:3, 3] = [t.x, t.y, t.z]
        return result

    def _initialize_frontend(self, transform):
        height_correction_z = float(self.get_parameter(
            'sim_geometry.camera_height_correction_z').value)
        pitch_offset_deg = float(self.get_parameter(
            'projection.pitch_offset_deg').value)
        corrected = apply_projection_corrections(
            self._transform_matrix(transform),
            camera_height_correction_z=height_correction_z,
            pitch_offset_deg=pitch_offset_deg,
            pitch_correction_frame=str(self.get_parameter(
                'projection.pitch_correction_frame').value),
        )
        projector = MetricGroundProjector(
            self.camera, self.grid, corrected,
            ground_z=float(self.get_parameter('ground_z').value),
        )
        self.frontend = BevFrontend(self.camera, projector)
        extractor = CanonicalComponentExtractor(
            self.grid,
            ComponentExtractionConfig(
                min_component_area=int(self.get_parameter('component.min_area').value),
                min_valid_pixels=int(self.get_parameter('component.min_valid_pixels').value),
                min_valid_overlap=float(self.get_parameter('component.min_valid_overlap').value),
                canonical_spacing=float(self.get_parameter('canonical.spacing').value),
                duplicate_tolerance=float(self.get_parameter('canonical.duplicate_tolerance').value),
            ),
        )
        ranges = {
            'WHITE': (HsvRange(
                tuple(self.get_parameter('hsv.white.lower').value),
                tuple(self.get_parameter('hsv.white.upper').value)),),
            'ORANGE': (HsvRange(
                tuple(self.get_parameter('hsv.orange.lower').value),
                tuple(self.get_parameter('hsv.orange.upper').value)),),
        }
        self.segmentation = ColorComponentPipeline(
            ranges,
            int(self.get_parameter('morphology.open_kernel').value),
            int(self.get_parameter('morphology.close_kernel').value),
            extractor,
        )
        self.both_geometry = FrameLocalBothGeometry(BothGeometryConfig(
            usable_min_support=float(self.get_parameter('both.usable_min_support').value),
            usable_min_points=int(self.get_parameter('both.usable_min_points').value),
            min_width=float(self.get_parameter('both.min_width').value),
            max_width=float(self.get_parameter('both.max_width').value),
            min_correspondences=int(self.get_parameter('both.min_correspondences').value),
            min_overlap_support=float(self.get_parameter('both.min_overlap_support').value),
            max_tangent_angle=float(self.get_parameter('both.max_tangent_angle').value),
            min_side_consistency=float(self.get_parameter('both.min_side_consistency').value),
            max_width_spread=float(self.get_parameter('both.max_width_spread').value),
            ambiguity_score_margin=float(self.get_parameter('both.ambiguity_score_margin').value),
            center_spacing=float(self.get_parameter('both.center_spacing').value),
        ))
        self.identity = TrustedBoundaryIdentity(
            self.both_geometry, IdentityConfig(
                initialization_frames=int(self.get_parameter('identity.initialization_frames').value),
                initialization_min_overlap_fraction=float(self.get_parameter('identity.initialization_min_overlap_fraction').value),
                distance_gate=float(self.get_parameter('identity.distance_gate').value),
                tangent_gate=float(self.get_parameter('identity.tangent_gate').value),
                min_overlap_support=float(self.get_parameter('identity.min_overlap_support').value),
                min_accepted_support=float(self.get_parameter('identity.min_accepted_support').value),
                max_continuation=float(self.get_parameter('identity.max_continuation').value),
                max_gap=float(self.get_parameter('identity.max_gap').value),
                conflict_min_side_consistency=float(self.get_parameter('identity.conflict_min_side_consistency').value),
                conflict_max_lateral_residual=float(self.get_parameter('identity.conflict_max_lateral_residual').value),
            ), self.arc_prior)
        self.single_reconstruction = TrustedSingleReconstruction(
            ValidatedWidth(WidthConfig(
                initialization_frames=int(self.get_parameter('width.initialization_frames').value),
                ema_alpha=float(self.get_parameter('width.ema_alpha').value),
                update_gate=float(self.get_parameter('width.update_gate').value),
            )),
            curvature_support=float(self.get_parameter('single.curvature_support').value),
            persistence_span=float(self.get_parameter('single.persistence_span').value),
            min_curvature_samples=int(self.get_parameter('single.min_curvature_samples').value),
        )
        fraction = float(np.mean(self.frontend.bev_valid_map))
        origin = corrected[:3, 3]
        self.get_logger().info(
            '[BEV V2 MAPPING] '
            f'height_correction_z={height_correction_z} '
            f'pitch_offset_deg={pitch_offset_deg} '
            f'camera_origin={origin[0]:.4f},{origin[1]:.4f},{origin[2]:.4f} '
            f'valid_fraction={fraction:.6f}'
        )

    def _process_image(self, message, transform_camera,
                       transform_odom_base=None):
        try:
            if transform_camera is None:
                raise ValueError('exact-stamp camera transform is required')
            started = time.perf_counter()
            corrected = apply_projection_corrections(
                self._transform_matrix(transform_camera),
                camera_height_correction_z=float(self.get_parameter(
                    'sim_geometry.camera_height_correction_z').value),
                pitch_offset_deg=float(self.get_parameter(
                    'projection.pitch_offset_deg').value),
                pitch_correction_frame=str(self.get_parameter(
                    'projection.pitch_correction_frame').value))
            self.frontend.update_projector(MetricGroundProjector(
                self.camera, self.grid, corrected,
                ground_z=float(self.get_parameter('ground_z').value)))
            map_seconds = time.perf_counter()-started
            self.dynamic_map_times.append(map_seconds)
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
            output = self.frontend.process(image)
            segmented = self.segmentation.process(
                output.bev, self.frontend.bev_valid_map,
                include_white_stages=self.white_continuity_capture is not None,
            )
            stamp_seconds = (float(message.header.stamp.sec)
                             + 1e-9*float(message.header.stamp.nanosec))
            pre_states = (self.identity.left_state, self.identity.right_state)
            pre_center = self.identity.trusted_center
            identity = self.identity.process(
                segmented.component_frame.candidates, stamp_seconds,
                transform_odom_base)
            single = self.single_reconstruction.process(
                identity, self.identity, stamp_seconds)
            if self.pan_association_characterizer is not None:
                try:
                    self.pan_association_characterizer.capture(
                        stamp_seconds, self.current_pan,
                        segmented.component_frame.candidates, pre_states,
                        pre_center, transform_odom_base, identity, self.identity)
                except Exception as diagnostic_error:
                    if self.frame_count == 0 or self.frame_count % 30 == 0:
                        self.get_logger().warning(
                            '[BEV V2 PAN ASSOCIATION CHARACTERIZATION] '
                            f'unavailable={diagnostic_error!r} '
                            'production_unchanged=True')
            if self.side_gate_characterizer is not None:
                try:
                    self.side_gate_characterizer.capture(
                        stamp_seconds, self.current_pan,
                        segmented.component_frame.candidates, pre_states,
                        pre_center, self.identity, identity)
                except Exception as diagnostic_error:
                    if self.frame_count == 0 or self.frame_count % 30 == 0:
                        self.get_logger().warning(
                            '[BEV V2 SIDE GATE CHARACTERIZATION] '
                            f'unavailable={diagnostic_error!r} '
                            'production_unchanged=True')
            if self.arc_support_capture is not None:
                captured_support = self.arc_support_capture.capture(
                    message.header.stamp,
                    segmented.component_frame.candidates, identity,
                    self.arc_prior, transform_odom_base)
                if captured_support and self.frame_count % 15 == 0:
                    self.get_logger().info(
                        '[BEV V2 ARC SUPPORT] '
                        f'scene={self.arc_support_capture.scene} '
                        f'files={len(captured_support)}')
            if self.geometry_capture is not None:
                captured = self.geometry_capture.capture(
                    message.header.stamp, segmented.component_frame,
                    identity, self.identity, single)
                if captured is not None and self.frame_count % 15 == 0:
                    self.get_logger().info(
                        f'[BEV V2 CAPTURE] scene={self.geometry_capture.scene} '
                        f'file={captured.name}')
            both = (identity.center_result if identity.both_accepted
                    else self.both_geometry.process(()))
            both_overlay = self._draw_both_overlay(segmented.overlay, both)
            identity_overlay = self._draw_identity_overlay(
                segmented.overlay, identity)
            single_overlay = self._draw_single_overlay(identity_overlay, single)
            if self.white_continuity_capture is not None:
                captured = self.white_continuity_capture.capture(
                    message.header.stamp, image, output.undistorted, output.bev,
                    output.validity_mask, segmented.white_stages,
                    segmented.overlay)
                if captured is not None and self.frame_count % 30 == 0:
                    self.get_logger().info(
                        '[BEV V2 WHITE CONTINUITY] '
                        f'file={captured.name}')
        except Exception as error:
            self.get_logger().error(f'BEV V2 frame failed: {error!r}')
            self.ready_pub.publish(Bool(data=False))
            return
        if self.arc_shadow is not None:
            try:
                self._run_arc_shadow(
                    message, segmented.component_frame.candidates,
                    identity, single)
            except Exception as error:
                # Shadow failure cannot reject or alter the production frame.
                if self.frame_count == 0 or self.frame_count % 30 == 0:
                    self.get_logger().warning(
                        f'[BEV V2 ARC SHADOW] unavailable={error!r} '
                        'production_unchanged=True')
        self._publish_image(
            self.undistorted_pub, output.undistorted, message, 'bgr8', CAMERA_FRAME
        )
        self._publish_image(self.bev_pub, output.bev, message, 'bgr8', VEHICLE_FRAME)
        self._publish_image(
            self.validity_pub, output.validity_mask, message, 'mono8', VEHICLE_FRAME
        )
        self._publish_image(
            self.white_mask_pub, segmented.white_mask, message, 'mono8', VEHICLE_FRAME
        )
        self._publish_image(
            self.orange_mask_pub, segmented.orange_mask, message, 'mono8', VEHICLE_FRAME
        )
        if segmented.white_stages is not None:
            self._publish_image(self.white_raw_pub, segmented.white_stages.raw,
                                message, 'mono8', VEHICLE_FRAME)
            self._publish_image(
                self.white_post_validity_pub,
                segmented.white_stages.post_validity,
                message, 'mono8', VEHICLE_FRAME)
            self._publish_image(
                self.white_after_open_pub, segmented.white_stages.after_open,
                message, 'mono8', VEHICLE_FRAME)
            self._publish_image(
                self.white_after_close_pub, segmented.white_stages.after_close,
                message, 'mono8', VEHICLE_FRAME)
        self._publish_image(
            self.component_overlay_pub, segmented.overlay, message, 'bgr8', VEHICLE_FRAME
        )
        self._publish_image(
            self.both_overlay_pub, both_overlay, message, 'bgr8', VEHICLE_FRAME
        )
        self._publish_image(
            self.identity_overlay_pub, identity_overlay, message, 'bgr8', VEHICLE_FRAME
        )
        self._publish_image(
            self.single_overlay_pub, single_overlay, message, 'bgr8', VEHICLE_FRAME
        )
        fraction = float(np.mean(output.validity_mask > 0))
        self.ready_pub.publish(Bool(data=True))
        self.dynamic_ready_pub.publish(Bool(data=True))
        self.valid_fraction_pub.publish(Float32(data=fraction))
        self.candidate_count_pub.publish(UInt32(data=len(segmented.component_frame.candidates)))
        self.usable_count_pub.publish(UInt32(data=len(both.usable_candidates)))
        self.both_valid_pub.publish(Bool(data=single.center is not None))
        self.identity_initialized_pub.publish(
            Bool(data=identity.identity_initialized))
        self.initialization_streak_pub.publish(
            UInt32(data=identity.initialization_streak))
        center_count = 0 if single.center is None else len(single.center.points)
        self.center_point_count_pub.publish(UInt32(data=center_count))
        if both.center_path is not None:
            self.observed_width_pub.publish(Float32(data=both.center_path.width_median))
        self.observation_mode_pub.publish(String(data=single.observation_mode))
        self.provenance_pub.publish(String(
            data=f'LEFT={single.left_provenance},RIGHT={single.right_provenance},CENTER={single.center_provenance}'))
        self.width_update_allowed_pub.publish(
            Bool(data=single.width_update_allowed))
        if single.trusted_width is not None:
            self.trusted_width_pub.publish(Float32(data=single.trusted_width))
        self.frame_count += 1
        stamp = (f'{message.header.stamp.sec}.'
                 f'{message.header.stamp.nanosec:09d}')
        log_component_details = self.frame_count == 1 or self.frame_count % 15 == 0
        for observation in segmented.component_frame.observations:
            if not log_component_details:
                continue
            metadata = observation.metadata
            candidate = observation.candidate
            if candidate is None:
                self.get_logger().info(
                    '[BEV V2 CANDIDATE] '
                    f'stamp={stamp} color={metadata.color} '
                    f'candidate_id={metadata.component_id} area={metadata.area_pixels} '
                    f'valid_overlap={metadata.valid_overlap:.3f} geometry_valid=False '
                    f'canonicalizable=False reason={metadata.rejection_reason}'
                )
                continue
            x_diff = np.diff(candidate.raw_ordered_points[:, 0])
            non_x_monotonic = bool(np.any(x_diff > 1e-9) and np.any(x_diff < -1e-9))
            self.get_logger().info(
                '[BEV V2 CANDIDATE] '
                f'stamp={stamp} color={metadata.color} '
                f'candidate_id={metadata.component_id} area={metadata.area_pixels} '
                f'valid_overlap={metadata.valid_overlap:.3f} geometry_valid=True '
                f'canonicalizable=True reason=valid '
                f'raw_points={candidate.raw_point_count} '
                f'canonical_points={candidate.canonical_point_count} '
                f'support_length={candidate.support_length:.6f} '
                f'near={candidate.near_endpoint[0]:.4f},{candidate.near_endpoint[1]:.4f} '
                f'far={candidate.far_endpoint[0]:.4f},{candidate.far_endpoint[1]:.4f} '
                f'raw_spacing={candidate.raw_spacing_min:.6f},'
                f'{candidate.raw_spacing_median:.6f},{candidate.raw_spacing_max:.6f} '
                f'canonical_spacing={candidate.canonical_spacing:.6f} '
                f'non_x_monotonic={non_x_monotonic}'
            )
        if log_component_details:
            self._log_both(stamp, segmented.component_frame, both)
            self._log_identity(stamp, identity)
            self._log_single(stamp, single)
        if self.frame_count == 1 or self.frame_count % 30 == 0:
            waits = np.asarray(self.camera_tf_waits[-300:], dtype=float)
            maps = np.asarray(self.dynamic_map_times[-300:], dtype=float)
            self.get_logger().info(
                f'[BEV V2 DEBUG] stamp={message.header.stamp.sec}.'
                f'{message.header.stamp.nanosec:09d} frames={self.frame_count} '
                f'shape={output.bev.shape[1]}x{output.bev.shape[0]} '
                f'valid_fraction={fraction:.6f} '
                f'components={len(segmented.component_frame.observations)} '
                f'candidates={len(segmented.component_frame.candidates)}'
                f' camera_tf_immediate={self.camera_tf_immediate}'
                f' camera_tf_eventual={self.camera_tf_eventual}'
                f' camera_tf_timeout={self.camera_tf_timeout}'
                f' pending_replaced={self.pending.replaced}'
                f' tf_wait_ms_median={0.0 if waits.size == 0 else 1e3*np.median(waits):.3f}'
                f' tf_wait_ms_p95={0.0 if waits.size == 0 else 1e3*np.percentile(waits, 95):.3f}'
                f' map_ms_median={0.0 if maps.size == 0 else 1e3*np.median(maps):.3f}'
                f' map_ms_p95={0.0 if maps.size == 0 else 1e3*np.percentile(maps, 95):.3f}'
            )

    def _lookup_odom_transform(self, source_stamp):
        """Nonblocking exact-stamp lookup used by the bounded pending queue."""
        stamp = Time.from_msg(source_stamp)
        try:
            transform = self.tf_buffer.lookup_transform(
                'odom', VEHICLE_FRAME, stamp,
                timeout=Duration(seconds=0.0))
        except Exception:
            return None
        matrix = self._transform_matrix(transform)
        return np.asarray([
            [matrix[0, 0], matrix[0, 1], matrix[0, 3]],
            [matrix[1, 0], matrix[1, 1], matrix[1, 3]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)

    def _run_arc_shadow(self, message, candidates, identity, single):
        timestamp = float(message.header.stamp.sec)+1e-9*float(
            message.header.stamp.nanosec)
        if (self.arc_shadow_pending
                and timestamp < self.arc_shadow_pending[-1][0]):
            self.arc_shadow_pending.clear()
        self.arc_shadow_pending.append((
            timestamp, message.header.stamp, candidates, identity, single))
        while self.arc_shadow_pending:
            pending = self.arc_shadow_pending[0]
            if timestamp-pending[0] > 1.0:
                self.arc_shadow_pending.pop(0)
                continue
            if not self._process_pending_arc_shadow(*pending[1:]):
                break
            self.arc_shadow_pending.pop(0)

    def _process_pending_arc_shadow(self, source_stamp, candidates, identity, single):
        stamp = Time.from_msg(source_stamp)
        ready, debug = self.tf_buffer.can_transform(
            'odom', VEHICLE_FRAME, stamp,
            timeout=Duration(seconds=0.0), return_debug_tuple=True)
        if not ready:
            return False
        transform = self.tf_buffer.lookup_transform(
            'odom', VEHICLE_FRAME, stamp, timeout=Duration(seconds=0.0))
        matrix = self._transform_matrix(transform)
        transform_odom_base = np.array([
            [matrix[0, 0], matrix[0, 1], matrix[0, 3]],
            [matrix[1, 0], matrix[1, 1], matrix[1, 3]],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        timestamp = float(source_stamp.sec)+1e-9*float(source_stamp.nanosec)
        outputs = self.arc_shadow.process(
            timestamp, transform_odom_base,
            self._arc_shadow_associations(identity, candidates), candidates)
        captured = self.arc_shadow_capture.write(
            source_stamp, transform_odom_base, outputs,
            identity, single, candidates)
        if self.frame_count == 0 or self.frame_count % 15 == 0:
            def details(value):
                best = value.best_comparison
                return (
                    f'category={value.category} streak={value.confirm_streak} '
                    f'age={value.age_frames}/{value.age_seconds:.3f} '
                    f'radius={None if value.memory is None else value.memory.radius} '
                    f'candidate={None if best is None else f"{best.color}:{best.candidate_id}"} '
                    f'radial={None if best is None else best.radial_median} '
                    f'tangent={None if best is None else best.tangent_error_median} '
                    f'covered={None if best is None else best.covered_support} '
                    f'margin={value.identity_margin}')
            self.get_logger().info(
                '[BEV V2 ARC SHADOW] '
                f'stamp={source_stamp.sec}.{source_stamp.nanosec:09d} '
                f'scene={self.arc_shadow_capture.scene} '
                f'production_mode={single.observation_mode} '
                f'LEFT[{details(outputs["LEFT"])}] '
                f'RIGHT[{details(outputs["RIGHT"])}] file={captured.name}')
        return True

    def _arc_shadow_associations(self, identity, candidates):
        associations = {'LEFT': identity.left, 'RIGHT': identity.right}
        if all(value is not None and value.valid and value.accepted is not None
               for value in associations.values()):
            return associations
        # Shadow-only fallback: a valid current-frame physical pair supplies
        # side labels for arc acquisition without changing trusted identity.
        frame_local = self.both_geometry.process(candidates)
        pair = frame_local.selected_pair
        if pair is None:
            return associations
        for side, candidate in (('LEFT', pair.left), ('RIGHT', pair.right)):
            value = associations[side]
            if value is None or not value.valid or value.accepted is None:
                associations[side] = SimpleNamespace(
                    side=side, valid=True, candidate=candidate,
                    accepted=candidate, shadow_source='frame_local_both')
        return associations

    def _metric_pixels(self, points):
        col, row = self.grid.metric_to_pixel(points[:, 0], points[:, 1])
        return np.rint(np.column_stack((col, row))).astype(np.int32)

    def _draw_both_overlay(self, source, result):
        overlay = source.copy()
        selected = result.selected_pair
        selected_candidates = set()
        if selected is not None:
            selected_candidates = {id(selected.left), id(selected.right)}
        for usable in result.usable_boundaries:
            if not usable.usable or id(usable.candidate) in selected_candidates:
                continue
            cv2.polylines(overlay, [self._metric_pixels(
                usable.candidate.canonical_points)], False, (160, 160, 160), 1)
        if selected is None:
            return overlay
        cv2.polylines(overlay, [self._metric_pixels(
            selected.left.canonical_points)], False, (255, 80, 0), 3)
        cv2.polylines(overlay, [self._metric_pixels(
            selected.right.canonical_points)], False, (0, 220, 255), 3)
        corr = selected.correspondence
        for first, second in zip(corr.first_points[::2], corr.second_points[::2]):
            pixels = self._metric_pixels(np.vstack((first, second)))
            cv2.line(overlay, tuple(pixels[0]), tuple(pixels[1]), (0, 255, 0), 1)
        cv2.polylines(overlay, [self._metric_pixels(
            selected.center.points)], False, (0, 0, 255), 3)
        return overlay

    def _draw_identity_overlay(self, source, result):
        overlay = source.copy()
        for state, color in ((self.identity.left_state, (160, 80, 0)),
                             (self.identity.right_state, (0, 160, 160))):
            if state.geometry is not None:
                cv2.polylines(overlay, [self._metric_pixels(
                    state.geometry.canonical_points)], False, color, 1)
        for association, color in ((result.left, (255, 80, 0)),
                                   (result.right, (0, 220, 255))):
            if association is None or association.accepted is None:
                continue
            cv2.polylines(overlay, [self._metric_pixels(
                association.candidate.canonical_points)], False, (80, 80, 80), 2)
            cv2.polylines(overlay, [self._metric_pixels(
                association.accepted.canonical_points)], False, color, 3)
        if result.both_accepted and result.center_result.center_path is not None:
            center = result.center_result.center_path
            cv2.polylines(overlay, [self._metric_pixels(
                center.points)], False, (0, 0, 255), 3)
        return overlay

    def _draw_single_overlay(self, source, result):
        overlay = source.copy()
        if result.missing is not None:
            cv2.polylines(overlay, [self._metric_pixels(
                result.missing.points)], False, (255, 0, 255), 2)
        if result.center is not None:
            center_color = ((0, 0, 255) if result.center_provenance == 'BOTH_CENTER'
                            else (0, 128, 255))
            cv2.polylines(overlay, [self._metric_pixels(
                result.center.points)], False, center_color, 3)
        return overlay

    def _log_single(self, stamp, result):
        center_support = 0.0 if result.center is None else result.center.support_length
        missing_support = 0.0 if result.missing is None else result.missing.support_length
        self.get_logger().info(
            '[BEV V2 SINGLE] '
            f'stamp={stamp} observation_mode={result.observation_mode} '
            f'left_provenance={result.left_provenance} '
            f'right_provenance={result.right_provenance} '
            f'center_provenance={result.center_provenance} '
            f'observed_side={result.observed_side} '
            f'trusted_width={result.trusted_width} '
            f'width_update_allowed={result.width_update_allowed} '
            f'normal_sign={result.normal_sign} '
            f'normal_sign_source={result.normal_sign_source} '
            f'center_valid={result.center is not None} '
            f'center_reason={result.center_safety.reason} '
            f'missing_valid={result.missing is not None} '
            f'missing_reason={result.missing_safety.reason} '
            f'center_support={center_support:.6f} '
            f'missing_support={missing_support:.6f} reason={result.reason}'
        )

    def _log_identity(self, stamp, result):
        def details(value):
            if value is None:
                return 'candidate=none association_valid=False reason=no_association'
            candidate = value.candidate
            return (
                f'candidate={candidate.color}:{candidate.component_id} '
                f'association_attempted={value.attempted} '
                f'association_valid={value.valid} reason={value.reason} '
                f'mean_distance={value.mean_distance} '
                f'overlap_support={value.overlap_support:.6f} '
                f'tangent_consistency={value.tangent_consistency:.6f} '
                f'interval={value.interval_start_s},{value.interval_end_s} '
                f'accepted_support={value.accepted_support:.6f} '
                f'rejected_tail_support={value.rejected_tail_support:.6f} '
                f'reference_support={value.reference_support:.6f} '
                f'sliding_association_used={value.sliding_association_used} '
                f'continuation_gap={value.continuation_gap} '
                f'side_state={value.side_state} '
                f'signed_lateral_median={value.signed_lateral_median} '
                f'expected_signed_lateral={value.expected_signed_lateral} '
                f'lateral_residual={value.lateral_residual} '
                f'side_consistent_support={value.side_consistent_support:.6f} '
                f'opposite_side_support={value.opposite_side_support:.6f} '
                f'center_crossing={value.center_crossing} '
                f'reference_update_allowed={value.reference_update_allowed} '
                f'side_reason={value.side_reason} '
                f'association_source={value.association_source} '
                f'arc_nearest_error={value.arc_nearest_error} '
                f'arc_radial_error={value.arc_radial_error} '
                f'arc_tangent_error={value.arc_tangent_error} '
                f'arc_compatible_support={value.arc_compatible_support:.6f} '
                f'arc_age_seconds={value.arc_age_seconds}'
            )
        left_reference = (self.identity.left_state.association_reference
                          or self.identity.left_state.geometry)
        right_reference = (self.identity.right_state.association_reference
                           or self.identity.right_state.geometry)
        reference_details = (
            f'LEFT_REFERENCE[long_term_support={self.identity.left_state.physical_support:.6f} '
            f'short_term_support={0.0 if left_reference is None else left_reference.support_length:.6f}] '
            f'RIGHT_REFERENCE[long_term_support={self.identity.right_state.physical_support:.6f} '
            f'short_term_support={0.0 if right_reference is None else right_reference.support_length:.6f}]'
        )
        conflict = result.conflict_resolution
        if conflict is None:
            conflict_details = 'conflict_resolution=none'
        else:
            def evidence(value):
                if value is None:
                    return 'valid=False reason=unavailable'
                return (
                    f'valid={value.valid} reason={value.reason} '
                    f'side_consistency={value.side_consistency:.6f} '
                    f'lateral_residual={value.lateral_residual:.6f} '
                    f'tangent_consistency={value.tangent_consistency:.6f} '
                    f'support={value.support_length:.6f} '
                    f'signed_lateral_median={value.signed_lateral_median} '
                    f'expected_signed_lateral={value.expected_signed_lateral} '
                    f'side_consistent_support={value.side_consistent_support:.6f} '
                    f'opposite_side_support={value.opposite_side_support:.6f} '
                    f'center_crossing={value.center_crossing}'
                )
            candidate = conflict.candidate
            conflict_details = (
                f'conflict_component={candidate.color}:{candidate.component_id} '
                f'conflict_center_reference_valid={conflict.center_reference_valid} '
                f'conflict_result={conflict.result} '
                f'conflict_reason={conflict.reason} '
                f'conflict_left_interval={conflict.left.interval_start_s},'
                f'{conflict.left.interval_end_s} '
                f'conflict_right_interval={conflict.right.interval_start_s},'
                f'{conflict.right.interval_end_s} '
                f'CONFLICT_LEFT[{evidence(conflict.left_evidence)}] '
                f'CONFLICT_RIGHT[{evidence(conflict.right_evidence)}]'
            )
        self.get_logger().info(
            '[BEV V2 IDENTITY] '
            f'stamp={stamp} initialized={result.identity_initialized} '
            f'initialization_streak={result.initialization_streak} '
            f'identity_conflict={result.identity_conflict} '
            f'both_accepted={result.both_accepted} '
            f'trusted_update_allowed={result.trusted_update_allowed} '
            f'center_valid={result.center_result.center_path is not None and result.both_accepted} '
            f'reason={result.reason} LEFT[{details(result.left)}] '
            f'RIGHT[{details(result.right)}] {conflict_details} '
            f'{reference_details}'
        )
        if self.arc_prior is not None:
            def arc_details(side):
                memory = self.arc_prior.memory[side]
                streak = len(self.arc_prior.pending[side])
                hypothesis = self.arc_prior.last_hypothesis[side]
                fit = ('fit=none' if hypothesis is None else
                       f'fit_reason={hypothesis.reason} '
                       f'fit_ratio={hypothesis.inlier_ratio:.6f} '
                       f'fit_rms={hypothesis.rms:.6f} '
                       f'fit_radius={hypothesis.radius:.6f} '
                       f'fit_span={hypothesis.angular_span:.6f} '
                       f'fit_support={hypothesis.contiguous_support:.6f}')
                if memory is None:
                    return f'valid=False streak={streak} {fit}'
                return (
                    f'valid=True streak={streak} radius={memory.radius:.6f} '
                    f'center_odom={memory.center_odom[0]:.6f},'
                    f'{memory.center_odom[1]:.6f} '
                    f'last_actual={memory.last_actual_time:.9f} {fit}')
            counts = self.arc_prior.counts
            self.get_logger().info(
                '[BEV V2 ARC PRIOR] '
                f'stamp={stamp} LEFT[{arc_details("LEFT")}] '
                f'RIGHT[{arc_details("RIGHT")}] '
                f'DIRECT_SUCCESS={counts["DIRECT_SUCCESS"]} '
                f'SLIDING_SUCCESS={counts["SLIDING_SUCCESS"]} '
                f'ARC_RESCUE_SUCCESS={counts["ARC_RESCUE_SUCCESS"]} '
                f'ARC_RESCUE_REJECTED={counts["ARC_RESCUE_REJECTED"]} '
                f'NO_ASSOCIATION={counts["NO_ASSOCIATION"]}')

    def _log_both(self, stamp, component_frame, result):
        reject_reasons = {}
        for pair in result.pair_evaluations:
            if not pair.valid:
                reject_reasons[pair.reason] = reject_reasons.get(pair.reason, 0) + 1
        rejected = ','.join(
            f'{reason}:{count}' for reason, count in sorted(reject_reasons.items())
        ) or 'none'
        if result.selected_pair is None:
            self.get_logger().info(
                '[BEV V2 BOTH] '
                f'stamp={stamp} total_candidates={len(component_frame.candidates)} '
                f'usable_candidates={len(result.usable_candidates)} '
                f'pair_candidates={len(result.pair_evaluations)} selected=none '
                f'reason={result.reason} pair_reject_reasons={rejected}'
            )
            return
        pair, center = result.selected_pair, result.center_path
        self.get_logger().info(
            '[BEV V2 BOTH] '
            f'stamp={stamp} total_candidates={len(component_frame.candidates)} '
            f'usable_candidates={len(result.usable_candidates)} '
            f'pair_candidates={len(result.pair_evaluations)} '
            f'selected={pair.first.color}:{pair.first.component_id},'
            f'{pair.second.color}:{pair.second.component_id} '
            f'left={center.left_color}:{center.left_component_id} '
            f'right={center.right_color}:{center.right_component_id} '
            f'left_support={pair.left.support_length:.6f} '
            f'right_support={pair.right.support_length:.6f} '
            f'pair_overlap_support={center.pair_overlap_support:.6f} '
            f'correspondences={center.correspondence_count} '
            f'width={center.width_min:.6f},{center.width_median:.6f},'
            f'{center.width_max:.6f} center_points={len(center.points)} '
            f'center_support={center.support_length:.6f} reason=valid '
            f'pair_reject_reasons={rejected}'
        )

    def _publish_image(self, publisher, image, source, encoding, frame):
        message = self.bridge.cv2_to_imgmsg(image, encoding=encoding)
        message.header.stamp = source.header.stamp
        message.header.frame_id = frame
        publisher.publish(message)


def main(args=None):
    rclpy.init(args=args)
    node = BevFrontendNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        if node.geometry_capture is not None:
            node.geometry_capture.close()
        if node.arc_shadow_capture is not None:
            node.arc_shadow_capture.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
