"""Minimal V3 ROS front-end: exact TF, dynamic metric BEV, direct CENTER."""
from collections import deque
import json
import math
import time
import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import (qos_profile_sensor_data, QoSProfile,
                       ReliabilityPolicy, HistoryPolicy)
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState, LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import PointStamped, PoseStamped
from std_msgs.msg import Bool, Float64, String
from cv_bridge import CvBridge
import tf2_ros
from physicar_track_perception_v2.geometry import CameraModel, BevGrid, MetricGroundProjector, apply_projection_corrections
from physicar_track_perception_v2.frontend import BevFrontend
from physicar_track_perception_v2.components import CanonicalComponentExtractor, ComponentExtractionConfig
from physicar_track_perception_v2.segmentation import ColorComponentPipeline, HsvRange
from .geometry import OrderedPolyline
from .roles import Component, RoleConfig, classify, CENTER, LEFT, RIGHT
from .path_selector import (select_orange, select_unknown_white,
                            DIRECT_CENTER_OBSERVED)
from .proximity import validate_start
from .white_propagation import WhiteShadow, seed_from_center, propagate, LEFT as WLEFT, RIGHT as WRIGHT
from .avoidance import AvoidanceConfig, deform_path
from .circle_avoidance import (
    AVOID_LEFT, AVOID_RIGHT, CircleAvoidanceConfig,
    CircleAvoidanceEngine, CENTER as CIRCLE_CENTER, LEFT as CIRCLE_LEFT,
    RIGHT as CIRCLE_RIGHT, classify_circle, relevant_circles)
from .obstacle_tracks import (
    MultiObstacleTracker, ObstacleObservation, ObstacleTrackConfig)
from .active_obstacle import (
    ACTIVE_LOST, ActiveObstacleLifecycle, ActiveTrackView,
    PHYSICAR_ROBOT_BOUNDING_RADIUS)
from .center_hybrid import (
    CURRENT_HYBRID_ORANGE_WHITE,
    CURRENT_HYBRID_WITH_HISTORY_PREFIX,
    CenterHybridConfig,
    RecentCenterHistory,
    orient_fragment_chain,
    stitch_current_frame,
    transform_xy)
from .low_vote_recovery import (
    DETERMINISTIC_DEFAULT,
    HEADING_RECOVERY_VOTE,
    LowVoteRecoveryConfig,
    LowVoteRecoveryManager,
    RecoveryTrackView,
    NORMAL_VOTE,
    SLOW_VOTING_LOCK,
    WHITE_PROXIMITY,
    WhiteComponentView)
from .control_integration import (
    build_active_avoidance_geometry, choose_control_mode)
from .lidar_bev import (expand_bev_canvas, filter_bev_bounds,
                        scan_to_lidar_points, transform_matrix,
                        transform_points)

# 카메라는 항상 **가장 최신 프레임만** 본다.
#
# qos_profile_sensor_data 는 큐 깊이가 5다. 카메라가 30 Hz 로 밀어넣는데
# 인지가 15 Hz 로 소화하면 큐에 다섯 장이 밀리고, 그러면 인지가 보는 건
# 5/15 = 333 ms 전의 세상이다. 1.2 m/s 에서 40 cm 다. 차는 이미 지나간
# 자리를 기준으로 조향한다 -- 코너에서 늘 늦게 꺾고 직선에서 넘어섰다
# 되돌아온다.
#
# 깊이를 1로 두면 밀린 프레임을 붙잡는 대신 버린다. 처리 속도는 그대로고
# (CPU 가 내주는 만큼 나온다) 지연만 한 프레임으로 고정된다. 프레임을
# 버리는 게 손해처럼 보이지만, 어차피 늦게 처리할 프레임이라 쓸모가 없다.
#
# BEST_EFFORT 는 원래대로 둔다. RELIABLE 로 올리면 재전송을 기다리느라
# 오히려 밀린다.
NEWEST_IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


class V3Node(Node):
    def __init__(self):
        super().__init__('physicar_track_perception_v3')
        defaults={
            'camera.width':480,'camera.height':360,
            'camera.K':[201.38988018035889,0.,240.,0.,201.38988733291626,180.,0.,0.,1.],
            'camera.D':[-.045,-.0001,-.0003,-.0001,.001],
            'bev.x_min':.1,'bev.x_max':2.,'bev.y_min':-.75,
            'bev.y_max':.75,'bev.resolution':.01,'ground_z':0.,
            'sim_geometry.camera_height_correction_z':-.018,
            'projection.pitch_offset_deg':2.8,
            'tf_wait.max_pending_age':.25,'tf_wait.allow_latest':False,'tf_wait.timer_period':.02,
            'path_proximity.max_start_distance':.60,
            'center_hybrid.enabled':True,
            'center_hybrid.join_gap':.30,
            'center_hybrid.tangent_angle_limit':.75,
            'center_history.enabled':True,
            'center_history.max_age':.50,
            'center_history.max_entries':8,
            'white.track_width':.70,'white.expected_half_width':.37,
            'white.half_width_tolerance':.10,
            'white.reference_fallback_enabled':True,
            'lidar.scan_topic':'/scan','lidar.fixed_frame':'odom',
            'lidar.pair_slop':.03,'lidar.tf_timeout':.10,
            'lidar.overlay_radius_px':2,'lidar.overlay_visual_stride':3,
            'lidar.overlay_x_min':-.5,'lidar.overlay_x_max':4.,
            'lidar.overlay_y_min':-2.,'lidar.overlay_y_max':2.,
            'lidar.path_overlay_color_bgr':[255,255,0],
            # Stage 5.1 baseline remains available for shadow comparison.
            'avoidance.shadow_enabled':True,
            'avoidance.path_near_distance':.20,
            'avoidance.representative_window':.30,
            'avoidance.influence_radius':.60,'avoidance.max_offset':.25,
            'avoidance.center_deadband':.03,'avoidance.tangent_window':.10,
            'avoidance.resample_spacing':.05,
            'avoidance.path_color_bgr':[255,255,255],
            'avoidance.obstacle_color_bgr':[0,165,255],
            # Stage 5.1R component/circle/lane-vote shadow parameters.
            'avoidance_circle.enabled':True,
            'lidar_component.gap_threshold':.12,
            'lidar_component.max_obstacle_support':.70,
            'lidar_component.min_circle_points':3,
            'avoidance_circle.min_radius':.02,
            'avoidance_circle.max_radius':.40,
            'avoidance_circle.max_fit_residual':.05,
            'avoidance_circle.path_near_distance':.20,
            'avoidance.direction_freeze_distance':1.5,
            'avoidance.component_continuity_distance':.45,
            'avoidance.default_side':'LEFT',
            'avoidance.safety_margin':.20,
            'avoidance.additional_clearance':.05,
            'avoidance.approach_length':.80,
            'avoidance.return_length':.80,
            'avoidance.circle_tangent_window':.20,
            'avoidance.circle_resample_spacing':.05,
            'avoidance.termination_rear_x':0.0,
            # Stage 5.2A independent multi-obstacle vote/lock shadow.
            'obstacle_track.enabled':True,
            'obstacle_track.association_distance':.12,
            'obstacle_track.retention_age':.50,
            'obstacle_track.max_voting_tracks':2,
            # Stage 5.2B raw-component active lifecycle shadow.
            'active_lifecycle.enabled':True,
            'active_lifecycle.robot_bounding_radius':(
                PHYSICAR_ROBOT_BOUNDING_RADIUS),
            'active_lifecycle.lost_release_radius_multiplier':1.2,
            # Stage 5.2C low-vote recovery shadow/request parameters.
            'avoidance_recovery.enabled':True,
            'avoidance_recovery.slow_voting_max_frames':4,
            'avoidance_recovery.heading_max_frames':5,
            'avoidance_recovery.max_heading_correction_rad':.25,
            'avoidance_recovery.emergency_distance':.45,
        }
        for key,value in defaults.items(): self.declare_parameter(key,value)
        p=lambda k: self.get_parameter(k).value
        self.camera=CameraModel(np.asarray(p('camera.K')).reshape(3,3),np.asarray(p('camera.D')),int(p('camera.width')),int(p('camera.height')))
        self.grid=BevGrid(float(p('bev.x_min')),float(p('bev.x_max')),float(p('bev.y_min')),float(p('bev.y_max')),float(p('bev.resolution')))
        self.lidar_overlay_grid=BevGrid(float(p('lidar.overlay_x_min')),float(p('lidar.overlay_x_max')),float(p('lidar.overlay_y_min')),float(p('lidar.overlay_y_max')),float(p('bev.resolution')))
        self.path_lidar_color=tuple(int(value) for value in p('lidar.path_overlay_color_bgr'))
        if (len(self.path_lidar_color) != 3
                or any(value < 0 or value > 255
                       for value in self.path_lidar_color)):
            raise ValueError('lidar.path_overlay_color_bgr must contain three bytes')
        self.avoidance_config=AvoidanceConfig(float(p('avoidance.path_near_distance')),float(p('avoidance.representative_window')),float(p('avoidance.influence_radius')),float(p('avoidance.max_offset')),float(p('avoidance.center_deadband')),float(p('avoidance.tangent_window')),float(p('avoidance.resample_spacing')))
        default_side = str(p('avoidance.default_side')).upper()
        default_avoidance_side = (AVOID_LEFT if default_side == 'LEFT'
                                  else AVOID_RIGHT if default_side == 'RIGHT'
                                  else default_side)
        self.circle_avoidance_config=CircleAvoidanceConfig(
            component_gap=float(p('lidar_component.gap_threshold')),
            max_obstacle_support=float(p('lidar_component.max_obstacle_support')),
            min_circle_points=int(p('lidar_component.min_circle_points')),
            min_circle_radius=float(p('avoidance_circle.min_radius')),
            max_circle_radius=float(p('avoidance_circle.max_radius')),
            max_circle_residual=float(p('avoidance_circle.max_fit_residual')),
            path_near_distance=float(p('avoidance_circle.path_near_distance')),
            direction_freeze_distance=float(p('avoidance.direction_freeze_distance')),
            component_continuity_distance=float(p('avoidance.component_continuity_distance')),
            default_avoidance_side=default_avoidance_side,
            safety_margin=float(p('avoidance.safety_margin')),
            additional_clearance=float(p('avoidance.additional_clearance')),
            approach_length=float(p('avoidance.approach_length')),
            return_length=float(p('avoidance.return_length')),
            tangent_window=float(p('avoidance.circle_tangent_window')),
            resample_spacing=float(p('avoidance.circle_resample_spacing')),
            termination_rear_x=float(p('avoidance.termination_rear_x')))
        self.circle_avoidance=CircleAvoidanceEngine(
            self.circle_avoidance_config)
        self.obstacle_track_config=ObstacleTrackConfig(
            association_distance=float(
                p('obstacle_track.association_distance')),
            retention_age=float(p('obstacle_track.retention_age')),
            direction_freeze_distance=float(
                p('avoidance.direction_freeze_distance')),
            default_avoidance_side=default_avoidance_side,
            max_voting_tracks=int(p('obstacle_track.max_voting_tracks')))
        self.multi_obstacle_tracker=MultiObstacleTracker(
            self.obstacle_track_config)
        self.active_obstacle_lifecycle=ActiveObstacleLifecycle(
            robot_radius=float(p('active_lifecycle.robot_bounding_radius')),
            lost_release_radius_multiplier=float(
                p('active_lifecycle.lost_release_radius_multiplier')),
            activation_distance=float(
                p('avoidance.direction_freeze_distance')))
        self.low_vote_recovery_config=LowVoteRecoveryConfig(
            freeze_distance=float(p('avoidance.direction_freeze_distance')),
            slow_voting_max_frames=int(
                p('avoidance_recovery.slow_voting_max_frames')),
            heading_recovery_max_frames=int(
                p('avoidance_recovery.heading_max_frames')),
            max_heading_correction=float(
                p('avoidance_recovery.max_heading_correction_rad')),
            emergency_distance=float(
                p('avoidance_recovery.emergency_distance')),
            center_start_distance=float(
                p('path_proximity.max_start_distance')),
            default_avoidance_side=default_avoidance_side)
        self.low_vote_recovery=LowVoteRecoveryManager(
            self.low_vote_recovery_config)
        self.center_hybrid_config=CenterHybridConfig(
            max_start_distance=float(p('path_proximity.max_start_distance')),
            join_gap=float(p('center_hybrid.join_gap')),
            tangent_angle_limit=float(
                p('center_hybrid.tangent_angle_limit')),
            history_max_age=float(p('center_history.max_age')),
            history_max_entries=int(p('center_history.max_entries')))
        self.center_history=RecentCenterHistory(self.center_hybrid_config)
        self.avoidance_path_color=tuple(int(value) for value in p('avoidance.path_color_bgr'))
        self.avoidance_obstacle_color=tuple(int(value) for value in p('avoidance.obstacle_color_bgr'))
        for name,color in (('avoidance.path_color_bgr',self.avoidance_path_color),('avoidance.obstacle_color_bgr',self.avoidance_obstacle_color)):
            if len(color) != 3 or any(value < 0 or value > 255 for value in color):
                raise ValueError(f'{name} must contain three bytes')
        self.frontend=None; self.stage={}; self.bridge=CvBridge(); self.pending=[]; self.pending_replaced=0; self.tfbuf=tf2_ros.Buffer(cache_time=Duration(seconds=10)); tf2_ros.TransformListener(self.tfbuf,self,spin_thread=True)
        self.frame_count = 0
        self._last_stats_log = 0.0
        self.stats = {k: 0 for k in ('images_received','immediate_tf_success','pending_enqueued','pending_retry_attempts','pending_eventual_success','pending_timeout','pending_replaced','frames_processed','bev_published','orange_processed','path_overlay_published','duplicate_processed','lidar_scans_received','lidar_no_pair','lidar_tf_success','lidar_tf_failure','lidar_tf_wait','lidar_pending_replaced','lidar_overlay_published','path_lidar_overlay_published','avoidance_published','avoidance_active','multi_track_published')}
        self.processed_stamps = set()
        self.previous_white = None
        self.boundary_hold_count = 0
        self.lidar_scans = deque(maxlen=3)
        self.lidar_pending = []
        self.last_lidar_diagnostic = None
        self.last_avoidance_result = None
        self.last_multi_track_result = None
        self.last_active_lifecycle_result = None
        self.last_low_vote_recovery_result = None
        self.last_low_vote_recovery_track = None
        self.last_control_path = np.empty((0, 2), dtype=np.float64)
        self.last_control_stamp = None
        self.last_control_diagnostic = {
            'schema_version':1,'controller_mode':'STOP',
            'reason':'NOT_PROCESSED','stamp_ns':None,
            'perception_ready':False}
        self.last_legacy_avoidance_result = None
        self.last_avoidance_reference = np.empty((0, 2), dtype=np.float64)
        self.last_avoidance_diagnostic = {'active':False,'reason':'NOT_PROCESSED'}
        self.last_multi_track_diagnostic = {'reason':'NOT_PROCESSED'}
        self.last_active_lifecycle_diagnostic = {'reason':'NOT_PROCESSED'}
        self.last_low_vote_recovery_diagnostic = {'reason':'NOT_PROCESSED'}
        self.last_center_hybrid_diagnostic = {'reason':'NOT_PROCESSED'}
        self.create_subscription(Image,'/camera/image_raw',self.image_cb,NEWEST_IMAGE_QOS); self.create_subscription(JointState,'/joint_states',self.joint_cb,qos_profile_sensor_data); self.create_subscription(LaserScan,str(p('lidar.scan_topic')),self.scan_cb,qos_profile_sensor_data); self.create_timer(float(p('tf_wait.timer_period')),self.retry)
        self.bev_pub=self.create_publisher(Image,'/perception_v3/debug/bev',2); self.lidar_bev_pub=self.create_publisher(Image,'/perception_v3/debug/bev_lidar_overlay',2); self.path_lidar_pub=self.create_publisher(Image,'/perception_v3/debug/path_lidar_overlay',2); self.lidar_diag_pub=self.create_publisher(String,'/perception_v3/debug/lidar_diagnostics',10); self.white_pub=self.create_publisher(Image,'/perception_v3/debug/white_mask',2); self.orange_pub=self.create_publisher(Image,'/perception_v3/debug/orange_mask',2); self.role_pub=self.create_publisher(Image,'/perception_v3/debug/role_overlay',2); self.path_pub=self.create_publisher(Image,'/perception_v3/debug/path_overlay',2); self.valid_pub=self.create_publisher(Bool,'/perception_v3/debug/path_valid',10); self.source_pub=self.create_publisher(String,'/perception_v3/debug/path_source',10); self.geometry_pub=self.create_publisher(Path,'/perception_v3/path',10); self.avoidance_path_pub=self.create_publisher(Path,'/avoidance_v3/debug/path',10); self.avoidance_active_pub=self.create_publisher(Bool,'/avoidance_v3/debug/active',10); self.avoidance_obstacle_pub=self.create_publisher(PointStamped,'/avoidance_v3/debug/obstacle_point',10); self.avoidance_offset_pub=self.create_publisher(Float64,'/avoidance_v3/debug/offset',10); self.avoidance_diag_pub=self.create_publisher(String,'/avoidance_v3/debug/diagnostics',10); self.avoidance_overlay_pub=self.create_publisher(Image,'/avoidance_v3/debug/overlay',2)
        self.avoidance_components_pub=self.create_publisher(String,'/avoidance_v3/debug/components',10)
        self.avoidance_selected_pub=self.create_publisher(String,'/avoidance_v3/debug/selected_obstacle',10)
        self.avoidance_circle_pub=self.create_publisher(String,'/avoidance_v3/debug/fitted_circle',10)
        self.avoidance_safety_pub=self.create_publisher(String,'/avoidance_v3/debug/safety_circle',10)
        self.avoidance_votes_pub=self.create_publisher(String,'/avoidance_v3/debug/lane_votes',10)
        self.avoidance_state_pub=self.create_publisher(String,'/avoidance_v3/debug/state',10)
        self.avoidance_locked_pub=self.create_publisher(Bool,'/avoidance_v3/debug/direction_locked',10)
        self.avoidance_target_pub=self.create_publisher(PointStamped,'/avoidance_v3/debug/target',10)
        self.multi_track_pub=self.create_publisher(
            String,'/avoidance_v3/debug/multi_tracks',10)
        self.multi_track_diag_pub=self.create_publisher(
            String,'/avoidance_v3/debug/multi_track_diagnostics',10)
        self.active_lifecycle_pub=self.create_publisher(
            String,'/avoidance_v3/debug/active_lifecycle',10)
        self.low_vote_recovery_pub=self.create_publisher(
            String,'/avoidance_v3/debug/low_vote_recovery',10)
        self.recovery_mode_pub=self.create_publisher(
            String,'/avoidance_v3/debug/recovery_mode',10)
        self.control_path_pub=self.create_publisher(
            Path,'/avoidance_v3/control/path',10)
        self.control_status_pub=self.create_publisher(
            String,'/avoidance_v3/control/status',10)
        self.center_hybrid_diag_pub=self.create_publisher(
            String,'/perception_v3/debug/center_hybrid_diagnostics',10)
        self.center_hybrid_overlay_pub=self.create_publisher(
            Image,'/perception_v3/debug/center_hybrid_overlay',2)
        extractor=CanonicalComponentExtractor(self.grid,ComponentExtractionConfig(min_component_area=8,min_valid_pixels=3,min_valid_overlap=.70,canonical_spacing=.05)); ranges={'WHITE':[HsvRange((0,0,170),(179,90,255))],'ORANGE':[HsvRange((5,100,100),(30,255,255))]}; self.seg=ColorComponentPipeline(ranges,3,5,extractor)
    def joint_cb(self,msg): pass
    def scan_cb(self,msg):
        self.lidar_scans.append(msg)
        self.stats['lidar_scans_received'] += 1

    @staticmethod
    def stamp_seconds(stamp):
        return float(stamp.sec) + 1e-9 * float(stamp.nanosec)

    def paired_scan(self, image_stamp):
        if not self.lidar_scans:
            return None, None
        image_time = self.stamp_seconds(image_stamp)
        scan = min(
            self.lidar_scans,
            key=lambda item: abs(self.stamp_seconds(item.header.stamp)
                                 - image_time))
        delta = self.stamp_seconds(scan.header.stamp) - image_time
        if abs(delta) > float(self.get_parameter('lidar.pair_slop').value):
            return None, delta
        return scan, delta

    def enqueue_lidar_overlay(self, bev, path_bev, reference_path,
                              current_white_components, history_reference,
                              image_stamp, scan, delta):
        entry = (bev.copy(), path_bev.copy(), reference_path.copy(),
                 tuple(current_white_components), history_reference.copy(),
                 image_stamp, scan, delta,
                 time.monotonic())
        if len(self.lidar_pending) < 2:
            self.lidar_pending.append(entry)
        else:
            self.lidar_pending[-1] = entry
            self.stats['lidar_pending_replaced'] += 1

    def render_lidar_overlay(self, bev, path_bev, reference_path, image_stamp,
                             current_white_components=(),
                             history_reference=None, scan=None, delta=None,
                             allow_enqueue=True):
        if history_reference is None:
            history_reference=np.empty((0,2),dtype=np.float64)
        overlay, _ = expand_bev_canvas(
            bev, self.grid, self.lidar_overlay_grid)
        path_overlay, _ = expand_bev_canvas(
            path_bev, self.grid, self.lidar_overlay_grid)
        avoidance_overlay = path_overlay.copy()
        if scan is None:
            scan, delta = self.paired_scan(image_stamp)
        diagnostic = {
            'image_stamp': self.stamp_seconds(image_stamp),
            'scan_stamp': None,
            'delta': delta,
            'scan_frame': None,
            'total_beams': 0,
            'valid_ranges': 0,
            'transformed_points': 0,
            'in_bounds_points': 0,
            'camera_bev_in_bounds_points': 0,
            'overlay_in_bounds_points': 0,
            'dropped_tf_points': 0,
            'tf_success': False,
        }
        if scan is None:
            self.stats['lidar_no_pair'] += 1
            self.last_lidar_diagnostic = diagnostic
            return overlay, path_overlay, avoidance_overlay
        diagnostic['scan_stamp'] = self.stamp_seconds(scan.header.stamp)
        diagnostic['scan_frame'] = scan.header.frame_id
        diagnostic['total_beams'] = len(scan.ranges)
        points_lidar, valid = scan_to_lidar_points(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max)
        valid_beam_indices = np.flatnonzero(valid)
        diagnostic['valid_ranges'] = int(np.count_nonzero(valid))
        try:
            full_tf = self.tfbuf.lookup_transform_full(
                target_frame='base_footprint',
                target_time=Time.from_msg(image_stamp),
                source_frame=scan.header.frame_id,
                source_time=Time.from_msg(scan.header.stamp),
                fixed_frame=str(self.get_parameter('lidar.fixed_frame').value),
                timeout=Duration(
                    seconds=(float(self.get_parameter('lidar.tf_timeout').value)
                             if allow_enqueue else 0.0)))
            tr = full_tf.transform.translation
            rot = full_tf.transform.rotation
            matrix = transform_matrix(
                [tr.x, tr.y, tr.z], [rot.x, rot.y, rot.z, rot.w])
            points_base = transform_points(points_lidar, matrix)
        except Exception as exc:
            diagnostic['dropped_tf_points'] = len(points_lidar)
            diagnostic['tf_error'] = str(exc)
            if allow_enqueue:
                self.enqueue_lidar_overlay(
                    bev, path_bev, reference_path, current_white_components,
                    history_reference, image_stamp, scan, delta)
                self.stats['lidar_tf_wait'] += 1
            self.last_lidar_diagnostic = diagnostic
            return None
        diagnostic['tf_success'] = True
        diagnostic['transformed_points'] = len(points_base)
        camera_in_bounds, _ = filter_bev_bounds(points_base, self.grid)
        in_bounds, _ = filter_bev_bounds(
            points_base, self.lidar_overlay_grid)
        diagnostic['camera_bev_in_bounds_points'] = len(camera_in_bounds)
        diagnostic['overlay_in_bounds_points'] = len(in_bounds)
        # Preserve the existing key for diagnostic consumers. It now means
        # points visible in the expanded debug overlay.
        diagnostic['in_bounds_points'] = len(in_bounds)
        self.stats['lidar_tf_success'] += 1
        if len(in_bounds):
            visual_stride = max(1, int(self.get_parameter(
                'lidar.overlay_visual_stride').value))
            display_points = in_bounds[::visual_stride]
            cols, rows = self.lidar_overlay_grid.metric_to_pixel(
                display_points[:, 0], display_points[:, 1])
            radius = int(self.get_parameter('lidar.overlay_radius_px').value)
            for col, row in zip(cols, rows):
                pixel = (int(round(col)), int(round(row)))
                cv2.circle(overlay, pixel, radius, (0, 0, 255), -1,
                           cv2.LINE_AA)
                cv2.circle(path_overlay, pixel, radius + 1, (0, 0, 0), -1,
                           cv2.LINE_AA)
                cv2.circle(path_overlay, pixel, radius,
                           self.path_lidar_color, -1, cv2.LINE_AA)
            diagnostic['rendered_raw_points'] = len(display_points)
            diagnostic['visual_stride'] = visual_stride
        avoidance_started=time.perf_counter()
        self.compute_avoidance(
            reference_path, points_base[:, :2], valid_beam_indices,
            (int(scan.header.stamp.sec), int(scan.header.stamp.nanosec)),
            image_stamp, current_white_components, history_reference)
        processing_ms=1000.0*(time.perf_counter()-avoidance_started)
        diagnostic['avoidance_processing_ms']=processing_ms
        self.last_avoidance_diagnostic['processing_ms']=processing_ms
        self.last_avoidance_diagnostic['rendered_raw_points']=(
            diagnostic.get('rendered_raw_points',0))
        result = self.last_avoidance_result
        if result is not None:
            self.draw_circle_avoidance_overlay(avoidance_overlay, result)
        if self.last_multi_track_result is not None:
            self.draw_multi_track_overlay(
                avoidance_overlay, self.last_multi_track_result)
        if (result is not None
                and self.last_active_lifecycle_result is not None):
            self.draw_active_lifecycle_overlay(
                avoidance_overlay, result,
                self.last_active_lifecycle_result)
        if self.last_low_vote_recovery_result is not None:
            self.draw_low_vote_recovery_overlay(
                avoidance_overlay, self.last_low_vote_recovery_result,
                self.last_low_vote_recovery_track)
        self.last_lidar_diagnostic = diagnostic
        return overlay, path_overlay, avoidance_overlay

    def compute_avoidance(self, reference_path, lidar_xy, beam_indices,
                          measurement_key, image_stamp,
                          current_white_components=(),
                          history_reference=None):
        if history_reference is None:
            history_reference=np.empty((0,2),dtype=np.float64)
        reference=np.asarray(reference_path,dtype=np.float64).reshape(-1,2)
        self.last_avoidance_reference=reference.copy()
        self.last_avoidance_result=None
        self.last_legacy_avoidance_result=None
        self.last_multi_track_result=None
        if not bool(self.get_parameter('avoidance_circle.enabled').value):
            self.circle_avoidance.reset()
            self.last_avoidance_diagnostic={
                'active':False,'reason':'CIRCLE_SHADOW_DISABLED',
                'state':'NORMAL','direction_locked':False,
                'raw_valid_lidar_point_count':int(len(lidar_xy))}
            self.multi_obstacle_tracker.reset()
            self.active_obstacle_lifecycle.reset()
            self.low_vote_recovery.reset()
            self.last_multi_track_diagnostic={
                'reason':'CIRCLE_SHADOW_DISABLED'}
            self.last_active_lifecycle_result=None
            self.last_active_lifecycle_diagnostic={
                'reason':'CIRCLE_SHADOW_DISABLED'}
            self.last_low_vote_recovery_result=None
            self.last_low_vote_recovery_track=None
            self.last_low_vote_recovery_diagnostic={
                'reason':'CIRCLE_SHADOW_DISABLED'}
            return
        try:
            # Keep Stage 5.1 as a diagnostic-only comparison. Its geometry is
            # never mixed into the Stage 5.1R path or /perception_v3/path.
            if (len(reference) >= 2
                    and bool(self.get_parameter('avoidance.shadow_enabled').value)):
                self.last_legacy_avoidance_result=deform_path(
                    reference,lidar_xy,self.avoidance_config)
            result=self.circle_avoidance.process(
                lidar_xy,beam_indices,reference,
                measurement_key=measurement_key)
            self.last_avoidance_result=result
            selected=result.selected
            selected_fit=(None if selected is None else selected.fit)
            selected_component=(None if selected is None
                                else selected.component)
            valid_fits=sum(item.valid for item in result.fits)
            wall_count=sum(item.wall_like for item in result.components)
            rear_components=sum(
                bool(np.any(item.points[:,0] < 0.0))
                for item in result.components)
            rear_points=int(np.count_nonzero(
                np.asarray(lidar_xy)[:,0] < 0.0))
            rendered_components=sum(
                item.wall_like or item.point_count >= self.circle_avoidance_config.min_circle_points
                for item in result.components)
            legacy=self.last_legacy_avoidance_result
            self.last_avoidance_diagnostic={
                'active':bool(result.active),'reason':result.reason,
                'measurement_key':list(measurement_key),
                'raw_valid_lidar_point_count':int(len(lidar_xy)),
                'component_count':len(result.components),
                'wall_like_count':int(wall_count),
                'rear_point_count':rear_points,
                'rear_component_count':int(rear_components),
                'rendered_component_count':int(rendered_components),
                'obstacle_sized_component_count':int(
                    len(result.components)-wall_count),
                'valid_circle_count':int(valid_fits),
                'selected_component_id':(None if selected_component is None
                                         else selected_component.component_id),
                'selected_component_support':(None if selected_component is None
                                              else selected_component.support),
                'selected_component_point_count':(None if selected_component is None
                                                  else selected_component.point_count),
                'circle_center':(None if selected_fit is None
                                 else selected_fit.center.tolist()),
                'fitted_radius':(None if selected_fit is None
                                 else selected_fit.radius),
                'fit_residual':(None if selected_fit is None
                                else selected_fit.residual),
                'safety_radius':result.safety_radius,
                'd_obs':(None if selected is None
                         else selected.vehicle_center_distance),
                'signed_lateral':(None if selected is None
                                  else selected.signed_lateral),
                'instantaneous_lane_side':result.instantaneous_side,
                'left_votes':result.left_votes,
                'right_votes':result.right_votes,
                'state':result.state,
                'direction_locked':result.direction_locked,
                'locked_avoidance_side':result.locked_avoidance_side,
                'lock_reason':result.lock_reason,
                'avoidance_target':(None if result.target is None
                                    else result.target.tolist()),
                'target_lateral_offset':result.target_lateral_offset,
                'clearance_original':result.clearance_original,
                'clearance_avoidance':result.clearance_avoidance,
                'safety_clearance_avoidance':result.safety_clearance_avoidance,
                'max_heading_step_original':result.max_heading_step_original,
                'max_heading_step_avoidance':result.max_heading_step_avoidance,
                'steering_original':result.steering_original,
                'steering_avoidance':result.steering_avoidance,
                'legacy_stage5_1':({
                    'active':bool(legacy.active),
                    'reason':legacy.reason,
                    'signed_lateral':legacy.signed_lateral,
                    'signed_offset':legacy.signed_offset,
                    'clearance_original':legacy.clearance_original,
                    'clearance_avoidance':legacy.clearance_avoidance,
                } if legacy is not None else None),
            }
            self.compute_multi_tracks(
                result,reference,image_stamp,measurement_key,
                current_white_components,history_reference)
        except Exception as exc:
            self.last_avoidance_diagnostic={
                'active':False,'reason':'CIRCLE_GEOMETRY_ERROR',
                'error':str(exc),'state':self.circle_avoidance.latch.state,
                'direction_locked':self.circle_avoidance.latch.locked,
                'raw_valid_lidar_point_count':int(len(lidar_xy))}
            self.get_logger().error(f'V3 circle avoidance shadow error: {exc}')

    @staticmethod
    def _track_side(signed_lateral):
        if signed_lateral > 1e-9:
            return CIRCLE_LEFT
        if signed_lateral < -1e-9:
            return CIRCLE_RIGHT
        return CIRCLE_CENTER

    def compute_multi_tracks(self,circle_result,reference,image_stamp,
                             measurement_key,current_white_components=(),
                             history_reference=None):
        if history_reference is None:
            history_reference=np.empty((0,2),dtype=np.float64)
        if not bool(self.get_parameter('obstacle_track.enabled').value):
            self.multi_obstacle_tracker.reset()
            self.active_obstacle_lifecycle.reset()
            self.low_vote_recovery.reset()
            self.last_multi_track_result=None
            self.last_multi_track_diagnostic={'reason':'MULTI_TRACK_DISABLED'}
            self.last_active_lifecycle_result=None
            self.last_active_lifecycle_diagnostic={
                'reason':'MULTI_TRACK_DISABLED'}
            self.last_low_vote_recovery_result=None
            self.last_low_vote_recovery_track=None
            self.last_low_vote_recovery_diagnostic={
                'reason':'MULTI_TRACK_DISABLED'}
            return
        started=time.perf_counter()
        try:
            base_from_odom,odom_from_base=self.exact_odom_matrices(image_stamp)
            relevant=relevant_circles(
                circle_result.components,circle_result.fits,reference,
                self.circle_avoidance_config)
            relevant_by_id={item.fit.component_id:item for item in relevant}
            observations=[]
            for fit in circle_result.fits:
                if not fit.valid:
                    continue
                selected=relevant_by_id.get(fit.component_id)
                if selected is None:
                    frame=classify_circle(
                        fit.center,reference,self.circle_avoidance_config)
                    signed_lateral=float(frame['signed_lateral'])
                    side=frame['side']
                    is_relevant=False
                else:
                    signed_lateral=float(selected.signed_lateral)
                    side=self._track_side(signed_lateral)
                    is_relevant=True
                center_base=np.asarray(fit.center,dtype=np.float64)
                homogeneous=np.array(
                    [center_base[0],center_base[1],0.0,1.0],
                    dtype=np.float64)
                center_odom=(odom_from_base@homogeneous)[:2]
                observations.append(ObstacleObservation(
                    observation_id=int(fit.component_id),
                    center_odom=center_odom,
                    center_base=center_base,
                    radius=float(fit.radius),
                    signed_lateral=signed_lateral,
                    instantaneous_side=side,
                    distance_to_vehicle=float(np.linalg.norm(center_base)),
                    relevant=is_relevant))
            association_started=time.perf_counter()
            track_result=self.multi_obstacle_tracker.update(
                observations,self.stamp_seconds(image_stamp),
                measurement_key=measurement_key)
            association_ms=1000.0*(time.perf_counter()-association_started)
            total_ms=1000.0*(time.perf_counter()-started)
            self.last_multi_track_result=track_result
            self.last_multi_track_diagnostic={
                'reason':('DUPLICATE_SCAN_NO_UPDATE'
                          if track_result.duplicate_measurement else 'OK'),
                'timestamp':track_result.timestamp,
                'measurement_key':list(measurement_key),
                'fitted_circle_count':int(sum(
                    item.valid for item in circle_result.fits)),
                'relevant_circle_count':track_result.relevant_observation_count,
                'active_track_count':track_result.active_track_count,
                'new_track_count':track_result.new_track_count,
                'associated_track_count':track_result.associated_track_count,
                'unmatched_circle_count':track_result.unmatched_observation_count,
                'capacity_rejected_circle_count':(
                    track_result.capacity_rejected_observation_count),
                'expired_track_count':track_result.expired_track_count,
                'expired_track_ids':list(track_result.expired_track_ids),
                'association_distance_threshold':(
                    self.obstacle_track_config.association_distance),
                'retention_age':self.obstacle_track_config.retention_age,
                'max_voting_tracks':(
                    self.obstacle_track_config.max_voting_tracks),
                'association_processing_ms':association_ms,
                'multi_track_processing_ms':total_ms,
                'tracks':[{
                    'track_id':item.track_id,
                    'center_odom':list(item.center_odom),
                    'center_base':list(item.center_base),
                    'radius':item.radius,
                    'association_distance':item.association_distance,
                    'last_seen_age':item.last_seen_age,
                    'seen_count':item.seen_count,
                    'd_obs':item.last_distance_to_vehicle,
                    'signed_lateral':item.signed_lateral,
                    'instantaneous_side':item.instantaneous_side,
                    'left_votes':item.left_votes,
                    'right_votes':item.right_votes,
                    'vote_count':item.vote_count,
                    'direction_locked':item.direction_locked,
                    'locked_avoidance_side':item.locked_avoidance_side,
                    'lock_reason':item.lock_reason,
                    'lock_source':item.lock_source,
                    'observed_this_frame':item.observed_this_frame,
                    'current_component_id':item.current_component_id,
                    'current_relevant':item.current_relevant,
                } for item in track_result.tracks],
            }
            self.compute_active_lifecycle(
                circle_result,track_result,measurement_key,reference,
                current_white_components,history_reference,base_from_odom,
                image_stamp)
        except Exception as exc:
            self.last_multi_track_result=None
            self.last_multi_track_diagnostic={
                'reason':'MULTI_TRACK_ERROR','error':str(exc),
                'measurement_key':list(measurement_key)}
            self.last_low_vote_recovery_result=None
            self.last_low_vote_recovery_track=None
            self.last_low_vote_recovery_diagnostic={
                'reason':'MULTI_TRACK_ERROR','error':str(exc),
                'measurement_key':list(measurement_key)}
            self.get_logger().error(f'V3 multi-track shadow error: {exc}')

    def compute_active_lifecycle(self,circle_result,track_result,
                                 measurement_key,reference=(),
                                 current_white_components=(),
                                 history_reference=None,
                                 base_from_odom=None,image_stamp=None):
        if history_reference is None:
            history_reference=np.empty((0,2),dtype=np.float64)
        if not bool(self.get_parameter('active_lifecycle.enabled').value):
            self.active_obstacle_lifecycle.reset()
            self.low_vote_recovery.reset()
            self.last_active_lifecycle_result=None
            self.last_active_lifecycle_diagnostic={
                'reason':'ACTIVE_LIFECYCLE_DISABLED'}
            self.last_low_vote_recovery_result=None
            self.last_low_vote_recovery_track=None
            self.last_low_vote_recovery_diagnostic={
                'reason':'ACTIVE_LIFECYCLE_DISABLED'}
            return
        if track_result.duplicate_measurement:
            self.last_active_lifecycle_diagnostic={
                **self.last_active_lifecycle_diagnostic,
                'reason':'DUPLICATE_SCAN_HOLD',
                'measurement_key':list(measurement_key)}
            return
        started=time.perf_counter()
        views=tuple(ActiveTrackView(
            track_id=item.track_id,
            center_base=item.center_base,
            distance_to_vehicle=item.last_distance_to_vehicle,
            vote_count=item.vote_count,
            direction_locked=item.direction_locked,
            locked_avoidance_side=item.locked_avoidance_side,
            observed_this_frame=item.observed_this_frame,
            current_relevant=item.current_relevant,
            current_component_id=item.current_component_id,
            center_odom=item.center_odom)
            for item in track_result.tracks)
        component_points={
            int(item.component_id):item.points
            for item in circle_result.components}
        lost_center_base=None
        last_active=self.active_obstacle_lifecycle.last_active_view
        if (last_active is not None and last_active.center_odom is not None
                and base_from_odom is not None):
            matrix=np.asarray(base_from_odom,dtype=np.float64).reshape(4,4)
            homogeneous=np.array([
                last_active.center_odom[0],last_active.center_odom[1],0.0,1.0],
                dtype=np.float64)
            lost_center_base=(matrix@homogeneous)[:2]
        lifecycle=self.active_obstacle_lifecycle.update(
            views,component_points,current_path=reference,
            lost_center_base=lost_center_base)
        lifecycle_ms=1000.0*(time.perf_counter()-started)
        self.last_active_lifecycle_result=lifecycle
        termination=lifecycle.termination
        lost_release=lifecycle.lost_release
        active_points=component_points.get(lifecycle.active_component_id)
        active_point_count=(0 if active_points is None else len(active_points))
        self.last_active_lifecycle_diagnostic={
            'reason':'OK',
            'timestamp':track_result.timestamp,
            'measurement_key':list(measurement_key),
            'state':lifecycle.state,
            'active_track_id':lifecycle.active_track_id,
            'evaluated_track_id':lifecycle.evaluated_track_id,
            'terminated_track_id':lifecycle.terminated_track_id,
            'released_lost_track_id':lifecycle.released_lost_track_id,
            'candidate_track_ids':list(lifecycle.candidate_track_ids),
            'next_candidate_id':lifecycle.next_candidate_id,
            'active_direction_locked':lifecycle.active_direction_locked,
            'active_locked_avoidance_side':(
                lifecycle.active_locked_avoidance_side),
            'active_center_base':(None if lifecycle.active_center_base is None
                                  else list(lifecycle.active_center_base)),
            'active_d_obs':lifecycle.active_distance_to_vehicle,
            'active_component_id':lifecycle.active_component_id,
            'active_component_point_count':int(active_point_count),
            'evaluated_component_point_count':termination.point_count,
            'd_surface':termination.distance_surface,
            'max_component_x':termination.max_x,
            'robot_bounding_radius':(
                self.active_obstacle_lifecycle.robot_radius),
            'passed':termination.passed,
            'surface_clear':termination.surface_clear,
            'termination':termination.termination,
            'termination_hold':lifecycle.termination_hold,
            'lost_center_base':(None if lost_release.center_base is None else
                                list(lost_release.center_base)),
            'lost_center_distance':lost_release.distance_to_center,
            'lost_release_radius_multiplier':(
                self.active_obstacle_lifecycle.
                lost_release_radius_multiplier),
            'active_activation_distance':(
                self.active_obstacle_lifecycle.activation_distance),
            'lost_release_distance':lost_release.release_distance,
            'lost_path_valid':lost_release.path_valid,
            'lost_vehicle_s':lost_release.vehicle_s,
            'lost_obstacle_s':lost_release.obstacle_s,
            'lost_progress_delta':lost_release.progress_delta,
            'lost_path_passed':lost_release.path_passed,
            'lost_distance_clear':lost_release.distance_clear,
            'lost_release':lost_release.release,
            'events':list(lifecycle.events),
            'completed_track_ids':list(lifecycle.completed_track_ids),
            'active_lifecycle_processing_ms':lifecycle_ms,
            'termination_evaluation_ms_included':lifecycle_ms,
        }
        self.compute_low_vote_recovery(
            track_result,lifecycle,reference,current_white_components,
            history_reference,measurement_key)
        self.compute_stage5_3_control(
            circle_result,track_result,lifecycle,reference,image_stamp)

    def compute_low_vote_recovery(self,track_result,lifecycle,reference,
                                  current_white_components,
                                  history_reference,measurement_key):
        self.low_vote_recovery.retain_tracks(
            item.track_id for item in track_result.tracks)
        if not bool(self.get_parameter('avoidance_recovery.enabled').value):
            self.low_vote_recovery.reset()
            self.last_low_vote_recovery_result=None
            self.last_low_vote_recovery_track=None
            self.last_low_vote_recovery_diagnostic={
                'reason':'LOW_VOTE_RECOVERY_DISABLED'}
            return
        if lifecycle.state == ACTIVE_LOST:
            self.last_low_vote_recovery_result=None
            self.last_low_vote_recovery_track=None
            self.last_low_vote_recovery_diagnostic={
                'reason':'ACTIVE_LOST_HOLD',
                'active_track_id':lifecycle.active_track_id,
                'measurement_key':list(measurement_key)}
            return
        active=next((item for item in track_result.tracks
                     if item.track_id==lifecycle.active_track_id),None)
        if active is None:
            self.last_low_vote_recovery_result=None
            self.last_low_vote_recovery_track=None
            self.last_low_vote_recovery_diagnostic={
                'reason':'NO_ACTIVE',
                'active_track_id':lifecycle.active_track_id,
                'measurement_key':list(measurement_key)}
            return
        view=RecoveryTrackView(
            track_id=active.track_id,
            center_base=active.center_base,
            distance_to_vehicle=active.last_distance_to_vehicle,
            left_votes=active.left_votes,
            right_votes=active.right_votes,
            vote_count=active.vote_count,
            direction_locked=active.direction_locked,
            locked_avoidance_side=active.locked_avoidance_side,
            lock_source=active.lock_source)
        started=time.perf_counter()
        recovery=self.low_vote_recovery.update(
            view,reference,current_white_components,history_reference,
            measurement_key=measurement_key)
        elapsed_ms=1000.0*(time.perf_counter()-started)
        lock_applied=False
        if recovery.lock_requested:
            if recovery.lock_source in (
                    NORMAL_VOTE,SLOW_VOTING_LOCK,HEADING_RECOVERY_VOTE):
                lock_applied=self.multi_obstacle_tracker.lock_track_from_votes(
                    active.track_id,recovery.lock_source)
            elif recovery.lock_source in (
                    WHITE_PROXIMITY,DETERMINISTIC_DEFAULT):
                lock_applied=self.multi_obstacle_tracker.lock_track_side(
                    active.track_id,recovery.chosen_side,
                    recovery.lock_source)
        tracker_state=self.multi_obstacle_tracker.tracks.get(active.track_id)
        locked_after=bool(
            tracker_state is not None and tracker_state.direction_locked)
        side_after=(None if tracker_state is None else
                    tracker_state.locked_avoidance_side)
        source_after=(None if tracker_state is None else
                      tracker_state.lock_source)
        self.last_low_vote_recovery_result=recovery
        self.last_low_vote_recovery_track=active
        self.last_low_vote_recovery_diagnostic={
            'reason':recovery.reason,
            'timestamp':track_result.timestamp,
            'measurement_key':list(measurement_key),
            'active_track_id':active.track_id,
            'd_obs':active.last_distance_to_vehicle,
            'vote_count':active.vote_count,
            'left_votes':active.left_votes,
            'right_votes':active.right_votes,
            'recovery_state':recovery.state,
            'requested_mode':recovery.requested_mode,
            'recovery_frame_count':recovery.recovery_frame_count,
            'no_vote_progress_count':recovery.no_vote_progress_count,
            'obstacle_bearing':recovery.obstacle_bearing,
            'center_path_valid':recovery.center_path_valid,
            'center_start_distance':recovery.center_start_distance,
            'recovery_heading_target':recovery.recovery_heading_target,
            'recovery_heading_error':recovery.recovery_heading_error,
            'heading_correction_limit':(
                self.low_vote_recovery_config.max_heading_correction),
            'nearest_white_component_id':(
                recovery.nearest_white_component_id),
            'nearest_white_point':(None if recovery.nearest_white_point is None
                                   else list(recovery.nearest_white_point)),
            'obstacle_to_white_distance':(
                recovery.obstacle_to_white_distance),
            'white_fallback_vector':(
                None if recovery.white_fallback_vector is None else
                list(recovery.white_fallback_vector)),
            'local_reference_source':recovery.reference_source,
            'chosen_side':(side_after if locked_after else
                           recovery.chosen_side),
            'lock_source':(source_after if locked_after else
                           recovery.lock_source),
            'lock_requested':recovery.lock_requested,
            'lock_applied':lock_applied,
            'direction_locked':locked_after,
            'emergency_distance':(
                self.low_vote_recovery_config.emergency_distance),
            'emergency_distance_reached':(
                recovery.emergency_distance_reached),
            'current_white_component_count':len(current_white_components),
            'history_reference_point_count':len(history_reference),
            'recovery_evaluation_ms':elapsed_ms,
            'white_search_upper_bound_ms':(
                elapsed_ms if recovery.nearest_white_component_id is not None
                else 0.0),
        }

    def compute_stage5_3_control(self,circle_result,track_result,lifecycle,
                                 reference,image_stamp):
        """Create an authoritative ACTIVE-specific path/control contract."""
        reference=np.asarray(reference,dtype=np.float64).reshape(-1,2)
        center_valid=bool(
            len(reference)>=2 and np.all(np.isfinite(reference)))
        active_snapshot=next((item for item in track_result.tracks
                              if item.track_id==lifecycle.active_track_id),None)
        active_state=self.multi_obstacle_tracker.tracks.get(
            lifecycle.active_track_id)
        direction_locked=bool(
            active_state is not None and active_state.direction_locked)
        locked_side=(None if active_state is None else
                     active_state.locked_avoidance_side)
        active_observed=bool(
            active_snapshot is not None
            and active_snapshot.observed_this_frame
            and active_snapshot.current_component_id is not None)
        component_id=(None if active_snapshot is None else
                      active_snapshot.current_component_id)

        geometry=build_active_avoidance_geometry(
            circle_result.components,circle_result.fits,reference,
            component_id,locked_side,self.circle_avoidance_config)
        recovery=self.last_low_vote_recovery_result
        recovery_matches=bool(
            recovery is not None
            and recovery.track_id==lifecycle.active_track_id)
        requested_mode=(recovery.requested_mode if recovery_matches else 'NONE')
        heading_target=(recovery.recovery_heading_target
                        if recovery_matches else None)
        recovery_state=(recovery.state if recovery_matches else
                        'NORMAL_VOTING')
        decision=choose_control_mode(
            lifecycle.state,active_snapshot is not None,active_observed,
            direction_locked,geometry.valid,center_valid,requested_mode,
            heading_target)

        self.last_control_path=(geometry.path.copy() if geometry.valid else
                                np.empty((0,2),dtype=np.float64))
        if image_stamp is None:
            self.last_control_stamp=None
            stamp_ns=None
        else:
            self.last_control_stamp=(
                int(image_stamp.sec),int(image_stamp.nanosec))
            stamp_ns=(self.last_control_stamp[0]*1000000000
                      + self.last_control_stamp[1])
        termination=lifecycle.termination
        self.last_control_diagnostic={
            'schema_version':1,
            'stamp_ns':stamp_ns,
            'controller_mode':decision.mode,
            'reason':decision.reason,
            'perception_ready':True,
            'selected_path':decision.mode,
            'center_path_valid':center_valid,
            'center_path_point_count':int(len(reference)),
            'avoidance_path_valid':bool(geometry.valid),
            'avoidance_path_reason':geometry.reason,
            'avoidance_path_point_count':int(len(self.last_control_path)),
            'lifecycle_state':lifecycle.state,
            'lifecycle_events':list(lifecycle.events),
            'active_track_id':lifecycle.active_track_id,
            'active_component_id':component_id,
            'active_observed':active_observed,
            'active_direction_locked':direction_locked,
            'active_locked_avoidance_side':locked_side,
            'active_d_obs':(None if active_snapshot is None else
                            active_snapshot.last_distance_to_vehicle),
            'recovery_state':recovery_state,
            'requested_mode':requested_mode,
            'recovery_heading_target':heading_target,
            'lock_source':(None if active_state is None else
                           active_state.lock_source),
            'avoidance_target':(None if geometry.target is None else
                                geometry.target.tolist()),
            'target_lateral_offset':geometry.target_lateral_offset,
            'clearance_original':geometry.clearance_original,
            'clearance_avoidance':geometry.clearance_avoidance,
            'safety_clearance_avoidance':(
                geometry.safety_clearance_avoidance),
            'steering_original_shadow':geometry.steering_original,
            'steering_avoidance_shadow':geometry.steering_avoidance,
            'd_surface':termination.distance_surface,
            'termination':termination.termination,
            'termination_hold':lifecycle.termination_hold,
            'terminated_track_id':lifecycle.terminated_track_id,
            'next_candidate_id':lifecycle.next_candidate_id,
        }

    def _draw_metric_polyline(self, image, points, color, width):
        values=np.asarray(points,dtype=np.float64).reshape(-1,2)
        if not len(values):
            return
        cols,rows=self.lidar_overlay_grid.metric_to_pixel(
            values[:,0],values[:,1])
        pixels=np.rint(np.c_[cols,rows]).astype(np.int32)
        if len(pixels)==1:
            cv2.circle(image,tuple(pixels[0]),max(1,width),color,-1,
                       cv2.LINE_AA)
        else:
            cv2.polylines(image,[pixels],False,color,width,cv2.LINE_AA)

    def _draw_metric_circle(self,image,center,radius,color,width):
        col,row=self.lidar_overlay_grid.metric_to_pixel(center[0],center[1])
        pixel=(int(round(col)),int(round(row)))
        radius_px=max(1,int(round(
            float(radius)/self.lidar_overlay_grid.resolution)))
        cv2.circle(image,pixel,radius_px,color,width,cv2.LINE_AA)

    def draw_circle_avoidance_overlay(self,image,result):
        fit_by_id={item.component_id:item for item in result.fits}
        selected_id=(None if result.selected is None
                     else result.selected.component.component_id)
        minimum_points=self.circle_avoidance_config.min_circle_points
        for component in result.components:
            if not component.wall_like and component.point_count < minimum_points:
                continue
            if component.component_id==selected_id:
                color=(0,165,255); width=3
            elif component.wall_like:
                color=(100,100,100); width=1
            elif fit_by_id[component.component_id].valid:
                color=(255,120,0); width=2
            else:
                color=(120,70,0); width=1
            self._draw_metric_polyline(image,component.points,color,width)

        selected=result.selected
        if selected is not None:
            center=selected.fit.center
            self._draw_metric_circle(
                image,center,selected.fit.radius,(0,255,255),2)
            if result.safety_radius is not None:
                self._draw_metric_circle(
                    image,center,result.safety_radius,(255,0,128),2)
            col,row=self.lidar_overlay_grid.metric_to_pixel(center[0],center[1])
            pixel=(int(round(col)),int(round(row)))
            cv2.circle(image,pixel,5,(0,0,0),-1,cv2.LINE_AA)
            cv2.circle(image,pixel,3,(0,255,255),-1,cv2.LINE_AA)
        if result.target is not None:
            col,row=self.lidar_overlay_grid.metric_to_pixel(
                result.target[0],result.target[1])
            pixel=(int(round(col)),int(round(row)))
            cv2.circle(image,pixel,7,(0,0,0),-1,cv2.LINE_AA)
            cv2.circle(image,pixel,5,(203,192,255),-1,cv2.LINE_AA)
        if result.active:
            self._draw_metric_polyline(
                image,result.shadow_path,(0,0,0),5)
            self._draw_metric_polyline(
                image,result.shadow_path,self.avoidance_path_color,2)

        selected_fit=(None if selected is None else selected.fit)
        distance=(None if selected is None
                  else selected.vehicle_center_distance)
        lines=[
            f'{result.state} lock={result.direction_locked} side={result.locked_avoidance_side}',
            f'vote L={result.left_votes} R={result.right_votes} instant={result.instantaneous_side}',
            ('circle none' if selected_fit is None else
             f'circle n={selected_fit.point_count} r={selected_fit.radius:.3f} '
             f'e={selected_fit.residual:.3f} d={distance:.3f}'),
        ]
        for index,text_value in enumerate(lines):
            origin=(8,18+17*index)
            cv2.putText(image,text_value,origin,cv2.FONT_HERSHEY_SIMPLEX,
                        .42,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(image,text_value,origin,cv2.FONT_HERSHEY_SIMPLEX,
                        .42,(255,255,255),1,cv2.LINE_AA)

    def draw_multi_track_overlay(self,image,result):
        active_id=(None if self.last_active_lifecycle_result is None else
                   self.last_active_lifecycle_result.active_track_id)
        for track in result.tracks:
            center=track.center_base
            col,row=self.lidar_overlay_grid.metric_to_pixel(
                center[0],center[1])
            pixel=(int(round(col)),int(round(row)))
            color=((40,220,40) if track.direction_locked else (255,80,220))
            width=2
            if track.track_id==active_id:
                color=(0,80,255); width=3
            if not track.observed_this_frame:
                color=(120,120,120)
            self._draw_metric_circle(image,center,track.radius,color,width)
            state=(track.locked_avoidance_side
                   if track.direction_locked else 'VOTING')
            active_label=(' ACTIVE' if track.track_id==active_id else '')
            lines=(f'#{track.track_id}{active_label} L={track.left_votes} R={track.right_votes}',
                   f'd={track.last_distance_to_vehicle:.2f} {state}')
            for index,text_value in enumerate(lines):
                origin=(pixel[0]+7,pixel[1]-8+14*index)
                cv2.putText(image,text_value,origin,
                            cv2.FONT_HERSHEY_SIMPLEX,.36,(0,0,0),3,
                            cv2.LINE_AA)
                cv2.putText(image,text_value,origin,
                            cv2.FONT_HERSHEY_SIMPLEX,.36,color,1,
                            cv2.LINE_AA)

    def draw_active_lifecycle_overlay(self,image,circle_result,lifecycle):
        self._draw_metric_circle(
            image,(0.0,0.0),self.active_obstacle_lifecycle.robot_radius,
            (0,255,80),2)
        component_by_id={
            item.component_id:item for item in circle_result.components}
        component=component_by_id.get(lifecycle.active_component_id)
        if component is not None:
            self._draw_metric_polyline(
                image,component.points,(0,80,255),4)
        termination=lifecycle.termination
        lost_release=lifecycle.lost_release
        distance=('none' if termination.distance_surface is None else
                  f'{termination.distance_surface:.3f}')
        max_x=('none' if termination.max_x is None else
               f'{termination.max_x:.3f}')
        lost_distance=('none' if lost_release.distance_to_center is None else
                       f'{lost_release.distance_to_center:.3f}')
        lost_delta=('none' if lost_release.progress_delta is None else
                    f'{lost_release.progress_delta:.3f}')
        lines=(
            f'5.2B {lifecycle.state} active={lifecycle.active_track_id} '
            f'side={lifecycle.active_locked_avoidance_side}',
            f'raw d={distance} max_x={max_x} '
            f'hold={lifecycle.termination_hold}',
            f'passed={termination.passed} clear={termination.surface_clear} '
            f'term={termination.termination}',
            f'lost d={lost_distance} ds={lost_delta} '
            f'release={lost_release.release}',
        )
        for index,text_value in enumerate(lines):
            origin=(8,70+17*index)
            cv2.putText(image,text_value,origin,cv2.FONT_HERSHEY_SIMPLEX,
                        .40,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(image,text_value,origin,cv2.FONT_HERSHEY_SIMPLEX,
                        .40,(0,255,80),1,cv2.LINE_AA)

    def draw_low_vote_recovery_overlay(self,image,recovery,track):
        if track is None:
            return
        if recovery.recovery_heading_target is not None:
            angle=recovery.recovery_heading_target
            endpoint=np.array([.45*math.cos(angle),.45*math.sin(angle)])
            self._draw_metric_polyline(
                image,np.array([[0.0,0.0],endpoint]),(255,255,0),2)
        if recovery.nearest_white_point is not None:
            point=np.asarray(recovery.nearest_white_point,dtype=np.float64)
            self._draw_metric_circle(image,point,.035,(255,0,255),-1)
            if recovery.white_fallback_vector is not None:
                away=np.asarray(
                    recovery.white_fallback_vector,dtype=np.float64)
                self._draw_metric_polyline(
                    image,np.array([point,point+.25*away]),(255,0,255),2)
        lines=(
            f'5.2C #{track.track_id} {recovery.state} '
            f'mode={recovery.requested_mode}',
            f'vote={track.vote_count} L={track.left_votes} R={track.right_votes} '
            f'frame={recovery.recovery_frame_count}',
            f'bearing={recovery.obstacle_bearing:.2f} '
            f'side={recovery.chosen_side} source={recovery.lock_source}',
        )
        for index,text_value in enumerate(lines):
            origin=(8,124+17*index)
            cv2.putText(image,text_value,origin,cv2.FONT_HERSHEY_SIMPLEX,
                        .38,(0,0,0),3,cv2.LINE_AA)
            cv2.putText(image,text_value,origin,cv2.FONT_HERSHEY_SIMPLEX,
                        .38,(255,255,0),1,cv2.LINE_AA)

    def retry_lidar(self):
        max_age = float(self.get_parameter('tf_wait.max_pending_age').value)
        while self.lidar_pending:
            (bev,path_bev,reference_path,current_white_components,
             history_reference,image_stamp,scan,delta,
             queued_at)=self.lidar_pending[0]
            if time.monotonic() - queued_at > max_age:
                self.lidar_pending.pop(0)
                self.stats['lidar_tf_failure'] += 1
                continue
            overlays = self.render_lidar_overlay(
                bev,path_bev,reference_path,image_stamp,
                current_white_components=current_white_components,
                history_reference=history_reference,
                scan=scan,delta=delta,allow_enqueue=False)
            if overlays is None:
                break
            self.lidar_pending.pop(0)
            self.publish_lidar_overlay(*overlays, image_stamp)

    def debug_wanted(self, *publishers):
        """구독자가 하나라도 있으면 True.

        이걸로 감싸는 것들은 전부 **그림과 진단 문자열**이다. 경로 계산이
        끝난 뒤에 그리는 것들이라, 안 그려도 /perception_v3/path 는 한
        글자도 안 바뀐다.

        실측으로 tail 구간이 33.8 ms 였다 -- 한 프레임 140.9 ms 중 24%.
        라이다 진단 줄의 overlays=118 이 그게 매 프레임 돌고 있었다는
        증거다.

        스위치가 아니라 구독자 수로 판단하는 이유: 끄는 걸 까먹을 스위치가
        없고, rqt 를 열면 저절로 다시 나온다. cone_bev_node 가 구독하는
        /perception_v3/debug/bev 와 white_mask 도 회피를 켜면 저절로
        살아난다.
        """
        return any(pub.get_subscription_count() for pub in publishers)

    def publish_lidar_overlay(self, overlay, path_overlay,
                              avoidance_overlay, image_stamp):
        message = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
        message.header.stamp = image_stamp
        message.header.frame_id = 'base_footprint'
        self.lidar_bev_pub.publish(message)
        path_message = self.bridge.cv2_to_imgmsg(path_overlay, 'bgr8')
        path_message.header.stamp = image_stamp
        path_message.header.frame_id = 'base_footprint'
        self.path_lidar_pub.publish(path_message)
        avoidance_message=self.bridge.cv2_to_imgmsg(avoidance_overlay,'bgr8')
        avoidance_message.header.stamp=image_stamp
        avoidance_message.header.frame_id='base_footprint'
        self.avoidance_overlay_pub.publish(avoidance_message)
        diagnostic = String()
        diagnostic.data = json.dumps(
            self.last_lidar_diagnostic, sort_keys=True)
        self.lidar_diag_pub.publish(diagnostic)
        self.stats['lidar_overlay_published'] += 1
        self.stats['path_lidar_overlay_published'] += 1
        self.publish_avoidance(image_stamp)

    def publish_avoidance(self,image_stamp):
        result=self.last_avoidance_result
        points=(result.shadow_path if result is not None
                else self.last_avoidance_reference)
        path=Path(); path.header.stamp=image_stamp; path.header.frame_id='base_footprint'
        for point in points:
            pose=PoseStamped(); pose.header=path.header
            pose.pose.position.x=float(point[0]); pose.pose.position.y=float(point[1]); pose.pose.orientation.w=1.0
            path.poses.append(pose)
        self.avoidance_path_pub.publish(path)
        active=Bool(); active.data=bool(result is not None and result.active)
        self.avoidance_active_pub.publish(active)
        offset=Float64(); offset.data=(
            float(result.target_lateral_offset)
            if result is not None and result.target_lateral_offset is not None
            else 0.0)
        self.avoidance_offset_pub.publish(offset)
        if result is not None and result.selected is not None:
            obstacle=PointStamped(); obstacle.header=path.header
            obstacle.point.x=float(result.selected.fit.center[0]); obstacle.point.y=float(result.selected.fit.center[1])
            self.avoidance_obstacle_pub.publish(obstacle)
        if result is not None and result.target is not None:
            target=PointStamped(); target.header=path.header
            target.point.x=float(result.target[0]); target.point.y=float(result.target[1])
            self.avoidance_target_pub.publish(target)
        diagnostic=String(); diagnostic.data=json.dumps(self.last_avoidance_diagnostic,sort_keys=True)
        self.avoidance_diag_pub.publish(diagnostic)
        multi_diagnostic=String(); multi_diagnostic.data=json.dumps(
            self.last_multi_track_diagnostic,sort_keys=True)
        self.multi_track_diag_pub.publish(multi_diagnostic)
        active_lifecycle=String(); active_lifecycle.data=json.dumps(
            self.last_active_lifecycle_diagnostic,sort_keys=True)
        self.active_lifecycle_pub.publish(active_lifecycle)
        recovery_message=String(); recovery_message.data=json.dumps(
            self.last_low_vote_recovery_diagnostic,sort_keys=True)
        self.low_vote_recovery_pub.publish(recovery_message)
        recovery_mode=String(); recovery_mode.data=str(
            self.last_low_vote_recovery_diagnostic.get(
                'requested_mode','NONE'))
        self.recovery_mode_pub.publish(recovery_mode)
        self.publish_stage5_3_control(image_stamp)
        multi_tracks=String(); multi_tracks.data=json.dumps({
            'timestamp':self.last_multi_track_diagnostic.get('timestamp'),
            'active_track_count':self.last_multi_track_diagnostic.get(
                'active_track_count',0),
            'tracks':self.last_multi_track_diagnostic.get('tracks',[])},
            sort_keys=True)
        self.multi_track_pub.publish(multi_tracks)
        components=String(); components.data=json.dumps({
            'count':0 if result is None else len(result.components),
            'items':([] if result is None else [{
                'id':item.component_id,'points':item.point_count,
                'support':item.support,'span':item.span,
                'nearest':item.nearest_distance,
                'wall_like':item.wall_like,
            } for item in result.components if item.wall_like or item.point_count >= self.circle_avoidance_config.min_circle_points])},sort_keys=True)
        self.avoidance_components_pub.publish(components)
        selected=String(); selected.data=json.dumps({
            key:self.last_avoidance_diagnostic.get(key) for key in (
                'selected_component_id','selected_component_support',
                'selected_component_point_count','circle_center','d_obs',
                'signed_lateral','instantaneous_lane_side')},sort_keys=True)
        self.avoidance_selected_pub.publish(selected)
        circle=String(); circle.data=json.dumps({
            key:self.last_avoidance_diagnostic.get(key) for key in (
                'circle_center','fitted_radius','fit_residual')},sort_keys=True)
        self.avoidance_circle_pub.publish(circle)
        safety=String(); safety.data=json.dumps({
            'center':self.last_avoidance_diagnostic.get('circle_center'),
            'radius':self.last_avoidance_diagnostic.get('safety_radius')},sort_keys=True)
        self.avoidance_safety_pub.publish(safety)
        votes=String(); votes.data=json.dumps({
            key:self.last_avoidance_diagnostic.get(key) for key in (
                'left_votes','right_votes','instantaneous_lane_side',
                'locked_avoidance_side','lock_reason')},sort_keys=True)
        self.avoidance_votes_pub.publish(votes)
        state=String(); state.data=(
            'NOT_PROCESSED' if result is None else result.state)
        self.avoidance_state_pub.publish(state)
        locked=Bool(); locked.data=bool(
            result is not None and result.direction_locked)
        self.avoidance_locked_pub.publish(locked)
        self.stats['avoidance_published']+=1
        self.stats['multi_track_published']+=1
        if active.data: self.stats['avoidance_active']+=1

    def publish_stage5_3_control(self,image_stamp):
        """Publish the last new-scan control contract without restamping it.

        A camera frame may reuse the same LaserScan.  Keeping the original
        control stamp allows the controller watchdog to detect a genuinely
        stale LiDAR/ACTIVE decision instead of treating repeated debug output
        as a fresh obstacle observation.
        """
        stamp=(self.last_control_stamp if self.last_control_stamp is not None
               else (int(image_stamp.sec),int(image_stamp.nanosec)))
        path=Path()
        path.header.stamp.sec=stamp[0]
        path.header.stamp.nanosec=stamp[1]
        path.header.frame_id='base_footprint'
        for point in self.last_control_path:
            pose=PoseStamped(); pose.header=path.header
            pose.pose.position.x=float(point[0])
            pose.pose.position.y=float(point[1])
            pose.pose.orientation.w=1.0
            path.poses.append(pose)
        self.control_path_pub.publish(path)
        diagnostic=dict(self.last_control_diagnostic)
        if diagnostic.get('stamp_ns') is None:
            diagnostic['stamp_ns']=stamp[0]*1000000000+stamp[1]
        message=String()
        message.data=json.dumps(diagnostic,sort_keys=True)
        self.control_status_pub.publish(message)
    def lookup(self,msg):
        """이미지 시각에 딱 맞는 TF 를 찾는다.

        원래는 정확한 시각만 허용한다. 카메라가 주행 중에 움직이면 그게
        맞다 -- 100 ms 전 자세로 BEV 를 만들면 지면이 통째로 틀어진다.

        그런데 우리 차는 주행 중 카메라가 **고정**이다(틸트 -30도, 팬 0도
        로 잡아둔다). 그 경우 정확한 시각을 고집하는 값이 너무 크다.
        실측:

            images=71  immediate=5  pending=66      93%가 대기 큐

        원인은 TF 스탬프가 joint_states 스탬프라는 것이다. 이미지는 30 ms
        만에 도착하는데 그 시각의 joint_states 가 아직 안 와 있으면,
        TF 가 따라올 때까지 프레임을 붙잡는다.

        tf_allow_latest 를 켜면 정확한 시각이 없을 때 **가장 최근 TF** 로
        대신한다. 카메라가 안 움직이면 최근 TF 와 정확한 시각의 TF 가
        같으므로 잃는 것이 없다.

        **카메라를 주행 중에 움직인다면 켜면 안 된다.** 그래서 기본은
        꺼짐이고, 런치가 카메라를 고정으로 잡을 때만 켠다.
        """
        stamp=Time.from_msg(msg.header.stamp)
        latest_ok=bool(self.get_parameter('tf_wait.allow_latest').value)
        ready,reason=self.tfbuf.can_transform('base_footprint','camera_optical_frame_corrected',stamp,timeout=Duration(seconds=0.0),return_debug_tuple=True)
        if not ready and latest_ok:
            zero=Time()
            if self.tfbuf.can_transform('base_footprint','camera_optical_frame_corrected',zero,timeout=Duration(seconds=0.0)):
                self.stats['tf_latest_used'] = self.stats.get('tf_latest_used',0)+1
                return self.tfbuf.lookup_transform('base_footprint','camera_optical_frame_corrected',zero,timeout=Duration(seconds=0.0))
        if not ready:
            # tf2 가 왜 안 된다고 하는지를 그대로 찍는다. 추측하지 않기
            # 위해서다. "extrapolation into the future" 면 TF 가 아직 안 온
            # 것이고, "past" 면 너무 늦게 온 것이고, 프레임 이름이 나오면
            # 아예 다른 문제다. 셋의 고칠 곳이 전부 다르다.
            self.get_logger().warn(
                'TF 대기 이유: %s  (이미지 %.3f, 지금 %.3f, 차이 %.3f초)'
                % (str(reason).strip() or '(이유 없음)',
                   stamp.nanoseconds / 1e9,
                   self.get_clock().now().nanoseconds / 1e9,
                   (self.get_clock().now().nanoseconds
                    - stamp.nanoseconds) / 1e9),
                throttle_duration_sec=2.0)
            raise RuntimeError('exact_tf_not_ready')
        if bool(self.get_parameter('center_history.enabled').value):
            fixed=str(self.get_parameter('lidar.fixed_frame').value)
            odom_ready,_=self.tfbuf.can_transform(
                'base_footprint',fixed,stamp,timeout=Duration(seconds=0.0),
                return_debug_tuple=True)
            if not odom_ready:
                raise RuntimeError('exact_tf_not_ready: odom')
        return self.tfbuf.lookup_transform('base_footprint','camera_optical_frame_corrected',stamp,timeout=Duration(seconds=0.0))

    @staticmethod
    def _failure_kind(exc):
        text = str(exc).lower()
        if 'exact_tf_not_ready' in text: return 'FUTURE_EXTRAPOLATION'
        if 'future' in text and 'extrapolat' in text: return 'FUTURE_EXTRAPOLATION'
        if 'past' in text and 'extrapolat' in text: return 'PAST_EXTRAPOLATION'
        if 'frame' in text and ('exist' in text or 'found' in text or 'invalid' in text): return 'FRAME_NOT_FOUND'
        if 'connect' in text: return 'CONNECTIVITY'
        return 'OTHER'

    def _enqueue(self, msg):
        self.stats['pending_enqueued'] += 1
        entry = (msg, time.monotonic())
        if len(self.pending) < 2:
            self.pending.append(entry)
        else:
            self.pending[-1] = entry
            self.pending_replaced += 1
            self.stats['pending_replaced'] += 1
    @staticmethod
    def matrix(t):
        q=t.transform.rotation; x,y,z,w=q.x,q.y,q.z,q.w; tr=t.transform.translation
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),tr.x],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),tr.y],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y),tr.z],[0,0,0,1]],float)

    def exact_odom_matrices(self, stamp):
        """Return base(t)<-odom and its exact inverse; never use latest TF."""
        fixed=str(self.get_parameter('lidar.fixed_frame').value)
        transform=self.tfbuf.lookup_transform(
            'base_footprint',fixed,Time.from_msg(stamp),
            timeout=Duration(seconds=float(
                self.get_parameter('lidar.tf_timeout').value)))
        base_from_odom=self.matrix(transform)
        return base_from_odom,np.linalg.inv(base_from_odom)

    @staticmethod
    def orange_fragments(items, orange_result):
        by_id={item.component_id:item for item in items
               if item.color == 'ORANGE'}
        fragments=[by_id[ident].polyline.points
                   for ident in orange_result.stitched_component_ids
                   if ident in by_id]
        return orient_fragment_chain(fragments) if fragments else tuple()

    def publish_center_hybrid_debug(self, role_image, temporary, recovery,
                                    prediction, final_result, diagnostic,
                                    stamp):
        overlay=role_image.copy()
        if temporary.valid and temporary.path is not None:
            self.draw_path_points(
                overlay,temporary.path.points,(255,0,255),1)
        if recovery is not None and len(recovery.prefix):
            self.draw_path_points(overlay,recovery.prefix,(255,255,0),2)
        if final_result.valid and final_result.path is not None:
            self.draw_path_points(
                overlay,final_result.path.points,(0,255,0),2)
        if prediction is not None and len(prediction.suffix):
            # Prediction is part of the final path, but stays visibly
            # magenta so measured geometry and extrapolation are distinct.
            self.draw_path_points(overlay,prediction.suffix,(255,0,255),2)
        message=self.bridge.cv2_to_imgmsg(overlay,'bgr8')
        message.header.stamp=stamp
        message.header.frame_id='base_footprint'
        self.center_hybrid_overlay_pub.publish(message)
        value=String(); value.data=json.dumps(diagnostic,sort_keys=True)
        self.center_hybrid_diag_pub.publish(value)
    def image_cb(self,msg):
        self.stats['images_received'] += 1
        try:
            tfmsg = self.lookup(msg)
            self.stats['immediate_tf_success'] += 1
            self.process(msg, tfmsg)
        except Exception as exc:
            self._enqueue(msg)
            # Exact-stamp TF commonly arrives just after the image in SIM. The
            # pending queue/retry path remains unchanged; only bound the
            # expected diagnostic so it cannot flood the terminal per frame.
            self.get_logger().warning(
                f'V3 frame pending ({self._failure_kind(exc)}): {exc}',
                throttle_duration_sec=2.0)
            self.retry()
    def retry(self):
        now=time.monotonic(); max_age=float(self.get_parameter('tf_wait.max_pending_age').value)
        while self.pending:
            msg, queued_at=self.pending[0]
            self.stats['pending_retry_attempts'] += 1
            if now-queued_at > max_age:
                self.pending.pop(0)
                self.stats['pending_timeout'] += 1
                continue
            try:
                tfmsg = self.lookup(msg)
                self.pending.pop(0)
                self.stats['pending_eventual_success'] += 1
                self.process(msg, tfmsg)
            except Exception as exc:
                # Keep the oldest pending image until its exact TF arrives.
                # No latest-TF fallback is permitted.
                break
        self.retry_lidar()
    def process(self,msg,tfmsg):
        t_start=time.perf_counter()
        stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
        if stamp in self.processed_stamps:
            self.stats['duplicate_processed'] += 1
            return
        self.processed_stamps.add(stamp)
        self.stats['frames_processed'] += 1
        p=lambda k:self.get_parameter(k).value; tf=apply_projection_corrections(self.matrix(tfmsg),camera_height_correction_z=float(p('sim_geometry.camera_height_correction_z')),pitch_offset_deg=float(p('projection.pitch_offset_deg')),pitch_correction_frame='pan_local_y')
        # 프레임마다 BevFrontend 를 새로 만들고 있었다. 그 생성자는
        # cv2.initUndistortRectifyMap 으로 480x360 왜곡보정 맵을 통째로
        # 계산하는데, frontend.py:28 에 민석이가 직접 써놨듯이 그 맵은
        # **자세와 무관**하다. 그래서 update_projector() 가 있고 v2 노드는
        # 그걸 쓴다(v2/bev_frontend_node.py:590). v3 만 빠져 있었다.
        # 게다가 실차 yaml 의 D 는 전부 0이라 그 맵은 항등사상이다 --
        # 아무것도 안 바꾸는 맵을 매 프레임 다시 만들고 있었다.
        #
        # 자세를 타는 것은 BEV 소스 맵뿐이고, 그건 update_projector 가
        # 그대로 다시 만든다. 결과는 완전히 같다.
        t_a=time.perf_counter()
        projector=MetricGroundProjector(self.camera,self.grid,tf,float(p('ground_z')))
        if self.frontend is None:
            self.frontend=BevFrontend(self.camera,projector)
        else:
            self.frontend.update_projector(projector)
        t_b=time.perf_counter()
        out=self.frontend.process(self.bridge.imgmsg_to_cv2(msg,'bgr8'))
        t_c=time.perf_counter()
        # Publish the front-end image before the intentionally heavier
        # component graph extraction, so BEV diagnostics remain observable.
        if self.debug_wanted(self.bev_pub):
            self.bev_pub.publish(self.bridge.cv2_to_imgmsg(out.bev,'bgr8'))
            self.stats['bev_published'] += 1
        # include_overlay=False: draw_overlay 결과를 v3 는 안 쓴다.
        # 그리고 그건 컴포넌트마다 파이썬 픽셀 루프를 300회 돈다.
        seg=self.seg.process(out.bev,out.validity_mask>0,
                             include_overlay=False)
        t_d=time.perf_counter()
        # 한 프레임 치를 통째로 모았다가 끝에서 한 번에 갈아끼운다.
        # 앞서는 total 만 프레임 끝에서 넣는 바람에, 로그가 이전 프레임의
        # total 에서 이번 프레임의 head 를 빼서 tail 이 엉뚱하게 나왔다.
        pending_stage={'map':t_b-t_a,'remap':t_c-t_b,'seg':t_d-t_c,
                       'head':t_d-t_start}
        pending_stage.update(self.seg.last_times)
        items=[]
        for obs in seg.component_frame.observations:
            if obs.candidate is not None:
                poly=OrderedPolyline.from_points(obs.candidate.canonical_points); items.append(Component(obs.candidate.component_id,obs.candidate.color,poly,obs.candidate.support_length))
        orange_result=select_orange(items,RoleConfig())
        role=self.draw_roles(out.bev,items)
        # Proximity is now a coverage diagnostic/trigger.  A valid far
        # current measurement is retained so WHITE/history can recover its
        # near prefix; it is no longer discarded before stitching.
        orange_proximity_ok, orange_start_dist, proximity_reason = (
            validate_start(orange_result.path,
                           float(p('path_proximity.max_start_distance'))))
        if orange_result.valid:
            self.boundary_hold_count = 0
        elif self.previous_white is not None:
            self.boundary_hold_count += 1
            if self.boundary_hold_count > 30:
                self.previous_white = None
        whites=[item for item in items if item.color == 'WHITE']
        # Always derive the temporary center from this image's WHITE
        # measurements.  It may serve as the fallback or as a same-frame
        # prefix, but is never retained for a later frame.
        current_temporary = select_unknown_white(
            items, float(p('white.track_width')),
            reference_path=(orange_result.path
                            if orange_result.valid else None))
        current_hybrid=None
        if orange_result.valid and orange_result.path is not None:
            fragments=self.orange_fragments(items,orange_result)
            if bool(p('center_hybrid.enabled')) and fragments:
                white_points=(current_temporary.path.points
                              if current_temporary.valid
                              and current_temporary.path is not None else None)
                current_hybrid=stitch_current_frame(
                    orange_result.path.points,fragments,white_points,
                    self.center_hybrid_config)
            if current_hybrid is not None and current_hybrid.white_used:
                current_result=orange_result.__class__(
                    True,OrderedPolyline.from_points(current_hybrid.path),
                    CURRENT_HYBRID_ORANGE_WHITE,CENTER,
                    current_hybrid.reason,
                    orange_result.stitched_component_ids,
                    orange_result.bridged_gap_count)
            else:
                current_result=orange_result
        else:
            current_result=current_temporary
        if not orange_result.valid:
            # Frame-local unknown-boundary fallback.  It does not assign a
            # LEFT/RIGHT identity and never overrides an observed ORANGE path.
            self.get_logger().warning(
                'V3 MAGENTA fallback stamp=%d.%09d orange_reason=%s '
                'orange_components=%d orange_stitched=%d orange_start=%s '
                'white_components=%d fallback_valid=%s fallback_ids=%s' % (
                    stamp[0], stamp[1], orange_result.reason,
                    sum(x.color == 'ORANGE' for x in items),
                    len(orange_result.stitched_component_ids),
                    ('none' if orange_start_dist is None
                     else round(orange_start_dist, 4)),
                    sum(x.color == 'WHITE' for x in items),
                    current_result.valid,
                    current_result.stitched_component_ids))

        # Preserve the current measurement used by odom history.  Terminal
        # tangent/WHITE prediction is intentionally disabled: WHITE remains
        # available only for near-prefix and between-fragment stitching.
        current_measurement_result=current_result
        prediction=None

        current_start_dist=(None if not current_result.valid
                            or current_result.path is None else float(
                                np.linalg.norm(current_result.path.points[0])))
        current_path_min_dist=(None if not current_result.valid
                               or current_result.path is None else float(
                                   np.min(np.linalg.norm(
                                       current_result.path.points,axis=1))))
        result=current_result
        recovery=None
        odom_tf_error=None
        history_stored=False
        history_reference=np.empty((0,2),dtype=np.float64)
        base_from_odom_current=None
        now_stamp=self.stamp_seconds(msg.header.stamp)
        prior_history_entry=(
            None if not self.center_history.entries else
            self.center_history.entries[-1])
        if (current_result.valid and current_result.path is not None
                and bool(p('center_history.enabled'))):
            try:
                base_from_odom,odom_from_base=(
                    self.exact_odom_matrices(msg.header.stamp))
                base_from_odom_current=base_from_odom
                recovery=self.center_history.recover(
                    current_result.path.points,base_from_odom,now_stamp)
                if recovery.used:
                    result=current_result.__class__(
                        True,OrderedPolyline.from_points(recovery.path),
                        CURRENT_HYBRID_WITH_HISTORY_PREFIX,
                        current_result.role,
                        recovery.reason,current_result.stitched_component_ids,
                        current_result.bridged_gap_count)
                # Store only this frame's measured ORANGE/WHITE geometry.
                # Never recursively store the history-augmented final path.
                self.center_history.store(
                    current_measurement_result.path.points,
                    odom_from_base,now_stamp)
                history_stored=True
            except Exception as exc:
                odom_tf_error=str(exc)
        if (bool(p('center_history.enabled'))
                and prior_history_entry is not None
                and now_stamp-prior_history_entry.stamp
                <= self.center_hybrid_config.history_max_age):
            try:
                if base_from_odom_current is None:
                    base_from_odom_current,_=(
                        self.exact_odom_matrices(msg.header.stamp))
                history_reference=transform_xy(
                    prior_history_entry.points_odom,
                    base_from_odom_current)
            except Exception as exc:
                if odom_tf_error is None:
                    odom_tf_error=str(exc)

        final_start_dist=(None if not result.valid or result.path is None
                          else float(np.linalg.norm(result.path.points[0])))
        final_support=(None if not result.valid or result.path is None
                       else float(result.path.support))
        self.last_center_hybrid_diagnostic={
            'timestamp':now_stamp,
            'orange_fragment_count':int(sum(
                item.color == 'ORANGE' for item in items)),
            'orange_stitched_fragment_count':int(len(
                orange_result.stitched_component_ids)),
            'orange_near_coverage_sufficient':bool(orange_proximity_ok),
            'orange_start_distance':orange_start_dist,
            'current_white_temporary_valid':bool(current_temporary.valid),
            'current_white_temporary_reason':current_temporary.reason,
            'current_white_temporary_component_ids':list(
                current_temporary.stitched_component_ids),
            'current_hybrid_valid':bool(current_result.valid),
            'current_source':current_result.source,
            'current_path_start_distance':current_start_dist,
            'current_path_min_distance':current_path_min_dist,
            'current_white_used':bool(current_hybrid is not None
                                      and current_hybrid.white_used),
            'current_white_gap_bridge_count':int(
                0 if current_hybrid is None else
                current_hybrid.white_gap_bridge_count),
            'current_white_near_prefix_points':int(
                0 if current_hybrid is None else
                current_hybrid.white_near_prefix_points),
            'current_white_far_suffix_points':int(
                0 if current_hybrid is None else
                current_hybrid.white_far_suffix_points),
            'current_join_gaps':([] if current_hybrid is None else
                                 list(current_hybrid.join_gaps)),
            'current_tangent_differences':(
                [] if current_hybrid is None else
                list(current_hybrid.tangent_differences)),
            'history_available':bool(
                recovery is not None and recovery.history_available),
            'history_age':(None if recovery is None else
                           recovery.history_age),
            'history_point_count':int(
                0 if recovery is None else recovery.history_point_count),
            'transformed_history_point_count':int(
                0 if recovery is None else recovery.transformed_point_count),
            'history_prefix_point_count':int(
                0 if recovery is None else len(recovery.prefix)),
            'history_current_join_gap':(
                None if recovery is None else recovery.join_gap),
            'history_current_tangent_difference':(
                None if recovery is None else recovery.tangent_difference),
            'history_used':bool(recovery is not None and recovery.used),
            'history_reason':('HISTORY_DISABLED_OR_NO_CURRENT_PATH'
                              if recovery is None and odom_tf_error is None
                              else ('EXACT_ODOM_TF_UNAVAILABLE'
                                    if recovery is None else recovery.reason)),
            'history_stored':history_stored,
            'history_buffer_entries':len(self.center_history.entries),
            'history_tf_source_frame':'odom',
            'history_tf_target_frame':'base_footprint',
            'history_tf_fixed_frame':'odom',
            'history_tf_source_timestamp':(
                None if recovery is None or recovery.history_age is None
                else now_stamp-recovery.history_age),
            'history_tf_target_timestamp':now_stamp,
            'exact_odom_tf_error':odom_tf_error,
            'predicted_suffix_point_count':int(
                0 if prediction is None else len(prediction.suffix)),
            'predicted_suffix_length':(
                0.0 if prediction is None else prediction.length),
            'predicted_suffix_reason':(
                'PREDICTION_NOT_REQUESTED' if prediction is None
                else prediction.reason),
            'predicted_suffix_selected_source':(
                None if prediction is None else prediction.selected_source),
            'predicted_tangent_candidate_point_count':int(
                0 if prediction is None else
                len(prediction.tangent_candidate)),
            'predicted_white_candidate_point_count':int(
                0 if prediction is None else
                len(prediction.white_candidate)),
            'predicted_tangent_boundary_score':(
                None if prediction is None else
                prediction.tangent_boundary_score),
            'predicted_white_boundary_score':(
                None if prediction is None else
                prediction.white_boundary_score),
            'predicted_tangent_boundary_matches':int(
                0 if prediction is None else
                prediction.tangent_boundary_matches),
            'predicted_white_boundary_matches':int(
                0 if prediction is None else
                prediction.white_boundary_matches),
            'predicted_white_candidate_valid':bool(
                prediction is not None and
                prediction.white_candidate_valid),
            'predicted_white_candidate_reason':(
                None if prediction is None else
                prediction.white_candidate_reason),
            'predicted_white_candidate_join_angle':(
                None if prediction is None else
                prediction.white_candidate_join_angle),
            'final_source':result.source,
            'final_path_point_count':int(
                0 if not result.valid or result.path is None
                else len(result.path.points)),
            'final_start_distance':final_start_dist,
            'final_support':final_support,
        }
        if orange_result.valid and orange_result.path is not None:
            white_shadow=seed_from_center(orange_result.path, whites, float(p('white.expected_half_width')), float(p('white.half_width_tolerance')), self.previous_white)
        elif bool(p('white.reference_fallback_enabled')) and self.previous_white is not None:
            # Reference propagation is optional and remains a shadow-only
            # selection; the UNKNOWN path fallback remains available below.
            white_shadow=propagate(self.previous_white, whites)
        else:
            white_shadow=None
        if white_shadow is None and bool(p('white.reference_fallback_enabled')) and self.previous_white is not None:
            # Keep the last known reference visible even when this frame has
            # no usable WHITE continuation at all.
            white_shadow = self.previous_white
        if white_shadow is not None:
            # Keep the last valid side visible when a single propagation
            # attempt misses a fragmented WHITE observation.  This is only
            # debug/reference overlay; it is not used as a new measurement.
            old = self.previous_white
            if old is not None and (white_shadow.left is None or white_shadow.right is None):
                white_shadow = WhiteShadow(
                    labels=dict(white_shadow.labels),
                    left=white_shadow.left if white_shadow.left is not None else old.left,
                    right=white_shadow.right if white_shadow.right is not None else old.right,
                    ambiguous=white_shadow.ambiguous,
                    reason=white_shadow.reason,
                    diagnostics=white_shadow.diagnostics)
            # Preserve each side independently. A one-sided propagation
            # result must not erase the other side's last valid reference.
            old = self.previous_white
            if white_shadow.left is not None or white_shadow.right is not None:
                self.previous_white = WhiteShadow(
                    labels=dict(white_shadow.labels),
                    left=white_shadow.left if white_shadow.left is not None else (old.left if old is not None else None),
                    right=white_shadow.right if white_shadow.right is not None else (old.right if old is not None else None),
                    ambiguous=white_shadow.ambiguous,
                    reason=white_shadow.reason,
                    diagnostics=white_shadow.diagnostics)
            # Render the assembled side geometry, not only the seed component.
            # The latter made fragmented WHITE dashes look unconnected even
            # when _stitch_side had already produced a continuous chain.
            if white_shadow.left is not None:
                self.draw_polyline(role, white_shadow.left, (255,0,0), 2)
            if white_shadow.right is not None:
                self.draw_polyline(role, white_shadow.right, (0,0,255), 2)
        # Copy after shadow labels are drawn so path_overlay contains the
        # same blue/red propagation evidence as role_overlay.
        path=role.copy()
        self.stats['orange_processed'] += sum(item.color == 'ORANGE' for item in items)
        self.frame_count += 1
        if self.frame_count % 30 == 0:
            centers = [(item.component_id, round(item.support, 3), tuple(np.round(item.polyline.points[0], 3)), tuple(np.round(item.polyline.points[-1], 3))) for item in items if item.color == 'ORANGE']
            usable = len(centers)
            self.get_logger().info('V3 WHITE shadow total=%d labels=%s lateral=%s left=%s right=%s reason=%s' % (len(whites), dict(white_shadow.labels) if white_shadow is not None else {}, white_shadow.diagnostics if white_shadow is not None else {}, bool(white_shadow and white_shadow.left), bool(white_shadow and white_shadow.right), white_shadow.reason if white_shadow is not None else 'NO_SHADOW'))
            # py-spy 대신 노드가 직접 찍는다. 178 ms 가 어디로 가는지
            # 이름으로 보이게 하는 게 목적이다.
            st = self.stage
            self.get_logger().info(
                'V3 timing total=%.1fms | map=%.1f remap=%.1f '
                'seg=%.1f(hsv=%.1f extract=%.1f overlay=%.1f comp=%d) '
                'tail=%.1f'
                % (st.get('total',0)*1e3, st.get('map',0)*1e3,
                   st.get('remap',0)*1e3, st.get('seg',0)*1e3,
                   st.get('hsv',0)*1e3, st.get('extract',0)*1e3,
                   st.get('overlay',0)*1e3, st.get('components',0),
                   (st.get('total',0)-st.get('head',0))*1e3))
            diag = self.last_lidar_diagnostic or {}
            self.get_logger().info('V3 LiDAR image=%.9f scan=%s delta=%s frame=%s beams=%d valid=%d transformed=%d in_bounds=%d dropped_tf=%d tf_success=%s tf_error=%s overlays=%d no_pair=%d tf_wait=%d tf_fail=%d pending=%d replaced=%d' % (diag.get('image_stamp', 0.0), 'none' if diag.get('scan_stamp') is None else '%.9f' % diag['scan_stamp'], 'none' if diag.get('delta') is None else '%.6f' % diag['delta'], diag.get('scan_frame'), diag.get('total_beams', 0), diag.get('valid_ranges', 0), diag.get('transformed_points', 0), diag.get('in_bounds_points', 0), diag.get('dropped_tf_points', 0), diag.get('tf_success', False), diag.get('tf_error'), self.stats['lidar_overlay_published'], self.stats['lidar_no_pair'], self.stats['lidar_tf_wait'], self.stats['lidar_tf_failure'], len(self.lidar_pending), self.stats['lidar_pending_replaced']))
            self.get_logger().info('V3 stats images=%d immediate=%d pending=%d retry=%d eventual=%d timeout=%d replaced=%d processed=%d bev=%d overlays=%d orange=%d usable=%d stitched=%d bridges=%d current_start=%s final_start=%s orange_proximity=%s history_used=%s pending_now=%d' % (self.stats['images_received'], self.stats['immediate_tf_success'], self.stats['pending_enqueued'], self.stats['pending_retry_attempts'], self.stats['pending_eventual_success'], self.stats['pending_timeout'], self.stats['pending_replaced'], self.stats['frames_processed'], self.stats['bev_published'], self.stats['path_overlay_published'], self.stats['orange_processed'], usable, len(result.stitched_component_ids), result.bridged_gap_count, 'none' if current_start_dist is None else round(current_start_dist, 4), 'none' if final_start_dist is None else round(final_start_dist, 4), proximity_reason, bool(recovery is not None and recovery.used), len(self.pending)))
        if result.valid:
            path_color = ((0,255,0) if result.source in (
                DIRECT_CENTER_OBSERVED,CURRENT_HYBRID_ORANGE_WHITE,
                CURRENT_HYBRID_WITH_HISTORY_PREFIX) else (255,0,255))
            self.draw_path_points(path, result.path.points, path_color, 2)
        # role_image.copy() 로 시작해 그림만 그리고 끝난다. self 에
        # 대입하는 곳이 없어 호출째로 건너뛰어도 안전하다.
        if self.debug_wanted(self.center_hybrid_overlay_pub,
                             self.center_hybrid_diag_pub):
            self.publish_center_hybrid_debug(
                role,current_temporary,recovery,prediction,result,
                self.last_center_hybrid_diagnostic,msg.header.stamp)
        reference_path=(result.path.points if result.valid and result.path is not None
                        else np.empty((0,2),dtype=np.float64))
        # Publish the current center geometry before the same-stamp Stage 5.3
        # arbitration status.  This prevents a controller timer from seeing a
        # new CENTER decision before its required path has reached the cache.
        gp=Path(); gp.header.stamp=msg.header.stamp; gp.header.frame_id='base_footprint'
        if result.valid:
            for point in result.path.points:
                pose=PoseStamped(); pose.header=gp.header; pose.pose.position.x=float(point[0]); pose.pose.position.y=float(point[1]); pose.pose.orientation.w=1.0; gp.poses.append(pose)
        self.geometry_pub.publish(gp); b=Bool(); b.data=result.valid; self.valid_pub.publish(b); s=String(); s.data=result.source; self.source_pub.publish(s)
        current_white_components=tuple(
            WhiteComponentView(item.component_id,item.polyline.points)
            for item in whites)
        # tail 33.8 ms 의 주범. render_lidar_overlay 는 시작하자마자
        # expand_bev_canvas 를 두 번 부르고 한 번 더 copy() 한다. 결과가
        # 가는 곳은 오버레이 토픽 셋뿐이고, 여기서 갱신되는
        # last_lidar_diagnostic 도 30프레임마다 찍는 로그에만 쓰인다.
        if self.debug_wanted(self.lidar_bev_pub, self.path_lidar_pub,
                             self.avoidance_overlay_pub, self.lidar_diag_pub):
            lidar_overlays = self.render_lidar_overlay(
                out.bev,path,reference_path,msg.header.stamp,
                current_white_components=current_white_components,
                history_reference=history_reference)
            if (lidar_overlays is not None
                    and self.last_lidar_diagnostic.get('tf_success', False)):
                self.publish_lidar_overlay(
                    *lidar_overlays, msg.header.stamp)
        if self.debug_wanted(self.white_pub):
            self.white_pub.publish(self.bridge.cv2_to_imgmsg(seg.white_mask,'mono8'))
        if self.debug_wanted(self.orange_pub):
            self.orange_pub.publish(self.bridge.cv2_to_imgmsg(seg.orange_mask,'mono8'))
        if self.debug_wanted(self.role_pub):
            self.role_pub.publish(self.bridge.cv2_to_imgmsg(role,'bgr8'))
        if self.debug_wanted(self.path_pub):
            self.path_pub.publish(self.bridge.cv2_to_imgmsg(path,'bgr8'))
            self.stats['path_overlay_published'] += 1
        pending_stage['total']=time.perf_counter()-t_start
        self.stage=pending_stage

    def draw_roles(self, bev, items):
        """V3-only overlay: no inherited endpoint markers or stale colors."""
        image = bev.copy(); cfg = RoleConfig()
        colors = {CENTER:(0,255,255), LEFT:(255,180,0), RIGHT:(255,0,255)}
        for item in items:
            # WHITE is not classified by the legacy global-Y role colors in
            # STEP 3A; only path-relative propagation below may label it.
            if item.color != 'ORANGE':
                continue
            role = classify(item, cfg)
            if role not in colors:
                continue
            col,row = self.grid.metric_to_pixel(item.polyline.points[:,0], item.polyline.points[:,1])
            points = np.rint(np.c_[col,row]).astype(np.int32)
            cv2.polylines(image, [points], False, colors[role], 1)
        return image
    def draw_polyline(self, image, item, color, width):
        col,row=self.grid.metric_to_pixel(item.polyline.points[:,0],item.polyline.points[:,1])
        cv2.polylines(image,[np.rint(np.c_[col,row]).astype(np.int32)],False,color,width)
    def draw_path_points(self, image, points, color, width):
        col,row=self.grid.metric_to_pixel(points[:,0],points[:,1])
        cv2.polylines(image,[np.rint(np.c_[col,row]).astype(np.int32)],False,color,width)
def main(args=None):
    rclpy.init(args=args); node=V3Node(); rclpy.spin(node); node.destroy_node(); rclpy.shutdown()
