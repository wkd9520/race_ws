"""Minimal V3 ROS front-end: exact TF, dynamic metric BEV, direct CENTER."""
from collections import deque
import json
import time
import numpy as np, cv2, rclpy
from rclpy.node import Node
from rclpy.duration import Duration
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import Image, JointState, LaserScan
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String
from cv_bridge import CvBridge
import tf2_ros
from physicar_track_perception_v2.geometry import CameraModel, BevGrid, MetricGroundProjector, apply_projection_corrections
from physicar_track_perception_v2.frontend import BevFrontend
from physicar_track_perception_v2.components import CanonicalComponentExtractor, ComponentExtractionConfig
from physicar_track_perception_v2.segmentation import ColorComponentPipeline, HsvRange
from .geometry import OrderedPolyline
from .roles import Component, RoleConfig, classify, CENTER, LEFT, RIGHT
from .path_selector import (select_orange, select_unknown_white,
                            DIRECT_CENTER_OBSERVED, INVALID)
from .proximity import validate_start
from .white_propagation import WhiteShadow, seed_from_center, propagate, LEFT as WLEFT, RIGHT as WRIGHT
from .lidar_bev import (expand_bev_canvas, filter_bev_bounds,
                        scan_to_lidar_points, transform_matrix,
                        transform_points)

class V3Node(Node):
    def __init__(self):
        super().__init__('physicar_track_perception_v3')
        defaults={'camera.width':480,'camera.height':360,'camera.K':[201.38988018035889,0.,240.,0.,201.38988733291626,180.,0.,0.,1.],'camera.D':[-.045,-.0001,-.0003,-.0001,.001],'bev.x_min':.1,'bev.x_max':2.,'bev.y_min':-.75,'bev.y_max':.75,'bev.resolution':.01,'ground_z':0.,'sim_geometry.camera_height_correction_z':-.018,'projection.pitch_offset_deg':2.8,'tf_wait.max_pending_age':.25,'tf_wait.timer_period':.02,'path_proximity.max_start_distance':.60,'white.track_width':.70,'white.expected_half_width':.37,'white.half_width_tolerance':.10,'white.reference_fallback_enabled':True,'lidar.scan_topic':'/scan','lidar.fixed_frame':'odom','lidar.pair_slop':.03,'lidar.tf_timeout':.10,'lidar.overlay_radius_px':2,'lidar.overlay_x_min':-.5,'lidar.overlay_x_max':4.,'lidar.overlay_y_min':-2.,'lidar.overlay_y_max':2.,'lidar.path_overlay_color_bgr':[255,255,0]}
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
        self.bridge=CvBridge(); self.pending=[]; self.pending_replaced=0; self.tfbuf=tf2_ros.Buffer(cache_time=Duration(seconds=10)); tf2_ros.TransformListener(self.tfbuf,self,spin_thread=True)
        self.frame_count = 0
        self._last_stats_log = 0.0
        self.stats = {k: 0 for k in ('images_received','immediate_tf_success','pending_enqueued','pending_retry_attempts','pending_eventual_success','pending_timeout','pending_replaced','frames_processed','bev_published','orange_processed','path_overlay_published','duplicate_processed','lidar_scans_received','lidar_no_pair','lidar_tf_success','lidar_tf_failure','lidar_tf_wait','lidar_pending_replaced','lidar_overlay_published','path_lidar_overlay_published')}
        self.processed_stamps = set()
        self.previous_white = None
        self.boundary_hold_count = 0
        self.lidar_scans = deque(maxlen=3)
        self.lidar_pending = []
        self.last_lidar_diagnostic = None
        self.create_subscription(Image,'/camera/image_raw',self.image_cb,qos_profile_sensor_data); self.create_subscription(JointState,'/joint_states',self.joint_cb,qos_profile_sensor_data); self.create_subscription(LaserScan,str(p('lidar.scan_topic')),self.scan_cb,qos_profile_sensor_data); self.create_timer(float(p('tf_wait.timer_period')),self.retry)
        self.bev_pub=self.create_publisher(Image,'/perception_v3/debug/bev',2); self.lidar_bev_pub=self.create_publisher(Image,'/perception_v3/debug/bev_lidar_overlay',2); self.path_lidar_pub=self.create_publisher(Image,'/perception_v3/debug/path_lidar_overlay',2); self.lidar_diag_pub=self.create_publisher(String,'/perception_v3/debug/lidar_diagnostics',10); self.white_pub=self.create_publisher(Image,'/perception_v3/debug/white_mask',2); self.orange_pub=self.create_publisher(Image,'/perception_v3/debug/orange_mask',2); self.role_pub=self.create_publisher(Image,'/perception_v3/debug/role_overlay',2); self.path_pub=self.create_publisher(Image,'/perception_v3/debug/path_overlay',2); self.valid_pub=self.create_publisher(Bool,'/perception_v3/debug/path_valid',10); self.source_pub=self.create_publisher(String,'/perception_v3/debug/path_source',10); self.geometry_pub=self.create_publisher(Path,'/perception_v3/path',10)
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

    def enqueue_lidar_overlay(self, bev, path_bev, image_stamp, scan, delta):
        entry = (bev.copy(), path_bev.copy(), image_stamp, scan, delta,
                 time.monotonic())
        if len(self.lidar_pending) < 2:
            self.lidar_pending.append(entry)
        else:
            self.lidar_pending[-1] = entry
            self.stats['lidar_pending_replaced'] += 1

    def render_lidar_overlay(self, bev, path_bev, image_stamp,
                             scan=None, delta=None, allow_enqueue=True):
        overlay, _ = expand_bev_canvas(
            bev, self.grid, self.lidar_overlay_grid)
        path_overlay, _ = expand_bev_canvas(
            path_bev, self.grid, self.lidar_overlay_grid)
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
            return overlay, path_overlay
        diagnostic['scan_stamp'] = self.stamp_seconds(scan.header.stamp)
        diagnostic['scan_frame'] = scan.header.frame_id
        diagnostic['total_beams'] = len(scan.ranges)
        points_lidar, valid = scan_to_lidar_points(
            scan.ranges, scan.angle_min, scan.angle_increment,
            scan.range_min, scan.range_max)
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
                    bev, path_bev, image_stamp, scan, delta)
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
            cols, rows = self.lidar_overlay_grid.metric_to_pixel(
                in_bounds[:, 0], in_bounds[:, 1])
            radius = int(self.get_parameter('lidar.overlay_radius_px').value)
            for col, row in zip(cols, rows):
                pixel = (int(round(col)), int(round(row)))
                cv2.circle(overlay, pixel, radius, (0, 0, 255), -1,
                           cv2.LINE_AA)
                cv2.circle(path_overlay, pixel, radius + 1, (0, 0, 0), -1,
                           cv2.LINE_AA)
                cv2.circle(path_overlay, pixel, radius,
                           self.path_lidar_color, -1, cv2.LINE_AA)
        self.last_lidar_diagnostic = diagnostic
        return overlay, path_overlay

    def retry_lidar(self):
        max_age = float(self.get_parameter('tf_wait.max_pending_age').value)
        while self.lidar_pending:
            bev, path_bev, image_stamp, scan, delta, queued_at = self.lidar_pending[0]
            if time.monotonic() - queued_at > max_age:
                self.lidar_pending.pop(0)
                self.stats['lidar_tf_failure'] += 1
                continue
            overlays = self.render_lidar_overlay(
                bev, path_bev, image_stamp, scan, delta,
                allow_enqueue=False)
            if overlays is None:
                break
            self.lidar_pending.pop(0)
            self.publish_lidar_overlay(*overlays, image_stamp)

    def publish_lidar_overlay(self, overlay, path_overlay, image_stamp):
        message = self.bridge.cv2_to_imgmsg(overlay, 'bgr8')
        message.header.stamp = image_stamp
        message.header.frame_id = 'base_footprint'
        self.lidar_bev_pub.publish(message)
        path_message = self.bridge.cv2_to_imgmsg(path_overlay, 'bgr8')
        path_message.header.stamp = image_stamp
        path_message.header.frame_id = 'base_footprint'
        self.path_lidar_pub.publish(path_message)
        diagnostic = String()
        diagnostic.data = json.dumps(
            self.last_lidar_diagnostic, sort_keys=True)
        self.lidar_diag_pub.publish(diagnostic)
        self.stats['lidar_overlay_published'] += 1
        self.stats['path_lidar_overlay_published'] += 1
    def lookup(self,msg):
        stamp=Time.from_msg(msg.header.stamp)
        ready,_=self.tfbuf.can_transform('base_footprint','camera_optical_frame_corrected',stamp,timeout=Duration(seconds=0.0),return_debug_tuple=True)
        if not ready: raise RuntimeError('exact_tf_not_ready')
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
    def image_cb(self,msg):
        self.stats['images_received'] += 1
        try:
            tfmsg = self.lookup(msg)
            self.stats['immediate_tf_success'] += 1
            self.process(msg, tfmsg)
        except Exception as exc:
            self._enqueue(msg)
            self.get_logger().warning(f'V3 frame pending ({self._failure_kind(exc)}): {exc}')
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
        stamp=(msg.header.stamp.sec,msg.header.stamp.nanosec)
        if stamp in self.processed_stamps:
            self.stats['duplicate_processed'] += 1
            return
        self.processed_stamps.add(stamp)
        self.stats['frames_processed'] += 1
        p=lambda k:self.get_parameter(k).value; tf=apply_projection_corrections(self.matrix(tfmsg),camera_height_correction_z=float(p('sim_geometry.camera_height_correction_z')),pitch_offset_deg=float(p('projection.pitch_offset_deg')),pitch_correction_frame='pan_local_y'); out=BevFrontend(self.camera,MetricGroundProjector(self.camera,self.grid,tf,float(p('ground_z')))).process(self.bridge.imgmsg_to_cv2(msg,'bgr8'))
        # Publish the front-end image before the intentionally heavier
        # component graph extraction, so BEV diagnostics remain observable.
        self.bev_pub.publish(self.bridge.cv2_to_imgmsg(out.bev,'bgr8'))
        self.stats['bev_published'] += 1
        seg=self.seg.process(out.bev,out.validity_mask>0)
        items=[]
        for obs in seg.component_frame.observations:
            if obs.candidate is not None:
                poly=OrderedPolyline.from_points(obs.candidate.canonical_points); items.append(Component(obs.candidate.component_id,obs.candidate.color,poly,obs.candidate.support_length))
        result=select_orange(items,RoleConfig()); orange_result=result; role=self.draw_roles(out.bev,items)
        # Validate observed ORANGE before using it as the WHITE side
        # reference.  A far/noisy ORANGE component must not trigger
        # CENTER_SEED classification.
        proximity_ok, start_dist, proximity_reason = validate_start(result.path, float(p('path_proximity.max_start_distance')))
        path_min_dist = None if result.path is None else float(np.min(np.linalg.norm(result.path.points, axis=1)))
        if result.valid and not proximity_ok:
            result = result.__class__(False, None, INVALID, result.role, proximity_reason,
                                      result.stitched_component_ids, result.bridged_gap_count)
            orange_result = result
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
            items, float(p('white.track_width')))
        if not orange_result.valid:
            # Frame-local unknown-boundary fallback.  It does not assign a
            # LEFT/RIGHT identity and never overrides an observed ORANGE path.
            result=current_temporary
            self.get_logger().warning(
                'V3 MAGENTA fallback stamp=%d.%09d orange_reason=%s '
                'orange_components=%d orange_stitched=%d orange_start=%s '
                'white_components=%d fallback_valid=%s fallback_ids=%s' % (
                    stamp[0], stamp[1], orange_result.reason,
                    sum(x.color == 'ORANGE' for x in items),
                    len(orange_result.stitched_component_ids),
                    'none' if start_dist is None else round(start_dist, 4),
                    sum(x.color == 'WHITE' for x in items), result.valid,
                    result.stitched_component_ids))
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
            diag = self.last_lidar_diagnostic or {}
            self.get_logger().info('V3 LiDAR image=%.9f scan=%s delta=%s frame=%s beams=%d valid=%d transformed=%d in_bounds=%d dropped_tf=%d tf_success=%s tf_error=%s overlays=%d no_pair=%d tf_wait=%d tf_fail=%d pending=%d replaced=%d' % (diag.get('image_stamp', 0.0), 'none' if diag.get('scan_stamp') is None else '%.9f' % diag['scan_stamp'], 'none' if diag.get('delta') is None else '%.6f' % diag['delta'], diag.get('scan_frame'), diag.get('total_beams', 0), diag.get('valid_ranges', 0), diag.get('transformed_points', 0), diag.get('in_bounds_points', 0), diag.get('dropped_tf_points', 0), diag.get('tf_success', False), diag.get('tf_error'), self.stats['lidar_overlay_published'], self.stats['lidar_no_pair'], self.stats['lidar_tf_wait'], self.stats['lidar_tf_failure'], len(self.lidar_pending), self.stats['lidar_pending_replaced']))
            self.get_logger().info('V3 stats images=%d immediate=%d pending=%d retry=%d eventual=%d timeout=%d replaced=%d processed=%d bev=%d overlays=%d orange=%d usable=%d stitched=%d bridges=%d start_dist=%s path_min_dist=%s proximity=%s pending_now=%d' % (self.stats['images_received'], self.stats['immediate_tf_success'], self.stats['pending_enqueued'], self.stats['pending_retry_attempts'], self.stats['pending_eventual_success'], self.stats['pending_timeout'], self.stats['pending_replaced'], self.stats['frames_processed'], self.stats['bev_published'], self.stats['path_overlay_published'], self.stats['orange_processed'], usable, len(result.stitched_component_ids), result.bridged_gap_count, 'none' if start_dist is None else round(start_dist, 4), 'none' if path_min_dist is None else round(path_min_dist, 4), proximity_reason, len(self.pending)))
        if result.valid:
            path_color = ((0,255,0) if result.source == DIRECT_CENTER_OBSERVED
                          else (255,0,255))
            self.draw_path_points(path, result.path.points, path_color, 2)
        lidar_overlays = self.render_lidar_overlay(
            out.bev, path, msg.header.stamp)
        if (lidar_overlays is not None
                and self.last_lidar_diagnostic.get('tf_success', False)):
            self.publish_lidar_overlay(
                *lidar_overlays, msg.header.stamp)
        self.white_pub.publish(self.bridge.cv2_to_imgmsg(seg.white_mask,'mono8')); self.orange_pub.publish(self.bridge.cv2_to_imgmsg(seg.orange_mask,'mono8')); self.role_pub.publish(self.bridge.cv2_to_imgmsg(role,'bgr8')); self.path_pub.publish(self.bridge.cv2_to_imgmsg(path,'bgr8')); self.stats['path_overlay_published'] += 1
        gp=Path(); gp.header.stamp=msg.header.stamp; gp.header.frame_id='base_footprint'
        if result.valid:
            for point in result.path.points:
                pose=PoseStamped(); pose.header=gp.header; pose.pose.position.x=float(point[0]); pose.pose.position.y=float(point[1]); pose.pose.orientation.w=1.0; gp.poses.append(pose)
        self.geometry_pub.publish(gp); b=Bool(); b.data=result.valid; self.valid_pub.publish(b); s=String(); s.data=result.source; self.source_pub.publish(s)

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
