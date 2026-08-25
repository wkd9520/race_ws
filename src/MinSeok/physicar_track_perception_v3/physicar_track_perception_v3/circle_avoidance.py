"""Stage 5.1R shadow-only LiDAR component and circle avoidance geometry.

This module deliberately has no ROS dependency.  The input points must retain
LaserScan angular order and already be expressed in ``base_footprint`` at the
camera/BEV timestamp.  No angular sector, including the rear sector, is
discarded here.
"""

from dataclasses import dataclass
import math

import numpy as np

from .avoidance import (local_arc_tangents, max_adjacent_heading_change,
                        project_to_polyline)
from .geometry import cumulative_s


NORMAL = 'NORMAL'
VOTING = 'VOTING'
AVOID_LEFT = 'AVOID_LEFT'
AVOID_RIGHT = 'AVOID_RIGHT'
LEFT = 'LEFT'
RIGHT = 'RIGHT'
CENTER = 'CENTER'


@dataclass(frozen=True)
class CircleAvoidanceConfig:
    component_gap: float = 0.12
    max_obstacle_support: float = 0.70
    min_circle_points: int = 3
    min_circle_radius: float = 0.02
    max_circle_radius: float = 0.40
    max_circle_residual: float = 0.05
    path_near_distance: float = 0.20
    direction_freeze_distance: float = 1.5
    component_continuity_distance: float = 0.45
    default_avoidance_side: str = AVOID_LEFT
    safety_margin: float = 0.20
    additional_clearance: float = 0.05
    approach_length: float = 0.80
    return_length: float = 0.80
    tangent_window: float = 0.20
    resample_spacing: float = 0.05
    termination_rear_x: float = 0.0
    lookahead_distance: float = 0.70
    wheelbase: float = 0.18
    steering_limit: float = 0.3490658504

    def __post_init__(self):
        positive = np.asarray([
            self.component_gap, self.max_obstacle_support,
            self.min_circle_radius, self.max_circle_radius,
            self.max_circle_residual, self.path_near_distance,
            self.direction_freeze_distance,
            self.component_continuity_distance, self.safety_margin,
            self.approach_length, self.return_length, self.tangent_window,
            self.resample_spacing, self.lookahead_distance, self.wheelbase,
            self.steering_limit,
        ], dtype=np.float64)
        if not np.all(np.isfinite(positive)) or np.any(positive <= 0.0):
            raise ValueError('positive Stage 5.1R parameters must be finite and > 0')
        if (not np.isfinite(self.additional_clearance)
                or self.additional_clearance < 0.0):
            raise ValueError('additional_clearance must be finite and >= 0')
        if not np.isfinite(self.termination_rear_x):
            raise ValueError('termination_rear_x must be finite')
        if self.min_circle_points < 3:
            raise ValueError('min_circle_points must be at least 3')
        if self.max_circle_radius <= self.min_circle_radius:
            raise ValueError('max_circle_radius must exceed min_circle_radius')
        if self.default_avoidance_side not in (AVOID_LEFT, AVOID_RIGHT):
            raise ValueError('default_avoidance_side must be AVOID_LEFT or AVOID_RIGHT')


@dataclass(frozen=True)
class LidarComponent:
    component_id: int
    beam_indices: np.ndarray
    points: np.ndarray
    point_count: int
    nearest_distance: float
    centroid: np.ndarray
    span: float
    support: float
    wall_like: bool


@dataclass(frozen=True)
class CircleFit:
    component_id: int
    valid: bool
    reason: str
    center: np.ndarray | None
    radius: float | None
    residual: float | None
    point_count: int
    support: float


@dataclass(frozen=True)
class PathRelativeCircle:
    component: LidarComponent
    fit: CircleFit
    nearest_path_point: np.ndarray
    tangent: np.ndarray
    normal_left: np.ndarray
    s_obstacle: float
    signed_lateral: float
    path_center_distance: float
    path_surface_distance: float
    vehicle_center_distance: float


@dataclass(frozen=True)
class CircleAvoidanceResult:
    active: bool
    reason: str
    state: str
    direction_locked: bool
    locked_avoidance_side: str | None
    lock_reason: str | None
    instantaneous_side: str | None
    left_votes: int
    right_votes: int
    original: np.ndarray
    shadow_path: np.ndarray
    components: tuple
    fits: tuple
    selected: PathRelativeCircle | None
    safety_radius: float | None
    target: np.ndarray | None
    target_lateral_offset: float | None
    clearance_original: float | None
    clearance_avoidance: float | None
    safety_clearance_avoidance: float | None
    max_heading_step_original: float | None
    max_heading_step_avoidance: float | None
    steering_original: dict | None
    steering_avoidance: dict | None


def _points_xy(values, name='points', minimum=0):
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f'{name} must have shape (N,2), N>={minimum}')
    if not np.all(np.isfinite(points)):
        raise ValueError(f'{name} must be finite')
    return points


def build_components(points_xy, beam_indices, config=CircleAvoidanceConfig()):
    """Split angular-order scan points at beam discontinuities or metric gaps."""
    points = _points_xy(points_xy)
    beams = np.asarray(beam_indices, dtype=np.int64).reshape(-1)
    if len(points) != len(beams):
        raise ValueError('points and beam_indices lengths differ')
    if len(beams) and np.any(np.diff(beams) <= 0):
        raise ValueError('beam_indices must be strictly increasing')
    if not len(points):
        return tuple()

    split = [0]
    for index in range(1, len(points)):
        gap = float(np.linalg.norm(points[index] - points[index - 1]))
        if beams[index] != beams[index - 1] + 1 or gap > config.component_gap:
            split.append(index)
    split.append(len(points))

    components = []
    for component_id, (begin, end) in enumerate(zip(split[:-1], split[1:])):
        component_points = points[begin:end].copy()
        component_beams = beams[begin:end].copy()
        if len(component_points) > 1:
            support = float(np.linalg.norm(
                np.diff(component_points, axis=0), axis=1).sum())
            span = float(np.linalg.norm(
                component_points[-1] - component_points[0]))
        else:
            support = 0.0
            span = 0.0
        components.append(LidarComponent(
            component_id=component_id,
            beam_indices=component_beams,
            points=component_points,
            point_count=len(component_points),
            nearest_distance=float(np.min(
                np.linalg.norm(component_points, axis=1))),
            centroid=np.mean(component_points, axis=0),
            span=span,
            support=support,
            wall_like=bool(support >= config.max_obstacle_support),
        ))
    return tuple(components)


def fit_circle(component, config=CircleAvoidanceConfig()):
    """Algebraic least-squares circle fit with minimal sanity checks."""
    points = component.points
    base = dict(component_id=component.component_id,
                point_count=component.point_count,
                support=component.support)
    if component.wall_like:
        return CircleFit(valid=False, reason='WALL_LIKE', center=None,
                         radius=None, residual=None, **base)
    if len(points) < config.min_circle_points:
        return CircleFit(valid=False, reason='INSUFFICIENT_POINTS',
                         center=None, radius=None, residual=None, **base)
    matrix = np.column_stack((2.0 * points[:, 0],
                              2.0 * points[:, 1],
                              np.ones(len(points))))
    rhs = np.sum(np.square(points), axis=1)
    solution, _, rank, _ = np.linalg.lstsq(matrix, rhs, rcond=None)
    if rank < 3 or not np.all(np.isfinite(solution)):
        return CircleFit(valid=False, reason='DEGENERATE_FIT', center=None,
                         radius=None, residual=None, **base)
    center = solution[:2]
    radius_squared = float(solution[2] + np.dot(center, center))
    if not np.isfinite(radius_squared) or radius_squared <= 0.0:
        return CircleFit(valid=False, reason='INVALID_RADIUS', center=center,
                         radius=None, residual=None, **base)
    radius = math.sqrt(radius_squared)
    radial_error = np.linalg.norm(points - center, axis=1) - radius
    residual = float(np.sqrt(np.mean(np.square(radial_error))))
    if not np.isfinite(radius) or not np.isfinite(residual):
        return CircleFit(valid=False, reason='NONFINITE_FIT', center=center,
                         radius=radius, residual=residual, **base)
    if radius < config.min_circle_radius or radius > config.max_circle_radius:
        return CircleFit(valid=False, reason='RADIUS_OUT_OF_RANGE',
                         center=center, radius=radius, residual=residual,
                         **base)
    if residual > config.max_circle_residual:
        return CircleFit(valid=False, reason='RESIDUAL_TOO_LARGE',
                         center=center, radius=radius, residual=residual,
                         **base)
    return CircleFit(valid=True, reason='VALID', center=center,
                     radius=radius, residual=residual, **base)


def analyze_components(points_xy, beam_indices,
                       config=CircleAvoidanceConfig()):
    components = build_components(points_xy, beam_indices, config)
    fits = tuple(fit_circle(component, config) for component in components)
    return components, fits


def _resample_with_arc(path_points, spacing, extra_s=None):
    path = _points_xy(path_points, 'path_points', minimum=2)
    path_s = cumulative_s(path)
    total = float(path_s[-1])
    if total <= 1e-9:
        raise ValueError('path must contain non-degenerate geometry')
    sample_s = np.arange(0.0, total, float(spacing), dtype=np.float64)
    sample_s = np.r_[sample_s, total]
    if extra_s is not None:
        sample_s = np.r_[sample_s, np.clip(float(extra_s), 0.0, total)]
    sample_s = np.unique(np.round(sample_s, 12))
    points = np.column_stack((
        np.interp(sample_s, path_s, path[:, 0]),
        np.interp(sample_s, path_s, path[:, 1]),
    ))
    return points, sample_s


def classify_circle(center, path_points, config=CircleAvoidanceConfig()):
    """Return a stable local path frame and signed lane side for a circle."""
    path, path_s = _resample_with_arc(path_points, config.resample_spacing)
    projection = project_to_polyline(np.asarray(center)[None, :], path)
    raw_arc = float(projection['s'][0])
    index = int(np.argmin(np.abs(path_s - raw_arc)))
    tangent = local_arc_tangents(path, config.tangent_window)[index]
    normal_left = np.array([-tangent[1], tangent[0]], dtype=np.float64)
    # The raw nearest segment may be a millimetre-scale dash connector whose
    # tangent is not representative of the ordered path. Project onto the
    # stable local tangent line through the resampled anchor instead.
    anchor = path[index]
    along = float(np.dot(np.asarray(center) - anchor, tangent))
    nearest = anchor + along * tangent
    arc = float(np.clip(path_s[index] + along, 0.0, path_s[-1]))
    lateral = float(np.dot(np.asarray(center) - nearest, normal_left))
    if lateral > 1e-9:
        side = LEFT
    elif lateral < -1e-9:
        side = RIGHT
    else:
        side = CENTER
    return {
        'nearest': nearest,
        'tangent': tangent,
        'normal_left': normal_left,
        's': arc,
        'signed_lateral': lateral,
        'side': side,
        'distance': float(projection['distance'][0]),
    }


def relevant_circles(components, fits, path_points,
                     config=CircleAvoidanceConfig()):
    by_id = {component.component_id: component for component in components}
    result = []
    for fit in fits:
        if not fit.valid:
            continue
        frame = classify_circle(fit.center, path_points, config)
        surface_distance = max(0.0, frame['distance'] - fit.radius)
        if surface_distance > config.path_near_distance:
            continue
        component = by_id[fit.component_id]
        result.append(PathRelativeCircle(
            component=component,
            fit=fit,
            nearest_path_point=frame['nearest'],
            tangent=frame['tangent'],
            normal_left=frame['normal_left'],
            s_obstacle=frame['s'],
            signed_lateral=frame['signed_lateral'],
            path_center_distance=frame['distance'],
            path_surface_distance=surface_distance,
            vehicle_center_distance=float(np.linalg.norm(fit.center)),
        ))
    return tuple(sorted(result,
                        key=lambda item: item.vehicle_center_distance))


class DirectionLatch:
    """Minimal vote/freeze/latch state; no obstacle tracking is performed."""

    def __init__(self, config=CircleAvoidanceConfig()):
        self.config = config
        self.reset()

    @property
    def locked(self):
        return self.locked_side is not None

    def reset(self):
        self.state = NORMAL
        self.left_votes = 0
        self.right_votes = 0
        self.locked_side = None
        self.lock_reason = None

    def reset_votes(self):
        self.state = VOTING
        self.left_votes = 0
        self.right_votes = 0
        self.locked_side = None
        self.lock_reason = None

    def _tie_break(self, left_clearance, right_clearance):
        if (left_clearance is not None and right_clearance is not None
                and np.isfinite(left_clearance)
                and np.isfinite(right_clearance)
                and not np.isclose(left_clearance, right_clearance)):
            return ((AVOID_LEFT if left_clearance > right_clearance
                     else AVOID_RIGHT), 'TIE_TRACK_CLEARANCE')
        return self.config.default_avoidance_side, 'TIE_DETERMINISTIC_DEFAULT'

    def observe(self, instantaneous_side, obstacle_distance,
                left_clearance=None, right_clearance=None):
        if self.locked:
            return self.locked_side
        if self.state == NORMAL:
            self.state = VOTING
        distance = float(obstacle_distance)
        if distance > self.config.direction_freeze_distance:
            if instantaneous_side == LEFT:
                self.left_votes += 1
            elif instantaneous_side == RIGHT:
                self.right_votes += 1
            return None

        if self.left_votes > self.right_votes:
            self.locked_side = AVOID_RIGHT
            self.lock_reason = 'LEFT_OBSTACLE_VOTE_MAJORITY'
        elif self.right_votes > self.left_votes:
            self.locked_side = AVOID_LEFT
            self.lock_reason = 'RIGHT_OBSTACLE_VOTE_MAJORITY'
        else:
            self.locked_side, self.lock_reason = self._tie_break(
                left_clearance, right_clearance)
        self.state = self.locked_side
        return self.locked_side


def avoidance_target(selected, locked_side, config=CircleAvoidanceConfig()):
    if locked_side not in (AVOID_LEFT, AVOID_RIGHT):
        raise ValueError('avoidance target requires a locked side')
    safety_radius = selected.fit.radius + config.safety_margin
    target_radius = safety_radius + config.additional_clearance
    sign = 1.0 if locked_side == AVOID_LEFT else -1.0
    target = (selected.fit.center
              + sign * target_radius * selected.normal_left)
    lateral_offset = float(np.dot(
        target - selected.nearest_path_point, selected.normal_left))
    return target, safety_radius, lateral_offset


def build_shadow_path(path_points, selected, target,
                      config=CircleAvoidanceConfig()):
    path, path_s = _resample_with_arc(
        path_points, config.resample_spacing, selected.s_obstacle)
    tangents = local_arc_tangents(path, config.tangent_window)
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    obstacle_index = int(np.argmin(np.abs(path_s - selected.s_obstacle)))
    obstacle_normal = normals[obstacle_index]
    desired_offset = float(np.dot(
        target - path[obstacle_index], obstacle_normal))
    weights = np.zeros(len(path), dtype=np.float64)
    before = path_s <= selected.s_obstacle
    before_distance = selected.s_obstacle - path_s[before]
    before_valid = before_distance < config.approach_length
    before_indices = np.flatnonzero(before)[before_valid]
    weights[before_indices] = 0.5 * (1.0 + np.cos(
        np.pi * before_distance[before_valid] / config.approach_length))
    after = path_s > selected.s_obstacle
    after_distance = path_s[after] - selected.s_obstacle
    after_valid = after_distance < config.return_length
    after_indices = np.flatnonzero(after)[after_valid]
    weights[after_indices] = 0.5 * (1.0 + np.cos(
        np.pi * after_distance[after_valid] / config.return_length))
    shadow = path + weights[:, None] * desired_offset * normals
    return shadow, weights


def pure_pursuit_shadow(path_points, config=CircleAvoidanceConfig()):
    """Exact controller-v3 lookahead/Pure Pursuit equations for diagnostics."""
    points = _points_xy(path_points, 'path_points')
    if not len(points):
        return {'target': None, 'distance': 0.0, 'alpha': 0.0,
                'steering_raw': 0.0, 'steering_saturated': 0.0}
    previous = np.zeros(2, dtype=np.float64)
    total = 0.0
    target = points[-1].copy()
    target_distance = total
    for point in points:
        segment = float(np.linalg.norm(point - previous))
        if total + segment >= config.lookahead_distance:
            fraction = (0.0 if segment == 0.0
                        else (config.lookahead_distance - total) / segment)
            target = previous + fraction * (point - previous)
            target_distance = config.lookahead_distance
            break
        total += segment
        target_distance = total
        previous = point
    alpha = math.atan2(float(target[1]), float(target[0]))
    distance = float(np.linalg.norm(target))
    raw = math.atan2(2.0 * config.wheelbase * math.sin(alpha),
                     max(distance, 1e-9))
    saturated = max(-config.steering_limit,
                    min(config.steering_limit, raw))
    return {
        'target': target.tolist(),
        'distance': float(target_distance),
        'alpha': float(alpha),
        'steering_raw': float(raw),
        'steering_saturated': float(saturated),
    }


class CircleAvoidanceEngine:
    """Single relevant-obstacle Stage 5.1R shadow state machine."""

    def __init__(self, config=CircleAvoidanceConfig()):
        self.config = config
        self.latch = DirectionLatch(config)
        self.previous_center = None
        self.last_measurement_key = None

    def reset(self):
        self.latch.reset()
        self.previous_center = None
        self.last_measurement_key = None

    def _match_locked(self, components, fits, path_points):
        if self.previous_center is None:
            return None
        candidates = relevant_circles(components, fits, path_points,
                                      self.config)
        valid = list(candidates)
        # Once locked, retain rear observations even though the current front
        # path no longer makes them path-relevant.
        candidate_ids = {item.fit.component_id for item in valid}
        by_id = {component.component_id: component for component in components}
        for fit in fits:
            if not fit.valid or fit.component_id in candidate_ids:
                continue
            frame = classify_circle(fit.center, path_points, self.config)
            valid.append(PathRelativeCircle(
                by_id[fit.component_id], fit, frame['nearest'],
                frame['tangent'], frame['normal_left'], frame['s'],
                frame['signed_lateral'], frame['distance'],
                max(0.0, frame['distance'] - fit.radius),
                float(np.linalg.norm(fit.center))))
        if not valid:
            return None
        selected = min(valid, key=lambda item: float(np.linalg.norm(
            item.fit.center - self.previous_center)))
        separation = float(np.linalg.norm(
            selected.fit.center - self.previous_center))
        return (selected if separation <= self.config.component_continuity_distance
                else None)

    def process(self, points_xy, beam_indices, path_points,
                measurement_key=None, left_clearance=None,
                right_clearance=None):
        reference = _points_xy(path_points, 'path_points')
        components, fits = analyze_components(
            points_xy, beam_indices, self.config)
        original_heading = (max_adjacent_heading_change(reference)
                            if len(reference) >= 2 else None)
        new_measurement = (measurement_key is None
                           or measurement_key != self.last_measurement_key)
        if new_measurement:
            self.last_measurement_key = measurement_key

        def result(reason, selected=None, shadow=None, target=None,
                   safety_radius=None, lateral_offset=None,
                   clearance_original=None, clearance_avoidance=None,
                   safety_clearance=None):
            output_path = (reference.copy() if shadow is None else shadow)
            instant = None
            if selected is not None:
                instant = (LEFT if selected.signed_lateral > 1e-9 else
                           RIGHT if selected.signed_lateral < -1e-9 else CENTER)
            return CircleAvoidanceResult(
                active=bool(shadow is not None), reason=reason,
                state=self.latch.state,
                direction_locked=self.latch.locked,
                locked_avoidance_side=self.latch.locked_side,
                lock_reason=self.latch.lock_reason,
                instantaneous_side=instant,
                left_votes=self.latch.left_votes,
                right_votes=self.latch.right_votes,
                original=reference.copy(), shadow_path=output_path,
                components=components, fits=fits, selected=selected,
                safety_radius=safety_radius, target=target,
                target_lateral_offset=lateral_offset,
                clearance_original=clearance_original,
                clearance_avoidance=clearance_avoidance,
                safety_clearance_avoidance=safety_clearance,
                max_heading_step_original=original_heading,
                max_heading_step_avoidance=(
                    max_adjacent_heading_change(output_path)
                    if len(output_path) >= 2 else None),
                steering_original=(pure_pursuit_shadow(reference, self.config)
                                   if len(reference) else None),
                steering_avoidance=(pure_pursuit_shadow(output_path, self.config)
                                    if len(output_path) else None),
            )

        if len(reference) < 2:
            return result('NO_VALID_REFERENCE_PATH')

        if self.latch.locked:
            selected = self._match_locked(components, fits, reference)
            if selected is None:
                return result('LOCKED_OBSTACLE_NOT_OBSERVED')
            if new_measurement and selected.fit.center[0] < self.config.termination_rear_x:
                self.reset()
                return result('AVOIDANCE_COMPLETE_REAR_PASS')
        else:
            candidates = relevant_circles(
                components, fits, reference, self.config)
            if not candidates:
                return result('NO_RELEVANT_CIRCLE')
            selected = candidates[0]
            if (self.previous_center is not None and self.latch.state == VOTING
                    and float(np.linalg.norm(selected.fit.center
                                             - self.previous_center))
                    > self.config.component_continuity_distance):
                self.latch.reset_votes()
            self.previous_center = selected.fit.center.copy()
            if new_measurement:
                instant = (LEFT if selected.signed_lateral > 1e-9 else
                           RIGHT if selected.signed_lateral < -1e-9 else CENTER)
                self.latch.observe(
                    instant, selected.vehicle_center_distance,
                    left_clearance, right_clearance)
            if not self.latch.locked:
                return result('VOTING', selected=selected)

        self.previous_center = selected.fit.center.copy()
        target, safety_radius, lateral_offset = avoidance_target(
            selected, self.latch.locked_side, self.config)
        shadow, _ = build_shadow_path(
            reference, selected, target, self.config)
        original_center_distance = selected.path_center_distance
        avoidance_center_distance = float(project_to_polyline(
            selected.fit.center[None, :], shadow)['distance'][0])
        clearance_original = original_center_distance - selected.fit.radius
        clearance_avoidance = avoidance_center_distance - selected.fit.radius
        safety_clearance = avoidance_center_distance - safety_radius
        return result(
            'LOCKED_ACTIVE', selected=selected, shadow=shadow,
            target=target, safety_radius=safety_radius,
            lateral_offset=lateral_offset,
            clearance_original=clearance_original,
            clearance_avoidance=clearance_avoidance,
            safety_clearance=safety_clearance)
