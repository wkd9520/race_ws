"""ROS-independent Stage 5.2B active-obstacle lifecycle.

Track voting and direction locks remain owned by :mod:`obstacle_tracks`.
This module only selects/latches one active track and evaluates termination
from that track's current, associated raw LaserScan component.
"""

from dataclasses import dataclass
import math

import numpy as np


NO_ACTIVE = 'NO_ACTIVE'
ACTIVE = 'ACTIVE'
ACTIVE_LOST = 'ACTIVE_LOST'

ACTIVE_SELECTED = 'ACTIVE_SELECTED'
ACTIVE_LATCHED = 'ACTIVE_LATCHED'
ACTIVE_SCAN_MISSING = 'ACTIVE_SCAN_MISSING'
ACTIVE_TERMINATED = 'ACTIVE_TERMINATED'
ACTIVE_LOST_RELEASED = 'ACTIVE_LOST_RELEASED'
NEXT_ACTIVE_SELECTED = 'NEXT_ACTIVE_SELECTED'


# Live source geometry:
# /opt/physicar/src/physicar-ros/physicar_description/urdf/
# physicar.urdf.xacro and physicar_macros.xacro
PHYSICAR_BODY_LENGTH = 0.27
PHYSICAR_BODY_WIDTH = 0.09
PHYSICAR_WHEELBASE = 0.18
PHYSICAR_TRACK_WIDTH = 0.16
PHYSICAR_WHEEL_RADIUS = 0.0375
PHYSICAR_WHEEL_WIDTH = 0.035
PHYSICAR_STEERING_LIMIT = 0.3491


def physicar_collision_footprint_vertices(steering_angle=0.0):
    """Return planar body/wheel collision extrema in ``base_footprint``.

    The body collision is a box. Wheel cylinders have their axes along local
    Y, so their planar extrema are the four corners (radius, half-width) after
    applying the front steering rotation.
    """
    angle = float(steering_angle)
    if not np.isfinite(angle) or abs(angle) > PHYSICAR_STEERING_LIMIT + 1e-12:
        raise ValueError('steering_angle is outside the URDF limit')
    body_half = np.array(
        [PHYSICAR_BODY_LENGTH / 2.0, PHYSICAR_BODY_WIDTH / 2.0])
    local_wheel = np.array([
        [sx * PHYSICAR_WHEEL_RADIUS, sy * PHYSICAR_WHEEL_WIDTH / 2.0]
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)], dtype=np.float64)
    vertices = [
        [sx * body_half[0], sy * body_half[1]]
        for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
    for x in (-PHYSICAR_WHEELBASE / 2.0,):
        for y in (-PHYSICAR_TRACK_WIDTH / 2.0,
                  PHYSICAR_TRACK_WIDTH / 2.0):
            vertices.extend((local_wheel + [x, y]).tolist())
    rotation = np.array([
        [math.cos(angle), -math.sin(angle)],
        [math.sin(angle), math.cos(angle)]], dtype=np.float64)
    steered_wheel = local_wheel @ rotation.T
    for x in (PHYSICAR_WHEELBASE / 2.0,):
        for y in (-PHYSICAR_TRACK_WIDTH / 2.0,
                  PHYSICAR_TRACK_WIDTH / 2.0):
            vertices.extend((steered_wheel + [x, y]).tolist())
    return np.asarray(vertices, dtype=np.float64)


def physicar_robot_bounding_radius():
    """Exact origin-centred planar bound for all URDF collision geometry."""
    body_corner = math.hypot(
        PHYSICAR_BODY_LENGTH / 2.0, PHYSICAR_BODY_WIDTH / 2.0)
    rear_wheel_corner = math.hypot(
        PHYSICAR_WHEELBASE / 2.0 + PHYSICAR_WHEEL_RADIUS,
        PHYSICAR_TRACK_WIDTH / 2.0 + PHYSICAR_WHEEL_WIDTH / 2.0)
    # The front wheel can steer until its outer corner is radially aligned
    # with the wheel-centre vector, so this continuous-angle envelope is the
    # exact maximum rather than an endpoint-only sample.
    front_wheel_envelope = (
        math.hypot(PHYSICAR_WHEELBASE / 2.0,
                   PHYSICAR_TRACK_WIDTH / 2.0)
        + math.hypot(PHYSICAR_WHEEL_RADIUS,
                     PHYSICAR_WHEEL_WIDTH / 2.0))
    # LiDAR, pan/tilt and their collision boxes/cylinder are all within this
    # wheel-dominated envelope in the live URDF.
    return max(body_corner, rear_wheel_corner, front_wheel_envelope)


PHYSICAR_ROBOT_BOUNDING_RADIUS = physicar_robot_bounding_radius()


@dataclass(frozen=True)
class SurfaceTermination:
    component_present: bool
    point_count: int
    distance_surface: float | None
    max_x: float | None
    passed: bool
    surface_clear: bool
    termination: bool


@dataclass(frozen=True)
class LostActiveRelease:
    center_present: bool
    path_valid: bool
    center_base: tuple | None
    distance_to_center: float | None
    release_distance: float
    vehicle_s: float | None
    obstacle_s: float | None
    progress_delta: float | None
    path_passed: bool
    distance_clear: bool
    release: bool


def evaluate_surface_termination(points_base, robot_radius):
    """Evaluate passed-and-clear termination from one raw scan component."""
    radius = float(robot_radius)
    if not np.isfinite(radius) or radius <= 0.0:
        raise ValueError('robot_radius must be finite and > 0')
    if points_base is None:
        return SurfaceTermination(False, 0, None, None, False, False, False)
    points = np.asarray(points_base, dtype=np.float64)
    if points.size == 0:
        return SurfaceTermination(False, 0, None, None, False, False, False)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('component points must have shape (N,2)')
    if not np.all(np.isfinite(points)):
        raise ValueError('component points must be finite')
    distance = float(np.min(np.linalg.norm(points, axis=1)))
    max_x = float(np.max(points[:, 0]))
    passed = max_x < 0.0
    surface_clear = distance > radius
    return SurfaceTermination(
        True, len(points), distance, max_x, passed, surface_clear,
        passed and surface_clear)


def _path_progress(query, path_points):
    """Return ordered-polyline progress with endpoint tangent extension."""
    query = np.asarray(query, dtype=np.float64).reshape(2)
    path = np.asarray(path_points, dtype=np.float64)
    if (path.ndim != 2 or path.shape[1] != 2 or len(path) < 2
            or not np.all(np.isfinite(path)) or not np.all(np.isfinite(query))):
        raise ValueError('path progress requires finite path shape (N,2), N>=2')
    keep = np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 1e-9]
    path = path[keep]
    if len(path) < 2:
        raise ValueError('path progress requires a non-degenerate segment')
    segment = np.diff(path, axis=0)
    length = np.linalg.norm(segment, axis=1)
    tangent = segment / length[:, None]
    cumulative = np.r_[0.0, np.cumsum(length)]
    relative = query[None, :] - path[:-1]
    raw_fraction = np.sum(relative * segment, axis=1) / np.square(length)
    fraction = np.clip(raw_fraction, 0.0, 1.0)
    projected = path[:-1] + fraction[:, None] * segment
    distance_squared = np.sum(np.square(projected-query), axis=1)
    index = int(np.argmin(distance_squared))
    if index == 0 and fraction[index] <= 1e-12:
        extension = float(np.dot(query-path[0], tangent[0]))
        if extension < 0.0:
            return extension
    last = len(segment)-1
    if index == last and fraction[index] >= 1.0-1e-12:
        extension = float(np.dot(query-path[-1], tangent[-1]))
        if extension > 0.0:
            return float(cumulative[-1]+extension)
    return float(cumulative[index]+fraction[index]*length[index])


def evaluate_lost_active_release(center_base, path_points, robot_radius,
                                 radius_multiplier=1.7):
    """Evaluate a conservative path-relative release for a lost ACTIVE.

    ``center_base`` must be the last ACTIVE odom center reprojected into
    ``base_footprint`` at the current measurement time.  This is deliberately
    separate from the raw-surface normal termination contract.
    """
    radius = float(robot_radius)
    multiplier = float(radius_multiplier)
    if (not np.isfinite(radius) or radius <= 0.0
            or not np.isfinite(multiplier) or multiplier <= 1.0):
        raise ValueError('lost release radius and multiplier are invalid')
    release_distance = radius*multiplier
    if center_base is None:
        return LostActiveRelease(
            False, False, None, None, release_distance,
            None, None, None, False, False, False)
    center = np.asarray(center_base, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(center)):
        raise ValueError('lost ACTIVE center must be finite')
    distance = float(np.linalg.norm(center))
    distance_clear = distance > release_distance
    try:
        vehicle_s = _path_progress((0.0, 0.0), path_points)
        obstacle_s = _path_progress(center, path_points)
    except ValueError:
        return LostActiveRelease(
            True, False, tuple(float(x) for x in center), distance,
            release_distance, None, None, None, False, distance_clear, False)
    progress_delta = float(vehicle_s-obstacle_s)
    path_passed = progress_delta > 0.0
    return LostActiveRelease(
        True, True, tuple(float(x) for x in center), distance,
        release_distance, vehicle_s, obstacle_s, progress_delta,
        path_passed, distance_clear, path_passed and distance_clear)


@dataclass(frozen=True)
class ActiveTrackView:
    track_id: int
    center_base: tuple
    distance_to_vehicle: float
    vote_count: int
    direction_locked: bool
    locked_avoidance_side: str | None
    observed_this_frame: bool
    current_relevant: bool
    current_component_id: int | None
    center_odom: tuple | None = None

    def __post_init__(self):
        center = np.asarray(self.center_base, dtype=np.float64).reshape(2)
        if (not np.all(np.isfinite(center))
                or not np.isfinite(self.distance_to_vehicle)
                or self.distance_to_vehicle < 0.0
                or isinstance(self.vote_count, bool)
                or not isinstance(self.vote_count, (int, np.integer))
                or self.vote_count < 0):
            raise ValueError('active track view geometry is invalid')
        object.__setattr__(self, 'center_base', tuple(float(x) for x in center))
        if self.center_odom is not None:
            odom = np.asarray(self.center_odom, dtype=np.float64).reshape(2)
            if not np.all(np.isfinite(odom)):
                raise ValueError('active track odom center is invalid')
            object.__setattr__(
                self, 'center_odom', tuple(float(x) for x in odom))


@dataclass(frozen=True)
class ActiveLifecycleResult:
    state: str
    active_track_id: int | None
    evaluated_track_id: int | None
    terminated_track_id: int | None
    released_lost_track_id: int | None
    candidate_track_ids: tuple
    next_candidate_id: int | None
    active_direction_locked: bool
    active_locked_avoidance_side: str | None
    active_center_base: tuple | None
    active_distance_to_vehicle: float | None
    active_component_id: int | None
    termination: SurfaceTermination
    lost_release: LostActiveRelease
    termination_hold: bool
    events: tuple
    completed_track_ids: tuple


class ActiveObstacleLifecycle:
    """Deterministic single-active latch over independent obstacle tracks."""

    def __init__(self, robot_radius=PHYSICAR_ROBOT_BOUNDING_RADIUS,
                 lost_release_radius_multiplier=1.2,
                 activation_distance=1.5):
        self.robot_radius = float(robot_radius)
        self.lost_release_radius_multiplier = float(
            lost_release_radius_multiplier)
        self.activation_distance = float(activation_distance)
        if (not np.isfinite(self.robot_radius) or self.robot_radius <= 0.0
                or not np.isfinite(self.lost_release_radius_multiplier)
                or self.lost_release_radius_multiplier <= 1.0
                or not np.isfinite(self.activation_distance)
                or self.activation_distance <= 0.0):
            raise ValueError('active lifecycle parameters are invalid')
        self.reset()

    def reset(self):
        self.active_track_id = None
        self.active_lost = False
        self.completed_track_ids = set()
        self.last_active_view = None

    def _candidates(self, tracks):
        return tuple(sorted(
            (track for track in tracks
             if track.observed_this_frame and track.current_relevant
             and (track.direction_locked
                  or track.distance_to_vehicle <= self.activation_distance)
             and track.track_id not in self.completed_track_ids),
            key=lambda track: (track.distance_to_vehicle, track.track_id)))

    @staticmethod
    def _missing_termination():
        return SurfaceTermination(False, 0, None, None,
                                  False, False, False)

    def _missing_lost_release(self):
        return evaluate_lost_active_release(
            None, (), self.robot_radius,
            self.lost_release_radius_multiplier)

    def _result(self, state, active, evaluated_track_id, termination,
                termination_hold, events, candidates,
                terminated_track_id=None, released_lost_track_id=None,
                lost_release=None):
        other_candidates = tuple(
            item.track_id for item in candidates
            if active is None or item.track_id != active.track_id)
        return ActiveLifecycleResult(
            state=state,
            active_track_id=(None if active is None else active.track_id),
            evaluated_track_id=evaluated_track_id,
            terminated_track_id=terminated_track_id,
            released_lost_track_id=released_lost_track_id,
            candidate_track_ids=tuple(item.track_id for item in candidates),
            next_candidate_id=(other_candidates[0]
                               if other_candidates else None),
            active_direction_locked=(False if active is None
                                     else active.direction_locked),
            active_locked_avoidance_side=(None if active is None
                                          else active.locked_avoidance_side),
            active_center_base=(None if active is None else active.center_base),
            active_distance_to_vehicle=(None if active is None else
                                        active.distance_to_vehicle),
            active_component_id=(None if active is None else
                                 active.current_component_id),
            termination=termination,
            lost_release=(self._missing_lost_release()
                          if lost_release is None else lost_release),
            termination_hold=termination_hold,
            events=tuple(events),
            completed_track_ids=tuple(sorted(self.completed_track_ids)))

    def _release_lost(self, candidates, evaluation, events=()):
        released_id = self.active_track_id
        if released_id is not None:
            self.completed_track_ids.add(released_id)
        self.active_lost = False
        self.active_track_id = None
        self.last_active_view = None
        next_active = candidates[0] if candidates else None
        output_events = list(events)+[ACTIVE_LOST_RELEASED]
        if next_active is not None:
            self.active_track_id = next_active.track_id
            self.last_active_view = next_active
            output_events.append(NEXT_ACTIVE_SELECTED)
        return self._result(
            ACTIVE if next_active is not None else NO_ACTIVE,
            next_active, released_id, self._missing_termination(), False,
            output_events, candidates,
            released_lost_track_id=released_id, lost_release=evaluation)

    def update(self, track_views, component_points_by_id,
               current_path=(), lost_center_base=None):
        tracks = tuple(track_views)
        track_by_id = {track.track_id: track for track in tracks}
        if len(track_by_id) != len(tracks):
            raise ValueError('track IDs must be unique')
        components = dict(component_points_by_id)
        live_ids = set(track_by_id)
        self.completed_track_ids.intersection_update(live_ids)
        candidates = self._candidates(tracks)
        lost_release = evaluate_lost_active_release(
            lost_center_base, current_path, self.robot_radius,
            self.lost_release_radius_multiplier)

        if self.active_lost:
            if lost_release.release:
                return self._release_lost(candidates, lost_release)
            return self._result(
                ACTIVE_LOST, self.last_active_view, self.active_track_id,
                self._missing_termination(), True, (), candidates,
                lost_release=lost_release)

        if self.active_track_id is None:
            if not candidates:
                return self._result(
                    NO_ACTIVE, None, None, self._missing_termination(),
                    False, (), candidates)
            active = candidates[0]
            self.active_track_id = active.track_id
            self.last_active_view = active
            points = components.get(active.current_component_id)
            evaluation = evaluate_surface_termination(points, self.robot_radius)
            return self._result(
                ACTIVE, active, active.track_id, evaluation,
                not evaluation.component_present, (ACTIVE_SELECTED,), candidates)

        active = track_by_id.get(self.active_track_id)
        if active is None:
            self.active_lost = True
            if lost_release.release:
                return self._release_lost(
                    candidates, lost_release, events=(ACTIVE_LOST,))
            return self._result(
                ACTIVE_LOST, self.last_active_view, self.active_track_id,
                self._missing_termination(), True, (ACTIVE_LOST,), candidates,
                lost_release=lost_release)

        self.last_active_view = active
        points = (None if not active.observed_this_frame else
                  components.get(active.current_component_id))
        evaluation = evaluate_surface_termination(points, self.robot_radius)
        if not evaluation.component_present:
            return self._result(
                ACTIVE, active, active.track_id, evaluation, True,
                (ACTIVE_SCAN_MISSING,), candidates)
        if not evaluation.termination:
            return self._result(
                ACTIVE, active, active.track_id, evaluation, False,
                (ACTIVE_LATCHED,), candidates)

        terminated_id = active.track_id
        self.completed_track_ids.add(terminated_id)
        self.active_track_id = None
        self.last_active_view = None
        next_candidates = self._candidates(tracks)
        events = [ACTIVE_TERMINATED]
        next_active = None
        if next_candidates:
            next_active = next_candidates[0]
            self.active_track_id = next_active.track_id
            self.last_active_view = next_active
            events.append(NEXT_ACTIVE_SELECTED)
        return self._result(
            ACTIVE if next_active is not None else NO_ACTIVE,
            next_active, terminated_id, evaluation, False, events,
            next_candidates, terminated_track_id=terminated_id)
