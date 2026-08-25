"""ROS-independent Stage 5.2C low-vote recovery decision chain.

This module emits shadow requests and deterministic direction-lock requests.
It never publishes actuator commands and never changes path geometry.
"""

from dataclasses import dataclass
import math

import numpy as np

from .center_hybrid import project_point
from .circle_avoidance import AVOID_LEFT, AVOID_RIGHT


NORMAL_VOTING = 'NORMAL_VOTING'
SLOW_VOTING = 'SLOW_VOTING'
HEADING_RECOVERY = 'HEADING_RECOVERY'
WHITE_PROXIMITY_FALLBACK = 'WHITE_PROXIMITY_FALLBACK'
LOCKED = 'LOCKED'
RECOVERY_FAILED_DEFAULT = 'RECOVERY_FAILED_DEFAULT'

MODE_NONE = 'NONE'
MODE_CRAWL = 'CRAWL'
MODE_HEADING_RECOVERY = 'HEADING_RECOVERY'

NORMAL_VOTE = 'NORMAL_VOTE'
SLOW_VOTING_LOCK = 'SLOW_VOTING'
HEADING_RECOVERY_VOTE = 'HEADING_RECOVERY_VOTE'
WHITE_PROXIMITY = 'WHITE_PROXIMITY'
DETERMINISTIC_DEFAULT = 'DETERMINISTIC_DEFAULT'


@dataclass(frozen=True)
class LowVoteRecoveryConfig:
    freeze_distance: float = 1.5
    slow_voting_max_frames: int = 4
    heading_recovery_max_frames: int = 5
    max_heading_correction: float = 0.25
    emergency_distance: float = 0.45
    center_start_distance: float = 0.60
    default_avoidance_side: str = AVOID_LEFT

    def __post_init__(self):
        values = np.asarray((
            self.freeze_distance, self.max_heading_correction,
            self.emergency_distance, self.center_start_distance),
            dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError('recovery distances/angle must be finite and > 0')
        if self.emergency_distance >= self.freeze_distance:
            raise ValueError('emergency distance must be below freeze distance')
        if (not isinstance(self.slow_voting_max_frames, int)
                or self.slow_voting_max_frames <= 0
                or not isinstance(self.heading_recovery_max_frames, int)
                or self.heading_recovery_max_frames <= 0):
            raise ValueError('recovery frame bounds must be positive')
        if self.default_avoidance_side not in (AVOID_LEFT, AVOID_RIGHT):
            raise ValueError('default avoidance side is invalid')


@dataclass(frozen=True)
class RecoveryTrackView:
    track_id: int
    center_base: tuple
    distance_to_vehicle: float
    left_votes: int
    right_votes: int
    vote_count: int
    direction_locked: bool
    locked_avoidance_side: str | None
    lock_source: str | None = None

    def __post_init__(self):
        center = np.asarray(self.center_base, dtype=np.float64).reshape(2)
        values = np.r_[center, self.distance_to_vehicle]
        if not np.all(np.isfinite(values)) or self.distance_to_vehicle < 0.0:
            raise ValueError('track recovery geometry is invalid')
        if min(self.left_votes, self.right_votes, self.vote_count) < 0:
            raise ValueError('vote counts must be nonnegative')
        if self.vote_count != self.left_votes + self.right_votes:
            raise ValueError('vote_count must equal LEFT+RIGHT')
        object.__setattr__(self, 'center_base', tuple(float(x) for x in center))


@dataclass(frozen=True)
class WhiteComponentView:
    component_id: int
    points: np.ndarray

    def __post_init__(self):
        points = np.asarray(self.points, dtype=np.float64)
        if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
                or not np.all(np.isfinite(points))):
            raise ValueError('WHITE component must be finite shape (N,2), N>=2')
        object.__setattr__(self, 'points', points.copy())


@dataclass
class _RecoveryMemory:
    state: str = NORMAL_VOTING
    frame_count: int = 0
    no_vote_progress_count: int = 0
    last_vote_count: int = 0
    last_measurement_key: object = None


@dataclass(frozen=True)
class RecoveryResult:
    track_id: int | None
    state: str
    requested_mode: str
    recovery_frame_count: int
    no_vote_progress_count: int
    obstacle_bearing: float | None
    recovery_heading_target: float | None
    recovery_heading_error: float | None
    center_path_valid: bool
    center_start_distance: float | None
    nearest_white_component_id: int | None
    nearest_white_point: tuple | None
    obstacle_to_white_distance: float | None
    white_fallback_vector: tuple | None
    reference_source: str | None
    chosen_side: str | None
    lock_source: str | None
    lock_requested: bool
    direction_locked: bool
    emergency_distance_reached: bool
    duplicate_measurement: bool
    reason: str


def _path_or_empty(values):
    points = np.asarray(values, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2 or len(points) < 2
            or not np.all(np.isfinite(points))):
        return np.empty((0, 2), dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9]
    points = points[keep]
    return points if len(points) >= 2 else np.empty((0, 2), dtype=np.float64)


def _near_to_far(points):
    values = _path_or_empty(points)
    if len(values) and np.linalg.norm(values[-1]) < np.linalg.norm(values[0]):
        return values[::-1].copy()
    return values


def _vote_side(track, default_side):
    if track.left_votes > track.right_votes:
        return AVOID_RIGHT
    if track.right_votes > track.left_votes:
        return AVOID_LEFT
    return default_side


def _nearest_white(center, components):
    best = None
    query = np.asarray(center, dtype=np.float64)
    for component in components:
        points = np.asarray(component.points, dtype=np.float64)
        distances = np.linalg.norm(points-query, axis=1)
        index = int(np.argmin(distances))
        key = (float(distances[index]), int(component.component_id), index)
        if best is None or key < best[0]:
            best = (key, component, points[index].copy())
    return best


def _reference_tangent(center, center_path, white_component, history_path):
    candidates = (
        ('CURRENT_CENTER', _near_to_far(center_path)),
        ('CURRENT_WHITE', (np.empty((0, 2), dtype=np.float64)
                           if white_component is None else
                           _near_to_far(white_component.points))),
        ('RECENT_CENTER_HISTORY', _near_to_far(history_path)),
    )
    for source, points in candidates:
        if len(points) < 2:
            continue
        try:
            return np.asarray(project_point(center, points).tangent), source
        except ValueError:
            continue
    return None, None


def white_proximity_decision(center, components, center_path=(),
                             history_path=(), default_side=AVOID_LEFT):
    """Return nearest-WHITE opposite-side decision in a local path frame."""
    values = tuple(components)
    nearest = _nearest_white(center, values) if values else None
    if nearest is None:
        return {
            'side': default_side, 'source': DETERMINISTIC_DEFAULT,
            'component_id': None, 'point': None, 'distance': None,
            'away': None, 'reference_source': None,
            'reason': 'NO_CURRENT_WHITE'}
    key, component, point = nearest
    center_value = np.asarray(center, dtype=np.float64)
    away = center_value-point
    norm = float(np.linalg.norm(away))
    tangent, reference_source = _reference_tangent(
        center_value, center_path, component, history_path)
    if norm <= 1e-9 or tangent is None:
        return {
            'side': default_side, 'source': DETERMINISTIC_DEFAULT,
            'component_id': component.component_id,
            'point': tuple(float(x) for x in point), 'distance': key[0],
            'away': (None if norm <= 1e-9 else
                     tuple(float(x) for x in away/norm)),
            'reference_source': reference_source,
            'reason': ('WHITE_VECTOR_DEGENERATE' if norm <= 1e-9
                       else 'LOCAL_REFERENCE_UNAVAILABLE')}
    away /= norm
    tangent /= np.linalg.norm(tangent)
    left_normal = np.array([-tangent[1], tangent[0]])
    signed = float(np.dot(away, left_normal))
    if abs(signed) <= 1e-9:
        side = default_side
        source = DETERMINISTIC_DEFAULT
        reason = 'WHITE_VECTOR_LATERAL_UNDEFINED'
    else:
        side = AVOID_LEFT if signed > 0.0 else AVOID_RIGHT
        source = WHITE_PROXIMITY
        reason = 'NEAREST_WHITE_OPPOSITE_SIDE'
    return {
        'side': side, 'source': source,
        'component_id': component.component_id,
        'point': tuple(float(x) for x in point), 'distance': key[0],
        'away': tuple(float(x) for x in away),
        'reference_source': reference_source, 'reason': reason}


class LowVoteRecoveryManager:
    """Bounded ACTIVE-only recovery with per-track independent memory."""

    def __init__(self, config=LowVoteRecoveryConfig()):
        self.config = config
        self.memories = {}
        self.last_results = {}

    def reset(self):
        self.memories.clear()
        self.last_results.clear()

    def retain_tracks(self, track_ids):
        live = {int(value) for value in track_ids}
        self.memories = {key: value for key, value in self.memories.items()
                         if key in live}
        self.last_results = {key: value for key, value in self.last_results.items()
                             if key in live}

    def _result(self, track, memory, **overrides):
        center_path = overrides.pop('center_path', np.empty((0, 2)))
        center = _path_or_empty(center_path)
        start = None if not len(center) else float(np.linalg.norm(center[0]))
        values = dict(
            track_id=track.track_id,
            state=memory.state,
            requested_mode=MODE_NONE,
            recovery_frame_count=memory.frame_count,
            no_vote_progress_count=memory.no_vote_progress_count,
            obstacle_bearing=math.atan2(
                track.center_base[1], track.center_base[0]),
            recovery_heading_target=None,
            recovery_heading_error=None,
            center_path_valid=bool(len(center)),
            center_start_distance=start,
            nearest_white_component_id=None,
            nearest_white_point=None,
            obstacle_to_white_distance=None,
            white_fallback_vector=None,
            reference_source=None,
            chosen_side=None,
            lock_source=None,
            lock_requested=False,
            direction_locked=track.direction_locked,
            emergency_distance_reached=(
                track.distance_to_vehicle <= self.config.emergency_distance),
            duplicate_measurement=False,
            reason='OK')
        values.update(overrides)
        result = RecoveryResult(**values)
        self.last_results[track.track_id] = result
        return result

    def update(self, track, center_path=(), white_components=(),
               history_path=(), measurement_key=None):
        memory = self.memories.setdefault(
            int(track.track_id), _RecoveryMemory(
                last_vote_count=track.vote_count))
        if (measurement_key is not None
                and memory.last_measurement_key == measurement_key):
            previous = self.last_results.get(track.track_id)
            if previous is not None:
                return RecoveryResult(
                    **{**previous.__dict__, 'duplicate_measurement': True,
                       'reason': 'DUPLICATE_MEASUREMENT_HOLD'})
        memory.last_measurement_key = measurement_key

        if track.direction_locked:
            memory.state = LOCKED
            memory.frame_count = 0
            return self._result(
                track, memory, center_path=center_path,
                chosen_side=track.locked_avoidance_side,
                lock_source=track.lock_source,
                direction_locked=True, reason='ALREADY_LOCKED')

        if track.distance_to_vehicle > self.config.freeze_distance:
            memory.state = NORMAL_VOTING
            memory.frame_count = 0
            memory.no_vote_progress_count = 0
            memory.last_vote_count = track.vote_count
            return self._result(
                track, memory, center_path=center_path,
                reason='OUTSIDE_FREEZE_DISTANCE')

        center = _path_or_empty(center_path)
        center_start = (None if not len(center) else
                        float(np.linalg.norm(center[0])))
        memory.last_vote_count = track.vote_count

        if track.vote_count > 0:
            memory.state = LOCKED
            side = _vote_side(track, self.config.default_avoidance_side)
            return self._result(
                track, memory, center_path=center,
                chosen_side=side, lock_source=NORMAL_VOTE,
                lock_requested=True, reason='AVAILABLE_VOTE_LOCK')

        # No approach vote exists at activation distance.  Do not wait or
        # steer toward the obstacle: choose the side away from the nearest
        # current WHITE component immediately.
        decision = white_proximity_decision(
            track.center_base, white_components, center, history_path,
            self.config.default_avoidance_side)
        memory.state = (WHITE_PROXIMITY_FALLBACK
                        if decision['source'] == WHITE_PROXIMITY
                        else RECOVERY_FAILED_DEFAULT)
        return self._result(
            track, memory, center_path=center,
            nearest_white_component_id=decision['component_id'],
            nearest_white_point=decision['point'],
            obstacle_to_white_distance=decision['distance'],
            white_fallback_vector=decision['away'],
            reference_source=decision['reference_source'],
            chosen_side=decision['side'],
            lock_source=decision['source'],
            lock_requested=True,
            reason=decision['reason'])
