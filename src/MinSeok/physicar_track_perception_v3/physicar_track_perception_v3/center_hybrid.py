"""Current-frame center stitching and short exact-odom path recovery.

This module is ROS independent.  Paths are ordered near-to-far polylines;
there is deliberately no X sorting, spline fitting, or long-term map state.
"""

from collections import deque
from dataclasses import dataclass
import math

import numpy as np

from .geometry import cumulative_s


CURRENT_ORANGE_ONLY = 'CURRENT_ORANGE_ONLY'
CURRENT_HYBRID_ORANGE_WHITE = 'CURRENT_HYBRID_ORANGE_WHITE'
CURRENT_HYBRID_WITH_HISTORY_PREFIX = 'CURRENT_HYBRID_WITH_HISTORY_PREFIX'


@dataclass(frozen=True)
class CenterHybridConfig:
    max_start_distance: float = 0.60
    join_gap: float = 0.30
    tangent_angle_limit: float = 0.75
    history_max_age: float = 0.50
    history_max_entries: int = 8

    def __post_init__(self):
        values = np.asarray((
            self.max_start_distance, self.join_gap,
            self.tangent_angle_limit, self.history_max_age), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError('hybrid geometry thresholds must be finite and > 0')
        if self.tangent_angle_limit >= math.pi:
            raise ValueError('tangent_angle_limit must be less than pi')
        if self.history_max_entries < 1:
            raise ValueError('history_max_entries must be positive')


@dataclass(frozen=True)
class Projection:
    point: np.ndarray
    distance: float
    arc: float
    tangent: np.ndarray
    segment_index: int
    fraction: float


@dataclass(frozen=True)
class CurrentHybridResult:
    path: np.ndarray
    source: str
    reason: str
    white_used: bool
    white_gap_bridge_count: int
    white_near_prefix_points: int
    white_far_suffix_points: int
    join_gaps: tuple
    tangent_differences: tuple


@dataclass(frozen=True)
class CenterHistoryEntry:
    stamp: float
    points_odom: np.ndarray


@dataclass(frozen=True)
class HistoryRecoveryResult:
    path: np.ndarray
    used: bool
    reason: str
    history_available: bool
    history_age: float | None
    history_point_count: int
    transformed_point_count: int
    prefix: np.ndarray
    join_gap: float | None
    tangent_difference: float | None


@dataclass(frozen=True)
class PredictedSuffixResult:
    path: np.ndarray
    suffix: np.ndarray
    length: float
    reason: str


@dataclass(frozen=True)
class BoundaryAlignedSuffixResult:
    path: np.ndarray
    suffix: np.ndarray
    length: float
    reason: str
    selected_source: str
    tangent_candidate: np.ndarray
    white_candidate: np.ndarray
    tangent_boundary_score: float | None
    white_boundary_score: float | None
    tangent_boundary_matches: int
    white_boundary_matches: int
    white_candidate_valid: bool
    white_candidate_reason: str
    white_candidate_join_angle: float | None


def _path(values, name='path', minimum=2):
    points = np.asarray(values, dtype=np.float64)
    if (points.ndim != 2 or points.shape[1] != 2
            or len(points) < minimum or not np.all(np.isfinite(points))):
        raise ValueError(f'{name} must be finite shape (N,2), N>={minimum}')
    if minimum >= 2 and np.any(np.linalg.norm(np.diff(points, axis=0), axis=1)
                               <= 1e-9):
        # Consecutive duplicates make endpoint tangent tests ambiguous.
        keep = np.r_[True, np.linalg.norm(np.diff(points, axis=0), axis=1)
                     > 1e-9]
        points = points[keep]
        if len(points) < minimum:
            raise ValueError(f'{name} is degenerate')
    return points.copy()


def _unit(vector):
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError('zero tangent')
    return vector / norm


def _angle(first, second):
    return float(np.arccos(np.clip(np.dot(_unit(first), _unit(second)),
                                   -1.0, 1.0)))


def _orient_near(points):
    points = _path(points)
    return (points[::-1].copy() if np.linalg.norm(points[-1])
            < np.linalg.norm(points[0]) else points)


def orient_fragment_chain(fragments):
    """Orient an already selected fragment chain without changing its order."""
    result = []
    reference = None
    for index, values in enumerate(fragments):
        points = _path(values, f'fragment_{index}')
        if reference is None:
            reverse = np.linalg.norm(points[-1]) < np.linalg.norm(points[0])
        else:
            reverse = (np.linalg.norm(points[-1] - reference)
                       < np.linalg.norm(points[0] - reference))
        if reverse:
            points = points[::-1].copy()
        result.append(points)
        reference = points[-1]
    return tuple(result)


def project_point(point, path):
    points = _path(path)
    query = np.asarray(point, dtype=np.float64).reshape(2)
    segment = np.diff(points, axis=0)
    length_squared = np.einsum('ij,ij->i', segment, segment)
    relative = query - points[:-1]
    fraction = np.einsum('ij,ij->i', relative, segment) / length_squared
    fraction = np.clip(fraction, 0.0, 1.0)
    projected = points[:-1] + fraction[:, None] * segment
    distances = np.linalg.norm(projected - query, axis=1)
    index = int(np.argmin(distances))
    lengths = np.sqrt(length_squared)
    arc = cumulative_s(points)
    return Projection(
        point=projected[index], distance=float(distances[index]),
        arc=float(arc[index] + fraction[index] * lengths[index]),
        tangent=_unit(segment[index]), segment_index=index,
        fraction=float(fraction[index]))


def _interval(path, begin_projection, end_projection):
    """Return an ordered inclusive polyline interval between projections."""
    points = _path(path)
    if end_projection.arc <= begin_projection.arc + 1e-9:
        return np.empty((0, 2), dtype=np.float64)
    arc = cumulative_s(points)
    selected = [begin_projection.point]
    for index, value in enumerate(arc):
        if begin_projection.arc + 1e-9 < value < end_projection.arc - 1e-9:
            selected.append(points[index])
    selected.append(end_projection.point)
    output = np.asarray(selected, dtype=np.float64)
    keep = np.r_[True, np.linalg.norm(np.diff(output, axis=0), axis=1) > 1e-9]
    return output[keep]


def _append_unique(target, values):
    for point in np.asarray(values, dtype=np.float64).reshape(-1, 2):
        if not target or np.linalg.norm(point - target[-1]) > 1e-9:
            target.append(point.copy())


def stitch_current_frame(orange_path, orange_fragments, white_path=None,
                         config=CenterHybridConfig()):
    """Use current WHITE only in ORANGE gaps or as a missing near prefix."""
    orange = _path(orange_path, 'orange_path')
    fragments = orient_fragment_chain(orange_fragments)
    if not fragments:
        raise ValueError('orange_fragments must not be empty')
    if white_path is None or len(white_path) < 2:
        return CurrentHybridResult(
            orange, CURRENT_ORANGE_ONLY, 'NO_CURRENT_WHITE_CENTER', False,
            0, 0, 0, (), ())
    white = _orient_near(white_path)
    assembled = []
    bridge_count = 0
    join_gaps = []
    tangent_differences = []
    for index, fragment in enumerate(fragments):
        if index:
            previous = fragments[index - 1]
            endpoint_a = previous[-1]
            endpoint_b = fragment[0]
            on_a = project_point(endpoint_a, white)
            on_b = project_point(endpoint_b, white)
            angle_a = _angle(previous[-1] - previous[-2], on_a.tangent)
            angle_b = _angle(on_b.tangent, fragment[1] - fragment[0])
            valid = (
                on_a.distance <= config.join_gap
                and on_b.distance <= config.join_gap
                and on_b.arc > on_a.arc + 1e-9
                and angle_a <= config.tangent_angle_limit
                and angle_b <= config.tangent_angle_limit)
            if valid:
                _append_unique(assembled, _interval(white, on_a, on_b))
                bridge_count += 1
                join_gaps.extend((on_a.distance, on_b.distance))
                tangent_differences.extend((angle_a, angle_b))
            else:
                _append_unique(
                    assembled, np.linspace(endpoint_a, endpoint_b, 4)[1:-1])
        _append_unique(assembled, fragment)
    assembled = np.asarray(assembled, dtype=np.float64)

    prefix_count = 0
    if np.linalg.norm(orange[0]) > config.max_start_distance:
        on_orange = project_point(fragments[0][0], white)
        angle = _angle(on_orange.tangent,
                       fragments[0][1] - fragments[0][0])
        origin_on_white = project_point(np.zeros(2), white)
        prefix = _interval(white, origin_on_white, on_orange)
        prefix_valid = (
            len(prefix) >= 2
            and on_orange.distance <= config.join_gap
            and on_orange.arc > origin_on_white.arc + 1e-9
            and angle <= config.tangent_angle_limit
            and np.linalg.norm(prefix[0]) <= config.max_start_distance
            and np.linalg.norm(prefix[0]) < np.linalg.norm(fragments[0][0]))
        if prefix_valid:
            merged = []
            _append_unique(merged, prefix)
            _append_unique(merged, assembled)
            assembled = np.asarray(merged, dtype=np.float64)
            prefix_count = len(prefix)
            join_gaps.append(on_orange.distance)
            tangent_differences.append(angle)

    # WHITE is intentionally limited to missing geometry before/between
    # observed ORANGE fragments.  Do not append a far suffix after the last
    # observed ORANGE point: the final path must end at current stitched
    # measurement rather than being forced toward a WHITE endpoint.
    far_suffix_count = 0

    white_used = bool(bridge_count or prefix_count)
    if not white_used:
        # Exact parity: use the existing selector output byte-for-byte when
        # WHITE did not satisfy the join contract.
        assembled = orange
    return CurrentHybridResult(
        path=assembled,
        source=(CURRENT_HYBRID_ORANGE_WHITE if white_used
                else CURRENT_ORANGE_ONLY),
        reason=('CURRENT_WHITE_GAP_OR_PREFIX' if white_used
                else 'CURRENT_WHITE_JOIN_REJECTED'),
        white_used=white_used,
        white_gap_bridge_count=bridge_count,
        white_near_prefix_points=prefix_count,
        white_far_suffix_points=far_suffix_count,
        join_gaps=tuple(float(value) for value in join_gaps),
        tangent_differences=tuple(float(value)
                                  for value in tangent_differences))


def extend_predicted_suffix(path, bounds, spacing=0.05,
                            tangent_window=0.15):
    """Append a tangent suffix, retaining its first point beyond the bounds."""
    points = _path(path)
    x_min, x_max, y_min, y_max = (float(value) for value in bounds)
    values = np.asarray((x_min, x_max, y_min, y_max,
                         spacing, tangent_window), dtype=float)
    if (not np.all(np.isfinite(values)) or x_max <= x_min or y_max <= y_min
            or spacing <= 0.0 or tangent_window <= 0.0):
        raise ValueError('predicted suffix parameters are invalid')
    arc = cumulative_s(points)
    begin_arc = max(0.0, float(arc[-1]) - float(tangent_window))
    begin = int(np.searchsorted(arc, begin_arc, side='left'))
    direction = points[-1] - points[begin]
    if np.linalg.norm(direction) <= 1e-12:
        direction = points[-1] - points[-2]
    direction = _unit(direction)
    predicted = []
    # Any ray starting inside a rectangle exits within its diagonal.  This
    # bound protects the loop without imposing a semantic path-length cap.
    maximum_steps = int(math.ceil(math.hypot(
        x_max - x_min, y_max - y_min) / float(spacing))) + 2
    for step in range(1, maximum_steps + 1):
        distance = float(step) * float(spacing)
        candidate = points[-1] + distance * direction
        # Bounds define the useful sensor/debug horizon, not a clipping
        # contract for nav_msgs/Path.  Preserve the first crossing sample so
        # an offset/extrapolated center does not disappear at the rendered
        # BEV edge.  The overrun is bounded naturally by one sample spacing.
        predicted.append(candidate)
        if not (x_min <= candidate[0] < x_max
                and y_min <= candidate[1] < y_max):
            break
    suffix = np.asarray(predicted, dtype=np.float64).reshape(-1, 2)
    if not len(suffix):
        return PredictedSuffixResult(
            points, suffix, 0.0, 'PREDICTION_UNAVAILABLE')
    merged = np.vstack((points, suffix))
    return PredictedSuffixResult(
        merged, suffix, float(np.linalg.norm(suffix[-1] - points[-1])),
        'CURRENT_TANGENT_SUFFIX')


def _suffix_boundary_score(suffix, boundary_paths, expected_half_width,
                           half_width_tolerance):
    """Compare candidate shape with nearby current WHITE boundaries.

    The score combines direction mismatch and W/2 offset error.  Candidate
    portions without a nearby current-frame boundary are deliberately ignored;
    they carry no measurement evidence for choosing between extrapolations.
    """
    points = np.asarray(suffix, dtype=np.float64).reshape(-1, 2)
    boundaries = []
    for values in boundary_paths:
        try:
            boundaries.append(_orient_near(values))
        except ValueError:
            continue
    if len(points) < 2 or not boundaries:
        return None, 0
    expected = float(expected_half_width)
    tolerance = float(half_width_tolerance)
    if (not np.isfinite(expected) or expected <= 0.0
            or not np.isfinite(tolerance) or tolerance <= 0.0):
        raise ValueError('WHITE score geometry must be finite and positive')
    maximum_distance = expected+tolerance
    values = []
    for first, second in zip(points[:-1], points[1:]):
        tangent = second-first
        if np.linalg.norm(tangent) <= 1e-9:
            continue
        query = 0.5*(first+second)
        nearest = None
        for boundary in boundaries:
            projection = project_point(query, boundary)
            key = projection.distance
            if nearest is None or key < nearest.distance:
                nearest = projection
        if nearest is None or nearest.distance > maximum_distance:
            continue
        # WHITE component orientation is not a trusted LEFT/RIGHT identity;
        # parallel and anti-parallel tangents therefore represent equal shape.
        tangent_error = float(np.arccos(np.clip(abs(np.dot(
            _unit(tangent), _unit(nearest.tangent))), 0.0, 1.0)))
        offset_error = abs(nearest.distance-expected)/expected
        values.append(tangent_error+offset_error)
    if not values:
        return None, 0
    return float(np.mean(values)), len(values)


def _white_geometry_candidate(path, white_center_path, bounds, spacing,
                              tangent_window, join_gap,
                              tangent_angle_limit):
    base = _path(path)
    if white_center_path is None or len(white_center_path) < 2:
        return None, 'NO_CURRENT_WHITE_CENTER', False, None
    try:
        white = _orient_near(white_center_path)
        on_end = project_point(base[-1], white)
        base_tangent = _unit(base[-1]-base[-2])
        angle = _angle(base_tangent, on_end.tangent)
    except ValueError:
        return None, 'WHITE_CANDIDATE_GEOMETRY_INVALID', False, None
    if on_end.distance > join_gap:
        return None, 'WHITE_CANDIDATE_JOIN_GAP', False, angle
    compatible = angle <= tangent_angle_limit

    white_end = project_point(white[-1], white)
    tail = _interval(white, on_end, white_end)
    merged = base.copy()
    measured_suffix = np.empty((0, 2), dtype=np.float64)
    if len(tail) >= 2:
        # Translate only the candidate suffix so its first point is exactly the
        # final hybrid endpoint.  This keeps the WHITE-observed curvature while
        # avoiding a connector jump between two near-parallel measurements.
        shifted = tail+(base[-1]-tail[0])
        measured_suffix = shifted[1:].copy()
        if len(measured_suffix):
            merged = np.vstack((merged, measured_suffix))
    else:
        # The current WHITE may end at the same longitudinal location as the
        # final hybrid.  Its local tangent still supplies an independent,
        # current-frame boundary-shaped extrapolation candidate.
        seed = base[-1]+float(spacing)*_unit(on_end.tangent)
        measured_suffix = seed[None, :]
        merged = np.vstack((merged, seed))
    predicted = extend_predicted_suffix(
        merged, bounds, spacing=spacing, tangent_window=tangent_window)
    suffix = np.vstack((
        measured_suffix,
        predicted.suffix,
    ))
    if not len(suffix):
        return None, 'WHITE_CANDIDATE_EMPTY', False, angle
    keep = np.r_[True, np.linalg.norm(np.diff(suffix, axis=0), axis=1) > 1e-9]
    suffix = suffix[keep]
    reason = ('CURRENT_WHITE_GEOMETRY_SUFFIX' if compatible else
              'WHITE_CANDIDATE_TANGENT_MISMATCH')
    return np.vstack((base, suffix)), reason, compatible, angle


def select_boundary_aligned_suffix(
        path, white_center_path, white_boundary_paths, bounds, spacing=0.05,
        tangent_window=0.15, join_gap=0.30, tangent_angle_limit=0.75,
        expected_half_width=0.37, half_width_tolerance=0.10):
    """Choose tangent or current-WHITE suffix by boundary-shape similarity."""
    base = _path(path)
    tangent = extend_predicted_suffix(
        base, bounds, spacing=spacing, tangent_window=tangent_window)
    white_path, white_reason, white_compatible, white_join_angle = (
        _white_geometry_candidate(
        base, white_center_path, bounds, spacing, tangent_window, join_gap,
        tangent_angle_limit))
    tangent_score, tangent_matches = _suffix_boundary_score(
        tangent.suffix, white_boundary_paths, expected_half_width,
        half_width_tolerance)
    white_suffix = np.empty((0, 2), dtype=np.float64)
    white_score = None
    white_matches = 0
    if white_path is not None:
        white_suffix = white_path[len(base):]
        white_score, white_matches = _suffix_boundary_score(
            white_suffix, white_boundary_paths, expected_half_width,
            half_width_tolerance)

    choose_white = bool(
        white_path is not None and white_compatible and white_score is not None
        and (tangent_score is None or white_score <= tangent_score+1e-12))
    if choose_white:
        selected_path = white_path
        selected_suffix = white_suffix
        selected_source = 'WHITE_GEOMETRY'
        reason = 'CURRENT_WHITE_BOUNDARY_ALIGNED_SUFFIX'
    else:
        selected_path = tangent.path
        selected_suffix = tangent.suffix
        selected_source = 'PATH_TANGENT'
        reason = ('CURRENT_TANGENT_SUFFIX_NO_WHITE_EVIDENCE'
                  if white_path is not None and white_score is None
                  else 'CURRENT_TANGENT_SUFFIX_SELECTED')
    length = (0.0 if not len(selected_suffix) else float(
        np.linalg.norm(selected_suffix[-1]-base[-1])))
    return BoundaryAlignedSuffixResult(
        path=selected_path,
        suffix=selected_suffix,
        length=length,
        reason=reason,
        selected_source=selected_source,
        tangent_candidate=tangent.suffix,
        white_candidate=white_suffix,
        tangent_boundary_score=tangent_score,
        white_boundary_score=white_score,
        tangent_boundary_matches=tangent_matches,
        white_boundary_matches=white_matches,
        white_candidate_valid=bool(white_path is not None and white_compatible),
        white_candidate_reason=white_reason,
        white_candidate_join_angle=white_join_angle,
    )


def transform_xy(points, target_from_source):
    points = _path(points)
    matrix = np.asarray(target_from_source, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError('transform must be finite shape (4,4)')
    xyz = np.column_stack((points, np.zeros(len(points)), np.ones(len(points))))
    transformed = (matrix @ xyz.T).T
    return transformed[:, :2]


class RecentCenterHistory:
    """Bounded odom-frame measurements; never stores recovered history."""

    def __init__(self, config=CenterHybridConfig()):
        self.config = config
        self.entries = deque(maxlen=config.history_max_entries)

    def clear(self):
        self.entries.clear()

    def store(self, points_base, odom_from_base, stamp):
        points_odom = transform_xy(points_base, odom_from_base)
        self.entries.append(CenterHistoryEntry(float(stamp), points_odom))

    def _recent_entries(self, now):
        now = float(now)
        return tuple(entry for entry in self.entries
                     if 0.0 <= now - entry.stamp <= self.config.history_max_age)

    def recover(self, current_path, base_from_odom, now):
        current = _path(current_path, 'current_path')
        recent = self._recent_entries(now)
        if np.linalg.norm(current[0]) <= self.config.max_start_distance:
            return HistoryRecoveryResult(
                current, False, 'CURRENT_NEAR_COVERAGE_SUFFICIENT',
                bool(recent), None, 0, 0, np.empty((0, 2)), None, None)
        newest_age = (None if not recent else float(now) - recent[-1].stamp)
        if not recent:
            return HistoryRecoveryResult(
                current, False, 'NO_RECENT_HISTORY', False, None, 0, 0,
                np.empty((0, 2)), None, None)

        last_transformed = 0
        last_points = 0
        for entry in reversed(recent):
            transformed = transform_xy(entry.points_odom, base_from_odom)
            last_transformed = len(transformed)
            last_points = len(entry.points_odom)
            origin = project_point(np.zeros(2), transformed)
            join = project_point(current[0], transformed)
            tangent_difference = _angle(join.tangent,
                                        current[1] - current[0])
            prefix = _interval(transformed, origin, join)
            valid = (
                len(prefix) >= 2
                and join.arc > origin.arc + 1e-9
                and join.distance <= self.config.join_gap
                and tangent_difference <= self.config.tangent_angle_limit
                and np.linalg.norm(prefix[0]) <= self.config.max_start_distance
                and np.linalg.norm(prefix[0]) < np.linalg.norm(current[0]))
            if not valid:
                continue
            merged = []
            _append_unique(merged, prefix)
            _append_unique(merged, current)
            return HistoryRecoveryResult(
                np.asarray(merged), True, 'RECENT_ODOM_PREFIX_JOINED', True,
                float(now) - entry.stamp, len(entry.points_odom),
                len(transformed), prefix, join.distance,
                tangent_difference)
        return HistoryRecoveryResult(
            current, False, 'HISTORY_JOIN_REJECTED', True, newest_age,
            last_points, last_transformed, np.empty((0, 2)), None, None)
