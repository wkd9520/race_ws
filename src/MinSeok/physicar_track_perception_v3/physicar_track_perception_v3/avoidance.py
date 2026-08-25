"""Current-frame, shadow-only LiDAR repulsive path deformation."""

from dataclasses import dataclass

import numpy as np

from .geometry import cumulative_s, tangents


@dataclass(frozen=True)
class AvoidanceConfig:
    path_near_distance: float = 0.20
    representative_window: float = 0.30
    influence_radius: float = 0.60
    avoidance_offset: float = 0.25
    center_deadband: float = 0.03
    tangent_window: float = 0.10
    resample_spacing: float = 0.05

    def __post_init__(self):
        values = np.asarray([
            self.path_near_distance,
            self.representative_window,
            self.influence_radius,
            self.avoidance_offset,
            self.center_deadband,
            self.tangent_window,
            self.resample_spacing,
        ], dtype=np.float64)
        if not np.all(np.isfinite(values)) or np.any(values < 0.0):
            raise ValueError('avoidance configuration must be finite and non-negative')
        if self.path_near_distance == 0.0:
            raise ValueError('path_near_distance must be positive')
        if self.representative_window == 0.0:
            raise ValueError('representative_window must be positive')
        if self.influence_radius == 0.0:
            raise ValueError('influence_radius must be positive')
        if self.tangent_window == 0.0:
            raise ValueError('tangent_window must be positive')
        if self.resample_spacing == 0.0:
            raise ValueError('resample_spacing must be positive')


@dataclass(frozen=True)
class AvoidanceResult:
    active: bool
    reason: str
    original: np.ndarray
    deformed: np.ndarray
    obstacle: np.ndarray | None
    nearest_path_point: np.ndarray | None
    candidate_count: int
    signed_lateral: float | None
    s_obstacle: float | None
    signed_offset: float
    weights: np.ndarray
    clearance_original: float | None
    clearance_avoidance: float | None
    max_heading_step_original: float | None
    max_heading_step_avoidance: float | None


def _points_xy(values, name, minimum=1):
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < minimum:
        raise ValueError(f'{name} must have shape (N,2), N>={minimum}')
    if not np.all(np.isfinite(points)):
        raise ValueError(f'{name} must be finite')
    return points


def project_to_polyline(query_points, path_points):
    """Project XY query points onto an ordered polyline.

    Returns nearest points, distances, arc lengths, tangents, and signed
    lateral distances. Positive lateral is the local LEFT normal.
    """
    query = _points_xy(query_points, 'query_points')
    path = _points_xy(path_points, 'path_points', minimum=2)
    segment = np.diff(path, axis=0)
    length = np.linalg.norm(segment, axis=1)
    valid = length > 1e-9
    if not np.any(valid):
        raise ValueError('path must contain a non-degenerate segment')
    path_s = cumulative_s(path)
    start = path[:-1][valid]
    vector = segment[valid]
    segment_length = length[valid]
    segment_s = path_s[:-1][valid]

    delta = query[:, None, :] - start[None, :, :]
    fraction = np.sum(delta * vector[None, :, :], axis=2)
    fraction /= np.square(segment_length)[None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = start[None, :, :] + fraction[:, :, None] * vector[None, :, :]
    residual = query[:, None, :] - closest
    distance_sq = np.sum(np.square(residual), axis=2)
    nearest_index = np.argmin(distance_sq, axis=1)
    row = np.arange(len(query))
    nearest = closest[row, nearest_index]
    nearest_fraction = fraction[row, nearest_index]
    nearest_length = segment_length[nearest_index]
    nearest_tangent = vector[nearest_index] / nearest_length[:, None]
    normal_left = np.column_stack((-nearest_tangent[:, 1],
                                   nearest_tangent[:, 0]))
    signed_lateral = np.sum((query - nearest) * normal_left, axis=1)
    arc_length = (segment_s[nearest_index]
                  + nearest_fraction * nearest_length)
    return {
        'nearest': nearest,
        'distance': np.sqrt(distance_sq[row, nearest_index]),
        's': arc_length,
        'tangent': nearest_tangent,
        'signed_lateral': signed_lateral,
    }


def max_adjacent_heading_change(path_points):
    path = _points_xy(path_points, 'path_points', minimum=2)
    segment = np.diff(path, axis=0)
    valid = np.linalg.norm(segment, axis=1) > 1e-9
    segment = segment[valid]
    if len(segment) < 2:
        return 0.0
    heading = np.unwrap(np.arctan2(segment[:, 1], segment[:, 0]))
    return float(np.max(np.abs(np.diff(heading))))


def local_arc_tangents(path_points, window):
    """Estimate local tangents with an arc-length chord around each point."""
    path = _points_xy(path_points, 'path_points', minimum=2)
    path_s = cumulative_s(path)
    half_window = 0.5 * float(window)
    result = np.empty_like(path)
    fallback = tangents(path)
    for index, arc in enumerate(path_s):
        before = int(np.searchsorted(path_s, arc - half_window,
                                     side='left'))
        after = int(np.searchsorted(path_s, arc + half_window,
                                    side='right') - 1)
        before = max(0, min(before, index))
        after = min(len(path) - 1, max(after, index))
        if before == after:
            before = max(0, index - 1)
            after = min(len(path) - 1, index + 1)
        chord = path[after] - path[before]
        norm = float(np.linalg.norm(chord))
        result[index] = chord / norm if norm > 1e-9 else fallback[index]
    return result


def resample_polyline(path_points, spacing):
    """Linearly resample ordered geometry at uniform arc-length spacing."""
    path = _points_xy(path_points, 'path_points', minimum=2)
    path_s = cumulative_s(path)
    total = float(path_s[-1])
    sample_s = np.arange(0.0, total, float(spacing), dtype=np.float64)
    if len(sample_s) == 0 or not np.isclose(sample_s[-1], total):
        sample_s = np.r_[sample_s, total]
    return np.column_stack((
        np.interp(sample_s, path_s, path[:, 0]),
        np.interp(sample_s, path_s, path[:, 1]),
    ))


def deform_path(path_points, lidar_points, config=AvoidanceConfig()):
    """Repel an ordered path from the first current-frame path-near return."""
    input_path = _points_xy(path_points, 'path_points', minimum=2)
    lidar = np.asarray(lidar_points, dtype=np.float64)
    if lidar.ndim != 2 or lidar.shape[1] != 2:
        raise ValueError('lidar_points must have shape (N,2)')
    if not np.all(np.isfinite(lidar)):
        raise ValueError('lidar_points must be finite')
    zero_weights = np.zeros(len(input_path), dtype=np.float64)
    original_heading = max_adjacent_heading_change(input_path)

    def inactive(reason, *, obstacle=None, nearest=None, count=0,
                 lateral=None, s_obstacle=None, clearance=None):
        return AvoidanceResult(
            False, reason, input_path.copy(), input_path.copy(), obstacle, nearest,
            int(count), lateral, s_obstacle, 0.0, zero_weights.copy(),
            clearance, clearance, original_heading, original_heading)

    path = input_path
    if len(lidar) == 0:
        return inactive('NO_LIDAR_POINTS')

    path = resample_polyline(input_path, config.resample_spacing)

    projection = project_to_polyline(lidar, path)
    candidate_mask = projection['distance'] <= config.path_near_distance
    candidate_indices = np.flatnonzero(candidate_mask)
    if len(candidate_indices) == 0:
        return inactive('NO_PATH_NEAR_OBSTACLE')

    # Process only the first relevant obstacle along the ordered near->far
    # path. Aggregate its local scan surface with a median; this is not a
    # persistent cluster and creates no temporal state.
    seed_index = candidate_indices[
        np.argmin(projection['s'][candidate_indices])]
    seed_s = projection['s'][seed_index]
    local = candidate_indices[
        np.abs(projection['s'][candidate_indices] - seed_s)
        <= 0.5 * config.representative_window]
    obstacle = np.median(lidar[local], axis=0)
    obstacle_projection = project_to_polyline(obstacle[None, :], path)
    nearest = obstacle_projection['nearest'][0]
    distance = float(obstacle_projection['distance'][0])
    lateral = float(obstacle_projection['signed_lateral'][0])
    s_obstacle = float(obstacle_projection['s'][0])
    if abs(lateral) <= config.center_deadband:
        return inactive(
            'CENTER_DIRECTION_UNDEFINED', obstacle=obstacle,
            nearest=nearest, count=len(candidate_indices), lateral=lateral,
            s_obstacle=s_obstacle, clearance=distance)

    signed_offset = (-config.avoidance_offset
                     if lateral > 0.0 else config.avoidance_offset)
    path_s = cumulative_s(path)
    distance_s = np.abs(path_s - s_obstacle)
    weights = np.zeros(len(path), dtype=np.float64)
    influenced = distance_s < config.influence_radius
    weights[influenced] = 0.5 * (
        1.0 + np.cos(np.pi * distance_s[influenced]
                     / config.influence_radius))
    path_tangent = local_arc_tangents(path, config.tangent_window)
    normal_left = np.column_stack((-path_tangent[:, 1],
                                   path_tangent[:, 0]))
    deformed = path + weights[:, None] * signed_offset * normal_left
    clearance_avoidance = float(
        project_to_polyline(obstacle[None, :], deformed)['distance'][0])
    return AvoidanceResult(
        True, 'ACTIVE', path.copy(), deformed, obstacle, nearest,
        int(len(candidate_indices)), lateral, s_obstacle,
        float(signed_offset), weights, distance, clearance_avoidance,
        original_heading, max_adjacent_heading_change(deformed))
