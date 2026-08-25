"""ROS-independent Stage 5.3 perception-to-controller contract.

The geometry in this module is not a new planner.  It selects one current
Stage 5.1R circle component by the Stage 5.2B ACTIVE identity and applies the
existing safety-target/shadow-path functions with the Stage 5.2A/C locked
avoidance side.
"""

from dataclasses import dataclass
import math

import numpy as np

from .active_obstacle import ACTIVE, ACTIVE_LOST, NO_ACTIVE
from .avoidance import project_to_polyline
from .circle_avoidance import (
    AVOID_LEFT,
    AVOID_RIGHT,
    CircleAvoidanceConfig,
    PathRelativeCircle,
    avoidance_target,
    build_shadow_path,
    classify_circle,
    pure_pursuit_shadow,
)
from .low_vote_recovery import (
    MODE_CRAWL,
    MODE_HEADING_RECOVERY,
    MODE_NONE,
)


CENTER_PATH = 'CENTER'
CENTER_CRAWL = 'CENTER_CRAWL'
HEADING_RECOVERY_CONTROL = 'HEADING_RECOVERY'
AVOIDANCE_PATH = 'AVOIDANCE'
STOP = 'STOP'


@dataclass(frozen=True)
class ActiveAvoidanceGeometry:
    valid: bool
    reason: str
    component_id: int | None
    locked_side: str | None
    path: np.ndarray
    target: np.ndarray | None
    safety_radius: float | None
    target_lateral_offset: float | None
    clearance_original: float | None
    clearance_avoidance: float | None
    safety_clearance_avoidance: float | None
    steering_original: dict | None
    steering_avoidance: dict | None


@dataclass(frozen=True)
class ControlModeDecision:
    mode: str
    reason: str


def _invalid_geometry(reason, component_id=None, locked_side=None):
    return ActiveAvoidanceGeometry(
        False, str(reason), component_id, locked_side,
        np.empty((0, 2), dtype=np.float64), None, None, None,
        None, None, None, None, None)


def build_active_avoidance_geometry(
        components, fits, path_points, component_id, locked_side,
        config=CircleAvoidanceConfig()):
    """Build the existing avoidance path for one explicitly selected ACTIVE.

    ``component_id`` must be the current raw/circle observation associated
    with the latched ACTIVE track.  No nearest-obstacle selection or direction
    voting occurs here.
    """
    reference = np.asarray(path_points, dtype=np.float64)
    if (reference.ndim != 2 or reference.shape[1] != 2 or len(reference) < 2
            or not np.all(np.isfinite(reference))):
        return _invalid_geometry(
            'INVALID_CENTER_PATH', component_id, locked_side)
    if component_id is None:
        return _invalid_geometry(
            'ACTIVE_COMPONENT_UNAVAILABLE', None, locked_side)
    if locked_side not in (AVOID_LEFT, AVOID_RIGHT):
        return _invalid_geometry(
            'ACTIVE_DIRECTION_NOT_LOCKED', component_id, locked_side)

    component_by_id = {
        int(item.component_id): item for item in tuple(components)}
    fit_by_id = {int(item.component_id): item for item in tuple(fits)}
    component = component_by_id.get(int(component_id))
    fit = fit_by_id.get(int(component_id))
    if component is None:
        return _invalid_geometry(
            'ACTIVE_COMPONENT_UNAVAILABLE', component_id, locked_side)
    if fit is None or not fit.valid or fit.center is None or fit.radius is None:
        reason = ('ACTIVE_CIRCLE_UNAVAILABLE' if fit is None
                  else f'ACTIVE_CIRCLE_{fit.reason}')
        return _invalid_geometry(reason, component_id, locked_side)

    try:
        frame = classify_circle(fit.center, reference, config)
        selected = PathRelativeCircle(
            component=component,
            fit=fit,
            nearest_path_point=frame['nearest'],
            tangent=frame['tangent'],
            normal_left=frame['normal_left'],
            s_obstacle=frame['s'],
            signed_lateral=frame['signed_lateral'],
            path_center_distance=frame['distance'],
            path_surface_distance=max(0.0, frame['distance']-fit.radius),
            vehicle_center_distance=float(np.linalg.norm(fit.center)),
        )
        target, safety_radius, lateral_offset = avoidance_target(
            selected, locked_side, config)
        shadow, _ = build_shadow_path(reference, selected, target, config)
        avoidance_center_distance = float(project_to_polyline(
            fit.center[None, :], shadow)['distance'][0])
        original_clearance = float(frame['distance']-fit.radius)
        avoidance_clearance = float(avoidance_center_distance-fit.radius)
        safety_clearance = float(avoidance_center_distance-safety_radius)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        return _invalid_geometry(
            f'ACTIVE_GEOMETRY_ERROR:{exc}', component_id, locked_side)

    if (not np.all(np.isfinite(shadow)) or len(shadow) < 2
            or target is None or not np.all(np.isfinite(target))):
        return _invalid_geometry(
            'ACTIVE_GEOMETRY_NONFINITE', component_id, locked_side)
    return ActiveAvoidanceGeometry(
        valid=True,
        reason='ACTIVE_LOCKED_GEOMETRY',
        component_id=int(component_id),
        locked_side=locked_side,
        path=shadow,
        target=np.asarray(target, dtype=np.float64),
        safety_radius=float(safety_radius),
        target_lateral_offset=float(lateral_offset),
        clearance_original=original_clearance,
        clearance_avoidance=avoidance_clearance,
        safety_clearance_avoidance=safety_clearance,
        steering_original=pure_pursuit_shadow(reference, config),
        steering_avoidance=pure_pursuit_shadow(shadow, config),
    )


def choose_control_mode(lifecycle_state, active_track_present,
                        active_observed, direction_locked,
                        avoidance_path_valid, center_path_valid,
                        requested_mode=MODE_NONE,
                        recovery_heading_target=None):
    """Return a fail-closed Stage 5.3 controller mode."""
    if lifecycle_state == ACTIVE_LOST:
        return ControlModeDecision(STOP, 'ACTIVE_LOST')
    if lifecycle_state == NO_ACTIVE:
        if center_path_valid:
            return ControlModeDecision(CENTER_PATH, 'NO_ACTIVE_CENTER_PATH')
        return ControlModeDecision(STOP, 'CENTER_PATH_INVALID')
    if lifecycle_state != ACTIVE:
        return ControlModeDecision(STOP, 'UNKNOWN_LIFECYCLE_STATE')
    if not active_track_present:
        return ControlModeDecision(STOP, 'ACTIVE_TRACK_UNAVAILABLE')
    if not active_observed:
        return ControlModeDecision(STOP, 'ACTIVE_COMPONENT_UNAVAILABLE')
    if direction_locked:
        if avoidance_path_valid:
            return ControlModeDecision(
                AVOIDANCE_PATH, 'ACTIVE_LOCKED_AVOIDANCE_PATH')
        return ControlModeDecision(STOP, 'ACTIVE_AVOIDANCE_PATH_INVALID')
    if requested_mode == MODE_HEADING_RECOVERY:
        if (recovery_heading_target is not None
                and math.isfinite(float(recovery_heading_target))):
            return ControlModeDecision(
                HEADING_RECOVERY_CONTROL, 'BOUNDED_HEADING_RECOVERY')
        return ControlModeDecision(STOP, 'HEADING_RECOVERY_TARGET_INVALID')
    if requested_mode == MODE_CRAWL:
        if center_path_valid:
            return ControlModeDecision(CENTER_CRAWL, 'SLOW_VOTING_CRAWL')
        return ControlModeDecision(STOP, 'CENTER_PATH_INVALID_FOR_CRAWL')
    if requested_mode == MODE_NONE:
        if center_path_valid:
            return ControlModeDecision(CENTER_PATH, 'ACTIVE_NORMAL_VOTING')
        return ControlModeDecision(STOP, 'CENTER_PATH_INVALID')
    return ControlModeDecision(STOP, 'UNKNOWN_RECOVERY_MODE')
