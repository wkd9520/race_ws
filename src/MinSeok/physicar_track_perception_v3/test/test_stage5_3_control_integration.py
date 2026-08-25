from dataclasses import replace

import numpy as np

from physicar_track_perception_v3.active_obstacle import (
    ACTIVE, ACTIVE_LOST, NO_ACTIVE)
from physicar_track_perception_v3.circle_avoidance import (
    AVOID_LEFT, AVOID_RIGHT, CircleAvoidanceConfig, LidarComponent,
    fit_circle)
from physicar_track_perception_v3.control_integration import (
    AVOIDANCE_PATH, CENTER_CRAWL, CENTER_PATH, HEADING_RECOVERY_CONTROL,
    STOP, build_active_avoidance_geometry, choose_control_mode)
from physicar_track_perception_v3.low_vote_recovery import (
    MODE_CRAWL, MODE_HEADING_RECOVERY, MODE_NONE)


def _path():
    return np.column_stack((np.linspace(0.1, 2.0, 39), np.zeros(39)))


def _circle_component(component_id=7, center=(0.8, 0.10), radius=0.08):
    angles = np.linspace(-2.4, 2.4, 31)
    points = np.asarray(center)+radius*np.column_stack((
        np.cos(angles), np.sin(angles)))
    support = float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())
    component = LidarComponent(
        component_id=component_id,
        beam_indices=np.arange(31),
        points=points,
        point_count=len(points),
        nearest_distance=float(np.min(np.linalg.norm(points, axis=1))),
        centroid=np.mean(points, axis=0),
        span=float(np.linalg.norm(points[-1]-points[0])),
        support=support,
        wall_like=False,
    )
    fitted = replace(fit_circle(component), component_id=component_id)
    assert fitted.valid
    return component, fitted


def test_active_identity_and_locked_side_drive_existing_geometry():
    component, fitted = _circle_component()
    right = build_active_avoidance_geometry(
        (component,), (fitted,), _path(), 7, AVOID_RIGHT)
    left = build_active_avoidance_geometry(
        (component,), (fitted,), _path(), 7, AVOID_LEFT)
    assert right.valid and left.valid
    assert right.component_id == left.component_id == 7
    assert right.target[1] < fitted.center[1]
    assert left.target[1] > fitted.center[1]
    assert np.min(right.path[:, 1]) < 0.0
    assert np.max(left.path[:, 1]) > 0.0
    assert right.clearance_avoidance > right.clearance_original
    assert left.clearance_avoidance > left.clearance_original


def test_active_geometry_fails_closed_on_wrong_or_missing_provenance():
    component, fitted = _circle_component()
    missing = build_active_avoidance_geometry(
        (component,), (fitted,), _path(), 8, AVOID_RIGHT)
    unlocked = build_active_avoidance_geometry(
        (component,), (fitted,), _path(), 7, None)
    bad_path = build_active_avoidance_geometry(
        (component,), (fitted,), [[0.1, 0.0]], 7, AVOID_RIGHT)
    assert not missing.valid and missing.reason == 'ACTIVE_COMPONENT_UNAVAILABLE'
    assert not unlocked.valid and unlocked.reason == 'ACTIVE_DIRECTION_NOT_LOCKED'
    assert not bad_path.valid and bad_path.reason == 'INVALID_CENTER_PATH'


def test_control_mode_normal_crawl_heading_avoidance_and_handoff():
    normal = choose_control_mode(
        NO_ACTIVE, False, False, False, False, True)
    crawl = choose_control_mode(
        ACTIVE, True, True, False, False, True, MODE_CRAWL)
    heading = choose_control_mode(
        ACTIVE, True, True, False, False, True,
        MODE_HEADING_RECOVERY, 0.2)
    avoidance = choose_control_mode(
        ACTIVE, True, True, True, True, True, MODE_NONE)
    next_avoidance = choose_control_mode(
        ACTIVE, True, True, True, True, True, MODE_NONE)
    assert normal.mode == CENTER_PATH
    assert crawl.mode == CENTER_CRAWL
    assert heading.mode == HEADING_RECOVERY_CONTROL
    assert avoidance.mode == next_avoidance.mode == AVOIDANCE_PATH


def test_control_mode_stops_for_lost_missing_or_invalid_required_geometry():
    cases = (
        choose_control_mode(ACTIVE_LOST, False, False, False, False, True),
        choose_control_mode(ACTIVE, True, False, True, True, True),
        choose_control_mode(ACTIVE, True, True, True, False, True),
        choose_control_mode(NO_ACTIVE, False, False, False, False, False),
        choose_control_mode(
            ACTIVE, True, True, False, False, True,
            MODE_HEADING_RECOVERY, None),
    )
    assert all(item.mode == STOP for item in cases)
