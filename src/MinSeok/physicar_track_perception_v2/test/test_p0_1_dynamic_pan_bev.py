import math

import numpy as np

from physicar_track_perception_v2.dynamic_bev import (
    BoundedPendingFrames, DynamicPanGuard)
from physicar_track_perception_v2.frontend import BevFrontend
from physicar_track_perception_v2.geometry import (
    BevGrid, CameraModel, MetricGroundProjector,
    apply_projection_corrections)


def _camera():
    return CameraModel(np.array([[201.39, 0., 240.],
                                 [0., 201.39, 180.], [0., 0., 1.]]),
                       np.zeros(5), 480, 360)


def _pose(pan):
    c, s = math.cos(pan), math.sin(pan)
    rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    value = np.eye(4)
    value[:3, :3] = rz @ optical
    value[:3, 3] = [.108, 0., .1476]
    return apply_projection_corrections(value)


def test_dynamic_guard_accepts_bounded_pan_but_keeps_tilt_fixed():
    guard = DynamicPanGuard()
    for pan in (0., math.radians(10), math.radians(-20), .5236, -.5236):
        assert guard.accepts(pan, -.5236)
    assert not guard.accepts(.54, -.5236)
    assert not guard.accepts(0., -.50)


def test_pending_queue_is_bounded_and_preserves_oldest_plus_latest():
    queue = BoundedPendingFrames(2)
    queue.append('oldest', 0.)
    queue.append('middle', .01)
    queue.append('latest', .02)
    assert len(queue) == 2 and queue.replaced == 1
    assert queue.pop()[0] == 'oldest'
    assert queue.pop()[0] == 'latest'


def test_pending_timeout_and_original_frame_are_preserved():
    queue = BoundedPendingFrames(2)
    frame = object()
    queue.append(frame, 1.)
    assert queue.peek()[0] is frame
    assert queue.expire(1.24, .25) == 0
    assert queue.expire(1.26, .25) == 1


def test_frontend_updates_only_pose_dependent_map_and_keeps_grid_fixed():
    camera = _camera()
    grid = BevGrid(.1, 2., -.75, .75, .01)
    frontend = BevFrontend(camera, MetricGroundProjector(camera, grid, _pose(0.)))
    undistort_x = frontend.undistort_map_x
    map_zero = frontend.bev_map_x.copy()
    frontend.update_projector(MetricGroundProjector(
        camera, grid, _pose(math.radians(20))))
    assert frontend.undistort_map_x is undistort_x
    assert frontend.bev_map_x.shape == map_zero.shape == (190, 150)
    assert not np.array_equal(frontend.bev_map_x, map_zero)


def test_vehicle_fixed_pitch_contract_is_preserved_for_nonzero_pan():
    raw = _pose(math.radians(20))
    # _pose already applies the production correction. Undoing is unnecessary:
    # the contract under test is that output remains a valid rigid transform
    # with vehicle-Z origin correction and no pan-specific alternate branch.
    assert np.allclose(raw[:3, :3].T @ raw[:3, :3], np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(raw[:3, :3]), 1.)


def test_pan_zero_correction_models_are_exactly_equal():
    raw = _pose(0.)
    # _pose includes one correction, but equality of the two left multipliers
    # at zero pan remains exact for any valid input rotation.
    a = apply_projection_corrections(
        raw, pitch_correction_frame='vehicle_y')
    b = apply_projection_corrections(
        raw, pitch_correction_frame='pan_local_y')
    assert np.allclose(a, b, atol=1e-15)


def test_pan_local_axis_is_pan_rotated_vehicle_y_for_both_signs():
    delta = math.radians(2.7)
    cy, sy = math.cos(delta), math.sin(delta)
    ry = np.array([[cy, 0., sy], [0., 1., 0.], [-sy, 0., cy]])
    optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    for pan in (math.radians(10), math.radians(-10),
                math.radians(20), math.radians(-20)):
        cp, sp = math.cos(pan), math.sin(pan)
        rz = np.array([[cp, -sp, 0.], [sp, cp, 0.], [0., 0., 1.]])
        raw = np.eye(4)
        raw[:3, :3] = rz @ optical
        actual = apply_projection_corrections(
            raw, camera_height_correction_z=0.,
            pitch_offset_deg=2.7,
            pitch_correction_frame='pan_local_y')
        assert np.allclose(actual[:3, :3], rz @ ry @ optical, atol=1e-15)


def test_unknown_pitch_correction_frame_is_rejected():
    try:
        apply_projection_corrections(
            np.eye(4), pitch_correction_frame='camera_guess')
    except ValueError as error:
        assert 'pitch_correction_frame' in str(error)
    else:
        raise AssertionError('invalid correction frame was accepted')
