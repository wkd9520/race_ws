import numpy as np

from physicar_track_perception_v2.trusted_corridor import (
    TrustedCorridorReference,)


def tf(tx=0., ty=0., yaw=0.):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, tx], [s, c, ty], [0., 0., 1.]])


def reference():
    return TrustedCorridorReference.from_center(
        np.array([[0., 0.], [.5, 0.], [1., 0.]]), .7, tf(2., -1., .2), 10.)


def test_motion_compensated_current_geometry():
    value = reference()
    current_tf = tf(2., -1., .2)
    expected = np.array([[0., 0.], [.5, 0.], [1., 0.]])
    current = value.current_center(current_tf)
    assert np.allclose(current, expected, atol=1e-9)


def test_inside_endpoint_and_outside_coverage_are_distinct():
    value = reference()
    current_tf = tf(2., -1., .2)
    inside = np.array([[.25, .35], [.75, .35]])
    outside = np.array([[1.3, .35], [1.5, .35]])
    assert value.coverage(inside, current_tf).state == 'INTERIOR_CORRESPONDENCE'
    cov = value.coverage(outside, current_tf)
    assert cov.state == 'OUT_OF_CORRIDOR_REFERENCE_COVERAGE'
    assert np.all(cov.outside_mask)


def test_outside_never_reuses_endpoint_tangent():
    value = reference()
    current_tf = tf(2., -1., .2)
    points = np.array([[1.3, .35], [1.5, .35]])
    result = value.evaluate_side('LEFT', points, current_tf)
    assert result.state == 'SIDE_REFERENCE_OUT_OF_COVERAGE'
    assert result.coverage.outside_fraction == 1.0


def test_side_and_opposite_protection():
    value = reference()
    current_tf = tf(2., -1., .2)
    left = value.evaluate_side('LEFT', np.array([[.25, .35], [.75, .35]]), current_tf)
    right_as_left = value.evaluate_side('LEFT', np.array([[.25, -.35], [.75, -.35]]), current_tf)
    assert left.state == 'SIDE_CONSISTENT'
    assert right_as_left.state == 'SIDE_OPPOSITE'


def test_combined_motion_transform_is_supported():
    value = reference()
    current_tf = tf(2.4, -.7, -.3)
    center = value.current_center(current_tf)
    assert center.shape == (3, 2)
    assert np.all(np.isfinite(center))
