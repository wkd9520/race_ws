import numpy as np

from physicar_track_perception_v3.avoidance import (
    AvoidanceConfig,
    deform_path,
    project_to_polyline,
)


def straight_path():
    return np.column_stack((np.linspace(0.0, 2.0, 21), np.zeros(21)))


def config():
    return AvoidanceConfig(
        path_near_distance=0.20,
        representative_window=0.30,
        influence_radius=0.50,
        avoidance_offset=0.25,
        center_deadband=0.01)


def test_left_obstacle_deforms_right_and_improves_clearance():
    result = deform_path(straight_path(), [[1.0, 0.10]], config())
    assert result.active
    assert result.signed_lateral > 0.0
    assert result.signed_offset < 0.0
    assert result.deformed[int(np.argmax(result.weights)), 1] < 0.0
    assert result.clearance_avoidance > result.clearance_original


def test_right_obstacle_is_mirror():
    left = deform_path(straight_path(), [[1.0, 0.10]], config())
    right = deform_path(straight_path(), [[1.0, -0.10]], config())
    assert right.active
    assert right.signed_lateral < 0.0
    assert right.signed_offset > 0.0
    assert np.allclose(left.deformed[:, 0], right.deformed[:, 0])
    assert np.allclose(left.deformed[:, 1], -right.deformed[:, 1])


def test_far_obstacle_is_inactive_and_path_is_unchanged():
    path = straight_path()
    result = deform_path(path, [[1.0, 0.30]], config())
    assert not result.active
    assert result.reason == 'NO_PATH_NEAR_OBSTACLE'
    assert np.array_equal(result.deformed, path)


def test_deformation_is_local_and_returns_to_original():
    path = straight_path()
    result = deform_path(path, [[1.0, 0.10]], config())
    assert np.array_equal(result.deformed[0], path[0])
    assert np.array_equal(result.deformed[-1], path[-1])
    peak = int(np.argmax(result.weights))
    assert result.weights[peak] == 1.0
    assert result.weights[0] == 0.0
    assert result.weights[-1] == 0.0
    assert np.all(np.diff(result.weights[:peak + 1]) >= -1e-12)
    assert np.all(np.diff(result.weights[peak:]) <= 1e-12)


def test_curved_non_x_monotonic_path_uses_local_normal_and_arc_length():
    path = np.array([
        [0.0, 0.0], [0.25, 0.0], [0.50, 0.0], [0.75, 0.0],
        [1.0, 0.0], [1.0, 0.25], [1.0, 0.50], [1.0, 0.75],
        [1.0, 1.0],
    ])
    # On the vertical section, local LEFT is -X. The obstacle is LEFT, so
    # avoidance must move toward +X, not a fixed global +/-Y direction.
    result = deform_path(path, [[0.90, 0.75]], config())
    assert result.active
    assert result.signed_lateral > 0.0
    vertical = result.deformed[:, 1] > 0.50
    assert np.max(result.deformed[vertical, 0]) > 1.0
    assert np.isclose(result.deformed[0, 1], path[0, 1])
    assert result.clearance_avoidance > result.clearance_original


def test_polyline_projection_uses_segments_not_only_samples():
    projection = project_to_polyline([[0.5, 0.2]], [[0.0, 0.0], [1.0, 0.0]])
    assert np.allclose(projection['nearest'][0], [0.5, 0.0])
    assert np.isclose(projection['distance'][0], 0.2)
    assert np.isclose(projection['s'][0], 0.5)
    assert np.isclose(projection['signed_lateral'][0], 0.2)


def test_center_obstacle_is_deterministically_inactive_without_side_rule():
    result = deform_path(straight_path(), [[1.0, 0.0]], config())
    assert not result.active
    assert result.reason == 'CENTER_DIRECTION_UNDEFINED'
    assert np.array_equal(result.deformed, result.original)


def test_local_arc_tangent_does_not_amplify_tiny_connector_reversal():
    path = np.array([
        [0.0, 0.0], [0.20, 0.0], [0.40, 0.0],
        [0.43, 0.01], [0.432, 0.008], [0.45, 0.02],
        [0.60, 0.05], [0.80, 0.10], [1.0, 0.15],
    ])
    result = deform_path(path, [[0.60, -0.05]], config())
    assert result.active
    assert (result.max_heading_step_avoidance
            < result.max_heading_step_original)
