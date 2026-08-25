import math

import numpy as np

from physicar_track_perception_v3.center_hybrid import (
    CURRENT_HYBRID_ORANGE_WHITE,
    CURRENT_ORANGE_ONLY,
    CenterHybridConfig,
    RecentCenterHistory,
    extend_predicted_suffix,
    select_boundary_aligned_suffix,
    stitch_current_frame,
    transform_xy,
)


def matrix(x=0.0, y=0.0, yaw=0.0):
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.array([
        [cosine, -sine, 0.0, x],
        [sine, cosine, 0.0, y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


def baseline(fragments):
    points = []
    for index, fragment in enumerate(fragments):
        if index:
            points.extend(np.linspace(fragments[index - 1][-1],
                                      fragment[0], 4)[1:-1])
        points.extend(fragment)
    return np.asarray(points)


def config(**kwargs):
    values = dict(max_start_distance=0.60, join_gap=0.30,
                  tangent_angle_limit=0.75, history_max_age=0.50,
                  history_max_entries=8)
    values.update(kwargs)
    return CenterHybridConfig(**values)


def test_continuous_orange_is_exact_parity_even_with_white():
    orange = np.array([[0.10, 0.0], [0.30, 0.0], [0.60, 0.0]])
    white = np.array([[0.10, 0.02], [0.35, 0.02], [0.70, 0.02]])
    result = stitch_current_frame(orange, [orange], white, config())
    assert result.source == CURRENT_ORANGE_ONLY
    assert not result.white_used
    assert np.array_equal(result.path, orange)


def test_current_white_replaces_only_orange_gap_geometry():
    first = np.array([[0.10, 0.0], [0.20, 0.0], [0.30, 0.0]])
    second = np.array([[0.55, 0.0], [0.70, 0.0], [0.90, 0.0]])
    orange = baseline([first, second])
    white = np.array([[0.10, 0.0], [0.25, 0.0], [0.40, 0.04],
                      [0.55, 0.0], [0.90, 0.0]])
    result = stitch_current_frame(orange, [first, second], white, config())
    assert result.source == CURRENT_HYBRID_ORANGE_WHITE
    assert result.white_gap_bridge_count == 1
    # Every observed point remains exact and the curved WHITE gap sample is used.
    for point in np.vstack((first, second)):
        assert np.any(np.all(np.isclose(result.path, point), axis=1))
    assert np.any(np.all(np.isclose(result.path, [0.40, 0.04]), axis=1))


def test_bad_white_bridge_keeps_exact_orange_baseline():
    first = np.array([[0.10, 0.0], [0.20, 0.0], [0.30, 0.0]])
    second = np.array([[0.55, 0.0], [0.70, 0.0], [0.90, 0.0]])
    orange = baseline([first, second])
    white = np.array([[0.10, 0.60], [0.50, 0.60], [0.90, 0.60]])
    result = stitch_current_frame(orange, [first, second], white, config())
    assert result.source == CURRENT_ORANGE_ONLY
    assert not result.white_used
    assert np.array_equal(result.path, orange)


def test_vehicle_near_white_prefix_connects_to_far_orange():
    orange = np.array([[0.80, 0.0], [1.00, 0.0], [1.20, 0.0]])
    white = np.array([[0.15, 0.0], [0.35, 0.0], [0.60, 0.0],
                      [0.80, 0.0], [1.10, 0.0]])
    result = stitch_current_frame(orange, [orange], white, config())
    assert result.source == CURRENT_HYBRID_ORANGE_WHITE
    assert result.white_near_prefix_points >= 2
    assert np.linalg.norm(result.path[0]) <= 0.60
    assert np.array_equal(result.path[-len(orange):], orange)


def test_history_not_used_when_current_coverage_is_sufficient():
    history = RecentCenterHistory(config())
    old = np.array([[0.05, 0.0], [0.40, 0.0], [1.0, 0.0]])
    history.store(old, np.eye(4), 10.0)
    current = np.array([[0.20, 0.0], [0.70, 0.0], [1.1, 0.0]])
    result = history.recover(current, np.eye(4), 10.1)
    assert not result.used
    assert result.reason == 'CURRENT_NEAR_COVERAGE_SUFFICIENT'
    assert np.array_equal(result.path, current)


def test_recent_odom_history_supplies_only_near_prefix():
    history = RecentCenterHistory(config())
    observed = np.array([[0.10, 0.0], [0.35, 0.0], [0.60, 0.0],
                         [0.85, 0.0], [1.10, 0.0]])
    history.store(observed, np.eye(4), 20.0)
    current = np.array([[0.80, 0.02], [1.05, 0.02], [1.30, 0.02]])
    result = history.recover(current, np.eye(4), 20.1)
    assert result.used
    assert np.linalg.norm(result.path[0]) <= 0.60
    # Current observation owns the overlap and the complete suffix.
    assert np.array_equal(result.path[-len(current):], current)
    assert result.join_gap <= config().join_gap


def test_history_translation_and_yaw_use_odom_not_old_base_coordinates():
    old_base = np.array([[0.50, 0.0], [1.00, 0.0], [1.50, 0.0]])
    odom_from_old_base = matrix(x=2.0, y=1.0, yaw=math.pi / 2.0)
    stored = transform_xy(old_base, odom_from_old_base)
    # The current vehicle is translated 0.5 m along the old path direction.
    odom_from_current_base = matrix(x=2.0, y=1.5, yaw=math.pi / 2.0)
    current_base_from_odom = np.linalg.inv(odom_from_current_base)
    transformed = transform_xy(stored, current_base_from_odom)
    assert np.allclose(transformed, [[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]])
    assert not np.allclose(transformed, old_base)


def test_stale_history_is_rejected():
    history = RecentCenterHistory(config(history_max_age=0.20))
    history.store(np.array([[0.1, 0.0], [0.6, 0.0], [1.0, 0.0]]),
                  np.eye(4), 1.0)
    current = np.array([[0.8, 0.0], [1.0, 0.0], [1.2, 0.0]])
    result = history.recover(current, np.eye(4), 1.21)
    assert not result.used
    assert result.reason == 'NO_RECENT_HISTORY'


def test_incompatible_history_join_is_rejected():
    history = RecentCenterHistory(config())
    history.store(np.array([[0.1, 0.5], [0.5, 0.5], [1.0, 0.5]]),
                  np.eye(4), 2.0)
    current = np.array([[0.8, 0.0], [1.0, 0.0], [1.2, 0.0]])
    result = history.recover(current, np.eye(4), 2.1)
    assert not result.used
    assert result.reason == 'HISTORY_JOIN_REJECTED'
    assert np.array_equal(result.path, current)


def test_90_degree_history_join_preserves_order_without_x_sorting():
    history = RecentCenterHistory(config())
    curve = np.array([[0.10, 0.0], [0.40, 0.0], [0.70, 0.0],
                      [0.85, 0.10], [0.85, 0.40], [0.85, 0.75]])
    history.store(curve, np.eye(4), 5.0)
    current = np.array([[0.85, 0.40], [0.85, 0.65], [0.85, 0.90]])
    result = history.recover(current, np.eye(4), 5.1)
    assert result.used
    assert np.array_equal(result.path[-len(current):], current)
    assert np.any(np.diff(result.path[:, 0]) > 0.0)
    assert np.any(np.isclose(np.diff(result.path[:, 0]), 0.0))


def test_90_degree_current_white_gap_bridge_uses_local_order():
    first = np.array([[0.10, 0.0], [0.35, 0.0], [0.55, 0.0]])
    second = np.array([[0.70, 0.15], [0.70, 0.40], [0.70, 0.65]])
    orange = baseline([first, second])
    white = np.array([[0.10, 0.0], [0.35, 0.0], [0.55, 0.0],
                      [0.65, 0.05], [0.70, 0.15], [0.70, 0.40],
                      [0.70, 0.65]])
    result = stitch_current_frame(orange, [first, second], white, config())
    assert result.source == CURRENT_HYBRID_ORANGE_WHITE
    assert result.white_gap_bridge_count == 1
    assert np.array_equal(result.path[:len(first)], first)
    assert np.array_equal(result.path[-len(second):], second)
    # The ordered result turns upward; it is not globally sorted by X.
    assert np.any(np.isclose(np.diff(result.path[:, 0]), 0.0))


def test_used_white_does_not_extend_beyond_last_orange_fragment():
    first = np.array([[0.10, 0.0], [0.20, 0.0], [0.30, 0.0]])
    second = np.array([[0.55, 0.0], [0.70, 0.0], [0.80, 0.0]])
    orange = baseline([first, second])
    white = np.array([[0.10, 0.0], [0.30, 0.0], [0.42, 0.03],
                      [0.55, 0.0], [0.80, 0.0], [1.00, 0.0],
                      [1.20, 0.0]])
    result = stitch_current_frame(
        orange, [first, second], white, config())
    assert result.white_gap_bridge_count == 1
    assert result.white_far_suffix_points == 0
    assert np.allclose(result.path[-1], second[-1])
    assert not np.allclose(result.path[-1], white[-1])
    # Every observed ORANGE point is still preserved exactly.
    for point in np.vstack((first, second)):
        assert np.any(np.all(np.isclose(result.path, point), axis=1))


def test_predicted_suffix_retains_first_point_beyond_bev_edge():
    measured = np.array([[0.20, 0.0], [0.60, 0.0], [1.00, 0.0]])
    result = extend_predicted_suffix(
        measured, (0.1, 2.0, -0.75, 0.75), spacing=0.05)
    assert result.reason == 'CURRENT_TANGENT_SUFFIX'
    assert result.length > 0.70
    assert np.allclose(result.path[:len(measured)], measured)
    assert np.isclose(result.suffix[-1, 0], 2.0)
    assert result.suffix[-2, 0] < 2.0
    assert np.allclose(result.suffix[:, 1], 0.0)


def test_predicted_suffix_follows_90_degree_endpoint_tangent_to_bounds():
    measured = np.array([[0.10, 0.0], [0.50, 0.0], [0.70, 0.20],
                         [0.70, 0.40]])
    result = extend_predicted_suffix(
        measured, (0.1, 2.0, -0.75, 0.75), spacing=0.05,
        tangent_window=0.15)
    assert len(result.suffix) > 0
    assert np.allclose(result.suffix[:, 0], 0.70)
    assert np.all(np.diff(result.suffix[:, 1]) > 0.0)
    assert result.suffix[-2, 1] < 0.75
    assert result.suffix[-1, 1] >= 0.75


def test_predicted_suffix_is_preserved_beyond_camera_bev_for_lidar_voting():
    measured = np.array([[1.20, 0.0], [1.60, 0.0], [1.90, 0.0]])
    result = extend_predicted_suffix(
        measured, (-0.5, 4.0, -2.0, 2.0), spacing=0.05)
    assert np.any(result.suffix[:, 0] >= 2.0)
    assert np.isclose(result.suffix[-1, 0], 4.0)
    assert result.suffix[-2, 0] < 4.0
    assert np.allclose(result.path[:len(measured)], measured)


def test_predicted_suffix_keeps_small_non_aligned_bounds_overrun():
    measured = np.array([[1.20, 0.0], [1.60, 0.0], [1.98, 0.0]])
    result = extend_predicted_suffix(
        measured, (0.1, 2.0, -0.75, 0.75), spacing=0.05)
    assert np.isclose(result.suffix[-1, 0], 2.03)
    assert 0.0 < result.suffix[-1, 0] - 2.0 <= 0.05 + 1e-12


def test_white_shaped_suffix_wins_when_it_matches_curved_boundary():
    measured = np.array([[0.10, 0.0], [0.50, 0.0], [0.90, 0.0]])
    white_center = np.array([
        [0.60, 0.00], [0.90, 0.00], [1.05, 0.03],
        [1.20, 0.12], [1.35, 0.27], [1.45, 0.45],
    ])
    tangent = np.gradient(white_center, axis=0)
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    white_boundary = white_center + 0.37 * np.column_stack((
        -tangent[:, 1], tangent[:, 0]))

    result = select_boundary_aligned_suffix(
        measured, white_center, [white_boundary],
        (-0.5, 2.0, -2.0, 2.0))

    assert result.selected_source == 'WHITE_GEOMETRY'
    assert result.white_candidate_valid
    assert result.white_boundary_matches > 0
    assert result.white_boundary_score < result.tangent_boundary_score
    assert np.array_equal(result.path[:len(measured)], measured)


def test_tangent_suffix_wins_when_white_candidate_disagrees_with_boundary():
    measured = np.array([[0.10, 0.0], [0.50, 0.0], [0.90, 0.0]])
    curved_white_center = np.array([
        [0.60, 0.00], [0.90, 0.00], [1.05, 0.03],
        [1.20, 0.12], [1.35, 0.27], [1.45, 0.45],
    ])
    straight_boundary = np.array([
        [0.60, 0.37], [1.00, 0.37], [1.50, 0.37], [2.00, 0.37],
    ])

    result = select_boundary_aligned_suffix(
        measured, curved_white_center, [straight_boundary],
        (-0.5, 2.0, -2.0, 2.0))

    assert result.selected_source == 'PATH_TANGENT'
    assert result.tangent_boundary_score < result.white_boundary_score
    assert np.allclose(result.suffix[:, 1], 0.0)
    assert np.array_equal(result.path[:len(measured)], measured)


def test_suffix_selector_without_white_preserves_existing_tangent_result():
    measured = np.array([[0.20, 0.0], [0.60, 0.0], [1.00, 0.0]])
    previous = extend_predicted_suffix(
        measured, (0.1, 2.0, -0.75, 0.75), spacing=0.05)
    result = select_boundary_aligned_suffix(
        measured, None, [], (0.1, 2.0, -0.75, 0.75), spacing=0.05)

    assert result.selected_source == 'PATH_TANGENT'
    assert not result.white_candidate_valid
    assert result.white_candidate_reason == 'NO_CURRENT_WHITE_CENTER'
    assert np.array_equal(result.path, previous.path)


def test_incompatible_white_shape_is_scored_but_cannot_create_a_kink():
    measured = np.array([[0.10, 0.0], [0.50, 0.0], [0.90, 0.0]])
    perpendicular_white = np.array([
        [0.90, 0.0], [0.90, 0.20], [0.90, 0.45], [0.90, 0.70],
    ])
    white_boundary = perpendicular_white + np.array([-0.37, 0.0])

    result = select_boundary_aligned_suffix(
        measured, perpendicular_white, [white_boundary],
        (-0.5, 2.0, -2.0, 2.0))

    assert len(result.white_candidate) > 0
    assert result.white_boundary_score is not None
    assert not result.white_candidate_valid
    assert result.white_candidate_reason == 'WHITE_CANDIDATE_TANGENT_MISMATCH'
    assert result.white_candidate_join_angle > 0.75
    assert result.selected_source == 'PATH_TANGENT'
