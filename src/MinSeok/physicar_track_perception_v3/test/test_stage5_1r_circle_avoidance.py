import numpy as np

from physicar_track_perception_v3.circle_avoidance import (
    AVOID_LEFT,
    AVOID_RIGHT,
    LEFT,
    NORMAL,
    RIGHT,
    VOTING,
    CircleAvoidanceConfig,
    CircleAvoidanceEngine,
    DirectionLatch,
    LidarComponent,
    analyze_components,
    avoidance_target,
    build_components,
    build_shadow_path,
    classify_circle,
    fit_circle,
    pure_pursuit_shadow,
    relevant_circles,
)


def config(**overrides):
    values = dict(
        component_gap=0.08,
        max_obstacle_support=0.70,
        min_circle_points=3,
        min_circle_radius=0.01,
        max_circle_radius=0.40,
        max_circle_residual=0.05,
        path_near_distance=0.20,
        direction_freeze_distance=1.0,
        component_continuity_distance=0.45,
        default_avoidance_side=AVOID_LEFT,
        safety_margin=0.10,
        additional_clearance=0.05,
        approach_length=0.80,
        return_length=0.80,
        tangent_window=0.20,
        resample_spacing=0.05,
    )
    values.update(overrides)
    return CircleAvoidanceConfig(**values)


def straight_path(length=2.0):
    return np.column_stack((np.linspace(0.1, length, 39), np.zeros(39)))


def circle_points(center=(1.2, 0.1), radius=0.08,
                  begin=-2.5, end=2.5, count=31, noise=0.0):
    angles = np.linspace(begin, end, count)
    points = np.asarray(center) + radius * np.column_stack((
        np.cos(angles), np.sin(angles)))
    if noise:
        points = points + np.random.default_rng(7).normal(
            scale=noise, size=points.shape)
    return points


def component_from_points(points, support=None, wall_like=False):
    points = np.asarray(points, dtype=np.float64)
    actual_support = (float(np.linalg.norm(
        np.diff(points, axis=0), axis=1).sum()) if len(points) > 1 else 0.0)
    if support is None:
        support = actual_support
    return LidarComponent(
        component_id=0,
        beam_indices=np.arange(len(points)),
        points=points,
        point_count=len(points),
        nearest_distance=float(np.min(np.linalg.norm(points, axis=1))),
        centroid=np.mean(points, axis=0),
        span=float(np.linalg.norm(points[-1] - points[0])) if len(points) > 1 else 0.0,
        support=float(support),
        wall_like=wall_like,
    )


def scan_circle(center, measurement, path=None):
    points = circle_points(center=center)
    beams = np.arange(200, 200 + len(points))
    return points, beams, straight_path() if path is None else path, measurement


def test_consecutive_nearby_beams_form_one_component_and_gap_splits():
    points = np.array([[0.0, 0.0], [0.02, 0.0], [0.04, 0.0],
                       [0.20, 0.0], [0.22, 0.0]])
    components = build_components(points, np.arange(5), config())
    assert [item.point_count for item in components] == [3, 2]


def test_missing_beam_index_splits_even_when_metric_points_are_close():
    points = np.array([[1.0, 0.0], [1.01, 0.0], [1.02, 0.0]])
    components = build_components(points, [10, 11, 13], config())
    assert [item.point_count for item in components] == [2, 1]


def test_curved_short_surface_is_one_component():
    points = circle_points(count=20)
    components = build_components(points, np.arange(20), config())
    assert len(components) == 1
    assert not components[0].wall_like


def test_long_wall_is_one_component_and_wall_like():
    points = np.column_stack((np.linspace(0.0, 1.2, 121),
                              np.zeros(121)))
    components = build_components(
        points, np.arange(len(points)), config(component_gap=0.02))
    assert len(components) == 1
    assert components[0].support > 1.0
    assert components[0].wall_like


def test_wall_rejection_support_boundary():
    for support, expected in ((0.20, False), (0.60, False),
                              (0.69, False), (0.70, True),
                              (0.71, True)):
        points = np.column_stack((
            np.linspace(0.0, support, int(round(support * 100)) + 1),
            np.zeros(int(round(support * 100)) + 1)))
        component = build_components(
            points, np.arange(len(points)),
            config(component_gap=0.02))[0]
        assert component.wall_like is expected
        fit = fit_circle(component, config(component_gap=0.02))
        if expected:
            assert not fit.valid and fit.reason == 'WALL_LIKE'


def test_rear_component_is_retained_without_angle_filter():
    points = circle_points(center=(-1.0, 0.0), count=17)
    components, _ = analyze_components(
        points, np.arange(500, 517), config())
    assert len(components) == 1
    assert np.all(components[0].points[:, 0] < 0.0)


def test_ideal_partial_circle_fit():
    fit = fit_circle(component_from_points(circle_points()), config())
    assert fit.valid
    assert np.allclose(fit.center, [1.2, 0.1], atol=1e-8)
    assert np.isclose(fit.radius, 0.08, atol=1e-8)
    assert fit.residual < 1e-8


def test_noisy_partial_circle_fit_and_larger_angular_span():
    narrow = fit_circle(component_from_points(
        circle_points(begin=-0.8, end=0.8, noise=0.001)), config())
    wide = fit_circle(component_from_points(
        circle_points(begin=-2.5, end=2.5, noise=0.001)), config())
    assert narrow.valid and wide.valid
    assert np.linalg.norm(wide.center - [1.2, 0.1]) < 0.01
    assert abs(wide.radius - 0.08) < 0.01


def test_circle_fit_rejects_insufficient_degenerate_and_bad_radius():
    insufficient = fit_circle(component_from_points(
        [[1.0, 0.0], [1.0, 0.1]]), config())
    assert insufficient.reason == 'INSUFFICIENT_POINTS'
    degenerate = fit_circle(component_from_points(
        [[1.0, 0.0], [1.0, 0.1], [1.0, 0.2], [1.0, 0.3]]), config())
    assert degenerate.reason == 'DEGENERATE_FIT'
    too_large = fit_circle(component_from_points(
        circle_points(radius=0.08)), config(max_circle_radius=0.05))
    assert too_large.reason == 'RADIUS_OUT_OF_RANGE'


def test_circle_fit_rejects_extreme_residual():
    points = circle_points(count=20)
    points[::2] += np.array([0.08, 0.0])
    fit = fit_circle(component_from_points(points),
                     config(max_circle_residual=0.005))
    assert not fit.valid
    assert fit.reason == 'RESIDUAL_TOO_LARGE'


def test_lane_side_straight_and_mirror():
    left = classify_circle([1.0, 0.1], straight_path(), config())
    right = classify_circle([1.0, -0.1], straight_path(), config())
    assert left['side'] == LEFT and left['signed_lateral'] > 0.0
    assert right['side'] == RIGHT and right['signed_lateral'] < 0.0


def test_lane_side_on_vertical_90_degree_path_uses_local_normal():
    path = np.array([[0.1, 0.0], [0.5, 0.0], [1.0, 0.0],
                     [1.0, 0.4], [1.0, 0.8], [1.0, 1.2]])
    frame = classify_circle([0.9, 0.8], path, config())
    assert frame['side'] == LEFT
    assert frame['normal_left'][0] < -0.9


def test_vote_sequence_counts_each_valid_frame():
    latch = DirectionLatch(config())
    for side in (LEFT, LEFT, RIGHT, LEFT, LEFT):
        latch.observe(side, 1.2)
    assert latch.state == VOTING
    assert (latch.left_votes, latch.right_votes) == (4, 1)


def test_freeze_occurs_at_first_distance_below_one_meter():
    latch = DirectionLatch(config())
    for distance in (1.50, 1.25, 1.08, 1.01):
        assert latch.observe(LEFT, distance) is None
        assert not latch.locked
    assert latch.observe(LEFT, 0.99) == AVOID_RIGHT
    assert latch.locked and latch.state == AVOID_RIGHT


def test_narrow_majority_locks_without_margin():
    latch = DirectionLatch(config())
    latch.left_votes, latch.right_votes = 21, 20
    latch.state = VOTING
    assert latch.observe(RIGHT, 0.99) == AVOID_RIGHT


def test_exact_tie_uses_clearance_then_deterministic_default():
    latch = DirectionLatch(config(default_avoidance_side=AVOID_LEFT))
    latch.left_votes = latch.right_votes = 20
    latch.state = VOTING
    assert latch.observe(LEFT, 0.99, 0.2, 0.4) == AVOID_RIGHT
    latch.reset(); latch.left_votes = latch.right_votes = 20
    latch.state = VOTING
    assert latch.observe(RIGHT, 0.99) == AVOID_LEFT


def test_direction_latch_ignores_later_instantaneous_side_changes():
    latch = DirectionLatch(config())
    for _ in range(4):
        latch.observe(LEFT, 1.2)
    assert latch.observe(RIGHT, 0.9) == AVOID_RIGHT
    for side in (LEFT, RIGHT, RIGHT, LEFT):
        assert latch.observe(side, 0.7) == AVOID_RIGHT
    assert (latch.left_votes, latch.right_votes) == (4, 0)


def test_reset_clears_votes_lock_and_state():
    latch = DirectionLatch(config())
    latch.observe(LEFT, 1.2)
    latch.observe(LEFT, 0.9)
    latch.reset()
    assert latch.state == NORMAL
    assert (latch.left_votes, latch.right_votes) == (0, 0)
    assert not latch.locked


def _selected(center=(0.9, 0.1), path=None):
    if path is None:
        path = straight_path()
    points = circle_points(center=center)
    beams = np.arange(len(points))
    components, fits = analyze_components(points, beams, config())
    return relevant_circles(components, fits, path, config())[0]


def test_avoidance_target_is_outside_safety_circle_on_locked_side():
    selected = _selected()
    left, safe, _ = avoidance_target(selected, AVOID_LEFT, config())
    right, safe_mirror, _ = avoidance_target(selected, AVOID_RIGHT, config())
    expected_radius = selected.fit.radius + config().safety_margin
    assert np.isclose(safe, expected_radius)
    assert np.isclose(safe_mirror, expected_radius)
    assert np.linalg.norm(left - selected.fit.center) > safe
    assert np.linalg.norm(right - selected.fit.center) > safe
    assert left[1] > selected.fit.center[1]
    assert right[1] < selected.fit.center[1]


def test_target_on_90_degree_path_follows_local_normal_not_global_y():
    path = np.array([[0.1, 0.0], [0.5, 0.0], [1.0, 0.0],
                     [1.0, 0.4], [1.0, 0.8], [1.0, 1.2]])
    selected = _selected(center=(0.9, 0.8), path=path)
    target, _, _ = avoidance_target(selected, AVOID_LEFT, config())
    assert target[0] < selected.fit.center[0]
    assert abs(target[1] - selected.fit.center[1]) < 0.05


def test_shadow_path_reaches_target_returns_and_improves_clearance():
    path = straight_path()
    selected = _selected(path=path)
    target, safe, _ = avoidance_target(selected, AVOID_RIGHT, config())
    shadow, weights = build_shadow_path(path, selected, target, config())
    closest_target = np.min(np.linalg.norm(shadow - target, axis=1))
    original_distance = selected.path_center_distance
    from physicar_track_perception_v3.avoidance import project_to_polyline
    shadow_distance = project_to_polyline(
        selected.fit.center[None, :], shadow)['distance'][0]
    assert closest_target < 1e-9
    assert shadow_distance > original_distance
    assert weights[0] == 0.0
    assert weights[-1] == 0.0
    assert np.allclose(shadow[0], path[0])
    assert np.allclose(shadow[-1], path[-1])
    assert shadow_distance >= safe - 1e-6


def test_shadow_path_creates_clearer_pure_pursuit_steering():
    path = straight_path()
    selected = _selected(center=(0.85, 0.1), path=path)
    target, _, _ = avoidance_target(selected, AVOID_RIGHT, config())
    shadow, _ = build_shadow_path(path, selected, target, config())
    original = pure_pursuit_shadow(path, config())
    avoidance = pure_pursuit_shadow(shadow, config())
    assert np.isclose(original['steering_saturated'], 0.0)
    assert avoidance['steering_saturated'] < -0.05


def test_circle_target_does_not_create_kink_at_tiny_dash_connector():
    from physicar_track_perception_v3.avoidance import max_adjacent_heading_change
    path = np.array([
        [0.10, 0.00], [0.30, 0.00], [0.50, 0.00],
        [0.53, 0.01], [0.532, 0.008], [0.55, 0.02],
        [0.70, 0.05], [0.90, 0.10], [1.20, 0.15],
    ])
    selected = _selected(center=(0.80, -0.08), path=path)
    target, _, _ = avoidance_target(selected, AVOID_LEFT, config())
    shadow, _ = build_shadow_path(path, selected, target, config())
    assert np.min(np.linalg.norm(shadow - target, axis=1)) < 0.01
    assert (max_adjacent_heading_change(shadow)
            < max_adjacent_heading_change(path))


def test_engine_votes_locks_holds_direction_and_terminates_on_rear_pass():
    engine = CircleAvoidanceEngine(config())
    path = straight_path()
    for measurement, center in enumerate(((1.35, 0.1), (1.20, 0.1),
                                          (1.05, 0.1))):
        result = engine.process(*scan_circle(center, measurement, path)[:3],
                                measurement_key=measurement)
        assert result.state == VOTING
    points, beams, path, measurement = scan_circle((0.95, 0.1), 3, path)
    result = engine.process(points, beams, path, measurement_key=measurement)
    assert result.direction_locked
    assert result.locked_avoidance_side == AVOID_RIGHT
    assert result.active

    # Move the current circle to the other instantaneous side. The latched
    # avoidance direction must remain RIGHT.
    points, beams, path, measurement = scan_circle((0.70, -0.1), 4, path)
    result = engine.process(points, beams, path, measurement_key=measurement)
    assert result.instantaneous_side == RIGHT
    assert result.locked_avoidance_side == AVOID_RIGHT

    for measurement, center in enumerate(((0.35, -0.1), (-0.05, -0.1)), 5):
        points, beams, path, _ = scan_circle(center, measurement, path)
        result = engine.process(points, beams, path,
                                measurement_key=measurement)
    assert result.reason == 'AVOIDANCE_COMPLETE_REAR_PASS'
    assert result.state == NORMAL
    assert not result.direction_locked
    assert (result.left_votes, result.right_votes) == (0, 0)


def test_duplicate_scan_measurement_does_not_add_duplicate_vote():
    engine = CircleAvoidanceEngine(config())
    points, beams, path, _ = scan_circle((1.2, 0.1), 10)
    first = engine.process(points, beams, path, measurement_key=10)
    duplicate = engine.process(points, beams, path, measurement_key=10)
    assert first.left_votes == 1
    assert duplicate.left_votes == 1
