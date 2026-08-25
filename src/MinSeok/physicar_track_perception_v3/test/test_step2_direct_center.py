import numpy as np
from physicar_track_perception_v3.geometry import OrderedPolyline
from physicar_track_perception_v3.roles import Component
from physicar_track_perception_v3.path_selector import (select, select_orange,
    select_unknown_white, DIRECT_CENTER_OBSERVED, INVALID)
from physicar_track_perception_v3.roles import CENTER

def comp(color, points, ident=1):
    path = OrderedPolyline.from_points(points)
    return Component(ident, color, path, path.support)

def test_direct_center_is_primary_over_boundaries():
    x = np.linspace(.1, 1.4, 40)
    center = comp('ORANGE', np.c_[x, .02*np.sin(x)], 3)
    left = comp('WHITE', np.c_[x, np.full_like(x, .34)], 1)
    right = comp('WHITE', np.c_[x, np.full_like(x, -.34)], 2)
    result = select([left, right, center])
    assert result.valid and result.source == DIRECT_CENTER_OBSERVED
    assert np.allclose(result.path.points, center.polyline.points)

def test_non_x_monotonic_center_is_preserved_as_ordered_geometry():
    points = np.array([[.1, 0.0], [.3, .04], [.3, .06], [.1, .02], [-.1, .03]])
    result = select([comp('ORANGE', points)])
    assert result.valid
    assert np.allclose(result.path.points, points)

def test_center_absent_is_invalid_in_step2():
    x = np.linspace(.1, 1.0, 20)
    result = select([comp('WHITE', np.c_[x, np.full_like(x, .34)])])
    assert not result.valid and result.source == INVALID

def test_near_seed_stitches_dashed_center_instead_of_longest_only():
    near = comp('ORANGE', np.array([[.10,0.], [.20,0.], [.25,0.], [.30,0.]]), 1)
    far = comp('ORANGE', np.array([[.27,0.], [.50,0.], [.80,0.], [1.1,0.]]), 2)
    from physicar_track_perception_v3.roles import RoleConfig
    result = select([far, near], RoleConfig(minimum_support=.19), gap_limit=.10)
    assert result.valid and result.stitched_component_ids == (1, 2)
    assert result.path.points[0, 0] == .10
    assert result.bridged_gap_count == 1

def test_large_gap_stops_near_to_far_dash_chain():
    near = comp('ORANGE', np.array([[.10,0.], [.20,0.], [.30,0.]]), 1)
    far = comp('ORANGE', np.array([[.50,.30], [.70,.30], [.90,.30]]), 2)
    from physicar_track_perception_v3.roles import RoleConfig
    result = select_orange([near, far], RoleConfig(minimum_support=.19), gap_limit=.15)
    assert result.stitched_component_ids == (1,)

def test_reversed_fragment_is_used_only_once():
    near = comp('ORANGE', np.array([[.10,0.], [.20,0.], [.30,0.]]), 1)
    # Canonical graph order can oppose the selected chain direction.
    reversed_far = comp('ORANGE', np.array([[.80,0.], [.55,0.], [.35,0.]]), 2)
    result = select_orange([near, reversed_far], gap_limit=.30)
    assert result.stitched_component_ids == (1, 2)
    assert len(set(result.stitched_component_ids)) == 2

def test_orange_direct_path_does_not_depend_on_generic_role_classifier():
    orange = comp('ORANGE', np.array([[.1, .3], [.2, .3], [.3, .3], [.4, .3]]), 7)
    # It is deliberately lateral and would fail the old center role heuristic.
    result = select_orange([orange])
    assert result.valid and result.source == DIRECT_CENTER_OBSERVED

def test_unknown_white_uses_nearest_component_even_when_another_is_longer():
    nearest_short = comp('WHITE', np.array([
        [.10, .4], [.20, .4]]), 1)
    farther_long = comp('WHITE', np.array([
        [.12, -.4], [.40, -.4], [.70, -.4]]), 2)
    result = select_unknown_white(
        [farther_long, nearest_short], .70)
    assert result.valid
    assert nearest_short.support < farther_long.support
    assert result.stitched_component_ids == (1,)


def test_unknown_white_noisy_endpoint_cannot_flip_complete_offset_outward():
    # The first short diagonal reproduces the observed 45-degree endpoint
    # tangent.  The stable interior of this LEFT boundary runs along +X.
    boundary = comp('WHITE', np.array([
        [.325, .295], [.360, .330], [.450, .365], [.550, .365],
        [.800, .365], [1.100, .370], [1.400, .375]]), 1)
    result = select_unknown_white([boundary], .70)
    assert result.valid
    assert result.reason == 'unknown_white_vehicle_median_offset'
    # W/2 must go toward the track center, not to y=+0.7 outside.
    assert np.max(np.abs(result.path.points[:, 1])) < .15


def test_unknown_white_orange_reference_has_first_priority():
    x = np.linspace(.15, 1.50, 20)
    boundary = comp('WHITE', np.c_[x, np.full_like(x, .35)], 1)
    orange = OrderedPolyline.from_points(np.c_[x, np.zeros_like(x)])
    result = select_unknown_white(
        [boundary], .70, reference_path=orange)
    assert result.valid
    assert result.reason == 'unknown_white_orange_reference_offset'
    assert np.allclose(result.path.points[:, 1], 0.0, atol=1e-9)


def test_unknown_white_exact_vehicle_side_tie_forces_one_candidate():
    # A radial boundary has zero vehicle projection on its normal.  Priority
    # 3 must still return one deterministic current-frame candidate.
    boundary = comp('WHITE', np.array([
        [.20, 0.0], [.40, 0.0], [.60, 0.0], [.80, 0.0]]), 1)
    result = select_unknown_white([boundary], .70)
    assert result.valid
    assert result.reason == 'unknown_white_vehicle_forced_offset'
    assert np.allclose(np.abs(result.path.points[:, 1]), .35)


def test_unknown_white_reference_offset_follows_90_degree_local_normal():
    angles = np.linspace(0.0, np.pi / 2.0, 30)
    center_points = np.c_[np.cos(angles), np.sin(angles)]
    boundary_points = 1.35 * center_points
    boundary = comp('WHITE', boundary_points, 1)
    orange = OrderedPolyline.from_points(center_points)
    result = select_unknown_white(
        [boundary], .70, reference_path=orange)
    radii = np.linalg.norm(result.path.points, axis=1)
    assert result.reason == 'unknown_white_orange_reference_offset'
    assert np.allclose(radii, 1.0, atol=.02)
    assert np.any(np.diff(result.path.points[:, 0]) < 0.0)
    assert np.any(np.diff(result.path.points[:, 1]) > 0.0)
