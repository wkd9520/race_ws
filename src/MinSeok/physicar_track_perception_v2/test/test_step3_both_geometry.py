import numpy as np
import pytest

from physicar_track_perception_v2.both_geometry import (
    BothGeometryConfig,
    FrameLocalBothGeometry,
)
from physicar_track_perception_v2.components import (
    CanonicalComponentExtractor,
    ComponentExtractionConfig,
    ORANGE,
    WHITE,
)
from physicar_track_perception_v2.geometry import BevGrid


GRID = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)
EXTRACTOR = CanonicalComponentExtractor(
    GRID, ComponentExtractionConfig(canonical_spacing=0.05)
)


def candidate(points, component_id, color=WHITE):
    result, reason = EXTRACTOR.canonicalize_ordered_points(
        points, component_id=component_id, color=color
    )
    assert reason == 'valid'
    return result


def straight(length=1.2, width=0.70, yaw=0.0, lateral=0.0,
             samples=121, colors=(WHITE, WHITE), reverse=False):
    s = np.linspace(0.25, 0.25 + length, samples)
    center = np.column_stack((s, np.full_like(s, lateral)))
    tangent = np.array([np.cos(yaw), np.sin(yaw)])
    normal = np.array([-np.sin(yaw), np.cos(yaw)])
    center = np.array([0.25, lateral]) + (s - 0.25)[:, None] * tangent
    left = center + 0.5 * width * normal
    right = center - 0.5 * width * normal
    if reverse:
        left, right = left[::-1], right[::-1]
    return candidate(left, 1, colors[0]), candidate(right, 2, colors[1]), center


def arc(radius=1.0, width=0.70, angle=np.pi/2, samples=181,
        mirror=False, colors=(WHITE, WHITE), ids=(1, 2)):
    theta = np.linspace(0.0, angle, samples)
    sign = -1.0 if mirror else 1.0
    center = np.column_stack((0.20 + radius*np.sin(theta),
                              sign*radius*(1.0-np.cos(theta))))
    tangent = np.column_stack((np.cos(theta), sign*np.sin(theta)))
    left_normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left = center + 0.5*width*left_normal
    right = center - 0.5*width*left_normal
    return (candidate(left, ids[0], colors[0]),
            candidate(right, ids[1], colors[1]), center)


def solve(*candidates, config=None):
    return FrameLocalBothGeometry(config or BothGeometryConfig()).process(candidates)


def radial_error(points, radius, mirror=False):
    circle_center = np.array([0.20, -radius if mirror else radius])
    return np.abs(np.linalg.norm(points-circle_center, axis=1)-radius)


def test_straight_parallel_boundaries_center_exactly_between():
    left, right, _ = straight()
    result = solve(left, right)
    assert result.reason == 'valid'
    assert np.max(np.abs(result.center_path.points[:, 1])) < 1e-12
    assert result.center_path.width_median == pytest.approx(0.70, abs=1e-12)


def test_lateral_offset_keeps_physical_corridor_midpoint():
    left, right, _ = straight(lateral=0.16)
    center = solve(left, right).center_path
    assert np.max(np.abs(center.points[:, 1]-0.16)) < 1e-12


def test_vehicle_yaw_offset_keeps_side_assignment_and_center():
    left, right, _ = straight(yaw=0.28)
    result = solve(right, left)
    assert result.center_path.left_component_id == left.component_id
    expected_y = np.tan(0.28) * (result.center_path.points[:, 0]-0.25)
    assert np.max(np.abs(result.center_path.points[:, 1]-expected_y)) < 2e-3


def test_gradual_concentric_curve_follows_middle_arc():
    left, right, _ = arc(radius=1.5, angle=0.55)
    center = solve(left, right).center_path
    assert np.max(radial_error(center.points, 1.5)) < 0.004


def test_ninety_degree_concentric_curve_is_continuous_without_x_assumption():
    left, right, _ = arc(radius=1.0)
    center = solve(left, right).center_path
    assert center is not None and center.support_length > 1.3
    assert np.max(radial_error(center.points, 1.0)) < 0.005
    assert np.all(np.diff(center.s) > 0)


def test_left_turn_inner_outer_pair_has_middle_radius():
    left, right, _ = arc(radius=1.0)
    result = solve(left, right)
    assert result.center_path.left_component_id == left.component_id
    assert np.median(radial_error(result.center_path.points, 1.0)) < 0.002


def test_right_turn_mirror_has_middle_radius_and_mirrored_side():
    left, right, _ = arc(radius=1.0, mirror=True)
    result = solve(left, right)
    assert result.center_path.left_component_id == left.component_id
    assert np.max(radial_error(result.center_path.points, 1.0, mirror=True)) < 0.005


def test_non_x_monotonic_boundaries_preserve_correspondence_order():
    theta = np.linspace(-0.7, 3.5, 500)
    center = np.column_stack((0.9 + 0.45*np.sin(theta), 0.45*np.cos(theta)))
    tangent = np.column_stack((np.cos(theta), -np.sin(theta)))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    left = candidate(center + 0.31*normal, 1)
    right = candidate(center - 0.31*normal, 2)
    result = solve(left, right)
    assert result.reason == 'valid'
    corr = result.selected_pair.correspondence
    assert np.all(np.diff(corr.first_indices) > 0)
    assert np.all(np.diff(corr.second_indices) > 0)


def test_longer_boundary_uses_only_common_overlap():
    left, right, _ = straight(length=1.4)
    short_right = candidate(right.raw_ordered_points[25:-20], 2)
    center = solve(left, short_right).center_path
    assert center.support_length < left.support_length
    assert center.pair_overlap_support <= short_right.support_length + 1e-9


def test_partial_overlap_does_not_extrapolate_center_tails():
    left, right, _ = straight(length=1.2)
    partial = candidate(right.raw_ordered_points[40:90], 2)
    center = solve(left, partial).center_path
    assert center.points[0, 0] >= partial.near_endpoint[0]-0.03
    assert center.points[-1, 0] <= partial.far_endpoint[0]+0.03


def test_dense_and_sparse_boundary_sampling_give_similar_center():
    dense = arc(samples=361)[:2]
    sparse = arc(samples=45, ids=(3, 4))[:2]
    center_dense = solve(*dense).center_path
    center_sparse = solve(*sparse).center_path
    assert center_dense.support_length == pytest.approx(center_sparse.support_length, abs=0.03)
    assert np.median(radial_error(center_sparse.points, 1.0)) < 0.005


def test_candidate_input_order_and_raw_order_reversal_are_invariant():
    left, right, _ = straight(reverse=True)
    first, second = solve(left, right), solve(right, left)
    assert first.center_path.left_component_id == second.center_path.left_component_id == 1
    assert np.allclose(first.center_path.points, second.center_path.points)


def test_multiple_candidates_selects_long_geometry_consistent_pair():
    left, right, _ = straight(length=1.2)
    external_points = np.column_stack((np.linspace(0.3, 0.58, 20),
                                       np.full(20, 0.70)))
    external = candidate(external_points, 9, ORANGE)
    result = solve(external, right, left)
    assert result.reason == 'valid'
    assert {result.center_path.left_component_id,
            result.center_path.right_component_id} == {1, 2}


def test_equal_geometry_pairs_are_reported_ambiguous_not_forced():
    x = np.linspace(0.25, 1.3, 120)
    candidates = [candidate(np.column_stack((x, np.full_like(x, y))), index)
                  for index, y in enumerate((-0.70, 0.0, 0.70), 1)]
    result = solve(*candidates)
    assert result.reason == 'ambiguous_pair'
    assert result.center_path is None


def test_impossible_width_is_rejected():
    left, right, _ = straight(width=0.30)
    result = solve(left, right)
    assert result.center_path is None
    assert result.pair_evaluations[0].reason == 'width_gate'


def test_crossing_correspondence_candidate_is_invalid():
    x = np.linspace(0.25, 1.45, 100)
    first = candidate(np.column_stack((x, np.full_like(x, 0.35))), 1)
    second = candidate(np.column_stack((x, np.linspace(-0.35, 0.50, len(x)))), 2)
    result = solve(first, second)
    assert result.center_path is None


def test_color_swap_changes_only_provenance_not_geometry():
    white_orange = straight(colors=(WHITE, ORANGE))[:2]
    orange_white = straight(colors=(ORANGE, WHITE))[:2]
    first, second = solve(*white_orange).center_path, solve(*orange_white).center_path
    assert np.allclose(first.points, second.points)
    assert first.width_median == pytest.approx(second.width_median)
    assert (first.left_color, first.right_color) != (second.left_color, second.right_color)


def test_short_usable_pair_has_explicit_support_without_temporal_state():
    left, right, _ = straight(length=0.22, samples=24)
    result = solve(left, right)
    assert result.reason == 'valid'
    assert result.center_path.pair_overlap_support >= 0.15
    assert not hasattr(result, 'previous_frame')
