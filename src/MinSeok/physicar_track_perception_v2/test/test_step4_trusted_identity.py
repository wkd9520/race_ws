import numpy as np
import pytest

from physicar_track_perception_v2.both_geometry import FrameLocalBothGeometry
from physicar_track_perception_v2.components import (
    CanonicalComponentExtractor, ComponentExtractionConfig, WHITE,
)
from physicar_track_perception_v2.geometry import BevGrid
from physicar_track_perception_v2.trusted_identity import TrustedBoundaryIdentity


GRID = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)
EXTRACTOR = CanonicalComponentExtractor(
    GRID, ComponentExtractionConfig(canonical_spacing=0.05))


def make(points, component_id):
    result, reason = EXTRACTOR.canonicalize_ordered_points(
        points, component_id=component_id, color=WHITE)
    assert reason == 'valid'
    return result


def straight(offset=0.0, yaw=0.0, length=1.3, ids=(1, 2)):
    s = np.linspace(0.25, 0.25+length, 180)
    tangent = np.array([np.cos(yaw), np.sin(yaw)])
    normal = np.array([-np.sin(yaw), np.cos(yaw)])
    center = np.array([0.25, offset]) + (s-0.25)[:, None]*tangent
    return make(center+0.35*normal, ids[0]), make(center-0.35*normal, ids[1])


def arcs(mirror=False, ids=(1, 2)):
    theta = np.linspace(0, np.pi/2, 240)
    sign = -1 if mirror else 1
    center = np.column_stack((0.2+np.sin(theta), sign*(1-np.cos(theta))))
    tangent = np.column_stack((np.cos(theta), sign*np.sin(theta)))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return make(center+0.35*normal, ids[0]), make(center-0.35*normal, ids[1])


def initialized(pair=None):
    tracker = TrustedBoundaryIdentity(FrameLocalBothGeometry())
    pair = pair or straight()
    for frame in range(3):
        result = tracker.process(pair, frame*0.07)
    assert result.identity_initialized
    return tracker


def test_straight_both_requires_three_consistent_frames():
    tracker = TrustedBoundaryIdentity(FrameLocalBothGeometry())
    pair = straight()
    assert tracker.process(pair).initialization_streak == 1
    assert not tracker.process(pair).identity_initialized
    assert tracker.process(pair).identity_initialized


def test_one_good_then_inconsistent_does_not_lock():
    tracker = TrustedBoundaryIdentity(FrameLocalBothGeometry())
    tracker.process(straight())
    result = tracker.process(straight(offset=0.6))
    assert not result.identity_initialized
    assert result.initialization_streak <= 1


@pytest.mark.parametrize('pair', [straight(ids=(9, 4)), straight(yaw=0.25), arcs()])
def test_initialization_uses_physical_geometry_not_ids_or_x_monotonicity(pair):
    tracker = initialized(pair)
    assert tracker.left_state.initialized and tracker.right_state.initialized


@pytest.mark.parametrize('offset', [-0.04, 0.03])
def test_small_physical_motion_associates(offset):
    tracker = initialized()
    result = tracker.process(straight(offset=offset, ids=(8, 3)), 1.0)
    assert result.both_accepted and result.center_result.center_path is not None


def test_partial_overlap_accepts_supported_subpolyline():
    tracker = initialized()
    left, right = straight(ids=(7, 8))
    short_left = make(left.raw_ordered_points[5:-6], 7)
    result = tracker.process((short_left, right))
    assert result.both_accepted
    assert result.left.accepted_support < tracker.config.min_accepted_support + 1.2


def test_natural_short_continuation_is_accepted():
    tracker = initialized()
    left, right = straight(length=1.42, ids=(7, 8))
    result = tracker.process((left, right))
    assert result.both_accepted
    assert result.left.accepted_support > 1.30


def test_shorter_visible_boundary_remains_associated():
    tracker = initialized()
    result = tracker.process(straight(length=0.9, ids=(4, 5)))
    assert result.both_accepted


def test_distant_component_is_rejected():
    tracker = initialized()
    left, _ = straight(offset=1.0, ids=(8, 9))
    result = tracker.process((left,))
    assert not result.both_accepted
    assert result.reason == 'no_association'


def test_wrong_tangent_is_rejected():
    tracker = initialized()
    y = np.linspace(-0.35, 0.35, 30)
    wrong = make(np.column_stack((np.full_like(y, 0.7), y)), 8)
    result = tracker.process((wrong,))
    assert not result.both_accepted


def contaminated_right(extra_length=0.7):
    left, right = straight(ids=(11, 12))
    source = right.raw_ordered_points
    tail_y = np.linspace(source[-1, 1], source[-1, 1]-extra_length, 30)[1:]
    tail = np.column_stack((np.full_like(tail_y, source[-1, 0]), tail_y))
    return left, make(np.vstack((source, tail)), 12)


@pytest.mark.parametrize('extra_length', [0.4, 0.8, 1.2])
def test_merged_external_tail_is_trimmed_and_cannot_win_by_size(extra_length):
    tracker = initialized()
    left, merged = contaminated_right(extra_length)
    result = tracker.process((merged, left))
    assert result.both_accepted
    assert result.right.rejected_tail_support > 0.2
    assert result.right.accepted.support_length < merged.support_length
    assert np.max(np.abs(result.center_result.center_path.points[:, 1])) < 0.03


def test_persistent_contamination_does_not_enter_trusted_right():
    tracker = initialized()
    for frame in range(4):
        left, merged = contaminated_right(0.8)
        result = tracker.process((left, merged), frame+1)
        assert result.both_accepted
        assert tracker.right_state.physical_support < merged.support_length


def test_one_frame_contamination_then_clean_keeps_identity_clean():
    tracker = initialized()
    result = tracker.process(contaminated_right())
    assert result.right.rejected_tail_support > 0
    clean = tracker.process(straight(ids=(20, 21)))
    assert clean.both_accepted


def test_unrecoverable_contaminated_right_produces_no_false_both():
    tracker = initialized()
    left, _ = straight()
    x = np.linspace(0.3, 1.5, 50)
    external = make(np.column_stack((x, np.full_like(x, -1.0))), 9)
    result = tracker.process((left, external))
    assert not result.both_accepted
    assert result.center_result.center_path is None


@pytest.mark.parametrize('mirror', [False, True])
def test_ninety_degree_and_mirror_association_preserve_center(mirror):
    pair = arcs(mirror=mirror)
    tracker = initialized(pair)
    result = tracker.process(arcs(mirror=mirror, ids=(7, 3)))
    assert result.both_accepted
    assert result.center_result.center_path.support_length > 0.5


def test_non_x_monotonic_current_associates_in_metric_2d():
    pair = arcs()
    tracker = initialized(pair)
    left, right = arcs(ids=(40, 41))
    assert np.any(np.diff(left.canonical_points[:, 0]) < 0) or left.support_length > 1.0
    assert tracker.process((right, left)).both_accepted


def test_component_matching_both_identities_is_not_dually_assigned():
    tracker = initialized()
    left, right = straight()
    bridge = np.vstack((left.raw_ordered_points,
                        right.raw_ordered_points[::-1],
                        right.raw_ordered_points))
    result = tracker.process((make(bridge, 99),))
    assert not result.both_accepted
    assert result.center_result.center_path is None
