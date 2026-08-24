import numpy as np
import pytest

from physicar_track_perception_v2.both_geometry import FrameLocalBothGeometry
from physicar_track_perception_v2.components import (
    CanonicalComponentExtractor, ComponentExtractionConfig, WHITE,
)
from physicar_track_perception_v2.geometry import BevGrid
from physicar_track_perception_v2.single_reconstruction import (
    LEFT_ONLY_OBSERVED, RIGHT_ONLY_OBSERVED, TrustedSingleReconstruction,
    ValidatedWidth,
)
from physicar_track_perception_v2.trusted_identity import (
    TrustedBoundaryIdentity, TrustedBoundaryState,
)


GRID = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)
EXTRACTOR = CanonicalComponentExtractor(
    GRID, ComponentExtractionConfig(canonical_spacing=0.05))


def make(points, component_id):
    result, reason = EXTRACTOR.canonicalize_ordered_points(
        points, component_id=component_id, color=WHITE)
    assert reason == 'valid'
    return result


def straight(length=1.3, ids=(1, 2)):
    x = np.linspace(0.25, 0.25+length, 180)
    return (make(np.column_stack((x, np.full_like(x, 0.35))), ids[0]),
            make(np.column_stack((x, np.full_like(x, -0.35))), ids[1]))


def arcs(mirror=False, ids=(1, 2)):
    theta = np.linspace(0.0, np.pi/2, 240)
    sign = -1.0 if mirror else 1.0
    center = np.column_stack((0.2+np.sin(theta), sign*(1-np.cos(theta))))
    tangent = np.column_stack((np.cos(theta), sign*np.sin(theta)))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    return make(center+0.35*normal, ids[0]), make(center-0.35*normal, ids[1])


def initialized(pair):
    tracker = TrustedBoundaryIdentity(FrameLocalBothGeometry())
    single = TrustedSingleReconstruction(ValidatedWidth())
    output = None
    # Three frames establish identity; three subsequent actual BOTH frames
    # establish the separately protected width state.
    for frame in range(6):
        identity = tracker.process(pair, frame*.07)
        output = single.process(identity, tracker, frame*.07)
    assert tracker.initialized and tracker.trusted_center is not None
    assert single.width.state.initialized
    return tracker, single, output


def force_cross_identity(tracker, survivor, survivor_side):
    """Model a stale opposite identity overlapping the visible survivor."""
    state = TrustedBoundaryState(
        'RIGHT' if survivor_side == 'LEFT' else 'LEFT', True, survivor,
        0.5, survivor.support_length)
    if survivor_side == 'LEFT':
        tracker.right_state = state
    else:
        tracker.left_state = state


@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_clear_corridor_side_resolves_single_without_width_update(side):
    pair = straight()
    tracker, single, _ = initialized(pair)
    survivor = pair[0] if side == 'LEFT' else pair[1]
    missing_before = (tracker.right_state.geometry if side == 'LEFT'
                      else tracker.left_state.geometry)
    force_cross_identity(tracker, survivor, side)
    identity = tracker.process((survivor,), 1.0)
    assert identity.conflict_resolution.result == 'RESOLVED_'+side
    assert identity.reason == 'cross_identity_resolved_'+side.lower()
    # Resolution itself does not alter the opposite actual identity; restore
    # the realistic last actual opposite geometry after the synthetic stale
    # overlap used to force the conflict.
    if side == 'LEFT':
        tracker.right_state = TrustedBoundaryState(
            'RIGHT', True, missing_before, .5, missing_before.support_length)
    else:
        tracker.left_state = TrustedBoundaryState(
            'LEFT', True, missing_before, .5, missing_before.support_length)
    output = single.process(identity, tracker, 1.0)
    assert output.observation_mode == (
        LEFT_ONLY_OBSERVED if side == 'LEFT' else RIGHT_ONLY_OBSERVED)
    assert output.center is not None
    assert not output.width_update_allowed
    # A reconstructed opposite never overwrites its physical identity state.
    opposite_after = (tracker.right_state.geometry if side == 'LEFT'
                      else tracker.left_state.geometry)
    assert opposite_after is missing_before


@pytest.mark.parametrize('pair', [arcs(), arcs(mirror=True)])
@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_ninety_degree_short_visible_segment_resolves_by_local_corridor(pair, side):
    tracker, single, _ = initialized(pair)
    source = pair[0] if side == 'LEFT' else pair[1]
    opposite = tracker.right_state.geometry if side == 'LEFT' else tracker.left_state.geometry
    short = make(source.raw_ordered_points[35:165], 31)
    force_cross_identity(tracker, short, side)
    identity = tracker.process((short,), 1.0)
    assert identity.conflict_resolution.result == 'RESOLVED_'+side
    if side == 'LEFT':
        tracker.right_state = TrustedBoundaryState(
            'RIGHT', True, opposite, .5, opposite.support_length)
    else:
        tracker.left_state = TrustedBoundaryState(
            'LEFT', True, opposite, .5, opposite.support_length)
    output = single.process(identity, tracker, 1.0)
    assert output.center is not None


def test_component_with_supported_left_and_right_intervals_stays_ambiguous():
    pair = straight()
    tracker, _, _ = initialized(pair)
    bridge = make(np.vstack((pair[0].raw_ordered_points,
                             pair[1].raw_ordered_points[::-1],
                             pair[1].raw_ordered_points)), 99)
    left = tracker._associate('LEFT', tracker.left_state.geometry, pair[0])
    right = tracker._associate('RIGHT', tracker.right_state.geometry, pair[1])
    resolution = tracker._resolve_conflict(bridge, left, right)
    assert resolution.result == 'AMBIGUOUS_CONFLICT'
    assert resolution.reason == 'cross_identity_ambiguous'


def test_center_crossing_short_component_is_not_forced_to_a_side():
    pair = straight()
    tracker, _, _ = initialized(pair)
    x = np.linspace(0.45, 0.75, 40)
    crossing = make(np.column_stack((x, np.linspace(0.08, -0.08, 40))), 50)
    left = tracker._associate('LEFT', crossing, crossing)
    right = tracker._associate('RIGHT', crossing, crossing)
    resolution = tracker._resolve_conflict(crossing, left, right)
    assert resolution.result == 'UNSUPPORTED'
    assert not resolution.left_evidence.valid
    assert not resolution.right_evidence.valid


def test_missing_trusted_center_preserves_safe_conflict_reject():
    pair = straight()
    tracker, _, _ = initialized(pair)
    survivor = pair[0]
    force_cross_identity(tracker, survivor, 'LEFT')
    tracker.trusted_center = None
    result = tracker.process((survivor,), 1.0)
    assert result.conflict_resolution.result == 'UNSUPPORTED'
    assert result.reason == 'cross_identity_center_reference_unavailable'
    assert result.left is None and result.right is None


def test_resolved_single_does_not_accept_external_tail():
    pair = straight()
    tracker, _, _ = initialized(pair)
    left = pair[0]
    tail_x = np.full(30, left.raw_ordered_points[-1, 0])
    tail_y = np.linspace(left.raw_ordered_points[-1, 1], 0.72, 30)
    merged = make(np.vstack((left.raw_ordered_points,
                             np.column_stack((tail_x, tail_y))[1:])), 77)
    force_cross_identity(tracker, merged, 'LEFT')
    result = tracker.process((merged,), 1.0)
    assert result.conflict_resolution.result == 'RESOLVED_LEFT'
    assert result.left.accepted.support_length < merged.support_length
    assert result.left.rejected_tail_support > 0.0


def test_clean_both_after_ambiguous_conflict_resumes_without_identity_drift():
    pair = straight()
    tracker, single, _ = initialized(pair)
    bridge = make(np.vstack((pair[0].raw_ordered_points,
                             pair[1].raw_ordered_points[::-1],
                             pair[1].raw_ordered_points)), 99)
    left = tracker._associate('LEFT', tracker.left_state.geometry, pair[0])
    right = tracker._associate('RIGHT', tracker.right_state.geometry, pair[1])
    conflict = tracker._resolve_conflict(bridge, left, right)
    assert conflict.result == 'AMBIGUOUS_CONFLICT'
    recovered = tracker.process(straight(ids=(70, 71)), 1.1)
    output = single.process(recovered, tracker, 1.1)
    assert recovered.both_accepted
    assert output.center is not None
