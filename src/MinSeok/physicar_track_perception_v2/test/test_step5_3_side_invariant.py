import numpy as np
import pytest

from physicar_track_perception_v2.both_geometry import FrameLocalBothGeometry
from physicar_track_perception_v2.components import (
    CanonicalComponentExtractor, ComponentExtractionConfig, WHITE,
)
from physicar_track_perception_v2.geometry import BevGrid
from physicar_track_perception_v2.single_reconstruction import (
    TrustedSingleReconstruction, ValidatedWidth,
)
from physicar_track_perception_v2.trusted_identity import (
    TrustedBoundaryIdentity, TrustedBoundaryState,
)


GRID = BevGrid(.1, 2., -.75, .75, .01)
EXTRACTOR = CanonicalComponentExtractor(
    GRID, ComponentExtractionConfig(canonical_spacing=.05))


def make(points, component_id):
    result, reason = EXTRACTOR.canonicalize_ordered_points(
        points, component_id=component_id, color=WHITE)
    assert reason == 'valid'
    return result


def line(y, component_id, x0=.25, x1=1.55):
    x = np.linspace(x0, x1, 180)
    return make(np.column_stack((x, np.full_like(x, y))), component_id)


def pair(ids=(1, 2), x0=.25, x1=1.55):
    return line(.35, ids[0], x0, x1), line(-.35, ids[1], x0, x1)


def initialized():
    tracker = TrustedBoundaryIdentity(FrameLocalBothGeometry())
    single = TrustedSingleReconstruction(ValidatedWidth())
    observations = pair()
    for frame in range(6):
        result = tracker.process(observations, frame*.07)
        single.process(result, tracker, frame*.07)
    return tracker, single


@pytest.mark.parametrize('side,index,state', [
    ('LEFT', 0, 'SIDE_CONSISTENT'),
    ('RIGHT', 1, 'SIDE_CONSISTENT'),
])
def test_physical_boundary_passes_matching_side_invariant(side, index, state):
    tracker, _ = initialized()
    result = tracker.process((pair(ids=(10, 11))[index],), 1.0)
    association = result.left if side == 'LEFT' else result.right
    assert association.valid
    assert association.side_state == state
    assert association.reference_update_allowed


@pytest.mark.parametrize('hypothesis,physical_y', [('LEFT', -.35), ('RIGHT', .35)])
def test_raw_wrong_side_association_is_invalidated(hypothesis, physical_y):
    tracker, _ = initialized()
    current = line(physical_y, 20)
    # Reproduce the 67642.675 precondition: a poisoned short-term reference is
    # close enough that raw distance/tangent association alone succeeds.
    state = tracker.left_state if hypothesis == 'LEFT' else tracker.right_state
    poisoned = TrustedBoundaryState(
        hypothesis, True, state.geometry, state.timestamp,
        state.physical_support, current, current)
    if hypothesis == 'LEFT':
        tracker.left_state = poisoned
    else:
        tracker.right_state = poisoned
    raw = tracker._associate(hypothesis, current, current)
    assert raw.valid
    validated = tracker._validate_identity_side(raw)
    assert not validated.valid
    assert validated.side_state == 'SIDE_OPPOSITE'
    assert validated.reason == 'association_rejected_wrong_side'
    assert not validated.reference_update_allowed


def test_67642_pattern_cannot_poison_left_reference():
    tracker, single = initialized()
    right_fragment = line(-.25, 30, 1.05, 1.42)
    original_long_term = tracker.left_state.geometry
    tracker.left_state = TrustedBoundaryState(
        'LEFT', True, original_long_term, .5,
        original_long_term.support_length, right_fragment, right_fragment)
    result = tracker.process((right_fragment,), 1.0)
    assert result.left is None
    assert result.right is not None
    assert result.reason == 'cross_identity_resolved_right'
    single.process(result, tracker, 1.0)
    assert tracker.left_state.association_reference is right_fragment
    # The rejected frame cannot advance the poisoned precondition further.
    assert tracker.left_state.geometry is original_long_term


def test_center_crossing_geometry_is_rejected():
    tracker, _ = initialized()
    x = np.linspace(.3, 1.2, 120)
    crossing = make(np.column_stack((x, np.linspace(.3, -.3, len(x)))), 40)
    raw = tracker._associate('LEFT', crossing, crossing)
    checked = tracker._validate_identity_side(raw)
    assert not checked.valid
    assert checked.center_crossing or checked.side_state != 'SIDE_CONSISTENT'


def test_strong_both_updates_long_term_corridor():
    tracker, _ = initialized()
    old = tracker.left_state.geometry
    result = tracker.process(pair(ids=(50, 51)), 1.0)
    assert result.both_accepted and result.trusted_update_allowed
    assert tracker.left_state.geometry is not old


def test_short_both_center_may_be_used_without_long_term_update():
    tracker, _ = initialized()
    old_left, old_right = tracker.left_state.geometry, tracker.right_state.geometry
    short = pair(ids=(60, 61), x0=.5, x1=.95)
    result = tracker.process(short, 1.0)
    assert result.both_accepted
    assert result.center_result.center_path is not None
    assert not result.trusted_update_allowed
    assert result.reason == 'valid_long_term_update_insufficient_support'
    assert tracker.left_state.geometry is old_left
    assert tracker.right_state.geometry is old_right


def test_multi_frame_wrong_side_cannot_walk_reference_across_corridor():
    tracker, single = initialized()
    correct_left = line(.35, 70, .4, 1.2)
    output = single.process(tracker.process((correct_left,), 1.0), tracker, 1.0)
    assert output.center is not None
    safe_reference = tracker.left_state.association_reference
    for frame, y in enumerate((.05, -.10, -.25, -.35), start=1):
        fragment = line(y, 70+frame, .45, 1.15)
        result = tracker.process((fragment,), 1.0+frame*.07)
        single.process(result, tracker, 1.0+frame*.07)
        assert tracker.left_state.association_reference is safe_reference


def test_reconstructed_opposite_still_never_updates_identity_reference():
    tracker, single = initialized()
    right_reference = tracker.right_state.association_reference
    output = single.process(
        tracker.process((line(.35, 90),), 1.0), tracker, 1.0)
    assert output.center is not None
    assert tracker.right_state.association_reference is right_reference
