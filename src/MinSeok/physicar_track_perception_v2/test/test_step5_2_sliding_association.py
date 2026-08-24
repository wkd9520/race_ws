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
from physicar_track_perception_v2.trusted_identity import TrustedBoundaryIdentity


GRID = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)
EXTRACTOR = CanonicalComponentExtractor(
    GRID, ComponentExtractionConfig(canonical_spacing=0.05))


def make(points, component_id):
    result, reason = EXTRACTOR.canonicalize_ordered_points(
        points, component_id=component_id, color=WHITE)
    assert reason == 'valid'
    return result


def segment(x0, x1, y, component_id):
    x = np.linspace(x0, x1, max(30, int((x1-x0)/.005)))
    return make(np.column_stack((x, np.full_like(x, y))), component_id)


def initial_pair():
    return segment(.20, 1.20, .35, 1), segment(.20, 1.20, -.35, 2)


def initialized():
    tracker = TrustedBoundaryIdentity(FrameLocalBothGeometry())
    single = TrustedSingleReconstruction(ValidatedWidth())
    pair = initial_pair()
    for frame in range(6):
        identity = tracker.process(pair, frame*.07)
        single.process(identity, tracker, frame*.07)
    assert tracker.initialized and single.width.state.initialized
    return tracker, single


def arc_segment(theta0, theta1, side, mirror, component_id):
    theta = np.linspace(theta0, theta1, 180)
    turn = -1.0 if mirror else 1.0
    center = np.column_stack((.2+np.sin(theta), turn*(1-np.cos(theta))))
    tangent = np.column_stack((np.cos(theta), turn*np.sin(theta)))
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))
    offset = .35 if side == 'LEFT' else -.35
    return make(center+offset*normal, component_id)


def initialized_arc(mirror=False):
    tracker = TrustedBoundaryIdentity(FrameLocalBothGeometry())
    pair = (arc_segment(0, 1.25, 'LEFT', mirror, 1),
            arc_segment(0, 1.25, 'RIGHT', mirror, 2))
    for frame in range(3):
        tracker.process(pair, frame*.07)
    return tracker


def test_partial_overlap_slides_short_term_reference_not_long_term_identity():
    tracker, _ = initialized()
    long_term = tracker.left_state.geometry
    current = segment(.55, 1.55, .35, 10)
    result = tracker.process((current,), 1.0)
    assert result.left.valid and not result.left.sliding_association_used
    tracker.update_single_observation('LEFT', result.left.accepted, 1.0)
    assert tracker.left_state.geometry is long_term
    assert tracker.left_state.association_reference is result.left.accepted


def test_progressively_shifted_single_observation_has_no_overlap_cliff():
    tracker, single = initialized()
    supports = []
    for index, start in enumerate((.35, .50, .65, .80, .95)):
        current = segment(start, min(1.95, start+.70), .35, 20+index)
        identity = tracker.process((current,), 1.0+index*.07)
        output = single.process(identity, tracker, 1.0+index*.07)
        assert identity.left is not None and identity.left.valid
        assert output.observation_mode == LEFT_ONLY_OBSERVED
        assert output.center is not None
        supports.append(identity.left.accepted_support)
    assert min(supports) > .5


@pytest.mark.parametrize('side,y,mode', [
    ('LEFT', .35, LEFT_ONLY_OBSERVED),
    ('RIGHT', -.35, RIGHT_ONLY_OBSERVED),
])
def test_near_zero_overlap_bounded_endpoint_continuation(side, y, mode):
    tracker, single = initialized()
    first = segment(.20, .75, y, 30)
    identity = tracker.process((first,), 1.0)
    single.process(identity, tracker, 1.0)
    current = segment(.82, 1.42, y, 31)
    identity = tracker.process((current,), 1.07)
    assert identity.reason == 'no_association'
    association = identity.left if side == 'LEFT' else identity.right
    assert association is not None and association.sliding_association_used
    output = single.process(identity, tracker, 1.07)
    assert output.observation_mode == mode and output.center is not None


def test_endpoint_proximity_with_wrong_tangent_is_rejected():
    tracker, _ = initialized()
    first = segment(.20, .75, .35, 40)
    result = tracker.process((first,), 1.0)
    tracker.update_single_observation('LEFT', result.left.accepted, 1.0)
    y = np.linspace(.35, .70, 70)
    wrong = make(np.column_stack((np.full_like(y, .80), y)), 41)
    result = tracker.process((wrong,), 1.07)
    assert result.left is None


def test_large_endpoint_gap_is_rejected():
    tracker, _ = initialized()
    first = segment(.20, .65, .35, 50)
    result = tracker.process((first,), 1.0)
    tracker.update_single_observation('LEFT', result.left.accepted, 1.0)
    result = tracker.process((segment(.90, 1.40, .35, 51),), 1.07)
    assert result.left is None


@pytest.mark.parametrize('mirror', [False, True])
@pytest.mark.parametrize('side', ['LEFT', 'RIGHT'])
def test_ninety_degree_visible_interval_progression_is_metric_2d(mirror, side):
    tracker = initialized_arc(mirror)
    for index, bounds in enumerate(((.15, .85), (.35, 1.05), (.55, 1.25))):
        current = arc_segment(*bounds, side, mirror, 60+index)
        result = tracker.process((current,), 1.0+index*.07)
        association = result.left if side == 'LEFT' else result.right
        assert association is not None and association.valid
        tracker.update_single_observation(side, association.accepted,
                                          1.0+index*.07)


def test_similar_tangent_external_on_wrong_corridor_side_cannot_continue():
    tracker, _ = initialized()
    first = segment(.20, .75, .35, 70)
    result = tracker.process((first,), 1.0)
    tracker.update_single_observation('LEFT', result.left.accepted, 1.0)
    external = segment(.80, 1.35, -.35, 71)
    result = tracker.process((external,), 1.07)
    assert result.left is None


def test_reconstructed_side_never_slides_actual_reference():
    tracker, single = initialized()
    right_reference = tracker.right_state.association_reference
    left = segment(.35, 1.25, .35, 80)
    output = single.process(tracker.process((left,), 1.0), tracker, 1.0)
    assert output.center is not None
    assert tracker.right_state.association_reference is right_reference
