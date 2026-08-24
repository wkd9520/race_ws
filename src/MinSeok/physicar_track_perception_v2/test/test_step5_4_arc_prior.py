import numpy as np
import pytest

from physicar_track_perception_v2.arc_prior import (
    ArcPriorConfig, ArcPriorMemory, OdomArcPrior,
)
from physicar_track_perception_v2.arc_shadow import transform_points
from physicar_track_perception_v2.both_geometry import FrameLocalBothGeometry
from physicar_track_perception_v2.components import (
    CanonicalComponentExtractor, ComponentExtractionConfig, WHITE,
)
from physicar_track_perception_v2.geometry import BevGrid
from physicar_track_perception_v2.trusted_identity import (
    TrustedBoundaryIdentity, TrustedBoundaryState,
)


GRID = BevGrid(.1, 2., -.75, .75, .01)
EXTRACTOR = CanonicalComponentExtractor(
    GRID, ComponentExtractionConfig(canonical_spacing=.05))


def transform(x=0.0, y=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def make(points, component_id=1):
    result, reason = EXTRACTOR.canonicalize_ordered_points(
        np.asarray(points), component_id=component_id, color=WHITE)
    assert reason == 'valid'
    return result


def arc(center=(.8, .7), radius=.55, start=-2.5, stop=.1,
        count=240, component_id=1):
    theta = np.linspace(start, stop, count)
    points = np.asarray(center)+radius*np.column_stack(
        (np.cos(theta), np.sin(theta)))
    return make(points, component_id)


def line(y, component_id, x0=.25, x1=1.55):
    x = np.linspace(x0, x1, 180)
    return make(np.column_stack((x, np.full_like(x, y))), component_id)


@pytest.mark.parametrize('side,mirror', [('LEFT', False), ('RIGHT', True)])
def test_actual_trusted_arc_confirms_after_three_frames(side, mirror):
    prior = OdomArcPrior()
    for frame in range(3):
        value = arc(center=(.8, -.7 if mirror else .7), component_id=frame+1)
        prior.observe_actual(
            side, value, np.eye(3), frame*.1, actual_observed=True)
    assert prior.memory[side] is not None
    assert prior.memory[side].radius == pytest.approx(.55, abs=.01)


def test_single_frame_and_unstable_radius_do_not_confirm():
    prior = OdomArcPrior()
    prior.observe_actual('LEFT', arc(), np.eye(3), 0.0,
                         actual_observed=True)
    assert prior.memory['LEFT'] is None


def test_odom_center_inconsistent_does_not_confirm():
    prior = OdomArcPrior()
    source = arc()
    for frame, x in enumerate((0.0, .25, .50)):
        prior.observe_actual(
            'LEFT', source, transform(x=x), frame*.1,
            actual_observed=True)
    assert prior.memory['LEFT'] is None
    for frame, radius in enumerate((.55, .9, .5), start=1):
        prior.observe_actual(
            'LEFT', arc(radius=radius), np.eye(3), frame*.1,
            actual_observed=True)
    assert prior.memory['LEFT'] is None


def test_reconstructed_geometry_cannot_create_memory():
    prior = OdomArcPrior()
    for frame in range(4):
        prior.observe_actual(
            'LEFT', arc(), np.eye(3), frame*.1, actual_observed=False)
    assert prior.memory['LEFT'] is None
    assert not prior.pending['LEFT']


@pytest.mark.parametrize('motion', [
    transform(.3, 0.0, 0.0),
    transform(0.0, 0.0, .4),
    transform(.3, -.2, .4),
])
def test_se2_propagation_rescues_transformed_current_arc(motion):
    prior = OdomArcPrior()
    source = arc()
    for frame in range(3):
        prior.observe_actual(
            'LEFT', source, np.eye(3), frame*.1, actual_observed=True)
    current_points = transform_points(
        source.canonical_points, np.linalg.inv(motion))
    current = make(current_points, 10)
    result = prior.rescue('LEFT', (current,), motion, .4)
    assert result.valid
    assert result.radial_median < .01
    assert result.nearest_median < .01
    assert prior.memory['LEFT'].radius == pytest.approx(.55, abs=.01)


def test_straight_and_wrong_tangent_are_not_rescued():
    prior = OdomArcPrior()
    source = arc()
    for frame in range(3):
        prior.observe_actual(
            'LEFT', source, np.eye(3), frame*.1, actual_observed=True)
    straight = line(.0, 20)
    assert not prior.rescue('LEFT', (straight,), np.eye(3), .4).valid
    center = prior.memory['LEFT'].center_odom
    radial = np.linspace(.45, .65, 100)
    radial_line = make(center+np.column_stack((radial, np.zeros_like(radial))), 21)
    result = prior.rescue('LEFT', (radial_line,), np.eye(3), .4)
    assert not result.valid


def test_attached_external_tail_is_excluded_from_arc_interval():
    prior = OdomArcPrior()
    source = arc()
    for frame in range(3):
        prior.observe_actual(
            'LEFT', source, np.eye(3), frame*.1, actual_observed=True)
    tail = source.canonical_points[-1]+np.column_stack((
        np.linspace(0.0, .8, 100), np.linspace(0.0, .8, 100)))
    merged = make(np.vstack((source.canonical_points, tail[1:])), 22)
    result = prior.rescue('LEFT', (merged,), np.eye(3), .4)
    assert result.valid
    assert result.accepted.support_length >= .5
    assert result.rejected_tail_support >= .5


def test_age_and_transform_unavailable_are_bounded():
    prior = OdomArcPrior(ArcPriorConfig(max_age_seconds=.5))
    source = arc()
    for frame in range(3):
        prior.observe_actual(
            'LEFT', source, np.eye(3), frame*.1, actual_observed=True)
    assert not prior.rescue('LEFT', (source,), None, .3).valid
    assert not prior.rescue('LEFT', (source,), np.eye(3), 1.0).valid


def _initialized_tracker(prior):
    tracker = TrustedBoundaryIdentity(FrameLocalBothGeometry(), arc_prior=prior)
    pair = (line(.35, 100), line(-.35, 101))
    for frame in range(3):
        tracker.process(pair, frame*.1, np.eye(3))
    return tracker


def test_direct_success_is_not_overridden_by_arc():
    prior = OdomArcPrior()
    tracker = _initialized_tracker(prior)
    current = line(.35, 110)
    result = tracker.process((current,), .4, np.eye(3))
    assert result.left.valid
    assert result.left.association_source == 'DIRECT'


def test_arc_rescue_still_passes_side_invariant_and_cannot_refresh_itself():
    prior = OdomArcPrior()
    tracker = _initialized_tracker(prior)
    # A large-radius arc is locally a LEFT boundary around y=+0.35.
    center = np.array([.9, 20.0])
    radius = 19.65
    theta = np.linspace(-np.pi/2-.03, -np.pi/2+.03, 180)
    current = make(center+radius*np.column_stack(
        (np.cos(theta), np.sin(theta))), 120)
    prior.memory['LEFT'] = ArcPriorMemory(
        'LEFT', center, radius, current.canonical_points.copy(),
        .3, .3, 0.0, 0.0)
    # Poison only the short-term direct reference so direct/sliding fails;
    # the independent long-term center remains the side invariant.
    old = tracker.left_state
    tracker.left_state = TrustedBoundaryState(
        'LEFT', True, old.geometry, old.timestamp, old.physical_support,
        line(.65, 121), old.last_observed)
    before_actual = prior.memory['LEFT'].last_actual_time
    result = tracker.process((current,), .4, np.eye(3))
    assert result.left is not None and result.left.valid
    assert result.left.association_source == 'ARC_ASSISTED'
    assert result.left.side_state == 'SIDE_CONSISTENT'
    assert prior.memory['LEFT'].last_actual_time == before_actual


def test_wrong_side_arc_rescue_is_rejected_by_existing_invariant():
    prior = OdomArcPrior()
    tracker = _initialized_tracker(prior)
    center = np.array([.9, 20.0])
    radius = 20.35
    theta = np.linspace(-np.pi/2-.03, -np.pi/2+.03, 180)
    wrong = make(center+radius*np.column_stack(
        (np.cos(theta), np.sin(theta))), 130)
    # Deliberately create a LEFT memory on the physical RIGHT geometry.
    prior.memory['LEFT'] = ArcPriorMemory(
        'LEFT', center, radius,
        wrong.canonical_points.copy(), .3, .3, 0.0, 0.0)
    tracker.left_state = TrustedBoundaryState(
        'LEFT', True, tracker.left_state.geometry, .3,
        tracker.left_state.physical_support, line(.65, 131), None)
    result = tracker.process((wrong,), .4, np.eye(3))
    assert result.left is None
    assert prior.counts['ARC_RESCUE_SUCCESS'] == 0
    assert prior.counts['ARC_RESCUE_REJECTED'] > 0
