import numpy as np
import pytest

from physicar_track_perception_v2.both_geometry import FrameLocalBothGeometry
from physicar_track_perception_v2.components import CanonicalComponentExtractor, ComponentExtractionConfig, WHITE
from physicar_track_perception_v2.geometry import BevGrid
from physicar_track_perception_v2.single_reconstruction import (
    BOTH_OBSERVED, LEFT_ONLY_OBSERVED, RIGHT_ONLY_OBSERVED,
    DEGENERATE, REGULAR, UNKNOWN, TrustedSingleReconstruction,
)
from physicar_track_perception_v2.trusted_identity import TrustedBoundaryIdentity


GRID = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)
EXTRACTOR = CanonicalComponentExtractor(GRID, ComponentExtractionConfig(canonical_spacing=0.05))


def make(points, cid):
    value, reason = EXTRACTOR.canonicalize_ordered_points(points, component_id=cid, color=WHITE)
    assert reason == 'valid'
    return value


def straight(yaw=0.0, offset=0.0, length=1.3, ids=(1, 2)):
    s = np.linspace(0.25, 0.25+length, 180)
    t = np.array([np.cos(yaw), np.sin(yaw)])
    n = np.array([-np.sin(yaw), np.cos(yaw)])
    c = np.array([0.25, offset])+(s-0.25)[:, None]*t
    return make(c+0.35*n, ids[0]), make(c-0.35*n, ids[1])


def arcs(radius=1.0, mirror=False, ids=(1, 2)):
    theta = np.linspace(0, np.pi/2, 240)
    sign = -1 if mirror else 1
    c = np.column_stack((0.2+radius*np.sin(theta), sign*radius*(1-np.cos(theta))))
    t = np.column_stack((np.cos(theta), sign*np.sin(theta)))
    n = np.column_stack((-t[:, 1], t[:, 0]))
    return make(c+0.35*n, ids[0]), make(c-0.35*n, ids[1])


def ready(pair=None):
    pair = pair or straight()
    identity = TrustedBoundaryIdentity(FrameLocalBothGeometry())
    single = TrustedSingleReconstruction()
    outputs = []
    for frame in range(7):
        result = identity.process(pair, frame*.07)
        outputs.append(single.process(result, identity, frame*.07))
    assert identity.initialized and single.width.state.initialized
    return identity, single, outputs


def test_stable_both_initializes_width_and_updates_only_from_both():
    identity, single, outputs = ready()
    assert single.width.state.width == pytest.approx(0.70, abs=0.01)
    assert outputs[-1].observation_mode == BOTH_OBSERVED
    before = single.width.state.update_count
    result = identity.process((straight(ids=(4, 5))[0],), 1.0)
    output = single.process(result, identity, 1.0)
    assert not output.width_update_allowed
    assert single.width.state.update_count == before


@pytest.mark.parametrize('survivor, mode', [(0, LEFT_ONLY_OBSERVED), (1, RIGHT_ONLY_OBSERVED)])
def test_straight_single_reconstructs_center_and_missing(survivor, mode):
    identity, single, _ = ready()
    boundary = straight(ids=(7, 8))[survivor]
    result = single.process(identity.process((boundary,), 1.0), identity, 1.0)
    assert result.observation_mode == mode
    assert result.center is not None and result.missing is not None
    assert np.max(np.abs(result.center.points[:, 1])) < 0.02
    expected = -0.35 if survivor == 0 else 0.35
    assert np.median(result.missing.points[:, 1]) == pytest.approx(expected, abs=0.02)


@pytest.mark.parametrize('yaw', [np.pi/4, -np.pi/4])
def test_rotated_single_uses_local_normal_not_global_y(yaw):
    pair = straight(yaw=yaw)
    identity, single, _ = ready(pair)
    result = single.process(identity.process((pair[0],), 1.0), identity, 1.0)
    observed = pair[0].canonical_points[:len(result.center.points)]
    delta = result.center.points-observed
    tangent = np.array([np.cos(yaw), np.sin(yaw)])
    assert np.max(np.abs(delta@tangent)) < 0.02


@pytest.mark.parametrize('mirror,survivor', [(False, 0), (False, 1), (True, 0), (True, 1)])
def test_ninety_degree_inner_outer_and_mirror_are_concentric(mirror, survivor):
    pair = arcs(mirror=mirror)
    identity, single, _ = ready(pair)
    output = single.process(identity.process((pair[survivor],), 1.0), identity, 1.0)
    assert output.center is not None and output.missing is not None
    circle_center = np.array([0.2, -1.0 if mirror else 1.0])
    center_radius = np.linalg.norm(output.center.points-circle_center, axis=1)
    assert np.median(center_radius) == pytest.approx(1.0, abs=0.015)


def test_normal_sign_comes_from_trusted_opposite_and_is_stable():
    identity, single, _ = ready(arcs())
    signs = []
    for cid in (10, 11, 12):
        left = arcs(ids=(cid, cid+20))[0]
        output = single.process(identity.process((left,), cid), identity, cid)
        signs.append(output.normal_sign)
        assert output.normal_sign_source == 'trusted_opposite_boundary'
    assert len(set(signs)) == 1


def test_ambiguous_normal_has_no_false_center():
    identity, single, _ = ready()
    identity.right_state = type(identity.right_state)('RIGHT', True,
        make([[0.5, 0.35], [1.0, 0.35]], 90), 0.0, 0.5)
    left = straight()[0]
    output = single.process(identity.process((left,), 1.0), identity, 1.0)
    assert output.center is None


def test_center_regular_missing_degenerate_are_separated():
    _, single, _ = ready()
    outer = arcs(radius=0.2)[1]
    center = single._safety(outer, 0.35)
    missing = single._safety(outer, 0.70)
    assert center.state == REGULAR
    assert missing.state == DEGENERATE


def test_genuine_center_degeneracy_is_rejected():
    _, single, _ = ready()
    theta = np.linspace(0, np.pi, 160)
    tight = make(np.column_stack((0.5+0.2*np.sin(theta),
                                  0.2*(1-np.cos(theta)))), 70)
    assert single._safety(tight, 0.35).state == DEGENERATE


def test_short_geometry_is_explicit_unknown():
    _, single, _ = ready()
    short = make([[0.2, 0.35], [0.25, 0.35], [0.30, 0.35]], 30)
    assert single._safety(short, -0.35).state == UNKNOWN


def test_external_candidate_does_not_acquire_missing_identity_or_update_width():
    identity, single, _ = ready()
    left = straight(ids=(7, 8))[0]
    x = np.linspace(.3, 1.4, 80)
    external = make(np.column_stack((x, np.full_like(x, -1.0))), 99)
    before_right = identity.right_state.geometry
    before_width = single.width.state
    output = single.process(identity.process((left, external), 1.0), identity, 1.0)
    assert output.observation_mode == LEFT_ONLY_OBSERVED
    assert identity.right_state.geometry is before_right
    assert single.width.state == before_width


@pytest.mark.parametrize('side', [0, 1])
def test_repeated_single_never_overwrites_missing_trusted_state(side):
    identity, single, _ = ready()
    missing_before = identity.right_state.geometry if side == 0 else identity.left_state.geometry
    for frame in range(3):
        boundary = straight(ids=(20+frame, 30+frame))[side]
        single.process(identity.process((boundary,), frame+1), identity, frame+1)
    missing_after = identity.right_state.geometry if side == 0 else identity.left_state.geometry
    assert missing_after is missing_before


def test_return_to_actual_both_resumes_both_and_width_update():
    identity, single, _ = ready()
    single.process(identity.process((straight()[0],), 1.0), identity, 1.0)
    output = single.process(identity.process(straight(ids=(40, 41)), 1.1), identity, 1.1)
    assert output.observation_mode == BOTH_OBSERVED
    assert output.width_update_allowed
