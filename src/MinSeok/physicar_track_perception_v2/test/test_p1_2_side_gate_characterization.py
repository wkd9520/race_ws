import json

import numpy as np

from physicar_track_perception_v2.both_geometry import CanonicalCenterPath
from physicar_track_perception_v2.components import CanonicalBoundaryCandidate
from physicar_track_perception_v2.side_gate_characterization import (
    SideGateCharacterizer, boundary_corridor_shadow, characterize_side_gate,
    segment_correspondence)
from physicar_track_perception_v2.trusted_identity import (
    AssociationResult, IdentityConfig)


def candidate(points, component_id=1):
    points = np.asarray(points, dtype=float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    spacing = np.diff(s)
    return CanonicalBoundaryCandidate(
        component_id, 'WHITE', points, s, points, s, float(s[-1]),
        len(points), len(points), float(spacing.min()),
        float(np.median(spacing)), float(spacing.max()), .05,
        points[0], points[-1])


def center(points, width=.7):
    points = np.asarray(points, dtype=float)
    s = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]
    return CanonicalCenterPath(
        points, s, float(s[-1]), 1, 2, 'WHITE', 'WHITE', float(s[-1]),
        len(points), np.full(len(points), width), width, width, width,
        True, 'test')


def association(side, value):
    return AssociationResult(
        side, value, True, True, 'valid', value, 0.0,
        value.support_length, 1.0, 0.0, value.support_length,
        value.support_length, 0.0)


def test_endpoint_clamp_detection_and_coverage():
    result = segment_correspondence(
        [[.5, .35], [1.2, .35]], [[0., 0.], [1., 0.]])
    assert not result['endpoint_clamped'][0]
    assert result['far_clamped'][1]


def test_signed_lateral_and_side_support_are_exact():
    observed = candidate([[0., .35], [.25, .35], [.5, .35]])
    result = characterize_side_gate(
        'LEFT', association('LEFT', observed),
        center([[0., 0.], [.5, 0.]]), IdentityConfig(), observed)
    assert result['exact_side_state'] == 'SIDE_CONSISTENT'
    assert abs(result['signed_lateral_median']-.35) < 1e-9
    assert result['opposite_support'] == 0.0
    json.dumps(result)


def test_opposite_and_center_crossing_are_distinguished():
    wrong = candidate([[0., -.35], [.25, -.35], [.5, -.35]])
    result = characterize_side_gate(
        'LEFT', association('LEFT', wrong),
        center([[0., 0.], [.5, 0.]]), IdentityConfig(), wrong)
    assert result['exact_side_state'] == 'SIDE_OPPOSITE'
    crossing = candidate([[0., -.35], [.25, .35], [.5, .35]])
    result = characterize_side_gate(
        'LEFT', association('LEFT', crossing),
        center([[0., 0.], [.5, 0.]]), IdentityConfig(), crossing)
    assert result['exact_side_state'] == 'CENTER_CROSSING'


def test_boundary_shadow_recovers_local_corridor_relation():
    observed = candidate([[1., .35], [1.25, .35], [1.5, .35]])
    trusted = candidate([[1., .35], [1.25, .35], [1.5, .35]])
    result = boundary_corridor_shadow(
        'LEFT', observed.canonical_points, trusted, .35)
    assert result['available']
    assert result['side_consistency_fraction'] == 1.0
    assert result['lateral_residual_median'] < 1e-9


def test_representative_npz_integrity(tmp_path):
    writer = SideGateCharacterizer(tmp_path)
    observed = candidate([[0., -.35], [.25, -.35], [.5, -.35]])
    evidence = characterize_side_gate(
        'LEFT', association('LEFT', observed),
        center([[0., 0.], [.5, 0.]]), IdentityConfig(), observed)
    writer._capture_representative(
        1.0, .35, 'LEFT', association('LEFT', observed),
        center([[0., 0.], [.5, 0.]]), observed, evidence)
    files = list((tmp_path/'representative_npz').glob('*.npz'))
    assert len(files) == 1
    with np.load(files[0]) as archive:
        assert archive['accepted_points'].shape == (3, 2)
        assert archive['trusted_center_points'].shape == (2, 2)
