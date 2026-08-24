import json

import numpy as np

from physicar_track_perception_v2.pan_association_characterization import (
    PanAssociationCharacterizer, compare_candidate_reference)


def _line(y, start=0., end=.5, count=11):
    x = np.linspace(start, end, count)
    return np.column_stack((x, np.full(count, y))), x-start


def test_reference_comparison_decomposes_direct_pass():
    points, s = _line(.3)
    metric = compare_candidate_reference(points, s, points.copy())
    assert metric['distance_pass'] and metric['tangent_pass']
    assert metric['overlap_pass'] and metric['raw_valid']
    assert metric['first_failed_gate'] == 'PASS'


def test_reference_comparison_reports_distance_before_other_gates():
    points, s = _line(.3)
    reference, _ = _line(-.3)
    metric = compare_candidate_reference(points, s, reference)
    assert not metric['raw_valid']
    assert metric['first_failed_gate'] == 'DISTANCE_FAIL'
    assert metric['nearest_median'] > .5


def test_short_term_can_fail_while_motion_compensated_reference_passes():
    current, s = _line(.3, 1., 1.5)
    stale, _ = _line(.3, 0., .5)
    propagated = stale+np.array([1., 0.])
    assert not compare_candidate_reference(current, s, stale)['raw_valid']
    assert compare_candidate_reference(current, s, propagated)['raw_valid']


def test_characterizer_has_no_production_decision_api(tmp_path):
    value = PanAssociationCharacterizer(tmp_path)
    assert value.path.name == 'reference_comparison.jsonl'
    assert not hasattr(value, 'associate') and not hasattr(value, 'rescue')


def test_json_metric_values_are_serializable():
    points, s = _line(.3)
    payload = compare_candidate_reference(points, s, points)
    json.dumps(payload, allow_nan=False)
