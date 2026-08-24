import json
from types import SimpleNamespace

import numpy as np

from physicar_track_perception_v2.arc_support_capture import (
    ArcSupportCapture, detailed_circle_fit, retention_metrics)
from physicar_track_perception_v2.components import (
    CanonicalComponentExtractor, ComponentExtractionConfig, WHITE)
from physicar_track_perception_v2.geometry import BevGrid


def candidate(points, component_id=1):
    extractor = CanonicalComponentExtractor(
        BevGrid(0.10, 2.00, -0.75, 0.75, 0.01),
        ComponentExtractionConfig(canonical_spacing=0.05))
    value, reason = extractor.canonicalize_ordered_points(
        points, component_id, WHITE)
    assert reason == 'valid'
    return value


def arc(radius=.6, span=np.pi/2, count=100):
    angles = np.linspace(-span/2, span/2, count)
    return np.column_stack((1.0+radius*np.cos(angles),
                            radius*np.sin(angles)))


def test_retention_calculation_uses_physical_support():
    value = retention_metrics(2.0, .5, .4, np.pi/2, np.pi/6)
    assert value == {'geometry': .25, 'circle': .8,
                     'total': .2, 'span': 1/3}


def test_detailed_fit_exposes_inlier_geometry_without_changing_fit():
    value = detailed_circle_fit(arc(), np.eye(3))
    assert value is not None
    assert value['hypothesis'].strong
    assert value['inlier_mask'].dtype == bool
    assert value['inlier_run_count'] == 1
    assert value['longest_start'] == 0
    assert value['longest_end'] == len(arc())-1


def test_capture_full_and_accepted_integrity(tmp_path):
    full = candidate(arc(count=140))
    accepted = candidate(full.canonical_points[4:16], component_id=1)
    association = SimpleNamespace(
        candidate=full, accepted=accepted,
        accepted_support=accepted.support_length,
        interval_start_s=.2, interval_end_s=.75,
        overlap_support=.3,
        rejected_tail_support=full.support_length-accepted.support_length,
        reference_support=.5, association_source='DIRECT', reason='valid',
        sliding_association_used=False)
    identity = SimpleNamespace(left=association, right=None, reason='valid')
    tracker = SimpleNamespace(
        pending={'LEFT': [], 'RIGHT': []},
        memory={'LEFT': None, 'RIGHT': None})
    stamp = SimpleNamespace(sec=12, nanosec=34)
    capture = ArcSupportCapture(tmp_path)
    paths = capture.capture(stamp, (full,), identity, tracker, np.eye(3))
    assert paths
    left_path = next(path for path in paths if '_LEFT_' in path.name)
    data = np.load(left_path)
    metadata = json.loads(str(data['metadata_json']))
    assert np.array_equal(data['full_canonical_points'], full.canonical_points)
    assert np.array_equal(data['accepted_points'], accepted.canonical_points)
    assert metadata['retention']['geometry'] < 1.0
    assert metadata['accepted']['direct_overlap_support'] == .3


def test_diagnostic_fit_cannot_mutate_geometry():
    points = arc()
    before = points.copy()
    detailed_circle_fit(points, np.eye(3))
    assert np.array_equal(points, before)
