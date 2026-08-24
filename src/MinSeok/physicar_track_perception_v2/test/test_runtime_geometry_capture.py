import json
from types import SimpleNamespace

import numpy as np

from physicar_track_perception_v2.runtime_geometry_capture import (
    RuntimeGeometryCapture,
    rejected_canonical_points,
)


def geometry(points):
    points = np.asarray(points, dtype=np.float64)
    return SimpleNamespace(canonical_points=points)


def test_rejected_points_are_derived_without_mutating_candidate():
    full = geometry([[0., 0.], [.1, 0.], [.2, 0.], [.3, 0.]])
    accepted = geometry([[.1, 0.], [.2, 0.]])
    before = full.canonical_points.copy()
    rejected = rejected_canonical_points(full, accepted)
    assert np.array_equal(rejected, [[0., 0.], [.3, 0.]])
    assert np.array_equal(full.canonical_points, before)


def test_scene_markers_gate_capture_state_and_are_preserved(tmp_path):
    capture = RuntimeGeometryCapture(tmp_path)
    assert not capture.active
    capture.set_scene('C_90_LEFT', 12.5)
    assert capture.active
    capture.set_scene('STOP', 13.0)
    assert not capture.active
    capture.close()
    markers = [json.loads(line) for line in
               (tmp_path/'scene_markers.jsonl').read_text().splitlines()]
    assert markers == [
        {'wall_time': 12.5, 'scene': 'C_90_LEFT'},
        {'wall_time': 13.0, 'scene': 'STOP'},
    ]
