import numpy as np
from physicar_track_perception_v3.geometry import OrderedPolyline
from physicar_track_perception_v3.proximity import start_distance, validate_start


def path(points):
    return OrderedPolyline.from_points(np.asarray(points, dtype=float))


def test_near_start_passes():
    p = path([[.20, .02], [.50, .02]])
    ok, distance, reason = validate_start(p, .30)
    assert ok and np.isclose(distance, np.hypot(.20, .02)) and reason == 'START_NEAR'


def test_far_start_rejected():
    ok, distance, reason = validate_start(path([[.60, 0.], [.9, 0.]]), .30)
    assert not ok and np.isclose(distance, .60) and reason == 'START_TOO_FAR'


def test_middle_near_does_not_pass():
    ok, _, reason = validate_start(path([[.60, 0.], [.10, 0.], [.8, 0.]]), .30)
    assert not ok and reason == 'START_TOO_FAR'


def test_disabled_gate_only_diagnoses():
    ok, distance, reason = validate_start(path([[.60, 0.], [.9, 0.]]), -1.0)
    assert ok and np.isclose(distance, .60) and reason == 'DISABLED'


def test_missing_path_invalid():
    ok, distance, reason = validate_start(None, .30)
    assert not ok and distance is None and reason == 'NO_START_POINT'
