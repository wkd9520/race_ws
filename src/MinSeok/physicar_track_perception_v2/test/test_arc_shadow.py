from types import SimpleNamespace

import numpy as np
import pytest

from physicar_track_perception_v2.arc_shadow import (
    ArcShadowConfig,
    ArcShadowTracker,
    fit_circle_shadow,
    inverse_transform,
    transform_points,
)


def transform(x=0.0, y=0.0, yaw=0.0):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, x], [s, c, y], [0.0, 0.0, 1.0]])


def arc(center=(0.5, 0.4), radius=0.65, start=-1.2, stop=1.0, count=80):
    theta = np.linspace(start, stop, count)
    return np.asarray(center)+radius*np.column_stack((np.cos(theta), np.sin(theta)))


def candidate(points, cid=1, color='WHITE'):
    points = np.asarray(points, dtype=np.float64)
    support = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    return SimpleNamespace(
        component_id=cid, color=color, canonical_points=points,
        canonical_point_count=len(points), support_length=support)


def association(value, side='LEFT'):
    return SimpleNamespace(
        side=side, valid=True, candidate=value, accepted=value)


def test_rigid_transform_circle_center_and_radius():
    points = arc()
    T = transform(2.0, -0.3, 0.6)
    result = fit_circle_shadow(points, T)
    expected = transform_points(np.asarray([[0.5, 0.4]]), T)[0]
    assert result.strong
    assert result.radius == pytest.approx(0.65, abs=1e-10)
    assert np.allclose(result.center_odom, expected, atol=1e-10)


def test_known_odom_motion_returns_expected_current_base_center():
    center_odom = np.asarray([[2.0, 1.0]])
    T_odom_current = transform(0.8, -0.2, 0.4)
    actual = transform_points(center_odom, inverse_transform(T_odom_current))[0]
    expected = transform_points(center_odom, np.linalg.inv(T_odom_current))[0]
    assert np.allclose(actual, expected)


@pytest.mark.parametrize('mirror', [False, True])
def test_left_right_mirror_circle_fit(mirror):
    points = arc(center=(0.5, -0.4 if mirror else 0.4))
    if mirror:
        points[:, 1] *= -1.0
    result = fit_circle_shadow(points, np.eye(3))
    assert result.strong and result.radius == pytest.approx(0.65, abs=1e-10)


def test_three_consistent_frames_confirm_model_and_age_increments():
    tracker = ArcShadowTracker()
    fixed_center = np.array([2.0, 1.0])
    last = None
    for frame in range(3):
        T = transform(0.03*frame, 0.0, 0.01*frame)
        center_base = transform_points(
            fixed_center[None, :], inverse_transform(T))[0]
        value = candidate(arc(center=center_base))
        last = tracker.process(frame*0.1, T, {'LEFT': association(value)}, (value,))['LEFT']
    assert last.memory is not None
    assert last.confirm_streak == 3
    assert np.allclose(last.memory.center_odom, fixed_center, atol=1e-9)
    empty = tracker.process(0.3, transform(.09), {}, ())['LEFT']
    assert empty.age_frames == 1
    assert empty.age_seconds == pytest.approx(.1)


def test_unstable_radius_does_not_confirm():
    tracker = ArcShadowTracker()
    for frame, radius in enumerate((0.6, 0.9, 0.55, 1.0)):
        value = candidate(arc(radius=radius))
        result = tracker.process(
            frame*.1, np.eye(3), {'LEFT': association(value)}, (value,))['LEFT']
    assert result.memory is None
    assert result.confirm_streak == 1


def test_shadow_does_not_mutate_candidate_or_production_result():
    tracker = ArcShadowTracker()
    value = candidate(arc())
    before = value.canonical_points.copy()
    production = SimpleNamespace(reason='unchanged', candidate=value)
    tracker.process(0.0, np.eye(3), {'LEFT': association(value)}, (value,))
    assert np.array_equal(value.canonical_points, before)
    assert production.reason == 'unchanged' and production.candidate is value


def test_weak_short_arc_is_not_confirmed():
    config = ArcShadowConfig(min_angular_span=np.deg2rad(60.0))
    result = fit_circle_shadow(
        arc(start=0.0, stop=0.15, count=20), np.eye(3), config)
    assert result is not None and not result.strong
    assert result.reason in ('angular_span', 'contiguous_support')
