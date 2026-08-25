import numpy as np

from physicar_track_perception_v3.circle_avoidance import (
    AVOID_LEFT, AVOID_RIGHT)
from physicar_track_perception_v3.low_vote_recovery import (
    DETERMINISTIC_DEFAULT,
    LOCKED,
    NORMAL_VOTING,
    RECOVERY_FAILED_DEFAULT,
    NORMAL_VOTE,
    WHITE_PROXIMITY,
    WHITE_PROXIMITY_FALLBACK,
    LowVoteRecoveryConfig,
    LowVoteRecoveryManager,
    RecoveryTrackView,
    WhiteComponentView,
    white_proximity_decision,
)
from physicar_track_perception_v3.obstacle_tracks import (
    MultiObstacleTracker, ObstacleObservation, ObstacleTrackConfig)


def config(**overrides):
    values = dict(
        freeze_distance=1.0,
        slow_voting_max_frames=4,
        heading_recovery_max_frames=5,
        max_heading_correction=0.25,
        emergency_distance=0.45,
        center_start_distance=0.60,
        default_avoidance_side=AVOID_LEFT)
    values.update(overrides)
    return LowVoteRecoveryConfig(**values)


def track(ident=1, center=(0.9, 0.1), left=0, right=0,
          locked=False, side=None, source=None):
    return RecoveryTrackView(
        ident, center, float(np.linalg.norm(center)), left, right,
        left+right, locked, side, source)


def center_straight():
    return np.array([[0.1, 0.0], [0.8, 0.0], [1.5, 0.0]])


def test_zero_vote_inside_freeze_immediately_uses_white_proximity():
    white = WhiteComponentView(7, [[0.4, 0.3], [1.0, 0.3]])
    result = LowVoteRecoveryManager(config()).update(
        track(), center_straight(), [white], measurement_key=1)
    assert result.state == WHITE_PROXIMITY_FALLBACK
    assert result.chosen_side == AVOID_RIGHT
    assert result.lock_source == WHITE_PROXIMITY
    assert result.lock_requested


def test_tracker_preserves_zero_votes_inside_freeze_for_white_fallback():
    tracker = MultiObstacleTracker(ObstacleTrackConfig())
    observation = ObstacleObservation(
        1, np.array([2.0, 0.0]), np.array([0.9, 0.0]), 0.08,
        0.1, 'LEFT', 0.9, True)
    result = tracker.update([observation], 0.0, measurement_key=1)
    assert result.tracks[0].vote_count == 0
    assert not result.tracks[0].direction_locked


def test_any_available_vote_locks_from_majority():
    manager = LowVoteRecoveryManager(config())
    result = manager.update(
        track(left=5, right=2), center_straight(), measurement_key=1)
    assert result.state == LOCKED
    assert result.chosen_side == AVOID_RIGHT
    assert result.lock_source == NORMAL_VOTE
    assert result.lock_requested


def test_outside_freeze_keeps_normal_voting():
    result = LowVoteRecoveryManager(config()).update(
        track(center=(1.2, 0.1)), center_straight(), measurement_key=1)
    assert result.state == NORMAL_VOTING
    assert not result.lock_requested


def test_nearest_white_left_means_avoid_right_and_mirror():
    center = center_straight()
    left = WhiteComponentView(1, [[0.6, 0.3], [1.2, 0.3]])
    decision = white_proximity_decision(
        (0.8, 0.0), [left], center, default_side=AVOID_LEFT)
    assert decision['side'] == AVOID_RIGHT
    assert decision['source'] == WHITE_PROXIMITY

    right = WhiteComponentView(2, [[0.6, -0.3], [1.2, -0.3]])
    decision = white_proximity_decision(
        (0.8, 0.0), [right], center, default_side=AVOID_LEFT)
    assert decision['side'] == AVOID_LEFT


def test_curved_local_frame_does_not_use_global_y():
    curved_center = np.array([[0.0, 0.1], [0.0, 0.8], [0.0, 1.5]])
    # For tangent +Y, local LEFT is -X. WHITE at -X means avoid local RIGHT.
    white = WhiteComponentView(3, [[-0.3, 0.6], [-0.3, 1.2]])
    decision = white_proximity_decision(
        (0.0, 0.8), [white], curved_center,
        default_side=AVOID_LEFT)
    assert decision['side'] == AVOID_RIGHT
    assert decision['reference_source'] == 'CURRENT_CENTER'


def test_white_then_recent_history_tangent_fallback_priority():
    white = WhiteComponentView(4, [[0.7, 0.3], [1.2, 0.3]])
    decision = white_proximity_decision(
        (0.8, 0.0), [white], [], [], AVOID_LEFT)
    assert decision['reference_source'] == 'CURRENT_WHITE'

    degenerate_white = WhiteComponentView(5, [[0.7, 0.3], [0.7, 0.3]])
    history = np.array([[0.1, 0.0], [1.2, 0.0]])
    decision = white_proximity_decision(
        (0.8, 0.0), [degenerate_white], [], history, AVOID_LEFT)
    assert decision['reference_source'] == 'RECENT_CENTER_HISTORY'
    assert decision['side'] == AVOID_RIGHT


def test_no_white_uses_deterministic_default():
    decision = white_proximity_decision(
        (0.8, 0.0), [], center_straight(), [], AVOID_LEFT)
    assert decision['side'] == AVOID_LEFT
    assert decision['source'] == DETERMINISTIC_DEFAULT


def test_white_lock_is_latched_when_track_later_has_opposite_votes():
    manager = LowVoteRecoveryManager(config())
    locked = track(left=0, right=0, locked=True, side=AVOID_LEFT,
                   source=WHITE_PROXIMITY)
    first = manager.update(locked, center_straight(), measurement_key=1)
    opposite = track(left=10, right=0, locked=True, side=AVOID_LEFT,
                     source=WHITE_PROXIMITY)
    second = manager.update(opposite, center_straight(), measurement_key=2)
    assert first.chosen_side == second.chosen_side == AVOID_LEFT
    assert second.lock_source == WHITE_PROXIMITY


def test_per_track_recovery_memory_is_isolated():
    manager = LowVoteRecoveryManager(config())
    a = manager.update(track(1), center_straight(), measurement_key=1)
    b = manager.update(
        track(2, center=(1.3, 0.1), left=2), center_straight(),
        measurement_key=1)
    c = manager.update(
        track(3, locked=True, side=AVOID_RIGHT, source='NORMAL_VOTE'),
        center_straight(), measurement_key=1)
    assert a.state == RECOVERY_FAILED_DEFAULT
    assert b.state == NORMAL_VOTING
    assert c.state == LOCKED
    assert manager.memories[1].state == RECOVERY_FAILED_DEFAULT


def test_duplicate_measurement_does_not_advance_recovery_frames():
    manager = LowVoteRecoveryManager(config())
    first = manager.update(track(), center_straight(), measurement_key=7)
    duplicate = manager.update(track(), center_straight(), measurement_key=7)
    assert duplicate.duplicate_measurement
    assert duplicate.recovery_frame_count == first.recovery_frame_count
