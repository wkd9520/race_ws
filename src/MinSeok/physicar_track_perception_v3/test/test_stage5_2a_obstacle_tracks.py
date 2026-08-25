import numpy as np

from physicar_track_perception_v3.circle_avoidance import (
    AVOID_LEFT, AVOID_RIGHT, CENTER, LEFT, RIGHT, DirectionLatch)
from physicar_track_perception_v3.obstacle_tracks import (
    MultiObstacleTracker, ObstacleObservation, ObstacleTrackConfig)


def config(**overrides):
    values = dict(association_distance=0.12, retention_age=0.50,
                  direction_freeze_distance=1.5,
                  default_avoidance_side=AVOID_LEFT,
                  max_voting_tracks=2)
    values.update(overrides)
    return ObstacleTrackConfig(**values)


def observation(ident, odom, base=None, side=LEFT, relevant=True):
    odom = np.asarray(odom, dtype=float)
    base = odom.copy() if base is None else np.asarray(base, dtype=float)
    lateral = 0.1 if side == LEFT else -0.1 if side == RIGHT else 0.0
    return ObstacleObservation(
        ident, odom, base, 0.08, lateral, side,
        float(np.linalg.norm(base)), relevant)


def update(tracker, values, stamp, key=None):
    return tracker.update(values, stamp,
                          measurement_key=stamp if key is None else key)


def by_id(result):
    return {item.track_id: item for item in result.tracks}


def test_single_obstacle_parity_with_existing_vote_freeze_latch():
    tracker = MultiObstacleTracker(config())
    latch = DirectionLatch()
    for index, (distance, side) in enumerate((
            (2.2, LEFT), (2.1, LEFT), (2.0, RIGHT), (1.9, LEFT),
            (1.8, LEFT), (1.7, RIGHT), (1.6, LEFT), (1.45, LEFT))):
        result = update(tracker, [observation(
            0, [2.0, 0.0], [distance, 0.1], side)], 0.1*index)
        latch.observe(side, distance)
    track = result.tracks[0]
    assert (track.left_votes, track.right_votes) == (
        latch.left_votes, latch.right_votes)
    assert track.direction_locked == latch.locked
    assert track.locked_avoidance_side == latch.locked_side == AVOID_RIGHT


def test_two_independent_circles_create_two_tracks():
    result = update(MultiObstacleTracker(config()), [
        observation(1, [1.0, 0.2]), observation(2, [1.6, -0.2], side=RIGHT)], 0)
    assert result.active_track_count == 2
    assert result.new_track_count == 2


def test_capacity_keeps_two_nearest_voting_candidates():
    result = update(MultiObstacleTracker(config()), [
        observation(30, [3.0, 0.0], [3.0, 0.0]),
        observation(10, [1.0, 0.1], [1.0, 0.1]),
        observation(20, [2.0, -0.1], [2.0, -0.1], side=RIGHT)], 0)
    assert result.active_track_count == 2
    assert result.new_track_count == 2
    assert result.capacity_rejected_observation_count == 1
    centers = sorted(round(item.last_distance_to_vehicle, 2)
                     for item in result.tracks)
    assert centers == [1.0, 2.0]


def test_capacity_does_not_accumulate_a_third_track_later():
    tracker = MultiObstacleTracker(config())
    update(tracker, [
        observation(1, [1.0, 0.1]),
        observation(2, [1.5, -0.1], side=RIGHT)], 0)
    result = update(tracker, [
        observation(1, [1.01, 0.1]),
        observation(2, [1.51, -0.1], side=RIGHT),
        observation(3, [2.0, 0.0])], 0.1)
    assert result.active_track_count == 2
    assert result.capacity_rejected_observation_count == 1


def test_same_obstacle_small_odom_jitter_keeps_track_id():
    tracker = MultiObstacleTracker(config())
    first = update(tracker, [observation(1, [2.0, 0.1])], 0)
    second = update(tracker, [observation(4, [2.03, 0.08])], 0.1)
    assert first.tracks[0].track_id == second.tracks[0].track_id
    assert second.tracks[0].seen_count == 2


def test_vehicle_motion_uses_odom_not_base_for_association():
    tracker = MultiObstacleTracker(config())
    first = update(tracker, [observation(1, [2.0, 0.1], [1.5, 0.1])], 0)
    second = update(tracker, [observation(2, [2.01, 0.1], [0.7, -0.2])], 0.1)
    assert first.tracks[0].track_id == second.tracks[0].track_id


def test_two_obstacle_votes_are_independent():
    tracker = MultiObstacleTracker(config())
    a_sides = (LEFT, LEFT, RIGHT, LEFT)
    b_sides = (RIGHT, RIGHT, RIGHT, LEFT)
    for index, (a, b) in enumerate(zip(a_sides, b_sides)):
        result = update(tracker, [
            observation(1, [2.0, 0.2], [1.6, 0.2], a),
            observation(2, [2.6, -0.2], [1.8, -0.2], b)], 0.1*index)
    tracks = by_id(result)
    assert (tracks[1].left_votes, tracks[1].right_votes) == (3, 1)
    assert (tracks[2].left_votes, tracks[2].right_votes) == (1, 3)


def test_independent_freeze_and_simultaneous_freeze():
    tracker = MultiObstacleTracker(config())
    for index in range(7):
        update(tracker, [
            observation(1, [2.0, 0.2], [1.6, 0.2], LEFT),
            observation(2, [2.6, -0.2], [1.7, -0.2], RIGHT)],
            0.05*index)
    result = update(tracker, [
        observation(1, [2.0, 0.2], [0.94, 0.1], LEFT),
        observation(2, [2.6, -0.2], [1.6, -0.2], RIGHT)], 0.4)
    tracks = by_id(result)
    assert tracks[1].direction_locked and not tracks[2].direction_locked
    result = update(tracker, [
        observation(1, [2.0, 0.2], [0.8, 0.1], RIGHT),
        observation(2, [2.6, -0.2], [0.9, -0.1], RIGHT)], 0.5)
    tracks = by_id(result)
    assert tracks[1].direction_locked and tracks[2].direction_locked


def test_latch_independence_survives_opposite_instantaneous_side():
    tracker = MultiObstacleTracker(config())
    for index in range(7):
        update(tracker, [observation(
            1, [2.0, 0.2], [1.6, 0.2], LEFT)], 0.01*index)
    locked = update(tracker, [observation(
        1, [2.0, 0.2], [0.9, 0.2], LEFT)], 0.1).tracks[0]
    result = update(tracker, [
        observation(1, [2.0, 0.2], [0.7, -0.2], RIGHT),
        observation(2, [2.6, -0.2], [1.3, -0.2], RIGHT)], 0.2)
    first = by_id(result)[1]
    assert first.locked_avoidance_side == locked.locked_avoidance_side
    assert first.left_votes == locked.left_votes


def test_narrow_majority_and_exact_tie_are_deterministic():
    tracker = MultiObstacleTracker(config(default_avoidance_side=AVOID_LEFT))
    update(tracker, [observation(1, [2.0, 0.1], [1.2, 0.1])], 0)
    track = tracker.tracks[1]
    track.left_votes, track.right_votes, track.vote_count = 21, 20, 41
    result = update(tracker, [observation(
        1, [2.0, 0.1], [0.9, 0.1], RIGHT)], 0.1)
    assert result.tracks[0].locked_avoidance_side == AVOID_RIGHT

    tracker = MultiObstacleTracker(config(default_avoidance_side=AVOID_LEFT))
    initial = observation(1, [2.0, 0.0], [1.2, 0.0], CENTER)
    update(tracker, [initial], 0)
    track = tracker.tracks[1]
    track.left_votes, track.right_votes, track.vote_count = 20, 20, 40
    result = tracker.update(
        [observation(1, [2.0, 0.0], [0.9, 0.0], CENTER)], 0,
        measurement_key=1, left_clearance=0.4, right_clearance=0.2)
    assert result.tracks[0].locked_avoidance_side == AVOID_LEFT
    tracker = MultiObstacleTracker(config(default_avoidance_side=AVOID_LEFT))
    update(tracker, [initial], 0)
    track = tracker.tracks[1]
    track.left_votes, track.right_votes, track.vote_count = 20, 20, 40
    result = update(tracker, [observation(
        1, [2.0, 0.0], [0.9, 0.0], CENTER)], 0.1)
    assert result.tracks[0].locked_avoidance_side == AVOID_LEFT


def test_one_frame_dropout_preserves_track_votes_and_lock():
    tracker = MultiObstacleTracker(config(retention_age=0.30))
    first = update(tracker, [observation(1, [2.0, 0.1], [1.7, 0.1])], 0)
    missing = update(tracker, [], 0.1)
    returned = update(tracker, [observation(2, [2.02, 0.1], [1.6, 0.1])], 0.2)
    assert missing.active_track_count == 1
    assert returned.tracks[0].track_id == first.tracks[0].track_id
    assert returned.tracks[0].left_votes == 2


def test_expiration_removes_stale_track():
    tracker = MultiObstacleTracker(config(retention_age=0.20))
    update(tracker, [observation(1, [2.0, 0.1])], 0)
    result = update(tracker, [], 0.21)
    assert result.expired_track_count == 1
    assert result.active_track_count == 0


def test_merge_prevention_and_split_prevention():
    tracker = MultiObstacleTracker(config(association_distance=0.20))
    first = update(tracker, [
        observation(1, [1.0, 0.1]), observation(2, [1.5, 0.1])], 0)
    assert first.active_track_count == 2
    second = update(tracker, [
        observation(4, [1.04, 0.08]), observation(5, [1.47, 0.12])], 0.1)
    assert second.active_track_count == 2
    assert all(item.seen_count == 2 for item in second.tracks)


def test_rear_observation_is_associated_and_retained():
    tracker = MultiObstacleTracker(config(association_distance=0.20))
    first = update(tracker, [observation(
        1, [2.0, 0.1], [0.2, 0.1], LEFT)], 0)
    result = update(tracker, [observation(
        2, [2.02, 0.1], [-0.1, 0.1], RIGHT, relevant=False)], 0.1)
    assert result.tracks[0].track_id == first.tracks[0].track_id
    assert result.tracks[0].center_base[0] < 0.0


def test_duplicate_measurement_does_not_duplicate_votes():
    tracker = MultiObstacleTracker(config())
    first = update(tracker, [observation(1, [2.0, 0.1], [1.7, 0.1])], 0, key=7)
    duplicate = update(tracker, [observation(1, [2.0, 0.1], [1.7, 0.1])], 0.1, key=7)
    assert first.tracks[0].left_votes == duplicate.tracks[0].left_votes == 1
    assert duplicate.duplicate_measurement
