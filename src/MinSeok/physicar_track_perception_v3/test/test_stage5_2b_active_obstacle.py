import numpy as np

from physicar_track_perception_v3.active_obstacle import (
    ACTIVE,
    ACTIVE_LOST,
    ACTIVE_LOST_RELEASED,
    ACTIVE_SCAN_MISSING,
    ACTIVE_TERMINATED,
    NEXT_ACTIVE_SELECTED,
    ActiveObstacleLifecycle,
    ActiveTrackView,
    PHYSICAR_BODY_LENGTH,
    PHYSICAR_BODY_WIDTH,
    PHYSICAR_ROBOT_BOUNDING_RADIUS,
    PHYSICAR_STEERING_LIMIT,
    evaluate_lost_active_release,
    evaluate_surface_termination,
    physicar_collision_footprint_vertices,
)
from physicar_track_perception_v3.circle_avoidance import (
    AVOID_LEFT, AVOID_RIGHT, LEFT, RIGHT)
from physicar_track_perception_v3.obstacle_tracks import (
    MultiObstacleTracker, ObstacleObservation, ObstacleTrackConfig)


def view(ident, distance, component=None, relevant=True, observed=True,
         locked=False, side=None, center=None, center_odom=None, votes=7):
    return ActiveTrackView(
        track_id=ident,
        center_base=((distance, 0.0) if center is None else center),
        distance_to_vehicle=distance,
        vote_count=votes,
        direction_locked=locked,
        locked_avoidance_side=side,
        observed_this_frame=observed,
        current_relevant=relevant,
        current_component_id=component,
        center_odom=(center if center_odom is None else center_odom))


def ahead_points(distance=1.0):
    return np.array([[distance, -0.05], [distance, 0.05]])


def behind_clear_points():
    return np.array([[-0.25, -0.05], [-0.22, 0.05]])


def test_robot_bounding_radius_contains_live_collision_vertices():
    radius = PHYSICAR_ROBOT_BOUNDING_RADIUS
    for angle in np.linspace(-PHYSICAR_STEERING_LIMIT,
                             PHYSICAR_STEERING_LIMIT, 1001):
        vertices = physicar_collision_footprint_vertices(angle)
        assert np.max(np.linalg.norm(vertices, axis=1)) <= radius + 1e-12
    assert radius > np.hypot(PHYSICAR_BODY_LENGTH / 2.0,
                             PHYSICAR_BODY_WIDTH / 2.0)
    assert np.isclose(radius, 0.16179830918104007)


def test_surface_distance_is_minimum_raw_point_norm():
    result = evaluate_surface_termination(
        [[0.5, 0.0], [0.3, 0.4], [-0.2, 0.0]], 0.16)
    assert np.isclose(result.distance_surface, 0.2)
    assert np.isclose(result.max_x, 0.5)


def test_ahead_obstacle_never_terminates_from_distance_alone():
    result = evaluate_surface_termination([[0.5, 0.0], [0.6, 0.1]], 0.16)
    assert result.surface_clear
    assert not result.passed
    assert not result.termination


def test_side_overlap_with_any_nonnegative_x_does_not_terminate():
    result = evaluate_surface_termination([[-0.3, 0.2], [0.01, 0.2]], 0.16)
    assert result.surface_clear
    assert not result.passed
    assert not result.termination


def test_fully_behind_but_inside_robot_radius_does_not_terminate():
    result = evaluate_surface_termination([[-0.05, 0.0], [-0.10, 0.0]], 0.16)
    assert result.passed
    assert not result.surface_clear
    assert not result.termination


def test_fully_behind_and_surface_clear_terminates():
    result = evaluate_surface_termination(behind_clear_points(), 0.16)
    assert result.passed
    assert result.surface_clear
    assert result.termination


def test_scan_missing_holds_active_without_done():
    lifecycle = ActiveObstacleLifecycle(0.16)
    lifecycle.update([view(1, 1.0, 11)], {11: ahead_points()})
    result = lifecycle.update(
        [view(1, 1.1, None, relevant=False, observed=False)], {})
    assert result.active_track_id == 1
    assert result.state == ACTIVE
    assert result.termination_hold
    assert not result.termination.termination
    assert result.events == (ACTIVE_SCAN_MISSING,)


def test_active_latch_survives_a_closer_waiting_obstacle():
    lifecycle = ActiveObstacleLifecycle(0.16)
    selected = lifecycle.update(
        [view(1, 1.5, 11, locked=True, side=AVOID_RIGHT),
         view(2, 2.0, 22)],
        {11: ahead_points(1.5), 22: ahead_points(2.0)})
    assert selected.active_track_id == 1
    latched = lifecycle.update(
        [view(1, 1.4, 11), view(2, 0.8, 22)],
        {11: ahead_points(1.4), 22: ahead_points(0.8)})
    assert latched.active_track_id == 1


def test_termination_hands_off_to_next_candidate():
    lifecycle = ActiveObstacleLifecycle(0.16)
    lifecycle.update(
        [view(1, 1.0, 11), view(2, 1.4, 22)],
        {11: ahead_points(), 22: ahead_points(1.4)})
    result = lifecycle.update(
        [view(1, 0.3, 11, relevant=False),
         view(2, 1.2, 22, locked=True, side=AVOID_RIGHT)],
        {11: behind_clear_points(), 22: ahead_points(1.2)})
    assert result.terminated_track_id == 1
    assert result.active_track_id == 2
    assert result.active_direction_locked
    assert result.active_locked_avoidance_side == AVOID_RIGHT
    assert result.events == (ACTIVE_TERMINATED, NEXT_ACTIVE_SELECTED)


def test_multiple_waiting_tracks_use_distance_then_track_id():
    lifecycle = ActiveObstacleLifecycle(0.16)
    first = lifecycle.update(
        [view(3, 1.0, 33), view(2, 1.0, 22), view(1, 1.5, 11)],
        {11: ahead_points(), 22: ahead_points(), 33: ahead_points()})
    assert first.active_track_id == 2
    assert first.candidate_track_ids == (2, 3, 1)


def test_far_unlocked_track_cannot_become_active_even_with_many_votes():
    lifecycle = ActiveObstacleLifecycle(0.16, activation_distance=1.0)
    result = lifecycle.update(
        [view(13, 4.3, 13, votes=100)], {13: ahead_points(4.3)})
    assert result.state != ACTIVE
    assert result.active_track_id is None
    assert result.candidate_track_ids == ()


def test_inside_activation_distance_accepts_zero_vote_track():
    lifecycle = ActiveObstacleLifecycle(0.16, activation_distance=1.0)
    result = lifecycle.update(
        [view(1, 0.9, 11, votes=0)], {11: ahead_points(0.9)})
    assert result.active_track_id == 1


def test_locked_track_is_active_candidate_outside_one_meter():
    lifecycle = ActiveObstacleLifecycle(0.16, activation_distance=1.0)
    result = lifecycle.update(
        [view(2, 1.4, 22, locked=True, side=AVOID_LEFT, votes=3)],
        {22: ahead_points(1.4)})
    assert result.active_track_id == 2
    assert result.active_direction_locked
    assert result.active_locked_avoidance_side == AVOID_LEFT


def test_active_track_expiration_holds_while_last_center_is_path_ahead():
    lifecycle = ActiveObstacleLifecycle(0.16, 1.2)
    lifecycle.update([view(1, 1.0, 11)], {11: ahead_points()})
    path = np.array([[0.1, 0.0], [1.0, 0.0], [1.5, 0.2]])
    lost = lifecycle.update(
        [view(2, 0.8, 22)], {22: ahead_points()},
        current_path=path, lost_center_base=(2.18, 0.1))
    assert lost.state == ACTIVE_LOST
    assert lost.active_track_id == 1
    assert lost.terminated_track_id is None
    assert not lost.termination.termination
    assert lost.lost_release.distance_clear
    assert not lost.lost_release.path_passed
    assert not lost.lost_release.release
    still_lost = lifecycle.update(
        [view(2, 0.7, 22)], {22: ahead_points()},
        current_path=path, lost_center_base=(2.0, 0.1))
    assert still_lost.state == ACTIVE_LOST
    assert still_lost.active_track_id == 1


def test_lost_active_releases_after_path_pass_and_1p2_robot_radii():
    lifecycle = ActiveObstacleLifecycle(0.16, 1.2)
    lifecycle.update([view(1, 1.0, 11)], {11: ahead_points()})
    path = np.array([[0.1, 0.0], [0.6, 0.0], [0.9, 0.3]])
    result = lifecycle.update(
        [view(2, 0.8, 22, locked=True, side=AVOID_RIGHT)],
        {22: ahead_points(0.8)}, current_path=path,
        lost_center_base=(-0.30, 0.0))
    assert result.state == ACTIVE
    assert result.released_lost_track_id == 1
    assert result.terminated_track_id is None
    assert result.active_track_id == 2
    assert result.active_direction_locked
    assert result.active_locked_avoidance_side == AVOID_RIGHT
    assert result.lost_release.path_passed
    assert result.lost_release.distance_clear
    assert result.lost_release.release
    assert result.events == (
        ACTIVE_LOST, ACTIVE_LOST_RELEASED, NEXT_ACTIVE_SELECTED)


def test_lost_active_path_passed_but_inside_1p2_radius_holds():
    lifecycle = ActiveObstacleLifecycle(0.16, 1.2)
    lifecycle.update([view(1, 1.0, 11)], {11: ahead_points()})
    result = lifecycle.update(
        [view(2, 0.8, 22)], {22: ahead_points(0.8)},
        current_path=[[0.1, 0.0], [0.8, 0.0]],
        lost_center_base=(-0.18, 0.0))
    assert result.state == ACTIVE_LOST
    assert result.lost_release.path_passed
    assert not result.lost_release.distance_clear
    assert not result.lost_release.release


def test_lost_active_invalid_path_holds_even_when_distance_is_clear():
    lifecycle = ActiveObstacleLifecycle(0.16, 1.2)
    lifecycle.update([view(1, 1.0, 11)], {11: ahead_points()})
    result = lifecycle.update(
        [view(2, 0.8, 22)], {22: ahead_points(0.8)},
        current_path=[], lost_center_base=(-0.4, 0.0))
    assert result.state == ACTIVE_LOST
    assert result.lost_release.distance_clear
    assert not result.lost_release.path_valid
    assert not result.lost_release.release


def test_lost_release_uses_ordered_path_progress_on_curve():
    path = np.array([
        [0.1, 0.0], [0.5, 0.0], [0.5, 0.4], [0.3, 0.7]])
    behind = evaluate_lost_active_release((-0.30, 0.0), path, 0.16, 1.2)
    ahead = evaluate_lost_active_release((0.50, 0.30), path, 0.16, 1.2)
    assert behind.path_valid and behind.path_passed and behind.release
    assert ahead.path_valid and not ahead.path_passed and not ahead.release


def test_tracker_provenance_addition_preserves_vote_and_rear_association():
    config = ObstacleTrackConfig(association_distance=0.2)
    tracker = MultiObstacleTracker(config)
    first_observation = ObstacleObservation(
        7, np.array([2.0, 0.1]), np.array([1.7, 0.1]), 0.08,
        0.1, LEFT, np.hypot(1.7, 0.1), True)
    first = tracker.update([first_observation], 0.0, measurement_key=1)
    assert first.tracks[0].left_votes == 1
    assert first.tracks[0].current_component_id == 7
    assert first.tracks[0].current_relevant

    rear_observation = ObstacleObservation(
        9, np.array([2.01, 0.1]), np.array([-0.2, 0.1]), 0.08,
        -0.1, RIGHT, np.hypot(0.2, 0.1), False)
    rear = tracker.update([rear_observation], 0.1, measurement_key=2)
    assert rear.tracks[0].track_id == first.tracks[0].track_id
    assert rear.tracks[0].left_votes == 1
    assert rear.tracks[0].current_component_id == 9
    assert not rear.tracks[0].current_relevant
    assert rear.tracks[0].center_base[0] < 0.0


def test_next_track_vote_lock_state_is_not_mutated_by_lifecycle():
    lifecycle = ActiveObstacleLifecycle(0.16)
    waiting = view(2, 1.3, 22, locked=True, side=AVOID_LEFT)
    lifecycle.update(
        [view(1, 1.0, 11), waiting],
        {11: ahead_points(), 22: ahead_points(1.3)})
    result = lifecycle.update(
        [view(1, 0.3, 11, relevant=False), waiting],
        {11: behind_clear_points(), 22: ahead_points(1.3)})
    assert result.active_track_id == 2
    assert result.active_direction_locked == waiting.direction_locked
    assert result.active_locked_avoidance_side == waiting.locked_avoidance_side
