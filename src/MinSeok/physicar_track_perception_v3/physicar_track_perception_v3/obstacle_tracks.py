"""ROS-independent Stage 5.2A multi-obstacle vote/lock state.

Circle centers are associated in ``odom``.  ``base_footprint`` coordinates
are retained only for the current distance/lane diagnostic.  This is a short
bounded local identity mechanism, not an obstacle map or motion tracker.
"""

from dataclasses import dataclass
import math

import numpy as np

from .circle_avoidance import (
    AVOID_LEFT, AVOID_RIGHT, CENTER, LEFT, RIGHT, VOTING)


@dataclass(frozen=True)
class ObstacleTrackConfig:
    association_distance: float = 0.12
    retention_age: float = 0.50
    direction_freeze_distance: float = 1.5
    default_avoidance_side: str = AVOID_LEFT
    max_voting_tracks: int = 2

    def __post_init__(self):
        values = np.asarray((self.association_distance, self.retention_age,
                             self.direction_freeze_distance), dtype=float)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError('track distances/retention must be finite and > 0')
        if self.default_avoidance_side not in (AVOID_LEFT, AVOID_RIGHT):
            raise ValueError('default_avoidance_side is invalid')
        if (not isinstance(self.max_voting_tracks, int)
                or self.max_voting_tracks <= 0):
            raise ValueError('max_voting_tracks must be a positive integer')


@dataclass(frozen=True)
class ObstacleObservation:
    observation_id: int
    center_odom: np.ndarray
    center_base: np.ndarray
    radius: float
    signed_lateral: float
    instantaneous_side: str
    distance_to_vehicle: float
    relevant: bool = True

    def __post_init__(self):
        odom = np.asarray(self.center_odom, dtype=np.float64).reshape(2)
        base = np.asarray(self.center_base, dtype=np.float64).reshape(2)
        values = np.r_[odom, base, self.radius, self.signed_lateral,
                       self.distance_to_vehicle]
        if not np.all(np.isfinite(values)):
            raise ValueError('obstacle observation must be finite')
        if self.radius <= 0.0 or self.distance_to_vehicle < 0.0:
            raise ValueError('obstacle radius/distance is invalid')
        if self.instantaneous_side not in (LEFT, RIGHT, CENTER):
            raise ValueError('instantaneous_side is invalid')
        object.__setattr__(self, 'center_odom', odom.copy())
        object.__setattr__(self, 'center_base', base.copy())


@dataclass
class ObstacleTrack:
    track_id: int
    center_odom: np.ndarray
    last_center_base: np.ndarray
    radius: float
    last_seen_stamp: float
    seen_count: int = 0
    left_votes: int = 0
    right_votes: int = 0
    vote_count: int = 0
    direction_locked: bool = False
    locked_avoidance_side: str | None = None
    last_distance_to_vehicle: float = math.inf
    last_signed_lateral: float | None = None
    last_instantaneous_side: str | None = None
    last_association_distance: float | None = None
    lock_reason: str | None = None
    lock_source: str | None = None
    last_observation_id: int | None = None
    last_observation_relevant: bool = False


@dataclass(frozen=True)
class ObstacleTrackSnapshot:
    track_id: int
    center_odom: tuple
    center_base: tuple
    radius: float
    last_seen_age: float
    seen_count: int
    left_votes: int
    right_votes: int
    vote_count: int
    direction_locked: bool
    locked_avoidance_side: str | None
    last_distance_to_vehicle: float
    signed_lateral: float | None
    instantaneous_side: str | None
    association_distance: float | None
    lock_reason: str | None
    lock_source: str | None
    observed_this_frame: bool
    current_component_id: int | None
    current_relevant: bool


@dataclass(frozen=True)
class MultiTrackResult:
    timestamp: float
    measurement_key: object
    duplicate_measurement: bool
    observation_count: int
    relevant_observation_count: int
    active_track_count: int
    new_track_count: int
    associated_track_count: int
    unmatched_observation_count: int
    capacity_rejected_observation_count: int
    expired_track_count: int
    expired_track_ids: tuple
    tracks: tuple


class MultiObstacleTracker:
    """Greedy one-to-one odom association with independent vote latches."""

    def __init__(self, config=ObstacleTrackConfig()):
        self.config = config
        self.tracks = {}
        self.next_track_id = 1
        self.last_measurement_key = None

    def reset(self):
        self.tracks.clear()
        self.next_track_id = 1
        self.last_measurement_key = None

    def _tie_break(self, left_clearance, right_clearance):
        if (left_clearance is not None and right_clearance is not None
                and np.isfinite(left_clearance)
                and np.isfinite(right_clearance)
                and not np.isclose(left_clearance, right_clearance)):
            return ((AVOID_LEFT if left_clearance > right_clearance
                     else AVOID_RIGHT), 'TIE_TRACK_CLEARANCE')
        return (self.config.default_avoidance_side,
                'TIE_DETERMINISTIC_DEFAULT')

    def _observe_vote(self, track, observation,
                      left_clearance=None, right_clearance=None):
        if track.direction_locked or not observation.relevant:
            return
        if observation.distance_to_vehicle > self.config.direction_freeze_distance:
            self._add_vote(track, observation.instantaneous_side)
            return
        # Freeze uses whatever approach evidence already exists.  A track
        # first observed inside the activation distance intentionally remains
        # unlocked with zero votes so current-WHITE proximity can decide it.
        if track.vote_count == 0:
            return
        self._lock_from_votes(
            track, 'NORMAL_VOTE', left_clearance, right_clearance)

    @staticmethod
    def _add_vote(track, side):
        if side == LEFT:
            track.left_votes += 1
        elif side == RIGHT:
            track.right_votes += 1
        track.vote_count = track.left_votes + track.right_votes

    def _lock_from_votes(self, track, source,
                         left_clearance=None, right_clearance=None):
        if track.direction_locked:
            return False
        if track.left_votes > track.right_votes:
            track.locked_avoidance_side = AVOID_RIGHT
            track.lock_reason = 'LEFT_OBSTACLE_VOTE_MAJORITY'
        elif track.right_votes > track.left_votes:
            track.locked_avoidance_side = AVOID_LEFT
            track.lock_reason = 'RIGHT_OBSTACLE_VOTE_MAJORITY'
        else:
            track.locked_avoidance_side, track.lock_reason = self._tie_break(
                left_clearance, right_clearance)
        track.direction_locked = True
        track.lock_source = str(source)
        return True

    def lock_track_from_votes(self, track_id, source,
                              left_clearance=None, right_clearance=None):
        """Latch one track from its accumulated votes; never relock it."""
        track = self.tracks.get(int(track_id))
        if track is None:
            return False
        return self._lock_from_votes(
            track, source, left_clearance, right_clearance)

    def lock_track_side(self, track_id, side, source):
        """Latch an externally recovered deterministic avoidance side."""
        track = self.tracks.get(int(track_id))
        if track is None or track.direction_locked:
            return False
        if side not in (AVOID_LEFT, AVOID_RIGHT):
            raise ValueError('avoidance side is invalid')
        track.locked_avoidance_side = side
        track.lock_reason = str(source)
        track.lock_source = str(source)
        track.direction_locked = True
        return True

    def _update_track(self, track, observation, stamp,
                      association_distance, left_clearance, right_clearance):
        track.center_odom = observation.center_odom.copy()
        track.last_center_base = observation.center_base.copy()
        track.radius = float(observation.radius)
        track.last_seen_stamp = float(stamp)
        track.seen_count += 1
        track.last_distance_to_vehicle = float(
            observation.distance_to_vehicle)
        track.last_signed_lateral = float(observation.signed_lateral)
        track.last_instantaneous_side = observation.instantaneous_side
        track.last_association_distance = association_distance
        track.last_observation_id = int(observation.observation_id)
        track.last_observation_relevant = bool(observation.relevant)
        self._observe_vote(track, observation, left_clearance, right_clearance)

    def _new_track(self, observation, stamp,
                   left_clearance, right_clearance):
        ident = self.next_track_id
        self.next_track_id += 1
        track = ObstacleTrack(
            track_id=ident,
            center_odom=observation.center_odom.copy(),
            last_center_base=observation.center_base.copy(),
            radius=float(observation.radius),
            last_seen_stamp=float(stamp))
        self._update_track(track, observation, stamp, None,
                           left_clearance, right_clearance)
        self.tracks[ident] = track
        return track

    def _snapshots(self, stamp, observed_ids):
        output = []
        for ident in sorted(self.tracks):
            track = self.tracks[ident]
            observed = ident in observed_ids
            output.append(ObstacleTrackSnapshot(
                track_id=track.track_id,
                center_odom=tuple(float(x) for x in track.center_odom),
                center_base=tuple(float(x) for x in track.last_center_base),
                radius=float(track.radius),
                last_seen_age=max(0.0, float(stamp)-track.last_seen_stamp),
                seen_count=track.seen_count,
                left_votes=track.left_votes,
                right_votes=track.right_votes,
                vote_count=track.vote_count,
                direction_locked=track.direction_locked,
                locked_avoidance_side=track.locked_avoidance_side,
                last_distance_to_vehicle=track.last_distance_to_vehicle,
                signed_lateral=track.last_signed_lateral,
                instantaneous_side=track.last_instantaneous_side,
                association_distance=track.last_association_distance,
                lock_reason=track.lock_reason,
                lock_source=track.lock_source,
                observed_this_frame=observed,
                current_component_id=(track.last_observation_id
                                      if observed else None),
                current_relevant=(track.last_observation_relevant
                                  if observed else False)))
        return tuple(output)

    def update(self, observations, stamp, measurement_key=None,
               left_clearance=None, right_clearance=None):
        now = float(stamp)
        if not np.isfinite(now):
            raise ValueError('stamp must be finite')
        values = tuple(observations)
        ids = [item.observation_id for item in values]
        if len(set(ids)) != len(ids):
            raise ValueError('observation_id must be unique within a frame')
        if (measurement_key is not None
                and measurement_key == self.last_measurement_key):
            return MultiTrackResult(
                timestamp=now,
                measurement_key=measurement_key,
                duplicate_measurement=True,
                observation_count=len(values),
                relevant_observation_count=sum(
                    item.relevant for item in values),
                active_track_count=len(self.tracks),
                new_track_count=0,
                associated_track_count=0,
                unmatched_observation_count=len(values),
                capacity_rejected_observation_count=0,
                expired_track_count=0,
                expired_track_ids=(),
                tracks=self._snapshots(now, set()))
        self.last_measurement_key = measurement_key

        expired = tuple(sorted(
            ident for ident, track in self.tracks.items()
            if now-track.last_seen_stamp > self.config.retention_age))
        for ident in expired:
            del self.tracks[ident]

        pairs = []
        for observation_index, observation in enumerate(values):
            for ident, track in self.tracks.items():
                distance = float(np.linalg.norm(
                    observation.center_odom-track.center_odom))
                if distance <= self.config.association_distance:
                    pairs.append((distance, observation_index, ident))
        pairs.sort(key=lambda item: (item[0], item[1], item[2]))
        used_observations = set()
        used_tracks = set()
        for distance, observation_index, ident in pairs:
            if observation_index in used_observations or ident in used_tracks:
                continue
            self._update_track(
                self.tracks[ident], values[observation_index], now, distance,
                left_clearance, right_clearance)
            used_observations.add(observation_index)
            used_tracks.add(ident)

        new_count = 0
        new_candidates = sorted(
            (index for index, observation in enumerate(values)
             if index not in used_observations and observation.relevant),
            key=lambda index: (
                values[index].distance_to_vehicle,
                values[index].observation_id))
        available_slots = max(
            0, self.config.max_voting_tracks-len(self.tracks))
        for index in new_candidates[:available_slots]:
            observation = values[index]
            track = self._new_track(observation, now,
                                    left_clearance, right_clearance)
            used_observations.add(index)
            used_tracks.add(track.track_id)
            new_count += 1
        capacity_rejected = sum(
            index not in used_observations and observation.relevant
            for index, observation in enumerate(values))

        return MultiTrackResult(
            timestamp=now,
            measurement_key=measurement_key,
            duplicate_measurement=False,
            observation_count=len(values),
            relevant_observation_count=sum(item.relevant for item in values),
            active_track_count=len(self.tracks),
            new_track_count=new_count,
            associated_track_count=len(used_tracks)-new_count,
            unmatched_observation_count=len(values)-len(used_observations),
            capacity_rejected_observation_count=capacity_rejected,
            expired_track_count=len(expired),
            expired_track_ids=expired,
            tracks=self._snapshots(now, used_tracks))
