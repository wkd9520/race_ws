"""Temporal LEFT/RIGHT identity and association-supported observations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional

import numpy as np

from .both_geometry import (
    FrameLocalBothGeometry, polyline_left_normals, polyline_tangents,
)
from .components import CanonicalBoundaryCandidate, cumulative_arc_length


@dataclass(frozen=True)
class IdentityConfig:
    initialization_frames: int = 3
    initialization_min_overlap_fraction: float = 0.50
    distance_gate: float = 0.12
    tangent_gate: float = 0.45
    min_overlap_support: float = 0.15
    min_accepted_support: float = 0.20
    max_continuation: float = 0.15
    max_gap: float = 0.075
    conflict_min_side_consistency: float = 0.80
    conflict_max_lateral_residual: float = 0.12


@dataclass(frozen=True)
class ConflictSideEvidence:
    side: str
    valid: bool
    reason: str
    side_consistency: float
    lateral_residual: float
    tangent_consistency: float
    support_length: float
    signed_lateral_median: float
    expected_signed_lateral: float
    side_consistent_support: float
    opposite_side_support: float
    center_crossing: bool


@dataclass(frozen=True)
class ConflictResolution:
    candidate: object
    left: AssociationResult
    right: AssociationResult
    center_reference_valid: bool
    left_evidence: Optional[ConflictSideEvidence]
    right_evidence: Optional[ConflictSideEvidence]
    result: str
    reason: str


@dataclass(frozen=True)
class AssociationResult:
    side: str
    candidate: object
    attempted: bool
    valid: bool
    reason: str
    accepted: Optional[CanonicalBoundaryCandidate]
    mean_distance: float
    overlap_support: float
    tangent_consistency: float
    interval_start_s: float
    interval_end_s: float
    accepted_support: float
    rejected_tail_support: float
    sliding_association_used: bool = False
    continuation_gap: float = float('nan')
    reference_support: float = 0.0
    side_state: str = 'NOT_EVALUATED'
    signed_lateral_median: float = float('nan')
    expected_signed_lateral: float = float('nan')
    lateral_residual: float = float('nan')
    side_consistent_support: float = 0.0
    opposite_side_support: float = 0.0
    center_crossing: bool = False
    reference_update_allowed: bool = False
    side_reason: str = 'not_evaluated'
    association_source: str = 'DIRECT'
    arc_nearest_error: float = float('nan')
    arc_radial_error: float = float('nan')
    arc_tangent_error: float = float('nan')
    arc_compatible_support: float = 0.0
    arc_age_seconds: float = float('nan')


@dataclass(frozen=True)
class TrustedBoundaryState:
    side: str
    initialized: bool
    geometry: Optional[CanonicalBoundaryCandidate]
    timestamp: float
    physical_support: float
    association_reference: Optional[CanonicalBoundaryCandidate] = None
    last_observed: Optional[CanonicalBoundaryCandidate] = None


@dataclass(frozen=True)
class IdentityFrameResult:
    identity_initialized: bool
    initialization_streak: int
    left: Optional[AssociationResult]
    right: Optional[AssociationResult]
    identity_conflict: bool
    both_accepted: bool
    center_result: object
    trusted_update_allowed: bool
    reason: str
    conflict_resolution: Optional[ConflictResolution] = None


class TrustedBoundaryIdentity:
    def __init__(self, both_geometry: FrameLocalBothGeometry,
                 config=IdentityConfig(), arc_prior=None):
        self.both = both_geometry
        self.config = config
        self.initialization_streak = 0
        self._pending_left = None
        self._pending_right = None
        self.left_state = TrustedBoundaryState('LEFT', False, None, 0.0, 0.0)
        self.right_state = TrustedBoundaryState('RIGHT', False, None, 0.0, 0.0)
        self.trusted_center = None
        self.arc_prior = arc_prior
        self._transform_odom_base = None
        self._timestamp = 0.0

    @property
    def initialized(self):
        return self.left_state.initialized and self.right_state.initialized

    def process(self, candidates, timestamp=0.0, transform_odom_base=None):
        candidates = tuple(candidates)
        self._transform_odom_base = transform_odom_base
        self._timestamp = float(timestamp)
        if not self.initialized:
            return self._initialize(candidates, timestamp)
        return self._associate_frame(candidates, timestamp)

    def update_single_observation(self, side, accepted, timestamp):
        """Slide only the short-term reference using an actual observation."""
        previous = self.left_state if side == 'LEFT' else self.right_state
        state = TrustedBoundaryState(
            side, True, previous.geometry, timestamp,
            previous.physical_support, accepted, accepted)
        if side == 'LEFT':
            self.left_state = state
        elif side == 'RIGHT':
            self.right_state = state
        else:
            raise ValueError('unknown boundary side')

    def update_single_association(self, association, timestamp):
        """Apply a guarded short-term update from an actual observation."""
        if not association.reference_update_allowed:
            return False
        self.update_single_observation(
            association.side, association.accepted, timestamp)
        return True

    def _initialize(self, candidates, timestamp):
        frame = self.both.process(candidates)
        if frame.selected_pair is None:
            self._clear_pending()
            return IdentityFrameResult(False, 0, None, None, False, False,
                                       frame, False, 'identity_not_initialized')
        pair = frame.selected_pair
        overlap_fraction = pair.correspondence.overlap_support / max(
            min(pair.first.support_length, pair.second.support_length), 1e-9)
        if overlap_fraction < self.config.initialization_min_overlap_fraction:
            self._clear_pending()
            return IdentityFrameResult(False, 0, None, None, False, False,
                                       frame, False,
                                       'initialization_overlap_insufficient')
        left, right = pair.left, pair.right
        if self._pending_left is None:
            consistent = True
        else:
            left_match = self._associate('LEFT', self._pending_left, left)
            right_match = self._associate('RIGHT', self._pending_right, right)
            consistent = left_match.valid and right_match.valid
        if not consistent:
            self.initialization_streak = 1
        else:
            self.initialization_streak += 1
        self._pending_left, self._pending_right = left, right
        if self.initialization_streak < self.config.initialization_frames:
            return IdentityFrameResult(False, self.initialization_streak,
                                       None, None, False, False, frame, False,
                                       'initialization_pending')
        self.left_state = TrustedBoundaryState(
            'LEFT', True, left, timestamp, left.support_length, left, left)
        self.right_state = TrustedBoundaryState(
            'RIGHT', True, right, timestamp, right.support_length, right, right)
        self.trusted_center = pair.center
        self._observe_arc_actual('LEFT', left)
        self._observe_arc_actual('RIGHT', right)
        return IdentityFrameResult(True, self.initialization_streak,
                                   None, None, False, False, frame, True,
                                   'identity_initialized')

    def _clear_pending(self):
        self.initialization_streak = 0
        self._pending_left = self._pending_right = None

    def _associate_frame(self, candidates, timestamp):
        left_reference = (self.left_state.association_reference
                          or self.left_state.geometry)
        right_reference = (self.right_state.association_reference
                           or self.right_state.geometry)
        left_raw = [self._associate('LEFT', left_reference, item)
                    for item in candidates]
        right_raw = [self._associate('RIGHT', right_reference, item)
                     for item in candidates]
        conflict_ids = {
            id(candidate) for candidate in candidates
            if any(item.valid and item.candidate is candidate for item in left_raw)
            and any(item.valid and item.candidate is candidate for item in right_raw)
        }
        conflict_resolutions = []
        for candidate in candidates:
            if id(candidate) not in conflict_ids:
                continue
            left_match = next(item for item in left_raw
                              if item.candidate is candidate and item.valid)
            right_match = next(item for item in right_raw
                               if item.candidate is candidate and item.valid)
            conflict_resolutions.append(
                self._resolve_conflict(candidate, left_match, right_match))
        resolved = [item for item in conflict_resolutions
                    if item.result in ('RESOLVED_LEFT', 'RESOLVED_RIGHT')]
        unresolved = [item for item in conflict_resolutions
                      if item.result not in ('RESOLVED_LEFT', 'RESOLVED_RIGHT')]
        if len(resolved) == 1 and not unresolved:
            resolution = resolved[0]
            left = (self._mark_side_consistent(
                resolution.left, resolution.left_evidence)
                    if resolution.result == 'RESOLVED_LEFT' else None)
            right = (self._mark_side_consistent(
                resolution.right, resolution.right_evidence)
                     if resolution.result == 'RESOLVED_RIGHT' else None)
            empty = self.both.process(())
            return IdentityFrameResult(
                True, self.initialization_streak, left, right, True, False,
                empty, False, resolution.reason, resolution)
        left_all = [self._validate_identity_side(item) for item in left_raw]
        right_all = [self._validate_identity_side(item) for item in right_raw]
        left = self._best([item for item in left_all if id(item.candidate) not in conflict_ids])
        right = self._best([item for item in right_all if id(item.candidate) not in conflict_ids])
        conflict = bool(conflict_ids)
        if not conflict:
            left, right = self._arc_rescue_missing(
                candidates, left, right)
        if left is None or right is None:
            empty = self.both.process(())
            reason = 'cross_identity_conflict' if conflict else 'no_association'
            resolution = (conflict_resolutions[0]
                          if len(conflict_resolutions) == 1 else None)
            if resolution is not None:
                reason = resolution.reason
            result = IdentityFrameResult(
                True, self.initialization_streak, left, right, conflict,
                False, empty, False, reason, resolution)
            self._observe_frame_actual(left, right)
            if self.arc_prior is not None:
                if left is None and right is None:
                    self.arc_prior.counts['NO_ASSOCIATION'] += 1
            return result
        if left.candidate is right.candidate:
            empty = self.both.process(())
            return IdentityFrameResult(True, self.initialization_streak,
                                       left, right, True, False, empty, False,
                                       'cross_identity_conflict')
        center = self.both.process((left.accepted, right.accepted))
        if center.selected_pair is None:
            result = IdentityFrameResult(
                True, self.initialization_streak, left, right, False, False,
                center, False, 'pair_invalid_after_identity_gate')
            self._observe_frame_actual(left, right)
            return result
        width = float(center.center_path.width_median)
        long_term_update = (
            left.accepted_support >= width
            and right.accepted_support >= width
            and center.center_path.pair_overlap_support >= width
            and left.side_state == 'SIDE_CONSISTENT'
            and right.side_state == 'SIDE_CONSISTENT'
            and left.association_source != 'ARC_ASSISTED'
            and right.association_source != 'ARC_ASSISTED')
        if long_term_update:
            self.left_state = TrustedBoundaryState(
                'LEFT', True, left.accepted, timestamp, left.accepted_support,
                left.accepted, left.accepted)
            self.right_state = TrustedBoundaryState(
                'RIGHT', True, right.accepted, timestamp, right.accepted_support,
                right.accepted, right.accepted)
            self.trusted_center = center.center_path
        else:
            self.left_state = replace(
                self.left_state, timestamp=timestamp,
                association_reference=left.accepted,
                last_observed=left.accepted)
            self.right_state = replace(
                self.right_state, timestamp=timestamp,
                association_reference=right.accepted,
                last_observed=right.accepted)
        result = IdentityFrameResult(
            True, self.initialization_streak, left, right, False, True,
            center, long_term_update,
            ('valid' if long_term_update
             else 'valid_long_term_update_insufficient_support'))
        self._observe_frame_actual(left, right)
        return result

    def _arc_rescue_missing(self, candidates, left, right):
        if self.arc_prior is None or self._transform_odom_base is None:
            return left, right
        used = [value.candidate for value in (left, right)
                if value is not None and value.valid]
        rescued = {}
        for side, current in (('LEFT', left), ('RIGHT', right)):
            if current is not None and current.valid:
                continue
            value = self.arc_prior.rescue(
                side, candidates, self._transform_odom_base,
                self._timestamp, used)
            if value is None or not value.valid:
                continue
            provisional = AssociationResult(
                side, value.candidate, True, True, 'arc_compatible',
                value.accepted, value.nearest_median,
                value.compatible_support,
                float(np.cos(value.tangent_error_median)),
                value.interval_start_s, value.interval_end_s,
                value.compatible_support, value.rejected_tail_support,
                False, float('nan'), 0.0,
                association_source='ARC_ASSISTED',
                arc_nearest_error=value.nearest_median,
                arc_radial_error=value.radial_median,
                arc_tangent_error=value.tangent_error_median,
                arc_compatible_support=value.compatible_support,
                arc_age_seconds=value.age_seconds)
            checked = self._validate_identity_side(provisional)
            if checked.valid:
                rescued[side] = checked
                used.append(value.candidate)
            else:
                self.arc_prior.counts['ARC_RESCUE_REJECTED'] += 1
        left = rescued.get('LEFT', left)
        right = rescued.get('RIGHT', right)
        if (left is not None and right is not None
                and left.candidate is right.candidate):
            self.arc_prior.counts['ARC_RESCUE_REJECTED'] += 1
            if left.association_source == 'ARC_ASSISTED':
                left = None
            if right.association_source == 'ARC_ASSISTED':
                right = None
        for value in rescued.values():
            if value is left or value is right:
                self.arc_prior.counts['ARC_RESCUE_SUCCESS'] += 1
        return left, right

    def _observe_frame_actual(self, left, right):
        for side, association in (('LEFT', left), ('RIGHT', right)):
            if association is None or not association.valid:
                continue
            if association.association_source == 'ARC_ASSISTED':
                continue
            if association.sliding_association_used:
                if self.arc_prior is not None:
                    self.arc_prior.counts['SLIDING_SUCCESS'] += 1
            else:
                if self.arc_prior is not None:
                    self.arc_prior.counts['DIRECT_SUCCESS'] += 1
            self._observe_arc_actual(side, association.accepted)

    def _observe_arc_actual(self, side, accepted):
        if self.arc_prior is None or self._transform_odom_base is None:
            return
        self.arc_prior.observe_actual(
            side, accepted, self._transform_odom_base, self._timestamp,
            actual_observed=True)

    def _resolve_conflict(self, candidate, left, right):
        if self.trusted_center is None:
            return ConflictResolution(
                candidate, left, right, False, None, None, 'UNSUPPORTED',
                'cross_identity_center_reference_unavailable')
        left_evidence = self._corridor_evidence('LEFT', left)
        right_evidence = self._corridor_evidence('RIGHT', right)
        if left_evidence.valid and not right_evidence.valid:
            return ConflictResolution(
                candidate, left, right, True, left_evidence, right_evidence,
                'RESOLVED_LEFT', 'cross_identity_resolved_left')
        if right_evidence.valid and not left_evidence.valid:
            return ConflictResolution(
                candidate, left, right, True, left_evidence, right_evidence,
                'RESOLVED_RIGHT', 'cross_identity_resolved_right')
        if left_evidence.valid and right_evidence.valid:
            result, reason = 'AMBIGUOUS_CONFLICT', 'cross_identity_ambiguous'
        else:
            result = 'UNSUPPORTED'
            reason = 'cross_identity_insufficient_side_evidence'
        return ConflictResolution(
            candidate, left, right, True, left_evidence, right_evidence,
            result, reason)

    def _corridor_evidence(self, side, association):
        points = np.asarray(association.accepted.canonical_points)
        center_points = np.asarray(self.trusted_center.points)
        try:
            center_tangent = polyline_tangents(center_points)
            current_tangent = polyline_tangents(points)
            center_normals = polyline_left_normals(center_points)
        except ValueError:
            return ConflictSideEvidence(
                side, False, 'geometry_invalid', 0.0, float('inf'), 0.0,
                association.accepted_support, float('nan'), float('nan'),
                0.0, 0.0, False)
        distance = np.linalg.norm(
            points[:, None, :] - center_points[None, :, :], axis=2)
        nearest = np.argmin(distance, axis=1)
        delta = points-center_points[nearest]
        lateral = np.einsum('ij,ij->i', delta, center_normals[nearest])
        expected_sign = 1.0 if side == 'LEFT' else -1.0
        side_consistency = float(np.mean(expected_sign*lateral > 0.0))
        expected_distance = 0.5*float(self.trusted_center.width_median)
        expected_signed = expected_sign*expected_distance
        lateral_residual = float(np.median(np.abs(
            lateral-expected_signed)))
        tangent = np.abs(np.einsum(
            'ij,ij->i', current_tangent, center_tangent[nearest]))
        tangent_consistency = float(np.median(tangent))
        valid = (
            side_consistency >= self.config.conflict_min_side_consistency
            and lateral_residual <= self.config.conflict_max_lateral_residual
            and tangent_consistency >= np.cos(self.config.tangent_gate)
            and association.accepted_support >= self.config.min_accepted_support
        )
        if side_consistency < self.config.conflict_min_side_consistency:
            reason = 'side_inconsistent'
        elif lateral_residual > self.config.conflict_max_lateral_residual:
            reason = 'lateral_residual_mismatch'
        elif tangent_consistency < np.cos(self.config.tangent_gate):
            reason = 'tangent_mismatch'
        elif association.accepted_support < self.config.min_accepted_support:
            reason = 'physical_support_short'
        else:
            reason = 'supported'
        spacing = (association.accepted_support/max(len(points)-1, 1))
        consistent_support = float(np.count_nonzero(
            expected_sign*lateral > 0.0)*spacing)
        opposite_support = float(np.count_nonzero(
            expected_sign*lateral < 0.0)*spacing)
        center_crossing = bool(
            consistent_support > self.config.max_gap
            and opposite_support > self.config.max_gap)
        if center_crossing:
            valid = False
            reason = 'center_crossing'
        return ConflictSideEvidence(
            side, valid, reason, side_consistency, lateral_residual,
            tangent_consistency, association.accepted_support,
            float(np.median(lateral)), expected_signed,
            consistent_support, opposite_support, center_crossing)

    def _validate_identity_side(self, association):
        if not association.valid:
            return association
        if self.trusted_center is None:
            return replace(
                association, valid=False, reason='corridor_reference_unavailable',
                side_state='SIDE_AMBIGUOUS',
                side_reason='corridor_reference_unavailable')
        evidence = self._corridor_evidence(association.side, association)
        if evidence.center_crossing:
            state, reason = 'SIDE_AMBIGUOUS', 'association_rejected_center_crossing'
        elif evidence.side_consistency <= 1.0-self.config.conflict_min_side_consistency:
            state, reason = 'SIDE_OPPOSITE', 'association_rejected_wrong_side'
        elif evidence.valid:
            state, reason = 'SIDE_CONSISTENT', 'association_valid_side_consistent'
        else:
            state, reason = 'SIDE_AMBIGUOUS', 'association_valid_side_ambiguous'
        valid = state == 'SIDE_CONSISTENT'
        return replace(
            association, valid=valid, reason=reason, side_state=state,
            signed_lateral_median=evidence.signed_lateral_median,
            expected_signed_lateral=evidence.expected_signed_lateral,
            lateral_residual=evidence.lateral_residual,
            side_consistent_support=evidence.side_consistent_support,
            opposite_side_support=evidence.opposite_side_support,
            center_crossing=evidence.center_crossing,
            reference_update_allowed=valid, side_reason=reason)

    @staticmethod
    def _mark_side_consistent(association, evidence):
        return replace(
            association, side_state='SIDE_CONSISTENT',
            signed_lateral_median=evidence.signed_lateral_median,
            expected_signed_lateral=evidence.expected_signed_lateral,
            lateral_residual=evidence.lateral_residual,
            side_consistent_support=evidence.side_consistent_support,
            opposite_side_support=evidence.opposite_side_support,
            center_crossing=evidence.center_crossing,
            reference_update_allowed=True,
            side_reason='association_valid_side_consistent')

    @staticmethod
    def _best(results):
        valid = [item for item in results if item.valid]
        if not valid:
            return None
        return max(valid, key=lambda item: (
            item.overlap_support, item.accepted_support,
            -item.mean_distance, item.tangent_consistency,
        ))

    def _associate(self, side, trusted, current):
        points = np.asarray(current.canonical_points)
        trusted_points = np.asarray(trusted.canonical_points)
        try:
            current_tangent = polyline_tangents(points)
            trusted_tangent = polyline_tangents(trusted_points)
        except ValueError:
            return self._invalid(side, current, 'tangent_mismatch')
        distance = np.linalg.norm(
            points[:, None, :] - trusted_points[None, :, :], axis=2)
        nearest = np.argmin(distance, axis=1)
        nearest_distance = distance[np.arange(len(points)), nearest]
        alignment = np.einsum(
            'ij,ij->i', current_tangent, trusted_tangent[nearest])
        supported = ((nearest_distance <= self.config.distance_gate)
                     & (alignment >= np.cos(self.config.tangent_gate)))
        runs = self._runs(current.canonical_s, supported)
        if not runs:
            continued = self._endpoint_continuation(
                side, trusted, current, current_tangent, trusted_tangent)
            if continued is not None:
                return continued
            reason = ('distance_mismatch' if np.count_nonzero(
                nearest_distance <= self.config.distance_gate) == 0
                else 'tangent_mismatch')
            return self._invalid(side, current, reason)
        start, end = max(runs, key=lambda item:
                         current.canonical_s[item[1]]-current.canonical_s[item[0]])
        overlap = float(current.canonical_s[end]-current.canonical_s[start])
        if overlap < self.config.min_overlap_support:
            continued = self._endpoint_continuation(
                side, trusted, current, current_tangent, trusted_tangent)
            if continued is not None:
                return continued
            return self._invalid(side, current, 'insufficient_overlap')
        first, last = self._extend(points, current_tangent, start, end)
        accepted_points = points[first:last+1]
        accepted_s = cumulative_arc_length(accepted_points)
        accepted_support = float(accepted_s[-1])
        if accepted_support < self.config.min_accepted_support:
            return self._invalid(side, current, 'insufficient_accepted_support')
        accepted = self._subcandidate(current, accepted_points, accepted_s)
        rejected = max(0.0, current.support_length-accepted_support)
        return AssociationResult(
            side, current, True, True, 'valid', accepted,
            float(np.mean(nearest_distance[start:end+1])), overlap,
            float(np.mean(alignment[start:end+1])),
            float(current.canonical_s[first]), float(current.canonical_s[last]),
            accepted_support, rejected, False, float('nan'),
            float(trusted.support_length),
        )

    def _runs(self, s, mask):
        indices = np.flatnonzero(mask)
        if not len(indices):
            return []
        runs, start, previous = [], int(indices[0]), int(indices[0])
        for value in indices[1:]:
            value = int(value)
            if value != previous+1 or s[value]-s[previous] > self.config.max_gap:
                runs.append((start, previous))
                start = value
            previous = value
        runs.append((start, previous))
        return runs

    def _extend(self, points, tangents, start, end):
        first, last = start, end
        while first > 0:
            if np.linalg.norm(points[first]-points[first-1]) > self.config.max_gap:
                break
            if np.dot(tangents[first-1], tangents[first]) < np.cos(self.config.tangent_gate):
                break
            first -= 1
        while last+1 < len(points):
            if np.linalg.norm(points[last+1]-points[last]) > self.config.max_gap:
                break
            if np.dot(tangents[last+1], tangents[last]) < np.cos(self.config.tangent_gate):
                break
            last += 1
        return first, last

    def _endpoint_continuation(self, side, trusted, current,
                               current_tangent, trusted_tangent):
        current_points = np.asarray(current.canonical_points)
        trusted_points = np.asarray(trusted.canonical_points)
        pairs = []
        for current_index in (0, len(current_points)-1):
            for trusted_index in (0, len(trusted_points)-1):
                gap = float(np.linalg.norm(
                    current_points[current_index]-trusted_points[trusted_index]))
                alignment = float(np.dot(
                    current_tangent[current_index], trusted_tangent[trusted_index]))
                pairs.append((gap, -alignment, current_index, trusted_index))
        gap, negative_alignment, current_index, _ = min(pairs)
        alignment = -negative_alignment
        if (gap > self.config.max_continuation
                or alignment < np.cos(self.config.tangent_gate)):
            return None
        local_alignment = np.einsum(
            'ij,ij->i', current_tangent[:-1], current_tangent[1:])
        if (len(local_alignment)
                and np.any(local_alignment < np.cos(self.config.tangent_gate))):
            return None
        s = cumulative_arc_length(current_points)
        accepted = self._subcandidate(current, current_points, s)
        provisional = AssociationResult(
            side, current, True, True, 'sliding_continuation', accepted,
            gap, 0.0, alignment, 0.0, float(s[-1]), float(s[-1]),
            0.0, True, gap, float(trusted.support_length),
            association_source='SLIDING')
        evidence = self._corridor_evidence(side, provisional)
        if not evidence.valid:
            return None
        return provisional

    @staticmethod
    def _subcandidate(source, points, s):
        spacing = np.diff(s)
        return CanonicalBoundaryCandidate(
            component_id=source.component_id, color=source.color,
            raw_ordered_points=points, raw_s=s,
            canonical_points=points, canonical_s=s,
            support_length=float(s[-1]), raw_point_count=len(points),
            canonical_point_count=len(points),
            raw_spacing_min=float(np.min(spacing)),
            raw_spacing_median=float(np.median(spacing)),
            raw_spacing_max=float(np.max(spacing)),
            canonical_spacing=source.canonical_spacing,
            near_endpoint=points[0], far_endpoint=points[-1],
        )

    @staticmethod
    def _invalid(side, candidate, reason):
        return AssociationResult(side, candidate, True, False, reason, None,
                                 float('nan'), 0.0, 0.0,
                                 float('nan'), float('nan'), 0.0,
                                 float(candidate.support_length), False,
                                 float('nan'), 0.0)
