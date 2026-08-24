"""Bounded odom-frame circle prior used only to rescue current observations."""

from dataclasses import dataclass
import math

import numpy as np

from .arc_shadow import fit_circle_shadow, inverse_transform, transform_points
from .both_geometry import polyline_tangents
from .components import CanonicalBoundaryCandidate, cumulative_arc_length


@dataclass(frozen=True)
class ArcPriorConfig:
    # R2 OBSERVED: good short-visible median errors were about 0.036/0.029 m;
    # stale straight errors were about 1.10/0.95 m. These conservative initial
    # production values remain subject to repeated runtime validation.
    max_nearest_distance: float = 0.05
    max_radial_residual: float = 0.05
    max_tangent_error: float = 0.35
    min_compatible_support: float = 0.20
    max_point_gap: float = 0.075
    confirmation_frames: int = 3
    max_confirmation_radius_fraction: float = 0.05
    max_confirmation_center_spread: float = 0.10
    # Compatibility is the primary lifetime gate. This is only a final safety
    # bound and is deliberately longer than the R2-validated 15-frame horizon.
    max_age_seconds: float = 15.0


@dataclass(frozen=True)
class ArcPriorMemory:
    side: str
    center_odom: np.ndarray
    radius: float
    arc_points_odom: np.ndarray
    confirmed_time: float
    last_actual_time: float
    confirmation_center_spread: float
    confirmation_radius_std: float


@dataclass(frozen=True)
class ArcRescue:
    side: str
    valid: bool
    reason: str
    candidate: object
    accepted: CanonicalBoundaryCandidate | None
    nearest_median: float
    radial_median: float
    tangent_error_median: float
    compatible_support: float
    interval_start_s: float
    interval_end_s: float
    rejected_tail_support: float
    age_seconds: float


class OdomArcPrior:
    """Identity-owned memory; never creates identity without trusted state."""

    def __init__(self, config=ArcPriorConfig()):
        self.config = config
        self.pending = {'LEFT': [], 'RIGHT': []}
        self.memory = {'LEFT': None, 'RIGHT': None}
        self.last_hypothesis = {'LEFT': None, 'RIGHT': None}
        self.counts = {
            'DIRECT_SUCCESS': 0,
            'SLIDING_SUCCESS': 0,
            'ARC_RESCUE_SUCCESS': 0,
            'ARC_RESCUE_REJECTED': 0,
            'NO_ASSOCIATION': 0,
        }

    def observe_actual(self, side, accepted, transform_odom_base, timestamp,
                       *, actual_observed):
        """Update hypotheses from an already trusted actual observation only."""
        if not actual_observed:
            return None
        if accepted is None or transform_odom_base is None:
            self.pending[side] = []
            return None
        hypothesis = fit_circle_shadow(
            accepted.canonical_points, transform_odom_base)
        self.last_hypothesis[side] = hypothesis
        if hypothesis is None or not hypothesis.strong:
            self.pending[side] = []
            return hypothesis
        pending = self.pending[side]
        if pending:
            radius = float(np.median([item.radius for item in pending]))
            center = np.median(
                np.asarray([item.center_odom for item in pending]), axis=0)
            consistent = (
                abs(hypothesis.radius-radius)/max(radius, 1e-9)
                <= self.config.max_confirmation_radius_fraction
                and np.linalg.norm(hypothesis.center_odom-center)
                <= self.config.max_confirmation_center_spread)
            if not consistent:
                pending = []
        pending = (pending+[hypothesis])[-self.config.confirmation_frames:]
        self.pending[side] = pending
        if len(pending) < self.config.confirmation_frames:
            return hypothesis
        centers = np.asarray([item.center_odom for item in pending])
        radii = np.asarray([item.radius for item in pending])
        center = np.median(centers, axis=0)
        self.memory[side] = ArcPriorMemory(
            side, center, float(np.median(radii)),
            pending[-1].inlier_points_odom.copy(), timestamp, timestamp,
            float(np.max(np.linalg.norm(centers-center, axis=1))),
            float(np.std(radii)))
        return hypothesis

    def rescue(self, side, candidates, transform_odom_base, timestamp,
               excluded_candidates=()):
        memory = self.memory[side]
        if memory is None:
            return None
        age = float(timestamp-memory.last_actual_time)
        if age < 0.0 or age > self.config.max_age_seconds:
            self.counts['ARC_RESCUE_REJECTED'] += 1
            return ArcRescue(side, False, 'arc_prior_age_limit', None, None,
                             math.inf, math.inf, math.inf, 0.0,
                             math.nan, math.nan, 0.0, age)
        if transform_odom_base is None:
            self.counts['ARC_RESCUE_REJECTED'] += 1
            return ArcRescue(side, False, 'arc_prior_transform_unavailable',
                             None, None, math.inf, math.inf, math.inf, 0.0,
                             math.nan, math.nan, 0.0, age)
        transform_base_odom = inverse_transform(transform_odom_base)
        center = transform_points(
            np.asarray([memory.center_odom]), transform_base_odom)[0]
        predicted_arc = transform_points(
            memory.arc_points_odom, transform_base_odom)
        excluded = {id(item) for item in excluded_candidates if item is not None}
        attempts = [self._compatible_interval(
            side, item, center, memory.radius, predicted_arc, age)
            for item in candidates if id(item) not in excluded]
        valid = [item for item in attempts if item.valid]
        if not valid:
            self.counts['ARC_RESCUE_REJECTED'] += 1
            if not attempts:
                return ArcRescue(side, False, 'arc_prior_no_candidate', None,
                                 None, math.inf, math.inf, math.inf, 0.0,
                                 math.nan, math.nan, 0.0, age)
            return min(attempts, key=lambda item: (
                item.radial_median, item.nearest_median))
        return max(valid, key=lambda item: (
            item.compatible_support, -item.radial_median,
            -item.nearest_median, -item.tangent_error_median))

    def _compatible_interval(self, side, candidate, center, radius,
                             predicted_arc, age):
        points = np.asarray(candidate.canonical_points, dtype=np.float64)
        if len(points) < 3 or len(predicted_arc) < 2:
            return self._invalid(side, candidate, 'arc_geometry_short', age)
        radial = np.abs(np.linalg.norm(points-center, axis=1)-radius)
        distance = np.linalg.norm(
            points[:, None, :]-predicted_arc[None, :, :], axis=2)
        nearest = np.min(distance, axis=1)
        try:
            tangent = polyline_tangents(points)
        except ValueError:
            return self._invalid(side, candidate, 'arc_tangent_invalid', age)
        radial_vector = points-center
        radial_vector /= np.maximum(
            np.linalg.norm(radial_vector, axis=1, keepdims=True), 1e-12)
        circle_tangent = np.column_stack(
            (-radial_vector[:, 1], radial_vector[:, 0]))
        alignment = np.clip(np.abs(np.einsum(
            'ij,ij->i', tangent, circle_tangent)), 0.0, 1.0)
        tangent_error = np.arccos(alignment)
        mask = (
            (nearest <= self.config.max_nearest_distance)
            & (radial <= self.config.max_radial_residual)
            & (tangent_error <= self.config.max_tangent_error))
        runs = self._runs(candidate.canonical_s, mask)
        if not runs:
            reason = self._failure_reason(nearest, radial, tangent_error)
            return self._invalid(
                side, candidate, reason, age, nearest, radial, tangent_error)
        start, end = max(runs, key=lambda value:
                         candidate.canonical_s[value[1]]
                         - candidate.canonical_s[value[0]])
        support = float(candidate.canonical_s[end]-candidate.canonical_s[start])
        if support < self.config.min_compatible_support:
            return self._invalid(
                side, candidate, 'arc_insufficient_compatible_support', age,
                nearest[start:end+1], radial[start:end+1],
                tangent_error[start:end+1], support)
        accepted_points = points[start:end+1]
        accepted_s = cumulative_arc_length(accepted_points)
        accepted = self._subcandidate(candidate, accepted_points, accepted_s)
        return ArcRescue(
            side, True, 'arc_compatible', candidate, accepted,
            float(np.median(nearest[start:end+1])),
            float(np.median(radial[start:end+1])),
            float(np.median(tangent_error[start:end+1])), support,
            float(candidate.canonical_s[start]),
            float(candidate.canonical_s[end]),
            max(0.0, float(candidate.support_length)-support), age)

    def _failure_reason(self, nearest, radial, tangent):
        if float(np.median(nearest)) > self.config.max_nearest_distance:
            return 'arc_nearest_mismatch'
        if float(np.median(radial)) > self.config.max_radial_residual:
            return 'arc_radial_mismatch'
        if float(np.median(tangent)) > self.config.max_tangent_error:
            return 'arc_tangent_mismatch'
        return 'arc_discontinuous_support'

    def _runs(self, s, mask):
        indices = np.flatnonzero(mask)
        if not len(indices):
            return []
        result, start, previous = [], int(indices[0]), int(indices[0])
        for value in indices[1:]:
            value = int(value)
            if (value != previous+1
                    or s[value]-s[previous] > self.config.max_point_gap):
                result.append((start, previous))
                start = value
            previous = value
        result.append((start, previous))
        return result

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
            near_endpoint=points[0], far_endpoint=points[-1])

    @staticmethod
    def _invalid(side, candidate, reason, age, nearest=None, radial=None,
                 tangent=None, support=0.0):
        def median(value):
            return math.inf if value is None or not len(value) else float(
                np.median(value))
        return ArcRescue(
            side, False, reason, candidate, None, median(nearest),
            median(radial), median(tangent), float(support), math.nan,
            math.nan, 0.0 if candidate is None else candidate.support_length,
            age)
