"""Validated width and trusted single-boundary local-normal reconstruction."""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .both_geometry import CanonicalCenterPath, polyline_left_normals
from .components import cumulative_arc_length


BOTH_OBSERVED = 'BOTH_OBSERVED'
LEFT_ONLY_OBSERVED = 'LEFT_ONLY_OBSERVED'
RIGHT_ONLY_OBSERVED = 'RIGHT_ONLY_OBSERVED'
NO_USABLE_BOTH_OR_SINGLE = 'NO_USABLE_BOTH_OR_SINGLE'
OBSERVED = 'OBSERVED'
RECONSTRUCTED = 'RECONSTRUCTED'
UNAVAILABLE = 'UNAVAILABLE'
BOTH_CENTER = 'BOTH_CENTER'
LEFT_SINGLE_RECONSTRUCTED = 'LEFT_SINGLE_RECONSTRUCTED'
RIGHT_SINGLE_RECONSTRUCTED = 'RIGHT_SINGLE_RECONSTRUCTED'
INVALID = 'INVALID'
REGULAR = 'REGULAR'
DEGENERATE = 'DEGENERATE'
UNKNOWN = 'UNKNOWN'


@dataclass(frozen=True)
class WidthConfig:
    initialization_frames: int = 3
    ema_alpha: float = 0.20
    update_gate: float = 0.12


@dataclass(frozen=True)
class TrustedWidthState:
    initialized: bool
    width: Optional[float]
    initialization_samples: tuple[float, ...]
    update_count: int


class ValidatedWidth:
    def __init__(self, config=WidthConfig()):
        self.config = config
        self.state = TrustedWidthState(False, None, (), 0)

    def observe_both(self, width):
        width = float(width)
        if not np.isfinite(width) or width <= 0:
            return False
        if not self.state.initialized:
            samples = self.state.initialization_samples + (width,)
            if len(samples) < self.config.initialization_frames:
                self.state = TrustedWidthState(False, None, samples, 0)
                return True
            value = float(np.median(samples))
            self.state = TrustedWidthState(True, value, samples,
                                           len(samples))
            return True
        if abs(width-self.state.width) > self.config.update_gate:
            return False
        value = (self.config.ema_alpha*width
                 + (1-self.config.ema_alpha)*self.state.width)
        self.state = TrustedWidthState(True, float(value),
                                       self.state.initialization_samples,
                                       self.state.update_count+1)
        return True


@dataclass(frozen=True)
class OffsetSafety:
    state: str
    reason: str
    valid_curvature_count: int
    minimum_factor: float
    irregular_span: float


@dataclass(frozen=True)
class ReconstructedBoundary:
    points: np.ndarray
    s: np.ndarray
    support_length: float
    provenance: str
    valid: bool
    reason: str


@dataclass(frozen=True)
class SingleFrameOutput:
    observation_mode: str
    left_provenance: str
    right_provenance: str
    center_provenance: str
    observed_side: str
    trusted_width: Optional[float]
    width_update_allowed: bool
    normal_sign: float
    normal_sign_source: str
    center: Optional[CanonicalCenterPath]
    missing: Optional[ReconstructedBoundary]
    center_safety: OffsetSafety
    missing_safety: OffsetSafety
    reason: str


def signed_curvature(points, s, support=0.10):
    points, s = np.asarray(points), np.asarray(s)
    curvature = np.full(len(points), np.nan)
    for index in range(len(points)):
        before = np.flatnonzero(s <= s[index]-support)
        after = np.flatnonzero(s >= s[index]+support)
        if not len(before) or not len(after):
            continue
        a, b, c = points[before[-1]], points[index], points[after[0]]
        ab, bc, ac = np.linalg.norm(b-a), np.linalg.norm(c-b), np.linalg.norm(c-a)
        area2 = np.cross(b-a, c-a)
        denominator = ab*bc*ac
        if denominator > 1e-9:
            curvature[index] = 2*area2/denominator
    return curvature


class TrustedSingleReconstruction:
    def __init__(self, width=None, curvature_support=0.10,
                 persistence_span=0.15, min_curvature_samples=3):
        self.width = width or ValidatedWidth()
        self.curvature_support = curvature_support
        self.persistence_span = persistence_span
        self.min_curvature_samples = min_curvature_samples

    def process(self, identity, identity_tracker, timestamp=0.0):
        if identity.both_accepted and identity.center_result.center_path is not None:
            center = identity.center_result.center_path
            updated = self.width.observe_both(center.width_median)
            return SingleFrameOutput(
                BOTH_OBSERVED, OBSERVED, OBSERVED, BOTH_CENTER, 'BOTH',
                self.width.state.width, updated, 0.0, 'not_applicable',
                center, None, self._na(), self._na(), 'valid')
        valid_left = identity.left is not None and identity.left.valid
        valid_right = identity.right is not None and identity.right.valid
        if valid_left == valid_right or not self.width.state.initialized:
            reason = ('trusted_width_unavailable' if valid_left != valid_right
                      else identity.reason)
            return self._invalid(reason)
        if valid_left:
            observed, opposite = identity.left.accepted, identity_tracker.right_state.geometry
            side, mode = 'LEFT', LEFT_ONLY_OBSERVED
            left_prov, right_prov = OBSERVED, RECONSTRUCTED
            center_prov = LEFT_SINGLE_RECONSTRUCTED
        else:
            observed, opposite = identity.right.accepted, identity_tracker.left_state.geometry
            side, mode = 'RIGHT', RIGHT_ONLY_OBSERVED
            left_prov, right_prov = RECONSTRUCTED, OBSERVED
            center_prov = RIGHT_SINGLE_RECONSTRUCTED
        normals, sign, source = self._interior_normals(observed, opposite)
        if normals is None:
            return self._invalid('normal_direction_ambiguous', mode, side,
                                 left_prov, right_prov)
        width = self.width.state.width
        center_safety = self._safety(observed, sign*0.5*width)
        missing_safety = self._safety(observed, sign*width)
        if center_safety.state != REGULAR:
            return SingleFrameOutput(
                mode, left_prov, right_prov, INVALID, side, width, False,
                sign, source, None, None, center_safety, missing_safety,
                'center_geometry_'+center_safety.state.lower())
        center_points = observed.canonical_points + 0.5*width*normals
        center_s = cumulative_arc_length(center_points)
        center = CanonicalCenterPath(
            center_points, center_s, float(center_s[-1]),
            observed.component_id if side == 'LEFT' else -1,
            observed.component_id if side == 'RIGHT' else -1,
            observed.color if side == 'LEFT' else RECONSTRUCTED,
            observed.color if side == 'RIGHT' else RECONSTRUCTED,
            observed.support_length, len(center_points), np.empty(0),
            width, width, width, True, 'valid')
        missing = None
        if missing_safety.state == REGULAR:
            missing_points = observed.canonical_points + width*normals
            missing_s = cumulative_arc_length(missing_points)
            missing = ReconstructedBoundary(
                missing_points, missing_s, float(missing_s[-1]),
                RECONSTRUCTED, True, 'valid')
        association = identity.left if side == 'LEFT' else identity.right
        identity_tracker.update_single_association(association, timestamp)
        return SingleFrameOutput(
            mode, left_prov, right_prov, center_prov, side, width, False,
            sign, source, center, missing, center_safety, missing_safety,
            'valid' if missing is not None else 'missing_geometry_'+missing_safety.state.lower())

    def _interior_normals(self, observed, opposite):
        if opposite is None:
            return None, 0.0, 'unavailable'
        points = observed.canonical_points
        normals = polyline_left_normals(points)
        reference = opposite.canonical_points
        distance = np.linalg.norm(points[:, None]-reference[None, :], axis=2)
        vector = reference[np.argmin(distance, axis=1)]-points
        dot = np.einsum('ij,ij->i', normals, vector)
        usable = np.abs(dot) > 1e-4
        if np.count_nonzero(usable) < 3:
            return None, 0.0, 'trusted_opposite'
        positive = float(np.mean(dot[usable] > 0))
        if max(positive, 1-positive) < 0.80:
            return None, 0.0, 'trusted_opposite'
        sign = 1.0 if positive >= 0.5 else -1.0
        return sign*normals, sign, 'trusted_opposite_boundary'

    def _safety(self, observed, signed_offset):
        curvature = signed_curvature(
            observed.canonical_points, observed.canonical_s,
            self.curvature_support)
        valid = np.isfinite(curvature)
        if np.count_nonzero(valid) < self.min_curvature_samples:
            return OffsetSafety(UNKNOWN, 'insufficient_physical_support',
                                int(np.count_nonzero(valid)), float('nan'), 0.0)
        factor = 1-signed_offset*curvature
        irregular = valid & (factor <= 0)
        max_span = 0.0
        indices = np.flatnonzero(irregular)
        if len(indices):
            start = previous = int(indices[0])
            for value in list(indices[1:])+[None]:
                if value is None or int(value) != previous+1:
                    max_span = max(max_span, float(
                        observed.canonical_s[previous]-observed.canonical_s[start]))
                    if value is None:
                        break
                    start = int(value)
                previous = int(value)
        if max_span >= self.persistence_span:
            return OffsetSafety(DEGENERATE, 'sustained_offset_singularity',
                                int(np.count_nonzero(valid)),
                                float(np.nanmin(factor)), max_span)
        return OffsetSafety(REGULAR, 'regular', int(np.count_nonzero(valid)),
                            float(np.nanmin(factor)), max_span)

    @staticmethod
    def _na():
        return OffsetSafety(UNKNOWN, 'not_applicable', 0, float('nan'), 0.0)

    def _invalid(self, reason, mode=NO_USABLE_BOTH_OR_SINGLE,
                 side='NONE', left=UNAVAILABLE, right=UNAVAILABLE):
        return SingleFrameOutput(
            mode, left, right, INVALID, side, self.width.state.width, False,
            0.0, 'unavailable', None, None, self._na(), self._na(), reason)
