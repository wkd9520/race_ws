"""Frame-local BOTH-boundary assignment and canonical centre geometry.

This module deliberately contains no temporal identity, trusted width, target,
single-boundary reconstruction, or output state machine.
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Optional

import numpy as np

from .components import cumulative_arc_length


def _frozen(values, dtype=None):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def polyline_tangents(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError('polyline must have shape (N,2), N>=2')
    delta = np.empty_like(points)
    delta[0] = points[1] - points[0]
    delta[-1] = points[-1] - points[-2]
    if len(points) > 2:
        delta[1:-1] = points[2:] - points[:-2]
    norm = np.linalg.norm(delta, axis=1)
    if np.any(norm <= 1e-9) or not np.all(np.isfinite(norm)):
        raise ValueError('polyline tangent is undefined')
    return delta / norm[:, None]


def polyline_left_normals(points):
    """Canonical unit normals derived from the shared tangent kernel."""
    tangent = polyline_tangents(points)
    return np.column_stack((-tangent[:, 1], tangent[:, 0]))


def _resample(points, spacing):
    source_s = cumulative_arc_length(points)
    support = float(source_s[-1])
    targets = np.arange(0.0, support + 1e-12, spacing)
    if targets[-1] < support - 1e-12:
        targets = np.append(targets, support)
    else:
        targets[-1] = support
    sampled = np.column_stack((
        np.interp(targets, source_s, points[:, 0]),
        np.interp(targets, source_s, points[:, 1]),
    ))
    return sampled, targets


@dataclass(frozen=True)
class BothGeometryConfig:
    usable_min_support: float = 0.20
    usable_min_points: int = 5
    min_width: float = 0.60
    max_width: float = 0.95
    min_correspondences: int = 4
    min_overlap_support: float = 0.15
    max_tangent_angle: float = 0.45
    min_side_consistency: float = 0.80
    max_width_spread: float = 0.15
    ambiguity_score_margin: float = 0.05
    center_spacing: float = 0.05

    def __post_init__(self):
        if self.usable_min_support <= 0 or self.usable_min_points < 2:
            raise ValueError('usable boundary limits are invalid')
        if not 0 < self.min_width < self.max_width:
            raise ValueError('track width range is invalid')
        if self.min_correspondences < 2 or self.min_overlap_support <= 0:
            raise ValueError('pair support limits are invalid')
        if not 0 < self.max_tangent_angle < np.pi / 2:
            raise ValueError('tangent gate is invalid')
        if not 0.5 <= self.min_side_consistency <= 1.0:
            raise ValueError('side consistency is invalid')
        if self.max_width_spread <= 0 or self.ambiguity_score_margin < 0:
            raise ValueError('pair quality limits are invalid')
        if self.center_spacing <= 0:
            raise ValueError('center spacing is invalid')


@dataclass(frozen=True)
class UsableBoundary:
    candidate: object
    usable: bool
    reason: str


@dataclass(frozen=True)
class BoundaryCorrespondence:
    first_indices: np.ndarray
    second_indices: np.ndarray
    first_points: np.ndarray
    second_points: np.ndarray
    widths: np.ndarray
    overlap_support: float
    tangent_alignment_median: float
    side_consistency: float

    def __post_init__(self):
        for name in ('first_indices', 'second_indices', 'first_points',
                     'second_points', 'widths'):
            object.__setattr__(self, name, _frozen(getattr(self, name)))


@dataclass(frozen=True)
class CanonicalCenterPath:
    points: np.ndarray
    s: np.ndarray
    support_length: float
    left_component_id: int
    right_component_id: int
    left_color: str
    right_color: str
    pair_overlap_support: float
    correspondence_count: int
    width_samples: np.ndarray
    width_min: float
    width_median: float
    width_max: float
    geometry_valid: bool
    generation_reason: str

    def __post_init__(self):
        for name in ('points', 's', 'width_samples'):
            object.__setattr__(self, name, _frozen(getattr(self, name)))


@dataclass(frozen=True)
class PairEvaluation:
    first: object
    second: object
    valid: bool
    reason: str
    correspondence: Optional[BoundaryCorrespondence]
    left: Optional[object]
    right: Optional[object]
    center: Optional[CanonicalCenterPath]
    score: float


@dataclass(frozen=True)
class BothFrameResult:
    usable_boundaries: tuple[UsableBoundary, ...]
    pair_evaluations: tuple[PairEvaluation, ...]
    selected_pair: Optional[PairEvaluation]
    reason: str

    @property
    def usable_candidates(self):
        return tuple(item.candidate for item in self.usable_boundaries if item.usable)

    @property
    def center_path(self):
        return self.selected_pair.center if self.selected_pair is not None else None


class FrameLocalBothGeometry:
    def __init__(self, config=BothGeometryConfig()):
        self.config = config

    def process(self, candidates):
        usable = tuple(self._usable(candidate) for candidate in candidates)
        available = [item.candidate for item in usable if item.usable]
        evaluations = tuple(
            self._evaluate_pair(first, second)
            for first, second in combinations(available, 2)
        )
        valid = sorted(
            (item for item in evaluations if item.valid),
            key=lambda item: (-item.score, self._pair_key(item)),
        )
        if not valid:
            return BothFrameResult(usable, evaluations, None, 'no_valid_pair')
        if len(valid) > 1 and valid[0].score - valid[1].score < self.config.ambiguity_score_margin:
            return BothFrameResult(usable, evaluations, None, 'ambiguous_pair')
        return BothFrameResult(usable, evaluations, valid[0], 'valid')

    @staticmethod
    def _candidate_key(candidate):
        return (candidate.color, int(candidate.component_id))

    @classmethod
    def _pair_key(cls, evaluation):
        return tuple(sorted((cls._candidate_key(evaluation.first),
                             cls._candidate_key(evaluation.second))))

    def _usable(self, candidate):
        if candidate is None:
            return UsableBoundary(candidate, False, 'geometry_unavailable')
        if candidate.canonical_point_count < self.config.usable_min_points:
            return UsableBoundary(candidate, False, 'canonical_points_short')
        if candidate.support_length < self.config.usable_min_support:
            return UsableBoundary(candidate, False, 'physical_support_short')
        if not np.all(np.isfinite(candidate.canonical_points)):
            return UsableBoundary(candidate, False, 'geometry_nonfinite')
        return UsableBoundary(candidate, True, 'usable')

    def _evaluate_pair(self, first, second):
        correspondence, reason = self._correspond(first, second)
        if correspondence is None:
            return PairEvaluation(first, second, False, reason, None,
                                  None, None, None, float('-inf'))
        left, right, reason = self._assign_sides(first, second, correspondence)
        if left is None:
            return PairEvaluation(first, second, False, reason, correspondence,
                                  None, None, None, float('-inf'))
        center = self._make_center(left, right, first, correspondence)
        width_spread = float(np.ptp(correspondence.widths))
        score = (correspondence.overlap_support
                 + 0.02 * len(correspondence.widths)
                 - 0.5 * width_spread
                 + 0.1 * correspondence.tangent_alignment_median)
        return PairEvaluation(first, second, True, 'valid', correspondence,
                              left, right, center, float(score))

    def _correspond(self, first, second):
        a = np.asarray(first.canonical_points, dtype=np.float64)
        b = np.asarray(second.canonical_points, dtype=np.float64)
        try:
            tangent_a, tangent_b = polyline_tangents(a), polyline_tangents(b)
        except ValueError:
            return None, 'tangent_invalid'
        distance = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        nearest_b = np.argmin(distance, axis=1)
        nearest_a = np.argmin(distance, axis=0)
        pairs = [(i, int(j)) for i, j in enumerate(nearest_b)
                 if nearest_a[int(j)] == i]
        if len(pairs) < self.config.min_correspondences:
            return None, 'mutual_support_short'
        ia = np.asarray([item[0] for item in pairs], dtype=np.int32)
        ib = np.asarray([item[1] for item in pairs], dtype=np.int32)
        if np.any(np.diff(ia) <= 0) or np.any(np.diff(ib) <= 0):
            return None, 'correspondence_inversion'
        widths = distance[ia, ib]
        in_width = ((widths >= self.config.min_width)
                    & (widths <= self.config.max_width))
        alignment = np.einsum('ij,ij->i', tangent_a[ia], tangent_b[ib])
        aligned = alignment >= np.cos(self.config.max_tangent_angle)
        keep = in_width & aligned
        ia, ib, widths, alignment = ia[keep], ib[keep], widths[keep], alignment[keep]
        if len(ia) < self.config.min_correspondences:
            if np.count_nonzero(in_width) < self.config.min_correspondences:
                return None, 'width_gate'
            return None, 'tangent_gate'
        if np.any(np.diff(ia) <= 0) or np.any(np.diff(ib) <= 0):
            return None, 'correspondence_inversion'
        span_a = float(first.canonical_s[ia[-1]] - first.canonical_s[ia[0]])
        span_b = float(second.canonical_s[ib[-1]] - second.canonical_s[ib[0]])
        overlap = min(span_a, span_b)
        if overlap < self.config.min_overlap_support:
            return None, 'overlap_support_short'
        if float(np.ptp(widths)) > self.config.max_width_spread:
            return None, 'width_inconsistent'
        vector = b[ib] - a[ia]
        average_tangent = tangent_a[ia] + tangent_b[ib]
        average_tangent /= np.linalg.norm(average_tangent, axis=1)[:, None]
        cross = average_tangent[:, 0] * vector[:, 1] - average_tangent[:, 1] * vector[:, 0]
        nonzero = np.abs(cross) > 1e-6
        if np.count_nonzero(nonzero) < self.config.min_correspondences:
            return None, 'side_geometry_ambiguous'
        positive_fraction = float(np.mean(cross[nonzero] > 0))
        side_consistency = max(positive_fraction, 1.0 - positive_fraction)
        result = BoundaryCorrespondence(
            ia, ib, a[ia], b[ib], widths, overlap,
            float(np.median(alignment)), side_consistency,
        )
        return result, 'valid'

    def _assign_sides(self, first, second, correspondence):
        if correspondence.side_consistency < self.config.min_side_consistency:
            return None, None, 'side_assignment_ambiguous'
        tangent_first = polyline_tangents(first.canonical_points)[correspondence.first_indices]
        tangent_second = polyline_tangents(second.canonical_points)[correspondence.second_indices]
        tangent = tangent_first + tangent_second
        tangent /= np.linalg.norm(tangent, axis=1)[:, None]
        vector = correspondence.second_points - correspondence.first_points
        cross = tangent[:, 0] * vector[:, 1] - tangent[:, 1] * vector[:, 0]
        if float(np.median(cross)) > 0:
            return second, first, 'valid'
        return first, second, 'valid'

    def _make_center(self, left, right, first, correspondence):
        midpoints = 0.5 * (correspondence.first_points + correspondence.second_points)
        points, s = _resample(midpoints, self.config.center_spacing)
        widths = correspondence.widths
        return CanonicalCenterPath(
            points=points, s=s, support_length=float(s[-1]),
            left_component_id=int(left.component_id),
            right_component_id=int(right.component_id),
            left_color=left.color, right_color=right.color,
            pair_overlap_support=correspondence.overlap_support,
            correspondence_count=len(widths), width_samples=widths,
            width_min=float(np.min(widths)), width_median=float(np.median(widths)),
            width_max=float(np.max(widths)), geometry_valid=True,
            generation_reason='valid',
        )
