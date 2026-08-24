"""Decision-neutral fixed-frame circle memory and comparison helpers."""

from dataclasses import dataclass, replace
import math

import numpy as np

from .both_geometry import polyline_tangents


@dataclass(frozen=True)
class ArcShadowConfig:
    # PROPOSED SHADOW VALUES: characterization only, never production gates.
    ransac_threshold: float = 0.02
    min_inlier_ratio: float = 0.60
    max_rms: float = 0.014
    min_angular_span: float = math.radians(60.0)
    min_contiguous_support: float = 0.50
    confirmation_frames: int = 3
    max_confirmation_radius_fraction: float = 0.05
    max_confirmation_center_spread: float = 0.10
    max_age_frames: int = 15


@dataclass(frozen=True)
class CircleHypothesis:
    center_base: np.ndarray
    center_odom: np.ndarray
    radius: float
    inlier_ratio: float
    rms: float
    angular_span: float
    contiguous_support: float
    inlier_points_base: np.ndarray
    inlier_points_odom: np.ndarray
    strong: bool
    reason: str


@dataclass(frozen=True)
class ShadowArcMemory:
    valid: bool
    side: str
    center_odom: np.ndarray
    radius: float
    arc_points_odom: np.ndarray
    confirmed_frame: int
    confirmed_time: float
    last_strong_frame: int
    last_strong_time: float
    confirmation_center_spread: float
    confirmation_radius_std: float


@dataclass(frozen=True)
class ArcComparison:
    candidate_id: int
    color: str
    radial_median: float
    radial_p95: float
    nearest_median: float
    nearest_p95: float
    tangent_error_median: float
    covered_support: float
    covered_fraction: float


@dataclass(frozen=True)
class SideShadowResult:
    side: str
    hypothesis: CircleHypothesis | None
    confirm_streak: int
    memory: ShadowArcMemory | None
    age_frames: int
    age_seconds: float
    predicted_center_base: np.ndarray | None
    predicted_arc_base: np.ndarray | None
    best_comparison: ArcComparison | None
    correct_comparison: ArcComparison | None
    wrong_comparison: ArcComparison | None
    identity_margin: float
    production_association_valid: bool
    acquisition_source: str
    category: str
    reason: str


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (3, 3) or points.ndim != 2 or points.shape[1] != 2:
        raise ValueError('2D homogeneous transform/points shape invalid')
    return (transform @ np.column_stack((points, np.ones(len(points)))).T).T[:, :2]


def inverse_transform(transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (3, 3):
        raise ValueError('2D homogeneous transform shape invalid')
    return np.linalg.inv(transform)


def _circle_from_three(points):
    a = 2.0*(points[1:]-points[0])
    b = np.sum(points[1:]**2, axis=1)-np.sum(points[0]**2)
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    center = np.linalg.solve(a, b)
    radius = float(np.linalg.norm(points[0]-center))
    return None if not math.isfinite(radius) or radius < 1e-5 else (center, radius)


def _refit_circle(points):
    a = np.column_stack((2.0*points, np.ones(len(points))))
    b = np.sum(points**2, axis=1)
    value = np.linalg.lstsq(a, b, rcond=None)[0]
    radius2 = value[2]+value[0]**2+value[1]**2
    if radius2 <= 1e-10:
        return None
    return value[:2], float(math.sqrt(radius2))


def _runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    result, start, previous = [], int(indices[0]), int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous+1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result


def fit_circle_shadow(points, transform_odom_base, config=ArcShadowConfig(), seed=20260823):
    points = np.asarray(points, dtype=np.float64)
    if len(points) < 4:
        return None
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(500):
        model = _circle_from_three(points[rng.choice(len(points), 3, replace=False)])
        if model is None:
            continue
        center, radius = model
        residual = np.abs(np.linalg.norm(points-center, axis=1)-radius)
        mask = residual <= config.ransac_threshold
        runs = _runs(mask)
        support = [float(s[end]-s[start]) for start, end in runs]
        key = (max(support, default=0.0), sum(support), int(np.count_nonzero(mask)))
        if best is None or key > best[0]:
            best = key, mask
    if best is None or np.count_nonzero(best[1]) < 3:
        return None
    model = _refit_circle(points[best[1]])
    if model is None:
        return None
    center, radius = model
    residual = np.abs(np.linalg.norm(points-center, axis=1)-radius)
    mask = residual <= config.ransac_threshold
    model = _refit_circle(points[mask]) if np.count_nonzero(mask) >= 3 else model
    center, radius = model
    residual = np.abs(np.linalg.norm(points-center, axis=1)-radius)
    mask = residual <= config.ransac_threshold
    if not np.any(mask):
        return None
    run_values = _runs(mask)
    longest = max(run_values, key=lambda item: s[item[1]]-s[item[0]])
    contiguous = float(s[longest[1]]-s[longest[0]])
    inlier_points = points[longest[0]:longest[1]+1]
    angles = np.unwrap(np.arctan2(
        inlier_points[:, 1]-center[1], inlier_points[:, 0]-center[0]))
    span = float(np.ptp(angles))
    rms = float(np.sqrt(np.mean(residual[mask]**2)))
    ratio = float(np.mean(mask))
    strong = (ratio >= config.min_inlier_ratio and rms <= config.max_rms
              and span >= config.min_angular_span
              and contiguous >= config.min_contiguous_support)
    if ratio < config.min_inlier_ratio:
        reason = 'inlier_ratio'
    elif rms > config.max_rms:
        reason = 'radial_rms'
    elif span < config.min_angular_span:
        reason = 'angular_span'
    elif contiguous < config.min_contiguous_support:
        reason = 'contiguous_support'
    else:
        reason = 'strong'
    center_odom = transform_points(np.asarray([center]), transform_odom_base)[0]
    inlier_odom = transform_points(inlier_points, transform_odom_base)
    return CircleHypothesis(
        center, center_odom, radius, ratio, rms, span, contiguous,
        inlier_points, inlier_odom, strong, reason)


def compare_candidate(candidate, center_base, radius, predicted_arc, threshold=0.03):
    points = np.asarray(candidate.canonical_points, dtype=np.float64)
    radial = np.abs(np.linalg.norm(points-center_base, axis=1)-radius)
    distance = np.linalg.norm(points[:, None, :]-predicted_arc[None, :, :], axis=2)
    nearest = np.min(distance, axis=1)
    tangent = polyline_tangents(points)
    radial_vector = points-center_base
    radial_vector /= np.maximum(np.linalg.norm(radial_vector, axis=1, keepdims=True), 1e-12)
    circle_tangent = np.column_stack((-radial_vector[:, 1], radial_vector[:, 0]))
    alignment = np.clip(np.abs(np.einsum('ij,ij->i', tangent, circle_tangent)), 0.0, 1.0)
    tangent_error = np.arccos(alignment)
    s = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    covered = radial <= threshold
    covered_support = float(sum(
        s[end]-s[start] for start, end in _runs(covered)))
    return ArcComparison(
        int(candidate.component_id), candidate.color,
        float(np.median(radial)), float(np.percentile(radial, 95)),
        float(np.median(nearest)), float(np.percentile(nearest, 95)),
        float(np.median(tangent_error)), covered_support,
        covered_support/max(float(candidate.support_length), 1e-12))


class ArcShadowTracker:
    """Independent shadow state; inputs are read but never mutated."""

    def __init__(self, config=ArcShadowConfig()):
        self.config = config
        self.frame_index = 0
        self.pending = {'LEFT': [], 'RIGHT': []}
        self.memory = {'LEFT': None, 'RIGHT': None}

    def process(self, timestamp, transform_odom_base, associations, candidates):
        self.frame_index += 1
        outputs = {}
        for side in ('LEFT', 'RIGHT'):
            association = associations.get(side)
            production_valid = bool(
                association is not None and association.valid
                and association.accepted is not None)
            hypothesis = None
            if production_valid:
                hypothesis = fit_circle_shadow(
                    association.accepted.canonical_points,
                    transform_odom_base, self.config)
            self._update(side, timestamp, hypothesis)
            outputs[side] = self._result(
                side, timestamp, transform_odom_base, hypothesis,
                candidates, association, production_valid)
        return outputs

    def _update(self, side, timestamp, hypothesis):
        if hypothesis is None or not hypothesis.strong:
            self.pending[side] = []
            return
        pending = self.pending[side]
        if pending:
            radius_reference = float(np.median([item.radius for item in pending]))
            center_reference = np.median(
                np.asarray([item.center_odom for item in pending]), axis=0)
            consistent = (
                abs(hypothesis.radius-radius_reference)/max(radius_reference, 1e-9)
                <= self.config.max_confirmation_radius_fraction
                and np.linalg.norm(hypothesis.center_odom-center_reference)
                <= self.config.max_confirmation_center_spread)
            if not consistent:
                pending = []
        pending = (pending+[hypothesis])[-self.config.confirmation_frames:]
        self.pending[side] = pending
        if len(pending) < self.config.confirmation_frames:
            return
        centers = np.asarray([item.center_odom for item in pending])
        radii = np.asarray([item.radius for item in pending])
        center = np.median(centers, axis=0)
        radius = float(np.median(radii))
        points = pending[-1].inlier_points_odom
        existing = self.memory[side]
        confirmed_frame = (self.frame_index if existing is None
                           else existing.confirmed_frame)
        confirmed_time = timestamp if existing is None else existing.confirmed_time
        self.memory[side] = ShadowArcMemory(
            True, side, center, radius, points, confirmed_frame, confirmed_time,
            self.frame_index, timestamp,
            float(np.max(np.linalg.norm(centers-center, axis=1))),
            float(np.std(radii)))

    def _result(self, side, timestamp, transform_odom_base, hypothesis,
                candidates, association, production_valid):
        memory = self.memory[side]
        if memory is None:
            return SideShadowResult(
                side, hypothesis, len(self.pending[side]), None, 0, 0.0,
                None, None, None, None, None, float('nan'), production_valid,
                getattr(association, 'shadow_source', 'production_identity'),
                'UNCONFIRMED', 'model_unconfirmed')
        age_frames = self.frame_index-memory.last_strong_frame
        age_seconds = timestamp-memory.last_strong_time
        transform_base_odom = inverse_transform(transform_odom_base)
        center_base = transform_points(
            np.asarray([memory.center_odom]), transform_base_odom)[0]
        arc_base = transform_points(memory.arc_points_odom, transform_base_odom)
        comparisons = [compare_candidate(
            candidate, center_base, memory.radius, arc_base)
            for candidate in candidates if candidate.canonical_point_count >= 4]
        best = max(comparisons, key=lambda item: (
            item.covered_support, -item.radial_median,
            -item.tangent_error_median)) if comparisons else None
        correct = None
        if association is not None:
            correct = next((item for item in comparisons
                            if item.candidate_id == association.candidate.component_id
                            and item.color == association.candidate.color), None)
        other = self.memory['RIGHT' if side == 'LEFT' else 'LEFT']
        wrong = None
        margin = float('nan')
        if best is not None and other is not None:
            other_center = transform_points(
                np.asarray([other.center_odom]), transform_base_odom)[0]
            other_arc = transform_points(other.arc_points_odom, transform_base_odom)
            candidate = next((value for value in candidates
                              if value.component_id == best.candidate_id
                              and value.color == best.color), None)
            if candidate is not None:
                wrong = compare_candidate(
                    candidate, other_center, other.radius, other_arc)
                margin = wrong.radial_median-best.radial_median
        accurate = bool(best is not None and best.radial_median <= 0.03
                        and best.tangent_error_median <= 0.30
                        and best.covered_support >= 0.20)
        if production_valid and accurate:
            category = 'A_PRODUCTION_VALID_SHADOW_ACCURATE'
        elif not production_valid and accurate:
            category = 'B_PRODUCTION_INVALID_SHADOW_ACCURATE'
        elif production_valid:
            category = 'C_PRODUCTION_VALID_SHADOW_STALE'
        else:
            category = 'D_BOTH_FAIL'
        reason = ('age_limit' if age_frames > self.config.max_age_frames
                  else 'comparison_available' if best is not None
                  else 'no_candidate')
        return SideShadowResult(
            side, hypothesis, len(self.pending[side]), memory,
            age_frames, age_seconds, center_base, arc_base, best, correct,
            wrong, margin, production_valid,
            getattr(association, 'shadow_source', 'production_identity'),
            category, reason)
