"""Shadow-only motion-compensated, coverage-aware corridor reference.

This module deliberately has no connection to production identity decisions.
It separates a finite physical identity anchor from the local corridor evidence
used by side validation.  A finite model never extrapolates endpoint tangents.
"""

from dataclasses import dataclass
import math

import numpy as np

from .arc_shadow import inverse_transform, transform_points
from .both_geometry import polyline_left_normals, polyline_tangents


@dataclass(frozen=True)
class CorridorCoverage:
    state: str
    projected_points: np.ndarray
    segment_indices: np.ndarray
    parameters: np.ndarray
    distances: np.ndarray
    interior_mask: np.ndarray
    outside_mask: np.ndarray
    near_endpoint_mask: np.ndarray
    far_endpoint_mask: np.ndarray

    @property
    def interior_fraction(self):
        return float(np.mean(self.interior_mask)) if len(self.interior_mask) else 0.0

    @property
    def outside_fraction(self):
        return float(np.mean(self.outside_mask)) if len(self.outside_mask) else 0.0


@dataclass(frozen=True)
class CorridorSideEvidence:
    state: str
    reason: str
    coverage: CorridorCoverage
    signed_lateral: np.ndarray
    tangent_consistency: np.ndarray
    expected_signed_lateral: float
    side_consistent_fraction: float
    opposite_fraction: float
    lateral_residual_median: float


@dataclass(frozen=True)
class TrustedCorridorReference:
    """Finite corridor model stored in odom, independent of identity anchor."""

    valid: bool
    acquisition_stamp: float
    frame_id: str
    model_type: str
    center_odom: np.ndarray
    trusted_width: float
    observed_support: float
    valid_support_min: float
    valid_support_max: float
    provenance: str

    @classmethod
    def from_center(cls, center_points_base, width, transform_odom_base,
                    acquisition_stamp, provenance='ACTUAL_BOTH'):
        points = np.asarray(center_points_base, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError('center polyline must contain at least two points')
        if not math.isfinite(float(width)) or float(width) <= 0.0:
            raise ValueError('trusted width must be positive')
        support = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
        if support <= 0.0:
            raise ValueError('center support must be positive')
        odom = transform_points(points, transform_odom_base)
        return cls(True, float(acquisition_stamp), 'odom', 'GENERIC_LOCAL',
                   odom, float(width), support, 0.0, support, provenance)

    def current_center(self, transform_odom_base):
        if not self.valid:
            raise ValueError('invalid corridor reference')
        return transform_points(self.center_odom,
                                inverse_transform(transform_odom_base))

    def coverage(self, points_base, transform_odom_base):
        center = self.current_center(transform_odom_base)
        points = np.asarray(points_base, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('current points must have shape (N,2)')
        if len(points) == 0:
            empty = np.zeros(0, dtype=bool)
            return CorridorCoverage('EMPTY', points, np.zeros(0, dtype=int),
                                     np.zeros(0), np.zeros(0), empty, empty,
                                     empty, empty)
        vectors = np.diff(center, axis=0)
        lengths2 = np.einsum('ij,ij->i', vectors, vectors)
        if np.any(lengths2 <= 1e-12):
            raise ValueError('degenerate corridor segment')
        start = center[:-1]
        relative = points[:, None, :]-start[None, :, :]
        raw = np.einsum('nsi,si->ns', relative, vectors)/lengths2
        parameter = np.clip(raw, 0.0, 1.0)
        projected = start[None, :, :] + parameter[:, :, None]*vectors[None, :, :]
        distance = np.linalg.norm(points[:, None, :]-projected, axis=2)
        rows = np.arange(len(points))
        segment = np.argmin(distance, axis=1)
        chosen_parameter = parameter[rows, segment]
        chosen_raw = raw[rows, segment]
        chosen = projected[rows, segment]
        near = (segment == 0) & (chosen_parameter <= 1e-9)
        far = ((segment == len(vectors)-1)
               & (chosen_parameter >= 1.0-1e-9))
        interior = (chosen_raw >= 0.0) & (chosen_raw <= 1.0)
        outside = ~interior
        state = 'INTERIOR_CORRESPONDENCE' if np.all(interior) else (
            'OUT_OF_CORRIDOR_REFERENCE_COVERAGE' if np.all(outside)
            else 'PARTIAL_CORRIDOR_REFERENCE_COVERAGE')
        return CorridorCoverage(state, chosen, segment, chosen_parameter,
                                distance[rows, segment], interior, outside,
                                near, far)

    def evaluate_side(self, side, points_base, transform_odom_base,
                      min_side_fraction=0.80, max_lateral_residual=0.12,
                      tangent_gate=0.45):
        center = self.current_center(transform_odom_base)
        coverage = self.coverage(points_base, transform_odom_base)
        points = np.asarray(points_base, dtype=np.float64)
        if not len(points) or not np.any(coverage.interior_mask):
            empty = np.full(len(points), np.nan)
            return CorridorSideEvidence(
                'SIDE_REFERENCE_OUT_OF_COVERAGE', 'no_interior_correspondence',
                coverage, empty, empty, 0.5*self.trusted_width, 0.0, 0.0,
                float('inf'))
        tangent = polyline_tangents(center)
        normal = polyline_left_normals(center)
        idx = coverage.segment_indices
        delta = points-coverage.projected_points
        lateral = np.einsum('ij,ij->i', delta, normal[idx])
        alignment = np.abs(np.einsum(
            'ij,ij->i', polyline_tangents(points), tangent[idx]))
        expected_sign = 1.0 if side == 'LEFT' else -1.0
        expected = expected_sign*0.5*self.trusted_width
        valid = coverage.interior_mask
        sign = expected_sign*lateral[valid] > 0.0
        side_fraction = float(np.mean(sign)) if len(sign) else 0.0
        opposite = float(np.mean(~sign)) if len(sign) else 0.0
        residual = float(np.median(np.abs(lateral[valid]-expected))) if len(sign) else float('inf')
        tangent_ok = float(np.median(alignment[valid])) >= math.cos(tangent_gate) if len(sign) else False
        if coverage.state != 'INTERIOR_CORRESPONDENCE':
            state, reason = 'SIDE_REFERENCE_OUT_OF_COVERAGE', 'partial_coverage'
        elif side_fraction <= 1.0-min_side_fraction:
            state, reason = 'SIDE_OPPOSITE', 'opposite_side_support'
        elif side_fraction < min_side_fraction or residual > max_lateral_residual or not tangent_ok:
            state, reason = 'SIDE_AMBIGUOUS', 'corridor_evidence_insufficient'
        else:
            state, reason = 'SIDE_CONSISTENT', 'supported'
        return CorridorSideEvidence(state, reason, coverage, lateral, alignment,
                                    expected, side_fraction, opposite, residual)

