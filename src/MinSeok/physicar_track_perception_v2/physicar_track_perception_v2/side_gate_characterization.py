"""Decision-neutral rejected raw-match side/corridor characterization."""

from dataclasses import asdict
import json
import math
from pathlib import Path

import numpy as np

from .both_geometry import polyline_left_normals, polyline_tangents


def _finite(value):
    value = float(value)
    return value if math.isfinite(value) else None


def _percentile(values, percentile):
    return _finite(np.percentile(values, percentile))


def segment_correspondence(points, center_points):
    """Project points onto a finite polyline and expose endpoint clamping."""
    points = np.asarray(points, dtype=np.float64)
    center = np.asarray(center_points, dtype=np.float64)
    if len(points) == 0 or len(center) < 2:
        return None
    start, vector = center[:-1], np.diff(center, axis=0)
    length_sq = np.einsum('ij,ij->i', vector, vector)
    if np.any(length_sq <= 1e-12):
        return None
    relative = points[:, None, :]-start[None, :, :]
    parameter_raw = np.einsum('nsi,si->ns', relative, vector)/length_sq
    parameter = np.clip(parameter_raw, 0.0, 1.0)
    projected = start[None, :, :]+parameter[:, :, None]*vector[None, :, :]
    distance = np.linalg.norm(points[:, None, :]-projected, axis=2)
    segment = np.argmin(distance, axis=1)
    row = np.arange(len(points))
    selected_parameter = parameter[row, segment]
    selected_raw = parameter_raw[row, segment]
    selected_points = projected[row, segment]
    near = (segment == 0) & (selected_parameter <= 1e-9)
    far = ((segment == len(vector)-1)
           & (selected_parameter >= 1.0-1e-9))
    endpoint = near | far
    return {
        'segment_index': segment,
        'projection_parameter': selected_parameter,
        'projection_parameter_raw': selected_raw,
        'projected_points': selected_points,
        'distance': distance[row, segment],
        'near_clamped': near,
        'far_clamped': far,
        'endpoint_clamped': endpoint,
        'interior': ~endpoint,
    }


def characterize_side_gate(side, association, center, config,
                           boundary_reference=None):
    """Reproduce the exact point-nearest gate and add continuous diagnostics."""
    if not association.valid or association.accepted is None:
        return {'available': False, 'reason': 'RAW_ASSOCIATION_INVALID'}
    if center is None:
        return {
            'available': False, 'reason': 'CENTER_REFERENCE_INVALID',
            'exact_side_state': 'CENTER_REFERENCE_INVALID',
        }
    points = np.asarray(association.accepted.canonical_points, dtype=np.float64)
    center_points = np.asarray(center.points, dtype=np.float64)
    try:
        current_tangent = polyline_tangents(points)
        center_tangent = polyline_tangents(center_points)
        center_normal = polyline_left_normals(center_points)
    except ValueError:
        return {'available': False, 'reason': 'GEOMETRY_INVALID',
                'exact_side_state': 'OTHER'}

    # This block intentionally mirrors TrustedBoundaryIdentity._corridor_evidence.
    distance = np.linalg.norm(
        points[:, None, :]-center_points[None, :, :], axis=2)
    nearest = np.argmin(distance, axis=1)
    delta = points-center_points[nearest]
    lateral = np.einsum('ij,ij->i', delta, center_normal[nearest])
    expected_sign = 1.0 if side == 'LEFT' else -1.0
    expected_distance = 0.5*float(center.width_median)
    expected_signed = expected_sign*expected_distance
    residual = np.abs(lateral-expected_signed)
    tangent = np.abs(np.einsum(
        'ij,ij->i', current_tangent, center_tangent[nearest]))
    sign_consistent = expected_sign*lateral > 0.0
    sign_opposite = expected_sign*lateral < 0.0
    spacing = association.accepted_support/max(len(points)-1, 1)
    consistent_support = float(np.count_nonzero(sign_consistent)*spacing)
    opposite_support = float(np.count_nonzero(sign_opposite)*spacing)
    unclassified_support = float(np.count_nonzero(
        ~(sign_consistent | sign_opposite))*spacing)
    side_consistency = float(np.mean(sign_consistent))
    lateral_residual = float(np.median(residual))
    tangent_consistency = float(np.median(tangent))
    crossing = bool(consistent_support > config.max_gap
                    and opposite_support > config.max_gap)
    if crossing:
        exact, subreason = 'CENTER_CROSSING', 'center_crossing'
    elif side_consistency <= 1.0-config.conflict_min_side_consistency:
        exact, subreason = 'SIDE_OPPOSITE', 'side_inconsistent'
    elif (side_consistency < config.conflict_min_side_consistency
          or lateral_residual > config.conflict_max_lateral_residual
          or tangent_consistency < math.cos(config.tangent_gate)
          or association.accepted_support < config.min_accepted_support):
        exact = 'SIDE_AMBIGUOUS'
        if side_consistency < config.conflict_min_side_consistency:
            subreason = 'low_side_consistent_fraction'
        elif lateral_residual > config.conflict_max_lateral_residual:
            subreason = 'lateral_residual_mismatch'
        elif tangent_consistency < math.cos(config.tangent_gate):
            subreason = 'center_tangent_mismatch'
        else:
            subreason = 'insufficient_physical_support'
    else:
        exact, subreason = 'SIDE_CONSISTENT', 'supported'

    projection = segment_correspondence(points, center_points)
    endpoint_count = int(np.count_nonzero(projection['endpoint_clamped']))
    result = {
        'available': True,
        'exact_side_state': exact,
        'subreason': subreason,
        'candidate_id': f'{association.candidate.color}:{association.candidate.component_id}',
        'raw_source': association.association_source,
        'full_support': _finite(association.candidate.support_length),
        'accepted_support': _finite(association.accepted_support),
        'candidate_near': np.asarray(association.candidate.near_endpoint).tolist(),
        'candidate_far': np.asarray(association.candidate.far_endpoint).tolist(),
        'center_support': _finite(center.support_length),
        'center_near': center_points[0].tolist(),
        'center_far': center_points[-1].tolist(),
        'trusted_width': _finite(center.width_median),
        'expected_signed_lateral': _finite(expected_signed),
        'signed_lateral_median': _finite(np.median(lateral)),
        'signed_lateral_p10': _percentile(lateral, 10),
        'signed_lateral_p90': _percentile(lateral, 90),
        'lateral_residual_median': _finite(lateral_residual),
        'lateral_residual_p95': _percentile(residual, 95),
        'tangent_consistency_median': _finite(tangent_consistency),
        'side_consistency_fraction': _finite(side_consistency),
        'side_consistent_support': _finite(consistent_support),
        'side_consistent_fraction': _finite(consistent_support/max(
            association.accepted_support, 1e-9)),
        'opposite_support': _finite(opposite_support),
        'opposite_fraction': _finite(opposite_support/max(
            association.accepted_support, 1e-9)),
        'unclassified_support': _finite(unclassified_support),
        'center_crossing': crossing,
        'sign_change_count': int(np.count_nonzero(
            np.diff(np.sign(lateral)) != 0)),
        'production_point_nearest_index_p10': _percentile(nearest, 10),
        'production_point_nearest_index_p90': _percentile(nearest, 90),
        'production_center_distance_median': _finite(np.median(
            distance[np.arange(len(points)), nearest])),
        'endpoint_clamped_count': endpoint_count,
        'endpoint_clamped_fraction': _finite(endpoint_count/len(points)),
        'clamped_to_near_count': int(np.count_nonzero(
            projection['near_clamped'])),
        'clamped_to_far_count': int(np.count_nonzero(
            projection['far_clamped'])),
        'clamped_support': _finite(endpoint_count*spacing),
        'interior_correspondence_fraction': _finite(np.mean(
            projection['interior'])),
        'segment_index_p10': _percentile(projection['segment_index'], 10),
        'segment_index_p90': _percentile(projection['segment_index'], 90),
        'segment_distance_median': _finite(np.median(projection['distance'])),
    }
    if boundary_reference is not None:
        result['boundary_shadow'] = boundary_corridor_shadow(
            side, points, boundary_reference, expected_distance)
    return result


def boundary_corridor_shadow(side, points, boundary, half_width):
    """Analysis-only local corridor inferred from the matched identity boundary."""
    reference = np.asarray(boundary.canonical_points, dtype=np.float64)
    try:
        tangent = polyline_tangents(reference)
        normal = polyline_left_normals(reference)
    except ValueError:
        return {'available': False}
    points = np.asarray(points, dtype=np.float64)
    distance = np.linalg.norm(points[:, None, :]-reference[None, :, :], axis=2)
    nearest = np.argmin(distance, axis=1)
    expected_sign = 1.0 if side == 'LEFT' else -1.0
    inferred_center = reference[nearest]-expected_sign*half_width*normal[nearest]
    lateral = np.einsum(
        'ij,ij->i', points-inferred_center, normal[nearest])
    residual = np.abs(lateral-expected_sign*half_width)
    return {
        'available': True,
        'signed_lateral_median': _finite(np.median(lateral)),
        'lateral_residual_median': _finite(np.median(residual)),
        'side_consistency_fraction': _finite(np.mean(expected_sign*lateral > 0.0)),
        'reference_nearest_median': _finite(np.median(np.min(distance, axis=1))),
    }


class SideGateCharacterizer:
    """Observer-only JSONL writer; no result is returned to production."""

    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/'side_gate_evidence.jsonl'
        self.representative_directory = self.directory/'representative_npz'
        self.representative_directory.mkdir(exist_ok=True)
        self._captured_keys = set()

    def capture(self, timestamp, pan, candidates, pre_states, pre_center,
                tracker, production_result):
        evidence = []
        conflict_ids = set()
        raw_by_side = {}
        for side, state in zip(('LEFT', 'RIGHT'), pre_states):
            reference = state.association_reference or state.geometry
            raw = ([tracker._associate(side, reference, item)
                    for item in candidates] if reference is not None else [])
            raw_by_side[side] = raw
        for candidate in candidates:
            if (any(v.valid and v.candidate is candidate for v in raw_by_side['LEFT'])
                    and any(v.valid and v.candidate is candidate for v in raw_by_side['RIGHT'])):
                conflict_ids.add(id(candidate))
        for side, state in zip(('LEFT', 'RIGHT'), pre_states):
            for raw in raw_by_side[side]:
                if not raw.valid:
                    continue
                item = characterize_side_gate(
                    side, raw, pre_center, tracker.config, state.geometry)
                item['hypothesis'] = side
                item['cross_identity_conflict'] = id(raw.candidate) in conflict_ids
                evidence.append(item)
                if item.get('exact_side_state') != 'SIDE_CONSISTENT':
                    self._capture_representative(
                        timestamp, pan, side, raw, pre_center, state.geometry,
                        item)
        record = {
            'timestamp': float(timestamp),
            'pan': _finite(pan),
            'production_reason': production_result.reason,
            'production_left': (None if production_result.left is None else
                                production_result.left.candidate.component_id),
            'production_right': (None if production_result.right is None else
                                 production_result.right.candidate.component_id),
            'trusted_center_valid': pre_center is not None,
            'rejected_raw_matches': evidence,
        }
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, separators=(',', ':'))+'\n')

    def _capture_representative(self, timestamp, pan, side, association,
                                center, boundary, evidence):
        pan_bin = ('zero' if abs(float(pan)) < .05 else
                   'plus10' if .12 < float(pan) < .25 else
                   'plus20' if float(pan) >= .25 else 'other')
        key = (pan_bin, side, evidence['exact_side_state'])
        if key in self._captured_keys or center is None:
            return
        name = (f'{float(timestamp):.6f}_{pan_bin}_{side}_'
                f'{evidence["exact_side_state"]}.npz')
        np.savez_compressed(
            self.representative_directory/name,
            accepted_points=np.asarray(
                association.accepted.canonical_points, dtype=np.float64),
            full_candidate_points=np.asarray(
                association.candidate.canonical_points, dtype=np.float64),
            trusted_center_points=np.asarray(center.points, dtype=np.float64),
            trusted_boundary_points=(np.empty((0, 2), dtype=np.float64)
                                     if boundary is None else np.asarray(
                                         boundary.canonical_points,
                                         dtype=np.float64)),
            evidence_json=np.asarray(json.dumps(evidence)),
        )
        self._captured_keys.add(key)
