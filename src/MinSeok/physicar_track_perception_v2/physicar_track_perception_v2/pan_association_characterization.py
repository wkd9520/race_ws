"""Decision-neutral A/B/C reference comparison for manual-pan scenes."""

from dataclasses import dataclass
import json
import math
from pathlib import Path

import numpy as np

from .arc_shadow import inverse_transform, transform_points
from .both_geometry import polyline_tangents


def _value(value):
    value = float(value)
    return value if math.isfinite(value) else None


@dataclass(frozen=True)
class ReferenceMetricConfig:
    distance_gate: float = 0.12
    tangent_gate: float = 0.45
    min_overlap_support: float = 0.15
    min_accepted_support: float = 0.20
    max_gap: float = 0.075
    max_continuation: float = 0.15


def compare_candidate_reference(candidate_points, candidate_s,
                                reference_points,
                                config=ReferenceMetricConfig()):
    """Decompose the production direct-reference gates without mutation."""
    points = np.asarray(candidate_points, dtype=np.float64)
    s = np.asarray(candidate_s, dtype=np.float64)
    reference = np.asarray(reference_points, dtype=np.float64)
    result = {
        'available': len(points) >= 2 and len(reference) >= 2,
        'candidate_support': _value(s[-1]) if len(s) else 0.0,
        'reference_support': _value(np.sum(np.linalg.norm(
            np.diff(reference, axis=0), axis=1))) if len(reference) >= 2 else 0.0,
    }
    if not result['available']:
        result.update(raw_valid=False, first_failed_gate='REFERENCE_UNAVAILABLE')
        return result
    try:
        tangent = polyline_tangents(points)
        reference_tangent = polyline_tangents(reference)
    except ValueError:
        result.update(raw_valid=False, first_failed_gate='TANGENT_INVALID')
        return result
    distance = np.linalg.norm(
        points[:, None, :]-reference[None, :, :], axis=2)
    nearest_index = np.argmin(distance, axis=1)
    nearest = distance[np.arange(len(points)), nearest_index]
    alignment = np.einsum(
        'ij,ij->i', tangent, reference_tangent[nearest_index])
    distance_mask = nearest <= config.distance_gate
    tangent_mask = alignment >= math.cos(config.tangent_gate)
    supported = distance_mask & tangent_mask
    runs, indices = [], np.flatnonzero(supported)
    if len(indices):
        start = previous = int(indices[0])
        for index in indices[1:]:
            index = int(index)
            if index != previous+1 or s[index]-s[previous] > config.max_gap:
                runs.append((start, previous))
                start = index
            previous = index
        runs.append((start, previous))
    overlap = max((float(s[b]-s[a]) for a, b in runs), default=0.0)
    endpoint_gap = min(float(np.linalg.norm(points[i]-reference[j]))
                       for i in (0, len(points)-1)
                       for j in (0, len(reference)-1))
    endpoint_alignment = max(float(np.dot(tangent[i], reference_tangent[j]))
                             for i in (0, len(points)-1)
                             for j in (0, len(reference)-1))
    continuation_pass = (
        endpoint_gap <= config.max_continuation
        and endpoint_alignment >= math.cos(config.tangent_gate))
    if not np.any(distance_mask):
        first = 'DISTANCE_FAIL'
    elif not np.any(supported):
        first = 'TANGENT_FAIL'
    elif overlap < config.min_overlap_support:
        first = ('DIRECT_OVERLAP_FAIL_CONTINUATION_PASS'
                 if continuation_pass else 'OVERLAP_FAIL')
    else:
        first = 'PASS'
    result.update(
        nearest_median=_value(np.median(nearest)),
        nearest_p95=_value(np.percentile(nearest, 95)),
        nearest_min=_value(np.min(nearest)),
        tangent_consistency_median=_value(np.median(alignment)),
        tangent_delta_median=_value(np.median(np.arccos(
            np.clip(alignment, -1.0, 1.0)))),
        distance_consistent_fraction=_value(np.mean(distance_mask)),
        tangent_consistent_fraction=_value(np.mean(tangent_mask)),
        overlap_support=overlap,
        overlap_ratio=_value(overlap/max(float(s[-1]), 1e-9)),
        contiguous_runs=len(runs),
        endpoint_gap=endpoint_gap,
        endpoint_alignment=endpoint_alignment,
        continuation_pass=continuation_pass,
        distance_pass=bool(np.any(distance_mask)),
        tangent_pass=bool(np.any(supported)),
        overlap_pass=overlap >= config.min_overlap_support,
        raw_valid=(overlap >= config.min_overlap_support or continuation_pass),
        first_failed_gate=first)
    return result


class PanAssociationCharacterizer:
    """Capture comparisons only; never returns a production decision."""

    def __init__(self, directory, config=ReferenceMetricConfig()):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory/'reference_comparison.jsonl'
        self.config = config
        self.odom_snapshots = {'LEFT': None, 'RIGHT': None}
        self.last_long_ids = {'LEFT': None, 'RIGHT': None}

    def capture(self, timestamp, pan, candidates, pre_states,
                pre_center, transform_odom_base, production_result,
                tracker_after):
        transform = (None if transform_odom_base is None else
                     np.asarray(transform_odom_base, dtype=np.float64))
        frame = {
            'timestamp': float(timestamp), 'pan': _value(pan),
            'production_reason': production_result.reason,
            'production_mode': self._mode(production_result),
            'transform_odom_base_available': transform is not None,
            'sides': {},
        }
        for side, state in zip(('LEFT', 'RIGHT'), pre_states):
            short = state.association_reference or state.geometry
            long = state.geometry
            odom = self.odom_snapshots[side]
            odom_points = None
            if odom is not None and transform is not None:
                odom_points = transform_points(
                    odom['points_odom'], inverse_transform(transform))
            records = []
            for candidate in candidates:
                metrics = {}
                for name, reference in (
                        ('short_term', short), ('long_term', long)):
                    metrics[name] = compare_candidate_reference(
                        candidate.canonical_points, candidate.canonical_s,
                        [] if reference is None else reference.canonical_points,
                        self.config)
                metrics['odom_long_term'] = compare_candidate_reference(
                    candidate.canonical_points, candidate.canonical_s,
                    [] if odom_points is None else odom_points, self.config)
                records.append({
                    'candidate_id': f'{candidate.color}:{candidate.component_id}',
                    'support': float(candidate.support_length),
                    'near': np.asarray(candidate.near_endpoint).tolist(),
                    'far': np.asarray(candidate.far_endpoint).tolist(),
                    'metrics': metrics,
                })
            frame['sides'][side] = {
                'short_reference_support': (0.0 if short is None else
                                            float(short.support_length)),
                'long_reference_support': (0.0 if long is None else
                                           float(long.support_length)),
                'odom_reference_available': odom_points is not None,
                'candidates': records,
            }
        with self.path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(frame, separators=(',', ':'))+'\n')
        self._update_long_snapshots(tracker_after, transform, timestamp)

    def _update_long_snapshots(self, tracker, transform, timestamp):
        if transform is None:
            return
        for side, state in (('LEFT', tracker.left_state),
                            ('RIGHT', tracker.right_state)):
            geometry = state.geometry
            if geometry is None or id(geometry) == self.last_long_ids[side]:
                continue
            self.odom_snapshots[side] = {
                'points_odom': transform_points(
                    geometry.canonical_points, transform),
                'timestamp': float(timestamp),
            }
            self.last_long_ids[side] = id(geometry)

    @staticmethod
    def _mode(result):
        if result.both_accepted:
            return 'BOTH_OBSERVED'
        if result.left is not None and result.left.valid:
            return 'LEFT_ONLY_OBSERVED'
        if result.right is not None and result.right.valid:
            return 'RIGHT_ONLY_OBSERVED'
        return 'NO_USABLE_BOTH_OR_SINGLE'
