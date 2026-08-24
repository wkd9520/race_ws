"""STEP 5.4A decision-neutral full/accepted/circle support capture."""

from pathlib import Path
import json
import math

import numpy as np

from .arc_shadow import ArcShadowConfig, fit_circle_shadow
from .components import cumulative_arc_length


def _runs(mask):
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs, start, previous = [], int(indices[0]), int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            runs.append((start, previous))
            start = value
        previous = value
    runs.append((start, previous))
    return runs


def detailed_circle_fit(points, transform_odom_base,
                        config=ArcShadowConfig()):
    """Use the production fitter, then expose its exact final inlier geometry."""
    points = np.asarray(points, dtype=np.float64)
    hypothesis = fit_circle_shadow(points, transform_odom_base, config)
    if hypothesis is None:
        return None
    s = cumulative_arc_length(points)
    radial = np.abs(np.linalg.norm(
        points-hypothesis.center_base, axis=1)-hypothesis.radius)
    mask = radial <= config.ransac_threshold
    runs = _runs(mask)
    supports = [float(s[end]-s[start]) for start, end in runs]
    longest = max(range(len(runs)), key=lambda i: supports[i])
    start, end = runs[longest]
    all_angles = np.unwrap(np.arctan2(
        points[:, 1]-hypothesis.center_base[1],
        points[:, 0]-hypothesis.center_base[0]))
    inlier_support = float(sum(supports))
    return {
        'hypothesis': hypothesis,
        'inlier_mask': mask,
        'inlier_indices': np.flatnonzero(mask),
        'longest_start': int(start),
        'longest_end': int(end),
        'inlier_run_count': len(runs),
        'inlier_support': inlier_support,
        'outlier_support': max(0.0, float(s[-1])-inlier_support),
        'all_point_angular_span': float(np.ptp(all_angles)),
    }


def retention_metrics(full_support, accepted_support, circle_support,
                      full_span, accepted_span):
    def ratio(value, denominator):
        return float(value/max(float(denominator), 1e-12))
    return {
        'geometry': ratio(accepted_support, full_support),
        'circle': ratio(circle_support, accepted_support),
        'total': ratio(circle_support, full_support),
        'span': ratio(accepted_span, full_span),
    }


class ArcSupportCapture:
    def __init__(self, directory, stride=1):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, int(stride))
        self.frame_index = 0
        self.scene = 'UNMARKED'
        self.records = self.directory / 'arc_support_index.jsonl'

    def set_scene(self, value):
        value = str(value).strip().upper()
        self.scene = value if value else 'UNMARKED'

    @staticmethod
    def _fit_metadata(value):
        if value is None:
            return {
                'valid': False, 'strong': False, 'reason': 'fit_unavailable'}
        fit = value['hypothesis']
        return {
            'valid': True, 'strong': bool(fit.strong), 'reason': fit.reason,
            'inlier_ratio': fit.inlier_ratio, 'rms': fit.rms,
            'radius': fit.radius, 'center_base': fit.center_base.tolist(),
            'angular_span_rad': fit.angular_span,
            'angular_span_deg': math.degrees(fit.angular_span),
            'all_point_span_rad': value['all_point_angular_span'],
            'all_point_span_deg': math.degrees(value['all_point_angular_span']),
            'contiguous_support': fit.contiguous_support,
            'inlier_support': value['inlier_support'],
            'outlier_support': value['outlier_support'],
            'inlier_run_count': value['inlier_run_count'],
            'longest_interval': [value['longest_start'], value['longest_end']],
        }

    def capture(self, stamp, candidates, identity, tracker,
                transform_odom_base):
        frame = self.frame_index
        self.frame_index += 1
        if frame % self.stride or transform_odom_base is None:
            return ()
        candidates = tuple(candidates)
        associations = {'LEFT': identity.left, 'RIGHT': identity.right}
        relevant = []
        for side, association in associations.items():
            if association is not None and association.candidate is not None:
                relevant.append((side, association.candidate, association))
        whites = [item for item in candidates if item.color == 'WHITE']
        if whites:
            longest = max(whites, key=lambda item: item.support_length)
            if not any(item[1] is longest for item in relevant):
                relevant.append(('UNASSIGNED_LONGEST_WHITE', longest, None))
        outputs = []
        for side, candidate, association in relevant:
            full_points = np.asarray(candidate.canonical_points)
            accepted = None if association is None else association.accepted
            accepted_points = (np.empty((0, 2), np.float64) if accepted is None
                               else np.asarray(accepted.canonical_points))
            full_fit = detailed_circle_fit(full_points, transform_odom_base)
            accepted_fit = (None if len(accepted_points) < 4 else
                            detailed_circle_fit(
                                accepted_points, transform_odom_base))
            full_meta = self._fit_metadata(full_fit)
            accepted_meta = self._fit_metadata(accepted_fit)
            full_support = float(candidate.support_length)
            accepted_support = (0.0 if association is None else
                                float(association.accepted_support))
            circle_support = float(accepted_meta.get(
                'contiguous_support', 0.0))
            retention = retention_metrics(
                full_support, accepted_support, circle_support,
                full_meta.get('angular_span_rad', 0.0),
                accepted_meta.get('angular_span_rad', 0.0))
            memory_side = side if side in ('LEFT', 'RIGHT') else None
            pending = (0 if memory_side is None or tracker is None else
                       len(tracker.pending[memory_side]))
            confirmed = bool(memory_side is not None and tracker is not None
                             and tracker.memory[memory_side] is not None)
            source = ('NONE' if association is None else
                      association.association_source)
            metadata = {
                'stamp': float(stamp.sec)+1e-9*float(stamp.nanosec),
                'scene': self.scene, 'side': side,
                'candidate_id': int(candidate.component_id),
                'color': candidate.color,
                'full': {
                    'raw_point_count': int(candidate.raw_point_count),
                    'canonical_point_count': int(candidate.canonical_point_count),
                    'support': full_support,
                    'near': candidate.near_endpoint.tolist(),
                    'far': candidate.far_endpoint.tolist(),
                    **full_meta,
                },
                'accepted': {
                    'point_count': int(len(accepted_points)),
                    'support': accepted_support,
                    'interval_start_s': (math.nan if association is None else
                                         association.interval_start_s),
                    'interval_end_s': (math.nan if association is None else
                                       association.interval_end_s),
                    'direct_overlap_support': (0.0 if association is None else
                                               association.overlap_support),
                    'continuation_support': (0.0 if association is None else max(
                        0.0, association.accepted_support-
                        association.overlap_support)),
                    'rejected_tail_support': (full_support if association is None
                                              else association.rejected_tail_support),
                    'reference_support': (0.0 if association is None else
                                          association.reference_support),
                    'association_source': source,
                    'association_reason': ('unassigned' if association is None
                                           else association.reason),
                    'sliding_used': bool(association is not None and
                                         association.sliding_association_used),
                    **accepted_meta,
                },
                'retention': retention,
                'confirmation_streak': pending,
                'memory_confirmed': confirmed,
                'identity_frame_reason': identity.reason,
            }
            stem = (f'{int(stamp.sec):010d}_{int(stamp.nanosec):09d}_'
                    f'{side}_{candidate.color}_{candidate.component_id}')
            path = self.directory / f'{stem}.npz'
            arrays = {
                'full_canonical_points': full_points,
                'accepted_points': accepted_points,
                'metadata_json': np.asarray(json.dumps(metadata)),
            }
            for prefix, fit in (('full', full_fit), ('accepted', accepted_fit)):
                if fit is not None:
                    arrays[f'{prefix}_inlier_mask'] = fit['inlier_mask']
                    arrays[f'{prefix}_inlier_indices'] = fit['inlier_indices']
            np.savez_compressed(path, **arrays)
            with self.records.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(metadata, sort_keys=True) + '\n')
            outputs.append(path)
        return tuple(outputs)
