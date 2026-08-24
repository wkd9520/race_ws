"""JSONL/NPZ writer for decision-neutral odom arc shadow results."""

import json
from pathlib import Path

import numpy as np


def _comparison(value):
    if value is None:
        return None
    return {
        'candidate_id': value.candidate_id, 'color': value.color,
        'radial_median': value.radial_median, 'radial_p95': value.radial_p95,
        'nearest_median': value.nearest_median, 'nearest_p95': value.nearest_p95,
        'tangent_error_median': value.tangent_error_median,
        'covered_support': value.covered_support,
        'covered_fraction': value.covered_fraction,
    }


class ArcShadowCapture:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.scene = 'UNMARKED'
        self.jsonl_path = self.directory/'arc_shadow.jsonl'
        self.marker_path = self.directory/'scene_markers.jsonl'
        self._jsonl = self.jsonl_path.open('a', encoding='utf-8', buffering=1)
        self._markers = self.marker_path.open('a', encoding='utf-8', buffering=1)

    def set_scene(self, scene, wall_time):
        self.scene = str(scene).strip().upper() or 'UNMARKED'
        self._markers.write(json.dumps(
            {'wall_time': float(wall_time), 'scene': self.scene},
            separators=(',', ':'))+'\n')

    def write(self, stamp, transform_odom_base, outputs, identity, single, candidates):
        stamp_text = f'{stamp.sec}_{stamp.nanosec:09d}'
        arrays = {'transform_odom_base': np.asarray(transform_odom_base, np.float64)}
        if single.center is not None:
            arrays['production_center'] = np.asarray(single.center.points, np.float64)
        if single.missing is not None:
            arrays['production_missing'] = np.asarray(single.missing.points, np.float64)
        sides = {}
        for side, result in outputs.items():
            hypothesis = result.hypothesis
            memory = result.memory
            data = {
                'confirm_streak': result.confirm_streak,
                'age_frames': result.age_frames, 'age_seconds': result.age_seconds,
                'production_association_valid': result.production_association_valid,
                'acquisition_source': result.acquisition_source,
                'category': result.category, 'reason': result.reason,
                'identity_margin': result.identity_margin,
                'best_comparison': _comparison(result.best_comparison),
                'correct_comparison': _comparison(result.correct_comparison),
                'wrong_comparison': _comparison(result.wrong_comparison),
            }
            if hypothesis is not None:
                data['hypothesis'] = {
                    'strong': hypothesis.strong, 'reason': hypothesis.reason,
                    'radius': hypothesis.radius,
                    'inlier_ratio': hypothesis.inlier_ratio, 'rms': hypothesis.rms,
                    'angular_span': hypothesis.angular_span,
                    'contiguous_support': hypothesis.contiguous_support,
                    'center_base': hypothesis.center_base.tolist(),
                    'center_odom': hypothesis.center_odom.tolist(),
                }
            else:
                data['hypothesis'] = None
            if memory is not None:
                data['memory'] = {
                    'radius': memory.radius,
                    'center_odom': memory.center_odom.tolist(),
                    'confirmed_frame': memory.confirmed_frame,
                    'confirmed_time': memory.confirmed_time,
                    'last_strong_frame': memory.last_strong_frame,
                    'last_strong_time': memory.last_strong_time,
                    'confirmation_center_spread': memory.confirmation_center_spread,
                    'confirmation_radius_std': memory.confirmation_radius_std,
                }
                arrays[f'{side.lower()}_memory_arc_odom'] = memory.arc_points_odom
            else:
                data['memory'] = None
            if result.predicted_center_base is not None:
                arrays[f'{side.lower()}_predicted_center_base'] = result.predicted_center_base
            if result.predicted_arc_base is not None:
                arrays[f'{side.lower()}_predicted_arc_base'] = result.predicted_arc_base
            sides[side] = data
        for index, candidate in enumerate(candidates):
            arrays[f'candidate_{index}_canonical'] = np.asarray(
                candidate.canonical_points, np.float64)
        filename = f'shadow_{stamp_text}.npz'
        record = {
            'schema': 1, 'scene': self.scene,
            'stamp': {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)},
            'npz': filename,
            'production': {
                'identity_initialized': identity.identity_initialized,
                'identity_reason': identity.reason,
                'identity_conflict': identity.identity_conflict,
                'both_accepted': identity.both_accepted,
                'observation_mode': single.observation_mode,
                'center_provenance': single.center_provenance,
                'trusted_width': single.trusted_width,
            },
            'candidate_keys': [{
                'array': f'candidate_{index}_canonical',
                'component_id': int(candidate.component_id),
                'color': candidate.color,
                'support': float(candidate.support_length),
            } for index, candidate in enumerate(candidates)],
            'sides': sides,
        }
        arrays['metadata_json'] = np.asarray(json.dumps(record, separators=(',', ':')))
        np.savez_compressed(self.directory/filename, **arrays)
        self._jsonl.write(json.dumps(record, separators=(',', ':'))+'\n')
        return self.directory/filename

    def close(self):
        self._jsonl.close()
        self._markers.close()
