"""Decision-neutral runtime geometry capture for offline analysis."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np


IDLE_SCENES = frozenset(('', 'IDLE', 'STOP'))


def rejected_canonical_points(candidate, accepted, tolerance=1e-9):
    """Return candidate samples not present in an accepted subpolyline."""
    if candidate is None:
        return np.empty((0, 2), dtype=np.float64)
    full = np.asarray(candidate.canonical_points, dtype=np.float64)
    if accepted is None:
        return full.copy()
    kept = np.asarray(accepted.canonical_points, dtype=np.float64)
    if not len(kept):
        return full.copy()
    distance = np.linalg.norm(full[:, None, :] - kept[None, :, :], axis=2)
    return full[np.min(distance, axis=1) > tolerance].copy()


@dataclass
class RuntimeGeometryCapture:
    directory: Path
    scene: str = 'IDLE'

    def __post_init__(self):
        self.directory = Path(self.directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / 'geometry_manifest.jsonl'
        self.marker_path = self.directory / 'scene_markers.jsonl'
        self._manifest = self.manifest_path.open('a', encoding='utf-8', buffering=1)
        self._markers = self.marker_path.open('a', encoding='utf-8', buffering=1)

    @property
    def active(self):
        return self.scene.upper() not in IDLE_SCENES

    def set_scene(self, scene, wall_time):
        value = str(scene).strip().upper()
        self.scene = value or 'IDLE'
        self._markers.write(json.dumps(
            {'wall_time': float(wall_time), 'scene': self.scene},
            separators=(',', ':')) + '\n')

    @staticmethod
    def _association_metadata(value):
        if value is None:
            return None
        candidate = value.candidate
        return {
            'side': value.side,
            'candidate_id': int(candidate.component_id),
            'color': candidate.color,
            'valid': bool(value.valid),
            'reason': value.reason,
            'accepted_support': float(value.accepted_support),
            'rejected_tail_support': float(value.rejected_tail_support),
            'side_state': value.side_state,
            'sliding_association_used': bool(value.sliding_association_used),
        }

    def capture(self, stamp, component_frame, identity, identity_tracker, single):
        if not self.active:
            return None
        stamp_text = f'{stamp.sec}_{stamp.nanosec:09d}'
        name = f'{self.scene}_{stamp_text}.npz'
        path = self.directory / name
        arrays = {}
        candidates = []
        for index, observation in enumerate(component_frame.observations):
            metadata, candidate = observation.metadata, observation.candidate
            prefix = f'component_{index}'
            arrays[f'{prefix}_raw_pixels_rc'] = np.asarray(
                metadata.raw_pixels_rc, dtype=np.int32)
            item = {
                'index': index,
                'component_id': int(metadata.component_id),
                'color': metadata.color,
                'area_pixels': int(metadata.area_pixels),
                'valid_area_pixels': int(metadata.valid_area_pixels),
                'valid_overlap': float(metadata.valid_overlap),
                'geometry_valid': bool(metadata.geometry_valid),
                'reason': metadata.rejection_reason,
            }
            if candidate is not None:
                arrays[f'{prefix}_raw_ordered_points'] = np.asarray(
                    candidate.raw_ordered_points, dtype=np.float64)
                arrays[f'{prefix}_canonical_points'] = np.asarray(
                    candidate.canonical_points, dtype=np.float64)
                arrays[f'{prefix}_raw_s'] = np.asarray(candidate.raw_s, dtype=np.float64)
                arrays[f'{prefix}_canonical_s'] = np.asarray(
                    candidate.canonical_s, dtype=np.float64)
                item.update({
                    'support_length': float(candidate.support_length),
                    'non_x_monotonic': bool(
                        np.any(np.diff(candidate.raw_ordered_points[:, 0]) > 1e-9)
                        and np.any(np.diff(candidate.raw_ordered_points[:, 0]) < -1e-9)),
                    'near_endpoint': candidate.near_endpoint.tolist(),
                    'far_endpoint': candidate.far_endpoint.tolist(),
                })
            candidates.append(item)

        for side, association in (('left', identity.left), ('right', identity.right)):
            if association is not None:
                arrays[f'{side}_hypothesis_full'] = np.asarray(
                    association.candidate.canonical_points, dtype=np.float64)
                if association.accepted is not None:
                    arrays[f'{side}_accepted'] = np.asarray(
                        association.accepted.canonical_points, dtype=np.float64)
                arrays[f'{side}_rejected'] = rejected_canonical_points(
                    association.candidate, association.accepted)
            state = (identity_tracker.left_state if side == 'left'
                     else identity_tracker.right_state)
            if state.geometry is not None:
                arrays[f'{side}_trusted_long_term'] = np.asarray(
                    state.geometry.canonical_points, dtype=np.float64)
            if state.association_reference is not None:
                arrays[f'{side}_association_reference'] = np.asarray(
                    state.association_reference.canonical_points, dtype=np.float64)
        if identity_tracker.trusted_center is not None:
            arrays['trusted_center'] = np.asarray(
                identity_tracker.trusted_center.points, dtype=np.float64)

        frame_metadata = {
            'schema': 1,
            'scene': self.scene,
            'stamp': {'sec': int(stamp.sec), 'nanosec': int(stamp.nanosec)},
            'observation_mode': single.observation_mode,
            'identity_initialized': bool(identity.identity_initialized),
            'identity_reason': identity.reason,
            'left_hypothesis': self._association_metadata(identity.left),
            'right_hypothesis': self._association_metadata(identity.right),
            'candidates': candidates,
        }
        arrays['metadata_json'] = np.asarray(
            json.dumps(frame_metadata, separators=(',', ':')))
        np.savez_compressed(path, **arrays)
        self._manifest.write(json.dumps({
            'scene': self.scene,
            'stamp': frame_metadata['stamp'],
            'file': name,
            'observation_mode': single.observation_mode,
            'candidate_count': sum(item['geometry_valid'] for item in candidates),
            'left_hypothesis': frame_metadata['left_hypothesis'],
            'right_hypothesis': frame_metadata['right_hypothesis'],
        }, separators=(',', ':')) + '\n')
        return path

    def close(self):
        self._manifest.close()
        self._markers.close()
