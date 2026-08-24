"""Decision-neutral STEP 2.1 WHITE continuity runtime capture."""

from pathlib import Path
import json

import cv2
import numpy as np


VARIANTS = (
    'none', 'open3', 'close5', 'close3_open3', 'close5_open3',
    'open3_close7',
)


def _morph(mask, operations):
    result = np.asarray(mask, dtype=np.uint8)
    for operation, size in operations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        result = cv2.morphologyEx(result, operation, kernel, iterations=1)
    return result


def morphology_variants(raw):
    """Analysis-only masks; none of these are fed into perception."""
    return {
        'none': np.asarray(raw, dtype=np.uint8).copy(),
        'open3': _morph(raw, ((cv2.MORPH_OPEN, 3),)),
        'close5': _morph(raw, ((cv2.MORPH_CLOSE, 5),)),
        'close3_open3': _morph(
            raw, ((cv2.MORPH_CLOSE, 3), (cv2.MORPH_OPEN, 3))),
        'close5_open3': _morph(
            raw, ((cv2.MORPH_CLOSE, 5), (cv2.MORPH_OPEN, 3))),
        'open3_close7': _morph(
            raw, ((cv2.MORPH_OPEN, 3), (cv2.MORPH_CLOSE, 7))),
    }


def component_summary(mask):
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        np.asarray(mask > 0, dtype=np.uint8), connectivity=8)
    areas = sorted((int(value) for value in stats[1:, cv2.CC_STAT_AREA]),
                   reverse=True)
    return {
        'component_count': int(count - 1),
        'foreground_pixels': int(np.count_nonzero(mask)),
        'largest_areas': areas[:20],
        'small_component_count_lt8': int(sum(value < 8 for value in areas)),
    }


class WhiteContinuityCapture:
    def __init__(self, directory, stride=5):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stride = max(1, int(stride))
        self.frame_index = 0
        self.records = self.directory / 'capture_index.jsonl'

    def capture(self, stamp, source_bgr, undistorted, bev, validity, stages,
                component_overlay):
        index = self.frame_index
        self.frame_index += 1
        if stages is None or index % self.stride:
            return None
        stem = f'{int(stamp.sec):010d}_{int(stamp.nanosec):09d}'
        variants = morphology_variants(stages.raw)
        path = self.directory / f'{stem}.npz'
        np.savez_compressed(
            path, source_bgr=source_bgr, undistorted=undistorted,
            bev_bgr=bev, hsv=stages.hsv, validity=validity,
            white_raw=stages.raw,
            white_post_validity=stages.post_validity,
            white_after_open=stages.after_open,
            white_after_close=stages.after_close,
            component_overlay=component_overlay,
            **{f'variant_{key}': value for key, value in variants.items()},
        )
        record = {
            'stamp': float(stamp.sec) + 1e-9 * float(stamp.nanosec),
            'file': path.name,
            'production': {
                'raw': component_summary(stages.raw),
                'post_validity_diagnostic': component_summary(stages.post_validity),
                'after_open': component_summary(stages.after_open),
                'after_close_final': component_summary(stages.after_close),
            },
            'analysis_variants': {
                key: component_summary(value) for key, value in variants.items()
            },
        }
        with self.records.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
        return path
