"""Legacy-parity HSV masks plus V2 component geometry extraction."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from .components import COLORS, ORANGE, WHITE


@dataclass(frozen=True)
class HsvRange:
    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass(frozen=True)
class SegmentationOutput:
    white_mask: np.ndarray
    orange_mask: np.ndarray
    combined_mask: np.ndarray
    component_frame: object
    overlay: np.ndarray
    white_stages: Optional[object] = None


@dataclass(frozen=True)
class WhiteMaskStages:
    """Decision-neutral snapshots of the exact WHITE mask pipeline.

    ``post_validity`` is intentionally diagnostic-only.  Production applies
    validity after connected-component labelling, while selecting component
    pixels; it does not feed this image into morphology or labelling.
    """

    hsv: np.ndarray
    raw: np.ndarray
    post_validity: np.ndarray
    after_open: np.ndarray
    after_close: np.ndarray


class ColorComponentPipeline:
    def __init__(self, ranges, open_kernel, close_kernel, extractor):
        self.ranges = {color: tuple(ranges[color]) for color in COLORS}
        self.open_kernel = self._kernel_size(open_kernel)
        self.close_kernel = self._kernel_size(close_kernel)
        self.extractor = extractor

    @staticmethod
    def _kernel_size(value):
        value = int(value)
        if value <= 1:
            return 0
        return value if value % 2 else value + 1

    def process(self, bev, valid_map, include_white_stages=False):
        hsv = cv2.cvtColor(bev, cv2.COLOR_BGR2HSV)
        masks = {}
        white_stages = None
        for color in COLORS:
            mask = np.zeros(bev.shape[:2], dtype=np.uint8)
            for value in self.ranges[color]:
                mask = cv2.bitwise_or(mask, cv2.inRange(
                    hsv, np.asarray(value.lower, np.uint8),
                    np.asarray(value.upper, np.uint8),
                ))
            if color == WHITE and include_white_stages:
                after_open = self._operation(mask, cv2.MORPH_OPEN,
                                             self.open_kernel)
                after_close = self._operation(after_open, cv2.MORPH_CLOSE,
                                              self.close_kernel)
                white_stages = WhiteMaskStages(
                    hsv=hsv.copy(), raw=mask.copy(),
                    post_validity=cv2.bitwise_and(
                        mask, np.asarray(valid_map, dtype=np.uint8) * 255),
                    after_open=after_open.copy(),
                    after_close=after_close.copy(),
                )
                masks[color] = after_close
            else:
                masks[color] = self._morph(mask)
        frame = self.extractor.extract(masks, valid_map)
        overlay = self.draw_overlay(bev, frame)
        return SegmentationOutput(
            masks[WHITE], masks[ORANGE],
            cv2.bitwise_or(masks[WHITE], masks[ORANGE]), frame, overlay,
            white_stages,
        )

    def _morph(self, mask):
        result = mask
        for operation, size in ((cv2.MORPH_OPEN, self.open_kernel),
                                (cv2.MORPH_CLOSE, self.close_kernel)):
            result = self._operation(result, operation, size)
        return result

    @staticmethod
    def _operation(mask, operation, size):
        if not size:
            return mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        return cv2.morphologyEx(mask, operation, kernel, iterations=1)

    def draw_overlay(self, bev, frame):
        overlay = bev.copy()
        for observation in frame.observations:
            metadata = observation.metadata
            base_color = (240, 240, 240) if metadata.color == WHITE else (0, 140, 255)
            for row, col in metadata.raw_pixels_rc[::max(1, len(metadata.raw_pixels_rc)//300)]:
                overlay[int(row), int(col)] = base_color
            candidate = observation.candidate
            if candidate is None:
                continue
            raw_pixels = self._metric_pixels(candidate.raw_ordered_points)
            canonical_pixels = self._metric_pixels(candidate.canonical_points)
            cv2.polylines(overlay, [raw_pixels], False, base_color, 1)
            canonical_color = (255, 255, 0) if metadata.color == WHITE else (255, 0, 255)
            cv2.polylines(overlay, [canonical_pixels], False, canonical_color, 2)
            cv2.circle(overlay, tuple(canonical_pixels[0]), 4, (0, 255, 0), -1)
            cv2.circle(overlay, tuple(canonical_pixels[-1]), 4, (0, 0, 255), -1)
        return overlay

    def _metric_pixels(self, points):
        col, row = self.extractor.grid.metric_to_pixel(points[:, 0], points[:, 1])
        return np.rint(np.column_stack((col, row))).astype(np.int32)
