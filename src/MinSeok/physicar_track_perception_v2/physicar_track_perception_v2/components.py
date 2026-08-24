"""Color-provenance components and canonical ordered boundary candidates."""

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np


WHITE = 'WHITE'
ORANGE = 'ORANGE'
COLORS = (WHITE, ORANGE)
DEFAULT_CANONICAL_SPACING = 0.05


def _frozen(values, dtype=None):
    result = np.asarray(values, dtype=dtype).copy()
    result.setflags(write=False)
    return result


def cumulative_arc_length(points):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError('ordered points must have shape (N,2), N>=2')
    if not np.all(np.isfinite(points)):
        raise ValueError('ordered points must be finite')
    return np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))


@dataclass(frozen=True)
class ComponentMetadata:
    component_id: int
    color: str
    area_pixels: int
    valid_area_pixels: int
    pixel_bbox_xywh: tuple[int, int, int, int]
    valid_overlap: float
    metric_bbox_xyxy: Optional[tuple[float, float, float, float]]
    metric_extent_xy: tuple[float, float]
    raw_pixels_rc: np.ndarray
    extracted: bool
    geometry_valid: bool
    canonicalizable: bool
    rejection_reason: str

    def __post_init__(self):
        object.__setattr__(self, 'raw_pixels_rc', _frozen(self.raw_pixels_rc, np.int32))


@dataclass(frozen=True)
class CanonicalBoundaryCandidate:
    component_id: int
    color: str
    raw_ordered_points: np.ndarray
    raw_s: np.ndarray
    canonical_points: np.ndarray
    canonical_s: np.ndarray
    support_length: float
    raw_point_count: int
    canonical_point_count: int
    raw_spacing_min: float
    raw_spacing_median: float
    raw_spacing_max: float
    canonical_spacing: float
    near_endpoint: np.ndarray
    far_endpoint: np.ndarray

    def __post_init__(self):
        for name in ('raw_ordered_points', 'raw_s', 'canonical_points',
                     'canonical_s', 'near_endpoint', 'far_endpoint'):
            object.__setattr__(self, name, _frozen(getattr(self, name)))


@dataclass(frozen=True)
class ComponentObservation:
    metadata: ComponentMetadata
    candidate: Optional[CanonicalBoundaryCandidate]


@dataclass(frozen=True)
class ComponentFrame:
    observations: tuple[ComponentObservation, ...]
    filtered_masks: dict[str, np.ndarray]

    @property
    def candidates(self):
        return tuple(item.candidate for item in self.observations if item.candidate is not None)


@dataclass(frozen=True)
class ComponentExtractionConfig:
    min_component_area: int = 8
    min_valid_pixels: int = 3
    min_valid_overlap: float = 0.70
    canonical_spacing: float = DEFAULT_CANONICAL_SPACING
    duplicate_tolerance: float = 1e-9

    def __post_init__(self):
        if self.min_component_area < 1 or self.min_valid_pixels < 1:
            raise ValueError('component pixel thresholds must be positive')
        if not 0.0 <= self.min_valid_overlap <= 1.0:
            raise ValueError('min_valid_overlap must be in [0,1]')
        if self.canonical_spacing <= 0.0 or self.duplicate_tolerance < 0.0:
            raise ValueError('canonical spacing/tolerance is invalid')


class CanonicalComponentExtractor:
    """Create topology-ordered candidates without track-role assumptions."""

    def __init__(self, grid, config=ComponentExtractionConfig()):
        self.grid = grid
        self.config = config

    def extract(self, masks, valid_map):
        valid = np.asarray(valid_map, dtype=bool)
        if valid.shape != (self.grid.height, self.grid.width):
            raise ValueError('BEV valid map shape mismatch')
        observations = []
        filtered = {}
        for color in COLORS:
            mask = np.asarray(masks[color], dtype=np.uint8)
            if mask.shape != valid.shape:
                raise ValueError(f'{color} mask shape mismatch')
            color_observations, color_filtered = self._extract_color(mask, valid, color)
            observations.extend(color_observations)
            filtered[color] = color_filtered
        return ComponentFrame(tuple(observations), filtered)

    def _extract_color(self, mask, valid_map, color):
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.asarray(mask > 0, dtype=np.uint8), connectivity=8
        )
        observations = []
        filtered = np.zeros_like(mask)
        for component_id in range(1, count):
            rows, cols = np.nonzero(labels == component_id)
            area = int(len(rows))
            inside = valid_map[rows, cols]
            valid_rows, valid_cols = rows[inside], cols[inside]
            valid_area = int(len(valid_rows))
            overlap = valid_area / max(area, 1)
            bbox = tuple(int(value) for value in stats[component_id, :4])
            metric_bbox = None
            extent = (0.0, 0.0)
            if valid_area:
                x, y = self.grid.pixel_to_metric(valid_cols, valid_rows)
                metric_bbox = (float(np.min(x)), float(np.min(y)),
                               float(np.max(x)), float(np.max(y)))
                extent = (float(np.ptp(x) + self.grid.resolution),
                          float(np.ptp(y) + self.grid.resolution))
            reason = self._noise_rejection_reason(area, valid_area, overlap)
            candidate = None
            geometry_reason = reason
            if not reason:
                filtered[valid_rows, valid_cols] = 255
                raw = self._ordered_geodesic_polyline(valid_rows, valid_cols)
                candidate, geometry_reason = self.canonicalize_ordered_points(
                    raw, component_id=component_id, color=color
                )
            metadata = ComponentMetadata(
                component_id=component_id, color=color, area_pixels=area,
                valid_area_pixels=valid_area, pixel_bbox_xywh=bbox,
                valid_overlap=float(overlap), metric_bbox_xyxy=metric_bbox,
                metric_extent_xy=extent,
                raw_pixels_rc=np.column_stack((rows, cols)), extracted=True,
                geometry_valid=candidate is not None,
                canonicalizable=candidate is not None,
                rejection_reason=geometry_reason,
            )
            observations.append(ComponentObservation(metadata, candidate))
        return observations, filtered

    def _noise_rejection_reason(self, area, valid_area, overlap):
        if area < self.config.min_component_area:
            return 'area_small'
        if valid_area < self.config.min_valid_pixels:
            return 'valid_area_small'
        if overlap < self.config.min_valid_overlap:
            return 'valid_overlap'
        return ''

    def _ordered_geodesic_polyline(self, rows, cols):
        pixels = {(int(row), int(col)) for row, col in zip(rows, cols)}
        if len(pixels) < 2:
            return np.empty((0, 2), dtype=np.float64)

        def farthest(start, with_parent=False):
            queue = [start]
            distance = {start: 0}
            parent = {} if with_parent else None
            for current in queue:
                row, col = current
                neighbours = sorted(
                    (row + dr, col + dc)
                    for dr in (-1, 0, 1) for dc in (-1, 0, 1)
                    if (dr or dc) and (row + dr, col + dc) in pixels
                )
                for neighbour in neighbours:
                    if neighbour in distance:
                        continue
                    distance[neighbour] = distance[current] + 1
                    if parent is not None:
                        parent[neighbour] = current
                    queue.append(neighbour)
            maximum = max(distance.values())
            endpoint = min(point for point, value in distance.items() if value == maximum)
            return endpoint, parent

        seed = min(pixels)
        first, _ = farthest(seed)
        second, parent = farthest(first, with_parent=True)
        path = [second]
        while path[-1] != first:
            path.append(parent[path[-1]])
        path.reverse()
        path = np.asarray(path, dtype=np.int32)
        x, y = self.grid.pixel_to_metric(path[:, 1], path[:, 0])
        return self._orient_near_to_far(np.column_stack((x, y)))

    @staticmethod
    def _endpoint_key(point):
        return (float(np.linalg.norm(point)), float(point[0]),
                float(abs(point[1])), float(point[1]))

    @classmethod
    def _orient_near_to_far(cls, points):
        result = np.asarray(points, dtype=np.float64)
        return result[::-1].copy() if cls._endpoint_key(result[-1]) < cls._endpoint_key(result[0]) else result.copy()

    def canonicalize_ordered_points(self, raw, component_id=0, color=WHITE):
        """Canonicalize an already topology-ordered metric observation.

        This public boundary is also used by characterization tests.  It does
        not infer identity or track-side role; ``color`` remains provenance.
        """
        if color not in COLORS:
            return None, 'color_provenance_invalid'
        try:
            points = np.asarray(raw, dtype=np.float64)
            if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
                return None, 'ordered_geometry_short'
            if not np.all(np.isfinite(points)):
                return None, 'ordered_geometry_nonfinite'
            kept = [points[0]]
            for point in points[1:]:
                if np.linalg.norm(point - kept[-1]) > self.config.duplicate_tolerance:
                    kept.append(point)
            if len(kept) < 2:
                return None, 'ordered_geometry_degenerate'
            ordered = self._orient_near_to_far(np.asarray(kept))
            raw_s = cumulative_arc_length(ordered)
            support = float(raw_s[-1])
            if support <= self.config.duplicate_tolerance:
                return None, 'ordered_geometry_degenerate'
            targets = np.arange(
                0.0, support + 1e-12, self.config.canonical_spacing,
                dtype=np.float64,
            )
            if len(targets) == 0 or targets[-1] < support - 1e-12:
                targets = np.append(targets, support)
            else:
                targets[-1] = support
            canonical = np.column_stack((
                np.interp(targets, raw_s, ordered[:, 0]),
                np.interp(targets, raw_s, ordered[:, 1]),
            ))
            spacing = np.diff(raw_s)
            return CanonicalBoundaryCandidate(
                component_id=component_id, color=color,
                raw_ordered_points=ordered, raw_s=raw_s,
                canonical_points=canonical, canonical_s=targets,
                support_length=support, raw_point_count=len(ordered),
                canonical_point_count=len(canonical),
                raw_spacing_min=float(np.min(spacing)),
                raw_spacing_median=float(np.median(spacing)),
                raw_spacing_max=float(np.max(spacing)),
                canonical_spacing=float(self.config.canonical_spacing),
                near_endpoint=canonical[0], far_endpoint=canonical[-1],
            ), 'valid'
        except (ValueError, FloatingPointError):
            return None, 'canonicalization_error'
