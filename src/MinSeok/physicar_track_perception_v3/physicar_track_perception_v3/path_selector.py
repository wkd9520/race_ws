"""Minimal direct-center path policy (STEP 2; no boundary fallback yet)."""
from dataclasses import dataclass
from .roles import CENTER, LEFT, RIGHT, UNKNOWN, classify, RoleConfig
import numpy as np
from .geometry import tangents

DIRECT_CENTER_OBSERVED = 'DIRECT_CENTER_OBSERVED'
WHITE_HALF_WIDTH_OFFSET = 'WHITE_HALF_WIDTH_OFFSET'
INVALID = 'INVALID'

@dataclass(frozen=True)
class PathResult:
    valid: bool
    path: object
    source: str
    role: str
    reason: str
    stitched_component_ids: tuple = ()
    bridged_gap_count: int = 0

def _stitch_candidates(centers, *, gap_limit=0.30,
           tangent_angle_limit=0.75, lateral_limit=0.15):
    if not centers:
        return PathResult(False, None, INVALID, UNKNOWN, 'center_unobserved_step2')
    # Seed by the closest observed endpoint, never by longest support.
    def orient(component, reference=None):
        points = component.polyline.points
        if reference is None:
            reverse = np.linalg.norm(points[-1]) < np.linalg.norm(points[0])
        else:
            reverse = np.linalg.norm(points[-1]-reference) < np.linalg.norm(points[0]-reference)
        if not reverse:
            return component
        from .geometry import OrderedPolyline
        return type(component)(component.component_id, component.color,
                               OrderedPolyline.from_points(points[::-1]),
                               component.support)
    seed = orient(min(centers, key=lambda c: min(np.linalg.norm(c.polyline.points[0]),
                                                  np.linalg.norm(c.polyline.points[-1]))))
    chain = [seed]
    used = {id(seed)}
    bridges = 0
    while True:
        current = chain[-1].polyline.points
        ta = tangents(current)[-1]
        a = current[-1]
        choices = []
        for candidate in centers:
            if id(candidate) in used:
                continue
            candidate = orient(candidate, a)
            points = candidate.polyline.points
            b = points[0]
            gap = float(np.linalg.norm(b-a))
            tb = tangents(points)[0]
            tangent_delta = float(np.arccos(np.clip(np.dot(ta, tb), -1., 1.)))
            normal = np.asarray([-ta[1], ta[0]])
            lateral = abs(float(np.dot(b-a, normal)))
            longitudinal = float(np.dot(b-a, ta))
            # STEP 2R.1: endpoint distance is the only primary ordering
            # criterion.  Tangent/lateral values remain diagnostics; they do
            # not drop an otherwise usable ORANGE dash.
            choices.append((gap, tangent_delta, lateral, candidate))
        if not choices:
            break
        # A dashed marking is represented by short components.  Keep the
        # nearest continuation only while it remains within the explicit
        # physical dash-gap limit; support length is not a validity gate.
        choices = [item for item in choices if item[0] <= gap_limit]
        if not choices:
            break
        _, _, _, chosen = min(choices, key=lambda item: item[:3])
        chain.append(chosen); used.add(id(chosen)); bridges += 1
    points = []
    for index, item in enumerate(chain):
        current = item.polyline.points
        if index:
            previous = chain[index-1].polyline.points[-1]
            gap = float(np.linalg.norm(current[0]-previous))
            if gap > 1e-9:
                points.extend(np.linspace(previous, current[0], 4)[1:-1])
        points.extend(current)
    from .geometry import OrderedPolyline
    stitched = OrderedPolyline.from_points(np.asarray(points))
    return PathResult(True, stitched, DIRECT_CENTER_OBSERVED, CENTER,
                      'observed_center_stitched',
                      tuple(item.component_id for item in chain), bridges)

def select_orange(components, config=RoleConfig(), *, gap_limit=0.30,
                  tangent_angle_limit=0.75, lateral_limit=0.15):
    """Primary V3 STEP 2R policy: ORANGE is the center-dash source."""
    # ORANGE is the dashed center marking.  Short dashes are expected, so do
    # not apply the generic role minimum-support filter here.  Component
    # extraction has already removed invalid/noisy geometry; assembly uses
    # only near-to-far endpoint distance and the explicit gap limit.
    orange = [item for item in components if item.color == 'ORANGE']
    return _stitch_candidates(orange, gap_limit=gap_limit,
                              tangent_angle_limit=tangent_angle_limit,
                              lateral_limit=lateral_limit)

def select_unknown_white(components, track_width):
    """Create a temporary center path from the nearest observed WHITE line.

    This is deliberately frame-local: no LEFT/RIGHT identity is inferred.
    The vehicle-origin vector selects the interior-facing normal direction.
    """
    whites = [item for item in components if item.color == 'WHITE' and len(item.polyline.points) >= 2]
    if not whites:
        return PathResult(False, None, INVALID, UNKNOWN, 'no_white_boundary')
    def vehicle_distance(item):
        return float(np.min(np.linalg.norm(item.polyline.points, axis=1)))
    boundary = min(
        whites, key=lambda item: (vehicle_distance(item),
                                  int(item.component_id)))
    points = boundary.polyline.points
    # Use a small, evenly spaced representative set rather than every noisy
    # raster point.  The resulting center samples are then smoothed while
    # preserving the observed near/far endpoints.
    if len(points) > 8:
        sample_idx = np.linspace(0, len(points) - 1, 8).round().astype(int)
        sample_idx = np.unique(sample_idx)
    else:
        sample_idx = np.arange(len(points))
    samples = points[sample_idx]
    ts = tangents(points)[sample_idx]
    centers = []
    half = 0.5 * float(track_width)
    previous_offset = None
    for point, tangent in zip(samples, ts):
        normal = np.asarray([-tangent[1], tangent[0]], dtype=float)
        toward_vehicle = -np.asarray(point, dtype=float)
        projected = float(np.dot(toward_vehicle, normal)) * normal
        if np.linalg.norm(projected) < 1e-6 and previous_offset is not None:
            offset_dir = previous_offset
        else:
            offset_dir = projected / max(np.linalg.norm(projected), 1e-12)
            if previous_offset is not None and float(np.dot(offset_dir, previous_offset)) < 0.0:
                offset_dir = -offset_dir
        previous_offset = offset_dir
        centers.append(point + half * offset_dir)
    centers = np.asarray(centers)
    if len(centers) >= 3:
        smoothed = centers.copy()
        smoothed[1:-1] = (centers[:-2] + centers[1:-1] + centers[2:]) / 3.0
        centers = smoothed
    from .geometry import OrderedPolyline
    return PathResult(True, OrderedPolyline.from_points(np.asarray(centers)),
                      WHITE_HALF_WIDTH_OFFSET, UNKNOWN,
                      'unknown_white_vehicle_normal_offset', (boundary.component_id,), 0)

def select(components, config=RoleConfig(), **kwargs):
    """Compatibility wrapper for non-production role experiments."""
    classified = [(classify(c, config), c) for c in components]
    return _stitch_candidates([c for role, c in classified if role == CENTER], **kwargs)
