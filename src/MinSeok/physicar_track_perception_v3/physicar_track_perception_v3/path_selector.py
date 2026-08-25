"""Minimal direct-center path policy (STEP 2; no boundary fallback yet)."""
from dataclasses import dataclass
from .roles import CENTER, LEFT, RIGHT, UNKNOWN, classify, RoleConfig
import numpy as np
from .geometry import cumulative_s, tangents

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
    # Orientation may create a temporary Component object.  Track the stable
    # component ID, otherwise a reversed original can be selected repeatedly.
    used = {seed.component_id}
    bridges = 0
    while True:
        current = chain[-1].polyline.points
        ta = tangents(current)[-1]
        a = current[-1]
        choices = []
        for candidate in centers:
            if candidate.component_id in used:
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
        chain.append(chosen); used.add(chosen.component_id); bridges += 1
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

def _windowed_tangents(points, half_window=0.15):
    """Return ordered secant tangents without trusting one endpoint edge."""
    points = np.asarray(points, dtype=float)
    arc = cumulative_s(points)
    output = []
    previous = None
    for index, value in enumerate(arc):
        begin = int(np.searchsorted(
            arc, value - float(half_window), side='left'))
        end = int(np.searchsorted(
            arc, value + float(half_window), side='right') - 1)
        if begin == end:
            begin = max(0, index - 1)
            end = min(len(points) - 1, index + 1)
        direction = points[end] - points[begin]
        length = float(np.linalg.norm(direction))
        if length <= 1e-12:
            direction = tangents(points)[index]
        else:
            direction = direction / length
        # Component graph order is authoritative.  Prevent only a local
        # 180-degree tangent-axis flip; do not sort in global X/Y.
        if previous is not None and float(np.dot(direction, previous)) < 0.0:
            direction = -direction
        output.append(direction)
        previous = direction
    return np.asarray(output, dtype=float)


def _nearest_polyline_distances(points, reference):
    points = np.asarray(points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    segments = np.diff(reference, axis=0)
    length_squared = np.einsum('ij,ij->i', segments, segments)
    result = []
    for point in points:
        fraction = np.einsum(
            'ij,ij->i', point - reference[:-1], segments)
        fraction = np.divide(
            fraction, length_squared, out=np.zeros_like(fraction),
            where=length_squared > 1e-12)
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = reference[:-1] + fraction[:, None] * segments
        result.append(float(np.min(np.linalg.norm(
            projected - point, axis=1))))
    return np.asarray(result, dtype=float)


def select_unknown_white(components, track_width, reference_path=None):
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
    ts = _windowed_tangents(points)[sample_idx]
    half = 0.5 * float(track_width)
    normals = np.column_stack((-ts[:, 1], ts[:, 0]))
    plus = samples + half * normals
    minus = samples - half * normals

    # Priority 1: when current ORANGE overlaps this boundary, select the
    # complete +/-W/2 hypothesis that agrees with the observed center.  The
    # 0.30 m overlap gate reuses the current ORANGE fragment join scale.
    selected = None
    reason = None
    if reference_path is not None:
        reference = np.asarray(
            getattr(reference_path, 'points', reference_path), dtype=float)
        if (reference.ndim == 2 and reference.shape[1] == 2
                and len(reference) >= 2 and np.all(np.isfinite(reference))):
            plus_distance = _nearest_polyline_distances(plus, reference)
            minus_distance = _nearest_polyline_distances(minus, reference)
            overlap = np.minimum(plus_distance, minus_distance) <= 0.30
            if int(np.count_nonzero(overlap)) >= 2:
                plus_score = float(np.median(plus_distance[overlap]))
                minus_score = float(np.median(minus_distance[overlap]))
                if abs(plus_score - minus_score) > 1e-6:
                    selected = plus if plus_score < minus_score else minus
                    reason = 'unknown_white_orange_reference_offset'

    # Priority 2: without a usable ORANGE overlap, use a component-level
    # median.  One noisy endpoint can no longer seed and reverse every later
    # normal as happened with the old previous_offset propagation.
    signed_vehicle_side = np.einsum('ij,ij->i', -samples, normals)
    if selected is None:
        median_side = float(np.median(signed_vehicle_side))
        if abs(median_side) > 1e-3:
            selected = plus if median_side > 0.0 else minus
            reason = 'unknown_white_vehicle_median_offset'

    # Priority 3: user-requested non-empty fallback.  Pick the most
    # informative interior sample and force the normal toward the vehicle.
    # LEFT/RIGHT identity is deliberately not required here.
    if selected is None:
        candidates = (np.arange(1, len(samples) - 1)
                      if len(samples) > 2 else np.arange(len(samples)))
        representative = int(candidates[np.argmax(
            np.abs(signed_vehicle_side[candidates]))])
        side = float(signed_vehicle_side[representative])
        if abs(side) <= 1e-12:
            plus_distance = float(np.linalg.norm(plus[representative]))
            minus_distance = float(np.linalg.norm(minus[representative]))
            side = 1.0 if plus_distance <= minus_distance else -1.0
        selected = plus if side > 0.0 else minus
        reason = 'unknown_white_vehicle_forced_offset'

    centers = np.asarray(selected, dtype=float)
    if len(centers) >= 3:
        smoothed = centers.copy()
        smoothed[1:-1] = (centers[:-2] + centers[1:-1] + centers[2:]) / 3.0
        centers = smoothed
    from .geometry import OrderedPolyline
    return PathResult(True, OrderedPolyline.from_points(np.asarray(centers)),
                      WHITE_HALF_WIDTH_OFFSET, UNKNOWN,
                      reason, (boundary.component_id,), 0)

def select(components, config=RoleConfig(), **kwargs):
    """Compatibility wrapper for non-production role experiments."""
    classified = [(classify(c, config), c) for c in components]
    return _stitch_candidates([c for role, c in classified if role == CENTER], **kwargs)
