"""Vehicle-proximity validation for an observed ORANGE path."""
import numpy as np


def start_distance(path):
    """Return Euclidean distance from base_footprint origin to path start."""
    if path is None or not hasattr(path, 'points') or len(path.points) == 0:
        return None
    point = np.asarray(path.points[0], dtype=float)
    if point.shape != (2,) or not np.all(np.isfinite(point)):
        return None
    return float(np.linalg.norm(point))


def validate_start(path, max_distance):
    """Validate only the first observed path point, never the whole path."""
    distance = start_distance(path)
    if distance is None:
        return False, None, 'NO_START_POINT'
    if max_distance is None or float(max_distance) <= 0.0:
        return True, distance, 'DISABLED'
    if distance > float(max_distance):
        return False, distance, 'START_TOO_FAR'
    return True, distance, 'START_NEAR'
