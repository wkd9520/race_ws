"""Pure 2-D LiDAR geometry helpers for the Stage 5.0 debug overlay."""

import numpy as np


def scan_to_lidar_points(ranges, angle_min, angle_increment,
                         range_min, range_max):
    """Return valid LaserScan samples as XYZ points in the scan frame."""
    values = np.asarray(ranges, dtype=np.float64)
    angles = (float(angle_min)
              + np.arange(len(values), dtype=np.float64)
              * float(angle_increment))
    valid = (np.isfinite(values)
             & (values >= float(range_min))
             & (values <= float(range_max)))
    selected_ranges = values[valid]
    selected_angles = angles[valid]
    points = np.column_stack((
        selected_ranges * np.cos(selected_angles),
        selected_ranges * np.sin(selected_angles),
        np.zeros_like(selected_ranges),
    ))
    return points, valid


def quaternion_matrix_xyzw(quaternion):
    """Return a homogeneous rotation matrix for an XYZW quaternion."""
    q = np.asarray(quaternion, dtype=np.float64).reshape(-1)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError('quaternion must be a finite XYZW vector')
    norm = float(np.linalg.norm(q))
    if norm <= 0.0:
        raise ValueError('zero-norm quaternion')
    x, y, z, w = q / norm
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = np.array([
        [1.0 - 2.0 * (y*y + z*z), 2.0 * (x*y - z*w),
         2.0 * (x*z + y*w)],
        [2.0 * (x*y + z*w), 1.0 - 2.0 * (x*x + z*z),
         2.0 * (y*z - x*w)],
        [2.0 * (x*z - y*w), 2.0 * (y*z + x*w),
         1.0 - 2.0 * (x*x + y*y)],
    ], dtype=np.float64)
    return result


def transform_matrix(translation, quaternion_xyzw):
    """Build target<-source homogeneous transform."""
    translation = np.asarray(translation, dtype=np.float64).reshape(-1)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError('translation must be a finite XYZ vector')
    result = quaternion_matrix_xyzw(quaternion_xyzw)
    result[:3, 3] = translation
    return result


def transform_points(points, target_from_source):
    """Apply a target<-source homogeneous transform to XYZ points."""
    points = np.asarray(points, dtype=np.float64)
    transform = np.asarray(target_from_source, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N,3)')
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError('transform must be a finite 4x4 matrix')
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return (transform @ homogeneous.T).T[:, :3]


def filter_bev_bounds(points, grid):
    """Keep XYZ points within the existing V3 half-open metric grid."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('points must have shape (N,3)')
    inside = ((points[:, 0] >= grid.x_min)
              & (points[:, 0] < grid.x_max)
              & (points[:, 1] >= grid.y_min)
              & (points[:, 1] < grid.y_max))
    return points[inside], inside


def expand_bev_canvas(bev, source_grid, target_grid):
    """Place a metric BEV on a larger, grid-aligned black canvas."""
    image = np.asarray(bev)
    if image.shape[:2] != (source_grid.height, source_grid.width):
        raise ValueError('BEV image shape does not match source grid')
    if not np.isclose(source_grid.resolution, target_grid.resolution):
        raise ValueError('source and target grid resolutions must match')
    if (target_grid.x_min > source_grid.x_min
            or target_grid.x_max < source_grid.x_max
            or target_grid.y_min > source_grid.y_min
            or target_grid.y_max < source_grid.y_max):
        raise ValueError('target grid must contain the source grid')

    resolution = source_grid.resolution
    row_value = (target_grid.x_max - source_grid.x_max) / resolution
    col_value = (target_grid.y_max - source_grid.y_max) / resolution
    row_offset = int(round(row_value))
    col_offset = int(round(col_value))
    if (not np.isclose(row_value, row_offset)
            or not np.isclose(col_value, col_offset)):
        raise ValueError('source grid must align to target grid pixels')

    shape = (target_grid.height, target_grid.width) + image.shape[2:]
    canvas = np.zeros(shape, dtype=image.dtype)
    row_end = row_offset + source_grid.height
    col_end = col_offset + source_grid.width
    if row_end > target_grid.height or col_end > target_grid.width:
        raise ValueError('source grid does not fit target grid pixels')
    canvas[row_offset:row_end, col_offset:col_end] = image
    return canvas, (row_offset, col_offset)
