import numpy as np

from physicar_track_perception_v2.geometry import BevGrid
from physicar_track_perception_v3.lidar_bev import (
    expand_bev_canvas,
    filter_bev_bounds,
    scan_to_lidar_points,
    transform_matrix,
    transform_points,
)


def test_scan_cardinal_angles_and_invalid_ranges():
    points, valid = scan_to_lidar_points(
        [1.0, 1.0, 1.0, np.nan, np.inf, 0.05, 2.1],
        -np.pi / 2.0, np.pi / 2.0, 0.1, 2.0)
    assert valid.tolist() == [True, True, True, False, False, False, False]
    assert np.allclose(points, [[0.0, -1.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0]], atol=1e-12)


def test_transform_translation_and_yaw():
    transform = transform_matrix(
        [1.0, 2.0, 0.0],
        [0.0, 0.0, np.sin(np.pi / 4.0), np.cos(np.pi / 4.0)])
    result = transform_points([[1.0, 0.0, 0.0]], transform)
    assert np.allclose(result, [[1.0, 3.0, 0.0]], atol=1e-12)


def test_scan_to_image_time_motion_compensation_semantics():
    # A stationary point measured 2 m forward at t_scan appears 1 m forward
    # in base_footprint@t_image after the vehicle advances by 1 m in odom.
    target_at_image_from_source_at_scan = transform_matrix(
        [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    result = transform_points(
        [[2.0, 0.0, 0.0]], target_at_image_from_source_at_scan)
    assert np.allclose(result, [[1.0, 0.0, 0.0]])


def test_existing_bev_mapping_preserves_left_right_sign():
    grid = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)
    left_col, row = grid.metric_to_pixel(1.0, 0.5)
    right_col, right_row = grid.metric_to_pixel(1.0, -0.5)
    assert left_col < right_col
    assert np.isclose(row, right_row)
    assert np.isclose(left_col, 24.5)
    assert np.isclose(right_col, 124.5)
    assert np.isclose(row, 99.5)


def test_bev_bounds_are_half_open():
    grid = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)
    points = np.array([
        [0.1, -0.75, 0.0],
        [1.99, 0.74, 0.0],
        [0.099, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [1.0, 0.75, 0.0],
    ])
    selected, inside = filter_bev_bounds(points, grid)
    assert inside.tolist() == [True, True, False, False, False]
    assert np.allclose(selected, points[:2])


def test_expanded_overlay_embeds_existing_bev_at_metric_location():
    source = BevGrid(0.0, 2.0, -1.0, 1.0, 1.0)
    target = BevGrid(-1.0, 3.0, -2.0, 2.0, 1.0)
    bev = np.array([[1, 2], [3, 4]], dtype=np.uint8)
    canvas, offset = expand_bev_canvas(bev, source, target)
    assert canvas.shape == (4, 4)
    assert offset == (1, 1)
    assert np.array_equal(canvas[1:3, 1:3], bev)
    assert np.count_nonzero(canvas) == 4


def test_expanded_overlay_includes_points_before_camera_bev():
    camera_grid = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)
    overlay_grid = BevGrid(-0.5, 4.0, -2.0, 2.0, 0.01)
    points = np.array([
        [3.0, 0.0, 0.0],
        [1.0, 1.5, 0.0],
        [-0.25, 0.0, 0.0],
    ])
    camera_points, _ = filter_bev_bounds(points, camera_grid)
    overlay_points, _ = filter_bev_bounds(points, overlay_grid)
    assert len(camera_points) == 0
    assert np.array_equal(overlay_points, points)
    cols, rows = overlay_grid.metric_to_pixel(
        overlay_points[:, 0], overlay_points[:, 1])
    assert np.all((cols >= 0) & (cols < overlay_grid.width))
    assert np.all((rows >= 0) & (rows < overlay_grid.height))
