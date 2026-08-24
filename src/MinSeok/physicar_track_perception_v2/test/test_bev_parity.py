import math

import numpy as np

from physicar_track_perception.geometry import (
    BevGrid as LegacyGrid,
    CameraModel as LegacyCamera,
    MetricGroundProjector as LegacyProjector,
)
from physicar_track_perception_v2.frontend import BevFrontend
from physicar_track_perception_v2.geometry import (
    BevGrid,
    CameraModel,
    MetricGroundProjector,
    apply_projection_corrections,
)


K = np.array([
    [201.38988018035889, 0.0, 240.0],
    [0.0, 201.38988733291626, 180.0],
    [0.0, 0.0, 1.0],
])
D = np.array([-0.045, -0.0001, -0.0003, -0.0001, 0.001])


def _grid(cls=BevGrid):
    return cls(0.10, 2.00, -0.75, 0.75, 0.01)


def _camera(cls=CameraModel):
    return cls(K, D, 480, 360)


def _rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rotation_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _corrected_camera_tf(pan=0.0, joint_tilt=-0.5236):
    pan_rotation = _rotation_z(pan)
    tilt_rotation = _rotation_y(-joint_tilt)
    optical = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], dtype=float)
    result = np.eye(4)
    result[:3, :3] = pan_rotation @ tilt_rotation @ optical
    result[:3, 3] = (
        np.array([0.05, 0, 0.1375])
        + pan_rotation @ np.array([0.025, 0, 0.013])
        + pan_rotation @ tilt_rotation @ np.array([0.030, 0, 0.014])
    )
    return result


def _legacy_projection_corrections(transform):
    result = transform.copy()
    result[:3, 3] += [0, 0, -0.018]
    result[:3, :3] = _rotation_y(math.radians(2.7)) @ result[:3, :3]
    return result


def test_existing_bev_bounds_resolution_and_dimensions_parity():
    new, old = _grid(), _grid(LegacyGrid)
    assert (new.x_min, new.x_max, new.y_min, new.y_max, new.resolution) == (
        old.x_min, old.x_max, old.y_min, old.y_max, old.resolution
    )
    assert (new.width, new.height) == (old.width, old.height) == (150, 190)


def test_pixel_metric_mapping_is_numerically_identical():
    col = np.array([0.0, 74.5, 149.0])
    row = np.array([0.0, 94.5, 189.0])
    assert np.array_equal(_grid().pixel_to_metric(col, row), _grid(LegacyGrid).pixel_to_metric(col, row))
    x, y = np.array([0.105, 0.8, 1.995]), np.array([-0.745, 0, 0.745])
    assert np.array_equal(_grid().metric_to_pixel(x, y), _grid(LegacyGrid).metric_to_pixel(x, y))


def test_pitch_offset_2p7_degree_sign_and_application_semantics():
    source = _corrected_camera_tf()
    actual = apply_projection_corrections(source)
    expected = _legacy_projection_corrections(source)
    assert np.allclose(actual, expected, atol=1e-15)
    assert np.array_equal(actual[:3, 3], source[:3, 3] + [0, 0, -0.018])


def test_projection_tolerance_and_horizon_behavior_parity():
    transform = np.eye(4)
    transform[:3, :3] = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
    transform[:3, 3] = [0.1, 0, 0.15]
    new = MetricGroundProjector(_camera(), _grid(), transform)
    old = LegacyProjector(_camera(LegacyCamera), _grid(LegacyGrid), transform)
    pixels = np.array([[240, 180], [0, 0], [479, 359]], dtype=float)
    new_points, new_valid = new.rectified_pixels_to_ground(pixels)
    old_points, old_valid = old.rectified_pixels_to_ground(pixels)
    assert np.array_equal(new_valid, old_valid)
    assert np.allclose(new_points, old_points, equal_nan=True)


def test_representative_camera_transform_source_maps_match_legacy():
    transform = apply_projection_corrections(_corrected_camera_tf())
    new = MetricGroundProjector(_camera(), _grid(), transform).build_bev_source_map()
    old = LegacyProjector(
        _camera(LegacyCamera), _grid(LegacyGrid), transform
    ).build_bev_source_map()
    for new_value, old_value in zip(new, old):
        assert np.array_equal(new_value, old_value)


def test_nonzero_pan_and_tilt_frontend_parity():
    for pan, tilt in ((0.3, -0.5236), (-0.2, -0.35), (0.15, 0.2)):
        transform = apply_projection_corrections(_corrected_camera_tf(pan, tilt))
        new = MetricGroundProjector(_camera(), _grid(), transform)
        old = LegacyProjector(_camera(LegacyCamera), _grid(LegacyGrid), transform)
        points = np.array([[0.3, 0], [0.8, 0.2], [1.5, -0.4]])
        new_pixels, new_valid = new.vehicle_ground_to_rectified_pixels(points)
        old_pixels, old_valid = old.vehicle_ground_to_rectified_pixels(points)
        assert np.array_equal(new_valid, old_valid)
        assert np.allclose(new_pixels, old_pixels, equal_nan=True)


def test_invalid_map_sentinel_and_validity_parity():
    transform = apply_projection_corrections(_corrected_camera_tf())
    map_x, map_y, valid = MetricGroundProjector(
        _camera(), _grid(), transform
    ).build_bev_source_map()
    assert np.any(valid) and np.any(~valid)
    assert np.all(map_x[~valid] == -1.0)
    assert np.all(map_y[~valid] == -1.0)


def test_bev_frontend_output_dimensions_and_validity_mask():
    transform = apply_projection_corrections(_corrected_camera_tf())
    frontend = BevFrontend(
        _camera(), MetricGroundProjector(_camera(), _grid(), transform)
    )
    image = np.zeros((360, 480, 3), dtype=np.uint8)
    output = frontend.process(image)
    assert output.undistorted.shape == (360, 480, 3)
    assert output.bev.shape == (190, 150, 3)
    assert output.validity_mask.shape == (190, 150)
    assert set(np.unique(output.validity_mask)) <= {0, 255}


def test_ground_projection_roundtrip_matches_metric_points():
    transform = apply_projection_corrections(_corrected_camera_tf())
    projector = MetricGroundProjector(_camera(), _grid(), transform)
    points = np.array([[0.35, 0], [0.6, 0.1], [1.2, -0.2]])
    pixels, valid = projector.vehicle_ground_to_rectified_pixels(points)
    recovered, ray_valid = projector.rectified_pixels_to_ground(pixels)
    usable = valid & ray_valid
    assert np.any(usable)
    assert np.allclose(recovered[usable, :2], points[usable], atol=1e-9)


def test_old_new_frontend_numerical_comparison_across_ground_samples():
    transform = apply_projection_corrections(_corrected_camera_tf())
    new = MetricGroundProjector(_camera(), _grid(), transform)
    old = LegacyProjector(_camera(LegacyCamera), _grid(LegacyGrid), transform)
    x = np.linspace(0.1, 2.0, 20)
    y = np.linspace(-0.75, 0.75, 17)
    xx, yy = np.meshgrid(x, y)
    points = np.column_stack((xx.ravel(), yy.ravel()))
    new_pixels, new_valid = new.vehicle_ground_to_rectified_pixels(points)
    old_pixels, old_valid = old.vehicle_ground_to_rectified_pixels(points)
    assert np.array_equal(new_valid, old_valid)
    assert np.allclose(new_pixels, old_pixels, equal_nan=True, atol=1e-12)
