import math

import numpy as np

from physicar_track_perception_v2.geometry import (
    BevGrid, CameraModel, MetricGroundProjector,
    apply_projection_corrections)


def optical_transform(pan):
    # ^base T_optical: optical +Z is camera forward; positive pan rotates it
    # toward vehicle +Y. Fixed optical basis at pan zero is conventional ROS.
    cp, sp = math.cos(pan), math.sin(pan)
    rz = np.array([[cp, -sp, 0.], [sp, cp, 0.], [0., 0., 1.]])
    optical = np.array([[0., 0., 1.], [-1., 0., 0.], [0., -1., 0.]])
    value = np.eye(4)
    value[:3, :3] = rz @ optical
    value[:3, 3] = [.108, 0., .1476]
    return value


def test_positive_pan_view_vector_points_vehicle_left():
    value = optical_transform(math.radians(10.))
    assert value[1, 2] > 0.
    negative = optical_transform(math.radians(-10.))
    assert negative[1, 2] < 0.


def test_dynamic_projectors_recover_same_vehicle_ground_point():
    camera = CameraModel(np.array([[201.39, 0., 240.],
                                   [0., 201.39, 180.], [0., 0., 1.]]),
                         np.zeros(5), 480, 360)
    grid = BevGrid(.1, 2., -.75, .75, .01)
    point = np.array([[1.2, .20]])
    for pan in (0., math.radians(10.), math.radians(-10.)):
        projector = MetricGroundProjector(
            camera, grid, apply_projection_corrections(optical_transform(pan)))
        pixel, valid = projector.vehicle_ground_to_rectified_pixels(point)
        assert valid[0]
        recovered, valid_ray = projector.rectified_pixels_to_ground(pixel)
        assert valid_ray[0]
        assert np.allclose(recovered[0, :2], point[0], atol=1e-9)


def test_height_and_pitch_values_remain_unchanged_for_pan_inputs():
    for pan in (0., .2, -.2):
        original = optical_transform(pan)
        value = apply_projection_corrections(original)
        assert np.isclose(value[2, 3], original[2, 3]-.018)
        # Translation correction must not introduce lateral displacement.
        assert np.allclose(value[:2, 3], original[:2, 3])
