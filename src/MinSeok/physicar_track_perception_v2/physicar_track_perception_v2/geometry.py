"""ROS-independent verified metric camera/ground-plane geometry."""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray
    D: np.ndarray
    width: int
    height: int

    def __post_init__(self):
        k = np.asarray(self.K, dtype=np.float64)
        d = np.asarray(self.D, dtype=np.float64).reshape(-1)
        if k.shape != (3, 3) or not np.all(np.isfinite(k)):
            raise ValueError('K must be a finite 3x3 matrix')
        if d.size not in (4, 5, 8, 12, 14) or not np.all(np.isfinite(d)):
            raise ValueError('D must be a finite OpenCV distortion vector')
        if self.width <= 0 or self.height <= 0:
            raise ValueError('camera dimensions must be positive')
        object.__setattr__(self, 'K', k)
        object.__setattr__(self, 'D', d)


@dataclass(frozen=True)
class BevGrid:
    """Metric BEV in base_footprint: +X forward, +Y left."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    resolution: float

    def __post_init__(self):
        values = np.asarray([
            self.x_min, self.x_max, self.y_min, self.y_max, self.resolution
        ])
        if not np.all(np.isfinite(values)):
            raise ValueError('BEV limits and resolution must be finite')
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError('BEV maximum limits must exceed minimum limits')
        if self.resolution <= 0.0:
            raise ValueError('BEV resolution must be positive')

    @property
    def width(self):
        return int(np.ceil((self.y_max - self.y_min) / self.resolution))

    @property
    def height(self):
        return int(np.ceil((self.x_max - self.x_min) / self.resolution))

    def metric_to_pixel(self, x, y) -> Tuple[np.ndarray, np.ndarray]:
        col = (self.y_max - np.asarray(y, dtype=np.float64)) / self.resolution - 0.5
        row = (self.x_max - np.asarray(x, dtype=np.float64)) / self.resolution - 0.5
        return col, row

    def pixel_to_metric(self, col, row) -> Tuple[np.ndarray, np.ndarray]:
        y = self.y_max - (np.asarray(col, dtype=np.float64) + 0.5) * self.resolution
        x = self.x_max - (np.asarray(row, dtype=np.float64) + 0.5) * self.resolution
        return x, y

    def centre_mesh(self):
        rows, cols = np.indices((self.height, self.width), dtype=np.float64)
        return self.pixel_to_metric(cols, rows)


def validate_transform(transform):
    result = np.asarray(transform, dtype=np.float64)
    if result.shape != (4, 4) or not np.all(np.isfinite(result)):
        raise ValueError('T_vehicle_camera must be a finite 4x4 matrix')
    if not np.allclose(result[3], [0, 0, 0, 1], atol=1e-9):
        raise ValueError('invalid homogeneous transform last row')
    rotation = result[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
        raise ValueError('transform rotation is not orthonormal')
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
        raise ValueError('transform rotation determinant must be +1')
    return result


def apply_vehicle_translation_correction(transform, translation_vehicle):
    correction = np.eye(4)
    translation = np.asarray(translation_vehicle, dtype=np.float64).reshape(-1)
    if translation.shape != (3,) or not np.all(np.isfinite(translation)):
        raise ValueError('translation_vehicle must be a finite 3-vector')
    correction[:3, 3] = translation
    return correction @ validate_transform(transform)


def apply_projection_corrections(
    transform,
    *,
    camera_height_correction_z=-0.018,
    pitch_offset_deg=2.7,
    pitch_correction_frame='vehicle_y',
):
    """Apply live Stage 3 corrections with identical signs and frame semantics."""

    corrected = apply_vehicle_translation_correction(
        transform, [0.0, 0.0, float(camera_height_correction_z)]
    )
    if not math.isfinite(pitch_offset_deg):
        raise ValueError('pitch_offset_deg must be finite')
    angle = math.radians(float(pitch_offset_deg))
    c, s = math.cos(angle), math.sin(angle)
    rotation_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)
    if pitch_correction_frame == 'vehicle_y':
        correction = rotation_y
    elif pitch_correction_frame == 'pan_local_y':
        # Exact-stamp ^base R_optical has optical +Z as its viewing direction.
        # Its horizontal azimuth is the actual pan pose represented by this TF,
        # not the command or latest JointState value.
        view = corrected[:3, 2]
        horizontal = math.hypot(float(view[0]), float(view[1]))
        if horizontal <= 1e-12:
            raise ValueError('camera pan is undefined for vertical viewing axis')
        pan = math.atan2(float(view[1]), float(view[0]))
        cp, sp = math.cos(pan), math.sin(pan)
        rotation_z = np.array(
            [[cp, -sp, 0], [sp, cp, 0], [0, 0, 1]], dtype=np.float64)
        correction = rotation_z @ rotation_y @ rotation_z.T
    else:
        raise ValueError(
            'pitch_correction_frame must be vehicle_y or pan_local_y')
    corrected = corrected.copy()
    corrected[:3, :3] = correction @ corrected[:3, :3]
    return corrected


class MetricGroundProjector:
    def __init__(self, camera, grid, T_vehicle_camera, ground_z=0.0):
        self.camera = camera
        self.grid = grid
        self.T_vehicle_camera = validate_transform(T_vehicle_camera).copy()
        self.ground_z = float(ground_z)
        if not np.isfinite(self.ground_z):
            raise ValueError('ground_z must be finite')
        self.R_vehicle_camera = self.T_vehicle_camera[:3, :3]
        self.t_vehicle_camera = self.T_vehicle_camera[:3, 3]
        self.R_camera_vehicle = self.R_vehicle_camera.T
        self.K_inv = np.linalg.inv(camera.K)

    def rectified_pixels_to_ground(self, pixels_uv):
        pixels = np.asarray(pixels_uv, dtype=np.float64)
        if pixels.ndim != 2 or pixels.shape[1] != 2:
            raise ValueError('pixels_uv must have shape (N, 2)')
        rays_camera = (self.K_inv @ np.column_stack((pixels, np.ones(len(pixels)))).T).T
        rays_vehicle = (self.R_vehicle_camera @ rays_camera.T).T
        dz = rays_vehicle[:, 2]
        scale = np.full(len(pixels), np.nan)
        nonparallel = np.abs(dz) > 1e-12
        scale[nonparallel] = (self.ground_z - self.t_vehicle_camera[2]) / dz[nonparallel]
        points = self.t_vehicle_camera + scale[:, None] * rays_vehicle
        valid = nonparallel & np.isfinite(scale) & (scale > 0.0)
        points[~valid] = np.nan
        return points, valid

    def vehicle_ground_to_rectified_pixels(self, points_xy):
        points = np.asarray(points_xy, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError('points_xy must have shape (N, 2)')
        vehicle = np.column_stack((points, np.full(len(points), self.ground_z)))
        camera = (self.R_camera_vehicle @ (vehicle - self.t_vehicle_camera).T).T
        depth = camera[:, 2]
        projected = (self.camera.K @ camera.T).T
        pixels = np.full((len(points), 2), np.nan)
        in_front = depth > 1e-9
        pixels[in_front] = projected[in_front, :2] / depth[in_front, None]
        valid = (in_front & (pixels[:, 0] >= 0) & (pixels[:, 0] < self.camera.width)
                 & (pixels[:, 1] >= 0) & (pixels[:, 1] < self.camera.height))
        return pixels, valid

    def build_bev_source_map(self):
        x, y = self.grid.centre_mesh()
        pixels, valid = self.vehicle_ground_to_rectified_pixels(
            np.column_stack((x.ravel(), y.ravel()))
        )
        map_x = pixels[:, 0].reshape(self.grid.height, self.grid.width).astype(np.float32)
        map_y = pixels[:, 1].reshape(self.grid.height, self.grid.width).astype(np.float32)
        valid_map = valid.reshape(self.grid.height, self.grid.width)
        map_x[~valid_map] = -1.0
        map_y[~valid_map] = -1.0
        return map_x, map_y, valid_map
