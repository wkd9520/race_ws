"""Verified undistortion and metric-BEV remap; no HSV or track detection."""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class BevFrontendOutput:
    undistorted: np.ndarray
    bev: np.ndarray
    validity_mask: np.ndarray


class BevFrontend:
    def __init__(self, camera, projector):
        self.camera = camera
        self.projector = projector
        self.undistort_map_x, self.undistort_map_y = cv2.initUndistortRectifyMap(
            camera.K, camera.D, np.eye(3), camera.K,
            (camera.width, camera.height), cv2.CV_32FC1,
        )
        self.bev_map_x, self.bev_map_y, self.bev_valid_map = (
            projector.build_bev_source_map()
        )

    def update_projector(self, projector):
        """Replace only the pose-dependent BEV map.

        Camera calibration and undistortion maps are pose invariant.  The BEV
        source map is not: it must follow the exact camera pose of each image
        while the output grid remains fixed in base_footprint.
        """
        if projector.camera is not self.camera:
            raise ValueError('projector camera must match frontend camera')
        self.projector = projector
        self.bev_map_x, self.bev_map_y, self.bev_valid_map = (
            projector.build_bev_source_map()
        )

    def process(self, image_bgr):
        image = np.asarray(image_bgr)
        if image.shape[:2] != (self.camera.height, self.camera.width):
            raise ValueError('image shape does not match configured camera')
        undistorted = cv2.remap(
            image, self.undistort_map_x, self.undistort_map_y,
            interpolation=cv2.INTER_LINEAR,
        )
        bev = cv2.remap(
            undistorted, self.bev_map_x, self.bev_map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0),
        )
        return BevFrontendOutput(
            undistorted=undistorted,
            bev=bev,
            validity_mask=(self.bev_valid_map.astype(np.uint8) * 255),
        )
