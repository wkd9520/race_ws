import cv2
import numpy as np

from physicar_track_perception_v2.components import (
    CanonicalComponentExtractor, ComponentExtractionConfig)
from physicar_track_perception_v2.geometry import BevGrid
from physicar_track_perception_v2.segmentation import (
    ColorComponentPipeline, HsvRange)
from physicar_track_perception_v2.white_continuity_capture import (
    morphology_variants)


def pipeline():
    grid = BevGrid(0.10, 2.00, -0.75, 0.75, 0.01)
    extractor = CanonicalComponentExtractor(
        grid, ComponentExtractionConfig())
    return ColorComponentPipeline({
        'WHITE': (HsvRange((0, 0, 170), (179, 90, 255)),),
        'ORANGE': (HsvRange((5, 100, 100), (30, 255, 255)),),
    }, 3, 5, extractor), grid


def test_stage_capture_is_exact_and_does_not_change_final_mask():
    subject, grid = pipeline()
    image = np.zeros((grid.height, grid.width, 3), np.uint8)
    cv2.line(image, (20, 160), (120, 30), (220, 220, 220), 2)
    valid = np.ones(image.shape[:2], bool)
    normal = subject.process(image, valid)
    diagnostic = subject.process(image, valid, include_white_stages=True)
    assert normal.white_stages is None
    assert np.array_equal(normal.white_mask, diagnostic.white_mask)
    assert np.array_equal(normal.orange_mask, diagnostic.orange_mask)
    assert np.array_equal(normal.overlay, diagnostic.overlay)
    assert np.array_equal(
        diagnostic.white_stages.after_close, diagnostic.white_mask)


def test_stage_masks_match_exact_baseline_operations():
    subject, grid = pipeline()
    image = np.zeros((grid.height, grid.width, 3), np.uint8)
    image[50:54, 20:80] = 200
    valid = np.ones(image.shape[:2], bool)
    valid[:, :25] = False
    output = subject.process(image, valid, include_white_stages=True)
    stages = output.white_stages
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    raw = cv2.inRange(hsv, np.array([0, 0, 170], np.uint8),
                     np.array([179, 90, 255], np.uint8))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(raw, cv2.MORPH_OPEN, open_kernel, iterations=1)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, close_kernel,
                             iterations=1)
    assert np.array_equal(stages.hsv, hsv)
    assert np.array_equal(stages.raw, raw)
    assert np.array_equal(stages.post_validity, raw * valid.astype(np.uint8))
    assert np.array_equal(stages.after_open, opened)
    assert np.array_equal(stages.after_close, closed)


def test_analysis_variants_do_not_alias_or_mutate_raw():
    raw = np.zeros((40, 40), np.uint8)
    raw[10:12, 5:35] = 255
    before = raw.copy()
    variants = morphology_variants(raw)
    assert set(variants) == {
        'none', 'open3', 'close5', 'close3_open3', 'close5_open3',
        'open3_close7'}
    assert np.array_equal(raw, before)
    for value in variants.values():
        assert not np.shares_memory(value, raw)
