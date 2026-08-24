import cv2
import numpy as np
import pytest

from physicar_track_perception.candidate_extraction import (
    CandidateExtractionConfig as LegacyExtractionConfig,
    MetricCandidateExtractor as LegacyExtractor,
)
from physicar_track_perception_v2.components import (
    CanonicalComponentExtractor,
    ComponentExtractionConfig,
    ORANGE,
    WHITE,
)
from physicar_track_perception_v2.geometry import BevGrid
from physicar_track_perception_v2.segmentation import (
    ColorComponentPipeline,
    HsvRange,
)


GRID = BevGrid(0.1, 2.0, -0.75, 0.75, 0.01)


def extractor(**overrides):
    values = dict(min_component_area=1, min_valid_pixels=1,
                  min_valid_overlap=0.0, canonical_spacing=0.05)
    values.update(overrides)
    return CanonicalComponentExtractor(GRID, ComponentExtractionConfig(**values))


def canonical(points, color=WHITE):
    result, reason = extractor().canonicalize_ordered_points(points, color=color)
    assert reason == 'valid'
    return result


def masks(*, white=None, orange=None):
    shape = (GRID.height, GRID.width)
    result = {WHITE: np.zeros(shape, np.uint8), ORANGE: np.zeros(shape, np.uint8)}
    if white is not None:
        result[WHITE][tuple(np.asarray(white).T)] = 255
    if orange is not None:
        result[ORANGE][tuple(np.asarray(orange).T)] = 255
    return result


def test_straight_connected_component_orders_and_canonicalizes():
    pixels = [(row, 40) for row in range(40, 121)]
    frame = extractor().extract(masks(white=pixels), np.ones((GRID.height, GRID.width), bool))
    candidate = frame.candidates[0]
    assert candidate.raw_point_count == 81
    assert candidate.support_length == pytest.approx(0.80)
    assert candidate.near_endpoint[0] < candidate.far_endpoint[0]
    assert np.allclose(candidate.canonical_points[:, 1], candidate.canonical_points[0, 1])


def test_gradual_curve_preserves_connected_order():
    pixels = [(120-i, 40 + round(i*i/100.0)) for i in range(30)]
    frame = extractor().extract(masks(white=pixels), np.ones((GRID.height, GRID.width), bool))
    candidate = frame.candidates[0]
    assert candidate.raw_point_count >= 28
    assert np.all(np.diff(candidate.raw_s) > 0)


def test_l_shape_is_one_non_x_monotonic_polyline_without_x_sort():
    pixels = ([(130, col) for col in range(30, 71)]
              + [(row, 70) for row in range(90, 130)])
    candidate = extractor().extract(
        masks(orange=pixels), np.ones((GRID.height, GRID.width), bool)
    ).candidates[0]
    delta_x = np.diff(candidate.raw_ordered_points[:, 0])
    assert np.any(np.isclose(delta_x, 0.0))
    assert candidate.color == ORANGE
    assert candidate.support_length > 0.75


def test_increase_decrease_increase_x_topology_is_preserved():
    pixels = []
    for col, row in list(zip(range(20, 41), range(140, 119, -1))) \
            + list(zip(range(41, 61), range(120, 140))) \
            + list(zip(range(61, 82), range(139, 118, -1))):
        pixels.append((row, col))
    candidate = extractor().extract(
        masks(white=pixels), np.ones((GRID.height, GRID.width), bool)
    ).candidates[0]
    dx = np.diff(candidate.raw_ordered_points[:, 0])
    assert np.count_nonzero(np.diff(np.sign(dx[np.abs(dx) > 1e-9]))) >= 2
    assert candidate.raw_point_count >= len(set(pixels)) - 3


def test_dense_and_sparse_sampling_create_similar_canonical_geometry():
    theta_dense = np.linspace(0, np.pi / 2, 161)
    theta_sparse = np.linspace(0, np.pi / 2, 17)
    make = lambda t: np.column_stack((0.4 + 0.6*np.sin(t), -0.2 + 0.6*(1-np.cos(t))))
    dense, sparse = canonical(make(theta_dense)), canonical(make(theta_sparse))
    assert dense.support_length == pytest.approx(sparse.support_length, abs=8e-4)
    assert np.max(np.linalg.norm(dense.canonical_points - sparse.canonical_points, axis=1)) < 0.002


def test_duplicate_and_near_zero_segments_are_removed():
    points = np.array([[0.2, 0.0], [0.2, 0.0], [0.2000000001, 0.0], [0.4, 0.0]])
    candidate = canonical(points)
    assert candidate.raw_point_count == 2
    assert candidate.support_length == pytest.approx(0.2)


def test_disconnected_components_remain_distinct_candidates():
    pixels = ([(row, 20) for row in range(30, 50)]
              + [(row, 90) for row in range(100, 120)])
    frame = extractor().extract(masks(white=pixels), np.ones((GRID.height, GRID.width), bool))
    assert len(frame.observations) == 2
    assert len(frame.candidates) == 2


def test_white_and_orange_share_geometry_algorithm_only_provenance_differs():
    pixels = [(row, 50) for row in range(60, 100)]
    frame = extractor().extract(
        masks(white=pixels, orange=pixels), np.ones((GRID.height, GRID.width), bool)
    )
    white, orange = frame.candidates
    assert {white.color, orange.color} == {WHITE, ORANGE}
    assert np.array_equal(white.canonical_points, orange.canonical_points)


def test_short_component_keeps_quality_metadata_without_identity():
    pixels = [(80, 80), (81, 80)]
    frame = extractor().extract(masks(white=pixels), np.ones((GRID.height, GRID.width), bool))
    observation = frame.observations[0]
    assert observation.metadata.extracted
    assert observation.metadata.geometry_valid
    assert observation.candidate.support_length == pytest.approx(0.01)
    assert not hasattr(observation.candidate, 'boundary_side')


def test_reversed_order_has_same_near_to_far_canonical_geometry():
    points = np.array([[0.4, 0.1], [0.5, 0.14], [0.7, 0.3], [0.9, 0.45]])
    forward, reverse = canonical(points), canonical(points[::-1])
    assert np.allclose(forward.canonical_points, reverse.canonical_points)
    assert np.linalg.norm(forward.near_endpoint) <= np.linalg.norm(forward.far_endpoint)


@pytest.mark.parametrize('points, reason', [
    ([[0.2, 0.0]], 'ordered_geometry_short'),
    ([[0.2, 0.0], [np.nan, 0.1]], 'ordered_geometry_nonfinite'),
    ([[0.2, 0.0], [0.2, 0.0]], 'ordered_geometry_degenerate'),
])
def test_invalid_geometry_returns_explicit_result(points, reason):
    candidate, actual = extractor().canonicalize_ordered_points(points)
    assert candidate is None
    assert actual == reason


def test_hsv_ranges_and_morphology_match_existing_verified_values():
    pipeline = ColorComponentPipeline(
        {
            WHITE: (HsvRange((0, 0, 170), (179, 90, 255)),),
            ORANGE: (HsvRange((5, 100, 100), (30, 255, 255)),),
        }, 3, 5, extractor()
    )
    image = np.zeros((GRID.height, GRID.width, 3), np.uint8)
    image[60:70, 30:40] = cv2.cvtColor(
        np.uint8([[[0, 0, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    image[100:110, 80:90] = cv2.cvtColor(
        np.uint8([[[15, 220, 220]]]), cv2.COLOR_HSV2BGR)[0, 0]
    output = pipeline.process(image, np.ones(image.shape[:2], bool))
    assert output.white_mask[65, 35] == 255
    assert output.orange_mask[105, 85] == 255
    assert len(output.component_frame.candidates) == 2


def test_valid_overlap_filter_is_metadata_not_track_geometry_filter():
    pixels = [(row, 20) for row in range(30, 50)]
    valid = np.zeros((GRID.height, GRID.width), bool)
    valid[30:35, 20] = True
    frame = extractor(min_valid_overlap=0.70).extract(masks(white=pixels), valid)
    observation = frame.observations[0]
    assert observation.candidate is None
    assert observation.metadata.rejection_reason == 'valid_overlap'
    assert observation.metadata.valid_overlap == pytest.approx(0.25)


def test_canonical_support_is_raw_physical_support_not_resampled_count():
    candidate = canonical([[0.2, 0.0], [0.21, 0.0]])
    assert candidate.canonical_point_count == 2
    assert candidate.support_length == pytest.approx(0.01)
    assert candidate.canonical_s[-1] == pytest.approx(candidate.support_length)


def test_legacy_topology_order_is_preserved_without_legacy_x_bins():
    pixels = np.asarray(
        [(130, col) for col in range(30, 71)]
        + [(row, 70) for row in range(90, 130)], dtype=np.int32
    )
    legacy = LegacyExtractor(
        GRID, LegacyExtractionConfig(x_bin_size=0.05)
    )._ordered_component_polyline(pixels[:, 0], pixels[:, 1])
    new = extractor()._ordered_geodesic_polyline(pixels[:, 0], pixels[:, 1])
    assert len(legacy) >= 2 and len(new) > len(legacy)
    assert np.allclose(new[[0, -1]], legacy[[0, -1]], atol=GRID.resolution)
    assert np.any(np.isclose(np.diff(new[:, 0]), 0.0))
