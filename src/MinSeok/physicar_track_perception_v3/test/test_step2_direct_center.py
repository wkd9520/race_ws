import numpy as np
from physicar_track_perception_v3.geometry import OrderedPolyline
from physicar_track_perception_v3.roles import Component
from physicar_track_perception_v3.path_selector import (select, select_orange,
    select_unknown_white, DIRECT_CENTER_OBSERVED, INVALID)
from physicar_track_perception_v3.roles import CENTER

def comp(color, points, ident=1):
    path = OrderedPolyline.from_points(points)
    return Component(ident, color, path, path.support)

def test_direct_center_is_primary_over_boundaries():
    x = np.linspace(.1, 1.4, 40)
    center = comp('ORANGE', np.c_[x, .02*np.sin(x)], 3)
    left = comp('WHITE', np.c_[x, np.full_like(x, .34)], 1)
    right = comp('WHITE', np.c_[x, np.full_like(x, -.34)], 2)
    result = select([left, right, center])
    assert result.valid and result.source == DIRECT_CENTER_OBSERVED
    assert np.allclose(result.path.points, center.polyline.points)

def test_non_x_monotonic_center_is_preserved_as_ordered_geometry():
    points = np.array([[.1, 0.0], [.3, .04], [.3, .06], [.1, .02], [-.1, .03]])
    result = select([comp('ORANGE', points)])
    assert result.valid
    assert np.allclose(result.path.points, points)

def test_center_absent_is_invalid_in_step2():
    x = np.linspace(.1, 1.0, 20)
    result = select([comp('WHITE', np.c_[x, np.full_like(x, .34)])])
    assert not result.valid and result.source == INVALID

def test_near_seed_stitches_dashed_center_instead_of_longest_only():
    near = comp('ORANGE', np.array([[.10,0.], [.20,0.], [.25,0.], [.30,0.]]), 1)
    far = comp('ORANGE', np.array([[.27,0.], [.50,0.], [.80,0.], [1.1,0.]]), 2)
    from physicar_track_perception_v3.roles import RoleConfig
    result = select([far, near], RoleConfig(minimum_support=.19), gap_limit=.10)
    assert result.valid and result.stitched_component_ids == (1, 2)
    assert result.path.points[0, 0] == .10
    assert result.bridged_gap_count == 1

def test_large_gap_stops_near_to_far_dash_chain():
    near = comp('ORANGE', np.array([[.10,0.], [.20,0.], [.30,0.]]), 1)
    far = comp('ORANGE', np.array([[.50,.30], [.70,.30], [.90,.30]]), 2)
    from physicar_track_perception_v3.roles import RoleConfig
    result = select_orange([near, far], RoleConfig(minimum_support=.19), gap_limit=.15)
    assert result.stitched_component_ids == (1,)

def test_orange_direct_path_does_not_depend_on_generic_role_classifier():
    orange = comp('ORANGE', np.array([[.1, .3], [.2, .3], [.3, .3], [.4, .3]]), 7)
    # It is deliberately lateral and would fail the old center role heuristic.
    result = select_orange([orange])
    assert result.valid and result.source == DIRECT_CENTER_OBSERVED

def test_unknown_white_uses_nearest_component_even_when_another_is_longer():
    nearest_short = comp('WHITE', np.array([
        [.10, .4], [.20, .4]]), 1)
    farther_long = comp('WHITE', np.array([
        [.12, -.4], [.40, -.4], [.70, -.4]]), 2)
    result = select_unknown_white(
        [farther_long, nearest_short], .70)
    assert result.valid
    assert nearest_short.support < farther_long.support
    assert result.stitched_component_ids == (1,)
