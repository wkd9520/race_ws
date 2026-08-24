import numpy as np
from physicar_track_perception_v3.geometry import OrderedPolyline
from physicar_track_perception_v3.roles import Component
from physicar_track_perception_v3.white_propagation import seed_from_center, propagate, LEFT, RIGHT

def c(i, pts):
    p=OrderedPolyline.from_points(np.asarray(pts,float))
    return Component(i,'WHITE',p,p.support)

def test_center_seed_labels_path_relative_sides():
    center=OrderedPolyline.from_points([[.2,0],[.6,0]])
    shadow=seed_from_center(center,[c(1,[[.2,.2],[.6,.2]]),c(2,[[.2,-.2],[.6,-.2]])])
    assert shadow.labels == {1: LEFT, 2: RIGHT}

def test_frame_local_propagation_replaces_seed():
    previous=seed_from_center(OrderedPolyline.from_points([[.2,0],[.6,0]]),[c(1,[[.2,.2],[.6,.2]]),c(2,[[.2,-.2],[.6,-.2]])])
    current=propagate(previous,[c(3,[[.6,.2],[1.0,.2]]),c(4,[[.6,-.2],[1.0,-.2]])])
    assert current.labels == {3: LEFT, 4: RIGHT}

def test_missing_current_observation_has_no_geometry():
    previous=seed_from_center(OrderedPolyline.from_points([[.2,0],[.6,0]]),[c(1,[[.2,.2],[.6,.2]])])
    current=propagate(previous,[])
    assert current.left is None and current.right is None and current.labels == {}
