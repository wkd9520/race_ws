# PhysiCar Track Perception V3 — STEP 0/1/2

V3 is independent of V2 production state.  The initial core is frame-local:
an observed CENTER ordered metric polyline is the path; without one the path
is invalid.  Boundary half-width reconstruction, temporal state, identity,
RANSAC, odometry and controller integration are deferred.

The verified front-end contract to be integrated next is exact image-stamp TF,
`base_footprint` metric BEV, dynamic pan with pan-local-Y pitch correction,
and the existing HSV/component/canonical spacing contract.  No X sorting or
`y=f(x)` representation is permitted.
