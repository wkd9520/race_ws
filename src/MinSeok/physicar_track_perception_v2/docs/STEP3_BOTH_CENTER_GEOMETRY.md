# STEP 3 Frame-local BOTH Center Geometry

STEP 3 consumes STEP 2 canonical metric polylines and produces a center only
when one current frame contains an unambiguous, physically consistent pair. It
does not create track lock, trusted identity/width, reconstruction, target, or
temporal state.

## Usable evidence and verified values

- minimum canonical points: 5 (existing legacy center evidence count)
- minimum support: 0.20 m (physical derivation: four 0.05 m intervals)
- width range: 0.60–0.95 m (**EXISTING VERIFIED VALUE**)
- minimum correspondences: 4 (existing legacy observed-pair count)
- minimum overlap: 0.15 m (three 0.05 m physical intervals)

The tangent angle, side-consistency, width-spread, and ambiguity margins are
**PROPOSED INITIAL VALUES / RUNTIME VALIDATION REQUIRED**. They are frame-local
quality gates, not identity or temporal policy.

## Correspondence decision

V2 uses mutual-nearest samples in metric 2D, then requires consistent local
tangents, physical width, strictly increasing indices, and common arc-length
support. Mutual nearest prevents one-way nearest's excessive many-to-one
matches. Unlike normalized arc length, it does not force inner and outer arcs
of different lengths to share the same normalized parameter. A full dynamic
time-warping or normal-intersection solver was not selected because canonical
0.05 m sampling already gives comparable density and the simpler mutual rule
has explicit no-crossing behavior.

LEFT/RIGHT is assigned from the sign of `cross(local_tangent, pair_vector)`
over the correspondence run. This is vehicle-yaw invariant and does not use
global mean Y. Mixed signs are ambiguous and do not produce a center.

The center is the 2D midpoint of accepted physical pairs, ordered by the
monotone correspondence and resampled by its own cumulative arc length. Width
samples and min/median/max are current-frame observations only.
