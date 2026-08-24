# STEP 4 Trusted Boundary Identity

STEP 4 locks LEFT/RIGHT only after three consistent BOTH frames. Connected
component IDs remain frame-local measurements. The initial three-frame streak
and 50% pair-overlap coverage are proposed runtime values; three frames are
about 0.2 s at the observed camera rate. The 0.12 m distance gate reuses the
legacy metric continuity gate. Other association values require runtime
validation.

Current canonical points are compared with trusted canonical geometry using
physical nearest distance and tangent agreement. Supported points must form
one contiguous arc-length interval. The interval anchors at least 0.15 m and
may extend no more than 0.15 m along a locally tangent-consistent continuation.
Only this accepted subpolyline can update trusted state or enter the unchanged
STEP 3 center generator.

Canonical trimming was selected before graph extraction because it directly
addresses the observed merged tail with minimal topology policy. If STEP 2's
geodesic diameter omits the real branch entirely, STEP 4 rejects that side and
publishes no center. Skeleton graph branch recovery is intentionally deferred
until runtime evidence demonstrates that safe trimming is insufficient.
