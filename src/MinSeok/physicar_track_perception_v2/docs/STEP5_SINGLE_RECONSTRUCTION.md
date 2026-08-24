# STEP 5 Trusted Single-Boundary Reconstruction

Validated width is initialized by the median of three identity-gated BOTH
observations. Later BOTH observations use the existing 0.20 EMA and 0.12 m
metric residual gate. Single and reconstructed geometry never update width.

Interior direction is selected from the vector to the last trusted actual
opposite boundary, projected onto canonical normals. At least 80% of usable
normal signs must agree. Center and missing geometry are direct offsets of the
same accepted observed source by `W/2` and `W`; missing is never chained from
center and never becomes an observed identity.

Canonical curvature uses a 0.10 m balanced physical neighbourhood. Offset
factor evidence is REGULAR, DEGENERATE, or UNKNOWN. Sustained non-positive
factor over 0.15 m is degenerate. Insufficient evidence remains UNKNOWN and
does not publish a single center. Center and missing safety are independent.
These physical-support values are proposed and require Gazebo validation.
