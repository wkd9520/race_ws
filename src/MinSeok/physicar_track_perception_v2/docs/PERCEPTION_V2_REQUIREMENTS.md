# PhysiCar Perception V2 Requirements Freeze

Status: STEP 0 frozen requirements; STEP 1 implements only the verified metric-BEV front end. Later behavior requires explicit approval.

## 1. Purpose and operating conditions

V2 shall use a camera-derived metric BEV to identify the physical LEFT and RIGHT track boundaries and eventually publish one canonical metric center path and an arc-length lookahead target. Straight, gradual, sharp and near-90-degree non-X-monotonic curves are normal operating conditions. The design shall also cover one-boundary FOV loss, external WHITE objects, bounded dropout, LOST, and controlled recovery.

STEP 1 is not driving perception: it stops at `Camera image -> undistortion -> metric BEV -> V2-only debug topics`. It implements no HSV segmentation, components, boundaries, identity, centerline, target, reconstruction, or temporal policy.

## 2. Verified front-end contract (existing verified values)

The reference is the live `physicar_track_perception` source and `config/stage3_params.yaml` as inspected on 2026-08-23.

| Contract | Existing verified value/behavior |
|---|---|
| Image | `/camera/image_raw`, `sensor_msgs/msg/Image`, sensor-data QoS |
| CameraInfo | No CameraInfo subscription; SIM has no CameraInfo bridge. Static YAML intrinsics are authoritative. |
| Image model | 480x360; K=`[201.38988018035889,0,240; 0,201.38988733291626,180; 0,0,1]`; D=`[-0.045,-0.0001,-0.0003,-0.0001,0.001]` |
| Source/target TF | exact timestamp lookup of `base_footprint <- camera_optical_frame_corrected`, i.e. camera coordinates transformed into vehicle coordinates |
| Timestamp | original nonzero image stamp; zero stamp and latest-TF fallback forbidden; bounded pending age 0.25 s; readiness timer 0.02 s |
| Vehicle convention | `base_footprint`: +X forward, +Y left, +Z up |
| Ground | vehicle-frame Z=0.0 m |
| BEV bounds | X=[0.10,2.00] m, Y=[-0.75,0.75] m |
| Resolution/dimensions | 0.01 m/pixel, width 150, height 190; far +X is image-top and +Y left is image-left |
| Projection tolerances | ray-plane parallel `abs(dz)>1e-12`; forward ray scale `>0`; camera depth `>1e-9`; transform last row atol `1e-9`; orthonormal/determinant atol `1e-7` |
| Height correction | `sim_geometry.camera_height_correction_z=-0.018` m; left-multiplied vehicle-frame Z translation; camera rotation unchanged |
| Pitch correction | `projection.pitch_offset_deg=+2.7`; converted by `math.radians`; `R_y(+2.7°) @ R_vehicle_camera`; origin unchanged; rotation is about `base_footprint` +Y |
| Fixed pose | pan 0.0 rad, tilt -0.5236 rad, tolerance 0.01 rad, required |
| Undistortion/remap | OpenCV `initUndistortRectifyMap` and `remap`, `INTER_LINEAR`; BEV invalid border black and source maps -1 |

The front-end mapping is frozen from the first exact timestamped corrected-camera TF, matching the reference. Calibration, tolerance, pitch, and height values are not tuning parameters in STEP 1.

## 3. Canonical geometry requirements (future)

Every boundary shall use `B(s)=[X(s),Y(s)]`, an arc-length-parametric ordered metric 2D polyline in `base_footprint`. It shall preserve raw observations, near-to-far ordered points without X-monotonic assumptions, cumulative arc length, physical support, controlled resampling, canonical points, tangent, normal, signed curvature, confidence/support, provenance, and identity metadata. Global `y=f(x)` is forbidden as the canonical representation.

One geometry kernel shall provide tangent, normal, and curvature to association and reconstruction. Raw raster triplet curvature shall not directly represent physical track curvature. Path ordering and target selection shall use physical arc length, including 90-degree curves.

## 4. Observation, identity, and initialization (future)

The vehicle starts inside a normal track. Stable multi-frame BOTH evidence initializes LEFT identity, RIGHT identity, center geometry, and validated width; one frame cannot lock. After lock, candidates associate with trusted physical boundaries rather than being freely reassigned.

`DETECTED` and `USABLE` are distinct. A short or sparse component cannot make the state BOTH_OBSERVED merely by existing. Required observation states are BOTH_OBSERVED, LEFT_ONLY_OBSERVED, RIGHT_ONLY_OBSERVED, PREDICTED_ONLY, and LOST. Boundary provenance is OBSERVED, RECONSTRUCTED, PREDICTED, or LOST. Observation state and output validity are separate.

An external WHITE object cannot acquire missing-boundary identity through temporal persistence alone. Missing-boundary search cannot freely promote arbitrary WHITE components.

## 5. Reconstruction and physical safety (future)

For one usable trusted boundary and validated width W, use the trusted interior normal `n_in(s)`:

- `C(s)=B(s)+(W/2)n_in(s)`
- `B_missing(s)=B(s)+W n_in(s)`

Both offsets originate independently from the same observation. Interior direction shall use trusted opposite boundary, trusted center, identity, and/or previous interior direction—not an unverified left/right hard-code.

Ideal circular offsets share one curvature center. For outer observation, `R_center=R_outer-W/2`, `R_inner=R_outer-W`; for inner observation, `R_center=R_inner+W/2`, `R_outer=R_inner+W`. The rule is mirror invariant.

Curvature safety is physical-support based, not sample-count based. Known characterization: raster artifact span about 0.05 m with immediate sign reversal; genuine R=0.20 m arc has |kappa| about 5 m^-1 with same-sign support about 0.30 m. Fixed-support and evidence-coverage cliffs are forbidden. Geometry evidence distinguishes REGULAR, DEGENERATE, and UNKNOWN/INSUFFICIENT. UNKNOWN is neither automatically INVALID nor automatically REGULAR.

Center and missing safety remain separate: regular center plus regular missing permits both; regular center plus degenerate/unknown missing may retain the center but cannot use the missing prior; degenerate center rejects that reconstructed center.

## 6. Width, center, BOTH, and target (future)

Validated width updates require strong actual BOTH observation. Reconstructed, predicted, single-reconstructed, or identity-conflicted geometry cannot update width.

Pair, WHITE, and single observations shall converge to one `CanonicalCenterPath`; color is provenance, not a separate final path algorithm. BOTH processing builds the physical center between canonical boundaries without global X bins or `y=f(x)`. Lookahead uses cumulative center-path arc length, never target X.

## 7. Temporal architecture and heading (future)

Observation, identity, geometry, confidence, candidate path, and accepted temporal output are distinct layers; one `raw_valid` boolean cannot carry all meanings. Heading continuity shall compare a physical overlapping region. Legacy first-N-point heading is not reused. Evidence: legacy delta 0.489 rad while the same geometry produced physical-heading deltas approximately 0 at 0.1 m, 0.041 rad at 0.2 m, and 0 at 0.3 m due to visible-start change. No particular support length is frozen yet.

Short dropout protection is bounded; stale paths cannot be held indefinitely. LOST cannot reinitialize from one external component and requires strong BOTH evidence. Intermittent short SINGLE visibility during recovery must not automatically reset all recovery evidence.

Odom propagation is optional in the first version, but interfaces shall allow future `base(t0) -> fixed odom -> base(t1)` propagation without coupling it to observation logic.

## 8. Debug contract (future)

Structured diagnostics shall expose observation and identity state; LEFT/RIGHT detected, usable, associated, provenance, and support; canonical quality; width and update reason; center source; curvature/support/confidence; temporal state; target; and the exact rejecting layer/reason. A reason must name the layer that actually failed.

## 9. Deferred features

Until explicitly requested: obstacle avoidance, safe corridor/racing line, shortest path, camera pan/tilt control, real-vehicle roll/pitch compensation, IMU compensation, advanced curvature-aware controller, and mandatory odom propagation.

## 10. Lessons from Legacy Stage 3 — mandatory regression requirements

1. External WHITE was falsely promoted as a boundary; identity needs physical association.
2. Mere pair presence blocked an otherwise usable single reconstruction; availability and usability must be separate.
3. `y=f(x)`, X sorting, and X-monotonic assumptions failed near 90 degrees.
4. A raster zigzag produced false discrete curvature near 13.19 m^-1.
5. Sample-count irregular runs changed with raster density and endpoint duplication.
6. Fixed 0.30 m curvature support caused a 0.306-to-0.296 m evidence cliff.
7. A later evidence-coverage rule caused another 0.15-to-0.10 m cell-count cliff.
8. Canonical insufficiency re-entered known-bad raw curvature fallback.
9. First-five-point heading reacted to visible-start changes, not physical overlap.
10. Intermittent short SINGLE frames repeatedly reset recovery after a vehicle reset.
11. `identity_unassociated` was reported when association succeeded and reconstruction failed.
12. Reconstructed/predicted/conflicted measurements could poison width unless explicitly barred.

All are acceptance-test requirements for later V2 steps. STEP 1 deliberately does not implement their downstream algorithms.

## 11. STEP 1 ROS isolation

Node: `physicar_track_perception_v2_bev`. V2-only topics:

- `/perception_v2/debug/undistorted`
- `/perception_v2/debug/bev`
- `/perception_v2/debug/bev_validity`
- `/perception_v2/debug/bev_ready`
- `/perception_v2/debug/bev_valid_fraction`

No `/perception/track/*` production topic is published. Legacy and V2 BEV nodes may run concurrently for A/B comparison.
