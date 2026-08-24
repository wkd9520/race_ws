# STEP 2 Component Geometry Contract

## Existing verified segmentation values

The live legacy Stage 3 source and `stage3_params.yaml` define the values below.
V2 uses them without tuning:

- WHITE HSV: `[0, 0, 170]` through `[179, 90, 255]`
- ORANGE HSV: `[5, 100, 100]` through `[30, 255, 255]`
- morphology: one elliptical `OPEN` with size 3, then one elliptical
  `CLOSE` with size 5
- masks remain separate by color; their union is debug-only
- the projection validity map is evaluated per connected component; accepted
  ordered geometry contains valid BEV pixels only

Color is observation provenance. It is not a boundary identity or a separate
path algorithm.

## Component and geometry policy

V2 labels each color mask with 8-connectivity. It rejects only obvious image
noise using the existing minimum area (8 pixels), minimum valid area (3 pixels),
and valid-overlap fraction (0.70). Legacy longitudinal, lateral, elongation,
X-bin, width, pairing, and role filters are deliberately excluded because they
encode later track assumptions.

The raw ordered metric polyline is a deterministic approximate geodesic
diameter of the connected pixel graph. It is never X-sorted and is oriented by
the endpoint closer to the vehicle origin. Equal-distance ties use a stable
metric endpoint key. This candidate-only convention cannot guarantee temporal
identity; that belongs to a later step.

Canonical geometry removes only duplicate/near-zero consecutive segments,
retains raw geometry separately, computes cumulative physical arc length, and
resamples along arc length. The current 0.05 m spacing is a **PROPOSED INITIAL
VALUE / RUNTIME VALIDATION REQUIRED**, based on the verified 0.01 m BEV cells
and the legacy ordered-component physical sampling scale. Physical support is
always the raw polyline arc length and is not inflated by resampling.

No LEFT/RIGHT identity, pair, width, centerline, target, reconstruction,
regularity, or temporal decision exists in STEP 2.
