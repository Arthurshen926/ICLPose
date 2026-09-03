# Plane-specific geometry and cross-atlas fusion (2026-09-03)

## Outcome

This round implemented the attachment's first two priorities without changing
the source-image-free map contract.  It found and fixed a real mapping-side
geometry bug, but also showed that replacing the established appearance atlas
with the corrected geometry is not yet a held-set performance win.  The new
geometry is retained as an independent hypothesis/uncertainty channel; it does
not replace the current mainline atlas.

## P0: plane-specific observation geometry

The old observation bank decided that a RADIO token belonged to a plane when
at least eight pixels carried that plane label, but then averaged **all** valid
depth pixels in the 4x4 token to obtain its 3D point.  At plane boundaries this
mixed facade, tree/window foreground and background geometry.

`build_goal_maplet_plane_pnp_observation_bank.py` now accepts the exact rendered
plane-observation inventory and computes 3D statistics only over pixels carrying
the observation's exact plane label.  V2 stores, per token:

- plane-specific world point;
- full 3x3 world covariance;
- plane-pixel purity;
- within-plane depth dispersion.

The 482,484 RADIO descriptors and token IDs are byte-identical to V1.  Geometry
is not: 35.71% of points move by more than 5 cm, 8.05% by more than 10 cm,
0.626% by more than 25 cm, and the maximum old/new displacement is 22.18 m.
Only 14.83% of accepted tokens are 100% plane-pure.  Displacement increases as
purity falls and depth dispersion rises.

Artifact:

`output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1/stmarys_plane_specific_pnp_observation_bank_v2.npz`

- file SHA256: `3cf4acc23a50a88355bb285465153d56bd51d140afc2eaf5ecb5baf5d6b503ff`
- content SHA256: `2447d3d68188bf62236f3b012ae0d64ab2f892f7a072944ad39732cc53bc2247`

## Two geometry interpretations

V7 supports two explicit, non-confusable constructions:

1. **plane-specific atlas**: corrected plane pixels determine both texel
   assignment and prototype geometry;
2. **decoupled atlas**: the established RADIO observation determines appearance
   texel/mode assignment, while the row-aligned plane-specific bank independently
   supplies metric UV, signed height, range and view direction.

Both store anonymous descriptors and geometry only; neither stores source RGB,
source paths, or runtime source-view identity.

Plane-specific V7 file/content:
`3c4c00005ea83710f4386e182d4fd12b68e610556e9cccf8aec13cbee5835a5c` /
`a96571a781abc5ac7adb28a1f519bf5ddd298f71df93454f50083dbb1f5826c8`.

Decoupled V7 file/content:
`772284b7710f7663d3b39294719636ea7390d167ac64c969c04d8806c002b86b` /
`085856fe0b19f39759a167d8584b0423afaaf5d39604362eb2dd3a59364fb9ed`.

## Frozen seq10 development results

The previous dual-scale legacy atlas is 0.2367 m / 0.5337 deg with hit counts
10/46/69/80/85 at 0.1/0.25/0.5/1/2 m.

Plane-specific V7 alone is 0.2574 m / 0.5188 deg with 9/44/68/83/85.  It gains
three 1 m cases but loses fine cases.  Decoupled V7 alone is 0.2571 m / 0.5998
deg with 8/43/70/81/86.

The post-label oracle over legacy and plane-specific candidates is much stronger:
0.1763 m / 0.4187 deg and 18/59/75/85/88 hits.  This locates substantial
remaining error in candidate scoring/selection rather than candidate existence.

A label-free cross-atlas selector scores both frozen candidates on all four
loose/strict correspondence inventories.  On seq10 it reaches 0.2069 m / 0.5300
deg and 10/52/69/85/87 hits.  This is a genuine development improvement.  A
Gaussian soft-likelihood alternative was tested and killed because it reduced
both coarse and fine recall; softening an incomplete likelihood is not enough.

## Frozen held replay: seq13 shard3 (87 queries)

No constant was changed after seq10.  Results are:

| system | median m / deg | 0.1m | 0.25m | 0.5m | 1m | 2m |
|---|---:|---:|---:|---:|---:|---:|
| legacy dual-scale | 0.2382 / 0.9337 | 10 | 46 | 68 | 76 | 78 |
| plane-specific dual-scale | 0.2758 / 0.9927 | 9 | 39 | 68 | 76 | 77 |
| decoupled dual-scale | 0.2663 / 0.8507 | 10 | 41 | 65 | 74 | 76 |
| legacy + plane-specific cross-atlas | 0.2759 / 0.9421 | 11 | 41 | 68 | 76 | 77 |
| legacy + decoupled cross-atlas | 0.2550 / 0.8507 | 10 | 43 | 68 | 76 | 78 |
| three-candidate post-label oracle | 0.1748 / 0.5791 | 21 | 58 | 75 | 77 | 79 |

Therefore the cross-atlas selector and V7 replacements are **KILL as mainline
replacements** on this held shard.  The P0 geometry fix remains valid, and the
large oracle gap shows it should remain an independent geometry/covariance
channel.  The failure is specifically that raw reprojection inlier ratio cannot
calibrate when corrected geometry should override the appearance-biased field.

## Next constrained step

Do not add more hard branches or retune the 1.0/0.25 m thresholds.  The next
implementation should propagate V2 purity/covariance/depth dispersion to atlas
prototypes and correspondences, then evaluate an uncertainty-normalized surface
likelihood.  Candidate selection should be trained/calibrated only on a disjoint
development route and must expose retrieval, matching, geometry and solver
oracles separately.  MoGe3 scale remains useful in the existing plane/scale
optimizer; unrestricted token-level Sim(3) remains killed.

## Verification

- 14 focused tests pass;
- all modified modules pass `py_compile`;
- `git diff --check` passes;
- all selectors freeze their output before opening pose labels;
- runtime maps still store no source image or source-view identity.
