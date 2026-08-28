# G25 real-query planar recovery gate (2026-08-27)

## Decision

The planar localization route is **conditionally feasible**, but the current
zero-shot monocular query geometry is **not sufficient for pose estimation**.

The experiment deliberately separates four questions:

1. Can a dedicated 2DGS planar side map and a plane solver recover pose when
   query pixel geometry and correspondences are correct? **Yes.**
2. Can an off-the-shelf pose-free monocular model replace the query geometry
   while retaining oracle masks and correspondences? **No.**
3. Is the failure caused by matching? **No.** Matching is bypassed by oracle
   region correspondence.
4. Is the failure caused by the map/solver? **No.** Replacing both query plane
   normals and offsets by their ideal values recovers all 88 queries.

Therefore the next mainline problem is query-side, map-conditioned recovery of
plane offset and normal. A learned plane matcher, local optimizer, PnP, ALIKE,
or a global pose lattice must not be added before that gate passes.

## Frozen experiment

- Route: `seq10`, 88 images.
- RGB/intrinsics only are read by the geometry builder; no pose or GT is read.
- Query geometry: official MoGe-2 normal checkpoints at 256x144.
- Evaluation grants two oracles to isolate geometry:
  - GT 2DGS pixel visibility masks;
  - GT planar side-map region correspondence.
- At most 16 planes; normal-diverse selection; metric and scale-aware plane
  solvers; robust IRLS translation.
- No ALIKE, PnP, pose lattice, Gaussian reconstruction, or model training.

The map/solver upper bound is the balanced dedicated planar side map with GT
pixel surface geometry. Its robust solver reaches 100% at 1m/10 degrees in the
fixed-metric oracle and has median translation/rotation errors of about 0.057m
and 0.297 degrees. This establishes that the representation and algebra are
capable when the measurements are accurate.

## Real pose-free geometry results

### MoGe-2-S

With GT masks and GT region correspondence, the balanced robust solver obtains:

- 1m/10 degrees: 0/88;
- 2m/45 degrees: 2/88 (2.27%);
- median rotation error: 21.19 degrees;
- median translation error: 10.81m;
- median plane-normal error: 26.10 degrees;
- median plane-offset error: 4.51m.

### MoGe-2-L capacity control

The official `Ruicheng/moge-2-vitl-normal` checkpoint also fails:

- 1m/10 degrees: 0/88;
- 2m/45 degrees: 0/88;
- median rotation error: 18.73 degrees;
- median translation error: 9.13m;
- median plane-normal error: 23.93 degrees;
- median plane-offset error: 4.22m.

The large model runs at median 0.161s/image (p90 0.166s) and peaks at about
2.62GB CUDA allocation, so runtime is not the blocker. Accuracy is.

### MoGe-3-L

After MoGe-3 was released, the official `Ruicheng/moge-3-vitl` model was run
with true FOV and its default three sparse volumetric refinement steps. Relative
to MoGe-2-L it improves the final rotation estimate but does not fix metric
plane offset:

- 1m/10 degrees: 0/88;
- 2m/45 degrees: 0/88;
- median rotation error: 15.65 degrees (better than 18.73 degrees);
- median translation error: 9.02m;
- median plane-normal error: 25.70 degrees;
- median plane-offset error: 4.67m (worse than 4.22m).

With ideal offsets, MoGe-3 normals reach 95.45% at 2m/45 degrees, compared with
82.95% for MoGe-2. Thus the refinement is useful for coarse orientation and
fine surface shape, but the tested localization remains blocked by absolute
metric plane offset.

## Causal parameter ablation

Using the same selected regions and robust solver on the balanced side map:

| Query plane parameters | 1m/10deg | 2m/45deg | Median rotation | Median translation |
|---|---:|---:|---:|---:|
| MoGe normal + MoGe offset | 0.00% | 0.00% | 18.73deg | 9.13m |
| ideal normal + MoGe offset | 0.00% | 0.00% | 0.00deg | 9.13m |
| MoGe normal + ideal offset | 28.41% | 82.95% | 18.73deg | numerical zero |
| ideal normal + ideal offset | 100.00% | 100.00% | 0.00deg | numerical zero |

This is decisive:

- the primary failure is metric plane offset/depth, not matching;
- normal error is a secondary but still material blocker for 1m/10 degrees;
- an unknown global scale does not fix the offsets, because their errors are
  region-dependent rather than a single image-wide scale factor;
- the plane solver itself is verified by the ideal/ideal control.

## ZeroPlane audit

The official ZeroPlane implementation predicts plane segmentation, normals,
offsets and planar depth, so it is a relevant specialized baseline. It is not a
drop-in lightweight check: the released Dust3R checkpoint is 7,105,876,663
bytes and the inference stack requires Detectron2 plus a compiled
MSDeformAttn extension. It has not been used to authorize any result here.

ZeroPlane remains worth one bounded baseline only if it is evaluated by the
same GT-mask/GT-correspondence parameter gate. A better mask alone cannot pass:
MoGe already fails when the mask and correspondence are perfect. The required
promotion thresholds are therefore on metric offset and normal accuracy, not
segmentation IoU.

## Camera-contract and plane-source audit

The six-way P0 audit requested after the MoGe-3 result is now complete.  The
main MoGe-3 run used deterministically undistorted ideal-pinhole RGB and the
exact SIMPLE_RADIAL-derived pinhole camera.  A second full 88-query run used
the raw radial images with the same FOV contract.  The raw run changes the
numbers slightly (balanced robust: 3.41% at 1m/10 degrees, 5.68% at 2m/45
degrees, median offset error 4.14m), but remains catastrophically below the
80% gate.  Radial distortion is therefore a real nuisance, not the root cause
of the multi-metre error.

On the ideal-pinhole run, fitting planes directly to the MoGe-3 point map and
fitting planes after reconstructing points from MoGe depth with frozen exact K
are numerically equivalent at the reported scale.  Both give about 15.13
degrees median rotation error, 11.89m median translation error, and 4.77m
median offset error.  This rejects a hidden point-map versus depth/K coordinate
contract mismatch.  The normal-head plus robust-offset branch remains the
better tested source (15.65 degrees / 9.02m), but it also fails metric pose.

## Low-capacity metric-correction ladder

An image-wide affine scale plus shift was added as a five-unknown analytic
control.  It is ill-conditioned and worse than the raw metric solver: the
balanced mass-WLS branch has median translation error 16.74m and the
normal-diverse branches about 20.51m, with 0% recall at both pose thresholds.
This formally kills global scale+shift as the correction mechanism.

The first map-conditioned control is also complete.  Pose-free MoGe-3 geometry
was built for all 98 mapping images on seq9.  GT masks/correspondences were
used only to form a label-bearing fit artifact.  Regularization and model
family were selected on an interleaved seq9 validation subset; seq10 was a
different, held route.  Two deliberately small models were compared:

- global affine plus an L2-shrunk intercept for each matched map region;
- a continuous ridge using predicted offset, map normal and map offset.

Neither passes.  The selected region model sees only 38.84% of held seq10
plane observations, gives 0% at 1m/10 degrees and 3.41% at 2m/45 degrees, and
changes median translation only from 9.019m to 9.005m (P90 32.76m to 26.41m).
The continuous map-geometry model is already worse on seq9 validation.  Thus
simple region-ID memorization and low-order map geometry are not the proposed
map-conditioned recovery head; the missing information must be query-region
appearance/shape/context (RADIO, bounded polygon cues, and uncertainty).

This result does not kill map-conditioned recovery in general.  It kills the
cheap low-capacity shortcuts and narrows the next model to a route-disjoint
region-level predictor with actual query evidence.  Matcher development and
predicted masks remain blocked until that predictor passes 80% at 1m/10
degrees under GT masks and correspondence.

A subsequent route-disjoint query-evidence control used the complete
RADIO-final grid.  GT plane masks were inverse-SIMPLE_RADIAL warped onto the
36x64 raw RADIO token grid.  Full 1280-D region descriptors were pooled, PCA
was fit only on seq9 and frozen at 64 dimensions, and the residual regressor
also received MoGe normals/offsets, map plane geometry, mask extent and an
analytic projected-area distance proxy.  This still obtained 0% at both pose
gates on seq9 validation, so the held seq10 selector correctly retained the
region-intercept control.  The failure is not caused by the earlier 32-group
RADIO compression.

Finally, a weighted 3x3 normal calibration learned on seq9 was evaluated on
seq10.  It changes the fraction below 10 degrees only from 26.14% to 27.27%
and worsens median rotation from 15.65 to 16.35 degrees.  Combined with the
selected offset correction it remains 0% at 1m/10 degrees and 3.41% at
2m/45 degrees.  Thus all low-capacity P1 controls are now closed: neither
distance nor normal can be repaired by global, region-ID, low-order geometry,
pooled-RADIO ridge, projected-area proxy, or a fixed normal transform.

The next experiment must preserve within-region token layout and explicitly
predict structured plane geometry/uncertainty (or optimize bounded polygon
reprojection), rather than regress one scalar from an average descriptor.

## Cross-route map/solver audit

The strong pixel-surface oracle was replayed on seq12 and seq14 without fitting
query geometry.  Balanced robust reaches 89.36% / 94.68% (1m10 / 2m45) on
seq12 and 94.44% / 97.22% on seq14, with median translation 0.050m and 0.173m.
Strict robust is less available but remains precise when usable.  The planar
side map and solver are therefore not a seq10-only accident; the remaining
failure is still query measurement recovery.

## Mainline architecture after this gate

The justified architecture is:

1. RADIO retrieves a small set of map regions/submaps.
2. The retrieved 2DGS planar side map supplies finite planar regions and strong
   metric geometry priors.
3. A query-side head predicts plane masks and normals, but plane offsets are
   recovered or corrected *conditioned on the retrieved metric map*, rather
   than trusted as zero-shot monocular metric depth.
4. A structural matcher reasons jointly over plane descriptors, relative normal
   relations, finite boundaries and adjacency.
5. A small set of plane correspondences yields direct pose hypotheses through
   the verified robust plane solver.
6. Original 2DGS visibility/appearance performs final verification/refinement.

This preserves the successful RADIO retrieval front end and avoids enumerating
millions of global poses. The immediate research gate is map-conditioned plane
parameter recovery. Matching should be implemented only after its oracle-mask
parameter accuracy passes a meaningful pose threshold.

## Artifacts

- Strong pixel-surface oracle:
  `output/g25_pose_transport/map_disjoint_seq12_seq14_backend_v4_coordinate_calibration_disjoint/planar_pixel_oracle_seq10_v2/report_source_bound_v4_final.json`
- MoGe-2-S geometry:
  `output/g25_pose_transport/planar_query_geometry/moge2_vits_normal_seq10_full_v1/manifest.json`
- MoGe-2-L geometry:
  `output/g25_pose_transport/planar_query_geometry/moge2_vitl_normal_seq10_full_v1/manifest.json`
- MoGe-2-L causal ablation:
  `output/g25_pose_transport/planar_query_geometry/moge2_vitl_normal_seq10_full_v1/gt_mask_planar_pose_parameter_ablation_v2.json`
- MoGe-3-L geometry and causal ablation:
  `output/g25_pose_transport/planar_query_geometry/moge3_vitl_seq10_full_v1/manifest.json`
  and `gt_mask_planar_pose_parameter_ablation_v1.json`
- Camera-contract controls:
  `output/g25_pose_transport/planar_query_geometry/moge3_vitl_seq10_raw_radial_full_v1/gt_mask_planar_pose_normal_head_v1.json`,
  `output/g25_pose_transport/planar_query_geometry/moge3_vitl_seq10_full_v1/gt_mask_planar_pose_point_pca_v1.json`,
  and `gt_mask_planar_pose_depth_exact_k_pca_v1.json`
- Map-conditioned low-capacity control:
  `output/g25_pose_transport/planar_query_geometry/map_conditioned_offset_v1/seq9fit_seq10held_plane_parameter_recovery_v6.json`
- Cross-route strong oracles:
  `output/g25_pose_transport/planar_query_geometry/planar_pixel_oracle_seq12_crossroute_v1.json`
  and `planar_pixel_oracle_seq14_crossroute_v1.json`
