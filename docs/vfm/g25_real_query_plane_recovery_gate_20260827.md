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

