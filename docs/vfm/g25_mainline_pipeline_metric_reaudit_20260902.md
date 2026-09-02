# G25 VFM–plane localization mainline and metric re-audit (2026-09-02)

## Corrected scope

The production research mainline is **not** HLoc, LoFTR, SIFT, or a conventional
point-landmark localization system.  The intended representation is a retrievable
physical surface entity between an image and an individual landmark:

`2DGS surface -> finite plane/chart -> view-conditioned RADIO local field -> query
plane retrieval -> within-plane RADIO matching -> metric 2D--3D pose`.

HLoc is retained only as an external traditional-localization upper control.  SIFT is
retained only as a high-resolution local-feature diagnostic/refinement control.  Neither
number is allowed in the mainline metric column.

## Current mainline pipeline

### Map side

1. Build `PrimitiveSurfaceTable` from the route-disjoint training 2DGS.  No voxel,
   parent, or child identity is consumed by the active plane pipeline.
2. Render exact depth, normal, and primitive ownership from source-only mapping cameras.
3. Extract connected, bounded finite plane observations per view.
4. Fuse observations across views while retaining exact primitive membership.  Plane
   hulls are summaries, not the geometry authority.
5. Store each finite plane as the retrieval entity and retain its source-view RADIO
   descriptors, visibility support, and RADIO-token metric world points.

The OldHospital 480-view map has 1,149 planes, 13,192 fused observations, 182,554
assigned 2DGS primitives (26.293% of the global primitive inventory), and median finite
plane area 0.340 square metres.  After visibility/token filtering, the runtime field has
9,751 plane observations from 467 source views and the metric bank has 1,131,900 RADIO
token points.  This is a mid-level map, but it is not yet a compact canonical atlas: its
appearance is still stored as multiple view-conditioned observations.

### Query side

1. Extract dense RADIO tokens from the query RGB.
2. Use MoGe3 only to form connected plane-region masks.  MoGe3 depth and scale are not
   passed to PnP.
3. Pool RADIO within each query region and retrieve finite plane IDs against the
   source-view plane observation bank.
4. Match query RADIO tokens to source-view RADIO tokens inside retrieved planes and run
   a per-plane homography RANSAC.
5. Lift mapping tokens with frozen 2DGS depth, producing query 2D pixel to map 3D point
   correspondences.
6. Estimate pose with PnP-RANSAC and LM, followed by frozen reliability/candidate rules.

OldHospital query plane masks cover 31.03% of seq4 and 35.39% of seq8 pixels and contain
1,595/3,553 regions, or roughly 28 regions per query.  This fragmentation/low support is
an active bottleneck.  The sparse-foreground carrier groups observed components without
inventing hidden pixels, but did not improve the final operating point and remains KILL.

## Corrected OldHospital metric table (182 route-disjoint queries)

Threshold order is 0.1m/1deg, 0.25m/2deg, 0.5m/5deg, 1m/10deg, 2m/45deg.

| role | hit counts | recalls | median t/R | scientific status |
|---|---|---|---|---|
| 480-view raw Top5 RADIO-plane | 21/58/103/139/167 | 11.54/31.87/56.59/76.37/91.76% | 0.409m/0.750deg | cleanest current plane operating point |
| RADIO-plane historical best | 29/92/137/162/173 | 15.93/50.55/75.27/89.01/95.05% | 0.248m/0.468deg | mainline mechanism evidence; not pristine blind |
| RADIO-plane + SIFT agreement | 40/99/138/162/173 | 21.98/54.40/75.82/89.01/95.05% | 0.226m/0.446deg | traditional local-refinement control only |
| HLoc and HLoc hybrids | excluded | excluded | excluded | external traditional upper control only |

The authoritative mainline historical artifact is
`map_density_final_multiscale_spatial_strict_inlier_cascade_v1.npz`, file SHA256
`0781207c6529c7fbd83156d12cef37cebf81b7f9e01e2b0353df75fb80dc9e83`, content
SHA256 `ae339ebd4c4ec4bf82ea92cdc9d56c53758c9e6ae7a4073fae37589cf0ffabd8`.
Its independent replay is byte-identical.  The evaluation file SHA256 is
`3dda8080b39669473812356c68b11ab278942860334d01f146e7ed3c40071d13`.

Route split for the mainline historical best:

- seq4: 10/33/46/50/55 of 56; median 0.222m/0.580deg.
- seq8: 19/59/91/112/118 of 126; median 0.279m/0.381deg.

## Accuracy and lineage audit

- OldHospital mapping routes are seq1/2/3/5/6/7/9; queries are seq4/8.
- The 2DGS map, mapping cameras, plane field, and metric point bank contain no test route.
- Query camera artifacts expose intrinsics without reading pose members.
- Query plane extraction, RADIO retrieval, correspondence construction, PnP, and branch
  selection are frozen before pose labels are opened.
- Query depth/scale is not used by the pose solver.
- The mainline pose NPZ replays byte-for-byte.
- Numerical accuracy is therefore valid for the frozen artifact.
- Scientific generalization is weaker: OldHospital has been inspected repeatedly while
  choosing map densities and candidate/refinement branches.  Its best result is historical
  validation, not a preregistered blind claim.

ShopFacade independently supports route viability: a 60-view map reaches 100% at
2m/45deg and 99.03% at 1m/10deg over 103 queries, while the 231-view Top5 accuracy
control reaches 100%, 99.03%, 97.09%, and 91.26% at 2m/45deg, 1m/10deg, 0.5m/5deg,
and 0.25m/2deg.  These are cross-scene mechanism results, not production promotion.

## What is and is not solved

Established:

- Direct finite-plane RADIO retrieval is viable and much stronger than the old
  voxel-child-to-plane identity bridge.
- Increasing source-only plane observation density causally improves hard-scene recall.
- MoGe3 metric depth is not required to obtain a metric pose; map-side 2DGS depth supplies
  the 3D coordinate.
- The old voxel/child hierarchy is not required by the active localization path.

Not solved:

- Plane appearance is not yet a compact, canonical, viewpoint-robust feature atlas.
- Dense maps fragment one physical facade into many small plane instances and aliases.
- Query MoGe3 plane masks cover only about one third of OldHospital pixels and fragment
  each image into about 28 regions.
- Repeated facades can produce high-inlier wrong PnP poses.
- The MAtCha/chart route has improved source topology at stride2, but current artifacts
  are paired-initializer diagnostics and have not passed a model-neutral production gate.

## Correct next experiments

1. Keep direct RADIO plane retrieval as the global/mid-level mainline.
2. Replace view-max/mean plane appearance with a source-only canonical chart/plane local
   field that preserves multiple view-conditioned prototypes and metric UV position.
3. Use RADIO or another VFM local descriptor for within-plane high-resolution matching;
   SIFT remains only an oracle/control for the missing spatial-resolution capability.
4. Merge/deduplicate coplanar map fragments using exact support adjacency, not infinite
   plane or convex-hull identity.
5. Improve query plane support with boundary-aware sparse-occlusion reasoning while never
   hallucinating through dense foliage.
6. Freeze the complete pipeline on development data, then evaluate once on a new scene or
   genuinely unseen route.  HLoc may be reported beside it only as an external baseline.
