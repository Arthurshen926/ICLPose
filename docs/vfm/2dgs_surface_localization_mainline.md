# VFM-Centric 2DGS Surface Localization Mainline

## Scope

The production representation is now **SfM-point-free and SfM-track-free**.
Mapping poses and camera intrinsics remain necessary, but no sparse SfM point,
track identity, track observation, or track covisibility enters the
localization map.

The two persistent identities are:

```text
surface_maplet_id
    a RADIO-final-scale 3D surface region used for global/coarse retrieval

stable_surface_anchor_id
    a fixed, quality-filtered 2DGS surface element used for local matching/PnP
```

Raw 2D Gaussian indices are not exposed as retrieval landmarks. A VFM maplet
may contain many primitives, and only multi-view stable surface elements become
metric anchors.

RADIO intermediate is not part of this mainline. RADIO final supplies
multi-scale context; ALIKE or an explicitly configured local FPN supplies
pixel-accurate descriptors.

## Offline map

```text
high-quality 2DGS PLY
  -> opacity/geometry/class filtering
  -> persistent surface elements and optional virtual cells
  -> real mapping camera alpha-contribution buffers

real mapping RGB
  -> frozen RADIO final full map
  -> full-map surface-maplet mapper trained from 2DGS cross-view identity
  -> 1/3/5/9-token final-layer region encoding

RADIO token footprints + 2DGS surface contribution
  -> graph-fused VFM surface maplets
  -> per-view RADIO-final descriptors and real support regions

each maplet's supported surface elements
  -> multi-view contribution/normal/opacity/geometry filtering
  -> spatially balanced stable surface anchors
  -> exact real-view projections and local-feature support descriptors
```

The old `Vfm2DgsAnchorMap` is treated as a mapping-time fused region layer. It
is no longer sent directly to PnP. `VfmSurfaceMapletBank` and
`StableSurfaceAnchorMap` are the production artifacts.

## Online localization

```text
query RGB
  └─ RADIO final -> multi-scale query regions

RADIO-final regions
  -> top-K surface maplets
  -> multi-region support-layout consistency
  -> top real mapping support views

query/support RGB pairs selected only by RADIO final
  -> LoFTR local correspondences
  ├─ official 2DGS support-view depth -> dense metric 2DGS surface points
  └─ stable-anchor projection index -> sparse stable 2DGS anchors

dense-depth PnP + stable-anchor PnP
  -> independent cross-modal pose-consistency gate
  -> selected metric pose

if the two 2DGS lifting paths do not agree
  -> reject the dense hypothesis
  -> stable-anchor/ALIKE fallback
```

The whole-image support-layout step is important for repeated windows. It
requires several neighboring query regions to agree with one real support
view's 2D arrangement. A single semantically similar window cannot receive the
same phase-confidence boost on its own.

LoFTR is not used as an unconstrained whole-map retriever. RADIO final first
selects four support views; LoFTR only measures correspondences in those
VFM-selected pairs. This preserves the VFM's semantic/context advantage while
using a high-resolution matcher for the pixel correspondence that a final-layer
VFM token does not directly provide.

## Module boundaries

### RADIO final

RADIO final answers which surface region is visible. Its feature map keeps full
image context through the mapper before observation sampling. Multi-scale
pooling is performed on final features only, so the production descriptor stays
semantically stable while observing larger non-periodic context.

### VFM-conditioned local correspondence

LoFTR answers which pixels correspond inside the support pairs selected by
RADIO final. The primary metric lift samples the official 2DGS renderer depth
for the mapping-view pixel and back-projects it to the common 2DGS world frame.
A second path associates the same match with projected stable 2DGS anchors.
ALIKE remains a lazy fallback and never defines global retrieval identity.

### 2DGS geometry

2DGS supplies surface position, normal, tangent extent, visibility contribution,
and covariance. Dense primitives are a candidate supply, not independent
retrieval entries. Stable anchors must have multiple real-view observations and
pass opacity, geometry-confidence, surface-normal, scale, and coverage checks.

### PnP

PnP remains the metric solver because the fine stage returns explicit
`(query_pixel, stable_surface_xyz)` pairs. Multiple candidates from one query
point are mutually exclusive and are never passed as independent matches.

## Production invariants

- `vfm_layer == radio_final`.
- `uses_radio_intermediate == false`.
- `uses_sfm_points == false`.
- `uses_sfm_tracks == false`.
- LoFTR support images are selected only by RADIO final.
- Dense metric points come from the official 2DGS depth bank.
- Dense-depth and stable-anchor poses are checked independently before
  selection.
- Maplet retrieval and anchor assignment have separate scores and nulls.
- A maplet centroid is never used as a PnP point.
- A stable anchor ID denotes one fixed surface element.
- Real support views remain multi-modal until matching/likelihood
  marginalization.
- Pose verification uses the same fixed candidate pool for every hypothesis.
- Fit and verification query groups are disjoint.
- Query GT pose is absent from inference artifacts.

The artifact constructors enforce the first four invariants and reject an
incompatible metadata manifest.

## Mapper supervision

The production mapper is not the previous SfM-track/landmark mapper. Its only
positive relation is:

```text
same surface_maplet_id + different real mapping image
```

All other maplets are negatives, with an extra separation margin for nearby
surface maplets. Fitting and checkpoint selection use disjoint mapping image
sets. Checkpoint selection is lexicographic Recall@5, Recall@1, then MRR.
The mapper always consumes the complete RADIO-final map before any region is
sampled; the 1/3/5/9 context pooling remains differentiable during training.
Neither query/test pose nor query/test image is an input to mapper fitting.

## StMaryChurch build

The first supplied reconstruction is:

```text
/root/StMaryChurch2dgs.ply
```

The canonical build is four explicit steps because mapper supervision,
renderer contribution, surface identity, and local measurement support are
separately inspectable:

```bash
PYTHONPATH=. python feature_extract/tools/vfm/train_surface_maplet_mapper.py \
  --surface_maplets <bootstrap-mapping-dir>/surface_maplets.npz \
  --radio_final_manifest <radio-final-mapping-manifest.json> \
  --output_checkpoint <mapping-dir>/surface_maplet_mapper.pt \
  --summary_json <mapping-dir>/surface_maplet_mapper_training.json

PYTHONPATH=. python feature_extract/tools/vfm/build_vfm_2dgs_anchor_map.py \
  --gaussian_ply /root/StMaryChurch2dgs.ply \
  --reference_manifest <radio-final-mapping-manifest.json> \
  --reference_pose_file /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_train.txt \
  --camera_model_dir /hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/StMarysChurch/model_train \
  --layer_name radio_final \
  --surface_maplet_mapper_checkpoint <mapping-dir>/surface_maplet_mapper.pt \
  --require_full_map_mapper \
  --canonical_vfm_2dgs \
  --surface_adjacency_element_radius_cap 0.10 \
  --output_npz <mapping-dir>/region_map.npz \
  --surface_npz <mapping-dir>/surface_elements.npz \
  --observation_bank_npz <mapping-dir>/observation_bank.npz \
  --summary_json <mapping-dir>/region_summary.json

PYTHONPATH=. python feature_extract/tools/vfm/build_2dgs_surface_map.py \
  --surface_elements <mapping-dir>/surface_elements.npz \
  --region_map <mapping-dir>/region_map.npz \
  --observation_bank <mapping-dir>/observation_bank.npz \
  --radio_final_manifest <radio-final-mapping-manifest.json> \
  --reference_pose_file /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_train.txt \
  --camera_model_dir /hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/StMarysChurch/model_train \
  --gaussian_ply /root/StMaryChurch2dgs.ply \
  --surface_maplet_mapper_checkpoint <mapping-dir>/surface_maplet_mapper.pt \
  --require_full_map_mapper \
  --output_maplets <mapping-dir>/surface_maplets.npz \
  --output_anchors <mapping-dir>/stable_surface_anchors.npz \
  --summary_json <mapping-dir>/surface_map_summary.json

PYTHONPATH=. python feature_extract/tools/vfm/build_2dgs_anchor_local_bank.py \
  --anchors <mapping-dir>/stable_surface_anchors.npz \
  --image_root /hy-tmp/Cambridge_stdloc/StMarysChurch \
  --camera_model_dir /hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/StMarysChurch/model_train \
  --output_bank <mapping-dir>/anchor_alike_bank.npz \
  --summary_json <mapping-dir>/anchor_alike_summary.json

PYTHONPATH=. python feature_extract/tools/vfm/build_2dgs_mapping_depth_bank.py \
  --ply /root/StMaryChurch2dgs.ply \
  --mapping_pose_file /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_train.txt \
  --camera_model_dir /hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/StMarysChurch/model_train \
  --output_root <mapping-dir>/mapping_depth \
  --manifest <mapping-dir>/mapping_depth_bank.json
```

The first mapper can be bootstrapped from a geometry-fused map made with a
frozen prior mapper. After training, rebuild the region/maplet artifacts with
the surface mapper. Renderer contribution buffers and surface topology may be
reused with `--reuse_contribution_dir` and `--reuse_surface_npz`; this keeps the
token/surface candidate pool fixed during descriptor ablations.

## Completed StMaryChurch training and mapping evidence

The complete mapping run uses all 1,487 mapping images and all 1,031,773
supplied 2DGS primitives:

- 538,369 filtered surface elements;
- 864 VFM surface maplets and 23,150 stable 2DGS anchors;
- 17,591 maplet-view observations and 293,268 stable-anchor observations;
- a 1,487-view, 1024x576 official-2DGS depth bank with a per-view checksum;
- the image-disjoint surface mapper was selected at epoch 120 with
  R@1/R@5/R@10 `73.02%/95.60%/98.20%`, MRR `0.8303`, over 4,433 validation
  queries and 857 prototypes;
- final leave-one-view-out maplet retrieval reached
  R@1/R@5/R@10 `77.27%/97.88%/99.27%`;
- ALIKE stable-anchor identity within the correct maplet, with the support
  image excluded, reached R@1/R@3/R@5 `25.20%/45.87%/57.14%` over 140,547
  observations;
- exact projected-anchor PnP had `0.0000141 cm / 0°` median error; with one
  pixel Gaussian noise it had `1.20 cm / 0.0417°` median error.

## Frozen 530-query StMaryChurch result

The final pose-only inference completed all 530 query images before
`dataset_test.txt` was joined. Source and artifact hashes matched the pre-run
freeze. All 530 images produced a pose.

| Metric | Final 2DGS/VFM | Frozen previous version |
|---|---:|---:|
| Median translation | 0.154 m | 1.178 m |
| P90 translation | 0.441 m | 6.059 m |
| Median rotation | 0.404° | 3.519° |
| P90 rotation | 1.437° | 20.210° |
| 10 cm / 5° | 25.47% | 6.23% |
| 25 cm / 10° | 73.96% | 15.28% |
| 50 cm / 10° | 91.51% | 31.13% |
| 5 m / 10° | 97.17% | 74.15% |
| Pose success | 100% | 100% |

The required non-regression gates—median translation, P90 translation, median
rotation, and success rate—all pass. The paired result improves translation on
469/530 images, rotation on 481/530, and both on 458/530.

The catastrophic tail is not hidden: the maximum error is
`50.57 m / 177.16°`. The primary official-depth branch was selected on 509
images; the sparse stable-anchor fallback was selected on 21 and is the main
remaining long-tail weakness. The worst depth-branch case is a repeated-view
alias with only eight coarse VFM-PnP inliers. Future work should add a
GT-independent confidence/abstention gate and multi-support depth consensus,
and must be evaluated on a fresh scene or a new frozen split.

Canonical reports:

- `output/vfm/2dgs_surface/StMarysChurch/full_train/evaluation_final/evaluation.json`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/evaluation_final/pose_free_inference_summary.json`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/evaluation_final/long_tail_analysis.json`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/evaluation_final/inference_freeze.json`

Protocol disclosure: the final architecture and fixed cross-modal thresholds
were selected using a 32-query stratified diagnostic probe after the original
v1 530-query result had been opened. Therefore this is a complete frozen
non-regression evaluation, but not a claim of a pristine one-shot untouched
test.

## Accuracy gates

No end-to-end promotion is allowed from mapping-only recall. Report:

1. renderer depth/normal/contribution consistency;
2. stable-anchor view count, coverage, scale, and geometry quality;
3. maplet Recall@1/5/10;
4. whether the retrieved maplet contains the correct metric anchor;
5. local anchor precision/recall, dustbin AUPRC, and pixel EPE;
6. maplet oracle, anchor oracle, grouped-hypothesis oracle, and selected pose;
7. median/P90 translation, median rotation, success rate, and catastrophic
   tail/abstention.

The final method was paired against the frozen previous version on all 530
identical query IDs and passed median translation, P90 translation, median
rotation, and success-rate gates. StMaryChurch is now consumed for development;
additional architecture or threshold selection requires a new frozen
evaluation scope.
