# VFM-Centric Feature-Only 2DGS Surface Localization

## Scope

### Current execution scope

The active engineering path is inference against the already-frozen 2DGS
surface map and frozen RADIO-derived artifacts. It does **not** retrain or
reconstruct a Gaussian scene, run MAtCha/SfM alignment, or require a five-fold
geometry rebuild. Those pipelines are separate diagnostics and are not
needed to execute or validate the mainline localizer.

This is the production localization mainline. Its persistent map contains
2DGS geometry and features, but no mapping RGB:

```text
2DGS surface maplets
+ multi-view feature-aligned 2DGS surface anchors
+ RADIO-final surface/maplet feature prototypes
+ ALIKE anchor feature prototypes
+ visibility and view-conditioned feature-mode metadata
```

The two metric identities are:

```text
surface_maplet_id
    a VFM-footprint-scale 3D surface region used for coarse retrieval

stable_surface_anchor_id
    a multi-view ALIKE observation cluster lifted onto a quality-filtered
    2DGS surfel and used as a fixed metric point for fine matching and PnP
```

The production path does not use SfM points, SfM tracks, RADIO intermediate,
mapping-view depth, mapping RGB retrieval, or pairwise query/reference image
matching. In particular, LoFTR is not instantiated by the production
localizer.

## Offline map

Mapping RGB and calibrated mapping poses are used only during offline feature
attachment:

```text
high-quality 2DGS
  -> stable surface elements
  -> VFM-footprint-aligned surface maplets

mapping RGB --offline only-->
  RADIO final full maps
  -> RADIO-final surface observations
  -> maplet descriptors and view-conditioned feature modes

mapping RGB --offline only-->
  ALIKE keypoints and descriptors
  -> calibrated surfel ray-plane lifting
  -> multi-view geometric clustering
  -> feature-aligned stable-anchor prototypes

feature-aligned anchors
  -> feature-first per-maplet deployment index
  -> bounded geometry-anchor coverage fill
```

The deployment artifacts contain descriptors, coordinates, qualities,
visibility labels, and metric geometry. They contain no image paths that the
runtime can use to open a mapping image.

A stored mapping image ID is only a persistent label for one multi-modal
appearance/visibility feature mode. Selecting that label never retrieves or
opens the corresponding image.

## Online localization

```text
query RADIO final
  -> full-map learned projection
  -> multi-scale region descriptors
  -> top-K 2DGS surface maplets

query ALIKE points
  + RADIO-final descriptors sampled at the same query points
  -> query-aligned maplet posterior
  -> maplet-conditioned anchor sets
  -> query-conditioned support-prototype attention
  -> partial optimal transport with query/anchor dustbins
  -> transfer omitted top-L probability mass to the null state

stored RADIO-final 2DGS surface feature modes
  -> several query-to-map affine layout modes
  -> predicted stable-anchor locations

query RGB only
  -> ALIKE descriptor measurement near predicted anchor locations
  -> explicit query-pixel/stable-anchor candidates

all pose modes
  -> grouped four-point AP3P/PnP
  -> fixed maplet+ALIKE candidate evidence
  -> independent 2DGS feature-map likelihood
  -> soft weighted geometric EM
  -> optional pose-guided ALIKE feature refinement
  -> accept only when fixed evidence improves and feature evidence does not
     regress

low feature-support query, detected without GT
  -> broader feature-map-only anchor index
  -> same calibrated query features and geometry backend

new and stable map-only pose candidates
  -> one fixed feature-aligned 2DGS anchor map
  -> independent dense ALIKE likelihood and spatial coverage
  -> fixed score-margin and pose-disagreement safety policy
  -> final pose
```

The learned set matcher is deliberately budgeted to the retrieved scene
maplets. It is not run independently for every maplet touched by every query
point. Its PnP branch has an explicit conditional identity-confidence gate:
only groups whose retained anchor mass exceeds their conditional null mass
count toward the minimum needed to start PnP. Merely having non-zero candidate
mass is not confidence. Null-dominated points remain uncertainty evidence and
do not trigger low-information hypothesis generation. The gate also requires
the mean *absolute* null probability to stay below its configured ceiling;
conditional normalization alone cannot turn a 1e-6 retained mass into a
confident correspondence. Absolute posterior mass is retained for hypothesis
scoring, and every gate failure records whether conditional support or
absolute-null mass was the limiting condition.

Mapping-view retrieval uses the same conservation rule: view probabilities are
normalized over the complete graph before the returned view budget is applied;
unreturned view mass is carried into the mapping-view null. The serialized
anchor prefix includes its probability vector and null mass, so a smaller view
budget cannot manufacture confidence for a retained identity.

## Why the branches are complementary

RADIO final supplies semantic and whole-image context. It decides which
surface regions and which stored appearance modes are plausible. It is not
treated as a pixel-accurate keypoint descriptor.

ALIKE supplies high-resolution query measurements and stable-anchor identity
evidence inside the retrieved maplets. It never defines global scene identity
on its own.

The feature-mode layout branch preserves multi-view appearance modes instead
of averaging them into one descriptor. It uses an affine RANSAC model over
RADIO-final surface observations, then measures the corresponding stored
stable-anchor prototypes in the query. No support RGB is available to this
operation.

The fixed verifier is constructed before pose-mode generation from query ALIKE
features, retrieved maplets, and stored anchor prototypes. Every generated pose
is scored against this identical candidate pool and an independent 2DGS anchor
feature-map likelihood. This rejects repeated-structure modes that have
many self-consistent generation inliers but little independent map evidence.

Support descriptors are not reduced with a maximum over mapping views. The
likelihood uses a normalized mixture whose prior combines descriptor quality
and the stored anchor-to-camera direction. This prevents a single accidental
high-similarity view from dominating repeated-facade evidence.

## Production invariants

- `vfm_layer == radio_final`.
- `uses_radio_intermediate == false`.
- `uses_sfm_points == false`.
- `uses_sfm_tracks == false`.
- `uses_mapping_rgb_at_inference == false`.
- `uses_mapping_image_retrieval == false`.
- `uses_pairwise_image_matching == false`.
- `uses_view_depth_at_inference == false`.
- A maplet centroid is never a PnP point.
- A stable anchor ID denotes one fixed metric point attached to a 2DGS surfel.
- Mapping image IDs are feature-mode labels, not runtime file handles.
- Maplet retrieval and anchor assignment have separate probabilities and
  explicit null states.
- Layout evidence is estimated from a bounded raw-descriptor maplet pool
  wider than the returned maplet Top-K, then applied before the final Top-K;
  the pool size is frozen in the serialized matcher config so a layout
  explanation cannot be lost solely because descriptor ranking was truncated
  first.
- Truncating anchor candidates to top-L preserves probability: all omitted
  probability mass is added to the explicit null state.
- Fit and verification query groups are disjoint.
- Pose-mode selection uses one fixed candidate pool.
- Pose selection also uses an independently constructed 2DGS feature-map
  likelihood; generation inlier count alone cannot promote a pose.
- Pose-guided EM is restricted to RADIO-retrieved maplet anchors.
- A generated PnP pose cannot be promoted by itself: every branch, including
  direct RADIO surface-observation fallback, must pass finite fixed-map
  likelihood and fixed-map inlier support. If all independent evidence fails,
  the query abstains rather than retaining the proposal success flag.
- Query intrinsics and distortion are loaded from an exact per-query manifest;
  no median-intrinsics substitution is allowed in the canonical run.
- A GT-free feature-support gate may route coverage-poor queries to a broader
  stored anchor index. The gate never reads pose labels.
- If and only if every strict maplet-scoped pose mode fails, the selected
  RADIO feature modes may be remeasured against the full stable-anchor feature
  map; this remains map-only and never introduces support RGB.
- If stable-anchor measurement still produces no pose, RADIO-final query
  tokens may match the stored metric 2DGS surface observations directly. This
  last-resort grouped-PnP branch uses cached feature+XYZ map entries, not SfM
  points, mapping images, view depth, or back-projection.
- Query GT pose is absent from inference inputs.

The localizer validates map/checkpoint metadata and rejects artifacts that set
any forbidden runtime flag.

## Training

### Surface-maplet mapper

The full RADIO-final map is projected before any region sampling. Mapper
positives are observations of the same 2DGS surface maplet from different
mapping images. The fit and validation image sets are disjoint.

The StMaryChurch checkpoint was selected at epoch 120:

| Metric | Image-disjoint validation |
|---|---:|
| R@1 | 73.02% |
| R@5 | 95.60% |
| R@10 | 98.20% |
| MRR | 0.8303 |

### Surface-anchor set matcher

Training examples replay deployment: ALIKE is detected on each held-out mapping
image, RADIO-final descriptors are sampled at those detected positions, and the
same retrieval and top-L truncation used online constructs the candidate set.
Positive anchor identity is assigned by calibrated 2DGS geometry; unmatched
detected points provide the null examples. Validation images are excluded from
the anchor support prototypes used in their examples.

The matcher uses cross-attention, query-conditioned support-descriptor
attention, partial optimal transport, one-to-one anchor capacity, explicit
dustbins, and class-balanced assignment/no-match loss. The selected
feature-preferred StMaryChurch deployment-replay checkpoint achieved:

| Metric | Image-disjoint validation |
|---|---:|
| learned R@1 / R@3 / R@5 | 43.49% / 69.52% / 77.74% |
| cosine R@1 / R@5 | 43.15% / 77.40% |
| dustbin AUPRC | 99.38% |

The RADIO-final identity expert samples the mapped RADIO-final field at the
same ALIKE pixels and concatenates it with the local descriptor. The persistent
map stores these fused prototypes, never the source images. Its image-disjoint
deployment-replay validation achieved:

| Metric | ALIKE + RADIO-final identity expert |
|---|---:|
| learned R@1 / R@3 / R@5 | 53.97% / 78.91% / 86.62% |
| cosine R@1 / R@3 / R@5 | 51.25% / 74.38% / 83.45% |
| dustbin AUPRC | 99.82% |

The expert is not a drop-in replacement for the ALIKE main candidate. It adds
candidate diversity; final promotion still requires a fixed, ALIKE-only 2DGS
feature-map score margin and the pose-disagreement safety rule.

## StMaryChurch map

The current experiment uses `/root/StMaryChurch2dgs_clean.ply`. This clean file
retains source indices into the full 2DGS parameterization, so normals, disk
bases, and scales are recovered without nearest-neighbor remapping.

The feature-preferred deployment map contains:

- 591,182 clean 2DGS primitives;
- 864 VFM-aligned surface maplets;
- 17,220 deployment anchors;
- 11,582 multi-view feature-aligned anchors;
- 5,638 bounded geometry coverage anchors;
- 37,426 ALIKE observations attached to the feature-aligned subset;
- 1,476 deployment-replay examples containing 755,712 detected query nodes.
- 10,580 anchors and 25,890 exact replay-aligned 192-D
  ALIKE+RADIO-final support descriptors in the semantic identity expert.

The persistent deployment artifacts contain no mapping RGB. Historical
mapping-depth or image artifacts may remain elsewhere in experiment storage,
but are neither accepted by the production CLI nor listed as runtime inputs.

## Canonical localization command

```bash
PYTHONPATH=. python feature_extract/tools/vfm/localize_2dgs_surface_queries.py \
  --query_manifest <query-radio-final-manifest.json> \
  --query_image_root /hy-tmp/Cambridge_stdloc/StMarysChurch \
  --surface_mapper_checkpoint <surface-maplet-mapper.pt> \
  --maplets <surface-maplets.npz> \
  --anchors <stable-surface-anchors.npz> \
  --local_descriptor_bank <anchor-alike-bank.npz> \
  --radio_surface_feature_bank <observation-bank.npz> \
  --surface_anchor_matcher_checkpoint <surface-anchor-set-matcher.pt> \
  --camera_model_dir <training-camera-intrinsics-model> \
  --query_camera_manifest <exact-query-camera-manifest.json> \
  --anchor_top_l 20 \
  --hypothesis_count 64 \
  --output_jsonl <pose-results.jsonl> \
  --summary_json <runtime-summary.json>
```

The only RGB path accepted by this CLI is `query_image_root`.
For the complementary semantic identity expert, use its 192-D descriptor bank
and matcher checkpoint together with `--augment_alike_with_radio_final`.

## Validation status

StMaryChurch is a development scene: query GT had already been consumed during
diagnosis and parameter selection, so the final 530-query result is a complete
development/non-regression evaluation, not an untouched test claim.

The final ALIKE+RADIO-final identity expert completed all eight query shards.
Its strict conditional-null gate returned a pose for 516/530 queries and
abstained on 14 instead of forcing ambiguous PnP. The final selector compares
that expert with the previous safe feature-map result against one fixed,
oriented ALIKE-anchor map. It keeps the safe candidate unless the expert
improves fixed-map log likelihood by at least `0.05`; disagreement above
`5 m` or `10°` also retains the safe candidate. These tests use no GT.

The selector retained the expert on 82 queries and the safe candidate on 448.
Final coverage and pose success are 530/530.

| Metric | Final identity-expert selector | Previous fixed selector |
|---|---:|---:|
| Median translation | **0.197 m** | 0.216 m |
| Mean translation | **0.794 m** | 0.828 m |
| P90 translation | **1.627 m** | 1.890 m |
| P95 translation | 3.702 m | 3.702 m |
| Maximum translation | 25.068 m | 25.068 m |
| Median rotation | **0.619°** | 0.680° |
| Mean rotation | **2.444°** | 2.557° |
| P90 rotation | **4.031°** | 5.203° |
| P95 rotation | **13.113°** | 13.462° |
| Maximum rotation | 53.537° | 53.537° |
| 5 cm / 5° recall | **7.17%** | 5.85% |
| 10 cm / 5° recall | **24.15%** | 21.13% |
| 25 cm / 10° recall | **58.49%** | 54.53% |
| 50 cm / 10° recall | **78.87%** | 77.55% |
| 5 m / 10° recall | **93.77%** | 93.40% |
| Pose success | 100% | 100% |

The candidate-pose oracle over the two experts reaches a `0.183 m` translation
median. The oracle over every stored hypothesis reaches `0.105 m`, but only
15.85% of all queries have a stored hypothesis within `4 cm / 1°`. Therefore
selection regret remains measurable, while the dominant 4 cm bottleneck is
still candidate identity/generation and a missing reliable local metric
refinement basin. The result must not be presented as meeting a 4 cm target.

Every final row records selector evidence and all forbidden runtime flags as
false. No mapping RGB is stored or opened by either candidate generator or the
final verifier.

Canonical reports:

- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/full_vfm_identity_mixture_selector_safe/full_results.jsonl`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/full_vfm_identity_mixture_selector_safe/merge_summary.json`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/full_vfm_identity_mixture_selector_safe/evaluation.json`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/full_vfm_identity_mixture_selector_safe/errors.jsonl`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/full_vfm_identity_mixture_selector_safe/oracle_audit.json`
