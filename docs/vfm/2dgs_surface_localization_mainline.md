# VFM-Centric Feature-Only 2DGS Surface Localization

## Scope

This is the production localization mainline. Its persistent map contains
2DGS geometry and features, but no mapping RGB:

```text
2DGS surface maplets
+ stable 2DGS surface anchors
+ RADIO-final surface/maplet feature prototypes
+ ALIKE anchor feature prototypes
+ visibility and view-conditioned feature-mode metadata
```

The two metric identities are:

```text
surface_maplet_id
    a VFM-footprint-scale 3D surface region used for coarse retrieval

stable_surface_anchor_id
    a quality-filtered 2DGS surface element used for fine matching and PnP
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
  -> fixed surface anchors

mapping RGB --offline only-->
  RADIO final full maps
  -> RADIO-final surface observations
  -> maplet descriptors and view-conditioned feature modes

mapping RGB --offline only-->
  ALIKE/FPN descriptors
  -> multi-view stable-anchor feature prototypes
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
  -> maplet-conditioned anchor sets
  -> query-conditioned support-prototype attention
  -> partial optimal transport with query/anchor dustbins

stored RADIO-final 2DGS surface feature modes
  -> several query-to-map affine layout modes
  -> predicted stable-anchor locations

query RGB only
  -> ALIKE descriptor measurement near predicted anchor locations
  -> explicit query-pixel/stable-anchor candidates

all pose modes
  -> grouped PnP
  -> fixed maplet+ALIKE candidate evidence
  -> optional pose-guided ALIKE feature EM
  -> accept refinement only when the original fixed evidence improves
```

The learned set matcher is deliberately budgeted to the top 12 scene maplets.
It is not run independently for every maplet touched by every query point. Its
PnP branch also has an explicit confidence gate: a high-dustbin candidate set
is retained as uncertainty evidence but does not trigger expensive,
low-information hypothesis generation.

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
features, retrieved maplets, and stored anchor prototypes. Every generated
pose is scored against this identical candidate pool. This rejects repeated
structure modes that have many self-consistent generation inliers but little
independent map evidence.

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
- A stable anchor ID denotes one fixed 2DGS surface element.
- Mapping image IDs are feature-mode labels, not runtime file handles.
- Maplet retrieval and anchor assignment have separate probabilities and
  explicit null states.
- Fit and verification query groups are disjoint.
- Pose-mode selection uses one fixed candidate pool.
- Pose-guided EM is restricted to RADIO-retrieved maplet anchors.
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

Training episodes contain query ALIKE points and the anchors of one correct or
hard-negative surface maplet. Validation image descriptors are excluded from
the anchor support prototypes used in that validation episode.

The matcher uses cross-attention, query-conditioned support-descriptor
attention, partial optimal transport, one-to-one anchor capacity, and explicit
dustbins. Its selected StMaryChurch checkpoint achieved:

| Metric | Image-disjoint validation |
|---|---:|
| learned R@1 / R@3 / R@5 | 30.21% / 52.78% / 62.65% |
| cosine R@1 / R@3 / R@5 | 28.05% / 49.03% / 59.91% |
| dustbin AUPRC | 0.8297 |

## StMaryChurch map

The supplied reconstruction is `/root/StMaryChurch2dgs.ply`.

The completed map contains:

- 1,031,773 supplied 2DGS primitives;
- 538,369 filtered surface elements;
- 864 VFM-aligned surface maplets;
- 23,150 stable surface anchors;
- 17,591 maplet-view observations;
- 293,268 stable-anchor observations;
- 140,547 stored ALIKE descriptors from 1,478 mapping feature modes;
- 176,199 stored RADIO-final 2DGS surface observations.

The historical mapping-depth artifact may remain in the experiment directory,
but it is neither accepted by the production CLI nor listed as a runtime
artifact.

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
  --anchor_top_l 20 \
  --hypothesis_count 128 \
  --output_jsonl <pose-results.jsonl> \
  --summary_json <runtime-summary.json>
```

The only RGB path accepted by this CLI is `query_image_root`.

## Validation status

StMaryChurch is a development scene: query GT had already been consumed during
diagnosis and parameter selection, so the final 530-query result is a complete
development/non-regression evaluation, not an untouched test claim.

Before the full run, an eight-query cross-shard probe achieved 100% pose
success with `0.501 m / 1.313°` median and `0.843 m / 2.251°` P90 using the
frozen 128-hypothesis configuration. The same queries under the previous
feature-only implementation had `2.103 m / 6.997°` median and
`3.857 m / 9.321°` P90.

All eight frozen query shards then completed. One strict maplet/anchor failure
was rerun with the map-only RADIO-final 2DGS surface-observation fallback. The
merger accepted that replacement only because the original row failed and the
replacement succeeded. Final coverage and pose success are both 530/530.

| Metric | Restored map-only mainline | Previous feature-only version |
|---|---:|---:|
| Median translation | 0.302 m | 1.178 m |
| P90 translation | 2.040 m | 6.059 m |
| P95 translation | 3.739 m | 7.085 m |
| Mean translation | 0.932 m | 2.305 m |
| Median rotation | 0.948° | 3.519° |
| P90 rotation | 6.109° | 20.210° |
| P95 rotation | 13.462° | 26.736° |
| Mean rotation | 2.870° | 7.852° |
| 10 cm / 5° recall | 9.25% | 6.23% |
| 25 cm / 10° recall | 40.94% | 15.28% |
| 50 cm / 10° recall | 70.19% | 31.13% |
| 5 m / 10° recall | 92.83% | 74.15% |
| Pose success | 100% | 100% |

The new pose has lower translation error on 423/530 queries, lower rotation
error on 428/530, and improves both on 396/530. It improves translation by at
least 0.1 m on 370 queries; the previous version is better by at least 0.1 m on
64.

The strict 5 cm / 5° recall is `1.13%`, below the previous version's `1.89%`.
This is reported explicitly rather than hidden by the much better median and
tail statistics.

The catastrophic tail is also not hidden: maximum error is
`25.07 m / 53.54°`. The remaining worst cases are repeated-structure feature
mode aliases. A future confidence/abstention policy must be developed on a
different scene or frozen split; it must not be tuned on these 530 labels.

Final branch counts are:

- 267 view-conditioned RADIO-maplet + ALIKE stable-anchor poses;
- 262 fixed-evidence-approved pose-guided ALIKE feature-EM poses;
- 1 RADIO-final 2DGS surface-observation fallback pose.

Every final row reports all forbidden runtime flags as false. Per-query runtime
is 14.57 s mean, 14.89 s median, and 16.60 s P90 under the four-worker
evaluation configuration.

Canonical reports:

- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v2/full_h128/full_results.jsonl`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v2/full_h128/merge_summary.json`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v2/full_h128/evaluation.json`
- `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v2/full_h128/errors.jsonl`
