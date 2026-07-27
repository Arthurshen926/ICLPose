# V4: anchor-free 2DGS surface feature alignment

## Decision

V3 is frozen as a baseline. V4 does not extend its anchor/PnP chain.

The V4 map is:

```text
clean 2DGS metric surface
+ a bounded RADIO-final distribution mixture per metric maplet for coarse retrieval
+ one fine RADIO distribution per retained surface cell
+ surface confidence / uncertainty
```

It does not contain stable anchor identity, per-anchor descriptor lists, ALIKE
descriptors, mapping RGB, mapping image paths, SfM points/tracks, RADIO
intermediate features, or pairwise image matches.

ALIKE is a detector only. Offline, a detection ray is assigned to a clean
2DGS surfel with 2DGS depth and then intersected with that surfel plane. Its
RADIO-final descriptor—not its ALIKE descriptor—is fused into the surface
field. At query time, the implementation evaluates only ALIKE's score-head
channel and never materializes its dense descriptor map. ALIKE contributes
only a detector heat map; matching evidence remains RADIO-final.

## Why an anchor does not store multiple descriptors

A list of descriptors is evidence that a purported point identity is not a
stable measurement primitive. V4 represents each fine surface cell as one
normalized feature mean, one coherence-derived uncertainty, and one
support/confidence value. View observations are offline measurements and are
not deployment identities.

Maplets remain useful without anchors: they select disconnected local surface
patches that can be rendered and aligned jointly. They are regions, not
fine-point identities. A maplet may therefore use a bounded mixture (at most
four weighted RADIO centroids) to represent genuine regional viewpoint
multimodality. These components have no view/image identity and are not raw
observation lists. This is categorically different from storing several
descriptors behind a claimed stable point.

## Fine objective

For a pose hypothesis, the selected maplet cells are projected with the exact
query calibration. A depth-buffer pass keeps the visible surface sample in each
feature cell. The objective combines:

- RADIO metric similarity with a smooth null/outlier mixture;
- ALIKE detector heat only as a sub-pixel spatial weight;
- confidence and feature uncertainty;
- disjoint fit and held-out surface subsets.

The optimizer is bounded in SE(3) and accepts a step only when both fit and
held-out evidence improve. No 2D–3D correspondence set or final PnP is formed.

## Metric adapter

The optional fine adapter is a pointwise RADIO-final residual mapper. Positives
are observations of the same clean 2DGS surfel from different mapping views.
ALIKE descriptor values are never read. On the StMaryChurch held-out
observation split it changed:

| metric | raw mapped RADIO | fine metric adapter |
|---|---:|---:|
| R@1 | 37.80% | 42.11% |
| R@5 | 69.65% | 77.42% |
| R@10 | 79.37% | 87.71% |
| MRR | 0.521 | 0.576 |

This validates improved identity discrimination. It does not by itself prove a
metric pose basin.

## Required gate before full localization

The fine stage must be tested with oracle-visible maplets so retrieval and
coarse initialization cannot hide a bad local field:

- initial translation at most 20 cm: at least 80% must converge to 4 cm / 1°;
- initial translation at most 30 cm: at least 60% must converge to 4 cm / 1°.

A full 530-query localization run is not a valid next step when this gate
fails. The basin report explicitly labels its use of query ground truth for
controlled perturbation and oracle maplet visibility.

## Canonical construction

```bash
PYTHONPATH=. python feature_extract/tools/vfm/build_detector_weighted_2dgs_surface_field.py \
  --gaussian_ply /root/StMaryChurch2dgs.ply \
  --clean_gaussian_ply /root/StMaryChurch2dgs_clean.ply \
  --mapping_manifest output/vfm_tokens/StMarysChurch/full_1024x576/train_manifest.json \
  --mapping_pose_file /hy-tmp/Cambridge_stdloc/StMarysChurch/dataset_train.txt \
  --mapping_camera_manifest output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/mapping_camera_manifest.json \
  --mapping_depth_bank output/vfm/2dgs_surface/StMarysChurch/full_train/final_surface/mapping_depth_bank.json \
  --alike_detection_cache output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/cache \
  --surface_mapper_checkpoint output/vfm/2dgs_surface/StMarysChurch/full_train/surface_maplet_mapper_full.pt \
  --metric_mapper_checkpoint output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v4_surface_field/surface_metric_mapper.pt \
  --maplets output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/calibrated_feature_preferred_surface_maplets.npz \
  --output_field output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v4_surface_field/production_surface_feature_field.npz \
  --summary_json output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v4_surface_field/production_surface_feature_field_summary.json
```

Mapping depth, poses, detector cache, and mapping tokens are builder inputs
only. They are not accepted by the fine alignment module or stored in the
surface field.

The production retrieval artifact is separately stripped from the historical
construction container:

```bash
PYTHONPATH=. python feature_extract/tools/vfm/build_surface_retrieval_maplet_bank.py \
  --legacy_maplets output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v3_feature_aligned/calibrated_feature_preferred_surface_maplets.npz \
  --maximum_components 4 \
  --output_maplets output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v4_surface_field/production_surface_retrieval_maplets.npz \
  --summary_json output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v4_surface_field/production_surface_retrieval_maplets_summary.json
```

The output has 864 maplets and 3,226 anonymous mixture components. It stores
maplet ID/geometry, mixture offsets/centroids/weights, quality, uncertainty,
and contract metadata. Anchor arrays, surface-element lists, per-view
descriptors, view IDs, and view coordinates are absent.

## StMaryChurch validation result

The production field used all 1,479 mapping views and the clean 2DGS prior.
It fused 151,477 accepted detector observations into 21,630 multi-view surface
cells. The fine metric adapter was applied before fusion.

On all 530 test queries, Top-12 compact-mixture retrieval hit at least one of
the twelve most visible GT maplets in 70.75% of queries and covered 11.29% of
that visible set on average. The single-centroid version reached 68.87% /
11.01%; applying the point-level metric adapter to maplet centroids degraded it
to 54.53% / 6.70%. The mixture is therefore the retained compression, but
coarse retrieval is still materially below production quality.

The 16-query, 64-trial oracle-maplet basin test failed the required gate:

| initial perturbation | final translation median | final translation P90 | 4 cm / 1° |
|---:|---:|---:|---:|
| 5 cm / 1° | 5.33 cm | 14.99 cm | 6.25% |
| 10 cm / 2° | 10.00 cm | 14.60 cm | 6.25% |
| 20 cm / 4° | 20.00 cm | 26.06 cm | 0% |
| 30 cm / 6° | 30.00 cm | 32.18 cm | 0% |

The decisive failure is translational: rotation often improves, while
translation commonly remains at its initialization or moves in the wrong
direction. Thus one RADIO-final vector per detector-repeatable surfel improves
cross-view identity but does not provide a sufficiently smooth and
discriminative tangential metric gradient. Full 530-query promotion is
intentionally blocked; V3 remains the measured deployment baseline.
Removing ALIKE detector weighting did not repair the basin: its 5 cm median
became 6.64 cm and the 4 cm / 1° rate remained 6.25%. The detector term is
therefore not the root cause.

Two independent blockers are now isolated: coarse maplet recall and the fine
translation basin. The next experiment must change their learned measurement
objectives, not add another matcher. A valid direction is joint
surface-conditioned regional retrieval plus a high-resolution feature decoder
trained with explicit pose-Jacobian/basin supervision, so both wrong-region
negatives and nearby tangential displacements are calibrated. It must pass the
retrieval audit and the same oracle gate before any end-to-end claim.

`localize_2dgs_surface_feature_field.py` is therefore a refinement entry, not a
promoted end-to-end locator: it strictly requires coarse poses declaring a
RADIO-final maplet source. A new clean coarse 6-DoF pose-hypothesis model has
not been claimed or substituted with V3 PnP output. Building one before the two
measurement gates pass would only reconnect an invalid chain.
