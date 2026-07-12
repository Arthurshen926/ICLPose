# Real-Image Landmark Localization Mainline

## Scope

The active method localizes a real query image against an SfM landmark map. It
does not use rendered query/reference pairs, place-image retrieval, or submaps in
the production path. SfM tracks provide the 3D identity and XYZ used by PnP.

The staged pipeline is:

```text
offline map
  real mapping RGB -> frozen RADIO -> retrieval mapper full feature map
  -> sample mapped descriptors at SfM observations
  -> TrackPrototypeBuilder aggregation
  -> global landmark prototype bank + local maplets + support-view index

online query
  real query RGB -> RADIO final/intermediate + ALIKE detector features
  -> heatmap/spatially diverse query points
  -> global full-bank top-L landmark proposals (no ratio hard gate)
  -> candidate-conditioned local maplet matcher
  -> KEEP / SWITCH / DROP-compatible resolved 2D-3D identities
  -> optional selected-only RGB measurement (not promoted yet)
  -> conflict-resolved, coverage-safe geometry backend
```

## Module Responsibilities

### Global retrieval

- `JointFeatureMapper` maps RADIO final feature maps into the indexed descriptor
  space.
- Query and map descriptors must share the same descriptor-space manifest.
- Production map descriptors use full-map projected observations only:
  full feature map -> mapper -> sample each SfM observation -> aggregate by
  track.
- Post-aggregate 1x1 projection is diagnostic-only because it is not the same
  distribution as full-map projection.
- Global retrieval returns top-L mutually exclusive track hypotheses per query
  point. Ratio and nearest-neighbor margin are features, not default hard gates.

### Landmark representation

- SfM track ID is the physical identity and SfM XYZ is the PnP anchor.
- `TrackPrototypeBuilder` is shared by retrieval training and offline bank
  construction.
- The current default is a normalized single prototype built from projected
  observations. Mean, robust mean, medoid, geometric median, and multi-prototype
  variants remain ablations.
- A maplet is a local 16-64-track context around a proposal. It is not an image
  retrieval submap.

### Local assignment

- `CandidateMapletMatcher` consumes query-neighborhood features, candidate
  maplet tracks, multiple real support views, RADIO intermediate context, and
  static candidate quality features.
- It performs query/support cross-context matching, partial assignment with a
  dustbin, learned support-view aggregation, and top-L candidate-set reasoning.
- Geometry heads predict nested real-residual events (`p<=1px`, `p<=2px`, and
  `p<=5px`). Visibility metrics are valid only when visibility loss is enabled.
- The rescue policy has an explicit action target: keep the coarse baseline, or
  switch to any candidate that is within the configured residual threshold when
  the baseline is geometrically invalid. KEEP is the set-level dustbin.
- Assignment score and pose-selection confidence are separate tensors. Candidate
  identity must never be selected with one score and silently filtered with a
  semantically unrelated row score.

### Pixel measurement

- RGB measurement is disabled until assignment-only center PnP passes its gate.
- When enabled, it must run only on accepted high-value identities, use real RGB
  support observations, and output a multimodal location likelihood plus
  KEEP/UPDATE/DROP and uncertainty.
- The legacy proposal-independent patch offset adapter is not the production
  measurement path.

### Geometry backend

- Duplicate query observations and duplicate tracks are resolved before PnP.
- Uniform OpenCV RANSAC receives canonical token/track order. Learned score order
  must not leak into uniform-RANSAC sampling.
- Confidence may choose conflict winners or an explicit subset, but the solver
  input is canonicalized afterward.
- The next backend gate is coverage-safe multi-hypothesis PnP with held-out
  verification. Uncalibrated network scores are not treated as covariance.

## Data

All active training inputs are derived from real Cambridge/OldHospital imagery
and the retriangulated COLMAP model:

- mapping RGB and SfM observations for projected landmark construction;
- real query RGB and frozen RADIO final/intermediate caches;
- ALIKE query and support observation descriptors;
- global full-bank top-L proposals, including real hard negatives;
- SfM observation xy/XYZ/visibility/reprojection residuals used as targets only;
- maplet covisibility, local geometry, track quality, and support-view metadata.

Query pose is never a model input. It is used only to generate training labels
and evaluation errors. Query images are excluded from prototype support images
by the bank manifest.

The active S4 development split is sequence-balanced rather than a contiguous
block from one video. It holds out 15 images from each mapping sequence
(`seq1`, `seq2`, `seq3`, `seq5`, `seq6`, `seq7`, and `seq9`) and explicitly
partitions them into 9 train, 3 validation, and 3 late-test images per sequence.
The resulting 63/21/21 query sets are disjoint, all 105 query images are removed
from both support-token and support-observation manifests, and they have zero
image overlap with the historical `seq4`/`seq8` full182 query set. The split
manifest is a hashed model input; count-based implicit slicing is not accepted
for this experiment.

This is leakage-free for S4 prototype support and local-matcher supervision, not
an end-to-end untouched global-mapper test. The frozen R1 mapper predates this
split and its image-pair training saw 57/63 train, 19/21 validation, and 18/21
late images. Therefore the current 105-query run isolates whether broader S4
episodes improve local assignment; it cannot promote the complete localization
system. A later end-to-end run must rebuild the R1 pair data, frozen negative
snapshot, projected bank, and S4 artifacts after excluding all validation/test
images upstream.

The image-referenced lazy episode store loads feature/RGB data per candidate
group and keeps bounded caches. Full feature/RGB duplication is not required.

### Coordinate and inference protocol

- Detector xy, local-window radii, residual thresholds, camera intrinsics, and
  PnP must use one canonical image grid. The current OldHospital protocol is
  1024x576. Native 1920x1080 and canonical artifacts cannot share absolute
  pixel thresholds.
- Source RGB may remain at native resolution, but RADIO feature sampling uses
  the COLMAP model that defines the detector/query xy grid. The source RGB size
  is not treated as the xy coordinate space.
- RADIO intermediate caches record a coordinate-space ID, coordinate-model
  hashes, explicit checkpoint hash/load mode, sampling convention, and source
  manifests. A mismatch is an error, not a warning.
- Production inference artifacts omit nearest-visible tracks, GT residuals,
  labels, and projected assignment targets. Global proposal generation, support
  reranking, and local matcher forward run without query pose supervision.
  A separate evaluator joins frozen scores to GT artifacts after inference.

## Training Stages

1. **R0/R1 global mapper**: image correspondence warm-up followed by
   query-to-landmark episodic specialization. Measurement/fine losses are off.
2. **S2/S3 proposals and representation**: full-bank top-L recall/oracle audits,
   aggregation sweep, maplets, and support-view indices.
3. **S4 assignment**: freeze the global mapper; train local partial assignment,
   geometry validity, dustbin, and rescue/keep policy on real hard proposals.
4. **S5 measurement**: train selected-only pixel likelihood after S4 passes.
5. **S6 geometry**: coverage-safe multi-hypothesis PnP and held-out verification.

Candidate-maplet checkpoints are selected independently:

- `best.pt`: best overall validation policy;
- `best_rescue.pt`: best fixed rescue/keep policy using its own pose/calibration
  key;
- `best_global.pt`: best validation checkpoint under whole-image unique-track
  assignment and a fixed pose budget;
- `last.pt`: final optimization state for diagnostics.

## Evaluation Discipline

- OldHospital queries already used for method and threshold selection are a
  development set, not an untouched production test.
- A development run can pass a development gate but can never set
  `production_promoted=true`.
- Policy sweeps use validation only. A cross-block development audit may inspect
  both reused blocks, but then neither block is a held-out test.
- ANN `nprobe`, proposal L, assignment policy, and pose budget are frozen before
  query-set GT is joined. The fixed-`nprobe=16` full182 inference-only proposal
  artifact is byte-identical to the earlier no-GT artifact.
- The learned assignment gate compares every learned budget/mode to the single
  best validation-frozen baseline across all allowed budgets and modes. Comparing
  only to a weaker same-budget baseline is invalid. A trial with no finite PnP
  pose is invalid and cannot abort or pass a sweep.
- Assignment training consumes the hashed baseline-sweep summary. That summary
  must match the proposal, candidate artifact, projected bank, explicit query
  split, and baseline score; it freezes the global match budget/mode and its pose
  must replay exactly before epoch selection starts.
- Every pose report includes median and P90 translation, median rotation,
  success rate, 25 cm/2 deg, 10 cm/5 deg, 5 cm/5 deg, match count, and inliers.
- Every assignment report includes p1/p2 identity rank, pair AP, wrong-pool
  rejection, switch count, and real-residual calibration.
- Geometry reports distinguish three denominators: detector mappability means a
  visible GT track exists near the query point; proposal-pool availability means
  a correct track is present in the fixed top-L pool; selected recall measures
  the final assignment/PnP subset. These quantities must not share an ambiguous
  `mappable` label.

## Current Gate Status

- S0 descriptor-space/cache safety: implemented, including canonical-resolution
  and GT-free inference artifacts. Legacy native-resolution and old RADIO
  intermediate caches are diagnostic-only.
- S1 query-to-landmark alignment: implementation is present, but its full-epoch
  checkpoint is not promoted. It reaches mapping-observation full-bank
  Recall@1/@20 of `77.98%/98.99%`, yet full182 no-measurement pose is only
  `43.4 cm` median, `124.7 cm` P90, and `0.797 deg`. This is worse than the R1
  development path and demonstrates that train-observation recall is not an
  external-query checkpoint metric. A fully upstream-disjoint validation set is
  still required.
- S2 no-ratio global top-L proposal/oracle audits: implemented and passed for
  entering local assignment development.
- S3 projected-observation bank, aggregation audits, maplets, and support views:
  implemented.
- S4 assignment: the coordinate-correct ALIKE-only two-seed ablation did not pass
  the strict whole-image gate. On the old seq9-only validation, the best frozen
  baseline is global unique-track assignment with K=48 (`30.9 cm` median,
  `68.0 cm` P90, `0.405 deg`). The best seed improved translation
  (`27.2/63.1 cm`) but worsened rotation (`0.458 deg`), and no learned trial beat
  all three frozen-baseline errors. A corrected fresh-RGB RADIO-intermediate
  two-seed ablation improved row-level training and one validation block, but
  still produced zero strict global-policy passes and did not generalize to the
  late block. This rules out stale RADIO coordinates as the sole cause.
- S4 multi-sequence rebuild: complete for the current development protocol. The
  105-query split uses
  seven mapping sequences with an explicit 63/21/21 partition. Its frozen global
  proposal baseline is `23.8 cm` median, `72.5 cm` P90, and `0.485 deg`; 2 px
  correct-track recall rises from `8.45%` at top1 to `39.68%` at top20. The fixed
  proposal-pool oracle reaches `3.19 cm` median and `7.13 cm` P90, while the map
  oracle reaches `0.84 cm` median. On the 21-image validation block, the best
  assignment baseline is now frozen at K=128/score-topk (`27.18 cm` median,
  `65.23 cm` P90, `0.540 deg`); unchanged late-block replay gives `25.27 cm`,
  `167.12 cm`, and `0.565 deg`. The later frozen L97 identity policy is the
  active production-safe baseline: validation is `18.93 cm` median, `56.89 cm`
  P90, and `0.414 deg`; reused-late replay is `22.47 cm`, `118.50 cm`, and
  `0.467 deg`.
- S5 measurement geometry verification: implemented with real RGB only. A
  frozen support selector fuses observations, and a pose-free verifier is
  trained from true GT-pose projection residual labels. Query pose, coarse-PnP
  residual, assignment score, and target fields are forbidden model inputs.
  Measurement-only train-OOF/validation AUROC is `0.803/0.802`; adding
  inference-safe support/track quality reaches `0.839/0.842`. Frozen late-block
  application reaches AUROC `0.853`, AUPRC `0.853`, Brier `0.156`, and ECE
  `0.022`. At the validation-frozen `p>=0.64` threshold, correspondence
  precision is `80.0%` on validation and `82.6%` on late replay.
- S5 pose-free offset action verification: implemented separately from geometry
  correctness. The validation-frozen update threshold is `0.70`; validation and
  late update precision are `82.1%/82.3%`, with selected residual medians
  changing from `3.96 -> 1.90 px` and `3.73 -> 1.77 px`. Coordinate updates are
  not applied globally. The experimental refined mode requires geometry
  probability `>=0.85` and update probability `>=0.70`, and changes xy only for
  the measured selected track.
- S6 multi-hypothesis held-out geometry: implemented. Candidate fitting,
  rank-verification, and optional final-audit partitions are explicit. A prior
  bug that refit on verification rows and accepted on the same rows is fixed;
  final refit is disabled by default. P13 adds measurement-verified hypotheses
  while retaining the original coarse hypotheses. It is the first run to beat
  L97 on all three validation pose errors (`18.78 cm`, `55.00 cm`, `0.387 deg`).
  Reused-late replay improves translation to `18.26 cm` median and `107.66 cm`
  P90, but rotation regresses to `0.527 deg` and 25 cm recall regresses, so it
  is not production-promoted. Double-gated refined hypotheses improve the
  validation hypothesis oracle from `9.61 cm` to `8.87 cm`, but neither a
  retrained global ranker nor a frozen hierarchical margin gate generalizes
  across the reused late block. L97 remains the mainline policy.
- The next S6 step is candidate-specific selective RGB verification: measure a
  small pose-conditioned set of top-L alternatives rather than only the frozen
  hard-selected track. Any added candidate pool must preserve the original P13
  ranking as an invariant fallback; optional candidates may not silently change
  the baseline's cross-hypothesis features or score.

## Forbidden Production Combinations

- raw query descriptors against a projected-observation landmark bank;
- full-map query projection against post-aggregate 1x1 landmark projection;
- stale projected banks whose checkpoint/config/source hashes do not match;
- mixing native-resolution detector xy with canonical-resolution radii,
  residual thresholds, RADIO sampling, or camera intrinsics;
- model inference that loads query GT pose, candidate residuals, or assignment
  labels, even when those tensors are not consumed by the forward pass;
- image-retrieval or reference-visibility submaps in the no-retrieval preset;
- hard ratio rejection before top-L local assignment;
- multiple tracks from one query point passed as independent PnP matches;
- score-sorted input to uniform RANSAC;
- visibility, dustbin, or uncertainty claimed as calibrated when its supervised
  loss is disabled;
- RGB measurement enabled before assignment-only center PnP passes;
- treating a missing RGB measurement as a negative rather than unknown;
- applying a selected-track RGB offset to every top-L candidate that shares the
  query token;
- appending optional hypotheses in a way that changes the frozen fallback's
  cross-hypothesis features or score;
- any reused OldHospital development block labeled as an untouched test.

## Focused Verification

```bash
PYTHONPATH=. pytest -q \
  tests/measurement_v1/test_action_calibration.py \
  tests/measurement_v1/test_pose_free_geometry_verifier.py \
  tests/test_measurement_pose_evidence.py \
  tests/test_pose_hypothesis_verifier.py \
  tests/test_pose_hypothesis_ranking.py \
  tests/test_pose_backend_selection.py
```
