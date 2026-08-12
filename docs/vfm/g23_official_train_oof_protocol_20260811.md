# G23 official-train OOF protocol and recommendation audit

## Decision

G23 uses the Cambridge St Mary's Church split as follows:

- development and hyperparameter selection: all 1,487 official training
  images, each evaluated once out of fold;
- no permanent validation subset;
- final fit: rebuild all learned components and the map from all 1,487 official
  training images after configuration freeze;
- final benchmark: evaluate once on all 530 official test images without using
  test feedback to change the frozen G23 configuration.

The earlier 17-image seq12/seq14 set is a subset of official train, not test.
It remains only a historical failure/stress slice.  Its previous results are
diagnostic and cannot select a G23 setting.

The executable validator is
`feature_extract/tools/vfm/build_goal_maplet_official_oof_protocol.py`; the
generated immutable input partition is
`configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json`.  The validator
checks that pose files, RADIO token manifests, mapping cameras and mapping
depths agree exactly.  Current audited counts are 1,487/1,487/1,487/1,487 for
train and 530/530 for official test, with no train/test image-ID overlap.

## Five route-grouped folds

| Fold | Held query trajectories | Query images | Mapping images |
|---|---|---:|---:|
| F0 | seq2 | 352 | 1,135 |
| F1 | seq4 | 329 | 1,158 |
| F2 | seq1, seq8, seq14 | 281 | 1,206 |
| F3 | seq7, seq12 | 268 | 1,219 |
| F4 | seq6, seq9, seq10, seq11 | 257 | 1,230 |

The folds are route groups, not random frames.  Across the five folds every
official training image contributes one post-selection OOF outcome.  A
calibrator sees only these OOF outcomes; it is never fitted on a winner that
was produced by a model containing the same query observation.

## Geometry tiers

Two experiment tiers are deliberately separated:

1. `fixed_full_train_2dgs_non_publishable_screening` uses the existing full
   train 2DGS to reject clearly bad parameter settings cheaply.  Because a held
   train query helped optimize this geometry, these numbers are not evidence
   for the paper.
2. `fold_rebuilt_2dgs_excluding_held_query_trajectories` rebuilds geometry,
   clean primitives, hierarchy, canonical field, view topology and learned
   components per fold.  Only this tier may confirm a selected configuration.

The historical full-train PLY hashes are screening lineage only.  The final
official-test map must be the `final_alltrain` output of the same checked
strict builder, and its PLY/clean-surface hashes are recorded only after that
run completes.  No old `/root/StMaryChurch2dgs*.ply` artifact is accepted by
the final-fit entrypoint.

## Tuning order and objective

Correctness semantics are frozen and are not tunable: all geometry occludes,
H0 is normalized separately from views, invalid parent IDs cannot match
padding, camera intrinsics follow the declared model, and post-selection
success is not called a pose posterior.

The first bounded sweep changes only compute allocation and basin survival:

- exact pool: 64, 128, 256;
- refinement budget: 1, 2, 4, 8, 16, 32;
- protected states per anchor: 2, 4, 8;
- protected anchors: 8, 16, 32;
- pose NMS: (0.2 m, 3 deg), (0.5 m, 5 deg), (1 m, 10 deg).

Only configurations surviving that sweep open the representation parameters:
mapping anchors 8/16/32, hypotheses 2/4/8, support pairs 32/64/128, sparse
primitives 4/8/16, splat radius 0/1/2, and geometry confidence
0.02/0.05/0.10.  Refinement scale/iteration settings and minimum accepted
exact-energy improvement (1e-5/1e-4/5e-4) are tuned last.

Candidate-budget selection is lexicographic: catastrophic Top-1 regressions
first, then strict and loose basin recall at K=32, strict 0.5 m/5 deg and loose
1 m/10 deg Top-1 yield, translation/rotation P90, and finally compute.  The
later refinement-policy selection is separately catastrophe-first and then
success-yield-first.  The output includes basin recall at K, opportunity capture,
risk--coverage, success--compute and confidence intervals grouped by route.
The selected report stores the complete candidate inference argument vector,
not only a human-readable tag.  Strict geometry confirmation renders that
vector into its runner environment and refuses to start without the selection
artifact; it cannot silently fall back to the Exact-128 default.
Long candidate and refinement jobs snapshot the repository commit, tracked
status and tracked-diff hash at process start.  That start-state hash is part
of every atomic partial-checkpoint contract, so resuming after a source edit is
rejected instead of mixing rows produced by two implementations.  Freeze does
not accept the older end-of-process source snapshot semantics.  Because the
research worktree may contain newly added files before a paper commit, the
snapshot also hashes every untracked source/config/script file rather than
recording only tracked diffs.
Shard merging preserves the standard manifest schema, verifies identical
process-start source identities and numeric contracts, and records both source
manifests.  Freeze additionally requires one source identity across all five
candidate folds and every refinement shard; a code edit mid-OOF therefore
invalidates the run instead of being hidden by aggregation.
The frozen numeric contract uses FP64 pose matrices, FP32 feature/render
arithmetic (mixed precision is disabled), a full-clean-geometry depth prepass,
and persistent primitive IDs for all 1e-5-depth z ties.  Candidate/refinement
freeze rejects any other dtype, tie or image-ID seed policy.  Unit tests also
check primitive-input permutation and pose-batch-size invariance; the official
test is still repeated three times to measure GPU decision and score drift.
The common candidate/refinement source identity is itself frozen.  Final-fit
verification, final candidate verification and frozen-policy refinement all
reject a different commit/diff/untracked-source identity, so the test runner
cannot silently execute edited code with merely the same numeric arguments.

Route disjointness is complemented by a fixed acquisition-extrapolation audit.
Against each fold's mapping trajectories, the 1,487 held OOF queries contain
456 poses within 0.5 m of a mapping camera, 708 at 0.5--2 m, 270 at 2--5 m and
53 beyond 5 m.  A predeclared joint proxy (nearest mapping camera farther than
2 m, view-direction gap above 60 degrees, or height gap above 0.75 m) marks
331/1,487 queries as outside the mapping-view manifold.  There are 10 queries
at 60--90 degrees but none beyond 90 degrees, so this split can support a
route/interpolation claim but not a true reverse-view claim.  Strict OOF
success and catastrophe are reported with Wilson intervals in every stratum;
freeze requires that analysis.  A reverse-view claim therefore remains
blocked until another scene or deliberately acquired route supplies it.

## Attachment recommendation audit

| Recommendation | Current state | G23 action |
|---|---|---|
| mapping-view H0/null | implemented and unit-tested | frozen correctness contract |
| full-geometry sparse occlusion | implemented and synthetic-tested | frozen correctness contract |
| local-frame row/column | implemented and rotation-tested | frozen correctness contract |
| `-1` parent padding | implemented and tested | frozen correctness contract |
| camera-model intrinsics | implemented and tested | frozen correctness contract |
| repository replay/run manifest | versioned P0 replay, five-fold G23 build/evaluation entrypoints, fold-map leakage audit, checked strict-MAtCha runner, and the exact RTX3090/CUDA11.6 source patch are implemented | add final-test manifest only after OOF freeze |
| joint post-selection calibration | typed high-capacity experiment rejected on 17 train images; one low-capacity L2 logistic over selection, coverage, view-null and geometry-quality evidence implemented | fit only from 1,487 OOF outcomes |
| typed missingness/null | typed event schema v2 implemented | retain types; calibrate their intervention value OOF |
| view-conditioned field | rank-4 field tested; no gain on historical slice | rerun only after OOF baseline, do not promote yet |
| continuous visibility chart | bounded chart tested; no gain on historical slice | retain as ablation, not current mainline |
| basin/equivalence candidate units | proposal exact-pool basin accounting implemented | extend same units to adaptive refinement |
| shared verification/refinement energy | final exact verifier accepts/rejects refinements; production six-axis convergence evaluator and five-fold aggregator are implemented | the predeclared strict-fold route-balanced basin grid is a mandatory freeze gate |
| refinement K curve | K=1..32 prefix-replay evaluator and adaptive basin-cover policy implemented | run from OOF candidate pools |
| frame00139 taxonomy | generator absence established for common feature fold; automatic G/S/V/R OOF attribution and freeze gate implemented | populate it from the complete strict OOF run |
| geometry/query independence | fixed-geometry OOF screen remains non-publishable | strict fold/all-train map, contributor and feature rebuilds are now executable and mandatory before freeze |
| strict MAtCha dense-data completeness | fixed: all fold-mapping images are retained for dense supervision; 64 route-balanced views are chart initializers only | immutable per-fold COLMAP hashes and zero held-route overlap audited |
| multi-map/route extrapolation | pending | mandatory P3 experiment after St Mary's freeze |

The rejected or negative experiments remain results: typed likelihood on the
17-image historical slice degraded strict/loose success from 7/11 to 5/7 and
accepted the catastrophic frame, while the view-conditioned field and the
visibility-chart proposal did not recover the missing basin.  They are not
silently folded into the mainline.

After fixing the refinement baseline identity (the baseline is union candidate
zero, not merely the first mode from source report zero), the frozen historical
stress replay changed from 13/17 strict, 14/17 loose and 2/17 catastrophic to
14/17 strict, 16/17 loose and 1/17 catastrophic.  The gate selected refinement
for 7/17 queries.  This is a regression result only: it neither selects G23
hyperparameters nor substitutes for the 1,487-image OOF evaluation.  The
lineage-bound result is
`goal_maplet/g23_refinement_baseline_fix_replay/selective_gate_evaluation.json`.

## Immediate execution state

The previous contributor cache contains only 128 set-cover-selected mapping
views (including two seq8 views); the official source data themselves are
complete.  G23 therefore first builds an exact clean-2DGS contributor cache
for all 1,487 official training images, split deterministically over both GPUs.
This cache is construction data for OOF and final fitting; it is not a
validation set and stores no RGB.

The all-train contributor build is complete: two shards contain 744 and 743
queries, and the merged audit verifies all 1,487 image IDs, contributor
indices, clean-primitive IDs, cameras, depth, RADIO token geometry, weight
normalization, and finite values.  Exact 128x72 geometry supervision was then
derived from the contributor z-buffer on both GPUs.  Its 128-image overlap
against the older approximate renderer has median relative depth error 0.0079
and median normal cosine 0.9878; unlike the older label path it preserves the
same front-surface semantics used by the runtime contributor renderer.

Surface-mapper checkpoints use `fixed_epoch_no_selection`: every available
maplet observation from the mapping trajectories is fitted for 120 fixed
epochs, with no validation frames and no checkpoint selection.  Six official
train frames lack surface-maplet-bank observations, so the mapper has 1,481
unique supervised images; those frames are still present in the 1,487-image
contributor/canonical-field construction and in OOF evaluation.  This is
missing mapper supervision, not a hidden validation split.

The historical physical-instance readout cache covered only 128 set-cover
views.  G23 does not treat that subset as the training set: it extracts the
three offline teacher readouts for all 1,487 official-train frames, audits the
exact image-ID set, and then fits each fold only from its mapping trajectories
with `fixed_step_no_selection`.  Teacher features are discarded after
distillation; neither mapping RGB nor downstream teacher embeddings enter the
deployed map.  Each readout records the count and hash of teacher-supervised
image IDs so missing supervision cannot silently masquerade as a validation
split.

The 530 official-test query input packages are built and hash-audited only
after the train-side G23 configuration has been frozen and before final
evaluation.  No test contributor package or localization metric is produced
by the pre-freeze strict confirmation chain.  In the actual-parent/actual-child runtime the loader does not
deserialize contributor identity labels at all; these arrays are available
only to explicitly requested oracle ablations.  GT pose is used solely for
offline error/stage diagnostics and never for candidate selection.

Post-selection reports now carry the complete query-local evidence needed by
the success calibrator: winner/runner-up and baseline gaps, refinement gain,
cross-discretization stability, parent out-of-map/tail mass, exact rendered and
feature coverage, mapping-view typed-null/entropy, query geometry confidence,
and distinct-basin margin/count.  These enter one regularized logistic model;
they are not separately calibrated and added as pseudo-LLRs.  The deployed
interface reports threshold-specific success probability, never a continuous
pose posterior.  It also exposes a typed diagnostic vector for out-of-map,
canonical-field unsupported, low-quality query, outside-view-atlas and
repeated/symmetric ambiguity.  Occlusion remains physically resolved by the
all-geometry z-buffer rather than being fabricated as another independent
probability.

The joint calibrator's L2 strength is not selected on the route whose
probability is being evaluated.  Each outer held-route prediction chooses
`C` from 0.01/0.03/0.1/0.3/1.0 by inner route-cross-fit negative
log-likelihood on the remaining routes.  The final deployment head performs
the analogous route-cross-fit selection using all train OOF outcomes and then
fits once on those outcomes.  This is hyperparameter selection over full
official train without introducing a permanent validation subset.
Risk-threshold semantics follow the same separation: unbiased train-side
risk--coverage uses nested-policy OOF probabilities, whereas each deployment
threshold is fitted from route-cross-fit probabilities of the final
all-train-selected policy.  The latter is labeled fit-only because policy
selection saw all train OOF outcomes; it is never reported as unbiased OOF
performance.

Coverage is attached to each refined state under the same exact renderer.  If
an adaptive K replay changes the winner, rendered/feature coverage and the
derived unsupported-surface null change with it.  Pool-average coverage remains
only a proposal diagnostic and is not used as if it described the selected
pose.

A single score-Top-32 OOF refinement pass supports ground-truth-free prefix
replay for K=1,2,4,8,16,32 and basin-aware adaptive policies.  Policy
allocation uses only initial score, anchor and SE(3) basin separation; OOF pose
errors enter only the declared catastrophe-first configuration selection.
The source reports also serialize the complete optimizer schedule (translation
and rotation steps, iterations per scale and minimum exact-score gain), both
renderer radii, basin radii and cross-discretization rule.  Freeze requires
these fields to be identical across all folds, and final test renders its CLI
from that frozen record rather than relying on Python defaults.

Policy selection itself is nested by route.  For every held fold, the
refinement budget/margin policy is selected from the other four folds and then
replayed on the held routes; only these nested rows are eligible for train-side
risk--coverage and calibration evaluation.  A second policy selected from all
1,487 OOF outcomes is labeled fit-only: its replayed rows train the final
success calibrator and freeze the deployment configuration, but its metrics
are not reported as unbiased OOF estimates.  This prevents a route from
influencing both adaptive-policy selection and its own reported confidence.

The OOF chain emits a mandatory oracle-only G/S/V/R failure taxonomy.  It uses
ground truth only after proposal, ranking, refinement and winner selection are
fixed.  `G` means that no correct basin was generated, `S` that a raw basin was
removed before the exact pool, `V` that a basin entered the exact pool but was
lost by exact ranking/NMS, and `R` that a retained basin was not converted into
the final successful output; refinement/gate regressions are separated from
ordinary final-selection failures.  Configuration freeze refuses an incomplete
taxonomy, so claims about an atlas or verifier bottleneck cannot be based on a
hand-picked failure.

The production coordinate-search refiner also has a dedicated six-axis basin
entrypoint and strict-OOF aggregator.  It evaluates both signs of camera-frame `tx/ty/tz` at
0.1/0.25/0.5/1/2 m and `rx/ry/rz` at 2/5/10/20 degrees, records directional
asymmetry, strict/loose Wilson intervals and every exact-score accepted update,
and asserts that the common surface score is monotonic.  Query selection is a
deterministic, route-balanced subset fixed before outcomes are observed.  The
two-GPU launcher is `scripts/run_goal_maplet_g23_six_axis_basin.sh`; this is an
oracle convergence diagnostic and is never used as a localization proposal or
configuration selector.  Every official-train route contributes exactly four
deterministically chosen held-fold queries, every declared signed perturbation
must be present, and configuration freeze rejects a missing, incomplete or
non-monotonic aggregate.

Before final all-train fitting, `audit_goal_maplet_official_oof_maps.py`
checks every mapper, geometry head, canonical field, readout, typed graph,
mapping-view graph and validity artifact against the route split.  It rejects
held/test-route overlap, any validation-selected checkpoint, incomplete
teacher supervision, and cross-artifact field lineage drift.  Only after that
audit, complete candidate OOF evaluation, nested refinement selection and
route-crossfit success calibration does
`freeze_goal_maplet_g23_configuration.py` create
`frozen_configuration.json`.  The all-train fit now verifies every frozen
lineage hash, the protocol hash, calibrator hash and canonical final-fit recipe
instead of accepting the mere presence of a file named “frozen”.

Selective output is also frozen train-side rather than assigned an arbitrary
0.5 probability cutoff.  For strict and loose success heads, the calibrator
selects the maximum-coverage threshold whose 95% Wilson lower success bound
reaches the predeclared 0.80/0.90/0.95 targets.  Its reported train-side
performance uses another route-nested threshold selection; the threshold fit
on all OOF outcomes is labeled deployment-only and is never presented as an
unbiased OOF estimate.  Final inference emits both calibrated probabilities
and an accept/abstain decision at each frozen operating point.

The strict geometry rebuild is no longer an unspecified or PLY-only future step.  The
fold-COLMAP builder materializes a posed, point-free MAtCha input from only the
mapping trajectories, performs deterministic route-proportional selection
(the builder permits up to 512 initial charts; this run freezes 64), keeps every mapping image in
the COLMAP dataset for dense RGB supervision, scales the declared intrinsics
to the native image resolution, and hash-records both exact image lists.  A smoke
round-trip verifies its COLMAP binary model contains the expected cameras and
poses and zero SfM points.  The checked downstream chain now continues through
raw-RADIO bootstrap identities, fixed-epoch fold mapper fitting, mapper-space
surface reconstruction, clean physical hierarchy construction, exact
train/query contributor caches, geometry-head labels, canonical field,
physical readout, typed graph, mapping-view graph and validity calibration.
`audit_goal_maplet_strict_map_fold.py` binds every stage to the strict PLY and
route-clean inputs.  The expensive executions are pending; until all five
audits complete, current candidate numbers remain explicitly non-publishable
screening rather than geometry-independent validation.

The materialized strict inputs contain 1,135/1,158/1,206/1,219/1,230 dense
mapping images for F0--F4 and 1,487 for `final_alltrain`, exactly 64 initial
charts per map, zero held-route images and zero SfM points.  Their filtered
RADIO manifests and Cambridge pose files have the same per-fold image hashes.
Flattened MAtCha names (`seq__frame`) are explicitly aliased back to Cambridge
IDs (`seq/frame`) and strict map building fails if any view would fall back to
a default camera.  This fixes an earlier builder mistake where the
chart cap also truncated dense supervision to the chart subset; that behavior
would not have satisfied the agreed full-training protocol.

The pinned MAtCha commit alone was insufficient because cuRoPE must be compiled
for the two RTX 3090 cards under CUDA 11.6.  The sole tracked source change is
stored as `configs/vfm/matcha_b119fd96_rtx3090_cuda116.patch`; the setup helper
applies it only to commit `b119fd96e484fc81eb40623c1ea92ad3dbd3c21e`, and the
strict runner rejects any other tracked diff.  Each map audit binds the applied
diff hash as well as the upstream commit and checkpoint hashes.  Untracked
smoke configurations are explicitly not runtime inputs.

Configuration freeze now requires
`all_folds_strict_geometry_confirmation=true`; a leakage-clean fixed-geometry
screen is still insufficient.  The immutable final recipe includes the pinned
MAtCha commit, 64-chart selection, all-image dense supervision, 30k Gaussian
iterations, disabled virtual primitives, surface thresholds and clean
occluder contributor policy.  `run_goal_maplet_g23_final_alltrain_fit.sh` is a
verification boundary: it refuses historical geometry and accepts only the
audited 1,487-frame strict map plus the packaged-but-not-evaluated 530 test
inputs.

The post-freeze test entrypoint is
`scripts/run_goal_maplet_g23_final_test_frozen.sh`.  It runs the exact frozen
candidate and adaptive-refinement configuration three times, with each repeat
sharded deterministically over both GPUs.  Every candidate report is compared
structurally with the frozen train-OOF configuration before refinement.  The
final evaluator reports strict/loose success, catastrophic risk with Wilson
intervals, selective risk--coverage, winner/acceptance consistency, maximum
pose deviation and maximum score/probability range across the three runs.
Test outcomes are written only after the freeze and cannot update it.

The official MAtCha repository is pinned to commit
`b119fd96e484fc81eb40623c1ea92ad3dbd3c21e`.  A direct three-image smoke run
now completes MASt3R-SfM, chart alignment and 2D Gaussian training, producing
three dense pointmaps, `charts_data.npz`, and a nonempty 2DGS PLY.  The
repository runner
`feature_extract/tools/vfm/run_goal_maplet_strict_matcha_fold.py` deliberately
bypasses the upstream `train.py`/`run_sfm.py` wrappers because they discard
child exit codes.  It validates the point-free fold input, exact mapping/chart
counts and source commit; invokes each child with checked return status; and
hash-records every required output.  Its setup also restricts cuRoPE to the
deployed RTX 3090 `sm_86`, because the upstream setup asks CUDA 11.6 to compile
the unsupported `compute_90` target.

Validity calibration previously spent most of its time materializing an
`N_token x N_maplet` distribution and then discarded it.  The replacement
uses integral images of total contributor mass and mass owned by any maplet.
It is mathematically the exact marginal required by the validity target:
single-image comparison has maximum null-probability deviation
`5.96e-8` (mean `5.83e-10`) and measured 27.45 s versus 0.01896 s, or
1,447.7x for the target computation.  All five fold artifacts identify this
contract as `exact_owned_contributor_mass_integral_image_v1`.
