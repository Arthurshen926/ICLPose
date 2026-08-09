# ICLPose

Real-image VFM localization against a feature-bearing 2DGS surface map.

## Active Mainline

The active research line is **Goal-Maplet structured physical-surface
localization**. V6/V8/V8.1 are frozen diagnostics and V3 remains the frozen
production baseline.

```text
mapping RGB + calibrated poses + clean 2DGS --offline only-->
  exact primitive -> child surface tile -> context-parent hierarchy
  + one canonical RADIO-final-derived code per observed primitive
  + typed geometry/continuity/co-visibility/context graph

query RGB
  -> RADIO-final all-token physical-maplet posterior
  -> post-retrieval correlated-support grouping
  -> parent/child surface posterior + typed graph configurations
  -> fixed Top-16 coarse SE(3) modes
  -> exact full-scene canonical-field likelihood
```

The target prior stores only metric 2DGS geometry and bounded anonymous
feature-map statistics. It stores no mapping RGB, mapping image path/ID,
stable-anchor identity, per-view descriptor list, or ALIKE descriptor map.
Runtime uses no SfM
points/tracks, RADIO intermediate features, LoFTR, pairwise query/reference
matching, single-point cosine pose energy, or final point-correspondence PnP.

## Current Status

The [Goal-Maplet report](docs/vfm/goal_maplet_localization_v1.md) defines the
active contracts, artifacts, oracle ladder and promotion decision. Exact clean
geometry and PFIR retrieval pass their current gates. On the trajectory-disjoint
48-query development set, current Top-32 proposals contain a 0.253 m median
oracle candidate and cover 97.92% within 1 m / 10 degrees.  The G17 exact
full-scene feature likelihood is a real tail improvement: it reduces the
catastrophic rate from G16's 6.25% to 2.08% and repeated-facade seq13 from
18.75% to 6.25%.  It is not promoted, because Dev48 strict success is only
39.58% and translation is 0.596/1.495 m median/P90, below the frozen graph/G16
accuracy gates.  The deployable Top-1 therefore remains graph v9 at
0.555/1.680 m, and the continuous RADIO surface-flow refiner remains closed.
Goal-Maplet is an active research line, not a production or paper-ready
accuracy claim.

The earlier independent configuration logistic improves Dev48 median to
0.515 m and catastrophic errors to 6.25%, but is also rejected because P90 is
1.690 m and median selection regret remains 0.263 m. A parent-conditioned
child-local Top-8 likelihood reuses the single canonical primitive feature and
improves the oracle-child pose from 0.326/0.459 m to 0.265/0.406 m
median/P90; its Top-8 mode oracle is 0.151 m at the point-measurement level,
but mode selection does not yet meet the 0.15–0.20 m gate. These results and
the exact artifact contracts are recorded in the Goal-Maplet report.

The latest probability pass fixes candidate-pose self-conditioning by freezing
VFM local modes before geometry. Pairwise local-mode ranking reaches 0.194 m
point median on oracle-child Dev48, although independently selected points
still give 0.287 m pose median. A five-way typed-null factor is trained from
exact round-trips and hard wrong-child/pose negatives. Its predefined
cross-group valid-mass aggregation gives the current best Goal-Maplet research
diagnostic: 0.455 m median / 1.499 m P90, 1.667° / 5.058°, and 6.25%
catastrophic failures. The learned factor ranker does not preserve the P90
gain, median selection regret remains 0.217 m, and there is no untouched test;
the frozen graph proposal therefore remains the production default.

The G14 sparse mode-relation pass now adds a query-only local/long-range/
depth-normal edge graph over fixed Top-16 child options.  Paired relation LLRs
reach 0.711 edge concordance and 0.902 aggregate candidate concordance on held
out seq10. Exact tree inference uses Sum-Product for pose evidence, Max-Sum for
the interpretable configuration, and disjoint edges for verification.  A
zero-LLR safety gate improves Dev48 over the G13 Top-16 baseline from 0.638 m
to 0.582 m median while preserving its 1.711 m P90 and 6.25% catastrophic
rate, but it remains worse in translation than the frozen graph result
(0.555/1.680 m). The unbounded verification diagnostic reaches 0.512/1.600 m
on Dev48 but fails the independent validation tail gate, so neither relation
policy is promoted and no untouched test or continuous refiner is opened.

The G15 probability-semantics pass fixes G14's connected-component support
chaining, conserves all child/mode/geometry/field null mass, and uses exact
fit-tree joint marginals for held-out edges.  The predefined joint score
improves Dev48 to 0.461/1.596 m and 1.565/4.547 degrees with 52.08% strict
success, but catastrophic errors rise to 8.33% and validation P90 is 1.572 m.
Exact-GT non-null posterior remains only 3.65% on Dev48 and Max-Sum is still
all-null.  G15 is therefore a correctness and diagnostic advance, not a
production promotion; untouched test, proposal-family expansion and the
continuous refiner remain closed.

The G16 endpoint pass replaces fixed parent/child/mode quotas with a
mass-adaptive 16-leaf hierarchy and explicit parent/child/mode tails.  It also
collapses complete-link descriptor, posterior and image footprint into one
coherent endpoint, validates child/mode temperatures with proper log score,
and retrains relation LLRs on the exact deployment state distribution.  On
seq12/14 validation, node+fit improves from G15's 0.381/1.509 m and 64.71%
strict success to 0.359/1.509 m and 70.59% without worsening Top-3 or
catastrophic rate.  Dev48 improves P90 from 1.596 m to 1.370 m and catastrophic
rate from 8.33% to 6.25%, but regresses median/success/Top-3 to
0.563 m/45.83%/62.50%.  GT non-null mass also fails to separate repeated-
facade phase errors on Dev48.  G16 is therefore a method-level partial pass,
not a promotion; graph v9 remains production and the next blocker is
cross-trajectory endpoint identity at the query-support/local-readout boundary.

The G17 physical-instance pass uses the three RADIO downstream adaptors only
as offline mapping teachers: SigLIP supervises parent/context identity, DINO
supervises child/local identity, and SAM supervises support-boundary affinity.
Deployment still stores exactly one canonical RADIO-final code per observed
2DGS primitive; two small context/local readout heads are regenerated from
that code and no RGB, teacher embedding, image path, ALIKE descriptor or RADIO
intermediate is retained.  On independent seq12/14, the readout raises child
R@16 from 69.43% to 77.44% and joint parent-R@32/child-R@16 from 68.23% to
76.15%.  A proposed breadth-first 16-leaf allocator is rejected because it
reduces exact primitive and two-endpoint relation coverage.  Retaining the
validated mass-only leaf allocation with the new readout gives the current
G17 validation result: 0.255/1.028 m median/P90, 70.59% strict success,
76.47% Top-3 and 5.88% catastrophic failures.  This improves every frozen G16
ranking metric except that success is tied, but GT non-null mass still trails
phase-near-miss mass (0.0758 vs 0.1132).  G17 is therefore a method-level
advance under Dev stress testing, not a production or untouched-test
promotion.

The G17.1 verification pass removes a remaining implementation mismatch: the
learned 9x9 context readout is now applied symmetrically to the query and the
rendered canonical field with explicit missing-surface masks.  It renders the
complete clean 2DGS scene, uses a fixed full-query denominator, creates no
2D--3D point correspondences and invokes no PnP.  On seq12/14 it improves the
frozen candidate Top-1 from 0.441/4.710 m and 52.94% strict success to
0.366/0.837 m and 76.47%, while catastrophic failures fall from 17.65% to
5.88%.  The cross-trajectory Dev48 result above confirms the tail reduction
but not a success-rate promotion.  Top-32 verification, a second G16 proposal
branch and teacher-consistency reweighting of canonical map observations all
fail their benefit/complexity gates and remain ablations.

## Frozen V6 Status

See [the V6 mainline](docs/vfm/2dgs_maplet_atlas_localization_v6.md) for its
architecture, artifact contracts and strict G0–G4 promotion gates.
P0 identity/spatial probability semantics are corrected. Retrieval regions
and metric charts are now separate map entities with a strict many-to-many
index, overlapping query-region evidence is deduplicated, and the current
research bridge is `MapletFrameAlignment`. The route-repair audit normalizes
Region-to-Chart probability, uses projective chart observations and a robust
correlated chart-factor solver, and connects runtime atlas re-rendering and
held-out verification. Complete 4,096-mode generation plus a lossless sparse
Top-64 atlas screen has removed the earlier meter-scale implementation
failure: on the strict 12-query set the current Top-1 translation is
0.405 m median / 0.730 m P90 and rotation is 0.665 / 2.551 degrees. The full
generated-pool oracle is 0.279 m / 0.940 degrees median, with 4/12 and 7/12
inside 20 cm and 30 cm respectively. A 16 cm q7 component control remains
stable through normal atlas refinement, localizing the residual losses to
finite candidate selection/final ranking and, more fundamentally, the
cross-trajectory RADIO chart-frame distribution. An attempted `seq11`
ranking calibration was rejected because 0/6 complete Top-64 pools contained
a 30 cm / 3 degree mode. It is not production-qualified and no 530-query
accuracy claim is made. Structured frame/flow learning on multiple
trajectories, typed correlation nulls, adaptive charts and full-scene runtime
occlusion remain open.
The V5 point-similarity route is retained as a failed diagnostic: it never
implemented atlas/query correlation and is not extended.
The [V3 anchor/PnP mainline](docs/vfm/2dgs_surface_localization_mainline.md) is
frozen production baseline, not extended with new modules. The
[SfM landmark mainline](docs/vfm/landmark_localization_mainline.md) is retained
only as a historical baseline.

The [V8.1 structured-region diagnostic](docs/vfm/nonredundant_region_surface_localization_v81.md)
is isolated from the active V6 line. It fixes directed-edge duplication,
correlated-support counting, footprint semantics, virtual-pose truncation and
SE(3) mode collapse, but is not promoted: its final Strict12 result remains
1.72 m median translation. Its O1--O4 oracles identify continuous
maplet-interior RADIO alignment, rather than a larger pose lattice or another
stored embedding, as the next required research module.

## Active Code

- `feature_extract/vfm/localization_goal_maplet/`
- `feature_extract/tools/vfm/build_goal_maplet_physical_map.py`
- `feature_extract/tools/vfm/build_goal_maplet_canonical_field_from_contributors.py`
- `feature_extract/tools/vfm/build_goal_maplet_typed_graph.py`
- `feature_extract/tools/vfm/train_goal_maplet_physical_instance_readout.py`
- `feature_extract/tools/vfm/calibrate_goal_maplet_validity.py`
- `feature_extract/tools/vfm/train_goal_maplet_endpoint_hierarchy_calibration.py`
- `feature_extract/tools/vfm/evaluate_goal_maplet_canonical_pfir.py`
- `feature_extract/tools/vfm/evaluate_goal_maplet_oracle_ladder.py`
- `feature_extract/tools/vfm/evaluate_goal_maplet_pose_modes.py`
- `feature_extract/tools/vfm/verify_goal_maplet_pose_modes_with_surface_field.py`
- `feature_extract/tools/vfm/evaluate_goal_maplet_surface_basin.py`
- `feature_extract/tools/vfm/build_goal_maplet_feature_contract.py`
- `feature_extract/tools/vfm/build_goal_maplet_child_eligibility.py`
- `feature_extract/tools/vfm/train_goal_maplet_configuration_ranker.py`
- `feature_extract/tools/vfm/apply_goal_maplet_configuration_ranker.py`
- `feature_extract/tools/vfm/evaluate_goal_maplet_child_local_likelihood.py`
- `feature_extract/tools/vfm/train_goal_maplet_child_local_mode_ranker.py`
- `feature_extract/tools/vfm/train_goal_maplet_child_local_pairwise_ranker.py`
- `feature_extract/tools/vfm/build_goal_maplet_child_local_factor_samples.py`
- `feature_extract/tools/vfm/train_goal_maplet_child_local_factor_calibrator.py`
- `feature_extract/tools/vfm/train_goal_maplet_pose_likelihood_ratio.py`
- `feature_extract/tools/vfm/build_goal_maplet_mode_relation_samples.py`
- `feature_extract/tools/vfm/train_goal_maplet_mode_relation_likelihood_ratio.py`
- `feature_extract/tools/vfm/augment_goal_maplet_configuration_evidence.py`
- `feature_extract/tools/vfm/evaluate_goal_maplet_mode_relation.py`
- `feature_extract/tools/vfm/train_goal_maplet_configuration_pairwise_ranker.py`
- `feature_extract/tools/vfm/evaluate_goal_maplet_latent_configuration.py`
- `feature_extract/tools/vfm/train_goal_maplet_latent_safety_selector.py`
- `feature_extract/tools/vfm/apply_goal_maplet_latent_safety_selector.py`

## Frozen V6 Code

- `feature_extract/tools/vfm/build_v6_canonical_maplet_atlas.py`
- `feature_extract/tools/vfm/build_surface_retrieval_maplet_bank.py`
- `feature_extract/tools/vfm/calibrate_v6_retrieval_probabilities.py`
- `feature_extract/tools/vfm/build_v6_region_chart_index.py`
- `feature_extract/tools/vfm/audit_v6_primitive_contributors.py`
- `feature_extract/tools/vfm/build_v6_contributor_cache.py`
- `feature_extract/tools/vfm/train_v6_metric_encoder.py`
- `feature_extract/tools/vfm/bake_v6_metric_atlas.py`
- `feature_extract/tools/vfm/evaluate_v6_oracle_basin.py`
- `feature_extract/tools/vfm/evaluate_v6_retrieval_pose_basin.py`
- `feature_extract/tools/vfm/evaluate_v6_maplet_frame_alignment.py`
- `feature_extract/tools/vfm/evaluate_v6_radio_frame_source.py`
- `feature_extract/tools/vfm/evaluate_v6_projection_model_gap.py`
- `feature_extract/tools/vfm/evaluate_v6_radio_atlas_basin.py`
- `feature_extract/tools/vfm/merge_v6_radio_frame_reports.py`
- `feature_extract/tools/vfm/replay_v6_stage_c.py`
- `feature_extract/tools/vfm/diagnose_v6_stage_c_pool_coverage.py`
- `feature_extract/tools/vfm/merge_v6_stage_c_reports.py`
- `feature_extract/tools/vfm/fit_v6_stage_c_score_calibration.py`
- `feature_extract/tools/vfm/train_v6_structured_frame_adapter.py`
- `feature_extract/tools/vfm/train_v6_global_frame_encoder.py`
- `feature_extract/tools/vfm/bake_v6_retrieval_atlas.py`
- `feature_extract/vfm/localization_v6/`

## Historical And Reference Lines

Synthetic 2DGS RADIO-MATCHA, SfM tracks, rendered-map verification, image
retrieval/submaps, LoFTR, and legacy patch-offset measurement remain reference
or ablation paths. They must not be mixed into production 2DGS surface results.

## Evaluation Discipline

Every result must identify its 2DGS source, surface-field and mapper artifacts,
camera calibration, query manifest, maplet budget, pose policy, and split role.
A full 530-query localization run is forbidden until G0 geometry, G1
correlation, G2 local basin and G3 coarse-proposal gates all pass. Validation
trajectories must be disjoint from encoder training trajectories.

## Verification

The current repository expects the project root on `PYTHONPATH` when running the
focused tests from this checkout:

```bash
PYTHONPATH=. pytest -q \
  tests/test_goal_maplet_*.py \
  tests/test_v6_maplet_atlas_correlation.py \
  tests/test_v6_maplet_frame_alignment.py

python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```
