# G22 P1 outer-feature-cross-fit likelihood audit

Date: 2026-08-11

## Decision

The typed surface candidate likelihood is **not promoted**. Under reciprocal
trajectory-disjoint training it reduces both strict and loose localization
yield, learns incompatible null behavior in the two directions, and still
accepts the catastrophic `seq12/frame00139.png` result. The experiment is a
useful falsification of the current P1 design, not a paper accuracy result.

This audit establishes feature-pipeline and likelihood-model separation only.
The fixed physical 2DGS geometry was not rebuilt per outer fold, so complete
map/query independence remains unproven and is explicitly marked false in the
result artifact.

## Protocol

A common map feature fold excludes seq3, seq5, seq9, seq12, seq13 and seq14
from the deployed canonical field. It contains 102 mapping images from seq1,
seq2, seq4, seq6, seq7, seq8, seq10 and seq11:

- canonical field SHA-256:
  `2c4b13e91805970a4ec331d4ac4745735d44a5928517f2b1e95cca56da0d79ff`;
- 291,178 feature-bearing primitives, or 57.14% of the physical primitives;
- 102 anonymous mapping-view nodes and 7,604 view-parent relations;
- mapper checkpoint selected on seq9 at epoch 90, with Recall@1/5/10 of
  63.99%/92.10%/96.71%;
- the physical readout is trained on seq1/2/4/6/7/8/11, selected on seq10 and
  evaluated on seq9; its validation joint parent32-child16-primitive8 recall
  is 67.60%, versus 62.72% for the frozen baseline.

The exact candidate budget and scoring protocol are frozen before the
likelihood experiment: 16 structural mapping-view anchors, four protected
states per anchor, a 128-state exact pool, all-geometry visibility and the
single canonical RADIO-derived field. The query lists are the already-used
11 seq12 and six seq14 stress frames.

Two likelihood models use identical fixed hyperparameters and the final
predeclared epoch 20. Checkpoint selection never consults a target or
validation pose label:

- seq12 is evaluated with the model trained only on seq14;
- seq14 is evaluated with the model trained only on seq12.

Both the model-training trajectory and the target trajectory are absent from
the canonical feature map. Offline DINO/SAM/SigLIP caches only weight typed
training events; they are not runtime inputs or stored map embeddings.

## Probability and event-semantics corrections

The event vocabulary is versioned as v2:

`surface_match`, `wrong_phase`, `grazing_surface`, `field_missing`,
`outside_render_support`, and `unresolved`.

Two earlier labels were not observable from a single rendered hypothesis and
were removed:

1. a low-incidence but z-buffer-visible surface is grazing, not occluded;
2. absence of render support cannot be called query-unmapped/dynamic without
   independent query evidence.

True occlusion is resolved by the all-geometry z-buffer. Appearance similarity
does not assign physical phase; match/wrong-phase targets come only from the
frozen candidate pose error. Old v1 models fail closed on the new schema.

The model is correctly described as a same-query listwise candidate reranker
with a typed null. Its reported confidence is posterior concentration, **not**
a calibrated `P(success_tau)`. The cross-fit artifact records
`calibrated_success_probability: false` and forbids paper/deployment
promotion.

## Candidate recall before learned reranking

The threshold oracle is success-consistent: it first selects a 0.5 m/5 degree
candidate when one exists, then a 1 m/10 degree candidate, and only otherwise
falls back to minimum normalized pose error. This fixes the previous diagnostic
that could choose a 0.51 m pose over an available strict candidate and thereby
under-report basin recall.

| exact rank budget | seq12 strict / loose | seq14 strict / loose | combined strict / loose |
|---:|---:|---:|---:|
| 1 | 6/11 / 7/11 | 1/6 / 4/6 | 7/17 / 11/17 |
| 2 | 7/11 / 7/11 | 2/6 / 6/6 | 9/17 / 13/17 |
| 4 | 7/11 / 8/11 | 3/6 / 6/6 | 10/17 / 14/17 |
| 8 | 7/11 / 8/11 | 4/6 / 6/6 | 11/17 / 14/17 |
| 16 | 7/11 / 8/11 | 4/6 / 6/6 | 11/17 / 14/17 |
| 32 | 7/11 / 8/11 | 5/6 / 6/6 | 12/17 / 14/17 |

The strict curve is not saturated at Top-2. A fixed Top-2 continuous-refinement
budget is therefore unsupported by the present evidence. This diagnostic does
not tune a replacement cutoff on the same development frames; the next
evaluation must use a frozen basin-aware adaptive policy on new folds.

## Reciprocal cross-fit result

| selector | accepted | strict 0.5 m / 5 deg | loose 1 m / 10 deg | catastrophic | catastrophic or abstain |
|---|---:|---:|---:|---:|---:|
| frozen exact Top-1 | 17/17 | 7/17 | 11/17 | 1/17 | 1/17 |
| fixed-grid cosine | 17/17 | 4/17 | 10/17 | 1/17 | 1/17 |
| typed likelihood + null | 11/17 | **5/17** | **7/17** | 1/17 | **7/17** |
| exact-pool threshold oracle | 17/17 | 12/17 | 14/17 | 1/17 | 1/17 |

Relative to frozen exact Top-1, typed likelihood changes strict yield by
`-2/17`, loose yield by `-4/17`, and catastrophic-or-abstain by `+6/17`.
Among accepted outputs, strict precision is only 5/11 and loose precision is
7/11. At approximately 50% confidence-ordered coverage it accepts 9/17, has
strict precision 4/9 and still includes the catastrophic pose.

The null behavior collapses by training direction:

- the model trained on seq14, whose six training queries all contain a loose
  candidate, accepts all 11 seq12 targets, including frame00139;
- the model trained on seq12 abstains on all six seq14 targets, including the
  four correct loose Top-1 poses.

This is not useful selective risk control. It demonstrates that six to eleven
query-level listwise outcomes cannot identify a transferable candidate/null
model, regardless of the much larger number of token labels. The token count
must not be mistaken for the number of independent localization outcomes.

## Frame 00139 and runtime parity

The common-fold generator produces no one-metre state for frame00139. Exact
Top-1 is 7.886 m/7.170 degrees and the best threshold-oracle candidate remains
about 6.56 m away. The seq14-trained null model nevertheless accepts candidate
zero with candidate probability 0.04577 versus null probability 0.0163.

The real runtime verifier re-renders all 62 valid candidates and returns the
same selected candidate, probability and catastrophic pose as the cached
cross-fit evaluator. The negative result is therefore not caused by a
sample-only evaluation/runtime mismatch.

This also refines the failure taxonomy. In the earlier seq12-only feature fold,
a 1.059 m near state existed but failed exact-pool survival and verification.
When both target acquisitions are excluded from the common field, that state
disappears and the same image becomes a generator-absence failure. Failure
attribution is protocol-dependent and must be reported per complete map fold.

## Consequences for the mainline

The following paths remain closed:

- the v2 typed likelihood and its null head;
- treating candidate posterior concentration as success probability;
- fitting a post-hoc success calibrator on these same 17 out-of-fold
  predictions and evaluating it in-sample;
- selecting an adaptive refinement cutoff on this development set;
- any paper claim of complete map/query independence from the common fold.

The next defensible mainline requires new data/protocol, not another score
weight:

1. permanently freeze the 17 frames as a development stress set;
2. create at least three route/map-disjoint outer folds and rebuild the 2DGS
   geometry, hierarchy, field, view graph and every calibrator per fold;
3. generate out-of-fold predictions for the complete
   proposal--exact-ranking--refinement--winner process;
4. fit a very low-capacity `P(success_tau | post-selection z)` model only on
   independent outcomes, then evaluate risk--coverage once on untouched data;
5. evaluate shared-energy, basin-aware adaptive refinement through a
   success--compute curve, because the current Top-2 budget is not saturated.

The existing primitive refiner already performs derivative-free SE(3) updates
under a common fixed-grid, all-geometry primitive-VFM score and accepts only
strict score improvements. The remaining gap is not another KNN alignment
implementation; it is an untouched convergence-basin/compute study and a
policy that selects which distinct basins deserve refinement.

## Reproducible artifacts

Single-entry replay:

`scripts/run_goal_maplet_g22_crossfit_likelihood.sh`

Result root:

`output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/map_crossfit_g22_1/hold_seq12_seq14/`

Key files:

- `seq12_exact128.json`, `seq14_exact128.json`;
- `seq12_surface_samples.npz`, `seq14_surface_samples.npz`;
- `surface_likelihood_train_seq12.{pt,json}`;
- `surface_likelihood_train_seq14.{pt,json}`;
- `surface_likelihood_outer_crossfit.json`;
- `seq12_runtime_smoke_frame00139.json`.

The replay uses GPU 0 and GPU 1 concurrently for the two target trajectories.
Candidate rendering remains partially CPU/preparation bound, so instantaneous
GPU utilization is bursty even though both devices execute independent work.
