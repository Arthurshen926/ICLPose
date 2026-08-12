# G21 P0 Method and Correctness Contract

## Paper-level method definition

G21 is a selective proposal--verification system for global localization with
contextual visual-foundation-model features. It is not claimed to be a
normalized continuous pose posterior. The deployed computation is

\[
q_\phi(\mathcal B,\mathcal C\mid I_q,\mathcal M)
\;\longrightarrow\;
E_\theta(T;I_q,\mathcal M)
\;\longrightarrow\;
p_\psi(\mathrm{success}_\tau\mid z_{\mathrm{post-select}}),
\]

where the first term proposes distinct SE(3) basins and structured physical
configurations, the second is a full-map physical-surface verification energy,
and the last term controls whether the selected output is accepted or the
system abstains/falls back. `posterior` is reserved for probability-conserving
discrete variables. Discriminative candidate scores are called energies or
ranking scores; independently calibrated log-odds are not added and called an
LLR.

This supports the intended paper claim:

> A hard-correspondence-free, basin-retrieval and full-map surface-verification
> framework with selective risk control, designed for contextual VFM features.

The current primitive refiner uses latent soft assignments, so it is described
as hard-correspondence-free and PnP-free, not correspondence-free.

## Required stage semantics

### 1. Mapping-view H0 and H1

Mapping-view retrieval factorizes

\[
p(H_0\mid I_q), \qquad p(v\mid I_q,H_1).
\]

`p(H0)` is computed from unresolved query-support mass before view
normalization. It therefore cannot change when `maximum_views` or the graph
node count changes. Returned view probabilities are
`p(H1) * p(view | H1)`. With no resolved in-map support, H0 is one and no
mapping-view pose mode is emitted.

### 2. Sparse and exact visibility

All clean physical primitives participate in visibility at every screen.
Sparse stages may reduce descriptor evaluation, but may not remove occluders.
The P0 sparse primitive screen therefore uses a chunked all-primitive depth
prepass and evaluates canonical codes only for sampled feature primitives.
An unsampled or featureless foreground surface creates field-missing evidence;
it never reveals the descriptor of a rear surface.

The sparse screen is still an approximation to exact 2DGS alpha compositing.
Its global ranking therefore cannot be allowed to erase entire physical
proposal basins. The exact-verification pool first protects a fixed number of
sparse-ranked states from each represented structural anchor, then fills its
remaining budget by the global sparse ranking. These are proposal quotas, not
posterior mass or additive score terms; every retained state is compared by
the same exact surface verifier.
The G21 P0 replay uses the 16 highest-scoring mapping-view anchors, four
protected sparse states per anchor, and 128 exact slots. Thus 64 slots preserve
structural coverage and 64 remain a globally ranked fallback channel. On the
development stress set, protecting all 32 anchors with four states consumed
the fallback and deleted a strict basin; protecting three states from all 32
anchors retained only 32 fallback slots and deleted a different loose basin.
The selected 16-by-4 mixture preserves both failure basins at Top-16/32.
It is required to be conservative in basin survival and must not be presented
as the same likelihood at lower resolution.

### 3. Candidate units are basins

Candidate budgets and priors attach to SE(3)/visibility equivalence classes,
not the number of samples emitted by an anchor. Before paper promotion the
runtime must record raw state count, unique basin count, per-anchor basin
count, and correct-basin rank before and after deduplication. Top-K is a compute
budget over basins.

### 4. Selection probability is post-selection

The success calibrator consumes winner-level features after the complete
proposal, pruning, exact verification, optional refinement, and winner
selection path. Its target is

\[
P(e_t < \tau_t, e_r < \tau_r\mid z_{\mathrm{post-select}}).
\]

Wrong-pose negatives must be produced by the same deployed search, rather
than by random pose sampling. Radius agreement is one correlated stability
feature, not a second independent likelihood. Calibration is fitted only from
outer-fold predictions, with trajectory/map grouping preserved.

### 5. Missingness and ambiguity remain typed

The observation state distinguishes at least: out-of-map, in-map but
canonical-field-unsupported, supported but occluded/mixed, low-quality query,
repeated/symmetric ambiguity, and outside-view-atlas. Repeated states may stay
as an equivalence class. A unique pose is emitted only when selective risk is
below the declared threshold.

## Evaluation gates

Every frozen run reports the following, separately for strict (0.5 m/5 deg)
and loose (1 m/10 deg) thresholds:

| Stage | Primary metric | Required diagnostic |
|---|---|---|
| basin proposal | basin recall at K | raw/unique basin count and per-anchor survival |
| sparse screen | conditional basin survival | bypass-sparse exact oracle |
| exact verifier | opportunity capture | winner regret and repeated-facade subset |
| refinement | success versus pre-refine rank K | six-axis convergence and exact-energy acceptance |
| selective output | risk--coverage and success | catastrophic-risk confidence interval |

Continuous refinement is not opened merely because a local GT perturbation
test passes. It also requires convergence from the actual proposal
distribution and exact verifier acceptance that is monotone along accepted
updates.

## Data and lineage gates

The 17-query seq12/seq14 block consists of official **training** images.  It is
retained only as a named historical stress subset and is no longer a validation
split.  G23 development uses every one of the 1,487 official training images
exactly once as a query under five-fold route-grouped out-of-fold evaluation.
For a publishable OOF result, the held query trajectories must be excluded from
2DGS optimization, primitive selection, canonical-field construction, view
topology, learned readouts, and calibration.  A fixed full-train 2DGS may be
used only for explicitly labelled, non-publishable screening.

After OOF hyperparameters are frozen, the final map and learned components are
rebuilt from all 1,487 official training images and evaluated on all 530
official test images.  The official test was used in historical pre-G23
diagnostics, so it is the standard benchmark test rather than newly untouched
evidence.  Final generalization claims still require additional unseen maps or
routes whose RGB, pose, depth, masks, teacher features, and camera states did
not contribute to any development decision.

The machine-checkable split and fold definition is
`configs/vfm/goal_maplet_stmarys_official_train_oof_v1.json`.

Each result embeds a `goal_maplet_run_manifest_v1` containing the git commit,
tracked-worktree state, full argv, normalized configuration and hash, artifact
lineages, query-list hash, query RADIO-token hash, device/dtype contract, seed
policy, and candidate counts available at each recorded stage. The two-fold
P0 stress replay is versioned in
`scripts/run_goal_maplet_g21_p0_replay.sh` and dispatches seq12/seq14 to two
GPUs without overwriting outputs unless `G21_FORCE=1` is explicitly set.

## Innovation path after P0

The next method change is not another score fusion or larger Top-K. It is a
joint representation upgrade with two bounded-capacity components:

1. a continuous visibility atlas, where a mapping view is a local SE(3) chart
   with a validity region and learned/analytic chart offset rather than a pose
   lookup;
2. a view-conditioned primitive field, storing a canonical mean plus a small
   low-rank/view-prototype residual, observation cone and uncertainty rather
   than multiple full embeddings.

These components are promoted only if they increase outer-fold basin recall
or verification opportunity capture on route-disjoint data. Score-only gains
on the 17-query stress set are diagnostic and cannot justify opening the
untouched test.
