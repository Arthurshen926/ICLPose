# G22 P1 view field, visibility chart, and pose-basin audit

Date: 2026-08-11

## Decision

The three tested P2-style extensions are **not promoted**:

1. the continuous visibility-chart update moves the query toward an incorrect
   repeated facade on the audited failure;
2. the rank-4 view-conditioned primitive field improves held mapping-feature
   reconstruction but does not change localization success on the frozen
   seven-query stress block;
3. SE(3) basin de-duplication and an independent held-out geometry channel do
   not recover the near-correct raw state on `seq12/frame00139.png`.

All three paths remain disabled by default. The negative result is useful: the
current tail is not repaired by adding more continuous chart capacity, a
small view residual, or a second score whose evidence is unavailable in the
correct repeated-facade phase.

## Fixed protocol

The audit uses the two G20.1 outer feature-pipeline folds. `hold_seq12` excludes
seq12 from the RADIO-final mapper, canonical field, readout, graph and
candidate construction; `hold_seq14` does the same for seq14. The seven-query
block is the already-observed development stress set, not an untouched paper
test. The fixed physical 2DGS geometry is not rebuilt, so this is feature-
pipeline cross-fit and cannot establish complete map/query independence.

The corrected exact pool contains 128 states:

- 16 mapping-view structural anchors;
- exactly four protected states per anchor (64 states);
- 64 global sparse-score fills;
- exact all-geometry visibility scoring determines the final ranking.

Mapping-anchor generation count and protected-anchor count are separate
parameters. Increasing generated anchors therefore no longer silently changes
the exact-verification quota.

## Low-rank view-conditioned field

Each primitive retains one canonical RADIO code plus a shared rank-4 tangent
residual basis and five per-primitive coefficient vectors: intercept, two local
tangent view coordinates, local normal view coordinate, and centered log
projected scale. Conditioning is valid only within the observed direction cone
and scale interval; the canonical code is the fail-closed fallback. The
artifact stores no mapping RGB, image path, per-view descriptor, extra VFM
embedding, correspondence, SfM point, or track.

| fold | mapping views | observations | conditioned primitives | coverage | rank-4 residual energy | sample cosine canonical -> conditioned |
|---|---:|---:|---:|---:|---:|---:|
| hold_seq12 | 117 | 2,188,446 | 233,411 | 78.80% | 14.31% | 0.79079 -> 0.79958 |
| hold_seq14 | 122 | 2,234,478 | 236,313 | 79.22% | 14.29% | 0.78291 -> 0.79171 |

The mean reconstruction gain is `+0.00880` cosine in both folds. At runtime,
however, only about 20.9% of exact rendered samples on the audited seq12
failure use a conditioned code, and the small score changes do not repair the
repeated-facade ambiguity.

On the fixed seven-query block, canonical and view-conditioned fields have
identical outcomes:

| rank budget | strict 0.5 m / 5 deg | loose 1 m / 10 deg |
|---|---:|---:|
| Top-1 | 2 / 7 | 5 / 7 |
| Top-16 | 6 / 7 | 7 / 7 |
| Top-32 | 6 / 7 | 7 / 7 |

This implementation is therefore evidence that a small per-primitive view
residual alone is not the missing paper contribution. It should not be claimed
as an accuracy improvement.

Artifacts:

- `map_crossfit_g20_1/hold_seq12/view_conditioned_field_g22_r4.{npz,json}`
- `map_crossfit_g20_1/hold_seq14/view_conditioned_field_g22_r4.{npz,json}`
- `g22_view_conditioned_20260811/seq12_exact128_viewfield.json`
- `g22_view_conditioned_20260811/seq14_exact128_viewfield.json`

## Continuous visibility chart

The feature-free local chart update is implemented as an optional mapping-view
visibility residual with bounded pose increments. On frame 00139 it follows a
wrong but visibility-compatible facade rather than the correct physical phase.
This demonstrates that local visibility smoothness is not an identifying
signal in a repeated facade. The option remains off.

## Pose basins and independent geometry

Greedy camera-center/rotation NMS was tested before the exact pool:

- 0.5 m / 5 deg rejects 152 near duplicates;
- 1 m / 10 deg rejects 273 near duplicates;
- neither setting admits the best raw near-GT state to the exact 128.

Frame 00139 contains a raw state at 1.059 m / 3.946 deg. It is close, but it is
outside the declared 1 m success threshold. It is generated from three
isolated low-retrieval supports. Two geometry alternatives were audited:

1. scoring a candidate from its own three generating supports makes almost
   every hypothesis self-validating and is circular, so it is rejected;
2. excluding those supports and using mapping-anchor-conditioned held-out
   supports is causal, but the near-GT phase has zero independent child support
   for both K=8 and K=16.

The causal geometry branch consequently cannot protect that state. The
candidate-conditioned branch can only do so by reusing its own construction
evidence, which is not defensible as verification. Both geometry-union and
geometry-interleaved final ranking stay disabled.

The failure taxonomy for frame 00139 is therefore:

- not a strict generator absence: a near state exists;
- exact-pool survival failure under the current evidence;
- repeated-facade verifier ambiguity;
- no independent held-out geometry evidence for the near state;
- no demonstrated refinement/gate opportunity because the state never enters
  the exact pool.

## Consequence for the next mainline

The next defensible step is selective localization, not another additive
ranking factor. Candidate ranking, null/abstention, and success calibration
must be evaluated after the complete proposal and exact-selection process,
with query trajectories excluded from every learned map feature. Existing G18
models do not meet that map/query contract and are not reused. A common map
fold excluding seq12 and seq14 is being rebuilt for a strict reciprocal
fixed-epoch diagnostic; its 17 queries remain too small for a final calibrated
risk claim.
