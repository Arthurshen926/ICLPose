# Mainline first-principles repair record (2026-08-13)

## Scope

This record applies only to the frozen-prior VFM localization path in
`/root/ICLPose`: frozen 2DGS geometry/map artifacts, frozen RADIO-final query
features, maplet retrieval, stable-anchor candidate generation, grouped PnP,
fixed evidence, and conservative refinement. It does not start or authorize
five-fold geometry rebuilding, MAtCha/SfM alignment, Gaussian retraining, or
any new map reconstruction.

## Repairs completed

### 1. Probability conservation under truncation

Every maplet/anchor top-ᴸ operation now computes the posterior over the full
candidate set first. Retained candidates keep their mass; omitted valid mass is
added to an explicit null state. Invalid/non-finite/empty descriptor rows have
zero candidate mass and cannot become a top-ᴸ identity. Duplicate metric
anchors are merged before normalization rather than receiving probability twice.

This prevents the common failure mode

```text
full posterior -> discard tail -> renormalize prefix -> artificial certainty
```

### 2. Layout evidence is causally usable

Support-layout fitting uses a bounded raw-descriptor pool wider than the final
returned maplet Top-ᴸ. Layout compatibility is applied inside that pool
before the final Top-ᴸ cut. The pool size is serialized in the matcher
configuration. A lower descriptor-ranked maplet can therefore be promoted by
independent whole-query layout evidence, while runtime cost remains bounded.

### 3. Pose generation cannot consume zero evidence

Grouped PnP and random minimal-set generation consume only candidates with
strictly positive posterior mass. Valid placeholders with zero mass, all-null
rows, and zero total sampling mass now produce an abstention/no-hypothesis
result instead of a random or degenerate pose.

### 4. Conditional confidence is not absolute confidence

The set-matcher pose gate now requires both:

1. enough groups whose conditional identity mass exceeds their conditional null;
2. mean absolute null probability below the configured ceiling.

The second condition prevents a tiny retained mass from looking confident merely
because it was normalized by an even larger null mass. Gate failures record the
limiting reason.

### 5. Mapping-view artifact and posterior validation

Mapping-view poses are checked as proper finite SE(3) matrices (orthonormal
rotation, positive determinant, homogeneous last row). Mapping-view posterior
objects validate shape, finite values, bounds, and probability conservation.
Input candidate probability rows are also rejected when their mass exceeds one.
View truncation continues to transfer omitted view mass to the mapping null.

### 6. Deterministic ties

Top-ᴸ selection uses stable descending score plus metric-ID tie-breaking. This
removes dependence on unspecified `argpartition` tie order and makes repeated
runs reproducible at the candidate boundary.

### 7. Fixed-map promotion cannot be bypassed by fallback

Every generated pose proposal, including the direct RADIO surface-observation
fallback, must have finite fixed-map evidence and at least the configured
minimum number of fixed-map inliers before it enters final selection. The
per-query result is initialized as unaccepted while proposals are being
evaluated, so a successful PnP proposal cannot survive merely because all
independent verification branches failed.

## Verification

The focused mainline suite currently passes **49 tests**. The expanded
surface/pose/renderer regression scope passes **78 tests**, including the new
probability, gate, layout-pool, zero-mass-PnP, SE(3), posterior, and tie
counterexamples. Python compilation and `git diff --check` pass.

These are correctness/regression results, not a new real-image accuracy claim.
No new long experiment, Gaussian reconstruction, five-fold run, MAtCha run, or
matcher/alignment run was started in this repair pass.

## Remaining first-principles blockers

The repairs remove implementation-level confidence inflation and invalid-input
paths, but they do not prove that RADIO identifies the correct physical basin.
The remaining risks are:

- repeated façades can still produce genuinely similar full-map evidence;
- candidate generation recall is not yet measured on a newly frozen replay after
  these changes;
- score components are still a calibrated engineering likelihood, not a single
  learned posterior over SE(3);
- continuous refinement is local and cannot recover a missing basin;
- the historical 530-query numbers are development/non-regression evidence and
  must not be silently relabeled as a fresh result.

The next admissible experiment is therefore a bounded replay using the existing
frozen map and RADIO artifacts, with candidate recall, null/ambiguity rates,
runtime by stage, and pose success reported together. It must not change the
map, retrain the Gaussian scene, or introduce a new fold protocol.
