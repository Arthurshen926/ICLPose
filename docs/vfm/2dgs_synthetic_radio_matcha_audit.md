# 2DGS Synthetic RADIO-MATCHA Audit

## Verdict

Current state: **promising but not publishable as localization yet**.

The phase-1 design is now coherent: train a VFM/RADIO geometry matcher in a
controlled 2DGS synthetic domain before attempting real transfer or compact 3D
feature aggregation. That is a defensible staged research plan.

The current evidence, however, supports only a narrower statement:

> The implementation can overfit a synthetic pair and shows encouraging
> OldHospital render-to-render held-out correspondence/PnP metrics on small
> validation sets.

It does not yet prove robust synthetic generalization, cross-scene behavior,
real-image transfer, compact 3D mapability, or final localization accuracy.

## Core Problems

1. **[S | Confirmed] Synthetic render-to-render success is not real localization.**

   Both source and target images are rendered by the same 2DGS pipeline. The
   model sees matched renderer statistics, matched depth convention, matched
   lighting artifacts, and no real-camera appearance gap. This is appropriate
   for phase 1, but any claim beyond synthetic geometry learning would be
   methodologically invalid.

   Destructive impact: real localization claims would be unsupported even if the
   synthetic PnP median error is centimeter-level.

2. **[A | Confirmed] Gate 2 and Gate 3 are not passed.**

   The current held-out reports use 16 pairs. Gate 2 requires at least 32
   held-out micro/small pairs; Gate 3 requires at least 128 mixed pairs. The
   documented small-bin precision16 is 0.9469, slightly below the 0.95 Gate 2
   threshold.

   Destructive impact: current validation is still a smoke-scale diagnostic, not
   a stable generalization result.

3. **[A | Highly likely] The model may learn renderer-specific shortcuts.**

   2DGS source and target images share rendering artifacts, alpha/depth behavior,
   view-dependent appearance approximation, and synthetic boundary statistics.
   The model may exploit these regularities rather than learning a feature
   mapping that transfers to real query images.

   Destructive impact: high synthetic precision can collapse on real GT-render
   diagnostics.

4. **[A | Confirmed] The theory-to-code chain does not yet reach compact 3D
   representation.**

   The stated long-term theory is that VFM tokens can become mapable,
   interpretable, localization-useful features. Current code tests pairwise
   synthetic matching and PnP. It does not yet show that the learned mapping can
   be aggregated into SfM points, Gaussian anchors, or another compact 3D
   representation without losing correspondence quality.

   Destructive impact: the current phase solves only the feature-learning
   precondition, not the full map-conditioned localization problem.

5. **[A | Confirmed] Old VFM-MapLoc documents and gates can contaminate the
   claim.**

   Historical selector/verifier/status files use a different method narrative:
   selected feature maps, risk-aware handoff, rendered-map verification, and
   candidate reranking. Those are useful references, but they are not the active
   phase-1 method.

   Destructive impact: mixing these tables with synthetic MATCHA tables creates
   a false impression of an end-to-end validated localization system.

6. **[B | Confirmed] Patch-correlation remains unstable as a primary fine path.**

   The design already records the earlier regression: patch-correlation
   fine accuracy was extremely low in a previous path. Current held-out small
   and medium summaries include patch-correlation diagnostics, but the active
   acceptance logic still needs to treat patch-correlation as an ablation until
   it reliably passes frozen synthetic gates.

   Destructive impact: changing the fine matcher after inspecting validation
   metrics will make the reported recipe look overfit to the validation set.

7. **[B | Confirmed] Current evidence is OldHospital-only.**

   The phase is allowed to start on one scene, but a paper-facing claim about
   VFM/RADIO geometry learning cannot rest on one 2DGS reconstruction.

   Destructive impact: scene-specific geometry, texture, and renderer quality
   may be driving the result.

8. **[B | Needs experiment] Synthetic pose sampling may leave the camera
   manifold.**

   The sampler jitters camera centers and rotations in continuous space. This is
   useful, but some samples may be physically implausible relative to the real
   capture path or scene visibility. Overlap filters reduce the damage; they do
   not prove the distribution is representative.

   Destructive impact: the model may train on views that are easy for the
   renderer but irrelevant to real localization.

9. **[B | Needs experiment] Expected depth from 2DGS is not necessarily a hard
   surface depth.**

   PnP supervision and correspondence evaluation use rendered depth. 2DGS
   expected depth can mix surfaces near boundaries or semi-transparent regions.
   Alpha/depth filters help, but boundary supervision remains a possible noise
   source.

   Destructive impact: the model may be penalized or rewarded for renderer-depth
   conventions rather than geometric correctness.

10. **[C | Confirmed] Test commands require explicit `PYTHONPATH=.`.**

    Direct pytest invocation from this checkout may fail to import
    `feature_extract` unless the repository root is on `PYTHONPATH`.

    Destructive impact: reproducibility friction, not a method failure.

## Chain Consistency Check

```text
Theory:
  VFM/RADIO tokens can support geometry-aware, localization-useful features.

Current method:
  Train MATCHA-style correspondence on source/target views rendered from 2DGS.

Current code:
  Synthetic manifest -> 2DGS renderer -> RADIO tokens -> MATCHA training/eval
  -> PnP from predicted correspondences.

Current results:
  Strong single-pair overfit and promising 16-pair held-out synthetic metrics.

Current conclusion boundary:
  Synthetic geometry matching is plausible. Real localization, compact 3D
  aggregation, and map-conditioned verification remain unproven.
```

The chain is coherent only if the conclusion stops at synthetic geometry
learning. It breaks if the conclusion claims real localization or compact
mapability.

## Minimum Promotion Standard

Before this line is paper-facing as a main result:

1. Freeze one training/evaluation recipe before scaling validation.
2. Pass Gate 2 on at least 32 held-out micro/small synthetic pairs.
3. Pass Gate 3 on at least 128 held-out mixed synthetic pairs with per-bin
   reporting.
4. Run at least 3 seeds after recipe freeze.
5. Add manifest overlap checks to prove train and validation pair identities do
   not collide.
6. Repeat on at least one additional scene or explicitly state this is an
   OldHospital-only engineering milestone.
7. Only then run real query to GT-render diagnostics.

## What To Keep

- Deterministic synthetic manifest generation.
- 2DGS render-to-render isolation as phase 1.
- Pair-level correspondence metrics before PnP.
- PnP inlier precision, not only pose error.
- Historical 3D anchor/SfM/Gaussian aggregation as future reference, not active
  evidence.

## What To Stop Claiming

- Do not claim final localization.
- Do not claim map-conditioned verification.
- Do not claim compact 3D mapability.
- Do not claim cross-domain real-image transfer.
- Do not mix old selector/verifier metrics with synthetic MATCHA metrics in one
  contribution table.
