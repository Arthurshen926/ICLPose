# ICLPose

2DGS Synthetic RADIO-MATCHA for learning geometry-aware VFM feature mappings.

## Active Mainline

The current mainline is **2DGS Synthetic RADIO-MATCHA**. The phase-1 question is
narrow and deliberately staged:

> Can RADIO/VFM dense tokens be trained, in a controlled 2DGS simulator, into a
> geometry-matchable feature mapping that supports accurate source-to-target
> correspondence and PnP before we attempt real-image domain transfer or compact
> 3D feature aggregation?

This phase does **not** claim full real-image localization, SOTA pose refinement,
or a deployable map-conditioned verifier. It first isolates feature learning:
both source and target images are rendered from the same 2DGS scene so the system
can be tested without the real-image-to-render domain gap.

The active pipeline is:

```text
2DGS scene and calibrated poses
  -> deterministic synthetic source/target pose sampling
  -> 2DGS RGB/depth/alpha rendering for both views
  -> frozen RADIO/VFM dense token extraction
  -> MATCHA-style geometry matching and fine correspondence learning
  -> pair-level correspondence evaluation
  -> PnP from predicted correspondences and rendered depth
  -> later real-image transfer and compact 3D aggregation
```

## Current Status

See `docs/vfm/2dgs_synthetic_radio_matcha_status.md` for the current
implementation status, metrics, acceptance gates, and next experiments.

As of the latest local audit:

- Gate 1 single-pair synthetic overfit is passed on OldHospital micro pairs.
- Gate 2 is not yet passed because held-out validation currently uses 16 pairs,
  below the required 32, and small-pose precision is still slightly under the
  threshold in the best documented run.
- Gate 3 is not yet passed because medium-pose validation currently uses 16
  pairs, below the required 128.
- No real-image localization claim is active.

## Active Code

- `feature_extract/vfm/matcha_synthetic_pairs.py`: deterministic synthetic
  source/target pose sampling and pose-bin metadata.
- `feature_extract/tools/vfm/build_matcha_2dgs_synthetic_manifest.py`: tensor-free
  synthetic manifest generation.
- `feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`: streaming
  MATCHA training entrypoint, including `pair_source=2dgs_synthetic` routing.
- `feature_extract/tools/vfm/eval_matcha_2dgs_synthetic_pairs.py`: synthetic
  pair-level correspondence and PnP evaluation.
- `feature_extract/vfm/matcha_*`: MATCHA model, supervision, cache, and fine
  matching components.
- `feature_extract/vfm/official_2dgs_renderer.py`: official 2DGS rendering
  adapter used for synthetic RGB/depth generation.

## Historical And Reference Lines

Previous VFM-MapLoc selector, selected-track aggregation, SfM/Gaussian anchor
mapping, rendered-map verifier, candidate-bank reranking, and hard-case utility
experiments are retained as reference infrastructure. They are not the active
paper narrative for this branch.

Those experiments may still be useful later for compact 3D feature aggregation,
for example aggregating learned features onto SfM points or Gaussian anchors and
then doing patch-level matching plus PnP. They must not be mixed into the phase-1
2DGS Synthetic RADIO-MATCHA claim table unless a document explicitly states the
bridge experiment and evaluation protocol.

## Evaluation Discipline

Every result must state which stage it belongs to:

- `synthetic_overfit`: one deterministic pair, diagnostic only.
- `synthetic_heldout_micro_small`: held-out micro/small synthetic pairs.
- `synthetic_heldout_medium`: held-out micro/small/medium synthetic pairs.
- `real_gt_render_diagnostic`: real query image to GT-pose 2DGS render,
  allowed only after synthetic gates pass.
- `real_candidate_localization`: reference/retrieval/init candidate localization,
  not active until the real GT-render diagnostic is credible.

Synthetic render-to-render pose error is a simulator-domain diagnostic. It must
not be reported as real localization accuracy.

## Verification

The current repository expects the project root on `PYTHONPATH` when running the
focused tests from this checkout:

```bash
PYTHONPATH=. pytest tests/test_matcha_synthetic_pairs.py \
  tests/test_matcha_streaming_manifest.py \
  tests/test_matcha_joint_training.py \
  tests/test_matcha_coarse_supervision.py \
  tests/test_matcha_multiview_supervision.py -q

PYTHONPATH=. pytest tests/test_vfm_*.py -q

python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```

## Strict Audit

See `docs/vfm/2dgs_synthetic_radio_matcha_audit.md` for the current technical
and methodological audit. The audit is intentionally conservative: results are
promoted only when the relevant gate passes with the required sample count and
protocol separation.
