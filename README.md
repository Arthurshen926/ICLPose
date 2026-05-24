# VFM-MapLoc

Localizable foundation feature selection for prior-map visual localization.

This branch is a clean reset of the previous RADIO/POFD/CPR exploration. The
active project studies a narrower and more defensible question:

> Can we learn a compact, interpretable, and mapable subspace from frozen vision
> foundation model dense tokens that provides reliable map-conditioned evidence
> for localization hypothesis verification?

The project does not claim to be a new camera pose refinement solver. Fixed
retrieval, matching, PnP, photometric, or 3DGS-style solvers may be used as
candidate generators or downstream handoff modules, but the contribution is the
selected feature evidence used to verify and reject hypotheses.

## Mainline

The active pipeline is:

```text
raw VFM token bank
  -> localizable feature selector
  -> selected feature lifting into an explicit 3D map
  -> map-conditioned hypothesis verification
  -> risk-aware handoff to fixed downstream solvers
```

Core terms used in this branch:

- raw VFM token bank
- localizable feature subspace
- candidate hypothesis
- map-conditioned hypothesis verification
- selected feature map
- hard negative rejection
- risk calibration
- fixed solver handoff

Terms from the old branch, such as POFD, CPR, coarse/fine feature stages, and
GT-centered q50 lattice results, are not part of the active method narrative.

## Active Code

- `feature_extract/vfm/`: protocol records, hypothesis schema, selector, metrics,
  selected-track aggregation, verifier interfaces, and reporting helpers.
- `feature_extract/tools/vfm/`: small command-line entrypoints for the new
  experiment surface.
- `feature_extract/configs/vfm/`: clean data and protocol configs.
- `docs/vfm/`: method definition, evaluation protocol, experiment plan, and
  cleanup manifest.

Reusable infrastructure remains available in:

- `feature_extract/extractors/`: frozen VFM extractors.
- `feature_extract/utils/`: model loading utilities.
- `feature_retrieval/`: candidate generation infrastructure.
- `feature_gaussian/`: fixed 3D Gaussian map/rendering infrastructure.
- `data/`: dataset loaders.

## Evaluation Gates

Every result must be tagged with one protocol kind:

- `controlled_lattice`: GT-centered diagnostic only.
- `reference_pose`: reference-image or reference-pose ranking.
- `real_retrieval`: deployment-like retrieval candidates without GT candidate
  generation.
- `rendered_pose`: explicit rendered pose candidates around a declared init.

The main gates are:

1. Feature Utility: selected features beat raw VFM, PCA, random projection, and
   metadata-only baselines on hypothesis ranking.
2. Causal Selection: high-utility removal hurts, low-utility removal does not,
   and feature shuffle/wrong-scene controls collapse toward baseline.
3. Mapability: selected features aggregate into stable explicit 3D tracks and
   retain ranking signal after rendering/projection from the map.
4. Hard-Case Utility: hard false accepts are reduced on retrieval false
   positives, repeated structures, weak texture, and solver traps.
5. Final Localization: fixed downstream solver handoff is reported separately
   from solver-free hypothesis verification.

See `docs/vfm/evaluation_protocol.md` for metrics and promotion gates.
