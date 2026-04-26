# Legacy branches

This directory collects older or comparison-only lines that are **not** the current optimization target.

## Subdirectories

- `gsff_baseline/`
  - GSFF reproduction and GSFF+RADIO comparison experiments.
  - Kept only as a baseline / historical reference.

- `feature_ablations/`
  - Early FlowFeat and DA3 feature pipelines.
  - Useful for ablation history, but not part of the current RADIO mainline.

## Mainline reminder

The active project direction is:

- reconstruction / feature learning: `RADIO + DCFF`
- localization: `concat_loc`
- formulation: `dense correspondence / feature flow -> geometric solve -> iterative refinement`
