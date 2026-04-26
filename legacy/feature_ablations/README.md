# Feature ablations (legacy)

This directory archives early feature pipelines that were explored before the project converged on RADIO as the teacher / representation source.

## Included lines

- FlowFeat extraction / compression entrypoints
- DA3 extraction / pipeline entrypoints
- corresponding configs for old pose-training attempts

## Why they are archived

These lines were useful for exploration, but they did **not** become the strongest localization path.

The current project conclusion is that RADIO features transfer better into the downstream localization stack than these earlier alternatives.

## Remaining helper code

Some low-level helper modules still remain under top-level directories like `feature_extraction/` and `feature_3dgs/` for reference. The archived entrypoints and configs have been moved here to reduce clutter around the current mainline.
