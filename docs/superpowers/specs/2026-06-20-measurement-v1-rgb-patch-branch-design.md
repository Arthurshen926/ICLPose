# Measurement V1 RGB Patch Branch Design

## Scope

This phase replaces whole-image stride-4 measurement training with a local, high-resolution RGB patch measurement branch. It does not change reference retrieval, pose scoring, confidence ranking, or PnP protocols.

## Design

- Read `measurement_v1` rows with fixed render anchors and query GT projections.
- Load raw query RGB from `image_root / query_id` and render RGB from `render_cache_manifest.csv`.
- Crop only local RGB patches around the render anchor and query prior center.
- Encode both sides with a shared lightweight texture CNN.
- Build a local correlation likelihood over query-side residual offsets relative to the center prior.
- Train with continuous target residual NLL, covariance from spatial moments, and a dustbin target when GT is outside the search window.
- Always report center-prior baseline and measurement improvement ratio on the same held-out validation split.

## Gates

- Near-GT validation must beat the center baseline, not merely report low absolute EPE.
- The first meaningful gate is median EPE clearly below `0.5px` and improvement ratio approaching `>80%`.
- Reference-topK, learned scorer, and PnP optimization remain blocked until the measurement branch passes this gate.
