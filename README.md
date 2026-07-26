# ICLPose

Real-image VFM localization against a feature-bearing 2DGS surface map, with
RADIO-final maplet retrieval, ALIKE anchor assignment, and a geometry-safe PnP
backend.

## Active Mainline

The current mainline is **real-image, feature-map-only 2DGS localization**:

```text
mapping RGB + calibrated poses + high-quality 2DGS --offline only-->
  RADIO-final surface/maplet modes + multi-view ALIKE surface anchors

query RGB
  -> RADIO-final query-aligned maplet posterior
  -> ALIKE point-to-anchor partial assignment with explicit null mass
  -> independent 2DGS feature-map pose evidence
  -> grouped AP3P/PnP + soft geometric EM
  -> feature-support-gated map-only coverage fallback
```

The deployed prior stores metric 2DGS geometry and feature statistics. It does
not store or retrieve mapping RGB. The runtime does not use SfM points/tracks,
RADIO intermediate features, LoFTR, or any pairwise query/reference image
matching.

## Current Status

See [the 2DGS surface mainline](docs/vfm/2dgs_surface_localization_mainline.md)
for the architecture, data contracts, prohibited path combinations, and
StMaryChurch development-validation status. The
[SfM landmark mainline](docs/vfm/landmark_localization_mainline.md) is retained
only as a historical baseline.

## Active Code

- `feature_extract/tools/vfm/build_feature_aligned_surface_anchors.py`
- `feature_extract/tools/vfm/build_hybrid_feature_surface_map.py`
- `feature_extract/tools/vfm/build_surface_anchor_deployment_replay.py`
- `feature_extract/tools/vfm/train_surface_anchor_set_matcher.py`
- `feature_extract/tools/vfm/localize_2dgs_surface_queries.py`
- `feature_extract/tools/vfm/cascade_surface_localization_results.py`
- `feature_extract/tools/vfm/select_surface_pose_by_feature_map.py`
- `feature_extract/vfm/localization/surface_anchor_set_matcher.py`
- `feature_extract/vfm/localization/surface_localization.py`

## Historical And Reference Lines

Synthetic 2DGS RADIO-MATCHA, SfM tracks, rendered-map verification, image
retrieval/submaps, LoFTR, and legacy patch-offset measurement remain reference
or ablation paths. They must not be mixed into production 2DGS surface results.

## Evaluation Discipline

Every result must identify its 2DGS source, feature-map artifacts, camera
calibration manifest, query manifest, maplet proposal budget, anchor assignment
checkpoint, pose policy, and split role. Reused development data cannot produce
an untouched-test or production-promotion claim.

## Verification

The current repository expects the project root on `PYTHONPATH` when running the
focused tests from this checkout:

```bash
PYTHONPATH=. pytest -q \
  tests/test_vfm_2dgs_surface_map.py \
  tests/test_vfm_surface_anchor_set_matcher.py \
  tests/test_cascade_surface_localization_results.py

python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```
