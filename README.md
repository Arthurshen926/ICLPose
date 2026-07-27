# ICLPose

Real-image VFM localization against a feature-bearing 2DGS surface map.

## Active Mainline

The development mainline is **anchor-free 2DGS surface feature alignment**:

```text
mapping RGB + calibrated poses + high-quality 2DGS --offline only-->
  RADIO-final maplets
  + one RADIO-final distribution per detector-repeatable clean 2DGS surface cell

query RGB
  -> RADIO-final maplet retrieval / coarse pose
  -> render disconnected selected maplet feature patches
  -> continuous SE(3) RADIO feature-field alignment
  -> ALIKE detector response as a spatial weight only
```

The deployed prior stores metric 2DGS geometry and feature statistics. It does
not store or retrieve mapping RGB. It has no stable anchor identity, ALIKE
descriptor, descriptor list, or mapping image path. The runtime does not use
SfM points/tracks, RADIO intermediate features, LoFTR, pairwise
query/reference image matching, or final point-correspondence PnP.
The V4 query path evaluates only ALIKE's detector score channel; it does not
materialize an ALIKE dense descriptor map.

## Current Status

See [the anchor-free V4 mainline](docs/vfm/2dgs_surface_feature_field_v4.md)
for its architecture, contracts, construction, and mandatory local-basin gate.
The [V3 anchor/PnP mainline](docs/vfm/2dgs_surface_localization_mainline.md) is
frozen as a measured baseline, not extended with new modules. The
[SfM landmark mainline](docs/vfm/landmark_localization_mainline.md) is retained
only as a historical baseline.

## Active Code

- `feature_extract/tools/vfm/build_detector_weighted_2dgs_surface_field.py`
- `feature_extract/tools/vfm/build_surface_retrieval_maplet_bank.py`
- `feature_extract/tools/vfm/train_surface_metric_feature_mapper.py`
- `feature_extract/tools/vfm/apply_surface_metric_mapper_to_field.py`
- `feature_extract/tools/vfm/evaluate_2dgs_surface_alignment_basin.py`
- `feature_extract/tools/vfm/evaluate_surface_maplet_retrieval.py`
- `feature_extract/tools/vfm/localize_2dgs_surface_feature_field.py`
- `feature_extract/vfm/localization/surface_feature_field.py`
- `feature_extract/vfm/localization/surface_retrieval_maplets.py`
- `feature_extract/vfm/localization/alike_detector_only.py`
- `feature_extract/vfm/localization/surface_metric_feature_mapper.py`
- `feature_extract/vfm/localization/continuous_surface_alignment.py`

## Historical And Reference Lines

Synthetic 2DGS RADIO-MATCHA, SfM tracks, rendered-map verification, image
retrieval/submaps, LoFTR, and legacy patch-offset measurement remain reference
or ablation paths. They must not be mixed into production 2DGS surface results.

## Evaluation Discipline

Every result must identify its 2DGS source, surface-field and mapper artifacts,
camera calibration, query manifest, maplet budget, pose policy, and split role.
A full localization run is not promoted when the oracle-maplet local-basin gate
fails. Reused development data cannot produce an untouched-test claim.

## Verification

The current repository expects the project root on `PYTHONPATH` when running the
focused tests from this checkout:

```bash
PYTHONPATH=. pytest -q \
  tests/test_vfm_surface_feature_field.py \
  tests/test_vfm_continuous_surface_alignment.py

python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```
