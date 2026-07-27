# ICLPose

Real-image VFM localization against a feature-bearing 2DGS surface map.

## Active Mainline

The sole active research line is **V6 maplet-atlas correlation localization**:

```text
mapping RGB + calibrated poses + high-quality 2DGS --offline only-->
  fixed canonical 2DGS maplet geometry
  + raster-contributor-verified dense metric feature atlases

query RGB
  -> RADIO-final maplet retrieval with complete candidate groups
  -> region-level coarse pose distribution
  -> area-render selected canonical maplet atlases
  -> local multi-modal displacement correlation
  -> analytic robust joint-SE(3) update
  -> maplet-disjoint held-out verification
```

The target prior stores only metric 2DGS geometry and bounded anonymous
feature-map statistics. It stores no mapping RGB, mapping image path/ID,
stable-anchor identity, per-view descriptor list, or ALIKE descriptor map.
Runtime uses no SfM
points/tracks, RADIO intermediate features, LoFTR, pairwise query/reference
matching, single-point cosine pose energy, or final point-correspondence PnP.

## Current Status

See [the V6 mainline](docs/vfm/2dgs_maplet_atlas_localization_v6.md) for its
architecture, artifact contracts and strict G0–G4 promotion gates.
P0 identity/spatial probability semantics are corrected, but the current
region-level coarse proposal is still 0/12 at the 30 cm / 3 degree strict
audit gate. Typed correlation nulls, iterative fine alignment, adaptive
charts and full-scene runtime occlusion are also still open. V6 is therefore
not production-qualified and no 530-query accuracy claim is made.
The V5 point-similarity route is retained as a failed diagnostic: it never
implemented atlas/query correlation and is not extended.
The [V3 anchor/PnP mainline](docs/vfm/2dgs_surface_localization_mainline.md) is
frozen production baseline, not extended with new modules. The
[SfM landmark mainline](docs/vfm/landmark_localization_mainline.md) is retained
only as a historical baseline.

## Active Code

- `feature_extract/tools/vfm/build_v6_canonical_maplet_atlas.py`
- `feature_extract/tools/vfm/build_surface_retrieval_maplet_bank.py`
- `feature_extract/tools/vfm/calibrate_v6_retrieval_probabilities.py`
- `feature_extract/tools/vfm/build_v6_anonymous_pose_vote_bank.py`
- `feature_extract/tools/vfm/audit_v6_primitive_contributors.py`
- `feature_extract/tools/vfm/build_v6_contributor_cache.py`
- `feature_extract/tools/vfm/train_v6_metric_encoder.py`
- `feature_extract/tools/vfm/bake_v6_metric_atlas.py`
- `feature_extract/tools/vfm/evaluate_v6_oracle_basin.py`
- `feature_extract/tools/vfm/evaluate_v6_retrieval_pose_basin.py`
- `feature_extract/vfm/localization_v6/`

## Historical And Reference Lines

Synthetic 2DGS RADIO-MATCHA, SfM tracks, rendered-map verification, image
retrieval/submaps, LoFTR, and legacy patch-offset measurement remain reference
or ablation paths. They must not be mixed into production 2DGS surface results.

## Evaluation Discipline

Every result must identify its 2DGS source, surface-field and mapper artifacts,
camera calibration, query manifest, maplet budget, pose policy, and split role.
A full 530-query localization run is forbidden until G0 geometry, G1
correlation, G2 local basin and G3 coarse-proposal gates all pass. Validation
trajectories must be disjoint from encoder training trajectories.

## Verification

The current repository expects the project root on `PYTHONPATH` when running the
focused tests from this checkout:

```bash
PYTHONPATH=. pytest -q \
  tests/test_v6_maplet_atlas_correlation.py

python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```
