# ICLPose

Real-image VFM localization against a feature-bearing 2DGS surface map.

## Active Mainline

The sole active research line is **V6 maplet-atlas correlation localization**:

```text
mapping RGB + calibrated poses + high-quality 2DGS --offline only-->
  fixed canonical 2DGS maplet geometry
  + raster-contributor-verified dense metric feature atlases

query RGB
  -> RADIO-final RetrievalRegion posterior with complete candidate groups
  -> overlap-aware scene evidence
  -> normalized Region-to-Chart posterior over disconnected MetricSurfaceCharts
  -> per-chart projective frame distributions
  -> correlated chart-factor coarse SE(3) distribution
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
P0 identity/spatial probability semantics are corrected. Retrieval regions
and metric charts are now separate map entities with a strict many-to-many
index, overlapping query-region evidence is deduplicated, and the current
research bridge is `MapletFrameAlignment`. The route-repair audit normalizes
Region-to-Chart probability, uses projective chart observations and a robust
correlated chart-factor solver, and connects runtime atlas re-rendering and
held-out verification. Complete 4,096-mode generation plus a lossless sparse
Top-64 atlas screen has removed the earlier meter-scale implementation
failure: on the strict 12-query set the current Top-1 translation is
0.405 m median / 0.730 m P90 and rotation is 0.665 / 2.551 degrees. The full
generated-pool oracle is 0.279 m / 0.940 degrees median, with 4/12 and 7/12
inside 20 cm and 30 cm respectively. A 16 cm q7 component control remains
stable through normal atlas refinement, localizing the residual losses to
finite candidate selection/final ranking and, more fundamentally, the
cross-trajectory RADIO chart-frame distribution. An attempted `seq11`
ranking calibration was rejected because 0/6 complete Top-64 pools contained
a 30 cm / 3 degree mode. It is not production-qualified and no 530-query
accuracy claim is made. Structured frame/flow learning on multiple
trajectories, typed correlation nulls, adaptive charts and full-scene runtime
occlusion remain open.
The V5 point-similarity route is retained as a failed diagnostic: it never
implemented atlas/query correlation and is not extended.
The [V3 anchor/PnP mainline](docs/vfm/2dgs_surface_localization_mainline.md) is
frozen production baseline, not extended with new modules. The
[SfM landmark mainline](docs/vfm/landmark_localization_mainline.md) is retained
only as a historical baseline.

The [V8.1 structured-region diagnostic](docs/vfm/nonredundant_region_surface_localization_v81.md)
is isolated from the active V6 line. It fixes directed-edge duplication,
correlated-support counting, footprint semantics, virtual-pose truncation and
SE(3) mode collapse, but is not promoted: its final Strict12 result remains
1.72 m median translation. Its O1--O4 oracles identify continuous
maplet-interior RADIO alignment, rather than a larger pose lattice or another
stored embedding, as the next required research module.

## Active Code

- `feature_extract/tools/vfm/build_v6_canonical_maplet_atlas.py`
- `feature_extract/tools/vfm/build_surface_retrieval_maplet_bank.py`
- `feature_extract/tools/vfm/calibrate_v6_retrieval_probabilities.py`
- `feature_extract/tools/vfm/build_v6_region_chart_index.py`
- `feature_extract/tools/vfm/audit_v6_primitive_contributors.py`
- `feature_extract/tools/vfm/build_v6_contributor_cache.py`
- `feature_extract/tools/vfm/train_v6_metric_encoder.py`
- `feature_extract/tools/vfm/bake_v6_metric_atlas.py`
- `feature_extract/tools/vfm/evaluate_v6_oracle_basin.py`
- `feature_extract/tools/vfm/evaluate_v6_retrieval_pose_basin.py`
- `feature_extract/tools/vfm/evaluate_v6_maplet_frame_alignment.py`
- `feature_extract/tools/vfm/evaluate_v6_radio_frame_source.py`
- `feature_extract/tools/vfm/evaluate_v6_projection_model_gap.py`
- `feature_extract/tools/vfm/evaluate_v6_radio_atlas_basin.py`
- `feature_extract/tools/vfm/merge_v6_radio_frame_reports.py`
- `feature_extract/tools/vfm/replay_v6_stage_c.py`
- `feature_extract/tools/vfm/diagnose_v6_stage_c_pool_coverage.py`
- `feature_extract/tools/vfm/merge_v6_stage_c_reports.py`
- `feature_extract/tools/vfm/fit_v6_stage_c_score_calibration.py`
- `feature_extract/tools/vfm/train_v6_structured_frame_adapter.py`
- `feature_extract/tools/vfm/train_v6_global_frame_encoder.py`
- `feature_extract/tools/vfm/bake_v6_retrieval_atlas.py`
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
  tests/test_v6_maplet_atlas_correlation.py \
  tests/test_v6_maplet_frame_alignment.py

python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```
