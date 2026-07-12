# ICLPose

Real-image RADIO landmark localization with staged global retrieval, local
assignment, optional pixel measurement, and a geometry-safe PnP backend.

## Active Mainline

The current mainline is **real-image, no-render landmark localization**:

```text
real mapping RGB + SfM tracks
  -> RADIO full-map projected-observation landmark bank
real query RGB
  -> global full-bank top-L landmark proposals (no image retrieval/submap)
  -> candidate-conditioned local maplet assignment + KEEP/SWITCH/DROP
  -> selected-only RGB measurement after its gate
  -> conflict-resolved, coverage-safe PnP
```

## Current Status

See [the landmark mainline](docs/vfm/landmark_localization_mainline.md) for the
architecture, data contracts, staged gates, prohibited path combinations, and
current status. OldHospital results already used for method selection are
development results and are not production claims.

## Active Code

- `feature_extract/tools/vfm/build_projected_observation_landmark_bank.py`
- `feature_extract/vfm/landmark_retrieval_training.py`
- `feature_extract/vfm/localization/global_landmark_ann.py`
- `feature_extract/vfm/localization/candidate_maplet_data.py`
- `feature_extract/vfm/localization/candidate_maplet_matcher.py`
- `feature_extract/tools/vfm/train_candidate_maplet_matcher.py`
- `feature_extract/tools/vfm/eval_candidate_scores_pose_safe.py`
- `feature_extract/vfm/localization/pose_safe_selection.py`
- `feature_extract/vfm/query_to_3d_matching.py`

## Historical And Reference Lines

Synthetic 2DGS RADIO-MATCHA, rendered-map verification, Gaussian anchors, image
retrieval/submaps, and legacy patch-offset measurement remain reference or
ablation paths. They must not be mixed into real-image landmark results without
an explicit bridge protocol.

## Evaluation Discipline

Every result must identify its descriptor space, bank manifest, aggregation,
proposal L, query selector, assignment policy, measurement state, PnP policy,
and split role. Reused development data cannot produce a production promotion.

## Verification

The current repository expects the project root on `PYTHONPATH` when running the
focused tests from this checkout:

```bash
PYTHONPATH=. pytest -q \
  tests/test_candidate_maplet_data.py \
  tests/test_candidate_maplet_matcher.py \
  tests/test_pose_safe_selection.py \
  tests/test_vfm_query_to_3d_matching.py

python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```
