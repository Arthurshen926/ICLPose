# Real Render-Query Protocol Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the RADIO-MATCHA render-query pipeline auditable before scaling training: explicit protocol metadata, all-real-query manifest hygiene, topK/perturbation evaluation support, and frozen full-test matrix commands.

**Architecture:** Add a small protocol module that validates render-query manifest metadata without changing tensor storage. Extend existing evaluation pose-selection helpers for generic reference topK and deterministic rotation offsets. Add a matrix-runner CLI that emits reproducible commands instead of hiding protocol choices in ad hoc shell history.

**Tech Stack:** Python 3, argparse, dataclasses, pytest, existing `feature_extract.vfm` helpers.

---

## File Structure

- Create: `feature_extract/vfm/matcha_render_query_protocol.py`
  - Defines protocol constants, metadata validation, split/query overlap summaries, and helper constructors.
- Modify: `feature_extract/vfm/matcha_streaming_manifest.py`
  - Calls render-query metadata validation only when `metadata["pair_source"]` is present.
- Modify: `feature_extract/tools/vfm/build_matcha_streaming_manifest.py`
  - Adds `--pair_source`, `--validation_manifest`, and summary metadata for real render-query protocols.
- Modify: `feature_extract/vfm/render_pose_protocol.py`
  - Adds fixed-axis rotation-offset pose construction and generic reference topK selection labels.
- Modify: `feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`
  - Accepts `reference_top10`, `gt_rotation_offset`, and `--render_pose_rotation_offset_deg`.
- Create: `feature_extract/tools/vfm/run_matcha_real_render_query_eval_matrix.py`
  - Builds full-test evaluation commands for GT, translation offsets, rotation offsets, and reference topK.
- Test: `tests/test_matcha_render_query_protocol.py`
- Test: `tests/test_render_pose_protocol.py`
- Test: `tests/test_render_rgb_pose_update_protocol.py`
- Test: `tests/test_matcha_real_render_query_eval_matrix.py`

## Task 1: Render-Query Metadata Validation

**Files:**
- Create: `feature_extract/vfm/matcha_render_query_protocol.py`
- Modify: `feature_extract/vfm/matcha_streaming_manifest.py`
- Test: `tests/test_matcha_render_query_protocol.py`

- [ ] **Step 1: Write failing metadata tests**

```python
def test_render_query_metadata_requires_pair_source_and_counts() -> None:
    records = (
        MatchaStreamingPairRecord("q1.png", "train", "A_gt", 0, 0, 0, 3),
        MatchaStreamingPairRecord("q2.png", "train", "A_gt", 0, 1, 0, 3),
    )
    manifest = MatchaStreamingPairManifest(
        records=records,
        metadata=build_render_query_metadata(
            pair_source="real_gt_render",
            source_query_manifest="train_manifest.json",
            query_pose_file="poses_train.txt",
            train_query_ids=["q1.png", "q2.png"],
            validation_query_ids=["q3.png"],
            pair_type_counts={"A_gt": 2},
        ),
    )
    assert manifest.metadata["train_query_count"] == 2
    assert manifest.metadata["validation_query_count"] == 1
    assert manifest.metadata["train_validation_query_overlap_count"] == 0
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/test_matcha_render_query_protocol.py -q`

Expected: import failure for `feature_extract.vfm.matcha_render_query_protocol`.

- [ ] **Step 3: Implement protocol module and optional manifest validation**

Add constants for `2dgs_synthetic`, `real_gt_render`, `real_perturbed_render`, and `real_reference_render`; reject unknown sources; require `source_query_manifest`, `query_pose_file`, `train_query_count`, `validation_query_count`, `train_validation_query_overlap_count`, and `pair_type_counts`; require `candidate_bank` only for `real_reference_render`.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=. pytest tests/test_matcha_render_query_protocol.py tests/test_matcha_streaming_manifest.py -q`

Expected: all tests pass.

## Task 2: Builder Metadata Hygiene

**Files:**
- Modify: `feature_extract/tools/vfm/build_matcha_streaming_manifest.py`
- Test: `tests/test_matcha_streaming_manifest.py`

- [ ] **Step 1: Write failing CLI and summary tests**

```python
def test_build_streaming_manifest_accepts_real_pair_source() -> None:
    args = parse_streaming_args([
        "--query_manifest", "train_manifest.json",
        "--output_manifest", "streaming.json",
        "--summary_json", "summary.json",
        "--split_name", "train",
        "--pair_types", "A_gt,B_trans025",
        "--pair_source", "real_perturbed_render",
        "--query_pose_file", "poses_train.txt",
        "--validation_manifest", "val_manifest.json",
    ])
    assert args.pair_source == "real_perturbed_render"
    assert args.query_pose_file == "poses_train.txt"
    assert args.validation_manifest == "val_manifest.json"
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/test_matcha_streaming_manifest.py::test_build_streaming_manifest_accepts_real_pair_source -q`

Expected: argparse rejects `--pair_source`.

- [ ] **Step 3: Implement minimal CLI flags and metadata call**

Use `build_render_query_metadata(...)` when `--pair_source` is non-empty. Load validation manifest only to compute query IDs and overlap counts. Existing synthetic/legacy manifests without pair source must keep working.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=. pytest tests/test_matcha_streaming_manifest.py -q`

Expected: all streaming manifest tests pass.

## Task 3: Pose Initializer Coverage

**Files:**
- Modify: `feature_extract/vfm/render_pose_protocol.py`
- Modify: `feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`
- Test: `tests/test_render_pose_protocol.py`
- Test: `tests/test_render_rgb_pose_update_protocol.py`

- [ ] **Step 1: Write failing pose-selection tests**

```python
def test_select_render_pose_supports_rotation_offset() -> None:
    gt_pose = np.eye(4, dtype=np.float64)
    selected = select_render_pose(
        "q.png",
        gt_pose,
        mode="gt_rotation_offset",
        rotation_offset_deg=np.asarray([0.0, 3.0, 0.0], dtype=np.float64),
    )
    assert selected.label == "gt_rotation_offset:0.000,3.000,0.000"
    assert selected.render_translation_error_m == pytest.approx(0.0)
    assert selected.render_rotation_error_deg == pytest.approx(3.0)
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/test_render_pose_protocol.py::test_select_render_pose_supports_rotation_offset -q`

Expected: `select_render_pose` does not accept `rotation_offset_deg`.

- [ ] **Step 3: Implement rotation offsets and top10 parser exposure**

Add `rotate_pose_camera_frame(...)` or equivalent fixed-axis helper. Extend eval parser choices to include `gt_rotation_offset` and `reference_top10`; set `reference_top_k=10` when that mode is used.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=. pytest tests/test_render_pose_protocol.py tests/test_render_rgb_pose_update_protocol.py -q`

Expected: all pose protocol tests pass.

## Task 4: Full-Test Evaluation Matrix Runner

**Files:**
- Create: `feature_extract/tools/vfm/run_matcha_real_render_query_eval_matrix.py`
- Test: `tests/test_matcha_real_render_query_eval_matrix.py`

- [ ] **Step 1: Write failing matrix tests**

```python
def test_eval_matrix_dry_run_contains_required_oldhospital_modes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main([
        "--query_manifest", "test_manifest.json",
        "--query_pose_file", "dataset_test.txt",
        "--image_root", "OldHospital",
        "--gaussian_rgb_ply", "point_cloud.ply",
        "--output_root", str(tmp_path),
        "--matcha_joint_checkpoint", "model.pt",
        "--candidate_bank", "candidates.jsonl",
        "--dry_run",
    ])
    out = capsys.readouterr().out
    assert "--render_pose_mode gt" in out
    assert "--render_pose_mode gt_offset --render_pose_world_offset 0.250,0.000,0.000" in out
    assert "--render_pose_mode gt_rotation_offset --render_pose_rotation_offset_deg 0.000,6.000,0.000" in out
    assert "--render_pose_mode reference_top10" in out
    assert "--max_queries 0" in out
```

- [ ] **Step 2: Verify RED**

Run: `PYTHONPATH=. pytest tests/test_matcha_real_render_query_eval_matrix.py -q`

Expected: import failure for the new matrix runner.

- [ ] **Step 3: Implement dry-run and optional execution**

Generate one command per required initializer: GT, four translation offsets, four rotation offsets, reference top1/top5/top10. Default `--max_queries` must be `0` so full test split is evaluated unless the caller explicitly overrides it.

- [ ] **Step 4: Verify GREEN**

Run: `PYTHONPATH=. pytest tests/test_matcha_real_render_query_eval_matrix.py tests/test_render_rgb_pose_update_protocol.py -q`

Expected: all matrix-runner and parser tests pass.

## Task 5: Batch Verification

**Files:**
- Existing tests only.

- [ ] **Step 1: Run focused regression tests**

Run:

```bash
PYTHONPATH=. pytest \
  tests/test_matcha_render_query_protocol.py \
  tests/test_matcha_streaming_manifest.py \
  tests/test_render_pose_protocol.py \
  tests/test_render_rgb_pose_update_protocol.py \
  tests/test_matcha_real_render_query_eval_matrix.py \
  -q
```

Expected: all tests pass.

- [ ] **Step 2: Run syntax and whitespace checks**

Run:

```bash
python -m compileall feature_extract/vfm feature_extract/tools/vfm tests -q
git diff --check
```

Expected: both commands exit 0.

## Self-Review

- Spec coverage: this plan covers protocol hygiene and full-test matrix tooling. It intentionally does not implement the next training-scale jobs, no-match target generation, or fine-head redesign; those need a second plan after this protocol layer is merged.
- Placeholder scan: no `TBD`, `TODO`, or undefined placeholder task remains.
- Type consistency: all new public names are introduced in Task 1 or Task 4 before being used by tests.
