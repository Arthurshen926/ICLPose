# Render-Perturbation IF Lines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run parallel, non-invasive IF-line experiments to improve local render-conditioned VFM localization under 5-25cm render pose perturbation.

**Architecture:** Keep the current MATCHA-style render RGB/depth pipeline unchanged as the control path. Add isolated experimental flags/tools for render-side search, pose residual heads, wider-context render sampling, and geometry-aware supervision. Every IF line writes to its own `output/vfm/stage_r_matcha_joint/oldhospital/if_*` directory and must be compared against the frozen conservative head-only baseline.

**Tech Stack:** Python, PyTorch, NumPy, OpenCV PnP, official 2DGS renderer, RADIO dual features, existing `feature_extract/vfm/*` MATCHA-style modules.

---

## Common Baseline And Gate

**Frozen baseline checkpoint:**
`/root/ICLPose/output/vfm/stage_r_matcha_joint/oldhospital/streaming_full_v1/train_full_conservative25_headonly_120/best_joint.pt`

**Control eval commands:**
- GT full182:
  `/root/ICLPose/output/vfm/stage_r_matcha_joint/oldhospital/streaming_full_v1/eval_train_full_conservative25_headonly_120_gt_full182_confpnp_diag_1280/summary.json`
- +25cm full182:
  `/root/ICLPose/output/vfm/stage_r_matcha_joint/oldhospital/streaming_full_v1/eval_train_full_conservative25_headonly_120_gt_offset025_full182_confpnp_diag_1280/summary.json`

**Known control metrics:**
- GT full182: median `0.0187m`, S@10 `1.0`, S@25 `1.0`
- +25cm full182: median `0.2432m`, S@10 `0.011`, S@25 `0.731`
- +25cm fine offset: `12.74px -> 12.71px`, improved ratio `0.419`
- +25cm confidence inlier/outlier gap: `0.0029`

**Promotion gate for any IF line:**
- q32 +25cm: median t improves by at least `3cm` or S@25 improves by at least `10% absolute`.
- full182 +25cm after q32 pass: median t <= `0.20m` or S@25 >= `0.85`.
- GT render must not regress below S@10 `0.98`.
- PnP-inlier GT@16 must not decrease by more than `2% absolute`.

---

### Task 1: IF-A Render-Side Local Offset Search

**Hypothesis:** Query-side fine offset is not the right correction for render-conditioned depth. Render-side offset changes the sampled depth and therefore the 3D point, so it may recover true metric correspondence under perturbation.

**Files:**
- Modify: `/root/ICLPose/feature_extract/vfm/matcha_coarse_to_fine.py`
- Modify: `/root/ICLPose/feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`
- Test: `/root/ICLPose/tests/test_matcha_coarse_to_fine.py`

- [ ] **Step 1: Add render-side local topK offset API**

Add a new function in `matcha_coarse_to_fine.py`:

```python
def expand_matches_with_render_local_offsets(
    matches: Sequence[KeypointFeatureMatch],
    *,
    render_image_width: int,
    render_image_height: int,
    render_grid_width: int,
    render_grid_height: int,
    cell_radius: int = 1,
    offset_bins: int = 8,
    max_candidates_per_match: int = 9,
) -> list[KeypointFeatureMatch]:
    """Expand each coarse match into neighboring render-cell/subcell candidates.

    Query xy stays fixed. Render xy moves within a local cell window. Similarity
    and confidence are copied; later scoring chooses candidates using depth,
    confidence, and PnP.
    """
```

- [ ] **Step 2: Write unit test for render candidate expansion**

Add to `tests/test_matcha_coarse_to_fine.py`:

```python
def test_expand_matches_with_render_local_offsets_keeps_query_fixed_and_moves_render():
    match = KeypointFeatureMatch(
        query_index=5,
        render_index=5,
        query_xy=np.asarray([20.0, 20.0]),
        render_xy=np.asarray([20.0, 20.0]),
        similarity=0.7,
        ratio=0.0,
        dual_softmax_confidence=0.8,
    )
    expanded = expand_matches_with_render_local_offsets(
        [match],
        render_image_width=64,
        render_image_height=64,
        render_grid_width=4,
        render_grid_height=4,
        cell_radius=1,
        max_candidates_per_match=9,
    )
    assert len(expanded) == 9
    assert all(np.allclose(item.query_xy, match.query_xy) for item in expanded)
    assert any(not np.allclose(item.render_xy, match.render_xy) for item in expanded)
```

- [ ] **Step 3: Add eval flags**

In `eval_render_rgb_feature_keypoint_pose.py` add:

```python
parser.add_argument("--render_side_local_offset_radius_cells", type=int, default=0)
parser.add_argument("--render_side_local_offset_max_candidates", type=int, default=9)
```

When radius > 0, call `expand_matches_with_render_local_offsets` after coarse matching and before `keypoint_feature_matches_to_pnp_matches`.

- [ ] **Step 4: Run q32 +25cm smoke**

Run:

```bash
PYTHONPATH=. python feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py \
  --query_manifest output/vfm/stage_r_matcha_joint/oldhospital/clean_protocol_v1/split_manifests/test_full182_manifest.json \
  --query_pose_file /hy-tmp/Cambridge_stdloc/OldHospital/dataset_test.txt \
  --image_root /hy-tmp/Cambridge_stdloc/OldHospital \
  --gaussian_rgb_ply /root/ICLPose/result/result/feature_gaussian/joint_rgb_geometry_cambridge_oldhospital_processed_rebuild_v4_4gpu_1280_safe_rgbft_30k/point_cloud/best/point_cloud.ply \
  --output_dir output/vfm/stage_r_matcha_joint/oldhospital/if_a_render_side_search/q32_offset025_r1 \
  --feature_mode radio_dual \
  --extract_query_features_from_image \
  --matcha_joint_checkpoint output/vfm/stage_r_matcha_joint/oldhospital/streaming_full_v1/train_full_conservative25_headonly_120/best_joint.pt \
  --matcha_eval_preset conservative_confidence \
  --matcha_use_pair_fine_head \
  --matcha_pair_fine_side query \
  --matcha_cell_offset_side both \
  --render_side_local_offset_radius_cells 1 \
  --render_side_local_offset_max_candidates 9 \
  --render_pose_mode gt_offset \
  --render_pose_world_offset 0.25,0,0 \
  --render_width 1280 \
  --render_height 720 \
  --renderer official_2dgs \
  --match_mode matcha_c2f \
  --max_matches 1000 \
  --coverage_filter_grid 8 \
  --coverage_filter_max_per_cell 8 \
  --coverage_filter_max_total 512 \
  --max_queries 32 \
  --device cuda:0
```

Expected: summary written; compare median/S@25 to q32 control.

---

### Task 2: IF-B Render-Side Learned Fine Head

**Hypothesis:** The current checkpoint was optimized for query-side pair fine. A render-side head must be explicitly trained because render-side offset affects depth and 3D point selection.

**Files:**
- Modify: `/root/ICLPose/feature_extract/tools/vfm/train_matcha_joint_streaming_model.py`
- Modify: `/root/ICLPose/feature_extract/vfm/matcha_joint_training.py`
- Test: `/root/ICLPose/tests/test_matcha_joint_training.py`

- [ ] **Step 1: Add explicit render-side fine loss switch**

Expose:

```python
parser.add_argument("--render_pair_fine_loss_weight", type=float, default=0.0)
parser.add_argument("--query_pair_fine_loss_weight", type=float, default=1.0)
```

Ensure `render_pair_fine_loss_weight > 0` trains the render-side pair fine head on render offset soft labels.

- [ ] **Step 2: Add test that render loss is included**

In `test_matcha_joint_training.py`, construct a toy training set with `render_offset_soft_labels` and assert loss dict includes nonzero `render_pair_fine_loss`.

- [ ] **Step 3: Train a short render-head-only model**

Use frozen descriptor and curriculum with more 25cm exposure:

```bash
PYTHONPATH=. python feature_extract/tools/vfm/train_matcha_joint_streaming_model.py \
  --streaming_manifest output/vfm/stage_r_matcha_joint/oldhospital/streaming_full_v1/train_full_conservative25_streaming_manifest.json \
  --query_pose_file /hy-tmp/Cambridge_stdloc/OldHospital/dataset_train.txt \
  --image_root /hy-tmp/Cambridge_stdloc/OldHospital \
  --gaussian_rgb_ply /root/ICLPose/result/result/feature_gaussian/joint_rgb_geometry_cambridge_oldhospital_processed_rebuild_v4_4gpu_1280_safe_rgbft_30k/point_cloud/best/point_cloud.ply \
  --output_model output/vfm/stage_r_matcha_joint/oldhospital/if_b_render_fine/train_render_head_240/adapter.pt \
  --output_joint_model output/vfm/stage_r_matcha_joint/oldhospital/if_b_render_fine/train_render_head_240/model_joint.pt \
  --output_best_joint_model output/vfm/stage_r_matcha_joint/oldhospital/if_b_render_fine/train_render_head_240/best_joint.pt \
  --summary_json output/vfm/stage_r_matcha_joint/oldhospital/if_b_render_fine/train_render_head_240/summary.json \
  --warm_start_joint_checkpoint output/vfm/stage_r_matcha_joint/oldhospital/streaming_full_v1/train_full_agt_streaming_300/best_joint.pt \
  --query_feature_cache_dir output/vfm/stage_r_matcha_joint/oldhospital/streaming_full_v1/query_radio_dual_f16_train \
  --steps 240 \
  --batch_size 1024 \
  --freeze_descriptor_steps 240 \
  --descriptor_lr_scale 0.0 \
  --head_lr_scale 1.0 \
  --query_pair_fine_loss_weight 0.25 \
  --render_pair_fine_loss_weight 1.0 \
  --pair_type_curriculum uniform \
  --collect_visibility_no_match \
  --device cuda:0
```

- [ ] **Step 4: Evaluate render-side fine**

Run q32 +25cm with:

```bash
--matcha_joint_checkpoint output/vfm/stage_r_matcha_joint/oldhospital/if_b_render_fine/train_render_head_240/best_joint.pt \
--matcha_use_pair_fine_head \
--matcha_pair_fine_side render
```

Expected positive signal: render-side q32 S@25 > current render-side direct switch `0.094`.

---

### Task 3: IF-C Geometry-Constrained Residual Pose Head Diagnostic

**Hypothesis:** Current fine heads are trained independently per match. A pose residual module should learn residuals/confidences that jointly support one global `T_query = exp(Delta) * T_render`.

**Files:**
- Create: `/root/ICLPose/feature_extract/vfm/render_pose_residual_solver.py`
- Create: `/root/ICLPose/feature_extract/tools/vfm/eval_render_pose_residual_oracle.py`
- Test: `/root/ICLPose/tests/test_render_pose_residual_solver.py`

- [ ] **Step 1: Implement non-learned weighted SE(3) residual solver**

Create `render_pose_residual_solver.py` with:

```python
def solve_render_pose_delta_from_matches(
    matches: Sequence[QueryTo3DMatch],
    render_pose_w2c: np.ndarray,
    camera: ColmapCamera,
    *,
    max_iterations: int = 10,
    damping: float = 1e-4,
) -> np.ndarray:
    """Solve a small SE(3) update initialized at render_pose_w2c.

    The returned pose is global query pose, not a local-frame pose.
    """
```

Use Gauss-Newton reprojection residuals with numerical Jacobian first; this diagnostic prioritizes correctness over speed.

- [ ] **Step 2: Add synthetic coordinate test**

Test that correct correspondences with render pose offset 0.25m recover GT within `1e-4m`.

- [ ] **Step 3: Oracle residual diagnostic**

Create `eval_render_pose_residual_oracle.py` that consumes rows/matches from a single eval run, optionally filters GT-correct matches using GT pose, and compares:

- OpenCV PnP from all matches
- OpenCV PnP from GT-correct matches
- weighted residual solver from all matches
- weighted residual solver from GT-correct matches

Expected: GT-correct matches should pull +25cm close to GT. If not, geometry/depth is wrong.

---

### Task 4: IF-D Wider-FOV / Render Canvas Diagnostic

**Hypothesis:** Bigger same-FOV resolution does not increase capture range, but a wider-FOV render canvas may preserve candidates near borders and support larger render-side search.

**Files:**
- Modify: `/root/ICLPose/feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py`
- Create: `/root/ICLPose/tests/test_render_canvas_intrinsics.py`

- [ ] **Step 1: Add render canvas scale flag**

Add:

```python
parser.add_argument("--render_canvas_scale", type=float, default=1.0)
```

When `>1`, render with larger width/height and adjusted intrinsics preserving focal length and principal point offset into the larger canvas.

- [ ] **Step 2: Test intrinsics**

If original camera is 1280x720 and scale=1.25, output canvas is 1600x900, focal length remains scaled as before, principal point shifts by `(160, 90)`.

- [ ] **Step 3: Run diagnostic only**

Evaluate q32 +25cm with canvas scale `1.25` and `1.5`, no training changes.

Expected: If S@25 improves only with render-side search, then border/candidate support matters. If not, main issue is descriptor/measurement.

---

### Task 5: IF-E Pose-Offset Regression Head Smoke

**Hypothesis:** A small pose head can predict `Delta SE3` from aggregated match residual evidence, but it risks scene-prior overfitting. Treat it as diagnostic, not main method.

**Files:**
- Create: `/root/ICLPose/feature_extract/vfm/render_pose_offset_head.py`
- Create: `/root/ICLPose/feature_extract/tools/vfm/train_render_pose_offset_head.py`
- Test: `/root/ICLPose/tests/test_render_pose_offset_head.py`

- [ ] **Step 1: Build match-level feature table**

For each query-render pair store:

```text
query_id, pair_type, render_pose_error, match_count,
mean_similarity, mean_confidence, pnp_inlier_count,
mean_flow_proxy, spatial_coverage, depth_median,
gt_delta_translation, gt_delta_rotation
```

- [ ] **Step 2: Train small MLP**

Input aggregated features; output `6D delta`. Loss:

```text
L = ||translation_delta||_1 + 0.1 * ||rotation_delta_deg||_1
```

- [ ] **Step 3: Evaluate only on held-out test**

Report:

- render pose error
- regressed pose error
- PnP pose error
- failure cases where regression worsens

Expected: If MLP works only on q32 train-like samples but fails full182, reject as pose prior.

---

### Task 6: IF-F Match Confidence Re-Targeting

**Hypothesis:** Current confidence target is too close to patch correctness, not "usable for pose." Re-target confidence to pose-usefulness.

**Files:**
- Modify: `/root/ICLPose/feature_extract/vfm/matcha_coarse_supervision.py`
- Modify: `/root/ICLPose/feature_extract/vfm/matcha_joint_training.py`
- Test: `/root/ICLPose/tests/test_matcha_coarse_supervision.py`

- [ ] **Step 1: Add pose-usable confidence labels**

Define confidence label:

```text
1 if GT reprojection error <= 8px and alpha valid and depth-edge safe
0 if GT reprojection error > 24px or visibility invalid
ignore otherwise
```

- [ ] **Step 2: Add ignored confidence mask**

Training must skip ambiguous 8-24px cases in BCE.

- [ ] **Step 3: Train head-only q25 model**

Same as conservative head-only, but confidence target is pose-usable.

- [ ] **Step 4: Evaluate learned-confidence PnP**

Success criterion: +25cm confidence inlier/outlier gap > `0.03` and S@25 improves.

---

## Parallel Execution Layout

Run these in parallel because they mostly do not touch shared state:

- Worker A: IF-A render-side local search
- Worker B: IF-D wider-FOV/canvas diagnostic
- Worker C: IF-C residual solver diagnostic
- Worker D: IF-F confidence retargeting

Run after IF-A/IF-C evidence:

- Worker E: IF-B render-side learned fine head
- Worker F: IF-E pose-offset regression smoke

Each worker must write:

```text
output/vfm/stage_r_matcha_joint/oldhospital/if_<letter>_<name>/
  summary.json
  rows.csv
  visualizations/
  conclusion.md
```

`conclusion.md` must include:

```markdown
# IF-X Conclusion

Control:
- q32 +25cm median:
- q32 +25cm S@25:

Experiment:
- q32 +25cm median:
- q32 +25cm S@25:
- PnP-inlier GT@16:
- Fine offset before/after:

Decision:
- promote / reject / needs full182

Reason:
- one paragraph
```

---

## Recommended Order

1. IF-A render-side local offset search: cheapest and directly tests the user's hypothesis.
2. IF-C residual solver oracle: determines whether geometry can pull pose back when correspondences are correct.
3. IF-F confidence retargeting: addresses current confidence gap `~0.003`.
4. IF-D wider-FOV render canvas: useful only if border/candidate support matters.
5. IF-B render-side learned fine head: train only after IF-A shows render-side search has signal.
6. IF-E pose-offset regression: diagnostic only; do not make it the method unless it generalizes.

---

## Self-Review

Spec coverage:
- Render perturbation health check is already implemented in `diagnose_render_pose_perturbation.py`; this plan builds follow-up IF lines.
- Render-side offset is covered by IF-A and IF-B.
- Larger/wider render is covered by IF-D.
- Geometry-constrained residual pose head is covered by IF-C.
- Direct pose offset head is covered by IF-E as a risky diagnostic.
- Confidence/no-match retargeting is covered by IF-F.
- All IF lines are isolated from current mainline via new flags/tools and separate output directories.

Known risk:
- IF-D requires careful intrinsics handling; do not trust pose metrics until `test_render_canvas_intrinsics.py` passes.
- IF-E can overfit scene priors; it must never be promoted without full182 and cross-scene diagnostics.
