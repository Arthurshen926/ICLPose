# DCFF Localization Handoff - 2026-05-01

This document summarizes the current session for the next agent. The user wants
DCFF-only localization/refinement, with emphasis on localization-driven DCFF
feature reconstruction/matching. Do not use rendered RGB + LoFTR as the main
route.

## Current Goal

- Target: local centimeter-level refine using existing DCFF feature/depth.
- First benchmark target: start from local perturbations, eventually around
  `1 deg / 5 cm`, and push median translation below centimeter level.
- The user explicitly wants a first-principles review, not just hyperparameter
  tuning.
- Environment: use `conda run -n iclpose ...`.
- GPU request: user asked for GPU1, but in this session GPU1 was unusable:
  `Unable to determine the device handle for GPU1: 0000:68:00.0: Unknown Error`.
  All actual runs used visible physical GPU0 via `CUDA_VISIBLE_DEVICES=0`.

## Main Technical Conclusion

The current translation bottleneck is not primarily the depth/WLS backend.
The backend can reach centimeter-level when correspondence is correct.

The main bottleneck is real query-to-DCFF-map matching:

- Self-rendered DCFF map-to-map features form a strong local signal.
- Real query features still do not produce a stable local correlation peak.
- Query-only descriptor training can reduce soft-flow EPE to about `1.3 px`
  and WLS translation to roughly `17-25 mm`, but `peak_acc` remains near random.
- Low-resolution features and underused depth matter, but the deeper problem is
  that the feature/matcher pair is not yet trained to expose a depth-aware,
  metric, local correspondence signal.

Implication: continuing to tune descriptor-only local correlation losses is
unlikely to reach centimeter accuracy. The next useful step should be an
explicit depth-aware matcher / flow-confidence localizer trained with WLS/GN
pose loss.

## Architecture Understanding

High-level localization stack:

- `feature_retrieval`: retrieval/init/candidate generation and direct pose
  regression experiments.
- `feature_extract`: query student training, map/query supervision, current
  localization-driven feature training scaffolding.
- `feature_field` / `feature_gaussian`: DCFF / 2DGS feature reconstruction and
  rendering.
- `pose_refine`: dense correspondence, depth/Jacobian WLS, ConcatPoseNet,
  feature-metric and render-compare evaluation paths.

Coarse/fine interpretation to preserve:

- Coarse should mean globally discriminative / basin-expanding / retrieval or
  candidate gating. It should not be directly hard-bound to rotation only.
- Fine should mean locally correspondable and geometry-solvable. Translation
  precision depends on parallax, depth scale, and local correspondence, so fine
  features must be used together with rendered depth/Jacobian/WLS/GN.
- Coarse/fine is not a direct rotation/translation split. It affects rotation
  and translation indirectly through spatial scale and geometric observability.

Feature retrieval direct pose regression:

- This already exists historically.
- Relevant files:
  - `feature_retrieval/pose_regressor.py`
  - `feature_retrieval/pose_regressor_v*.py`
  - `feature_retrieval/patch_regressor_v7.py`
  - `feature_retrieval/build_learned_init_poses.py`
- This is initialization/candidate behavior, not the route expected to solve
  centimeter-level local refine.

## Important Prior Evidence

Diagnostics from earlier in the session:

- Self-render DCFF sensitivity:
  - 1/2/5/10 cm ranking accuracy was 100%.
  - Corr-WLS reached about `0.040 deg / 8.1 mm`.
  - Flow EPE around `0.908 px`.
  - This proves DCFF map + depth + WLS has a centimeter-level upper bound when
    the query side is also rendered/map-like.
- Real dataset query diagnostics:
  - Hard-negative gap was near zero or negative.
  - Ranking: 1/2 cm 0%, 5 cm about 31%, 10 cm about 50%.
  - Corr-WLS about `0.736 deg / 99.4 mm`, flow EPE about `4.55 px`.
- Dataset query with trained projection:
  - Corr-WLS about `0.611 deg / 95.5 mm`, flow EPE about `4.31 px`.
  - Projection alone does not fix query-map matching.
- Existing pose-refine corr-WLS probe:
  - From `1 deg / 5 cm` synthetic init, median init translation was about
    `76.3 mm`.
  - Learned refine worsened to about `111.9 mm`.
  - Flow/corr EPE stayed too high.

## Code Changes Made In This Session Family

Main files touched or extended:

- `feature_extract/students/radio_query_student.py`
  - Added optional `fine_loc` localization head/high-res branch.
  - Added optional scene-coordinate head variants.
  - Supports high-resolution query localization output.
- `feature_extract/train_impl.py`
  - Added localization-driven map/query losses.
  - Added scene-coordinate supervision helpers.
  - Added rendered map world-position outputs.
  - Added depth/translation observability weighting.
  - Added correlation subpixel / soft-flow / peak / WLS pose losses.
  - Added feature-metric pose loss path.
  - Fixed high-res metric logging memory issue by moving cosine metric logging
    under `torch.no_grad()`.
  - Added `local_correlation_joint_losses`, which computes subpixel, soft-flow,
    peak, and WLS-pose loss from a single local correlation volume.
  - Rewired query/self correlation branches to share the correlation volume.
    This reduced memory/time enough to run 544x960 high-res v28 smoke.
- `pose_refine/tools/diag_dcff_cm_sensitivity.py`
  - Fixed projection diagnostic bug:
    `proj_mode == "shared_linear"` should use `model.proj_shared`.
- `feature_extract/export_impl.py`, `feature_extract/evaluate_impl.py`
  - Threaded new model args.
- `tests/test_adaptive_joint_query_map.py`
  - Added coverage for scene coord head/losses and correlation/WLS paths.
  - Added equivalence test for `local_correlation_joint_losses`.
- `tests/test_pose_refine_geometry.py`
  - Existing geometry tests were extended in earlier work.

Current visible `git status --short` at handoff showed:

```text
 M feature_extract/train_impl.py
 M feature_field/visualize_feature_comparison.py
 M tests/test_adaptive_joint_query_map.py
 M tests/test_feature_visualization_pca.py
```

Note: some config files may be ignored/untracked depending on repo ignore rules.
Inspect the workspace before editing and do not revert user/other-agent changes.

## New / Relevant Configs

Recent configs created:

- `feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v27_locdriven_highres_stage1.yaml`
- `feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v28_locdriven_544_peak_stage1.yaml`
- `feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v29_locdriven_jointmap_peak.yaml`
- `feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v30_locdriven_largepert_peak.yaml`

Earlier configs in this line:

- `v24_scenecoord_corrwls`
- `v25_scenecoord_gridctx_corrwls`
- `v26_scenecoord_bootstrap`

## Recent Experiments In Detail

All runs below used:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.6+PTX \
PYTHONPATH=/root/ICLPose \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n iclpose python -u -m feature_extract.train ...
```

Warmstart used:

```text
/root/ICLPose/result/feature_extract/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v17_selfmap_s2highres_zero_fm/checkpoints/best.pth
```

### v27 - 408x720 Query-Local Stage

Config:

```text
feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v27_locdriven_highres_stage1.yaml
```

Intent:

- Freeze map.
- Train query high-res `fine_loc` against DCFF rendered features.
- Use query corr subpixel, soft-flow, peak, WLS pose, flow-warp, and self-map
  diagnostics/losses.

Result summary:

- Stopped before full epoch because trend was already clear.
- Soft-flow improved, but hard peak stayed random.
- WLS hovered around `23-34 mm`.

Parsed train statistics:

```text
v27 n=53 last_step=530
001-050: epe=2.349 peak_gap=-0.0287 peak_acc=0.0052 wls=33.7mm
051-100: epe=1.795 peak_gap=-0.0182 peak_acc=0.0042 wls=29.8mm
101-150: epe=1.485 peak_gap=-0.0146 peak_acc=0.0046 wls=26.4mm
151-200: epe=1.426 peak_gap=-0.0128 peak_acc=0.0048 wls=25.4mm
201-250: epe=1.323 peak_gap=-0.0127 peak_acc=0.0042 wls=23.5mm
301-350: epe=1.787 peak_gap=-0.0135 peak_acc=0.0046 wls=27.8mm
401-450: epe=1.443 peak_gap=-0.0130 peak_acc=0.0048 wls=28.9mm
501-550: epe=1.289 peak_gap=-0.0132 peak_acc=0.0047 wls=23.9mm
```

Interpretation:

- Learned a broad/soft displacement response.
- Did not learn a discrete local matching peak.
- Not enough for centimeter-level refine.

### v28 - 544x960 Peak-Focused Query-Only Stage

Config:

```text
feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v28_locdriven_544_peak_stage1.yaml
```

Intent:

- Use higher feature resolution to increase translation sensitivity.
- Freeze map.
- Reduce broad soft-flow incentives, strengthen peak/subpixel objectives.
- Uses the new shared-correlation implementation.

Smoke:

- Passed at 544x960, no OOM.
- GPU memory around `16.5 GB`.

Result summary:

- Best short-run behavior so far for WLS: around `17-18 mm`.
- Still did not form hard correlation peak.

Parsed train statistics:

```text
v28 n=23 last_step=230
001-050: epe=2.121 peak_gap=-0.0188 peak_acc=0.0028 wls=21.4mm
051-100: epe=1.472 peak_gap=-0.0086 peak_acc=0.0024 wls=17.0mm
101-150: epe=1.325 peak_gap=-0.0075 peak_acc=0.0026 wls=17.9mm
151-200: epe=1.289 peak_gap=-0.0072 peak_acc=0.0026 wls=17.3mm
201-250: epe=1.367 peak_gap=-0.0071 peak_acc=0.0030 wls=18.2mm
```

Interpretation:

- Higher resolution helps metric WLS but still leaves `peak_acc` near random.
- This supports the user's suspicion that feature size matters, but it is not
  sufficient by itself.

### v29 - Joint Map Feature Fine-Tuning

Config:

```text
feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v29_locdriven_jointmap_peak.yaml
```

Intent:

- Train localization head and low-LR DCFF map reconstruction side together.
- `trainable: true`
- `train_fine_decoder: true`
- `train_feat_sharp: true`
- `train_fsm: true`
- Keep geometry/latent/hash frozen.
- Add teacher/map anchors to avoid degeneracy.

Smoke:

- Passed at 408x720.
- GPU memory around `19.8 GB`.

Result summary:

- Did not improve over v28.
- WLS was worse than v28.
- Peak still random.

Parsed train statistics:

```text
v29 n=18 last_step=180
001-050: epe=2.239 peak_gap=-0.0222 peak_acc=0.0042 wls=28.8mm
051-100: epe=1.538 peak_gap=-0.0118 peak_acc=0.0046 wls=22.7mm
101-150: epe=1.487 peak_gap=-0.0111 peak_acc=0.0048 wls=21.0mm
151-200: epe=1.495 peak_gap=-0.0108 peak_acc=0.0053 wls=23.1mm
```

Interpretation:

- Simply letting localization loss update DCFF decoder/FSM does not solve
  query-map matching.
- It likely needs a better localization consumer/matcher objective before map
  fine-tuning helps.

### v30 - Larger-Perturbation Peak Curriculum

Config:

```text
feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v30_locdriven_largepert_peak.yaml
```

Intent:

- Move from tiny 0.5/1/2 cm perturbations to larger, more observable
  3/5/8 cm perturbations.
- Increase radius to 12.
- Hypothesis: tiny perturbations create subpixel shifts that make hard-peak
  supervision poorly observable early in training.

Smoke:

- Passed.
- Initial smoke had large EPE, as expected for larger perturbations.

Short-run result:

- Did not improve peak formation.
- WLS stayed around `54-60 mm`.

Parsed train statistics:

```text
v30 n=15 last_step=150
001-050: epe=4.001 peak_gap=-0.0270 peak_acc=0.0034 wls=60.3mm
051-100: epe=3.244 peak_gap=-0.0151 peak_acc=0.0034 wls=55.2mm
101-150: epe=3.245 peak_gap=-0.0141 peak_acc=0.0034 wls=54.1mm
```

Interpretation:

- Larger perturbation alone also does not rescue descriptor-only correlation.
- Stop treating this as a weighting/course-tuning problem.

## Verification Already Run

Compile:

```bash
conda run -n iclpose python -m py_compile \
  feature_extract/train_impl.py \
  tests/test_adaptive_joint_query_map.py \
  pose_refine/tools/diag_dcff_cm_sensitivity.py
```

Selected tests:

```bash
PYTHONPATH=/root/ICLPose conda run -n iclpose python -c "
import importlib.util, pathlib
spec=importlib.util.spec_from_file_location('t', pathlib.Path('tests/test_adaptive_joint_query_map.py').resolve())
mod=importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
names=[
 'test_joint_local_correlation_losses_match_individual_losses',
 'test_map_supervision_applies_depth_aware_local_correlation_loss',
 'test_map_supervision_can_use_fine_loc_for_localization_losses',
 'test_local_correlation_peak_margin_loss_is_low_for_correct_peak',
 'test_local_correlation_wls_pose_loss_empty_mask_is_finite',
 'test_compute_w2c_flow_is_zero_for_identical_poses',
]
[getattr(mod,n)() for n in names]
print('selected tests passed')
"
```

Both passed. Warnings about deprecated `torch.cuda.amp.autocast` are known and
not blockers.

No `feature_extract.train` process was running at the time this handoff was
written.

## Commands Useful For Next Agent

Check current training processes:

```bash
ps -eo pid,ppid,stat,etime,cmd | grep -F 'feature_extract.train' | grep -v grep || true
```

Check GPU:

```bash
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
```

Run v28 smoke:

```bash
CUDA_VISIBLE_DEVICES=0 TORCH_CUDA_ARCH_LIST=8.6+PTX \
PYTHONPATH=/root/ICLPose PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n iclpose python -u -m feature_extract.train \
  --config feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v28_locdriven_544_peak_stage1.yaml \
  --warmstart /root/ICLPose/result/feature_extract/joint_radio_dcff_cambridge_oldhospital_processed_pca64_depthaware_v17_selfmap_s2highres_zero_fm/checkpoints/best.pth \
  --smoke-test
```

Run DCFF sensitivity diagnostics:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/ICLPose \
conda run -n iclpose python -u pose_refine/tools/diag_dcff_cm_sensitivity.py \
  --config pose_refine/configs/concat_loc_cambridge_oldhospital_processed_locguided_dcff_highres_depthaware_v17_selfmap_export_corrwls_probe.yaml \
  --gpu 0 --split test --batch_size 4 --max_batches 4 \
  --query_source self_render \
  --dist_cm 1 2 5 10 \
  --no_feature_metric \
  --corr_radius 8 --corr_temperature 0.04
```

For real query:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/root/ICLPose \
conda run -n iclpose python -u pose_refine/tools/diag_dcff_cm_sensitivity.py \
  --config pose_refine/configs/concat_loc_cambridge_oldhospital_processed_locguided_dcff_highres_depthaware_v17_selfmap_export_corrwls_probe.yaml \
  --gpu 0 --split test --batch_size 4 --max_batches 4 \
  --query_source dataset \
  --dist_cm 1 2 5 10 \
  --no_feature_metric \
  --corr_radius 8 --corr_temperature 0.04
```

## Recommended Next Direction

Do not continue mainly with descriptor-only `rendered_feat dot query_feat` local
correlation plus stronger margins. v27-v30 show that this path can improve soft
EPE but does not produce the hard local matching peak needed for centimeter
translation.

Recommended next implementation:

1. Add an explicit depth-aware local matcher/refiner.
   - Inputs: rendered DCFF fine feature, query fine feature, rendered depth,
     alpha/prior mask, translation-observability map, maybe coarse/context.
   - Output: local flow distribution or direct flow + confidence.
   - Consumer: `diff_pose_solve` / depth-WLS must be the main translation path.
   - Losses: subpixel flow, confidence calibration, WLS pose loss, and
     perturb-ranking. Keep map/query teacher anchors.

2. Use depth and observability explicitly in the matcher, not just as a mask.
   - Concatenate depth, inverse depth, normalized pixel coordinates, and
     translational Jacobian/observability channels into the matching head.
   - This aligns with the user's first-principles argument: translation is
     observable through parallax and depth, not through semantic feature
     similarity alone.

3. Stage training:
   - Stage 1: freeze map/DCFF, train matcher/query head from 3-5 cm perturbations
     until WLS update direction is consistently correct.
   - Stage 2: introduce 0.5/1/2 cm perturbations and sharpen confidence.
   - Stage 3: low-LR fine-tune DCFF decoder/FeatSharp/FSM with strong teacher
     anchor.

4. Keep reporting these metrics:
   - `corr_flow_epe`
   - hard/argmax `peak_acc`
   - `peak_gap`
   - WLS translation median/mm
   - WLS gain
   - confidence mean/calibration
   - invalid depth / empty mask NaN checks

5. Treat feature-metric GN as a diagnostic and possible auxiliary loss, but not
   as sufficient unless real query residuals show correct update direction.

## Practical Cautions

- Use `apply_patch` for edits.
- Do not revert unrelated dirty files.
- GPU1 is not actually usable on this machine right now; use physical GPU0
  unless the machine state changes.
- Several earlier configs/runs are research probes. The strongest current
  evidence is the self-render upper bound vs real-query failure.
- If a future run reports soft WLS around `15-20 mm` but `peak_acc` remains
  random, that is not centimeter readiness. It means the solver is using a broad
  soft expectation, not reliable correspondence.

