# Geometry Mainline Matrix

Date: 2026-04-23
Scene: OldHospital
Goal: consolidate the current paper-usable localization baseline around retrieval init + geometry refinement, and separate it from the still-unreliable learned pose-refine branch.

---

## 1. Mainline decision

Current evidence supports the following split:

- **paper mainline**: strong initialization + geometric refinement / verification
- **research side-branch**: learned pose refinement / flow-based correction

Reason:

- perfect/oracle flow proves the geometry backend is capable of much lower error
- corrected real-init evaluation shows current learned refine is still weak
- `full_pose_weight`-style attempts do not recover the gap
- top-k fusion is not currently helping and can fail badly

---

## 2. Corrected real-init benchmark (current deployable baseline)

Protocol:

- init cache: `output/feature_retrieval/pose_regression/exp29a_direct_both_seed123/init_top5_learned.npz`
- effective protocol: `retrieval_topk=1`, `pose_fusion=none`
- corrected translation metric: camera-center distance
- test set size: `182`

### 2.1 Init only

- source: `exp29a_direct_both_seed123`
- median: `1.833 deg / 1406.7 mm`

Reference:

- `logs/exp29_report_20260420.md`
- echoed in:
  - `output/feature_retrieval/real_init_exp29a_top1_coarsefine_v1_full16_learnedinit_flowsmall_oi2_gi6_fixedmetric_eval/real_init/summary.json`

### 2.2 Current learned refine checkpoint (default solver)

- config: `pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall.yaml`
- checkpoint: `output/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall/checkpoints/best.pth`
- median: `2.172 deg / 1472.1 mm`

Reference:

- `output/feature_retrieval/real_init_exp29a_top1_coarsefine_v1_full16_learnedinit_flowsmall_oi2_gi6_fixedmetric_eval/real_init/summary.json`

### 2.3 Same checkpoint with `wls_full`

- median: `2.170 deg / 1379.5 mm`

Reference:

- `output/feature_retrieval/real_init_exp29a_top1_coarsefine_v1_full16_learnedinit_flowsmall_oi2_gi6_fixedmetric_wlsfull_eval/real_init/summary.json`

### 2.4 Interpretation

- current deployable learned-init baseline is still **meter-level**
- trusting the WLS/full-geometry branch at eval time helps translation a bit
- current learned pose head is **not** the source of strong gains

---

## 3. Oracle / geometry upper bound evidence

### 3.1 Perfect-flow diagnosis on real-init samples

Conclusion already established:

- with correct flow, the current geometry/update rule can converge to near-zero pose error on real OldHospital samples
- therefore the main bottleneck is **not** SE(3) convention or Jacobian formulation

Artifact:

- `output/feature_retrieval/real_init_exp29a_top1_oracle_perfectflow_sweep.json`

### 3.2 Smoke-split one-step oracle-flow upper bound

On the 8-sample smoke split used for quick diagnosis:

- init: `1.345 deg / 1353.0 mm`
- oracle-flow one-step geometry update: `0.040 deg / 177.4 mm`

Interpretation:

- the split is not inherently hard
- the remaining gap is almost entirely front-end / correspondence quality, not backend geometry

---

## 4. Strongest reproduced geometry pipeline (oracle retrieval setting)

These are not the learned-init deployable numbers; they are the best reproduced geometry-side evidence under oracle retrieval / strong LoFTR settings.

Source:

- `logs/exp30_report_20260421.md`

### 4.1 Clean oracle LoFTR baseline

Settings:

- `K=30`
- `loftr_mode=accumulated`
- `selection=inlier`
- `outer_iters=0`

Result:

- `0.366 deg / 205.2 mm`

### 4.2 Best reproduced translation median with render-and-rematch

Settings:

- same as above
- `rematch=4`

Result:

- `0.360 deg / 177.9 mm`

Recommended strongest translation baseline:

- `oracle_k30_accumulated_oi0_inlier_rematch4`

### 4.3 Best nearby rotation frontier

Settings:

- `rematch=4`
- `reproj_threshold=6.0`

Result:

- `0.340 deg / 179.2 mm`

### 4.4 What these results mean

- the major oracle gain comes from **stronger LoFTR settings + render-and-rematch**
- learned refinement is only a tiny effect under strongest oracle init
- historical `0.30 deg / 166 mm` is still not reproduced in the current worktree

---

## 5. Learned pose-refine branch status (do not use as paper mainline yet)

### 5.1 Real-init status

- current corrected benchmark is still about `1.38–1.47 m` median
- far from the geometry-side oracle frontier

### 5.2 Smoke diagnostics

Repeated smoke experiments show:

- `full_pose_weight=1.0` does not beat baseline
- `full_pose_weight=0.1` does not beat baseline
- `pose_weight=0, full_pose_weight=1.0` still does not recover useful pose gains

Observed behavior:

- `delta_xi_full` can be made larger
- predicted flow magnitude can be made larger
- but flow EPE stays poor and pose quality does not improve enough

### 5.3 Current conclusion

- learned pose refinement remains a **mechanism-research branch**, not a paper-ready main result

---

## 6. Top-k / fusion status

Smoke-split evidence with `topk=5` on the current baseline checkpoint:

- `top5 + none`: same as top-1 outcome in practice on the tested split
- `top5 + centroid`: catastrophic degradation
- `top5 + consensus`: catastrophic degradation
- `top5 + consensus_centroid`: catastrophic degradation

Interpretation:

- current top-k candidates are not providing additional usable oracle headroom after refine on the tested split
- current fusion methods should **not** be part of the paper mainline

---

## 7. Paper-usable matrix right now

### Main table candidates

1. **Init only (learned init exp29a)**
   - `1.833 deg / 1406.7 mm`

2. **Init + current learned refiner (default)**
   - `2.172 deg / 1472.1 mm`

3. **Init + current learned refiner (`wls_full`)**
   - `2.170 deg / 1379.5 mm`

4. **Oracle retrieval + strong LoFTR baseline**
   - `0.366 deg / 205.2 mm`

5. **Oracle retrieval + render-and-rematch (best reproduced translation)**
   - `0.360 deg / 177.9 mm`

6. **Oracle retrieval + render-and-rematch (best nearby rotation)**
   - `0.340 deg / 179.2 mm`

### Recommended narrative

- learned-init deployment baseline is still weak but stable and correctly measured
- geometry-side oracle analysis proves substantial remaining headroom
- render-and-rematch is the strongest reproduced source of large gains
- current learned refinement has not yet converted oracle headroom into deployable gains

---

## 8. Recommended next execution block

### For paper mainline

1. standardize the geometry-side table and regenerate any missing synced artifacts if needed
2. run a very small remaining sweep only around LoFTR-side filtering if searching for a better reproduced oracle point
3. otherwise freeze:
   - best translation baseline: `0.360 / 177.9 mm`
   - best nearby rotation baseline: `0.340 / 179.2 mm`

### For learned refine side-branch

next smallest useful experiment should target **flow learning failure directly**, not more `full_pose_weight` tuning.

Suggested next mechanism experiment:

- inspect / redesign flow supervision validity and confidence coupling
- explicitly test whether zero/small-flow bias is induced by mask coverage and current loss weighting
