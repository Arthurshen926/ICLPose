# 2026-05-12 NVS PoseEnergy Stage-1 Progress

## Implemented

- Added NVS-supervised pose-conditioned training entrypoint:
  - `feature_extract/tools/train_nvs_pose_feature_adapter.py`
  - query/render localization adapter
  - `PoseEnergyNet` candidate energy + residual head
  - GT/candidate NVS rendering with position maps
  - candidate score-map/vector feature construction
  - checkpoint roundtrip for adapter, energy net, and selected query prefixes
- Added gradient health logging:
  - `grad_nonfinite_elems`
  - `optimizer_step`
  - AMP skip detection
- Default NVS Stage-1 config now uses `amp: false` and disables GT candidate append to avoid fake learning shortcuts.
- Added residual-target modes:
  - `all`
  - `top1`
  - `soft_topk`
- Fixed residual eval bug: residual metrics now apply `pose_energy_update_scale`.

## Critical Bug Found

Earlier AMP runs were invalid. With AMP enabled, gradients became non-finite and `GradScaler` skipped optimizer steps. The logs changed because metrics were recomputed, but checkpoint weights did not update.

Evidence:

- AMP reproduction: `grad_nonfinite_elems=428502/428631`, optimizer step skipped.
- FP32 reproduction: finite gradients, nonzero parameter deltas.

All earlier `*_res0_*` AMP-style results should be treated as diagnostics only unless verified by checkpoint weight deltas.

## Valid FP32 Results

| Experiment | Bucket | Eval step | Selected | Oracle | Spearman | AUC | Residual |
|---|---:|---:|---:|---:|---:|---:|---:|
| `nvs_poseenergy_stage1_25cm5deg_uniform_nogt_fp32_b6_20260512` | 25cm/5deg | 20 | 115.0mm | 65.1mm | 0.324 | 0.765 | 117.6mm |
| same | 25cm/5deg | 40 | 135.7mm | 86.4mm | 0.193 | 0.851 | 134.0mm |
| same | 25cm/5deg | 60 | 130.8mm | 74.6mm | 0.358 | 0.975 | 132.0mm |
| `nvs_poseenergy_stage1_10cm2deg_residual_top1_scaledfix_fp32_b6_20260512` | 10cm/2deg | 20 | 46.0mm | 26.6mm | 0.673 | 1.000 | 45.8mm |
| `nvs_poseenergy_stage1_25cm5deg_residual_top1_scaledfix_fp32_b6_20260512` | 25cm/5deg | 20 | 115.0mm | 65.1mm | 0.218 | 0.778 | 114.9mm |
| `nvs_poseenergy_stage1_50cm10deg_uniform_nogt_fp32_b6_20260512` | 50cm/10deg | 20 | 357.0mm | 131.0mm | 0.077 | 0.446 | 360.2mm |
| `nvs_poseenergy_stage1_50cm10deg_energyonly_fp32_b6_20260512` | 50cm/10deg | 20 | 357.3mm | 131.0mm | 0.076 | 0.444 | 360.5mm |
| `nvs_poseenergy_stage1_25cm5deg_realonly_adapter_energy_fp32_b6_20260512` | 25cm/5deg | 20 | 156.3mm | 88.6mm | 0.278 | 0.701 | n/a |

## Conclusions

- The trainable PoseEnergy selector is a real improvement over hand local-corr scoring on 25cm/5deg:
  - hand score: about 169mm
  - PoseEnergy best: 115mm
- It has not reached the CPR target of `<100mm` at 25cm/5deg.
- Residual update is not a main breakthrough yet:
  - scale bug fixed
  - top1 residual mode added
  - best gains are only about 0.1-0.3mm on validation
- 50cm/10deg remains unsolved:
  - direct, energy-only, and curriculum-from-25 all fail around 357-520mm selected cost
  - topK/oracle exists, but selector evidence is not sufficient at this basin
- Real-only training does not improve validation; mixed synthetic/real early stopping is still better.

## Next Mainline

Do not continue tuning WLS scale, residual update scale, or 50cm direct training.

Next code direction:

1. Add cached candidate feature dataset for selector training to remove expensive render noise and allow larger batches.
2. Train selector with more candidates/samples and early stopping by validation selected cost.
3. Add hard-negative mining around candidates that hand score/energy consistently misselect.
4. Add reliability/calibration head and abstention/failure score before reintroducing residual updates.
5. Only return to 50cm after 25cm/5deg reliably reaches `<100mm`.


## 2026-05-13 Update: Expert-5 NVS Pose-Conditioned Adaptation

### Implemented from the latest expert plan

- Added localization-space uncertainty support in `PoseFeatureDomainAdapter` and propagated query/render uncertainty into fine selector / PoseEnergy vector features.
- Added factorized PoseEnergy heads: translation, rotation, and joint energy losses are now trainable alongside the joint logits.
- Added balanced candidate bank mode consistently for NVS training, standalone PoseEnergy training, and bucket eval.
- Added weighted `candidate_center_buckets` curriculum sampling, e.g. `10:2:0.5,25:5:0.5`, so Stage-1 no longer trains only on a single noisy-init bucket.
- Added NVS score monotonicity loss by rerendering the selected residual-updated pose once and requiring improved poses to score higher.
- Fixed checkpoint/eval plumbing:
  - NVS checkpoints can be loaded by `eval_pose_energy_buckets.py`.
  - best checkpoint selection now defaults to `pose_energy_pred_cost_m` when PoseEnergy is enabled.
  - independent bucket eval now respects balanced candidate banks.
- Added Stage-0 feature-pose audit tool `feature_extract/tools/eval_feature_pose_audit.py`.

### Verification

- `python -m py_compile` passed for:
  - `feature_extract/tools/train_nvs_pose_feature_adapter.py`
  - `feature_extract/tools/train_pose_energy.py`
  - `feature_extract/tools/eval_pose_energy_buckets.py`
  - `feature_extract/tools/eval_feature_pose_audit.py`
  - `feature_extract/students/pose_energy_net.py`
  - `feature_extract/train_impl.py`
- `pytest tests/test_pose_energy_net.py tests/test_nvs_pose_feature_adapter.py tests/test_adaptive_joint_query_map.py::test_fine_candidate_selector_features_include_uncertainty_stats -q`: 41 passed.

### Stage-0 audit result

Using the current frozen RADIO/student/map feature spaces on 16 val samples, K32:

| Bucket | Feature combo | Spearman | Selected trans | Oracle trans |
|---|---|---:|---:|---:|
| 25cm/5deg | teacher query vs map base | 0.190 | 258.0mm | 159.5mm |
| 25cm/5deg | student query vs map base | -0.060 | 249.9mm | 159.5mm |
| 25cm/5deg | projected student/map | -0.072 | 249.9mm | 159.5mm |
| 50cm/10deg | teacher query vs map base | 0.236 | 499.6mm | 242.0mm |
| 50cm/10deg | student query vs map base | -0.053 | 499.6mm | 242.0mm |
| 50cm/10deg | projected student/map | -0.107 | 499.6mm | 242.0mm |

Conclusion: raw RADIO/student/DCFF similarity is not pose-sortable enough. This supports the expert claim that RADIO should be a base anchor, not the final localization feature.

### NVS adapter experiments

1. Single-bucket smoke, batch 2, 12 steps:
   - Train/eval-in-loader PoseEnergy improved sharply over hand score.
   - Step 6 eval: hand score Spearman `-0.042`, pred cost `0.460m`; PoseEnergy Spearman `0.742`, AUC `0.886`, pred cost `0.108m`.
   - Found and fixed a best-checkpoint bug: the script was still selecting by old `pred_cost_m` instead of `pose_energy_pred_cost_m`.

2. Mixed wide basin from start (`10/25/50/100cm`), batch 4:
   - Negative result. Step 12 eval PoseEnergy Spearman `-0.495`, pred cost `0.618m`.
   - Stopped early. Conclusion: do not mix large basin at Stage-1; it destabilizes the energy evidence.

3. Curriculum Stage-1 (`10cm/2deg` + `25cm/5deg`), batch 4, 24 steps:
   - Step 12 eval is best by PoseEnergy cost: Spearman `0.553`, AUC `0.776`, pred cost `0.178m`, oracle `0.071m`.
   - Step 24 improves feature alignment (`align_cos` `0.168`) but worsens selection (`pose_energy_pred_cost_m` `0.209m`), so early stopping is necessary.
   - Monotonicity remains not solved: score gain is negative and monotonicity loss grows late in training.

### Independent bucket eval of curriculum checkpoint

Using `eval_pose_energy_buckets.py`, 16 val samples, K64:

| Bucket | Spearman | Selected trans | Init trans | Oracle trans | Gain+ | Comment |
|---|---:|---:|---:|---:|---:|---|
| 10cm/2deg | 0.560 | 145.9mm | 100.0mm | 42.2mm | 0.125 | Rank correlation exists, but top1 moves in wrong correction direction. |
| 25cm/5deg | 0.496 | 264.4mm | 249.9mm | 90.5mm | 0.438 | Evidence is partly sorted, but selected pose is worse than init. |
| 50cm/10deg | 0.221 | 507.7mm | 499.6mm | 229.7mm | 0.438 | Not ready for medium basin. |

### Current conclusion

The latest expert direction is partly validated but not yet a final method:

- Validated: NVS-supervised localization adapters + PoseEnergy can produce a much stronger rank signal than raw feature similarity.
- Not validated: top1 pose selection and residual update still do not improve final pose on independent bucket eval.
- Main failure mode: the model often chooses a fixed-magnitude non-center candidate whose correction direction is weak or wrong. Correlation can be positive while top1 is still bad.
- Residual update is still weak and sometimes harmful; keep it diagnostic until selected candidate improves.

### Next adjustment

Do not expand to 50cm/10deg yet. The next code change should add direction-aware supervision before more training:

1. Penalize selected candidate correction direction when its camera-center correction cosine to GT is low.
2. Add candidate-vector/correction-direction features explicitly, not only scalar delta magnitudes.
3. Use listwise/pairwise losses that compare candidates with similar magnitude but different correction direction.
4. Track and early-stop on independent bucket metrics: selected trans, selected correction cosine, and gain+, not only Spearman or PoseEnergy cost.
5. Temporarily reduce or disable monotonicity weight until selected top1 is reliable; current monotonicity rerender produces large negative score gain.

### 2026-05-13 follow-up: implemented expert backlog items

Implemented and verified:

- Added inference-available camera-center correction features for PoseEnergy / fine selector:
  `center_dx/dy/dz` plus per-row z-score variants.
- Propagated the new feature through standalone PoseEnergy train/eval and NVS adapter train.
- Added a supervised confidence calibration loss for the existing `PoseEnergyNet.confidence_head`; before this, the head was present but unused.
- Stage-1 config now enables center-delta features, confidence loss, uncertainty features, factorized energy heads, direction pairwise loss, and a small-lattice Stage-1 protocol.

Verification:

- `pytest tests/test_pose_energy_net.py tests/test_nvs_pose_feature_adapter.py tests/test_adaptive_joint_query_map.py::test_fine_candidate_selector_features_include_uncertainty_stats tests/test_adaptive_joint_query_map.py::test_fine_candidate_selector_features_can_include_center_delta_vector -q`: 45 passed.
- `python -m py_compile` passed for `pose_energy_net.py`, `train_impl.py`, `train_pose_energy.py`, `eval_pose_energy_buckets.py`, and `train_nvs_pose_feature_adapter.py`.

New experiments:

| Experiment | Internal eval result | Independent bucket result | Conclusion |
|---|---|---|---|
| center-delta + direction + confidence, synthetic+real, wide Stage-1 lattice | step16 PoseEnergy Spearman `0.483`, pred cost `0.184m`, oracle `0.068m` | 10cm: `113.6mm`, 25cm: `246.3mm`, 50cm: `496.7mm` | Stronger rank signal, but top1 still picks wrong direction/magnitude. |
| same, real-only | step16 PoseEnergy Spearman `0.186`, pred cost `0.205m`, oracle `0.070m` | 10cm: `113.6mm`, 25cm: `246.3mm`, 50cm: `498.8mm` | Synthetic+real is better internally, but independent final remains bottlenecked by candidate choice. |
| small Stage-1 lattice, synthetic+real | step16 PoseEnergy Spearman `-0.519`, pred cost `0.296m` | not promoted | Simply removing wide candidates collapses the learned energy signal. |
| small Stage-1 lattice, real-only | step16 PoseEnergy Spearman `-0.488`, pred cost `0.375m` | not promoted | Negative control; small lattice alone is not the fix. |

Current interpretation:

- The new center-delta + confidence additions are useful for training signal: wide-lattice internal Spearman improved to about `0.48`.
- They do not yet solve final pose because the selected candidate still has poor correction direction:
  - `10cm/2deg`: selected correction cosine `-0.11` while oracle is `0.95`.
  - `25cm/5deg`: selected correction cosine `0.14` while oracle is `0.92`.
- Residual update remains diagnostic only; it changes translation by only a few millimeters and does not repair wrong candidate selection.
- The strongest remaining unimplemented expert idea is not another shallow head, but a denser candidate-direction curriculum / cache that explicitly balances same-magnitude opposite-direction candidates and early-stops on independent selected correction cosine, not only Spearman.
