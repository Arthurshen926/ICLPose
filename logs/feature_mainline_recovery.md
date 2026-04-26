# Feature Mainline Recovery

## 2026-04-21 Batch 0-1

### Goals

- unify experiment artifacts so every run writes `config.yaml`, `results.json`, `report.md`, and representative qualitative outputs
- start the first coarse-supervision recovery sweep for the joint `student + DCFF + FSM` path
- decide whether coarse supervision is worth carrying into the next pose-refine integration step

### Infra changes completed

- added a reusable experiment-bundle helper in `feature_field/utils/loc_reporting.py`
- wired bundle output into:
  - `feature_extract/train_impl.py`
  - `pose_refine/evaluate_pipeline.py`
  - `pose_refine/train_impl.py`
- verified that new runs now emit:
  - `results.json`
  - `results.txt`
  - `report.md`
  - `report.txt`
  - existing qualitative dirs where available

### Batch 1 configs added

- `feature_extract/configs/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_fineonly.yaml`
- `feature_extract/configs/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_lightcoarse.yaml`
- `feature_extract/configs/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_strongcoarse.yaml`

Temporary choice for this batch:

- retrieval supervision disabled in these three configs
- reason: the configured retrieval teacher cache path was missing, and this batch was specifically for validating the `student <-> map coarse/fine` loop without unrelated path failures

### Smoke-test summary

All three smoke runs completed and produced the new unified bundle.

Key smoke signals:

- `fineonly`
  - `best_val = 3.7313`
  - `map_query_fine_cosine = 0.1644`
  - `map_query_coarse_cosine = -0.0388`
- `lightcoarse`
  - `best_val = 3.6618`
  - `map_query_fine_cosine = 0.2538`
  - `map_query_coarse_cosine = -0.0138`
- `strongcoarse`
  - `best_val = 4.1736`
  - `map_query_fine_cosine = 0.1955`
  - `map_query_coarse_cosine = 0.1202`

Interpretation:

- coarse supervision is real and moves the intended signal
- but strong coarse weighting hurts early overall stability
- `lightcoarse` looked like the safest full-run candidate after smoke

### Full-run results

Output dirs:

- `output/feature_extract/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_fineonly/`
- `output/feature_extract/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_lightcoarse/`
- `output/feature_extract/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_strongcoarse/`

Final metrics:

#### fineonly

- `best_val = 2.5147`
- `fine_cosine = 0.0867`
- `coarse_cosine = 0.2992`
- `map_query_fine_cosine = 0.9000`
- `map_query_coarse_cosine = 0.0910`
- `map_coarse_active = 0.0`

#### lightcoarse

- `best_val = 2.7950`
- `fine_cosine = 0.0874`
- `coarse_cosine = 0.2864`
- `map_query_fine_cosine = 0.9045`
- `map_query_coarse_cosine = 0.3671`
- `map_coarse_active = 1.0`

#### strongcoarse

- `best_val = 2.9747`
- `fine_cosine = 0.0997`
- `coarse_cosine = 0.2293`
- `map_query_fine_cosine = 0.8838`
- `map_query_coarse_cosine = 0.6075`
- `map_coarse_active = 1.0`

### Main conclusion from Batch 1

- `fineonly` still wins on the current aggregate validation loss.
- `lightcoarse` is the best balanced coarse-enabled variant.
- `strongcoarse` proves the coarse signal can be made strong, but it over-weights coarse alignment and degrades the broader objective.

Most important learning:

- the coarse path is now alive and measurable
- but the current scalar weighting makes it easy to optimize coarse alignment without improving the full objective

This means the next step should not be more blind coarse-weight sweeps first.

Instead, the next high-value move is:

- explicitly expose `rendered_coarse + FSM outputs` inside `pose_refine`
- so coarse information becomes part of the localization dataflow rather than only a map-supervision side signal

### Next step selected

Proceed to Batch 2:

- patch `pose_refine/train_impl.py` so render helpers return a bundle containing:
  - `fine_features`
  - `depth`
  - `coarse_features`
  - `fsm_spatial_conf`
  - `fsm_channel_weights`
- keep the current model API unchanged for now
- run smoke tests to confirm the aux tensors propagate through training / validation / visualization without breaking existing behavior

## 2026-04-22 Batch 3-4

### Goals

- verify whether feature-based reranking can make `student/DCFF/FSM` affect the actual localization outcome
- avoid the earlier no-op implementation where rerank only permuted candidate order but refinement still evaluated every candidate
- test the minimal gating variants first: rerank `top-1` and `top-2`

### Batch 3 diagnosis: why the first rerank had zero effect

The first rerank patch in `pose_refine/evaluate_pipeline.py` did compute meaningful candidate scores and reorder most images, but the pipeline still:

- refined all `K` LoFTR candidates
- selected the final pose by residual after refinement

That made the result permutation-invariant. Candidate order changed, but the set of refined candidates did not, so the final pose stayed identical.

### Batch 4 code changes

Patched `pose_refine/evaluate_pipeline.py` so that feature reranking now affects the candidate set, not just the order.

Main changes:

- each LoFTR candidate now records `original_ci`
- added CLI flag:
  - `--feature_rerank_top_m`
- when `feature_rerank != off` and `selection != inlier`:
  - compute coarse+fine rerank score per candidate
  - reorder candidates by score
  - refine only the top `M` candidates after rerank
- per-image reporting now records:
  - `selected_ci_original`
  - `selected_slot_after_rerank`
  - `rerank_top_original_ci`
  - `num_candidates_total`
  - `num_candidates_refined`
  - `selection_score_total`
  - `selection_score_fine`
  - `selection_score_coarse`

Follow-up cleanup:

- kept `feature_residual` as a fine-only residual for reporting continuity
- when rerank is enabled, final candidate selection now uses the same coarse+fine scoring family rather than dropping back to fine-only residual
- verified the updated evaluator with `py_compile` and a rerank smoke run:
  - `output/pipeline_eval/smoke_feature_rerank_top1_oi0_v2/`

### Gated rerank experiments

All runs used the same setup for comparability:

- config: `pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4.yaml`
- checkpoint: `output/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4/checkpoints/best.pth`
- `K=3`, `outer_iters=0`, `selection=residual`, `loftr_mode=individual`

Output dirs:

- baseline: `output/pipeline_eval/rerank_gate_cmp_baseline_k3_oi0/`
- top-1: `output/pipeline_eval/rerank_gate_cmp_feat_top1_k3_oi0/`
- top-1 + FSM: `output/pipeline_eval/rerank_gate_cmp_feat_fsm_top1_k3_oi0/`
- top-2: `output/pipeline_eval/rerank_gate_cmp_feat_top2_k3_oi0/`
- top-2 + FSM: `output/pipeline_eval/rerank_gate_cmp_feat_fsm_top2_k3_oi0/`

### Results summary

Reference baseline:

- `Refined S1 = 0.543° / 351.9 mm`
- `<1°/50mm = 3.85%`
- `<0.5°/30mm = 1.10%`
- `<5°/250mm = 40.66%`

#### top-1 rerank

- `0.573° / 336.7 mm`
- `<1°/50mm = 2.75%`
- `<0.5°/30mm = 0.55%`
- `<5°/250mm = 39.01%`

Conclusion:

- top-1 gating is too aggressive
- it does change the outcome, but hurts the accuracy thresholds and median rotation

#### top-1 rerank + FSM

- `0.582° / 348.8 mm`
- `<1°/50mm = 2.75%`
- `<0.5°/30mm = 0.55%`
- `<5°/250mm = 38.46%`

Conclusion:

- FSM weighting does not rescue the top-1 version
- this variant is worse than both baseline and plain top-1

#### top-2 rerank

- `0.538° / 322.2 mm`
- `<1°/50mm = 3.30%`
- `<0.5°/30mm = 0.55%`
- `<5°/250mm = 41.76%`

Conclusion:

- this is the first rerank setting that is actually useful
- median translation improves clearly: `351.9 mm -> 322.2 mm`
- median rotation improves slightly: `0.543° -> 0.538°`
- `<5°/250mm` also improves slightly
- but the tight thresholds `<1°/50mm` and `<0.5°/30mm` still regress versus baseline

#### top-2 rerank + FSM
- `0.538° / 322.2 mm`
- `<1°/50mm = 2.75%`
- `<0.5°/30mm = 0.55%`
- `<5°/250mm = 41.76%`

Conclusion:

- same median gains as plain top-2
- but worse tight-threshold accuracy than plain top-2
- FSM weighting is still not helping this stage of the pipeline

### Main conclusion from Batch 4

- feature rerank is no longer a no-op
- candidate gating can change the final localization result
- `top-2` is currently the only promising setting in this family
- `top-1` is too brittle
- FSM-weighted rerank does not currently help

Most important learning:

- coarse/fine feature scoring can now influence the actual refinement path
- but the gains are modest and mostly on median translation, not on the most important tight-accuracy regime
- this is enough evidence to keep `top-2` as a useful diagnostic branch, but not enough to treat rerank as the mainline solution

### Next step selected

Do not spend many more cycles on rerank-only tuning.

Next high-value move:

- push reconstruction features further into the core localization path, likely via one of:
  - feature-based init in `pose_refine/sparse_init.py`
  - or modifying `ConcatPoseNet` so `rendered_coarse` participates directly in coarse-to-fine pose updates

Current recommendation:

- keep `top-2` rerank as the best feature-gated baseline for comparison
- then pivot to feature-based init / feature-PnP prototype rather than continuing small rerank sweeps

## 2026-04-22 Batch 5

### Goal

- stop treating coarse as pooled side-context and switch the no-LoFTR mainline to an explicit `coarse -> pose update -> fine` refinement path
- keep the change config-gated so existing single-stage checkpoints and configs still load
- reuse the shared DCFF render-bundle runtime for both training and retrieval-time evaluation

### Code changes completed

- `feature_field/runtime.py`
  - extended `render_at_pose`, `render_batch`, `render_feature_bundle_at_pose`, and `render_feature_bundle_batch` with optional `render_coarse`
  - this lets coarse-only render passes happen without forking another render stack
- `pose_refine/models/concat_pose_net.py`
  - added `CoarsePoseStage`
  - added config flags:
    - `use_two_stage_refine`
    - `coarse_only_first_iter`
    - `coarse_stage_*`
  - coarse stage now explicitly consumes `query_coarse + rendered_coarse + optional FSM spatial confidence`
  - fine stage remains the existing flow/WLS or flow/MLP local refiner
- `pose_refine/runtime.py`
  - added shared helpers:
    - `apply_pose_delta(...)`
    - `run_model_refine_iteration(...)`
  - one refinement iteration is now:
    - optional coarse render
    - coarse pose delta
    - pose update to `pose_mid`
    - fine re-render at `pose_mid`
    - fine delta to `pose_next`
- `pose_refine/train_impl.py`
  - switched non-differentiable rendering to shared bundle helpers
  - training loop now uses the same explicit two-stage iteration helper
  - added coarse-stage pose supervision via `training.loss.coarse_pose_weight`
  - added `training.use_amp` and `training.loss.pose_loss_mode`
  - validation and visualization now also run through the new two-stage path
  - fixed missing `self.exp_name` assignment used by bundle export
- `feature_retrieval/evaluate_impl.py`
  - retrieval-time real-init refinement now uses the same explicit two-stage iteration helper before any solver-specific update
  - this makes learned-init evaluation structurally consistent with training
- new configs:
  - `pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1.yaml`
  - `pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_smoke.yaml`

### Debugging and stabilization

- fixed Python 3.9 import/runtime annotation issues introduced during the refactor
- found that AMP caused inf gradients in the new coarse-fine training path even when fp32 losses were finite
- added config-controlled AMP and disabled it in the new coarse-fine configs
- added a stable Lie-space supervision mode for the new coarse/fine pose losses while keeping old composed-pose loss behavior available

### Verification completed

- `python -m py_compile` passed for:
  - `feature_field/runtime.py`
  - `pose_refine/models/concat_pose_net.py`
  - `pose_refine/runtime.py`
  - `pose_refine/__init__.py`
  - `pose_refine/train_impl.py`
  - `feature_retrieval/evaluate_impl.py`
- CPU model build smoke passed:
  - `two_stage=True`
  - `coarse_stage=True`

### Smoke results

- reduced warmstart eval-only with the old single-stage checkpoint under the new two-stage architecture was poor, as expected:
  - `2.68° / 4707.3 mm` on the smoke validation path
  - confirms the new coarse head must be trained, not just warmstarted
- one-epoch smoke training on the 16/8 smoke split completed successfully after disabling AMP:
  - train: `loss=1.8378`, `flow_epe=1.41`, `rot=0.99°`, `trans=384.1 mm`
  - val: `rot_med=2.53°`, `trans_med=978.7 mm`
- small learned-init real-init eval (`max_samples=8`, `exp29a` init) now changes the final pose instead of staying exactly equal to init:
  - init median: `1.345° / 1154.0 mm`
  - final median: `1.164° / 1156.1 mm`

### Current status

- the codebase now has a real external coarse stage instead of the previous pseudo-coarse pooled-context path
- the exact no-op behavior is broken on the smoke real-init check: `final != init`
- translation has not improved yet in the smoke run; the current smoke checkpoint mainly shows the plumbing now works and the model can train stably
- because `retrieval_topk=1` in the smoke real-init eval, `top1_final == final == oracle_final` is expected there and no longer means the refiner is a no-op

### Immediate next step

- run a longer full-split training job with:
  - `pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1.yaml`
- then evaluate that checkpoint on the full test set with:
  - `feature_retrieval/evaluate_impl.py`
  - learned init from `exp29a_direct_both_seed123/init_top5_learned.npz`
- primary success criterion for the next batch:
  - improve translation under learned init while keeping the new path clearly non-no-op

## 2026-04-23 Learned-Init Follow-Up

### Goal

- continue from the learned-init coarse/fine results and test the smallest remaining retrains that could still improve full `exp29a` learned-init evaluation
- keep the evaluation protocol fixed:
  - `retrieval_topk=1`
  - `pose_fusion=none`
  - `init_poses_path=output/feature_retrieval/pose_regression/exp29a_direct_both_seed123/init_top5_learned.npz`

### Key diagnosis

- the planned `flow_init: small` retrains were not actually valid under the previous warmstart behavior
- reason:
  - `flow_init` only affects fresh model initialization
  - the warmstart checkpoint restored `gru_flow_head.2` because its shape matched
  - that silently overwrote the new flow-head initialization and made `flow_init: small` mostly ineffective

### Code change

- patched `pose_refine/train_impl.py` so warmstart can skip selected parameter prefixes via:
  - `training.warmstart_skip_prefixes`
- used:
  - `warmstart_skip_prefixes: [gru_flow_head.2]`
- this keeps the rest of the model warmstarted while leaving the final GRU flow-output layer at true fresh `flow_init: small`

### New configs added

- `pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall.yaml`
- `pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall_hybridrot.yaml`
- `pose_refine/configs/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1e_lowresw2_wls_learnedinit_flowsmall.yaml`

### Training outcomes

- canonical `flowsmall` (`full16_learnedinit_flowsmall`):
  - best val happened immediately at epoch 0
  - best val: `rot_med=1.99°`, `trans_med=1377.9 mm`
  - later epochs regressed and never beat epoch 0
- canonical `flowsmall + hybridrot`:
  - best val arrived later but remained weaker on translation
  - best val: `rot_med=1.95°`, `trans_med=1394.3 mm`
- lowres+wls `flowsmall` with real flow-head skip:
  - best val: `rot_med=1.96°`, `trans_med=1440.1 mm`
  - did not recover the earlier translation level of the canonical branch

### Full `exp29a` learned-init evaluation

- previous strict-translation best before this batch:
  - canonical learned-init: `1394.1 mm / 2.091°`
- new canonical `flowsmall`, epoch 0:
  - `oi=2 gi=4`: `1368.2 mm / 2.004°`
  - `oi=2 gi=6`: `1367.9 mm / 2.034°`
  - `oi=2 gi=8`: `1375.9 mm / 2.029°`
  - `oi=2 gi=10`: `1380.0 mm / 2.019°`
- new hybrid variant:
  - epoch 0, `oi=2 gi=4`: `1419.5 mm / 1.983°`
  - epoch 2, `oi=2 gi=4`: `1392.2 mm / 2.117°`
- new lowres+wls `flowsmall` with real flow-head skip:
  - epoch 2, `oi=2 gi=4`: `1440.4 mm / 1.957°`

### Best current result

- best strict translation found so far:
  - canonical `flowsmall` with real warmstart skip
  - `output/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall/checkpoints/best.pth`
  - full eval at `oi=2 gi=6`
  - `1367.9 mm / 2.034°`

### Conclusion

- the warmstart-aware `flow_init: small` fix was real and produced a new best full learned-init translation result
- the gain is modest but unambiguous versus the prior best `1394.1 mm / 2.091°`
- the follow-up evidence points to a plateau in this line:
  - canonical `flowsmall` peaks at epoch 0
  - extra GRU iterations beyond `gi=6` do not help
  - `hybridrot` and lowres+wls variants improve some rotation numbers but do not beat the new canonical translation best on the target metric
