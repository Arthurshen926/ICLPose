# 2026-05-16 ICLPose POFD Goal Audit

## Objective

Continue the project according to `ChatGPT-ICLPose (1).md`, use available GPUs
for experimental validation where useful, and do not call the work complete
until the method reaches the expert-file promote gates and a journal-grade
evidence chain.

## Prompt-To-Artifact Checklist

| requirement from expert file | current evidence | status |
|---|---|---|
| Implement PMED / pair-matcher-aware GeoNCE and cost-shaped ranking | Stage4 pair-flow losses, positive-only CE branch, listwise/GeoNCE D1-D5 configs and results are documented in `2026-05-16-pofd-stage4-single-render-continuous-cpr.md` | implemented, no-go |
| D1/D2/D3/D4/D5 recommendation matrix | q50/q25/q10 val128 results are logged in the Stage4 plan | completed, below promote target |
| q50 pred_cost <= 0.200 | correction reward reaches 0.160m on q50 val128 and 0.179m on q50 full182 | only controlled-cache reranker passes |
| q50 top1 >= 0.78 | correction reward reaches 0.898 on val128 and 0.874 on full182 | only controlled-cache reranker passes |
| q50 spearman >= 0.58 | correction reward is 0.500 on val128 and 0.481 on full182; offline 377,620 transform sweep found no step-prior transform >= 0.55 | missing |
| q50 succ@25cm/10deg >= 0.80 | correction reward reaches 0.914 on val128 and 0.890 on full182 | only controlled-cache reranker passes |
| q50 oracle_gap <= 0.070 | correction reward reaches 0.030m on val128 and 0.050m on full182 | only controlled-cache reranker passes |
| q50 rank/selection calibration without cache-PnP leakage | even/odd offline ridge on score+delta approaches the rank proxy but misses q50 cost/top1/success; POFD-only score-neighborhood smoothing is 0.214m/top1 0.766 when selection-tuned or 0.267m when rank-tuned; adding render-LoFTR/PnP cache fields recovers oracle but is a multi-render selector baseline | missing for POFD |
| q10/q25 regression protection | reward=2 reaches q25 0.073m / q10 0.028m on val128 and q25 0.071m / q10 0.028m on full182, but exact D3 q25 is 0.111m | mixed; main PMED branch fails q25 |
| Replace multi-candidate render as paper main path | Stage4 single-render render-once/match/solve evaluator is implemented and tested | implemented, not performant |
| Real-init deploy evaluation improves initializer by >=20% | best checked Stage4 real-init val128 gain is +6.00mm on a 37.0cm initializer (1.6% relative); full182 0.5cm translation-only iter1/iter2/iter3 are +3.96mm/+5.61mm/+4.19mm on a 44.0cm initializer; best full182 median translation is only about 21.7cm from a 22.3cm median initializer; translation-only avoids rotation drift but the best mean-cost setting regresses 5cm/2deg, 10cm/5deg, and 50cm/10deg success | missing |
| Stronger render-LoFTR top50 real-init check | top50 cache Stage4 pair-flow w2 gives +3.79mm with one SE(3) update, +5.36mm with two SE(3) updates, +6.00mm with two translation-only val128 updates, and at best +5.61mm on full182; full182 iter1/iter3 do not beat iter2, and the best single-render full182 median remains worse than local multi-render render-LoFTR/PnP baselines at 16.1-16.7cm | missing |
| Existing virtual trust-region fallback | top50 smoke32 with 0.01/0.02 score-gap thresholds regresses mean cost by 3.9-4.9cm | rejected |
| Proposal virtual acceptance gate | implemented and smoke-tested; one-step top50 smoke32 remains 1.7-2.7mm worse than init, iter3 with gate remains 1.9mm worse and drops success@10 from 0.781 to 0.750 | diagnostic only, rejected as promote path |
| Stage4 update-direction confidence test | per-update JSONL dump implemented; val128 threshold sweep shows no stable deployable confidence gate; full182 generated-update oracle gives only +16.49mm for 0.5cm and +9.10mm for 0.25cm, while held-out one-feature gates are only marginal | rejected for current proposal generator |
| Stage4 expected-offset proposal smoothing | top50 val128 expected-offset variants give only +0.3mm and lower correction cosine than argmax | rejected |
| Stage4 raw local-correlation proposal source | top50 val128 local_corr argmax has valid_frac 0.119, solver_success 0.008, accept 0.0, and is a no-op | rejected |
| Stage4 real-init pair-flow fine-tune | top1 random-cache pair-flow improves valid match coverage; translation-only two-update reaches the best mean gain (+6.00mm) but small-error and 10cm/50cm success regress; all-candidate top50/top16 training is computationally impractical in the current scoring path | partial diagnostic, not promotable |
| POFD-DenseFlow single-render proposal | dense-flow target/head/loss/checkpoint/Stage4 source are implemented and tested; two 80-step GPU smoke trainings produce valid checkpoints, but val128 real-init regresses 0.3700m -> 0.3767/0.3768m, drops 5cm/2deg and 10cm/5deg success, and accepts every update with only 3.1% match coverage | implemented, no-go |
| Single-render featuremetric GN real-init baseline | the existing featuremetric GN eval path was repaired with a v68 64d config because the old 96d matcher feature export is absent; two 64-sample GPU smokes on real NetVLAD+render-LoFTR top50 quality init regress median translation 104.5mm -> 124.9/128.2mm, with only 9/64 samples improved | existing-tool diagnostic, no-go |
| External render-at-init LoFTR+PnP single-render baseline | lightweight LoFTR diagnostic now supports real init caches and exports refined-pose caches; full182 strict setting improves the real initializer from 223.3mm/0.413deg median to 156.3mm/0.248deg, with 133/182 samples improved and a 182-pose teacher cache exported; this is a strong comparator/teacher but relies on external LoFTR, not POFD | baseline-positive, not main contribution |
| POFD Stage4 on external LoFTR refined cache | the refined-pose cache was wired into Stage4 as a one-candidate real-init cache. On full182, the real-init top50/w2 POFD updater regresses mean cost 0.274m -> 0.420m and drops 10cm/5deg success 0.335 -> 0.159; the q50 all-margin checkpoint rejects every update and exactly preserves the LoFTR-refined init. Current POFD proposals do not add value on top of the external teacher signal | diagnostic completed, no-go |
| Render-at-init LoFTR teacher-flow distillation | train/test LoFTR correspondence stores were exported and a query-centered teacher pair-flow loss was implemented. At 34x60, 99.5% of train teacher offsets quantize to zero, so a 136x240/min-offset branch was tested. Integer argmax regresses 0.0820m -> 0.0933/0.0934m; expected-offset softening reduces the regression to 0.0847/0.0849m but still fails. The continuous/subpixel branch learns sparse teacher EPE to about 0.382px, but full-grid expected proposals collapse to 0.375/0.382m with confidence about 4e-5 and 4.86px offsets; a 0.001 confidence gate is safe but exactly no-op | implemented, no-go for current teacher-flow proposal |
| Existing PoseEnergy residual/selector path | code already includes energy, confidence, factorized heads, residual deltas, direction/correction-cosine losses, and residual-update metrics; 25cm pair-heatmap eval works (about 0.065m, Spearman 0.71), but q50 pair-heatmap PoseEnergy eval is 0.484-0.512m with negative Spearman and zero residual-update gain | rejected for current q50 target |
| Single-render PoseEnergy residual pivot | direct real-init top1 residual training was isolated from pair-matcher and teacher-correlation losses, with a zero-init residual head for no-op safety. Smoke32 shows a small +3.8mm to +5.6mm gain, but explicit full182 eval after removing the inherited 32-sample cap gives only +0.47/+0.50mm and no success-rate change | diagnostic weak positive, not promotable |
| External correspondence + POFD reranking pivot | cache tools now gate refined poses, append render-at-init LoFTR refined poses as candidate 51 behind the original top50, and export compact top15+refined caches to keep training at the old 16-candidate memory footprint. Simple LoFTR inlier/ratio quality gating is no better than accepting all refined poses. The appended candidate improves full182 oracle median/mean from 78.9/115.6mm to 74.3/110.1mm in the top50 bank; compact top15+refined has a 0.1337m full182 oracle, but the best trained quality-s20 reranker is only 0.3136m on full182 versus 0.3172m baseline and still chooses worse-than-init poses on 59/182 rows. Score-prior/fallback gates improve robustness metrics only by falling back to init and do not improve the main cost. Longer 80-step rerank probes regress. LoFTR threshold sweeps (conf0.2/r4, conf0.3/r8) do not beat the existing conf0.3/r4 refined baseline. A new external iterative LoFTR render-at-current path exports iter2/iter3 caches; raw iter2/iter3 mean improves from 273.4mm to 264.5/264.3mm, but median worsens from 156.3mm to 175.5/181.9mm and no-GT inlier/raw gates cannot select the better subset. A no-GT pose-step gate using the one-pass->iter2 step selects iter3 for 35/182 rows and reaches 159.6mm median / 256.2mm mean with 10/25/50cm success 0.346/0.654/0.896. A DCFF feature-consistency scorer and score-delta gate show weak pose-level residual signal (corr with true iter3 gain 0.009); combined with the 0.20m pose-step gate it selects 27/182 rows and reaches 156.3mm median / 255.7mm mean, preserving the one-pass median but only improving the pose-step mean by 0.6mm. Stage4 all-margin q50 on that cache is still an exact no-op at 0.2563m init/pred cost and 0.0 accepted updates. A deeper Stage4 dump shows this no-op hides an unsafe proposal generator: lowering min-points to 16 gives solver success but all first-iteration updates worsen on smoke32, and a 5cm/0.5deg full182 accept gate accepts only 2.2% and no-ops at 0.25630m -> 0.25648m. Per-correspondence all-match scoring was then added; full182 DCFF residual has lower-is-inlier AUC 0.484 over 525,462 LoFTR matches, while LoFTR confidence has AUC 0.875 but confidence-filtered PnP still does not beat the 156.3mm external cache. Re-running all-match LoFTR PnP from the step+feature poses worsens to 171.4mm median. The real-image LoFTR+render-depth teacher exporter now preserves `pnp_inlier_mask` for all-depth-valid stores; full ShopFacade/OldHospital test top5 stores were exported with 204k/335k labeled correspondences and confidence AUC 0.621/0.766. A first confidence+query-position reliability head gives only weak same-scene AUC gains and mixed cross-scene transfer. Pose-level PnP filtering shows small external-baseline gains: ShopFacade best confidence filter reaches 55.7/103.6mm from 65.9/108.3mm, while OldHospital best cross-model filter only reaches 200.2/402.0mm from 206.8/421.8mm. Appending the step-gated pose as compact candidate 16 improves the top15 oracle only from 0.1337m to 0.1315m; Stage3a scorer improves 0.3172m->0.3160m and quality-s20 improves 0.3136m->0.3117m, still far worse than the external step-gated pose alone | pivot artifact implemented; old scorer/reranker/inlier gates and POFD residual correspondence filtering no-go; external pose-step+feature LoFTR is the best mean baseline/teacher but not POFD main contribution |
| Public SOTA comparison | public GS-CPR / GS-SMC / GSVisLoc / HLoc protocol audit plus local feature_retrieval baselines are added to the Stage4 plan; local multi-render top50 is 16.1-16.7cm median, local oracle top50 is 7.8cm, the single-render external LoFTR+PnP one-pass full182 baseline reaches 15.6cm median, and the new no-GT step+feature external cache reaches 15.6cm median / 25.6cm mean. HLoc is now cloned under `third_party/Hierarchical-Localization`, smoke-tested, and patched for a labeled Python 3.8 compatibility Cambridge run. Five-scene Cambridge HLoc completed and was exported into project pose-init caches: ShopFacade 103/103 localized, 0.042m / 0.206deg median, 99.03% at 50cm/5deg; OldHospital 182/182, 0.144m / 0.309deg, 86.81%; KingsCollege 343/343, 0.114m / 0.210deg, 91.25%; GreatCourt 760/760, 0.175m / 0.107deg, 80.39%; StMarysChurch 530/530, 0.075m / 0.224deg, 99.25%. The existing rendered-RGB LoFTR refiner does not pass the public-init improvement gate on HLoc: OldHospital median worsens 144.0mm -> 167.4mm despite mean improvement, and ShopFacade worsens 41.6mm/63.5mm median/mean -> 56.7mm/7.10m with only 28/103 PnP successes. POFD Stage4 on HLoc exact no-ops: OldHospital cost 0.2558m -> 0.2558m and ShopFacade 0.0641m -> 0.0641m, both with 0.0 solver success and 0.0 accepted updates. ACE, GLACE, MASt3R, and DUSt3R protocols remain incomplete. POFD Stage4 remains about 21.7cm median at best and does not improve the external or HLoc caches | protocol/baseline audit strengthened; HLoc five-scene Cambridge baseline done; HLoc/POFD refinement not SOTA |
| Full validation / uncapped validation | Stage4 top50 real-init full182 trans-only follow-up is logged (+5.61mm best mean gain; best median about 21.7cm/0.41deg); correction-reward controlled q50/q25/q10 full182 evals are logged (q50 0.179m/top1 0.874/Spearman 0.481, q25 0.071m, q10 0.028m); multi-scene public protocol and multi-seed validation are not complete | partial, not sufficient |
| Multi-seed stability | no 3-seed POFD Stage4 matrix | missing |
| At least one additional scene / dataset | workspace now contains Cambridge `ShopFacade` RADIO dual v68 cache, DCFF train/smoke/pilot configs, smoke/pilot/formal reconstruction configs, an 8-iteration smoke checkpoint, a 2GPU 100-iteration pilot checkpoint, and a longer 2GPU run whose best checkpoint is iter 500 / loss 1.7991 with 231/231 train + 103/103 test camera-to-feature matches. Cached RADIO and raw NetVLAD inits are weak, but the real-image LoFTR+render-depth PnP path over NetVLAD top5 now gives a strong external initializer: selected 84.5mm median / 130.4mm mean, reprojection-median selected 84.5mm / 111.1mm, and top5 oracle 43.2mm / 71.4mm. HLoc is stronger still at 41.6mm / 0.204deg median. Render-at-init RGB LoFTR diverges from both the 84.5mm external init and the 41.6mm HLoc init; same-image HLoc query-mode smoke16 reaches near-zero error, so the viable second-scene route is external real-image matching plus rendered depth. Full103 Stage4 POFD on the reprojection-median init exactly no-ops at 0.1120m cost with 0.0 solver success and 0.0 accepted updates; Stage4 on HLoc also no-ops at 0.0641m cost with 0.0 solver success and accepted updates. Train/test real-image LoFTR+depth teacher stores were exported, but direct teacher-quality tuning worsens selection, low-res teacher-pair-flow regresses Stage4 to 0.514m, and high-res subpixel teacher-pair-flow regresses to 0.211m at confidence 0; confidence thresholds 0.002/0.005/0.010 are safe only by no-op. This is useful second-scene evidence, but no promotable ShopFacade POFD evaluation table exists yet | partial prerequisite |
| Failure analysis | q50 failure/all-dump and fixed-rotation top16 dumps exist | partial; needs learned/continuous fix |
| Tests after code changes | `pytest tests/test_nvs_pose_feature_adapter.py -q -k 'teacher_pair_flow'` -> 5 passed; compact cache-tool/prior/fallback/scene-coordinate targeted regression -> 7 passed; broad targeted suite over checkpoint I/O, Stage4 adapter, LoFTR refine, init-cache export, localization protocol, scene-coordinate PnP, correspondence export, and the new cache tools -> 148 passed; latest cache-tool/RADIO-loader/DDP-init targeted regressions pass; feature-score/gate tests -> 9 passed; latest LoFTR correspondence mask + per-match feature-score tests -> 15 passed across the two targeted files; real-image LoFTR+depth teacher mask tests -> 6 passed; ShopFacade DCFF smoke, 2GPU pilot, and longer 2GPU map training completed/stopped with checkpoints; HLoc compatibility tests -> 5 passed; latest focused correspondence/filter/init-export/refine tests -> 27 passed, 2 warnings; latest `py_compile`, HLoc/init-cache artifact load check, HLoc Stage4 config/result check, `git diff --check`, process check, and final `nvidia-smi` passed | current targeted regression passes |

## Current Decision

The objective is not complete. The strongest selection numbers come from a
controlled-cache correction reward, not from the paper-preferred single-render
continuous refinement path. The correction reward also fails the rank
calibration gate on both val128 and full182, and the only offline calibration
that reaches oracle selection imports render-LoFTR/PnP cache metrics, so it
cannot be promoted as the POFD main method. The expert-file Stage4 stop/pivot
rule is now triggered for the
POFD-only continuous proposal path: multiple controlled configs improve far
less than 10mm, the real-init benchmark does not beat the initializer by a
publishable margin, and POFD updates on top of the strongest external refined
cache either harm or no-op. The workspace has LoFTR correspondence tooling,
staged HLoc, and now an official MASt3R checkout at `f5209af` with DUSt3R
`3cc8c88` and CroCo `d7de070`, but no native MASt3R/DUSt3R project adapter,
result table, completed ACE/GLACE public pose table, or external
MASt3R/DUSt3R comparator. Official MASt3R core/visloc imports work with the
local `PYTHONPATH`, optional visloc dependencies are installed, the official
checkpoint is downloaded, and a bundled-pair GPU smoke produces matches. The
immediate practical pivot is therefore a designed GS-CPR-style external
correspondence branch, with MASt3R or LoFTR supplying correspondences and POFD
restricted to uncertainty/reranking rather than another unsafe direct update
source. The first
pivot diagnostic is now implemented: simple LoFTR quality gating is insufficient,
top51 is too memory-heavy for the current pair-matcher training path, compact
top15+refined is trainable but only weakly positive on s64 and still weak on
full182, and direct score-prior/fallback uncertainty gates do not produce a
promotable main metric. A genuinely new uncertainty/reranking head or an
external MASt3R/DUSt3R-style correspondence refiner is required before this can
become a contribution; otherwise LoFTR+PnP must remain an external
baseline/teacher rather than the POFD core method.

## Next Gate

The virtual pose-energy acceptance gate, per-update confidence sweep, and
real-init pair-flow fine-tune have been implemented/evaluated, but none is
reliable enough to promote. The current Stage4 solver proposal generator has
only millimeter-scale upside on val128. Translation-only updates confirm that
rotation drift is one failure mode, but the best mean-cost run is still only
+6.00mm on val128 and +5.61mm on full182, with small-error and 10cm/50cm
success regressions at the larger two-step setting; full182 one-step and
three-step checks are only +3.96mm and +4.19mm. The full182 generated-update
oracle is also far below the 20% target, so the current proposal generator
itself is the bottleneck rather than only the accept gate. Median and baseline
audits reach the same conclusion: Stage4 full182 is about 21.7cm median at
best, behind local multi-render render-LoFTR/PnP baselines at 16.1-16.7cm and
the public HLoc Hospital reference at 15cm. The old PoseEnergy
residual/selector branch has already failed q50 eval, and the stricter
single-render residual pivot is only a sub-millimeter full182 gain after the
validation cap is removed. The new POFD-DenseFlow proposal source is
implemented, but its first two-seed val128 smoke is negative and should be
treated as a no-go rather than expanded to full182. A direct single-render
featuremetric GN baseline over the current v68 DCFF/RADIO features also
regresses the real render-LoFTR initializer on a 64-sample smoke, so the
missing ingredient is not merely a Gauss-Newton wrapper around existing
features. In contrast, an external render-at-init LoFTR+PnP baseline is
positive on full182 and reaches 15.6cm median translation, which makes it the
current SOTA-style comparator and a likely teacher/proposal signal. Feeding
that refined-pose cache back into current POFD Stage4 does not solve the
problem: top50/w2 actively harms the strong init, while all-margin q50 is a
safe no-op. A direct render-at-init LoFTR teacher-flow distillation branch has
also failed in integer, expected-offset, and current subpixel forms:
low-resolution grids quantize the teacher signal away, expected integer offsets
still regress mean cost, dense subpixel expected flow collapses without a
confidence gate, and the gate is a no-op. Promotion now requires either a
genuinely new POFD-owned proposal target or the expert-file pivot toward a
GS-CPR-style external-correspondence refiner with POFD uncertainty/reranking.
Given the current codebase, that means starting from the existing
render-at-init LoFTR+PnP path unless MASt3R/DUSt3R is added as a new dependency.
The multi-scene prerequisite is now partially unblocked by the ShopFacade
RADIO v68 cache, DCFF smoke/formal map wiring, a strong external
LoFTR+render-depth PnP initializer, render-at-init RGB failure diagnostics, and
a completed five-scene Cambridge HLoc baseline. This does not change the
decision: there is still no full second-scene POFD improvement table or
completed ACE/GLACE/MASt3R/DUSt3R SOTA comparison. The new HLoc bridge lowers
the setup barrier and provides public initializer baselines, but it is not
evidence of POFD SOTA performance.
