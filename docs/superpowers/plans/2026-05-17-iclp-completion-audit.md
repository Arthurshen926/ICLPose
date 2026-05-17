# ICLPose Expert-File Completion Audit

**Objective audited:** follow `ChatGPT-ICLPose (1).md`, continue optimization with
parallel GPU experiments where useful, and reach top-journal submission standard.

**Audit result:** not complete. The current repo now has extensive diagnostics,
external-LoFTR baselines, compact reranking probes, verified no-go evidence,
five-scene HLoc Cambridge pose tables, a staged official MASt3R/DUSt3R
checkout, and a first ShopFacade second-scene data/map/init smoke artifact. It still does
not satisfy the expert-file requirements for a clean single-render continuous
refinement method or public SOTA-grade evaluation.

## Prompt-To-Artifact Checklist

| Requirement / task from `ChatGPT-ICLPose (1).md` | Current artifact/evidence | Status |
|---|---|---|
| Do not use GT pose, GT camera center, GT candidate append, or oracle retrieval in deploy evaluation | Deploy eval configs use real init caches; GT/oracle variants are documented as diagnostics. Latest compact top15+refined appends an external refined proposal, so it is diagnostic/fallback, not clean deploy mainline | partial |
| Do not render K candidate poses as main inference path | Stage4 single-render paths, denseflow, featuremetric GN, and LoFTR refined-cache diagnostics were evaluated. Best current numbers still come from external LoFTR or compact candidate reranking, not a promoted single-render method | not met |
| Candidate rendering only for ablation/oracle/fallback | Candidate caches and top15/top50 experiments are documented as diagnostics. No clean main method supersedes them yet | partial |
| Preserve oracle/deploy/ablation separation | Plan/audit docs separate Stage4 deploy, controlled-cache oracle, external LoFTR, and append-refined diagnostics | partial |
| Keep POFD as one localization feature; do not reintroduce trainable topK selector as main method | Current branches use POFD/pair-matcher/denseflow/featuremetric tools, no trainable topK selector promoted | met |
| Add single-render CPR module | Existing Stage4 render-once refinement path and `denseflow` source exist; best full182 gain is only about +5.61mm and denseflow smokes regress | implemented, no-go |
| Pair matcher exposes single-render correspondence / offset signal | Stage4 pair-matcher correspondence path, pair-flow loss, expected-offset and subpixel teacher-flow branches exist; teacher-flow variants regress or no-op | implemented, no-go |
| Robust 2D-3D pose solver with damping, clamp, diagnostics | Stage4 robust update path and diagnostics exist. The latest step+feature-cache check shows the default `stage4_min_points=64` hides the issue by rejecting updates; reducing to 16 yields solver success but catastrophic drift (smoke32 0.059m -> 0.829m). A strict 5cm/0.5deg accept gate on full182 accepts only 2.2% of updates and slightly regresses 0.25630m -> 0.25648m, while the proposed ungated updates average -0.414m cost gain | implemented, weak/no-go |
| Iterative render-once refinement, 3-5 iterations, render count | POFD Stage4 iter1/iter2/iter3 full182 checks logged; iter2 translation-only is best but only +5.61mm and success regresses. External LoFTR render-at-current iter2/iter3 is now implemented and exported; raw iter2/iter3 improves mean slightly (264.5/264.3mm vs 273.4mm) but worsens median (175.5/181.9mm vs 156.3mm) and improves fewer samples. A no-GT pose-step gate using one-pass->iter2 step >=0.20m selects iter3 on 35/182 rows and reaches 159.6mm median / 256.2mm mean with 10/25/50cm success 0.346/0.654/0.896, but this remains an external LoFTR cache result | implemented; external baseline improved, POFD no-go |
| Optional render-once virtual trust-region fallback | Implemented and tested; smoke regressions and no stable promote setting | implemented, rejected |
| Training losses: offset CE/GeoNCE, robust update/final pose, weak teacher anchor, candidate score auxiliary only | Pair-flow, teacher-flow, denseflow, and reranking losses were implemented/tested. Current losses do not yield publishable refinement or reranking | partial/no-go |
| Evaluation: controlled q10/q25/q50 | Multiple q10/q25/q50 Stage3/Stage4 configs and audits exist. The latest correction-reward full182 controlled-cache eval reaches q50 0.179m/top1 0.874/oracle gap 0.050m/succ@25 0.890, q25 0.071m, and q10 0.028m, but q50 Spearman is only 0.481 and the branch is a cached-step reward rather than continuous refinement. q50 remains below promotion criteria for the paper-preferred continuous method | partial |
| Evaluation: real initial poses, ACE/GLACE/HLoc if available | OldHospital real NetVLAD/render-LoFTR caches are used. HLoc has now been cloned under `third_party/Hierarchical-Localization`, import-smoked, patched for the local Python 3.8/`pycolmap==3.12.5` compatibility path, and run on all five official retriangulated Cambridge model scenes. ShopFacade localizes 103/103 test images with 0.042m / 0.206deg median error and 99.03% at 50cm/5deg. OldHospital localizes 182/182 with 0.144m / 0.309deg and 86.81%. KingsCollege localizes 343/343 with 0.114m / 0.210deg and 91.25%. GreatCourt localizes 760/760 with 0.175m / 0.107deg and 80.39%. StMarysChurch localizes 530/530 with 0.075m / 0.224deg and 99.25%. These HLoc results were converted into project pose-init caches for all five scenes. Rendering at the HLoc pose and re-running the existing RGB LoFTR refiner does not pass the public-init improvement gate: OldHospital median worsens 144.0mm -> 167.4mm, and ShopFacade median/mean worsen 41.6/63.5mm -> 56.7mm/7.10m with only 28/103 successful PnP solves. POFD Stage4 on the HLoc caches also exact no-ops on the two scenes with current fields: OldHospital cost 0.2558m -> 0.2558m and ShopFacade 0.0641m -> 0.0641m, both with 0.0 solver success and 0.0 accepted updates. MASt3R is now staged from the official repo under `third_party/mast3r` with pinned DUSt3R/CroCo submodules; official visloc imports, synthetic PnP, CLI help, and a local-checkpoint GPU pair smoke pass, but no Cambridge MASt3R adapter/result exists yet. ACE/GLACE remain absent | partial public eval; HLoc five-scene Cambridge baseline done; HLoc refinement/POFD no-go; MASt3R dependency/checkpoint smoke done |
| Baselines: Stage3a, RADIO raw alignment, GS-CPR-style render+MASt3R if feasible, HLoc/ACE init | Stage3a and local featuremetric/LoFTR baselines are logged. Official MASt3R is now cloned at `f5209af` with DUSt3R `3cc8c88` and CroCo `d7de070`, and its README exposes a Cambridge `visloc.py` path that uses MASt3R correspondences plus 2D-3D PnP. No native project adapter or MASt3R result table exists yet. HLoc now has completed five-scene Cambridge public-baseline runs, but ACE/GLACE pose tables and external MASt3R/DUSt3R comparisons are not yet complete. The immediate bridge remains the existing real-image LoFTR+render-depth PnP path, now with all-depth-valid teacher correspondence masks preserved for future GS-CPR-style uncertainty training | missing/partial bridge; MASt3R adapter design required |
| Metrics: median trans/rot, success thresholds, init→final, render_count, runtime, solver inliers/residual/failure | Most internal metrics are logged for Stage4 and LoFTR. Runtime/render-count/public protocol tables are not complete enough for paper claims | partial |
| External correspondence + POFD uncertainty/reranking pivot | LoFTR refined cache, gate tool, append tool, compact top15+refined caches, score prior, identity fallback, s20/s80 rerank probes completed. Best full182 compact quality rerank is only 0.3136m vs 0.3172m baseline, while external refined-only is about 0.273m mean. The new external iterative LoFTR cache reduces raw iter3 mean to about 0.264m but loses median/10cm robustness; simple inlier/raw-match gates cannot choose it reliably. A new pose-step cache gate improves the external baseline to 0.256m mean / 0.160m median. DCFF feature consistency remains a no-go as an uncertainty signal: pose-cache residual delta has only 0.009 correlation with true iter3 gain, and the new all-match per-correspondence full182 scorer finds lower-residual-is-inlier AUC 0.484 (query mean 0.444), with outliers slightly lower residual than inliers. LoFTR confidence is a strong inlier classifier (AUC 0.875), but confidence/residual-filtered PnP sweeps still do not beat the step+feature cache; all-match LoFTR PnP is 0.171m median / 0.257m mean versus the 0.156m / 0.256m init. Real-image LoFTR+render-depth teacher payloads now preserve `pnp_inlier_mask` for all-depth-valid exports; full test top5 allvalid stores were exported for ShopFacade and OldHospital with 103/103 and 182/182 queries, 515/515 and 910/910 PnP-success candidates, and 204k/335k labeled correspondences. A non-leaky confidence+query-position logistic reliability head is only weak: same-scene AUC rises 0.621->0.640 on ShopFacade and 0.766->0.776 on OldHospital, but cross-scene is mixed and Brier worsens. q50 all-margin Stage4 on that cache no-ops at strict thresholds or regresses when thresholds are relaxed. Appending the step-gated pose to the compact top15 cache improves oracle 0.1337m->0.1315m and quality-s20 0.3136m->0.3117m, still far from the external baseline and not a POFD contribution | implemented; external baseline improved, POFD no-go |
| Top-journal standard | Requires a clean main method beating strong baselines, multi-scene/public protocol, stability, runtime, and failure analysis. Current strongest result is external LoFTR baseline/teacher, and the best controlled-cache selection result depends on an explicit correction-step reward rather than a learned continuous POFD contribution. HLoc now strengthens the public initializer baseline, but the existing rendered-RGB LoFTR refiner does not reliably improve it. ShopFacade now has RADIO v68 cache, DCFF smoke/formal map artifacts, a strong external real-image LoFTR+render-depth PnP initializer, render-at-init RGB failure diagnostics, a full103 Stage4 no-op check, teacher correspondence stores, and teacher-pair-flow closed-loop evals; no second-scene POFD result is promotable yet | not met |

## Current Blocking Gaps

1. **Main-method gap:** current POFD single-render correspondence and solver
   proposals do not improve enough over real init, and compact reranking does
   not beat the external LoFTR refined baseline. The full182 correction-reward
   controlled cache is a useful selection baseline, but it still fails q50 rank
   calibration and does not solve the single-render refinement requirement.
2. **Uncertainty/solver gap:** current POFD/DCFF residuals are not reliable
   correspondence-quality signals. Full182 all-match diagnostics show DCFF
   residual is anti-useful for inlier selection (AUC below 0.5), and LoFTR
   confidence filtering alone does not recover a better pose cache.
3. **External-correspondence gap:** the repo has LoFTR tooling and now an
   official MASt3R/DUSt3R checkout, but no native project
   MASt3R/DUSt3R/GS-CPR-style matcher entrypoint or result table. HLoc is now
   staged under `third_party/Hierarchical-Localization`, but ACE and GLACE
   remain unavailable in the workspace.
4. **Evaluation gap:** no complete ACE/GLACE public-init comparison, no
   multi-seed stability table, and no promoted public POFD refinement table.
   The HLoc Cambridge pipeline is now unblocked for a labeled Python 3.8
   compatibility run using the official retriangulated model layout.
   ShopFacade, OldHospital, KingsCollege, GreatCourt, and StMarysChurch HLoc
   have completed, but ACE/GLACE remain absent.
   ShopFacade is now partially prepared through feature export, DCFF map
   verification, real-init/refinement smoke diagnostics, and public HLoc pose
   tables exist for all five Cambridge model scenes, but the initializer basin is still
   too weak for a clean second-scene POFD claim.
5. **Paper-positioning gap:** candidate reranking remains a diagnostic/fallback;
   it is not yet a clean continuous refinement contribution.

## HLoc Public Protocol Setup

Status on 2026-05-17:

- Official HLoc was cloned recursively to
  `third_party/Hierarchical-Localization` at commit `c13273b`.
- The README path is `git clone --recursive`, `python -m pip install -e .`;
  it lists `pycolmap>=3.13.0` and notes that system COLMAP is no longer
  required for HLoc v1.3+.
- Project env smoke: `hloc==1.5`, `pycolmap==3.12.5`, `h5py==3.11.0`,
  `cv2==4.13.0`, `torch==1.13.1+cu116`, CUDA available on 2 GPUs. Importing
  `hloc.localize_sfm`, `hloc.extract_features`, `hloc.match_features`, and
  `hloc.pipelines.Cambridge.pipeline` succeeds, but HLoc warns that pycolmap is
  below the declared requirement. Treat this as a compatibility smoke only.
- Official-dependency smoke: conda env `hloc-py310` uses Python `3.10.20`,
  `hloc==1.5`, `pycolmap==4.0.4`, `h5py==3.16.0`, and `cv2==4.13.0`;
  `import hloc.localize_sfm` succeeds without the pycolmap warning. This env is
  still minimal and lacks full feature-extraction dependencies such as Torch,
  TorchVision, Kornia, SciPy, Matplotlib, Plotly, GDown, and LightGlue.
- A direct editable install in the project Python 3.8 env failed because
  `pycolmap>=3.13.0` has no Python 3.8 wheel. The optional LightGlue git
  dependency also hit a transient GitHub TLS clone error during pip install.
- The official retriangulated Cambridge layout was downloaded/extracted under
  `/hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px`
  and symlinked into `result/result/hloc/cambridge_input`.
- The Python 3.8 compatibility run needed two local HLoc patches:
  `HLOC_MATCH_NUM_WORKERS=0` support in `hloc/match_features.py` to avoid
  shared-memory DataLoader bus errors, and pycolmap path/database compatibility
  helpers in `hloc/triangulation.py`.
- ShopFacade completed with:
  `CUDA_VISIBLE_DEVICES=0 HLOC_MATCH_NUM_WORKERS=0 python -m hloc.pipelines.Cambridge.pipeline --scenes ShopFacade --dataset result/result/hloc/cambridge_input --outputs result/result/hloc/cambridge_py38_compat --num_covis 20 --num_loc 10 --overwrite`.
  Output:
  `result/result/hloc/cambridge_py38_compat/ShopFacade/results.txt` with
  103/103 localized queries, 0.042m / 0.206deg median error, and localization
  rates 0.97% at 1cm/1deg, 12.62% at 2cm/2deg, 33.01% at 3cm/3deg,
  59.22% at 5cm/5deg, 96.12% at 25cm/2deg, and 99.03% at 50cm/5deg.
- OldHospital completed with:
  `CUDA_VISIBLE_DEVICES=1 HLOC_MATCH_NUM_WORKERS=0 python -m hloc.pipelines.Cambridge.pipeline --scenes OldHospital --dataset result/result/hloc/cambridge_input --outputs result/result/hloc/cambridge_py38_compat --num_covis 20 --num_loc 10 --overwrite`.
  Output:
  `result/result/hloc/cambridge_py38_compat/OldHospital/results.txt` with
  182/182 localized queries, 0.144m / 0.309deg median error, and localization
  rates 0.00% at 1cm/1deg, 0.55% at 2cm/2deg, 2.75% at 3cm/3deg,
  8.24% at 5cm/5deg, 66.48% at 25cm/2deg, and 86.81% at 50cm/5deg.
- KingsCollege completed with:
  `CUDA_VISIBLE_DEVICES=0 HLOC_MATCH_NUM_WORKERS=0 python -m hloc.pipelines.Cambridge.pipeline --scenes KingsCollege --dataset result/result/hloc/cambridge_input --outputs result/result/hloc/cambridge_py38_compat --num_covis 20 --num_loc 10 --overwrite`.
  Output:
  `result/result/hloc/cambridge_py38_compat/KingsCollege/results.txt` with
  343/343 localized queries, 0.114m / 0.210deg median error, and localization
  rates 0.58% at 1cm/1deg, 2.04% at 2cm/2deg, 6.12% at 3cm/3deg,
  15.74% at 5cm/5deg, 73.47% at 25cm/2deg, and 91.25% at 50cm/5deg.
- GreatCourt completed with:
  `CUDA_VISIBLE_DEVICES=0 HLOC_MATCH_NUM_WORKERS=0 python -m hloc.pipelines.Cambridge.pipeline --scenes GreatCourt --dataset result/result/hloc/cambridge_input --outputs result/result/hloc/cambridge_py38_compat --num_covis 20 --num_loc 10 --overwrite`.
  Output:
  `result/result/hloc/cambridge_py38_compat/GreatCourt/results.txt` with
  760/760 localized queries, 0.175m / 0.107deg median error, and localization
  rates 0.66% at 1cm/1deg, 2.50% at 2cm/2deg, 5.79% at 3cm/3deg,
  12.11% at 5cm/5deg, 64.21% at 25cm/2deg, and 80.39% at 50cm/5deg.
- StMarysChurch completed with:
  `CUDA_VISIBLE_DEVICES=1 HLOC_MATCH_NUM_WORKERS=0 python -m hloc.pipelines.Cambridge.pipeline --scenes StMarysChurch --dataset result/result/hloc/cambridge_input --outputs result/result/hloc/cambridge_py38_compat --num_covis 20 --num_loc 10 --overwrite`.
  Output:
  `result/result/hloc/cambridge_py38_compat/StMarysChurch/results.txt` with
  530/530 localized queries, 0.075m / 0.224deg median error, and localization
  rates 0.57% at 1cm/1deg, 7.36% at 2cm/2deg, 15.09% at 3cm/3deg,
  32.08% at 5cm/5deg, 95.66% at 25cm/2deg, and 99.25% at 50cm/5deg.

HLoc-to-project cache bridge on 2026-05-17:

- Parsed the five `results.txt` pose tables and verified that the converted
  qvec/tvec project `w2c` matrices reproduce the HLoc metrics exactly.
- Exported single-candidate init caches under
  `result/result/feature_extract/pose_init_exports`:
  `shopfacade_hloc_superpoint_superglue_test_20260517.npz`,
  `oldhospital_hloc_superpoint_superglue_test_20260517.npz`,
  `kingscollege_hloc_superpoint_superglue_test_20260517.npz`,
  `greatcourt_hloc_superpoint_superglue_test_20260517.npz`, and
  `stmaryschurch_hloc_superpoint_superglue_test_20260517.npz`.
  The summary JSON is
  `cambridge_hloc_superpoint_superglue_test_20260517_summary.json`.
- HLoc render-at-init RGB LoFTR refinement was tested where DCFF reconstruction
  configs exist. OldHospital HLoc init is 144.0mm / 0.306deg median,
  255.0mm mean, and 86.81% at 50cm/5deg. The refined cache is
  167.4mm / 0.264deg median, 235.0mm mean, and 91.76% at 50cm/5deg
  with 182/182 PnP successes and 2791 median inliers. This improves mean and
  50cm/5deg but worsens median translation, so it is not a stable initializer
  improvement.
- ShopFacade HLoc init is 41.6mm / 0.204deg median, 63.5mm mean, and 99.03%
  at 50cm/5deg. The refined/fallback cache worsens to 56.7mm median,
  7.10m mean, and 71.84% at 50cm/5deg; the successful-refinement subset itself
  has 24.0m / 56.9deg median error with only 28/103 PnP successes. This is a
  clear no-go for the existing rendered-RGB refiner on strong HLoc inits.
- A same-image `ref_mode=query` smoke verifies the geometry path is not the
  blocker: ShopFacade smoke16 reaches 0.016mm / 0.000deg median and
  OldHospital smoke16 reaches 0.77mm / 0.000deg median. The failure is
  localized to rendered RGB query/reference matching.
- Added HLoc Stage4 eval configs:
  `feature_extract/configs/pofd_stage4_oldhospital_hloc_superpoint_superglue_full182_eval.yaml`
  and
  `feature_extract/configs/pofd_stage4_shopfacade_hloc_superpoint_superglue_full103_eval.yaml`.
  Full-split results are exact no-ops. OldHospital:
  `result/result/feature_extract/eval_pofd_stage4_oldhospital_hloc_superpoint_superglue_full182_20260517/stage4_eval_summary.json`
  has init/pred cost 0.2558m, translation 0.2550m, rotation 0.4108deg,
  solver success 0.0, accepted update 0.0, and 3 renders/query. ShopFacade:
  `result/result/feature_extract/eval_pofd_stage4_shopfacade_hloc_superpoint_superglue_full103_20260517/stage4_eval_summary.json`
  has init/pred cost 0.0641m, translation 0.0635m, rotation 0.3092deg,
  solver success 0.0, accepted update 0.0, and 3 renders/query.

Next public-baseline gate: compare the five-scene Cambridge HLoc initializer
table against POFD Stage4 and external LoFTR baselines before adding any public
SOTA claim. The current HLoc refinement result does not satisfy the
"real/public init improves initializer" promotion gate, and current POFD
Stage4 does not improve HLoc either.

## Multi-Scene Progress

The workspace contains the five official retriangulated Cambridge model scenes
under `/hy-tmp/Cambridge_stdloc`: `GreatCourt`, `KingsCollege`, `OldHospital`,
`ShopFacade`, and `StMarysChurch`. HLoc public-baseline pose tables now exist
for all five; the first non-OldHospital scene prepared for POFD/DCFF work in
this run is still `ShopFacade`.

Concrete ShopFacade artifacts:

- RADIO dual v68 cache:
  `result/result/feature_extract/features_radio_dual_v68_align_stop640/cambridge_shopfacade`
  with 334 `fine_geo`, 334 `coarse_sem`, 334 summaries, and
  `summary_matrix.pt` shape `(334, 2560)`.
- Train/test split check: the Cambridge split files contain two header lines
  each; after excluding headers, the scene has 231 train and 103 test frames.
- DCFF configs:
  `feature_field/configs/dcff_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_2gpu.yaml`,
  `feature_field/configs/dcff_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_pilot100_2gpu.yaml`,
  and `feature_field/configs/smoke_dcff_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68.yaml`.
- Reconstruction configs for smoke/pilot/formal checkpoints:
  `feature_field/configs/reconstruction_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_smoke.yaml`,
  `feature_field/configs/reconstruction_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_pilot100.yaml`,
  and
  `feature_field/configs/reconstruction_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68.yaml`.
- Smoke training result:
  `python -m feature_field.train --config feature_field/configs/smoke_dcff_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68.yaml`
  completed 8 iterations, matched 231/231 train and 103/103 test cameras to
  cached features, and saved best/latest checkpoints under
  `result/result/feature_field/smoke_dcff_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68/checkpoints`.
- 2GPU pilot result:
  `torchrun --standalone --nproc_per_node=2 -m feature_field.train --config feature_field/configs/dcff_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_pilot100_2gpu.yaml`
  completed 100 iterations with global batch 10, about 4.3 iter/s by the end,
  best total loss 3.1616 at iter 100, and validation smoke loss 3.0027 on 10
  frames. It saved best/latest checkpoints under
  `result/result/feature_field/dcff_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_pilot100_2gpu/checkpoints`.
  The pilot also exposed intermittent guarded non-finite gradient skips, so
  the long training config should be treated as a next experiment requiring
  conservative AMP/LR validation rather than a finished stable map.
- 2GPU longer training result:
  `torchrun --standalone --nproc_per_node=2 -m feature_field.train --config feature_field/configs/dcff_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68_2gpu.yaml`
  was run after reducing per-GPU batch size to 5. It saved `best.pth` at iter
  500 with best total loss 1.7991 and `latest.pth` at iter 2000 with
  10-frame validation smoke loss 2.4987. Continuing to iter 2434 did not
  improve the best checkpoint and accumulated 198 guarded non-finite gradient
  skips, so the run was stopped and the reconstruction config
  `feature_field/configs/reconstruction_cambridge_shopfacade_processed_cached64_v11_residual_spatial_v68.yaml`
  now points to the iter-500 best checkpoint.
- Real-init/export diagnostics:
  cached RADIO coarse/student top50 banks are weak on ShopFacade: top1 median
  translation is about 3.29m and the top50 spatial oracle is about 0.89m.
  NetVLAD is better but still coarse, with top1 median/mean 1.40/1.79m and
  48.5% recall at 10deg/2m; its top50 spatial oracle is about 0.55m median.
- External real-image LoFTR+render-depth PnP initializer:
  the existing `feature_retrieval.render_loftr_pnp_init_export` path was run
  over NetVLAD top5 in two GPU shards and merged into
  `result/result/feature_extract/pose_init_exports/shopfacade_netvlad_renderloftrpnp_top5_test_20260517.npz`.
  All 515 candidates produced PnP successes. The selected init reaches
  84.5mm median / 130.4mm mean translation with 0.403deg median rotation and
  0.650/0.922 success at 10cm/5deg and 25cm/10deg. The top5 translation oracle
  is 43.2mm median / 71.4mm mean. A deployable reprojection-median selector
  exports
  `result/result/feature_extract/pose_init_exports/shopfacade_netvlad_renderloftrpnp_top5_reprojmedian_test_20260517.npz`
  and improves the selected mean to 111.1mm while keeping the 84.5mm median.
- Render-at-init LoFTR smoke diagnostics:
  coarse/student top50 caches diverge on 32 test frames
  (2.60m init median -> 35.5m final median, 11/32 PnP successes). NetVLAD
  top50 starts closer but still diverges (1.18m init median -> 26.6m final
  median, 3/32 successes). A GT-pose/query-mode sanity check on the same map
  succeeds on 32/32 frames and refines 65.3mm synthetic noise to near zero,
  which localizes the failure to the real-init basin/correspondence selection
  rather than a totally broken ShopFacade renderer/map path. Re-running
  render-at-init LoFTR from the strong 84.5mm LoFTR+render-depth PnP cache still
  diverges: iter1/iter2 both produce 27/103 successes, 28.8m final median
  translation, and 0 improved samples. This indicates ShopFacade rendered RGB
  is not a reliable LoFTR reference even when the pose basin is good. The new
  HLoc public init is even stronger at 41.6mm / 0.204deg median, but the same
  render-at-init RGB LoFTR refiner still collapses: the refined/fallback cache
  has 56.7mm median, 7.10m mean, 71.84% at 50cm/5deg, and only 28/103 PnP
  successes. Same-image `ref_mode=query` smoke16 reaches near-zero error, so
  real-image matching plus rendered depth is the currently viable external route.
- Stage4 POFD on the strong ShopFacade external init:
  `feature_extract/configs/pofd_stage4_shopfacade_realinit_renderloftrpnp_top5_reprojmedian_full103_eval.yaml`
  was added after fixing the input height to 1088 for query-student stride
  alignment. The full103 eval in
  `result/result/feature_extract/eval_pofd_stage4_shopfacade_renderloftrpnp_reprojmedian_full103_20260517`
  exactly preserves the external init: init/pred cost 0.1120m, init/pred
  translation 0.1111m, init/pred rotation 0.5386deg, solver success 0.0,
  accepted updates 0.0, and 5/10/25/50cm success
  0.262/0.631/0.913/0.990. This is useful second-scene evidence, but still a
  POFD no-op rather than a contribution.
- Stage4 POFD on the stronger public HLoc init:
  `feature_extract/configs/pofd_stage4_shopfacade_hloc_superpoint_superglue_full103_eval.yaml`
  exactly no-ops as well. The full103 eval in
  `result/result/feature_extract/eval_pofd_stage4_shopfacade_hloc_superpoint_superglue_full103_20260517`
  has init/pred cost 0.0641m, init/pred translation 0.0635m, init/pred
  rotation 0.3092deg, solver success 0.0, accepted update 0.0, render count
  3.0, and unchanged 5/10/25/50cm success 0.592/0.903/0.961/0.990.
- Real-image LoFTR+render-depth teacher stores:
  `result/result/feature_extract/teacher_corr/shopfacade_train_netvlad_renderloftrpnp_top5_20260517`
  contains 229/231 train files and
  `result/result/feature_extract/teacher_corr/shopfacade_test_netvlad_renderloftrpnp_top5_20260517`
  contains 103/103 test files, each capped at 512 correspondences. The train
  export solves 1142/1155 candidates and the selected train pose is
  64.9mm median / 220.9mm mean; the test export solves 515/515 candidates and
  matches the 84.5mm / 130.4mm selected external init. These stores are useful
  teacher artifacts, but they still derive from external real-image matching.
  The payload code now also preserves `pnp_inlier_mask` when exporting
  all-depth-valid correspondences. After a 2-query smoke, full top5 allvalid
  test stores were exported in parallel on GPU0/GPU1:
  `shopfacade_test_netvlad_renderloftrpnp_top5_allvalid_20260517` contains
  103 files, 515/515 PnP-success candidates, 204,054 labeled points, 179,740
  inliers, 24,314 outliers, inlier ratio 0.881, and confidence AUC 0.621.
  `oldhospital_test_netvlad_renderloftrpnp_top5_allvalid_20260517` contains
  182 files, 910/910 PnP-success candidates, 334,559 labeled points, 296,450
  inliers, 38,109 outliers, inlier ratio 0.886, and confidence AUC 0.766.
  A first non-leaky reliability-head probe in
  `feature_retrieval/tools/score_correspondence_reliability.py` used only
  confidence plus query position. Training on ShopFacade improves its own AUC
  0.621 -> 0.640 but drops OldHospital AUC 0.766 -> 0.751; training on
  OldHospital improves its own AUC 0.766 -> 0.776 and ShopFacade AUC
  0.621 -> 0.626, while Brier is worse in all cases. These labels are suitable
  for future reliability/weighting research, but this first head is not strong
  enough to promote blindly. A follow-up PnP sweep in
  `feature_retrieval/tools/sweep_correspondence_pnp_filters.py` shows the
  signal has limited pose-level value: on ShopFacade, confidence top-25%
  improves median/mean translation from 65.9/108.3mm to 55.7/103.6mm and
  cross-model top-75% gives the best mean/success at 59.4/92.1mm with
  5deg/250mm success 0.951. On OldHospital, the best cross-model top-90%
  setting only improves 206.8/421.8mm to 200.2/402.0mm and 5deg/250mm success
  0.549 -> 0.571. This is a useful external baseline filter, but the gains are
  too small and scene-dependent to claim a robust POFD/Stage4 contribution.
- ShopFacade teacher-quality and teacher-pair-flow probes:
  `feature_extract/configs/pofd_shopfacade_teacherquality_renderloftrpnp_top5_s40.yaml`
  runs end-to-end but worsen candidate selection on full103 validation
  (best direct teacher-quality pred cost about 0.167m versus the 0.131m active
  selected external-cache cost, and worse success rates). Pair-flow-only configs
  were then added:
  `pofd_shopfacade_teacherpairflow_renderloftrpnp_top5_top1_pm_only_s40.yaml`
  and
  `pofd_shopfacade_teacherpairflow_renderloftrpnp_top5_top1_pm_only_hires136_subpixel_s40.yaml`.
  The low-resolution branch reaches teacher-flow acc 0.385 but only 3.8% of
  validation offsets are nonzero; Stage4 closed-loop eval on the reprojection-
  median init regresses 0.112m -> 0.514m. The high-resolution subpixel branch
  raises nonzero offset coverage to 44.0% and reaches 0.850px subpixel EPE, but
  Stage4 expected-offset eval still regresses 0.112m -> 0.211m at confidence 0.
  A confidence sweep confirms no usable gate: 0.001 still regresses to 0.126m,
  while 0.002/0.005/0.010 reject all updates and exactly no-op. This closes the
  current ShopFacade teacher-pair-flow branch as diagnostic-only/no-go.

This resolves the first map prerequisite for an additional-scene protocol, but
it does not count as a completed multi-scene POFD result. Missing pieces are
a POFD-owned refinement/uncertainty contribution that adds value beyond the
external initializer and a scene-level table with promotable multi-scene POFD
metrics comparable to OldHospital.

## MASt3R/DUSt3R Official Preflight

Status on 2026-05-17:

- Official MASt3R was cloned recursively to `third_party/mast3r` at
  `f5209afc300cec36239a7ac992263f36847bbba0`.
- The MASt3R parent pins DUSt3R to
  `3cc8c88c413bb9e34c41db0e0eef99c2ee010b12` even though the current DUSt3R
  `main` head is `4c24a6ebf04809f2cfe59915e51779c8984aaa40`. The initial
  shallow recursive clone fetched DUSt3R `main`; the pinned commit was then
  fetched directly and checked out.
- The nested DUSt3R CroCo submodule is initialized at
  `d7de0705845239092414480bd829228723bf20de`.
- Upstream README license is CC BY-NC-SA 4.0 and checkpoint use has additional
  dataset-license constraints in `CHECKPOINTS_NOTICE`; this is acceptable for
  research preflight but must be reflected in paper/release notes.
- Upstream example env is Python 3.11 with PyTorch CUDA 12.1, plus
  `requirements.txt`, `dust3r/requirements.txt`, and optional visloc packages.
  The current `iclpose` env is Python 3.8.10 with `torch==1.13.1+cu116`.
- The optional visloc dependencies `roma`, `numpy-quaternion`, `kapture`, and
  `kapture-localization` were installed into the current env after a pip
  dry-run. The install also brought `numba/llvmlite`, `opencv-python`,
  `cvxpy`, and solver wheels. `pip check` still reports the known HLoc
  dependency mismatch (`lightglue` absent and `pycolmap==3.12.5` below HLoc's
  declared `>=3.13.0`), but the local HLoc compatibility tests still pass.
- Current-env smoke with
  `PYTHONPATH=third_party/mast3r:third_party/mast3r/dust3r:third_party/mast3r/dust3r/croco`
  imports `mast3r.model.AsymmetricMASt3R`, `mast3r.fast_nn.fast_reciprocal_NNs`,
  `dust3r.inference.inference`, `dust3r_visloc.localization`,
  `dust3r_visloc.evaluation`, and the Cambridge dataloader. It warns that
  compiled RoPE kernels are absent and falls back to the slower PyTorch
  implementation.
- A synthetic PnP sanity check through official `dust3r_visloc.localization`
  succeeds with both `cv2` and `pycolmap` modes. `poselib` remains absent, so
  official README commands that request `--pnp_mode poselib` should be adapted
  to `pycolmap` or `cv2` unless `poselib` is added later.
- The official Naver checkpoint
  `MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth` was downloaded to
  `third_party/mast3r/checkpoints/` from the upstream README URL. A Hugging
  Face auto-download attempt was stopped after stalling at a 143MB incomplete
  blob; the Naver download completed at 2,754,910,614 bytes.
- A local-checkpoint GPU0 smoke on the bundled CroCo Chateau pair loads the
  model in 8.95s, produces descriptor maps of `(384, 512, 24)` for both images,
  and returns 1,047 raw reciprocal MASt3R matches.
- The official `visloc.py --help` command now works, but the available
  Cambridge HLoc/retriangulated data layout is `scene/{empty_all,model_train,
  list_db.txt,list_query.txt}`. Upstream `VislocCambridgeLandmarks` expects a
  prepared kapture/mapping/pairsfile root. Therefore a project adapter or data
  conversion layer is still required before running Cambridge MASt3R visloc.
- The official MASt3R `visloc.py` path is relevant to the expert-file pivot:
  it builds query-map dense MASt3R matches, maps rendered or SfM map pixels to
  3D points, and solves 2D-3D PnP with `cv2`, `poselib`, or `pycolmap`. A
  project adapter should reuse that correspondence/PnP logic but feed existing
  ICLPose render-depth/world-position tensors and HLoc/NetVLAD init caches.

This is only a dependency and design preflight. No MASt3R adapter, checkpoint
Cambridge visloc run, or MASt3R Cambridge result has been produced yet.

## Next Concrete Design Target

The next non-redundant implementation should change the correspondence source
or geometry model rather than add another DCFF residual scorer. The current
rendered-RGB LoFTR and POFD-residual weighting variants have been tested and
closed as no-go on OldHospital full182. A viable Stage4 external correspondence
refiner would need to:

- render once at the current pose;
- run an external matcher adapter (`LoFTR` now, `MASt3R/DUSt3R` when available);
- convert 2D query-render matches plus rendered depth/world positions into
  robust weighted 2D-3D correspondences;
- solve a guarded pose update with diagnostics;
- use POFD-derived uncertainty only to accept/reject or weight updates, not as
  a candidate topK selector.

## Latest Verification

Fresh checks after the HLoc cache/refiner update and MASt3R preflight on
2026-05-17:

- `pytest third_party/Hierarchical-Localization/tests/test_match_workers_env.py third_party/Hierarchical-Localization/tests/test_pycolmap_compat.py -q`
  passed: 5 tests.
- `pytest tests/test_sweep_correspondence_pnp_filters.py tests/test_score_correspondence_reliability.py tests/test_render_loftr_pnp_init_export.py tests/test_eval_render_loftr_refine.py tests/test_score_loftr_correspondence_feature_consistency.py -q`
  passed: 27 tests, 2 warnings.
- `python -m py_compile third_party/Hierarchical-Localization/hloc/match_features.py third_party/Hierarchical-Localization/hloc/triangulation.py data/radio_loc_retrieval_dataset.py feature_retrieval/localization_mainline.py pose_refine/tools/eval_render_loftr_refine.py feature_extract/tools/train_nvs_pose_feature_adapter.py`
  passed.
- HLoc Stage4 eval configs parse through YAML and the two Stage4 result JSONs
  report exact no-ops: OldHospital 0.255759m -> 0.255759m and ShopFacade
  0.064079m -> 0.064079m, both with 0.0 solver success and 0.0 accepted
  updates.
- `git diff --check` passed.
- HLoc init/refined caches load through `load_retrieval_init_entries` for all
  five HLoc init scenes plus OldHospital/ShopFacade refined caches.
- `git -C third_party/mast3r rev-parse HEAD`,
  `git -C third_party/mast3r/dust3r rev-parse HEAD`, and
  `git -C third_party/mast3r/dust3r/croco rev-parse HEAD` report
  `f5209afc300cec36239a7ac992263f36847bbba0`,
  `3cc8c88c413bb9e34c41db0e0eef99c2ee010b12`, and
  `d7de0705845239092414480bd829228723bf20de`.
- MASt3R visloc import smoke with the local `PYTHONPATH` imports
  `AsymmetricMASt3R`, `fast_reciprocal_NNs`, `dust3r.inference`,
  `dust3r_visloc.localization`, `dust3r_visloc.evaluation`, and
  `VislocCambridgeLandmarks`.
- Synthetic official PnP sanity check succeeds with `cv2` and `pycolmap`,
  each returning a 4x4 pose.
- `PYTHONPATH=... python third_party/mast3r/visloc.py --help` prints the
  expected CLI, and `python -m py_compile` passes for `visloc.py`,
  official localization/evaluation, and the Cambridge/base-colmap dataloaders.
- Local-checkpoint MASt3R pair smoke on GPU0 loads
  `third_party/mast3r/checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth`
  and returns 1,047 raw reciprocal matches on the bundled Chateau image pair.
- `pip check` exits nonzero only for the known HLoc package metadata issues:
  missing `lightglue` and `pycolmap==3.12.5` below HLoc's declared `>=3.13.0`.
  The targeted HLoc compatibility tests still pass.
- `nvidia-smi` showed both RTX 3090 GPUs idle at 1MiB and 0% utilization, and
  no HLoc/Cambridge/pytest/render-refine processes remained.
