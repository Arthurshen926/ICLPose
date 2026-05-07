# Implicit Coarse-To-Fine Localization Design

## Goal

Replace the external NetVLAD/LoFTR/PnP-driven pose-candidate entry point with an internal localization path driven by this project's query/map coarse and fine features.

The first milestone is not a full differentiable SE(3) optimizer. It is a measurable coarse pose bank that uses exported project features to produce pose anchors and reports top-k pose recall. This gives a clean test of whether the learned coarse space can provide a larger basin than the current fine-only refiner.

## Current Problem

The best deployable pipeline still depends on external candidates:

1. NetVLAD retrieves reference images.
2. LoFTR matches query/reference render pairs.
3. PnP converts each match set into a pose.
4. The refiner only performs local correction.

The top50 oracle reaches about 77.6mm median translation, so the candidate set can contain good poses. The deployable selector/refiner remains around 165-177mm because it cannot reliably identify or recover from the right basin. The deployed v17/v70 refiner configs also set `use_coarse: false` and `use_two_stage_refine: false`, so the project coarse features are not currently used as the large-basin stage.

## Proposed Architecture

The implicit route uses a project-owned pose energy:

1. Query student extracts query coarse and fine maps.
2. Map/DCFF or exported map/reference features provide a coarse pose bank.
3. Query coarse scores the bank with a differentiable cosine/InfoNCE objective during training.
4. Inference takes top-k anchors from this internal coarse bank.
5. A coarse local SE(3) grid or coarse pose stage pulls anchors into the fine basin.
6. Fine corr-WLS refines the final pose.

Training should be soft where gradients matter and hard only where runtime needs pruning. That means the training target is a soft pose distribution or ranking loss over anchors, not a hard NetVLAD/LoFTR/PnP candidate choice.

## First Milestone

Build `feature_extract/coarse_pose_bank_init_export.py`.

The tool loads exported `coarse_sem` feature maps, pools them into masked or spatial-average descriptors, searches train/reference descriptors for each query, and writes the same retrieval-init cache schema used by existing evaluation. It also reports pose recall metrics for top1/top5/top10/top20/top50 when ground-truth COLMAP poses are available.

Success criteria:

- Unit tests prove descriptor pooling, top-k search, retrieval-cache export, and top-k pose recall.
- OldHospital test export runs using `features_radio_dual_v68_align_stop640`.
- Top-k recall is compared against existing NetVLAD/LoFTR and oracle results.
- If top20/top50 recall shows enough good basins, wire this cache into existing `feature_retrieval.evaluate`.
- If recall is weak, train coarse ranking loss before investing in two-stage refiner changes.

## Non-Goals For This Milestone

- Do not remove NetVLAD/LoFTR yet.
- Do not rewrite `ConcatPoseNet` before coarse bank recall is measured.
- Do not claim end-to-end success from cosine alone.
- Do not make hard GT-assisted choices in deployable inference.

