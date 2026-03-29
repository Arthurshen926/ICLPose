# DCFF v4 OldHospital Plan

## Final Route

- Query side:
  - Use dual-scale RADIO teacher features.
  - Fine target is shallow RADIO block 10 (`fine_geo`).
  - Coarse target is final RADIO layer (`coarse_sem`).
  - Both targets remain in the same 64d PCA space used by training.
- Map side:
  - Geometry comes from pretrained 2DGS `v7_depth` PLY.
  - Fine branch is explicit: per-Gaussian latent -> rasterized `z_map` -> strong fine decoder.
  - Coarse branch is implicit: rasterized position/scale -> hash grid -> MLP.
  - Coarse branch does not consume latent or view direction.

## Why This Route

- It matches the original DCFF intent: explicit fine, implicit coarse.
- It keeps per-Gaussian storage compact while leaving capacity in the fine decoder.
- It protects geometry first, then optionally allows a low-LR late finetune.

## v4 Experiments

1. `dcff_oldhospital_v4_final`
   - Delayed geometry unfreeze at 12k iters.
   - Small geometry LR scale and no post-unfreeze densification.
2. `dcff_oldhospital_v4_frozen`
   - Same feature architecture.
   - Geometry remains frozen for the full run.

## Decision Gate

- Keep the delayed-unfreeze route only if it improves feature alignment without degrading RGB stability.
- If both runs are close, prefer the fully frozen geometry route for safety and reproducibility.