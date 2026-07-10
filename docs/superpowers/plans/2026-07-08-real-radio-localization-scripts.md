# Real RADIO Localization Train/Eval Scripts Plan

**Goal:** Replace the render-cache-facing actual measurement entrypoints with real-image-first scripts for training and evaluating the RADIO/MATCHA localization measurement branch.

**Constraints:**

- Training and evaluation must consume real dataset images through `query_id` plus `reference_image_id`/`support_image_id`.
- New public scripts must not require `render_cache_manifest_csv`.
- Keep legacy render-cache fusion behavior working for existing callers.

## Tasks

- [x] Add tests proving new train/eval CLIs default to real-pair rows and do not expose render-cache manifests.
- [x] Extend RGB patch fusion so evaluation can crop reference patches from real support images.
- [x] Add a real-pair evaluation wrapper that writes the same match table outputs without a render manifest.
- [x] Add `train_real_radio_localization.py` as a real-image wrapper around existing measurement branch training.
- [x] Add `eval_real_radio_localization.py` as a real-image wrapper around measurement fusion.
- [x] Run focused and combined pytest verification.
