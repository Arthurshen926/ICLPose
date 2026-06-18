# VFM Branch Cleanup Manifest

> **Superseded mainline note.** This cleanup manifest documents the earlier
> reset from RADIO/POFD/CPR exploration into VFM-MapLoc. The current branch has
> been re-scoped again: 2DGS Synthetic RADIO-MATCHA is now the active phase-1
> mainline. The retained VFM-MapLoc selector/verifier and 3D aggregation files
> are historical references unless a newer document explicitly promotes them.

The `vfm` branch removes the previous RADIO/POFD/CPR active mainline instead of
archiving it in this branch.

Removed active paths:

- `legacy/`
- `configs/`
- `logs/`
- `scripts/`
- old `docs/`
- `pose_refine/`
- `feature_extract/localizability/`
- old `feature_extract/configs/`
- old `feature_extract/scripts/`
- old `feature_extract/students/`
- old `feature_extract/tools/`
- old POFD/CPR/NVS/DCFF tests
- root-level historical discussion notes and old plans
- `feature_field/` DCFF reconstruction mainline
- `feature_retrieval/_archive/` and learned init/reranker scripts
- old `feature_gaussian/configs/`, scripts, and nested legacy code

Retained reusable infrastructure:

- `data/`
- `feature_extract/extractors/extractor_radio.py`
- `feature_extract/extractors/extractor_dino.py`
- `feature_extract/utils/radio_loader.py`
- `feature_retrieval/retrievers/`
- core `feature_gaussian/` model/evaluation utilities
- bundled RADIO checkpoint assets

New active paths:

- `feature_extract/vfm/`
- `feature_extract/tools/vfm/`
- `feature_extract/configs/vfm/`
- `docs/vfm/`
- `tests/test_vfm_*.py`

Cleanup rule:

New experiment code should not import deleted POFD/CPR/localizability modules or
reuse old stage names. If a fixed external solver is needed, wrap it as a
candidate generator or handoff baseline with explicit protocol metadata.
