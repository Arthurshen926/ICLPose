# ICLPose-loc

RADIO-guided feature-field reconstruction and visual localization.

## Quick Deploy

The fastest cloud path is:

1. Clone the repo to `/root/ICLPose-loc` if possible.
2. Create the environment with `conda env create -f environment.cloud.yml`.
3. Copy `dataset/` and `output/` checkpoints/features from the source machine.
4. Run the smoke configs in `docs/CLOUD_DEPLOYMENT.md`.

Current work is focused on the stable-map geometric feature path described in `docs/PROJECT_STATUS.md`.

## Mainline status

The current mainline is **RADIO + DCFF + concat localization**, organized around
five system interfaces:

1. **FeatureGaussian**
   - Gaussian geometry / appearance / latent carriers.
   - Primary entry: `feature_gaussian/`
   - Backing implementation: `dcff/hybrid_gaussian.py`, `feature_gaussian/models/`

2. **SceneFeatureField**
   - Scene-side feature field, renderer, map decoder, and train/eval shared DCFF runtime.
   - Primary entry: `scene_feature_field/`
   - Backing implementation: `dcff/`, `scene_feature_field/runtime.py`, `scripts/train_dcff_v5.py`

3. **FeatureExtract**
   - Teacher feature extraction, query-student training/export, and shared query-feature utilities.
   - Primary entry: `feature_extract/`
   - Backing implementation: `feature_extract/extractors/`, `feature_extract/students/`

4. **FeatureRetrieval**
   - Global retrieval / pose initialization / candidate filtering.
   - Primary entry: `feature_retrieval/`
   - Backing implementation: `feature_retrieval/retrievers/`, `data/radio_loc_retrieval_dataset.py`

5. **PoseRefine**
   - Dense correspondence, geometric solve, iterative pose refinement, and shared model loading.
   - Primary entry: `pose_refine/`
   - Backing implementation: `pose_refine/models/concat_pose_net.py`, `pose_refine/utils/geometry_solver.py`

Legacy directories such as `feature_extraction/`, `place_recognition/`, `feature_3dgs/`,
`ic_models/concat_pose_net.py`, and `modules/geometry_solver.py` are retained only as
compatibility facades during the migration.

The end-to-end localization stack remains:

`retrieval/init -> scene feature render -> dense correspondence / feature flow -> geometric solve -> iterative refinement`

## Current best results

### Mainline benchmark status

The current strongest OldHospital benchmark result is:

- `python -m pose_refine.evaluate_pipeline`
- **0.30° / 166 mm** median error
- pipeline: **RADIO features + DCFF map rendering + LoFTR accumulated correspondences + PnP + render-and-rematch**

This benchmark still uses **oracle retrieval** for candidate selection, so the
main remaining deployment gap is retrieval, not the geometric solver.

### Learned refinement status

On OldHospital, the best adapted student-query localizer is:

- `concat_loc_oh_v20k_v5g_student_querymap_adapt`
- **145.1 mm** median translation

This is essentially tied with the teacher-query baseline:

- `concat_loc_oh_v20d`
- **144.95 mm** median translation

A multi-hypothesis oracle-style fusion path also reached:

- **47.3 mm** median translation

This means the refinement stack itself is strong once initialization is in the right basin.

### Real init

Real deployment is **not** at the same level yet.

Best current read:

- `top1 CLS real-init`: **4493.5 mm** median translation
- `top10 oracle`: **2791.7 mm** median translation

So the current real-world bottleneck is **candidate scoring / selector quality**, not the downstream refiner alone.

## Mainline code paths

### System-layer facades

- `feature_gaussian/`
- `scene_feature_field/`
- `feature_extract/`
- `feature_retrieval/`
- `pose_refine/`

### Active training / evaluation entrypoints

- `python -m feature_gaussian.train`
- `python -m feature_gaussian.evaluate`
- `python -m scene_feature_field.train`
- `python -m scene_feature_field.evaluate`
- `python -m scene_feature_field.visualize_reconstruction`
- `python -m feature_extract.train`
- `python -m feature_extract.extract_radio_features`
- `python -m feature_extract.extract_radio_dual_features`
- `python -m feature_extract.export`
- `python -m feature_extract.evaluate`
- `python -m feature_retrieval.build_index`
- `python -m feature_retrieval.build_init_poses`
- `python -m feature_retrieval.train`
- `python -m feature_retrieval.evaluate`
- `python -m pose_refine.train`
- `python -m pose_refine.evaluate`
- `python -m pose_refine.evaluate_loftr_init`
- `python -m pose_refine.evaluate_pipeline`
- `python -m pose_refine.evaluate_render_compare`

The matching top-level `scripts/*.py` files are now mostly compatibility shims
or small support utilities; the canonical implementations should live inside
the five package directories. In particular, the strongest sparse/geometric
evaluation path is now `pose_refine.evaluate_pipeline`.

## Recommended working convention

If you are continuing the current project, start from:

- **FeatureGaussian**: `python -m feature_gaussian.train`, `python -m feature_gaussian.evaluate`
- **SceneFeatureField**: `python -m scene_feature_field.train`, `python -m scene_feature_field.evaluate`, `python -m scene_feature_field.visualize_reconstruction`
- **FeatureExtract**: `python -m feature_extract.train`, `python -m feature_extract.extract_radio_features`, `python -m feature_extract.extract_radio_dual_features`, `python -m feature_extract.export`, `python -m feature_extract.evaluate`
- **FeatureRetrieval / relocalization**: `python -m feature_retrieval.build_index`, `python -m feature_retrieval.build_init_poses`, `python -m feature_retrieval.train`, `python -m feature_retrieval.evaluate`
- **PoseRefine**: `python -m pose_refine.train`, `python -m pose_refine.evaluate`, `python -m pose_refine.evaluate_loftr_init`, `python -m pose_refine.evaluate_pipeline`, `python -m pose_refine.evaluate_render_compare`

### Canonical configs

- **SceneFeatureField**: `configs/dcff_oldhospital_v5a.yaml`
- **FeatureExtract**: `configs/joint_radio_dcff_oh_v5l_pointwise_featsharp_full.yaml`
- **FeatureExtract smoke/pilot**: `configs/joint_radio_dcff_oh_v5m_pointwise_teacher_anchor_pilot.yaml`
- **PoseRefine / FeatureRetrieval**: `configs/concat_loc_oh_v20k_v5g_student_querymap_adapt.yaml`

## Local asset convention

- Pretrained / reusable model assets should be read from repository-local paths only.
- Preferred locations:
  - checkpoints: `output/*/checkpoints/`
  - geometry / scene assets: `output/2dgs_models/`
  - extracted features: `output/features_*/`
  - bundled third-party weights: `models/`
- `/root/ICLPose/...` belongs to another branch and is **not** a migration source for this repo.
- Active configs must use local repo-relative paths only.

## Legacy / comparison-only branches

The following are **not** the current mainline:

- `legacy/gsff_baseline/`
  - GSFF reproduction and earlier comparison line
- `legacy/feature_ablations/`
  - early feature extraction ablations
- `_archive/`
  - historical notes and one-off investigations

These branches are preserved only for history and comparison. The active path is the five-module RADIO + DCFF + concat localization stack above.

Additional cleanup conventions:

- historical one-off scripts should be moved under `legacy/scripts/`
- experimental configs that are not canonical should be moved under `legacy/configs/experimental/`
- new top-level `scripts/` files should be reserved for compatibility shims or broadly useful support tools
