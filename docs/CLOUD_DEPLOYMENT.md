# Cloud Deployment

## Fast Path

The fastest way to run this repo on a rented server is:

1. Clone the repo to `/root/ICLPose-loc` if possible.
2. Copy `dataset/` and `output/` assets from the source machine.
3. Create the conda env from `environment.cloud.yml`.
4. Install submodule wheels / editable packages.
5. Export `PYTHONPATH` and `LD_LIBRARY_PATH`.
6. Run a smoke config first, then scale up.

## One-Time Setup

```bash
git clone <your-repo-url> /root/ICLPose-loc
cd /root/ICLPose-loc

conda env create -f environment.cloud.yml
conda activate iclpose-cloud

git submodule update --init --recursive
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install -U git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0

export PYTHONPATH=/root/ICLPose-loc
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages"
```

## Copy Required Assets

Copy these directories from the existing machine:

- `dataset/`
- `output/feature_field/`
- `output/feature_extract/`
- `output/feature_gaussian/`
- `output/feature_retrieval/`
- `output/pose_refine/`

Example:

```bash
rsync -avz source:/path/to/ICLPose-loc/dataset/ ./dataset/
rsync -avz source:/path/to/ICLPose-loc/output/feature_field/ ./output/feature_field/
rsync -avz source:/path/to/ICLPose-loc/output/feature_extract/ ./output/feature_extract/
rsync -avz source:/path/to/ICLPose-loc/output/feature_gaussian/ ./output/feature_gaussian/
rsync -avz source:/path/to/ICLPose-loc/output/feature_retrieval/ ./output/feature_retrieval/
rsync -avz source:/path/to/ICLPose-loc/output/pose_refine/ ./output/pose_refine/
```

If you cannot clone to `/root/ICLPose-loc`, rewrite the absolute paths inside the custom smoke configs first.

## Quick Verification

```bash
python - <<'PY'
import torch
import gsplat
print(torch.__version__)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
print('gsplat ok')
PY
```

## Smoke Run

Use the current lightweight validation configs first:

```bash
python -m feature_extract.train --config feature_extract/configs/joint_radio_dcff_oh_v5l_stablemap_geomcorr2e_smoke.yaml
python -m feature_extract.export_impl --config feature_extract/configs/joint_radio_dcff_oh_v5l_stablemap_geomcorr2e_smoke.yaml \
  --checkpoint output/feature_extract/joint_radio_dcff_oh_v5l_stablemap_geomcorr2e_smoke/checkpoints/best.pth \
  --output-dir output/feature_extract/features_query_student_v5l_stablemap_geomcorr2e_smoke_colmapid/OldHospital \
  --split all --name-mode colmap_image_id --colmap-dir dataset/OldHospital/sparse/0
python -m pose_refine.scripts.train_concat_loc --config pose_refine/configs/concat_loc_oh_v20k_v5l_stablemap_geomcorr2esmoke_fullwls_smoke.yaml
```

## Notes

- The current codebase is still optimized around repo-local paths.
- The quickest deployment path is to keep the same absolute repo path on the cloud host.
- For long runs, keep the `feature_extract -> export -> pose_refine` sequence separate and checkpoint each stage.
