# 环境配置指南

> 目标：在当前 `ICLPose-loc` 主线（RADIO + DCFF + concat localization）下完成可运行环境准备。

## 1. 硬件与版本建议

- Linux + NVIDIA GPU
- CUDA **11.8+**（4090 / Ada 建议至少 11.8）
- Python 3.9
- 推荐环境名：`geo-aware`

如果是 4090 / Ada 架构，不建议继续使用 PyTorch 1.13.1 + CUDA 11.6 组合。

## 2. 推荐安装方式

优先使用仓库内环境文件：

```bash
conda env create -f environment.key.yml
conda activate geo-aware
```

如果需要手动安装 PyTorch，推荐：

```bash
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

## 3. 子模块与关键依赖

```bash
git submodule update --init --recursive

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
```

如果 `gsplat` 与当前 PyTorch / CUDA 组合不兼容，优先重新编译而不是继续沿用旧 wheel。

## 4. 快速验证

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")

import gsplat
from gsplat import rasterization
print("gsplat ok")
PY
```

## 5. 本地资产约定

当前仓库只接受**仓库内本地路径**：

- checkpoints: `feature_field/output/*/checkpoints/`, `pose_refine/output/*/checkpoints/`
- 场景几何 / splats: `feature_gaussian/output/2dgs_models/`
- 提取特征: `feature_extract/output/features_*/`
- 第三方权重: `feature_extract/checkpoints/RADIO/`

`/root/ICLPose/...` 属于另一条分支，不应作为当前仓库的 checkpoint / 资产来源。

## 6. 当前主线入口

```bash
python -m feature_field.train
python -m feature_field.evaluate

python -m feature_extract.train
python -m feature_extract.export
python -m feature_extract.evaluate

python -m feature_retrieval.build_index
python -m feature_retrieval.train
python -m feature_retrieval.evaluate

python -m pose_refine.train
python -m pose_refine.evaluate
```

`feature_gaussian/` 也提供独立训练 / 评估入口，用于高斯场构建与渲染侧实验。

## 7. 多 GPU 使用建议

当前仓库尚未把所有主线训练统一收敛为一个标准 DDP 入口，推荐先采用：

1. **多实验并行**：每张卡跑一个独立实验
2. **模块分离执行**：SceneFeatureField / FeatureExtract / PoseRefine 分阶段运行
3. **优先保证路径本地化和 train/eval 一致性**，再做大规模 DDP 改造

## 8. 迁移到新机器时需要同步的内容

```bash
rsync -avz output/2dgs_models/ new_server:ICLPose-loc/output/2dgs_models/
rsync -avz output/features_*/ new_server:ICLPose-loc/output/
rsync -avz output/*/checkpoints/ new_server:ICLPose-loc/output/
rsync -avz dataset/ new_server:ICLPose-loc/dataset/
```

不要同步外部工作区绝对路径；只同步当前仓库目录内的数据和权重。
