# 环境配置详细指南

> 目标: 在 6×RTX 4090 服务器上复现 ICLPose 训练环境

---

## 1. 硬件要求

| 项目 | 当前 (开发) | 目标 (6×4090) |
|------|------------|--------------|
| GPU | 2× RTX 3090 24GB | 6× RTX 4090 24GB |
| CUDA Compute | sm_86 | sm_89 (Ada) |
| CUDA 最低版本 | 11.6 | **11.8** (必须升级) |
| 显存/卡 | 24GB | 24GB |
| 训练占用 | ~22GB/卡 | ~22GB/卡 |

## 2. ⚠️ CUDA 版本兼容性

**重要**: RTX 4090 基于 Ada Lovelace 架构 (sm_89), 需要 CUDA 11.8+。
当前环境 PyTorch 1.13.1 + CUDA 11.6 **无法直接在 4090 上运行**。

### 推荐升级方案

**方案 A: PyTorch 2.0+ (强烈推荐)**

PyTorch 2.0 有 `torch.compile()` 加速，且对 Ada 架构原生支持好:

```bash
conda create -n geo-aware python=3.9
conda activate geo-aware

# PyTorch 2.0.1 + CUDA 11.8
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

**方案 B: PyTorch 2.1+ (最新)**

```bash
pip install torch==2.1.2 torchvision==0.16.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

升级后需要检查:
- `gsplat 1.4.0` 是否与新 PyTorch 版本兼容 (可能需要从源码重新编译)
- `kornia 0.6.0` 在 PyTorch 2.x 下可能需要升级到 `0.7+`
- `detectron2` 需要重新编译

## 3. 完整安装步骤

```bash
# Step 1: 创建 conda 环境
conda create -n geo-aware python=3.9.23 -y
conda activate geo-aware

# Step 2: 安装 PyTorch (适配 4090)
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Step 3: 核心依赖
pip install numpy==1.23.5
pip install scipy==1.13.1
pip install opencv-python==4.8.1.78
pip install pillow==9.5.0
pip install matplotlib==3.9.4
pip install tqdm pyyaml

# Step 4: 深度学习库
pip install kornia==0.6.0        # 几何计算 (可能需升级到0.7+)
pip install timm==0.6.11         # Vision Transformer backbone
pip install einops==0.3.0        # tensor reshape
pip install lpips==0.1.4         # 感知损失 (可选)
pip install torchmetrics==0.6.0  # 指标计算

# Step 5: 3DGS 渲染 (关键!)
# gsplat 需要从源码编译以匹配 CUDA 版本
pip install git+https://github.com/nerfstudio-project/gsplat.git@v1.4.0
# 如果编译失败, 尝试:
# TORCH_CUDA_ARCH_LIST="8.9" pip install gsplat==1.4.0

# Step 6: 3D 处理
pip install open3d==0.17.0
pip install plyfile==1.1.3

# Step 7: 训练管理
pip install wandb==0.23.1
pip install tensorboard==2.20.0
pip install tensorboardx==2.6.4

# Step 8: NLP/Vision 模型 (Stable Diffusion 特征提取)
pip install transformers==4.26.1
pip install huggingface-hub==0.36.0

# Step 9: 可选依赖
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
```

## 4. 自编译包

项目依赖两个需要从源码编译的包:

```bash
# diff-gaussian-rasterization (自定义高斯光栅化)
cd submodules/diff-gaussian-rasterization
pip install .

# simple-knn
cd submodules/simple-knn
pip install .
```

如果 `submodules/` 不存在, 可能需要:
```bash
git submodule update --init --recursive
```

## 5. 验证安装

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"GPU count: {torch.cuda.device_count()}")

import gsplat
print(f"gsplat: {gsplat.__version__}")

# 测试渲染
from gsplat import rasterization
print("gsplat rasterization OK")

import kornia
print(f"kornia: {kornia.__version__}")
```

## 6. 当前完整包列表 (参考)

以下是当前 `geo-aware` 环境中与项目相关的核心包:

```
Package                   Version    Source
------------------------- ---------- ------
python                    3.9.23     conda-forge
torch (pytorch)           1.13.1     cu116 (需升级!)
torchvision               0.14.1     cu116 (需升级!)
numpy                     1.23.5     pypi
scipy                     1.13.1     pypi
opencv-python             4.8.1.78   pypi
pillow                    9.5.0      pypi
matplotlib                3.9.4      pypi
gsplat                    1.4.0      pypi (需重新编译)
diff-gauss                0.0.0      pypi (需重新编译)
kornia                    0.6.0      pypi
timm                      0.6.11     pypi
einops                    0.3.0      pypi
open3d                    0.17.0     pypi
plyfile                   1.1.3      pypi
wandb                     0.23.1     pypi
transformers              4.26.1     pypi
huggingface-hub           0.36.0     pypi
lpips                     0.1.4      pypi
detectron2                0.6        pypi (需重新编译)
tensorboard               2.20.0     pypi
tqdm                      4.67.1     pypi
pyyaml                    6.0.3      pypi
torchmetrics              0.6.0      pypi
```

## 7. 数据传输

需要传输到新服务器的数据:

```bash
# 1. 代码 (通过 git)
git clone git@github-sqy:Arthurshen926/ICLPose.git
cd ICLPose
git checkout v2-iterative-routing

# 2. 预提取特征 (~数GB)
rsync -avz output/features_multiscale/ new_server:ICLPose/output/features_multiscale/

# 3. 3DGS 模型
rsync -avz output/3dgs_models/ new_server:ICLPose/output/3dgs_models/

# 4. 深度图
rsync -avz output/depth_maps/ new_server:ICLPose/output/depth_maps/

# 5. 训练好的模型权重
rsync -avz output/exp032/ new_server:ICLPose/output/exp032/
rsync -avz output/exp031/ new_server:ICLPose/output/exp031/

# 6. Replica 数据集
rsync -avz dataset/Replica/ new_server:ICLPose/dataset/Replica/
```

## 8. 多 GPU 配置 (6×4090)

当前训练脚本 `scripts/train_ms_flow.py` 是**单 GPU** 的。
要利用 6 张卡, 有以下方案:

### 方案 A: 多实验并行 (最简单)

同时在 6 张卡上跑不同超参实验:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py --config configs/exp033.yaml &
CUDA_VISIBLE_DEVICES=1 python scripts/train_ms_flow.py --config configs/exp034.yaml &
# ...
```

### 方案 B: DDP 改造 (需改代码)

在 `scripts/train_ms_flow.py` 中添加 `torch.nn.parallel.DistributedDataParallel`:

```python
# 需要修改 MSFlowTrainer 以支持 DDP
# 关键挑战: MultiScaleRenderer 中的 4 个 GaussianFeatureModel
# 是冻结的 (不参与梯度), 只有 MSFlowPoseNet 需要 DDP 包装
```

### 方案 C: 更大 Batch Size

单卡 batch_size=1 已占 ~22GB。在不改代码的情况下:
- batch_size=1 × 6 cards = 有效 batch 6 (需 gradient accumulation)
- 或者在每张卡上跑不同数据子集
