# GSFF Reproduction on OldHospital (Cambridge Landmarks)

复现论文 **"Gaussian Splatting Feature Fields for (Privacy-Preserving) Visual Localization"** 在 OldHospital 场景上的结果。

## 快速开始

### 前提条件（已就绪）

| 组件 | 路径 | 状态 |
|------|------|------|
| 2DGS 几何模型 (300K Gaussians) | `output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply` | ✅ |
| 特征 3DGS (coarse 32d) | `output/feature_3dgs/oldhospital_ae_div_perscale/coarse/best_model.pth` | ✅ |
| 特征 3DGS (mid 64d) | `output/feature_3dgs/oldhospital_ae_div_perscale/mid/best_model.pth` | ✅ |
| 特征 3DGS (fine_sd 64d) | `output/feature_3dgs/oldhospital_ae_div_perscale/fine_sd/best_model.pth` | ✅ |
| 特征 3DGS (fine_dino 64d) | `output/feature_3dgs/oldhospital_ae_div_perscale/fine_dino/best_model.pth` | ✅ |
| AE 压缩特征 (1084帧) | `output/features_multiscale_compressed/OldHospital_indexed/` | ✅ |
| 官方训练集 (895帧) | `output/features_multiscale_compressed/OldHospital_indexed/train_indices.npy` | ✅ |
| 官方测试集 (182帧) | `output/features_multiscale_compressed/OldHospital_indexed/test_indices.npy` | ✅ |

### 训练

提供两种训练方案：

**方案 A: Cross-Frame Warp（推荐，测试时有效）**

用邻近训练帧的特征通过深度warp替代3DGS渲染，绕过3DGS特征渲染质量瓶颈：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/gsff_oldhospital.yaml
```

**方案 B: 标准 3DGS 渲染（纯 GSFF 论文方法）**

使用 3DGS 特征渲染，与原论文方法一致：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/gsff_oldhospital_render.yaml
```

### 监控训练

```bash
# 查看最新验证结果
grep 'Val E\|★' output/gsff_oldhospital_train.log | tail -10

# 查看 cosine similarity (反映特征匹配质量)
grep 'cos_sim' output/gsff_oldhospital_train.log | tail -5
```

### 评估（Cambridge Landmarks 官方协议）

```bash
# 同时测试3DGS渲染和检索warp两种模式
python scripts/eval_cambridge.py \
    --config configs/gsff_oldhospital.yaml \
    --checkpoint output/gsff_oldhospital/checkpoints/best.pth \
    --mode both --iters 1 3 5 10

# 仅测试检索warp模式
python scripts/eval_cambridge.py \
    --config configs/gsff_oldhospital.yaml \
    --checkpoint output/gsff_oldhospital/checkpoints/best.pth \
    --mode warp --iters 5 10

# 保存结果到 JSON
python scripts/eval_cambridge.py \
    --config configs/gsff_oldhospital.yaml \
    --checkpoint output/gsff_oldhospital/checkpoints/best.pth \
    --mode both --iters 5 10 \
    --output output/gsff_oldhospital_results.json
```

## 技术方案

### 系统架构

```
Query Image → SD+DINOv2 Feature Extraction → 多尺度特征 (coarse/mid/fine_sd/fine_dino)
                                                     ↓
                        ┌──────────────── 特征匹配 ────────────────┐
                        │  Query Features      Reference Features   │
                        │  (stored)            (rendered/warped)    │
                        └──────────────── ↓ ────────────────────────┘
                                  MSFlowPoseNet
                          Coarse→Mid→Fine 光流匹配
                                     ↓
                        Image Jacobian + WLS 几何求解
                                     ↓
                              SE(3) 位姿更新
                                     ↓
                         外层迭代精化 (3-5次)
```

### 两种参考特征生成方式

| | 3DGS 渲染 | 检索 Warp |
|---|---|---|
| **方法** | 从 3DGS 特征场渲染特征 | 从最近训练帧 warp 特征 |
| **特征质量** | cosine sim ≈ 0.33 (低) | cosine sim ≈ 0.75-0.90 (高) |
| **测试时有效** | ✅ | ✅ (仅用训练帧 + 已知位姿) |
| **隐私保护** | ✅ (无需存储图像) | ⚠️ (需存储特征向量) |
| **推理速度** | 快 (单次渲染) | 中等 (检索 + 深度渲染 + warp) |

### 训练参数

| 参数 | 值 | 说明 |
|------|------|------|
| 外层迭代 | 3 (train) / 5 (val) | 每个样本的渲染→匹配→更新循环 |
| fine_iters | 8 | GRU 精细光流迭代次数 |
| 分辨率 | 15×26 → 30×53 → 35×61 | 粗→中→细三级 |
| 噪声课程 | 2°/0.1m → 10°/1.0m | 50 epoch warmup |
| 旋转损失 | cosine (1-cos θ) | 避免 acos 梯度爆炸 |
| 置信度范围 | [0.3, 0.8] | OldHospital 最佳范围 |

### 数据集信息

- **场景**: Cambridge Landmarks - OldHospital
- **训练**: 895 帧 (seq1,2,3,5,6,7,9)
- **测试**: 182 帧 (seq4,8)
- **分辨率**: 1920×1080
- **相机内参**: fx=fy=1673.5, cx=960, cy=540
- **3DGS**: 300,845 个 2D Gaussian surfels

### 评估指标

Cambridge Landmarks 标准报告 **中位数** (median) 误差：
- 旋转误差 (°)
- 平移误差 (m)

## 文件结构

```
configs/
  gsff_oldhospital.yaml           # Cross-frame warp 训练配置
  gsff_oldhospital_render.yaml    # 标准 3DGS 渲染训练配置

scripts/
  eval_cambridge.py               # Cambridge Landmarks 官方评估脚本
  train_ms_flow.py                # 训练脚本 (支持 official split)
  eval_retrieval_warp.py          # 检索 warp 评估脚本

output/
  gsff_oldhospital/               # Cross-frame warp 训练输出
  gsff_oldhospital_render/        # 3DGS 渲染训练输出
```

## 已知问题与调优

1. **3DGS 特征渲染质量**: OldHospital 场景的 3DGS 渲染特征 cosine similarity 仅 ~0.33，
   远低于 room_0 等室内场景。这是纯 3DGS 渲染方案效果较差的根本原因。

2. **大位姿偏差**: 初始噪声 10°/1.0m 下存在部分样本收敛困难，
   导致均值误差远大于中位数。可尝试增加外层迭代次数 (val_outer_iters=10-20)。

3. **置信度范围**: OldHospital 对置信度范围敏感，[0.3, 0.8] 是验证过的最优范围。
   过宽的范围 (如 [0.05, 0.95]) 可能导致训练发散。
