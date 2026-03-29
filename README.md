# ICLPose — 基于 3DGS 特征图的 6-DOF 位姿估计

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-orange.svg)](https://pytorch.org/)
[![Branch](https://img.shields.io/badge/Branch-v2--iterative--routing-green.svg)]()

**多尺度光流 + 可微渲染 + 几何求解的迭代位姿精化框架**

</div>

---

## 当前最佳结果 (exp032, Replica room_0)

| 指标 | 数值 | vs CorrPoseNet 基线 |
|------|------|-------------------|
| **Rot Mean** | **0.33°** | -28% ✅ |
| Rot Median | 0.23° | — |
| **<1°** | **95.6%** | +12pp ✅ |
| Trans Mean | 20.7mm | -19% ✅ |

---

## 概览

ICLPose 通过 **3DGS 特征场** 的可微分渲染实现高精度位姿估计:

```
查询特征 + 初始位姿 → [外循环 ×3-5] → 精确 6-DOF 位姿
                         ↓
               可微渲染参考特征 (gsplat)
                         ↓
               多尺度光流 (coarse→mid→fine)
                         ↓
               Image Jacobian + WLS → Δξ
                         ↓
               更新位姿, 继续迭代
```

核心模型: **MSFlowPoseNet** (~4.17M 参数)
- SD+DINOv2 多尺度特征 (Coarse 7×10, Mid 15×20, Fine 35×46)
- RAFT-style ConvGRU 光流精化 (8 iterations)
- 置信度加权几何求解

---

## 快速开始

```bash
conda activate geo-aware
CUDA_VISIBLE_DEVICES=0 python scripts/train_ms_flow.py \
    --config configs/exp032_cosine_fiters8.yaml
```

---

## 文档

| 文档 | 内容 |
|------|------|
| **[docs/STABLE_IMPLEMENTATION_SUMMARY.md](docs/STABLE_IMPLEMENTATION_SUMMARY.md)** | 当前稳定实现的项目总览与原理说明 |
| **[docs/PROJECT_TRANSFER_GUIDE.md](docs/PROJECT_TRANSFER_GUIDE.md)** | 完整项目迁移指南 (推荐首读) |
| **[docs/AGENT_QUICKSTART.md](docs/AGENT_QUICKSTART.md)** | Agent 快速上手指南 |
| **[docs/ARCHITECTURE_DETAIL.md](docs/ARCHITECTURE_DETAIL.md)** | MSFlowPoseNet 架构详解 |
| **[docs/EXPERIMENT_HISTORY.md](docs/EXPERIMENT_HISTORY.md)** | 实验历史与改进记录 |
| **[docs/ENVIRONMENT_SETUP.md](docs/ENVIRONMENT_SETUP.md)** | 环境配置 (含 4090 适配) |

---

## 项目结构

```
ICLPose/
├── scripts/train_ms_flow.py      ★ 主训练脚本
├── ic_models/ms_flow_pose_net.py  ★ 核心模型
├── modules/
│   ├── geometry_solver.py         Image Jacobian + WLS
│   ├── multiscale_renderer.py     多尺度 3DGS 渲染
│   └── lie_algebra.py             SE(3) Lie 代数
├── data/dataset_v4.py             多尺度特征数据集
├── configs/exp032_*.yaml          最佳配置
├── feature_3dgs/                  3DGS 特征模型 + 渲染器
├── feature_extraction/            SD/DINO 特征提取
└── output/                        训练输出
```

---

## 致谢

- **RAFT** — 迭代光流精化
- **DROID-SLAM** — SE(3) Lie 代数
- **3D Gaussian Splatting / gsplat** — 可微渲染
- **Stable Diffusion / DINOv2** — 多尺度特征提取
- **STDLoc / SplatLoc** — Feature 3DGS 设计

---

## License

MIT License
