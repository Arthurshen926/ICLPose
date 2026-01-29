# Implicit Correspondence Pose Estimation - 项目概览

## 🎯 项目简介

基于隐式2D-3D对应关系的6-DOF相机位姿估计框架，完全对齐ICL-I2PReg架构，集成了SplatLoc模块，支持多GPU分布式训练。

---

## 📦 快速链接

| 文档 | 说明 |
|------|------|
| [README.md](README.md) | 项目主页 |
| [QUICKSTART.md](QUICKSTART.md) | 5分钟快速开始 |
| [TRAINING_GUIDE.md](TRAINING_GUIDE.md) | 完整训练指南 |
| [DISTRIBUTED_TRAINING.md](DISTRIBUTED_TRAINING.md) | 多GPU训练 |
| [REFACTOR_COMPLETE.md](REFACTOR_COMPLETE.md) | 重构完成说明 |

---

## 📂 目录结构

```
implicit_correspondence/
├── train.py                 ⭐ 训练入口
├── requirements.txt         📦 依赖列表
├── setup.py                 📦 安装脚本
│
├── configs/                 ⚙️  配置文件
├── data/                    💾 数据加载
├── ic_models/               🧠 核心模型
├── modules/                 🔧 模型组件
├── losses/                  📉 损失函数
├── utils/                   🛠️ 工具函数
├── splatloc_modules/        🎨 集成SplatLoc
├── scripts/                 📜 脚本工具
└── docs/                    📚 详细文档
```

---

## 🚀 核心功能

### 1. Keypoint-Based Pose Regression
- ✅ Soft-argmax关键点提取
- ✅ 关键点坐标编码（2D: 128维, 3D: 128维）
- ✅ 拼接特征：[kp_2d + query + kp_3d] = 384维

### 2. Diversity Loss
- ✅ 防止关键点坍缩
- ✅ 可配置margin和权重
- ✅ 自动记录到TensorBoard

### 3. 高级训练
- ✅ 相对位姿/绝对位姿
- ✅ Kendall's Loss自动权重
- ✅ 多GPU分布式训练
- ✅ 完整可视化工具

---

## 🎓 使用示例

### 训练
```bash
# 单GPU
python train.py --config configs/train_config.yaml --exp_name test

# 多GPU
bash scripts/train_distributed.sh
```

### 监控
```bash
tensorboard --logdir=output/test/logs
```

### 测试
```bash
python scripts/test_keypoint_extraction.py
python scripts/verify_installation.py
```

---

## 📊 性能指标

| 指标 | 典型值 |
|------|--------|
| 旋转误差（中位数） | ~2-5° |
| 平移误差（中位数） | ~0.05-0.1m |
| Diversity Loss | ~0.01 |
| 训练时间 | ~8h (4x GPU) |

---

## 🛠️ 依赖项

### 核心依赖
- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.7

### 主要包
- torchvision, numpy, scipy
- opencv-python, matplotlib
- tensorboard, tqdm
- plyfile, open3d

详见 [requirements.txt](requirements.txt)

---

## 📝 配置说明

主配置文件: `configs/train_config.yaml`

关键参数：
```yaml
dataset:
  data_root: "/path/to/data"
  train_scene: "Sequence_1"
  val_scene: "Sequence_2"

model:
  feature_dim: 256
  num_queries: 16
  fusion_layers: 2

loss:
  use_relative_pose: true
  diversity_weight: 0.01
  diversity_margin_2d: 10.0
  diversity_margin_3d: 0.1

training:
  batch_size: 32
  learning_rate: 1.0e-4
  num_epochs: 500
```

---

## 🔍 故障排查

### CUDA OOM
```yaml
training:
  batch_size: 16  # 减小
```

### 训练不稳定
```yaml
training:
  learning_rate: 5.0e-5  # 降低
  grad_clip: 5.0         # 增加
```

### Diversity Loss过大
```yaml
loss:
  diversity_weight: 0.005  # 降低
  diversity_margin_2d: 15.0  # 增大
```

---

## 📚 技术文档

### 核心文档
- [Keypoint Implementation](docs/KEYPOINT_IMPLEMENTATION_SUMMARY.md)
- [Architecture Alignment](docs/EXP013_ARCHITECTURE_ALIGNMENT.md)
- [Three Key Improvements](docs/THREE_KEY_IMPROVEMENTS.md)

### 更多文档
查看 [docs/](docs/) 目录获取25+技术文档

---

## 🎯 关键改进历史

### exp013 - 架构对齐
- ✅ 完全对齐ICL-I2PReg fusion module
- ✅ 2层fusion (img block + pcd block)
- ✅ Token projection layers
- ✅ Separated queries

### exp014 - Keypoint机制
- ✅ Keypoint坐标提取
- ✅ Keypoint-based regression
- ✅ Diversity loss

### 2026-01-29 - 仓库重构
- ✅ 集成SplatLoc模块
- ✅ 重新组织文件结构
- ✅ 标准Python包

---

## 🙏 致谢

- **ICL-I2PReg** - Keypoint架构
- **SplatLoc** - Gaussian Splatting
- **3DGS** - 渲染框架

---

## 📧 联系方式

- GitHub Issues
- Email: your.email@example.com

---

最后更新: 2026-01-29
