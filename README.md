# Implicit Correspondence Pose Estimation

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**6-DOF相机位姿估计框架，使用隐式2D-3D对应关系**

</div>

---

## 🎯 概览

本项目实现了基于隐式对应关系的6-DOF相机位姿估计，适用于3D Gaussian Splatting场景。通过学习图像特征和点云特征之间的隐式对应关系，无需显式特征匹配即可完成精确的位姿估计。

### 核心特性

✨ **Keypoint-Based Regression**
- 从attention heatmap提取2D/3D关键点坐标
- 使用关键点作为显式几何约束
- 完全对齐ICL-I2PReg架构

🔥 **Diversity Loss**
- 防止关键点坍缩到同一位置
- 确保关键点分散分布

⚡ **高级训练功能**
- 支持相对位姿和绝对位姿训练
- Kendall's Loss自动权重学习
- 多GPU分布式训练支持

🎨 **集成SplatLoc模块**
- 内置Gaussian Splatting渲染器
- 特征解码器集成
- 无需外部依赖

---

## 📦 安装

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/implicit_correspondence.git
cd implicit_correspondence
```

### 2. 安装依赖

```bash
# 创建conda环境
conda create -n ic-pose python=3.10
conda activate ic-pose

# 安装PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt
```

### 3. 准备数据

```
data/
└── room_0/
    ├── Sequence_1/          # 训练序列
    │   ├── rgb/
    │   ├── traj_w_c.txt
    │   └── features_compressed/fused/
    └── Sequence_2/          # 验证序列
```

---

## 🚀 快速开始

### 配置训练

编辑 `configs/train_config.yaml`:

```yaml
dataset:
  data_root: "/path/to/your/data/room_0"
  train_scene: "Sequence_1"
  val_scene: "Sequence_2"
```

### 开始训练

**单GPU：**
```bash
python train.py --config configs/train_config.yaml --exp_name my_exp
```

**多GPU（推荐）：**
```bash
bash scripts/train_distributed.sh
```

### 监控训练

```bash
tensorboard --logdir=output/your_exp_name/logs
```

---

## 📊 项目结构

```
implicit_correspondence/
├── configs/                 # 配置文件
├── data/                    # 数据加载
├── ic_models/               # 核心模型
├── modules/                 # 模型组件
├── losses/                  # 损失函数
├── splatloc_modules/        # 集成的SplatLoc模块
├── scripts/                 # 脚本和工具
├── docs/                    # 详细文档
├── train.py                 # 训练入口
└── requirements.txt
```

---

## 📖 文档

- **[TRAINING_GUIDE.md](TRAINING_GUIDE.md)** - 完整训练指南
- **[DISTRIBUTED_TRAINING.md](DISTRIBUTED_TRAINING.md)** - 多GPU训练
- **[docs/KEYPOINT_IMPLEMENTATION_SUMMARY.md](docs/KEYPOINT_IMPLEMENTATION_SUMMARY.md)** - 技术细节
- **[VISUALIZATION_README.md](VISUALIZATION_README.md)** - 可视化工具

---

## 🔬 架构

```
图像特征 + 点云特征 → FusionModule → Keypoint提取 → PoseRegressor → 6-DOF Pose
                                  ↓
                           Diversity Loss
```

核心改进：
1. **Keypoint提取** - Soft-argmax，可微分
2. **Keypoint编码** - 显式几何约束
3. **Diversity Loss** - 防止模式坍缩

---

## 🙏 致谢

- **ICL-I2PReg** - Keypoint架构灵感
- **SplatLoc** - Gaussian Splatting模块
- **3D Gaussian Splatting** - 渲染框架

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file

---

<div align="center">

[⬆ 返回顶部](#implicit-correspondence-pose-estimation)

</div>
