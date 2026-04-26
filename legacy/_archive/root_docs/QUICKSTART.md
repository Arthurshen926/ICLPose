# Quick Start Guide

## 🚀 5分钟快速上手

### 1. 安装

```bash
# 克隆仓库
git clone <repo_url>
cd implicit_correspondence

# 创建环境
conda create -n ic-pose python=3.10
conda activate ic-pose

# 安装依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### 2. 准备数据

确保你的数据结构如下：

```
/path/to/data/room_0/
├── Sequence_1/
│   ├── rgb/                    # RGB图像
│   ├── traj_w_c.txt           # 位姿文件
│   └── features_compressed/   # 预提取特征
│       └── fused/
└── Sequence_2/                 # 验证集（同样结构）
```

### 3. 配置

编辑 `configs/train_config.yaml`:

```yaml
dataset:
  data_root: "/path/to/your/data/room_0"
  train_scene: "Sequence_1"
  val_scene: "Sequence_2"

splatloc:
  gaussians_path: "/path/to/point_cloud.ply"
  decoder_path: "/path/to/decoder_weights.pth"
```

### 4. 训练

**单GPU：**
```bash
python train.py \
  --config configs/train_config.yaml \
  --exp_name my_first_experiment
```

**多GPU（推荐）：**
```bash
# 编辑 scripts/train_distributed.sh 设置参数
bash scripts/train_distributed.sh
```

或直接运行：
```bash
torchrun --nproc_per_node=4 train.py \
  --config configs/train_config.yaml \
  --exp_name exp_multi_gpu
```

### 5. 监控

```bash
tensorboard --logdir=output/my_first_experiment/logs
```

打开浏览器访问 http://localhost:6006

---

## 📊 关键指标

在TensorBoard中关注：

- **train/loss** - 总训练损失
- **train/diversity_loss_total** - Diversity loss（应稳定在~0.01）
- **val/rotation_error_mean** - 验证集角度误差（度）
- **val/translation_error_mean** - 验证集平移误差（米）

---

## 🔧 常见问题

### CUDA Out of Memory

降低batch size:
```yaml
training:
  batch_size: 16  # 默认32
```

### 训练不稳定

降低学习率:
```yaml
training:
  learning_rate: 5.0e-5  # 默认1.0e-4
```

### 导入错误

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/implicit_correspondence
```

---

## 📖 更多文档

- [TRAINING_GUIDE.md](TRAINING_GUIDE.md) - 完整训练指南
- [DISTRIBUTED_TRAINING.md](DISTRIBUTED_TRAINING.md) - 分布式训练详解
- [docs/](docs/) - 技术文档

---

## ✅ 测试安装

```bash
# 测试模型
python scripts/test_keypoint_extraction.py

# 检查环境
python scripts/verify_installation.py
```

---

祝你训练愉快！🎉
