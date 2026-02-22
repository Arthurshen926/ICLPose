# Implicit Correspondence - 隐式对应关系模块

用于学习2D-3D对应关系的深度学习模块，适配SplatLoc项目的数据格式。

## 📁 文件结构

```
implicit_correspondence/
├── data/
│   ├── __init__.py
│   └── dataset.py              # 对应关系数据集类
├── losses/
│   ├── __init__.py
│   └── pose_loss.py            # 位姿损失函数
├── models/
│   ├── __init__.py
│   └── ic_pose_net.py          # ICPoseNet模型
├── modules/
│   ├── __init__.py
│   ├── fusion_module.py        # 跨模态融合模块
│   ├── pose_regressor.py       # 位姿回归器
│   └── transformer.py          # Transformer模块
├── configs/
│   └── train_config.yaml       # 训练配置文件
├── train.py                    # 🆕 完整训练脚本
├── run_train.sh                # 🆕 快速启动脚本
├── check_environment.py        # 🆕 环境检查脚本
├── example_usage.py            # 使用示例
├── TRAINING_GUIDE.md           # 🆕 详细训练指南
├── TRAINING_SUMMARY.md         # 🆕 训练实现总结
└── README.md                   # 本文件
```

## 🚀 快速开始

### 训练ICPoseNet（完整训练流程）

**1. 检查环境**
```bash
python check_environment.py
```

**2. 配置训练参数**
编辑 `configs/train_config.yaml`，修改SplatLoc模型路径和数据集路径。

**3. 开始训练**
```bash
# 使用启动脚本（推荐）
./run_train.sh

# 或直接使用Python
python train.py --config configs/train_config.yaml
```

**4. 查看训练日志**
```bash
tensorboard --logdir output/exp001/logs --port 6006
```

详细的训练指南请参考：[TRAINING_GUIDE.md](TRAINING_GUIDE.md)

---

### 使用数据集和损失函数（手动训练）

#### 1. 数据集加载

```python
from implicit_correspondence.data import CorrespondenceDataset, collate_fn
from torch.utils.data import DataLoader

# 创建数据集
dataset = CorrespondenceDataset(
    data_root="/home/yons/Projects/data/room_0",
    scene_name="Sequence_1",
    image_size=(640, 480),
    augment=True,
    max_samples=100,
    use_depth=False,
    fx=320.0, fy=320.0, cx=319.5, cy=239.5,
)

# 创建DataLoader
loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=4,
)

# 迭代数据
for batch in loader:
    images = batch['image']          # [B, 3, H, W]
    poses = batch['pose']            # [B, 4, 4]
    points_2d = batch['points_2d']   # [N, 2]
    points_3d = batch['points_3d']   # [N, 3]
    # ... 训练代码
```

### 2. 损失函数使用

```python
from implicit_correspondence.losses import PoseLoss

# 创建损失函数
loss_fn = PoseLoss(
    rotation_loss_type='geodesic',    # 'geodesic', 'l2', 'cosine', 'quaternion'
    translation_loss_type='l2',       # 'l1', 'l2', 'smooth_l1'
    rotation_weight=1.0,
    translation_weight=1.0,
    reduction='mean',                  # 'mean', 'sum', 'none'
)

# 计算损失
pose_pred = model(images)              # [B, 4, 4]
pose_gt = batch['pose']                # [B, 4, 4]

losses = loss_fn(pose_pred, pose_gt, return_components=True)

total_loss = losses['loss']            # 总损失
rot_loss = losses['rotation_loss']     # 旋转损失
trans_loss = losses['translation_loss'] # 平移损失

# 反向传播
total_loss.backward()
```

## 📊 数据集详情

### CorrespondenceDataset

从SplatLoc场景加载训练数据，生成2D-3D对应关系。

**输入数据格式**：
```
data_root/scene_name/
├── rgb/
│   ├── rgb_0.png
│   ├── rgb_1.png
│   └── ...
├── depth/              # 可选
│   ├── depth_0.png
│   └── ...
└── traj_w_c.txt       # 相机位姿 (4x4矩阵，每行16个数)
```

**参数说明**：
- `data_root`: 数据根目录
- `scene_name`: 场景名称（如 "Sequence_1"）
- `image_size`: 图像尺寸 (width, height)
- `augment`: 是否启用数据增强（颜色抖动、灰度化等）
- `max_samples`: 最大样本数量
- `use_depth`: 是否使用深度图
- `fx, fy, cx, cy`: 相机内参

**返回数据**：
- `image`: [3, H, W] RGB图像（归一化）
- `pose`: [4, 4] 相机位姿矩阵
- `points_2d`: [N, 2] 图像坐标
- `points_3d`: [N, 3] 世界坐标
- `valid_mask`: [N] 有效点掩码
- `intrinsics`: [3, 3] 相机内参
- `depth`: [1, H, W] 深度图（可选）

### 数据增强

训练时（`augment=True`）自动应用：
- 颜色抖动（亮度、对比度、饱和度、色调）
- 随机灰度化（10%概率）
- ImageNet归一化

## 🎯 损失函数详情

### PoseLoss

支持多种旋转和平移损失的组合。

#### 旋转损失类型

| 类型 | 描述 | 适用场景 |
|------|------|---------|
| `geodesic` | SO(3)流形上的测地距离 | **推荐**，几何意义明确，单位为度 |
| `l2` | 旋转矩阵的L2距离 | 简单直接，但缺乏几何意义 |
| `cosine` | 旋转矩阵的余弦相似度 | 对大误差不敏感 |
| `quaternion` | 四元数L2距离 | 紧凑表示，训练稳定 |

#### 平移损失类型

| 类型 | 描述 | 适用场景 |
|------|------|---------|
| `l2` | 欧几里得距离 | **推荐**，几何意义明确 |
| `l1` | 曼哈顿距离 | 对异常值鲁棒 |
| `smooth_l1` | Smooth L1损失 | 结合L1和L2的优点 |

#### 推荐配置

```python
# 方案1: 通用配置（推荐）
PoseLoss(
    rotation_loss_type='geodesic',
    translation_loss_type='l2',
    rotation_weight=1.0,
    translation_weight=1.0,
)

# 方案2: 强调旋转精度
PoseLoss(
    rotation_loss_type='geodesic',
    translation_loss_type='l2',
    rotation_weight=10.0,  # 旋转权重更大
    translation_weight=1.0,
)

# 方案3: 稳定训练
PoseLoss(
    rotation_loss_type='quaternion',
    translation_loss_type='smooth_l1',
    rotation_weight=5.0,
    translation_weight=1.0,
)
```

## 🧪 测试

```bash
# 测试数据集
cd /home/yons/Projects/SplatLoc
python implicit_correspondence/data/dataset.py

# 测试损失函数
python implicit_correspondence/losses/pose_loss.py

# 运行完整示例
python implicit_correspondence/example_usage.py
```

## 💡 使用建议

1. **数据增强**：
   - 训练时设置 `augment=True`
   - 验证/测试时设置 `augment=False`

2. **损失函数选择**：
   - 位姿估计：`geodesic + l2`
   - 训练不稳定：`quaternion + smooth_l1`
   - 注重旋转：增大 `rotation_weight`

3. **权重调整**：
   - 根据任务需求调整旋转和平移的相对重要性
   - 旋转损失单位是度，平移损失单位是米
   - 建议先用均衡权重（1.0, 1.0），再根据结果调整

4. **深度信息**：
   - 如果有深度图，设置 `use_depth=True` 可以得到更准确的3D点
   - 没有深度图时，会生成随机深度（用于训练decoder）

5. **批处理**：
   - 必须使用提供的 `collate_fn` 来处理变长的点云
   - `sample_indices` 记录每个点属于哪个样本

## 📝 完整训练示例

```python
import torch
from torch.utils.data import DataLoader
from implicit_correspondence.data import CorrespondenceDataset, collate_fn
from implicit_correspondence.losses import PoseLoss

# 1. 准备数据
train_dataset = CorrespondenceDataset(
    data_root="/home/yons/Projects/data/room_0",
    scene_name="Sequence_1",
    augment=True,
    max_samples=500,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=8,
)

# 2. 定义模型（这里需要你自己实现）
model = YourPoseEstimationModel()
model = model.cuda()

# 3. 定义损失和优化器
pose_loss = PoseLoss(
    rotation_loss_type='geodesic',
    translation_loss_type='l2',
    rotation_weight=1.0,
    translation_weight=1.0,
).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. 训练循环
for epoch in range(100):
    model.train()
    
    for batch in train_loader:
        # 数据移到GPU
        images = batch['image'].cuda()
        poses_gt = batch['pose'].cuda()
        
        # 前向传播
        poses_pred = model(images)
        
        # 计算损失
        losses = pose_loss(poses_pred, poses_gt, return_components=True)
        loss = losses['loss']
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 打印日志
        if batch_idx % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, "
                  f"Rot: {losses['rotation_loss'].item():.2f}°, "
                  f"Trans: {losses['translation_loss'].item():.4f}m")
```

## 🔗 相关文件

- 主训练脚本：`train_decoder.py`
- 配置文件：`configs/replica_nerf/base_config.yaml`
- 自动编码器：`autoencoder/`
- 特征解码器：`models/decoders.py`

## 📚 依赖项

```
torch
torchvision
numpy
Pillow
opencv-python
```

## ⚠️ 注意事项

1. 确保相机位姿文件 `traj_w_c.txt` 格式正确（每行16个数字）
2. RGB图像命名格式必须是 `rgb_<数字>.png`
3. 如果使用深度图，命名格式必须是 `depth_<数字>.png`
4. 图像索引必须与位姿文件的行号对应
5. 损失函数中的旋转矩阵必须是正交矩阵（行列式为1）

## 🐛 常见问题

**Q: 位姿数量与图像数量不匹配？**  
A: 检查RGB文件命名和位姿文件行数，确保索引对应。

**Q: 损失函数返回NaN？**  
A: 检查预测的旋转矩阵是否正交，可以尝试使用 `quaternion` 损失。

**Q: 训练不稳定？**  
A: 尝试使用 `smooth_l1` 平移损失，或调整损失权重。

**Q: 如何可视化结果？**  
A: 可以使用 `batch['points_2d']` 和 `batch['points_3d']` 进行可视化。

## 📧 联系

如有问题，请参考项目主README或提交issue。
