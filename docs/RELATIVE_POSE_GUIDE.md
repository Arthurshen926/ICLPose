# Relative Pose Training 实现说明

## 当前问题

当前的relative pose实现（`use_relative_pose=true`）有一个重大缺陷：

```python
# 当前实现（不合理）
pose_init = gt_poses_abs[0:1].expand(gt_poses_abs.shape[0], 4, 4).clone()
gt_poses = compute_relative_pose(gt_poses_abs, pose_init)
```

**问题**：所有样本都使用batch第一帧作为参考位姿，这不合理因为：
1. Batch内的帧可能时间跨度很大（例如第1帧和第100帧）
2. 无法反映真实应用场景（每帧应该有自己的初始位姿估计）

## 正确的实现方式

你的理解完全正确！应该实现为：

### 方案1：使用上一帧作为初始位姿（SLAM/Tracking场景）

```python
# 每帧使用上一帧的位姿作为初始估计
if idx == 0:
    pose_init = torch.eye(4)  # 第一帧使用单位矩阵
else:
    pose_init = poses[idx-1]  # 使用上一帧位姿

# 学习增量变换
pose_delta = model(...)
pose_pred = pose_delta @ pose_init  # 组合得到最终位姿
```

### 方案2：使用每帧自己的初始位姿（General定位场景）

```python
# 数据集返回每帧的初始位姿估计
sample = {
    'pose_gt': gt_pose,        # Ground truth
    'pose_init': init_pose,    # 初始估计（如粗定位、SLAM估计等）
    ...
}

# 训练时学习相对变换
pose_delta = model(...)
pose_pred = pose_delta @ pose_init
loss = compute_loss(pose_pred, pose_gt)
```

### 方案3：添加噪声位姿作为初始估计（数据增强）

```python
# 在GT位姿上添加噪声作为初始估计
def add_pose_noise(pose_gt, rotation_noise=10.0, translation_noise=0.5):
    # 添加旋转噪声（角度）
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    angle = np.random.uniform(-rotation_noise, rotation_noise) * np.pi / 180
    R_noise = rotation_matrix(axis, angle)
    
    # 添加平移噪声（米）
    t_noise = np.random.randn(3) * translation_noise
    
    pose_init = pose_gt.copy()
    pose_init[:3, :3] = R_noise @ pose_gt[:3, :3]
    pose_init[:3, 3] += t_noise
    
    return pose_init

# 训练时
pose_init = add_pose_noise(pose_gt)
pose_delta = model(...)
pose_pred = pose_delta @ pose_init
```

## 实现步骤

### Step 1: 修改数据集（推荐方案3）

在`data/dataset.py`的`__getitem__`中添加：

```python
def __getitem__(self, idx: int):
    # ... 现有代码 ...
    
    # 如果使用relative pose模式
    if self.use_relative_pose:
        pose_init = self._generate_noisy_pose(pose, 
                                               rot_noise=self.rotation_noise,
                                               trans_noise=self.translation_noise)
    else:
        pose_init = None
    
    sample = {
        'image': image,
        'pose': pose,              # GT位姿
        'pose_init': pose_init,    # 初始位姿（如果使用relative pose）
        ...
    }
    return sample

def _generate_noisy_pose(self, pose_gt, rot_noise=10.0, trans_noise=0.5):
    """生成带噪声的初始位姿"""
    pose_init = pose_gt.clone()
    
    # 添加旋转噪声
    axis = torch.randn(3)
    axis = axis / axis.norm()
    angle = torch.rand(1) * 2 * rot_noise - rot_noise  # [-rot_noise, +rot_noise]
    angle_rad = angle * np.pi / 180.0
    
    # Rodrigues公式
    K = torch.tensor([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
    R_noise = torch.eye(3) + torch.sin(angle_rad) * K + (1 - torch.cos(angle_rad)) * (K @ K)
    
    pose_init[:3, :3] = R_noise @ pose_init[:3, :3]
    
    # 添加平移噪声
    t_noise = torch.randn(3) * trans_noise
    pose_init[:3, 3] += t_noise
    
    return pose_init
```

### Step 2: 修改训练代码

在`train.py`的`train_epoch`和`validate`中：

```python
# 3. 准备GT
gt_poses_abs = batch['pose'].to(self.device)  # (B, 4, 4) 绝对位姿

use_relative_pose = self.config['loss'].get('use_relative_pose', False)

if use_relative_pose:
    # 获取初始位姿
    pose_init = batch['pose_init'].to(self.device)  # (B, 4, 4)
    
    # 计算GT的相对位姿（相对于初始位姿）
    # gt_relative = pose_init^{-1} @ gt_abs
    pose_init_inv = torch.inverse(pose_init)
    gt_poses = torch.bmm(pose_init_inv, gt_poses_abs)
    
    # 网络预测的也是相对位姿
    # 最终评估时需要转回绝对位姿
else:
    gt_poses = gt_poses_abs
    pose_init = None

# 4. 计算损失（使用相对位姿）
loss_dict = self.pose_loss(
    pose_pred=pose_matrix_pred,
    pose_gt=gt_poses,
    return_components=True
)

# 6. 计算位姿误差（转回绝对位姿评估）
if use_relative_pose:
    # pose_abs = pose_init @ pose_relative
    pose_pred_abs = torch.bmm(pose_init, pose_matrix_pred)
else:
    pose_pred_abs = pose_matrix_pred

rot_error, trans_error = self._compute_pose_error(pose_pred_abs, gt_poses_abs)
```

### Step 3: 添加配置选项

在`configs/train_config.yaml`中：

```yaml
dataset:
  # Relative pose相关配置
  use_relative_pose: false
  rotation_noise: 10.0      # 初始位姿旋转噪声（度）
  translation_noise: 0.5    # 初始位姿平移噪声（米）

loss:
  # 是否使用相对位姿训练
  use_relative_pose: false
```

## 使用场景

### 何时使用Absolute Pose (当前默认)
- ✅ 全局定位任务（Visual Localization）
- ✅ 没有初始位姿估计
- ✅ 需要直接输出世界坐标系位姿

### 何时使用Relative Pose
- ✅ SLAM / Visual Odometry（有上一帧位姿）
- ✅ Pose Refinement（有粗定位结果）
- ✅ 增量式tracking
- ✅ 训练更稳定（学习小的增量变换）

## 好处

使用Relative Pose训练的优势：

1. **更稳定的训练**：学习小的增量变换比学习大的绝对位姿更容易
2. **更好的泛化**：网络学到的是"如何修正"而不是"绝对位置"
3. **更realistic**：符合真实应用场景（总有某种初始估计）
4. **数据增强**：通过改变噪声水平可以训练不同容错能力的模型

## 实验建议

1. **先验证Absolute Pose收敛**（当前已完成✅）
2. **添加Relative Pose实现**（按上述方案）
3. **对比实验**：
   - Baseline: absolute pose（当前）
   - Relative pose with noise (10° rotation, 0.5m translation)
   - Relative pose with larger noise (30° rotation, 1.0m translation)
4. **消融实验**：
   - 不同噪声水平的影响
   - 是否使用噪声增强的影响

## 注意事项

⚠️ **重要**：
- Relative pose训练时，损失计算用相对位姿
- 评估指标（角度误差、平移误差）必须用绝对位姿
- 保存的checkpoint要明确是absolute还是relative模式
- 推理时需要提供初始位姿估计（除非是absolute模式）

## 下一步

如果你想实现relative pose训练，我可以帮你：
1. 修改`data/dataset.py`添加噪声位姿生成
2. 修改`train.py`适配relative pose训练
3. 创建新的配置文件用于relative pose实验

需要我现在就实现吗？
