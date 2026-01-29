# 隐式对应关系训练 - 完整实现流程分析

## 修订时间: 2026-01-19 (视锥裁剪 + 位置编码版本)

**重要更新**:
- ✅ 融合特征使用原始35x46尺寸（不再上采样）
- ✅ 添加2D Sine/Cosine位置编码（256-dim）
- ✅ 添加3D Sine/Cosine位置编码（258-dim → 256-dim）

---

## 数据流程图

```
训练图像 (rgb_X.png)
    ↓
Dataset加载
    ├─ 加载RGB图像 [3, 640, 480]
    ├─ 加载相机位姿 c2w [4, 4]
    ├─ 加载融合特征 [256, 480, 640]
    └─ 生成2D-3D对应 (视锥裁剪)
         ↓
    [1] 从Gaussian点云视锥裁剪
         - Gaussian点云 (417,758点)
         - 投影到当前视角
         - 过滤: 深度>0 且在图像内
         - 随机采样1024个点
         ↓
    [2] 得到对应关系
         - points_2d: [1024, 2] (u, v)
         - points_3d: [1024, 3] (world)
         ↓
DataLoader批处理
    ├─ 图像: [B, 3, 640, 480]
    ├─ 位姿: [B, 4, 4]
    ├─ 融合特征: [B, 256, 480, 640]
    ├─ 2D点: [total_N, 2] 拼接
    ├─ 3D点: [total_N, 3] 拼接
    └─ sample_indices: [total_N]
         ↓
[3] 特征提取 (_extract_features)
    ├─ 3D特征: 
    │   pts_3d [total_N, 3]
    │       ↓
    │   FeatureDecoder(pts_3d)
    │       ↓
    │   pcd_feats_flat [total_N, 256]
    │       ↓
    │   按sample_indices重组
    │       ↓
    │   pcd_feats [B, max_N, 256] (padded)
    │       ↓
    │   [NEW] 添加3D位置编码
    │       coords_3d [B, N, 3] (世界坐标)
    │           ↓
    │       PositionalEncoding3D (258-dim)
    │           ↓
    │       Linear投影 (258 → 256)
    │           ↓
    │       pcd_feats = pcd_feats + pos_enc_3d
    │
    └─ 2D特征:
        fused_features [B, 256, 35, 46]  ← 注意：不再上采样
            ↓
        对每个样本i:
            pts_2d_i [N_i, 2] (图像坐标 640x480)
                ↓
            坐标映射: 图像 → 特征
                u_feat = u * (46/640)
                v_feat = v * (35/480)
                ↓
            归一化到[-1,1]
                ↓
            grid_sample采样 (从35x46特征图)
                ↓
            sampled_feats [N_i, 256]
            ↓
        img_feats [B, max_N, 256] (padded)
            ↓
        [NEW] 添加2D位置编码
            coords_2d_norm [B, N, 2] (归一化到[0,1])
                ↓
            PositionalEncoding2D (256-dim)
                ↓
            img_feats = img_feats + pos_enc_2d
         ↓
[4] ICPoseNet前向传播
    img_feats [B, N, 256]
    pcd_feats [B, N, 256]
         ↓
    可学习Query [128, 256]
         ↓
    扩展到batch: [B, 128, 256]
         ↓
    [5] 跨模态融合 (6层)
         层0: Query ← Image (cross-attention)
         层1: Query ← PointCloud (cross-attention)
         层2: Query ← Image
         层3: Query ← PointCloud
         层4: Query ← Image
         层5: Query ← PointCloud
         ↓
    融合后Query: [B, 128, 256]
         ↓
    [6] 位姿回归
         全局平均池化
              ↓
         MLP [256 → 512]
              ↓
         分支1: rotation_head → [B, 3] (轴角)
         分支2: translation_head → [B, 3]
              ↓
         pose_6d: [B, 6] = [tx,ty,tz, rx,ry,rz]
              ↓
         转换为4x4矩阵
              ↓
         pose_matrix: [B, 4, 4]
         ↓
[7] 损失计算
    pose_pred [B, 4, 4]
    pose_gt [B, 4, 4]
         ↓
    旋转损失: geodesic_distance(R_pred, R_gt)
    平移损失: mse_loss(t_pred, t_gt)
         ↓
    total_loss = rot_loss + trans_loss
         ↓
[8] 反向传播与优化
    loss.backward()
         ↓
    梯度裁剪 (max_norm=5.0)
         ↓
    optimizer.step() (AdamW, lr=5e-5)
         ↓
    scheduler.step() (Cosine)
```

---

## 关键模块详解

### 1. Dataset: 视锥裁剪生成对应关系

**文件**: `data/dataset.py`

```python
def _generate_2d_3d_pairs(self, idx, num_samples=1024):
    """
    使用视锥裁剪从Gaussian点云生成2D-3D对应
    参考: SplatLoc/test.py 的 get_frusm_pts()
    """
    # 1. 获取相机位姿
    c2w = self.poses[idx]  # camera-to-world
    w2c = np.linalg.inv(c2w)  # world-to-camera
    
    # 2. 获取所有Gaussian点 (417,758个)
    all_pts = self.gaussian_points  # [N, 3] world coords
    
    # 3. 转换到相机坐标系
    points_camera = (all_pts @ w2c[:3, :3].T) + w2c[:3, 3]
    
    # 4. 投影到图像平面
    projected_points = (K @ points_camera.T).T  # [N, 3]
    projected_points = projected_points[:, :2] / projected_points[:, 2:3]  # [N, 2] (u,v)
    
    # 5. 视锥裁剪: 过滤可见点
    mask = (points_camera[:, 2] > 0.05) &  # 深度 > 0
           (0 <= u < 640) & (0 <= v < 480)  # 在图像内
    
    visible_pts_3d = all_pts[mask]  # [N_vis, 3]
    visible_pts_2d = projected_points[mask]  # [N_vis, 2]
    
    # 6. 随机采样1024个点
    indices = np.random.choice(N_vis, 1024, replace=False)
    
    return visible_pts_2d[indices], visible_pts_3d[indices]
```

**关键点**:
- ✅ 3D点来自Gaussian点云，不是深度图反投影
- ✅ 2D坐标是3D点的投影，保证真实对应
- ✅ 与测试时（SplatLoc/test.py）流程一致

**对比旧实现**:
| 方面 | 旧实现（深度反投影）❌ | 新实现（视锥裁剪）✅ |
|------|---------------------|------------------|
| 3D点来源 | 深度图反投影 | Gaussian点云 |
| 训练/测试一致性 | 不一致 | 一致 |
| 是否需要深度图 | 必须 | 不需要 |
| 对应关系质量 | 依赖深度图质量 | 依赖点云密度 |

---

### 2. Feature Extraction: 双模态特征提取

**文件**: `train.py` → `_extract_features()`

#### 2.1 3D特征提取

```python
# 输入: pts_3d [total_N, 3] (world coords)
# 来源: 从Gaussian点云视锥裁剪得到的3D点

pcd_feats_flat = self.feat_decoder(pts_3d)  # [total_N, 256]
# FeatureDecoder: HashGrid编码 + MLP
#   - HashGrid: 将3D坐标编码为潜在特征
#   - MLP: 4层[embed_dim → 1235, 46] (预提取融合特征，保持原始尺寸)
#   pts_2d [total_N, 2] (u, v像素坐标 in 640x480)

for b in range(batch_size):
    pts_2d_b = pts_2d[sample_indices == b]  # [N_b, 2]
    
    # 坐标映射: 图像(640x480) → 特征(46x35)
    H_img, W_img = 480, 640
    H_feat, W_feat = 35, 46
    u_feat = pts_2d_b[:, 0] * (W_feat / W_img)  # [N_b]
    v_feat = pts_2d_b[:, 1] * (H_feat / H_img)  # [N_b]
    
    # 归一化到[-1, 1] for grid_sample
    grid_x = 2.0 * u_feat / (W_feat-1) - 1.0
    grid_y = 2.0 * v_feat / (H_feat-1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [N_b, 2]
    
    # 双线性插值采样
    sampled_feats = F.grid_sample(
        fused_features[b:b+1],  # [1, 256, 35, 46]
        grid.view(1, 1, N_b, 2),
        mode='bilinear',
        align_corners=True
    )  # [1, 256, 1, N_b]
    
    img_feats[b, :N_b] = sampled_feats.squeeze()

# 添加2D位置编码
coords_2d_norm = pts_2d / torch.tensor([639.0, 479.0])  # 归一化到[0,1]
coords_2d_batch = reorganize_to_batch(coords_2d_norm)  # [B, N, 2]
pos_enc_2d = PositionalEncoding2D(coords_2d_batch)  # [B, N, 256]
img_feats = img_feats + pos_enc_2d  # 特征 + 位置编码

# 输出: img_feats [B, max_N, 256]
```

**融合特征来源**: SD+DINO特征融合
- **路径**: `data/room_0/Sequence_1/features_compressed/fused/`
- **格式**: `rgb_X_fused_768x35x46_compressed.pt`
- **原始尺寸**: [35, 46, 256]
- **处理**: 转换为[256, 35, 46]，**不再上采样**

**特点**:
- ✅ 真实的2D图像特征（预提取）
- ✅ 保持原始35x46分辨率，避免上采样引入的模糊
- ✅ 双线性插值保证平滑性
- ✅ 添加2D位置编码，弥补低分辨率的几何信息损失
- ✅ 与3D特征维度一致（256-dim）

---

### 2.3 位置编码

#### 2.3.1 为什么需要位置编码？

**问题**: 35×46特征图分辨率太低，丢失了精细的几何结构信息

**解决方案**:
1. **2D位置编码**: 告诉网络特征点在画面中的准确(u,v)坐标
2. **3D位置编码**: 告诉网络3D点在空间中的准确(x,y,z)坐标

#### 2.3.2 2D Sine/Cosine位置编码

```python
class PositionalEncoding2D(nn.Module):
    def __init__(self, embed_dim=256, temperature=10000):
        # 频率系数
        dim_t = temperature ** (2 * torch.arange(embed_dim//4) / (embed_dim//4))
        
    def forward(self, coords_2d):
        # coords_2d: [B, N, 2] 归一化到[0,1]的(u,v)
        u = coords_2d[..., 0]  # [B, N]
        v = coords_2d[..., 1]  # [B, N]
        
   pos_enc_2d: PositionalEncoding2D (256-dim)  ← NEW
├─ pos_enc_3d: PositionalEncoding3D (258-dim)  ← NEW
├─ pos_enc_3d_proj: Linear(258 → 256)          ← NEW
├─      # u方向: 128维
        pos_u = u.unsqueeze(-1) / self.dim_t  # [B, N, 64]
        pos_u = torch.cat([pos_u.sin(), pos_u.cos()], dim=-1)  # [B, N, 128]
        
        # v方向: 128维
        pos_v = v.unsqueeze(-1) / self.dim_t  # [B, N, 64]
        pos_v = torch.cat([pos_v.sin(), pos_v.cos()], dim=-1)  # [B, N, 128]
        
        # 拼接
        return torch.cat([pos_u, pos_v], dim=-1)  # [B, N, 256]
```

**特性**:
- 使用不同频率的正弦/余弦函数编码位置
- 每个坐标维度使用128个频率（64个sin + 64个cos）
- 总维度: 256（u方向128 + v方向128）
- 参考DETR的位置编码设计

#### 2.3.3 3D Sine/Cosine位置编码

```python
class PositionalEncoding3D(nn.Module):
    def __init__(self, embed_dim=258, normalize=True, scale_factor=0.1):
        # embed_dim=258可被3整除，每个轴86维
        dim_t = temperature ** (2 * torch.arange(86//2) / (86//2))
        
    def forward(self, coords_3d):
        # coords_3d: [B, N, 3] 世界坐标(x,y,z)
        
        # 可选归一化（tanh到[-1,1]）
        if self.normalize:
            coords_norm = torch.tanh(coords_3d * self.scale_factor)
        
        # x方向: 86维
        pos_x = x.unsqueeze(-1) / self.dim_t  # [B, N, 43]
        pos_x = torch.cat([pos_x.sin(), pos_x.cos()], dim=-1)  # [B, N, 86]
        
        # y, z方向同理
        # ...
        
        # 拼接 [B, N, 258] = 86*3
        return torch.cat([pos_x, pos_y, pos_z], dim=-1)
```

**投影到256维**:
```python
self.pos_enc_3d_proj = nn.Linear(258, 256)
pos_enc_3d_256 = self.pos_enc_3d_proj(pos_enc_3d_258)
```

**特性**:
- 3个轴均匀分配编码维度（258/3 = 86）
- 使用scale_factor=0.1和tanh归一化，适应Replica场景尺度
- 通过Linear投影到256维，与2D特征对齐[b:b+1],  # [1, 256, 480, 640]
        grid.view(1, 1, N_b, 2),
        mode='bilinear',
        align_corners=True
    )  # [1, 256, 1, N_b]
    
    img_feats[b, :N_b] = sampled_feats.squeeze()

# 输出: img_feats [B, max_N, 256]
```

**融合特征来源**: SD+DINO特征融合
- **路径**: `data/room_0/Sequence_1/features_compressed/fused/`
- **格式**: `rgb_X_fused_768x35x46_compressed.pt`
- **原始尺寸**: [35, 46, 256]
- **上采样**: → [480, 640, 256]

**特点**:
- ✅ 真实的2D图像特征（预提取）
- ✅ 双线性插值保证平滑性
- ✅ 与3D特征维度一致（256-dim）

---

### 3. ICPoseNet: 跨模态融合与位姿回归

**文件**: `ic_models/ic_pose_net.py`

#### 3.1 网络架构

```python
ICPoseNet(
    feature_dim=256,
    num_queries=128,
    fusion_layers=6,
    num_heads=8,
    dropout=0.1
)

组件:
├─ query_embed: [1, 128, 256] 可学习查询向量
├─ fusion_module: CrossModalFusionModule (6层)
│   ├─ img_encoder: TransformerLayer (自注意力)
│   ├─ pcd_encoder: TransformerLayer (自注意力)
│   └─ transformer_layers: 6 × TransformerLayer (交叉注意力)
└─ pose_regressor: PoseRegressor
    ├─ feature_aggregation: [256 → 512]
    ├─ rotation_head: [512 → 256 → 3]
    └─ translation_head: [512 → 256 → 3]
```

#### 3.2 前向传播流程

```python
def forward(img_feats, pcd_feats):
    # img_feats: [B, N_img, 256]
    # pcd_feats: [B, N_pcd, 256]
    
    # 1. 扩展可学习query到batch
    query = self.query_embed.expand(B, -1, -1)  # [B, 128, 256]
    
    # 2. 跨模态融合
    query_list, img_tokens, pcd_tokens = self.fusion_module(
        query, img_feats, pcd_feats
    )
    # 融合策略（6层交替）:
    #   Layer 0: query ← img (cross-attn)
    #   Layer 1: query ← pcd (cross-attn)
    #   Layer 2: query ← img
    #   Layer 3: query ← pcd
    #   Layer 4: query ← img
    #   Layer 5: query ← pcd
    
    # 3. 位姿回归
    fused_feats = query_list[-1]  # [B, 128, 256]
    
    # 全局平均池化
    global_feat = fused_feats.mean(dim=1)  # [B, 256]
    
    # MLP特征聚合
    feat = self.feature_aggregation(global_feat)  # [B, 512]
    
    # 分支回归
    rotation = self.rotation_head(feat)  # [B, 3] 轴角表示
    translation = self.translation_head(feat)  # [B, 3]
    
    # 转换为4x4矩阵
    pose_matrix = pose_6d_to_matrix(rotation, translation)  # [B, 4, 4]
    
    return pose_matrix, pose_6d, rotation, translation
```

**参数量**: 9.1M
- query_embed: 128 × 256 = 32,768
- fusion_module: ~8.5M
- pose_regressor: ~0.6M

---

### 4. 损失函数: PoseLoss

**文件**: `losses/pose_loss.py`

```python
def forward(pose_pred, pose_gt):
    # 提取旋转和平移
    R_pred = pose_pred[:, :3, :3]  # [B, 3, 3]
    t_pred = pose_pred[:, :3, 3]   # [B, 3]
    R_gt = pose_gt[:, :3, :3]
    t_gt = pose_gt[:, :3, 3]
    
    # 1. 旋转损失: 测地距离 (SO(3)流形)
    rotation_loss = geodesic_distance(R_pred, R_gt)
    # 计算: arccos((trace(R_pred^T @ R_gt) - 1) / 2)
    # 单位: 弧度 (rad)
    
    # 2. 平移损失: L2距离
    translation_loss = F.mse_loss(t_pred, t_gt)
    # 单位: 米^2 (m²)
    
    # 3. 总损失
    loss = rotation_weight * rotation_loss + translation_weight * translation_loss
    #      ^^^^^^^^^^^^^^^^                   ^^^^^^^^^^^^^^^^^^^
    #      默认 = 1.0                         默认 = 1.0
    
    return {'loss': loss, 'rotation_loss': ..., 'translation_loss': ...}
```

**损失权重**: 均为1.0
- 旋转和平移同等重要
- 可根据任务调整（如SLAM更重视旋转）

---

### 5. 训练循环

**文件**: `train.py` → `train_epoch()`

```python + 位置编码
sample = dataset[0]
plt.figure(figsize=(18, 6))

# 子图1: 2D点投影
plt.subplot(1,3,1)
plt.imshow(sample['image'].permute(1,2,0))
plt.scatter(sample['points_2d'][:,0], sample['points_2d'][:,1], s=1, c='r')
plt.title('2D Points (视锥裁剪投影)')

# 子图2: 融合特征可视化
fused_feat = sample['fused_feature']  # [256, 35, 46]
feat_vis = fused_feat.mean(0).numpy()  # [35, 46]
plt.subplot(1,3,2)
plt.imshow(feat_vis, cmap='viridis')
plt.title('融合特征热图 (35x46)')

# 子图3: 位置编码可视化
from ic_models.positional_encoding import PositionalEncoding2D
pos_enc = PositionalEncoding2D(256)
coords_2d = sample['points_2d'] / torch.tensor([639.0, 479.0])
pos_emb = pos_enc(coords_2d.unsqueeze(0))  # [1, N, 256]
pos_vis = pos_emb[0].norm(dim=-1).numpy()
plt.subplot(1,3,3)
plt.scatter(sample['points_2d'][:,0], sample['points_2d'][:,1], 
            c=pos_vis, s=5, cmap='coolwarm')
plt.title('2D位置编码强度')
```

### 2. 统计分析
```python
# 特征分辨率vs精度
print(f"特征分辨率: 35x46 = {35*46} 像素")
print(f"图像分辨率: 480x640 = {480*640} 像素")
print(f"下采样比例: {(480*640)/(35*46):.1f}x")

# 位置编码有效性
pos_enc_std = pos_enc_2d.std().item()
print(f"2D位置编码标准差: {pos_enc_std:.4f}")  # 应该>0.1
```

### 3. 梯度监控（含位置编码层）
```python
# 检查各模块梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        if 'pos_enc' in name:
            print(f"[位置编码] {name}: {grad_norm:.4f}")
        else:
            print(f"{name}: {grad_norm:.4f}")
```

### 4. 特征质量检查（含位置编码）
```python
# 检查2D和3D特征相似度（加入位置编码后）
img_feats_with_pos = img_feats + pos_enc_2d
pcd_feats_with_pos = pcd_feats + pos_enc_3d_proj

cosine_sim = F.cosine_similarity(img_feats_with_pos, pcd_feats_with_pos, dim=-1)
print(f"特征+位置编码相似度: {cosine_sim.mean():.4f}")
# 期望: 对应点相似度高，非对应点相似度低
```

### 5. 位置编码频率分析
```python
# 可视化位置编码的频率分量
import matplotlib.pyplot as plt
pos_enc_2d = PositionalEncoding2D(256)
coords = torch.linspace(0, 1, 100).unsqueeze(0).unsqueeze(-1)  # [1, 100, 1]
coords_2d = coords.expand(-1, -1, 2)  # [1, 100, 2]
pos_emb = pos_enc_2d(coords_2d)  # [1, 100, 256]

plt.figure(figsize=(12, 4))
plt.imshow(pos_emb[0].T.numpy(), aspect='auto', cmap='seismic')
plt.xlabel('坐标位置 (0-1)')
plt.ylabel('编码维度 (0-255)')
plt.title('2D位置编码频率可视化')
plt.colorbar()
- Epochs: 100

---

## 关键差异对比

### 旧实现 vs 新实现

| 组件 | 旧实现❌ | 新实现✅ |
|------|---------|---------|
| **2D-3D对应生成** | 深度图反投影 | Gaussian点云视锥裁剪 |
| **3D点来源** | 深度→相机→世界坐标 | Gaussian点云（世界坐标） |
| **2D点来源** | 随机像素坐标 | 3D点投影坐标 |
| **深度图依赖** | 必需 | 不需要 |
| **训练/测试一致性** | 不一致（训练用深度，测试用点云） | 一致（都用点云） |
| **2D特征** | 复制3D特征 | 从融合特征采样 |
| **HashGrid配置** | 不匹配（6.7M vs 8.9M） | 匹配 |

---

## 潜在问题检查清单

### ✅ 已解决
1. ✅ 2D-3D对应关系真实性 - 使用视锥裁剪
2. ✅ 2D和3D特征模态差异 - 融合特征 vs 点云特征
3. ✅ HashGrid配置匹配 - scene.bound已修正
4. ✅ 训练/测试一致性 - 都使用Gaussian点云

### ⚠️ 需要检查
1. **视锥裁剪可见点数量**
   - 如果某些视角可见点过少（<1024），会重复采样
   - 建议: 记录每帧可见点数量分布

2. **融合特征质量**
   - 上采样从35×46→480×640可能引入模糊
   - 建议: 可视化采样特征，检查是否有异常

3. **位姿表示**
   - 旋转使用轴角表示（3-param），训练初期可能不稳定
   - 可能改进: 使用四元数（4-param）或6D连续表示

4. **可学习Query初始化**
   - 当前使用随机初始化
   - 可能改进: 使用预训练或特定初始化策略

5. **损失权重**
   - 旋转和平移权重都是1.0
   - 不同任务可能需要调整（如室内定位更重视平移）

6. **过拟合风险**
   - 训练集900张，验证集50张，同一场景
   - 建议: 监控train/val loss差距

---

## 下一步调试建议

### 1. 可视化检查
```python
# 检查2D-3D对应关系
sample = dataset[0]
plt.figure(figsize=(12, 6))
plt.subplot(1,2,1)
plt.imshow(sample['image'].permute(1,2,0))
plt.scatter(sample['points_2d'][:,0], sample['points_2d'][:,1], s=1, c='r')
5. ✅ **保持原始35x46特征分辨率**（避免上采样模糊）
6. ✅ **2D Sine/Cosine位置编码**（补偿几何信息损失）
7. ✅ **3D Sine/Cosine位置编码**（空间位置信息）

训练现在基于更合理的数据流程，预期能学到更准确的隐式对应关系。

### 关键参数

| 参数 | 值 | 说明 |
|------|---|------|
| 融合特征分辨率 | 35×46 | 原始分辨率，不上采样 |
| 图像分辨率 | 640×480 | Replica标准分辨率 |
| 2D位置编码维度 | 256 | 与特征维度一致 |
| 3D位置编码维度 | 258→256 | 线性投影到特征维度 |
| 位置编码温度 | 10000 | 标准Transformer设置 |
| 3D归一化scale | 0.1 | 适配Replica场景尺度 |
plt.subplot(1,2,2)
# 可视化3D点分布
```

### 2. 统计分析
```python
# 统计每帧可见点数量
visible_counts = []
for idx in range(len(dataset)):
    sample = dataset[idx]
    visible_counts.append(len(sample['points_3d']))

print(f"可见点数量: min={min(visible_counts)}, max={max(visible_counts)}, mean={np.mean(visible_counts)}")
```

### 3. 梯度监控
```python
# 检查各模块梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        print(f"{name}: {grad_norm:.4f}")
```

### 4. 特征质量检查
```python
# 检查2D和3D特征相似度
cosine_sim = F.cosine_similarity(img_feats, pcd_feats, dim=-1)
print(f"2D-3D特征余弦相似度: {cosine_sim.mean():.4f}")
# 期望: 对应点相似度高，非对应点相似度低
```

---

## 总结

当前实现使用**视锥裁剪**从Gaussian点云生成2D-3D对应关系，与SplatLoc/test.py的测试流程一致。主要改进：

1. ✅ 真实对应关系（点云投影 vs 深度反投影）
2. ✅ 训练/测试一致性（都用Gaussian点云）
3. ✅ 双模态特征（融合特征 vs 点云特征）
4. ✅ 配置匹配（HashGrid）

训练现在基于更合理的数据流程，预期能学到更准确的隐式对应关系。
