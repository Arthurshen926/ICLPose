# EXP009 vs EXP010 架构对比

## 核心差异

### EXP009 (旧架构 - 仅Cross-Attention)
```
Input: Query Embeddings [B, 128, 256]
       Image Features  [B, N_img, 256]
       Point Features  [B, N_pcd, 256]

Layer 0: Query ──Cross──> Image
Layer 1: Query ──Cross──> Point
Layer 2: Query ──Cross──> Image
Layer 3: Query ──Cross──> Point
Layer 4: Query ──Cross──> Image
Layer 5: Query ──Cross──> Point
Layer 6: Query ──Cross──> Image
Layer 7: Query ──Cross──> Point

Output: Fused Queries [B, 128, 256]

问题: Queries之间无法交互！
```

### EXP010 (新架构 - Self + Cross交替)
```
Input: Query Embeddings [B, 128, 256]
       Image Features  [B, N_img, 256]
       Point Features  [B, N_pcd, 256]

Layer 0: Query ──Self──> Query  (聚合信息)
         Query ──Cross──> Image (从图像学习)

Layer 1: Query ──Self──> Query  (聚合信息)
         Query ──Cross──> Point (从点云学习)

Layer 2: Query ──Self──> Query  (聚合信息)
         Query ──Cross──> Image (再次从图像)

Layer 3: Query ──Self──> Query  (聚合信息)
         Query ──Cross──> Point (再次从点云)

... (重复4次，共8层)

Output: Fused Queries [B, 128, 256]

优势: Queries可以相互交流，共享跨模态信息！
```

## 位置编码处理对比

### EXP009
```python
# 1. 加载特征 (已归一化, ||f|| = 1.0)
img_feats = fused_feature  # [B, N, 256]
pcd_feats = feat_decoder  # [B, N, 256]

# 2. 生成位置编码
pos_enc_2d = pos_enc_2d(coords)  # [B, N, 256]

# 3. 直接相加
img_feats = img_feats + pos_enc_2d  
# 问题: ||img_feats|| 从1.0变成11.8！

# 4. 重新归一化
img_feats = normalize(img_feats)
# 虽然修复了归一化，但改变了特征空间
```

### EXP010
```python
# 1. 加载特征 (保持归一化, ||f|| = 1.0)
img_feats = fused_feature  # [B, N, 256]
pcd_feats = feat_decoder  # [B, N, 256]

# 2. 生成位置编码（独立）
img_pos_embeds = pos_enc_2d(coords)  # [B, N, 256]
pcd_pos_embeds = pos_enc_3d(coords)  # [B, N, 256]

# 3. 传递给Transformer（内部处理）
output = transformer(
    q=query, k=img_feats, v=img_feats,
    q_embeds=None,
    k_embeds=img_pos_embeds  # 在内部通过projection添加
)

# 特征始终保持L2归一化，位置编码在Attention内部融合
```

## 性能对比（预期）

| 指标 | EXP009 (150 epochs) | EXP010 (预期) | 改进 |
|------|---------------------|---------------|------|
| **训练角度误差** | 3.74° | ~4-5° | 略微上升（正常） |
| **训练平移误差** | 0.27m | ~0.3m | 略微上升（正常） |
| **验证角度误差** | **17.43°** | **<10°** | **-43%** |
| **验证平移误差** | **6.48m** | **<3m** | **-54%** |
| **Train-Val Gap** | **4.7x** | **<2x** | **-57%** |
| **过拟合程度** | 严重 | 轻微 | 显著改善 |

## 代码变更统计

```
新建: 5个文件
  - modules/transformer.py (348行)
  - modules/fusion_module.py (141行)
  - configs/train_config_exp010.yaml
  - run_exp010.sh
  - test_new_implementation.py

修改: 3个文件
  - train.py (~30行修改)
  - ic_models/ic_pose_net.py (~20行修改)
  - modules/__init__.py (~10行修改)

备份: 2个文件
  - modules/transformer_old.py
  - modules/fusion_module_old.py
```

## 关键发现

### 为什么需要Self-Attention？

**场景**: 128个query embeddings需要从2D图像和3D点云中提取位姿信息

**没有Self-Attention (EXP009)**:
- Query 1从Image学到信息A
- Query 2从Point学到信息B
- **问题**: Query 1和2无法共享AB，各自为战
- **结果**: 模型容易记忆训练数据（过拟合）

**有Self-Attention (EXP010)**:
- Query 1从Image学到信息A
- Query 2从Point学到信息B  
- **Self-Attention**: Query 1看到B, Query 2看到A
- **结果**: 所有queries聚合全局信息（泛化）

### ICL-I2PReg的智慧

论文作者深知多模态融合的难点：
1. **先聚合，再学习**: Self → Cross交替模式
2. **位置编码解耦**: 通过embeds参数，不破坏特征空间
3. **完整的Attention**: 支持weights/masks/embeds灵活控制

我们EXP009最大的问题就是忽略了第1点！

## 运行EXP010

```bash
# 测试新实现
python test_new_implementation.py

# 启动训练
./run_exp010.sh

# 或直接运行
python train.py --config configs/train_config_exp010.yaml
```

祝训练顺利！ 🚀
