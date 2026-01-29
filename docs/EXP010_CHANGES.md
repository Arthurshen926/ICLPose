# EXP010 - 完全对齐ICL-I2PReg实现

## 修改日期
2026-01-28

## 修改原因
EXP009训练时发现严重过拟合（train: 3.74° vs val: 17.43°），通过与ICL-I2PReg对比发现架构偏差：
- **缺少Query Self-Attention层**（最关键）
- Position encoding应通过embeds参数传递，而非直接相加
- TransformerLayer缺少embeds/weights/masks支持

## 架构变更

### 1. TransformerLayer完全重写 (`modules/transformer.py`)

**新增功能**:
- `MultiHeadAttention`支持embeds参数:
  - `q_embeds`, `k_embeds`: 绝对位置编码 (APE)
  - `qk_embeds`: 相对位置编码 (RPE)
  - `v_embeds`: Value embeddings
  - `weights`: 注意力权重
  - `masks`: 注意力掩码
  - `return_attn`: 返回attention scores
  
- 新增`AttentionLayer`和`AttentionOutput`模块
- `TransformerLayer`完整结构: Attention → Output → FFN

**对齐ICL-I2PReg**: 完全匹配vision3d/layers/transformer.py

### 2. FusionModule改为Self-Cross交替 (`modules/fusion_module.py`)

**旧架构** (EXP009):
```
Layer 0: Cross(query, img)
Layer 1: Cross(query, pcd)
Layer 2: Cross(query, img)
...
```
Queries无法相互交互，信息聚合能力弱

**新架构** (EXP010 - 对齐ICL-I2PReg):
```
Layer 0: Self(query, query) → Cross(query, img)
Layer 1: Self(query, query) → Cross(query, pcd)
Layer 2: Self(query, query) → Cross(query, img)
...
```

**关键改进**:
- 每次Cross-Attention前，queries先通过Self-Attention聚合信息
- Self-Attention让queries可以相互交流，共享来自img/pcd的信息
- 这是ICL-I2PReg的核心设计，我们之前遗漏了

### 3. 位置编码传递方式 (`train.py`)

**旧方式** (EXP009):
```python
# 特征 + 位置编码，再归一化
img_feats_with_pe = img_feats + pos_enc_2d
img_feats = normalize(img_feats_with_pe)
```
问题: 破坏了特征空间，需要重新归一化

**新方式** (EXP010 - 对齐ICL-I2PReg):
```python
# 特征和位置编码分开传递
img_pos_embeds = pos_enc_2d
pcd_pos_embeds = pos_enc_3d_proj

# Transformer内部通过embeds参数处理
output = transformer(q, k, v, q_embeds=..., k_embeds=...)
```

**优势**:
- 特征保持L2归一化（||f|| = 1.0）
- 位置编码在Attention内部通过projection后再添加
- 更符合ICL-I2PReg的设计理念

### 4. ICPoseNet接口更新 (`ic_models/ic_pose_net.py`)

**新forward签名**:
```python
def forward(self, img_feats, pcd_feats, 
           img_pos_embeds=None, pcd_pos_embeds=None):
```

**改动**:
- 接收独立的位置编码参数
- 传递给FusionModule的embeds参数
- 输出改为9D (3平移 + 6D旋转)

## 文件变更列表

### 新建文件
- `modules/transformer.py` (完全重写)
- `modules/fusion_module.py` (完全重写)
- `configs/train_config_exp010.yaml`
- `run_exp010.sh`
- `test_new_implementation.py` (测试脚本)
- `EXP010_CHANGES.md` (本文档)

### 备份旧文件
- `modules/transformer_old.py` (EXP009版本)
- `modules/fusion_module_old.py` (EXP009版本)

### 修改文件
- `train.py`:
  - `_extract_features()`: 返回位置编码
  - `train_epoch()`: 传递位置编码
  - `validate()`: 传递位置编码
  
- `ic_models/ic_pose_net.py`:
  - `forward()`: 接收位置编码参数
  
- `modules/__init__.py`:
  - 移除`OverlapEstimator`
  - 添加`LearnableQueryEmbedding`
  - 添加`rotation_6d_to_matrix`, `matrix_to_rotation_6d`

## 参数配置 (exp010)

```yaml
output_dir: exp010
model:
  feature_dim: 256
  num_queries: 128
  fusion_layers: 8  # 4个Self-Cross pairs
  num_heads: 8
  dropout: 0.1

training:
  batch_size: 24
  num_epochs: 300
  learning_rate: 1.0e-4

loss:
  use_kendall: true
  init_log_var_rotation: 0.0
  init_log_var_translation: 0.0
```

## 代码测试

运行测试脚本验证:
```bash
python test_new_implementation.py
```

**测试结果**: ✓ All tests passed!
- TransformerLayer: ✓
- With position embeds: ✓
- CrossModalFusionModule: ✓ (8层输出)
- ICPoseNet: ✓ (pose_matrix, pose_9d, rotation_6d, translation)

## 预期改进

### 1. 解决过拟合
- **原因**: 缺少Self-Attention导致queries表达能力弱，模型记忆训练数据
- **预期**: Self-Attention让queries聚合多源信息，提升泛化能力
- **目标**: train-val gap从4.7x降低到<2x

### 2. 提升验证性能
- **当前**: Rot=17.43°, Trans=6.48m
- **目标**: Rot<10°, Trans<3m

### 3. 更稳定的训练
- 位置编码不再破坏特征归一化
- Kendall Loss平衡（log_var=0.0）
- 符合ICL-I2PReg的best practice

## 启动训练

```bash
# 方式1: 使用启动脚本
./run_exp010.sh

# 方式2: 直接运行
python train.py --config configs/train_config_exp010.yaml

# 查看日志
tail -f output/exp010/training.log
```

## 对比ICL-I2PReg检查清单

✅ **已对齐**:
- [x] TransformerLayer支持embeds/weights/masks
- [x] MultiHeadAttention完整实现
- [x] AttentionLayer + AttentionOutput
- [x] Self-Cross交替模式
- [x] Query自注意力层
- [x] 位置编码通过embeds传递

❓ **待验证**:
- [ ] 训练稳定性
- [ ] 验证集性能
- [ ] 过拟合问题解决

🔄 **可选改进**:
- [ ] 使用qk_embeds实现RPE (Relative Position Encoding)
- [ ] 添加attention_scores可视化
- [ ] 实现完整的vision3d TransformerLayer所有功能

## 备注

- EXP009已停止训练（epoch 150/300）
- 如需继续EXP009，使用旧配置: `configs/train_config.yaml`
- EXP010为全新实验，从头开始训练
- 保留旧代码在`*_old.py`文件中，方便回滚
