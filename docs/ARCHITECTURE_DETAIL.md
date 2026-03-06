# MSFlowPoseNet 架构详解

> ICLPose 核心位姿估计网络的完整技术文档

---

## 1. 总体架构

MSFlowPoseNet (Multi-Scale Flow Pose Network) 是一个 **Coarse-to-Fine 多尺度光流** 位姿估计网络。
总参数量: **~4.17M** (全部可训练)。

### 设计哲学

参考 RAFT (Recurrent All-Pairs Field Transforms) 的光流估计思想:
1. **构建相关性体** (correlation volume) 而非直接回归
2. **迭代精化** (iterative refinement) 而非一次预测
3. **几何先验** (Image Jacobian) 将光流转为 6-DOF 位姿

### 与 CorrPoseNet 的关键区别

| 特性 | CorrPoseNet | MSFlowPoseNet |
|------|------------|---------------|
| 特征尺度 | 单尺度 (fine) | 多尺度 (coarse/mid/fine) |
| 相关性 | 全局 all-pairs | 粗→全局, 细→局部 |
| 光流精化 | 简单卷积 | RAFT-style ConvGRU |
| 外循环 | ✅ | ✅ |
| 置信度 | 无 | ✅ (网络预测) |

---

## 2. 数据流

```
         查询特征                        参考特征 (可微渲染)
    ┌──────────┐                    ┌──────────┐
    │ coarse   │ 1280d, 7×10       │ coarse   │ ← render(T_curr, 3DGS_coarse)
    │ mid      │ 1280d, 15×20      │ mid      │ ← render(T_curr, 3DGS_mid)
    │ fine_sd  │ 640d,  35×46      │ fine_sd  │ ← render(T_curr, 3DGS_fine_sd)
    │ fine_dino│ 768d,  35×46      │ fine_dino│ ← render(T_curr, 3DGS_fine_dino)
    └──────────┘                    └──────────┘
         │                                │
         ▼                                ▼
    ScaleDecoder (→64-d)          ScaleDecoder (→64-d)
         │                                │
         ▼ ─────────── Stage 1: Coarse ──────────── ▼
    global_correlation(q_coarse, r_coarse)  → corr [B, 70, 7, 10]
    conv layers → flow_coarse [B, 2, 7, 10]
         │
         ▼ ─────────── Stage 2: Mid ──────────────── ▼
    upsample flow_coarse → [B, 2, 15, 20]
    guided_local_correlation(q_mid, r_mid, flow, r=4) → corr [B, 81, 15, 20]
    conv layers → flow_mid [B, 2, 15, 20]
         │
         ▼ ─────────── Stage 3: Fine (RAFT GRU) ──── ▼
    upsample flow_mid → [B, 2, 35, 46]
    FineDualDecoder(fine_sd, fine_dino) → 64-d features
    for i in range(fine_iters):  # 8次
        guided_local_correlation(q_fine, r_fine, flow, r=4) → corr
        corr_feat = corr_encoder(corr)
        hidden = ConvGRU(hidden, corr_feat + context)
        delta_flow, confidence = flow_head(hidden)
        flow = flow + delta_flow
         │
         ▼ ─────────── Stage 4: Geometry Solve ───── ▼
    diff_pose_solve(flow, depth, confidence, intrinsics)
    → Δξ ∈ se(3) → exp(Δξ) → ΔT ∈ SE(3)
    T_curr = ΔT · T_curr
```

---

## 3. 模块详解

### 3.1 ScaleDecoder

将高维 SD 特征降到 64 维用于相关性计算:

```python
class ScaleDecoder(nn.Module):
    # 1×1 Conv (in_dim → 256) → ReLU → 1×1 Conv (256 → 128) → ReLU → 1×1 Conv (128 → 64)
    # 最后 L2 normalize
    # 参数: ~300K per decoder (coarse, mid, ref_coarse, ref_mid 共4个)
```

### 3.2 FineDualDecoder

Fine 尺度同时使用 SD s3 和 DINOv2 两种特征:

```python
class FineDualDecoder(nn.Module):
    # SD 分支:   1×1 Conv (640 → 128) → ReLU
    # DINO 分支: 1×1 Conv (768 → 128) → ReLU
    # 融合:      concat (256) → 1×1 Conv (256 → 64) → L2 normalize
    # 比 ScaleDecoder 稍复杂, 但输出同为 64-d
```

### 3.3 相关性计算

**全局相关性** (coarse 尺度):
```python
def global_correlation(feat_q, feat_r):
    """
    feat_q: [B, 64, H, W] query features
    feat_r: [B, 64, H, W] reference features
    output: [B, H*W, H, W] all-pairs correlation
    
    for pixel (i,j) in query, compute dot product with ALL ref pixels
    """
    return torch.einsum('bcij,bckl->bijkl', feat_q, feat_r)  # 重塑为 [B, H²W², H, W]
```

**引导局部相关性** (mid/fine 尺度):
```python
def guided_local_correlation(feat_q, feat_r, flow, r=4):
    """
    使用 flow 将 ref 特征 warp 到 query 空间,
    然后在 (2r+1)² 邻域内计算局部相关性
    
    output: [B, (2r+1)², H, W] = [B, 81, H, W]
    比全局相关性更高效 O(H·W·r²) vs O(H²·W²)
    """
```

### 3.4 ConvGRU

```python
class ConvGRU(nn.Module):
    """3×3 Convolutional Gated Recurrent Unit
    
    hidden_dim = 128
    input: x (corr features + context), h (previous hidden state)
    output: h_new (updated hidden state)
    
    z = sigmoid(Conv([h, x]))  # update gate
    r = sigmoid(Conv([h, x]))  # reset gate
    h_tilde = tanh(Conv([r*h, x]))  # candidate
    h_new = (1-z)*h + z*h_tilde
    """
```

### 3.5 FlowRefinementHead

```python
class FlowRefinementHead(nn.Module):
    """RAFT-style iterative flow refinement
    
    corr_encoder: 相关性 → 128-d 特征
    gru: ConvGRU (hidden_dim=128)
    flow_head: 2×Conv → (du, dv, confidence)
    
    每次迭代:
    1. 对当前 flow 计算 guided_local_correlation
    2. corr_encoder 编码相关性
    3. GRU 更新 hidden state
    4. flow_head 预测 (delta_flow, confidence)
    5. flow += delta_flow
    """
```

### 3.6 Geometry Solver

数学原理:

对于像素 $(u, v)$, 深度 $z$, 相机内参 $(f_x, f_y, c_x, c_y)$:

**Image Jacobian**:
$$J = \begin{bmatrix}
-\frac{f_x}{z} & 0 & \frac{f_x \cdot X}{z^2} & \frac{f_x \cdot X \cdot Y}{z^2} & -f_x(1+\frac{X^2}{z^2}) & \frac{f_x \cdot Y}{z} \\
0 & -\frac{f_y}{z} & \frac{f_y \cdot Y}{z^2} & f_y(1+\frac{Y^2}{z^2}) & -\frac{f_y \cdot X \cdot Y}{z^2} & -\frac{f_y \cdot X}{z}
\end{bmatrix}$$

其中 $X = (u - c_x) \cdot z / f_x$, $Y = (v - c_y) \cdot z / f_y$

**加权最小二乘**:
$$\Delta\xi^* = \arg\min_{\Delta\xi} \sum_i w_i \| J_i \Delta\xi - f_i \|^2$$

闭式解 (带 LM 阻尼):
$$\Delta\xi = (J^T W J + \lambda I)^{-1} J^T W f$$

- $f$: 预测的光流 (flow)
- $w$: 网络预测的置信度 (经过 sigmoid)
- $\lambda = 10^{-4}$: Levenberg-Marquardt 阻尼
- 通过 `torch.linalg.solve` 求解 6×6 线性系统 → **完全可微**

---

## 4. 多尺度可微渲染

### MultiScaleRenderer

管理 4 个 `GaussianFeatureModel` 实例, 共享 Gaussian 几何:

```python
class MultiScaleRenderer:
    models:
        coarse:    GaussianFeatureModel (1280-d features, 7×10 渲染)
        mid:       GaussianFeatureModel (1280-d features, 15×20 渲染)
        fine_sd:   GaussianFeatureModel (640-d features, 35×46 渲染)
        fine_dino: GaussianFeatureModel (768-d features, 35×46 渲染)
    
    render_batch(viewmats):
        # 对每个尺度调用 gsplat.rasterization()
        # 额外渲染深度图 (render_mode='D', fine resolution)
        # 返回: {scale: [B, C, H, W]}, depth [B, 1, 35, 46]
```

渲染使用 [gsplat v1.4.0](https://github.com/nerfstudio-project/gsplat):
- 3DGS: `gsplat.rasterization()` — 椭球光栅化, 内置 channel_chunk
- 2DGS: `gsplat.rasterization_2dgs()` — surfel 光栅化 (手动分块)

### GaussianFeatureModel

```python
class GaussianFeatureModel(nn.Module):
    # 从 PLY 加载预训练 3DGS
    # 冻结: xyz, rotation, scaling, opacity, SH (register_buffer)
    # 可训练: _loc_feature [N, D] — 每个 Gaussian 的特征嵌入
    
    # 渲染前 L2 归一化特征
    # 支持 3DGS (3 scales) 和 2DGS (2 scales, 自动检测)
```

---

## 5. SE(3) Lie 代数

`modules/lie_algebra.py` 实现了完整的 SE(3) 操作:

| 函数 | 功能 | 输入→输出 |
|------|------|----------|
| `se3_exp(xi)` | 指数映射 | ℝ⁶ → SE(3) 4×4 |
| `se3_log(T)` | 对数映射 | SE(3) 4×4 → ℝ⁶ |
| `so3_exp(omega)` | 旋转指数映射 | ℝ³ → SO(3) 3×3 |
| `hat(omega)` | 反对称矩阵 | ℝ³ → so(3) 3×3 |
| `pose_compose(T1, T2)` | 位姿复合 | T1·T2 |
| `pose_inverse(T)` | 位姿求逆 | T → T⁻¹ |

约定: $\xi = [v_x, v_y, v_z, \omega_x, \omega_y, \omega_z]$ (平移在前, 旋转在后)

所有实现使用 `gradient-safe` 技巧: clamp+sqrt 避免 $\sqrt{0}$ 梯度为 inf。

---

## 6. 外循环迭代精化

这是 exp031 引入的**关键改进**，使 MSFlowPoseNet 从"不work"变为"超越基线":

```python
def _train_step(self, batch):
    T_curr = batch['initial_pose']  # 带噪声的初始位姿
    
    for outer_iter in range(self.outer_iters):  # 3次
        # 1. 用当前位姿渲染参考特征 (可微)
        ref_feats, depth = self.renderer.render_batch(T_curr)
        
        # 2. 网络前向: 预测光流 + 置信度
        flows, confidences = self.model(query_feats, ref_feats)
        
        # 3. 计算GT光流 (用T_curr和T_gt)
        gt_flow = compute_gt_flow(depth, T_gt, T_curr)
        
        # 4. 几何求解: 光流 → 位姿增量
        delta_xi = diff_pose_solve(flows[-1], depth, confidences[-1])
        delta_T = se3_exp(delta_xi)
        
        # 5. 更新位姿 (detach! 不将渲染梯度回传到上一步)
        T_curr = (delta_T @ T_curr).detach()
        
        # 6. 每步都反向传播光流损失
        loss += flow_loss(flows, gt_flow)
```

关键点:
- 每次外循环用更新后的位姿**重新渲染**参考特征
- `T_curr.detach()` 截断梯度, 避免爆炸
- 由于渲染对位姿可微, 光流监督信号随迭代越来越准确

---

## 7. 训练损失设计

### 光流损失 (主损失)

```python
def multiscale_flow_loss(pred_flows, gt_flows, valid_masks):
    """
    Coarse/Mid: 简单 L1 loss (有效像素)
    Fine (RAFT iterations): 
        L = Σ_i γ^(N-1-i) · L1(flow_i, gt_flow)
        γ=0.85, 最后几次迭代权重最大
    """
```

### 位姿损失 (Phase2)

```python
def pose_loss(pred_poses, gt_poses, rot_type='cosine'):
    """
    Rotation: 1 - cos(θ_error)  ← cosine, 梯度安全
    或:       acos(cos_theta)    ← acos, 有梯度爆炸风险
    
    Translation: ||t_pred - t_gt||₂
    
    Total: loss_rot + trans_weight * loss_trans
    """
```

---

## 8. 数据集 (PoseDatasetV4)

```python
class PoseDatasetV4(Dataset):
    """
    每帧返回:
    - query_feats: {coarse: [1280, 7, 10], mid: [1280, 15, 20],
                    fine_sd: [640, 35, 46], fine_dino: [768, 35, 46]}
    - pose_gt: [4, 4] w2c 位姿
    - initial_pose: [4, 4] 带噪声的初始位姿 (训练) 或 NetVLAD 位姿 (验证)
    - depth: [35, 46] GT 深度图 (用于 Image Jacobian)
    
    支持 v1/v2 特征格式自动检测
    noise_rot_deg 和 noise_trans_m 由训练器动态设置 (噪声课程)
    """
```
