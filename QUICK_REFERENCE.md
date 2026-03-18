# ICLPose: Quick Reference & Key Insights

## Critical Components Matrix

| Component | File | Lines | Purpose | Key Param |
|-----------|------|-------|---------|-----------|
| **ScaleDecoder** | ms_flow_pose_net.py | 115-135 | Per-scale feat reduction | in_dim→64d |
| **FineDualDecoder** | ms_flow_pose_net.py | 137-211 | Fuse SD+DINO | (640,768)→64d |
| **global_correlation** | ms_flow_pose_net.py | 213-240 | All-pairs sim (coarse) | (B,80,8,10) |
| **guided_local_corr** | ms_flow_pose_net.py | 315-357 | Warp+local (mid/fine) | (B,81,H,W) r=4 |
| **ConvGRU** | ms_flow_pose_net.py | 476-506 | Iterative gating | 3×3 conv |
| **FlowRefinementHead** | ms_flow_pose_net.py | 508-635 | Scale-specific refine | (corr,h)→(u,v,conf) |
| **Image Jacobian** | geometry_solver.py | 14-79 | Pixel→motion jacob | (B,N,6) |
| **WLS Solver** | geometry_solver.py | 82-202 | Pose from flow | J^TwJ ξ=J^Twr |
| **IRLS Reweighting** | geometry_solver.py | 172-193 | Robust outlier reject | Huber k=1.345 |
| **SE(3) Ops** | lie_algebra.py | 22-199 | Lie group conversions | exp/log/hat |

---

## Iterative Refinement Strategy

### Pipeline Structure
```
Coarse (global)  → Mid (coarse+local)  → Fine (mid+RAFT)   → Geometry Solver
 8×10             1 iter @16×20         4 iters @35×46      → 6-DOF δξ
 All-pairs        Warp+local r=4        Warp+local r=4      → (B,6)
 80 channels      81 channels           81+ channels        → Pose increment
```

### Iteration Counts
- **Coarse:** 1 GRU step (global correlation doesn't benefit from iteration)
- **Mid:** 1 GRU step (typically sufficient for refinement from coarse)
- **Fine:** 4 GRU steps (RAFT-style, recomputes warp-guided correlation each step)
- **Outer (inference):** N iterations (typically 3-5) with re-rendering between iterations

### Between-Iteration Changes
```
Inner loop (fine GRU iterations):
  for i in [1,2,3,4]:
    - Flow update only (network predicts Δflow)
    - Same pose (frozen)
    
Outer loop (iterative refinement):
  for i in [1,2,3,4,5]:
    - Render new features at updated pose
    - Forward pass (coarse→mid→fine→geometry)
    - Apply Δξ to pose
    - Repeat with new render
```

---

## Confidence Map Evolution

```
INIT:        conf ~ 0.5 (uniform)
COARSE:      conf_c = sigmoid(head(GRU(...)))
             Represents: spatial reliability of global correlation
             
MID:         conf_m_init = upsample(conf_c)
             conf_m_final = sigmoid(head(GRU(...))) after 1 iter
             Represents: confidence after coarse-guided local refinement
             
FINE:        conf_f_0 = upsample(conf_m)
             for i in [1,2,3,4]:
               inp = [corr + flow + conf_f_i-1]
               conf_f_i = sigmoid(head(GRU(h, enc(inp))))
             Represents: per-pixel flow prediction reliability
             
MULTISCALE:  if enabled:
               consistency = exp(-(dist(flow_c,flow_f) + dist(flow_m,flow_f)) / 2σ²)
               conf_f *= consistency
             Represents: agreement between scales
             
GEOMETRY:    w = conf_f * valid (mask invalid pixels)
             Used in: J^TwJ ξ = J^Tw(flow)
             Later: w *= Huber(residuals) for robustness
```

---

## Numerical Stability Measures

| Issue | Location | Mitigation | Severity |
|-------|----------|-----------|----------|
| sqrt(0) gradient | lie_algebra.py:58 | `clamp(θ²,1e-10) + torch.where` | Medium |
| Z=0 division | geometry_solver.py:54 | `Z.clamp(min=0.05)` | High |
| Singular J^TJ | geometry_solver.py:100 | LM damping λ*diag(J^TJ) | Medium |
| NaN in Huber median | geometry_solver.py:182 | Push invalid to 1e6 before median | Low |
| Grid extrapolation | ms_flow_pose_net.py:350 | Pad with 0, mark invalid | Medium |
| Depth matching | ms_flow_pose_net.py:1200 | F.interpolate with resize check | Low |

---

## Error Propagation Paths

### Path 1: Feature Encoding
```
Raw features (ODISE 1280d/DINO 768d)
  ↓ ScaleDecoder(1×1 conv 3 layers)
  → L2 normalized 64d
  → Correlation computation
  
Error source: Dimensionality reduction loses information
Mitigation: Deep decoder (256d bottleneck), L2 norm for scale invariance
```

### Path 2: Correlation Mismatch
```
Query features (ground-truth camera)
  ↓ correlation
Render features (hypothesis camera pose)

Error: If pose hypothesis is far from GT → poor correlation
Propagates via: confidence downweighting (soft), then geometry solver
```

### Path 3: Flow → Pose
```
Dense flow field (1610 pixels)
  ↓ Image Jacobian (1610×6 matrix)
  ↓ Weighted LS solver (weights from confidence)
  → δξ (6 unknowns)

Error sources:
  - Flow prediction error (primary)
  - Jacobian inaccuracy (depth dependent)
  - Outlier flows (IRLS mitigates)
  - Rank deficiency (LM damping mitigates)
```

---

## Confidence Weaknesses & Fixes

### Current Model
```python
conf: (B, 1, H, W) scalar per pixel
w = conf * valid  # scalar weight
J^T w J ξ = J^T w f  # diag(w) scaling
```

**Limitations:**
1. No directional uncertainty (u-ambiguous vs v-ambiguous)
2. No correlation between neighbors
3. Ignores Jacobian condition number
4. No depth uncertainty propagation

### Recommended Fix #1: Directional Confidence (Low Risk)
```python
# Current: 1 channel
# Proposed: 2 channels
conf_uv = head(hidden)  # (B, 2, H, W)
conf_u, conf_v = conf_uv.chunk(2)

# In solver:
w_u = conf_u * valid
w_v = conf_v * valid
J^T(w_u⊙Ju + w_v⊙Jv) ξ = J^T(w_u⊙flow_u + w_v⊙flow_v)
```

### Recommended Fix #2: Covariance Output (Medium Risk)
```python
# Current: (B, 1, H, W) conf
# Proposed: (B, 6, H, W) → [u, v, σ_u, σ_v, ρ_uv]
# or more simply: (B, 3, H, W) → [u, v, σ]

# Mahalanobis weighting:
sigma_safe = sigmoid(σ_raw) * 0.5 + 0.01  # (0.01, 0.51)
w = 1 / σ²

# In solver:
J^T(1/σ²⊙J) ξ = J^T(1/σ²⊙flow)
```

---

## Multi-Scale Consistency Reweighting: Math & Concerns

### Implementation
```python
# Upsample coarse/mid flows to fine resolution
flow_c_up = upsample(flow_c, fine_hw)
flow_c_up[:, 0] *= W_fine / W_coarse  # rescale pixel values
flow_m_up = upsample(flow_m, fine_hw)
flow_m_up[:, 1] *= H_fine / H_mid

# Disagreement (L2 distance)
diff_cf = (flow_f - flow_c_up)^2
diff_mf = (flow_f - flow_m_up)^2

# Consistency Gaussian
sigma = 1.0  # default
consistency = exp(-(diff_cf + diff_mf) / (2 * sigma^2))

# Apply
conf_f *= consistency
```

### Concerns
1. **Assumption:** Assumes flow error ~ Gaussian; not empirically validated
2. **Hyperparameter:** σ=1.0 is ad-hoc; no principled tuning
3. **Interpretation:** "Disagreement" confuses different types:
   - Coarse ambiguity ≠ fine ambiguity (different resolutions)
   - May spuriously downweight correct fine predictions

### Better Approach
```python
# Instead of assuming Gaussian:
# Compute per-scale error statistics from training data
# Use learned uncertainty model:

class ConsistencyReweighter(nn.Module):
    def forward(self, diff_cf, diff_mf):
        feat = [diff_cf, diff_mf, flow_c, flow_m, flow_f]
        weight = MLP(feat)  # [0, 1] output
        return weight
```

---

## IRLS Robust Estimation Analysis

### Current Implementation
```python
for irls_iter in range(3):  # default
    # Compute residuals
    pred = J @ ξ
    residual = |pred - flow|
    
    # Huber threshold (adaptive)
    threshold = median(residual) * 1.345
    
    # Huber weight: smooth transition at threshold
    w_huber = where(residual < threshold, 1.0, threshold / residual)
    
    # Combine with confidence
    w_irls = conf * w_huber
    
    # Resolve
    ξ = solve(J^T(w_irls⊙J), J^T(w_irls⊙flow))
```

### Strengths
- ✅ Soft weighting prevents discontinuities
- ✅ Adaptive threshold via weighted median
- ✅ Iterations converge for most scenes

### Weaknesses
- ❌ Fixed 3 iterations (may be premature/excessive)
- ❌ Median-based threshold sensitive to outlier ratio
- ❌ Assumes outliers are "random" not spatially clustered

### Improvement: Variance-Regularized IRLS
```python
# Current: w ∝ 1 / residual (Huber)
# Better: w ∝ 1 / (residual² + α²)  (softer tail)

# Reason: Avoids 1/ε singularity, more stable numerically
```

---

## Geometry Upsampling (Advanced Feature)

### Purpose
When `geometry_upsample > 1`, solve geometry at higher resolution than flow prediction.

### Implementation
```python
s = geometry_upsample  # e.g., 2
if s > 1:
    # Upsample flow
    flow = bilinear(flow, size * s) * s
    # Upsample confidence
    conf = bilinear(conf, size * s)
    # Upsample depth
    depth = bilinear(depth, size * s)
    
    # Compute Jacobian at finer grid
    J = jacobian(depth)  # now (B, N*s², 6)
    
    # Solve: more constraints, but same unknowns
    ξ = solve(J^T w J, J^T w f)  # (B, 6)
```

### Rationale
- Flow network outputs at 35×46
- But flow prediction error is correlated
- Upsampling → decorrelates, provides finer geometric constraints
- Cost: O(s²) more Jacobian computation

### Trade-off
- ✅ More constraints → better condition number
- ❌ Finer mesh may reveal solver limitations
- ❌ Interpolated flow/depth less accurate

---

## Loss Functions (From losses/ directory)

### PoseLoss (pose_loss.py)
- **Rotation losses:** geodesic, L2, cosine, quaternion
- **Translation losses:** L1, L2, smooth_L1
- **Weight combination:** rotation_weight * L_rot + trans_weight * L_trans
- **Typical:** geodesic rotation + L2 translation (differentiable, interpretable)

### PoseLossC2F (pose_loss_c2f.py)
- **Coarse-to-fine annealing:** progressively switch from coarse-level to fine-level loss
- **Warmup phase:** early epochs prioritize coarse levels
- **Dynamic clamping:** tanh-based soft clamping on errors
- **Purpose:** Prevent early-stage coarse flow prediction from dominating loss

### SequenceLoss (sequence_loss.py)
- **Multi-iteration supervision:** apply loss at each coarse/mid/fine iteration
- **Weighting:** typically decrease weight for earlier iterations
- **Purpose:** Enforce flow refinement at each scale, encourage consistent predictions

---

## Quick Debug Checklist

### If pose error is high:
1. **Check flow quality:**
   - Visualize flow_coarse, flow_mid, flow_fine
   - Do they form a coherent progression?
   - Are magnitudes reasonable (< image diagonal)?

2. **Check confidence:**
   - Is conf_fine spatially reasonable (high in textured regions)?
   - Does multiscale_consistency reweighting make sense?
   - Are invalid pixels (depth ≤ 0.05) properly masked?

3. **Check Jacobian:**
   - Depth clipping: are most pixels > 0.05?
   - Is Jacobian rank-deficient? (check condition number)
   - Are small rotations causing numerical issues?

4. **Check solver:**
   - Does IRLS converge? (check residual norms across iterations)
   - Is damping too aggressive? (try smaller values)
   - Do soft clamps trigger? (tanh saturation suggests large Δξ)

### If training loss oscillates:
1. **Correlation temperature:** If corr_temperature ≠ 1.0, check scaling
2. **Learning rate:** May be too high for pose_loss_c2f warmup phase
3. **Batch size:** Small batches → noisy IRLS median computation

### If inference is slow:
1. **Outer iterations:** Reduce num_iters in eval_iterative.py
2. **Fine iterations:** Reduce fine_iters in config (default 4)
3. **Correlation type:** If using multi-scale (dilations), simplify to single-scale

---

## File Organization Summary

```
ic_models/
  └─ ms_flow_pose_net.py (1316 lines)
     ├─ ScaleDecoder, FineDualDecoder (feature encoding)
     ├─ global_correlation, guided_local_correlation, dilated/multiscale variants
     ├─ ConvGRU, FlowRefinementHead, PoseRefinementHead
     ├─ PositionalEncoding2D, CrossScaleContext, ContextAdapter
     └─ MSFlowPoseNet (main network, forward pass)

modules/
  ├─ geometry_solver.py (203 lines)
  │  ├─ compute_image_jacobian
  │  ├─ _solve_weighted_normal_eq (WLS solver core)
  │  └─ diff_pose_solve (IRLS wrapper, pixel_stride subsampling)
 lie_algebra.py (200 lines)  
  │  ├─ hat, so3_exp, se3_exp
  │  ├─ se3_log, pose_inverse, pose_compose
  │  └─ compute_gt_flow (for ground truth supervision)

data/
  └─ dataset_v4.py (301 lines)
     ├─ PoseDatasetV4 (v1/v2 format auto-detection)
     ├─ perturb_pose, _orthogonalize_rotations
     └─ collate_v4 (custom batching)

losses/
  ├─ pose_loss.py (geodesic/L2 rotation, L1/L2/smooth-L1 translation)
  ├─ pose_loss_c2f.py (coarse-to-fine annealing, dynamic clamping)
  ├─ reprojection_loss.py (3D reprojection for auxiliary supervision)
  └─ sequence_loss.py (multi-iteration supervision)

scripts/
  └─ eval_iterative.py (evaluation with num_iters ∈ [1,3,5])
```

---

## Recommended Reading Order

1. **High-level overview:** MSFLOW_ARCHITECTURE.md (if available)
2. **Main network:** ms_flow_pose_net.py top-to-bottom (focus on forward method)
3. **Correlation functions:** understand global_correlation, guided_local_correlation
4. **Geometry solver:** geometry_solver.py (Image Jacobian + WLS + IRLS)
5. **SE(3) operations:** lie_algebra.py (for understanding pose updates)
6. **Data loading:** dataset_v4.py (how features/poses are organized)
7. **Evaluation:** eval_iterative.py (inference loop with iterations)

---

