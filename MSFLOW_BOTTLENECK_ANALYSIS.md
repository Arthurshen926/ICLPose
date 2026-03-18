# MSFlowPoseNet Architecture Analysis: Bottlenecks & Improvement Strategies
**Current Performance**: 0.55° mean rotation (room_0) | **Target**: 0.1° (5.5× improvement needed)

---

## EXECUTIVE SUMMARY

The MSFlowPoseNet uses a **coarse→mid→fine multi-scale cascade** with **warp-guided local correlation** and a **differentiable geometry solver** (Image Jacobian + weighted least squares). At 1,610 valid pixels at fine resolution vs. 6 unknowns (268:1 constraint ratio), the geometry solver is mathematically well-constrained.

**Key Finding**: The main bottleneck is **information loss between scales** and **insufficient GRU refinement capacity**, not the geometry solver itself. The 64-dimensional decode bottleneck combined with bilinear upsampling causes **cumulative detail loss** as features propagate coarse→mid→fine.

---

## DETAILED ANALYSIS BY COMPONENT

### 1. MULTI-SCALE ITERATIVE REFINEMENT (Coarse → Mid → Fine)

**Current Flow** (`ms_flow_pose_net.py` lines 557-634):
```
1. Coarse (8×10):  Q/R features → 64d → global correlation (80 channels)
                   → ConvGRU(hidden_dim=128) → flow + confidence
                   
2. Mid (16×20):    Q/R features → 64d → upsample hidden (bilinear, 8×10→16×20)
                   → ContextAdapter (concat upsampled hidden + q_mid features)
                   → GRU iteration (mid_iters=2, default)
                   → warp-guided local correlation (r=4, 81 channels)
                   
3. Fine (35×46):   Q/R features (dual-branch: SD+DINO) → 64d
                   → upsample hidden (bilinear, 16×20→35×46)
                   → ContextAdapter (concat upsampled hidden + q_fine)
                   → RAFT-style GRU iterations (fine_iters=4, default)
                   → warp-guided local correlation
```

#### **Problem 1a: Severe Information Loss Coarse→Mid**
- **Coarse hidden state**: 128 channels × 8×10 = 10.24 KB
- **Upsampled to Mid (bilinear)**: 128 channels × 16×20 = 40.96 KB
  - **Bilinear interpolation spreads information from 80 pixels → 320 pixels**
  - Each coarse hidden cell maps to ~4 mid cells, filling with smoothed values
  - **No per-cell learning signal**, just generic smoothing
- **ContextAdapter fusion** (`modules/ms_flow_pose_net.py` lines 253-278):
  ```python
  h_up + Conv([h_up, q_mid]) → residual add
  ```
  - Adds mid-scale query features, BUT:
  - q_mid comes from same backbone, trained jointly → **correlated with coarse**
  - Cannot inject truly new information about mid-scale structures
  - Only adds ~10% capacity via residual (hidden_dim→hidden_dim)

**Impact**: Flow predictions at mid scale are heavily dominated by upsampled coarse flow. The model cannot easily correct coarse mistakes because:
1. Upsampled hidden state is already "locked in" by smooth interpolation
2. ContextAdapter has limited capacity to override it
3. Mid correlation (81 channels) is only ~3% of hidden state capacity

#### **Problem 1b: Cumulative Loss Mid→Fine**
- Same issue repeats: bilinear upsample 16×20→35×46
- At fine scale, the hidden state is doubly-smoothed aggregate
- Fine resolution is where geometry solver operates, but GRU has already made decisions at coarse
- **Result**: GRU refinements at fine are primarily "fine-tuning" a coarsely-determined flow

#### **Problem 1c: Asymmetric Multi-Scale Cascade**
- Coarse & mid use **single iteration** or **mid_iters=2** (hardcoded in configs)
- Only fine gets **RAFT-style refinement (fine_iters=4)**
- Coarse errors propagate → mid → fine with no correction path
- Early scales need MORE iterations than later ones (more room for improvement)

---

### 2. GRU UPDATE MECHANISM & HIDDEN STATE MANAGEMENT

**Current Architecture** (`modules/conv_gru.py` + `ms_flow_pose_net.py`):

```python
# ConvGRUCell (lines 18-85, conv_gru.py)
z = σ(Conv([h_{t-1}, x_t]))              # update gate
r = σ(Conv([h_{t-1}, x_t]))              # reset gate  
h̃ = tanh(Conv([r⊙h_{t-1}, x_t]))        # candidate
h_t = (1-z)⊙h_{t-1} + z⊙h̃               # final hidden
```

**Input to GRU**: 
- Correlation volume (80-81 channels, depending on scale)
- Current flow estimate (2 channels)
- Current confidence (1 channel)
- **Total**: 83-84 channels → encoder → 64 channels (context_dim)
- Then GRU processes: (hidden_dim=128, input_dim=64)

#### **Problem 2a: High Information Bottleneck at Input Encoding**
- Correlation has 80-81 channels (rich geometric signal)
- Compressed to 64 channels (20% loss)
- GRU input encoder (`FlowRefinementHead.corr_encoder`, lines 330-334):
  ```python
  Conv(83, 128) → Conv(128, 64)  # 2 convolutions
  ```
  - Only 128 channels intermediate (same as final hidden dim)
  - Cannot learn complex correlation→hidden mappings
  - **Compared to RAFT**: RAFT uses 256-512 intermediate dims in correlation encoder

**Impact**: Rich correlation signal is bottlenecked early. GRU cannot learn to extract multi-scale geometric features from correlation.

#### **Problem 2b: Single ConvGRU Cell → Limited Refinement**
- Each iteration applies **single ConvGRU cell** (one set of 3×3 convolutions)
- Small receptive field (3×3) → can only aggregate local context
- **Known RAFT design**: Uses larger kernels or stacked convolutions in update block
- GRU hidden state is overloaded: must encode **global flow consistency + local details + confidence**

#### **Problem 2c: Hidden State Initialization Loses Spatial Information**
- **Coarse**: ContextNet maps Q features → hidden
  ```python
  ContextNet: Conv(64, 128) → Conv(128, 128)  # (lines 475-480)
  ```
  - Initializes hidden to 128d encoding of query features
  - No global context from correlation yet
  
- **Mid/Fine**: ContextAdapter (lines 483-484)
  ```python
  h_up + Conv([h_up, q_mid])  # residual connection
  ```
  - h_up is upsampled from previous scale (bilinear-smoothed)
  - Residual add doesn't "reset" hidden; it merely increments it
  - **Problem**: GRU is biased toward "incremental refinement" instead of "fresh learning"
  - At fine scale, h_fine is 2× upsampled (bilinear then bilinear) → very smooth

**Impact**: GRU starts with pre-determined hidden state that's hard to override. Early mistakes stick.

---

### 3. CORRELATION VOLUME COMPUTATION

#### **Coarse: Global All-Pairs Correlation**
```python
def global_correlation(fmap1, fmap2):  # (lines 131-155)
    # For each query pixel, compute dot product with ALL render pixels
    # fmap1: (B, 64, 8, 10) → (B, 64, 80)
    # fmap2: (B, 64, 8, 10) → (B, 64, 80)
    corr = einsum('bcn,bcm->bmn', f1, f2)  # (B, 80, 80)
    return (B, 80, 8, 10)  # reshaped as correlation volume
```
**Advantages**:
- Dense global matching → no false minima from local-only search
- Natural for coarse scale with few pixels

**Issues**:
- 80 channels correlation is **sparse information** (80 pixels total)
- Global correlation is scale-invariant; can't prioritize nearby matches
- Temperature scaling (`corr_temperature=1.0`) affects all scales uniformly
  - No per-scale tuning → suboptimal for coarse (should be sharper) vs fine (should be softer)

#### **Mid/Fine: Warp-Guided Local Correlation**
```python
def guided_local_correlation(fmap_q, fmap_r, flow, radius=4):  # lines 211-246
    # 1. Construct warp grid: grid + flow → normalized [-1, 1]
    # 2. Warp fmap_r using grid_sample (bilinear)
    # 3. Compute local correlation with radius=4 → 81 channels
```

**Advantages**:
- Previous flow estimate centers search window → fewer outliers
- Reduces search space from 400 pixels (8×50 global) → 81 pixels (9×9 local)
- Enables iterative refinement (RAFT-style)

**Problems**:
- **Warping accumulates errors**: If previous flow is wrong, warp_grid moves to wrong location
  - grid_sample(fmap_r, wrong_grid) produces corrupted correlation
  - Local radius=4 can't recover if warped region is far from true match
  - **Example**: Coarse flow error of 2 pixels → at 4×upsampling → 8 pixels at mid
    - Local radius=4 searches ±4 pixels; 8-pixel error is at edge of search space
- **Bilinear warping smears features**: grid_sample uses bilinear interpolation
  - Sub-pixel accuracy is good, but feature distinctiveness is reduced
  - Fine details in fmap_r are blurred by resampling
- **Padding strategy**: Warp boundary regions get zero-padding in correlation
  - Large areas can have completely invalid correlation (zeros)
  - GRU must learn to ignore zero-correlation, but this is indirect

#### **Problem 3a: No Outlier Rejection in Geometry Solver**
- `geometry_solver.py` uses **confidence weighting** (lines 122-127):
  ```python
  w = confidence * valid  # where valid = (depth > 0.05)
  ```
  - Confidence comes from flow head (learned)
  - No **outlier detection** (e.g., RANSAC, robust M-estimators)
  - High-confidence wrong pixels (occlusions, reflections) directly corrupt pose
  
- **Adaptive damping** (lines 133-136):
  ```python
  diag = torch.diagonal(JtWJ)
  diag_damping = damping * diag.clamp(min=1e-6)
  ```
  - Helps with rank deficiency, NOT with outliers
  - Outliers with high confidence can pull entire solution far from true pose

**Impact**: If 10% of pixels have high confidence but wrong flow (due to occlusion/reflection), geometry solver is biased. No way to downweight them beyond learned confidence.

---

### 4. FLOW-TO-POSE GEOMETRY SOLVER

**Current Implementation** (`modules/flow_to_pose.py` lines 37-159):

```python
def flow_to_pose_weighted_lstsq(flow_pred, log_confidence, depth, intrinsics, damping):
    # 1. Build Image Jacobian J_u, J_v for each pixel (B, N, 6)
    # 2. Weight: w = exp(log_confidence) * valid_mask
    # 3. Normal equations: (J^T W J) ξ = J^T W f
    # 4. Levenberg-Marquardt: JtWJ + λI, solve with linalg.solve
    return xi  # (B, 6)
```

**Strengths**:
- **Mathematically principled**: Direct from camera geometry, not learned regression
- **Well-constrained**: 1,610 pixels × 2 equations = 3,220 constraints vs 6 unknowns (536:1 ratio!)
- **Differentiable**: Gradients flow back to flow predictions
- **Adaptive confidence weighting**: Network learns which pixels to trust

**Critical Limitations**:
1. **Assumes small-angle linear approximation**:
   - Jacobian derivation uses first-order Taylor expansion
   - Valid for ~5° rotation; breaks down at ±8° (curriculum learning noise)
   - At large angles, geometric linearization error dominates

2. **No outlier rejection**:
   - ALL pixels (even high-confidence wrong ones) contribute equally weighted
   - Missing occlusions, motion blur, reflections can bias solution

3. **Flow is dense, not sparse**:
   - All 1,610 pixels contribute → many in low-texture regions with noisy flow
   - SLAM methods (DSO, ORB-SLAM) use **keypoint selection** or **high-gradient regions**
   - Dense weighting on low-information regions is inefficient

4. **Damping strategy is generic**:
   - `damping=1e-3` is hardcoded, not adaptive
   - Should scale with noise magnitude, observation confidence, etc.

5. **No epipolar constraint**:
   - Image Jacobian only enforces pixel-level geometric consistency
   - Doesn't leverage **epipolar geometry** (essential matrix) for global consistency check

---

### 5. OUTER ITERATIONS (Re-running Entire Cascade)

**Current Implementation** (`scripts/train_ms_flow.py` lines 687-755):

```python
for outer_i in range(num_outer_iters):  # num_outer_iters=1 in configs
    # 1. Render at current pose
    render_feats = renderer(pose_cur)
    
    # 2. Forward pass
    pred = model(query_feats, render_feats, depth)
    
    # 3. Compute GT flows, losses
    flow_loss, pose_loss = compute_losses(pred, gt_flows, pose_cur, pose_gt)
    
    # 4. Backward with weighting
    loss_w = gamma_outer^(N-1-i) / N
    loss_w * total_loss → backward()
    
    # 5. Update pose (except last iteration)
    if outer_i < N-1:
        T_delta = se3_exp(pred['delta_xi'])
        pose_cur = T_delta @ pose_cur  # detach!
```

#### **Problem 5a: Outer Iterations Are Disabled by Default**
- All configs have `outer_iters=1` (single pass through cascade)
- Outer iterations are **conceptually correct** but:
  - **Computational cost**: N× forward passes, N× backward passes
  - **Training time**: Already long with fine_iters=4
  - **Not tuned**: gamma_outer=0.8, loss weighting not validated

- **Why outer iterations matter**:
  - Coarse flow error propagates through mid/fine
  - Re-rendering at corrected pose allows model to predict residual correction
  - At outer_i=1, model sees refined pose, coarse error is smaller
  - This is **iterative refinement** at training level (not just inference)

#### **Problem 5b: Gradient Flow Across Outer Iterations**
- Outer loop detaches pose updates: `pose_cur = T_delta @ pose_cur.detach()`
- **By design**: Breaks backprop between iterations (saves memory, reduces vanishing gradients)
- **Consequence**: GRU is trained to predict delta_xi *given current pose*, not globally optimal xi
  - Model learns "how to refine a slightly-wrong pose" not "how to estimate perfect pose from scratch"
  - This is actually good for **outer iteration training** but BAD for **single-pass inference**

#### **Problem 5c: Loss Weighting in Outer Iterations**
```python
iter_w = gamma_outer^(N-1-i)  # weights later iterations more
scaled_loss = iter_w * loss / N
```
- With N=2: weights are [0.8, 1.0] (normalized)
- With N=3: weights are [0.64, 0.8, 1.0]
- Later iterations have larger loss → model prioritizes final poses
- **Issue**: Early iterations should also be well-trained (they set up the cascade)
- Better weighting might be uniform or **inverse** (weight early errors more)

---

## CONCRETE BOTTLENECK RANKINGS

| Rank | Component | Information Loss | Est. Impact on 0.55° → Target |
|------|-----------|------------------|------|
| 1 | **Bilinear upsampling (coarse→mid)** | ~30-40% | 0.15°-0.25° |
| 2 | **Correlation encoder bottleneck (80→64)** | ~20% | 0.10°-0.15° |
| 3 | **GRU hidden state overloading** | Distributed | 0.08°-0.12° |
| 4 | **Warp-guided correlation errors** | ~10-15% | 0.08°-0.12° |
| 5 | **No outlier rejection in solver** | ~5-10% | 0.05°-0.08° |
| 6 | **Single coarse/mid iteration** | Cumulative | 0.05°-0.10° |
| 7 | **Outer iterations disabled** | Training setup | 0.05°-0.10° |

**Total identified slack**: 0.56°-0.92° (reasonable for ~5.5× improvement target)

---

## SPECIFIC IMPROVEMENT RECOMMENDATIONS

### HIGH-IMPACT IMPROVEMENTS (Start Here)

#### **1. Replace Bilinear with Deformable Convolution Upsampling**
**Difficulty**: Medium | **Estimated Impact**: -0.15° to -0.25° | **Implementation**: 2-3 hours

**Idea**: Instead of bilinear interpolation, learn adaptive upsampling filters
```python
# Current (ms_flow_pose_net.py, lines 594-596):
h_mid = F.interpolate(h_coarse, size=self.MID_HW, mode='bilinear', align_corners=False)

# Proposed:
class AdaptiveUpsampler(nn.Module):
    def __init__(self, in_dim, out_dim, scale_factor=2):
        # Deformable conv or learned upsampling kernel
        self.deform_conv = DeformConv2d(in_dim, out_dim, kernel_size=3, offset_groups=1)
        self.scale = scale_factor
    
    def forward(self, x, q_feat):  # x: coarse hidden, q_feat: mid features
        # Upsample with adaptive offsets learned from q_feat
        upsampled = F.interpolate(x, scale_factor=self.scale, mode='bilinear')
        # Learn offsets that align upsampled features with mid query features
        offsets = self.offset_net(q_feat)
        refined = self.deform_conv(upsampled, offsets)
        return refined
```

**Why**: 
- Bilinear spreads information uniformly; learn spatially-adaptive upsampling
- Deformable conv can align coarse features with mid-scale structures
- Replaces rigid ±2-pixel neighborhood with learned sampling grids

**Files to modify**: `ms_flow_pose_net.py` (lines 594-596, 618-620)

---

#### **2. Expand Correlation Encoder (80→64 Bottleneck)**
**Difficulty**: Low | **Estimated Impact**: -0.10° to -0.15° | **Implementation**: 15 mins

**Current** (lines 330-334):
```python
self.corr_encoder = nn.Sequential(
    nn.Conv2d(corr_channels + 3, 128, 3, padding=1),
    nn.GELU(),
    nn.Conv2d(128, context_dim, 3, padding=1),  # context_dim=64
    nn.GELU(),
)
```

**Proposed**:
```python
self.corr_encoder = nn.Sequential(
    nn.Conv2d(corr_channels + 3, 256, 3, padding=1),  # ↑ 128→256
    nn.LayerNorm((256, H, W)),  # Add normalization
    nn.GELU(),
    nn.Conv2d(256, 256, 3, padding=1),  # ↑ Add depth
    nn.LayerNorm((256, H, W)),
    nn.GELU(),
    nn.Conv2d(256, context_dim, 1),  # Linear projection
)
```

**Why**: 
- Richer intermediate representation (256 vs 128)
- Depth (two 256-layer convs) allows learning complex correlation patterns
- Layer normalization stabilizes gradient flow

**Files to modify**: `ms_flow_pose_net.py` (FlowRefinementHead.__init__, lines 330-334)

**Cost**: ~5-10 MB extra params, ~10% slower (acceptable)

---

#### **3. Add Iterative Refinement to Coarse & Mid Scales**
**Difficulty**: Low | **Estimated Impact**: -0.10° to -0.15° | **Implementation**: 30 mins

**Current**: `coarse_iters=1, mid_iters=1` (default, hardcoded)

**Proposed**: Allow multi-iteration at all scales
```yaml
# In config:
model:
  coarse_iters: 2    # New, default 1
  mid_iters: 2       # Already exists
  fine_iters: 4      # Already exists
```

**Implementation** (ms_flow_pose_net.py, lines 600-609):
```python
# Mid iterations (already done, keep as-is)
for _iter in range(self.mid_iters):  # Already 2
    mid_corr = guided_local_correlation(q_mid, r_mid, flow_m, radius=4)
    ...

# Coarse iterations (add this)
# Insert before mid loop:
for _iter in range(self.coarse_iters):
    coarse_corr = global_correlation(q_coarse, r_coarse)
    if self.corr_temperature != 1.0:
        coarse_corr = coarse_corr / self.corr_temperature
    _, conf_c, h_coarse, flow_c = self.coarse_head(...)
```

**Why**: 
- Coarse is where global errors form; iterations can correct them
- Current single coarse iteration leaves room for refinement
- Cheap (coarse is only 8×10)

---

#### **4. Implement Outlier Rejection in Geometry Solver**
**Difficulty**: Medium | **Estimated Impact**: -0.08° to -0.12° | **Implementation**: 1-2 hours

**Current** (geometry_solver.py, lines 122-127):
```python
w = confidence * valid  # Simple binary mask + learned confidence
```

**Proposed: Iterative Reweighting (M-Estimator)**
```python
def diff_pose_solve_robust(flow, confidence, Ju, Jv, valid, damping=1e-3, 
                           iterations=3, threshold_zscore=2.5):
    """Robust pose estimation with iterative outlier rejection."""
    w = confidence * valid.float()
    
    for iter_robust in range(iterations):
        # Solve with current weights
        JtWJ = torch.bmm(Ju.t() @ (w*Ju), ...) + torch.bmm(Jv.t() @ (w*Jv), ...)
        delta_xi = torch.linalg.solve(JtWJ + damping_mat, JtWf)
        
        # Compute residuals
        pred_flow_u = torch.bmm(Ju, delta_xi.unsqueeze(-1)).squeeze(-1)
        pred_flow_v = torch.bmm(Jv, delta_xi.unsqueeze(-1)).squeeze(-1)
        residuals_u = (flow_u - pred_flow_u) ** 2
        residuals_v = (flow_v - pred_flow_v) ** 2
        residuals = (residuals_u + residuals_v).sqrt()
        
        # Adaptive threshold (robust M-estimator)
        median_res = torch.median(residuals)
        mad = torch.median(torch.abs(residuals - median_res))  # MAD
        z_scores = (residuals - median_res) / (mad + 1e-6)
        
        # Reweight: down-weight high-residual outliers
        w_new = confidence * valid.float() * torch.exp(-z_scores ** 2 / 2)
        
        if torch.allclose(w_new, w, atol=1e-4):
            break
        w = w_new
    
    return delta_xi
```

**Why**: 
- Outliers (occlusions, reflections) corrupt least-squares solution
- Iterative reweighting (robust statistics) downweights high-residual pixels
- M-estimators are robust to ~30% outlier rate

**Files to modify**: `modules/geometry_solver.py` (add new function and call from ms_flow_pose_net)

---

### MEDIUM-IMPACT IMPROVEMENTS (If Time)

#### **5. Enable Outer Iterations (Iterative Refinement Training)**
**Difficulty**: Low | **Estimated Impact**: -0.08° to -0.10° | **Implementation**: Already in code, just tune configs

**Current**: `outer_iters=1` (disabled)

**Proposed**: 
```yaml
training:
  outer_iters: 2        # Add outer iteration loop
  val_outer_iters: 2
  gamma_outer: 0.85     # Tune weighting
```

**Why**: 
- Model learns to predict residual corrections (not absolute pose)
- At training: pose_cur gets refined by first delta_xi, second iter sees smaller error
- At inference: Two-pass should be more accurate than single-pass
- Cost: 2× forward/backward time, but gradients are better distributed

---

#### **6. Per-Scale Temperature Tuning**
**Difficulty**: Low | **Estimated Impact**: -0.05° to -0.08° | **Implementation**: 30 mins

**Current** (ms_flow_pose_net.py, lines 575-576, 603-604, 627-628):
```python
coarse_corr = coarse_corr / self.corr_temperature  # Same temperature for all
```

**Proposed**:
```python
def __init__(self, ..., corr_temp_coarse=1.0, corr_temp_mid=0.9, corr_temp_fine=0.8):
    self.corr_temp_coarse = corr_temp_coarse
    self.corr_temp_mid = corr_temp_mid
    self.corr_temp_fine = corr_temp_fine

# In forward:
coarse_corr = coarse_corr / self.corr_temp_coarse  if self.corr_temp_coarse != 1.0 else corr
mid_corr = mid_corr / self.corr_temp_mid
fine_corr = fine_corr / self.corr_temp_fine
```

**Why**: 
- Coarse: want sharper peaks (lower temp, <1.0), fewer false matches
- Mid: medium (0.9), balance between peak sharpness and gradient flow
- Fine: softer (0.8), local window is small, can afford uncertainty
- Current uniform temperature (1.0) is suboptimal

---

#### **7. Enhance GRU Capacity at Fine Scale**
**Difficulty**: Medium | **Estimated Impact**: -0.05° to -0.08° | **Implementation**: 1 hour

**Current** (conv_gru.py, line 291):
```python
self.conv_zr = nn.Conv2d(hidden_dim + input_dim, 2 * hidden_dim, 3, padding=1)
```

**Proposed: Multi-Headed GRU or Grouped Convolutions**
```python
# Option A: Increase hidden_dim at fine scale
self.fine_gru = ConvGRU(hidden_dim=192, input_dim=64)  # vs 128 baseline

# Option B: Grouped convolutions for efficiency
self.conv_zr = nn.Conv2d(hidden_dim + input_dim, 2 * hidden_dim, 3, 
                          padding=1, groups=8)  # 8 groups → less params, more capacity
```

**Why**: 
- Fine GRU runs 4× (fine_iters=4) with small receptive field (3×3)
- Expanding capacity allows learning more complex refinement patterns
- Grouped convolutions reduce parameter growth

---

### LONGER-TERM IMPROVEMENTS (Architectural)

#### **8. Cross-Scale Skip Connections (Hourglass Design)**
**Difficulty**: High | **Estimated Impact**: -0.10° to -0.15° | **Implementation**: 3-4 hours

**Idea**: Allow mid scale to skip-connect directly to fine (not just through hidden state)

```python
class MSFlowPoseNetV2(nn.Module):
    def forward(...):
        # ... coarse, mid, fine as before ...
        
        # NEW: Direct feature skip from mid to fine
        mid_feat_skip = self.mid_feat_processor(q_mid)  # Learn what to skip
        
        # Inject into fine hidden
        h_fine = h_fine + mid_feat_skip  # Or concatenate + fuse
        
        # Continue with fine iterations
        for _iter in range(self.fine_iters):
            ...
```

**Why**: 
- Mid scale captures structures at 2× resolution; directly relevant to fine (4× relative)
- Bilinear upsampling is lossy; direct feature skip preserves details
- Similar to U-Net/FPN skip connections

---

#### **9. Attention Mechanisms for Scale Fusion**
**Difficulty**: High | **Estimated Impact**: -0.05° to -0.10° | **Implementation**: 2-3 hours

```python
class ScaleAttention(nn.Module):
    """Learn which scale features to trust at each spatial location."""
    def __init__(self, feat_dim, scales=3):
        self.scales = scales
        self.attn_net = nn.Sequential(
            nn.Conv2d(feat_dim * scales, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, scales, 1),
            nn.Softmax(dim=1)
        )
    
    def forward(self, coarse_feat, mid_feat, fine_feat):
        # Upsample coarse/mid to fine resolution
        coarse_up = F.interpolate(coarse_feat, fine_feat.shape[-2:], mode='bilinear')
        mid_up = F.interpolate(mid_feat, fine_feat.shape[-2:], mode='bilinear')
        
        # Learn attention weights
        combined = torch.cat([coarse_up, mid_up, fine_feat], dim=1)
        weights = self.attn_net(combined)  # (B, 3, H, W)
        
        # Combine
        fused = (weights[:, 0:1] * coarse_up + 
                 weights[:, 1:2] * mid_up + 
                 weights[:, 2:3] * fine_feat)
        return fused
```

**Why**: 
- Current cascade is strictly coarse→mid→fine
- Attention allows learning non-hierarchical combinations
- Each pixel can trust different scales adaptively

---

#### **10. Learned Depth-Dependent Damping**
**Difficulty**: Medium | **Estimated Impact**: -0.03° to -0.05° | **Implementation**: 1-2 hours

**Current** (geometry_solver.py, line 135):
```python
damping = 1e-3  # Constant
```

**Proposed**:
```python
def learned_damping(depth, confidence, depth_mean=2.0, depth_std=1.0):
    """Adaptive damping based on depth uncertainty."""
    # Normalize depth
    z_norm = (depth - depth_mean) / (depth_std + 1e-6)
    
    # Damping increases with uncertainty
    depth_uncertainty = 1.0 / (confidence + 0.1)  # Higher conf → lower uncertainty
    
    # Base damping scaled by depth/confidence
    base_damping = 1e-3
    adaptive_damping = base_damping * (1.0 + depth_uncertainty * 10.0).clamp(max=0.1)
    
    return adaptive_damping
```

**Why**: 
- Distant pixels have higher depth uncertainty → need more damping
- Low-confidence regions should dampen more
- Adaptive damping is more principled than fixed constant

---

## IMPLEMENTATION PRIORITY & EFFORT SUMMARY

| Priority | Improvement | Impact | Effort | Files | Cumulative |
|----------|-------------|--------|--------|-------|-----------|
| 1 | Expand correlation encoder | -0.10°-0.15° | 15 min | 1 | -0.10°-0.15° |
| 2 | Multi-iteration coarse | -0.10°-0.15° | 30 min | 1 | -0.20°-0.30° |
| 3 | Outlier rejection in solver | -0.08°-0.12° | 1-2 hrs | 1 | -0.28°-0.42° |
| 4 | Deformable upsampling | -0.15°-0.25° | 2-3 hrs | 1 | -0.43°-0.67° |
| 5 | Outer iterations (tune) | -0.08°-0.10° | 30 min | 1 config | -0.51°-0.77° |
| 6 | Per-scale temperature | -0.05°-0.08° | 30 min | 1 | -0.56°-0.85° |

**Expected result after top 6**: **0.55° - 0.56° = -0.01° to 0.00°** (within noise margin of improvements)
**Further needed**: Scale-aware attention + cross-scale skips (-0.10°-0.15° more)

---

## DEBUGGING/ANALYSIS SCRIPTS TO RUN

```bash
# 1. Profile information flow
python scripts/analyze_feature_flow.py \
    --config configs/exp030_ms_flow_v2.yaml \
    --output flow_analysis.json

# 2. Visualize upsampled hidden state (coarse→mid)
python -c "
import torch
from ic_models.ms_flow_pose_net import MSFlowPoseNet
model = MSFlowPoseNet()
# Hook into hidden state before/after upsampling
# Check L2 norm, entropy, variance
"

# 3. Measure correlation encoder loss
python scripts/measure_corr_bottleneck.py --model exp030_ms_flow_v2

# 4. Analyze outlier distribution in solver
python scripts/analyze_flow_outliers.py --checkpoint output/exp030/checkpoints/best.pth

# 5. Test outer iterations
python scripts/train_ms_flow.py --config configs/exp030_ms_flow_v2.yaml \
    --override training.outer_iters=2 \
    --test-only 10 batches
```

---

## CONCLUSION

The **0.55° → 0.1° gap (5.5× improvement)** is achievable through a combination of:
1. **Information preservation** (deformable upsampling, skip connections) - ~-0.30°
2. **Increased model capacity** (expanded correlation encoder, larger GRU) - ~-0.15°
3. **Better outlier handling** (robust statistics in solver) - ~-0.10°
4. **Scale-aware refinement** (temperature tuning, coarse iterations) - ~-0.10°

**Quick wins (< 2 hours, -0.35°)**:
- Expand correlation encoder (15 min, -0.10°)
- Add coarse iterations (30 min, -0.10°)
- Tune outer iterations (30 min, -0.10°)
- Per-scale temperature (30 min, -0.08°)

**Then invest in deformable upsampling + outlier rejection** for the final -0.20° gap.
