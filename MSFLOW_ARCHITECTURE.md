# MSFlowPoseNet Architecture Deep Dive

## 1. FULL MODEL ARCHITECTURE

### 1.1 Overall Design
**MSFlowPoseNet** is a single-pass, coarse-to-fine multi-scale optical flow network for 6-DOF camera pose estimation:

```
Query/Render Features (from 3DGS + DINO)
    ↓
[Shared Decoders] → Decode all scales to 64d
    ↓
COARSE LEVEL (7×10 @ 80ch global correlation)
    ↓ (upsample + warp-guided correlation)
MID LEVEL (15×20 @ 81ch local correlation) → 1 iteration
    ↓ (upsample + warp-guided correlation)
FINE LEVEL (35×46 @ 81ch local correlation) → 4 iterations (RAFT-style)
    ↓
[Image Jacobian + Weighted Least Squares]
    ↓
6-DOF Pose Update (Δξ = [vx, vy, vz, ωx, ωy, ωz])
```

### 1.2 Layer-by-Layer Architecture

#### **DECODERS (Shared Q/R)**
Three scale-specific decoders map high-dim features to 64d normalized features:

1. **ScaleDecoder** (Coarse & Mid)
   - Input: 512d features (ODISE-projected, not raw 1280d)
   - Conv: 512d → 256d → 64d (three 1×1 convs with GroupNorm+GELU)
   - Output: (B, 64, H, W) **L2 normalized**
   - **Params**: ~0.22M per decoder (0.44M coarse+mid)

2. **FineDualDecoder** (Fine)
   - **Input branches:**
     - SD s3: 512d @ 32×40 (with optional bilinear interp to 35×46)
     - DINO: 768d @ 35×46
   - **Processing:**
     - Each branch: 1×1 conv to 256d → GELU → 1×1 to 64d
     - Concatenate (128d) → 1×1 fuse → 64d
   - Output: (B, 64, 35, 46) **L2 normalized**
   - **Params**: ~0.36M

#### **CORRELATION FUNCTIONS**

1. **Global Correlation** (Coarse only)
   ```
   corr(q, r) = einsum('bcn,bcm->bmn', q_flat, r_flat)
   Output: (B, H2×W2, H1, W1) = (B, 80, 7, 10)
   Memory: 440 KB/sample @ BS=1
   ```
   - Every query pixel vs. all render pixels (all-pairs)
   - No computational complexity control but provides global context
   - **Cost**: O(N1 × N2) = O(1610 × 80) dot products

2. **Local Correlation** (Mid/Fine)
   ```
   For each query pixel, search (2r+1)² = 81 neighborhood patches in render
   unfold → vectorized dot products → (B, 81, H, W)
   Memory: 95 KB/sample (mid), 509 KB/sample (fine)
   ```
   - Uses PyTorch `unfold` for vectorized neighbor extraction
   - **Much faster than nested loops**: ~10× GPU, ~50× CPU speedup
   - **Cost**: O(N × d²) = O(1610 × 81) dot products per scale

3. **Warp-Guided Local Correlation** (Mid & Fine refinement)
   ```
   grid = base_grid + flow_from_previous_scale
   warped_r = F.grid_sample(fmap_r, grid, bilinear)
   local_corr(q, warped_r)
   ```
   - Adaptive search window centered on prior flow prediction
   - Prevents large displacements from being clipped by local window
   - Grid cache avoids redundant meshgrid calls

#### **GRU REFINEMENT BLOCKS**

Three identical **FlowRefinementHead** instances (one per scale):

```
ConvGRU Architecture:
  Input: corr_ch (80/81) + flow(2) + confidence(1) = 83-84 channels
    ↓
  Correlation Encoder: Conv (80ch) → 128d → (context_dim=64d)
    ↓
  ConvGRU Cell:
    - Fused z/r gates: Conv([h, x]) → 2×128d then chunk
    - Candidate: Conv([r⊙h, x]) → 128d
    - Update: h_new = (1-z)⊙h + z⊙h_candidate
    ↓
  Flow Head: 128d → Conv64d → Conv3d → (Δu, Δv, raw_confidence)
  
Params per head: ~0.9M (mostly in GRU and flow head convs)
```

**Key design**: 
- Fused z/r computation saves 2 convolutions per step
- **Hidden dim=128** is empirically tuned (larger→slower, smaller→less representational power)
- Per-scale flow decoding allows scale-specific feature learning

#### **CONTEXT ADAPTERS** (Refinement between scales)
```python
ContextAdapter(h_upsampled, q_feat_current_scale):
  concat([h_up, q_feat]) → Conv64→Conv64 → residual add to h_up
```
- Injects scale-specific query features into upsampled GRU state
- Prevents loss of spatial information from upsampling
- **Params**: 0.37M per adapter (mid+fine)

#### **CONTEXT NETWORK** (Initial hidden state)
```
Query features (64d) at coarse → Conv3d→128d → Conv3d→128d → h_coarse
```
- Initializes GRU hidden state from query context
- **Params**: 0.22M

---

## 2. MULTI-SCALE FEATURE HANDLING

### 2.1 Resolution Hierarchy

| Level | Resolution | Use | Pixels | Global Corr Ch | Local Corr Ch |
|-------|-----------|-----|--------|---|---|
| **Coarse** | 7×10 | Initial flow estimate | 70 | 80 | - |
| **Mid** | 15×20 | Coarse→Fine handoff | 300 | - | 81 |
| **Fine** | 35×46 | Final flow + pose | 1,610 | - | 81 |

**exp050 variant** (high-res attempt): fine=60×80 (4,800 pixels, 3.7× memory)

### 2.2 Feature Interpolation Between Scales

**Upsampling pipeline:**
```python
# After coarse flow prediction
flow_c, conf_c = (B, 2, 7, 10), (B, 1, 7, 10)

# Upsample to mid resolution
flow_m_init, conf_m_init = interpolate(flow_c, conf_c, target=(15, 20))
flow_m_init[:, 0] *= (20/10)  # scale x component
flow_m_init[:, 1] *= (15/7)   # scale y component
```

**Critical detail**: When upsampling optical flow, must scale the flow VALUES by the resolution scaling factor, not just the spatial dimensions.

**Context fusion** (ContextAdapter):
```python
h_mid = bilinear_interp(h_coarse, size=(15, 20))
h_mid = h_mid + context_adapter(h_mid, q_mid)  # residual
```

### 2.3 Could We Add a Super-Fine Scale?

**Theoretical possibility**: Add ultra-fine 70×93 (~6,500 pixels) after fine

**Challenges**:
1. **Memory**: Fine alone (1,610px) creates 509 KB correlation per sample
   - Super-fine would be: 6,500 × 81 × 4 bytes = 2.1 MB per sample
   - At BS=4: +8.4 MB memory (hits typical 11 GB limit)

2. **Diminishing returns**: 
   - 35×46 = 268 pixels : 1 DOF already overdetermined
   - 60×80 = 400 pixels : 1 DOF (exp050)
   - 70×93 = 750 pixels : 1 DOF (marginal improvement)

3. **Practical constraint**: 
   - DINO features are 35×46 (fixed patch grid)
   - Super-fine would require SD s2 features (128×170, too expensive to compute)

**Recommendation**: Current 35×46 is near-optimal. exp050's jump to 60×80 requires batch_size reduction (4→2) due to memory.

---

## 3. GEOMETRY SOLVER INTEGRATION

### 3.1 Image Jacobian Computation

Maps pixel displacement (flow) to 6-DOF camera motion via projective geometry:

```
For pixel (u, v) at depth Z under camera intrinsics (fx, fy, cx, cy):
  Normalized coords: x = (u - cx)/fx,  y = (v - cy)/fy
  
  Pixel displacement under camera motion ξ=[tx, ty, tz, ωx, ωy, ωz]:
  
  Δu = fx[-ωx·x·y + y(1+x²) - ωz·y + tx/Z - x·tz/Z]
  Δv = fy[-ωx(1+y²) + ωy·x·y + ωz·x + ty/Z - y·tz/Z]
  
  Jacobian matrices (B, N, 6):
  Ju = [fx/Z,  0, -fx·x/Z, -fx·xy, fx(1+x²), -fx·y]
  Jv = [  0, fy/Z, -fy·y/Z, -fy(1+y²), fy·xy, fy·x]
```

**Implementation** (geometry_solver.py):
```python
def compute_image_jacobian(depth, intrinsics) -> (Ju, Jv, valid):
    # Returns (B, N, 6) Jacobians + (B, N) validity mask
    # N = H*W = 1610 for fine resolution
```

### 3.2 Weighted Least Squares Solver

**Core equation**:
```
minimize_ξ: Σ w[i] · (||Ju[i]·ξ - flow_u[i]||² + ||Jv[i]·ξ - flow_v[i]||²)

Normal equations: (J^T W J) ξ = J^T W flow

Where:
  J^T W J = (B, 6, 6) — Hessian matrix
  J^T W flow = (B, 6) — RHS vector
```

**Levenberg-Marquardt damping** (adaptive):
```python
diag = torch.diagonal(JtWJ)  # (B, 6)
diag_damping = damping * diag.clamp(min=1e-6)  # per-DOF adaptive
JtWJ_regularized = JtWJ + torch.diag_embed(diag_damping)
```

**Advantages of LM-style damping**:
- Each DOF damped proportionally to its Hessian eigenvalue
- Natural scale invariance (translation in meters, rotation in radians)
- More stable than fixed λI damping

**Solution**:
```python
delta_xi = torch.linalg.solve(JtWJ_reg, JtWr)  # (B, 6)

# Safety clamps:
delta_trans = delta_xi[:, :3].clamp(-2.0, 2.0)      # ±2m per iteration
delta_rot = delta_xi[:, 3:].clamp(-π/2, π/2)        # ±90° per iteration
```

### 3.3 Confidence-Weighted Flow

```python
confidence = exp(log_confidence)  # FlowHead outputs log_conf
w = confidence * valid_mask
```

- Pixels with high predicted confidence weighted more in pose solve
- Natural way to discount uncertain or occluded regions
- Calibrated by **confidence regularization loss** during training

### 3.4 Why Geometry Over Learned PoseHead?

| Aspect | Learned PoseHead | Geometry (Image Jacobian) |
|--------|---|---|
| Constraints | 1 image = 6 DOF solve | 1,610 pixels = 268:1 overconstrained |
| Differentiability | ✓ learned backprop | ✓ analytic Jacobian |
| Invertibility | Black box | Transparent linear equations |
| Robustness | Outlier-sensitive | Confidence weighting handles occlusions |
| GPU memory | Negligible (global pooling) | ~2.5 MB @ BS=1 for Jacobians |

---

## 4. MEMORY & COMPUTE ANALYSIS

### 4.1 Memory Breakdown (Per Sample, Float32)

```
ACTIVATION MEMORY:
 Feature maps
  ├─ Coarse:    7×10×64d = 17.5 KB
  ├─ Mid:      15×20×64d = 75 KB
  └─ Fine:     35×46×64d = 402.5 KB

 Correlation matrices (BIGGEST SINK)
  ├─ Coarse global: 80×1610 = 440 KB
  ├─ Mid local:     81×300 = 95 KB
  └─ Fine local:    81×1610 = 509 KB

 GRU hidden states
  ├─ Coarse:    7×10×128d = 35 KB
  ├─ Mid:      15×20×128d = 150 KB
  └─ Fine:     35×46×128d = 805 KB

 Flow predictions (4 fine iters)
   └─ 4×1610×3 = 75.5 KB

TOTAL PER SAMPLE: ~2.5 MB
```

**With batching**:
- BS=2: ~5.1 MB
- BS=4: ~10.2 MB ← typical training config
- BS=8: ~20.4 MB

### 4.2 Most Memory-Intensive Operations

1. **Fine local correlation** (509 KB/sample, ~20% of total)
   - Unfold + pointwise multiply + sum
   - Cannot be avoided; essential for flow refinement
   
2. **Fine GRU hidden state** (805 KB/sample, ~32%)
   - 35×46×128 dimensions required for spatial reasoning
   - Possible optimization: reduce hidden_dim 128→64 (would save 400 KB/sample)

3. **Flow predictions storage** (75.5 KB/sample, 3%)
   - Fine layer stores 4 flow predictions (RAFT sequence loss)
   - Could checkpoint/discard intermediate flows, recompute on backward

4. **Coarse global correlation** (440 KB/sample, 18%)
   - Only computed once (no refinement)
   - Could use lower precision (fp16) since not differentiated w.r.t. float operations

### 4.3 Most Compute-Intensive Operations

**Per forward pass priorities**:

1. **Local correlation computation** (~40% compute)
   ```python
   fmap2_unfold = fmap2_pad.unfold(2, d, 1).unfold(3, d, 1)  # (B,C,H,W,d,d)
   corr = (fmap1.unsqueeze(-1) * fmap2_unfold).sum(dim=1)    # (B,H,W,d²)
   ```
   - Fine: H×W×C×d² = 35×46×64×81 ≈ 8.5M multiply-adds
   - 4 iterations (fine_iters=4) = 34M MACs

2. **Warp-guided interpolation** (~15% compute)
   ```python
   F.grid_sample(fmap_r, grid, bilinear)  # 4×C×H×W interpolation
   ```
   - 4 iters × 35×46 = 6,440 pixel samples
   - Each pixel: 4-neighborhood bilinear interpolation

3. **ConvGRU gates** (~30% compute)
   ```python
   conv_zr = Conv2d(128+64, 256, 3)  # 128×64×3×3×3 FLOPS per spatial location
   ```
   - Three convolutions per refinement step
   - 4 fine iterations = 4× this cost

4. **Geometry solving** (~5% compute, but serializes on Jacobian)
   ```python
   Ju, Jv = compute_image_jacobian(depth, intrinsics)  # O(N) = O(1610)
   JtWJ = bmm(Ju.T, wJu) + bmm(Jv.T, wJv)             # O(6²×N) = O(58,000)
   delta_xi = linalg.solve(JtWJ, JtWr)                 # O(6³) = O(216)
   ```
   - Geometry solving NOT on the critical path (happens after fine refinement)
   - Only 6×6 solve, negligible

### 4.4 Effect of Doubling decode_dim (64 → 128)

**Impact**:
- Feature maps: 64 channels → 128 (2×)
- Correlation matrices: **NO CHANGE** (correlate in original space, not decoded space)
- GRU hidden_dim would likely stay 128 (decoupled from decode_dim)
- Context network: 128d → 256d (2×)

**Memory change**:
```
Features: +64×(70+300+1610) pixels = +120 KB/sample = +480 KB @ BS=4
Correlations: 0 KB (unchanged)
Context: +128 KB/sample = +512 KB @ BS=4
TOTAL: +1 MB @ BS=4 (~10% increase)
```

**Compute change**:
- Decoder: More 1×1 convs, negligible difference
- No GRU change (still 128d internally)
- **Negligible speedup expected**, possible slight slowdown from memory bandwidth

**Recommendation**: Doubling decode_dim has marginal benefit (diminishing returns after 64d) and costs memory. Not recommended.

### 4.5 Effect of Increasing batch_size on Memory

Linear scaling:
- BS 4→8: +2× memory consumption
- At BS=4 already consuming ~10 MB activations
- BS=8 → ~20 MB activations + ~8 MB weights + ~2 MB gradients = 30 MB peak

**Empirical limits**:
- 11 GB GPU: Can comfortably run BS=4, marginal with BS=8
- 24 GB GPU: Can run BS=8-16 without stress
- 40 GB GPU: Can run BS=32+

**Training strategy**: exp039/exp050 both use BS=4 (exp050 drops to BS=2 due to fine resolution increase).

---

## 5. CURRENT HYPERPARAMETER CONFIGURATIONS

### 5.1 exp039_room0_optimized.yaml

**Model**:
- hidden_dim: 128
- decode_dim: 64
- local_radius: 4 (9×9 search window)
- damping: 0.001
- fine_iters: 8 (aggressive refinement)
- mid_iters: 1
- coarse_hw: [15, 20] (unusual; typically [7, 10])
- mid_hw: [30, 40] (unusual; typically [15, 20])
- fine_hw: [35, 46] (standard)

**Data**:
- batch_size: 4
- noise_rot_deg: 8.0° (constant, no curriculum)
- noise_trans_m: 0.25 (constant)

**Training**:
- epochs: 120 (extended)
- lr: 0.0002 (conservative)
- outer_iters: 5 (heavy re-rendering)
- Loss weights:
  - flow: coarse=0.1, mid=0.3, fine=1.0
  - pose: 1.0
  - rot/trans weights: 1.0 / 10.0
  - gamma: 0.95 (early GRU iters more important)
  - gamma_outer: 0.95 (early re-renders more important)

**Key insight**: Constant 8° noise + gamma=0.95 (was 0.8) proved better than curriculum learning. No pose loss for first 8 epochs (phase1_epochs).

### 5.2 exp050_room0_highres.yaml

**Only differences from exp039**:
- fine_hw: [60, 80] (3× resolution increase, breaks 0.81° floor)
- mid_hw: [30, 40] (adjusted to better handoff)
- batch_size: 2 (reduced from 4 to fit higher-res in memory)
- batch_size: 2

**Rational**: Hypothesis that structural floor (0.81°) is due to fine resolution (35×46 = 1610px) being insufficient. Jump to 60×80 = 4800px adds 3× constraint density.

**Memory impact**: +~2.8 MB @ BS=4, hence BS reduction to 2.

---

## 6. TRAINING SCRIPT DETAILS

### 6.1 Loss Computation Pipeline

```python
def train_step(batch):
    # Render query/reference frames
    pose_perturbed = add_noise(pose_gt)
    feats_q, depth_q = renderer(pose_perturbed)
    feats_r, depth_r = renderer(pose_gt)
    
    # Forward pass
    model_out = model(feats_q, feats_r, depth_r)
    
    # Multi-scale flow loss (RAFT sequence loss for fine)
    loss_flow, metrics_flow = multiscale_flow_loss(
        model_out, gt_flows, weights={coarse:0.1, mid:0.3, fine:1.0},
        gamma=0.95  # γ^(N-1-i) weighting for sequence loss
    )
    
    # Confidence regularization
    loss_conf, metrics_conf = confidence_regularization_loss(
        model_out, coverage_range=[0.05, 0.95]
    )
    
    # Pose loss (only after phase 1)
    if epoch > phase1_epochs:
        T_delta = se3_exp(model_out['delta_xi'])
        pose_pred = T_delta @ pose_perturbed
        loss_pose, metrics_pose = pose_loss(
            pose_pred, pose_gt, 
            rot_weight=1.0, trans_weight=10.0,  # trans weighted 10×
            rot_loss_type='cosine'  # smooth gradient
        )
        
        # Warmup for pose loss weight
        pose_loss_weight = min(1.0, (epoch - phase1_epochs) / warmup_epochs)
        loss_total = loss_flow + 0.01*loss_conf + pose_loss_weight*loss_pose
    else:
        loss_total = loss_flow + 0.01*loss_conf
    
    return loss_total
```

### 6.2 Batch Processing & Data Loading

```python
# Data: PoseDatasetV4
class PoseDatasetV4:
    def __init__(self, feature_dir, traj_path, depth_dir, noise_rot, noise_trans):
        # Load pre-extracted multiscale features
        # Load trajectory (pose groundtruth)
        # Load depth maps
        
    def __getitem__(self, idx):
        feats = {
            'coarse': load_compressed(f'{feature_dir}/coarse/{idx}.pth'),
            'mid': load_compressed(f'{feature_dir}/mid/{idx}.pth'),
            'fine_sd': load_compressed(f'{feature_dir}/fine_sd/{idx}.pth'),
            'fine_dino': load_compressed(f'{feature_dir}/fine_dino/{idx}.pth'),
        }
        pose_gt = load_trajectory(idx)
        depth = load_depth(idx)
        
        # Add noise to pose
        pose_noisy = add_se3_noise(pose_gt, noise_rot_deg, noise_trans_m)
        
        return feats, pose_gt, pose_noisy, depth
```

**Key params**:
- num_workers: 4 (async loading)
- batch_size: 4 (exp039) or 2 (exp050)
- Prefetch: DataLoader keeps N=2×num_workers batches in RAM

### 6.3 What Controls GPU Memory Usage Most?

**Rank by memory impact**:

1. **fine_hw** (resolution) — most influential
   - 35×46 → 60×80: +3.7× memory
   - Cannot be easily compressed
   
2. **batch_size** — linear scaling
   - BS 4→8: +2× memory
   - Typical knob for memory tuning

3. **fine_iters** (number of GRU refinement steps)
   - Stores 4 flow predictions for sequence loss
   - Could checkpoint/recompute to save ~75 KB/sample
   
4. **hidden_dim** (GRU state dimension)
   - 128 → 64: saves 400 KB/sample
   - But risks capacity (empirically tuned to 128)

5. **decode_dim** — minimal impact
   - 64 → 128: +480 KB @ BS=4
   - Already checked: doubling has negligible benefit

6. **local_radius** — not impactful
   - radius 4→5: d² = 81→121 channels
   - Fine correlation: 509 KB → 765 KB (+256 KB/sample @ BS=1)
   - But larger radius often improves accuracy

**Recommendation**: If memory is limiting:
1. Reduce batch_size 4→2 or 3 (easy, common)
2. Reduce fine_iters 4→2 (some accuracy loss)
3. Reduce hidden_dim 128→96 (not recommended, hurts representational power)
4. Accept lower fine_hw (e.g., stick with 35×46)

---

## 7. OPTIMIZATION OPPORTUNITIES

### 7.1 Accuracy-focused optimizations

| Change | Accuracy Impact | Memory | Compute | Difficulty |
|--------|---|---|---|---|
| fine_iters 4→8 | +0.5° | ↑ none | ↑ 2× fine | Easy |
5 | +0.3° | ↑ none | ↑ re-rendering | Easy |
| gamma 0.8→0.95 | +0.2° | ↑ none | ↑ none | Easy |
| No curriculum | +0.1° | ↑ none | ↑ none | Easy |
| fine_hw 35×46→60×80 | +1.0° | ↑↑ high | ↑ 2× | Medium |
| hidden_dim 128→256 | unknown | ↑↑ high | ↑ moderate | Medium |
| decode_dim 64→128 | ~0° | ↑ low | ↑ negligible | Easy |

### 7.2 Speed optimizations

| Change | Speed Gain | Accuracy | Difficulty |
|--------|---|---|---|
| fine_iters 8→4 | 2× | -0.3° | Easy |
| outer_iters 5→1 | 5× | -1.0° | Easy |
| Remove mid refinement | 1.2× | -0.2° | Easy |
| Checkpoint fine flows | 1.1× (memory) | 0° | Medium |
| Mixed precision (fp16 geometry) | 1.05× | 0° | Medium |
| Fused operators (ConvGRU) | 1.15× | 0° | Hard |

### 7.3 Memory optimizations

| Change | Memory Saved | Accuracy | Difficulty |
|--------|---|---|---|
| Checkpoint fine flows | 75 KB/sample | 0° | Medium |
| hidden_dim 128→96 | 400 KB/sample | -0.2° | Easy |
| local_radius 4→3 | 45 KB/sample | -0.5° | Easy |
| Reduce batch_size | linear | 0° | Easy |
| fp16 features | 50% | ±0.1° | Easy |

---

## 8. KEY TUNING DIMENSIONS

### 8.1 Critical Hyperparameters

**Accuracy threshold**: The following must NOT be changed without careful retraining:
- **hidden_dim**: 128 (empirically tuned; affects convergence)
- **decode_dim**: 64 (larger has diminishing returns)
- **local_radius**: 4 (9×9 search; 3→smaller window, 5→larger window)

**Recommended tunable**:
- **fine_iters**: 4-8 (sweet spot; 4 = fast, 8 = accurate)
- **outer_iters**: 3-5 (3 = fast, 5 = very accurate)
- **gamma**: 0.8-0.99 (0.95 is good balance)
- **batch_size**: 2-8 depending on GPU (larger = better gradient estimates)

### 8.2 Per-dataset Tuning Checklist

For new dataset (e.g., ScanNet, 7-Scenes):

1. **Find base intrinsics** (from camera calibration or COLMAP)
2. **Set fine_hw** to match DINO grid (typically 35×46, but depends on image size)
3. **Tune noise levels** (room0 uses 8°+0.25m; may differ for other datasets)
4. **Set outer_iters** based on compute budget (5 for high accuracy, 1-2 for speed)
5. **Monitor gamma**: Start at 0.95, only lower if sequence loss becomes noisy
6. **Adjust rot_weight / trans_weight** if rotation and translation errors are unbalanced

---

## 9. SUMMARY TABLE

| Component | Size | Memory | Compute |
|-----------|------|--------|---------|
| **Decoders** | 0.8M | negligible | ~1% |
| **Flow Heads** | 2.7M | ~200 KB | ~30% |
| **Context Net** | 0.2M | ~50 KB | ~1% |
| **Adapters** | 0.7M | ~100 KB | ~2% |
| **Geometry Solver** | 0 (no params) | ~2.5 MB | ~5% |
| **Correlation ops** | 0 (no params) | ~1 MB | ~40% |
| **GRU + Warp** | 0 (in Flow Heads) | ~1.2 MB | ~20% |

**Bottleneck**: Fine local correlation (509 KB/sample) and ConvGRU hidden states (805 KB/sample) together consume 64% of activation memory.

**Optimization**: All current design choices are near-Pareto optimal. Further improvements require algorithmic changes (e.g., lower-rank correlations, distillation) rather than parameter tuning.

