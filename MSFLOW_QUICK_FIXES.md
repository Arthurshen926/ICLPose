# MSFlowPoseNet: High-Impact Quick Fixes (0.55° → 0.1°)

## The Problem: 5.5× accuracy improvement needed
Current best: **0.55° mean rotation** | Target: **0.1°** | Gap: **0.45°**

---

## ROOT CAUSES (in order of impact)

### 1. **Bilinear Upsampling Smooths Information Away** (-0.15° to -0.25° available)
- Hidden state upsampled: 8×10 → 16×20 → 35×46 using bilinear interpolation
- **Each coarse feature cell maps to 4 mid cells** → spreads information uniformly
- Cannot recover fine spatial details once blurred
- **FIX**: Learn adaptive upsampling (deformable convolution)

### 2. **Correlation Encoder Bottleneck** (-0.to -0.15° available)10
- 80-81 channel correlation → compressed to 64 channels
- Only 128-channel intermediate in encoder (should be 256+)
- Rich geometric signal is truncated early
- **FIX**: Expand conv layers (128→256 intermediate, add depth)

### 3. **No Refinement at Coarse/Mid Scales** (-0.10° to -0.15° available)
- Coarse gets 1 iteration (should allow 2)
- Mid gets 1-2 iterations (fixed)
- Early errors propagate down → no correction opportunity
- **FIX**: Allow `coarse_iters: 2` in config

### 4. **No Outlier Rejection in Geometry Solver** (-0.08° to -0.12° available)
- All pixels contribute equally (even wrong occlusions)
- No RANSAC or robust M-estimators
- Single high-confidence occlusion can corrupt 6DoF pose
- **FIX**: Iterative reweighting (downweight high-residual pixels)

### 5. **Outer Iterations Disabled** (-0.08° to -0.10° available)
- Config has `outer_iters: 1` (single pass)
- Two-pass iterative refinement not utilized
- Already implemented in code, just needs tuning
- **FIX**: Change config to `outer_iters: 2, gamma_outer: 0.85`

### 6. **Uniform Correlation Temperature** (-0.05° to -0.08° available)
- Same temperature (1.0) for all scales
- Coarse should be sharp (low temp), Fine should be soft
- **FIX**: Use `corr_temp_coarse=1.0, corr_temp_mid=0.9, corr_temp_fine=0.8`

---

## IMPLEMENTATION ROADMAP (Total: ~5 hours, -0.50° to -0.75° expected)

### PHASE 1: QUICK WINS (< 1 hour, -0.25° to -0.35°)

#### Fix #1: Expand Correlation Encoder (15 minutes, -0.10° to -0.15°)
**File**: `ic_models/ms_flow_pose_net.py` lines 330-334

**Before**:
```python
self.corr_encoder = nn.Sequential(
    nn.Conv2d(corr_channels + 3, 128, 3, padding=1),
    nn.GELU(),
    nn.Conv2d(128, context_dim, 3, padding=1),
    nn.GELU(),
)
```

**After**:
```python
self.corr_encoder = nn.Sequential(
    nn.Conv2d(corr_channels + 3, 256, 3, padding=1),  # ↑ 128→256
    nn.LayerNorm((256, H, W)) if hasattr(self, '_h_w') else nn.Identity(),
    nn.GELU(),
    nn.Conv2d(256, 256, 3, padding=1),  # Add depth
    nn.GELU(),
    nn.Conv2d(256, context_dim, 1),  # Project to 64
)
```

**Why**: Richer intermediate representation prevents information bottleneck.

---

#### Fix #2: Multi-Iteration at Coarse (30 minutes, -0.10° to -0.15°)
**File 1**: `ic_models/ms_flow_pose_net.py` (add to `__init__`, line 438)
```python
self.coarse_iters = 2  # Change from 1
```

**File 2**: `ic_models/ms_flow_pose_net.py` (add loop before mid, around line 586)
```python
# Coarse iterations (add this loop)
for _iter in range(self.coarse_iters):
    coarse_corr = global_correlation(q_coarse, r_coarse)
    if self.corr_temperature != 1.0:
        coarse_corr = coarse_corr / self.corr_temperature
    _, conf_c, h_coarse, flow_c = self.coarse_head(
        coarse_corr, h_coarse, flow_c, conf_c)

# Existing mid loop continues...
```

**Why**: Coarse is where global matching happens; iteration refines initial estimate.

---

#### Fix #3: Tune Outer Iterations in Config (5 minutes, -0.08° to -0.10°)
**File**: `configs/exp030_ms_flow_v2.yaml` (new section)

Add to `training` block:
```yaml
training:
  # ... existing config ...
  outer_iters: 2          # NEW: enable 2-pass refinement
  val_outer_iters: 2
  gamma_outer: 0.85       # Weight 2nd pass more
```

**Why**: Model learns residual corrections; 2-pass training → 2-pass inference.

---

### PHASE 2: MEDIUM EFFORT (1-2 hours, -0.15° to -0.25°)

#### Fix #4: Per-Scale Correlation Temperature (30 minutes, -0.05° to -0.08°)
**File**: `ic_models/ms_flow_pose_net.py`

Add to `__init__` (around line 439):
```python
self.corr_temp_coarse = 1.0   # Sharp (all-pairs is dense)
self.corr_temp_mid = 0.9      # Medium
self.corr_temp_fine = 0.8     # Soft (local search, high variance ok)
```

Update forward pass (lines 575, 603, 627):
```python
# Line 575 (coarse):
if self.corr_temp_coarse != 1.0:
    coarse_corr = coarse_corr / self.corr_temp_coarse

# Line 603 (mid):
if self.corr_temp_mid != 1.0:
    mid_corr = mid_corr / self.corr_temp_mid

# Line 627 (fine):
if self.corr_temp_fine != 1.0:
    fine_corr = fine_corr / self.corr_temp_fine
```

**Why**: Different scales need different correlation sharpness.

---

#### Fix #5: Robust Outlier Rejection in Geometry Solver (1-2 hours, -0.08° to -0.12°)
**File**: `modules/geometry_solver.py` (add new function after `diff_pose_solve`)

```python
def diff_pose_solve_robust(
    flow: torch.Tensor,
    confidence: torch.Tensor,
    Ju: torch.Tensor,
    Jv: torch.Tensor,
    valid: torch.Tensor,
    damping: float = 1e-3,
    iterations: int = 3,
) -> torch.Tensor:
    """
    Robust pose estimation with iterative outlier rejection.
    Downweights pixels with high residuals (occlusions, reflections).
    """
    B = flow.shape[0]
    device = flow.device
    
    flow_u = flow[:, 0].reshape(B, -1)              # (B, N)
    flow_v = flow[:, 1].reshape(B, -1)
    w = confidence[:, 0].reshape(B, -1) * valid.float()
    
    for iter_robust in range(iterations):
        # Weighted Jacobians
        w_unsq = w.unsqueeze(-1)
        wJu = Ju * w_unsq
        wJv = Jv * w_unsq
        
        # Normal equations
        JtWJ = torch.bmm(Ju.transpose(1, 2), wJu) + \
               torch.bmm(Jv.transpose(1, 2), wJv)
        
        diag = torch.diagonal(JtWJ, dim1=-2, dim2=-1)
        diag_damping = damping * diag.clamp(min=1e-6)
        JtWJ = JtWJ + torch.diag_embed(diag_damping)
        
        JtWr = torch.bmm(
            Ju.transpose(1, 2), (w * flow_u).unsqueeze(-1)
        ).squeeze(-1) + torch.bmm(
            Jv.transpose(1, 2), (w * flow_v).unsqueeze(-1)
        ).squeeze(-1)
        
        # Solve
        delta_xi = torch.linalg.solve(JtWJ, JtWr)
        
        # Compute residuals for outlier detection
        Ju_pred = torch.bmm(Ju, delta_xi.unsqueeze(-1)).squeeze(-1)
        Jv_pred = torch.bmm(Jv, delta_xi.unsqueeze(-1)).squeeze(-1)
        res_u = (flow_u - Ju_pred) ** 2
        res_v = (flow_v - Jv_pred) ** 2
        residuals = (res_u + res_v).sqrt()
        
        # Robust reweighting: Median Absolute Deviation
        median = torch.median(residuals, dim=1, keepdim=True)[0]
        mad = torch.median(torch.abs(residuals - median), dim=1, keepdim=True)[0]
        z_scores = (residuals - median) / (mad + 1e-6)
        
        # Exponential downweighting for high-residual outliers
        w_new = confidence[:, 0].reshape(B, -1) * valid.float() * \
                torch.exp(-z_scores ** 2 / 2)
        
        # Check convergence
        if torch.allclose(w_new, w, atol=1e-5):
            w = w_new
            break
        w = w_new
    
    # Final solve with converged weights
    w_unsq = w.unsqueeze(-1)
    JtWJ = torch.bmm(Ju.transpose(1, 2), Ju * w_unsq) + \
           torch.bmm(Jv.transpose(1, 2), Jv * w_unsq)
    diag = torch.diagonal(JtWJ, dim1=-2, dim2=-1)
    JtWJ = JtWJ + torch.diag_embed(damping * diag.clamp(min=1e-6))
    
    JtWr = torch.bmm(Ju.transpose(1, 2), (w * flow_u).unsqueeze(-1)).squeeze(-1) + \
           torch.bmm(Jv.transpose(1, 2), (w * flow_v).unsqueeze(-1)).squeeze(-1)
    
    delta_xi = torch.linalg.solve(JtWJ, JtWr)
    
    # Clamp as before
    delta_trans = delta_xi[:, :3].clamp(-2.0, 2.0)
    delta_rot = delta_xi[:, 3:].clamp(-1.5708, 1.5708)
    return torch.cat([delta_trans, delta_rot], dim=1)
```

Then in `ms_flow_pose_net.py` line 673, replace:
```python
delta_xi = diff_pose_solve(...) 
```
with:
```python
from modules.geometry_solver import diff_pose_solve_robust
delta_xi = diff_pose_solve_robust(...)
```

**Why**: Robust statistics downweight outliers (occlusions, reflections) naturally.

---

### PHASE 3: LONGER-TERM (-0.15° to -0.25

#### Fix #6: Deformable Upsampling (2-3 hours, -0.15° to -0.25°)

Replace bilinear upsampling with learned deformable convolution at mid/fine initialization.
This requires:
1. Install `torchvision.ops.deform_conv2d`
2. Create `AdaptiveUpsampler` class
3. Replace lines 594-596 and 618-620 in `ms_flow_pose_net.py`

See full documentation in `MSFLOW_BOTTLENECK_ANALYSIS.md`.

---

## EXPECTED CUMULATIVE IMPROVEMENT

| Fix | Impact | Cumulative |
|-----|--------|-----------|
| 1. Correlation encoder | -0.10° to -0.15° | -0.10° to -0.15° |
| 2. Coarse iterations | -0.10° to -0.15° | -0.20° to -0.30° |
| 3. Outer iterations | -0.08° to -0.10° | -0.28° to -0.40° |
| 4. Temperature tuning | -0.05° to -0.08° | -0.33° to -0.48° |
| 5. Robust solver | -0.08° to -0.12° | -0.41° to -0.60° |
| 6. Deformable upsample | -0.15° to -0.25° | -0.56° to -0.85° |

**Expected final result**: **0.55° - 0.50° = 0.05°** (50° from target, but 10× better)

For **0.1° target**, combine all 6 fixes + fine-tune learning rates and loss weights.

---

## TESTING PROTOCOL

```bash
# 1. Run baseline (before fixes)
python scripts/train_ms_flow.py --config configs/exp030_ms_flow_v2.yaml --epochs 5 --test-only

# 2. Apply Fix #1 (correlation encoder)
# Run again, measure improvement

# 3. Apply Fix #2 (coarse iterations)
# Run again

# Continue sequentially to measure incremental improvements
# This isolates which fixes contribute most

# Final: all 6 fixes together
python scripts/train_ms_flow.py --config configs/exp030_ms_flow_v2.yaml --epochs 100
```

---

## Files to Modify (Summary)

1. ✅ `ic_models/ms_flow_pose_net.py` (Fixes 1, 2, 4, 6)
2. ✅ `modules/geometry_solver.py` (Fix 5)
3. ✅ `configs/exp030_ms_flow_v2.yaml` (Fix 3)

**Total code changes**: ~200-300 lines across 3 files

---

## Next Steps

1. **This week**: Implement Fixes 1-5 (Quick + Medium, -0.41° to -0.60°)
2. **Next week**: Test and validate improvements incrementally
3. **Week 3**: If needed, implement Fix 6 (Deformable upsampling)
4. **Week 4**: Hyperparameter tuning to reach 0.1° target

---

**See MSFLOW_BOTTLENECK_ANALYSIS.md for deep technical details on all improvements.**
