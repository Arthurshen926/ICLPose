# RADIO-to-Localization NVS Pose Energy Mainline

> **For agentic workers:** This is the active research mainline after `ChatGPT-特征训练与可微检索4.md`. Do not continue optimizing the old shallow selector path unless this plan explicitly asks for it.

**Goal:** Learn localization-oriented map/query features by transforming RADIO/DCFF supervision into a pose-conditioned neural energy and residual field for camera pose refinement.

**Core claim:** RADIO features are not directly localization features. The paper studies how to reconstruct and adapt RADIO-derived map/query features so they induce a useful pose energy landscape for learning-based camera pose refinement.

**Non-goals for the main paper path:** explicit 2D-3D/PnP-RANSAC as the primary solver, hand-crafted local-correlation topK selection, WLS step-size tuning, global full relocalization as the main claim.

---

## 1. Updated Research Claim

The main claim should be:

```text
We reconstruct a 3D feature field from RADIO/DCFF supervision, then optimize map/query features with pose-conditioned learning signals so the resulting feature space supports camera pose refinement.
```

Equivalent short framing:

```text
RADIO -> localization-oriented feature reconstruction
```

This keeps the original intent:

- learning-based CPR, not a traditional geometric pipeline;
- map-side and query-side features both matter;
- 3DGS/DCFF is not only a feature renderer but a source of dense pose-conditioned supervision;
- localization loss reshapes feature reconstruction instead of merely matching the teacher.

## 2. Why the Old Mainline Stops

The current evidence is sufficient to stop the old selector path:

| Evidence | Result | Decision |
|---|---:|---|
| K64 oracle | good, around `92mm` on the key diagnostic | candidate coverage exists |
| feature score Spearman | around `0.05` | raw fine similarity is not pose evidence |
| RGB L1 rerank | near random | not enough |
| query gate / score-map selector | no generalization | stop |
| scene-coordinate head | meter-level | not a short-term main path |
| WLS update | mean worsens / outliers | diagnostic only |

The bottleneck is not candidate coverage. It is:

```text
The learned/raw query-map feature evidence does not form a generalizable pose-conditioned energy landscape.
```

## 3. New Mainline: NVS-PoseEnergy

Name:

```text
NVS-PoseEnergy
```

Full name:

```text
3DGS Novel-View Supervised Pose Energy Learning
```

Data flow:

```text
3DGS/DCFF map
  -> render synthetic query RGB at GT pose T*
  -> query-only domain randomization
  -> query encoder/projector Q(I)

candidate pose bank {T_k}
  -> render map feature/depth/mask M(T_k)
  -> PoseEnergyNet(Q, M(T_k), pose_delta_k, aux_k)
  -> energy E_k, residual delta_xi_k, confidence sigma_k
```

Inference:

```text
real query + external/local T0
  -> K64 local pose bank
  -> render map features for candidates
  -> PoseEnergyNet scores candidates
  -> optional 1-3 learned residual updates
  -> refined pose
```

This remains learning-based and feature-field based. No explicit 2D-3D solver is the main stage.

## 4. Model Units

### Query Feature Path

Use the existing query student as the feature extractor. For the first stage:

- freeze the query backbone;
- allow only `local_corr_projector` or a new query pose-energy projector to train;
- keep a strong drift/teacher anchor.

Reason: previous runs showed direct query feature drift can destroy map-query alignment.

### Map Feature Path

Use current safe map/DCFF render as the stable base.

Stage A/B keep map frozen. Stage C may add a small rerank adapter or decoder fine-tune.

Never update:

```text
Gaussian geometry, opacity, scale, rotation, RGB color
```

### PoseEnergyNet

Inputs per candidate:

```text
projected query fine feature
projected rendered map fine feature
render mask / alpha
depth statistics
pose delta from bank center or T0
candidate rank/prior optional
```

Outputs:

```text
energy_logit: [B,K]
residual_delta: [B,K,6]
confidence: [B,K]
```

First implementation can reuse score-map/correlation infrastructure, but it must not be only a scalar selector. The residual head is required.

## 5. Losses

For each GT pose `T*`, sample candidates:

```text
T_k = exp(xi_k) T*
```

Pose cost:

```text
c_k = translation_error(T_k, T*) + rot_cost_weight * rotation_error(T_k, T*)
```

Energy target:

```text
p_k = softmax(-c_k / tau)
L_energy = CE(softmax(-E_k), p_k)
```

Residual target:

```text
delta_xi*_k = log(T* T_k^-1)
L_residual = Huber(delta_xi_k, delta_xi*_k)
```

Improvement loss:

```text
T'_k = exp(alpha * delta_xi_k) T_k
L_improve = max(0, d(T'_k, T*) - d(T_k, T*) + margin)
```

Anchors:

```text
L_real_align: real query feature aligns with map render at GT pose
L_teacher_anchor: weak RADIO/DCFF teacher anchor
L_query_drift: projected feature statistics do not collapse
L_map_drift: only in map fine-tune stage
```

## 6. Novel-View Training Data

Synthetic query creation:

```text
sample T* near training trajectory
render RGB_syn from 3DGS
apply strong query-only augmentation
sample K64/K128 candidate poses around T*
render map feature/depth/mask at candidates
supervise energy and residual from known pose labels
```

Query-only augmentation:

```text
brightness / contrast / gamma / color temperature
Gaussian noise / shot noise / blur / JPEG
vignetting / exposure shift
resize-crop / focal jitter / principal point jitter
random occlusion / patch erase / alpha dropout
feature channel dropout / spatial dropout
```

Important constraint:

```text
Do not apply the same augmentation to map render features.
```

Otherwise the model learns renderer-to-renderer shortcuts instead of real-query-to-map alignment.

## 7. Training Stages

### Stage A: Frozen Map, Pose Energy Pretraining

Train:

```text
query projector
PoseEnergyNet
residual head
confidence head
```

Freeze:

```text
map field
query backbone
coarse adapter/scorer
old selector
```

Batch mix:

```text
50% synthetic novel views
50% real training queries
```

Success gates:

| Eval | Target |
|---|---:|
| synthetic-heldout K64 Spearman | `> 0.60` |
| synthetic-heldout good/bad AUC | `> 0.85` |
| real 0.25m/5deg Spearman | `> 0.15` first gate, `> 0.25` strong gate |
| real 0.25m/5deg selected mean | `< 150mm` first gate |
| projected map-query fine cosine | `>= 0.78` |

### Stage B: Add Learned Residual Updates

Use PoseEnergyNet residual prediction for 1-3 learned updates:

```text
score K64 -> apply residual to topM -> re-render -> score/update again
```

Success gates:

| Bucket | Target |
|---|---:|
| 0.25m/5deg final mean | `< 120mm` |
| 0.25m/5deg gain+ | `> 65%` |
| 0.5m/10deg final mean | `< 180mm` |

### Stage C: Rerank Adapter, Base Map Frozen

Add a small rerank adapter:

```text
F_r = F_base + alpha * A_r(F_base)
```

Train:

```text
rerank_adapter
query projector
PoseEnergyNet
```

Freeze:

```text
base map decoder
latent
hash MLP
geometry
query backbone
```

Success gate:

```text
At least one main bucket improves > 10% over Stage B without reducing gain+.
```

### Stage D: Localization-Guided Map Fine-Tune

Only after Stage B/C passes.

LR order:

```text
pose_energy_net: 1.0x
query projector: 0.3x - 0.5x
rerank adapter: 0.3x - 0.5x
map decoder/FSM: 0.01x - 0.05x
map latent: 0.001x - 0.01x
geometry: 0
```

Must include replay:

```text
teacher anchor
real GT align
synthetic GT align
energy loss
map drift from safe checkpoint
```

Stop if pose improves only on synthetic but degrades real validation.

## 8. Implementation Targets

Create:

```text
feature_extract/students/pose_energy_net.py
feature_extract/tools/export_nvs_pose_energy_cache.py
feature_extract/tools/train_pose_energy.py
feature_extract/tools/eval_pose_energy_buckets.py
feature_extract/configs/nvs_pose_energy_stage_a_frozenmap.yaml
```

Reuse:

```text
feature_extract/train_impl.py::MapFeatureRenderer
feature_extract/train_impl.py::build_local_pose_lattice_candidates
feature_extract/train_impl.py::apply_pose_delta
feature_extract/train_impl.py::pose_error_tensors
feature_extract/train_impl.py::project_query_render_for_fine_selector
feature_extract/train_impl.py::local_render_score_feature_candidates
```

Do not delete old selector code. Keep it for ablation/no-go comparison.

## 9. First Minimum Experiment

Experiment name:

```text
nvs_pose_energy_stage_a_synthreal_frozenmap_k64
```

Scope:

```text
1000 synthetic train views
200 synthetic heldout views
all real train queries with GT perturb candidate banks
real val buckets: 0.1m/2deg, 0.25m/5deg, 0.5m/10deg
K=64
batch size as large as VRAM allows
```

Report:

```text
synthetic Spearman/AUC/oracle gap
real Spearman/AUC/oracle gap
selected pose mean/median
residual update gain
projected feature health
runtime/memory
```

Minimum go/no-go:

```text
Synthetic heldout must work first.
If synthetic Spearman < 0.6, fix PoseEnergyNet/loss before touching map/query.
If synthetic works and real fails, focus on domain randomization and real anchors.
If synthetic+real improves real Spearman but final pose is weak, add residual update loop.
```

## 10. Paper Story

Use this narrative:

1. RADIO is a strong general visual feature, but raw RADIO/DCFF similarity does not directly produce a robust pose energy for CPR.
2. We reconstruct a map-side feature field and query-side feature, then adapt them using pose-conditioned supervision.
3. 3DGS novel-view rendering supplies dense pose supervision without switching to an explicit 2D-3D/PnP pipeline.
4. A neural pose energy and residual field converts candidate pose banks into learnable refinement.
5. Map-side fine-tuning is only introduced after the energy field is stable, proving localization-guided reconstruction rather than uncontrolled feature drift.

## 11. Ablations

Required:

| Ablation | Purpose |
|---|---|
| raw RADIO/DCFF similarity selector | prove raw teacher feature is not enough |
| old score-map selector | preserve no-go baseline |
| synthetic only | measure domain gap |
| real only | measure coverage gap |
| synthetic + real anchor | prove NVS helps |
| no residual head | prove energy-only is insufficient |
| no query augmentation | prove domain randomization matters |
| frozen map vs rerank adapter | prove map-side localization adaptation |
| frozen map vs Stage D map fine-tune | prove localization-guided reconstruction |
| explicit PnP/RANSAC appendix only | clarify not the main claim |

## 12. Final Success Criteria

Minimum publishable CPR target:

| Item | Target |
|---|---:|
| synthetic heldout K64 Spearman | `> 0.60` |
| real 0.25m/5deg final mean | `< 120mm` |
| real 0.25m/5deg gain+ | `> 65%` |
| real 0.5m/10deg final mean | `< 180-200mm` |
| map-side contribution | `> 10%` gain in at least one main bucket |
| old selector no-go documented | yes |
| NVS augmentation contribution | synthetic+real beats real-only |

Stretch:

| Item | Target |
|---|---:|
| real 0.25m/5deg final mean | `< 100mm` |
| real 0.5m/10deg final mean | `< 150mm` |
| real 1.0m/20deg final mean | `< 350mm` |

## 13. Immediate Decisions

Stop:

```text
score-map selector tuning
query gate variants
RGB L1 rerank
WLS scale tuning
map candidate-gradient fine-tune
scene-coordinate head as main path
coarse adapter retraining
```

Continue:

```text
safe map checkpoint
K64 jittered eval
oracle gap diagnostics
projected feature visualization
Spearman/AUC diagnostics
map/query feature health logging
```

Main next implementation:

```text
PoseEnergyNet + NVS synthetic/real pose-energy training cache + Stage A eval.
```
