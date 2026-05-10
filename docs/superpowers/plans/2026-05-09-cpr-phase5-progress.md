# 2026-05-09 CPR Phase5 Progress

## Current Mainline

The strongest current CPR path is still:

```text
local pose lattice around T0
-> frozen coarse candidate scorer topK
-> fine local-correlation rerank over topK
-> no forced WLS pose update
```

`fine_update_scale=0` is intentional for the current mainline. The fine branch is
used to select the best candidate from topK; WLS/confidence is still a diagnostic
because previous runs showed confidence-selected WLS can make poses worse.

## Effective Results

Fixed full medium lattice evaluation, val32, top8 fine-score rerank,
`fine_update_scale=0`:

| checkpoint | bucket | coarse top1 median | final median | final mean | gain+ | top4 | top8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen medium | 0.25m/5deg | 353.5mm | 0.003mm | 173.5mm | 0.625 | 0.625 | 0.844 |
| Phase5 GT-align map | 0.25m/5deg | 353.5mm | 0.002mm | 146.5mm | 0.625 | 0.625 | 0.844 |
| Phase5 outer50 candidate map | 0.25m/5deg | 558.9mm | 0.002mm | 151.8mm | 0.625 | 0.625 | 0.938 |
| Phase5 oracle-subset49 map | 0.25m/5deg | 558.9mm | 0.002mm | 116.8mm | 0.688 | 0.625 | 0.938 |
| frozen medium | 0.5m/10deg | 558.4mm | 0.002mm | 175.1mm | 0.781 | 0.688 | 0.906 |
| Phase5 GT-align map | 0.5m/10deg | 558.4mm | 0.001mm | 167.3mm | 0.750 | 0.688 | 0.906 |
| Phase5 outer50 candidate map | 0.5m/10deg | 558.4mm | 0.001mm | 264.7mm | 0.625 | 0.625 | 0.875 |
| Phase5 oracle-subset49 map | 0.5m/10deg | 558.4mm | 0.001mm | 220.6mm | 0.688 | 0.688 | 0.875 |

Interpretation:

- Phase5 GT-align is the only useful map-side fine-tune result so far.
- It improves 0.25m/5deg final mean by about 15.5%, which is enough to keep as
  evidence that map-side can be optimized for the CPR feature space.
- It does not yet improve gain+ and only gives a small 0.5m/10deg mean gain.
- Oracle-subset candidate-gradient is a cleaner implementation than K32 uniform
  or outer-only K49, but it still shifts the map toward small-basin behavior:
  K49 improves 0.25m/5deg mean to 116.8mm and gain+ to 0.688, while worsening
  0.5m/10deg mean to 220.6mm and lowering top8 recall to 0.875.
- The exact 0mm medians are an artifact of the fixed-noise protocol where the
  lattice contains exact inverse candidates. Use mean, gain+, and topK recall
  for decisions.

## Negative Results

### K32 uniform candidate-gradient

The transient K32 uniform run was stopped and its output/config were deleted.
Validation oracle/top8 coverage was only 0.375, so the loss was training on a
polluted candidate set.

### Mixed K49 candidate-gradient

The transient mixed-K49 run was stopped before validation and its config was
deleted. Training oracle frequently dropped below 1.0 because mixing 25cm/5deg
and 50cm/10deg perturbations with a compact outer lattice does not guarantee
strict basin coverage under the actual SE(3) composition.

### Outer50 K49 candidate-gradient

The transient outer50-K49 run had clean training coverage: oracle/top4/top8
reached 1.0 at validation step32 with pred_trans 296mm. However, when evaluated
with the full medium lattice, it worsened the final mean on 0.5m/10deg from
167.3mm to 264.7mm and reduced gain+ to 0.625. The large checkpoint directory
and config were deleted after saving the JSON metrics because the result is a
negative control, not a checkpoint to continue from.

Conclusion: candidate-gradient map fine-tune is not ready for the mainline. It
can overfit a compact candidate set and damage full-lattice ranking.

### Oracle-subset candidate-gradient

Implemented a safer training-time candidate subset:

```text
generate full K169 medium lattice
-> compute GT pose error in pose space
-> render GT-nearest candidate + uniform negative subset
```

This avoids K169 map-gradient OOM while guaranteeing that the positive candidate
is not dropped. Unit coverage was added for this selection path.

Results:

- K33 validation pred_trans: 267.5mm.
- K49 validation pred_trans: 216.9mm.
- Full-lattice eval showed K49 helps 0.25m/5deg but hurts 0.5m/10deg.

Conclusion: keep the code path for controlled future experiments, but do not
promote oracle-subset map-gradient to the main result yet.

## Current Bottleneck

Candidate coverage is not the bottleneck when the lattice matches the bucket.
The bottleneck is still selection:

- coarse scorer top1 is not reliable enough;
- topK coverage is useful;
- fine-score rerank recovers many cases;
- WLS confidence is not reliable enough to drive selection or update;
- map-side candidate-gradient can corrupt ranking if candidate coverage is not
  carefully controlled.

## Next Adjustments

1. Keep Phase5 GT-align as the current map-side contribution checkpoint.
2. Do not use Phase5 candidate-gradient results in the mainline yet; report
   oracle-subset49 as an ablation/diagnostic only.
3. Make the default CPR evaluation/report path:
   `topK=8`, `fine_select=fine_score`, `fine_update_scale=0`.
4. For map localization fine-tune, try a safer objective next:
   full-lattice frozen scorer loss with gradient only through a small selected
   subset that is guaranteed to contain the GT-nearest candidate, plus a strong
   GT-align regularizer.
5. Stop any map fine-tune if validation gain+ drops or full-lattice final mean
   regresses, even if cosine or compact-lattice pred_trans improves.

## 2026-05-10 Continuation

### Evaluation Infrastructure Fixes

Added eval-only controls:

- `--map-checkpoint` to load model/scorer from one checkpoint and map renderer
  state from another.
- `--candidate-render-batch-size` to override candidate render chunking at eval
  time and avoid OOM/very slow full-lattice evaluation.
- `fine_score_prior` plus `--fine-prior-weight` for a conservative topK
  rerank diagnostic.
- `--fine-score-stat` for `mean`, `max`, `topk_mean`, and `peakiness`
  local-correlation rerank diagnostics.

Regression tests were added in `tests/test_eval_cpr_buckets.py`.

### Current Best Results

Plan default lattice, val32, top8 fine-score rerank, `fine_update_scale=0`,
using adapter model plus safe decoder-only map checkpoint:

| bucket | final mean | final median | gain+ | top4 | top8 | fine top8 oracle mean |
|---|---:|---:|---:|---:|---:|---:|
| 0.1m/2deg | 143.9mm | 141.4mm | 0.312 | 1.000 | 1.000 | 59.6mm |
| 0.25m/5deg | 164.1mm | 0.0mm | 0.625 | 0.688 | 0.844 | 80.1mm |
| 0.5m/10deg | 105.6mm | 0.0mm | 0.844 | 0.750 | 0.938 | 46.8mm |
| 1.0m/20deg | 337.0mm | 0.0mm | 0.750 | 0.688 | 0.719 | 227.8mm |

Safe decoder-only map without the adapter is worse on the medium bucket:
0.5m/10deg final mean is 171.9mm versus 105.6mm for adapter+safe-map.

Interpretation:

- Safe map-side fine decoder is useful, mainly when combined with the
  adapter/coarse candidate path.
- The adapter helps medium/large basin; it should not be treated as a
  small-basin refiner.
- Exact 0mm medians remain a fixed-lattice artifact; use means, gain+, and
  topK/oracle-topK for decisions.

### Negative/Diagnostic Results

Adapter-only hard/rank fine-tunes on plan-medium and small lattices did not
learn useful ranking. `cfrank` stayed around 0.693 and validation pred_trans
worsened or stayed worse than the current adapter checkpoint.

Training `candidate_score_fusion_head + candidate_basin_adapter` also failed to
improve validation. First validation:

- planhard: pred_trans 332.9mm, top8 0.969.
- smallguard: pred_trans 233.1mm, top8 0.875.

Fine WLS updates are not stable as a main path:

| update scale | 0.25m final mean | 0.5m final mean | note |
|---:|---:|---:|---|
| 0.0 | 164.1mm | 105.6mm | current best mean |
| 0.1 | 170.4mm | 117.3mm | median improves, mean worsens |
| 0.2 | 177.1mm | 129.0mm | worse |
| 0.5 | 199.3mm | 164.3mm | worse |
| 1.0 | 243.6mm | 223.8mm | over-shoot |

Even oracle-within-top8 worsens as update scale increases, so the issue is not
only selection; the WLS update itself creates outliers.

Hand-coded fine rerank variants did not solve selection:

- `fine_score_prior` improves 0.1m/2deg to about 98-106mm, but damages
  0.25m/0.5m/1.0m.
- `topk_mean` and `max` local-correlation statistics are worse than the
  original mean statistic.

### Updated Bottleneck

The current bottleneck is now:

```text
coarse local lattice has enough oracle coverage
-> coarse topK is adequate for 0.25/0.5 and partly 1.0
-> fine topK contains good candidates
-> current fine reranker selects the wrong one
-> WLS update is not robust enough to rescue selection
```

This is visible from oracle-top8 means: 0.25m/5deg can reach about 80mm and
0.5m/10deg about 47mm if the system chooses correctly within top8.

### Next Mainline Adjustment

Stop spending cycles on:

- residual adapter-only ranking;
- scorer-head retuning on the same coarse features;
- WLS step-size tuning;
- hand-picked fine correlation summary statistics.

Next implementation should be a trainable fine topK candidate selector:

```text
query fine + rendered candidate fine/depth/mask
-> local correlation score maps and confidence summaries
-> light candidate selector logits over coarse topK
-> supervised by GT pose error within topK
-> optionally followed by very small/gated WLS only when confidence is high
```

The selector target should be GT pose error in topK, not WLS-soft target. Map
and query can stay frozen for the first selector probe; after it works, repeat
with safe map-side fine-tune loaded.

### Storage

Removed two old implicit-retrieval feature export directories that are not used
by the current CPR configs:

- `features_radio_dual_v74_implicit_joint_pose_energy`
- `features_radio_dual_v75_implicit_joint_pose_energy_no_teacher`

This freed about 36GB; `/root/ICLPose/result` went from about 46GB free to
about 81GB free.
