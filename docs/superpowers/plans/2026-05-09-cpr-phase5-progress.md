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
