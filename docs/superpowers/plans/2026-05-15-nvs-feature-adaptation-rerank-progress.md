# 2026-05-15 NVS Feature Adaptation / Rerank Progress

## Main Checkpoint

Current valid small-basin checkpoint:

```text
result/result/feature_extract/nvs_matchingloc_teacherquality_mixed_10_25_paircorr_from_mixedbest_b6_s60_h128_20260515/checkpoints/best.pth
```

Correct paircorr-enabled bucket eval:

| bucket | pred trans | pred rot | top1 | oracle gap |
|---|---:|---:|---:|---:|
| q10 / 2deg | 44.4mm | 0.56deg | 0.722 | 19.5mm |
| q25 / 5deg | 95.5mm | 1.42deg | 0.750 | 33.3mm |
| q50 / 10deg | 353.7mm | 5.28deg | 0.444 | 233.5mm |

Shuffling candidate order does not change these numbers, so the current paircorr
score itself is order-invariant.  However, the fixed local-lattice protocol still
has a serious artifact: the GT oracle candidate is always candidate index 1 in
q10/q25/q50 caches.

## q50 Cache Expansion

Exported larger q50 local-lattice render-LoFTR/PnP caches:

```text
pose_init_exports/oldhospital_local_lattice_renderloftr_q50cm10deg_top16_noexact_train256.npz
pose_init_exports/oldhospital_local_lattice_renderloftr_q50cm10deg_top16_noexact_val128.npz
```

Both have 100% LoFTR/PnP success.  q50 val128 candidate oracle is:

```text
oracle trans mean ~= 125mm
oracle rot mean ~= 2.5deg
```

PNP composite teacher selection is only:

```text
trans mean ~= 292mm
top1 oracle match ~= 0.49
oracle in PNP top4 ~= 0.93
oracle in PNP top8 ~= 0.96
```

This explains the current q50 plateau: the learned paircorr scorer is roughly
matching the PNP-composite teacher level, not closing the topK oracle gap.

## Negative q50 Experiments

All resumed from the q10/q25 paircorr checkpoint.

| run | best/observed eval | conclusion |
|---|---:|---|
| q50 adapter+pairmatcher, PNP teacher | ~328mm / 6.48deg | negative |
| q50 pairmatcher-only, PNP teacher | ~320mm / 5.39deg | negative |
| q50 adapter+pairmatcher, GT pose target, train96/val32 | ~312mm / 4.69deg | negative |
| q50 adapter+pairmatcher, GT pose target, train256/val128 | ~296mm / 5.78deg | still negative |
| q50 PoseEnergy pairheatmap selector | ~484-515mm cost, negative Spearman | negative |

Do not continue q50 by only increasing steps or tuning WLS/adapter LR.  The
bottleneck is topK reranking signal/capacity, not candidate coverage or cache
size.

## Protocol Fix Added

Added:

```text
feature_extract/tools/shuffle_pose_candidate_cache.py
```

Generated shuffled q10/q25/q50 256/128 caches with seed `20260515`, and added
matching config files.  These preserve the candidate set and teacher fields while
removing fixed candidate order.

## Next Mainline

1. Treat q10/q25 paircorr as the current Stage-1 small-basin result.
2. Stop claiming q50 success from fixed local-lattice until a real reranker
   closes the gap on shuffled or jittered protocols.
3. Implement the next selector as an explicit trainable topK reranker with:
   pair heatmaps, pose delta vectors, PNP teacher quality as optional auxiliary,
   and GT pose-error listwise/pairwise as the primary target.
4. Validate against shuffled caches first, then export a true jittered candidate
   protocol where the oracle is not a fixed directed candidate.
