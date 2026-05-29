# VFM-MapLoc Status And Next Steps

## Current Status

The branch now has a clean VFM-MapLoc mainline:

```text
raw VFM token bank
  -> localizable feature selector
  -> selected feature map
  -> map-conditioned hypothesis verification
  -> fixed solver handoff
```

Old RADIO/POFD/CPR active paths and old experiment configs have been removed
from the tracked branch. The active code lives under `feature_extract/vfm/`,
`feature_extract/tools/vfm/`, `feature_extract/configs/vfm/`, and `docs/vfm/`.

Implemented infrastructure:

- protocol records and no-leak checks
- candidate hypothesis schema and JSONL bank IO
- raw token-bank manifest and checksum validation
- manifest CLI for existing NPZ token banks
- model-backed token extraction runner interface for RADIO/C-RADIO and DINOv2
- normalized JSON/CSV candidate adapters with query grouping
- native pose-init NPZ and reference-pose NPZ candidate adapters
- generic score-table JSONL adapter for labeled fixed-candidate evaluation
- candidate-bank metadata scoring baselines
- token-candidate dataset indexing with pose-label boundary checks
- selector module with group gates, projection, utility, and uncertainty
- selector losses for track consistency, hard-negative contrast, listwise pose
  ranking, basin BCE, sparsity, and uncertainty calibration
- explicit COLMAP track-supervised selector pretraining with consistency,
  track-ID contrast, and geometry-utility targets
- selected track feature aggregation
- selected track feature bank NPZ persistence
- rendered selected-map feature scoring against query selected features
- hard-case subset builder for retrieval-top1-wrong, near-identity false
  positives, and high-PnP-score wrong candidates
- score-table evaluation and protocol-kind checks
- metadata-only, feature shuffle, and high/low utility masking controls
- random and PCA same-dimension projection controls
- mapability metrics for coverage, track variance, separability, and storage
- Kendall, ECE, risk-coverage AUC, catastrophic failure rate, and ranking metrics
- paired bootstrap CI, McNemar exact test, and Wilcoxon signed-rank helpers
- descriptor-bank causality controls for query shuffle, map shuffle, and
  wrong-scene map features
- synthetic positive-control validation for the feature-utility gate
- synthetic selector-training positive control
- disk-level synthetic pipeline that writes token, candidate, selected-track,
  score, and report artifacts
- selected-descriptor cache builder for trained dense selectors
- multi-GPU job-array launcher for independent full-scene experiments
- resumable token extraction via `--skip_existing`
- provenance hashes in selector training reports, scoring reports, descriptor
  caches, and selected track-bank summaries
- full-scene training fast paths that skip repeated checksum scans, cache
  sampled dense tokens instead of full dense maps, and group COLMAP track
  sampling by image

Verification run:

```text
python -m pytest tests/test_vfm_*.py -q
152 passed
python -m compileall -q feature_extract/vfm feature_extract/tools/vfm feature_extract/extractors
```

Latest focused engineering verification:

```text
python -m pytest tests/test_vfm_selected_descriptor_bank.py \
  tests/test_vfm_gpu_job_array.py \
  tests/test_vfm_extract_tokens_runner.py -q
8 passed
```

Synthetic positive-control command:

```text
python -m feature_extract.tools.vfm.run_synthetic_gate_validation \
  --query_count 64 \
  --candidates_per_query 8 \
  --seed 0 \
  --output output/vfm/synthetic_feature_utility/result.json
```

Observed synthetic control summary:

| method | mean_top1_acc | mean_pred_cost_m | mean_spearman |
| --- | ---: | ---: | ---: |
| selected_feature | 1.000 | 0.0589 | 0.996 |
| raw_vfm | 0.516 | 0.2256 | 0.609 |
| query_shuffle | 0.203 | 0.5088 | 0.071 |
| metadata_only | 0.094 | 0.5927 | 0.010 |

This is not a localization result. It is a positive control showing that the
new reporting/controls path can expose a known selected-feature signal and
detect shuffle/metadata-only degradation.

Disk-level synthetic pipeline command:

```text
python -m feature_extract.tools.vfm.run_synthetic_pipeline \
  --output_dir output/vfm/synthetic_pipeline \
  --query_count 64 \
  --candidates_per_query 8 \
  --seed 0
```

Artifacts written:

- `token_manifest.json`
- `candidate_bank.jsonl`
- `selected_tracks.npz`
- `score_reports.json`
- `feature_utility.md`

Real fixed-candidate adapter smoke:

```text
python -m feature_extract.tools.vfm.build_hypothesis_bank \
  --protocol_name oldhospital_real_retrieval_top20_fixed \
  --protocol_kind real_retrieval \
  --candidates result/result/feature_extract/pose_init_exports/oldhospital_netvlad_renderloftr_top20_quality_fields_test.npz \
  --output output/vfm/candidate_banks/oldhospital_real_retrieval_top20_fixed.jsonl
```

```text
python -m feature_extract.tools.vfm.build_hypothesis_bank \
  --protocol_name oldhospital_reference_pose_top10_fixed \
  --protocol_kind reference_pose \
  --candidates result/result/feature_extract/localizability_banks/cambridge_reference_pose/oldhospital_hloc_netvlad_top10_reference_pose.npz \
  --output output/vfm/candidate_banks/oldhospital_reference_pose_top10_fixed.jsonl
```

Observed conversion counts:

| bank | candidates |
| --- | ---: |
| OldHospital real retrieval top20 fixed | 3640 |
| OldHospital reference-pose top10 fixed | 1820 |

Reference-pose retrieval-order baseline at `5m/10deg` basin:

| metric | value |
| --- | ---: |
| query_count | 182 |
| mean_pred_cost_m | 4.1710 |
| mean_oracle_cost_m | 2.6280 |
| mean_oracle_gap_m | 1.5431 |
| mean_top1_acc | 0.4945 |
| mean_spearman | 0.3156 |
| basin_recall_at_5 | 0.6868 |
| basin_recall_at_10 | 0.7582 |

OldHospital real retrieval labeled score-table baselines at `0.25m/5deg` basin:

| method | pred_m | oracle_m | gap_m | top1 | spearman | basin@5 | basin@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| retrieval_order | 0.4290 | 0.1532 | 0.2758 | 0.5275 | 0.2456 | 0.7088 | 0.8022 |
| candidate_prior | 0.3350 | 0.1532 | 0.1818 | 0.5769 | 0.5065 | 0.7637 | 0.8077 |

Hard-case split from the same labeled real retrieval candidate table:

| subset | query_count |
| --- | ---: |
| retrieval_top1_wrong | 54 |
| pnp_high_score_wrong | 149 |
| near_identity_false_positive | 0 |

Synthetic selector-training positive control, CUDA seeds 0/1/2:

| seed | initial_loss | final_loss | top1 | signal_gate | noise_gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 1.0093 | 0.7988 | 1.000 | 0.5538 | 0.4347 |
| 1 | 1.0209 | 0.7762 | 1.000 | 0.5511 | 0.4383 |
| 2 | 0.9828 | 0.7773 | 1.000 | 0.5501 | 0.4394 |

## Real Cambridge VFM Artifacts

The `/hy-tmp/Cambridge_stdloc` dataset is now wired through clean VFM data
configs for all five Cambridge scenes. C-RADIO runs offline from the local
checkpoint; DINOv2 is not cached locally yet. Full-resolution C-RADIO extraction
must currently use `batch_size=1`; `batch_size=2` OOMs on a 24GB RTX 3090
because the full 1920x1080 token grid makes attention memory too large.

Extracted fp16 `radio_final` token banks:

| scene | split | records | shape | bytes |
| --- | --- | ---: | --- | ---: |
| OldHospital | train | 895 | `[1280,68,120]` | 17.10G |
| OldHospital | test | 182 | `[1280,68,120]` | 3.48G |
| ShopFacade | train | 231 | `[1280,68,120]` | 4.42G |
| ShopFacade | test | 103 | `[1280,68,120]` | 1.97G |
| KingsCollege | train | 1220 | `[1280,68,120]` | extracted |
| KingsCollege | test | 343 | `[1280,68,120]` | 6.55G |
| GreatCourt | train | 1532 | `[1280,68,120]` | extracted |
| GreatCourt | test | 760 | `[1280,68,120]` | 14.47G |
| StMarysChurch | test | 530 | `[1280,68,120]` | extracted |

The dense banks are also pooled into lightweight L2-normalized descriptor banks
under `output/vfm_token_descriptors_radio/` so reference-pose scoring and
selector smoke runs no longer repeatedly scan the full dense token files.
Newly generated descriptor banks:

- `output/vfm_token_descriptors_radio/KingsCollege/{train,test}_mean_l2.npz`
- `output/vfm_token_descriptors_radio/GreatCourt/train_mean_l2.npz`
- `output/vfm_token_descriptors_radio/StMarysChurch/test_mean_l2.npz`

Dense selector scoring can also be cached after selector projection, e.g.
`output/vfm_selected_descriptors_radio/OldHospital/*seed0*.npz`. Cached selected
descriptors reproduce direct dense-selector scoring for seed0:

| scene | metric | direct | cached |
| --- | --- | ---: | ---: |
| OldHospital | pred_m | 3.2807 | 3.2807 |
| OldHospital | top1 | 0.5934 | 0.5934 |
| OldHospital | Spearman | 0.7044 | 0.7046 |
| ShopFacade | pred_m | 1.3212 | 1.3212 |
| ShopFacade | top1 | 0.8155 | 0.8155 |
| ShopFacade | Spearman | 0.6177 | 0.6177 |

Multi-GPU extraction is now exercised through
`feature_extract.tools.vfm.run_gpu_job_array`. A full extraction batch used both
RTX 3090s at 100% utilization and completed:

| job | records | elapsed |
| --- | ---: | ---: |
| StMarysChurch test C-RADIO | 530 | 1531s |
| KingsCollege train C-RADIO | 1220 | 3527s |
| GreatCourt train C-RADIO | 1532 | manual GPU0 run |

Raw C-RADIO descriptor score-table baselines on reference-pose top10, using the
same `5m/10deg` basin:

| scene | method | pred_m | oracle_m | top1 | spearman | basin@5 | basin@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | retrieval_order | 4.1710 | 2.6280 | 0.4945 | 0.3156 | 0.6868 | 0.7582 |
| OldHospital | raw_radio_descriptor_cosine | 4.3968 | 2.6280 | 0.5385 | 0.2217 | 0.7088 | 0.7582 |
| ShopFacade | retrieval_order | 1.6348 | 0.9141 | 0.8058 | 0.3959 | 0.9029 | 0.9515 |
| ShopFacade | raw_radio_descriptor_cosine | 2.0930 | 0.9141 | 0.7184 | 0.1924 | 0.9029 | 0.9515 |
| KingsCollege | retrieval_order | 3.4313 | 2.1796 | 0.7259 | 0.3209 | 0.8717 | 0.9038 |
| KingsCollege | raw_radio_descriptor_cosine | 4.6030 | 2.1796 | 0.6472 | 0.0208 | 0.8455 | 0.9038 |
| GreatCourt | retrieval_order | 14.0610 | 5.8401 | 0.3171 | 0.1657 | 0.5053 | 0.5947 |
| GreatCourt | raw_radio_descriptor_cosine | 15.7702 | 5.8401 | 0.2395 | -0.0060 | 0.4368 | 0.5947 |

This is a useful negative/diagnostic result: mean-pooled raw RADIO is not a
strong localization evidence signal by itself. It sometimes improves basin
coverage, but it degrades median prediction and rank correlation versus
retrieval order, especially on ShopFacade.

Real COLMAP track-observation exports are now available beyond the initial two
scenes:

| scene | available observations | available tracks | exported observations | exported tracks |
| --- | ---: | ---: | ---: | ---: |
| KingsCollege | 4,414,843 | 335,020 | 100,000 | 5,350 |
| GreatCourt | 2,946,464 | 352,774 | 100,000 | 7,994 |
| StMarysChurch | 4,078,801 | 495,250 | 100,000 | 13,447 |

Descriptor-level selector smoke is implemented as an intentionally lightweight
real-data training check over fixed reference-pose candidates. It uses an
internal query split and listwise combined pose-cost labels, so it is not a
paper claim. It does show that a compact selector can sometimes improve over
raw descriptor cosine on held-out queries, but the signal is high-variance:

| scene | seeds | raw eval top1 mean | selector eval top1 mean | note |
| --- | ---: | ---: | ---: | --- |
| OldHospital | 3 | 0.361 | 0.472 | 2/3 seeds improve |
| ShopFacade | 3 | 0.317 | 0.429 | 2/3 seeds improve |

The current conclusion is conservative: descriptor-only selection is a
debuggable positive smoke, not the final selected dense/map-conditioned result.

KingsCollege and GreatCourt now have 5-seed descriptor warm-start selectors on
full reference-pose top10 candidates:

| scene | seeds | raw eval top1 mean | descriptor selector eval top1 mean |
| --- | ---: | ---: | ---: |
| KingsCollege | 5 | 0.249 | 0.446 |
| GreatCourt | 5 | 0.154 | 0.370 |

Real COLMAP track-observation export and dense-token sampling are now in the
clean VFM path. Smoke track banks from 20k train-map observations:

| scene | available observations | available tracks | sampled observations | raw track bank tracks | dim | mean variance |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | 1,589,350 | 202,014 | 20,000 | 3,244 | 1280 | 0.0892 |
| ShopFacade | 389,745 | 68,028 | 20,000 | 2,877 | 1280 | 0.1083 |

This closes the first real-data mapability plumbing step: COLMAP
`images.bin/points3D.bin/cameras.bin` -> observation JSONL -> dense token
sampling -> `SelectedTrackFeatureBank` NPZ. It is still a raw-feature bank;
the trained selected-feature bank is the next mapability step.

Same-observation, L2-normalized mapability controls are also generated for the
20k-observation smoke. Variance is now comparable across transforms:

| scene | transform | tracks | dim | mean track variance |
| --- | --- | ---: | ---: | ---: |
| OldHospital | raw RADIO | 3,244 | 1280 | 0.000092 |
| OldHospital | first64 | 3,244 | 64 | 0.001250 |
| OldHospital | PCA64 | 3,244 | 64 | 0.001953 |
| OldHospital | random64 | 3,244 | 64 | 0.001912 |
| ShopFacade | raw RADIO | 2,877 | 1280 | 0.000109 |
| ShopFacade | first64 | 2,877 | 64 | 0.001898 |
| ShopFacade | PCA64 | 2,877 | 64 | 0.001760 |
| ShopFacade | random64 | 2,877 | 64 | 0.002023 |

The trained descriptor-selector checkpoint can now be lifted into a selected
track bank through the same sampler. Same-dim `16` mapability reports, using a
512-track sampled between-track distance for bounded memory, are:

| scene | bank | tracks | dim | storage | variance | separability |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital | raw RADIO | 3,244 | 1280 | 33.3M | 0.000092 | 11813.0 |
| OldHospital | first16 | 3,244 | 16 | 0.45M | 0.007213 | 138.4 |
| OldHospital | PCA16 | 3,244 | 16 | 0.45M | 0.004389 | 302.9 |
| OldHospital | random16 | 3,244 | 16 | 0.45M | 0.009055 | 124.3 |
| OldHospital | selector16 | 3,244 | 16 | 0.45M | 0.005807 | 196.7 |
| ShopFacade | raw RADIO | 2,877 | 1280 | 29.5M | 0.000109 | 10494.5 |
| ShopFacade | first16 | 2,877 | 16 | 0.40M | 0.008561 | 127.3 |
| ShopFacade | PCA16 | 2,877 | 16 | 0.40M | 0.003979 | 336.6 |
| ShopFacade | random16 | 2,877 | 16 | 0.40M | 0.008617 | 133.0 |
| ShopFacade | selector16 | 2,877 | 16 | 0.40M | 0.007680 | 164.6 |

Current interpretation: selected16 is compact and more mappable than first16
or random16 by separability, but it does not beat PCA16 yet. This is a useful
diagnostic and a clear next target, not a final mapability claim.

The selected 3D map plumbing has now been extended to KingsCollege and
GreatCourt using the same 100k COLMAP observation exports and dense selector
seed0 checkpoints. These are provenance-tracked full-scene selected track
banks:

| scene | observations | tracks | dim | mean obs/track | variance | mean utility |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KingsCollege | 100,000 | 5,350 | 16 | 18.69 | 0.007653 | 0.471 |
| GreatCourt | 100,000 | 7,994 | 16 | 12.51 | 0.012240 | 0.455 |

Selector-projected descriptor scoring now applies the same selector to query
and reference descriptors before fixed-candidate scoring. Full-query results
are strong but include training-query leakage from the internal smoke split, so
the stricter numbers below use only each seed's held-out query subset:

| scene | method | pred_m | top1 | Spearman | basin@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| OldHospital | retrieval order | 4.268 | 0.528 | 0.291 | 0.648 |
| OldHospital | raw descriptor | 4.504 | 0.528 | 0.196 | 0.676 |
| OldHospital | selector descriptor | 3.559 | 0.593 | 0.672 | 0.704 |
| ShopFacade | retrieval order | 1.529 | 0.794 | 0.351 | 0.857 |
| ShopFacade | raw descriptor | 2.086 | 0.762 | 0.193 | 0.873 |
| ShopFacade | selector descriptor | 1.471 | 0.698 | 0.583 | 0.905 |

This is the first clean positive evidence for the new mainline: selector
projection improves rank correlation on held-out queries in both scenes and
improves predicted translation on both scenes. It still loses top1 to retrieval
on ShopFacade, so the expected contribution should be framed as hypothesis
verification/risk evidence, not pure retrieval-top1 replacement.

## Dense Selector And Rendered Map Verifier

The mainline now includes dense selector training and a rendered selected-map
verifier. The dense trainer uses train-query-only pose-cost normalization,
stable spatial sampling, optional warm-start from a descriptor selector
checkpoint, and records the train normalizers in the run summary. The rendered
map verifier validates that the selected track bank has observation provenance,
rejects query/test image provenance by default, validates feature dimensions
before scoring, and caches query/reference descriptors to avoid repeated
manifest checksum scans.

Leak-free warm-start dense selector training, using descriptor-selector
checkpoints as initialization and 1024 sampled token positions per image.
Seeds 0/1/2 are complete:

| scene | seeds | raw eval top1 mean | dense eval top1 mean | dense eval range | train top1 mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital | 3 | 0.278 | 0.500 | 0.389-0.611 | 0.621 |
| ShopFacade | 3 | 0.286 | 0.429 | 0.381-0.476 | 0.748 |

Seed-level summaries are stored under
`output/vfm/dense_selector_training/*_dense_warm_seed{0,1,2}_s1024_leakfree.json`.
The current full-query score-table rows below use the best available
OldHospital seed1 and ShopFacade seed2 checkpoints; they should be expanded to
all three seeds before final reporting.

Full-query 2D dense-selected scoring on fixed reference-pose top10 is a clear
positive result against retrieval order and raw mean-pooled descriptors. The
table below reports seeds 0/1/2 mean and range:

| scene | method | pred_m | top1 | Spearman | basin@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| OldHospital | retrieval order | 4.171 | 0.495 | 0.316 | 0.687 |
| OldHospital | raw RADIO descriptor | 4.397 | 0.538 | 0.222 | 0.709 |
| OldHospital | dense selector16 mean | 3.240 | 0.623 | 0.704 | 0.751 |
| OldHospital | dense selector16 range | 3.178-3.281 | 0.593-0.648 | 0.701-0.707 | 0.747-0.758 |
| ShopFacade | retrieval order | 1.635 | 0.806 | 0.396 | 0.903 |
| ShopFacade | raw RADIO descriptor | 2.093 | 0.718 | 0.192 | 0.903 |
| ShopFacade | dense selector16 mean | 1.346 | 0.838 | 0.623 | 0.945 |
| ShopFacade | dense selector16 range | 1.321-1.395 | 0.816-0.854 | 0.618-0.629 | 0.942-0.951 |

This is currently the strongest solver-free evidence-selection result in the
clean VFM branch. It should be promoted to 5 seeds and additional scenes before
paper claims, but it already supports continuing the dense selector path.

Additional full-scene dense selector training is now complete for
KingsCollege and GreatCourt. The training summaries are internal split
diagnostics, not final held-out paper numbers, but all seeds improve over raw
dense descriptor top1:

| scene | seeds | raw eval top1 mean | dense eval top1 mean | dense eval range |
| --- | ---: | ---: | ---: | ---: |
| KingsCollege | 5 | 0.238 | 0.435 | 0.362-0.536 |
| GreatCourt | 5 | 0.120 | 0.370 | 0.316-0.414 |

Cached seed0 selected descriptors give full-query reference-pose top10 scoring
without rescanning dense tokens:

| scene | method | pred_m | top1 | Spearman | basin@5 | gap_m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| KingsCollege | retrieval order | 3.431 | 0.726 | 0.321 | 0.872 | 1.252 |
| KingsCollege | raw RADIO descriptor | 4.603 | 0.647 | 0.021 | 0.845 | 2.423 |
| KingsCollege | dense selected descriptor seed0 | 3.226 | 0.816 | 0.445 | 0.901 | 1.046 |
| GreatCourt | retrieval order | 14.061 | 0.317 | 0.166 | 0.505 | 8.221 |
| GreatCourt | raw RADIO descriptor | 15.770 | 0.239 | -0.006 | 0.437 | 9.930 |
| GreatCourt | dense selected descriptor seed0 | 7.854 | 0.443 | 0.527 | 0.582 | 2.014 |

The same dense selector checkpoints were lifted into provenance-preserving 3D
selected track banks from the 20k-observation COLMAP smoke subsets:

| scene | tracks | dim | variance | separability | storage |
| --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital | 3,244 | 16 | 0.006249 | 187.2 | 0.45M |
| ShopFacade | 2,877 | 16 | 0.008700 | 144.1 | 0.40M |

Rendered selected-map scoring with these dense-selector track banks closes the
map-conditioned smoke loop. Four verifier variants have now been tested:

- `global visible-track mean`: ignores 2D geometry and mean-pools all selected
  tracks visible in the candidate reference image.
- `sparse token-grid`: rasterizes selected 3D track features to the reference
  image token grid using COLMAP observation xy and scores local selected
  query-map evidence.
- `projected token-grid`: projects selected 3D track `xyz` into the query
  camera using each candidate `Tcw` pose and query intrinsics.
- `projected token-grid + feature inliers`: adds selector-utility-weighted
  splatting, feature inlier fraction, and inlier-derived risk.

| scene | method | pred_m | top1 | Spearman | basin@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| OldHospital | descriptor-selector rendered map | 5.269 | 0.319 | 0.099 | 0.632 |
| OldHospital | dense-selector rendered map | 4.926 | 0.335 | 0.149 | 0.643 |
| OldHospital | sparse-grid rendered map r0 | 5.421 | 0.379 | 0.062 | 0.676 |
| OldHospital | sparse-grid rendered map r4 | 5.182 | 0.407 | 0.081 | 0.687 |
| OldHospital | projected-grid rendered map r4 | 4.626 | 0.505 | 0.084 | 0.725 |
| OldHospital | projected-grid + inlier/utility r4 | 4.581 | 0.505 | 0.077 | 0.731 |
| ShopFacade | descriptor-selector rendered map | 2.266 | 0.573 | 0.035 | 0.845 |
| ShopFacade | dense-selector rendered map | 2.206 | 0.602 | 0.082 | 0.883 |
| ShopFacade | sparse-grid rendered map r0 | 2.861 | 0.524 | -0.041 | 0.913 |
| ShopFacade | sparse-grid rendered map r4 | 2.775 | 0.544 | -0.073 | 0.913 |
| ShopFacade | projected-grid rendered map r4 | 2.830 | 0.553 | -0.120 | 0.874 |
| ShopFacade | projected-grid + inlier/utility r4 | 2.790 | 0.553 | -0.117 | 0.874 |

Interpretation: dense selection has clear localization evidence in the 2D
query-reference protocol. On reference-pose top10, rendered-map variants are
provenance-safe diagnostics but fail retention. The important protocol lesson is
that reference-pose candidates are not query-view rendered-pose hypotheses:
spatial projection is not expected to align when the candidate is simply a
nearby database image pose.

A clean `controlled_lattice` sanity protocol was added to avoid mixing these
candidate types. It builds GT-centered rendered-pose lattices from Cambridge
pose files and explicitly marks them as GT-assisted. Each query has one exact
GT pose plus 0.25m and 0.50m world-translation perturbations. The projected
selected-map verifier is then evaluated with a strict `0.1m/5deg` basin, so
only the exact pose is positive.

| scene | protocol | pred_m | top1 | Spearman | basin@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| OldHospital | GT-centered rendered-pose lattice | 0.133 | 0.544 | 0.679 | 0.978 |
| ShopFacade | GT-centered rendered-pose lattice | 0.056 | 0.786 | 0.668 | 1.000 |

This is not a deployment result because the lattice is GT-centered. It is a
positive verifier sanity check: the projected selected-map evidence ranks exact
query-view poses above 0.25/0.50m perturbations far more often than chance, and
its score decreases monotonically from exact to near to far candidates. The
next map-conditioned step should replace GT-centered lattices with init-centered
rendered-pose proposals from retrieval/HLoc/VO and keep the same verifier.

That init-centered rendered-pose protocol is now implemented. The new
`feature_extract.tools.vfm.build_init_pose_lattice` tool consumes a real init
candidate bank, applies q-level world-translation lattices (`q10`, `q25`,
`q50`) around non-oracle init poses, and labels pose error from Cambridge GT
poses for evaluation only. Artifacts:

- `output/vfm/candidate_banks/oldhospital_realinit_q10_rendered_pose_top1.jsonl`
- `output/vfm/candidate_banks/oldhospital_realinit_q25_rendered_pose_top1.jsonl`
- `output/vfm/candidate_banks/oldhospital_realinit_q50_rendered_pose_top1.jsonl`
- `output/vfm/candidate_banks/oldhospital_realinit_q50_rendered_pose_top4.jsonl`

OldHospital real retrieval init-centered q-lattice summary:

| protocol | scorer | pred_m | top1 | Spearman | basin@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| q10 top1 | source init + offset prior | 0.429 | 0.253 | 0.206 | 0.302 |
| q10 top1 | oracle | 0.368 | 0.363 | 1.000 | 0.363 |
| q25 top1 | source init + offset prior | 0.429 | 0.527 | 0.402 | 0.626 |
| q25 top1 | oracle | 0.329 | 0.703 | 1.000 | 0.703 |
| q50 top1 | source init + offset prior | 0.429 | 0.527 | 0.555 | 0.659 |
| q50 top1 | projected selected-map ref-vis r0 | 0.467 | 0.489 | 0.295 | 0.643 |
| q50 top1 | projected selected-map ref-vis r4 | 0.548 | 0.247 | 0.073 | 0.626 |
| q50 top1 | oracle | 0.304 | 0.725 | 1.000 | 0.725 |
| q50 top4 | source init + offset prior | 0.323 | 0.604 | 0.326 | 0.714 |
| q50 top4 | oracle | 0.144 | 0.874 | 1.000 | 0.874 |

The candidate generator is now suitable for q-level verification experiments:
top4 q50 has a strong oracle (`0.144m`, basin@5 `0.874`) and a reasonable
source-prior baseline (`0.323m`, top1 `0.604`). The current projected
selected-map verifier still fails on this deployable protocol. A reference
visibility filter and smaller local search radius improve q50 top1 over the
unfiltered `r4` variant, but it remains below the source-prior baseline. The
`/hy-tmp/Cambridge_stdloc/OldHospital/sparse/0` camera model should not be used
with the current `model_train` track JSONL because the COLMAP point ids and
coordinates come from different reconstructions. Protocol C's next blocker is
selected 3D map evidence and visibility/occlusion scoring, not candidate
generation.

The projected-grid CLI now checks this provenance automatically when a sibling
`*_summary.json` exists for the track-observation JSONL. If the requested
`--camera_model_dir` differs from the observation export's `model_dir`, the
tool raises instead of silently mixing COLMAP reconstructions.

Projected-map score rows now carry diagnostic evidence fields:
`mean_similarity`, `inlier_fraction`, `match_count`, and
`visibility_fraction`; CLI reports also include an `evidence_summary`. For
OldHospital q50 top1 `ref-vis r0`, the summary is:

| evidence field | value |
| --- | ---: |
| empty evidence fraction | 0.376 |
| mean visibility fraction | 0.539 |
| mean match count | 136.4 |
| mean inlier fraction | 0.525 |
| mean similarity | 0.069 |

This makes the next failure mode measurable: many candidates have no projected
reference-visible evidence, and the remaining evidence is too weak to overrule
the init prior safely.

OldHospital real retrieval top20 is now label-joined by `(query_id,
retrieval_rank)`, preserving deployable `reference_image` records from
`oldhospital_real_retrieval_top20_fixed` and pose labels from
`oldhospital_real_retrieval_scoretable_fixed`. This produces 2,912 labeled
candidates over 182 queries. Reference-pose trained dense selectors do not
transfer to this deployment protocol:

| method | pred_m | top1 | Spearman | basin@5 |
| --- | ---: | ---: | ---: | ---: |
| retrieval order | 0.429 | 0.527 | 0.246 | 0.709 |
| candidate prior / POFD score | 0.335 | 0.577 | 0.506 | 0.764 |
| dense selector16 seed0 | 0.510 | 0.473 | 0.103 | 0.720 |
| dense selector16 seed1 | 0.457 | 0.500 | 0.104 | 0.709 |
| dense selector16 seed2 | 0.489 | 0.473 | 0.087 | 0.709 |

This is a strict negative transfer result. The real-retrieval claim should
remain downgraded until selector training includes real-retrieval candidates or
a candidate-prior + visual-evidence fusion scorer that beats metadata-only.

The same conclusion now holds on hard-case slices derived from the labeled
OldHospital real-retrieval table:

| subset | method | pred_m | top1 | Spearman | basin@5 | gap_m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| retrieval-top1-wrong | retrieval order | 0.630 | 0.000 | 0.129 | 0.611 | 0.474 |
| retrieval-top1-wrong | candidate prior | 0.446 | 0.278 | 0.446 | 0.796 | 0.291 |
| retrieval-top1-wrong | dense selector seed1 | 0.663 | 0.148 | -0.034 | 0.611 | 0.508 |
| PnP-high-score-wrong | retrieval order | 0.475 | 0.450 | 0.224 | 0.664 | 0.309 |
| PnP-high-score-wrong | candidate prior | 0.370 | 0.503 | 0.489 | 0.732 | 0.204 |
| PnP-high-score-wrong | dense selector seed1 | 0.498 | 0.416 | 0.077 | 0.664 | 0.332 |

Hard-slice artifacts are under `output/vfm/reports/hard_slices/`. They are
diagnostic evidence, not a positive claim: current dense selectors do not
reduce real-retrieval hard false accepts versus candidate prior.

Projected selected-map scoring has also been tested directly on OldHospital
real retrieval top20, using the seed1 selected track bank and query-camera
projection:

| method | pred_m | top1 | Spearman | basin@5 | basin@10 | gap_m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| retrieval order | 0.429 | 0.527 | 0.246 | 0.709 | 0.802 | 0.276 |
| candidate prior | 0.335 | 0.577 | 0.507 | 0.764 | 0.808 | 0.182 |
| dense selector seed1 | 0.457 | 0.500 | 0.104 | 0.709 | 0.775 | 0.303 |
| projected selected map seed1 r4 | 0.721 | 0.407 | 0.040 | 0.643 | 0.769 | 0.568 |

Paired bootstrap against retrieval order confirms this is a negative result:
`pred_cost_m` delta `+0.292m [0.109, 0.527]`, Spearman delta `-0.206
[-0.278, -0.134]`, and basin@5 delta `-0.066 [-0.115, -0.017]`.
Against candidate prior the degradation is larger. The projected selected-map
path remains a controlled-lattice sanity check, not a real-retrieval verifier.

A query-wise score fusion baseline has now been added for real retrieval. It
keeps metadata-only prior and visual evidence explicitly separated by fusing
aligned score-row tables after per-query normalization. This is exposed as
`feature_extract.tools.vfm.fuse_score_rows`, with `candidate_id` or
`query_rank` alignment so rows like `score:000` and `retrieval:000` can be
compared safely. `feature_extract.tools.vfm.fit_fusion_weight` calibrates the
fusion weight on held-out queries, supports prefix/random/explicit query-list
splits, and writes query-set hashes plus optional split query lists for
reproducibility.

OldHospital real retrieval top20, full 182-query set. This table is an
exploratory fixed-alpha sweep, not the promotion protocol:

| method | pred_m | top1 | Spearman | basin@5 | gap_m |
| --- | ---: | ---: | ---: | ---: | ---: |
| candidate prior | 0.335 | 0.577 | 0.506 | 0.764 | 0.182 |
| dense selector16 seed0 | 0.510 | 0.473 | 0.103 | 0.720 | 0.356 |
| prior + projected rendered a=0.25 | 0.333 | 0.593 | 0.488 | 0.753 | 0.180 |
| prior + dense selector16 seed0 a=0.10 | 0.328 | 0.599 | 0.499 | 0.769 | 0.174 |

Held-out split calibration gives a more conservative signal. Alpha is selected
only on the calibration split by `pred_cost_m`; the rows below are evaluation
queries only.

| split | alpha | eval queries | prior pred/top1/Spearman/basin@5 | fused pred/top1/Spearman/basin@5 |
| --- | ---: | ---: | --- | --- |
| seq4 -> seq8 | 0.10 | 126 | 0.386 / 0.540 / 0.446 / 0.706 | 0.376 / 0.563 / 0.438 / 0.706 |
| seq8 -> seq4 | 0.10 | 56 | 0.220 / 0.661 / 0.642 / 0.893 | 0.219 / 0.679 / 0.636 / 0.911 |

Paired bootstrap against candidate prior on these held-out splits:

| split | pred_m delta | top1 delta | Spearman delta | basin@5 delta |
| --- | ---: | ---: | ---: | ---: |
| seq4 -> seq8 | -0.010 [-0.039, 0.017] | +0.024 [0.000, 0.056] | -0.008 [-0.019, 0.003] | +0.000 [0.000, 0.000] |
| seq8 -> seq4 | -0.002 [-0.011, 0.007] | +0.018 [0.000, 0.054] | -0.006 [-0.017, 0.006] | +0.018 [0.000, 0.054] |

The conservative random query split grid (`alpha in {0, 0.02, 0.05, 0.08,
0.10}`, five seeds) selects `[0.05, 0.10, 0.08, 0.02, 0.02]` and has mean
held-out deltas: `pred_m -0.001`, `top1 +0.015`, `Spearman -0.002`,
`basin@5 +0.004`. A wider random grid overfits some splits, so the current
paper-facing claim should stay narrow: small auxiliary verifier gains after
calibration, not a dominant standalone visual ranking result.

The same split-calibrated protocol was applied to projected rendered selected
map evidence (`selector16 seed1 r4`) as the secondary score. It does not pass
real-retrieval promotion: `seq4 -> seq8` selects `alpha=0.05` and is effectively
flat (`pred_m +0.000`, `top1 +0.000`, `Spearman +0.002`, `basin@5 -0.016`);
`seq8 -> seq4` selects `alpha=0.25` but trades a small top1 increase for worse
ranking and basin retention (`pred_m +0.005`, `top1 +0.018`, `Spearman
-0.029`, `basin@5 -0.018`). Current map-conditioned scoring should therefore
remain a controlled-lattice/mapability diagnostic until the projected/rendered
feature construction is improved.

Implementation note: sparse rendered-map evidence now normalizes both rendered
and query token vectors before local matching, so scores are cosine-style and
cannot be dominated by raw feature magnitude. Recomputing OldHospital real
retrieval projected-grid rows with this fix (`selector16 seed1 r4`) did not
improve the result: standalone projected-map scoring is `pred 0.768 / top1
0.352 / Spearman 0.015 / basin@5 0.676`, and held-out fusion remains negative
or mixed (`seq4 -> seq8`: `pred +0.003`, `top1 +0.008`, `Spearman +0.002`,
`basin@5 -0.016`; `seq8 -> seq4`: `pred +0.005`, `top1 +0.000`, `Spearman
-0.024`, `basin@5 -0.018`). This shifts the next map-verifier work from score
normalization to better query-view geometry, visibility, occlusion handling,
and selector/track-bank quality.

Hard slices from the exploratory full-set `a=0.10` fusion:

| subset | method | pred_m | top1 | Spearman | basin@5 | gap_m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| retrieval-top1-wrong | candidate prior | 0.446 | 0.278 | 0.446 | 0.796 | 0.291 |
| retrieval-top1-wrong | prior + dense0 a=0.10 | 0.443 | 0.333 | 0.425 | 0.815 | 0.287 |
| PnP-high-score-wrong | candidate prior | 0.370 | 0.503 | 0.489 | 0.732 | 0.204 |
| PnP-high-score-wrong | prior + dense0 a=0.10 | 0.369 | 0.530 | 0.478 | 0.738 | 0.203 |

Paired bootstrap against candidate prior shows the gain is small and not yet a
promotion result: full-set pred delta `-0.007m [-0.028, 0.012]`, top1 delta
`+0.022 [0.000, 0.044]`, Spearman delta `-0.007 [-0.016, 0.002]`, and basin@5
delta `+0.005 [0.000, 0.016]`. This is the first positive real-retrieval
evidence for selected visual features as an auxiliary verifier, while the
standalone visual scorer remains weaker than candidate prior.

## Full-Data Training And GPU Utilization Note

The current dense selector trainer is not CUDA-bound. It trains a small
selector on pre-extracted dense token NPZ files; RADIO itself is frozen and is
not backpropagated. In full-scene train-query experiments, the runtime is
dominated by CPU-side NPZ reads, sampled-token caching, and Python loops over
query/candidate groups.

Confirmed changes:

- `run_gpu_job_array` now supports repeated GPU slots such as `0,0,1,1`, so
  multiple independent scene/seed jobs can share each GPU.
- dense sampled-token cache is capped at 256 entries to avoid CPU-memory
  blowups when increasing `spatial_samples`.
- `train_dense_selector` has `--diagnostic_query_limit`, which limits
  expensive initial/final train diagnostics without reducing the actual train
  query pool.
- train-query reference-pose banks now exist for all five Cambridge scenes
  under `output/vfm/candidate_banks/*_train_pose_neighbors_top10.jsonl`.

Observed behavior:

- `s2048/b24` with four concurrent jobs filled more memory but remained
  CPU/I/O-bound and was stopped before completion.
- `s512/b16/20-step` full-train smoke completed only ShopFacade in about
  288s; larger scenes remained dominated by NPZ reads and were stopped.
- sampled-feature preload plus batched query/candidate scoring completed
  split-clean full-train smoke on OldHospital, ShopFacade, KingsCollege, and
  GreatCourt. The same `s512/b16/20-step` job finished all four scenes with
  no failed jobs.

Next engineering step before long dense-selector experiments: precompute
sampled dense training tensors or vectorize batch/candidate scoring so the
training loop moves contiguous tensors to CUDA instead of repeatedly scanning
per-image NPZ files.

The first split-clean pose-nearest train bank was a useful negative control:
loss decreased, but held-out test ranking degraded because the train candidate
generator did not match the retrieval/reference-pose test candidate generator.
The active split-clean candidate generator is now descriptor retrieval:

```text
raw RADIO train descriptors -> top10 descriptor retrieval train candidates
  -> GT pose labels for supervised ranking
  -> dense selector training on train token banks
  -> held-out test reference-pose top10 scoring
```

Held-out reference-pose top10 results for the descriptor-retrieval trained
seed0 dense selector:

| scene | retrieval pred/top1/Spearman/basin@5 | raw RADIO pred/top1/Spearman/basin@5 | split-clean descriptor-retrieval selected pred/top1/Spearman/basin@5 |
| --- | --- | --- | --- |
| OldHospital | `4.171 / 0.495 / 0.316 / 0.687` | `4.397 / 0.538 / 0.222 / 0.709` | `4.066 / 0.593 / 0.316 / 0.736` |
| ShopFacade | `1.635 / 0.806 / 0.396 / 0.893` | `2.093 / 0.718 / 0.192 / 0.903` | `1.945 / 0.709 / 0.374 / 0.913` |
| KingsCollege | `3.431 / 0.726 / 0.321 / 0.872` | `4.603 / 0.647 / 0.021 / 0.846` | `4.109 / 0.749 / 0.269 / 0.898` |
| GreatCourt | `14.061 / 0.317 / 0.166 / 0.505` | `15.770 / 0.239 / -0.006 / 0.437` | `13.985 / 0.301 / 0.168 / 0.496` |

Interpretation: this is positive evidence for the clean split protocol relative
to raw RADIO and the pose-nearest train bank. It is not yet a final main result:
OldHospital beats retrieval on pred/top1/basin, Kings improves top1/basin but
not pred, GreatCourt is near retrieval with better pred/gap but weaker top1,
and ShopFacade improves over raw while remaining below retrieval median/top1.
The next selector objective should add basin/hard-negative weighting and
calibration, then run 5 seeds.

Basin-aware and hard-negative training has now been added to the dense selector
trainer:

```text
listwise pose-cost loss
  + basin BCE on absolute translation/rotation threshold labels
  + max hard-negative margin against the best in-basin candidate
```

This objective is exposed through `train_dense_selector` as
`--basin_bce_weight`, `--hard_negative_weight`,
`--basin_translation_threshold_m`, `--basin_rotation_threshold_deg`, and
`--hard_negative_margin`.

Held-out reference-pose top10 effect for seed0:

| scene | descriptor-retrieval selected | + basin/hard selected | interpretation |
| --- | --- | --- | --- |
| OldHospital | `4.066 / 0.593 / 0.316 / 0.736` | `4.113 / 0.593 / 0.366 / 0.747` | better rank correlation and basin recall, slightly worse pred |
| ShopFacade | `1.945 / 0.709 / 0.374 / 0.913` | `2.157 / 0.738 / 0.283 / 0.913` | top1 improves, median/Spearman regress |
| KingsCollege | `4.109 / 0.749 / 0.269 / 0.898` | `4.021 / 0.711 / 0.277 / 0.892` | pred/gap/Spearman improve, top1/basin regress |
| GreatCourt pose-valid | `13.985 / 0.301 / 0.168 / 0.496` | `13.221 / 0.330 / 0.190 / 0.522` | clean positive gain over retrieval and raw |

Values are `pred / top1 / Spearman / basin@5`. GreatCourt train poses contain
one corrupted 3e9-meter camera center (`seq5/frame00297.png`), so the clean
GreatCourt row uses a descriptor-retrieval train bank built with
`--max_abs_pose_center 1000000`. The unfiltered GreatCourt run is diagnostic
only and must not be used as a main result.

Full split-clean descriptor-retrieval training is now complete for four
Cambridge scenes with 5 seeds, `spatial_samples=768`, `batch_size=64`, and the
basin/hard-negative objective. This is the current clean full-data selector
result for scenes with both train/test RADIO token banks available:

| scene | selected64 5-seed mean pred/top1/Spearman/basin@5/gap | comparison |
| --- | --- | --- |
| OldHospital | `4.049±0.157 / 0.532±0.023 / 0.343±0.026 / 0.736±0.007 / 1.421±0.157` | beats retrieval on pred, basin@5, and gap; top1 below the seed0 small run |
| ShopFacade | `2.150±0.140 / 0.726±0.029 / 0.248±0.032 / 0.917±0.009 / 1.236±0.140` | beats raw RADIO and basin@5 retrieval; still below retrieval order on pred/top1/Spearman |
| KingsCollege | `4.053±0.151 / 0.731±0.009 / 0.289±0.008 / 0.892±0.004 / 1.873±0.151` | beats raw RADIO and retrieval on top1/basin@5; pred remains worse than retrieval |
| GreatCourt | `13.695±0.478 / 0.350±0.007 / 0.216±0.004 / 0.522±0.008 / 7.855±0.478` | beats raw RADIO and retrieval on pred/top1/Spearman/basin@5/gap |

This table uses held-out test queries and fixed reference-pose top10 candidate
banks; it is not a GT-centered q-lattice protocol. The current conclusion is
positive but conservative: selected features provide localization evidence
beyond raw RADIO, and they beat retrieval order on OldHospital and GreatCourt,
but ShopFacade and KingsCollege still show that retrieval prior remains a
strong baseline. Main-paper claims should therefore be framed as evidence
selection and hypothesis verification, not universal retrieval replacement.

The corresponding multi-GPU artifact summaries are:

- `output/vfm/job_arrays/train_dense_trainretrieval_seeds0_4_s768_b64_basin_hard_interleaved_summary.json`
- `output/vfm/job_arrays/build_selected_descriptors_trainretrieval_seeds0_4_s768_b64_basinhard_summary.json`
- `output/vfm/job_arrays/score_selected_descriptors_trainretrieval_seeds0_4_s768_b64_basinhard_summary.json`
- `output/vfm/reports/{scene}_reference_pose_top10_splitclean_trainretrieval_s768_b64_basinhard_selected64_5seed_summary.json`

All three summaries report `failed_count=0`.

Resource note from this run:

- two RTX 3090s were used through the job-array launcher;
- peak observed GPU memory reached about 22-23GB during GreatCourt training;
- GPU utilization was low for long stretches because the trainer repeatedly
  reads full-resolution dense token NPZ files and samples features on CPU before
  each CUDA training stage;
- selected-descriptor cache generation has the same I/O bottleneck;
- scoring on cached selected descriptors is fast, with all 20 reports generated
  in roughly one to two seconds per job.

The next engineering optimization is now explicit: promote sampled dense
training tensors or selector-projected descriptors to persistent, reusable
artifacts so 5-seed experiments no longer rescan dense token banks for every
seed and split.

That optimization has started. `train_dense_selector` now supports:

- `--sampling_seed`: decouples spatial token sampling from the optimization
  seed, so multiple training seeds can share the same sampled token positions;
- `--write_sample_cache`: writes the sampled dense training features used by a
  run into a persistent NPZ cache;
- `--sample_cache`: loads a persistent sampled-feature cache before training
  and avoids full dense NPZ reads when all required entries are present.
- `--sample_cache_dtype`: stores persistent cache arrays as `float16` by
  default, with `float32` available when exact full-precision cache artifacts
  are needed.

There is also a standalone cache builder:

```text
python -m feature_extract.tools.vfm.build_dense_sample_cache \
  --bank <train_candidate_bank.jsonl> \
  --query_manifest <train_manifest.json> \
  --map_manifest <train_manifest.json> \
  --spatial_samples 768 \
  --sampling_seed 0 \
  --output <sample_cache.npz> \
  --summary <sample_cache_summary.json>
```

Because full-scene `s768` caches can still be several GB even in fp16, cache
builds should be planned per scene and only after confirming disk headroom.

Repeated-seed report aggregation is also now a first-class CLI:

```text
python -m feature_extract.tools.vfm.summarize_seed_reports \
  --label <experiment_label> \
  --reports <seed0_report.json> ... <seed4_report.json> \
  --output_json <summary.json> \
  --output_md <summary.md>
```

The generated summary records mean/std/min/max for each metric plus input file
hashes, so 5-seed tables are reproducible rather than hand-copied.

Paired query-level significance is now also a first-class CLI:

```text
python -m feature_extract.tools.vfm.compare_score_rows \
  --method_rows <method_rows.json> \
  --baseline_rows <baseline_rows.json> \
  --label <method_vs_baseline> \
  --resamples 10000 \
  --output_json <paired_stats.json> \
  --output_md <paired_stats.md>
```

It aligns rows by `query_id`, recomputes per-query ranking metrics, reports
paired bootstrap confidence intervals, McNemar tests for binary success
metrics, and Wilcoxon signed-rank diagnostics for continuous metrics. Current
paired statistics for the 4-scene 5-seed selected64 runs are stored in
`output/vfm/reports/paired_stats/`.

Descriptor-level causality controls are now generated for the full split-clean
`selected64`, `s768/b64`, basin/hard-negative protocol over four Cambridge
scenes and five seeds. Each control preserves candidate labels and image ids
while corrupting only the visual evidence path. Artifacts are under
`output/vfm/reports/controls/`; paired bootstrap comparisons are under
`output/vfm/reports/paired_stats/*_vs_{query_shuffle,map_shuffle,wrong_scene_map}.*`.

| scene | control | control pred | control top1 | control Spearman | control basin@5 | selected-control pred delta | top1 delta | Spearman delta | basin@5 delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | query shuffle | 6.694 | 0.232 | -0.035 | 0.597 | -2.645 | +0.300 | +0.378 | +0.140 |
| OldHospital | map shuffle | 5.703 | 0.310 | 0.005 | 0.657 | -1.654 | +0.222 | +0.338 | +0.079 |
| OldHospital | wrong-scene map | 5.708 | 0.311 | -0.003 | 0.652 | -1.659 | +0.221 | +0.345 | +0.085 |
| ShopFacade | query shuffle | 2.752 | 0.456 | -0.015 | 0.870 | -0.602 | +0.270 | +0.263 | +0.047 |
| ShopFacade | map shuffle | 2.438 | 0.470 | 0.003 | 0.918 | -0.288 | +0.256 | +0.245 | -0.002 |
| ShopFacade | wrong-scene map | 2.487 | 0.470 | 0.004 | 0.862 | -0.337 | +0.256 | +0.244 | +0.054 |
| KingsCollege | query shuffle | 5.354 | 0.457 | 0.013 | 0.833 | -1.301 | +0.274 | +0.275 | +0.059 |
| KingsCollege | map shuffle | 4.619 | 0.511 | 0.001 | 0.847 | -0.566 | +0.220 | +0.288 | +0.044 |
| KingsCollege | wrong-scene map | 4.656 | 0.529 | 0.009 | 0.845 | -0.603 | +0.202 | +0.280 | +0.047 |
| GreatCourt | query shuffle | 16.993 | 0.174 | -0.015 | 0.461 | -3.299 | +0.176 | +0.231 | +0.061 |
| GreatCourt | map shuffle | 16.003 | 0.217 | -0.002 | 0.497 | -2.309 | +0.133 | +0.218 | +0.024 |
| GreatCourt | wrong-scene map | 16.445 | 0.213 | -0.002 | 0.488 | -2.750 | +0.137 | +0.217 | +0.033 |

Interpretation: the 2D selected-descriptor evidence path now passes the basic
causality sanity check. Query shuffle, map shuffle, and wrong-scene map
features drive Spearman close to zero and reduce top1 across all four scenes.
This supports the claim that the selected feature scorer is using query-map
visual correspondence rather than only candidate ordering. It does not solve
the still-negative real-retrieval and rendered-map transfer results.

Utility-channel counterfactual masking is also implemented in the descriptor
control CLI. It uses the dense selector checkpoint's absolute
`utility_head.weight` as a selected-channel attribution signal, then removes
the top or bottom utility channels from both query and map descriptors. This is
currently a negative interpretability result:

| scene | mask fraction | high-utility pred/top1/Spearman delta | low-utility pred/top1/Spearman delta | status |
| --- | ---: | --- | --- | --- |
| OldHospital | 25% | `-0.168 / +0.001 / +0.013` | `-0.105 / -0.002 / +0.014` | high/low not separated |
| ShopFacade | 25% | `-0.013 / +0.012 / +0.011` | `-0.080 / +0.014 / +0.014` | high/low not separated |
| KingsCollege | 25% | `+0.044 / +0.001 / +0.001` | `-0.007 / +0.010 / +0.008` | high/low not separated |
| GreatCourt | 25% | `-0.158 / +0.004 / +0.013` | `-0.133 / -0.003 / +0.009` | high/low not separated |
| OldHospital | 50% | `-0.077 / +0.014 / +0.014` | `-0.101 / +0.008 / +0.020` | high/low not separated |
| ShopFacade | 50% | `+0.072 / +0.017 / -0.010` | `-0.152 / +0.016 / +0.030` | high/low not separated |
| KingsCollege | 50% | `-0.083 / +0.008 / +0.009` | `-0.065 / +0.024 / +0.034` | high/low not separated |
| GreatCourt | 50% | `-0.007 / +0.015 / +0.017` | `-0.446 / +0.002 / +0.022` | high/low not separated |

The correct interpretation is that selected descriptor evidence is causal at
the query-map matching level, but the current utility head is not a sufficient
channel-level explanation. A paper claim about interpretable high-utility
channels still needs a better attribution objective, group sparsity, or direct
track/rendered-map utility supervision.

The dense selector now also supports `utility_weighted_pooling`, so the spatial
utility head participates in descriptor pooling and receives ranking gradients.
This is exposed through `train_dense_selector`,
`score_candidate_bank_dense_selector`, and `build_selected_descriptor_bank`.
A seed0 utility-weighted selected16 smoke was run on OldHospital and
ShopFacade from the existing split-clean seed0 checkpoints:

| scene | method | pred_m | top1 | Spearman | basin@5 | interpretation |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| OldHospital | selected64 seed0 | 3.881 | 0.527 | 0.369 | 0.747 | stronger baseline |
| OldHospital | utility-weighted selected16 seed0 | 4.063 | 0.538 | 0.333 | 0.736 | mixed, Spearman lower |
| ShopFacade | selected64 seed0 | 2.113 | 0.738 | 0.296 | 0.922 | stronger baseline |
| ShopFacade | utility-weighted selected16 seed0 | 2.255 | 0.748 | 0.244 | 0.922 | mixed, Spearman lower |

Training diagnostics show utility-weighted training can improve internal
train/eval top1 over raw descriptors, but held-out reference-pose ranking does
not improve over the existing selected64 baseline. Utility mask controls on
this utility-weighted seed0 checkpoint show limited top1 separation, but still
not a robust pred/Spearman attribution result:

| scene | mask fraction | high-utility pred/top1/Spearman delta | low-utility pred/top1/Spearman delta |
| --- | ---: | --- | --- |
| OldHospital | 25% | `-0.238 / +0.016 / -0.006` | `-0.183 / +0.005 / +0.012` |
| OldHospital | 50% | `-0.154 / +0.055 / +0.015` | `-0.165 / -0.016 / +0.012` |
| ShopFacade | 25% | `+0.034 / +0.019 / +0.001` | `-0.016 / -0.029 / +0.016` |
| ShopFacade | 50% | `+0.033 / +0.087 / +0.018` | `-0.031 / +0.000 / +0.008` |

Conclusion: utility-weighted pooling is now correctly wired and test-covered,
but it is not yet a paper-positive utility attribution objective. The next
useful direction is not more seed0 fine-tuning; it is an explicit spatial or
track/rendered-map utility supervision term.

Spatial utility masking is now implemented in the selected-descriptor cache
builder. It computes the selector's utility map, removes the highest- or
lowest-utility spatial token positions before pooling, and writes normal
descriptor-bank artifacts. This closes the missing Gate-2 control interface for
`mask high-utility spatial regions` and `mask low-utility spatial regions`.

Seed0 utility-weighted selected16 spatial mask control at 50% removal:

| scene | mask removed | masked pred | masked top1 | masked Spearman | selected-masked pred delta | top1 delta | Spearman delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | high 50% | 1.909 | 0.709 | 0.331 | +0.346 | +0.039 | -0.087 |
| ShopFacade | low 50% | 2.316 | 0.738 | 0.157 | -0.061 | +0.010 | +0.087 |
| OldHospital | high 50% | 4.458 | 0.505 | 0.199 | -0.395 | +0.033 | +0.134 |
| OldHospital | low 50% | 4.186 | 0.599 | 0.322 | -0.123 | -0.060 | +0.011 |

Interpretation: OldHospital shows the desired direction for high-utility
spatial removal on pred and Spearman, but ShopFacade is inverted on pred and
Spearman. The spatial counterfactual interface is now present, but the utility
map itself is not yet a robust cross-scene explanation. This reinforces the
next training requirement: supervise utility with explicit track consistency,
rendered-map inliers, or geometry-visible evidence instead of relying on
descriptor-ranking gradients alone.

That explicit track-supervised path is now implemented as
`feature_extract.tools.vfm.train_track_utility_selector`. It consumes COLMAP
track observations plus raw VFM token manifests, samples token features at
track observations, and trains `LocalizableFeatureSelector` with:

```text
same-track consistency
+ track-ID contrastive classification
+ geometry utility regression
+ group sparsity
```

On synthetic track data the trainer learns both track separation and utility
targets. On real 20k-observation OldHospital and ShopFacade COLMAP track
smokes, it gives a real positive mapability-training signal but not yet a final
mapability win over PCA:

| scene | init loss | final loss | pos sim init/final | neg sim init/final | utility corr init/final |
| --- | ---: | ---: | --- | --- | --- |
| OldHospital | 4.171 | 1.816 | `0.836 / 0.746` | `0.114 / 0.026` | `-0.021 / 0.020` |
| ShopFacade | 3.398 | 1.575 | `0.794 / 0.744` | `0.055 / 0.023` | `0.013 / 0.059` |

The learned track-supervised checkpoint was also lifted back into selected
track banks on the same 20k observations:

| scene | method | dim | variance | separability | note |
| --- | --- | ---: | ---: | ---: | --- |
| OldHospital | PCA64 L2 | 64 | 0.001953 | 665.9 | stronger same-dim mapability baseline |
| OldHospital | track-utility selector64 | 64 | 0.003094 | 402.8 | improves learned track separation, below PCA64 |
| ShopFacade | PCA64 L2 | 64 | 0.001760 | 746.3 | stronger same-dim mapability baseline |
| ShopFacade | track-utility selector64 | 64 | 0.003060 | 408.2 | improves learned track separation, below PCA64 |

The track objective has now been combined with the successful split-clean dense
ranking objective. The dense selector trainer accepts optional COLMAP track
observations and a warm-start anchor:

```text
listwise + basin/hard-negative ranking
  + lambda_track * same-track consistency
  + lambda_anchor * warm-start parameter anchor
```

The first contrastive joint run was too aggressive: it lowered inter-track
similarity but degraded selected-track mapability. The current best seed0 smoke
uses consistency-only track supervision (`track_contrastive_weight=0`,
`track_consistency_weight=10`, `track_supervision_weight=0.05`,
`init_anchor_weight=50`).

Held-out reference-pose top10, using the same `5m/10deg` basin label as the
existing selected64 reports:

| scene | method | pred_m | top1 | Spearman | basin@5 | gap_m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital | selected64 seed0 | 3.881 | 0.527 | 0.369 | 0.747 | 1.253 |
| OldHospital | joint consistency anchor | 3.843 | 0.533 | 0.373 | 0.753 | 1.215 |
| ShopFacade | selected64 seed0 | 2.113 | 0.738 | 0.296 | 0.922 | 1.199 |
| ShopFacade | joint consistency anchor | 2.149 | 0.757 | 0.255 | 0.922 | 1.235 |

Selected 3D track-bank mapability on the same 20k COLMAP observations:

| scene | method | variance | separability | note |
| --- | --- | ---: | ---: | --- |
| OldHospital | selected64 seed0 | 0.002075 | 586.4 | warm-start baseline |
| OldHospital | joint consistency anchor | 0.002009 | 601.4 | improves selected64, still below PCA64 |
| OldHospital | PCA64 L2 | 0.001953 | 665.9 | stronger mapability baseline |
| ShopFacade | selected64 seed0 | 0.002507 | 494.6 | warm-start baseline |
| ShopFacade | joint consistency anchor | 0.002480 | 501.0 | improves selected64, still below PCA64 |
| ShopFacade | PCA64 L2 | 0.001760 | 746.3 | stronger mapability baseline |

Interpretation: the joint consistency-anchor path is the first positive
evidence that the dense localization selector can be nudged toward explicit 3D
track mapability without losing solver-free evidence. It is not yet a promotion
result because PCA64 remains stronger on mapability and ShopFacade loses
pred/gap/Spearman while improving top1 and track variance.

## Raw VFM 3D Landmark Feature Aggregation Reset

We paused the selector/rendered-map/refinement branch and implemented a simpler
first-stage protocol: raw high-dimensional VFM token observations are sampled at
COLMAP track coordinates and aggregated directly into explicit 3D landmark
features. This follows the ULF-Loc/OpenGaFF-style premise that the first
artifact should be an unbiased 3D VFM landmark bank before any learned selector
or feature field is introduced.

Implemented artifacts:

- `feature_extract/vfm/landmark_feature_aggregation.py`
- `feature_extract/tools/vfm/build_raw_vfm_landmark_bank.py`
- `tests/test_vfm_landmark_feature_aggregation.py`

Supported aggregation methods:

- `mean`
- `random_observation`
- `geometry_weighted`
- `robust_trimmed_mean`
- `view_consistent`

The first real 20k-observation banks were generated for OldHospital and
ShopFacade under `output/vfm/raw_landmark_banks/`. Features are raw RADIO token
vectors with per-observation L2 normalization before aggregation.

| scene | method | utility mode | tracks | obs/track | variance | split cos | heldout R@1 | heldout R@5 | mean rank |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | mean | uniform | 3244 | 6.17 | 0.0000916 | 0.9460 | 0.269 | 0.652 | 9.14 |
| OldHospital | geometry_weighted | inv-reproj + center | 3244 | 6.17 | 0.0000876 | 0.9458 | 0.267 | 0.651 | 9.23 |
| OldHospital | robust_trimmed_mean | uniform | 3244 | 5.38 | 0.0000802 | 0.9434 | 0.260 | 0.629 | 9.86 |
| OldHospital | view_consistent | utility-weighted top views | 3244 | 3.72 | 0.0000646 | 0.9397 | 0.246 | 0.610 | 11.57 |
| ShopFacade | mean | uniform | 2877 | 6.95 | 0.0001090 | 0.9403 | 0.386 | 0.799 | 5.21 |
| ShopFacade | geometry_weighted | inv-reproj + center | 2877 | 6.95 | 0.0001046 | 0.9405 | 0.386 | 0.795 | 5.19 |
| ShopFacade | robust_trimmed_mean | uniform | 2877 | 6.01 | 0.0000954 | 0.9365 | 0.380 | 0.788 | 5.45 |
| ShopFacade | view_consistent | utility-weighted top views | 2877 | 3.63 | 0.0000743 | 0.9283 | 0.357 | 0.753 | 5.97 |

Immediate conclusion: the raw 3D VFM landmark feature bank is now buildable and
diagnosable. Mean aggregation is the strongest held-out observation retrieval
baseline on the current head20k tracks. Robust/view-consistent aggregation lowers
within-track variance, but it also drops held-out retrieval, so the first-stage
metric must report both stability/variance and retrieval retention. The original
`inverse_reprojection` utility was point-level in the exported COLMAP JSONL and
therefore degenerated to mean inside each track; the builder now exposes
`inverse_reprojection_center` as an explicit observation-level geometry proxy.

Next first-stage work:

1. Cache sampled raw VFM track observations so multiple aggregation methods do
   not repeatedly scan token NPZ files.
2. Add true camera-pose geometry weights from COLMAP image extrinsics: viewing
   angle, depth, and baseline diversity. The current center prior is only a
   deterministic geometry proxy.
3. Run 100k-observation banks after the sampled-observation cache exists.
4. Add query-image-to-3D-landmark retrieval as the next bridge toward
   localization, still without selector or refinement.

### Stage A v1: Bilinear Sampling + Camera View Metadata

We then upgraded the raw landmark bank builder toward the full Stage A design:

- COLMAP export now records per-observation `camera_center` and `viewing_ray`.
- raw VFM sampling supports `sample_mode=bilinear`.
- observation utility supports `view_consistency` and
  `inverse_reprojection_center_view`.
- aggregation supports `cosine_weighted_mean`, `geometric_median`, and `medoid`.

ULF-Loc reference point: their public implementation builds normalized dense
feature maps with bilinear interpolation and uses geometry/view consistency when
matching/fusing features. We adapt that idea to sparse SfM landmarks instead of
Gaussian landmarks.

Head20k Stage A v1 results:

| scene | method | sample | utility | tracks | variance | split cos | heldout R@1 | heldout R@5 | mean rank |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | mean | bilinear | inv-reproj + center + view | 3244 | 0.0000660 | 0.9626 | 0.351 | 0.737 | 7.08 |
| OldHospital | geometry_weighted | bilinear | inv-reproj + center + view | 3244 | 0.0000617 | 0.9638 | 0.351 | 0.737 | 7.02 |
| OldHospital | geometric_median | bilinear | inv-reproj + center + view | 3244 | 0.0000782 | 0.9503 | 0.330 | 0.692 | 8.80 |
| OldHospital | medoid | bilinear | inv-reproj + center + view | 3244 | 0.0001099 | 0.9250 | 0.288 | 0.636 | 11.36 |
| ShopFacade | mean | bilinear | inv-reproj + center + view | 2877 | 0.0000807 | 0.9578 | 0.476 | 0.865 | 3.73 |
| ShopFacade | geometry_weighted | bilinear | inv-reproj + center + view | 2877 | 0.0000757 | 0.9590 | 0.479 | 0.869 | 3.64 |
| ShopFacade | geometric_median | bilinear | inv-reproj + center + view | 2877 | 0.0000958 | 0.9450 | 0.448 | 0.837 | 4.48 |
| ShopFacade | medoid | bilinear | inv-reproj + center + view | 2877 | 0.0001363 | 0.9104 | 0.406 | 0.793 | 5.25 |

Interpretation: Stage A v1 is a clear improvement over the nearest-token v0
bank. The main gain comes from bilinear VFM sampling, with small additional
benefit from geometry/view weighting. Weighted mean/cosine-space mean is the
current best baseline. Geometric median and medoid are not better on head20k;
they are retained as robustness controls, not the default.

### Sampled Observation Cache

The Stage A builder now supports a reusable sampled-observation cache:

- `--write_sampled_observation_cache path.npz` writes sampled raw VFM
  observations after COLMAP-to-token bilinear sampling.
- `--sampled_observation_cache path.npz` builds landmark banks directly from the
  cached observations, without rescanning token NPZ files.

OldHospital head20k validation:

- direct mean bank vs cached mean bank: max feature diff `0.0`
- direct geometry-weighted bank vs cached geometry-weighted bank: max feature
  diff `0.0`
- cached geometry-weighted metrics match the direct Stage A v1 result:
  R@1 `0.350654`, R@5 `0.737255`, split cosine `0.963793`

This removes the main CPU/IO blocker for sweeping aggregation methods and for
running 100k-observation banks.

### Head100k Stage A v1 Banks

Using the sampled-observation cache, we generated head100k bilinear-view raw
landmark banks for OldHospital and ShopFacade:

| scene | method | sampled obs | tracks | obs/track | variance | split cos | heldout R@1 | heldout R@5 | mean rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | mean | 100000 | 21337 | 4.69 | 0.0000808 | 0.9494 | 0.373 | 0.713 | 12.75 |
| OldHospital | geometry_weighted | 100000 | 21337 | 4.69 | 0.0000767 | 0.9490 | 0.370 | 0.710 | 13.09 |
| ShopFacade | mean | 100000 | 20577 | 4.86 | 0.0000843 | 0.9426 | 0.294 | 0.676 | 13.67 |
| ShopFacade | geometry_weighted | 100000 | 20577 | 4.86 | 0.0000791 | 0.9433 | 0.292 | 0.679 | 13.71 |

Interpretation: head100k substantially improves 3D landmark coverage while
keeping split stability high. Geometry-weighting consistently lowers
within-track variance, but the retrieval difference is small and not uniformly
positive. For now, the default unbiased Stage A bank should remain bilinear
mean/cosine mean, with geometry-weighted reported as a variance-oriented
alternative rather than the primary retrieval baseline.

### Stage A v1 Visual Validation and Cleanup

The current Stage A raw VFM landmark bank now has qualitative artifacts in
`output/vfm/visualizations/raw_landmark_banks/`:

- `oldhospital_head100k_mean_pca_color.ply`
- `oldhospital_head100k_mean_variance_color.ply`
- `oldhospital_head100k_mean_pca_topdown.png`
- `shopfacade_head100k_mean_pca_color.ply`
- `shopfacade_head100k_mean_variance_color.ply`
- `shopfacade_head100k_mean_pca_topdown.png`

The PCA-color point clouds are non-empty and spatially structured. ShopFacade
forms a cleaner elongated facade-like geometry, while OldHospital contains a
dense central structure plus sparse far/outlier tracks. This is acceptable for a
first unbiased SfM-landmark VFM bank, but the OldHospital tail should be pruned
or down-weighted before using the bank as a final localization map.

Compared with the ULF-Loc-style first-stage target, the current implementation
matches the core requirements: clean SfM 2D-3D associations, bilinear dense VFM
sampling, L2-normalized features, multi-view aggregation, view/camera metadata,
and robust aggregation alternatives. It does not yet implement a full
ULF-Loc-equivalent local geometry verifier or learned localization model, so the
claim should remain "Stage A raw 3D VFM landmark feature bank is ready for
downstream query-to-3D evaluation", not "localization solved".

Old unused outputs from previous mainlines and non-current token banks were
removed. Current retained output footprint is approximately:

| path | size |
| --- | ---: |
| `output/` | 27G |
| `output/vfm_tokens_radio/` | 26G |
| `output/vfm/` | 1.9G |
| `output/vfm/raw_landmark_banks/` | 1.6G |

Disk headroom increased from about 24G available to about 104G available.

## Stage B: Query-to-3D Raw VFM Matching Baseline

The next validation step is now implemented as a solver-free descriptor
baseline followed by fixed PnP-RANSAC:

1. load query dense VFM token grid
2. restrict the 3D landmark map to a coarse reference-image visibility submap
3. match query tokens to raw 3D landmark VFM features with cosine top-K
4. apply ratio test, optional mutual-nearest filtering, and optional landmark
   variance filtering
5. estimate pose with PnP-RANSAC
6. report feature precision, hard false-match rate, PnP inlier ratio, and pose
   success thresholds

Code:

- `feature_extract/vfm/query_to_3d_matching.py`
- `feature_extract/tools/vfm/eval_query_to_3d_vfm_matching.py`

Current full-test results using head100k bilinear-mean raw landmark banks:

| scene | submap | filter | precision | false match | PnP success | 10cm/5deg | 25cm/10deg | med t | med r |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | ref top10 | ratio 0.95 | 0.215 | 0.785 | 0.956 | 0.000 | 0.088 | 1.613m | 2.439deg |
| OldHospital | ref top10 | mutual + ratio 0.80 | 0.414 | 0.586 | 0.934 | 0.022 | 0.088 | 1.480m | 2.302deg |
| ShopFacade | ref top5 | ratio 0.95 | 0.282 | 0.718 | 1.000 | 0.126 | 0.456 | 0.281m | 1.056deg |
| ShopFacade | ref top5 | mutual + ratio 0.80 | 0.534 | 0.466 | 1.000 | 0.117 | 0.485 | 0.260m | 0.981deg |

OldHospital is strongly split-dependent. With ref top10 + ratio 0.95:

| subset | queries | precision | 25cm/10deg | med t |
| --- | ---: | ---: | ---: | ---: |
| seq4 | 56 | 0.046 | 0.000 | 5.741m |
| seq8 | 126 | 0.290 | 0.127 | 1.014m |

Interpretation: raw VFM landmark features do produce a real query-to-3D
localization signal under a coarse submap, especially on ShopFacade. However,
false matches are still high. Mutual-nearest and stricter ratio filtering
improve feature precision but do not reliably solve OldHospital, so the next
step should focus on local geometry/visibility filtering and descriptor
selection, not only PnP tuning.

### OldHospital Coverage-Balanced Stage A/B Update

The initial OldHospital head100k bank had a severe coverage bias: `seq4`
queries often had only tens of bank landmarks visible from their top10
reference images, even though full COLMAP visibility contained thousands of
tracks. This was caused by prefix-limited sampled observations being reused as
the visibility source.

Fixes now in place:

- exported full COLMAP visibility to
  `output/vfm/colmap_tracks/OldHospital/model_train_full_visibility_min2_v1.npz`
- added coverage-balanced COLMAP observation sampling
- rebuilt OldHospital balanced300k raw VFM landmark bank
- Stage B now reports `full_visible_tracks`, `bank_visible_tracks`,
  `bank_visibility_coverage`, and `projected_landmarks` per query
- Stage B submaps can use full visibility rather than sampled-bank visibility

OldHospital full-test comparison:

| bank | filter | landmarks | bank coverage | projected | precision | false match | PnP success | 10cm/5deg | 25cm/10deg | med t | med r |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| head100k | ratio 0.95 | 21337 | n/a | n/a | 0.215 | 0.785 | 0.956 | 0.000 | 0.088 | 1.613m | 2.439deg |
| balanced300k | ratio 0.95 | 108561 | 0.692 | 6049 | 0.282 | 0.718 | 1.000 | 0.005 | 0.060 | 0.809m | 1.528deg |
| balanced300k | mutual + ratio 0.80 | 108561 | 0.692 | 6049 | 0.488 | 0.512 | 0.995 | 0.011 | 0.093 | 0.829m | 1.438deg |

The balanced bank fixes the sparse-render failure. For example,
`seq4/frame00019.png` now has 8847 submap landmarks and 6719 projected
landmarks, with PnP success and 0.625m translation error. This is much better
than the previous PnP failure, but still not accurate localization. The
remaining bottleneck is now descriptor ambiguity / local geometric consistency,
not missing 3D map coverage.

New visualizations:

- `output/vfm/visualizations/query_to_3d_matches/oldhospital_balanced300k_projection_rgb_draw60_contact_sheet.png`
- `output/vfm/visualizations/query_to_3d_matches/oldhospital_balanced300k_projection_feature_draw60_contact_sheet.png`

### Raw VFM Retrieval Full Validation

We also tested whether raw VFM features can provide the coarse sparse retrieval
stage, so Stage B does not depend on a GT-centered or manually provided initial
pose. This protocol uses only query/reference `radio_final` token banks to build
global descriptors and retrieve reference images. Cambridge poses are attached
after retrieval only to report pose recall; they are not used to generate the
candidate order.

New retrieval utilities:

- `feature_extract/vfm/retrieval_report.py`
- `feature_extract/tools/vfm/report_descriptor_retrieval.py`
- `feature_extract/tools/vfm/build_token_descriptor_bank.py`
  now supports `mean`, signed `gem`, and optional per-token L2 before pooling.

Raw VFM retrieval artifacts:

- descriptors: `output/vfm/raw_vfm_retrieval/descriptors/`
- candidate banks: `output/vfm/raw_vfm_retrieval/candidates/`
- retrieval reports: `output/vfm/raw_vfm_retrieval/reports/`

Retrieval-only full-test results:

| scene | descriptor | topK | top1 med t | oracle med t | R@1 1m/10deg | R@20 1m/10deg | R@50 1m/10deg | R@50 2m/20deg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | mean | 20 | 4.350m | 2.488m | 0.060 | 0.159 | n/a | n/a |
| OldHospital | signed-GeM | 20 | 4.346m | 2.547m | 0.082 | 0.159 | n/a | n/a |
| OldHospital | tokenL2+mean | 20 | 4.426m | 2.488m | 0.060 | 0.159 | n/a | n/a |
| OldHospital | mean | 50 | 4.350m | 2.061m | 0.060 | 0.159 | 0.181 | 0.478 |
| ShopFacade | mean | 20 | 2.019m | 0.736m | 0.175 | 0.369 | n/a | n/a |
| ShopFacade | signed-GeM | 20 | 2.036m | 0.779m | 0.204 | 0.359 | n/a | n/a |
| ShopFacade | tokenL2+mean | 20 | 2.019m | 0.740m | 0.175 | 0.369 | n/a | n/a |
| ShopFacade | mean | 50 | 2.019m | 0.655m | 0.175 | 0.369 | 0.379 | 0.874 |

Interpretation: raw global VFM retrieval is a weak replacement for a standard
image retrieval frontend. Mean, signed-GeM, and token-normalized mean are nearly
identical on these two scenes, so the bottleneck is not a simple pooling choice.
OldHospital remains especially weak: only 18.1% of queries have a 1m/10deg
reference within top50. ShopFacade has poor fine recall but usable broad
2m/20deg coverage in top50.

We then used the best mean raw-VFM retrieval candidate banks as Stage B submaps:

| scene | raw retrieval submap | bank coverage | projected | precision | PnP success | 10cm/5deg | 25cm/10deg | med t | med r |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | top20 | 0.665 | 9074.7 | 0.293 | 1.000 | 0.011 | 0.088 | 0.787m | 1.446deg |
| OldHospital | top50 | 0.647 | 16111.3 | 0.301 | 1.000 | 0.016 | 0.104 | 0.789m | 1.308deg |
| ShopFacade | top20 | 0.437 | 6403.6 | 0.317 | 1.000 | 0.165 | 0.544 | 0.226m | 0.896deg |
| ShopFacade | top50 | 0.460 | 11145.6 | 0.331 | 1.000 | 0.165 | 0.524 | 0.222m | 0.676deg |

Conclusion: raw VFM features can drive a complete sparse-retrieval + 2D-3D PnP
baseline without a pose init, but not at a competitive Cambridge localization
level. The positive signal is clearest on ShopFacade, where raw retrieval top20
plus raw 3D VFM matching reaches `54.4%` at `25cm/10deg` and `0.226m` median
translation. OldHospital still fails the expected fine-localization bar even
after coverage-balanced map construction. The next retrieval direction should be
regional/local aggregation or a standard retrieval frontend for coarse submaps,
while raw VFM 3D matching remains a downstream verifier rather than the sole
place-recognition engine.

### Gaussian VFM Field From 3D Landmark Features

We implemented a first Gaussian VFM field path whose purpose is to render dense
VFM feature maps from already-aggregated 3D VFM landmarks. This is deliberately
not a learned feature-Gaussian optimization and not full ray-contribution
attribution. It is the conservative ULF-Loc-style first step:

1. load a trained 3DGS/2DGS Gaussian PLY
2. load the raw 3D VFM landmark bank plus COLMAP track xyz
3. associate each Gaussian to nearby 3D VFM landmarks with KDTree radius search
4. keep only feature-bearing Gaussians with enough landmark support
5. render a dense feature map using a deterministic soft z-buffer splat renderer

Code:

- `feature_extract/vfm/gaussian_vfm_field.py`
- `feature_extract/tools/vfm/build_gaussian_vfm_field.py`
- `feature_extract/tools/vfm/render_gaussian_vfm_feature_map.py`
- `feature_extract/tools/vfm/visualize_gaussian_vfm_render.py`

OldHospital smoke artifacts:

- field:
  `output/vfm/gaussian_vfm_fields/OldHospital/oldhospital_gaussian_vfm_field_landmark_assoc_80k_r010.npz`
- render:
  `output/vfm/gaussian_vfm_fields/OldHospital/renders/seq9_frame00001_vfm_160x90.npz`
- PCA visualization:
  `output/vfm/gaussian_vfm_fields/OldHospital/renders/seq9_frame00001_vfm_160x90_pca.png`

Smoke metrics:

| input | value |
| --- | ---: |
| source Gaussians | 80,000 |
| feature-bearing Gaussians | 44,100 |
| coverage | 0.551 |
| feature dim | 1280 |
| mean landmark support | 2.49 |
| mean association distance | 0.071m |
| render size | 160x90 |
| visible rendered pixels | 7,822 |
| visible fraction | 0.543 |

The rendered feature map has shape `[1280, 90, 160]`; visible pixels are
L2-normalized. This gives the project a concrete dense-rendered VFM feature
artifact for later query-vs-rendered-feature matching. The next engineering step
is to replace the simplified splat renderer with a gsplat contribution-aware
renderer or export `_loc_feature` into the existing Gaussian renderer, then
evaluate query/render feature consistency against raw query VFM maps.

Follow-up implementation:

- added `export_gaussian_vfm_field_to_ply`
- added `feature_extract/tools/vfm/export_gaussian_vfm_field_ply.py`
- added gsplat-backed rendering via `render_gaussian_vfm_feature_map_gsplat`
- `render_gaussian_vfm_feature_map.py` now supports `--renderer soft|gsplat`

Exported OldHospital feature PLY:

- `output/vfm/gaussian_vfm_fields/OldHospital/oldhospital_gaussian_vfm_field_landmark_assoc_80k_r010_loc.ply`

The exported PLY keeps all `504,352` source Gaussian rows and appends
`loc_0 ... loc_1279`. Unassigned Gaussians receive zero features; the `44,100`
feature-bearing rows keep their associated VFM features. It is loadable by the
existing `GaussianFeatureModel.load_ply_with_features` path:

| field | value |
| --- | ---: |
| Gaussians | 504,352 |
| feature dim | 1280 |
| loc tensor shape | `[504352, 1280]` |
| detected geometry | 2DGS |

gsplat render smoke:

| renderer | size | visible pixels | visible fraction | mean alpha/weight |
| --- | --- | ---: | ---: | ---: |
| soft z-buffer splat | 160x90 | 7,822 | 0.543 | 0.195 |
| gsplat | 160x90 | 9,275 | 0.644 | 0.685 |

New gsplat artifacts:

- `output/vfm/gaussian_vfm_fields/OldHospital/renders/seq9_frame00001_vfm_160x90_gsplat.npz`
- `output/vfm/gaussian_vfm_fields/OldHospital/renders/seq9_frame00001_vfm_160x90_gsplat_pca.png`

The field can now be consumed in two ways: directly through the project NPZ
field renderer, or through an existing Gaussian PLY loader that expects per-row
`loc_*` attributes. The remaining limitation is that association is still
nearest-landmark based, not ray-contribution based; this should be treated as
the first reliable feature-bearing Gaussian baseline.

Hole diagnosis and full-radius sweep:

| source | visible fraction on `seq9/frame00001` |
| --- | ---: |
| feature field, first 80k, r=0.10 | 0.644 |
| all source Gaussian alpha, first 80k | 0.809 |
| all source Gaussian alpha, full 504k | 1.000 |

The holes are therefore primarily feature-assignment holes, not 3DGS geometry
holes. The initial field covered only 44,100 / 80,000 Gaussians.

We then ran full 504k landmark-to-Gaussian association with wider radii:

| radius | feature Gaussians | coverage | mean support | mean distance | gsplat visible fraction |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.10m | 241,352 | 0.479 | 2.34 | 0.072m | 0.743 |
| 0.15m | 342,441 | 0.679 | 3.01 | 0.099m | 0.774 |
| 0.20m | 395,775 | 0.785 | 3.38 | 0.118m | 0.795 |

Wider radius clearly improves feature-bearing coverage, but the rendered visible
fraction saturates near 0.80 on this view. `r=0.20m` is the best current
landmark-association field for dense rendering, but it may mix features across
nearby surfaces more than `r=0.10m`; downstream query/render matching should
compare both coverage and descriptor precision.

Ray-contribution aggregation first version:

- added `GaussianVFMRayContributionConfig`
- added `GaussianVFMFeatureView`
- added `aggregate_ray_contributed_gaussian_vfm_features`
- added `feature_extract/tools/vfm/build_ray_contributed_gaussian_vfm_field.py`

This first version works on the VFM token grid. It projects Gaussians into each
reference feature map, assigns each token to a dominant near-depth Gaussian
within a pixel radius, and averages the token features per Gaussian.

OldHospital ray-contribution smoke on first 80k Gaussians:

| views | selection | radius | min samples | feature Gaussians | coverage | mean samples | gsplat visible fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | prefix | 1px | 2 | 15,934 | 0.199 | 11.26 | n/a |
| 64 | prefix | 2px | 1 | 16,910 | 0.211 | 23.46 | n/a |
| 64 | uniform | 2px | 1 | 25,400 | 0.318 | 18.38 | 0.803 |

The ray-contribution field has lower global Gaussian coverage than landmark
association on the same first 80k Gaussians, but its rendered coverage on the
tested view is slightly higher than the full 504k landmark field at `r=0.20m`
(`0.803` vs `0.795`). Interpretation: ray contribution is a useful
view-conditioned hole-filling mechanism, but the current token-grid CPU
implementation is not yet a replacement for global landmark association. The
next version should combine both fields and move the dominant-contribution pass
to GPU / gsplat metadata rather than manual token-grid splatting.

Hybrid Gaussian VFM field:

- added `merge_gaussian_vfm_fields`
- added `feature_extract/tools/vfm/build_hybrid_gaussian_vfm_field.py`
- merge policy: landmark-associated field is primary; ray-contribution field
  only fills Gaussian indices missing from the primary field

OldHospital hybrid artifact:

- `output/vfm/gaussian_vfm_fields/OldHospital/hybrid/oldhospital_gaussian_vfm_hybrid_fullr020_ray80k_uniform64.npz`
- PCA render visualization:
  `output/vfm/gaussian_vfm_fields/OldHospital/hybrid/seq9_frame00001_vfm_160x90_hybrid_fullr020_ray80k_uniform64_gsplat_pca.png`

Hybrid composition:

| component | count |
| --- | ---: |
| primary landmark field, full r=0.20 | 395,775 |
| fallback ray field, 80k uniform64 | 25,400 |
| overlap | 21,641 |
| fallback added | 3,759 |
| hybrid total | 399,534 |

Render coverage on `seq9/frame00001`, `160x90`, gsplat:

| field | visible fraction | visible pixels | mean alpha/weight |
| --- | ---: | ---: | ---: |
| landmark full r=0.10 | 0.743 | 10,701 | n/a |
| landmark full r=0.15 | 0.774 | 11,141 | n/a |
| landmark full r=0.20 | 0.795 | 11,453 | n/a |
| ray 80k uniform64 | 0.803 | 11,561 | n/a |
| hybrid full r=0.20 + ray fill | 0.839 | 12,075 | 0.891 |

This validates the intended behavior: the ray field adds relatively few global
Gaussians, but they sit in view-critical holes and raise rendered dense feature
coverage. The next evaluation should measure query-vs-rendered VFM matching
precision, not only coverage, because `r=0.20m` and ray fill can both introduce
feature contamination if they bridge nearby but distinct surfaces.

### Query-to-Map Correspondence Audit

We then tested the direct question: after aggregating raw VFM features into a
3D map, do query VFM tokens form geometrically meaningful correspondences to
the map?

New code:

- `feature_extract/vfm/query_to_render_matching.py`
- `feature_extract/tools/vfm/eval_query_to_render_vfm_matching.py`
- `feature_extract/tools/vfm/visualize_query_to_render_vfm_matches.py`
- sparse landmark matching, reprojection audit, spatial degeneracy audit, and
  quality filtering in `feature_extract/vfm/query_to_3d_matching.py`
- sparse evaluation/visualization CLIs now auto-load COLMAP intrinsics from
  `<scene>/sparse/0/cameras.bin` when `--camera_model_dir` is not specified.

The evaluator now reports correspondence quality separately from final PnP:

- GT reprojection precision at 5/10/16/32 px
- mean/median/p90/p95 GT reprojection error
- PnP-inlier GT precision at 16 px
- all-match and PnP-inlier spatial distribution/degeneracy
- PnP reprojection residual
- per-match source/quality fields for bucket analysis
- final pose error after PnP-RANSAC

Current main path:

Dense render is no longer the main Stage B path. It remains only as an earlier
diagnostic artifact. The current baseline is sparse SfM landmark features:
query VFM token -> sparse 3D VFM landmark -> PnP-RANSAC.

Important protocol correction:

The initial audit accidentally used the fallback camera
`SIMPLE_RADIAL 1024x576 f=883 cx=512 cy=288`, while OldHospital raw images and
COLMAP cameras are `1920x1080` with `f ~= 1660-1673`, `cx=960`, `cy=540`. The
RADIO token grid is `[68,120]`, so the correct token-to-image scale is about
`16.1px`, not `8.6px`. The earlier numbers are kept only as a legacy diagnostic
because they explain why the visual correspondence looked worse than the
reported GT@16 suggested.

OldHospital reference-pose top1, first 20 test queries, `query_token_step=8`,
using the legacy fallback camera:

| method | mean matches | GT precision@5px | GT precision@16px | median match reproj | PnP-inlier GT@16px | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse 3D landmarks | 95.95 | 0.055 | 0.240 | 49.56px | 0.812 | 0.625m | 1.659deg |
| hybrid dense render | 28.75 | 0.083 | 0.357 | 32.47px | 0.707 | 1.261m | 2.324deg |

The same audit using COLMAP camera intrinsics from
`/hy-tmp/Cambridge_stdloc/OldHospital/sparse/0`:

| method | mean matches | GT precision@5px | GT precision@16px | GT precision@32px | median match reproj | PnP-inlier GT@16px | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| sparse 3D landmarks | 95.95 | 0.028 | 0.137 | 0.249 | 91.99px | 0.495 | 1.261m | 3.019deg |
| hybrid dense render | 30.25 | 0.023 | 0.175 | 0.341 | 58.51px | 0.381 | 2.069m | 5.617deg |

Sparse landmark filtering:

The sparse matcher now supports an interpretable landmark quality path:

`Q_X = w_track log(N_X) - w_var Var_X - w_reproj reproj_err_X + w_idf IDF_X - w_amb ambiguity_X`

where `ambiguity_X` is estimated as scene-level nearest-neighbor feature
ambiguity and is precomputed on the full landmark index before submap slicing.
The matcher can use:

- hard thresholds on track length, feature variance, reprojection error,
  scene-level ambiguity, boundary distance, and similarity margin
- optional `similarity * Q_X` weighted ranking
- optional MNN
- per-match logging of `landmark_quality`, `landmark_ambiguity`,
  `quality_weighted_similarity`, `landmark_reprojection_error`, and margin

OldHospital reference-pose top1, sparse only, COLMAP intrinsics:

| split/config | mean matches | GT@5 | GT@16 | GT@32 | median match reproj | PnP-inlier GT@16 | PnP success | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q20 baseline | 95.95 | 0.028 | 0.137 | 0.249 | 91.99px | 0.495 | 1.000 | 1.261m | 3.019deg |
| q20 balanced filters | 32.90 | 0.053 | 0.251 | 0.433 | 36.34px | 0.513 | 1.000 | 1.251m | 2.835deg |
| q20 strict filters | 20.40 | 0.063 | 0.327 | 0.504 | 32.37px | 0.596 | 0.950 | 1.247m | 2.486deg |
| full182 baseline | 93.16 | 0.018 | 0.104 | 0.210 | 109.85px | 0.469 | 0.852 | 3.616m | 6.307deg |
| full182 balanced filters | 23.62 | 0.039 | 0.218 | 0.409 | 41.65px | 0.445 | 0.890 | 4.266m | 7.383deg |

Patch-level sparse matching:

The new Stage B sparse baseline treats each VFM token as a patch-level proposal
instead of a pixel-accurate keypoint. For evaluation, a query token is correct
if the predicted 3D landmark belongs to the visible landmark set whose GT
projection falls inside that token patch:

`P(q) = {X_j | pi(T_gt X_j) in patch(q), visible(X_j)=1}`

New code:

- `feature_extract/vfm/patch_to_3d_matching.py`
- `feature_extract/tools/vfm/eval_patch_to_3d_vfm_matching.py`

The patch evaluator reports:

- `Patch@1`, `Patch@K`
- `GT@stride`, `GT@2stride`
- legacy `GT@5px`, `GT@16px`, and median reprojection error
- PnP-inlier `Patch@1` and PnP-inlier `GT@stride`
- positive-set difficulty: visible landmark count, positives per token,
  non-empty patch fraction, and zero-positive token ratio
- PnP solve rate separated from localization success at `10cm/5deg`,
  `25cm/10deg`, `50cm/10deg`, and `1m/10deg`
- PnP-inlier count, inlier ratio, spatial coverage, depth range, and 3D
  degeneracy statistics
- final pose using a stride-aware PnP-RANSAC threshold

OldHospital reference-pose top1, sparse patch baseline, balanced landmark
filters, COLMAP intrinsics:

Legacy q20 ablation:

| split/config | matches | Patch@1 | Patch@5 | GT@5 | GT@16 | GT@stride | GT@2stride | solve rate | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q20 patch NN, 1.5 stride PnP | 32.90 | 0.113 | 0.113 | 0.053 | 0.251 | 0.251 | 0.435 | 1.000 | 0.847m | 2.042deg |
| q20 patch MNN, 1.5 stride PnP | 31.25 | 0.118 | 0.118 | 0.055 | 0.262 | 0.262 | 0.447 | 1.000 | 1.191m | 2.680deg |

Updated soft-mutual top5 no-margin protocol with `max_matches=1000`:

| split/submap | landmarks | visible | Patch@1 | Patch@5 | GT@5 | GT@stride | GT@2stride | inliers | inlier Patch@1 | solve | S@10cm/5deg | S@25cm/10deg | S@50cm/10deg | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| q20 GT-visible oracle | 20000 | 20000 | 0.180 | 0.246 | 0.067 | 0.367 | 0.629 | 518.7 | 0.345 | 1.000 | 0.000 | 0.500 | 1.000 | 0.241m | 0.428deg |
| q20 reference top1 | 1813 | 1578 | 0.169 | 0.221 | 0.063 | 0.352 | 0.608 | 504.0 | 0.331 | 1.000 | 0.100 | 0.500 | 0.900 | 0.247m | 0.530deg |
| q20 reference top5 | 3386 | 2942 | 0.191 | 0.250 | 0.074 | 0.380 | 0.633 | 532.4 | 0.353 | 1.000 | 0.100 | 0.450 | 1.000 | 0.259m | 0.438deg |
| q20 reference top10 | 5158 | 4455 | 0.197 | 0.260 | 0.076 | 0.389 | 0.640 | 543.5 | 0.362 | 1.000 | 0.050 | 0.550 | 1.000 | 0.242m | 0.523deg |
| full182 GT-visible oracle | 20000 | 20000 | 0.160 | 0.242 | 0.059 | 0.326 | 0.539 | 447.7 | 0.335 | 1.000 | 0.000 | 0.154 | 0.429 | 0.579m | 0.966deg |
| full182 reference top1 | 1564 | 1407 | 0.162 | 0.223 | 0.059 | 0.339 | 0.559 | 467.4 | 0.329 | 1.000 | 0.016 | 0.165 | 0.478 | 0.530m | 0.904deg |
| full182 reference top5 | 4417 | 3849 | 0.188 | 0.263 | 0.071 | 0.374 | 0.590 | 503.9 | 0.357 | 1.000 | 0.016 | 0.137 | 0.484 | 0.522m | 0.875deg |
| full182 reference top10 | 7000 | 6003 | 0.194 | 0.275 | 0.075 | 0.381 | 0.592 | 508.0 | 0.367 | 1.000 | 0.005 | 0.159 | 0.500 | 0.499m | 0.892deg |

Interpretation:

- Patch-level metrics expose signal that point-level `GT@5px` hides. On
  full182 reference top10, `Patch@1=0.194` and `GT@stride=0.381`, while
  strict pixel-level `GT@5=0.075`.
- `pnp_solve_rate` must not be called localization success. The solver returns
  a pose for every full182 query in the updated soft-mutual protocol, but
  actual `success@25cm/10deg` is only `13.7-16.5%`, and
  `success@10cm/5deg` is below `2%`.
- The stride-aware PnP threshold gives a clear downstream improvement compared
  with the old point-level balanced sparse path (`4.266m/7.383deg` median), but
  the result remains a coarse localization signal rather than a precise local
  feature matcher.
- Strict one-to-one MNN is still not the best default. It marginally improves
  correspondence precision on q20 but gives worse pose than NN and soft mutual.
- Soft mutual top5 without margin and a fixed top-1000 match cap lowers
  all-match precision but improves pose, likely because PnP gets better spatial
  coverage and RANSAC can select a useful subset.
- GT-visible oracle is not automatically an upper bound. With a 20k visible
  landmark cap it introduces more true-visible but repetitive/ambiguous
  landmarks; full182 reference top10 is slightly better than full182
  GT-visible in median pose (`0.499m` vs `0.579m`).
- This supports the claim that raw VFM is better modeled as patch-level
  localization evidence than as pixel-level 2D-3D correspondence.

Follow-up ablations before feature selection:

The evaluator now supports `LandmarkQualityConfig` in the patch matcher, so
`similarity * Q_X` can be used as a quality-weighted matching score. It also
has a summary tool:

```text
python -m feature_extract.tools.vfm.summarize_patch_to_3d_ablation
```

which writes JSON/CSV/Markdown comparison tables from per-run summary JSON
files. This keeps Step 2/3/4 reporting reproducible instead of manually
copying metrics.

OldHospital q20 submap split with the same default candidate matcher
`MNN + reference/visible submap + stride-aware PnP`:

| submap | Patch@1 | inlier Patch@1 | S@25cm/10deg | S@50cm/10deg | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| GT-visible oracle | 0.227 | 0.426 | 0.850 | 1.000 | 0.166m | 0.364deg |
| reference top1 | 0.281 | 0.452 | 0.650 | 1.000 | 0.224m | 0.346deg |
| reference top5 | 0.285 | 0.462 | 0.500 | 1.000 | 0.221m | 0.292deg |
| reference top10 | 0.280 | 0.466 | 0.800 | 0.950 | 0.176m | 0.429deg |
| all-map, capped 20k | 0.201 | 0.423 | 0.800 | 1.000 | 0.196m | 0.411deg |

Interpretation: in the q20 slice, the descriptor/matcher is not the only
bottleneck. Coarse submap choice affects success, but the oracle-visible
submap is not overwhelmingly better than top10 because the 20k visible cap
adds many ambiguous landmarks.

OldHospital full182 matching-rule ablation on reference top10:

| matcher | Patch@1 | inlier Patch@1 | S@25cm/10deg | S@50cm/10deg | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNN | 0.258 | 0.437 | 0.209 | 0.577 | 0.435m | 0.664deg |
| soft mutual top3 + margin 0.02 + quality | 0.182 | 0.359 | 0.176 | 0.555 | 0.436m | 0.734deg |
| soft mutual top3 + margin 0.02 | 0.183 | 0.360 | 0.170 | 0.511 | 0.475m | 0.764deg |
| soft mutual top5, no margin | 0.194 | 0.367 | 0.159 | 0.500 | 0.499m | 0.892deg |

ShopFacade full103 matching-rule ablation on reference top10:

| matcher | Patch@1 | inlier Patch@1 | S@25cm/10deg | S@50cm/10deg | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNN | 0.259 | 0.407 | 0.786 | 0.932 | 0.170m | 0.445deg |
| soft mutual top3 + margin 0.02 | 0.190 | 0.330 | 0.786 | 0.942 | 0.172m | 0.552deg |
| soft mutual top5, no margin | 0.157 | 0.287 | 0.631 | 0.854 | 0.205m | 0.570deg |

Current default baseline before feature selection:

```text
reference top10 submap
+ sparse landmark raw VFM map
+ patch-level MNN
+ token-stride-aware PnP-RANSAC
```

This is deliberately conservative. It gives fewer correspondences than soft
mutual topK, but higher patch correctness and better full-split pose on both
OldHospital and ShopFacade.

Top-K reference pool and candidate-prior audit:

The patch evaluator now writes per-query and summary-level candidate-pool
diagnostics:

- `gt_visible_bank_tracks`, `submap_gt_visible_tracks`, and
  `visible_landmark_recall`
- `reference_prior.top1` and `reference_prior.oracle` pose errors from the
  fixed candidate bank
- PnP-inlier patch correctness, inlier count, spatial coverage, depth range,
  and degeneracy metrics in the same summary rows

The same-source HLoc/NetVLAD reference-pose bank only contains top10
candidates, so the clean deployment-like sweep is top1/top3/top5/top10 plus
GT-visible and all-map-capped20k stress tests. A separately generated
`pose_nearest_reference_top20_oracle` bank is included only as an oracle-like
top20 coverage upper bound; it must not be mixed into the same-source curve.

OldHospital full182, sparse raw VFM patch MNN, balanced300k mean bank:

| pool | visible recall med | ref top1 med t | ref oracle med t | matches | inliers | inlier Patch@1 | S@25cm/10deg | S@50cm/10deg | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HLoc-ref top1 | 0.017 | 4.002m | 4.002m | 497.5 | 302.2 | 0.427 | 0.198 | 0.511 | 0.459m | 0.759deg |
| HLoc-ref top3 | 0.031 | 4.002m | 3.173m | 717.0 | 427.4 | 0.435 | 0.214 | 0.566 | 0.445m | 0.646deg |
| HLoc-ref top5 | 0.045 | 4.002m | 2.869m | 814.5 | 482.9 | 0.435 | 0.198 | 0.626 | 0.414m | 0.617deg |
| HLoc-ref top10 | 0.067 | 4.002m | 2.322m | 914.3 | 531.3 | 0.437 | 0.209 | 0.577 | 0.435m | 0.664deg |
| pose-nearest top20 oracle | 0.085 | 1.505m | 1.505m | 948.9 | 558.2 | 0.442 | 0.253 | 0.566 | 0.429m | 0.613deg |
| all-map capped20k | 0.182 | 4.002m | 2.322m | 889.8 | 362.3 | 0.380 | 0.214 | 0.522 | 0.486m | 0.817deg |
| GT-visible capped20k | 0.222 | 4.002m | 2.322m | 901.9 | 422.8 | 0.391 | 0.203 | 0.484 | 0.512m | 0.719deg |

ShopFacade full103, sparse raw VFM patch MNN, head100k mean bank:

| pool | visible recall med | ref top1 med t | ref oracle med t | matches | inliers | inlier Patch@1 | S@25cm/10deg | S@50cm/10deg | med t | med r |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| HLoc-ref top1 | 0.057 | 1.377m | 1.377m | 385.2 | 268.3 | 0.400 | 0.670 | 0.825 | 0.194m | 0.585deg |
| HLoc-ref top3 | 0.131 | 1.377m | 0.972m | 565.3 | 394.5 | 0.409 | 0.709 | 0.903 | 0.191m | 0.499deg |
| HLoc-ref top5 | 0.181 | 1.377m | 0.785m | 616.9 | 431.0 | 0.408 | 0.738 | 0.913 | 0.182m | 0.512deg |
| HLoc-ref top10 | 0.285 | 1.377m | 0.736m | 678.4 | 467.9 | 0.407 | 0.786 | 0.932 | 0.170m | 0.445deg |
| pose-nearest top20 oracle | 0.422 | 0.746m | 0.746m | 698.2 | 481.7 | 0.406 | 0.767 | 0.932 | 0.159m | 0.439deg |
| all-map capped20k | 0.971 | 1.377m | 0.736m | 859.1 | 486.1 | 0.380 | 0.650 | 0.835 | 0.190m | 0.452deg |
| GT-visible | 1.000 | 1.377m | 0.736m | 840.0 | 500.3 | 0.387 | 0.621 | 0.816 | 0.185m | 0.438deg |

Interpretation:

- VFM+PnP is not merely copying reference top1 pose. On OldHospital, HLoc-ref
  top1 prior has median `4.002m`, while VFM+PnP reaches `0.459m`; on
  ShopFacade, `1.377m` becomes `0.194m`.
- Increasing K improves visible-landmark recall, but the pose curve saturates
  early. OldHospital top5 has the best median translation among same-source
  pools, while ShopFacade keeps improving through top10.
- All-map and GT-visible pools do not dominate despite higher coverage; they
  add many ambiguous landmarks and lower PnP-inlier patch correctness. This
  confirms that a coarse candidate pool is still necessary.
- The top20 pose-nearest result is an oracle-like coverage diagnostic, not a
  deployment claim. Its limited gain over top10 shows that raw VFM patch MNN is
  now partly matcher/outlier-limited, not only coverage-limited.

Map feature representation ablation with the fixed MNN baseline, head100k
landmark banks:

| scene | representation | Patch@1 | inlier Patch@1 | S@25cm/10deg | S@50cm/10deg | med t | med r |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital full182 | geometry weighted mean | 0.191 | 0.347 | 0.121 | 0.385 | 0.668m | 1.033deg |
| OldHospital full182 | mean | 0.189 | 0.345 | 0.133 | 0.403 | 0.701m | 0.900deg |
| OldHospital full182 | geometric median | 0.188 | 0.349 | 0.110 | 0.387 | 0.744m | 1.028deg |
| OldHospital full182 | medoid | 0.159 | 0.320 | 0.089 | 0.335 | 0.758m | 1.199deg |
| ShopFacade full103 | geometric median | 0.256 | 0.407 | 0.748 | 0.932 | 0.165m | 0.471deg |
| ShopFacade full103 | mean | 0.259 | 0.407 | 0.786 | 0.932 | 0.170m | 0.445deg |
| ShopFacade full103 | medoid | 0.221 | 0.381 | 0.689 | 0.903 | 0.181m | 0.443deg |
| ShopFacade full103 | geometry weighted mean | 0.262 | 0.410 | 0.777 | 0.922 | 0.185m | 0.463deg |

The robust representations do not provide a clear universal win yet. Mean or
geometry-weighted mean remain the safest default for OldHospital, while
ShopFacade shows a small median-translation gain for geometric median but not a
success-rate gain. Medoid is now evaluated correctly after fixing its
observation-count metadata, but it is not competitive as a default.

Selector-entry gate update:

- enter feature selection only on top of the fixed MNN patch baseline above;
- report MNN and soft-mutual top3+margin as non-selector baselines;
- require selector improvements in at least one of `Patch@1`, PnP-inlier
  Patch@1, `success@25cm/10deg`, or hard-case failure rate without degrading
  `success@50cm/10deg`;
- keep OldHospital and ShopFacade in the baseline table before adding any
  learned selector claim.

Stage C0 non-learned compression baseline:

Stage C0 is now implemented as a leakage-controlled preprocessing step: a
single transform is fit from 3D landmark mean features, then applied to both the
3D landmark bank and the query token bank before running the same sparse
patch-to-3D MNN evaluator. Code:

- `feature_extract/vfm/feature_compression.py`
- `feature_extract/tools/vfm/build_stage_c0_compressed_features.py`
- `feature_extract/tools/vfm/summarize_stage_c0_compression.py`

Implemented methods:

- raw high-D identity baseline
- PCA projection
- seeded Gaussian random projection
- first-channel control
- channel-variance selection
- IDF channel selection
- Fisher channel selection when a supervised label `.npy` is provided

Full artifacts are under `output/vfm/stage_c0_compression/`. The fixed
evaluator protocol is:

```text
reference top10 submap
+ sparse raw VFM landmark map or compressed map
+ patch-level MNN
+ token-stride-aware PnP-RANSAC
```

OldHospital full182, balanced300k mean landmark bank:

| method | dim | S@25cm/10deg | S@50cm/10deg | med t | med r | inlier Patch@1 | storage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 1280 | 0.209 | 0.577 | 0.435m | 0.664deg | 0.437 | 4.19G |
| PCA | 512 | 0.209 | 0.522 | 0.469m | 0.725deg | 0.414 | 2.98G |
| PCA | 256 | 0.176 | 0.516 | 0.490m | 0.756deg | 0.406 | 1.50G |
| PCA | 128 | 0.170 | 0.549 | 0.468m | 0.793deg | 0.388 | 0.75G |
| PCA | 64 | 0.209 | 0.462 | 0.547m | 0.826deg | 0.363 | 0.38G |
| random | 128 | 0.236 | 0.544 | 0.468m | 0.687deg | 0.414 | 0.75G |
| random | 64 | 0.209 | 0.533 | 0.479m | 0.695deg | 0.401 | 0.37G |
| variance | 128 | 0.165 | 0.440 | 0.593m | 0.905deg | 0.367 | 0.51G |
| IDF | 128 | 0.214 | 0.516 | 0.460m | 0.724deg | 0.394 | 0.51G |

ShopFacade full103, head100k mean landmark bank:

| method | dim | S@25cm/10deg | S@50cm/10deg | med t | med r | inlier Patch@1 | storage |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 1280 | 0.786 | 0.932 | 0.170m | 0.445deg | 0.407 | 2.01G |
| PCA | 512 | 0.767 | 0.922 | 0.168m | 0.449deg | 0.386 | 1.55G |
| PCA | 256 | 0.738 | 0.932 | 0.180m | 0.462deg | 0.364 | 0.78G |
| PCA | 128 | 0.641 | 0.874 | 0.195m | 0.541deg | 0.342 | 0.39G |
| PCA | 64 | 0.553 | 0.806 | 0.221m | 0.589deg | 0.306 | 0.19G |
| random | 128 | 0.757 | 0.942 | 0.173m | 0.477deg | 0.386 | 0.39G |
| random | 64 | 0.748 | 0.883 | 0.164m | 0.502deg | 0.372 | 0.19G |
| variance | 128 | 0.680 | 0.835 | 0.204m | 0.558deg | 0.348 | 0.25G |
| IDF | 128 | 0.748 | 0.883 | 0.174m | 0.518deg | 0.363 | 0.25G |

Interpretation:

- PCA-512 is the safest non-learned compression if the goal is minimal quality
  loss, but it still reduces PnP-inlier patch correctness and broad recall on
  OldHospital.
- Random projection is a surprisingly strong low-dimensional baseline:
  random-128 improves OldHospital `S@25cm/10deg` over raw and nearly preserves
  ShopFacade, while using about 18-19% of raw storage.
- Simple channel-variance selection is not competitive. IDF selection is better
  than variance but does not dominate random projection.
- PCA-64 and channel-selection-64 are too aggressive for this matcher. They
  may still be useful as storage/risk controls, but not as default baselines.
- Selector claims must beat random-128 and PCA-512, not only raw or variance
  selection. A good next selector target is random-128 storage with at least
  raw-level `S@50cm/10deg` and improved inlier Patch@1.

Spatial/PnP diagnostics with the corrected camera:

| method | PnP residual median | inlier bbox area | inlier 4x4 grid occupancy | inlier xy PCA minor/major | inlier xyz planarity |
| --- | ---: | ---: | ---: | ---: | ---: |
| sparse 3D landmarks | 5.83px | 0.356 | 0.353 | 0.286 | 0.00229 |
| hybrid dense render | 5.65px | 0.259 | 0.299 | 0.359 | 0.00084 |

Interpretation:

- The user's visual concern is correct. Most accepted raw VFM query-to-map
  correspondences are not geometrically correct at normal local-feature
  thresholds.
- The legacy fallback camera made the correspondence metrics look too
  optimistic. With the correct COLMAP camera, the all-match GT precision and
  PnP-inlier GT precision both drop sharply.
- Dense rendered matching has lower median correspondence error than sparse
  landmark matching, but its PnP inliers are less spatially spread and more
  planar, which explains why final pose can be worse despite some cleaner
  all-match statistics.
- PnP can still produce a plausible pose because RANSAC selects a smaller,
  more geometrically consistent subset from many bad matches. Therefore final
  pose accuracy alone is not sufficient evidence that the VFM correspondences
  are meaningful.
- The current raw VFM query-to-map path should be treated as a diagnostic
  verifier, not as a validated correspondence engine.
- Sparse landmark filtering is effective for correspondence quality: on
  full182, balanced filters roughly double `GT@16` and reduce median GT
  reprojection error from `109.85px` to `41.65px`.
- That improvement does not automatically solve pose. The full182 balanced
  filters slightly increase PnP success but worsen median final pose. This
  indicates that PnP still accepts spatially biased or geometrically weak
  match subsets, so local geometric consistency is the next sparse-stage
  requirement.
- In q20 experiments, direct `similarity * Q_X` re-ranking plus MNN was too
  brittle: it improved some all-match statistics but reduced usable match
  count and PnP success. For now, quality is better used as conservative hard
  filtering and reporting, not as the only ranking signal.
- Patch-level matching changes the downstream picture: using token-stride
  uncertainty in PnP gives positive full182 pose movement even without training
  a selector.

Bucket analysis with the corrected camera:

- Sparse landmark matches are strongly driven by descriptor similarity and
  track quality. The lowest-similarity quartile has `GT@16=0.000`, while the
  highest-similarity quartile reaches `GT@16=0.323`. The lowest feature-variance
  quartile reaches `GT@16=0.190`, compared with `0.092` for the highest
  variance quartile.
- Dense rendered matches are also similarity- and boundary-sensitive. The
  lowest-similarity quartile has `GT@16=0.079`, and middle/high similarity
  buckets reach about `0.22-0.25`. Near-boundary matches are worse
  (`GT@16=0.119`) than central matches (`0.18-0.20`).
- The current approximate alpha-entropy/top1-contribution diagnostics do not
  behave as a reliable rejection rule. Higher entropy and lower top1
  contribution can correlate with better matches, likely because the current
  implementation measures projected Gaussian neighborhood density rather than
  the exact renderer contribution for that pixel.
- Dense depth variance has a weak useful signal: very high ray-depth variance
  reduces PnP-inlier rate and GT@16, but it is not enough as a standalone
  filter.

RGB-render visualizations were added to avoid misleading interpretation from
VFM PCA colors. The right panel is now a 3DGS RGB render at the same candidate
pose, while the lines still show the accepted VFM query-to-render matches:

- good case:
  `output/vfm/visualizations/query_to_render_matches/hybrid_refpose_top1_q20_good_bad_rgb_render/seq8__frame00123.png_query_to_hybrid_render.png`
- bad case:
  `output/vfm/visualizations/query_to_render_matches/hybrid_refpose_top1_q20_good_bad_rgb_render/seq8__frame00119.png_query_to_hybrid_render.png`

Immediate implication for the next method step:

1. Do not optimize or claim only final PnP pose.
2. Add sparse patch-level local geometric consistency before PnP, such as Hough
   voting over token-cell displacement, image-neighborhood-consistent match
   filtering, or submap-relative pose voting.
3. Evaluate any selector/refinement by correspondence precision and
   PnP-inlier GT precision, not only translation/rotation error.
4. Treat VFM descriptors as patch-level semantic/geometric evidence unless a
   method explicitly proves sub-token correspondence quality.
5. Before changing the network, fix protocol plumbing so every OldHospital
   evaluator loads COLMAP camera intrinsics by default and logs token/render
   scale. The default fallback camera should only be used in toy tests.
6. Keep dense render out of the main path until sparse landmark matching has a
   reliable geometric verifier.

## Stage C1: Supervised Linear Patch Selector

Stage C1 is now implemented as the deliberately small learned selector baseline:

```text
raw VFM token / landmark feature
-> bias-free shared Linear(C, D)
-> L2-normalized descriptor
```

The training signal is patch-to-landmark-set supervision, not reference-pose
reranking. For each query token, positives are the GT-visible landmarks whose
projection falls inside the token patch. Negatives are hard same-submap
landmarks selected by raw VFM false-nearest similarity after excluding all
landmarks positive for that patch.

Implemented files:

- `feature_extract/vfm/patch_selector_training.py`
  - multi-positive patch InfoNCE
  - linear selector training and export as `FeatureCompressionTransform`
  - token-limited batched hard-negative mining
  - NPZ sample cache round-trip for repeated seeds without repeated mining
- `feature_extract/tools/vfm/train_stage_c1_patch_selector.py`
  - trains from train query tokens, COLMAP/SfM landmarks, GT train poses,
    fixed candidate submaps and optional visibility index
  - writes a learned transform
  - optionally exports compressed query tokens and a compressed landmark bank
- `feature_extract/tools/vfm/summarize_stage_c1_patch_selector.py`
  - aggregates learned/random/PCA/raw runs into seed mean/std/best tables
- `feature_extract/tools/vfm/summarize_patch_topk_diagnostics.py`
  - aggregates soft-mutual topK ranking diagnostics into scene/method/K tables
- `feature_extract/tools/vfm/summarize_patch_hard_cases.py`
  - builds patch-to-3D hard-case tables and rescue/worsen counts
- `feature_extract/tools/vfm/make_linear_selector_ablation.py`
  - creates top/bottom/random learned-linear channel-group ablations
- `feature_extract/tools/vfm/summarize_patch_map_quality_modes.py`
  - compares feature-only, stats-only, and feature+stats matching modes
- `feature_extract/tools/vfm/summarize_patch_mode_guard.py`
  - evaluates no-GT per-query guarded mode selection policies
- `feature_extract/vfm/patch_to_3d_matching.py`
  - optional `similarity_device=cuda:*` backend for the topK similarity search
    while keeping the same matcher and PnP protocol
  - explicit `match_score_mode` for `similarity`, `landmark_quality`, and
    `similarity_quality`
- `feature_extract/vfm/patch_selector_training.py`
  - optional input-channel group lasso and hard group-gate export
- `tests/test_vfm_stage_c1_patch_selector.py`
  - supervised toy convergence
  - patch-positive/hard-negative sample construction
  - token limit before hard-negative search
  - sample cache reuse
  - CLI transform/bank export smoke
- `tests/test_vfm_stage_c1_summary.py`
  - learned multi-seed summary
  - untrained control summary
- `tests/test_vfm_patch_topk_summary.py`
  - soft-mutual topK diagnostic table smoke

Protocol used for the full C1 held-out check:

- Train split only for selector fitting.
- Held-out test split for patch-to-3D localization evaluation.
- Same Stage C0 evaluator and same fixed `reference top10` submap protocol.
- Same sparse landmark path; no dense render.
- Same matching default: `MNN`, `top_k=1`, `mutual_top_k=1`,
  `min_similarity=0.2`, stride-aware PnP.
- Random controls use seeds `0..4`; seed0 is the Stage C0 run and seeds
  `1..4` are in the C1 final directory.
- Learned controls use seeds `0..4`.
- OldHospital now uses all 895 train-query records to build the sample cache,
  capped at `24000` sampled patch examples.
- ShopFacade uses all available train-query records to build the sample cache,
  capped at `12000` sampled patch examples.

Five-seed C1 summary:

| Scene | Method | Dim | Seeds | S@25 mean/std/best | S@50 mean | Median t mean/best | PnP-inlier Patch@1 mean |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | raw | 1280 | 1 | 0.209 / 0.000 / 0.209 | 0.577 | 0.435m / 0.435m | 0.437 |
| OldHospital | random | 128 | 5 | 0.199 / 0.029 / 0.236 | 0.551 | 0.457m / 0.427m | 0.414 |
| OldHospital | random | 64 | 5 | 0.195 / 0.019 / 0.214 | 0.519 | 0.478m / 0.447m | 0.400 |
| OldHospital | learned | 128 | 5 | 0.290 / 0.012 / 0.302 | 0.652 | 0.367m / 0.341m | 0.511 |
| OldHospital | learned | 64 | 5 | 0.279 / 0.029 / 0.324 | 0.638 | 0.377m / 0.360m | 0.507 |
| ShopFacade | raw | 1280 | 1 | 0.786 / 0.000 / 0.786 | 0.932 | 0.170m / 0.170m | 0.407 |
| ShopFacade | random | 128 | 5 | 0.728 / 0.031 / 0.767 | 0.936 | 0.177m / 0.168m | 0.387 |
| ShopFacade | random | 64 | 5 | 0.724 / 0.029 / 0.767 | 0.909 | 0.166m / 0.159m | 0.371 |
| ShopFacade | learned | 128 | 5 | 0.806 / 0.012 / 0.816 | 0.932 | 0.147m / 0.143m | 0.451 |
| ShopFacade | learned | 64 | 5 | 0.814 / 0.029 / 0.835 | 0.932 | 0.149m / 0.135m | 0.444 |

Full tables:

- `output/vfm/stage_c1_patch_selector_final/stage_c1_summary.md`
- `output/vfm/stage_c1_patch_selector_final/stage_c1_summary.csv`
- `output/vfm/stage_c1_patch_selector_final/stage_c1_summary.json`

Soft-mutual topK diagnostic:

- Same held-out test split and `reference top10` submap protocol.
- Same sparse landmark path; no dense render.
- Representative seed0 runs for `random128`, `learned128`, and `learned64`.
- `top_k == mutual_top_k`, no ratio test, `min_similarity=0.2`,
  `max_matches=2000`, stride-aware PnP.
- The diagnostic measures whether the learned descriptor improves ranking
  distribution, not only MNN nearest-neighbor sharpness.

| Scene | Method | topK | Patch@5 | GT@2stride | PnP-inlier Patch@5 | S@25 | S@50 | Median t |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | raw1280 | 2 | 0.251 | 0.611 | 0.422 | 0.170 | 0.566 | 0.448m |
| OldHospital | random128 | 2 | 0.224 | 0.563 | 0.407 | 0.165 | 0.511 | 0.492m |
| OldHospital | learned128 | 2 | 0.419 | 0.801 | 0.524 | 0.269 | 0.626 | 0.371m |
| OldHospital | learned64 | 2 | 0.410 | 0.795 | 0.516 | 0.291 | 0.621 | 0.388m |
| OldHospital | raw1280 | 3 | 0.264 | 0.601 | 0.429 | 0.198 | 0.549 | 0.472m |
| OldHospital | random128 | 3 | 0.237 | 0.543 | 0.412 | 0.170 | 0.511 | 0.489m |
| OldHospital | learned128 | 3 | 0.437 | 0.795 | 0.538 | 0.286 | 0.599 | 0.419m |
| OldHospital | learned64 | 3 | 0.430 | 0.794 | 0.532 | 0.264 | 0.615 | 0.368m |
| OldHospital | raw1280 | 5 | 0.267 | 0.570 | 0.429 | 0.176 | 0.505 | 0.498m |
| OldHospital | random128 | 5 | 0.244 | 0.502 | 0.418 | 0.143 | 0.440 | 0.571m |
| OldHospital | learned128 | 5 | 0.444 | 0.777 | 0.548 | 0.291 | 0.549 | 0.435m |
| OldHospital | learned64 | 5 | 0.438 | 0.780 | 0.541 | 0.231 | 0.621 | 0.404m |
| ShopFacade | raw1280 | 2 | 0.238 | 0.657 | 0.386 | 0.757 | 0.951 | 0.166m |
| ShopFacade | random128 | 2 | 0.217 | 0.620 | 0.372 | 0.689 | 0.893 | 0.188m |
| ShopFacade | learned128 | 2 | 0.295 | 0.710 | 0.433 | 0.816 | 0.951 | 0.147m |
| ShopFacade | learned64 | 2 | 0.281 | 0.684 | 0.427 | 0.806 | 0.922 | 0.153m |
| ShopFacade | raw1280 | 3 | 0.231 | 0.625 | 0.376 | 0.748 | 0.893 | 0.184m |
| ShopFacade | random128 | 3 | 0.212 | 0.591 | 0.364 | 0.709 | 0.913 | 0.176m |
| ShopFacade | learned128 | 3 | 0.298 | 0.705 | 0.427 | 0.796 | 0.922 | 0.155m |
| ShopFacade | learned64 | 3 | 0.286 | 0.683 | 0.425 | 0.806 | 0.922 | 0.172m |
| ShopFacade | raw1280 | 5 | 0.222 | 0.582 | 0.355 | 0.709 | 0.874 | 0.170m |
| ShopFacade | random128 | 5 | 0.206 | 0.547 | 0.346 | 0.641 | 0.864 | 0.207m |
| ShopFacade | learned128 | 5 | 0.297 | 0.692 | 0.418 | 0.748 | 0.932 | 0.164m |
| ShopFacade | learned64 | 5 | 0.286 | 0.672 | 0.415 | 0.748 | 0.922 | 0.169m |

TopK artifacts:

- `output/vfm/stage_c1_topk_diagnostics/topk_summary.md`
- `output/vfm/stage_c1_topk_diagnostics/topk_summary.json`

TopK conclusion:

- Learned128/64 improve Patch@5, GT@2stride, and PnP-inlier Patch@5 over
  raw1280 and random128 for every reported topK on both scenes. This supports
  the interpretation that C1 improves the descriptor ranking distribution, not
  only the MNN top1 decision.
- Larger K is not uniformly better for final pose. OldHospital learned128 keeps
  S@25 near `0.29` at K=5, but ShopFacade learned128 drops from `0.816` at K=2
  to `0.748` at K=5. The extra matches improve recall-style patch metrics but
  also add outliers; the main protocol should stay conservative until map-side
  quality scoring is added.

Hard-case diagnostic:

- Uses the conservative MNN top1 main protocol.
- Uses representative seed0 for learned/random controls, so it is a diagnostic
  split rather than the final statistical table.
- Subsets are built from the `random128` anchor plus raw rows:
  random failure, raw failure, top1-reference far, weak top10 coverage, low
  visible-landmark count, low random-inlier count, and large random pose error.

Key hard-case rows:

| Scene | Subset | Method | queries | S@25 | S@50 | median t | inlier Patch@1 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital | random128_fail_s25 | random128 | 139 | 0.000 | 0.403 | 0.543m | 0.406 |
| OldHospital | random128_fail_s25 | raw1280 | 139 | 0.079 | 0.482 | 0.520m | 0.428 |
| OldHospital | random128_fail_s25 | learned128 | 139 | 0.209 | 0.525 | 0.484m | 0.505 |
| OldHospital | random128_fail_s25 | learned64 | 139 | 0.173 | 0.518 | 0.487m | 0.497 |
| ShopFacade | random128_fail_s25 | random128 | 25 | 0.000 | 0.760 | 0.371m | 0.377 |
| ShopFacade | random128_fail_s25 | raw1280 | 25 | 0.320 | 0.720 | 0.296m | 0.401 |
| ShopFacade | random128_fail_s25 | learned128 | 25 | 0.200 | 0.720 | 0.338m | 0.444 |
| ShopFacade | random128_fail_s25 | learned64 | 25 | 0.480 | 0.720 | 0.258m | 0.452 |

Random128 failure rescue summary:

| Scene | Method | rescued / random-fail | worsened / random-success |
| --- | --- | ---: | ---: |
| OldHospital | learned128 | 29 / 139 | 20 / 43 |
| OldHospital | learned64 | 24 / 139 | 21 / 43 |
| ShopFacade | learned128 | 5 / 25 | 2 / 78 |
| ShopFacade | learned64 | 12 / 25 | 5 / 78 |

Hard-case conclusion:

- Learned selectors do rescue random128 failures, especially ShopFacade
  learned64 (`12/25`) and OldHospital learned128 (`29/139`).
- The rescue is not free on OldHospital seed0: learned128 worsens `20/43`
  random-success queries. This points to the next required step: add map-side
  quality/statistics scoring and calibrated guarding instead of using feature
  similarity alone for all cases.

Hard-case artifacts:

- `output/vfm/stage_c1_hard_cases/hard_case_summary.md`
- `output/vfm/stage_c1_hard_cases/hard_case_summary.json`

Channel-group causality smoke:

- Selector: `learned128_seed0`.
- Grouping: consecutive raw VFM channel groups of 64 dimensions.
- Ablation: zero 25% of input-channel groups in the learned linear projection,
  then re-export query tokens and landmark bank and run the same MNN top1
  patch-to-3D evaluator.
- Policies: highest weight-energy groups, lowest weight-energy groups, random
  groups.

| Scene | Method | removed energy | S@25 | S@50 | median t | inlier Patch@1 | inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | learned128 base |  | 0.286 | 0.626 | 0.412m | 0.512 | 782.5 |
| OldHospital | top-drop25 | 0.274 | 0.258 | 0.599 | 0.404m | 0.480 | 702.7 |
| OldHospital | bottom-drop25 | 0.228 | 0.253 | 0.610 | 0.396m | 0.485 | 709.7 |
| OldHospital | random-drop25 | 0.249 | 0.291 | 0.610 | 0.381m | 0.488 | 693.7 |
| ShopFacade | learned128 base |  | 0.786 | 0.932 | 0.144m | 0.449 | 565.3 |
| ShopFacade | top-drop25 | 0.267 | 0.777 | 0.932 | 0.165m | 0.418 | 509.4 |
| ShopFacade | bottom-drop25 | 0.237 | 0.845 | 0.951 | 0.147m | 0.424 | 519.0 |
| ShopFacade | random-drop25 | 0.244 | 0.796 | 0.922 | 0.165m | 0.422 | 513.0 |

Causality conclusion:

- Removing high-energy groups consistently hurts correspondence diagnostics
  and inlier count versus the base selector, which is weak positive evidence
  that the learned projection concentrates useful descriptor signal.
- The evidence is not yet strong enough to claim sparse channel selection:
  bottom/random removal also hurts inlier Patch@1, and final pose is not
  monotonic with energy removal. The learned128 linear weights are relatively
  distributed (`top 25%` groups contain only about `27%` of weight energy), so a
  stronger selection claim needs explicit group sparsity or hard gates.

Causality artifacts:

- `output/vfm/stage_c1_causality/causality_summary.md`
- `output/vfm/stage_c1_causality/causality_summary.json`
- `feature_extract/tools/vfm/make_linear_selector_ablation.py`

Map-side quality smoke:

- Selector: `learned128_seed0`.
- Main protocol: MNN top1, `reference top10`, sparse landmarks only.
- Quality terms: track length, feature variance, and COLMAP reprojection error.
- No scene-level ambiguity/IDF in this smoke, to keep the diagnostic cheap.
- Effect: `similarity * Q_X` controls match ordering before the fixed PnP
  handoff.

| Scene | Method | S@25 | S@50 | median t | inlier Patch@1 | inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital | feature only | 0.286 | 0.626 | 0.412m | 0.512 | 782.5 |
| OldHospital | feature + quality | 0.286 | 0.632 | 0.352m | 0.507 | 758.6 |
| ShopFacade | feature only | 0.786 | 0.932 | 0.144m | 0.449 | 565.3 |
| ShopFacade | feature + quality | 0.806 | 0.942 | 0.158m | 0.443 | 536.1 |

Map-quality conclusion:

- Lightweight map statistics improve broad success on both scenes and reduce
  OldHospital median translation, but they slightly reduce inlier Patch@1 and
  inlier count. This is useful as a guarded scorer feature, not yet a replacement
  for descriptor similarity.
- Next implementation should separate `feature-only`, `stats-only`, and
  calibrated `feature+stats` scoring, then evaluate it on the hard-case splits
  above.

Map-quality artifacts:

- `output/vfm/stage_c1_map_quality/map_quality_summary.md`
- `output/vfm/stage_c1_map_quality/map_quality_summary.json`

Map-quality formal mode sweep:

- Protocol: learned128 seed0, sparse landmarks only, `reference top10`.
- Candidate pool: soft-mutual top3, `max_matches=1000`.
- Modes:
  - `feature_only`: rank by descriptor similarity.
  - `stats_only`: rank by landmark quality from track length, feature variance,
    and reprojection error.
  - `feature_stats`: rank by descriptor similarity times landmark quality.

| Scene | Mode | S@25 | S@50 | median t | inlier Patch@1 | inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| OldHospital | feature_only | 0.242 | 0.593 | 0.396m | 0.470 | 771.7 |
| OldHospital | feature_stats | 0.253 | 0.577 | 0.421m | 0.431 | 698.2 |
| OldHospital | stats_only | 0.225 | 0.549 | 0.445m | 0.381 | 570.5 |
| ShopFacade | feature_only | 0.767 | 0.932 | 0.172m | 0.373 | 650.4 |
| ShopFacade | feature_stats | 0.738 | 0.913 | 0.164m | 0.353 | 586.0 |
| ShopFacade | stats_only | 0.796 | 0.922 | 0.151m | 0.322 | 441.5 |

Guarded mode-selection diagnostic:

- No-GT policies: choose per query by PnP inlier count, inlier ratio, or lowest
  PnP inlier median residual.
- `oracle_pose` is diagnostic only and uses GT pose error.

| Scene | Policy | chosen modes | S@25 | S@50 | median t |
| --- | --- | --- | ---: | ---: | ---: |
| OldHospital | inlier_count | feature_only:175,feature_stats:4,stats_only:3 | 0.236 | 0.593 | 0.400m |
| OldHospital | low_residual | feature_only:168,feature_stats:14 | 0.247 | 0.599 | 0.394m |
| OldHospital | oracle_pose | feature_only:69,feature_stats:54,stats_only:59 | 0.396 | 0.720 | 0.301m |
| ShopFacade | inlier_count | feature_only:89,feature_stats:8,stats_only:6 | 0.767 | 0.942 | 0.169m |
| ShopFacade | low_residual | feature_only:64,feature_stats:23,stats_only:16 | 0.835 | 0.961 | 0.158m |
| ShopFacade | oracle_pose | feature_only:29,feature_stats:33,stats_only:41 | 0.903 | 0.961 | 0.104m |

Formal map-quality conclusion:

- A global `feature+stats` ranker is not consistently better. OldHospital mostly
  prefers feature-only, while ShopFacade benefits from stats-only in S@25 and
  median translation.
- The residual-based no-GT guard is promising on ShopFacade (`0.835` S@25) but
  only marginal on OldHospital (`0.247` S@25). The oracle gap is large on both
  scenes, so the next useful model is a calibrated per-query guard, not another
  descriptor-only selector.

Formal map-quality artifacts:

- `output/vfm/stage_c1_map_quality_modes/map_quality_modes_summary.md`
- `output/vfm/stage_c1_map_quality_modes/map_quality_hard_cases.md`
- `output/vfm/stage_c1_map_quality_modes/mode_guard_summary.md`
- `feature_extract/tools/vfm/summarize_patch_map_quality_modes.py`
- `feature_extract/tools/vfm/summarize_patch_mode_guard.py`

## Stage C2 Safe Localizable Descriptor Selection

Implemented C2 infrastructure:

- `feature_extract/vfm/patch_selector_training.py`
  - `ResidualGatedPatchSelector`: LayerNorm -> input-channel group gate ->
    linear projection -> residual bottleneck MLP -> L2 descriptor.
  - query matchability, landmark reliability, and pairwise inlier head.
  - auxiliary pairwise BCE on patch-positive vs hard-negative pairs.
  - hard group-gate export mask for active-group Pareto diagnostics.
  - checkpoint save/load and batched descriptor encoding.
- `feature_extract/tools/vfm/train_stage_c2_safe_selector.py`
  - trains from the existing C1 patch-positive sample cache.
  - exports C2 query token descriptors and C2 landmark bank in the same format
    consumed by the patch-to-3D evaluator.
  - nonlinear landmark-bank variance is currently exported as zero, so
    map-quality variance terms should not be used for this smoke.
- `tests/test_vfm_stage_c2_safe_selector.py`
  - residual-gated descriptor shape/normalization and pairwise head smoke.
  - C2 training loss/top1/inlier-head smoke.
  - checkpoint round-trip.
  - CLI train/export smoke.

Current C2 smoke protocol:

- Sample cache:
  - ShopFacade: `output/vfm/stage_c1_patch_selector_final/shopfacade/samples_ref10_12k_seed0.npz`
  - OldHospital: `output/vfm/stage_c1_patch_selector_final/oldhospital/samples_ref10_24k_seed0.npz`
- Test protocol: same sparse patch-to-3D MNN top1, reference top10, stride-aware
  PnP as the C1 main table.
- C2 runs:
  - `c2_safe128_full_seed0`: residual-gated selector, pairwise auxiliary head,
    100% active groups.
  - `c2_safe128_keep60_seed0`: same model but hard-gated to 60% active channel
    groups at export.

| Scene | Method | S@25 | S@50 | median t | median r | Patch@1 | Inlier Patch@1 | matches | inliers |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | random128 | 0.757 | 0.942 | 0.173m | 0.477deg | 0.230 | 0.386 | 671.3 | 436.6 |
| ShopFacade | C1 learned128 | 0.786 | 0.932 | 0.144m | 0.401deg | 0.316 | 0.449 | 762.2 | 565.3 |
| ShopFacade | C2 full128 | 0.816 | 0.913 | 0.151m | 0.449deg | 0.293 | 0.438 | 758.8 | 540.9 |
| ShopFacade | C2 keep60 | 0.757 | 0.951 | 0.173m | 0.438deg | 0.256 | 0.402 | 706.0 | 481.8 |
| OldHospital | random128 | 0.236 | 0.544 | 0.468m | 0.687deg | 0.226 | 0.414 | 901.8 | 483.7 |
| OldHospital | C1 learned128 | 0.286 | 0.626 | 0.412m | 0.559deg | 0.411 | 0.512 | 1000.0 | 782.5 |
| OldHospital | C2 full128 | 0.253 | 0.632 | 0.384m | 0.605deg | 0.423 | 0.517 | 1000.0 | 798.1 |
| OldHospital | C2 keep60 | 0.231 | 0.621 | 0.387m | 0.566deg | 0.340 | 0.470 | 994.9 | 701.5 |

Hard-case rescue/break versus random128:

| Scene | Method | random-fail queries | rescued | rescue rate | random-success worsened | worsen rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | C1 learned128 | 25 | 5 | 0.200 | 2 | 0.026 |
| ShopFacade | C2 full128 | 25 | 13 | 0.520 | 7 | 0.090 |
| ShopFacade | C2 keep60 | 25 | 11 | 0.440 | 11 | 0.141 |
| OldHospital | C1 learned128 | 139 | 29 | 0.209 | 20 | 0.465 |
| OldHospital | C2 full128 | 139 | 21 | 0.151 | 18 | 0.419 |
| OldHospital | C2 keep60 | 139 | 15 | 0.108 | 16 | 0.372 |

C2 interpretation:

- C2 full128 is useful but not yet a drop-in replacement for C1 learned128.
  It improves ShopFacade S@25 and rescues many more ShopFacade random failures,
  and on OldHospital it improves broad S@50, median translation, Patch@1, and
  PnP-inlier Patch@1. However, OldHospital S@25 drops from `0.286` to `0.253`.
- The 60% active-group hard gate does not pass the proposed C2 Pareto target.
  It can retain broad S@50 on OldHospital and improve ShopFacade S@50, but it
  loses too much S@25 and correspondence precision.
- The pairwise inlier head is currently an auxiliary training signal and stored
  in the checkpoint; the exported patch-to-3D evaluator still ranks matches by
  descriptor cosine. The next C2 step is to use
  `cos(z_q, z_X) + alpha log p_inlier(q, X)` in candidate matching instead of
  only using the head as an auxiliary loss.

C2 artifacts:

- `output/vfm/stage_c2_safe_selector_smoke/shopfacade/c2_safe128_full_seed0/`
- `output/vfm/stage_c2_safe_selector_smoke/shopfacade/c2_safe128_keep60_seed0/`
- `output/vfm/stage_c2_safe_selector_smoke/oldhospital/c2_safe128_full_seed0/`
- `output/vfm/stage_c2_safe_selector_smoke/oldhospital/c2_safe128_keep60_seed0/`
- `output/vfm/stage_c2_safe_selector_smoke/c2_hard_case_summary.md`

Additional C2 completion work:

- Pairwise head scoring is now connected to the patch matcher through
  `match_score_mode=similarity_pairwise`.
  - Candidate generation remains cosine top-K.
  - Candidate score is `cos(z_q, z_X) + alpha * log sigmoid(h(q, X))`.
  - The evaluator accepts `--safe_pairwise_checkpoint`,
    `--pairwise_inlier_weight`, `--pairwise_device`, and
    `--pairwise_batch_size`.
- Added `feature_extract/tools/vfm/export_stage_c2_safe_selector.py`.
  - Re-exports descriptors from one trained C2 checkpoint at arbitrary
    active-group fractions.
  - This makes C2.4 Pareto sweeps cheaper and avoids retraining for every
    sparsity point.

Pairwise scorer alpha sweep:

- Model: C2 full128 seed0.
- Candidate protocol: cosine top3 candidate generation, per-token NN selection
  by `cos + alpha log p_inlier`, sparse patch-to-3D, reference top10.
- This is a scorer diagnostic, not the main protocol, because it changes MNN
  top1 into top3 rescoring.

| Scene | alpha | S@25 | S@50 | median t | median r | Patch@1 | Inlier Patch@1 | matches | inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 0.05 | 0.806 | 0.932 | 0.157m | 0.432deg | 0.220 | 0.350 | 1000.0 | 587.7 |
| ShopFacade | 0.10 | 0.738 | 0.932 | 0.172m | 0.424deg | 0.220 | 0.351 | 1000.0 | 585.4 |
| ShopFacade | 0.20 | 0.699 | 0.903 | 0.159m | 0.399deg | 0.221 | 0.351 | 1000.0 | 587.3 |
| OldHospital | 0.05 | 0.231 | 0.621 | 0.397m | 0.667deg | 0.369 | 0.478 | 1000.0 | 753.9 |
| OldHospital | 0.10 | 0.258 | 0.615 | 0.393m | 0.605deg | 0.371 | 0.479 | 1000.0 | 755.4 |
| OldHospital | 0.20 | 0.264 | 0.599 | 0.392m | 0.591deg | 0.374 | 0.483 | 1000.0 | 758.4 |

Pairwise scorer conclusion:

- Direct top3 pairwise rescoring is not yet safe enough for the default C2
  protocol. It raises some recall-style behavior but sharply lowers Patch@1 on
  ShopFacade, and does not beat the C2 full128 MNN baseline on the main
  success metrics.
- The pairwise head should next be calibrated as a guard/risk feature or used
  with stricter mutual/topK filtering, not simply dropped into every token's
  top3 candidate set.

C2 group-sparsity Pareto from one full checkpoint:

- Model: C2 full128 seed0.
- Export: same checkpoint re-exported at 80%, 60%, and 40% active input-channel
  groups.
- Eval: original MNN top1 protocol, sparse patch-to-3D, reference top10.

| Scene | active groups | S@25 | S@50 | median t | median r | Patch@1 | Inlier Patch@1 | matches | inliers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ShopFacade | 16/20 | 0.816 | 0.903 | 0.146m | 0.437deg | 0.284 | 0.421 | 724.4 | 521.0 |
| ShopFacade | 12/20 | 0.777 | 0.951 | 0.170m | 0.443deg | 0.256 | 0.402 | 706.0 | 482.0 |
| ShopFacade | 8/20 | 0.786 | 0.913 | 0.158m | 0.432deg | 0.244 | 0.402 | 699.7 | 459.3 |
| OldHospital | 16/20 | 0.275 | 0.654 | 0.363m | 0.550deg | 0.394 | 0.498 | 997.9 | 769.1 |
| OldHospital | 12/20 | 0.214 | 0.632 | 0.387m | 0.568deg | 0.340 | 0.470 | 994.9 | 701.4 |
| OldHospital | 8/20 | 0.247 | 0.637 | 0.415m | 0.573deg | 0.293 | 0.451 | 989.5 | 623.8 |

Pareto conclusion:

- 80% active groups is currently the best C2 sparse operating point. It matches
  ShopFacade C2 full128 S@25 and improves OldHospital over full128 on S@25,
  S@50, median translation, and median rotation.
- 60% active groups does not pass the proposed C2 target across scenes. It is
  acceptable on ShopFacade S@25 but loses too much on OldHospital S@25 and
  correspondence precision.
- 40% active groups is a diagnostic compression point, not a paper-ready
  default.

C2 pairwise/Pareto artifacts:

- `output/vfm/stage_c2_pairwise_scoring/`
- `output/vfm/stage_c2_group_pareto/`

Group-gated selector smoke:

- Training entry now supports input-channel group lasso plus hard group pruning.
- Smoke config: learned128, `group_size=64`, keep 50% of groups, lasso `1e-4`,
  seed0, same MNN top1 held-out evaluator.

| Scene | Method | active groups | active channels | gated eval top1 | S@25 | S@50 | median t | inlier Patch@1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | learned128 base |  |  | 0.420 | 0.286 | 0.626 | 0.412m | 0.512 |
| OldHospital | learned128 group-gated | 10 / 20 | 640 / 1280 | 0.290 | 0.253 | 0.604 | 0.389m | 0.462 |
| ShopFacade | learned128 base |  |  | 0.530 | 0.786 | 0.932 | 0.144m | 0.449 |
| ShopFacade | learned128 group-gated | 10 / 20 | 640 / 1280 | 0.436 | 0.777 | 0.893 | 0.169m | 0.414 |

Group-gated conclusion:

- Keeping only half of the raw VFM channel groups preserves a large fraction of
  downstream localization performance, but it clearly drops training-set
  discriminability and PnP-inlier correctness.
- This is now an explicit feature-selection mechanism, but the first smoke is a
  compression/selection tradeoff, not yet a positive causality claim. It should
  be swept over keep ratios and group-lasso weights before being presented as a
  main result.

Group-gated artifacts:

- `output/vfm/stage_c1_group_gate/group_gate_summary.md`
- `output/vfm/stage_c1_group_gate/group_gate_summary.json`

Artifacts:

- `output/vfm/stage_c1_patch_selector_final/shopfacade/learned64_seed{0..4}/`
- `output/vfm/stage_c1_patch_selector_final/shopfacade/learned128_seed{0..4}/`
- `output/vfm/stage_c1_patch_selector_final/shopfacade/random64_seed{1..4}/`
- `output/vfm/stage_c1_patch_selector_final/shopfacade/random128_seed{1..4}/`
- `output/vfm/stage_c1_patch_selector_final/oldhospital/learned64_seed{0..4}/`
- `output/vfm/stage_c1_patch_selector_final/oldhospital/learned128_seed{0..4}/`
- `output/vfm/stage_c1_patch_selector_final/oldhospital/random64_seed{1..4}/`
- `output/vfm/stage_c1_patch_selector_final/oldhospital/random128_seed{1..4}/`

Important caveats:

- This completes C1 for OldHospital and ShopFacade, not for all Cambridge
  scenes. Kings/Great/StMary still require matching raw token/landmark banks
  before they can enter this table.
- Patch@5 in the main pose table still collapses to Patch@1 because the fixed
  default matching protocol is MNN top1. Use the soft-mutual diagnostic table
  above when discussing ranking-quality metrics.
- Training and eval use different fixed candidate-submap sources:
  train uses `*_train_pose_neighbors_top10.jsonl`, held-out eval uses
  `*_reference_pose_top10_fixed.jsonl`. This should be reported explicitly in
  any paper table.
- The C1 compression/export summaries still reuse the Stage C0 export tool, so
  the JSON `stage` field can say `stage_c0_nonlearned_feature_compression` even
  when the input transform is learned. The surrounding C1 train/eval paths are
  the authoritative provenance.
- Learned64/128 beating random mean and improving PnP-inlier Patch@1 on both
  scenes is now the strongest evidence that supervised patch-level selector
  training is solving a descriptor problem rather than only compressing.

## Stage C1 Qualitative Comparisons

Added a reproducible qualitative visualizer:

```bash
PYTHONPATH=. python feature_extract/tools/vfm/visualize_stage_c1_selection_qualitative.py \
  --scene shopfacade \
  --query_id seq3/frame00045.png \
  --query_id seq1/frame00036.png \
  --output_dir output/vfm/stage_c1_qualitative/shopfacade \
  --similarity_device cuda:0
```

The same preset works for `--scene oldhospital`. The generated artifacts cover
three qualitative checks:

- `*_match_comparison_contact_sheet.png`: four-column raw1280 / random128 /
  learned128 / group-gated128 patch-match overlays. Green marks patch-positive
  matches, red marks patch-false matches, yellow rings mark PnP inliers.
- `*_margin_heatmap_contact_sheet.png`: top1-top2 cosine similarity margin
  heatmaps over the RGB query. These expose descriptor sharpness and repeated
  region ambiguity without comparing incompatible PCA color spaces.
- `group_energy_heatmap.png`: raw-channel group energy for learned128 and
  group-gated128 transforms. This makes the selector/gate behavior visible at
  the input-channel-group level.

Current generated examples:

- ShopFacade:
  `output/vfm/stage_c1_qualitative/shopfacade/seq3__frame00045_match_comparison_contact_sheet.png`
  and
  `output/vfm/stage_c1_qualitative/shopfacade/seq1__frame00036_match_comparison_contact_sheet.png`.
- OldHospital:
  `output/vfm/stage_c1_qualitative/oldhospital/seq4__frame00020_match_comparison_contact_sheet.png`
  and
  `output/vfm/stage_c1_qualitative/oldhospital/seq8__frame00027_match_comparison_contact_sheet.png`.

The qualitative examples match the quantitative diagnosis: learned128 often
raises descriptor margin and PnP inlier count on good cases, while hard cases
can still fail despite higher patch precision, so geometry degeneracy and
candidate-submap quality remain active bottlenecks.

## Remaining Paper-Critical Gaps

1. Real raw VFM token extraction:
   - StMarysChurch train is still missing and should wait for more disk headroom
   - StMarysChurch test/train pair should be completed before final five-scene
     selector training
   - cache or vendor DINOv2 if it remains a required VFM baseline
   - normalization/whitening without labels

2. Candidate hypothesis library:
   - train/eval/inference banks with protocol hashes
   - real hard-negative mining from fixed candidate generators

3. Real selector training:
   - extend the completed 4-scene 5-seed split-clean run to StMarysChurch once
     the train token bank exists
   - add bootstrap CIs and paired significance tests for the 5-seed reports
   - checkpoint manifest with protocol hash
   - no oracle inputs

4. Selected feature mapability:
   - selected/raw/PCA/random comparison for dense-trained selector
   - improve rendered selected-map retention with query-view projection

5. Map-conditioned verifier:
   - replace global/reference-frame renderers with candidate-pose projected
     query-view token-grid map features
   - feature inlier ratio and calibrated risk head
   - risk calibration

6. Final solver handoff:
   - fixed external solvers only
   - solver-free top1 and handoff reported separately
   - hard-case subsets reported independently

## Next Implementation Round

Priority order:

1. Add protocol hashes to token, descriptor, track-bank, checkpoint, and report
   manifests.
2. Add real hard-case mining from the converted OldHospital fixed retrieval bank
   once pose labels are joined.
3. Add selected-feature same-track bank construction from the same COLMAP
   observation JSONL.
4. Add dense selector inference over token banks, then compare selected/raw/PCA
   mapability variance and separability.
5. Add map-conditioned verifier evaluation over real rendered/projected selected
   track features and controls.
6. Add final fixed-solver handoff tables only after the solver-free verifier and
   mapability claims are stable.

The project now has real tokens, fixed reference-pose banks, descriptor-level
selector smoke, dense selector training with positive 2D evidence, and a
provenance-safe rendered selected-map smoke verifier. The next substantial step
is local geometry-aware rendered-map verification and causal controls, not more
global mean-pooled descriptor tuning.

Expected promotion metrics are tracked in `docs/vfm/expected_metrics.md`.
