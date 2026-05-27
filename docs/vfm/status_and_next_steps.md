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
