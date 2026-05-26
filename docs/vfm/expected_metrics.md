# VFM-MapLoc Expected Metrics

This file records the current paper-facing gates. These are expected promotion
targets, not claims. A result should be reported as a main result only if it
uses a fixed candidate generator, no oracle inputs, and a held-out query or
scene protocol.

## Solver-Free Hypothesis Verification

Protocol: Cambridge reference-pose top10/top20 and real retrieval top20/top50.
Primary scorer inputs must be query-map visual evidence plus whitelisted
candidate metadata when explicitly labeled as metadata-assisted.

| metric | near-term target | paper target |
| --- | ---: | ---: |
| Spearman / Kendall | +0.15 over retrieval order | +0.25 over retrieval order |
| pred_cost_m | 10% lower than retrieval order | 15-25% lower than retrieval order |
| oracle_gap_m | 10% lower than retrieval order | 20% lower than retrieval order |
| top1_acc | not worse by more than 3 pp | better than retrieval order |
| basin@5 | +3 pp over retrieval order | +5 pp over retrieval order |
| ECE | <= 0.10 | <= 0.07 |
| risk-coverage AUC | better than raw/PCA/random | best among non-oracle controls |

Current held-out descriptor-selector smoke:

| scene | pred vs retrieval | top1 vs retrieval | Spearman vs retrieval | basin@5 vs retrieval |
| --- | ---: | ---: | ---: | ---: |
| OldHospital | `4.268 -> 3.559` | `0.528 -> 0.593` | `0.291 -> 0.672` | `0.648 -> 0.704` |
| ShopFacade | `1.529 -> 1.471` | `0.794 -> 0.698` | `0.351 -> 0.583` | `0.857 -> 0.905` |

Interpretation: enough to continue the selector path; not enough to claim a
final verifier because ShopFacade top1 drops and the current model is
descriptor-level rather than rendered-map-conditioned.

Current dense-selector reference-pose top10 result:

| scene | retrieval pred/top1/Spearman | dense selector mean pred/top1/Spearman | status |
| --- | --- | --- | --- |
| OldHospital | `4.171 / 0.495 / 0.316` | `3.240 / 0.623 / 0.704` | clears near-term solver-free gate |
| ShopFacade | `1.635 / 0.806 / 0.396` | `1.346 / 0.838 / 0.623` | clears near-term solver-free gate |
| KingsCollege | `3.431 / 0.726 / 0.321` | seed0 full-query `3.226 / 0.816 / 0.445` | clears near-term solver-free gate |
| GreatCourt | `14.061 / 0.317 / 0.166` | seed0 full-query `7.854 / 0.443 / 0.527` | clears near-term solver-free gate |

This is the current strongest evidence-selection result. OldHospital and
ShopFacade already have multi-seed full-query reports; KingsCollege and
GreatCourt now have 5-seed training diagnostics plus seed0 cached full-query
reports. They still need 5-seed cached full-query scoring, controls, and
bootstrap CIs before paper claims.

Dense training seed stability check:

| scene | raw eval top1 mean | dense eval top1 mean | dense eval range |
| --- | ---: | ---: | ---: |
| OldHospital | 0.278 | 0.500 | 0.389-0.611 |
| ShopFacade | 0.286 | 0.429 | 0.381-0.476 |
| KingsCollege | 0.238 | 0.435 | 0.362-0.536 |
| GreatCourt | 0.120 | 0.370 | 0.316-0.414 |

Near-term next gate: add cached full-query scoring for all 5 seeds, bootstrap
CIs, and controls.

Split-clean train/test protocol update:

| scene | retrieval pred/top1/Spearman | raw RADIO pred/top1/Spearman | descriptor-retrieval trained selected64 pred/top1/Spearman | current status |
| --- | --- | --- | --- | --- |
| OldHospital | `4.171 / 0.495 / 0.316` | `4.397 / 0.538 / 0.222` | `4.066 / 0.593 / 0.316` | positive vs retrieval and raw on pred/top1 |
| ShopFacade | `1.635 / 0.806 / 0.396` | `2.093 / 0.718 / 0.192` | `1.945 / 0.709 / 0.374` | positive vs raw, below retrieval median/top1 |
| KingsCollege | `3.431 / 0.726 / 0.321` | `4.603 / 0.647 / 0.021` | `4.109 / 0.749 / 0.269` | positive top1/basin, pred still weak |
| GreatCourt | `14.061 / 0.317 / 0.166` | `15.770 / 0.239 / -0.006` | `13.985 / 0.301 / 0.168` | slight pred/Spearman gain vs retrieval, top1 weak |

This table supersedes the pose-nearest split-clean smoke as the active clean
training protocol. It is positive enough to continue, but not yet paper-grade:
run 5 seeds, add hard-negative/basin-aware loss weighting, and require paired
bootstrap CIs before claiming the selector is robustly better than retrieval.

Basin-aware + hard-negative loss is implemented and currently behaves as a
tradeoff knob rather than a universal replacement:

| scene | descriptor-retrieval selected64 | + basin/hard selected64 | useful gain |
| --- | --- | --- | --- |
| OldHospital | `4.066 / 0.593 / 0.316 / 0.736` | `4.113 / 0.593 / 0.366 / 0.747` | Spearman and basin@5 |
| ShopFacade | `1.945 / 0.709 / 0.374 / 0.913` | `2.157 / 0.738 / 0.283 / 0.913` | top1 only |
| KingsCollege | `4.109 / 0.749 / 0.269 / 0.898` | `4.021 / 0.711 / 0.277 / 0.892` | pred/gap/Spearman |
| GreatCourt pose-valid | `13.985 / 0.301 / 0.168 / 0.496` | `13.221 / 0.330 / 0.190 / 0.522` | pred/top1/Spearman/basin |

Values are `pred / top1 / Spearman / basin@5`. The next promotion target is not
to freeze these weights, but to sweep basin/hard-negative weights on held-out
train scenes and report 5-seed CIs.

Latest full-data split-clean 5-seed result, using descriptor-retrieval train
candidates, held-out reference-pose top10 test candidates, `selected64`,
`spatial_samples=768`, `batch_size=64`, and basin/hard-negative loss:

| scene | retrieval pred/top1/Spearman/basin@5/gap | raw RADIO pred/top1/Spearman/basin@5/gap | selected64 5-seed mean pred/top1/Spearman/basin@5/gap | status |
| --- | --- | --- | --- | --- |
| OldHospital | `4.171 / 0.495 / 0.316 / 0.687 / 1.543` | `4.397 / 0.538 / 0.222 / 0.709 / 1.769` | `4.049±0.157 / 0.532±0.023 / 0.343±0.026 / 0.736±0.007 / 1.421±0.157` | positive vs retrieval on pred, Spearman, basin@5, gap |
| ShopFacade | `1.635 / 0.806 / 0.396 / 0.893 / 0.721` | `2.093 / 0.718 / 0.192 / 0.903 / 1.179` | `2.150±0.140 / 0.726±0.029 / 0.248±0.032 / 0.917±0.009 / 1.236±0.140` | positive vs raw and basin@5; below retrieval prior |
| KingsCollege | `3.431 / 0.726 / 0.321 / 0.872 / 1.252` | `4.603 / 0.647 / 0.021 / 0.846 / 2.423` | `4.053±0.151 / 0.731±0.009 / 0.289±0.008 / 0.892±0.004 / 1.873±0.151` | positive vs raw and retrieval top1/basin@5; pred below retrieval |
| GreatCourt | `14.061 / 0.317 / 0.166 / 0.505 / 8.221` | `15.770 / 0.239 / -0.006 / 0.437 / 9.930` | `13.695±0.478 / 0.350±0.007 / 0.216±0.004 / 0.522±0.008 / 7.855±0.478` | positive vs retrieval and raw |

Interpretation: this is the first full 5-seed, split-clean, non-q-lattice
selector result on the available four Cambridge train/test token banks. It
supports the selected-feature evidence claim, but does not clear a universal
"better than retrieval" gate. Near-term paper framing should emphasize
localization evidence, risk verification, and hard-case utility rather than a
replacement for retrieval order.

Paired query-level bootstrap statistics sharpen that interpretation:

- vs raw RADIO, selected64 gives robust positive evidence on KingsCollege and
  GreatCourt: pred_cost, Spearman, and basin@5 confidence intervals are all in
  the favorable direction for most or all seeds.
- vs retrieval order, selected64 is mixed: OldHospital and GreatCourt improve
  basin/Spearman in several seeds, but ShopFacade and KingsCollege remain
  dominated by the retrieval prior on pred_cost.
- On OldHospital real retrieval, both hard-case slices and projected
  selected-map scoring are negative versus candidate prior. This blocks any
  real-retrieval hard-case utility claim until training or fusion explicitly
  targets real-retrieval candidates.

The report status is therefore:

| gate | current status |
| --- | --- |
| raw-vs-selected feature utility | positive on multiple scenes |
| descriptor causality controls | positive on 4 scenes / 5 seeds |
| selected-vs-retrieval universal ranking | not established |
| real-retrieval hard-case utility | negative so far |
| projected selected-map real retrieval | negative so far |
| controlled rendered-pose sanity | positive, but GT-centered only |

## Feature Selection Causality

Required controls:

- raw VFM
- same-dim first channels
- same-dim random projection
- same-dim PCA
- selector projection
- query feature shuffle
- map/reference feature shuffle
- wrong-scene map/reference feature
- high-utility removal
- low-utility removal

Expected promotion targets:

| metric | target |
| --- | ---: |
| selector vs raw Spearman | positive in at least 4/5 scenes |
| selector vs PCA/random pred_cost | >= 10% lower on mean paired queries |
| query/map shuffle degradation | >= 30% relative Spearman drop |
| wrong-scene degradation | close to random / retrieval-prior only |
| high-utility removal | >= 15% pred_cost degradation |
| low-utility removal | <= 5% pred_cost degradation |

Current 4-scene / 5-seed descriptor-level causality result for the split-clean
`selected64`, `s768/b64`, basin/hard-negative protocol:

| scene | query-shuffle delta pred/top1/Spearman | map-shuffle delta pred/top1/Spearman | wrong-scene-map delta pred/top1/Spearman | status |
| --- | --- | --- | --- | --- |
| OldHospital | `-2.645 / +0.300 / +0.378` | `-1.654 / +0.222 / +0.338` | `-1.659 / +0.221 / +0.345` | passes descriptor causality |
| ShopFacade | `-0.602 / +0.270 / +0.263` | `-0.288 / +0.256 / +0.245` | `-0.337 / +0.256 / +0.244` | passes descriptor causality; map-shuffle basin unchanged |
| KingsCollege | `-1.301 / +0.274 / +0.275` | `-0.566 / +0.220 / +0.288` | `-0.603 / +0.202 / +0.280` | passes descriptor causality |
| GreatCourt | `-3.299 / +0.176 / +0.231` | `-2.309 / +0.133 / +0.218` | `-2.750 / +0.137 / +0.217` | passes descriptor causality |

Deltas are selected minus corrupted-control. Lower pred deltas and higher
top1/Spearman deltas are favorable. The associated paired bootstrap files are
under `output/vfm/reports/paired_stats/`. This upgrades the selected-feature
claim from raw-vs-selected correlation to a causal descriptor-evidence sanity
check, but it is still not a rendered-map or real-retrieval success claim.

Utility-channel removal is implemented but currently fails the stronger
interpretability gate. Removing the top or bottom channels according to
`abs(utility_head.weight)` at 25% and 50% produces only small, inconsistent
changes; high-utility removal is not reliably worse than low-utility removal.
Therefore the current selector supports a query-map evidence claim, but not yet
a channel-level utility attribution claim. The high/low removal promotion gate
remains open.

The training and descriptor-cache path now has an optional
`utility_weighted_pooling` mode, which lets the spatial utility head influence
selected descriptor pooling and receive ranking gradients. A seed0
OldHospital/ShopFacade smoke confirms the path runs, but it is not a promotion
result: selected16 with utility-weighted pooling lowers Spearman versus the
existing selected64 baseline on both scenes, and high/low utility removal only
shows weak top1 separation. Treat it as infrastructure for the next
track/rendered-map utility supervision experiment, not as evidence that utility
maps are already interpretable.

Spatial high/low utility masking is also implemented. On seed0
utility-weighted selected16, OldHospital moves in the desired direction when
the top 50% utility positions are removed, but ShopFacade is inverted on
pred/Spearman. This means the Gate-2 spatial masking interface exists, while
the spatial utility map itself still fails the cross-scene promotion criterion.

Track/geometry-supervised utility pretraining is now implemented and tested.
It improves real same-track vs different-track separation on OldHospital and
ShopFacade 20k COLMAP-observation smokes, but the lifted 64D track banks still
trail PCA64 on variance and separability. Treat this as a positive training
signal and infrastructure milestone, not a passed mapability gate.

## Mapability

Protocol: same COLMAP tracks, same observation subset, same feature dimension,
L2-normalized per observation unless the representation is explicitly evaluated
as unnormalized.

Current 20k-observation smoke:

| scene | selector16 separability | PCA16 separability | random16 separability | first16 separability |
| --- | ---: | ---: | ---: | ---: |
| OldHospital | 196.7 | 302.9 | 124.3 | 138.4 |
| ShopFacade | 164.6 | 336.6 | 133.0 | 127.3 |

Expected promotion targets:

| metric | near-term target | paper target |
| --- | ---: | ---: |
| track coverage | match same-dim controls | match same-dim controls |
| storage | <= 10% raw full-dim | <= 5% raw full-dim |
| variance | lower than random and first-channels | lower than PCA/random/first |
| separability | higher than random and first-channels | higher than PCA/random/first |
| rendered-map retention | same trend as 2D selected scoring | <= 10% drop from 2D selected trend |

Current interpretation: selector16 is compact and beats random/first on
separability, but PCA16 remains stronger. The next selected-map target is to
beat PCA16 through dense, localization-supervised selector training rather than
descriptor-only smoke training.

Dense-trained selector16 mapability on the same 20k-observation smoke:

| scene | dense selector16 separability | dense selector16 variance | note |
| --- | ---: | ---: | --- |
| OldHospital | 187.2 | 0.006249 | lower separability than descriptor-selector16 and PCA16 |
| ShopFacade | 144.1 | 0.008700 | only slightly above random/first; below PCA16 |

Dense-trained selector16 selected track banks on 100k-observation full-scene
exports:

| scene | tracks | dim | variance | mean obs/track | note |
| --- | ---: | ---: | ---: | ---: | --- |
| KingsCollege | 5,350 | 16 | 0.007653 | 18.69 | selected 3D bank generated |
| GreatCourt | 7,994 | 16 | 0.012240 | 12.51 | selected 3D bank generated |

This shows that the current dense ranking objective improves 2D
hypothesis-verification strongly but does not automatically optimize 3D
mapability. The next selector loss should add track consistency/visibility or
map-rendered supervision.

Joint dense-ranking + track-consistency training is now implemented and should
be treated as the active mapability refinement line. Use `5m/10deg` basin labels
when comparing to existing reference-pose selected64 reports; stricter thresholds
must be reported as a separate protocol column.

Seed0 20k-track smoke:

| scene | method | pred_m | top1 | Spearman | basin@5 | variance | separability |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OldHospital | selected64 | 3.881 | 0.527 | 0.369 | 0.747 | 0.002075 | 586.4 |
| OldHospital | joint consistency anchor | 3.843 | 0.533 | 0.373 | 0.753 | 0.002009 | 601.4 |
| ShopFacade | selected64 | 2.113 | 0.738 | 0.296 | 0.922 | 0.002507 | 494.6 |
| ShopFacade | joint consistency anchor | 2.149 | 0.757 | 0.255 | 0.922 | 0.002480 | 501.0 |

Near-term promotion gate for this line:

| metric | target |
| --- | ---: |
| pred/gap | no worse than selected64 by more than 2% |
| Spearman | no worse than selected64 by more than 0.02 |
| top1 or basin@5 | improve selected64 on at least two scenes |
| selected-track variance | lower than selected64 on at least two scenes |
| separability | higher than selected64 on at least two scenes |
| PCA64 gap | reduce separability gap to PCA64 by 50% before paper claim |

Rendered selected-map retention with the current global visible-track mean:

| scene | 2D dense selected pred/top1/Spearman | rendered map pred/top1/Spearman | retention status |
| --- | --- | --- | --- |
| OldHospital | `3.178 / 0.648 / 0.707` | `4.926 / 0.335 / 0.149` | fails retention |
| ShopFacade | `1.395 / 0.854 / 0.629` | `2.206 / 0.602 / 0.082` | fails retention |

The rendered-map target remains: retain the 2D selected-feature trend with no
more than a 10% drop. The current global descriptor renderer is a smoke test,
not the final map-conditioned verifier.

Sparse reference-frame token-grid renderer has also been tested. It rasterizes
selected track features to reference image token coordinates and compares local
query selected features. It does not use candidate-pose projection yet:

| scene | sparse r0 pred/top1/Spearman | sparse r4 pred/top1/Spearman | status |
| --- | --- | --- | --- |
| OldHospital | `5.421 / 0.379 / 0.062` | `5.182 / 0.407 / 0.081` | fails retention |
| ShopFacade | `2.861 / 0.524 / -0.041` | `2.775 / 0.544 / -0.073` | fails retention |

This turns the next verifier requirement into a concrete engineering target:
project 3D track `xyz` into the query camera defined by the candidate pose and
camera intrinsics, then compute local selected-feature inlier evidence.

That projected verifier is now implemented. On reference-pose top10 it remains
a negative retention result, which is expected to be a candidate-type mismatch:
reference image poses are coarse database poses, not query-view rendered-pose
hypotheses.

| scene | projected r4 pred/top1/Spearman | projected r4 + inlier/utility | status |
| --- | --- | --- | --- |
| OldHospital | `4.626 / 0.505 / 0.084` | `4.581 / 0.505 / 0.077` | reference-pose retention fails |
| ShopFacade | `2.830 / 0.553 / -0.120` | `2.790 / 0.553 / -0.117` | reference-pose retention fails |

The same verifier gives a positive sanity result on a clearly disclosed
`controlled_lattice` protocol. The candidate generator uses GT query poses and
world-translation perturbations, so this is not a deployable localization
number; it only validates the projected selected-map evidence path.

| scene | protocol | pred_m | top1 exact@0.1m | Spearman | basin@5 |
| --- | --- | ---: | ---: | ---: | ---: |
| OldHospital | GT-centered rendered-pose lattice | 0.133 | 0.544 | 0.679 | 0.978 |
| ShopFacade | GT-centered rendered-pose lattice | 0.056 | 0.786 | 0.668 | 1.000 |

Promotion gate for rendered selected-map verification is now split:

| protocol | near-term target |
| --- | ---: |
| GT-centered sanity lattice | Spearman >= 0.60 on OldHospital and ShopFacade |
| init-centered rendered-pose proposals | beat candidate prior by >= 10% pred_cost |
| reference-pose top10 | diagnostic only; do not require spatial retention |

Init-centered rendered-pose proposal banks are now implemented through
`feature_extract.tools.vfm.build_init_pose_lattice`. They use real retrieval
or other non-oracle init poses to generate q-level translation lattices; GT
Cambridge poses are used only to label `pose_error`. On OldHospital real
retrieval top1/top4 init, the candidate generator has useful upper bound but
the current projected selected-map verifier still fails to exploit it:

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

This is a newly closed protocol gap, not a positive verifier result. The
candidate generator is strong enough for q-level verification, but the selected
3D map evidence must be improved before Protocol C can be promoted. Reference
visibility filtering and `local_radius=0` reduce the damage from local
max-pooling, but projected selected-map evidence still trails source-prior
ranking. The q50 top1 `ref-vis r0` report now includes evidence diagnostics:
empty evidence fraction `0.376`, mean visibility fraction `0.539`, mean match
count `136.4`, mean inlier fraction `0.525`, and mean similarity `0.069`.
Near-term Protocol C work should reduce empty evidence and improve
visibility/occlusion-aware scoring before adding a stronger learned scorer.
The projected-grid CLI now refuses mismatched COLMAP model/camera provenance
when the track-observation summary records a source `model_dir`.

## Real Retrieval Transfer

OldHospital real retrieval top20 is now label-joined into a deployable fixed
candidate bank with 2,912 labeled candidates over 182 queries.

| method | pred_m | top1 | Spearman | basin@5 |
| --- | ---: | ---: | ---: | ---: |
| retrieval order | 0.429 | 0.527 | 0.246 | 0.709 |
| candidate prior / POFD score | 0.335 | 0.577 | 0.506 | 0.764 |
| dense selector16 seed0 | 0.510 | 0.473 | 0.103 | 0.720 |
| dense selector16 seed1 | 0.457 | 0.500 | 0.104 | 0.709 |
| dense selector16 seed2 | 0.489 | 0.473 | 0.087 | 0.709 |
| candidate prior + dense0 zscore fusion a=0.10 | 0.328 | 0.599 | 0.499 | 0.769 |

Current interpretation: reference-pose trained dense selection alone does not
transfer to real retrieval; candidate-prior metadata remains stronger as a
standalone scorer. A lightweight query-wise fusion baseline now shows that
visual selected-feature evidence can add a small amount of useful signal when it
is treated as an auxiliary verifier rather than a replacement for the prior.
The full-set `a=0.10` number above is exploratory because the fusion weight is
chosen on the same query set.

The current best full-set real-retrieval fusion smoke is:

```text
score = zscore_per_query(candidate_prior)
      + 0.10 * zscore_per_query(dense selector16 seed0)
```

Against candidate prior, paired bootstrap over 182 queries gives:

| metric | delta | 95% CI |
| --- | ---: | ---: |
| pred_m | -0.007 | [-0.028, 0.012] |
| top1 | +0.022 | [0.000, 0.044] |
| Spearman | -0.007 | [-0.016, 0.002] |
| basin@5 | +0.005 | [0.000, 0.016] |

This is a narrow positive result, not a final claim: pred/top1/basin improve
slightly, but Spearman is lower and the confidence interval for pred still
crosses zero. The next promotion target is to make this visual+prior gain
larger and stable across seeds/scenes.

Split-calibrated real-retrieval check:

| split | selected alpha | eval queries | pred_m delta | top1 delta | Spearman delta | basin@5 delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| seq4 -> seq8 | 0.10 | 126 | -0.010 | +0.024 | -0.008 | +0.000 |
| seq8 -> seq4 | 0.10 | 56 | -0.002 | +0.018 | -0.006 | +0.018 |
| random 5-seed conservative mean | 0.02-0.10 | 91 each | -0.001 | +0.015 | -0.002 | +0.004 |

The split-calibrated result is the current paper-facing evidence. It supports
only a weak auxiliary-verifier claim: calibrated selected visual evidence can
recover a few top1/basin cases over a strong candidate-prior scorer, while
ranking correlation remains essentially unchanged or slightly worse.

Projected rendered selected-map fusion was also tested with the same held-out
protocol and remains negative/mixed: `seq4 -> seq8` is flat on pred/top1 and
loses basin@5, while `seq8 -> seq4` improves top1 by `+0.018` but degrades
pred, Spearman, and basin@5. The map-conditioned verifier is therefore not yet
a paper-facing real-retrieval result.

Sparse rendered-map evidence has been corrected to use cosine-style local
token matching for both query and rendered features. The corrected projected
rows are still negative on OldHospital real retrieval (`pred 0.768`, `top1
0.352`, `Spearman 0.015`, `basin@5 0.676`), so the expected indicator for the
next iteration is not another normalization tweak. The target is a geometry and
visibility improvement that at least restores projected-map fusion to non-worse
held-out `pred_m` and `basin@5` versus candidate prior.

## Hard-Case Utility

Report hard subsets separately:

- retrieval top1 wrong but topK contains a correct basin candidate
- high-score wrong false accept
- near-identity false positive
- repeated corridor/facade cases
- fixed solver handoff regressions

Expected targets:

| metric | target |
| --- | ---: |
| false-accept rate | 20% relative reduction vs retrieval/raw/PCA |
| hard subset Spearman | +0.20 over retrieval order |
| hard subset basin@5 | +5 pp over retrieval order |
| catastrophic failure rate | lower than fixed solver-only baseline |
| guarded handoff | not worse than identity selector top1 |

Current OldHospital real-retrieval hard-slice fusion smoke:

| subset | method | pred_m | top1 | Spearman | basin@5 | gap_m |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| retrieval-top1-wrong | candidate prior | 0.446 | 0.278 | 0.446 | 0.796 | 0.291 |
| retrieval-top1-wrong | prior + dense0 a=0.10 | 0.443 | 0.333 | 0.425 | 0.815 | 0.287 |
| PnP-high-score-wrong | candidate prior | 0.370 | 0.503 | 0.489 | 0.732 | 0.204 |
| PnP-high-score-wrong | prior + dense0 a=0.10 | 0.369 | 0.530 | 0.478 | 0.738 | 0.203 |

This supports a limited hard-case utility claim: selected visual evidence can
reduce a few candidate-prior false accepts when fused conservatively. It still
does not satisfy the paper target because the gains are small and Spearman
drops.

## Final Pose Handoff

Final localization should be reported in three separated rows:

- identity selector top1 pose
- selector-guarded fixed solver handoff
- fixed solver baseline without selector guard

Expected targets:

| metric | target |
| --- | ---: |
| median translation | match or improve fixed solver baseline on hard cases |
| R@5deg/250mm | +3 pp on hard cases |
| solver regression rate | <= 2% absolute queries |
| identity-vs-handoff gap | handoff must not degrade clean selector top1 systematically |

If final pose does not beat HLoc/fixed solver, the retained claim should be:
VFM-MapLoc provides compact map-conditioned verification and risk evidence, not
state-of-the-art final localization.
