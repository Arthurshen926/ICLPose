# V6: 2DGS Maplet-Atlas Localization

V6 is the active research line. It localizes a query against a feature-bearing
2DGS map; it does not retrieve or match a stored reference image.

## Frozen method definition

```text
Stage A  query RADIO-final regions
         -> full-map maplet identity posterior + identity null

Stage B  region identities + canonical maplet geometry
         -> several region-level coarse SE(3) modes

Stage C  render selected canonical feature atlases at every coarse mode
         -> multi-scale local displacement distributions
         -> robust joint SE(3) updates, retaining multiple flow modes

Stage D  maplet-disjoint evidence
         -> accept, reject, or retain several pose modes
```

RADIO regions are area/context observations. Their centers are not keypoints
and must not be interpreted as exact 2D measurements of surface cells.
Consequently, regional-center PnP is a diagnostic decomposition only. It is
not the coarse estimator and never becomes the final estimator.

The target runtime map contains 2DGS geometry and bounded anonymous feature
sufficient statistics. It contains no mapping RGB, mapping image paths or IDs,
per-view descriptor list, SfM point/track, stable-anchor identity, ALIKE
descriptor map, RADIO-intermediate feature, or pairwise image matcher input.
A maplet may retain a small anonymous appearance mixture; this is not a bundle
of reference-image descriptors.

## P0 probability and representation contract

The identity and spatial posteriors are different random variables.

```text
p(identity m | query region)

p(surface mode k | identity m, query region, spatial atlas available)
```

The identity bank is always the full bank. A maplet absent from the spatial
atlas remains an identity candidate with:

```text
spatial_available = false
p(spatial null | m, q) = 1
```

Every available identity candidate has its own conditional spatial null:

```text
sum_k p(surface mode k | m,q) + p(spatial null | m,q) = 1
```

Nearby spatial samples suppressed within the metric NMS radius describe one
wider resolved mode: their probability, mean and covariance are merged by the
law of total covariance. Only distinct modes removed by the top-M capacity
limit are transferred to spatial null. Retained modes are not renormalized. A
representative geometry value is a real MAP cell; a posterior mean of distinct
3D cells is forbidden.
Scene-level Top-K removal is transferred to identity null.

Identity-null and conditional-spatial-null probabilities come from a frozen,
trajectory-disjoint artifact fitted with unweighted Bernoulli NLL. Class
balanced classifier scores are not probabilities and are rejected by the
artifact loader. Conditional-spatial labels contain only true identities with
an available atlas; wrong identities are not relabelled as spatial nulls.

The compact identity map now retains canonical tangent frames as geometry.
This is required for an oriented projected-area model. Inferring a chart frame
from only a normal is explicitly non-canonical and is rejected by footprint
localization. Pose-vote artifacts hash both descriptor components and retrieval
geometry.

## Stage B implementation and status

The active research bridge is `MapletFrameAlignment`:

```text
RADIO-final RetrievalRegions
  -> overlap-aware Top-q / noisy-OR / block-balanced scene evidence
  -> many-to-many RegionChartIndex
  -> several independent MetricSurfaceCharts
  -> per-chart full-query translation / scale / rotation / shear correlation
  -> optional continuous local affine refinement
  -> additive independent-chart evidence + canonical frame controls
  -> IPPE or grouped regional pose modes
```

`RetrievalRegionBank` and `MetricSurfaceChartBank` are intentionally different
entities. Retrieval regions may be larger, overlapping and context-rich;
metric charts remain locally planar and continuous. `RegionChartIndex` is
strict CSR in both directions and its artifact loader rejects unknown versions,
non-canonical fields and any image/SfM/point-correspondence contract violation.
The current StMaryChurch retrieval bank is still a legacy single-scale
maplet proxy. Expanding its Region→Chart adjacency improves geometric support
but does **not** create a learned context-rich superregion descriptor; this is
now recorded explicitly by the index builder in newly generated metadata.

Overlapping RADIO regions are correlated observations. The default scene
ranking therefore uses spatial NMS followed by a capped Top-q sum. Legacy
unbounded sum, noisy-OR after NMS and block-balanced aggregation remain
reported side by side. Frame hypotheses use joint translation/affine NMS:
nearby translations may retain a bounded number of genuinely different
scale/rotation/shear modes instead of discarding them all as one center mode.
Composite charts preserve bounded per-texel anonymous appearance mixtures,
and local refinement marginalizes those modes with their priors. However,
flattening disconnected/non-coplanar charts into one virtual plane is retained
only as a diagnostic; the active bridge aligns the associated metric charts
independently and combines their regional evidence in pose space.

Cross-scale hypotheses are compared by mean regional correlation. The older
`support**0.35` factor was methodologically invalid: it rewarded larger
projected templates even when their mean match was worse. Independent chart
evidence is additive rather than averaged. Averaging caused single-chart
modes to swamp geometrically supported three/four-chart modes.

Three older coarse methods remain diagnostic only:

1. `anonymous_view_mode` reuses an anonymous mapping-camera pose distribution;
   it is a historical-view seed baseline, not the V6 coarse definition.
2. `maplet_footprint` is a pose scorer/refinement objective. It is not a
   from-scratch proposal generator.
3. `regional_center_pseudo_pnp` treats a region center as a pseudo point and
   cannot be reported as production localization.

The footprint foreground term is an image-background density ratio. The older
unnormalized expression was bounded by one, so a match could never beat the
unmatched branch; that implementation was invalid. The older spherical
maplet approximation was also invalid because canonical chart axes and
extents were discarded. The current model projects an oriented canonical
quad and uses its dilated area.

None of the Stage-B runtime methods is production-qualified yet. In
particular, stored anonymous view modes are not a substitute for
cross-trajectory pose extrapolation, and a high footprint likelihood does not
generate the missing global frame mode.

## Stage C/D contract and current blocker

Canonical atlases use fixed clean-2DGS geometry and raster contributor IDs.
Mapping observations update only anonymous feature sufficient statistics,
support and uncertainty; they never move canonical XYZ. Rendering uses
perspective-correct area interpolation and selected-maplet z-buffering. The
analytic solver uses the radial-camera Jacobian, robust covariance weighting,
per-maplet balancing, and maplet-disjoint verification.

The current correlation null remains:

```text
fixed pair-null logit + atlas-uncertainty term
```

and still uses a random-unit-vector partition approximation. It is not the
required typed calibration over correlation peak, top-2 margin, entropy,
periodicity, spread, identity probability, visibility, occlusion and
matchability. Therefore Stage C/D is not runtime-qualified. The missing typed
outcomes are at least:

```text
wrong identity
spatial unavailable
out of search window
occluded
query unmatchable
valid flow / zero flow
```

The runtime path also does not yet close two or three render/correlate/update/
verify iterations. A one-step oracle experiment is a component diagnostic,
not end-to-end localization.

## StMaryChurch artifacts used by the corrected audit

The corrected map lineage uses the clean 2DGS geometry and these generated
artifacts:

- full identity map: `identity_maplets_full864_mapper_k4_frames_v2.npz`
  (864 maplets, 3,226 anonymous appearance components);
- separate spatial map: 783/864 identities have at least one spatial atlas
  component;
- pose statistics:
  `anonymous_pose_votes_full_identity_region_conditioned_k4_frames_v5.npz`;
- probability calibration:
  `probability_calibration_seq11_full_identity_frames_conditional_proper_v6.json`.

The frozen audit lineage is:

| Artifact | SHA-256 |
| --- | --- |
| `/root/StMaryChurch2dgs_clean.ply` | `4127e187090baa3ab1f21cdba7e3b0de8b51c0e177d29a425120964f18c506af` |
| canonical atlas geometry | `43cbc9c0effb2932baa449ab72b45e185ef6241b989b54321f3de13b39b92c48` |
| full identity map | `be3b0617a9ffc0521f544e631fced3a13ae9ae4c73a3eee41f6b24bd9030eceb` |
| separate spatial map | `fbe0f54ce14b1442097b971df963daf6cefcc4db133c642932489d613848811f` |
| anonymous pose statistics | `e112f3f94bc411052a2be118d61f88befafdee5ef13a94d6c7940d4d8ead5780` |
| probability calibration | `319626aef6b65997e61a5e50c8148d84ded7c09beccd55cb5af25d3f188bcd6c` |
| corrected strict-12 dashboard (`dashboard_bearing_corrected_view_vote_strict12_v13.json`) | `8d53f233e65f548f1b091cedc9a844969383ae9e5d218696684cb95c164be177` |

The calibration trajectory is `seq11`; the fixed strict audit trajectories are
`seq3`, `seq5`, and `seq13`. They are disjoint. The calibration artifact has:

| Posterior | Examples | Null prevalence | AUPRC | Brier | ECE |
| --- | ---: | ---: | ---: | ---: | ---: |
| identity null | 768 | 42.19% | 61.11% | 0.2283 | 0.0667 |
| spatial null, conditional on true identity + atlas | 608 | 12.34% | 18.50% | 0.1072 | 0.0130 |

The conditional spatial probability is calibrated in prevalence but has weak
ranking power. The near-perfect AUPRC obtained by mixing wrong identities into
spatial-null labels was a metric-definition bug and is invalid.

Pose-vote artifacts now record every mapping trajectory. By default the
evaluator rejects any query trajectory that overlaps either probability
calibration or pose-vote map construction. The
`--allow_reference_trajectory_diagnostic` escape hatch is diagnostic-only:
such a run is reference replay, not independent localization evidence. In
particular, the earlier `seq11` result produced from an all-training pose bank
cannot support a generalization claim.

The 128-view bake covers 787 of 848 geometry-bearing maplets, but only 47.38%
of valid canonical texels. Median coverage among baked maplets is 51.97%;
physical texel size has a 4.81 cm P90. Adaptive 16/32/64 charts, automatic
split of disconnected/non-planar maplets, and full-scene runtime occlusion
remain open.

## Corrected fixed strict audit

The promotion audit contains 12 deterministic queries sampled from the
official test trajectories. It is not the 530-query result.

After full identity/spatial decoupling, proper null calibration, region
conditioning, symmetric pose kernels, canonical-frame geometry and bearing
correction:

| Metric | Result |
| --- | ---: |
| identity probability conservation max error | about `1.2e-7` |
| conditional spatial conservation max error | about `6.9e-7` |
| dominant identity recall @1 / @5 / @64 | 29.92% / 62.04% / 71.06% |
| visible surface coverage | 67.91% |
| true-identity conditional spatial-null AUPRC | 24.50% |
| anonymous region-mode Top-1 within 30 cm / 3 deg | 0/12 |
| anonymous region-mode oracle Top-16 within 30 cm / 3 deg | 0/12 |
| Top-1 translation median / P90 | 2.96 m / 8.52 m |
| Top-1 rotation median / P90 | 7.77 deg / 29.16 deg |

This is a failed coarse-pose gate, not a production result. The bearing update
is a small improvement over fixed-pose voting, but it does not solve
cross-trajectory extrapolation.

The oracle decomposition was also corrected. An older evaluator accidentally
read “true-maplet spatial components” from the identity bank, whose component
geometry collapses to the maplet center. It now reads them from the separate
spatial bank:

| Region-center diagnostic | Median translation / rotation | Within 30 cm / 3 deg |
| --- | ---: | ---: |
| nearest exact visible surface | 13.57 cm / 0.33 deg | 10/12 |
| true identity + oracle spatial component | 11.87 cm / 0.48 deg | 11/12 |
| retrieved true identity + oracle spatial component | 27.25 cm / 0.93 deg | 6/12 |
| true identity + descriptor-MAP spatial component | 58.18 cm / 2.01 deg | 2/12 |
| true maplet center | 1.25 m / 3.07 deg | 0/12 |

This corrected split is decisive: clean geometry is not the dominant failure.
Identity/component selection and the region-level pose proposal lose most of
the basin coverage. Even the two oracle rows remain region-center PnP
diagnostics and do not establish the final 4 cm observation model.

Ground-truth pose is also scored strictly after hypothesis generation as a
diagnostic only. With the oriented quad footprint, GT has higher likelihood
than every generated proposal on 10/12 queries. This shows that proposal
coverage/search is the dominant failure on most queries. On 2/12, a
repetitive wrong structure still has higher footprint likelihood, so merely
widening CEM cannot be promoted safely.

A fixed two-query CEM check confirmed that warning: it remained 0/2 inside
30 cm / 3 degrees (Top-1 median 4.29 m / 14.06 degrees), and its optimized
false modes outscored GT footprint likelihood on both queries. The footprint
optimizer is therefore retained as an experimental diagnostic, not selected
as the default coarse method.

The exact-surface and surface-component PnP rows in the dashboard are
observation-model diagnostics. They use RADIO region centers and therefore do
not establish final metric accuracy. Earlier tables that described regional
MAP-PnP as deployable are historical and invalid under the corrected method
definition.

## MapletFrameAlignment strict M1/M2/M3 audit

The latest audit uses clean 2DGS geometry, map trajectories
`seq1/2/4/6/7/8`, training trajectories `seq9/10/12/14`, calibration
trajectory `seq11`, and the fixed test trajectories `seq3/5/13`. Test
trajectories are disjoint from every map, training and calibration source.
Ground truth selects/scores the correct chart only in the explicitly named
oracle diagnostics and is never an input to frame-mode generation.

This audit corrected seven implementation/metric errors that could previously
hide the real bottleneck:

- dominant-region and visible-region metrics now traverse the many-to-many
  `RegionChartIndex`; chart IDs are never assumed to equal region IDs;
- nearby spatial posterior modes merge probability and covariance, while only
  distinct Top-M truncation contributes to null;
- frame NMS is joint over translation and affine state, so different
  scale/rotation/shear modes at one center are not accidentally deleted;
- disconnected composite charts retain bounded per-texel appearance mixtures,
  and refinement marginalizes those modes rather than collapsing them to one
  mean descriptor;
- cross-scale scores no longer include the invalid large-template support
  reward;
- grouped mode enumeration no longer duplicates every single-chart pose or
  truncate the intended per-chart Top-5 modes before grouping;
- evidence from independent charts is accumulated rather than averaged.

The generated Region→Chart index has 864 retrieval regions, 734 metric charts
and 2,582 edges. 468 regions contain multiple charts, while 573 charts can be
recalled by multiple regions. Charts per region have P10/P50/P90 of
0/2/8; physical chart texel size is 0.49/2.03/5.80 cm. This verifies the
many-to-many geometric representation, but the retrieval descriptors are
still the explicitly labelled single-scale proxy described above.

Scene evidence improves recall without double-counting overlapping query
regions:

| Aggregation | Dominant region R@16 | Visible surface @16 | Dominant region R@64 |
| --- | ---: | ---: | ---: |
| legacy unbounded sum | 8/12 | 51.14% | 11/12 |
| Top-q after NMS | 9/12 | 53.05% | 11/12 |
| noisy-OR after NMS | 9/12 | 53.42% | 11/12 |
| block balanced | 9/12 | 55.66% | 11/12 |

M1 holds region identity correct and measures whether the map/query feature
field recovers the chart's 2D frame. “Center” is only the projected chart
center; “control” requires the scale/orientation-bearing frame controls and is
the relevant gate:

| Correct-identity feature source | Examples | Center R@16 (32 px) | Control R@16 (32 px) | Median best control error @16 |
| --- | ---: | ---: | ---: | ---: |
| old support-biased frozen mapper | 12 charts | 11/12 | 2/12 | 73.09 px |
| corrected frozen mapper | 48 charts | 46/48 | 33/48 | 26.87 px |
| corrected frozen mapper + local refinement | 48 charts | 44/48 | 33/48 | 25.99 px |
| corrected fine metric student | 36 charts | 21/36 | 11/36 | 37.27 px |

The score correction is a material result, not a small parameter movement:
frozen-map chart-control R@16 rose from 16.67% to 68.75%. Its Top-5 center
error is 10.35 px median, while full frame controls remain 26.87 px; scale and
orientation are therefore now the dominant visual errors. The trained
stride-4 student is worse despite its finer grid, confirming that its current
pointwise global-location objective did not learn the required structured
frame field. Disconnected composite refinement remains invalid as a main
route.

M2 bypasses visual frame prediction and feeds exact projected controls to the
same frame-to-pose solver:

| Frame model | 1 chart | 2 charts | 3 charts |
| --- | ---: | ---: | ---: |
| oriented similarity, within 30 cm / 3 deg | 2/12 | 7/12 | 12/12 |
| affine, within 30 cm / 3 deg | 12/12 | 12/12 | 12/12 |
| homography, within 30 cm / 3 deg | 12/12 | 12/12 | 12/12 |
| oriented similarity median translation | 1.080 m | 0.161 m | 0.013 m |
| affine median translation | 0.681 mm | 0.537 mm | 0.025 mm |

Thus the clean geometry, chart controls and grouped pose solver are sufficient
when the visual frame is correct. M3 replaces the oracle frame with predicted
frames and keeps charts independent:

| M3 source | Basin recall @1/@5/@16 | Bounded candidate oracle | Top-1 median translation / rotation |
| --- | ---: | ---: | ---: |
| correct identities, four independent charts | 0/12, 0/12, 1/12 | 3/12; 0.54 m / 2.59 deg median | 2.88 m / 11.48 deg |
| actual Top-16 retrieval, up to six charts | 0/12, 0/12, 0/12 | 1/12; 1.40 m / 5.77 deg median | 10.91 m / 48.80 deg |

The candidate oracle is scored with ground truth only after a bounded set of
at most 1,024 frame-generated pose modes exists. The three correct-identity
successes and the one actual-retrieval success come from a separately labelled
multi-chart-center diagnostic. Those centers are projections estimated by
whole-chart atlas correlation—not RADIO region centers, stable anchors or
local point descriptors—but this diagnostic is not promoted as the final
observation model.

The new results show that valid signal now survives identity→frame→pose, but
mode precision and ranking remain insufficient. Meter-level Top-1 errors are
geometric amplification of wrong scale/orientation modes, not evidence that
the 2DGS map itself is meter-inaccurate. The first-principles bottleneck is now
localized to:

```text
context-rich retrieval-region identity
  -> query-conditioned metric-chart appearance field
  -> precise translation/scale/rotation/shear modes
  -> calibrated multi-chart geometric ranking
```

An 80-step full-query hard-negative encoder experiment selected step 20, but
trajectory-disjoint `seq11` radius-1 R@5 remained only 25.00% / 13.28% /
6.38% at coarse/middle/fine. It plateaued and was not promoted.

Key immutable results:

| Artifact | SHA-256 |
| --- | --- |
| corrected four-chart M1/M3 (`radio_mapper_four_chart_additive_evidence_m3_strict12_v23.json`) | `6f5415e5271f1f6fa680aaecb09d34cfb89c82073e12dd5ba5e5e9facd9bd6e5` |
| actual-retrieval M3 (`radio_mapper_runtime_retrieval_additive_center_m3_strict12_v24.json`) | `301a887d5f7d6a444575b066c91f71b09a432edfa5ccce04713b96cc4f947703` |
| corrected fine-student diagnostic (`frame_alignment_fine_scale_unbiased_independent_chart_strict12_v19.json`) | `84032fa22390c7ae8abaee657fddd042d4287354ccd611eba1876227ba9b74b4` |
| Region→Chart index | `95875d5500244b0f93790143d60b5c4ab3d948809c0d08a84b42f735cc221208` |
| hard-negative encoder | `46c2faf46c7eaff9e78d77346f7d98dbcd47d072fa27ccbbcbecf67d5880806a` |

The next justified implementation is a genuinely context-trained
RetrievalRegion representation plus structured joint map/query frame learning
whose loss is defined on complete chart transforms—not independent texel
classification—with rotation/scale/homography perturbations and
repeated-facade hard negatives. Pose-mode confidence must then be calibrated
on runtime retrieval replay.
Typed correlation nulls, P3 proposal integration and Stage-C fine alignment
remain downstream work; advancing them before M1 control recall passes would
only optimize around a broken observation model.

## Promotion policy

The 530-query run is not permitted while the following remain false:

- G0: sufficient adaptive atlas coverage, chart quality and full-scene
  visibility;
- G1: trajectory-disjoint typed-null correlation and reliable flow direction;
- G2: two or three verified fine iterations improve fixed 5/20/30 cm starts;
- G3: actual retrieval and region-level proposal put enough strict queries
  inside the 20–30 cm / 3 degree basin;
- G4: only then run and report all 530 end-to-end queries.

Scientific diagnostics continue below a gate, but a failed component cannot be
renamed production or used to claim final accuracy.

## Active implementation

- `feature_extract/tools/vfm/build_surface_retrieval_maplet_bank.py`
- `feature_extract/tools/vfm/calibrate_v6_retrieval_probabilities.py`
- `feature_extract/tools/vfm/build_v6_region_chart_index.py`
- `feature_extract/tools/vfm/bake_v6_retrieval_atlas.py`
- `feature_extract/tools/vfm/build_v6_anonymous_pose_vote_bank.py`
- `feature_extract/tools/vfm/evaluate_v6_retrieval_pose_basin.py`
- `feature_extract/tools/vfm/evaluate_v6_maplet_frame_alignment.py`
- `feature_extract/tools/vfm/evaluate_v6_radio_frame_source.py`
- `feature_extract/tools/vfm/train_v6_global_frame_encoder.py`
- `feature_extract/vfm/localization_v6/maplet_retrieval.py`
- `feature_extract/vfm/localization_v6/map_entities.py`
- `feature_extract/vfm/localization_v6/maplet_frame_alignment.py`
- `feature_extract/vfm/localization_v6/maplet_pose_voting.py`
- `feature_extract/vfm/localization_v6/maplet_footprint_pose.py`
- `feature_extract/vfm/localization_v6/maplet_pose_proposal.py`
- `feature_extract/vfm/localization_v6/probability_calibration.py`
- `feature_extract/vfm/localization_v6/atlas_renderer.py`
- `feature_extract/vfm/localization_v6/local_correlation.py`
- `feature_extract/vfm/localization_v6/se3_update.py`
