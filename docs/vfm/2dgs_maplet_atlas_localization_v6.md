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
  -> normalized P(chart | query) through the many-to-many RegionChartIndex
  -> adaptive set of independent MetricSurfaceCharts
  -> per-chart full-query translation / scale / rotation / shear modes
  -> fixed-support continuous projective refinement
  -> one correlated eight-dimensional control factor per chart
  -> bounded coarse SE(3) modes
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

Region scores are not summed over every adjacent edge. The runtime expansion
first normalizes `P(chart | region)` for each region and then computes
`P(chart | query) = sum_region P(region | query) P(chart | region)`. This
removes graph-degree bias and uses cumulative posterior mass with a bounded
12--24 chart budget. Cross-scale hypotheses are compared by mean regional
correlation. The older `support**0.35` factor was methodologically invalid:
it rewarded larger projected templates even when their mean match was worse.

IPPE/EPNP may convert the four projected controls of one or more charts into
an initial SE(3) seed. They are not fed independently matched points, stable
point identities, ALIKE descriptors or SfM tracks. Every seed is refined and
ranked by robust correlated chart-block factors. This seed conversion does
not make V6 a point-correspondence-PnP localization route; the final estimator
remains atlas correlation and continuous surface alignment.

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

The runtime evaluator now closes the intended loop:

```text
coarse pose mode
  -> select pose-visible disconnected maplets
  -> render maplet atlases
  -> local displacement distributions
  -> robust SE(3) proposals
  -> maplet-disjoint fit/held-out acceptance
  -> re-render at the accepted pose
```

This makes Stage C executable, but not yet accurate. The current
cross-trajectory smoke accepted no SE(3) update; atlas evidence only improved
the ranking of coarse modes. A connected implementation is not a passed G1/G2
gate, and a one-query smoke is not an end-to-end accuracy result.

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

## Route-repair D1--D6 audit

The current strict audit supersedes the preceding V24 M3 row for the active
implementation. It uses the same 12 trajectory-disjoint queries and the
clean-2DGS lineage, removes the chart-center SQPnP diagnostic from candidate
generation, and separates identity, frame observation, geometry and ranking.

The implementation fixes the following protocol errors:

- Region-to-Chart expansion is a normalized conditional posterior instead of
  a degree-biased sum of every neighboring region score;
- the chart budget covers 95% posterior mass subject to a 12--24 bound instead
  of truncating unconditionally to six;
- local frame refinement uses the observed support hull, a fixed canonical
  denominator, an explicit out-of-view penalty and a full homography;
- four chart corners form one correlated chart factor rather than four
  independent pseudo-points;
- IPPE/EPNP is labelled and retained only as a regional seed mechanism;
- raw cosine-score addition and the grouped chart-center SQPnP diagnostic no
  longer determine the reported production candidate set;
- Stage C is called by the actual runtime evaluator and saves every
  render/correlate/update/verify decision.

The decomposition is:

| Gate | Strict-12 result |
| --- | ---: |
| D1 dominant region in retrieval Top-16 | 9/12 |
| D1 dominant chart after normalized expansion, @12 / @24 | 5/12 / 7/12 |
| D1 visible-surface coverage, @12 / @24 | 21.78% / 40.73% |
| D2 actual retrieval charts + GT projective frames, basin @1/@5/@16 | 12/12 / 12/12 / 12/12 |
| D3 correct chart identity + best generated frame mode, candidate oracle | 0/12; 2.46 m / 6.99 deg median |
| D4 actual retrieval + best generated frame mode, candidate oracle | 0/12; 0.95 m / 3.72 deg median |
| actual runtime frame modes, bounded candidate oracle | 1/12; 1.12 m / 3.57 deg median |
| actual runtime frame modes, Top-1 | 5.78 m / 19.03 deg median |

D2 is the decisive control: although the dominant chart is not always
retrieved, the selected set contains other visible charts sufficient for the
same chart-factor solver on every query. D3 then fails even when chart identity
is supplied. Therefore retrieval coverage should still improve, but it is not
the immediate cause of the meter-scale pose error. The missing variable is a
precise, cross-trajectory chart-frame probability distribution.

Correct-identity local refinement now has 48 examples, control R@16 (32 px)
of 37/48 and median best-control error of 21.85 px. This threshold remains far
too loose for metric pose initialization. The geometry-only D6 control also
shows that affine-versus-homography approximation is small for these charts:
0.19 px median and 0.86 px P90 over 48 charts. Projective support is still the
correct representation, but it is not the current bottleneck.

One Stage-C connectivity smoke on `seq13/frame00001.png` used a diverse
64-mode coarse pool and refined the atlas-ranked Top-4. Atlas ranking changed
the selected pose from 1.10 m / 3.11 deg to 0.54 m / 1.56 deg, but no proposed
SE(3) update passed held-out verification. A 0.093 m / 1.16 deg coarse mode
existed only at raw rank 158, outside that render budget. This is evidence that
both frame-mode coverage/ranking and the visual flow likelihood remain weak;
it is not evidence that local atlas alignment can recover an arbitrary
meter-scale initialization.

The D5 local-basin control removes coarse retrieval and frame prediction
entirely. GT selects 12 visible charts and creates only a 5 cm / 0.33 degree
initial perturbation; subsequent rendering, correlation, SE(3) updates and
held-out checks receive no GT. With the original likelihood-only verifier,
7/12 cases accepted an update but only 1/12 improved, six became worse, and
the final translation median/P90 was 5.30/7.12 cm. Thus “accepted by held-out
likelihood” was not equivalent to “geometrically correct.”

The repaired verifier additionally requires fit and maplet-disjoint held-out
sets to estimate a consistent six-dimensional update before committing it.
This reduces accepted updates from 7/12 to 3/12 and final translation
median/P90 to 5.00/5.95 cm, while preserving the one genuine improvement.
However, two consistent updates are still geometrically wrong and the
4 cm / 1 degree success remains only 1/12. The guard is a useful safety fix,
not a solution: the same uncalibrated RADIO atlas likelihood can agree on a
wrong repeated-facade direction across both subsets.

The immutable reports are:

| Artifact | SHA-256 |
| --- | --- |
| `route_repair_d1_d4_strict12_merged_v28.json` | `41bcb6e49a94ff6ad18c59b853e3b4147a11737616d35fa36f0b5f40cd9f7fe8` |
| `route_repair_d6_projection_gap_strict12_v29.json` | `da4bfb012b43d1ac070b9c08a1e63964d992c2ba9d231f734ddc063a6ea562f6` |
| `route_repair_stage_c_prerank_smoke1_v27.json` | `1362a9a9b51211aa5937c327fd88aaf7800b26ef4d4ef72bf12921c96fea7d43` |
| `route_repair_stage_c_oracle_basin_strict12_merged_v31.json` | `c4d2490c58c53213401c5d6705a91d7952178d674761467c6b3706ace001afe4` |
| `route_repair_stage_c_consensus_basin_strict12_merged_v33.json` | `c8b93cb53ee0c082f377ae159611f844af494cd33b70d4ebb2dddbb22f7c6cd7` |

The next promotion target is not another candidate-count or threshold sweep.
It is a shared RADIO-final structured frame matcher trained on complete
translation/scale/rotation/projective modes, with chart-level soft-min
likelihood, repeated-facade hard negatives, explicit null outcomes and
trajectory-disjoint deployment replay. Until D3 begins to enter the coarse
basin and M1 control error falls to roughly 8--12 px, full Stage-C and
530-query runs are intentionally blocked.

A bounded 400-step precursor was implemented to test whether a query-side
linear RADIO metric adapter is sufficient. Its loss consumes complete chart
transform candidates (translation, rotation, scale, shear and global facade
modes), not independent texel labels. On 192 fixed `seq11` validation
episodes, transform Top-1 improved from 45.83% to 52.08%, but the hardest-mode
margin remained negative and control error changed only from 1.87 to 1.83
stride-16 cells. The strict `seq13/frame00001` replay then failed the real
promotion gate: refined M1 control error worsened from 9.23 px to 17.40 px,
D3 candidate-oracle translation worsened from 0.58 m to 0.96 m, and basin
recall remained zero. The adapter is therefore retained as a diagnostic and
is not enabled by default. A shallow channel metric cannot substitute for a
model that predicts a calibrated structured projection distribution and
typed nulls.

| Structured-training artifact | SHA-256 |
| --- | --- |
| `structured_frame_adapter_chart_transform_stable_v36.json` | `a5206b2fb8644b3d8448eb447d6012fc39ac4c1a691eb8903e8c0bad305140fd` |
| `structured_frame_adapter_strict_smoke1_v37.json` | `3d6bc3508353123f4e09458450fe9a42b5831f37074573d710d78e6cb324746c` |

## Repaired complete-pool Stage-C result

The latest strict run uses the clean 2DGS source
`/root/StMaryChurch2dgs_clean.ply`, a stride-8 RADIO-final atlas, 4,096
structured frame-derived pose hypotheses, a complete sparse Top-64 atlas
screen, the preserved four-exact/four-local candidate union, 24 fixed scoring
charts, 12 pose-visible refinement charts, three render/correlate/update
rounds and at most one accepted translation-bearing update. ALIKE supplies
detector scores only; no ALIKE descriptor is computed or stored.

| Strict-12 metric | Result |
| --- | ---: |
| Top-1 translation median / P90 | 0.405 m / 0.730 m |
| Top-1 rotation median / P90 | 0.665 deg / 2.551 deg |
| Top-1 recall at 30 cm / 3 deg | 2/12 |
| final eight-candidate oracle median | 0.293 m / 0.703 deg |
| final eight-candidate oracle recall at 30 cm / 3 deg | 7/12 |
| complete approximately 4k-pose oracle median | 0.279 m / 0.940 deg |
| complete-pool oracle recall at 20 cm / 3 deg | 4/12 |
| complete-pool oracle recall at 30 cm / 3 deg | 7/12 |

The complete-pool and sparse Top-64 oracle results are identical. The GPU
broad screen therefore does not remove a useful basin. Compression from 64
to the exact/local eight loses useful alternatives on individual queries,
and final ranking loses additional accuracy, but neither can lower the
complete-pool 27.9 cm median by itself.

A target-labelled component diagnostic selected the 16.0 cm / 1.92 degree
q7 pool mode. With target information removed after selection, the normal
Stage-C renderer, RADIO correlation and held-out verifier left the mode
unchanged. This establishes that Stage C can preserve a good basin; the mode
was lost by finite selection, not corrupted by the SE(3) optimizer.

An optional trajectory-disjoint score calibration path was implemented as a
safe supplement to the exact/local union, never as its replacement. It uses
within-query ranks of fixed-chart exact evidence, fit and held-out evidence,
local displacement-marginal evidence and the structured-frame coarse score.
The fitter requires at least two calibration trajectories, leaves out whole
trajectories during cross-validation and refuses all-negative candidate
pools. The available six-frame `seq11` replay was correctly rejected: none
of its complete Top-64 pools contained a 30 cm / 3 degree mode and their best
errors ranged from 0.57 m to 3.00 m. No score-calibration artifact was
promoted and the strict result was not tuned with target poses.

The immutable reports are:

| Artifact | SHA-256 |
| --- | --- |
| `stagec_strict12_score24_refine12_trans1_merged_v264.json` | `ecd2cbc0a28a7c9c58eeea3095f7ea8b4559c7070369700c64ddcf325d54f91e` |
| `stagec_q7_pool3903_16cm_diagnostic_v265.json` | `87f8a0165555b33a465f8ecf923a198a6a8ddce5feec9176527d9b506d848a4f` |
| `stagec_q7_full64_round0_v269.json` | `c6e2fd8a9d6d7363309afb69d3c2a45d73aaf65a4884583df60dcedc2aebfaf9` |
| `frame_modes_seq11_calibration6_v261.json` | `90cd7d8d06cb3d4c24e31a86f0778cf7650068596b40e564ccf318d9fce4b8bc` |

This run removes the previous meter-scale implementation failure, but it is
not a promotion result. The remaining first-order error is the structured
chart-frame observation: even perfect selection from the generated pool is
about 28 cm median. The next high-value experiment must improve the
query-conditioned projective chart distribution and runtime correlation
null/flow calibration on multiple training trajectories. More candidate
counts, translation steps or strict-set score tuning are explicitly not
justified. The full 530-query run remains blocked by G1--G3.

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
- `feature_extract/tools/vfm/replay_v6_stage_c.py`
- `feature_extract/tools/vfm/diagnose_v6_stage_c_pool_coverage.py`
- `feature_extract/tools/vfm/merge_v6_stage_c_reports.py`
- `feature_extract/tools/vfm/fit_v6_stage_c_score_calibration.py`
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
