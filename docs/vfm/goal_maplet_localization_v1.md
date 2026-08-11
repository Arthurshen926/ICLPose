# Goal-Maplet Structured Physical-Surface Localization

## Decision

Goal-Maplet replaces V6 as the active research namespace, but it is **not yet
paper-ready as an accuracy method and is not promoted over the frozen V3
production baseline**. The work establishes a much cleaner physical-map and
evaluation foundation and recovers useful Top-N coarse-pose coverage. It does
not yet turn those candidates into a reliable Top-1 pose or a convergent local
refinement.

This distinction is important:

- the map geometry, storage contract, exact contributor labels and oracle
  geometry are now credible;
- physical-region retrieval is useful and produces pose-sufficient sets;
- configuration proposal reaches a decimetre-scale candidate on most queries;
- configuration ranking and continuous VFM alignment remain the blocking
  scientific problems.

The current result supports a paper about a **mid-scale structured
representation and its diagnostic/oracle analysis** only if the ranking or
handoff gap is subsequently closed. It does not support a claim of superior
final localization accuracy today.

## Frozen method question

The active question is:

> Can one exact clean-2DGS map containing a single canonical VFM feature type
> retrieve physical maplet configurations and generate high-coverage Top-N
> 6DoF pose modes, without mapping RGB, SfM tracks, point-landmark banks,
> LoFTR, RADIO intermediate features or stored ALIKE descriptors?

The runtime map contains no mapping RGB, mapping image path/ID, per-view
descriptor list, downstream teacher embedding, SfM point/track, LoFTR match,
RADIO intermediate feature or ALIKE descriptor. ALIKE is permitted only as an
optional query-side detector; its tested detector-only/RADIO-only refiner was
rejected and is not in the active path.

## Architecture

```text
clean 2DGS primitives
  -> exact primitive membership
  -> small connected child surface tiles with local coordinates
  -> overlapping context parents
  -> one canonical RADIO-final-derived primitive field
  -> regenerable parent/child readouts
  -> typed physical graph

query RADIO-final
  -> all tokens
  -> calibrated multi-positive parent posterior
  -> grouping after retrieval (correlated evidence is averaged)
  -> parent-conditioned child surface posterior
  -> graph-conditioned configuration modes
  -> robust medium-scale region initializer
  -> diverse Top-16/Top-32 SE(3) modes
  -> optional ranking/refinement plugin
```

The numerical initializer may call PnP internally, but its observations are
medium-scale physical regions/tiles. The map has no keypoint landmark bank,
tracks or point descriptors, and final method identity is not defined by a
solver API.

## What was implemented

### G0 — isolated evidence line

- Independent code namespace: `feature_extract/vfm/localization_goal_maplet/`.
- Independent artifacts: `output/.../goal_maplet/`.
- Strict lineage checks bind physical geometry, canonical field, typed graph,
  calibration and split role.
- Strict12 is regression/diagnostic only.
- A trajectory-disjoint Dev48 uses 16 frames each from seq3, seq5 and seq13.

### G1 — exact physical hierarchy

The physical map contains 509,572 clean primitives, 864 context parents and
7,653 child surface tiles. Parent/child membership is exact CSR over original
clean-2DGS primitives; centers, normals and extents are derived summaries.

Audit results:

| Geometry audit | Result |
|---|---:|
| child normal-dispersion P90 | 20.88° |
| disconnected child fraction | 6.57% |
| depth-layer excess fraction | 7.79% |
| physical-map SHA-256 | `19a34e8c...f8f7e6e` |

Signed sidedness, full-scene occlusion and exact dominant contributors replace
unconditional absolute-normal visibility and summary-box proxies.

### G2 — PFIR and calibration

PFIR truth is the evaluator-only top-k clean-2DGS contributor distribution.
Each support is multi-positive. Truncated probability mass goes to null, and
correlated grouped tokens are averaged rather than multiplied.

| Exact grouped PFIR | Strict12 | Dev48 |
|---|---:|---:|
| weighted R@1 | 0.345 | 0.312 |
| weighted R@5 | 0.599 | 0.567 |
| weighted R@20 | 0.771 | 0.738 |
| weighted R@64 | 0.860 | 0.828 |
| whole-image visible coverage@64 | 0.865 | 0.842 |
| pose-sufficient set@64 | 1.000 | 1.000 |
| null ECE | 0.152 | 0.133 |

The result says retrieval is useful but physical-instance R@1 and calibration
are not solved. High pose-sufficient R@64 justifies multi-modal configuration
proposal; it does not justify forcing a retrieval Top-1 pose.

### G3/G4 — all-token support and mapper contract

- Retrieval runs on every RADIO token before grouping.
- Spatially adjacent tokens merge only after retrieval when identity and
  feature correlation agree.
- Group posterior uses average evidence and a complete null branch.
- The frozen `SurfaceMapletMapper` remains the retrieval adaptor. It is not
  treated as a local-coordinate or pose-regression head.
- Earlier pooling ablations show increasing context improves retrieval over
  1x1, but multi-scale support masks are evaluated as their exact weighted
  union rather than one average-size box.

### G5 — canonical field contract audit

The exact deployed retrieval field contains 298,705 directly observed
primitives (58.62% of the clean scene), with 98.15% non-empty parents and
95.51% non-empty children. It stores one 128-D RADIO-final-derived code and no
downstream embedding list.

The audit found that the v2 builder stored the retrieval mapper output while
calling it canonical RADIO-final. A task-neutral mapping-only PCA codec was
therefore implemented and evaluated:

- input: normalized 1280-D RADIO-final;
- output: one 256-D canonical code;
- fit: 122 mapping views only, excluding seq11/seq3/seq5/seq13;
- explained variance: 92.64%;
- codec SHA-256: `16ffa07b...fc8ceac`;
- field SHA-256: `5848ea4e...96bf0cb`.

This corrected representation contract but did not by itself provide a local
alignment basin, so it is a research artifact rather than the retrieval
default.

### G6 — typed graph

The graph contains 864 parents and 78,369 typed edges:

| Edge type | Count |
|---|---:|
| geometry adjacency | 4,228 |
| surface continuity | 2,064 |
| mapping co-visibility | 70,393 |
| distinctive context | 1,684 |

Same-parent compatibility is neutral rather than zero. Feature-nearest edges
do not replace the physical graph. Mapping view identities and observations
are not stored in the graph artifact.

### G7/G8 — child surface and Top-N proposal

The oracle ladder separates point-center approximations from continuous
surface information:

| Oracle diagnostic (Strict12) | translation median | P90 |
|---|---:|---:|
| exact primitive-center geometry | 0.171 m | 0.283 m |
| exact ray–surfel intersection | 0.000 m | 0.000 m |
| grouped parent center | 0.335 m | 0.529 m |
| grouped child center | 0.250 m | 0.404 m |
| current grouping + oracle child-local continuous surface | **0.020 m** | **0.081 m** |

Thus the physical hierarchy contains the information needed for the target;
the loss occurs in visual configuration inference and refinement.

Pair-seeded typed-graph configurations modestly improve candidate diversity on
Dev48. Current proposal coverage is:

| Dev48 proposal metric | Result |
|---|---:|
| Top-16 best translation median / P90 | 0.255 / 0.641 m |
| Top-32 best translation median / P90 | **0.241 / 0.489 m** |
| Top-16 ≤1m/10° coverage | 97.92% |
| Top-32 ≤1m/10° coverage | **97.92%** |
| Top-16 ≤0.5m/5° coverage | 85.42% |
| Top-32 ≤0.5m/5° coverage | **89.58%** |

This passes a useful 1 m handoff gate but does not reach universal 0.5 m
coverage.

### G8 — ranking probability audit

The rendered identity ranker double-counted context evidence because it scored
both `P(parent)` and the joint `P(parent,child)`. It now scores
`P(parent)` and `P(child|parent)` separately with a fixed all-token
denominator.

The semantic correction helped Strict12 but failed the independent Dev48 gate:

| Top-1 | frozen graph v9 | corrected conditional rank |
|---|---:|---:|
| Strict12 translation median | 0.649 m | **0.518 m** |
| Strict12 translation P90 | 2.502 m | **1.514 m** |
| Dev48 translation median | **0.555 m** | 0.660 m |
| Dev48 translation P90 | **1.680 m** | 1.990 m |
| Dev48 rotation median | **1.994°** | 2.100° |

The corrected ranker is therefore not enabled by default. Correct
factorization alone is insufficient: the child conditional likelihood is not
calibrated as a pose likelihood and child-splat geometry is approximate.

### G9 — refinement basin and render round-trip

Several refiners were tested and rejected:

- multiview-averaged mapper-code surface correlation;
- trained child local head;
- ALIKE detector-only + RADIO-only primitive matching;
- parent-center then pose-conditioned child refinement;
- uncalibrated joint local-evidence configuration factor.

The initial continuous renderer also violated its training protocol: field
contributors were rasterized at 144x256 and associated with 36x64 RADIO
tokens, while refinement rendered 2DGS directly at 36x64. Exact 4x rendering
followed by mask-aware token pooling fixed the GT acceptance failure:

| One-frame oracle-child diagnostic | low-resolution | exact 4x round-trip |
|---|---:|---:|
| GT final drift | 0.150 m / 1.76° | **0 / 0** (bad step rejected) |
| GT alignment score | 0.238 | 0.328 |
| median spurious flow | 0.51 token | 0.24 token |

However, a 10 cm perturbation was also rejected rather than improved. A 2x
flow has useful direction on that frame (10 cm to 4.41 cm) but drifts from GT;
using 4x acceptance does not close the loop. The current flow-to-SE(3)
refiner therefore has no validated convergence basin and is not promoted.

### G10 — frozen contracts, independent ranking and child-local likelihood

The latest pass followed the ranking/measurement gates without changing the
exact map, retrieval or graph backbone.

#### P0 implementation corrections

- A fail-closed field/readout contract now binds field SHA-256, query readout
  type/SHA-256 and render protocol. Evaluators no longer infer the query path
  from feature dimension. The retrieval contract SHA-256 is
  `f64a5d7b...e42664`.
- Child geometry has nested retrieval/proposal/refinement eligibility masks.
  Of 7,653 children, 95.51% / 92.53% / 70.01% pass the three roles. The
  eligibility SHA-256 is `42dd3d6a...32daa0`.
- Proposal RNG is now `sha256(image_id)` masked to a signed 31-bit OpenCV
  seed. This fixes both shard-dependent candidates and unsigned-C-int
  overflow on roughly half of image hashes.
- Configuration density medians use `nanmedian`; the earlier implementation
  included each candidate's NaN self-distance and silently zeroed both median
  distance features.
- Exact-cascade reranking now permutes pose and every diagnostic together and
  stores an explicit exact-evaluated mask. Previously exact Top-K reordered
  poses while leaving cheap features in the old order.
- Front-to-back 2DGS contribution compositing is vectorized. A reference-loop
  equivalence test confirms identical candidate score/coverage on overlapping
  exact renders.

These are implementation fixes, not accuracy claims.

#### P1 independent configuration ranker

A fixed 128-query mapping pool was split by trajectory: seq1/2/4/6/7/8 train,
seq9/10/12/14 validation, and seq11 excluded because it calibrated validity.
The ranker uses 16 runtime-only relative features and no GT, absolute position,
mapping RGB, image identity or stored downstream embedding. Logistic and one
small GBDT were compared once; logistic passed mapping validation but not the
Dev48 promotion gate.

| Frozen Dev48 Top-1 | translation median | P90 | rotation median | P90 | catastrophic >2m or >10° |
|---|---:|---:|---:|---:|---:|
| graph proposal | 0.555 m | **1.680 m** | 1.994° | **5.561°** | 8.33% |
| cheap identity | 0.650 m | 2.148 m | 2.170° | 5.810° | 12.50% |
| learned logistic | **0.515 m** | 1.690 m | **1.917°** | 6.455° | **6.25%** |
| Top-32 oracle | 0.242 m | 0.538 m | — | — | — |

The learned ranker improves median and catastrophic rate, but its 0.263 m
median selection regret and slightly worse translation/rotation tails fail the
5–8 cm regret and sub-metre P90 gates. It is retained as a diagnostic artifact,
not promoted over graph proposal.

Exact full-2DGS Top-4 reranking was also audited on every Dev48 query. It fired
on 37/48 cascade gates and produced 0.627 m / 1.986 m translation, worse than
the graph proposal. The exact renderer is trustworthy geometry evidence, but
raw exact identity remains a retrieval likelihood rather than pose
correctness. An explicit exact-feature ranker contract and always-exact mask
are implemented, but an exact Top-8 mapping pool was stopped: same-GPU gsplat
kernels serialized, no shard completed its first query after 2m17s, projected
runtime exceeded the evidence value, and cheap Top-4's 0.339 m oracle already
bounded that experiment above the required accuracy. No incomplete artifact
is used.

#### P2 parent-conditioned child-local surface likelihood

A new readout conditions one query descriptor on one candidate child and
scores only that child's exact primitive membership. It outputs a multi-modal
primitive distribution, MAP/expected 3D points, child-local covariance and
null. It reuses the single canonical primitive field and stores no dense ALIKE
map, point landmark descriptor bank or second per-map embedding.

The measurement was first evaluated with oracle child identity so child-ID
errors could not be confused with local-coordinate quality. Only
refinement-qualified children participate.

| Dev48 oracle-child diagnostic | query-balanced point median / P90 | pose translation median / P90 |
|---|---:|---:|
| child center | 0.299 / 0.406 m | 0.326 / 0.459 m |
| VFM primitive MAP | 0.234 / 0.338 m | 0.273 / 0.422 m |
| coarse-pose conditioned MAP | **0.224 / 0.328 m** | 0.279 / 0.438 m |
| learned Top-8 mode | 0.230 / **0.326 m** | **0.265 / 0.406 m** |
| Top-8 local-mode oracle | **0.151 / 0.222 m** | — |
| exact child-local surface | — | **0.016 / 0.053 m** |

The lightweight Top-8 mode GBDT uses only relative VFM, confidence,
visibility, incidence and normalized child geometry features. It consistently
improves child-center pose, but does not reach the requested 0.15–0.20 m
candidate-precision gate. A joint multi-modal solver was rejected at smoke:
its learned mixture score moved one frame from a 0.307 m child-center pose to
0.345 m, showing that the current probabilities are not calibrated pose-factor
likelihoods. It was not swept or run on Dev48.

P2 therefore establishes the right information hierarchy—correct local modes
often exist—but does not close independent mode selection. The next method
change must learn `p(u,v,null | query, child, geometry)` as a calibrated
pose-factor likelihood from exact contributor round-trips, rather than add
another fixed score or average multi-modal coordinates.

### G11 — pairwise utility, typed nulls and configuration factors

This pass retained the frozen physical map, canonical field, mapper, graph and
Top-32 pool.  Gates are now reported separately as correctness, scientific
signal and production promotion; a research module is no longer discarded
merely because it is not production-ready.

#### Correctness and probability semantics

- Child-local VFM modes now use a fixed denominator: `Top-M(query, child)` is
  generated once without candidate-pose geometry.  A pose may evaluate those
  modes but cannot change their identity or VFM probability and then use the
  changed result to validate itself.
- If every primitive is behind the camera or sidedness-incompatible, the
  geometry-conditioned diagnostic returns explicit null/invalid modes instead
  of normalizing identical invalid logits into a spurious uniform posterior.
- Joint multi-modal sampling is opt-in (`joint_pose_trials=0` by default).
- Configuration-factor training fails closed unless its upstream factor
  evidence is trajectory outer-cross-fitted or factor-training-pool disjoint.

The new typed-null target is

```text
valid | wrong_child | pose_incompatible | unresolved | field_missing
```

and is supervised by exact-GT factors, retrieved hard wrong children, frozen
graph Top-1 poses and the nearest out-of-basin Top-32 pose.  The 63,232 mapping
samples store 18 runtime statistics and training identities only; no RGB or
additional map embedding is stored.

#### Query-group pairwise local-mode ranking

The child-local ranker is trained on ordered feature differences and continuous
exact surface error rather than independent best-mode labels.  On mapping
validation its query-balanced point median improves from 0.242 m to 0.210 m.
On frozen Dev48:

| Oracle-child diagnostic | v1 learned mode | pairwise v2 |
|---|---:|---:|
| point median / P90 | 0.230 / 0.326 m | **0.194 / 0.285 m** |
| pose translation median / P90 | **0.265** / 0.406 m | 0.287 / **0.391 m** |
| pose rotation median / P90 | 0.583° / 1.063° | **0.567° / 1.049°** |

The point measurement reaches the 20 cm neighbourhood, while independent
group choices do not produce a more coherent pose.  This isolates the next
gap to group-aware latent-factor inference rather than point-mode recall.

#### Typed-null calibration

The final factor calibrator trains on seq1/2/4/6/7/8, calibrates on seq9 and
selects once on seq10; seq11/12/14 are excluded.  Frozen Dev48 reports NLL
1.012, ECE 0.080 and the following AUPRC:

| null type | AUPRC |
|---|---:|
| valid | 0.477 |
| wrong child | **0.784** |
| pose incompatible | 0.505 |
| unresolved | 0.372 |
| field missing | **0.943** |

Single-factor discrimination is therefore useful for wrong-child and missing
field evidence but weak for repetitive-structure near-miss poses.  In
particular, per-factor near-miss classification accuracy is only 9.0%; a
single region cannot reliably reject a globally wrong facade phase.

#### Configuration-level aggregation and listwise ranking

The configuration ranker now learns within-query pairwise preferences using
continuous utility

```text
translation_m / 0.5 + rotation_deg / 5
```

instead of independent 0.5 m / 5 degree labels.  The base pairwise Dev48
result is 0.535 / 1.591 m, 1.898° / 5.055°, and 6.25% catastrophic failures:
it improves the graph proposal tail but still has 0.252 m median regret.

Complete configuration evidence aggregates 64 groups and four latent children
per group: typed null mass, fixed local-mode uncertainty, reprojection,
retrieval probability, child capacity, eligibility and graph co-visibility.
Mapping training evidence is outer-cross-fitted by trajectory.  On the clean
seq12/14 mapping validation split, adding these factors changes median only
0.389 to 0.388 m but improves P90 1.477 to 1.301 m and P90 regret 0.763 to
0.473 m.

Frozen Dev48 separates the signal from the still-unstable learned fusion:

| Dev48 configuration policy | translation median / P90 | rotation median / P90 | success <=0.5m/5° | catastrophic |
|---|---:|---:|---:|---:|
| graph proposal | 0.555 / 1.680 m | 1.994° / 5.561° | 45.83% | 8.33% |
| pairwise base ranker | 0.535 / 1.591 m | 1.898° / 5.055° | 45.83% | **6.25%** |
| learned cross-fit factor ranker | 0.482 / 1.791 m | **1.622°** / 5.508° | 50.00% | 8.33% |
| predefined mean valid-factor mass | **0.455 / 1.499 m** | 1.667° / **5.058°** | **54.17%** | **6.25%** |

The predefined factor aggregation is the strongest Goal-Maplet research
diagnostic so far and has within-query pair concordance 0.659.  It is not
promoted: median regret remains 0.217 m, the learned fusion does not preserve
its tail gain, and no untouched test block has been opened.  A calibrated
group-capacity latent solver is justified next; another graph-weight sweep or
unconstrained joint PnP is not.

### G12 — fixed-denominator group-aware latent configurations

This pass froze the physical map, canonical field, parent/child retrieval,
typed graph, typed-null calibrator and every Top-M VFM local mode.  It adds no
map embedding, image cache, landmark bank or point matcher.  A candidate pose
can evaluate the frozen modes but cannot regenerate their feature identities
or probability denominator.

#### Protocol and implementation corrections

- G11 fixed the local-mode denominator but still averaged only groups that had
  an eligible child option.  G12 keeps all 64 selected query groups in every
  candidate denominator; missing options are explicit unresolved nulls.  In
  the measured mapping/Dev pools every group happened to have an eligible
  child, so this is a real protocol fix but does not explain the metric change.
- An all-geometry-invalid Top-M set now follows a tested empty-assignment path
  to null instead of indexing a synthetic option.
- Correlated receptive fields are de-duplicated without map identity.  A first
  connected-component implementation was rejected because overlap chains
  merged distant supports.  The retained complete-link rule requires every
  pair in a support cluster to have at least 0.5 box IoU.  Each cluster has unit
  evidence mass; the median 64-group query has 45 effective groups (typical
  10th--90th percentile 41--51 across mapping and Dev).
- Every group chooses one fixed child-local primitive mode or typed null.
  Independent support clusters cannot reuse one primitive.  Child capacity is
  derived from projected tile area relative to query support area and bounded
  to 1--6 clusters.  The solver is deterministic and does not update SE(3).

The implementation reports the old eligible-only mean, a fixed unweighted
mean, a correlation-corrected mean, uncapacitated latent assignment and
capacitated latent assignment separately.  This prevents a denominator fix,
support de-correlation and capacity constraint from being credited to one
another.

#### Mapping outer-cross-fit gate

All 106 mapping diagnostics use trajectory outer-cross-fitted typed factors;
seq12/14 (17 queries) remains the configuration validation partition.

| Predefined mapping policy | translation median / P90 | success <=0.5m/5° | catastrophic |
|---|---:|---:|---:|
| graph proposal | 0.533 / 2.144 m | 45.28% | 14.15% |
| legacy valid-factor mean | 0.434 / 1.535 m | 55.66% | 8.49% |
| correlation-corrected mean | **0.418** / 1.535 m | **57.55%** | 8.49% |
| uncapacitated latent | 0.489 / 1.486 m | 50.00% | 6.60% |
| capacitated latent | 0.456 / **1.442 m** | 52.83% | **5.66%** |

Capacity is active rather than cosmetic: it changes 73% of candidates in the
seq6/7/8 cross-fit shard.  It behaves as a safety/tail factor on the combined
mapping pool, but on clean seq12/14 validation it does not improve catastrophic
rate and reduces success.  It is therefore retained as evidence, not selected
as the primary Top-1 policy.

A low-capacity basin head and catastrophic head were trained as a correction
bounded to 0.25 of each query's base-score IQR.  Candidate-level validation
AUPRC is 0.959/0.968.  The correction deliberately leaves validation Top-1
unchanged.  It improves validation risk ordering at 70% coverage to 0.201 /
0.766 m, 75.0% success and zero catastrophic errors, but this full risk gain
does not transfer unchanged to Dev48.

#### Frozen Dev48 result

| Dev48 policy | translation median / P90 | rotation median / P90 | success <=0.5m/5° | catastrophic |
|---|---:|---:|---:|---:|
| graph proposal | 0.555 / 1.680 m | 1.994° / 5.561° | 45.83% | 8.33% |
| G11 legacy valid-factor mean | **0.455** / 1.499 m | 1.667° / 5.058° | **54.17%** | 6.25% |
| correlation-corrected mean | 0.495 / **1.301 m** | **1.599° / 3.760°** | 50.00% | **4.17%** |
| uncapacitated latent | 0.577 / 1.539 m | 1.925° / 4.216° | 39.58% | 8.33% |
| capacitated latent | 0.577 / 1.637 m | 2.020° / 4.548° | 45.83% | 8.33% |
| bounded safety residual | 0.495 / 1.301 m | 1.599° / 3.760° | 50.00% | 4.17% |

The correlation-corrected policy improves every robustness metric relative to
the graph proposal and halves the catastrophic count from four to two.  Its
Top-3 0.5 m / 5 degree recall is 72.92%, versus 70.83% for the legacy factor
mean.  Relative to G11 it explicitly trades 4 cm median and 4.17 percentage
points of Top-1 success for 20 cm P90, 1.30 degrees rotation P90 and one fewer
catastrophic failure.  This is aligned with a robustness-first claim, but it
does not recover the requested centimetre/decimetre precision.

The safety residual does not alter any Dev48 Top-1 choice.  At 70% accepted
coverage its confidence ordering reports 0.473 / 1.115 m and 2.94%
catastrophic rate.  Basin/catastrophic AUPRC falls from 0.959/0.968 on mapping
validation to 0.748/0.619 on Dev48, so abstention remains useful but is not yet
reliably calibrated enough for production.  The two remaining catastrophic frames are
both seq13: one has a 0.459 m Top-32 oracle but is mis-ranked by every factor
policy; the other has only a 1.030 m Top-32 oracle and is partly a proposal
coverage failure.

The group-aware solver therefore answers an important method question rather
than merely adding another score: support de-correlation is a stable tail
improvement; hard child/primitive capacity is not a stable Top-1 improvement.
Because the latent-selection gate fails on Dev48, no SE(3) update or continuous
refiner is started.  Doing so would again allow a weak assignment to move and
then validate its own pose.

### G13 — paired pose likelihood and deployment-replay audit

This pass implemented the probability correction requested after G12 instead
of another configuration-weight sweep.  The map remains the clean exact 2DGS
hierarchy with one canonical `radio_final` code.  No RGB, RADIO intermediate,
ALIKE descriptor, SfM track, image matcher or additional downstream map
embedding was introduced.

#### Two protocol bugs fixed

The old 63k factor sample artifact stored feature rows and class labels but
dropped query-group, tested-child and candidate-pose identity.  It therefore
could not train the claimed same-query/same-group density ratio.  Paired v3
samples now persist image, group, truth/tested child, runtime child rank,
candidate index, negative family and pose error.  Structured negatives include
frozen graph Top-1, the closest out-of-basin Top-32 pose, a low-rotation
translated facade phase, and the highest-probability wrong runtime child.

A second, more consequential mismatch was found during full replay.  Training
used the oracle-child member centroid as the query coordinate while deployment
uses the complete query-group centre.  Across 16,384 exact groups this hidden
offset is 0.82 px median, 16.14 px P90, 24.48 px P95 and 240.59 px maximum.
Labels still use exact contributor membership, but all factor inputs now use
the deployment group centre.  The oracle offset is retained only as a
non-deployment audit column.

The factor direction is fit from symmetric positive-minus-negative differences
inside the same image/group.  A disjoint trajectory then calibrates its affine
zero point, yielding

```text
log p(e | correct) / p(e | structured wrong)
```

rather than reinterpreting a typed multi-class confidence as additive pose
evidence.  Runtime uses no GT or absolute world-coordinate feature.  On held
out seq10 after fixing both deployment gaps, Top-4 pair concordance is 0.749;
the low-rotation facade and nearest-pose families are 0.739 and 0.664.  26 of
27 query/candidate-family aggregates rank exact GT over the paired wrong pose.

#### The previously unmeasured Top-C bottleneck

Exact replay also shows that the truth child is absent from the deployment
option set much more often than the G12 solver assumed:

| child retrieval cutoff | exact group recall |
|---:|---:|
| Top-1 | 18.98% |
| Top-4 | 42.55% |
| Top-8 | 55.62% |
| Top-16 | 67.41% |
| Top-32 | 77.15% |
| Top-64 | 84.47% |
| Top-128 | 89.64% |

Thus Top-4 deleted 57.5% of correct physical explanations before factor
scoring.  Factor positives and wrong-child negatives are now restricted to the
same runtime Top-C.  Top-16 was selected from this recall/cost curve before
pose evaluation as the single coverage diagnostic; Top-32/64 were not swept.

#### Soft assignment and configuration gate

G13 adds an entropy-regularized posterior over every fixed child/primitive mode
plus four explicit null types.  The per-child mode posterior only distributes
one factor Bayes mass, so geometry is not counted twice.  A soft-capacity
diagnostic uses projected dual penalties on expected child/primitive occupancy;
it never hard-deletes an option.  Every query group remains in the denominator.

The local factor gate does not transfer to full Top-32 configuration selection:

| seq12/14 validation policy (17 queries) | translation median / P90 | rotation median / P90 | success <=0.5m/5deg | catastrophic |
|---|---:|---:|---:|---:|
| graph proposal | 0.389 / 3.617 m | 1.472 / 9.970 deg | 52.94% | 17.65% |
| Top-4 legacy valid mass | 0.359 / 1.967 m | 1.384 / 8.703 deg | **64.71%** | **11.76%** |
| Top-4 paired LLR mean | 0.364 / 7.524 m | 1.384 / 24.116 deg | 58.82% | 17.65% |
| Top-4 legacy + LLR-median equal-rank fusion | 0.373 / **1.372 m** | 1.265 / **4.692 deg** | **64.71%** | **11.76%** |
| Top-16 legacy valid mass | 0.364 / 3.700 m | 1.091 / 9.373 deg | **64.71%** | 17.65% |
| Top-16 paired LLR mean | 0.381 / 3.290 m | **0.971** / 8.487 deg | 58.82% | 17.65% |
| Top-16 legacy + LLR-median equal-rank fusion | 0.389 / **1.486 m** | 1.321 / 7.129 deg | **64.71%** | **11.76%** |

The predefined equal-rank fusion is scale-free.  On the corrected Top-4
validation replay it improves the legacy P90 from 1.967 to 1.372 m and rotation
P90 from 8.703 to 4.692 degrees without changing success or catastrophic rate;
median regresses by 1.4 cm.  At Top-16 its P90 is 1.486 m, but its median is
0.389 m.  This is a useful robustness diagnostic, not a promotion: neither
version improves the three primary accuracy/safety decisions together.

The first-principles conclusion is that two different losses are currently
being conflated.  Correct-child/exact-group factors can learn pose
compatibility, while deployment first has to infer region identity from a
diffuse and repetitive child posterior.  Increasing Top-C restores coverage
but introduces ambiguity at nearly the same rate.  Independent factor means,
soft assignment and capacity cannot create the missing cross-group identity
constraint.  Because the configuration gate still fails, mode-relation,
continuous SE(3) refinement and Dev48 were deliberately not run in G13.

The next implementation should train a sparse mode-relation factor on full
deployment option sets: relative query displacement versus projected selected
primitive displacement, with local, long-range and depth/normal edge types.
Its gate must be configuration-level trajectory transfer, not exact-factor
pair accuracy.  Only if it improves both basin selection and tail risk should
the frozen Top-3 refiner handoff be reopened.

### G14 — sparse mode relations and exact tree inference

G14 freezes the G13 map, canonical field, mapper, graph, pose pool and local
LLR.  It adds no map embedding.  A relation edge joins two fixed query groups
and two fixed child-local primitive modes; the candidate pose may only project
and evaluate them.  The edge set is built before any pose is read and contains
three explicit families:

```text
local image neighbours
descriptor-distinct long-range supports
support-scale probes evaluated for depth / normal diversity
```

Strongly overlapping supports contribute one representative relation node.
The fit edges form a deterministic information tree/forest.  Long-range and
depth/normal edges not used by that tree are held out for verification.  The
two sets are fail-closed disjoint.  Runtime retrieves Top-16 children, retains
at most eight with no more than two per parent family, and retains the first
two modes from the fixed VFM Top-M distribution for each child.  This is a
family-preserving query-only shortlist, not a Top-K sweep or a pose-conditioned
mode search.

The analytic baseline exposed and fixed two protocol bugs before learning:

1. a local pair could be silently relabelled long/depth when it was also
   informative, eliminating the local family from the sample protocol;
2. visibility was initially renormalized over the modes left alive by each
   candidate pose, allowing a wrong pose to manufacture unit mode mass.

The corrected denominator is normalized exactly once from the fixed VFM
distribution.  Geometry can only remove probability mass into typed relation
null; it cannot redistribute that mass.  Invalid/missing, behind-camera,
back-facing and same-primitive conflicts have explicit meanings.  Only the
same-primitive choice by independent groups is a hard conflict.

The fixed analytic statistic is already directional.  Across the four mapping
shards its concordance is about 0.64 overall, 0.59--0.66 for local edges,
0.65--0.69 for long-range edges and 0.61--0.64 for depth/normal edges.  Phase
negatives reach 0.74--0.80, while one-wrong-endpoint identity negatives remain
hard at 0.54--0.56.  The learned same-query/same-edge density ratio transfers
to held-out seq10 as follows:

| paired relation gate | seq10 result |
|---|---:|
| all edge concordance | 0.711 |
| local / long-range / depth-normal | 0.722 / 0.756 / 0.667 |
| low-rotation phase | 0.773 |
| aggregate candidate concordance | 0.902 |

The first Max-Sum replay selected null for every group.  This identified a
probability-definition error rather than a weight problem: a unit null state
was being compared with each individual mode after the total valid mass had
been split across many modes.  G14 therefore uses exact Sum-Product on the fit
tree for the pose log-partition and uncertainty, while exact Max-Sum is kept
only for the interpretable configuration decode.  Held-out verification is
the predictive relation LLR conditional on both endpoints being valid; its
valid mass and null mass remain separate confidence/abstention outputs.  An
empty valid edge is neutral evidence, and equal-rank policies now assign true
average ranks to tied scores instead of injecting candidate-order bias.

On the frozen seq12/14 validation set, unrestricted held-out verification
shows that the relation signal genuinely changes configuration ranking, but
also that low-valid-mass overrides are unsafe:

| fixed policy (17 queries) | translation median / P90 | rotation median / P90 | success <=0.5m/5deg | catastrophic | median regret |
|---|---:|---:|---:|---:|---:|
| G13 Top-16 equal-rank | 0.389 / 1.486 m | 1.321 / 7.129 deg | 64.71% | 11.76% | 0.247 m |
| G13 + unrestricted held-out verification | **0.210** / 3.335 m | **0.971** / 9.329 deg | **70.59%** | 17.65% | **0.075 m** |
| G13 + positive-verification safety gate | 0.389 / **1.486 m** | 1.321 / **7.033 deg** | 64.71% | **11.76%** | **0.180 m** |

The safety gate has no fitted threshold: a relation override is permitted only
when the fused winner has LLR greater than the independently calibrated neutral
point zero.  It preserves P90, success and catastrophic rate, reduces selection
regret on both seq12 and seq14, and therefore was frozen before Dev48 was opened.

The single frozen Dev48 run gives:

| Dev48 policy | translation median / P90 | rotation median / P90 | success <=0.5m/5deg | catastrophic |
|---|---:|---:|---:|---:|
| frozen graph proposal | **0.555 / 1.680 m** | 1.994 / 5.561 deg | 45.83% | 8.33% |
| G13 Top-16 equal-rank | 0.638 / 1.711 m | 2.015 / **5.033 deg** | 43.75% | **6.25%** |
| unrestricted held-out verification (diagnostic) | 0.512 / **1.600 m** | **1.568** / 5.058 deg | **47.92%** | **6.25%** |
| frozen positive-verification gate | 0.582 / 1.711 m | 2.007 / **5.033 deg** | 45.83% | **6.25%** |

The unrestricted policy is not eligible for promotion because it failed the
validation tail gate before Dev48.  The frozen safe policy transfers modestly
but remains worse in translation than the graph proposal.  G14 is therefore a
scientific partial pass and a production fail: sparse relations do address
ranking failure, but the current endpoint-valid mass is too low for them to
reliably determine absolute facade identity.  The strict 0.5 m / 5 degree pose
pool coverage failure is 11.76% on seq12/14 and 12.5% on Dev48; relation ranking
cannot repair those queries.  Continuous refinement, relation-guided proposal
generation and untouched test remain closed.

This limitation is visible directly in the latent state, not inferred only
from pose error.  The unconditional Max-Sum decode assigns zero non-null groups
for every candidate in both final audits.  Median held-out endpoint-valid mass
is only 5.5% on seq12/14 and 8.6% on Dev48 (median null mass 94.5% and 91.4%).
Consequently the observed accuracy gain comes from conditional marginal
relation evidence, not from a solved hard physical identity configuration.
The next method change must raise and calibrate endpoint-valid mass or generate
relation-supported proposal families; forcing a non-null decode or tuning the
null prior would merely hide the unresolved identity uncertainty.

### G15 — relation probability semantics v2

G15 changes the probability definition without changing the clean 2DGS map,
the single canonical RADIO-final field, the frozen Top-32 pose pool, or either
learned LLR model.  It implements the five correctness fixes identified after
G14:

1. relation supports reuse the G12 complete-link overlap clusters; the
   representative is a query-only highest-quality support with a medoid
   tie-break rather than the first raster-order member;
2. retrieval null, omitted child, omitted mode, geometry-invalid and
   field-missing mass are explicit states, and together with all non-null
   child/mode states sum to one;
3. every state is scored as log prior plus likelihood-ratio evidence in one
   coordinate system rather than comparing conditional non-null logits with a
   unit null score;
4. a held-out endpoint pair uses its exact joint marginal along the fit tree,
   not the product of two node marginals;
5. physical invalidity has a frozen conservative likelihood loss, while query
   retrieval/shortlist uncertainty remains neutral.  A wrong pose can no
   longer delete an endpoint and receive the same pair score as an ordinary
   query-null event.

The state normalizer is numerically closed: the maximum residual over every
candidate in validation and Dev48 is `4.44e-16`.  A brute-force unit test also
checks non-adjacent pair marginals, and a tree-potential reparameterization
test leaves both the log partition and endpoint joint unchanged.

The overlap bug was much larger than expected.  Across seq12/14, complete-link
keeps 47.94 relation nodes per query versus 10.12 under the retired connected
components, restoring 643 nodes in 17 queries.  Dev48 keeps 45.71 versus
10.23, restoring 1,703 nodes.  Thus G14 had silently collapsed most of the
query relation graph through transitive overlap chains.

The newly fixed GT coverage ladder localizes the remaining probability loss:

| offline endpoint stage | seq12/14 validation | Dev48 |
|---|---:|---:|
| truth child in runtime Top-16 | 82.73% | 83.35% |
| survives per-parent Top-2 quota | 54.02% | 48.99% |
| survives final family Top-8 | 53.21% | 48.61% |
| truth primitive in VFM Top-8 | 42.57% | 39.05% |
| truth primitive in relation Top-2 | 22.89% | 20.37% |
| endpoint pose-valid | 21.69% | 20.37% |
| relation edge has two valid GT endpoints | 5.08% | 2.45% |

Visibility is not the main late-stage loss: almost every Top-2 truth primitive
is pose-valid.  The dominant losses are the parent-family quota and then the
fixed Top-2 local-mode truncation.  The probability audit agrees.  Mean prior
retrieval-null mass is 59.68%/59.25% and child-omitted mass is 24.27%/26.85%
on validation/Dev48.  Median posterior non-null mass over candidates is only
2.64%/3.32%; exact-GT medians are 4.27%/3.65%, barely above phase near misses
at 3.97%/3.57%.  Exact Max-Sum therefore still selects all-null for every
query.  This is now a trustworthy model failure, not the G14 coordinate bug.

The predefined v2 score is the sum of node, fit-tree incremental and exact
held-out predictive LLR evidence.  Its frozen-pool results are:

| policy | validation translation median/P90 | validation success/catastrophic | Dev48 translation median/P90 | Dev48 success/catastrophic |
|---|---:|---:|---:|---:|
| G13 equal-rank | 0.389 / 1.486 m | 64.71% / 11.76% | 0.638 / 1.711 m | 43.75% / 6.25% |
| G14 research relation | 0.210 / 3.335 m | 70.59% / 17.65% | 0.512 / 1.600 m | 47.92% / 6.25% |
| G15 node + fit + held-out | 0.381 / 1.572 m | 64.71% / 11.76% | **0.461 / 1.596 m** | **52.08%** / 8.33% |
| G15 node + fit only | 0.381 / 1.509 m | 64.71% / 11.76% | **0.457 / 1.596 m** | **54.17%** / 8.33% |

The main v2 score is a method-level improvement in Dev48 median, rotation
(`1.565/4.547` degrees), success and selection regret.  It is nevertheless not
promoted: validation P90 is worse than G13, Dev48 catastrophic rate is worse
than G13/G14, and seq13 carries the regression.  The exact held-out term is
especially non-transferable: by itself it is excellent on validation
(`0.270/1.260` m, 11.76% catastrophic) but fails on Dev48
(`0.592/3.064` m, 16.67% catastrophic).  It is retained as a diagnostic, not a
promotion signal.

A scale-free G13 plus G15-node/fit equal-rank fusion was inspected after these
research sets were already visible.  It reaches `0.381/1.301` m on validation
and `0.482/1.617` m on Dev48 while preserving the 6.25% Dev catastrophic rate.
This is the most robust observed G15 diagnostic, but it is explicitly post-hoc
and cannot be called frozen or production-qualified without a new disjoint
selection protocol.

G15 therefore passes probability mass conservation, exact inference and bug
repair, but fails the endpoint-identifiability and cross-trajectory promotion
gates.  No threshold sweep, larger relation model, Top-K expansion,
continuous refiner, relation-guided proposal generator or untouched test was
run.  The next justified work is a query-only, mass-adaptive child/mode state
representation or retraining/calibration under the corrected complete-link
edge distribution; proposal-family expansion remains Phase D only after the
endpoint model closes.

### G16 — mass-adaptive hierarchical endpoints

G16 implements the fixed-compute endpoint correction without changing the
clean 2DGS hierarchy, canonical RADIO-final field, query readout, typed graph,
or frozen Top-32 pose pool.  Each complete-link endpoint now evaluates the
full query-only child posterior and Top-8 local modes, then retains the 16
largest parent->child->mode leaf masses.  Omitted probability is represented
by separate support-invalid, parent-tail, child-tail, mode-tail,
geometry-invalid and field-missing states.  A candidate pose can score but
cannot change this universe.  Maximum normalization residual is
`6.66e-16`.

The support-correlation seam was tested on the same frozen validation pool.
Keeping every overlapping node and fractionally scaling its full unary raises
non-null mass but duplicates correlated observations; its node+fit P90 is
`10.464 m`.  The retired G15 convention (fractional LLR only) gives a strong
Dev diagnostic (`0.489/1.215 m`) but still multiplies the same retrieval prior
more than once.  Neither is admissible.  The active implementation collapses
each complete-link cluster once and averages its sparse identity posterior.
An additional audit found that the first collapse implementation averaged the
descriptor/posterior but retained a single representative's image center and
extent.  G16 now collapses descriptor, posterior and the union observation
geometry together; relation training and deployment use this identical
endpoint definition.

Hierarchy calibration is fitted with exact contributor truth and categorical
log score, not a null threshold.  `seq9` is the fit split and `seq10` is the
predefined level-selection split.  Support and parent calibration are rejected
because their validation NLL worsens.  Only child and mode calibration are
enabled:

| hierarchy level | seq10 NLL before | fitted NLL | decision |
|---|---:|---:|---|
| support-valid | 0.767 | 0.863 | identity |
| parent given valid | 1.513 | 1.557 | identity |
| child given parent | 1.570 | **1.549** | temperature 0.482 |
| mode given child | 2.000 | **1.654** | temperature 2.735 |

The relation LLR is then retrained on corrected cluster geometry and the exact
runtime adaptive options.  It uses a shared paired direction plus
family-specific affine calibration.  Pair/aggregate concordance is
`70.72%/79.93%` on its training trajectories and `78.60%/82.86%` on held-out
`seq10`.  It stores no RGB, path, extra embedding, SfM track, ALIKE descriptor,
or RADIO intermediate.

The endpoint funnel improves, but remains sparse:

| endpoint stage | G15 validation | G16 validation | G15 Dev48 | G16 Dev48 |
|---|---:|---:|---:|---:|
| truth child in full runtime posterior | 82.73% | 98.13% | 83.35% | 97.94% |
| truth child in relation budget | 53.21% | 53.15% | 48.61% | 42.70% |
| truth primitive in relation budget | 22.89% | 35.61% | 20.37% | 29.91% |
| endpoint pose-valid | 21.69% | 33.20% | 20.37% | 29.60% |
| relation edge has two valid GT endpoints | 5.08% | **10.34%** | 2.45% | **5.40%** |

Thus the adaptive mode allocation substantially improves primitive and
two-endpoint coverage, although the calibrated mode distribution expands only
5.91 children per Dev group on average and child-identity coverage is lower
than the old fixed family quota.  This is a real remaining allocation/identity
trade-off, not a visibility failure.

Frozen-pool accuracy is:

| policy | validation median/P90 | validation success / Top-3 / catastrophic | Dev48 median/P90 | Dev48 success / Top-3 / catastrophic |
|---|---:|---:|---:|---:|
| G15 node + fit | 0.381 / 1.509 m | 64.71% / 70.59% / 11.76% | **0.457** / 1.596 m | **54.17%** / **70.83%** / 8.33% |
| G16 node + fit | **0.359** / **1.509 m** | **70.59%** / 70.59% / 11.76% | 0.563 / **1.370 m** | 45.83% / 62.50% / **6.25%** |
| G16 joint, held-out diagnostic | 0.279 / 1.509 m | 70.59% / 70.59% / 11.76% | 0.447 / 1.370 m | 52.08% / 62.50% / 6.25% |

G16 passes the independent validation gate relative to G15 and improves the
Dev tail, but the predefined node+fit main score loses Dev median, success and
Top-3 recall.  The joint diagnostic cannot rescue promotion because held-out
evidence was explicitly diagnostic-only and its Top-3 also regresses.  Seq3
and seq5 remain controlled; seq13 still has `3.765 m` translation P90 and
18.75% catastrophic failures.

Most importantly, exact-GT versus phase-near-miss posterior non-null medians
are `0.0763/0.0708` on validation but reverse to `0.0705/0.0783` on Dev48.
Max-Sum is still all-null for every query.  G16 therefore proves that
mass-adaptive states and cross-group relations increase usable endpoint
coverage, but does not establish cross-trajectory physical instance identity.
Production remains graph v9; untouched test, continuous refinement and
relation-guided proposal generation remain closed.  The next justified method
work is at the query-support/local-readout boundary of the single canonical
field, especially the repeated-structure seq13 distribution, rather than more
back-end Top-K, null-prior or LLR tuning.

### G17 — offline multi-teacher physical-instance readout

G17 changes the map readout without changing the physical map or deployment
storage contract.  The physical map is built from
`/root/StMaryChurch2dgs_clean.ply` (SHA-256 prefix `4127e187`) and keeps exact
primitive membership.  Every observed primitive still stores one canonical
RADIO-final code.  During mapping only, three RADIO downstream adaptors provide
separate supervisory signals:

- SigLIP2-G: parent/context identity and long-support similarity;
- DINOv3-7B: child/local physical-instance identity and hard negatives;
- SAM3: within-image support-boundary affinity.

The teacher tensors are discarded after training.  The deployed artifact is a
pair of small residual context/local functions of the same canonical code, not
per-primitive downstream embeddings.  Query regions retain their relative 2D
token coordinates and masks; context uses the legacy 1/3/5/9 support mixture
as a zero-update baseline, while local uses a center-dominant 3x3 support.  The
same local transform is regenerated for the canonical primitive field before
query-map comparison.  An audit caught and fixed the earlier asymmetric case
where only the query was transformed.  Shard merging now also preserves and
checks the physical-readout SHA and endpoint-state policy.

The independently evaluated readout gain is:

| seq12/14 token identity | canonical baseline | G17 readout |
|---|---:|---:|
| parent R@32 | 93.19% | 94.01% |
| child R@16 | 69.43% | **77.44%** |
| joint parent-R@32 / child-R@16 | 68.23% | **76.15%** |
| child median rank | 6 | **4** |

Adding an exact nearest-primitive loss was separately tested.  It improves
within-child primitive R@8 by 0.83 points but reduces R@1 from 40.98% to
40.52%, so that checkpoint is rejected.  It confirms that a token's nearest
visible Gaussian is not a stable semantic identity target by itself.

G17 also implements the proposed breadth-then-depth fixed-16 endpoint frontier
with explicit support, parent, child, mode, geometry and field tails.  The
implementation conserves probability to a maximum residual of `6.66e-16`, but
the empirical first-principles test rejects it for this downstream task:

| seq12/14 endpoint funnel | breadth->depth | mass-only leaves |
|---|---:|---:|
| truth child in 16-state budget | **76.31%** | 51.70% |
| truth primitive in budget | 24.08% | **35.08%** |
| endpoint pose-valid | 22.91% | **33.38%** |
| relation edge with two valid endpoints | 5.89% | **8.65%** |
| mapping relation training pairs | 1,060 | **3,226** |

The reason is structural: relation inference needs a geometrically resolved
primitive at both ends.  Spending nearly all 16 leaves on one uncertain mode
from 14.7 different children improves identity breadth but destroys the depth
needed for a physical relation.  This is not repaired by more relation
training.  The selected G17 line therefore retains the new readout and unary
models but reuses the validated mass-only leaf frontier.  Its held-out seq10
GT-versus-phase relation concordance is 78.38%, compared with 37.50% for the
breadth frontier.

Fully rebuilt, lineage-consistent seq12/14 results are:

| policy | median/P90 | strict success | Top-3 | catastrophic |
|---|---:|---:|---:|---:|
| G16 node+fit | 0.359/1.509 m | 70.59% | 70.59% | 11.76% |
| G17 breadth node+fit | 0.255/1.091 m | 58.82% | 76.47% | 5.88% |
| **G17 selected mass-only node+fit** | **0.255/1.028 m** | **70.59%** | **76.47%** | **5.88%** |

Two failure-directed proposal tests are negative.  Expanding the retained
graph modes from 32 to all 42--44 available modes does not recover the
catastrophic seq12 frame 139, and activating 32 parent-pair seeds does not
recover either frame 139 or the 0.846 m near miss at frame 144.  Their failure
is upstream physical-instance evidence, not proposal truncation.

The probability gate remains closed: exact-GT non-null mass is `0.0758`, below
the phase-near-miss median `0.1132`, and Max-Sum remains all-null.  Consequently
the breadth allocator, pair-seed proposal change, primitive-loss checkpoint,
untouched test and continuous refiner are not promoted.  Dev48 is used only as
a trajectory stress test of the selected mass-only G17 line.

### G17.1 — exact physical-instance surface likelihood

The G17 relation result isolates the final ranking problem but does not solve
it.  A further audit found that the first surface verifier transformed each
query/map code independently with `project_flat`; it therefore bypassed the
position-preserving context attention trained from SigLIP and the rendered
missing-surface mask.  G17.1 fixes that mismatch:

1. generate a frozen Top-16 geometric pose set;
2. render the **complete** clean 2DGS scene from the one canonical field at
   every pose (no candidate-dependent maplet crop);
3. apply the same 9x9 context role head to query and rendered token sets;
4. mask missing rendered neighbours explicitly;
5. rank by full-query-grid mean cosine, so deleting difficult rendered support
   cannot improve the denominator.

This remains feature-map alignment.  It creates no 2D--3D correspondences,
uses no PnP/LoFTR/ALIKE descriptors, reads no mapping image and stores no second
map embedding.  The three offline teachers affect the learned context/local
readout, but their tensors are absent at runtime.

| seq12/14 validation | candidate identity rank | G17.1 spatial-context surface |
|---|---:|---:|
| translation median/P90 | 0.441 / 4.710 m | **0.366 / 0.837 m** |
| rotation median/P90 | 1.91 / 21.35 deg | **1.59 / 3.78 deg** |
| strict 0.5 m / 5 deg | 52.94% | **76.47%** |
| 1 m / 10 deg | 76.47% | **94.12%** |
| catastrophic | 17.65% | **5.88%** |

The independent Dev48 stress test proves a narrower but important claim:

| Dev48 policy | median/P90 | strict | 1 m / 10 deg | catastrophic |
|---|---:|---:|---:|---:|
| G16 node+fit | 0.569 / 1.405 m | **43.75%** | **77.08%** | 6.25% |
| G17 relation node+fit | 0.601 / 1.885 m | 41.67% | 72.92% | 8.33% |
| **G17.1 spatial-context Top-16** | 0.596 / 1.495 m | 39.58% | 72.92% | **2.08%** |

On the hardest repeated-facade `seq13`, catastrophic failure falls from
G16's 18.75% to 6.25% and translation P90 from 3.28 m to 1.54 m.  This is a
method-level tail improvement, but not a final-accuracy promotion: median and
strict success do not beat G16 or graph v9.

The following controlled extensions are rejected:

- verifying Top-32 leaves strict success at 39.58% but worsens P90 to 1.873 m
  and catastrophic rate to 8.33%; fixed Top-16 is retained;
- adding the G16-selected geometric pose as a seventeenth candidate leaves
  success unchanged and improves Dev P90 by only 0.032 m, which does not
  justify a second proposal branch;
- weighting canonical fusion observations by leave-one-view-out DINO/SAM/
  SigLIP consistency raises child R@16 only from 77.44% to 77.62%, while
  reducing validation pose Top-16 strict coverage from 88.24% to 76.47%.
  The teacher-weighted field is an ablation and does not replace the exact v2
  field.

The renderer hot path was also corrected: conversion of roughly 0.51 million
2DGS surface rotations from matrices to quaternions is now vectorized rather
than repeated in a Python loop for every pose.  Random 10,000-rotation
round-trip agreement has minimum absolute quaternion dot `0.9999998`, and all
52 2DGS mapping tests pass.

G17.1 is therefore the selected research verifier for failure reduction, not
the production default.  The relation probability gate still fails, the
surface score is not yet a calibrated non-null likelihood, and neither the
untouched test nor continuous refiner is opened.

### G17.2/G18 — exact-render audit and competing-phase surface likelihood

G17.2 freezes the G17.1 map, readout, candidate pool and Dev48 result, then
uses a new `seq11` trajectory-disjoint development block.  `seq11` is absent
from the canonical map, readout training, candidate-validity calibration and
G18 training.  The candidate-validity calibration was rebuilt on `seq10`
only; the selected G17 readout remains the lineaged `g17_v1` artifact.

The diagnostic dashboard writes RGB only for visual inspection and never to
the map.  It also writes depth, normal, primitive/parent/child identity,
canonical PCA, valid/missing masks, symmetric attention, per-token evidence,
candidate clouds and four-axis score landscapes.  One fixed `seq11` query
gives:

| exact-render audit | value |
|---|---:|
| primitive Top-1 / Top-K agreement | 98.91% / 99.79% |
| render/truth mask IoU | 99.98% |
| matched-primitive center-depth RMS | 1.65e-6 m |
| canonical feature coverage | 93.80% |
| GT local/global peak axes | 1 / 4 |
| raw score/coverage correlation | 0.801 |
| GT minus phase-near-miss score | +0.0233 |

The previous meter-scale depth audit was a diagnostic-definition error: it
mixed primitive-ID disagreements and compared gsplat primitive-center depth
with ray/primitive-plane intersection depth.  The corrected matched-identity
metric above shows no geometry-coordinate drift.  Plane-vs-center depth is
still reported separately because it is a definition difference, not a
renderer error.

G18 replaces independent mean cosine with a view/geometry-conditioned typed
surface likelihood.  Its runtime token evidence includes same-pixel RADIO
agreement, parent/child boundary, depth/gradient, camera-frame normal,
incidence, projected scale, primitive uncertainty, visibility and field
missing.  It predicts `surface_match`, `wrong_phase`, `occluded`,
`field_missing`, `query_unmapped_dynamic` and `unresolved`; all frozen
candidates compete in one posterior with a typed null.  The null is
conditioned on the fixed candidate-set evidence as well as query statistics,
because a query-only head cannot determine whether this particular set lacks
a valid pose.  A shared cross-candidate token-disagreement weight emphasizes
phase-discriminative regions without letting any candidate choose its own
denominator.

Offline DINO/SAM/SigLIP tensors remain training-only.  They are not averaged
into another static reliability map: DINO/SigLIP spatial cues supervise only
match/phase events, while SAM cues supervise boundary/visibility null events.
Only scalar role-directed supervision weights enter the training cache; no
teacher embedding enters the model or deployment map.  The map still stores
one canonical RADIO-derived primitive field and runtime uses no RGB, point
correspondence or PnP.

The frozen Top-16 pool and G18 result on the 11-query `seq11` block are:

| seq11 policy | median/P90 | strict 0.5 m/5 deg | 1 m/10 deg | catastrophic |
|---|---:|---:|---:|---:|
| frozen candidate Top-1 | 1.082 / 2.506 m | 9.09% | 36.36% | **0%** |
| fixed-grid cosine | 1.335 / 2.067 m | **18.18%** | 36.36% | 9.09% |
| **G18 competing-phase posterior** | 1.335 / 2.670 m | **18.18%** | 27.27% | **0%** |
| frozen Top-16 oracle | 0.270 / 0.593 m | 72.73% | 100% | n/a |

G18 training uses all 115 available queries from nine trajectories with equal
trajectory mass.  Its training strict/1 m success is 63.48%/86.09% and its
translation median/P90 is 0.394/1.181 m, but this does not transfer to
`seq11`.  Posterior risk is also not useful yet: reducing coverage to 50%
retains only one strict success and removes no additional catastrophic pose.
The initial 26-query smoke improved seq11 P90 to 1.743 m and 1 m success to
45.45%, but introduced one catastrophic output; it is rejected rather than
used to tune the development block.

This closes two implementation questions.  The exact clean-2DGS geometry
chain is sound, and same-query probabilistic competition can remove unsafe
null/catastrophic behavior.  The remaining failure is representational:
cross-trajectory canonical/readout features are too smooth and the score has
no reliable local SE(3) basin around GT.  G18 is not promoted, Dev48 is not
reused for model selection, the untouched test remains closed, and the old
surface-flow refiner is not started.

### G19-A/G19-B — phase-survival audit and coordinate-free phase readout

G19-A first removes an over-strong conclusion from G18: its failure did not
show that RADIO-final lacks metric phase.  It fixes GT versus a pose-defined
nearby phase negative (median 0.772 m / 1.63 deg on `seq11`) and audits the
same pair through the complete feature chain.  The negative is selected only
from frozen-candidate pose error; feature cosine is not used to define it.
For additional diagnosis only, an aligned 1280D RADIO-final canonical field
was built from the same 122 clean-2DGS contributors.  This 1.4 GB artifact is
not a deployment map and does not add a runtime feature.

| `seq11` phase pair level | GT > phase | candidate AUC | median margin | discriminative token fraction |
|---|---:|---:|---:|---:|
| 1280D RADIO-final canonical | **81.82%** | 0.603 | +0.0095 | 34.86% |
| PCA256 canonical | **81.82%** | **0.628** | +0.0101 | **39.65%** |
| mapper canonical | **81.82%** | 0.595 | **+0.0122** | 36.87% |
| local flat/spatial readout | 63.64% | 0.562/0.570 | +0.0112/+0.0134 | 37.02%/34.72% |
| 9x9 context readout | 63.64% | 0.562 | +0.0137 | 27.78% |

This is case A of the phase-survival decision tree: phase remains in the raw,
PCA and mapper canonical fields, while the learned local/context readouts
reduce cross-trajectory concordance and context pooling removes about one
quarter of the phase-discriminative tokens.  The audit does not claim to
separate native RADIO invariance from multi-view fusion—its raw level is still
post-fusion—but it is sufficient to reject a new view-conditioned map latent
as the next step.  The map is not the current bottleneck.

The coordinate audit also rejects a trajectory-framing dependency.  Fixed
tiny probes on G18 summaries have `seq11` AUC 0.438 with absolute grid XY,
0.463 with camera rays and 0.471 with no coordinate; all are near chance and
none justifies absolute image position.  G19-B therefore uses no absolute
coordinate.  G18's old typed token label was also circular (`cosine < 0.35`
defined `wrong_phase`).  New samples leave comparable tokens unresolved in
the feature extractor and assign `surface_match`/`wrong_phase` from frozen
candidate pose semantics; appearance is input evidence only.

G19-B keeps the one 128D mapper field and regenerates a high-frequency branch
at runtime.  For adjacent grid tokens it compares normalized mapper-feature
differences in horizontal and vertical directions.  The low-frequency
physical identity band is already supplied by Goal-Maplet proposal; adding
context cosine again at surface ranking double-counts it and transfers badly.
The phase score is a two-variable L2 pairwise logistic model trained on 115
mapping-trajectory candidate sets, with nearby 0.5--2.5 m physical negatives
upweighted.  It uses no RGB, keypoint, discrete correspondence, PnP, absolute
grid coordinate, second stored field or downstream embedding.

| frozen `seq11` Top-16 policy | translation median/P90 | rotation median/P90 | strict | 1 m/10 deg | catastrophic |
|---|---:|---:|---:|---:|---:|
| frozen proposal Top-1 | 1.082 / 2.506 m | 2.27 / 7.53 deg | 9.09% | 36.36% | 0% |
| context cosine | 1.335 / 2.067 m | 3.81 / 5.36 deg | 18.18% | 36.36% | 9.09% |
| G18 competing-phase posterior | 1.335 / 2.670 m | n/a | 18.18% | 27.27% | 0% |
| fixed single-scale directional phase | 0.673 / 1.732 m | 1.90 / 4.28 deg | 36.36% | 72.73% | 0% |
| **G19-B single-scale pairwise phase** | **0.625 / 0.854 m** | **1.43 / 2.23 deg** | **45.45%** | **90.91%** | **0%** |
| Top-16 oracle | 0.270 / 0.593 m | 1.06 / 1.49 deg | 72.73% | 100% | n/a |

The selected standardizer and two linear coefficients are serialized without
GT or feature tensors and loaded by the production verifier with strict map/
readout hashes.  A fresh two-GPU runtime replay reproduces the table exactly;
the gain is not an offline-only reorder.

This is a real method-level recovery: versus G18, median/P90 improve by
0.710/1.816 m and 1 m success by 63.64 percentage points without introducing
a catastrophic pose.  It still does not meet the requested centimetre-level
target and remains below the Top-16 oracle, so it is a research promotion, not
a production/final-test promotion.

A multiscale/diagonal relation extension was preselected by mapping-trajectory
leave-one-trajectory-out (0.967 m versus 1.198 m P90), but the locked `seq11`
promotion check rejected it: median improves to 0.535 m, while P90 regresses
to 1.082 m and 1 m success falls to 81.82%; strict and catastrophe remain
45.45%/0%.  The simpler single-scale phase readout is retained because this
round prioritizes eliminating localization failures and long tails.  This
ablation is also evidence that mapping-trajectory LOTO is not a substitute for
cross-acquisition validation.

G19-B made a continuous phase basin plausible but did not prove it.  G19-C
below closes that gate and corrects the raster protocol before any continuous
refiner is opened.

### G19-C — phase semantics, raster falsification and six-DoF basin gate

G19-C first removes two implementation footguns.  The historical fixed
`0.50 mapper + 0.20 context + 0.15 horizontal + 0.15 vertical` mixture is now
named `legacy_dual_band_score`; the active verifier fails closed unless a
serialized policy is supplied.  Directional evidence is also exposed as
`conditional phase agreement × phase observability`.  This preserves the
exact G19-B replay score while stopping future conditional energies from
silently treating missing evidence as phase disagreement.

The no-training six-DoF GT perturbation audit rejects continuous refinement:

| G19-C basin gate on `seq11` | result |
|---|---:|
| GT beats ±0.25 m surface-frame translations | 59.09% |
| GT beats ±0.50 m surface-frame translations | 77.27% |
| GT beats ±3° camera rotations | 84.85% |
| strict local maximum, tangent 1 / tangent 2 / normal | 36.36% / 36.36% / **0%** |
| strict local maximum, roll / pitch / yaw | 18.18% / 9.09% / 9.09% |

The score is therefore a discrete candidate ranker, not a differentiable
surface likelihood with a verified 0.25--0.5 m basin.  The continuous refiner
remains disabled.  The expanded pair benchmark agrees: GT beats frozen
negatives only 50% of the time in the 0.25--0.5 m band, versus 90.91% in each
of the 0.5--1.0 m and 1.0--2.5 m bands.  Phase helps coarse candidate
selection before it becomes a reliable local derivative.

The raster falsification audit then reveals the main G19-B implementation
error.  Common roll preserves only 54.55% of small-pool Top-1 choices at +20°;
1x versus mask-aware 2x/4x rendering preserves only 45.45%, with pairwise
order agreement 0.742/0.727.  About 98.43% of informative edges lie in the
one-pixel neighborhood of primitive boundaries.  Five-percent opacity jitter
is stable (100% Top-1), while deterministic five-percent primitive pruning
preserves 90.91%.  Thus the signal is not arbitrary opacity noise, but 1x
sampling is too coupled to the discrete fine-primitive raster.

Mask-aware 2x supersampling is the bounded fix: high-resolution 2DGS
compositing is pooled into the same query token grid, so the stored map,
query feature, two coefficients and candidate pool remain unchanged.

| frozen policy / split | translation median/P90 | rotation median/P90 | strict | 1 m/10 deg | catastrophic |
|---|---:|---:|---:|---:|---:|
| G19-B 1x, `seq11` | 0.625 / 0.854 m | 1.43 / 2.23° | 45.45% | 90.91% | 0% |
| **G19-C 2x, `seq11`** | **0.563 / 0.674 m** | **1.03 / 1.48°** | 45.45% | **100%** | 0% |
| graph conditional rank, strict12 | 0.518 / 1.514 m | 1.67 / 6.99° | 50.00% | 75.00% | 0% |
| G19-B 1x, strict12 | 0.626 / 1.576 m | 1.92 / 7.09° | 41.67% | 75.00% | 0% |
| **G19-C 2x, strict12** | **0.303 / 0.705 m** | **1.25 / 2.18°** | **75.00%** | **100%** | 0% |
| strict12 Top-16 oracle | 0.266 / 0.456 m | 0.90 / 1.66° | 91.67% | 100% | n/a |

The 2x protocol was chosen on `seq11`, then confirmed once on the disjoint
`seq3/seq5/seq13` strict12 trajectories; strict12 was not used to choose its
factor or coefficients and is now closed again.  A promoted policy artifact
records the required factor and the verifier rejects a 1x invocation.  This
is a real cross-trajectory ranking improvement, but it does not rescue the
continuous basin or meet the centimetre-level target.  G20 must use a
low-capacity conditional identity/phase/observability energy and a
surface-frame or orientation-equivariant phase field; it must not multiply
two falsely independent RADIO probabilities or reopen strict12 for tuning.

### G20 — fractional surface observation and Jacobian phase

G20 turns the empirical factor-2 fix into an explicit VFM-token observation
contract.  For token footprint `Omega_u`, the rendered canonical feature is
defined as the normalized surface-weighted footprint integral

```text
F_bar_T(u) = integral K_u A_T F_T dx / integral K_u A_T dx .
```

The factor-2 renderer is its current bounded quadrature.  It now exports
`p_feature`, `p_visible`, `p_missing` and `p_background`, with

```text
p_feature + p_missing + p_background = 1
p_feature + p_missing = p_visible .
```

Across 1,105 rendered candidates, the maximum closure residuals are
`3.35e-8` and `5.31e-8`.  Features remain an area-aware mixture.  Depth,
position and normal instead come from the frontmost sample of the dominant
physical component, so a wall/window boundary can no longer create a
non-physical averaged normal.  `mixed_surface` and dominant-component purity
remain explicit; median/P90 mixed-marker rates are 0.726/0.849 and are not
collapsed back to mutually exclusive feature/missing states.  These are token
marker rates, not non-dominant area mass.

The phase artifact schema is upgraded to
`goal_maplet_phase_readout_policy_v2`, with role `phase_residual_only`.
Loading fails closed unless the only active component is
`jacobian_phase_visible`; mapper cosine, context cosine and the legacy
dual-band score are forbidden.  Given the 128D feature Jacobian

```text
J_F = [dF/dx, dF/dy],
S_phase = <J_query,J_render>_F / (||J_query||_F ||J_render||_F + eps),
```

the phase is invariant when query and render share the same in-plane basis
rotation.  Carrier weight comes only from query gradient magnitude.  Phase
agreement, fractional observability and gradient log-scale are separate
measurements; the map still stores one canonical feature type and no
downstream embedding.

G20-C fits a non-negative three-term linear energy over the frozen Top-16
candidate set and normalizes candidates jointly with one explicit null.  The
rank weights are learned only from candidate-conditional targets.  Null is a
separate one-class open-set test on absolute candidate-set phase support;
query-local centering is not allowed to erase the case where every candidate
is poor.  Seq1/2/4 rows are excluded from query-side calibration.  Outer
leave-one-query-trajectory-out calibration uses seq6/7/8/9/12/14; closed
seq11 and seq3/5/13 strict12 are not reopened.

| 41-query fixed-map trajectory LOTO diagnostic | translation median/P90 | rotation median/P90 | strict | 1 m/10 deg | accepted catastrophic |
|---|---:|---:|---:|---:|---:|
| proposal identity only | 0.378 / 1.209 m | 1.34 / 3.37 deg | 63.41% | 87.80% | 2.50% |
| **Jacobian phase only** | **0.235 / 0.661 m** | **0.85 / 1.97 deg** | **87.80%** | **92.68%** | **0%** |
| identity + phase | 0.235 / 0.695 m | 0.85 / 1.97 deg | 85.37% | 92.68% | 0% |
| identity + phase + observation | 0.267 / 0.661 m | 0.99 / 2.14 deg | 85.37% | 92.68% | 0% |

All variants use the same absolute phase-support null.  It abstains on one of
41 queries (2.44%), correctly catching the 12.79 m/31.1 degree candidate-set
failure and giving 0% catastrophic rate among accepted full-energy results.
The other no-success candidate set has strong but wrong phase support and is
not rejected (2.17 m/11.9 degrees), so null classification is 97.56%, not
perfect.  Compared with identity, phase gains ten strict successes and loses
none.  The additive observation term loses one strict case relative to
phase-only; therefore G20 establishes the conditional-energy interface and
typed null, but the measured ranking gain belongs primarily to Jacobian phase.
The final diagnostic full-energy artifact has weights
`[0.0417, 0.9730, 0.9032]`.  A fresh verifier replay matches its offline
candidate and null probabilities exactly; this replay is distinct from the
outer-fold metrics and is not reported as held-out accuracy.  The artifact is
marked `deployment_allowed=false`; the verifier rejects it by default and
requires an explicit `--allow_self_map_diagnostic` override for such a replay.

G20-D reruns the predefined six-DoF basin audit on 18 queries, and reports a
12-query seq6/7/8/9/12/14 subset separately:

| fixed-map factor-2 Jacobian basin diagnostic | result |
|---|---:|
| GT beats +/-0.25 m surface-frame translations | 98.61% |
| GT beats +/-0.50 m surface-frame translations | 100% |
| GT beats +/-3 deg camera rotations | 100% |
| strict local maximum, tangent1 / tangent2 / normal | 66.67% / 66.67% / **83.33%** |
| strict local maximum, roll / pitch / yaw | 100% / 100% / 100% |

This is a qualitative change from G19-C's 0% surface-normal local maximum.
The separate gradient-scale diagnostic is weaker on the normal axis (41.67%)
than phase itself, so a scale/parallax term is not added merely for model
completeness.  Every numerical basin check passes.  The old surface-flow
refiner is not silently re-enabled, and G20 itself contains no continuous
refinement.

The final offline lineage audit exposes a more restrictive protocol fact.  The
canonical field was fused from 122 images in
`contributors_setcover128_clean` after the declared exclusions, and all 115
images in the G20 evidence report occur in that exact mapping-image set:

```text
mapping images:             122
evaluated images:           115
exact map/query overlap:    115 (100%)
map/query image disjoint:   false
```

The map still stores no image identity or RGB; this is an evaluation leakage,
not a deployment-storage violation.  Query-trajectory LOTO isolates only the
fit of the three energy weights.  It does not undo the fact that the query's
own VFM observation contributed to the canonical field or that the frozen
candidate pool was built against that field.  Consequently the 0.235/0.661 m
ranking result and the strong six-DoF basin are **self-map development
diagnostics**, not map-disjoint cross-acquisition localization evidence.
`g20_decision_record_v1.json` therefore records the numerical diagnostic pass
but sets both `map_disjoint_cross_acquisition_pass` and
`g21_continuous_refinement_open` to false.  A proper next run must rebuild the
canonical field, readout/proposals and evidence with the audited query
acquisition excluded before G21 is opened.  G20 remains a useful
method/observation-model implementation advance, but makes no production,
paper-accuracy or centimetre-precision claim.

#### Fixed-method map-disjoint replay

After discovering the self-map leakage, the serialized v2 phase policy was
replayed once, without fitting or threshold selection, on the two field-
excluded historical blocks.  The canonical field contains 122 images from
seq1/2/4/6/7/8/9/10/12/14; seq11 and strict12 (seq3/5/13) therefore have both
zero exact image overlap and zero trajectory overlap with field construction.
These blocks were used in earlier method development, so this is a
map-disjoint regression audit, not a new untouched test set.

| frozen phase replay | translation median/P90 | rotation median/P90 | strict | 1 m/10 deg | catastrophe |
|---|---:|---:|---:|---:|---:|
| seq11 G19-C directional 2x | 0.563 / 0.674 m | 1.03 / 1.48 deg | 45.45% | 100% | 0% |
| seq11 G20 Jacobian 2x | **0.483** / 0.723 m | **0.99** / 1.99 deg | **54.55%** | 90.91% | 0% |
| strict12 G19-C directional 2x | **0.303 / 0.705 m** | **1.25 / 2.18 deg** | **75.00%** | 100% | 0% |
| strict12 G20 Jacobian 2x | 0.335 / 0.715 m | 1.48 / 2.19 deg | 66.67% | 100% | 0% |

Seq11 uses the same graph proposal seed, typed graph, validity calibration,
identity renderer, cascade, map, field and physical readout as G20
calibration.  Applying the frozen full energy to that unchanged generator
does not transfer:

| seq11 frozen transfer | translation median/P90 | rotation median/P90 | strict | 1 m/10 deg | catastrophe / abstain |
|---|---:|---:|---:|---:|---:|
| identity only | 1.082 / 2.506 m | 2.27 / 7.53 deg | 9.09% | 36.36% | 0% / 100% |
| Jacobian phase only | **0.483 / 0.723 m** | **0.99 / 1.99 deg** | **54.55%** | **90.91%** | **0% / 100%** |
| frozen full energy | 0.854 / 5.643 m | 1.96 / 17.14 deg | 27.27% | 63.64% | 18.18% / 100% |

The all-abstain result is itself a failed transfer of the self-map-calibrated
absolute null, not a successful safety mechanism.  The strict12 pool uses an
older proposal seed/typed-graph/validity contract; the evaluator therefore
fails closed before applying the full energy, while the parameter-free phase
comparison remains reportable on its frozen candidate set.  Taken together,
the replay rejects the learned G20 energy and does not establish a consistent
Jacobian-phase improvement over G19-C.  G19-C remains selected for discrete
ranking, and neither a production promotion nor G21 continuous refinement is
opened.

#### G20.1 outer feature-pipeline cross-fit audit

The historical replay above still reused a mapper/readout trained with some
of the audited acquisitions.  G20.1 therefore rebuilds the complete feature
pipeline twice.  In the `hold_seq12` fold, seq12 is excluded from a mapper
trained from scratch, the canonical field, the physical-instance readout,
typed graph and validity/candidate calibration; `hold_seq14` applies the same
rule to seq14.  Query evidence is generated only after every fold-local
artifact has been hashed and checked for held-trajectory overlap.  Ranking
and abstention are now separate: these runs contain no conditional energy,
null, observation reward or refiner.

The old G19-C directional weights are not reused in the primary comparison:
their fit set included seq12/14.  Instead, the two cross-fit operators are
analytic and parameter-free: equal-weight horizontal/vertical directional
phase versus the unscaled visible Jacobian-phase cosine.  This changes the
question from transfer of a leaked historical calibration to the intrinsic
stability of the two phase definitions.

| held acquisition / factor-2 phase | translation median/P90 | rotation median/P90 | strict | 1 m/10 deg | catastrophe |
|---|---:|---:|---:|---:|---:|
| seq12 directional | 0.383 / 9.926 m | 1.41 / 10.86 deg | 54.55% | 63.64% | 27.27% |
| seq12 Jacobian | 0.383 / 9.016 m | 1.41 / 10.86 deg | 54.55% | 63.64% | 27.27% |
| seq14 directional | 0.261 / 0.449 m | 1.15 / 1.47 deg | 83.33% | 100% | 0% |
| seq14 Jacobian | 0.261 / 0.449 m | 1.15 / 1.47 deg | 83.33% | 100% | 0% |
| merged directional | **0.296 / 9.193 m** | **1.15 / 10.01 deg** | **64.71%** | **76.47%** | **17.65%** |
| merged Jacobian | **0.296 / 8.829 m** | 1.25 / 10.68 deg | **64.71%** | **76.47%** | **17.65%** |

The apparent tail is not a phase-operator failure.  The same fold-local
candidate pools provide the following exact ceiling/decomposition:

| candidate/selection event over 17 queries | strict | 1 m/10 deg |
|---|---:|---:|
| full Top-32 oracle | 70.59% | 82.35% |
| phase-evaluated Top-16 oracle | 70.59% | 76.47% |
| directional selected | 64.71% | 76.47% |
| no valid candidate in full pool | 5 | 3 |
| valid candidate only outside Top-16 | 0 | 1 |
| phase miss despite an available Top-16 candidate | 1 | **0** |

Thus directional phase captures 11/12 available strict successes and all
13/13 available one-metre successes.  Both operators have exactly the same
success set: 11 both-right, six both-wrong, zero directional-only and zero
Jacobian-only successes; 14/17 Top-1 poses are identical.  Per-query score
correlation is correspondingly high (median Kendall 0.905, Pearson 0.986).
Overall pose-quality pairwise concordance is 77.83% for directional and
78.04% for Jacobian.  Directional/Jacobian oracle-negative concordance is
90.0/90.0% at 0.25--0.5 m, 86.79/88.68% at 0.5--1 m, and
97.14/91.43% at 1--2.5 m.  There is no evidence for a learned phase mixture;
the simpler directional operator remains selected.

The typed factor-2 basin audit is run around GT on every one of the 17 held
queries, with only the predeclared ±0.25/±0.5 m surface-frame and ±3 degree
camera perturbations:

| outer-feature-cross-fit basin | ±0.25 m | ±0.50 m | ±3 deg | tangent1 / tangent2 / normal local max | roll / pitch / yaw local max |
|---|---:|---:|---:|---:|---:|
| seq12 directional | 98.48% | 100% | 100% | 100 / 90.91 / 100% | 100 / 100 / 100% |
| seq12 Jacobian | 98.48% | 100% | 100% | 100 / 90.91 / 100% | 100 / 100 / 100% |
| seq14 directional | 100% | 100% | 100% | 100 / 100 / 100% | 100 / 100 / 100% |
| seq14 Jacobian | 100% | 100% | 100% | 100 / 100 / 100% | 100 / 100 / 100% |
| merged, either operator | **99.02%** | **100%** | **100%** | **100 / 94.12 / 100%** | **100 / 100 / 100%** |

The two ranking operators therefore also have indistinguishable binary basin
gates.  In contrast, Jacobian log-scale agreement reaches only
69.61/75.49/84.31% on ±0.25 m/±0.50 m/±3 degrees, and raw observability only
50.98/57.84/50.98%.  This directly supports the revised semantics:
observability can parameterize phase uncertainty, but is not a monotonic pose
reward; scale agreement is not added merely for completeness.

This run also repaired two protocol-affecting implementation errors.  The
trajectory-split mapper evaluator previously demanded feature maps for strict
holdout observation rows even though those rows must not be loaded; it now
evaluates exactly `prototype | validation`.  The basin audit previously sent
new v1 directional replays through the v2 Jacobian evidence path after the
G20 integration, making their horizontal/vertical components identically
zero; dispatch is now explicitly policy-typed.  This does not invalidate the
older serialized G19-C basin artifact, whose stored directional components
are nonzero.  A misleading directional report label implying proposal-score
addition was also corrected: the serialized ranking score is phase-only.

G20.1 is still **feature-pipeline outer cross-fit, not geometry outer
cross-fit**.  `/root/StMaryChurch2dgs_clean.ply` and the physical hierarchy
were built before these folds and can contain held-acquisition geometry.
Therefore these numbers identify the current method bottleneck but cannot be
promoted as a final cross-acquisition paper result.  They make that bottleneck
unambiguous: Stage C nearly saturates its candidate set, while Stage A/B fails
to place a one-metre candidate in the full pool for three seq12 queries and a
strict candidate for five.  Top-K or phase-weight tuning cannot repair those
failures.

#### G20.3 candidate-coverage autopsy and configuration gate

G20.3 freezes the two G20.1 folds and decomposes every query before adding a
new Stage-B model.  It audits runtime parent/child recall, support topology,
raw versus post-NMS proposals, oracle parent/child assignments and exact
2DGS-surface geometry.  The resulting failure classification is unambiguous:

| 17-query cross-fit event | strict | 1 m/10 deg |
|---|---:|---:|
| graph proposal, post-NMS | 70.59% (12/17) | 82.35% (14/17) |
| parent-conditioned child fix | 70.59% (12/17) | 82.35% (14/17) |
| same proposals before pose NMS (up to 65) | 70.59% (12/17) | 82.35% (14/17) |
| oracle parent, runtime child readout | 88.24% (15/17) | **100% (17/17)** |
| oracle parent and child | 88.24% (15/17) | **100% (17/17)** |

All 17 queries pass the structural pose-sufficient parent and child recall
gate.  A runtime-posterior-supported child-center oracle is within one metre
on 16/17.  The three no-one-metre frames (`seq12/frame00139`, `00144`,
`00155`) are therefore all classified as **B1 configuration inference**:
the physical identities needed for a pose are present, but the independent
pre-pose hard assignment does not assemble the correct physical phase.
Neither B3 proposal retention nor a simple candidate-count shortage explains
the tail.

The support audit also rejects an indiscriminate G20.4 regrouping rewrite.
Those three frames retain respectively 70, 135 and 173 truth-child-supported
groups over 6, 7 and 9 image bins.  Their exact-surface bbox-center oracles
reach 7.5, 11.7 and 4.5 cm translation, while contributor-weighted exact
surface coordinates are at least as good.  In contrast, a child tile center
can be tens or hundreds of pixels from a support center.  Thus current support
topology is not what removes the coarse pose, although child centers remain an
inadequate final observation and the eventual solver must marginalize a
surface footprint.

Two real implementation errors were repaired during the autopsy:

1. disabling `local_evidence_weight` also disabled the already-computed
   parent-conditioned child enumeration.  Enumeration and score weighting are
   now independent;
2. `_score_pose()` returned compact Top-96 child rows and a refinement caller
   treated them as aligned to supports 0--95.  It now returns an explicit
   full-support array, preventing silent 2D/3D index mismatch on the 442--832
   group tail frames.

The first fix is necessary for correct method semantics but produces no
coverage gain, and the second does not alter the active graph generator's
candidate ceiling.  Increasing graph pair seeds to 32, disabling NMS, a
low-capacity covisibility beam and coherent physical-transport diagnostics
all fail to recover the three B1 frames.  The failed beam/transport variants
are not promoted into the production entry point.  This negative result is
important: the current typed graph stores co-visibility strength and scalar
distance, but the proposal does not preserve query-edge direction/order or a
pose-conditioned child mixture.  A larger beam over the same unary and
Jaccard factors is therefore not a principled solution.

Replaying the fixed conditioned candidate pool with the parameter-free
directional Stage C gives 0.329 m median, 5.868 m P90, 64.71% strict and
82.35% within one metre; all three catastrophes are exactly the B1 frames.
The score metadata now correctly records a parameter-free analytic operator.
G21 remains closed.  The next Stage-B implementation gate is a fixed Top-32,
assignment-diverse physical configuration generator whose pair factors retain
query displacement/order and whose child likelihood is marginalized under
pose; only after it raises candidate coverage should projected-tile
region-to-surface refinement be evaluated.

## G20.4 pose-conditioned query-edge audit

G20.4-A implements the missing factor without training or changing the
production candidate pool.  Query topology is fixed before a candidate is
examined: four nearest-neighbour edges and two long-range edges are built from
normalized query-region locations and RADIO context contrast.  For a pose and
surface assignment, child tangent rectangles are projected into the same
camera image and signed displacement is scored by an extent-normalized analytic
Huber loss.  Relative scale and left/right/above/below order are reported as
separate measurements, so no fitted weight can hide a failed displacement
hypothesis.

The three B1 frames all pass the oracle factor gate:

| frame | truth - current all-edge | truth - hardest near-phase all-edge | long-range | distinctive-context |
|---|---:|---:|---:|---:|
| 00139 | +0.346 | +0.119 | +1.559 | +0.968 |
| 00144 | +0.188 | +0.059 | +1.281 | +0.455 |
| 00155 | +0.063 | +0.033 | unavailable on the common assigned subset | +0.155 |

Here the truth configuration pose is taken from the already frozen
oracle-parent/runtime-child candidate pool.  It is independent of edge
evaluation.  A same-support PnP sanity control is deliberately not accepted:
on frame 00139 it creates a self-consistent edge residual at **20.32 m / 109.8
degrees**, demonstrating that using the same correspondences to generate and
validate a pose is circular.  This check fixes an important evaluation bug in
the first draft of the audit.

The factor result is positive, but the G20.4-B search gate is not.  An
assignment/SE(3)-diverse diagnostic expanded frame 00139 to 4,096 parent
configurations and retained 512 raw poses.  It still produced zero candidates
within 1 m / 10 degrees; the best joint candidate remains 5.87 m / 18.99
degrees, and even the minimum-translation candidate is 2.94 m with unusable
rotation.  An oracle physical-translation audit also rejects the historical
rigid-transport shortcut: although truth/current parents show a dominant
4--9 m facade shift, nearest translated candidates recover only 35--53% of
truth parents and solve to 7.97--14.45 m.

Therefore only the analytic query-edge factor and its fail-closed audit are
promoted.  The wide beam, random provisional-PnP and rigid transport branches
are not present in the production entry point.  This result narrows the next
foundational gate further: improve the pose-sufficient parent-support and
configuration proposal process under mapping-trajectory LOTO, then replay one
frozen Top-32 regression.  G20.5 child mixtures, G20.6 projected-tile
likelihood and G21 remain closed until that proposal gate improves 14/17
within-one-metre coverage.

## G20.5 RADIO geometry as a configuration proposal factor

G20.5 changes the missing variable rather than widening the old graph beam.
The clean PLY is used only for exact primitive membership because it has lost
the oriented 2DGS tangent frames; centers, tangents, scales, opacity and
normals are read from the original oriented 2DGS through `source_index`.
Using the clean PLY itself as oriented geometry was a real label-generation
error and is now rejected.  The resulting 509,572-element labels supervise a
small, regenerable RADIO-final head on mapping images.  Mapping RGB is not
stored.  Under outer trajectory holdout its scale-aligned depth AbsRel is
0.135/0.141 for seq12/seq14, versus 0.634/0.735 for a constant-depth baseline;
mean normal error is 26.23/26.76 degrees.

The proposal is deliberately not PnP.  Two query-region/physical-child
hypotheses define a relative metric relation, query geometry chooses a third
region, and a three-region Sim(3) alignment estimates rotation and translation
while marginalizing the one global query-depth scale.  A deterministic pair
beam then evaluates every provisional pose against all fixed query supports
using projected 2DGS surface support and the signed query-edge graph.  The
output remains a multimodal Top-32 physical configuration/SE(3) set; no point
descriptor, SfM track, raw mapping image or stored downstream embedding is
introduced.

The frozen 17-query outer-fold candidate result is a genuine coverage gain:

| fixed Top-32 candidate set | strict | 1 m/10 deg | median t | P90 t | P90 r |
|---|---:|---:|---:|---:|---:|
| G20.3 graph | 12/17 | 14/17 | 0.227 m | 4.491 m | 11.813 deg |
| G20.5 geometry | **13/17** | **15/17** | **0.210 m** | **3.286 m** | **4.819 deg** |

An oracle-parent/runtime-child control reaches 17/17 within one metre, and the
geometry proposal reaches 0.221 m/2.90 degrees and 0.089 m/1.14 degrees on the
previously failed 00139/00144 frames under that control.  This proves both the
query-geometry-to-pose path and the 2DGS geometry are usable.  With actual
retrieval, 00144 is newly recovered at 0.802 m/4.78 degrees; 00139 and 00155
remain configuration-identity misses.  Increasing candidate slots, using
identity-free regrouping or normal-guided initialization does not recover
them, so those variants are not promoted.

Parameter-free Stage C does not yet convert all of the candidate gain into one
pose:

| Stage-C Top-1 | strict | 1 m/10 deg | median t | P90 t | median r | P90 r |
|---|---:|---:|---:|---:|---:|---:|
| G20.3 graph | 11/17 | 14/17 | 0.329 m | 5.868 m | 1.149 deg | 6.878 deg |
| G20.5 geometry | **12/17** | 14/17 | **0.318 m** | **3.552 m** | 1.626 deg | **6.675 deg** |

The exact failure is now observable.  On 00144 the phase score places the
0.802 m candidate fourth and selects a 1.247 m candidate.  Six rounds of
continuous surface flow improve the selected pose only to 1.213 m; refining
the correct mode improves it to 0.697 m, but its final feature score remains
lower.  This is a repeated-facade likelihood ambiguity, not a missing local
optimizer basin.

A further dense audit compares RADIO-predicted query depth/normals with the
complete rendered 2DGS depth/normals.  It marginalizes one query-depth scale
and uses a fixed confident-query denominator.  This factor correctly prefers
the recovered 00144 mode (0.933 versus 0.919), but using it alone over all 17
queries falls to 9/17 strict and 13/17 within one metre.  No post-hoc threshold
is promoted.  It is retained as lineaged evidence for a future query-disjoint
calibration of a joint appearance/geometry likelihood.

Therefore G20.5 passes the fixed-budget **candidate proposal** gate and is the
new research proposal branch, but it does not replace the frozen graph Top-1
policy.  The next admissible experiment is not another beam or scalar weight
sweep: it is a mapping-trajectory-LOTO calibration of the joint phase and
dense-geometry likelihood, followed by one untouched outer-fold replay.  In
parallel, the two remaining candidate failures require better physical-parent
identity evidence, not Stage-C refinement.

## G20.6 map/query/head-disjoint likelihood and mapping-teacher audit

G20.6 executes that experiment with three independent leakage barriers.  The
canonical field excludes the evaluated query trajectory, the geometry head
excludes the evaluation and calibration trajectories, and likelihood fitting
uses trajectory LOTO over seq3/seq5/seq13 before frozen transfer to seq12/14.
Candidate pools are unchanged.  Every candidate exposes four typed quantities:
proposal identity, frozen surface phase, fractional observation quality and
scale-marginalized rendered/query geometry.  They are normalized only within
the same query by robust median/IQR statistics and receive non-negative
weights, so cross-map transfer does not depend on an absolute feature-score
scale.

This stricter protocol changes the conclusion suggested by the G20.5
single-frame audit.  Fold-specific heads give dense geometry positive LOTO
weights, but the frozen outer replay gains no strict or one-metre successes.
On seq12 its translation P90 changes from 7.011 m to 7.180 m and rotation P90
from 9.983 to 11.965 degrees; on seq14 the median/P90 translation changes from
0.182/0.326 m to 0.276/0.405 m.  The combined result is therefore rejected:

| frozen 17-query Stage C | strict | 1 m/10 deg | median t | P90 t | median r | P90 r |
|---|---:|---:|---:|---:|---:|---:|
| G20.5 phase | 12/17 | 14/17 | **0.318 m** | **3.552 m** | 1.626 deg | **6.675 deg** |
| phase + dense geometry | 12/17 | 14/17 | 0.334 m | 3.620 m | 1.855 deg | 7.468 deg |

The common-head fit drives both proposal identity and dense geometry to exactly
zero and retains only phase plus a small fractional-observation reliability
term.  This fallback passes the predeclared outer development gates: strict
success rises from 12/17 to **13/17**, rotation median improves from 1.626 to
**1.246 degrees**, no query is worsened, and one-metre success, catastrophic
rate and translation median/P90 remain exactly 14/17, 2/17 and
0.318/3.552 m.  It is the selected G20.6 development policy, but is not a
production promotion because no untouched test has been evaluated.  In
particular, seq12 00139/00155 still contain no correct candidate; reranking
cannot remove those failures.  On 00144 the runtime Top-16 contains a
0.802 m/4.78 degree mode but still selects 1.247 m/4.47 degrees, so successful
entry-point replay is not confused with a recovered strict localization.

The requested use of all three RADIO downstream adaptors at mapping time is
also audited without changing the deployment map contract.  Replacing the old
64-D block means with a deterministic signed 128-D teacher sketch greatly
improves preservation of teacher cosine geometry: DINO 0.899 to 0.940, SAM
0.747 to 0.951 and SigLIP 0.921 to 0.965.  The sketch is mapping-only and is
never stored in the deployed map.  Nevertheless, two readout seeds reach only
0.7587/0.7597 joint parent-R@32/child-R@16 versus **0.7615** for the frozen G17
readout.  The new compression and both readouts are therefore rejected.  This
separates representation fidelity from endpoint utility and avoids replacing a
validated readout merely because its teacher reconstruction is better.

The actual verifier has replayed the selected policy on a frozen seq12 query.
Its lineaged contract confirms one stored canonical VFM field, zero stored
downstream embeddings, no mapping RGB, no point correspondence and no PnP; the
unselected dense head is not loaded.  G20.6 consequently closes the Stage-C
fusion question: the next high-return change is upstream physical-parent/
configuration identity evidence that creates a valid mode for 00139/00155,
not a larger likelihood model or another scalar fusion sweep.

## G20.7 latent phase/child marginalization audit

G20.7 tests the upstream failure directly instead of widening the existing
beam again.  The experimental `soft_geometry` route aggregates retrieval mass
into anonymous physical-phase anchors, proposes metric SE(3) modes with the
cross-fit RADIO geometry head, and postpones parent/child identity until after
a pose exists.  Query edges and complete projected 2DGS child rectangles are
scored in the same normalized image frame.  The map contract is unchanged:
one canonical VFM primitive field, no stored downstream embedding, no mapping
RGB, no point correspondence and no PnP.

The audit found and fixed three correctness problems:

1. Stage C now has a typed all-candidates-wrong state.  Its v2 likelihood fits
   candidate weights and a null logit jointly; production replay fails closed
   when only a development policy is supplied, unless an explicit development
   override is requested.
2. The pair beam no longer scores an ambiguous phase by taking the best of K
   independent extension residuals.  It marginalizes explained probability
   mass, eliminating the look-elsewhere reward for diffuse wrong phases.
3. Runtime surface identity is marginalized over `8 parent x 4 child`
   alternatives from the same canonical field.  Out-of-map mass weights a
   support's pose evidence, while truncated-parent and unretained-child mass
   remain explicit missing identity.  No extra descriptor is stored per
   maplet or anchor.

The causal result is informative but does **not** pass promotion.  With the
same geometry generator and only oracle parent identities, seq12 frame00139
recovers 0.303 m/1.966 degrees at Top-1 and frame00155 contains
0.449 m/1.437 degrees at Top-16.  Actual retrieval therefore remains the
limiting variable.  Probability-marginal phase anchors improve frame00139's
raw seed beam from no one-metre candidate to four; the best is
0.967 m/8.119 degrees.  Frame00155 still has no one-metre seed.

Most importantly, retaining every bounded phase seed proves that the present
pose likelihood is not yet the missing solution.  Frame00139 carries all four
correct seeds through a 6,232-pose final scoring pool, but neither the
mass-conserving conditional likelihood nor the parent x child version places
one in Top-32.  Static per-anchor quotas also fail before final scoring.  The
full seq12/seq14 replay is intentionally stopped at this predeclared gate; it
would only repeat a known candidate-ranking failure at much higher cost.

Consequently `soft_geometry` remains an isolated research mode and the frozen
G20.6 development policy remains selected.  The next admissible method change
is a batched pose-conditioned canonical-VFM feature-map likelihood over sparse,
possibly disconnected rendered maplet regions.  It must compare query and map
VFM tensors after the pose, rather than reuse independent retrieval posterior
mass as a surrogate pose score.  Its gates are: recover the frame00139 seed in
Top-32, create a one-metre frame00155 seed, then improve the frozen 17-query
success/P90 before any untouched test.

### G20.8 — pose-conditioned primitive VFM and selective SE(3) refinement

G20.8 implements the missing pose-conditioned measurement without adding a
second map feature. Every clean-2DGS primitive centre participates in a
batched z-buffer; a visible primitive contributes the single canonical RADIO
code stored for that physical surface, and a primitive without a canonical
code remains an occluder. Query/map cosine is accumulated on the complete
48x48 query-token grid, so missing coverage cannot improve the score. This is
continuous-surface evidence, not retrieval posterior reuse, a point matcher or
PnP.

The first implementation audit exposed a protocol error in the experimental
replay: the old baseline used the symmetric `context` directional readout,
whereas the new report had defaulted to the `local` Jacobian readout. The
policy SHA and score scale differed. All G20.8 comparisons below were rerun
with the matching fold-local field/readout, context policy and 2x render. The
invalid cross-policy score union is retained only as a negative diagnostic.

Three representation levels were tested:

- one aggregated child descriptor destroys within-child phase and leaves the
  frame00139 correct mode around rank 1,394;
- eight stratified physical primitives per child cover only about 10% of the
  grid and do not recover Top-32;
- all 509,572 physical primitive centres recover the direction of the exact
  clean-2DGS score and evaluate a two-pose gate in about 0.06 s after load.

The dense-centre score is still not a valid replacement proposal expert. On
the frozen seq12/seq14 pools, exact Stage C gives 12/17 strict and 13/17 within
1 m, versus 12/17 and 14/17 for the old phase baseline. Combining old Top-16
and new Top-4 candidates by exact phase gives 11/17 strict; the G20.6 joint
posterior gives 12/17. A source-conditioned gate trained on seq3/seq5/seq13
also has no LOTO gain. Therefore the new expert is not merged into Stage B.

The same score is effective as a local optimization objective. The selected
operator evaluates the current pose and the positive/negative directions of
all six left-SE(3) axes in one batch, using a fixed
`0.60/0.40/0.25/0.12 m` and `5/3/2/1 degree` trust-region schedule. A step is
accepted only when the fixed-grid primitive VFM score increases. Because an
increasing coarse VFM score is not sufficient for decreasing metric error, an
explicit two-feature risk gate controls whether the refiner is called:

```text
joint phase posterior max <= 0.5374847939
and phase Top-2 margin >= 0.0031293709
```

Both features are defined within the same query candidate set; the gate
stores no scene feature. The thresholds are fitted on seq3/seq5/seq13. Outer
trajectory LOTO changes strict success from 8/12 to 9/12, retains 12/12 within
1 m/10 degrees and zero catastrophes, and does not lose a strict or loose
success in any fold. The gate therefore passes its declared development
promotion criterion.

Frozen seq12/seq14 replay against the G20.6 selected poses gives:

| 17-query development replay | strict 0.5 m/5 deg | within 1 m/10 deg | translation median/P90 | rotation median/P90 | catastrophic |
|---|---:|---:|---:|---:|---:|
| G20.6 | 13/17 | 14/17 | 0.318 / 3.552 m | 1.246 / 6.675 deg | 2/17 |
| G20.8 selective primitive refinement | **13/17** | **15/17** | 0.333 / **3.365 m** | **1.171** / 6.675 deg | 2/17 |

The new success is seq12 frame00144, refined from 1.247 m/4.470 degrees to
0.935 m/4.470 degrees. Unconditional refinement is explicitly rejected: it
would reduce strict seq12 success from 6/11 to 5/11. The remaining failure
ceiling is still upstream identity/configuration recall: frame00139 and
frame00155 remain catastrophic, and neither the primitive proposal pool nor
local refinement creates a valid basin for them. G20.8 is therefore the new
selected development policy, but production/paper promotion remains closed
until an untouched test is run and those two null modes are addressed.

### G21 — feature-free mapping-view topology and full-map VFM state verification

G21 addresses the remaining failure from first principles: independent
maplet identities do not encode which repeated surfaces were jointly visible
from one realizable camera state. A new mapping-view graph therefore stores
only calibrated mapping poses and sparse pose-to-physical-maplet incidence.
It stores no mapping RGB, image ID/path, VFM tensor, downstream embedding,
SfM point/track or correspondence. Query interaction remains deferred: the
view graph first turns the existing parent posterior into higher-order
structural pose hypotheses, and the sole canonical primitive RADIO field then
verifies those hypotheses against the query.

Three implementation errors exposed by the tail audit are now fixed:

1. global and within-anchor truncation used to delete structurally equivalent
   hypotheses before their appearance was evaluated; mapping-view anchor
   provenance and per-anchor survival are now explicit;
2. the first VFM verifier rendered only the seed's three hypothesized maplets,
   making its evidence circular; the selected verifier renders the complete
   physical map, uses an 8-primitives-per-child prescreen, then exact
   all-primitive verification for the retained 128 states;
3. the first union replay used the baseline report's rank one rather than the
   frozen G20.6 likelihood decision. In particular, this changed seq12
   frame00097 from 0.352 m to 0.582 m. The replay now resolves the deployed
   pre-order index through the report's stored ranking permutation, checks the
   report hash, and reads no metric-error label.

The causal frame00155 trace demonstrates the resulting chain. The
mapping-view hyperedge generator creates a 0.464 m/4.392 degree hypothesis,
but it is raw rank 4,590/16,033 and rank 67/508 within its structural anchor.
Full-map sparse VFM moves it to rank 104/3,840; exact all-primitive VFM moves
it to rank 2. Correspondence-free SE(3) alignment then produces
0.470 m/2.168 degrees. The remaining rank-one alias and the correct state
have nearly equal final primary scores, so arbitrary cross-expert maximum
score is not safe.

Deployment replay consequently treats the added graph as an optional state
expert, not a replacement pipeline. The existing query-local G20.8 gate
decides whether the expert is invoked. The frozen baseline state is always
kept. A calibration-only lower control limit on the baseline's full-grid
primitive score defines a typed null; a non-null baseline cannot be
overridden, while a null baseline can be replaced only when radius-0 and
radius-1 rendering select the same state. This protects frame00144 from a
41 m/178 degree repeated-facade alias and makes frame00155 the only changed
output among all 17 frozen queries.

| frozen 17-query replay | strict 0.5 m/5 deg | within 1 m/10 deg | translation median/P90 | rotation median/P90 | catastrophic |
|---|---:|---:|---:|---:|---:|
| G20.8 selected | 13/17 | 15/17 | 0.333 / 3.365 m | 1.171 / 6.675 deg | 2/17 |
| G21 protected state union | **14/17** | **16/17** | **0.333 / 0.805 m** | **1.171 / 3.731 deg** | **1/17** |

Frame00139 remains a genuine proposal-generation failure: the mapping-view
generator's best state is 1.059 m/3.946 degrees and does not survive exact
screening. G21 is therefore a credible method-level development gain, but
not a production or paper-test result. The absolute typed-null control limit
also transfers across fold-local fields; its next required replacement is a
trajectory- and map-fold-cross-fitted, coverage-conditioned likelihood ratio.
Only after that calibration and an untouched test may G21 be promoted.

## Development-set result

The best deployable Goal-Maplet Top-1 remains the frozen graph v9 result, not
the rejected corrected ranker:

| Dev48 v9 | translation median | P90 | rotation median | P90 |
|---|---:|---:|---:|---:|
| all | **0.555 m** | **1.680 m** | **1.994°** | **5.561°** |
| seq3 | 0.602 m | 1.546 m | 1.93° | 4.43° |
| seq5 | 0.474 m | 0.793 m | 1.66° | 3.76° |
| seq13 | 0.700 m | 2.421 m | 2.93° | 6.89° |

This is substantially worse in median accuracy than the frozen V3 full-scene
development baseline (0.197 m median), although the protocols are not directly
comparable: V3 reports all 530 development queries and uses a feature-aligned
anchor/PnP system, while Goal-Maplet Dev48 is trajectory-disjoint and designed
for architecture selection. V3 itself has a 1.627 m translation P90 and large
catastrophic tail, so Goal-Maplet's Top-N coverage remains scientifically
relevant, but it does not yet meet the requested accuracy.

## Milestone status

| Milestone | Decision | Evidence |
|---|---|---|
| M1 trustworthy map | pass | exact hierarchy, geometry/visibility audit, lineage |
| M2 trustworthy retrieval | partial pass | pose-sufficient R@64=1; R@1/calibration still weak |
| M3 structured coarse localization | partial pass | strong Top-32, unreliable Top-1 rank |
| M3.1 configuration ranking | scientific partial / production fail | robust aggregate 0.495/1.301 m and 4.17% catastrophic, but 0.235 m regret |
| M3.2 child-local measurement | scientific partial / production fail | grouped assignment implemented; hard capacity fails Dev promotion |
| M3.3 sparse mode relation | scientific partial / production fail | held-out relation improves Dev median/ranking, but safe policy remains below graph Top-1 |
| M3.4 relation probability semantics v2 | correctness pass / production fail | exact mass and pair marginals; 0.461/1.596 m Dev joint, but 8.33% catastrophic and all-null MAP |
| M3.5 mass-adaptive hierarchical endpoints | method pass / production fail | two-valid endpoints 2.45% -> 5.40%; 0.563/1.370 m Dev node+fit, but median/Top-3 regress and GT-vs-phase reverses |
| M3.6 physical-instance surface likelihood | tail pass / production fail | validation 0.366/0.837 m and 76.47% strict; Dev catastrophic 2.08%, but strict 39.58% |
| M3.7 typed competing-phase likelihood | correctness pass / production fail | exact geometry audit passes; seq11 strict 18.18% and zero catastrophe, but 1.335/2.670 m and no transferable phase basin |
| M3.8 phase-survival/readout | method pass / production fail | phase survives through mapper (81.82% pair concordance); selected coordinate-free phase readout reaches 0.625/0.854 m, 45.45% strict, 90.91% within 1 m and zero catastrophe on seq11 |
| M3.9 fractional Jacobian phase | implementation pass / protocol fail | self-map trajectory LOTO phase-only 0.235/0.661 m and 87.80% strict; all 115 evidence images contributed to the canonical field, so cross-acquisition accuracy is unproven |
| M3.10 pose-conditioned query edges | oracle factor pass / solver fail | truth beats current and eight near-phase configurations on all 3 B1 frames; 4,096-config/512-pose search still has zero 1 m candidates on 00139 |
| M3.11 RADIO geometry proposal | candidate pass / Top-1 partial | Top-32 rises from 14/17 to 15/17 within 1 m; Stage C improves strict 11/17 to 12/17 and P90 5.868 to 3.552 m, but remains 14/17 within 1 m |
| M3.12 disjoint joint likelihood | reliability pass / dense geometry fail | phase+observation raises strict 12/17 to 13/17 with zero losses; dense geometry gains no success and regresses outer P90; production remains closed |
| M3.13 latent phase/child marginalization | correctness pass / production fail | oracle parent gives 0.303/0.449 m on the two tail frames and phase marginalization creates four 1 m seeds on 00139, but a 6,232-pose all-phase audit still ranks none in Top-32 and 00155 remains uncovered |
| M3.14 primitive VFM trust region | development pass / production open | seq3/5/13 LOTO 8/12 -> 9/12 strict without success loss; held seq12/14 replay preserves 13/17 strict and improves 1 m success 14/17 -> 15/17, but two catastrophes remain |
| M3.15 mapping-view state expert | method pass / production open | corrected frozen replay changes only frame00155, reaching 14/17 strict, 16/17 within 1 m and one catastrophe; frame00139 and cross-map typed-null calibration remain open |
| M4 refiner handoff | self-map basin pass / closed | factor-2 Jacobian phase passes the numerical six-DoF checks, but query/map overlap blocks G21 until a disjoint field and candidate pool are rebuilt |
| M5 final paper claim | fail | no untouched test, hard-subset comparison or target accuracy |

## Rejected directions

Do not continue by tuning these components:

- single-parent or pair-parent seed weights alone;
- global child Top-K expansion (2,048 candidates worsened failures);
- local RADIO evidence before pose conditioning;
- parent-center aggregation as a geometric observation;
- current child-splat or full-scene identity ranker without calibration;
- mapper-code pointwise cosine flow;
- current child local head;
- detector-only ALIKE + RADIO matching;
- larger virtual pose lattices or graph-weight sweeps.
- wider static/transport configuration beams without a new proposal factor.

## Required foundational work

The next work should change two previously under-questioned foundations:

1. **Calibrate or learn the exact surface score as a pose likelihood.**  The
   fixed-denominator spatial-context score controls catastrophic phases, but
   does not yet convert score mass into a calibrated non-null probability.
   Keep one canonical code and add view/geometry conditioning only through a
   shared regenerable readout.  Its first gate is GT stability, GT-vs-phase
   NLL and a measured 0.1/0.25/0.5/1 m basin.
2. **Train/calibrate a configuration ranker against pose correctness.** It must
   consume complete Top-N assignments, typed structure, exact rendered
   coverage and calibrated factor LLRs. Selection is a different task from
   retrieval and cannot reuse retrieval probabilities as a pose score.

The lightweight versions of both items above have now been tested. The next
iteration needs stronger *probability semantics*, not additional weight
sweeps: a batched/cached exact renderer for tractable configuration evidence,
and a pose-likelihood/local-mode readout trained with exact deployment replay.

Only after both gates pass should the system run an untouched test and a fair
V3/STDLoc comparison. A useful paper claim is likely Top-N structured physical
localization under repetitive/low-texture conditions plus reliable uncertainty,
not an unsupported claim that VFM regions directly yield centimetre pose.

## Canonical artifacts

- physical map: `goal_maplet/physical_map_v4.npz`
- exact retrieval field: `goal_maplet/canonical_surface_field_exact_v2.npz`
- task-neutral codec: `goal_maplet/canonical_radio_pca256_mapping_v1.npz`
- task-neutral field: `goal_maplet/canonical_surface_field_radio_pca256_exact_v3.npz`
- typed graph: `goal_maplet/typed_parent_graph_exact_v1.npz`
- exact Strict12 PFIR: `goal_maplet/canonical_pfir_exact_grouped_strict12_calibrated_v2.json`
- exact Dev48 PFIR: `goal_maplet/canonical_pfir_exact_grouped_dev48_calibrated_v2.json`
- oracle ladder: `goal_maplet/oracle_ladder_strict12_surface_v2.json`
- frozen Dev48 pose result: `goal_maplet/pose_modes_graph_actual_dev48_exact_v9.json`
- reproducible candidate pool: `goal_maplet/config_rank_pool_dev48_v18.json`
- rejected configuration ranker: `goal_maplet/configuration_ranker_v1.joblib`
- configuration Dev48 audit: `goal_maplet/configuration_ranker_dev48_v1.json`
- field/readout contract: `goal_maplet/retrieval_field_exact_v2_contract.json`
- child eligibility: `goal_maplet/child_geometry_eligibility_exact_v1.npz`
- child-local mode ranker: `goal_maplet/child_local_mode_ranker_v1.joblib`
- child-local Dev48 audit: `goal_maplet/child_local_likelihood_ranked_dev48_v4.json`
- pairwise child-local Dev48: `goal_maplet/child_local_likelihood_pairwise_dev48_v5.json`
- typed-null calibrator: `goal_maplet/child_local_factor_calibrator_v2.joblib`
- typed-null Dev48 audit: `goal_maplet/child_local_factor_calibrator_dev48_v2.json`
- cross-fit factor pool: `goal_maplet/config_rank_pool_mapping_factor_crossfit_v22.json`
- configuration-factor Dev48 audit: `goal_maplet/configuration_evidence_dev48_v2.json`
- learned factor ranker Dev48: `goal_maplet/configuration_pairwise_ranker_factor_dev48_v3.json`
- latent mapping cross-fit pool: `goal_maplet/config_rank_pool_mapping_latent_crossfit_v23.json`
- latent mapping/Dev audits: `goal_maplet/latent_configuration_mapping_crossfit_v23.json`, `goal_maplet/latent_configuration_dev48_v23.json`
- bounded safety selector/audit: `goal_maplet/latent_safety_selector_v1.joblib`, `goal_maplet/latent_safety_selector_dev48_v1.json`
- corrected Top-4 paired factor/audit: `goal_maplet/pose_likelihood_ratio_top4_corrected_v2.joblib`, `goal_maplet/pose_likelihood_ratio_top4_corrected_v2.json`
- corrected Top-4 validation audit: `goal_maplet/likelihood_configuration_mapping_top4_corrected_validation_v26.json`
- Top-16 coverage diagnostic: `goal_maplet/pose_likelihood_ratio_top16_v1.json`, `goal_maplet/likelihood_configuration_mapping_top16_validation_v25.json`
- G14 paired relation model/audit: `goal_maplet/mode_relation_likelihood_ratio_top16_v1.joblib`, `goal_maplet/mode_relation_likelihood_ratio_top16_v1.json`
- G14 frozen validation: `goal_maplet/config_rank_pool_mapping_relation_top16_validation_v31.json`, `goal_maplet/mode_relation_mapping_top16_validation_v31.json`
- G14 single Dev48 audit: `goal_maplet/config_rank_pool_dev48_relation_top16_v32.json`, `goal_maplet/mode_relation_dev48_top16_v32.json`
- G15 validation probability audit: `goal_maplet/config_rank_pool_mapping_relation_semantics_v2_validation_v33.json`, `goal_maplet/mode_relation_semantics_v2_validation_v35.json`
- G15 Dev48 probability audit: `goal_maplet/config_rank_pool_dev48_relation_semantics_v2_v34.json`, `goal_maplet/mode_relation_semantics_v2_dev48_v35.json`
- G16 selected hierarchy calibration: `goal_maplet/endpoint_hierarchy_calibration_seq9_g16_v3.json`, `goal_maplet/endpoint_hierarchy_calibration_seq9_g16_v3_summary.json`
- G16 corrected relation model: `goal_maplet/mode_relation_likelihood_ratio_g16_calibrated_v3.joblib`, `goal_maplet/mode_relation_likelihood_ratio_g16_calibrated_v3.json`
- G16 validation audit: `goal_maplet/config_rank_pool_g16_calibrated_validation_v40.json`, `goal_maplet/mode_relation_g16_calibrated_validation_v40.json`
- G16 Dev48 audit: `goal_maplet/config_rank_pool_g16_calibrated_dev48_v40.json`, `goal_maplet/mode_relation_g16_calibrated_dev48_v40.json`
- G17 physical-instance readout: `goal_maplet/physical_instance_readout_g17_v1.pt`, `goal_maplet/physical_instance_readout_g17_v1.json`
- G17 selected validation relation audit: `goal_maplet/config_rank_pool_g17_massonly_validation_v44.json`, `goal_maplet/mode_relation_g17_massonly_validation_v44.json`
- G17 fully rebuilt Dev48 relation audit: `goal_maplet/config_rank_pool_g17_massonly_dev48_v46.json`, `goal_maplet/mode_relation_g17_massonly_dev48_v46.json`
- G17.1 selected validation surface audit: `goal_maplet/pose_modes_surface_spatial_context_g17_validation_v50.json`
- G17.1 selected Dev48 surface audit: `goal_maplet/pose_modes_surface_spatial_context_g17_dev48_v51.json`
- G17.2 corrected diagnostic: `goal_maplet/g17_2_diagnostics_seq11_v2.json`, `goal_maplet/g17_2_diagnostics_seq11_v2/`
- G18 trajectory-disjoint candidate pools: `goal_maplet/pose_modes_g18_train_v1.json`, `goal_maplet/pose_modes_g18_seq11_v1.json`
- G18 role-directed full training samples: `goal_maplet/surface_likelihood_samples_g18_train_full_shard{0..3}_v2.npz`
- G18 selected competing-phase likelihood: `goal_maplet/surface_pose_likelihood_g18_competing_phase_v4.pt`, `goal_maplet/surface_pose_likelihood_g18_competing_phase_v4.json`
- G18 runtime integration smoke: `goal_maplet/pose_modes_surface_likelihood_g18_seq11_smoke_v1.json`
- G19 raw RADIO-final diagnostic field: `goal_maplet/canonical_surface_field_radio_raw1280_exact_g19_v1.npz`
- G19-A phase-survival audit: `goal_maplet/phase_survival_g19_a_v2.json`
- G19-B full train/seq11 evidence: `goal_maplet/surface_dual_band_g19_train_v2.json`, `goal_maplet/surface_dual_band_g19_seq11_v2.json`
- G19-B promotion/model: `goal_maplet/dual_band_phase_readout_g19_b_v5.json`, `goal_maplet/phase_readout_g19_b_selected_v1.json`
- G19-B runtime replay: `goal_maplet/pose_modes_phase_readout_g19_b_seq11_v1.json`
- G19-C six-DoF basin: `goal_maplet/phase_basin_g19_c_seq11_v1.json`
- G19-C raster/roll/boundary audit: `goal_maplet/phase_correctness_g19_c_seq11_v1.json`
- G19-C 2x seq11/strict12 confirmation: `goal_maplet/pose_modes_phase_readout_g19_c_supersample2_seq11_v1.json`, `goal_maplet/pose_modes_phase_readout_g19_c_strict12_2x_v1.json`
- G19-C selected render-bound policy: `goal_maplet/phase_readout_g19_c_supersample2_selected_v1.json`
- G19-C final lineaged decision record: `goal_maplet/g19_c_decision_record_v1.json`
- G20 phase-only policy: `goal_maplet/phase_readout_g20_jacobian_v2.json`
- G20 full frozen evidence: `goal_maplet/pose_modes_g20_jacobian_train_v1.json`
- G20 self-map query-trajectory LOTO audit/policy: `goal_maplet/conditional_energy_g20_query_trajectory_loto_v1.json`, `goal_maplet/conditional_energy_g20_selected_v1.json`
- G20 six-DoF basin: `goal_maplet/phase_basin_g20_mapping_v1.json`
- G20 explicit self-map runtime posterior replay: `goal_maplet/pose_modes_g20_runtime_smoke_v1.json`
- G20 lineaged decision record: `goal_maplet/g20_decision_record_v1.json`
- G20 map-disjoint seq11 phase replay and frozen-energy transfer: `goal_maplet/pose_modes_g20_mapdisjoint_seq11_v1.json`, `goal_maplet/conditional_energy_g20_mapdisjoint_seq11_v1.json`
- G20 map-disjoint strict12 phase-only regression: `goal_maplet/pose_modes_g20_mapdisjoint_strict12_v1.json`
- G20.1 fold-local maps/readouts/evidence: `goal_maplet/map_crossfit_g20_1/hold_seq12`, `goal_maplet/map_crossfit_g20_1/hold_seq14`
- G20.1 merged operator decision: `goal_maplet/map_crossfit_g20_1/phase_operator_evaluation_merged.json`
- G20.1 typed outer-feature-cross-fit basins: `goal_maplet/map_crossfit_g20_1/directional_basin_merged.json`, `goal_maplet/map_crossfit_g20_1/jacobian_basin_merged.json`
- G20.3 merged candidate-coverage autopsy: `goal_maplet/map_crossfit_g20_1/candidate_coverage_autopsy_g20_3_merged.json`
- G20.3 fixed conditioned/pre-NMS/oracle ladders: `goal_maplet/map_crossfit_g20_1/hold_seq{12,14}/candidate_pool_conditioned_child_g20_3.json`, `candidate_pool_conditioned_child_pre_nms_g20_3.json`, `candidate_pool_oracle_ladder_g20_3.json`
- G20.3 corrected parameter-free Stage-C replays: `goal_maplet/map_crossfit_g20_1/hold_seq{12,14}/directional_report_conditioned_child_g20_3.json`
- G20.3 lineaged decision: `goal_maplet/map_crossfit_g20_1/g20_3_candidate_coverage_decision.json`
- G20.4-A query-edge oracle audit: `goal_maplet/map_crossfit_g20_1/hold_seq12/query_edge_oracle_audit_g20_4_a.json`
- G20.4-B rejected wide frame-00139 coverage audit: `goal_maplet/map_crossfit_g20_1/hold_seq12/candidate_pose_conditioned_wide_frame00139_g20_4_b.json`
- G20.4 lineaged decision: `goal_maplet/map_crossfit_g20_1/g20_4_query_edge_decision.json`
- G20.5 outer-fold RADIO geometry heads and reports: `goal_maplet/g20_5_geometry/head_relative_hold_seq{12,14}`, `goal_maplet/map_crossfit_g20_1/hold_seq{12,14}/candidate_pool_relative_geometry_g20_5_j.json`
- G20.5 Stage-C and dense rendered-geometry audits: `goal_maplet/map_crossfit_g20_1/hold_seq{12,14}/directional_report_relative_geometry_g20_5_j.json`, `directional_report_relative_geometry_dense_audit_g20_5_k.json`
- G20.5 lineaged decision: `goal_maplet/map_crossfit_g20_1/g20_5_geometry_decision.json`
- G20.6 strict LOTO likelihood and selected development policy: `goal_maplet/g20_6_joint/joint_loto_common_transfer.json`, `goal_maplet/g20_6_joint/joint_policy_common_transfer.json`
- G20.6 rejected fold-specific dense-geometry audits: `goal_maplet/g20_6_joint/joint_loto_head_seq{12,14}.json`
- G20.6 mapping-teacher compression/readout audit: `goal_maplet/g20_6_mapping/teacher_compression_audit.json`, `goal_maplet/g20_6_mapping/physical_instance_readout_signed128_seed{1720,1721}.json`
- G20.6 runtime smoke and lineaged decision: `goal_maplet/g20_6_joint/runtime_smoke_seq12_frame00144.json`, `goal_maplet/g20_6_decision.json`
- G20.7 latent-phase diagnostics: `goal_maplet/g20_7_soft/oracle_parent_geometry_frame001{39,55}.json`, `goal_maplet/g20_7_soft/candidate_marginal_phase_frame001{39,55}.json`, `goal_maplet/g20_7_soft/candidate_all_phase_likelihood_frame00139.json`, `goal_maplet/g20_7_soft/candidate_parent_child_marginal_all_phase_frame00139.json`
- G20.7 lineaged decision: `goal_maplet/g20_7_soft/g20_7_decision.json`
- G20.8 sparse primitive experiments and selected replay: `goal_maplet/g20_8_sparse_vfm/`
- G20.8 refinement gate: `goal_maplet/g20_8_sparse_vfm/primitive_refinement_gate.json`
- G20.8 frozen 17-query evaluation: `goal_maplet/g20_8_sparse_vfm/selective_refinement_test.json`
- G21 fold-local mapping-view graphs: `goal_maplet/map_crossfit_g20_1/hold_seq{12,14}/mapping_view_graph_g21.npz`
- G21 causal tail audits and gated candidates: `goal_maplet/g21_mapping_view/`
- G21 corrected frozen replays: `goal_maplet/g21_mapping_view/g21_protocol_corrected_state_union_seq{12,14}.json`
- G21 lineaged decision: `goal_maplet/g21_mapping_view/g21_decision.json`
- rejected teacher-weighted field/readout: `goal_maplet/canonical_surface_field_teacher_g17_v3.npz`, `goal_maplet/physical_instance_readout_teacher_g17_v3.pt`
- rejected conditional rank: `goal_maplet/pose_modes_graph_conditionalrank_dev48_v17.json`
- 4x GT round-trip audit: `goal_maplet/surface_basin_radio_pca256_supersample4_oracle_smoke_gt1round_v4.json`

Focused G20.1 verification: `333 passed` across Goal-Maplet, V6 atlas/frame and
2DGS mapping tests.  G20 additionally covers fractional mass closure,
dominant-surface pooling, common-roll Jacobian invariance, v2 fail-closed
policy loading, conditional null semantics and the basin gate.  G20.1 adds
strict holdout mapper evaluation, typed v1/v2 basin dispatch, parameter-free
operator contracts, exact fold merging, candidate-ceiling decomposition and
held-trajectory lineage rejection.

Focused G20.5 verification: `161 passed` across all Goal-Maplet tests plus the
2DGS geometry-label, canonical-field and RADIO high-resolution geometry-head
tests.  This includes clean/oriented PLY lineage, scale-invariant depth losses,
fixed-denominator dense geometry evidence, decomposed posterior mass, query
edges, configuration provenance and dominant-child render caching.

Focused G20.6 verification: `164 passed` on the same scope.  The added tests
cover monotonic joint-likelihood semantics, robust query-local normalization,
fail-closed component contracts and deterministic signed teacher sketches.

Focused G20.7 verification: `169 passed`.  The new coverage checks typed
candidate/null mass conservation, parent-conditioned top-L child
marginalization, recovery from an incorrect descriptor Top-1 child, and both
hard and soft geometry-proposal branches.  `compileall` and `git diff --check`
also pass.

Focused G20.8 verification adds the sparse child/primitive pose likelihood,
all-primitive fixed-grid occlusion semantics, monotone SE(3) trust-region and
low-capacity refinement-gate tests. The focused changed scope passes 16 tests;
the complete current `tests/test_goal_maplet_*` regression passes 157 tests.

Focused G21 verification adds feature-free mapping-view graph serialization,
conditional in-map posterior invariance, deployed-baseline permutation and
lineage checks, structural-anchor provenance, hierarchical full-map primitive
screening and protected multi-expert state replay. The current Goal-Maplet
suite passes 165 tests; including changed Gaussian-field and RADIO geometry
head scope gives **185 passed**. Python compilation and `git diff --check`
also pass.
