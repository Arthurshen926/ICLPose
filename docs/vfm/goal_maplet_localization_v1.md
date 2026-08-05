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
| M3.1 configuration ranking | fail | 0.515 m median but 1.690 m P90 and 0.263 m regret |
| M3.2 child-local measurement | partial/fail gate | oracle-child 0.265 m; Top-8 mode oracle 0.151 m |
| M4 refiner handoff | fail | no stable 0.1–0.5 m convergence basin |
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

## Required foundational work

The next work should change two previously under-questioned foundations:

1. **Learn a pose-likelihood field/readout, not another retrieval embedding.**
   Keep one canonical map code, but train a view/geometry-conditioned readout
   or explicit surface-flow head on exact 4x contributor round-trips. Its first
   gate is GT stability and a measured 0.1/0.25/0.5/1 m basin.
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
- rejected conditional rank: `goal_maplet/pose_modes_graph_conditionalrank_dev48_v17.json`
- 4x GT round-trip audit: `goal_maplet/surface_basin_radio_pca256_supersample4_oracle_smoke_gt1round_v4.json`

Focused verification: `33 passed` for `tests/test_goal_maplet_*.py` plus the
exact 2DGS compositing equivalence test.
