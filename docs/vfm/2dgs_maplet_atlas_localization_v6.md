# V6: 2DGS Maplet-Atlas Correlation Localization

V6 is the sole active research line. V3 is the frozen production baseline and
V5 is a failed point-similarity diagnostic.

## Runtime graph

```text
RADIO-final maplet retrieval
  -> grouped probabilistic maplet pose proposals
  -> selected canonical maplet atlas area rendering
  -> stride-16/8/4 local correlation distributions
  -> analytic robust joint-SE(3) update
  -> maplet-identity-disjoint held-out verification
```

The fine stage predicts a two-dimensional displacement distribution, including
an explicit null outcome. It does not convert descriptor cosine directly into
an SE(3) energy and does not end in point-correspondence PnP.

## Map contract

Canonical geometry is determined only by maplet coordinates and intersections
with declared clean 2DGS support disks. Mapping observations may update feature
mean, dispersion and support count; they never move a texel.

Feature baking accepts an observation only when the texel's primitive ID is
present in the declared clean-2DGS rasterizer's top-k source-index buffer.
Canonical geometry, contributor caches and baked atlases carry SHA-256 hashes
of the full source PLY, clean PLY and retained source-index mask; a mismatch is
fatal. KD-tree assignment is not a production fallback. The saved atlas
contains:

- fixed XYZ and primitive ID per texel;
- fused unit metric feature retained as the K=1 ablation;
- optionally 2 or 4 anonymous appearance modes, each with a unit feature
  mean, mixture weight, view-direction mean/covariance and feature dispersion;
- support count and valid mask;
- maplet frame, extent and identity.

It contains no mapping RGB, mapping image path, observation descriptor list,
SfM point/track, ALIKE descriptor or RADIO-intermediate feature.
Appearance modes are sufficient statistics, not stored per-view descriptors.
The atlas records the metric-encoder SHA-256; evaluation rejects a known
map/query encoder lineage mismatch.

## Metric encoder

The metric encoder uses a phase-preserving RGB stem:

```text
Conv3x3 stride2 -> residual
-> Conv3x3 stride2 -> residual blocks
```

RADIO-final context conditions the stride-4 phase features with FiLM and
spatial residual fusion. Stride-8 and stride-16 use independent downsampling
residual trunks and descriptor heads rather than average-pooling stride-4.
Query-location matchability is supervised independently from pairwise null.
Pairwise null is a class in the complete map/query correlation distribution;
it is not predicted by a query-only head. Displacement covariance comes from
that distribution and map uncertainty comes from atlas feature dispersion.
The unused legacy query-only null and variance heads have been removed.
Training uses coupled translation
and rotation perturbations up to 30 cm and 2 degrees and differentiates through
the analytic joint-6DoF normal equations for the one-step pose objective.
Episodes are selected by actual correlation-grid flow, not metric translation
alone. Validation is a fixed, trajectory-disjoint, scale-stratified set and
reports NLL, EPE, correct-mode recall, direction cosine and in-window rate.
With a frozen atlas, the default episode path invokes the production atlas
renderer (including selected-maplet z-buffer/coverage), applies the same candidate-offset
correlation and K-mode marginalisation, then differentiates through the
analytic one-step SE(3) objective. The old point-pair episode is an explicit
diagnostic opt-out only.
Wrong texels use a 20/30/50 mixture of random, same-maplet neighbouring and
appearance-nearest cross-maplet negatives. Query-unmatchable regions are
sampled separately and do not conflate pairwise null with query matchability.

## Correlation and geometry

Selected atlases are triangle-rasterized with perspective-correct feature,
XYZ and uncertainty interpolation. Conservative subpixel coverage prevents a
small surface chart from disappearing on coarser grids. Complete local
correlation probabilities are retained; mean and covariance are derived only
after normalization with the null outcome. Query matchability is unfolded at
every candidate offset. For K-mode atlases, correlation marginalises
`p(mode|view) * p(query candidate|mode)`; it never hard-selects an appearance
mode before observing the query. Atlas uncertainty increases correlation
temperature and null prior.

The solver uses the calibrated radial-camera projection Jacobian, covariance
weighting, robust IRLS and LM damping. Evidence is normalised within each
maplet before the joint system is formed, so projected area does not let one
large repetitive facade dominate. Fit and held-out maplets are split by
stable maplet identity rather than by view index.

The solver does not assume that a raster cell's integer index is the exact
projection of its perspective-interpolated XYZ. It converts
`raster center + correlation displacement` to original-image coordinates and
subtracts the XYZ's actual subpixel projection before applying the Jacobian.
This term is mandatory for conservative subpixel triangles and coarse grids.

## Retrieval and coarse proposal

Retrieval scores every maplet mixture with its component weights. Probability
outside retained candidates is transferred exactly to unknown/null.
`QueryMapletGroup` preserves region position, extent, candidate identities,
probabilities and query-conditioned real surface-location modes. Appearance
modes at the same 3D cell are probability-aggregated, spatially distinct modes
are retained, and no posterior-average virtual 3D point is sent to PnP. Coarse
grouped PnP samples these real surface modes and is used only to enter the
local basin; it is never the final estimator. Legacy banks without spatial
geometry may still fall back to a maplet centre, but that is not the active
map contract.

## Contract, diagnostic and promotion gates

Contract gates are strict: coordinate conventions, fixed canonical XYZ,
primitive contributor identity, probability conservation, trajectory
isolation, forbidden-path checks and map/query encoder lineage.

Scientific diagnostics continue when a metric is below target. They include
flow EPE, correct-mode recall, direction/parallel/orthogonal flow
decomposition, null prevalence/AUPRC, support-view gap and basin curves.
Production promotion and the 530-query claim remain gated:

- G0: exact contributor IDs, canonical reprojection RMS below 0.5 px,
  sufficient atlas coverage and consistent coordinates.
- G1: oracle-maplet flow EPE, correct-mode recall, null AUPRC and one-step pose
  improvement.
- G2: at least 90%, 80% and 60% of 5 cm, 20 cm and 30 cm starts respectively
  reach 4 cm / 1 degree.
- G3: retrieved maplet groups place at least 60% of queries inside the
  20–30 cm basin.
- G4: end-to-end evaluation is run only after G0–G3 pass.

Encoder validation is trajectory-disjoint. A failed scientific target is
reported but does not prevent the next targeted oracle/ablation; it only
prevents production replacement and the full-query accuracy claim.

## StMaryChurch P0/P1 diagnostic (2026-07-27)

Using the fixed 32x32 canonical atlas, the existing trajectory-disjoint
encoder, 12 seq2/seq4 mapping views and four held-out seq11 views:

- K=4 baking accepted 109,934 texel observations and produced 17,805 valid
  texels; no mapping RGB/path or per-view descriptor list is stored.
- The new per-maplet audit is much more revealing than global texel coverage:
  only 71/848 geometry-bearing maplets receive any baked feature (777 are
  empty). Median coverage among those 71 is 19.0%, while the median over all
  maplets is zero. Even among the 71 baked maplets, view-direction dispersion
  is small (median about 0.0027, p90 about 0.0066), so these 12 neighbouring
  views do not provide a genuine wide-baseline appearance-mode experiment.
- The previous unconditional-mean replay and query-conditioned K=4 replay
  both failed the 5 cm production gate (0/4 below 4 cm/1 degree).
- Mean coarse EPE/mode recall over the four cases were approximately
  2.06 cells/45.5%; K=4 marginalisation gave 2.01 cells/44.0%.
- Median final translation remained 5.0 cm. One case accepted a useful update
  (to 4.27 cm), while three held-out checks correctly rejected harmful steps.
- A real-data inference-matched smoke replay completed on 66 seq2/seq4 pairs
  and six trajectory-disjoint seq11 pairs. It also exposed that the older
  seq1/seq2 training cache has zero renderable-maplet overlap with this local
  12-view atlas; such pairs are no longer silently admitted as validation.

Therefore K-mode representation is supported and removes the premature
single-mean assumption, but this particular 12-view bake cannot test the full
hypothesis: it lacks both surface coverage and view-angle diversity. The next
required experiment is trajectory/view set-cover baking across the scene,
followed by inference-matched fine-tuning. The remaining blockers are atlas
coverage/view support, the atlas-render/query-correlation training
distribution and weak null/flow calibration—not the analytic geometry solver.
These are scientific diagnostics, not a production-accuracy claim.

## Implementation audit after the strict-test correction (2026-07-27)

The earlier `seq11` checks were validation diagnostics, not Cambridge's
official held-out test. The strict test protocol is now `dataset_test.txt`
(`seq3`, `seq5`, `seq13`, 530 queries); `seq11` remains encoder validation.
No 530-query production claim is made until the layered gates pass.

Implemented and active:

- clean 2DGS source-index lineage across canonical geometry, contributors and
  baked feature maps;
- exact contributor-ID atlas baking, fixed canonical XYZ and no KD-tree
  production fallback;
- RADIO-final-only region retrieval with exact omitted mass and no LoFTR,
  mapping RGB, SfM, RADIO-intermediate or ALIKE descriptor dependency;
- K=1/2/4 anonymous per-texel appearance sufficient statistics;
- exact canonical RADIO-final spatial retrieval texture, replacing the old
  whole-maplet-centre observation model;
- real query-conditioned surface-location modes for coarse pose, with
  component preservation and no virtual posterior-average point;
- per-offset query matchability, structured wrong-mode negatives, query-only
  unmatchable samples, independent pyramid heads and frozen-atlas
  render/correlate training;
- deterministic trajectory-disjoint validation with scale-level EPE,
  direction, mode, one-step and null calibration metrics;
- maplet-balanced multi-hypothesis SE(3), held-out null-mass correction and
  exact subpixel projection correction.

Not yet implemented or not yet production-qualified:

- adaptive 16/32/64 atlas resolution (the current fixed-32 physical texel
  p90 is about 4.81 cm);
- automatic splitting of multilayer, disconnected or strongly non-planar
  maplets into separate charts;
- complete-scene clean-2DGS occlusion at every runtime proposal (the renderer
  currently z-buffers selected maplets; exact full-scene visibility is used
  only as a training/evaluation label);
- 2–3-step unrolled training and per-null-type calibration;
- a successful hierarchical/context/set retrieval model; tested anonymous
  pose voting and global spatial priors did not pass the coarse-basin gate;
- production promotion on all 530 strict-test queries.

These omissions are intentionally reported as open work. Several historical
recommendations are mutually exclusive: the stable-anchor/ALIKE-descriptor
branch has been superseded by the anchor-free surface-atlas correlation
branch, rather than being simultaneously retained in the active graph.

## Cross-round audit and clean-2DGS result (2026-07-27)

The answer to “were all earlier recommendations implemented?” is **no**.
The active graph now implements the recommendations that are compatible with
the anchor-free surface-atlas design, explicitly rejects failed or superseded
branches, and keeps the remaining items below as open work. Earlier reports
that only compared nearby hyperparameters were insufficient evidence.

### Validation-chain defects found in this audit

The following were implementation or protocol defects, not tunable choices:

- `seq11` had previously been described as held-out test data. It is now used
  only for trajectory-disjoint model selection; official strict test images
  come from `dataset_test.txt` (`seq3`, `seq5`, `seq13`).
- Exact visibility labels had leaked query ground-truth depth into an
  evaluator path. Ground truth is now restricted to labels and post-hoc error
  measurement; deployable hypotheses see query RGB/RADIO-final features,
  camera calibration and the feature map only.
- Query-unmatchable null mass was counted twice in one held-out aggregation.
- The old query matchability label was derived from pair agreement instead of
  clean-2DGS surface support.
- Spatial components were selected before identity retrieval in one path,
  turning a full-scene search into an invalid location-conditioned search.
  Identity retrieval now precedes query-conditioned spatial decoding.
- Descriptor-component deduplication used a component's local rank. After
  sorting and NMS, equal local ranks can refer to different physical points.
  Deduplication now uses stable maplet identity plus rounded physical XYZ.
- `maximum_maplets` was once only reported while the proposal path still
  consumed a different fixed candidate budget. The budget now controls the
  candidates actually passed downstream and is recorded in every report.
- Validation paired view and scale indices, suppressing part of the intended
  cross-view/cross-scale distribution. They are sampled independently now.
- Coarse PnP required six sampled groups although calibrated PnP needs four,
  sampled groups almost uniformly, prohibited two distinct cells from the
  same maplet, and lacked a geometric-rank check. These defects made the
  success probability unnecessarily small. The active deterministic solver
  uses regional posterior guidance and physical-cell identity; the optional
  stochastic ablation uses four-point minimal samples.
- EM refinement unconditionally replaced the initial pose. A monotonic
  likelihood rollback was added, but `seq11` still showed lower true pose
  accuracy even when the internal likelihood improved. EM is therefore
  disabled by default; it remains an explicit diagnostic only.

All new strict-test diagnostics use the same calibrated PnP implementation.
This matters because it separates observation quality from solver quality:
using exact visible clean-2DGS surface observations solves all 12 audit
queries, with 6.78 cm / 0.256 degree median error. Replacing those locations
with true maplet centres solves 0/12 within 30 cm / 3 degrees and gives
1.271 m / 2.963 degrees median error. Maplet centres are consequently not a
valid observation model.

### Representation learning that is more than a parameter sweep

The raw RADIO-final descriptor is useful for identity retrieval but is not
spatially precise enough inside a surface chart. A small origin-preserving
linear projection is now trained using clean-2DGS cell identity. Its second
version replays the deployed K-mode atlas: each query surface cell is matched
against anonymous observations of that cell from at least two distinct
reference trajectories, appearance modes are marginalized, and negatives are
different physical cells balanced within maplets. Mapping images, image IDs
and per-view descriptor lists are discarded after sufficient statistics are
baked.

On fixed `seq11` validation episodes, the atlas-matched projection changes:

| Metric | Raw RADIO-final | Spatial projection V2 |
| --- | ---: | ---: |
| within-maplet cell recall | 18.63% | 21.32% |
| global cell recall | 12.17% | 14.95% |
| median surface error | 18.06 cm | 15.91 cm |
| surface error below 30 cm | 65.65% | 68.48% |

This is a structural, geometry-supervised change rather than temperature or
Top-K tuning. Its strict-test benefit is real but modest, so it is not
presented as a solved localization system.

### Layered strict-test audit on 12 deterministic queries

The 12-query set is only a fixed promotion audit, not the final 530-query
accuracy result. The following rows use the clean
`StMaryChurch2dgs_clean.ply` lineage and stable physical-point deduplication.

| Observation/proposal path | within 30 cm / 3 degrees | median translation | median rotation |
| --- | ---: | ---: | ---: |
| exact visible clean-2DGS surface + PnP | 12/12 | 6.78 cm | 0.256 deg |
| retrieved maplets + oracle surface cell, Top-64 | 12/12 | 10.14 cm | 0.330 deg |
| projected descriptor MAP cell, Top-64 | 6/12 | 31.36 cm | 1.103 deg |
| deployable regional MAP RANSAC, Top-1 | 3/12 | 51.93 cm | 1.980 deg |
| deployable regional MAP RANSAC, oracle Top-5 | 6/12 | -- | -- |

Top-24 was an architectural bottleneck rather than a harmless efficiency
setting. Widening identity recall to Top-64 changes mean visible-maplet recall
from 35.40% to 67.84%, visible-surface coverage from 44.35% to 75.67%, and
retrieved-maplet oracle-surface PnP from 9/12 to 12/12. Descriptor MAP
localization moves only from 5/12 to 6/12, proving that the remaining dominant
error is the map-feature-to-surface posterior, not geometry or candidate
coverage.

The deployable deterministic regional solver reaches 4/6 Top-1 and 6/6
oracle Top-5 on `seq11`, but only 3/12 Top-1 and 6/12 oracle Top-5 on strict
queries. This validation-to-test gap is why no 530-query production claim is
made. The previous random Top-24 proposal path was far worse (0/12 Top-1,
7.98 m / 22.31 degrees median); the new result is a genuine chain repair, but
still fails G3 and is not a satisfactory final accuracy level.

### Disposition of the accumulated recommendations

Implemented and retained:

- clean 2DGS lineage, exact surface texels, K-mode anonymous atlas statistics;
- identity-first wide recall, query-conditioned spatial posterior and
  probability conservation;
- RADIO-final-only matching, ALIKE detection only where an optional sampling
  policy needs it, and no ALIKE descriptor map;
- region-aware multi-hypothesis PnP with stable physical correspondences;
- trajectory-disjoint model selection, layered geometry/retrieval/spatial/
  proposal oracles, and explicit production gates.

Implemented but rejected by validation:

- a monolithic learned metric encoder (training loss fell while
  trajectory-disjoint spatial validation degraded);
- global RADIO pose priors and anonymous pose voting;
- multi-prefix consensus aggregation;
- likelihood-only EM pose refinement.

Superseded and absent from the production graph:

- SfM tracks, stable-anchor descriptor bundles, maplet-centre PnP,
  LoFTR/pairwise image matching, stored mapping RGB, RADIO-intermediate
  features and ALIKE/RADIO concatenated descriptors.

Still open:

- adaptive atlas resolution and automatic maplet chart splitting;
- full-scene occlusion for every runtime proposal;
- a better calibrated continuous surface posterior, especially across the
  `seq11` to strict-test appearance/domain gap;
- per-null-type calibration and 2–3-step unrolled local alignment;
- a production-qualified final estimator and the gated 530-query evaluation.

The next rational research target is therefore not another Top-K/temperature
sweep. It is to improve the continuous RADIO-final surface posterior under
strict trajectory shift, while keeping the exact-geometry and identity-first
contracts fixed. Until its fixed strict audit exceeds the G3 basin-entry gate,
running all 530 queries would produce a precise measurement of an unqualified
system rather than a valid production result.
