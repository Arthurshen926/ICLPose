# G25 chart-local RADIO-64 mainline upgrade (2026-09-03)

## Decision

The mapping-only 64-D chart-local RADIO projection is a **mainline GO**.  It
improves both the seq10 development route and the frozen seq13 shard3 replay,
while reducing the canonical atlas from 1280-D to 64-D.  The map still stores
finite plane entities, metric-UV texels, anonymous descriptors, geometry, and
uncertainty; it stores no mapping RGB, image path, or source-view identity.

This does not authorize a production claim.  Seq13 shard3 is a historical
cross-route validation inventory, not a pristine blind test.

## Method contract

The learned head is a bias-free `64 x 1280` linear projection.  Positive pairs
are mapping tokens observed in distinct mapping images that land on the same
finite plane and the same 0.5 m metric-UV cell.  Hard negatives lie on the same
plane but in a different metric-UV cell.  Training uses symmetric InfoNCE plus
a same-plane hard-negative softplus term.

The fit routes are seq1/2/4/6/7/8/11; the entire mapping route seq9 is reserved
for representation validation.  No query pose, query depth, query label, or
query RGB is read.  The runtime map keeps the projected anonymous prototypes,
not the source observations used to learn them.

Held mapping-route representation validation:

| descriptor | positive cosine | same-plane/different-UV cosine | positive wins | margin |
|---|---:|---:|---:|---:|
| raw RADIO-1280 | .7681 | .5776 | 80.67% | .1905 |
| learned RADIO-64 | .8133 | .3947 | 92.32% | .4186 |

The training run was repeated with the same sealed inputs and seed.  All logged
losses, the weight array, canonical content hash, and the complete compressed
NPZ file were byte-identical.

An earlier provisional run allowed 21/27,084 fit identities whose planes had no
second eligible UV cell to fall back to a cross-plane negative.  That 0.08%
contract violation was removed before the results below: such identities are
now excluded from triplet sampling, every hard negative is replay-verified to
be on the same physical plane, and the provisional v1 result is superseded.

## Pose results

The only changed representation in this comparison is the chart-local RADIO
head and the resulting anonymous prototype selection.  Plane-specific metric
geometry, covariance/purity/dispersion, query planes, PnP, view refinement,
MoGe3 scale/surface refinement, and the frozen dual-branch selector remain in
place.

| route | system | median translation | median rotation | 0.1m/1deg | .25m/2deg | .5m/5deg | 1m/10deg | 2m/45deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seq10 (88) | prior mainline | .2367 m | .5337 deg | 10 | 46 | 69 | 80 | 85 |
| seq10 (88) | learned RADIO-64 | **.1623 m** | **.4455 deg** | **20** | **64** | **83** | **87** | **88** |
| seq13 shard3 (87) | prior mainline | .2382 m | .9337 deg | 10 | 46 | 68 | 76 | 78 |
| seq13 shard3 (87) | learned RADIO-64 | **.1921 m** | **.6400 deg** | **11** | **58** | **75** | **77** | **81** |

On the cross-route inventory, median translation improves 19.4%, rotation
31.5%, and .25m/2deg recall improves by 12/87 (13.8 percentage points).

MoGe3 remains materially useful after the descriptor upgrade.  On seq13 shard3
for the 0.25 m support branch, view-only refinement gives 43/77/77/80 hits at
.25/.5/1/2 m; MoGe3 metric-scale plane refinement improves this to
55/77/78/80 hits and a .2004 m/.6545 deg median.  Thus query depth scale is not
being ignored: it is estimated per query and used as a surface constraint after
the RADIO/plane correspondences initialize the pose.

## Resource result

The atlas contains 2,010 finite planes, 28,817 metric texels, and 91,976
anonymous prototypes.  Its descriptor array shrinks from `(91976,1280)` to
`(91976,64)` float16.  The complete compressed atlas shrinks from 228,340,852
bytes to 21,743,646 bytes (10.5x); the projection artifact is 306,180 bytes.

## Post-label upper-bound decomposition

This diagnostic is never deployable and is not used for training or selection.
It asks whether either of the two already-computed MoGe3 support branches has a
correct pose.

| route | deployed .1/.25/.5/1/2m hits | branch-union oracle hits | recoverable selector misses |
|---|---|---|---|
| seq10 | 20/64/83/87/88 | 23/70/85/87/88 | 3/6/2/0/0 |
| seq13 shard3 | 11/58/75/77/81 | 19/66/78/79/81 | 8/8/3/2/0 |

The coarse basin is therefore close to saturated; the remaining measurable
headroom is predominantly tight-threshold, label-free branch selection and
sub-token/within-plane precision.  Enlarging the candidate pool is not the
first priority.

## Legacy uncertainty-weighted pose upgrade

This earlier bounded optimization was a mainline **GO** under its original
contract.  The solver used
the anonymous atlas fields that were previously stored but not consumed by the
pose optimizer: the 3x3 world-point covariance, plane-pixel purity, and metric
plane-depth dispersion.  It adds no mapping image, image path, source-view
identity, query label, or new learned parameter.

For every frozen query token, only the currently best valid 3D hypothesis was
retained.  World covariance is propagated through the pinhole Jacobian;
metric depth dispersion supplies a second conservative image-space variance;
their maximum is added to the exact variance of a uniform four-pixel RADIO
cell and divided by clipped plane purity.  A Huber six-DoF solve then uses
these fixed variances and plane-balanced weights.  It fails closed below 12
rows or two physical planes, must preserve 95% of the original unique-token
support, and cannot move the camera by more than 0.5 m or rotate it by more
than 5 degrees.

The frozen sequence was `RADIO/plane pose -> MoGe3 plane+scale -> uncertainty
reprojection refinement`.  Reversing the final two stages was tested and was
inferior.  The old and refined dual-surface endpoints are finally compared by
the already defined uncertainty-normalized token likelihood; labels are
opened only after that selection artifact is sealed.

| route | endpoint | median translation | median rotation | .1m/1deg | .25m/2deg | .5m/5deg | 1m/10deg | 2m/45deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| seq10 (88) | prior RADIO-64 | .1623 m | .4455 deg | 20 | 64 | 83 | 87 | 88 |
| seq10 (88) | uncertainty upgrade | **.1605 m** | **.3983 deg** | 19 | **70** | 82 | 87 | 88 |
| seq13 shard3 (87) | prior RADIO-64 | .1921 m | .6400 deg | 11 | 58 | 75 | 77 | 81 |
| seq13 shard3 (87) | uncertainty upgrade | **.1654 m** | **.5291 deg** | **17** | **63** | **77** | **78** | 81 |

The seq10 result is not Pareto-dominant at every discrete threshold, but its
primary .25m/2deg count and both medians improve.  The configuration was then
frozen and replayed once on seq13 shard3: translation and rotation medians
improve by 13.9% and 17.3%, and every reported held threshold is non-decreasing.
Both held uncertainty-refinement branches, the dual selector, and the final
old-vs-new selector were repeated byte-identically.

The post-label old/new union still reaches 21/66/77/79/81 held queries at the
five thresholds, versus the deployed 17/63/77/78/81.  The remaining selector
headroom is therefore only 4/3/0/1/0 queries; another candidate-pool expansion
is not justified by this result.

## Negative controls retained

- A seq10-fitted low-parameter branch-quality calibrator did not generalize to
  seq13 shard3 and is KILL.
- Unified loose/strict EM refinement improved seq10 but not seq13 shard3 and is
  KILL.
- Inverse-homography pseudo-subtoken measurements degraded the held result and
  are KILL; a homography residual is not a learned feature location.
- Replacing each RADIO token center by the centroid of observed same-plane mask
  pixels improved one loose-branch .5m count on seq10 but degraded the selected
  endpoint (.162m to .188m and 64 to 60 hits at .25m/2deg).  It is KILL as a
  default and held labels were not opened: segmentation geometry and the RADIO
  receptive-field center are not interchangeable feature measurements.
- Many query fragments per map plane is now an optional, mass-balanced MoGe3
  association policy, but it did not improve the default seq10 endpoint and is
  not the default.
- Cross-atlas hand weighting and uncertainty-normalized selection did not pass
  the cross-route gate.

## Sealed artifacts

- Projection: `stmarys_chart_local_radio_projection_64d_v2.npz`, file SHA256
  `cf422024b7ca736f9b66db9176c9cca5ac3d70208498e5cc4e889f89c18845b2`,
  content `d3a5e506d93ad5a65d21603841e938fbb5fcc2386e27483e8e1ed337624b9ef8`.
- Learned atlas: `stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz`,
  file `8083e4bbd607d98b6885eb2251cd44ae98f766a7f62d6101303f5368a883d853`,
  content `5ca4549a3c32e917bac41c7b6dcfd1a27fb8556d08171773592335b633feb641`.
- Seq10 selected pose: `learned64d_strict_seq10_dual.npz`, file
  `de827e2fa4d3ae82bfd6ea9a93baa4fb1ac773b12cc6d6176666b23a97fbce4c`,
  content `50e4bc0f3775c20c2696eade1cc82fdacac880f5399c94515e7938f9c8d1f3dc`.
- Seq13 shard3 selected pose: `learned64d_strict_shard3_dual.npz`, file
  `f69b28d0f1696009f878933a314b3453902f79de2bdf50fd2948482a12bb997b`,
  content `1ac7998fc4ee3cf3947d729594025872c1afb9b3a6e5268a04fd40498178bb9d`.
- Seq10 uncertainty-upgraded selected pose:
  `learned64d_strict_seq10_moge_uncertainty_dual_vs_old_uncertainty_selected.npz`,
  file `3eb302e9be632180037097d94db2edce6ccd1b8c6a528c6d22d8e2bfa95b22cc`,
  content `1443f92b89cc0fd5132eb55afc5209137223ab36170943df979106511e9f9a20`.
- Seq13 shard3 uncertainty-upgraded selected pose:
  `learned64d_strict_shard3_moge_uncertainty_vs_old_selected.npz`, file
  `60e84385f36946b730ee545b2fba742d2e2a69efe983ffc069107c9a615d1f13`,
  content `89f2a24d27ce15d68613d1df87dc65bdfa3a1daf0d57ac8f368962662c496b5f`.

All paths above are under
`output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1/`.

## Next bounded optimization

1. Replace the current heuristic two-branch selector with a mapping-only or
   separately calibrated uncertainty model; freeze it before another route.
2. Learn an explicit sub-token chart coordinate/confidence head rather than
   interpreting a homography fit as a measurement.
3. Make the surface optimizer consume the stored world covariance, purity, and
   depth dispersion directly in its residual covariance, not only in matching.
4. Add a multi-plane layout residual and a rank/conditioning report for query
   scale observability.  Keep single-plane scale fail-closed.
5. Perform the final claim on a new scene/route split that has not participated
   in development.

The third item was implemented, but a later uncertainty-semantics audit found
that the stored world covariance is the scatter of the finite surface
footprint, not the covariance of its estimated centroid.  Likewise, source
depth dispersion is view-conditioned surface spread rather than query-centroid
measurement uncertainty.  Those quantities remain useful map descriptors but
must not be added to a V5 centroid likelihood.  The superseding implementation
and results are below.

## Continuous-coordinate and probabilistic follow-up

The post-label continuous-coordinate oracle decisively passed the external
review's P0 screen.  On seq10, selecting the best already-generated h0.25
candidate with labels improves the token-centre endpoint from 19/70/82/87/88
to 35/81/85/88/88 hits at .1/.25/.5/1/2m; exact projection onto the associated
local surface reaches 88/88.  This is diagnostic only, but it locates the main
remaining headroom in within-token coordinate/association rather than another
global retrieval expansion.

Two deliberately bounded P2/P3 controls were then rejected:

- A uniform-prior, explicit-null, multi-hypothesis EM surface solver fell to
  19/62/81/87/88 on seq10 after correcting its null support from an erroneous
  8-pixel width to the actual 4-pixel correspondence boundary.
- A normalized Gaussian-plus-uniform-null selector with per-token arithmetic
  hypothesis marginalization avoided multiplicity bias but did not improve the
  final candidate choice.  The learned match head separates positive/negative
  mapping pairs only weakly and must not be used as a hard rejection gate.

Both controls remain useful fail-closed implementations and negative evidence;
neither was opened on a new held split.

## Mapping-only sub-token head and V5 mainline candidate

The accepted P1 implementation stores no mapping RGB or source-view identity.
It consumes anonymous RADIO-64 atlas prototypes and predicts a continuous
offset inside the original 4x4 query token plus an isotropic centroid variance.
The validation route seq9 is absent from prototype construction.  Its views are
split deterministically: even sorted observations calibrate one closed-form
coordinate shrinkage scalar, while odd observations evaluate it.  The frozen
scalar is 0.7745875684.

On the untouched half of mapping seq9, token-centre median/P90 error is
1.3413/2.0287 pixels and the predicted coordinate is 1.1317/1.9202 pixels;
the median improvement is 15.63%, 66.41% of pairs improve, and uncertainty
quartiles have monotonically increasing mean error.  A full rerun with the
same seed is byte-identical (file SHA256 `637472f9...1d05`).

The V5 pose sequence is:

`finite-plane retrieval -> anonymous RADIO-64 prototype match -> mapping-only
sub-token coordinate -> grouped PnP -> anonymous view-geometry filter ->
MoGe3 plane normal/offset + one scale -> centroid-uncertainty refinement`.

For V5, the uncertainty refinement consumes only the predicted query-centroid
variance (modulated by plane purity).  It explicitly excludes map footprint
scatter and source-view depth dispersion, fixing the semantic double use in the
legacy branch.

| route | endpoint | median translation | median rotation | P90 translation | P90 rotation | .1m/1deg | .25m/2deg | .5m/5deg | 1m/10deg | 2m/45deg |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| seq10 (88) | previous mainline | .1605 m | .3983 deg | — | — | 19 | 70 | 82 | 87 | 88 |
| seq10 (88) | V5 candidate | .1655 m | .4104 deg | **.3255 m** | **.9344 deg** | 19 | 70 | **85** | 87 | 88 |
| seq13 shard3 (87) | previous mainline | .1654 m | .5291 deg | .9576 m | 3.4678 deg | 17 | 63 | 77 | 78 | 81 |
| seq13 shard3 (87) | V5 candidate | .1675 m | **.5262 deg** | **.5240 m** | **1.8339 deg** | **19** | **66** | 77 | **80** | 81 |

The held direction is favorable but not statistically conclusive at this sample
size: exact McNemar tests do not reject equality.  Catastrophic retrieval
failures also dominate the mean, so claims use the pre-existing median, P90,
and threshold-recall protocol.  V5 is therefore a validated auxiliary/mainline
candidate, not a final production claim; it is Pareto non-decreasing in all
reported recall thresholds on seq10 and the frozen held shard.

A stricter mode-specific head that exactly replayed the atlas medoid-then-
farthest four-mode construction passed mapping validation (12.9% median pixel
improvement) but reduced the seq10 1m candidate oracle from 88/88 to 86/88 and
is KILL.  The mean-prototype training distribution plus shrinkage is the retained
operating point.

## MoGe3 scale observability

The MoGe3 scale latent now has an explicit rank diagnostic.  The robust data
Jacobian excludes the weak log-scale prior, then a Schur complement marginalizes
the six pose variables.  A solve is called data-observable when this conditional
information is at least the prior information.

- seq10: 88/88 diagnosed and observable; data/prior ratio min/median/max is
  540.7/2924.5/8292.2.
- seq13 shard3: 83/83 solvable cases observable; ratio is
  163.0/1972.0/7102.6.

Thus scale is genuinely constrained by the multi-plane data in these solved
queries; it is not merely inherited from the weak metric prior.  This paragraph
describes the original diagnostic run; the superseding implementation below
turns the same Schur-information test into an explicit acceptance gate.

## V5 sealed artifacts

All files are under the surface-coordinate upgrade output directory.

- Mapping-only head: `stmarys_mapping_canonical_subtoken_head_shrunk_v4.npz`,
  file SHA256 `637472f9397e14ff545b4a4b2713ad64fd8d8ec3017dcef8fbc1382e6b1e1d05`,
  content `87548a3bb282935e7a9822083bf262315e9a0be20ecdee047de9fe3b457a4276`.
- Seq10 V5 correspondence/final pose: files `42610d4b...572b` and
  `2fca7046...e6f`; final content `25c22117...938f`.
- Held seq13 shard3 V5 correspondence/final pose: files `6aa841c2...fdc0`
  and `48e605f3...cf0`; final content `169cc771...7739`.
- Paired post-label audits: seq10 content `6b563b3e...f7a1`; held content
  `02b5f327...6458`.
- Scale observability audits: seq10 content `fbb22c6b...fc5e`; held content
  `1539a22b...c1f3`.

## Updated next bounded optimization

1. Replace hard per-token candidate commitment with a learned, mapping-only
   mode posterior whose null calibration is demonstrably stronger than the
   current 0.52/0.48 positive/negative probabilities.
2. Predict map-centroid uncertainty separately from surface-footprint extent;
   the present atlas does not yet store a statistically valid map-centroid
   covariance.
3. Evaluate V5 on a new preregistered route/scene.  The current held shard gives
   consistent effect direction but insufficient power for a final claim.
4. Preserve MoGe3 scale and its Schur observability diagnostic; do not replace
   it with an assumed metric-depth truth.

## 2026-09-03 continuous surface-coordinate implementation and gates

The next external-review cycle has now been implemented through the full
correspondence/PnP/MoGe/uncertainty chain.  The outcome is deliberately split
between a proven mechanism and the retained deployment operating point.

### P0 oracle: decisive GO for continuous association headroom

On seq10, the frozen token-centre candidate has 19/70/82/87/88 hits at
.1/.25/.5/1/2m.  Selecting a correct continuous coordinate within the already
frozen association raises this to 35/81/85/88/88; the projective local-surface
oracle reaches 88/88.  This clears the predeclared five-point strict-recall
screen and proves that continuous coordinate/association error, rather than
another global candidate expansion, is a real limiting factor.

### Joint mapping-only query/subtoken + chart-UV head

A new head predicts five quantities from a query RADIO-64 descriptor and an
anonymous map prototype: query sub-token mean/variance, metric chart-UV
mean/variance, and match/null logit.  Seq9 is absent from prototype fitting;
even seq9 mapping observations calibrate scalar means/variances and odd seq9
observations are the untouched representation evaluation.

For the accepted mapping diagnostic, query sub-token median/P90 changes only
from .5476/.9148 px to .5378/.8899 px (1.80% median), while continuous chart UV
changes from .1214/.2215 m to .1144/.2122 m (5.79% median).  Both Gaussian NLLs
improve and both uncertainty quartile sequences are monotonic.  The latter is
the substantive signal.

The initial runtime V6 correspondence inventory stores the predicted continuous world
centroid plus a rank-two tangent-plane centroid covariance.  It explicitly
does not reinterpret the token surface footprint or source-view depth spread
as centroid noise.  Covariance propagation now uses the analytic
SIMPLE_RADIAL Jacobian; a finite-difference regression test covers the full
distortion derivative.

Directly using this head throughout PnP is unsafe: seq10 view refinement drops
.5m and 1m recall.  Using stable V5 PnP/MoGe only as initialization and enabling
continuous coordinates in the final bounded surface refinement gives:

| seq10 endpoint | median t | median R | P90 t | P90 R | .1 | .25 | .5 | 1 | 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retained V5 | .1655m | .4104deg | .3255m | .9344deg | 19 | 70 | 85 | 87 | 88 |
| conditional continuous mean | .1487m | .3847deg | .3426m | .9021deg | 24 | 68 | 85 | 87 | 88 |
| match-marginal continuous mean | .1466m | .3737deg | .3200m | .8934deg | 23 | 68 | 85 | 87 | 88 |

The match-marginal branch improves 52/88 translations and 54/88 rotations;
the mean rotation delta bootstrap interval excludes zero.  It nevertheless
loses two net .25m hits, so it is a high-precision research branch rather than
a promoted replacement, and held query data remained closed.

Three attempted safeguards were also frozen and rejected:

- mapping-only scalar chart-offset shrinkage passes seq9 but worsens seq10 PnP;
- a proper Gaussian+uniform-null hard-hypothesis posterior loses one .1m and
  one .25m hit, so nearest-reprojection remains the default and the posterior
  policy is opt-in only;
- scoring V5 and surface candidates on both coordinate inventories removes
  candidate self-grading.  Endpoint Pareto selection is recall-neutral but has
  no gain.  A fixed 0,1/4,1/2,3/4,1 SE(3) line search significantly improves
  mean translation/rotation and preserves .5/1/2m, but still loses one .25m
  hit; it therefore also remains non-promoted.

A four-mode, atlas-like mapping head was tested to address train/deploy mode
mismatch.  Its held mapping chart-UV median gain is only 3.34%, below the 5%
gate, so it was killed before any query evaluation.  This indicates the next
head must model a genuine multi-modal coordinate posterior or richer geometric
conditioning; merely duplicating the anonymous atlas mode inventory is not
enough.

### P2/P3 and P4 decisions

The uniform-prior null-aware EM solver and the proper null selector remain
negative controls; neither improved the hard V5 operating point.  The new
cross-coordinate selector makes candidate self-consistency explicit and
provides a reusable phase-separated likelihood harness, but is not promoted by
the current seq10 gate.

MoGe3 scale observability is now an actual acceptance condition rather than a
report-only diagnostic.  The data-only log-scale Schur information must be at
least as large as the weak prior information, otherwise the joint update is
rejected and the input pose/unit scale is retained.  Seq10 passes for all 88/88
queries, so the new gate reproduces the previous metrics exactly.  Existing
held audit data also show all 83 solvable cases above this threshold.  Thus the
current errors cannot be attributed to an unconstrained MoGe scale latent.

### Frozen decision

V5 remains the deployable research operating point.  Continuous chart
coordinates are now implemented with the correct covariance and solver
semantics and show real strict-accuracy headroom, but do not yet satisfy the
required `.25m improves AND .5/1/2m do not decrease` gate.  The next bounded
step is a 2--4 mode mapping-only coordinate posterior conditioned on query ray,
query plane geometry, anonymous prototype direction/range, and local map
uncertainty, followed by one unified null-aware surface solve.  No further
threshold or endpoint-selector search is justified on seq10.

Key new artifacts are under
`output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1/`:

- continuous head `stmarys_mapping_surface_coordinate_head_v7b.npz`, content
  `b4be8ec5...f7664`;
- V6 surface correspondences `learned64d_strict_seq10_h025_surfacecoord_v7_corr.npz`,
  file/content `5b9438e8...eaaa` / `9c2fd49a...c334`;
- local continuous endpoint `learned64d_strict_seq10_h025_surfacecoord_local_v9_final.npz`,
  file/content `9bc60abe...aea5` / `f1681fc8...ad6`;
- match-marginal endpoint and paired audit, content `25790118...dbca` and
  `523174cc...1fea`;
- cross-coordinate line-search endpoint and paired audit, content
  `9f9f8926...d8e6` and `63fdcde5...a1f`;
- scale-gated MoGe endpoint `learned64d_strict_seq10_h025_moge_scalegate_v17.npz`,
  content `85653873...e1e`.

At that checkpoint the complete goal-maplet regression suite passed 954 tests,
with only the pre-existing PyTorch scatter-reduce beta warning.

## Cell-bounded V7 correction and final 2026-09-03 decision

A final contract audit found that V6 bounded the *predicted offset* to
plus/minus 0.5 m but did not guarantee that the resulting chart coordinate
remained inside its original 0.5 m atlas cell.  This was a real semantic bug:
the prediction could cross a cell boundary and silently change the finite
surface support associated with an anonymous prototype.

V7 fixes the contract by clipping the final predicted chart coordinate to the
half-open metric bounds of the prototype's original cell.  The correspondence
artifact records `chart_uv_metric_cell_support_enforced=true` and stores both
the actual chart-UV measurement and its cell lower bound; downstream loaders
replay `lower <= uv < lower + cell_size` for every row.  The world-coordinate covariance remains a
rank-two tangent-plane centroid covariance and is propagated with the full
SIMPLE_RADIAL Jacobian.  Focused schema, covariance, finite-difference, and
adversarial tests pass.

The corrected seq10 result remains scientifically informative but does not
clear the promotion gate:

| seq10 endpoint | median t | median R | P90 t | P90 R | .1 | .25 | .5 | 1 | 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retained V5 | .1655m | .4104deg | .3255m | .9344deg | 19 | 70 | 85 | 87 | 88 |
| cell-bounded V7 conditional mean | .1495m | .3892deg | .3434m | .9011deg | 23 | 69 | 85 | 87 | 88 |
| cell-bounded V7 match-marginal mean | .1487m | .3771deg | .3255m | .8971deg | 23 | 68 | 85 | 87 | 88 |

Against V5, the conditional arm improves median translation by 1.59 cm and
median rotation by 0.021 degrees.  Its paired mean changes are smaller and not
conclusive: -0.00283 m (95% CI -0.00991,+0.00415) and -0.0142 degrees
(95% CI -0.0330,+0.00455).  It gains four 0.1 m cases but loses one net
0.25 m case, so neither V7 endpoint is promoted and held data remains closed.

The cross-coordinate selector now also has a zero-weight
`pareto_line_search`: a fixed SE(3) line step is admissible only if its
likelihood is no worse than the V5 endpoint under both the point-centre and
continuous-surface coordinate inventories.  On seq10 it selects the V5
endpoint for 49/88 queries and nonzero fractions for 39/88.  It improves the
median translation from .1655 m to .1618 m but leaves every reported recall
threshold exactly unchanged.  This closes the candidate-self-grading issue,
but it is not the requested five-point strict-recall gain and is therefore a
safe diagnostic selector rather than a new operating point.

Corrected V7 artifacts:

- replayable correspondence build: `learned64d_strict_seq10_h025_surfacecoord_cell_replayable_v23_corr.npz`,
  file/content `041d86a1...098f` / `bcf24b28...d99a`;
- conditional endpoint: `learned64d_strict_seq10_h025_surfacecoord_cell_replayable_v23_final.npz`,
  file/content `d103b7f4...3933` / `065b020d...30b8`;
- conditional-vs-V5 paired audit: content `475e9164...496f`;
- match-marginal endpoint (diagnostic, pre row-level replay arrays):
  `learned64d_strict_seq10_h025_surfacecoord_cell_marginal_v19_final.npz`,
  file/content `1b51a93d...b604` / `ad8c61cb...34b5`;
- replayable Pareto line selector: `learned64d_strict_seq10_h025_surfacecoord_cell_replayable_pareto_line_v24.npz`,
  file/content `8d5b65ff...ef26` / `48a29ac6...f8c9`;
- Pareto line paired audit: content `b26669f0...65c4`.

The earlier V18/V19/V22 artifacts reproduce the reported numerical poses but do not
carry row-level cell-bound replay arrays; they are superseded diagnostics and
must not be used as strict V7 authority.

Final decision: retain V5; retain V7 and the Pareto selector as bounded research
components.  Do not tune more seq10 thresholds or open held data.  The next
scientifically justified change is a mapping-only multi-modal coordinate
posterior with richer ray/plane/prototype conditioning, trained and calibrated
before a new preregistered route is opened.

After the row-level replay hardening and Pareto-line tests, the complete
goal-maplet suite passes 957 tests; `py_compile` and `git diff --check` also
pass.  The only warning is the pre-existing PyTorch scatter-reduce beta notice.

## Mapping-only ambiguity and geometry-context follow-up

The requested multi-modal coordinate posterior was implemented as a
three-component isotropic mixture and evaluated only on mapping data.  Its
held-mapping-route posterior mean improves chart-UV median error from
`.121401 m` to `.114334 m` (5.82%), while a post-label best-component oracle
would reach `.093609 m` (22.89%).  However, the learned posterior has mean
effective mode count only `1.00381`: almost all probability collapses onto one
component and the apparent oracle gain lives in negligible-probability shadow
modes.  The mapping gate now requires effective mode count at least `1.25`.
The non-collapse rerun therefore correctly returns **KILL**.  The earlier
mixture artifact that predated this gate is superseded and cannot be used as
evidence for ambiguity modelling.  No seq10 or held-query data was opened for
this arm.

A second mapping-only head adds five deployment-available geometric inputs to
the frozen RADIO-64 pair: undistorted query token-centre ray `(x,y)`, anonymous
prototype phase inside its metric 0.5 m cell `(u,v)`, and absolute query
plane/ray incidence.  It still stores no mapping RGB, path, or source-view
identity.  On the disjoint seq9 mapping validation route it improves query
subpixel median/P90 by 3.27%/6.39% and chart-UV median/P90 by 5.83%/5.00%; both
Gaussian NLLs improve and uncertainty-error quartiles are monotonic.  A
same-seed complete retraining reproduced all 13 arrays and the compressed NPZ
byte-for-byte.

On seq10, the direct context endpoint changes V5 recall from
`19/70/85/87/88` to `19/71/85/87/88`; the fixed Pareto line search reaches
`18/73/85/87/88`.  Thus the geometry context recovers three additional
0.25 m cases without any coarse-threshold loss, but loses one 0.1 m case and
does not meet the preregistered five-percentage-point promotion threshold.
It remains a useful continuous-coordinate research component; V5 remains the
operating point, selector tuning stops, and held data remains unopened.

Sealed follow-up artifacts:

- non-collapsed mixture gate: `stmarys_mapping_surface_coordinate_mixture3_head_v26_noncollapse_gate.npz`,
  content `747c3468...a9df0` (**KILL**, effective modes `1.00381`);
- geometry-context head: `stmarys_mapping_surface_coordinate_context_head_v27.npz`,
  file/content `246dc3e3...92a0` / `3d854721...74f0e`;
- byte-identical deterministic replay: `stmarys_mapping_surface_coordinate_context_head_v30_deterministic_replay.npz`;
- context correspondence/final pose: files `120e5a1f...00c2` / `e0356ef6...a5fd`,
  contents `4a1fc0a2...5488f` / `a3241805...f178`;
- context Pareto pose: file/content `8e423048...76c9` / `f9b893bd...5050`;
- context direct/Pareto paired audits: contents `8f538a86...58f6` /
  `42e2a5c8...a885`.

Scientific decision: the current pair representation is empirically close to
unimodal, so entropy forcing would fabricate modes rather than recover
evidence.  The next justified step is not another selector or another mixture
head.  It is to give the continuous head genuine local surface context (a
small neighbouring RADIO/UV field with physical offsets) and then consume its
calibrated coordinate posterior in one probabilistic surface solve.

## Frozen homography-context coordinate result

Two further bounded tests separate capacity from missing spatial evidence.
Simply increasing the geometry-context MLP to two 192-D hidden layers improves
mapping seq9 chart-UV median error by 9.09%, but degrades the seq10 pose to
`18/67/82/87/88`; even the fixed Pareto selector loses one case at each of
0.1/0.25/0.5 m.  This is a clean **KILL** for capacity scaling and evidence of
mapping-to-query overfit rather than under-capacity.

The successful alternative uses the already-computed coarse plane homography
as context, not as a coordinate label.  For every RADIO/prototype pair the
head receives the normalized two-dimensional residual between the homography
projection of the token centre and the prototype metric UV, plus a reliability
bit.  RADIO still predicts the actual sub-token/chart coordinate and calibrated
variance.  The homography is fit before labels and poses are opened; invalid
fits retain the historical correspondence row and contribute zero context.

The latter separation fixed a real implementation mistake in the first V11
build: reliability `<4` had accidentally deleted 161 correspondence rows.  The
bad V35 endpoint is invalid and superseded.  The corrected V37 inventory has
exactly 79,536 rows and is byte-identical to V5 for names, offsets, token IDs,
plane/texel/prototype identities, cameras, radial coefficient, and RADIO
score.  Only the learned continuous query/chart measurements differ.

Mapping-only seq9 validation is substantially better:

| coordinate | control median/P90 | V11 median/P90 | median improvement |
|---|---:|---:|---:|
| query sub-token | .547614/.914771 px | .526695/.859403 px | 3.82% |
| metric chart UV | .121401/.221495 m | .091121/.185968 m | **24.94%** |

Both NLLs improve and uncertainty-error quartiles remain monotonic.  A full
same-seed retraining reproduces all arrays, metadata, and compressed NPZ
byte-for-byte.

Corrected seq10 performance is:

| endpoint | median t | median R | P90 t | P90 R | .1 | .25 | .5 | 1 | 2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| retained V5 | .165459m | .410394deg | .325505m | .934385deg | 19 | 70 | 85 | 87 | 88 |
| V11 homography context | **.155924m** | **.405758deg** | .333088m | .953678deg | **25** | 70 | 84 | 87 | 88 |
| fixed Pareto line | .159258m | .445825deg | -- | -- | 19 | 71 | 84 | 87 | 88 |

V11 therefore clears the requested five-percentage-point strict-recall screen:
0.1 m gains 7 cases and loses 1, for net `+6/88 = +6.82pp`; the exact paired
McNemar p-value is `.0703125`.  It also improves both medians.  It does not yet
clear the complete promotion contract because one 0.5 m case is lost, and the
existing Pareto selector neither preserves the strict gain nor restores that
case.  The correct status is **strong research GO / mainline promotion HOLD**.
V5 remains the operating point and held data remains closed.

Sealed artifacts:

- V11 mapping head and deterministic replay: files `5e655a1e...1287` for both,
  content `0e26c026...e136`;
- corrected V37 correspondence/final pose: files `9ef8bc56...9f1b` /
  `31f1816b...eff`, contents `32227f9f...8394` / `036f37e7...6f78`;
- direct V37-vs-V5 paired audit: file/content `7e14194a...01e6` /
  `1553b0e6...06f4`;
- fixed Pareto V38 and audit: files `1f1d3edc...ecbf` /
  `12fd382f...5b57`, contents `d236b954...939f` / `f4ee77ba...105b`;
- deep-capacity negative head: file/content `e7a9b6eb...7960` /
  `2c1dd813...3d7d`.

The remaining bottleneck is no longer whether the coarse homography carries
useful spatial information—it clearly does.  It is how to propagate this
posterior without one tail regression.  The next implementation should unify
V5 and V11 as calibrated multi-hypothesis surface observations inside one
solver, rather than train another selector or use the homography as a hard
coordinate replacement.

After V8--V11, correspondence-inventory parity tests, and the invalid-context
regression test, the complete goal-maplet suite passes 967 tests;
`py_compile` and `git diff --check` pass.  The sole warning remains the
pre-existing PyTorch scatter-reduce beta notice.

## Cross-split coordinate posterior decision (V62--V66)

Later experiments tested whether V11 should replace V5, whether the two poses
could be collapsed by another verifier, and whether they should instead remain
an explicit small pose set.  The conclusion is now sharper: **V5 remains the
Top-1 operating point; `{V5,V11}` is a proposal-level Top-2 GO.**  V11 is not a
stable single-pose replacement, but it is a stable complementary hypothesis.

The evaluation includes seq10, three already-open seq13 shards, and a fourth
seq13 shard that had not been used by this V5/V11 upgrade before its pose
inventories were frozen.  Each Top-2 NPZ was written and hashed before query
poses were opened.  It stores two 4x4 poses and their validity bits, not source
RGB, source image paths, source-view identities, or labels.

| split | V5 Top-1 .1/.25/.5/1/2m hits | frozen Top-2 hits | net gain |
|---|---|---|---|
| seq10 (88) | 19/70/85/87/88 | 25/74/85/87/88 | +6/+4/0/0/0 |
| seq13 shard0 (88) | 18/63/78/82/82 | 28/69/80/82/82 | +10/+6/+2/0/0 |
| seq13 shard1 (88) | 16/65/75/79/80 | 22/67/78/79/81 | +6/+2/+3/0/+1 |
| seq13 shard2 (87, upgrade-blind) | 21/66/78/80/81 | 25/71/79/81/82 | +4/+5/+1/+1/+1 |
| seq13 shard3 (87) | 19/66/77/80/81 | 30/68/78/80/81 | +11/+2/+1/0/0 |

The shard2 result is particularly important.  V11 alone is worse than V5 at
the strict threshold (18 versus 21 hits), so choosing it as the new mainline
would have failed.  Nevertheless, the frozen two-pose set improves all five
thresholds on that split.  This is genuine proposal recall, not a claim that a
label-free final selector already achieves the oracle numbers.  Downstream
plane geometry must still collapse the set to one pose when a single answer is
required.

Several bounded attempts to perform that collapse were rejected:

- The mass-balanced MoGe3 verifier was changed to allow multiple disconnected
  query fragments to support the same physical map plane.  This is the right
  sparse-occlusion semantics, and every map plane retains unit total squared
  weight, but its four validation deltas were `+2/0/0/0/0`,
  `+2/-1/0/0/0`, `+3/-1/-1/-2/+1`, and `+1/0/0/0/0`.  It therefore remains a
  diagnostic and does not select the production Top-1 pose.
- A fixed logistic selector trained only on seq10 candidate-side evidence was
  also checked as a bounded prototype.  It improved strict recall on some
  splits but consistently lost medium-threshold cases, so it was not
  materialized or tuned further.
- A fixed quarter-step between the two poses and a two-coordinate Pareto line
  search improved medians on some splits but had inconsistent recall.  They
  remain negative controls.
- Expanding plane retrieval from Top-10 to Top-20 on shard1 increased median
  correspondence count from 802.5 to 1213.5, but raw PnP 2m/45 recall fell
  from 78 to 77 and its candidate oracle fell from 81 to 79.  Candidate volume
  is not the current bottleneck.
- A context-fused plane ranker improved seq10 raw 1m PnP by two cases but the
  complete final endpoint retained exactly `19/70/85/87/88`, with slightly
  worse translation median.  It was not opened on held shards.

The many-fragment run exposed and fixed a real fail-closed bug: a sealed
unusable non-finite pose could make `cv2.projectPoints` return `None`.  Such a
pose now produces zero verifier rows and infinite errors rather than crashing
or silently entering geometry selection; a directed regression test covers
the case.

Key sealed artifacts, all below the surface-coordinate upgrade directory:

- seq10 Top-2: `learned64d_strict_seq10_h025_coordinate_top2_v63.npz`,
  file/content `09314f72...37d8` / `397012a0...0cdd`;
- shard0/1/3 Top-2 files: `477a8072...4fe7`, `94497533...bea6`, and
  `069aac12...1dd8`;
- upgrade-blind shard2 V5 final: file/content `5337f166...ebf3` /
  `52755dc7...c577`;
- upgrade-blind shard2 V11 final: file/content `9124bc34...c1d` /
  `144ecfd0...73fc`;
- upgrade-blind shard2 Top-2: `learned64d_strict_shard2_h025_coordinate_top2_v66.npz`,
  file/content `1d368736...45b3` / `64187396...253b`.

This result changes the next optimization target.  More plane candidates,
hand-tuned interpolation, and another binary selector are not justified.  The
useful next step is an independent plane-layout/material-consistency verifier
that scores the frozen Top-2 set without reusing the same RADIO reprojection
residuals that generated it.  Until that exists, report V5 as Top-1 accuracy
and `{V5,V11}` separately as Top-2 proposal recall.

After the V62 many-fragment verifier, V63/V66 hypothesis-set implementation,
non-finite-pose regression, and the complete shard2 replay, the full
goal-maplet suite passes 994 tests.  All new pose-set artifacts pass their
strong loader; `py_compile` and `git diff --check` pass.  The only warning is
the pre-existing PyTorch scatter-reduce beta notice.

## Independent sparse/dense geometry consensus (V67--V74)

The previously unresolved Top-2-to-Top-1 collapse is now a reproducible
label-free improvement.  The frozen rule keeps V5 unless V11 is strictly
better under **both** of the following independent query/map geometry tests:

1. the V62 sparse physical-plane normal/offset objective after fitting one
   MoGe3 query scale; and
2. full-map 2DGS-rendered versus MoGe3 normal agreement within 20 degrees,
   scored as good rays divided by *all* valid query MoGe3 pixels.  Missing map
   render support therefore counts as failure rather than disappearing from
   the denominator.

There is no fitted selector, continuous fusion weight, held-split threshold,
source RGB, or source-view identity.  The selected NPZ is written, hashed, and
strongly reloaded before either label-bearing endpoint evaluation is opened.
The 20-degree convention and the strict two-test AND rule were fixed on seq10
and then replayed unchanged on all four seq13 shards.

| split | queries | V5 .1/.25/.5/1/2m hits | geometry-consensus hits | delta | V11 selected |
|---|---:|---|---|---|---:|
| seq10 | 88 | 19/70/85/87/88 | **21/71/85/87/88** | +2/+1/0/0/0 | 17 |
| seq13 shard0 | 88 | 18/63/78/82/82 | **21/65/79/82/82** | +3/+2/+1/0/0 | 21 |
| seq13 shard1 | 88 | 16/65/75/79/80 | **19/65/75/79/80** | +3/0/0/0/0 | 16 |
| seq13 shard2, upgrade-blind | 87 | 21/66/78/80/81 | **22/68/79/81/82** | +1/+2/+1/+1/+1 | 24 |
| seq13 shard3 | 87 | 19/66/77/80/81 | **21/67/77/80/81** | +2/+1/0/0/0 | 13 |
| total | **438** | **93/330/393/408/412** | **104/336/395/409/413** | **+11/+6/+2/+1/+1** | **91** |

All five splits are Pareto non-decreasing at every reported pose threshold.
The strict 0.1 m / 1 degree recall rises from `93/438 = 21.23%` to
`104/438 = 23.74%`, an absolute gain of `+2.51pp`; the 0.25 m / 2 degree
recall rises by `+1.37pp`.  This is the first real Top-1 gain produced from the
complementary V5/V11 hypotheses rather than an oracle Top-2 recall statement.
It is a research operating-point GO, while `production_eligible` remains false
until the full runtime and broader-scene contract is completed.
Paired transitions across all splits are gain/loss `14/3`, `9/3`, `2/0`,
`1/0`, and `1/0` from the strict through coarse thresholds: the aggregate
improvement is not an artifact of merely exchanging equal numbers of cases.

The seq10 consensus artifact is
`learned64d_strict_seq10_h025_geometry_consensus_v68.npz`, file/content
`6cab0b18...c48` / `9244849e...adb`; its evaluation content is
`127fc141...f99`.  The seq13 shard0--3 selected pose file/content pairs are
respectively `95f0861c...05a` / `a182e679...330`,
`81e77f6a...ed6` / `1dc099cb...e8`,
`7dac35d4...5c1` / `a1f58cf6...b0d`, and
`cf35ce21...6f0` / `e26b9d13...ce9`.

Two negative controls sharpen why the consensus works:

- A finite planar-support verifier used the exact 151,226 2DGS primitive
  memberships of the 2,010 fused planes and did not use voxel/child identity.
  On seq10 it changed `19/70/85/87/88` to `22/69/84/87/88`; the strict gain
  came with medium-threshold regressions.  Thin or incomplete finite plane
  support confounds map holes with pose error and is not a safe selector.
- Full-map rendered scale/log-depth alone changed seq10 to
  `20/68/85/87/88`; rendered normal agreement alone changed it to
  `22/70/84/87/88`.  Neither is safe independently.  Requiring agreement with
  the sparse physical-plane objective removes those regressions.

The full renderer also exposed and fixed a strict lineage bug.  Uncertainty-
refined final poses bind their camera indirectly through the frozen
correspondence artifact.  The renderer now validates the complete chain
`final pose -> correspondence file/content/arrays -> camera-only inventory`
and rejects a mismatched camera; it still never opens a query pose or GT.
Rendering the complete 509,572-primitive map costs about 1.5--1.6 seconds per
query and branch on the current GPUs.  This is acceptable for validation but
not the desired final online cost.  The next implementation target is a
lossless retrieved-plane/surfel render subset that reproduces the full-map
normal score, not a new tuned selector.

A first exact-member subset was tested and rejected before label evaluation.
It kept every 2DGS primitive assigned to every Top-10 plane of every query
region: 40,185--75,877 primitives per query, mean 58,500.  Runtime fell from
132.6 to 46.0 seconds for 88 candidate renders (`2.88x`), but its normal score
changed the final V5/V11 branch on 8/88 seq10 queries.  Unretrieved planes,
unassigned fragments, and occluding surfaces still affect the z-buffer and
the all-query-pixel denominator.  This shortcut is therefore KILL and was not
connected to the selector.  A future acceleration must conservatively retain
the complete pose-frustum/occlusion carrier and prove branch-level parity;
plain retrieved-plane membership is insufficient.

That lossless acceleration is now implemented.  The complete 509,572-element
physical map stays resident on the GPU, four independently frozen cameras are
projected per batch, and the packed raster hits are composited on-device with
the exact stable `(pixel, depth, primitive-row)` ordering and float64 prefix
transmittance used by the NumPy authority.  CPU signed-surface visibility,
dominant primitive tie-breaking, full-map occlusion, and the all-valid-query
denominator are unchanged.  This is a scheduling/data-movement optimization,
not another map subset or scoring approximation.

The dedicated no-label equivalence audit checks every per-query render field,
recomputes both dense normal scores, replays the V5/V11 decision, and verifies
the previously frozen consensus artifact.  It passed both the seq10
development split and the upgrade-blind seq13 shard2:

| split / branch | legacy seconds | resident-GPU seconds | speedup | row or decision changes |
|---|---:|---:|---:|---:|
| seq10 V5 | 132.604 | 53.780 | 2.466x | 0 |
| seq10 V11 | 133.091 | 54.656 | 2.435x | 0 |
| seq13 shard2 V5 | 142.294 | 60.720 | 2.343x | 0 |
| seq13 shard2 V11 | 143.556 | 61.063 | 2.351x | 0 |

Across 175 queries and four render runs, every row is exactly equal, the
normal-score maximum delta is zero, no V5/V11 branch changes, and both selected
branch arrays exactly reproduce the pre-existing frozen outputs.  The seq10
audit file/content hashes are `7946f5fa...a194` / `249bf248...c8d`; the
upgrade-blind shard2 hashes are `f0591ef1...1740` / `6eca9d71...88d`.
Consequently the accelerated renderer is a strict implementation replacement
for this verifier, while the rejected plane-only subset remains a diagnostic
negative control.  Batch size four and the deterministic GPU compositor are
now the CLI defaults; the scalar authority remains available explicitly with
`--resident_batch_size 1 --no_resident_gpu_composite`.

The complete goal-maplet regression suite now passes **1001 tests**; focused
compilation, strong-loader replay, and `git diff --check` also pass.  The only
warning is the pre-existing PyTorch scatter-reduce beta notice.

## Exact sparse-first dense verification (V89--V101)

The dense full-map verifier is now short-circuited by a separately sealed,
pose/label-free schedule without changing the V5/V11 decision rule.  V11 can
only win when both candidates are usable and its frozen sparse plane/scale
objective is strictly lower than V5.  Consequently both dense renders are
computed only for those queries; all other rows contain no dense score and
must fall back to V5 (except the pre-existing primary-unusable case).  The
selector replays the external plan bytes and independently recomputes the
mask from the two pose inventories and sparse-plane artifact before accepting
either render report.

| split | dense renders / queries | V11 selected | speedup V5 / V11 | selected pose and hits |
|---|---:|---:|---:|---|
| seq10 | 39 / 88 | 17 | 2.034x / 2.059x | exact |
| seq13 shard0 | 47 / 88 | 21 | 1.925x / 2.012x | exact |
| seq13 shard1 | 40 / 88 | 16 | 2.133x / 2.194x | exact |
| seq13 shard2 (upgrade-blind) | 49 / 87 | 24 | 1.680x / 1.721x | exact |
| seq13 shard3 | 32 / 87 | 13 | 2.531x / 2.616x | exact |
| total | **207 / 438 (47.26%)** | **91** | **2.026x / 2.086x** | **exact** |

The runtime baseline in this table is the already accelerated resident-GPU
full render.  Summed one-arm runtime falls from 305.99/310.70 seconds to
151.06/148.96 seconds.  On every split, all retained per-query render fields
are exactly equal to the full authority, omitted rows contain no dense score,
the branch vector is identical, and `names/pose_w2c/usable/selected_branch`
plus the sparse objectives and fitted scales exactly replay the existing
frozen consensus (including NaN-valued unusable poses).  Therefore aggregate
accuracy remains exactly `104/336/395/409/413`; this is an implementation GO,
not a new fitted selector or an accuracy claim.

The machine-readable audit file/content hashes are:

- seq10: `5925602e...9d81` / `5649c07e...0341`;
- shard0: `1cc780bb...2731` / `5b287f34...44f2`;
- shard1: `9ac2e880...52d2` / `8fa6823a...bfbd`;
- shard2: `89acf1f2...66a8` / `0a2a5665...b2d2`;
- shard3: `544695cb...4b8a` / `3fe8bbe5...8aaa`.

The new scheduler and equivalence auditor are
`build_goal_maplet_sparse_first_render_plan.py` and
`audit_goal_maplet_sparse_first_render_equivalence.py`.  Their current file
hashes are `b89a2518...ab94` and `d4809936...16f1`.  Adversarial contracts
reject a changed schedule mask, missing external plan, wrong pose/plane
lineage, score-bearing omitted row, retained-row drift, or a branch change.

Three tempting alternatives were also closed rather than silently retained:

- batch size eight was slower than the frozen batch-four implementation
  (61.02 versus 59.62 seconds on the measured arm), so batch four remains the
  default;
- restricting the dense normal score to only MoGe3 planar pixels changed the
  held aggregate from `104/336/395/409/413` to
  `103/337/395/409/413`; the strict loss makes plane-only substitution KILL;
- a consensus-initialized equal-prior cross-coordinate EM changed seq10 from
  `21/71/85/87/88` to `19/71/84/87/88`; direct scalar depth/scale gates were
  also worse.  These results reinforce that future accuracy work should target
  calibrated candidate posterior or additional independent geometry, not
  another uncalibrated scalar gate.

After the sparse-first plan, renderer, selector, external-plan replay, and
equivalence-audit regressions were added, the complete goal-maplet suite passes
**1008 tests**.  `py_compile`, strong artifact reloads, file/content-hash
replays, permissions, and `git diff --check` all pass; the only warning remains
the pre-existing PyTorch `scatter_reduce` beta notice.

## Same-ray depth/normal conjunction negative control (V102--V105)

A final bounded accuracy experiment tested whether MoGe3 depth becomes useful
when it is coupled spatially to the retained dense normal evidence, rather than
used as another independent scalar gate.  On each common rendered pixel it
required both scale-normalized depth error <=20% and unsigned normal error
<=20 degrees, then used its good-ray recall under the same sparse-plane AND
rule.  The thresholds were existing report conventions.  The alternative was
checked on seq10 first and opened on the four held shards exactly once without
held retuning.

Seq10 was non-regressing at `21/71/85/87/88`, but the four held shards changed
from `83/265/310/322/325` to `82/264/311/322/325`.  Including seq10, the result
is `103/335/396/409/413`, versus the retained mainline
`104/336/395/409/413`.  The strict and 0.25m losses violate the frozen
cross-held Pareto gate, so this conjunction is KILL and its experimental code
was removed from the default renderer/selector.  The immutable reports remain
as negative-control evidence (`joint_depth_normal_consensus_v103/v105`).

This is a useful boundary on how MoGe3 depth should enter the method: it has
real spatial signal (shard2 gains one strict case and the aggregate gains one
0.5m case), but one globally fitted scale followed by a hard per-pixel depth
band is not a calibrated pose likelihood.  Future depth use should be inside a
multi-plane scale/offset residual with uncertainty, or in a separately trained
candidate posterior; it should not be reintroduced as another hand threshold.

One additional exact runtime control parallelized the four CPU signed-surface
visibility masks in each resident batch.  On the same 39-query seq10 sparse
schedule, serial and four-worker reports had exactly equal rows, but elapsed
time changed only from 23.390 to 23.002 seconds (1.017x).  This is too small to
justify another concurrency path, so the implementation was removed and the
serial exact visibility authority remains the default.

Two further bounded controls also failed to improve the frozen operating
point.  First, a fixed 2x2 spatial median-of-means replaced the global dense
normal good-ray fraction while retaining the same 20-degree test and sparse
plane conjunction.  It passed seq10, but the held shard hit counts were
`21/66/79/82/82`, `20/66/75/79/80`, `22/67/79/81/82`, and
`20/66/77/80/81`.  The held aggregate and the five-split aggregate were
exactly unchanged from the current mainline, while two individual shards
regressed.  The option was therefore removed rather than exposed as another
post-hoc selector knob.  Requiring the observed-plane-only dense score in
addition to the global score removed two seq10 switches without improving any
threshold and was not advanced to held evaluation.

Second, the final uncertainty-refined V5/V11 poses were sent back through the
existing joint reprojection + MoGe3 multi-plane normal/offset + one-scale
solver.  This tests whether the sequential solver had simply left useful plane
constraints unsatisfied.  It instead changed V5 from `19/70/85/87/88` to
`19/68/85/87/88`, and V11 from `25/70/84/87/88` to
`18/69/84/87/88`.  Thus reapplying the plane solve after the calibrated
coordinate refinement is KILL.  A future unified optimizer must preserve the
calibrated coordinate likelihood inside one objective; alternating the two
existing optimizers is not a valid shortcut.

After all four held shards had already been opened, an exploratory intersection
of the global-normal and 2x2-spatial signs produced
`105/337/395/409/413`, one strict and one 0.25 m hit above the retained
mainline.  This number is explicitly post-hoc: shard2 loses one 0.25 m case,
and the rule was inspected after seeing every held outcome.  It is therefore
not a promotion result and no selected-pose artifact is designated as a new
operating point.  The exact three-way rule may be preregistered unchanged for
a genuinely unseen route, where it must pass per-route non-regression before
it can replace the current global-normal consensus.

## Candidate-anchored posterior and staged tail audit (V112--V119)

The new review cycle first closed two methodological gaps rather than tuning
another endpoint selector.

The cross-coordinate probabilistic solver now assigns probability mass to a
physical mode keyed by `(query token, physical plane, anonymous atlas
prototype)`. V5/V11 components split that mode mass and exact duplicate rows
split, rather than duplicate, the existing probability. Consequently adding
an accidental duplicate cannot increase a token's total prior. The fixed
`radio_gibbs_unit_temperature` prior uses no fitted temperature and retains an
explicit null. This is a materially better candidate-count-invariant contract
than uniform row priors, but it is not an empirical improvement: seq10 changes
from `21/71/85/87/88` to `19/71/84/87/88`. The arm is therefore **KILL** and
was not opened on a new held route. Alternating the current local EM update,
rather than prior normalization alone, remains the dominant problem.

A new post-label stage audit freezes all correspondence and pose inventories
before opening each contributor. Across 438 queries it records `413` coarse
successes and decomposes the `25` failures into:

- 15 hard-coordinate/PnP initialization failures;
- 6 PnP-candidate collapse or downstream-refinement regressions;
- 3 retrieved chart/within-chart UV support failures;
- 1 final V5/V11 selector failure;
- 0 candidate-geometry degeneracies under the exact-projection oracle.

The 15 initialization failures are not empty-map cases. Their existing
GT-consistent rows have min/median/max counts `6/16/94`, span `2/5/21`
physical planes, and have minimum GT reprojection error
`.141/.424/1.273 px`. Cambridge has no depth ground truth, so the three support
failures cannot honestly be split into "chart absent" versus "correct UV
absent"; the report preserves that limitation explicitly.

The aggregate audit is `pose_failure_attribution_all438_v115.json`, file
SHA256 `ddf636306314445748b2765f1a71e56b486c4c64f01f68e022f5f07dab910245`,
content SHA256
`32286fec4978668f219569e68dba489e6186bcc645003939e8ec5fbff262e196`.

The first bounded tail response uses a pose-free relative multi-plane layout
test. For two query-region/map-plane associations it compares the unsigned
angle between the query normals with the unsigned angle between the map
normals. This quantity is invariant to unknown camera rotation and normal
sign. Candidate generation is capped at 64 pairs and ranked lexicographically
by angle error, existing RADIO evidence, support, and IDs; there is no
continuous fusion weight. All candidate poses are hashed before labels open.

On the 350 held queries the retained mainline has `325/350` coarse hits. The
layout candidate pool alone has `322/350`, but its union with the existing pose
contains `332/350`: it recovers 7 of the 25 existing held failures. However,
the label-free supported-entity selection chooses only `310/350`, recovering
just 3 failures while losing other cases. A sparse MoGe3 plane-geometry
selection smoke on shard0 is coarse-neutral and heavily harms tighter
thresholds. The correct conclusion is **candidate-generation headroom, no
deployable selection gain**. The layout pool remains a bounded tail hypothesis
source; it does not replace the mainline and does not justify more Top-K or
pair enumeration.

The aggregate is `multiplane_layout_all350_aggregate_v119.json`, file SHA256
`d90c1a962a057a8ed1a70c5490f339154654df7b7deecd6931246420d62d6e80`,
content SHA256
`02da92371aff6ade98938c0b30e690bd3ef79406b6d39d26d6d291a825df8573`.
An intermediate V117 report had a local variable-shadowing bug in evaluation
bookkeeping (the frozen candidate NPZ was unaffected); it is superseded by
V118 and is not evidence.

The updated optimization target is precise. Coarse-tail work needs a
query-conditioned, repeated-facade-aware multi-plane association posterior,
not a larger map or an angle-only heuristic. Tight-threshold work still needs
the candidate-conditioned local RADIO/chart-UV correlation head, followed by
one joint null-aware surface solver rather than alternating hard PnP and local
EM. Both must be frozen on mapping-only data before the next unseen route is
opened.

### P3 local RADIO correlation: mapping improvement, no mainline promotion (2026-09-07)

Implemented a candidate-conditioned 3x3 query / 3x3 physical-chart RADIO64
correlation head. Its 107 context inputs comprise the existing eight geometric
values, 81 masked correlations, and 18 validity flags. The hidden width remains
96. Outputs are bounded query subtoken and metric chart-UV residuals, positive
measurement variances, and a match sigmoid. Runtime uses anonymous atlas
descriptors, not source RGB or source-view retrieval. Fit neighbors exclude the
query plane-observation; validation neighbors use fit routes only.

The mapping-only V121 checkpoint improves held plane-observation image median
error from 0.5267 to 0.4787 px (9.1%) and UV median error from 0.09112 to
0.08767 m (3.8%); image/UV P90 and NLL also improve. All nine reference gates
pass. These gates were introduced after the first local-correlation training
run, so this is exploratory evidence, not a preregistered acceptance result.
Calibration and evaluation alternate plane observations on seq9, not whole
images; they are not image-disjoint validation sets.

Seq10 results below use identical 79,536 correspondence candidates on 88
development queries. Counts are joint translation/rotation successes.

| Fixed selector | Coordinates | 0.1m/1deg | 0.25m/2deg | 0.5m/5deg | 1m/10deg | 2m/45deg | Median t (m) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| calibrated_gaussian_null | V11 | 25 | 70 | 84 | 87 | 88 | 0.1559 |
| calibrated_gaussian_null | V12 | 22 | 71 | 84 | 87 | 88 | 0.1409 |
| calibrated_gaussian_null | Pareto fusion | 24 | 73 | 84 | 87 | 88 | 0.1459 |
| nearest_reprojection | V11 | 24 | 69 | 84 | 87 | 88 | 0.1583 |
| nearest_reprojection | V12 | 21 | 72 | 84 | 86 | 88 | 0.1412 |
| nearest_reprojection | Pareto fusion | 21 | 72 | 84 | 87 | 88 | 0.1476 |

The original V122 paired report compared different selectors and must not be
used as a head-only ablation. Fair reports are
`learned64d_strict_seq10_h025_localcorr_v122_vs_v11cell_v123_paired.json`,
`learned64d_strict_seq10_h025_localcorr_pareto_fused_v124_vs_v11_paired.json`,
and the two `*_nearest_v125_paired.json` reports in the surface-coordinate
upgrade output directory. Nearest-policy mean translation change confidence
intervals include zero for both variants. Frame bootstrap also ignores temporal
dependence. These development results do not establish blind-test superiority.

Pareto fusion substitutes 32,923/79,536 coordinates (41.39%) only when predicted
image and UV variance do not increase and the match sigmoid does not decrease.
It has no query-GT gate, but cross-head sigmoid comparison is **not calibrated
probability comparison**. It is a diagnostic, not a theoretically guaranteed
non-regression rule. V126 corrects this documentation/lineage defect: records a
per-row `coordinate_source_head_index`, nests both heads' calibration metadata,
and removes misleading top-level single-head metadata. V124 is preserved as
the historical numerical experiment; no claim that V126 itself was pose-replayed.

Audit ruled out a suspected query-neighbor pixel-support discrepancy: both
the observation bank and runtime require eight plane pixels per token. A real
remaining distribution difference is training neighboring cell means from all
fit representatives versus runtime means over at most four anonymous atlas
modes. This needs a mapping-only deployment-equivalent ablation before any
more query-side selector experiments. Neither variant passes strict-threshold
non-regression; retain the existing mainline and do not expand to the remaining
350 reused development queries merely to select the best variant. Verification:
all 1,027 goal-maplet tests pass; `git diff --check` passes. V126's 19 original
numerical arrays exactly match V124; only provenance/metadata was repaired.
Next priority
is aligned chart-neighbor construction and image/route-disjoint uncertainty
validation, followed by a frozen unseen-route evaluation.

### P3 neighbor-mode aggregation alignment ablation (2026-09-07)

Added opt-in `--local_neighbor_policy atlas_modes_mean`. The default preserves
the previous all-representative mean. The ablation changes only neighboring
map-cell features: exclude the query observation first, select up to four modes
using the atlas medoid-then-farthest rule, serialize at float16 precision, and
normalize their mean. Candidate-center features, training targets, seed 260918,
1200 steps, hidden width 96, and the V11 mapping gate stay fixed. The artifact
records the policy and mode budget; runtime rejects a mismatched atlas budget.
Tests compare the training aggregation directly to runtime aggregation and
check observation exclusion under both policies. All 1,029 goal-maplet tests pass.

Scope caveat: this aligns mode selection/aggregation, not the full descriptor
pipeline. Training still uses the token nearest the cell center per
plane-observation/cell; atlas construction averages multiple tokens per
view/cell before selecting modes. That upstream difference is intentionally
unchanged in this one-factor experiment. V127 is an intermediate smoke;
V128 reruns with the exact runtime float64 accumulation / float32 normalization
and invalid-norm guard. Neither uses query labels in fitting.

V128 mapping validation image median/P90 = 0.478840/0.810099 px versus
V121 0.478654/0.809736; UV median/P90 = 0.088133/0.179419 m versus
0.087669/0.178657. Both pass the frozen V11-relative gate. Mode alignment
does not improve the mapping coordinate metric by itself.

The fixed nearest-reprojection seq10 replay V129 gives:

| Coordinates | 0.1m/1deg | 0.25m/2deg | 0.5m/5deg | 1m/10deg | 2m/45deg | Median t (m) | Median r (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V11 reference | 24 | 69 | 84 | 87 | 88 | 0.15827 | 0.39003 |
| V121 unaligned local head | 21 | 72 | 84 | 86 | 88 | 0.14115 | 0.36127 |
| V128 mode-aligned local head | 22 | 72 | 84 | 87 | 88 | 0.13934 | 0.35986 |

All 13 candidate/common arrays are exactly equal across old/new local heads
(79,536 rows). Against the unaligned head, strict-threshold and 1m successes
each recover one query with no losses at those thresholds. At 0.25m there are
two gains and two losses, not per-query non-regression. The translation mean
delta is -0.007676 m with frame-bootstrap 95% interval [-0.021846, +0.001001];
there is no statistically established improvement, and these are reused
development queries. Against V11, strict success remains two queries lower.
Decision: promising small alignment benefit, **no mainline promotion** and no
selector/fusion sweep. Next bounded ablation should replace representative-token
neighbor inputs with view/cell token means while preserving targets and splits;
whole-image/route-disjoint calibration remains separately required.

Head: `stmarys_mapping_surface_coordinate_local_modes_head_v128.npz`, content
SHA256 `7d1cd7e0c13a352036afcd1d12a4d8aa5b7d7aef59ff018b504846dc2db01e19`.
Pose: `learned64d_strict_seq10_h025_local_modes_v129_final.npz`, file SHA256
`e64fa99f6829ad1d7099c965bd463162875c4099e63650936abce46c1f4cf8af`.
Both paired reports use the prefix `learned64d_strict_seq10_h025_local_modes_v129_vs_`
in the existing surface-coordinate upgrade output directory.

### P3 view/cell token-mean alignment and source-view exclusion (2026-09-07)

Added opt-in `--local_neighbor_token_pooling view_cell_mean`, requiring the
atlas-mode neighbor policy. All bank tokens are projected and normalized before
averaging per physical cell and source view; the mean is normalized, then at
most four anonymous modes are selected and aggregated. Only neighbor context
changes: center descriptors, coordinate targets, query context, and pose backend
remain unchanged. This is offline mapping processing; no source-view IDs or
source images are introduced into runtime retrieval.

The audit found **642 excess observations in 553 repeated plane/source-view
groups across 111 planes**, among 13,043 observations. Plane observation IDs
therefore cannot substitute for source-view IDs. V130 is a superseded
observation-mean smoke and was not query-evaluated. V131 merges tokens by the
actual source-view name, deduplicates cell/view representatives, and excludes
the query's entire source view from neighbor mode construction before selection.
Validation neighbor modes continue to use fit routes only. New tests cover
multi-token means, cross-view isolation, and exclusion of all same-source rows.

Important remaining boundary: the unchanged candidate-center training dataset
still uses plane-observation exclusion; this experiment does not certify
whole-pipeline source-view isolation or whole-image-disjoint calibration. That
contract needs a separate audit/fix rather than calling this complete alignment.
Specifically, 1,584 of 225,296 fit representatives belong to 792 cell/source-view
groups containing multiple observations, before coordinate-target filtering.
This is potential same-source fit contamination, not query-GT or held-route
leakage. The atlas also requires two independent views per cell, whereas the
training neighbor code still accepts one remaining view after exclusion. This
support-validity discrepancy is unchanged and must not be hidden by the term
"aligned"; a follow-up should freeze its policy before evaluating more queries.

V131 mapping validation image median/P90 is 0.479406/0.809425 px; UV
median/P90 is 0.087486/0.179680 m. V128 was 0.478840/0.810099 px and
0.088133/0.179419 m. Thus token pooling slightly improves UV median, not every
coordinate metric. All nine V11-relative gates pass. Head content SHA256:
`53538f7e3f3ffb4451633d78b76300bf743000a2248a16bbb389f05b4db32885`.
All 1,031 goal-maplet tests pass; `git diff --check` is clean.

V132 fixed-nearest-reprojection replay uses exactly the same 79,536 candidates
as V129 (all 13 common arrays match bitwise). On the same 88 development queries:

| Coordinates | 0.1m/1deg | 0.25m/2deg | 0.5m/5deg | 1m/10deg | 2m/45deg | Median t (m) | Median r (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| V128 representative/mode mean | 22 | 72 | 84 | 87 | 88 | 0.13934 | 0.35986 |
| V131 source-view token mean | 20 | 69 | 84 | 87 | 88 | 0.14726 | 0.37461 |

Strict threshold loses two queries without gains; 0.25m has one gain/four
losses. Mean translation delta is +0.002121 m, frame-bootstrap interval
[-0.001377, +0.005493]. Thus there is no gain warranting promotion; do not
salvage it by searching selectors or expanding the reused development set.
The one-pass treatment combines token pooling and necessary true-source
deduplication/exclusion: this does not isolate which component causes the
regression, nor prove that view/cell pooling is inherently unsuitable. It does
show that improved mapping UV median alone is not a pose acceptance criterion.

Keep the current production mainline and the prior diagnostic checkpoints.
Next valuable work is a source-view-isolated center/neighbor training contract
with deployment-equivalent support validity, then image-disjoint calibration;
avoid interpreting another small seq10 fluctuation as a mainline-level result.
The new pooling policy remains opt-in; correcting fit isolation should not be
abandoned merely because this exploratory joint treatment regressed.

Pose file `learned64d_strict_seq10_h025_local_viewmean_v132_final.npz`, SHA256
`c8ac1311cd4d8b280d3b069e0bfe6a51ccdbd84b904af52fbf7a0412c8d540cd`.
Paired reports use prefix `learned64d_strict_seq10_h025_local_viewmean_v132_vs_`
in the existing surface-coordinate upgrade output directory. V130 was never
query-evaluated. Training used the existing seed/steps with one BLAS/OMP CPU
thread; inference candidate identity and scores were verified bitwise unchanged.

### P3 source-view training contract (2026-09-07)

The opt-in `--source_view_training_contract` unifies three previously mismatched
rules. Canonical center targets exclude every observation from the query source
view, average remaining observations within each source, then give sources equal
weight. At least two independent remaining sources are required. Local chart
neighbors also require two independent sources, matching the atlas admission
rule. Shrinkage/variance calibration and evaluation alternate sorted **source
images**, not plane observations; no image occurs in both subsets. These are
still adjacent images within seq9, not independent unseen-route validation.

Because targets and evaluation membership change, old V11/V12 mapping metrics
are not comparable to this protocol. V133 is a newly trained V11 reference;
the local head must use that reference with the same new source-view contract.
The artifact records its contract, support minimum, and partition semantics;
runtime checks the independent-view minimum against the atlas. Default legacy
behavior remains available for reproduction. This fixes source isolation, not
the still-distinct canonical-center versus anonymous-mode descriptor design.

Both V133 and V134 use 68,936 fit pairs and 17,034 held mapping pairs, of which
8,591 are calibration and 8,443 evaluation. With identical new targets and
partitions, V134 reduces image median/P90 from 0.531019/0.870341 to
0.485813/0.819919 px, and UV median/P90 from 0.092048/0.187220 to
0.088256/0.180786 m. Both NLLs improve; all nine V11-relative gates pass.
This supports a local-correlation coordinate benefit without same-image
calibration overlap, not yet an unseen-route localization claim.

V133 head content SHA256
`641ef6f2236d9e9dda2cfdff5344ee6ec67468b8914b5eec67c0bc525225987f`;
V134 `a9164c758f1c0666c59c4b4dd69ea6eb919e1489a5b2ef17c76289f900734005`.
All 1,033 goal-maplet tests pass after the changes. Tests verify source-balanced
center targets, complete query-source exclusion, and the two-source neighbor
validity rule. No historical head or mainline pose was overwritten.

Fixed-nearest-reprojection localization on the same 88 development queries:

| Training/head | 0.1m/1deg | 0.25m/2deg | 0.5m/5deg | 1m/10deg | 2m/45deg | Median t (m) | Median r (deg) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Legacy V11 | 24 | 69 | 84 | 87 | 88 | 0.15827 | 0.39003 |
| Source-isolated V11 (V135) | 22 | 73 | 84 | 87 | 88 | 0.15629 | 0.39604 |
| Source-isolated local head (V136) | 23 | 71 | 84 | 87 | 88 | 0.14685 | 0.39045 |

All 13 common correspondence arrays match exactly between V135 and V136
(79,536 candidates). V136 versus V135 has five strict gains/four losses and
zero 0.25m gains/two losses. Translation mean delta is -0.008415 m with ordinary
frame-bootstrap interval [-0.018088, -0.000154]; rotation delta is -0.022350 deg
with interval [-0.047678, -0.001497]. These narrow intervals do **not** establish
generalization: frames are temporally correlated and seq10 is repeatedly reused
development data. Threshold non-regression still fails. Do not promote either
new head merely from these results or search another fusion threshold.

The source-isolation/support/calibration corrections remain valuable independent
of this mixed pose result. Coordinate improvements now survive a stricter
mapping evaluation, but are not uniformly translated into localization success.
The canonical-center versus runtime mode target mismatch and downstream
hypothesis sensitivity remain unresolved; next work should diagnose those
under this frozen corrected training protocol, not revert source isolation.

V135 final pose SHA256
`c3b1d3502c7ca315a228a81717175585281f3cf1b181ef6c8516b4a77e11453b`;
V136 `668f2a33c8b4b68d5de18460a4127c53b987364a658127ed6632585b558645ae`.
Report `learned64d_strict_seq10_h025_sourceisolated_local_v136_vs_v135_paired.json`
and legacy-reference report `learned64d_strict_seq10_h025_sourceisolated_v11_v135_vs_legacy_paired.json`
are in the existing surface-coordinate upgrade output directory. Mainline
artifacts remain unchanged; these are diagnostic candidates only.

## Latest external-review completion audit and temporal inference (V137)

The latest attachment is `d5342896-3e23-48f2-a1ee-f5ac7da4395d/pasted-text.txt`.
Its recommendations are **not all complete or validated**:

| Recommendation | Verified status |
| --- | --- |
| P0 staged tail attribution | Completed aggregate of 438; 25 failures classified, but chart-missing versus UV-missing cannot be separated reliably without depth GT. |
| P1 physical candidate priors and null | Candidate-anchored, duplicate-count-invariant prototype exists; match sigmoid is not independently probability-calibrated. |
| P2 probability surface backend | Prototype tested and rejected for regression; no successful unified final solver. |
| P3 local correlation/residual head | Implemented, source-isolated mapping gains verified; mixed query thresholds prevent promotion. Boundary-neighbor posterior and runtime-mode center targets remain open. |
| P4 unified sparse/dense/depth/scale factors | Not complete; current sequential refinements/consensus are not one calibrated joint likelihood. |
| Tail layout response | Extra oracle-recoverable poses found, but label-free selection regresses; no deployable tail gain. |
| Final unseen-route/scene evaluation | Not done for these variants; 438 queries remain development evidence. |
| Commit-level reproducibility and single entrypoint | Partial artifact hashes and reports exist; current local HEAD is d7dbbbb, with substantial uncommitted/untracked research code. No complete new release/tag or verified end-to-end single entrypoint. Remote HEAD was not queried in this audit. |

V137 adds route-stratified circular moving-block bootstrap, ordered by numeric
frame index, with fixed exploratory sensitivity lengths 5/10/20 observed frames and
10,000 replicates each. Blocks preserve the paired method delta and never mix
routes. This is conditional inference on observed routes, not route-level
generalization; circular endpoint wrapping and block length are assumptions.
Missing paired errors are kept in sequence positions and excluded only from
each sampled mean. The threshold comparison also fixes a separate usability
bug: count each method's successes independently, rather than discarding both
when either is unusable. Both V135/V136 have 88 usable poses, so this repair
does not change their historical counts.

V136 minus V135 mean translation is -0.008415 m. Block 95% intervals:
5 frames [-0.018757, -0.000518], 10 [-0.017744, -0.000848],
20 [-0.017163, -0.001859]. The small translation benefit survives these
within-sequence sensitivity checks. Rotation's 10-frame interval crosses zero
[-0.048611, +0.000589] deg; do not claim robust rotation improvement. These
post-development checks cannot undo repeated seq10 model selection, and the
0.25m threshold regression still prevents promotion.

Report: `learned64d_strict_seq10_h025_sourceisolated_local_v137_block_audit.json`,
content SHA256 `adf028df0e554ee9e2a2f9f22e8363ce9a347b52512729d684cba5e28dd52bbe`.
Next algorithmic work should isolate mean-coordinate versus covariance versus
initialization effects using frozen candidates/initial poses, then align center
targets to physical anonymous modes and calibrate null before another joint
solver attempt. Do not treat more selector sweeps or larger universal Top-K as
the primary optimization direction.

## Fixed-initialization mean/covariance attribution (V138, 2026-09-08)

New diagnostic `audit_goal_maplet_coordinate_solver_factors.py` freezes all
eight poses before opening contributor GT. Factors are the V135/V136 upstream
MoGe-refined initialization, V11/local coordinate means, and V11/local predicted
coordinate covariance. Within each initial pose, selection always uses V11
nearest reprojection, and covariance propagation always uses the V11 world-point
Jacobian at that initial pose. Thus changing means cannot silently change row
identity or its covariance Jacobian. The physical map footprint, purity, and
dispersion are unchanged. Matchability does not participate in this fixed-row
nearest-reprojection experiment.

This runs the existing bounded local solver, not the production wrapper's
global-support acceptance check. It is a conditional diagnostic, not a claimed
new deployed pipeline. Row identities are equal across all four arms per
initialization. At initialization 0 the V11/V11 control reproduces production
V135's reported metrics. All 1,037 goal-maplet tests pass.

| Initial pose | Mean | Covariance | Median t (m) | 0.1m/1deg | 0.25m/2deg | 0.5m/5deg |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| V135 | V11 | V11 | 0.15629 | 22 | 73 | 84 |
| V135 | V11 | local | 0.14909 | 23 | 70 | 84 |
| V135 | local | V11 | 0.14980 | 19 | 72 | 84 |
| V135 | local | local | 0.14320 | 22 | 72 | 84 |
| V136 | V11 | V11 | 0.15933 | 24 | 73 | 84 |
| V136 | V11 | local | 0.14972 | 25 | 70 | 84 |
| V136 | local | V11 | 0.15789 | 22 | 72 | 83 |
| V136 | local | local | 0.14911 | 21 | 72 | 83 |

Every arm has 87/88 at 1m/10deg and 88/88 at 2m/45deg. This excludes a simple
"only covariance is broken" or "new means are useless" explanation. Both
factors improve some aggregate errors and worsen some thresholds; initialization
interacts with their effects. Choosing the nicest cell after opening labels
would be another development-set selector, not independent validation.

Post-label geometry audit on the fixed rows provides stronger coordinate evidence:
at initial 0, 29,079 rows have GT-pose reprojection median/P90 1.170529/2.691141 px
with V11 means, versus 1.118181/2.626327 with local means; 56.87% improve. At
initial 1, 29,081 rows give 1.170117/2.691923 versus 1.118445/2.627327; 56.69%
improve. This measures consistency with camera-pose GT, not outdoor depth GT.
If nearest-reprojection association is recomputed using local means at the same
initial pose, initial 0 loses 1,503 original rows and adds 1,609; initial 1 loses
1,539 and adds 1,658. Roughly 5% row removal shows a nontrivial discrete
association change, but does not by itself prove that the changed rows cause
every localization regression.

The next solver direction should address physical multi-hypothesis association
and calibrated correlated/anisotropic coordinate uncertainty, with fixed-pose
and fixed-row controls retained. Do not promote a post-label-selected hybrid
or discard the source-isolation fixes. Mainline remains unchanged.

Report `fixed_row_factor_audit_v138/report.json`, content SHA256
`d4e76361a17b01a71c86677a5a2b66716133f68e604e2f7c22b5f913017d83ae`, under the
surface-coordinate upgrade output directory; all eight row/pose inventories are
saved next to it with source hashes and factor IDs.

## Exact conditional marginal objective experiment (V139, 2026-09-08)

The existing EM prototype computes Gaussian-mixture responsibilities but its
M-step optimizes plane-balanced Huber residuals, and its final acceptance scores
the original mixture. Changing covariance with pose without differentiating its
normalizer is another obstacle to an exact EM interpretation. These are
surrogate-update limitations, not evidence that every previous pose is wrong.

V139 directly minimizes one weighted token mixture negative log likelihood with
an analytic radial-projection gradient. Covariance is frozen at initialization,
so its determinant and projection do not silently change inside the objective.
All candidate responsibilities are implicitly recomputed at every function
evaluation. There is no extra Huber loss in this new update. Token weights and
active tokens come from the same V11 nearest-reprojection anchor; all existing
physical candidates for those tokens remain available. Physical-mode normalized
RADIO priors split duplicate mass rather than creating extra evidence.

Four arms were fixed before query evaluation: V133/V134 coordinate inventories,
each with isotropic trace covariance or full 2x2 projected centroid covariance.
All start from V135's MoGe-refined pose. Coordinate means, candidate sets,
network parameters, and match priors are not tuned. The raw support and pose-step
limits are inherited; effective match mass must also retain 95% of its initial
value to limit null-only improvements. Match sigmoids remain uncalibrated
diagnostic priors; this is not a claimed fully calibrated generative model.
MoGe/scale is upstream, not a joint factor in this conditional experiment.

The solver and acceptance evaluate the same scalar objective. Tests check its
finite-difference gradient with radial distortion, duplicate-mass invariance,
all-null constant density/zero gradient, and single-candidate reduction to
weighted Gaussian reprojection (not exact reproduction of legacy Huber/PnP).
The isotropic trace of full covariance reproduces the previous propagated sigma.
The four pose inventories are all sealed before their query labels are opened.
Acceptance for further evaluation requires tight-threshold gains without coarse
regression against a same-head/same-initialization hard solve; objective descent
alone is not a pose-accuracy criterion. No arm is automatically promoted.

### V139--V143 outcomes: objective correctness is not sufficient

The bounded follow-up comprises 11 new localization arms (four exact marginal,
three hard-covariance controls, four pose-free-token marginal arms), plus two
post-label GT-objective audits. The existing V135 hard/isotropic output supplies
the twelfth table row/control. All runs use the same V135 MoGe-refined initial
poses and unchanged source-isolated V133/V134 correspondence inventories.

| Head | Solver / token support / covariance | Median t (m) | 0.1m/1deg | 0.25m/2deg | 0.5m/5deg |
| --- | --- | ---: | ---: | ---: | ---: |
| V11 | Hard / nearest / isotropic (control) | 0.15629 | 22 | 73 | 84 |
| V11 | Hard / nearest / full | 0.15826 | 19 | 72 | 84 |
| Local | Hard / nearest / isotropic | 0.14157 | 24 | 71 | 84 |
| Local | Hard / nearest / full | 0.14739 | 22 | 73 | 84 |
| V11 | Marginal / initial-nearest / isotropic | 0.15713 | 19 | 68 | 84 |
| V11 | Marginal / initial-nearest / full | 0.16822 | 21 | 70 | 84 |
| Local | Marginal / initial-nearest / isotropic | 0.14555 | 22 | 68 | 84 |
| Local | Marginal / initial-nearest / full | 0.15300 | 21 | 68 | 84 |
| V11 | Marginal / pose-free RADIO / isotropic | 0.15901 | 23 | 68 | 84 |
| V11 | Marginal / pose-free RADIO / full | 0.16857 | 21 | 66 | 83 |
| Local | Marginal / pose-free RADIO / isotropic | 0.15177 | 20 | 67 | 84 |
| Local | Marginal / pose-free RADIO / full | 0.15487 | 18 | 66 | 83 |

Every row is 87/88 at 1m/10deg and 88/88 at 2m/45deg. The full-covariance hard
control changes only whitening, retaining row selection and global acceptance.
Its symmetric inverse-square-root whitening reproduces the old solver for
isotropic matrices (tested), but full covariance does not uniformly improve
accuracy: for the local head it trades two strict successes for two 0.25m
successes. This is not a promotion result.

All logged exact-objective optimizer trajectories are nonincreasing within
1e-9. V139 has one failed line search, correctly rejected; the other 351 solves
terminate successfully. Nevertheless, 37--41 accepted V139 updates increase
translation error relative to initialization, compared with 29 for the same
local-head hard/isotropic control and 31 for the V11 control. Exact optimization
of this model has not solved objective-improving/error-worsening behavior.

V141 scores camera-pose GT under the **same frozen objective**, reconstructing
and asserting each initial loss against the saved solve report. GT is preferred
over initialization on only 4/5/6/9 queries (V11 isotropic/full, local
isotropic/full); the fitted output is preferred over GT on 88/88/87/88. Fitted
likelihood exceeding GT is normal with noisy measurements and is not alone a
proof of a bug or miscalibration. Here it establishes that stronger optimization
of this conditional score cannot be assumed to move toward GT.

The initial token screen itself conditions on pose residual. V142 removes that
screen without adding map candidates or changing Top-K: all existing RADIO
tokens are retained, and their balance-plane anchor is selected by RADIO score
with deterministic physical-ID ties. This raises the active token count from
29,079 to 49,162. It does not restore precision. V143 repeats the GT-objective
audit: GT beats initialization on 10/11/7/10 queries, and output still beats GT
on 88/87/88/87. Thus pose-conditioned token screening is not the sole explanation.
V142 has no optimizer termination failures; its rejected outputs fail the
unchanged safety gates. None of these arms is opened on an unseen route or
promoted; all remain exploratory development-set results.

### Concrete next modeling gap

Image and chart-UV coordinates come from the same head and shared geometric
supervision. Current residual covariance adds their marginals as if independent:
`Sigma_u + J Sigma_X J^T`. The general expression also contains
`-Sigma_uX J^T - J Sigma_Xu`. Neither the cross-covariance nor an independently
calibrated match/null prior is provided by the current head. The experiments do
not establish these as the unique cause of regression, but they make
mapping-only joint reprojection-residual calibration a more justified next
target than another optimizer, arbitrary likelihood weight, or selector sweep.
Calibration data should reproduce anonymous-mode runtime inputs and preserve
the corrected source-view isolation. Do not fit it on the repeatedly reused
seq10 labels or call a best table row a new generalization result.

Implementation delivered: exact marginal solver, full-covariance hard-refinement
option (default stays isotropic), pose-free token support option, and a frozen
objective/GT audit. 1,044 goal-maplet tests pass; `git diff --check` passes.
No mainline artifact, atlas, or network checkpoint was overwritten.

Artifacts under `surface_coordinate_upgrade_v1`:

- `exact_marginal_v139/report.json`, content SHA256
  `213e6273dbd5702ade18d8ee2183de4958eb33e0a04eee587e1d165e9b74232a`.
- `hard_covariance_v140_h{0,1}_{isotropic,full_centroid}.{npz,json}`;
  h0 isotropic uses the existing V135 control, not a new file.
- `exact_marginal_v141_gt_objective_audit.json`.
- `exact_marginal_posefree_v142/report.json`, content SHA256
  `cd3990d097557f5d979bb806402ab23e926dca867b4f368bd26db8d059074215`.
- `exact_marginal_v143_posefree_gt_objective_audit.json`.

## V144–V151: frozen joint-residual calibration and geometric error attribution

The V133/V134 source-isolated heads were replayed without any optimizer steps.
Checkpoint weight hashes and all four original coordinate/variance calibration
scalars are asserted identical. Replay also validates mapping bank, plane map,
visibility, projection, contributor inventory, source contract, seed, route,
and neighborhood policies. Existing replay outputs and the original checkpoint
cannot be overwritten. V144/V145 export 17,034 mapping pairs: 8,591 calibration
pairs and 8,443 evaluation pairs from disjoint complete source images. These
remain **canonical-center mapping inputs**, not anonymous runtime-mode inputs;
this important distribution-alignment item is not completed by this experiment.

### Calibration results, not a physical cross-correlation estimate

The image and UV targets were checked to derive from the same mapping world
point. The exported predicted UV is clipped using the actual runtime half-open
cell rule before radial-camera projection. The covariance uses the full radial
Jacobian. Five fixed families are fitted only on calibration images: unchanged
independent marginals; overall scale; overall scale plus scalar correlation;
separate image/UV scales; separate scales plus correlation. For projected UV
covariance P and image variance q, the correlation family is
`a*q*I + b*P - 2*rho*sqrt(a*q)*sqrt(b*P)`, with bounded rho and positive scales.
This is SPD by construction; rho=0, a=b=1 recovers the original full covariance.

| Held-image result | V133 baseline head | V134 local head |
|---|---:|---:|
| Independent NLL | 2.458981 | 2.301156 |
| Overall-scale NLL | 2.439721 | 2.266425 |
| Separate-scale NLL | 2.378315 | 2.228704 |
| Separate-scale 90% coverage | 88.06% | 88.90% |
| Joint separate-scale NLL | 2.376039 | 2.227793 |
| Empirical image/world error correlation x/y | .196/.308 | .280/.326 |

Separate scale factors (image, UV) are (2.8961, .13912) and (2.2159, .29581).
These optimize the **joint residual distribution** and are not standalone
evidence that one marginal predictor's variance is wrong by those factors.
Both learned correlation families saturate rho=-.99, despite positive empirical
error correlation. The constrained conditional covariance family is therefore
not an interpretable empirical cross-covariance estimator. The full joint model
fails the interior-solution gate and is NOT deployed. Its small NLL advantage
over separate scales does not justify adding the correlation parameter.

A numerical issue in the newly written calibrator was caught during development:
float32 variance scaling erased scipy finite-difference perturbations and could
leave the image scale at one. Calibration now converts inputs to float64 before
parameter multiplication; the synthetic calibration regression uses float32
input and verifies recovery and evaluation-label independence. This issue was
in the new experiment, not evidence of a pre-existing mainline optimizer bug.

### V148: mapping-frozen calibration does not establish a pose upgrade

Four query arms use the exact same V135 MoGe-refined initialization, unchanged
candidates and nearest-token associations, full centroid covariance, and existing
robust/support/step gates. Only overall or separate variance scales change. All
calibration reports are frozen before solving and head-hash checked. Query GT
is used only by evaluation, not fitting. Counts are on the reused 88-query seq10
development set; the mainline 438-query result remains unchanged.

| Head / covariance | Median translation m | .1m/1deg | .25m/2deg | .5m/5deg |
|---|---:|---:|---:|---:|
| V133 full, original V140 | .15826 | 19 | 72 | 84 |
| V133 full, overall scale | .15801 | 18 | 72 | 84 |
| V133 full, separate scales | .15389 | 20 | 72 | 84 |
| V134 full, original V140 | .14739 | 22 | 73 | 84 |
| V134 full, overall scale | .14441 | 22 | 72 | 84 |
| V134 full, separate scales | .14603 | 21 | 71 | 84 |

Every arm retains 87/88 at 1m/10deg and 88/88 at 2m/45deg. The prior V134
hard/isotropic control remains .14157m and 24/71/84/87/88; the new arms do not
dominate it. For V134 overall scale vs original full covariance, mean paired
translation change is -1.083mm, but the 10-frame block bootstrap 95% interval is
[-2.271,+.103]mm and .25m success loses one query. All four arms' 5/10/20-frame
translation intervals cross zero. No new pose artifact is promoted, and no
selector or thresholds were tuned to recover the lost query.

### V149/V150 geometric decomposition

Additional frozen replays separate image error, chart-tangent UV error, and the
normal-height mismatch between the mapping query point and the prototype's
local tangent sheet. Exact squared-error decomposition includes all cross terms;
component energies must not be interpreted as additive positive percentages.
Coordinate oracles are mapping-only diagnostics, never runtime measurements.

For V133, normal-height mismatch median/P90/P99 is .0202/.0557/.0969m.
Predicted reprojection median/P90 is .915/1.764px. Perfect image coordinates alone
give .915/1.717px; perfect UV alone gives .553/.964px; perfect normal height alone
gives .904/1.747px. Perfect image AND UV leaves .143/.437px. Thus tangent-coordinate
accuracy has more evident remaining leverage than scalar uncertainty tuning or
height-only correction on these held mapping pairs. The limited image-only
oracle gain reflects correlated errors; it is not proof image accuracy is useless.

V151 confirms the same ordering for V134: predicted .857/1.630px, ideal UV
.515/.913px, ideal normal height .841/1.623px, and ideal image alone
.883/1.654px (the loss of favorable error cancellation can hurt). With both
image and UV ideal, the same .143/.437px normal residual remains. Across 49 held
source images, local-minus-baseline mean reprojection error with equal image
weight is -.06805px; source-image bootstrap 95% CI is [-.07702,-.05880]px.
This independently supports better **mapping joint coordinates** from the
existing local head, but is not new query-pose improvement or an unseen-route
generalization claim. The bootstrap does not remove neighboring-image temporal
correlation. Full decomposition and input hashes are in
`mapping_joint_geometry_audit_v151.json`.

Delivered tools: frozen replay/export, joint covariance calibrator, optional
mapping-calibrated full-covariance refinement, and exact geometric decomposition.
Mainline source-image-free retrieval and MoGe factors are unchanged. 1,058
goal-maplet tests pass (one pre-existing scatter_reduce warning).

Artifacts under `surface_coordinate_upgrade_v1`: `mapping_joint_residual_v144/145.npz`,
`mapping_joint_calibration_v146/147.json`, four `mapping_calibrated_v148_h*` pose,
evaluation and paired-audit files, plus expanded `mapping_joint_residual_v149/150.npz`.
Priority remains source-isolated **anonymous-mode-aligned continuous UV measurement**
and joint-coordinate learning/evaluation, not more global solver-weight sweeps.

## V152–V164: anonymous center alignment, negative-label repair, and controlled validation

Implemented the previously missing `source_view_modes` center policy. It projects
all mapping tokens, averages by physical-cell/source-image, normalizes each view
mean, selects deterministic medoid/farthest modes, and retains each selected
mode's own mean 3D geometry. Training removes the entire query source image BEFORE
mode selection and requires two remaining independent images. Validation modes
use fit routes only. The query representative cap limits query examples, not the
mapping source pool. Unit tests compare the actual atlas fusion routine's mode
features and geometry, test complete source exclusion and minimum-view support.
Runtime rejects incompatible cell size, mode budget, view support or incomplete
center contracts. Source identities remain mapping-only and are not exported to
the runtime map or prediction head.

### A real label issue exposed by multi-mode expansion

The old within-plane rolled negative policy becomes particularly invalid after
expanding several modes for each query. On the new frozen dataset it selects a
mode from the SAME physical cell for **90.81% of fit negatives and 86.26% of
validation negatives**; 8.20% of fit negatives even use the query source image.
These percentages describe the new source-mode dataset under the legacy negative
rule, not an audit of every historical canonical experiment.

Added `source_isolated_nulls`: same physical plane, different cell, excluded query
source, and world distance beyond the positive threshold. Invalid negatives are
masked out of BCE and match-score diagnostics, not falsely labeled. Valid counts
are 194,467/194,725 fit and 50,921/50,934 validation; same-cell and near-positive
conflicts are zero among valid negatives. Source provenance exists transiently
in the mapping dataset only. These deliberately distant negatives are EASY and
do not calibrate the runtime match/null prior. High positive/negative separation
after this repair must not be advertised as improved deployed discrimination.

V152 replays V133 weights on the new positive examples and refits only mapping
calibration; V153 retrains with the legacy negative rule and fails the full
mapping gate. V154 retrains with repaired negatives and passes. V155 replays V133
with the same repaired negative diagnostic and exports the aligned residual bank.
V157 trains the local-correlation head against V154's frozen same-input reference;
all nine relative gates pass. V158 replays V154 and exports joint residuals.
Training remains 1,200 steps, batch 512, seed 260918; no query labels are used.
The dataset has 194,725 fit pairs and 50,934 validation pairs, with 25,147 evaluation
pairs on odd sorted source images (49 images). Modes share query tokens, so these
are not 25,147 statistically independent image observations.

| Same-input held mapping evaluation | Frozen old base V155 | Aligned base V154 | Aligned local V157 |
|---|---:|---:|---:|
| Image median px | .535960 | .532512 | .489155 |
| Image P90 px | .880058 | .886003 | .816613 |
| UV median m | .085611 | .083076 | .079881 |
| UV P90 m | .176305 | .171587 | .163122 |
| Joint reprojection median px | .838566 | .800819 | .753186 |
| Joint reprojection P90 px | 1.578685 | 1.538813 | 1.451507 |

V160: base alignment changes equal-image-weight mean joint error by -.03162px,
source-image bootstrap CI [-.03899,-.02430]. V161: local vs aligned base gives
-.04835px, CI [-.05772,-.03866]. Neither interval establishes query localization
improvement or handles temporal correlation between neighboring mapping images.
V154's image P90 still regresses slightly against the frozen old base; the local
head's nine-gate pass is relative to V154, not an automatic upgrade over V134.

V162 additionally freezes the OLD local V134 weights and evaluates them on these
same anonymous inputs, with mapping-only recalibration. Its image median/P90 is
.491420/.826746px, UV median/P90 .082873/.169174m, and joint reprojection
.783390/1.484087px. V157 improves all six values on this matched dataset. V164
compares their joint errors with equal image weights: -.02946px, source-image
bootstrap CI [-.03676,-.02181]. Thus the measurement gain survives the stronger
old-local rather than old-base control, while still failing to provide stable
query-pose gains. Full results are in `mapping_mode_geometry_local_transfer_v164.json`.

### Fixed query initialization: some gains, but no mainline promotion

V156/V159 preserve all 79,536 candidate identities/tokens and use the same V135
MoGe-refined poses. Retrieval, mode budget, source-image-free map, and support
gates remain unchanged. Only the new coordinate heads feed refinement. Tests
cover isotropic and full covariance as predefined controls, not a selector sweep.

| Head/refinement, 88 reused seq10 queries | Median m | .1m/1deg | .25m/2deg | .5m/5deg |
|---|---:|---:|---:|---:|
| Old base isotropic V135 | .15629 | 22 | 73 | 84 |
| Aligned base isotropic V156 | .16078 | 24 | 73 | 84 |
| Aligned base full V156 | .15691 | 22 | 73 | 84 |
| Old local isotropic V140 | .14157 | 24 | 71 | 84 |
| Aligned local isotropic V159 | .14095 | 20 | 71 | 84 |
| Old local full V140 | .14739 | 22 | 73 | 84 |
| Aligned local full V159 | .14173 | 21 | 72 | 84 |

All listed new arms retain 87/88 and 88/88 at the two loosest thresholds.
The aligned base's +2 strict successes comprise four gains and two losses
(McNemar p=.6875). The aligned local isotropic arm loses four strict successes
net. Paired 10-frame translation intervals cross zero in every arm. These mixed
results are NOT promoted and the existing 438-query mainline stays unchanged.

V163 freezes the old local head's correspondence rows and initial Jacobians,
then independently swaps means/covariances. Old/old, old/new, new/old, new/new
yield strict hits 24/23/20/22 and .25m hits 71/72/71/73. Thus strict regression
can occur with the new means even WITHOUT reselecting correspondences; row churn
alone is not the explanation. These diagnostic solves omit the production global
support gate and are not additional production candidates.

Remaining gaps are explicit: mapping pairs are conditioned on a correct physical
cell and the existing positive projection/distance screen; actual retrieval also
contains wrong planes/cells and ambiguous candidates. The newly repaired far
negatives do not model those deployment errors. Full center-feature alignment
does not by itself align this conditional candidate distribution, the null
prior, or the downstream pose sensitivity of coordinate errors. Next priorities
are mapping-held retrieved-candidate supervision and geometry-valid HARD
negatives, plus joint-coordinate evaluation; not another global weight sweep.

All artifacts use `surface_coordinate_upgrade_v1`: source-mode checkpoints
V152–V158, query inventories `source_modes_v156_*` and `source_modes_v159_*`,
`mapping_mode_geometry_baseline_v160.json`, `mapping_mode_geometry_local_v161.json`,
and `source_modes_fixed_row_v163/report.json`. V152's early residual export carries
the original weight-head identity despite recalibrated scalars; it is superseded
by V155 and is not used by any reported joint-residual audit. Recalibrated exports
now bind the delivered calibrated head's content hash. Original checkpoints and
mainline artifacts were not overwritten. 1,063 goal-maplet tests pass.

## V165–V171: retrieved hard negatives and decoupled match-score learning

Implemented a mapping-only candidate miner that uses the complete fit-route
source-view/cell pool, not just centers surviving other positive-pair screens.
Within the known physical plane it excludes the query source, requires two
independent views per cell, selects at most four anonymous modes, then ranks
them by frozen RADIO64. Query cell/world labels are used AFTER ranking to label
geometrically safe negatives. `radio_topk` takes the first valid negative among
the top 16; `pool_far` is the predefined full-pool farthest-negative control.
No held query image, query pose label, or runtime source image is used to fit it.
This is **known-plane within-chart retrieval**, not full physical-plane retrieval.
Early artifact flags named `pool_filtered_by_query_labels=false` refer only to
query-cell/world filtering; the known-plane conditioning is an explicit limit.
The code now names those fields precisely and records selected-row digests.

### Evidence for a real local ambiguity problem

On the held mapping route, the median candidate pool is 227 modes. Even given
the correct physical plane, the top-ranked candidate is in a different cell
and beyond the positive world-distance threshold on 30.54% of eligible unique
queries (22.65% on fit). This is not an end-to-end query failure rate.
Far-negative median RADIO cosine/distance is .064/5.784m on validation;
top-16 hard-negative values are .835/.523m. There are 50,932 valid validation
negative pairs. Entire-source exclusion, geometry-safe labels, deterministic
ranking, duplicate-query sharing, and atlas support constraints have tests.

V165 trains with complete-pool far negatives and passes mapping coordinate
gates. V166 trains jointly with retrieved hard negatives at the same seed,
1,200 steps and batch 512. V167 replays the former aligned base V154 against
the SAME hard-negative mining configuration without updating weights.
V168 audits 25,145 common held pairs:

| Hard-negative held metric | Old far-trained V167 | Joint hard-trained V166 |
|---|---:|---:|
| Sampled AUC | .86096 | .97053 |
| Balanced BCE | 1.47890 | .21071 |
| Positive recall at .5 | .98763 | .91796 |
| Negative false positive at .5 | .93633 | .05818 |
| Paired positive score above negative | .88284 | .95474 |

The high old-model false-positive rate is a concrete reason NOT to treat its
sigmoid as a deployed probability. The hard-trained classifier improves ranking,
not just score offset. But V166 fails the unchanged image-uncertainty monotonicity
gate: middle quartile mean errors are .5466625 and .5465763px. The difference is
tiny, yet the gate is not relaxed after seeing it. No V166 query pose experiment
is run. Its UV median/P90 .08314/.17203m also does not dominate the far control
(.08250/.17194m), so joint training is not sold as a coordinate upgrade.

### V169/V170: improve scores without changing coordinate weights

Added `match_only_finetune`: load the validated V154 checkpoint, freeze every
parameter except `match.weight` and `match.bias`, and train on the same hard
negatives. Every non-match state tensor is asserted bitwise identical afterward;
tests confirm AdamW cannot change the coordinate parameters or outputs.
Mapping coordinate gates remain passing. V170 compares on the same 25,145 pairs:
AUC .94242, BCE .32358, positive recall .90173 and negative false positive .10698
at .5. This trades some of V166's discrimination for exact coordinate preservation.
It is a useful learned-scoring improvement, not a calibrated deployment prior.

Separately, a score rejection threshold is fitted ONLY on even-source-image
calibration positives: the fixed 5th percentile is .1219096795. Odd-image
evaluation retains 95.67% of positives and accepts 27.67% of hard negatives.
The calibration target (95%) is fixed before query evaluation. The threshold
serves as a rejection rule, not as a Gaussian-mixture existence probability.

### V171 query switch test: late gating is not a pose upgrade

The new score head is applied to the original 79,536 candidate hypotheses.
World points, image coordinates, both covariance arrays, query tokens and
candidate provenance are bitwise equal to V156. Same V135 MoGe initialization,
same nearest-token selection, same isotropic refinement and safety gates.
The sole intervention is applying the frozen mapping threshold AFTER geometric
token-hypothesis selection. The no-gate control exactly reproduces V156 poses.

| 88 reused seq10 queries | No gate | Fixed mapping gate |
|---|---:|---:|
| Median translation m | .160781 | .157109 |
| Median rotation deg | .393792 | .396417 |
| .1m/1deg successes | 24 | 22 |
| .25m/2deg successes | 73 | 73 |
| .5m/5deg successes | 84 | 84 |
| 1m/10deg, 2m/45deg | 87 / 88 | 87 / 88 |
| Fixed correspondence count | 29,051 | 28,704 |

Mean paired translation change is +.525mm; 10-frame block CI [-.791,+2.000]mm.
Both strict changes are losses, with no gains. Thus no mainline promotion and
no threshold sweep follow this result. The existing 438-query baseline is intact.

Post-freeze GT-pose reprojection audit: retained points have median/P90
1.163/2.678px; rejected points 1.987/3.509px. The score identifies poorer points
on average, but 50.43% of the 347 rejected points still have <2px GT-pose residual.
This is a map-geometry consistency diagnostic, not sensor correspondence truth.
Only 1.19% of already geometrically screened points are removed. The evidence
supports testing learned scores EARLIER in candidate association/initialization;
it does not establish that more aggressive late pruning will improve localization.

Deliverables: complete-pool source-isolated mining, score-only fine-tuning,
matched mining score audit, mapping-only positive-quantile rejection, and a fixed
query on/off test. 1,069 goal-maplet tests pass (one existing warning), diff check
passes. Artifacts under `surface_coordinate_upgrade_v1`: checkpoints/residuals
`mapping_retrieved_*v165/166/167/169`, `retrieved_match_audit_v168/170.json`,
`mapping_match_gate_v170.json`, and `retrieved_match_v171_*` query/paired reports.
Remaining gaps: cross-plane retrieval negatives, hard positive examples outside
the current correct-cell projection screen, and calibrated early association
under the actual retrieved candidate population. No raw-source-image localization
or traditional-image retrieval path was introduced.

## V172–V177: implementation audit, unique-token LM, and early association

### Confirmed implementation inconsistency and repairs

`_score` already counts each query token once, but `_solve` previously sent ALL
RANSAC inlier hypotheses to LM, including mutually exclusive alternatives for
the same image token. On the 88 all-correspondence PnP seeds, there are 39,860
raw inliers but only 28,337 distinct-token inliers: **11,523 duplicate supports**.
All 88 seeds contain duplicates, with median multiplicity 1.398. This audit
does not use query GT. In addition, six rows could previously mean fewer than
six actual image measurements.

The corrected default `unique_token_lm` retains the lowest-residual hypothesis
per token at the RANSAC pose, requires six distinct tokens, and gives only those
rows to LM. The same fix is used in anonymous-view-geometry refinement. Tests
mock RANSAC/LM to verify six unique measurements rather than twelve alternatives,
stable ties and rejection when duplicate rows fake six measurements. Explicit
`--solver_policy legacy` preserves reproducible historical controls. The change
does NOT replace OpenCV RANSAC's internal flattened-hypothesis scoring; a fully
token-aware robust initializer remains a modeling gap. Do not claim the whole
probabilistic backend is now solved.

Two robustness/interpretation issues were also repaired:

- Probability range checks alone do not reject NaN. Correspondence loading now
  rejects nonfinite world/camera/distortion/measurement-variance/match arrays and
  validates offset endpoints/order, token range/dtype and row shapes. Tests use
  malformed inputs with recomputed valid hashes, so checks are not merely hash
  failures. No evidence was found that current sealed inputs contain these NaNs.
- The metric atlas's third provenance column is an anonymous prototype, not a
  source view. Candidate origin and metadata now say so. Historical `view_*`
  score keys are retained as compatibility aliases but explicitly do not mean
  independent-view evidence. No source-view IDs or RGB were added to the map.

### Early learned association, with unchanged scoring population

Four predefined arms use the same V169 match-only head, same 79,536 candidates,
same seed budgets and raw-inlier pose selection:
V172 legacy LM/all hypotheses; V173 unique-token LM/all; V174 unique-token LM
with the frozen V170 mapping score gate BEFORE PnP; V175 unique-token LM with
one highest-match-score hypothesis per query token. Candidate poses are scored
against ALL original correspondences with unique-token support, not only the
retained subset. Thus deleting candidates cannot inflate the scoring denominator.
All arms go through anonymous view geometry, MoGe plane/scale refinement and
uncertainty refinement with fixed settings. The score-gate threshold is not tuned.

| Complete backend, 88 reused seq10 queries | Median m | .1m/1deg | .25m/2deg | .5m/5deg | 1m/10deg | 2m/45deg |
|---|---:|---:|---:|---:|---:|---:|
| V172 legacy control | .15727 | 21 | 72 | 84 | 87 | 88 |
| V173 unique-token LM | .16088 | 25 | 72 | 84 | 87 | 88 |
| V174 early mapping gate | .15818 | 23 | 71 | 84 | 87 | 88 |
| V175 global token Top-1 | .15711 | 21 | 72 | 84 | 87 | 88 |

V173 gains four strict successes and loses none vs V172, but exact McNemar p=.125
and the paired 10-frame translation CI spans zero (mean +.825mm, CI
[-2.987,+4.685]mm). Median translation regresses slightly. This is a promising
strict-threshold change with a sound consistency fix, not a universal accuracy
upgrade. V174/V175 lose strict successes vs V173 and are not promoted.

At the initialization stage, global Top-1 also harms broad recall: 2m/45deg
successes drop from 88 to 86 and 1m/10deg from 87 to 84. Downstream geometry
recovers them, which is why reporting only final medians would hide the damage.

V176 repeats unique-token LM with the OLD V133 coordinate head and its original
V135 correspondences, independently of the new match classifier. The final
median improves .15629 -> .15178m; strict hits remain 22 (one gain/one loss),
.25m hits drop 73 -> 72, and other thresholds are unchanged. Mean translation
delta -3.393mm has 10-frame CI [-8.941,+1.862]mm. This cross-head control prevents
claiming that the consistency fix improves every metric or every head.

The initialization and full-backend tables are not directly comparable to V171's
fixed-initialization score switch test. These experiments RECOMPUTE initialization.
No branch is chosen per query using labels, and no new full438 score is claimed.

### Scope-correct association follow-up

The match head was trained on same-physical-plane negatives. Comparing its raw
scores across different candidate planes and keeping global Top-1 is therefore
an unsupported cross-plane calibration assumption. Added `match_top1_per_plane`:
rank only inside each (query token, physical plane), retaining cross-plane
ambiguity. Unit tests verify that two distinct planes survive while duplicate
within-plane alternatives are reduced. V177 tests this specific scope correction
after the negative global-Top1 diagnosis; it is exploratory, not a blind test.

V177 final median is .15985m with 24/71/84/87/88 hits. Versus V173 it loses
one strict success and one .25m success, with no gains at those thresholds.
Its 10-frame mean-translation CI also crosses zero. Preserving cross-plane
ambiguity limits the damage compared with global Top-1 but does not justify
hard within-plane pruning. It remains disabled by default. The next principled
target is a token-aware robust initializer that retains hypotheses while avoiding
duplicated measurement support, not more match-score threshold sweeps.

All 1,078 goal-maplet tests pass, with one existing scatter_reduce warning;
`git diff --check` passes. This is a scoped audit, not a guarantee that every
remaining implementation or design issue has been found.

Artifacts live under `surface_coordinate_upgrade_v1` as `front_association_v172`
through V175, their `_view/_moge/_final` stages and paired reports, and
`unique_token_v176*`. Code fixes are implemented, historical pose artifacts remain
untouched, and the 438-query mainline score is not replaced by this 88-query result.

## V178–V181: token-level initialization and solver correctness audit (2026-09-08)

Implemented `token_hypothesis_ransac.py`: canonical exact-hypothesis deduplication,
four distinct query-token sampling with AP3P solution enumeration, positive-depth
reprojection support counted once per token, and local LM proposals accepted only
when the same token-level consensus score improves. All alternatives remain
available; neither query GT nor match-score threshold tuning selects hypotheses.
The experimental policy is opt-in; the default remains `unique_token_lm`.

V178 uses 128 fixed trials per seed group, with the historical row-count group
budget, then the existing view-geometry / MoGe3 / uncertainty backend. V179 is a
fresh unique-token-LM control on the same V171 correspondence inventory. This is
not a compute-matched comparison to OpenCV's 1000-iteration / .999-confidence
adaptive initializer, nor a calibrated success-probability guarantee.

| 88-query seq10 development experiment | Median translation | Hits: .1m/1deg, .25m/2deg, .5m/5deg, 1m/10deg, 2m/45deg |
| --- | --- | --- |
| V179 refreshed control, final | .160880 m | 25 / 72 / 84 / 87 / 88 |
| V178 token RANSAC, initialization | .177595 m | 18 / 62 / 79 / 86 / 87 |
| V178 token RANSAC, final | .160728 m | 22 / 72 / 84 / 87 / 88 |
| V180 independent-token seed budget fix, final | .160880 m | 25 / 72 / 84 / 87 / 88 |

The corresponding V179 initialization median is .167449 m, with
20 / 64 / 82 / 87 / 88 hits. V178 loses three strict successes, gains none
(paired McNemar p=.25); mean translation delta is +1.333 mm and its 10-frame
block 95% CI is [-4.527, +8.303] mm. Final p90 translation worsens from
.340298 m to .363745 m. Do not promote this initializer on the basis of its
tiny median improvement. The negative result concerns this sampler/budget, not
the feasibility of a token-aware or plane-based backend in general.

Three additional implementation issues were fixed with regression tests:

- Unique-token LM now excludes behind-camera RANSAC inliers before refinement.
- A perfect zero reprojection median is no longer treated as a missing/worst
  score by Python truthiness during candidate tie-breaking.
- Seed groups are filtered and budgeted by independent tokens, not duplicated
  rows; at least six distinct tokens are required before applying the group cap.
  Explicit `--seed_group_support row_count` retains the historical control.

On actual V171 inputs, old grouping selected 2747 groups across 88 queries and
three group types, including nine groups with fewer than six independent tokens.
The corrected version selects 2740 groups; selected group sets differ in 79
query/type combinations. Nevertheless all three selected initial-pose rules,
and V180's entire final pose array, are exactly unchanged versus V179. This is
a correctness/budget fix, not a claimed accuracy gain on this sample.

Token consensus scoring was vectorized with `minimum.reduceat`. A 1000-call
first-query microbenchmark took 1.0894 s for reference scoring versus .1137 s
for vectorized scoring (~9.6x), with exactly equal scores. This is a kernel
microbenchmark, not an end-to-end runtime claim. Synthetic tests also check
duplicate/permutation invariance, radial projection, cheirality, collinear
samples, and rejection of harmful LM proposals.

V181 reruns the vectorized initializer on all 88 queries with V178's explicit
row-count group policy. Every non-metadata array is exactly equal to V178,
including the complete candidate pose pool, origins, support and all three
selected-pose rules. Thus the kernel optimization preserves actual experiment
outputs, not only synthetic scores. Artifacts: `token_ransac_v181_vectorized.*`.

Validation: `python -m pytest -q tests/test_goal_maplet*.py` passes 1087 tests
with one existing scatter_reduce warning. Full `pytest -q tests` cannot collect
`test_matcha_moge_validity_masks.py` because this environment lacks `pytorch3d`;
therefore this is not a claim that the entire repository passes. `git diff
--check` passes. Artifacts use `token_ransac_v178*`, `token_ransac_v179_control*`,
and `token_ransac_v180_groupfix*` under `surface_coordinate_upgrade_v1`, including
the V178 paired report. Reused seq10 remains development evidence; no new blind
test claim or replacement of the 438-query mainline 104/336/395/409/413 result.

Remaining design issue: legacy OpenCV RANSAC still scores flattened alternative
rows internally. The new initializer fixes this semantic mismatch but has not
improved accuracy. Future work should test proposal quality / geometric
conditioning under a predeclared budget and verify on additional frozen query
routes, rather than repeatedly tune thresholds against this same sequence.
