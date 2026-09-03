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
