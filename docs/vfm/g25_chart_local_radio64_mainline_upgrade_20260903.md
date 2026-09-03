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

## Uncertainty-weighted pose upgrade

The next bounded optimization is also a mainline **GO**.  The solver now uses
the anonymous atlas fields that were previously stored but not consumed by the
pose optimizer: the 3x3 world-point covariance, plane-pixel purity, and metric
plane-depth dispersion.  It adds no mapping image, image path, source-view
identity, query label, or new learned parameter.

For every frozen query token, only the currently best valid 3D hypothesis is
retained.  World covariance is propagated through the pinhole Jacobian;
metric depth dispersion supplies a second conservative image-space variance;
their maximum is added to the exact variance of a uniform four-pixel RADIO
cell and divided by clipped plane purity.  A Huber six-DoF solve then uses
these fixed variances and plane-balanced weights.  It fails closed below 12
rows or two physical planes, must preserve 95% of the original unique-token
support, and cannot move the camera by more than 0.5 m or rotate it by more
than 5 degrees.

The frozen sequence is `RADIO/plane pose -> MoGe3 plane+scale -> uncertainty
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

The third item is now implemented and validated.  The next highest-value item
is a learned sub-token chart coordinate with calibrated covariance; the
mask-centroid negative control shows that segmentation pixels must not be
substituted for feature coordinates.
