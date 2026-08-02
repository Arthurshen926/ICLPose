# V7 architecture decision: hierarchical maplets and measured pose basins

## Outcome

The desired deployable shape remains a two-compute-stage system:

```text
query RADIO-final + ALIKE detections
  -> one structured maplet retrieval / coarse-pose stage
  -> a few local 2DGS surface renders
  -> continuous feature-field alignment
  -> held-out maplet verification
```

The July V6 global chart-frame correlation bridge is frozen as a diagnostic
fallback. It is not the intended V7 production method. However, it cannot yet
be deleted: the current Stage-A statistics do not produce a pose inside a
measured Stage-C basin, and the current Stage C has no reliable translational
capture basin even from a 5 cm oracle initialization.

The map contract remains strict. No runtime artifact may store mapping RGB,
mapping image IDs or paths, per-view observation descriptors, SfM points or
tracks, ALIKE descriptors, RADIO intermediate features, LoFTR inputs, or point
correspondence PnP inputs. ALIKE is a detector only. Matching and alignment use
transformed RADIO-final features.

## What was implemented and tested

`localization_v7.pose_signature` implements a compact experiment requested by
the architecture review:

1. Run Stage A once and retain its Top-64 maplet result.
2. Compress it into maplet identity evidence, normalized query-region
   location/extent moments, variance, and mass.
3. Associate those statistics offline with mapping-sequence ground-truth
   poses.
4. At runtime compare only fixed sufficient statistics. There is no further
   query-to-map feature interaction.
5. Produce continuous poses with pose-mode averaging, local linear refitting,
   or pure geometric KDE voting.

The resulting 1,487-pose / 807-maplet prototype bank is 1.8 MB. Its loader
rejects artifacts that declare mapping RGB, image identity/path, per-view
descriptors, SfM/tracks, ALIKE descriptors, or RADIO intermediate features.

Two supervised regressors were also tested. Hyperparameters were selected on
the complete held-out mapping trajectory `seq11`, before evaluating the fixed
strict queries on `seq3`, `seq5`, and `seq13`:

- PCA plus ridge regression;
- nonlinear ExtraTrees regression.

This tests the strongest version of the suggestion that mapping-sequence poses
could refit a continuous coarse pose without another query/map feature pass.

## Coarse-pose result

All values below use the strict 12-query audit and the clean 2DGS lineage.

| Coarse estimator | Top-1 translation median | Top-1 rotation median | Any Top-16 30 cm / 3 deg |
| --- | ---: | ---: | ---: |
| identity-only pose prototypes | 4.34 m | 8.28 deg | 0/12 |
| identity + mean layout prototypes | 5.03 m | 11.09 deg | 0/12 |
| clustered SE(3) means | 5.15 m | 13.54 deg | 0/12 |
| local linear pose refit | 5.15 m | 13.72 deg | 0/12 |
| geometry-only KDE modes | 8.05 m | 19.77 deg | 0/12 |
| PCA-ridge pose regression | 5.85 m | 12.08 deg | 0/12 |
| ExtraTrees pose regression | 3.53 m | 9.60 deg | 0/12 |

The supervised failure is already present on held-out `seq11`: PCA-ridge has
11.58 m median translation error and ExtraTrees has 6.19 m. It is therefore
not a strict-test tuning issue.

The geometric audit explains why simple voting is insufficient:

- the nearest retrieved Top-64 mapping centre has 0.90 m median error;
- after requiring orientation within 3 deg, the nearest centre is 3.17 m away;
- the candidate-centre convex hull often contains the query, but only because
  widely scattered wrong modes surround it.

Ground truth can choose a convex combination inside that hull. Candidate
geometry alone cannot determine which combination is correct. A density vote
selects a repeated visual mode elsewhere in the scene.

The failed representation discards the critical variable: the multi-instance
region topology. One mean location and extent per maplet cannot distinguish
several disconnected occurrences, repeated church structure, or mutually
inconsistent maplet layouts. A stronger regressor cannot recover information
that Stage A already marginalized away.

## Measured Stage-C capture basin

The Stage-C evaluator was corrected before this audit. It previously reported
`maximum_committed_translation_updates = 1` while silently using the function
default of two. The call now explicitly uses one. Its former strict
floating-point comparison also counted unchanged errors at approximately
`1e-15 m` as translation improvements; the corrected diagnostic requires a
meaningful 1 mm reduction.

Every row below gives Stage C the oracle visible maplet identities. Thus this
isolates feature-field alignment from retrieval.

| Initial perturbation | Median initial flow | Final translation median | Final rotation median | Meaningful translation improvements |
| --- | ---: | ---: | ---: | ---: |
| 2 cm, 0 deg | 0.95 px | 2.0 cm | 0.250 deg | 0/12 |
| 5 cm, 0 deg | 2.36 px | 5.0 cm | 0.375 deg | 0/12 |
| 10 cm, 0 deg | 4.73 px | 10.0 cm | 0.250 deg | 0/12 |
| 15 cm, 0 deg | 7.11 px | 15.0 cm | 0.375 deg | 0/12 |
| 20 cm, 0 deg | 9.49 px | 20.0 cm | 0.500 deg | 0/12 |
| 30 cm, 0 deg | 14.27 px | 30.0 cm | 0.730 deg | 0/12 |
| 0 cm, 0.25 deg | 3.40 px | 0 cm | 0.250 deg | 0/12 |
| 0 cm, 0.5 deg | 6.80 px | 0 cm | 0.500 deg | 0/12 |
| 0 cm, 1 deg | 13.59 px | 0 cm | 0.716 deg | 0/12 |
| 0 cm, 2 deg | 27.17 px | 0 cm | 1.243 deg | 0/12 |

Translation is the decisive failure. There are no meaningful improvements at
any tested 2/5/10/15/20/30 cm start. At 2 cm, four queries are instead pushed
farther away and P90
translation reaches 26.2 cm. At 5 cm, Stage C accepts no translation update
and adds a false rotation on 8/12 queries. At 20 cm, its three accepted
translation updates all worsen the pose, and every query acquires false
rotation. Rotation from 1--2 deg has a useful coarse gradient, but does not
converge to a precise orientation and introduces translation drift on 3/12 and
4/12 queries respectively. Sub-degree rotation is inconsistent: at 0.5 deg,
four queries improve and four worsen.

Stage C is therefore a pose-preservation / partial orientation-ranking module,
not a qualified six-degree-of-freedom refiner. A coarse pose within 30 cm is
not a sufficient contract for the current implementation.

## First-principles architecture decision

Feature-field alignment is still theoretically appropriate. It avoids the
conflict between globally distinctive retrieval features and precise local
point identities. It does not require global feature uniqueness or one
connected planar maplet. Disconnected surface children can be rendered into
one query canvas and optimized jointly.

It does, however, require all of the following locally:

- a spatially varying correlation likelihood with the correct pose gradient;
- sufficient directional observability across several surfaces;
- view-consistent feature transport at the rendered surface cells;
- correct occlusion, visibility, uncertainty, and null semantics;
- geometry fine enough that a sub-token displacement is represented rather
  than quantized away.

Current RADIO atlas statistics do not satisfy that contract. Saying that
alignment needs less global distinctiveness does not mean it needs no local
discriminability, continuity, or calibrated spatial gradient.

The map should not contain two peer identity systems. It should contain one
hierarchical entity:

```text
HierarchicalMaplet
  retrieval parent
    context-rich, possibly non-planar/disconnected region identity
    bounded structured appearance statistics
  geometry-owned child surface tiles
    fixed clean-2DGS XYZ, normal/frame, support and uncertainty
    local transformed RADIO-final feature field
```

The parent is optimized for scene-level identity and layout. Child tiles are
optimized for local rendering and alignment. They share one identity lineage;
the parent-to-child relation is ownership, not a learned many-to-many bridge
between two maps.

## Next promotion gates

Further tuning of the current mean-layout pose signature or Stage-C acceptance
thresholds is stopped. The next implementation must pass these gates in order:

1. Preserve a bounded set/graph of query regions: normalized location, scale,
   posterior, null, overlap, and repeated-instance topology. Do not average all
   instances of one maplet into one location.
2. Train a permutation-equivariant set-to-pose-mode head with mapping poses,
   using complete trajectory holdouts. It may output several anonymous SE(3)
   modes and covariance, but no reference image identity.
3. Require held-out coarse candidate coverage before Stage C: Top-N must reach
   at least the empirically measured basin, not an assumed 30 cm basin.
4. Train/validate the child surface feature field for local pose equivariance.
   Report likelihood landscapes and update direction at 2/5/10/15/20/30 cm
   and 0.25/0.5/1/2 deg before end-to-end evaluation.
5. Require Stage C to improve, not merely preserve, both translation and
   rotation with maplet-disjoint verification and a no-op option.
6. Only after gates 1--5 pass may the old global Stage-B chart correlation be
   removed and the 530-query evaluation be opened.

## Reproducible artifacts

- pose bank: `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v7/pose_signatures_mapping1487.npz`
- strict coarse report: `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v7/pose_aware_retrieval_strict12.json`
- trajectory-held-out regression report: `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v7/pose_signature_regression_strict12.json`
- consolidated decision dashboard: `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v7/architecture_decision_strict12.json`
- Stage-C basin reports: `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v7/stagec_basin_*.json`
