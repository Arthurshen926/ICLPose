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
present in the full-scene 2DGS rasterizer's top-k source-index buffer. KD-tree
assignment is not a production fallback. The saved atlas contains:

- fixed XYZ and primitive ID per texel;
- fused unit metric feature;
- scalar feature dispersion;
- support count and valid mask;
- maplet frame, extent and identity.

It contains no mapping RGB, mapping image path, observation descriptor list,
SfM point/track, ALIKE descriptor or RADIO-intermediate feature.

## Metric encoder

The metric encoder uses a phase-preserving RGB stem:

```text
Conv3x3 stride2 -> residual
-> Conv3x3 stride2 -> residual blocks
```

RADIO-final context conditions the stride-4 phase features with FiLM and
spatial residual fusion. Stride-8 and stride-16 features are derived from the
same fused metric representation. Matchability, explicit null and uncertainty
are supervised alongside correlation NLL. Training uses coupled translation
and rotation perturbations up to 30 cm and 2 degrees and differentiates through
the analytic joint-6DoF normal equations for the one-step pose objective.
Wrong texels and repeated-surface assignments are explicit pairwise-null hard
negatives; they do not incorrectly suppress query-only matchability.

## Correlation and geometry

Selected atlases are triangle-rasterized with perspective-correct feature,
XYZ and uncertainty interpolation. Conservative subpixel coverage prevents a
small surface chart from disappearing on coarser grids. Complete local
correlation probabilities are retained; mean and covariance are derived only
after normalization with the null outcome.

The solver uses the calibrated radial-camera projection Jacobian, covariance
weighting, robust IRLS and LM damping. Fit and held-out maplets are split by
stable maplet identity rather than by view index.

The solver does not assume that a raster cell's integer index is the exact
projection of its perspective-interpolated XYZ. It converts
`raster center + correlation displacement` to original-image coordinates and
subtracts the XYZ's actual subpixel projection before applying the Jacobian.
This term is mandatory for conservative subpixel triangles and coarse grids.

## Retrieval and coarse proposal

Retrieval scores every maplet mixture with its component weights. Probability
outside retained candidates is transferred exactly to unknown/null.
`QueryMapletGroup` preserves region position, extent, candidate identities and
probabilities. Coarse grouped PnP treats maplet centers as uncertain regions
and is used only to enter the local basin; it is never the final estimator.

## Promotion gates

- G0: exact contributor IDs, canonical reprojection RMS below 0.5 px,
  sufficient atlas coverage and consistent coordinates.
- G1: oracle-maplet flow EPE, correct-mode recall, null AUPRC and one-step pose
  improvement.
- G2: at least 90%, 80% and 60% of 5 cm, 20 cm and 30 cm starts respectively
  reach 4 cm / 1 degree.
- G3: retrieved maplet groups place at least 60% of queries inside the
  20–30 cm basin.
- G4: end-to-end evaluation is run only after G0–G3 pass.

Encoder validation is trajectory-disjoint. A failed gate is reported as a
failure and is never promoted by running a downstream optimizer or the full
query set.
