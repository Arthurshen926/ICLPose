# Anchor-free 2DGS Surface Feature Alignment V5

V5 is a strict correction of V4, not another anchor/PnP variant. The deployed
map contains clean 2DGS geometry, tangent texels, feature statistics,
uncertainty, confidence, and maplet ownership. It contains no mapping RGB or
mapping image paths. Runtime matching does not use SfM points/tracks, LoFTR,
RADIO intermediate features, ALIKE descriptors, stable-anchor identity, or
point-correspondence PnP.

## Representation

The clean PLY is a quality mask: its `source_index` refers back to the complete
2DGS PLY. The complete PLY remains the source of disk rotation, normal and
anisotropic scale. Treating the clean PLY itself as full disk geometry is
invalid because the compact clean artifact deliberately omits those fields.

Mapping samples are assigned by camera-ray/2DGS-plane intersection. A sample is
accepted only when its rendered depth agrees with the intersection within
3 cm, the intersection lies inside the disk footprint, and the ray has a
non-degenerate incidence angle. The old nearest-centre 15 cm assignment is not
used.

Each retained primitive has an 8 by 8 tangent texture address space. Only
observed cells are materialized, and each cell stores one fused feature
distribution rather than a list of view descriptors. Dense feature-grid
samples establish the field; ALIKE detections may weight matchability but
never decide whether the surface exists.

## Feature branches

Retrieval and metric features are separate:

```text
raw RADIO final
  +-- retrieval mapper (low-resolution, invariant maplet retrieval)
  `-- stride-4 metric decoder
        + RADIO context half
        ` shallow RGB phase half
```

The decoder does not consume retrieval-mapped features. It emits normalized
stride-16/8/4 features plus matchability and uncertainty. Mapping RGB is an
offline baking input only; the resulting map stores the decoded features.

Legacy RADIO artifacts use their measured endpoint sampling convention. The
new decoder uses half-pixel centres (`align_corners=False`). The convention is
recorded in the field metadata and consumed by the alignment scorer. They must
not be silently mixed.

## Training

Cross-view supervision uses the same physical 2DGS surface point projected into
both images. Both projections must agree with independently rendered 2DGS
depth within 5 cm. Nominal texel pairs with fewer than four mutually visible
points are rejected rather than falling back to different observation
coordinates.

The objectives are:

- cross-view surface identity;
- displacement NLL and monotonic ranking at 0.5, 1, 2, 4 and 8 pixels;
- within-view spatial sharpness;
- direct SE(3) ranking at 5, 10 and 20 cm and 0.5 and 1 degree;
- matchability likelihood.

Training and validation view sets are disjoint.

## Alignment likelihood

The scorer uses an explicit signed inlier log likelihood and an
uncertainty-dependent null mixture. Out-of-view samples receive an explicit
likelihood. The selected surface sample set and denominator remain fixed for
every pose being compared. ALIKE detector response can modulate matchability;
it cannot contribute a separate positive reward.

The optimizer has a joint six-dimensional proposal in addition to bounded
trust-region probes. It is not evaluated as a precision result until the score
landscape is valid.

## Mandatory gate

`evaluate_2dgs_surface_score_landscape.py` evaluates oracle-visible maplets at
ground truth and translation offsets of 5, 10, 20 and 30 cm. Ground truth is
diagnostic-only. All translation axes must place ground truth near a local
maximum and must have the correct small-step direction before an optimizer or
the 530-query pipeline is run.

The first StMarysChurch audit exposed real failures rather than passing the
gate:

| field | mapping views | GT axis maximum | correct small direction |
|---|---:|---:|---:|
| legacy V4 field with corrected likelihood | existing | 20.83% | 27.08% |
| exact-disk tangent field, legacy metric | 100 | 25.00% | 25.00% |
| exact-disk tangent field, legacy metric | 400 | 14.58% | 18.75% |
| raw RADIO final tangent field | 100 | 2.08% | 4.17% |
| stride-4 factorized decoder tangent field | 100 | 16.67% | 16.67% |

The decoder validation likewise learned surface identity but did not learn the
required local/SE(3) peak. Those checkpoints are diagnostic artifacts and are
not production checkpoints. Consequently V5 has not been promoted and no
downstream optimizer/full-query number is reported as a V5 precision result.
This prevents another apparently complete run from hiding an invalid pose
objective.
