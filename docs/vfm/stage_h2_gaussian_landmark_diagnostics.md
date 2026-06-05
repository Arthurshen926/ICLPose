# Stage H2 Gaussian Landmark Diagnostics

## Scope

This check inspected:

- trained OldHospital 3DGS visualizations;
- the full 3DGS PLY used by Stage H2;
- current VFM-token-saliency Gaussian landmark votes;
- the raw-VFM aggregated H2 anchor map;
- camera-view visualizations of selected Gaussian anchors.

Main diagnostic output:

- `output/vfm/stage_h2_raw_gaussian_anchors/diagnostics/oldhospital_full_v96_top20k/summary.json`
- `output/vfm/stage_h2_raw_gaussian_anchors/diagnostics/oldhospital_full_v96_top20k/training_3dgs_visualization_montage.png`
- `output/vfm/stage_h2_raw_gaussian_anchors/diagnostics/oldhospital_full_v96_top20k/gaussian_sampling_distributions.png`
- `output/vfm/stage_h2_raw_gaussian_anchors/diagnostics/oldhospital_full_v96_top20k/*_h2_gaussian_landmark_overlay.png`
- `output/vfm/stage_h2_raw_gaussian_anchors/diagnostics/oldhospital_full_v96_top20k/*_3dgs_pseudo_rgb.png`

## 3DGS Visual Quality

The trained 3DGS is not completely broken. The training visualization montage shows:

- recognizable OldHospital facade geometry;
- windows, arches, columns, roof line, and depth structure are visible;
- RGB is mildly blurry and not photorealistic, with visible PSNR values around 15-18dB in the saved visualization panels.

The pseudo RGB z-buffer diagnostic is not a real 3DGS splat render, but it confirms that the PLY centers/colors project to the expected facade structure. Therefore the current H2 failure is not simply because the 3DGS map is unusable.

## Current H2 Sampling Statistics

Map:

- source Gaussians: 504,352
- voted Gaussians: 20,602
- sampled Gaussians: 20,000
- raw-VFM aggregated anchors: 19,991
- selected fraction of all Gaussians: 3.96%
- selected fraction of voted Gaussians: 97.0%

Vote distribution:

- nonzero vote median: 1
- nonzero vote mean: 1.82
- selected anchors with votes >= 2: 31.1%
- selected anchors with votes >= 3: 15.7%
- selected anchors with votes >= 5: 6.5%
- selected anchors with votes >= 10: 1.6%

Opacity:

- all Gaussian opacity median: 0.392
- selected anchor opacity median: 0.191
- selected anchors with opacity >= 0.5: 20.2%
- selected anchors with opacity >= 0.75: 11.9%

Important pattern:

- higher VFM-saliency vote thresholds do not increase opacity;
- for vote >= 5, selected opacity median is only about 0.133;
- this means current voting is not selecting high-contribution rendered Gaussians.

## Camera-View Findings

The overlay images show:

- selected anchors concentrate around high-contrast facade structures such as window frames, arches, columns, roof edges, and repeated vertical structures;
- many anchors also appear on sky/background/floating regions, black windows, tree-like structures, and repeated facade patterns;
- selected anchor density is high enough in camera view, but the anchors are not clearly filtered by render contribution or geometric reliability.

Visible anchor counts in four OldHospital views:

- visible all Gaussians: about 365k-412k
- visible selected anchors: about 12.9k-15.6k
- selected visible ratio: about 3.5%-3.8%, close to the global selected fraction

So the issue is not that too few selected anchors project into the query view. The issue is that many selected anchors are weak or ambiguous localization primitives.

## Root Cause Hypothesis

The current H2 sampler is still too weak as a ULF-Loc-style landmark selector.

Two concrete problems are visible:

1. Token-saliency votes are not alpha/contribution-aware.

   The current vote assigns a salient VFM token to the frontmost projected Gaussian owner. It does not ask whether that Gaussian contributes meaningful opacity/color to the pixel. Low-opacity floaters can receive votes if they are slightly in front.

2. Raw aggregation observation count is not true visibility.

   The H2 raw aggregation currently samples each selected Gaussian from every reference view where its projection lies inside the token map. Unless `require_token_owner_visibility` is enabled, this does not ensure that the Gaussian is actually the dominant visible contributor. This explains why anchor observation counts look high while localization reliability remains weak.

## Conclusion

The trained 3DGS is visually usable, but the current H2 Gaussian landmark selection is not yet selecting reliable localization landmarks.

The main blocker is not the number of Gaussians. It is that VFM token-saliency voting is acting as an edge/high-contrast heuristic, not as a render-contribution-aware landmark reliability score.

The next fix should be contribution-aware:

- vote only for high-alpha or dominant-contribution Gaussians;
- require owner visibility during raw VFM aggregation;
- rank anchors by a combined score: vote count, opacity, contribution consistency, multi-view visibility, feature variance, and local ambiguity;
- compare the resulting anchor set against the current H2 set with the same camera-view diagnostics before rerunning localization.

## Optimized Sampling Sweep

An opacity-aware H2 sampler was added after this diagnostic.

New controls:

- `vote_min_owner_opacity`: ignore low-opacity front owners during VFM token-saliency voting.
- `sample_opacity_power`: rank candidates by `votes * opacity^power`.
- `sample_min_opacity`: reject low-opacity Gaussian anchors before NMS.
- `aggregation_owner_min_opacity`: optional owner-visibility threshold for raw feature aggregation.

OldHospital q32 with old C2.5 selector projection:

| Sampling | Anchors | Aggregation | Submap | S@25 | S@50 | Median t | Inlier Patch@1 |
| --- | ---: | --- | --- | ---: | ---: | ---: | ---: |
| old H2 full | 19,991 | projection-in-image | all | 0.3125 | 0.6875 | 0.355m | 0.317 |
| opacity-aware top20k | 19,914 | projection-in-image | all | 0.3125 | 0.7813 | 0.342m | 0.333 |
| opacity-aware top10k | 9,995 | projection-in-image | all | 0.2500 | 0.6563 | 0.363m | 0.334 |
| opacity-aware top20k | 17,231 | owner-visible | all | 0.2500 | 0.6875 | 0.368m | 0.374 |
| old H2 full | 19,991 | projection-in-image | GT-visible | 0.2813 | 0.7500 | 0.324m | 0.332 |
| opacity-aware top20k | 19,914 | projection-in-image | GT-visible | 0.2813 | 0.7813 | 0.328m | 0.307 |
| opacity-aware top20k | 17,231 | owner-visible | GT-visible | 0.2813 | 0.8750 | 0.346m | 0.374 |

Sampling diagnostics for the recommended opacity-aware top20k variant:

- selected opacity median improved from 0.191 to 0.294;
- selected opacity p05 improved from 0.004 to 0.113;
- observation-count median stayed stable, 63 to 64, because raw aggregation still uses multi-view projection averaging;
- all-map S@50 improved from 0.6875 to 0.7813, while S@25 stayed unchanged.

The owner-visible aggregation variant is useful diagnostically: it raises PnP-inlier Patch@1 and GT-visible broad recall, but it reduces feature observation count too much and hurts all-map precision. It should not be the default raw aggregation path yet.

Current recommended H2 sampling default:

```bash
--vote_min_owner_opacity 0.1 \
--sample_opacity_power 1.0 \
--sample_min_opacity 0.05 \
--max_anchors 20000
```

Do not enable `--require_token_owner_visibility` as the default until descriptor aggregation is redesigned to handle sparse owner-visible observations, for example by multi-prototype or owner-visible plus fallback aggregation.
