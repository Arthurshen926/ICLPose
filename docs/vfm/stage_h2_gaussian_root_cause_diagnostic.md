# Stage H2 Gaussian Root-Cause Diagnostic

## Question

The current Gaussian feature-map branch gives weak localization despite corrected all-view raw aggregation. This diagnostic tests whether the bottleneck is:

1. Gaussian point geometry / patch-center PnP itself;
2. query-to-Gaussian descriptor matching;
3. correspondence confidence / outlier rejection.

## Tests

Scene: OldHospital q32, GT-visible submap, 20k landmark cap, 0.75-stride EPNP + LM.

Artifacts:

`output/vfm/stage_h2_gaussian_mainline/oldhospital/root_cause_diagnostics/oracle_patch_pnp/`

### 1. Oracle Patch-Positive PnP

This bypasses descriptor matching. For each query patch, it uses GT patch-positive 3D landmarks directly.

| Map | Oracle positives per patch | S@10cm/5deg | S@25cm/10deg | S@50cm/10deg | Median t | Mean matches |
|---|---:|---:|---:|---:|---:|---:|
| Gaussian | 1 | 1.000 | 1.000 | 1.000 | 0.0207m | 2941.0 |
| Gaussian | 3 | 1.000 | 1.000 | 1.000 | 0.0225m | 6062.7 |
| SfM | 1 | 1.000 | 1.000 | 1.000 | 0.0138m | 4390.2 |
| SfM | 3 | 1.000 | 1.000 | 1.000 | 0.0303m | 9536.4 |

Conclusion: the Gaussian centers are not fundamentally unusable for patch-center PnP when correspondences are correct.

### 2. Filter Current Feature Matches To GT-Correct Only

This keeps only GT-correct matches from the current feature matcher output.

| Method | S@10cm/5deg | S@25cm/10deg | S@50cm/10deg | Median t | Correct matches |
|---|---:|---:|---:|---:|---:|
| Gaussian raw1280 filtered | 0.438 | 1.000 | 1.000 | 0.1218m | 61.5 |
| Gaussian learned128 filtered | 0.812 | 1.000 | 1.000 | 0.0620m | 152.9 |
| SfM raw1280 filtered | 0.938 | 1.000 | 1.000 | 0.0579m | 203.5 |

Conclusion: current Gaussian feature matching already contains enough correct matches to localize, especially after learned128. The failure is not just recall; it is correspondence selection and false-match rejection.

### 3. Real Feature Matching Baseline

| Method | S@25cm/10deg | S@50cm/10deg | Median t | Mean matches | Inlier Patch@1 |
|---|---:|---:|---:|---:|---:|
| Gaussian raw1280 | 0.375 | 0.812 | 0.317m | 520.7 | 0.400 |
| Gaussian learned128 | 0.438 | 0.781 | 0.281m | 1000.0 | 0.531 |
| SfM raw1280 | 0.719 | 1.000 | 0.151m | 802.2 | 0.678 |

The learned128 selector improves correct-match density but also saturates the 1000-match cap. This explains why S@25 improves while S@50 and full182 median can degrade: it increases correct matches, but not enough to suppress spatially self-consistent false matches.

## Coverage Difference

GT-visible positive-set coverage is lower for Gaussian anchors:

| Map | Nonempty patch fraction | Visible landmarks | Positives per token |
|---|---:|---:|---:|
| Gaussian | 0.360 | 14084.8 | 1.726 |
| SfM | 0.538 | 20000.0 | 2.451 |

This means the Gaussian map has enough geometry for oracle PnP, but a weaker matching substrate than SfM: fewer patches have valid positives, and each patch has fewer correct candidates.

## Final Conclusion

The earlier hypothesis was too broad. The point-PnP measurement model is not the first bottleneck under GT-visible q32: oracle patch-positive Gaussian PnP reaches centimeter-level accuracy.

The core bottleneck is:

**correspondence selection under repeated / ambiguous Gaussian-VFM matches.**

More specifically:

- Gaussian raw features retrieve too few correct matches relative to SfM.
- learned128 increases correct-match density, but also pushes many matches to the cap and does not reliably reject false correspondences.
- C2-safe improved sample-level top1 but did not improve pose, which is consistent with a confidence/outlier-selection problem rather than a descriptor-capacity problem.

The next validated direction should be a correspondence verifier / confidence selector over the current match set, not another larger descriptor selector.

