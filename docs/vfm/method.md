# VFM-MapLoc Method

## Problem

Given a query image, a prior map, a frozen vision foundation model, and a set of
candidate localization hypotheses, learn a selected feature representation that
provides reliable map-conditioned evidence for deciding which hypotheses are
valid.

The selected features are not expected to replace geometry, matching, PnP, or
photometric refinement. They are expected to verify hypotheses, reject hard
false positives, calibrate risk, and supply stable map features.

## Terminology

- Raw VFM token bank: frozen dense tokens from RADIO/C-RADIO, DINOv2, or another
  foundation model layer/resolution.
- Localizable feature subspace: a compact selected representation that preserves
  evidence useful for localization hypothesis verification.
- Candidate hypothesis: a place, reference-pose, rendered-pose, or solver
  hypothesis to be verified.
- Selected feature map: explicit 3D tracks, surfels, Gaussians, or map
  primitives carrying aggregated selected features.
- Map-conditioned hypothesis verifier: a scorer that compares query selected
  features against rendered or projected selected map evidence.

## System Modules

### Candidate Generator

This is not the contribution. It may be HLoc, NetVLAD, AnyLoc/SALAD-style
retrieval, scene coordinate regression, VO/SLAM priors, rendered pose sampling,
or a fixed solver output.

It must record candidate type, candidate generator, split, whether GT was used,
and whether the candidate is solver-conditioned.

### Foundation Feature Selector

The selector consumes raw dense tokens and outputs:

- `z`: compact selected descriptor, typically 32 or 64 dimensions.
- `m`: spatial utility/evidence map.
- `sigma`: uncertainty/risk map.
- `g`: interpretable layer or channel-group gates.

The default architecture is a group gate plus 1x1 projection, followed by
normalization, spatial utility, and uncertainty heads. Large black-box heads are
not the default because they weaken the feature-selection claim.

### Selected Feature Map

Selected 2D features are lifted into explicit 3D support using COLMAP tracks,
visibility, and geometry validity. Each track stores mean feature, variance,
visibility count, and uncertainty/utility summaries.

### Map-Conditioned Verifier

For each candidate, render or project selected map features into the query view,
then score selected feature consistency, visibility, geometry consistency,
uncertainty, and declared candidate prior. The query and rendered map evidence
must use the same selector checkpoint.

### Fixed Solver Handoff

HLoc, PnP-RANSAC, LoFTR, photometric alignment, MASt3R-style matching, or
GS-CPR-style refinement are fixed validators. They are not trained as the main
method. Solver-free verifier top1 and solver handoff results are always reported
separately.

## Training Stages

1. Feature Localizability Audit: no training; compare raw VFM tokens, layers,
   PCA, random projections, and metadata-only baselines.
2. Localizable Subspace Learning: train selector with track consistency,
   hard-negative contrast, pose ranking, basin classification, sparsity, and
   calibration losses.
3. Selected Feature Lifting: aggregate selected features into explicit 3D tracks
   and validate variance/separability.
4. Map-Conditioned Verification: train or calibrate the verifier using query-map
   evidence and hard negatives.
5. Hard-Case Specialization: mine retrieval false positives, repeated
   structures, weak texture, photometric traps, and high-inlier wrong PnP cases.
6. Fixed Solver Handoff: evaluate whether selected evidence improves candidate
   selection, risk, and final localization with frozen downstream solvers.
