# VFM-MapLoc Experiment Plan

## Phase 0: Protocol Freeze

- Define protocol config for every artifact.
- Add leakage checks before training.
- Split controlled, reference-pose, real-retrieval, and rendered-pose reports.
- Use a single gate table schema for all methods.

## Phase 1: Raw VFM Token Bank

- Extract raw RADIO/C-RADIO spatial tokens.
- Extract raw DINOv2 spatial tokens as a second VFM baseline.
- Store layer, stride, channel count, model name, image id, split, scene, and
  checksum.
- Normalize and optionally whiten tokens without using localization labels.

## Phase 2: Candidate Hypothesis Library

- Build retrieval/reference candidates from fixed HLoc or retrieval outputs.
- Build rendered-pose candidates around declared init poses.
- Mine same-scene hard negatives.
- Record candidate type, generator, pose, pose cost label, basin label,
  render availability, solver label, and hard-case type.

## Phase 3: Feature Selector Training

- Default selector: layer/channel group gates plus 1x1 projection to 64D.
- Losses: track consistency, score-hard negative contrast, listwise pose rank,
  basin BCE, sparsity, and calibration.
- Run 5 seeds for trained selector components.
- Use Spearman and oracle gap as checkpoint gates, not only top1.

## Phase 4: Selected Feature Map

- Lift selected features into COLMAP tracks or equivalent explicit map
  primitives.
- Save mean feature, variance, visibility count, geometry validity, utility, and
  uncertainty.
- Compare selected tracks against raw, PCA, and random features.

## Phase 5: Map-Conditioned Verifier

- Render or project selected map features for every candidate.
- Score query-map selected feature evidence, visibility, geometry consistency,
  uncertainty, and declared prior.
- Run feature shuffle, wrong-scene, and metadata-only controls.

## Phase 6: Fixed Solver Handoff

- Evaluate OldHospital real retrieval top20/top50.
- Evaluate ShopFacade and at least one multi-scene reference-pose protocol.
- Keep HLoc/LoFTR/PNP/photometric/GS-style solvers fixed.
- Report hard-case subsets separately from full-set median.

## Statistics

- 5 training seeds for trained selector/verifier components.
- Query-level paired bootstrap with 10,000 resamples.
- McNemar test for binary success rates.
- Wilcoxon signed-rank test for paired continuous pose errors.
- ECE and risk-coverage AUC for calibration.
