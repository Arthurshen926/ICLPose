# VFM-MapLoc Evaluation Protocol

## Protocol Kinds

Every artifact must declare exactly one protocol kind:

- `controlled_lattice`: GT-centered local lattice. Diagnostic only.
- `reference_pose`: reference image or reference pose candidates.
- `real_retrieval`: retrieval candidates produced without GT candidate
  generation. This is the deployment-like protocol.
- `rendered_pose`: explicit rendered pose candidates around a declared init.

Reports must not merge different protocol kinds into one claim table.

## Input Whitelist

Training inputs may include:

- query raw or selected VFM tokens
- candidate raw or selected VFM tokens
- rendered selected map features
- declared candidate prior
- visibility, depth, normal, and geometry validity

Training inputs must not include:

- GT pose as scorer input
- oracle rank
- oracle cost
- pose error
- solver success label as an input feature

Labels may use GT pose only to build costs, basin labels, and evaluation
metrics. This usage must be declared in the protocol config.

## Gate 1: Feature Utility

Purpose: show that selected features are better localization evidence than raw
or trivial compressed features.

Baselines:

- retrieval order
- metadata-only scorer
- raw RADIO/C-RADIO
- raw DINOv2
- PCA-64
- same-dimension random projection
- teacher reconstruction projection
- selected feature

Metrics:

- `pred_cost_m`
- `oracle_gap_m`
- `top1_acc`
- `spearman`
- `kendall`
- `ndcg_at_10`
- `basin_recall_at_1/5/10`
- `hard_false_accept_rate`
- `ece`
- `risk_coverage_auc`

Expected OldHospital controlled diagnostic gate:

- `pred_cost_m <= 0.21`
- `top1_acc >= 0.75`
- `oracle_gap_m <= 0.09`
- `spearman >= 0.55`

Passing this gate on controlled data is not a deployment localization claim.

## Gate 2: Causal Selection

Controls:

- query feature shuffle
- map/render feature shuffle
- wrong-scene map feature
- remove high-utility channel groups
- remove low-utility channel groups
- mask high-utility spatial regions
- mask low-utility spatial regions
- full feature vs selected feature vs PCA/random same dimension

Expected behavior:

- high-utility removal degrades ranking or basin metrics clearly
- low-utility removal has limited effect
- shuffle/wrong-scene controls collapse toward metadata-only performance

## Gate 3: Mapability

Metrics:

- track within-feature variance
- between/within separability
- rendered selected feature residual
- rendered scoring retention
- selected feature storage cost
- valid rendered selected-feature coverage

Expected behavior:

- selected track variance is lower than raw/PCA/random at the same dimension
- rendered selected-map scoring retains the main trend of 2D selected scoring

## Gate 4: Hard-Case Utility

Hard subsets:

- retrieval top1 wrong but topK contains a good candidate
- repeated corridor or similar facade
- weak texture
- photometric ambiguity
- near-identity false positive
- high PnP inlier but wrong pose

Metrics:

- hard false accept rate
- hard top1 accuracy
- hard basin recall@K
- catastrophic failure rate
- risk-coverage AUC

Expected behavior:

- reduce hard false accepts by at least 20-30 percent relative to retrieval or
  metadata-only baselines
- improve risk coverage even if final median pose does not beat HLoc

## Gate 5: Final Localization

Final pose is evaluated with fixed downstream solvers only. Report:

- solver-free verifier top1 pose
- verifier guarded handoff
- fixed external solver result
- HLoc / retrieval / fixed-solver baselines

Expected behavior:

- fixed solver handoff must not be systematically worse than solver-free top1
- only claim final localization improvement when paired bootstrap and relevant
  success-rate tests support it
- otherwise claim risk/verification evidence, not SOTA localization
