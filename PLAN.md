# Feature-Guided Depth-Aware Localization Plan

## Summary

把当前定位路线调整成“coarse 负责找大概，fine+depth 负责精修”。核心目标不是继续强化纯特征相似度，而是让深度显式参与平移求解，让特征学习服务于几何可解的对应关系。

## Key Changes

- 将默认 refinement 从 MLP translation head 偏向 `full_wls=True` 或等价的 depth-aware 6DoF WLS/PnP 更新。
- render-compare 不再只用 feature cosine/L2 loss；加入显式几何项：
  - rendered depth / valid mask 约束；
  - feature correspondence 诱导的 3D-2D residual；
  - 或使用已有 `feature_metric_solve` 做 feature-metric Gauss-Newton 更新。
- coarse feature 只用于：
  - 初始化候选 rerank；
  - 粗位姿 basin 扩展；
  - 低分辨率 correlation prior。
- fine feature 用于：
  - dense/semidense correspondence；
  - confidence-weighted WLS/PnP；
  - 最终平移和旋转精修。

## End-to-End Training Scheme

- 第一阶段：冻结 feature field，训练 localization head 学会可靠 flow/confidence，启用 depth-aware WLS pose loss。
- 第二阶段：低学习率解冻 fine decoder / FSM，让定位 loss 反向指导特征重建，但保留 teacher feature reconstruction loss，防止特征坍缩成只服务训练集 pose 的捷径。
- 第三阶段：加入 pose perturbation ranking loss。对同一个 query，正确 pose 的 render feature 应该优于小扰动、大扰动和错误候选。
- 第四阶段：只让 coarse 学 rerank/candidate scoring，不让 strong coarse loss 直接压 final metric pose。

## Test Plan

- 对比 `full_wls=false`、`full_wls=true`、PnP、feature-metric GN 的 translation/rotation。
- 单独报告初始化误差、一次 refine 后误差、多次 refine 后误差。
- 做 oracle-flow / predicted-flow gap 分析，确认改动是否真的改善 correspondence，而不是只调了求解器。
- 做 coarse ablation：
  - fine-only；
  - light coarse rerank；
  - strong coarse final-pose loss。
- 核心验收指标优先看 translation median/mean，再看 rotation；目标是先把真实初始化场景从米级稳定拉回亚米级，再追 200mm 以内。

## Assumptions

- 当前最值得投入的是 correspondence quality + depth-aware solving，而不是继续扩大 MLP pose regressor。
- coarse/fine 不需要完全删除；但 coarse 的职责要收窄，避免它污染最终精定位特征。
- 端到端训练是合理的，但必须有 teacher anchor、几何约束和 staged unfreeze，否则容易学出“定位器喜欢但不可泛化”的特征。
