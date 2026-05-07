# ICLPose 专家评审简报

更新时间：2026-05-07

## 1. 一句话判断

当前项目已经完成了较完整的 map-side feature field、query-side student feature、候选初始化缓存、以及局部位姿精化链路，但“完整、可部署、单张图像 6-DoF 重定位”还没有真正闭环。系统现状更接近：

- 一个有潜力的定位导向特征场框架；
- 一个在好初始化附近表现较强的 pose refiner；
- 一个仍然偏弱、尚未稳定进入主线的初始化前端。

因此，当前最需要专家帮助判断的不是“还能不能继续堆模块”，而是：项目接下来应当继续冲完整 localization，还是先收敛为一个更自洽的 pose refinement / localization-oriented feature field 工作。

## 2. 项目目标与当前实现方案

### 2.1 目标

目标任务是：给定单张 RGB query 图像和预建场景地图，估计相机 6-DoF 位姿。当前主场景是 OldHospital 室内数据。

### 2.2 当前主线实现

项目现有主线可分为五个模块：

| 模块 | 当前职责 | 主要目录 | 当前状态 |
|---|---|---|---|
| FeatureGaussian | 2DGS 显式几何/载体表示 | `feature_gaussian/` | 已实现，作为 map-side 几何底座 |
| FeatureField | DCFF 场景特征场渲染与训练 | `feature_field/` | 已实现，是 map-side 主体 |
| FeatureExtract | RADIO 教师特征提取、query student 训练与导出 | `feature_extract/` | 已实现，query 侧主干已具备 |
| FeatureRetrieval | 全局检索、初始位姿缓存、真实初始化评估 | `feature_retrieval/` | 已实现，但仍是当前主瓶颈 |
| PoseRefine | 稠密对应、几何求解、迭代位姿精化 | `pose_refine/` | 已实现，局部精化能力较强 |

### 2.3 端到端流程

当前逻辑上的完整流程是：

`retrieval/init -> scene feature render -> dense correspondence / feature flow -> geometric solve -> iterative refinement`

对应到实现上，大致是：

1. Map side 用 2DGS + DCFF 存储和渲染 coarse/fine feature。
2. Query side 用 RADIO 教师监督的 student 输出 coarse/fine query feature。
3. Retrieval 模块给出 query 的候选初始位姿或 init cache。
4. PoseRefine 模块用 query-map feature 做局部对应，再通过 WLS/PnP/feature-metric solver 精化位姿。

### 2.4 当前技术特征

当前实现最核心的技术点包括：

- 用 RADIO 双尺度特征作为教师监督，强调 coarse semantic 与 fine geometric 分工。
- 用 2DGS 显式载体加 HashGrid 隐式补全构建 DCFF 场景特征场。
- 用 query student 降低推理期直接跑大模型教师的成本。
- 用 RAFT/GRU 风格的稠密 flow 估计加几何求解器做 pose refinement。
- 评估上同时保留 oracle init、真实 retrieval init、LoFTR/PnP 候选与 learned init cache 等多种路径。

## 3. 已有结果与当前证据

### 3.1 目前最强结果

当前 README 汇总的代表性结果如下：

- Oracle retrieval 条件下，`pose_refine.evaluate_pipeline` 可达到约 `0.30° / 166 mm` 中位误差。
- 学习式 refinement 在较好初始化附近可达到约 `145.1 mm` 中位平移误差，基本与 teacher-query 基线持平。
- 多假设 oracle-style 融合路径曾达到约 `47.3 mm` 中位平移误差。

这组结果说明：下游 refinement 本身不是完全没有能力，系统在“初始化已进入正确 basin”时是有效的。

### 3.2 当前真实部署问题

目前真实初始化效果与 oracle 条件差距很大：

- `top1 CLS real-init` 约为 `4493.5 mm` 中位平移误差。
- `top10 oracle` 约为 `2791.7 mm` 中位平移误差。

这说明当前真正限制部署效果的主要不是局部求解器，而是 candidate scoring / selector / initialization 质量。

### 3.3 近期内部判断

从近期阶段总结与设计说明来看，项目内部已经形成了几个相对稳定的判断：

- 现有 deployable pipeline 仍然依赖外部或半外部的候选入口，典型形态仍是 NetVLAD/LoFTR/PnP 再接 local refine。
- 当前 refiner 的本质仍是“local correction”，不是能从大误差初始化直接拉回来的大 basin 全局定位器。
- 项目的 coarse feature 还没有真正成为 deployable 的 coarse stage；当前最关键的缺口仍是大 basin 初始化，而不是 refinement 末端再多堆一点复杂性。

### 3.4 稳定图与 smoke 阶段结果

近期状态文档里，较新的 smoke / stable-map 结果大致在以下水平：

- `full_wls_smoke` 达到约 `1196.9 mm` 中位平移误差。
- stable-map geomcorr 2e + colmap-id export 达到约 `1331.4 mm` 中位平移误差。

这说明 stable-map + geometry-aware supervision 方向是有信号的，但离强结论还较远。

## 4. 当前核心瓶颈

### 4.1 初始化前端明显弱于后端精化

这是目前最主要的系统瓶颈。

- 当前系统最强结果仍依赖 oracle retrieval 或外部候选。
- real-init 下误差仍在米级，明显没有进入 refiner 的有效 basin。
- 现有 PoseRefine 更像一个“局部修正器”，不是一个完整的单阶段定位器。

如果这个问题不解决，就很难把项目稳妥地表述为完整 visual localization 系统。

### 4.2 coarse 阶段尚未真正接管大 basin 搜索

项目近期已经明确意识到：coarse stage 应该负责 candidate scoring、pose bank、粗 basin 扩展；fine stage 负责 correspondence 与最终几何精修。

但当前现实是：

- deployable pipeline 仍主要依赖旧式外部候选生成；
- coarse bank 还没有稳定进入主线；
- 部分 deployable config 仍关闭了 coarse / two-stage refine 路径。

所以 coarse-to-fine 的主张在思路上是清楚的，但在可复现的主线上还没有完全落地。

### 4.3 query-map gap 大于 map-side reconstruction gap

当前已有不少证据表明：map-side feature field 本身并非完全无效，问题更大概率出在 query-to-map 对齐与泛化。

也就是说，项目当前更像是：

- map side 上限存在；
- query student 也不是完全失效；
- 但二者之间的可泛化 correspondence 还不够稳定，导致 downstream solver 吃不到 map-side 的上限。

这意味着继续单独优化 feature reconstruction cosine，边际收益可能已经不如直接围绕 query-map matcher、flow confidence 和 pose solver 做训练闭环。

### 4.4 DCFF coarse 分支的职责和实现仍不够自洽

从当前文档总结看，DCFF coarse 分支仍有明显架构问题：

- coarse 路径更多被 implicit hash 分支主导；
- 显式 latent carrier 对 coarse 分支的真实贡献较弱；
- coarse feature 在系统级上尚未稳定承担“大范围候选筛选”职责。

这带来一个风险：论文里会说 coarse/fine 分工明确，但工程上 coarse 还没成为真正的第一阶段定位器。

### 4.5 论文叙事与工程主线尚未完全收敛

目前项目存在两条尚未完全统一的叙事：

- 一条是“完整 coarse-to-fine localization”；
- 一条是“定位导向特征场 + 局部 pose refinement”。

前者要求强初始化证据；后者更符合现有最强结果结构。

如果继续两条都讲，专家和 reviewer 很容易抓住 claim 不够收敛的问题。

### 4.6 文档和代码入口存在漂移

当前仓库还有一个次级但现实的工程问题：文档与实际主线存在一定漂移。

- README 仍使用 `scene_feature_field/` 命名，但当前仓库主目录实际是 `feature_field/`。
- README 中“真实检索评估”相关描述偏旧，而当前 real-init 主入口实际上已经更靠近 `feature_retrieval.evaluate`。

这类漂移不一定影响方法本身，但会显著影响复现、汇报和对外沟通效率。

## 5. 当前最关键的策略决策

### 5.1 路线 A：继续冲完整 localization

如果坚持把项目定义为完整 visual localization 方法，那么下一阶段最关键的投入应当是：

- coarse pose bank / retrieval recall；
- 候选排序与 selector；
- coarse-to-fine 两阶段 basin 扩展；
- 内部 coarse bank 替代外部 NetVLAD/LoFTR/PnP 入口。

这条路线的难点是：需要在短期内补齐最弱的一段，而且要拿出足够说服人的 real-init 指标。

### 5.2 路线 B：先收敛为 pose refinement / localization-oriented feature field

如果选择收敛论文本体，那么更自然的定义是：

- 给定初始位姿或固定 init 协议；
- 研究 map/query 特征如何服务局部 pose refinement；
- 强调 init-to-final 的 gain、basin of convergence、solver robustness 与 query-map correspondence。

这条路线与当前已有证据更匹配，也更接近 GS-SMC 一类工作的评价方式。

从当前仓库状态看，我个人更倾向把它作为近期更可行的论文收敛方向。

## 6. 建议请专家重点判断的问题

建议把下面几个问题明确抛给专家：

1. **论文范围应收敛到哪里？**
   当前结果更适合定义为完整 localization，还是更适合定义为 pose refinement / localization-oriented feature field？

2. **如果暂时不把初始化作为主贡献，学术上是否成立？**
   也就是采用固定 init protocol，对比不同特征场与 refinement 设计，只讨论 init-to-final gain，是否足够形成一篇自洽工作？

3. **最值得投入的下一步是 coarse init，还是 query-map matcher？**
   如果资源有限，应该优先补 coarse pose bank / retrieval，还是优先把 query-map local matcher + WLS 闭环做扎实？

4. **当前 coarse/fine 的职责定义是否合理？**
   coarse 只负责候选与大 basin，fine 负责 correspondence 与精修，这样的职责拆分是否足够清晰、是否符合当前领域经验？

5. **论文最需要的关键实验是什么？**
   是 real-init recall，还是 fixed-init pose gain 曲线，还是 basin of convergence / noise bucket / ablation report？

6. **是否应当继续 end-to-end 解冻 map side？**
   还是应先固定 map 侧，把 query-map matcher 和 pose solver 跑通，再考虑低学习率解冻 fine decoder / latent？

## 7. 当前建议的对外表述

如果需要在当前阶段向专家快速解释项目，我建议用下面这段话：

> 我们已经完成了一个以 RADIO 教师监督为基础、结合 2DGS + DCFF 场景特征场、query student 和局部 pose refinement 的视觉定位框架。项目当前最强的能力在于：给定较好的初始化时，局部 refinement 能显著改善位姿；但项目最弱的一段仍是大范围初始化与候选选择。因此我们现在需要判断：下一步应继续把 coarse retrieval / pose bank 补成完整 localization，还是先把工作收敛成一个更自洽的 pose refinement / localization-oriented feature field 方向。

## 8. 附：当前主入口与沟通注意事项

- map-side feature field 当前以 `feature_field/` 为主，而不是 README 中的 `scene_feature_field/` 命名。
- real-init 评估当前更接近 `feature_retrieval.evaluate` 这条主线。
- PoseRefine 当前更适合作为 local refiner 来理解，而不是独立完成大 basin 重定位的模块。
- 如果要做正式汇报，建议固定一版 mainline config、checkpoint 和结果表，避免旧 LoFTR/rematch、PCA64 路线与新 adaptive/DCFF 主线混在一起。