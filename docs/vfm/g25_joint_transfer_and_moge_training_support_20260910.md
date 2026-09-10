# G25：联合坐标跨路线验证与真实 MoGe 训练支撑审计

本轮保持匿名物理表面地图、RADIO 召回/局部匹配和几何后端不变；没有新 backbone、源图像检索或逐 query GT 选择。默认配置不变。

## 1. 联合坐标头：350 张验证不支持升级

固定上一轮训练完成的局部相关头 v157 与联合坐标头 v201，不再扫损失权重。四个 seq13 shard 分别重新生成两组坐标，共 350 张。两组使用完全相同的正确性修复后主分支 MoGe 初值、isotropic 后端；names、offsets、token、区域/平面/atlas 行、RADIO 匹配分数、内参和畸变均逐数组验证一致。

注意：这不是双分支共识最终系统；也不和上一轮 seq10 使用的历史固定初值混合汇总。seq13 是已有开发路线，不是新盲测路线。

| 350 张固定初值 | 0.1m/1° | 0.25m/2° | 0.5m/5° | 1m/10° | 2m/45° |
|---|---:|---:|---:|---:|---:|
| 原局部相关头 | 69 | 246 | 310 | 321 | 323 |
| 联合坐标头 | 66 | 245 | 311 | 321 | 323 |

严格门限 3 改善、6 退步，p=0.508；0.25m 为 4 改善、5 退步。平移中位数 0.173641→0.174501m；153/350 张平移改善。平均平移变化 -3.198mm，但 10 帧分块区间 [-8.586,+0.871]mm，跨零。

逐 shard 严格命中：17→17、16→17、19→15、17→17。不能因为平均值略降就忽略严格门限损失，也不能仅凭一个 shard 判定所有联合监督无效。

结论：上一轮 seq10 的 +4 严格命中没有在本次条件对照复现；当前权重/实现版本不升级。这个结果否定的是当前候选头的稳定收益，不是联合坐标几何原则。优先补训练输入分布，不继续在这些 query 上扫权重。

证据目录：`/root/ICLPose/output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1/joint_transfer_v210_shard{0,1,2,3}`。每个目录包含命令、日志、两臂冻结对应和位姿、control.json 与 paired.json。总表 `joint_transfer_v210_all350.json`。新增入口 `replay_goal_maplet_joint_coordinate_transfer.py` 与聚合审计 `audit_goal_maplet_joint_coordinate_transfer.py`。

## 2. 缓存来源校验：修复真实风险

`build_goal_maplet_query_plane_regions._work` 复用既有输出时，原先只检查 carrier 配置，未核对当前 MoGe 源文件以及基础区域来源，可能静默复用不同输入生成的旧区域。

现在复用前要求 MoGe 文件哈希一致、`uses_pose_or_ground_truth` 明确为 false、基础区域文件哈希一致；不匹配立即失败。新增 source/pose/base 三类拒绝及正确来源复用测试。未发现或宣称历史正式缓存已被污染；这是可触发的实现缺口修复，未重写旧缓存。

## 3. 真实 MoGe 区域与旧 mapping 监督不等价

利用既有 seq9 MoGe3 点图构造 98 张真实 query 区域，共 3095 个；运行的是现有默认有限平面提取器，无 query 位姿输入、无隐藏像素补造。

新目录：`/root/ICLPose/output/g25_pose_transport/planar_query_geometry/moge3_seq9_regions_v211`。

按运行时区域 token 支撑规则审计，以 (图像,token) 为单位，跨 region/多平面观测去重：

| 人口 | 数量 |
|---|---:|
| 真实 MoGe 区域 token | 134805 |
| 旧 mapping bank 可监督 token | 52241 |
| 两者交集 | 45539 |
| 真实区域有、bank 无监督 | 89266 |
| bank 有、真实区域外 | 6702 |

可监督覆盖率仅 **33.78%**；3095 个区域中 **1479 个不足 4 个可监督 token**。这不是地图定位覆盖率，也不是 MoGe 正确率；它衡量的是现有训练 bank 能覆盖真实区域输入的程度。

结论：只将旧 bank 换一个区域标签，仍不等于完整部署候选训练。缺失监督的 token 不能作为负例，不能从候选生成阶段删除后宣称分布已经对齐。下一步需要完整 RADIO token 的真实区域召回/匹配，以及单独的 mapping 可见性/几何监督 mask。

权威报告：`.../surface_coordinate_upgrade_v1/moge_mapping_support_v212.json`。开发中 v211 支撑报告曾因错误去掉文件名 `.npz` 后缀而出现零关联，已定位并更正；新增无 mapping 图像关联时 fail-fast，v211 计数作废，不是模型结果。

## 4. 真区域受限候选对照

已增加实际 MoGe 区域模式：源路线完全从 map prototype 中排除，区域由 MoGe 给出，token 固定取 mapping bank 可监督子集；跨全部平面召回后运行相同 MNN/homography 原语，先冻结候选再产生几何标签。

同一 token 在 mapping 多平面观测中有多个监督位置时，不按候选距离选择最有利目标；明确冲突的目标统一标为 ambiguous。这个实验是数据对齐中间步骤，仍不是完整部署候选训练。区域描述子也来自可监督子集，存在选择偏差。

v213 为初次候选归档；v214 增加冲突目标 ambiguity mask，用于后续关联探针。未使用任何 query-test GT 训练。

v214 实际保留 1616 个有足够监督的 MoGe 区域。homography 前 76541 行，后 50867 行；后者为 5417 正例、13811 负例、31639 模糊例，含 43786 跨平面行。50 个冲突监督候选行被统一 mask；v213/v214 冻结候选数组及文件哈希相同，仅监督标签更保守。

固定五参数 logistic 探针、49 图 fit/49 图 eval，9592/9636 非模糊样本，不扫超参数。相对余弦：AUC **0.94686→0.94424**；相对余弦单特征 logistic 标定，Brier **0.10417→0.08498**，BCE **0.34466→0.28649**。结论：条件标定改善，排序没有改善；不进行运行时硬过滤、不宣称位姿增益，也不将该概率解释为缺失/模糊 token 的整体正确率。

证据：`.../surface_coordinate_upgrade_v1/moge_crossplane_candidates_v214.{npz,json}`、对应 `.labels.npz`、`moge_association_probe_v215.json`。关联探针报告现在绑定候选报告哈希并明确区域来源，避免把不同输入分布混为一个实验。

## 5. 当前行动结论

1. 保留当前默认局部相关头与正确性基线；上一轮 LM 保护仍仅是可选候选，不能把本次条件实验指标混入全系统成绩。
2. 暂不升级联合坐标头，不用逐 query 指标决定用哪一个头。
3. 优先补完整真实区域候选与监督 mask 的分离，而非增加联合损失复杂度。
4. 缓存来源校验修复保留；原有脏工作树、历史产物均保留，未 commit/push。

相关 `goal_maplet` 测试 **1108 passed, 1 warning**；这不是全仓测试通过，此前缺少 pytorch3d 的全仓收集边界仍保留。
最后重点复核 **8 passed**，`git diff --check` 通过。
