# Cambridge：用户提供高斯先验下的完整实验（v418，五场景已完成）

本轮按用户要求使用 `/root/matcha_prior` 的五个 PLY，并保留旧 v414 全部已接受的推理祖先。v415 的简化流程不作为本轮方法或结果。构建完成、训练完成和完整定位评估分别记录，不能互相替代。

## 数据与实验约束

| 场景 | 官方 train | 官方 test |
|---|---:|---:|
| GreatCourt | 1532 | 760 |
| KingsCollege | 1220 | 343 |
| OldHospital | 895 | 182 |
| ShopFacade | 231 | 103 |
| StMarysChurch | 1487 | 530 |

GreatCourt 的 `seq5/frame00297.png` 缺少原生相机绑定，校准建图清单明确排除此一张训练图；不排除测试图。五场景共 1918 张官方 test，两个种子都必须覆盖完整清单。报告平均/中位数平移、旋转误差，以及 0.1m/1°、0.25m/2°、0.5m/5° 等联合召回率。无效位姿保留无穷误差，不从分母删除。

用户提供的 PLY 与旧 StMary 地图不同，不能套用旧 PLY 的 source-index 清理掩码，也不能把旧图结果当作新先验结果。输入 SHA256 固定在各场景 `full_map/input_contract.json`。用户已明确确认这五份 PLY 均仅由各场景官方 train 图像重建，声明及文件哈希见 `user_prior_provenance.json`；本轮未另行执行上游重建审计。

ShopFacade 官方 train 只有 seq2，不能伪称跨训练轨迹验证。使用预先固定的同轨迹图像分区（180 fit、5 间隔排除、46 validation），明确记录图像不重叠但轨迹不独立。其他场景采用训练轨迹划分。划分不依据 test 定位结果。

## 保留的完整流程

1. 高斯表面、区域邻接、初始表面地图单元；120 epoch 表面映射器；最终区域与物理表面层级。
2. 全训练视图贡献缓存、渲染平面、融合平面及可见性 atlas、平面专属观测库。
3. 每场景重新拟合 64D RADIO 投影、点坐标头、表面坐标头及坐标收缩校准；保留原映射验证门槛。
4. 重新提取与该投影绑定的 1024/1536 RADIO 读出，拟合 fine 校准和可靠性；构建原 source-mode 区域记忆与 native-context 区域记忆。
5. 原 wide8/global-token 分支及 risk9/native-context 分支；fine rank32、几何渲染、MoGe、共识与多候选保留（至 v309）。
6. v313/v319 结构与深度验证，v326/v327 救援，v344/v357 特征精修。
7. v402 LoD 候选（两种子）、v404 相对验证、v405 保留证据、v409 学习精度选择、v411 联合邻域、v412 多模态、v413 双支持、v414 双精修与多尺度一致性。

通用风险模型、LoD 匹配器、相对验证器、局部精度选择器保持原 StMary 训练权重；每场景的地图、投影、坐标头和 fine 校准重新构建。通用模型来源由 `runtime_base/frozen_generic_models.json` 明示。因此这是完整推理方法的跨场景实验，不是“全部模型都在各自场景重新训练”的实验。

## 已有一致性证据

- v417 旧完整链连接重放，原 seq10 与 seq13 四分片的全部检查数组与历史输出一致。
- v418 `original_memory_gate/equivalence.json`：旧输入重建 source-mode 拓扑、区域库、native-context 缓存与 native-context 库的全部共有数组精确一致。
- 邻接向量化保持原几何筛选规则；合成边界用例和真实先验局部邻域与标量实现一致，见 `adjacency_equivalence.json`。不声称整个 ShopFacade 图已与完整标量结果比对（耗时标量构建在完成前终止）。
- 原生相机内参输出在五场景与旧读取器一致。旧读取器获取内参时会顺带解码 `images.bin` 外参，然后丢弃；早期 StMary/Shop 建图进程使用了此读取器。已改为跳过外参字段，只读相机绑定。`camera_lookup_audit.json` 明确纠正早期 `test_pose_values_read=False` 字段过宽的表述；实际未将这些测试外参用于建图或定位。不能把它改写成“任何进程从未解码测试外参”。
- 全 1918 张查询的既有 MoGe 与查询平面缓存清单已检查：原 official-test level5/FP16、256×144；查询平面无位姿输入。新版五列平面清单按原四列接口导出，保留父清单哈希。

## 已完成但属于旧地图的 StMary 官方 test 结果

v417 的全部 530 张已完成原完整链。种子1：平移 mean/median 0.183820/0.112914m，旋转 mean/median 0.715305/0.391650°；R10=43.2075%，R25=91.1321%，R50=97.3585%。种子2：R10=43.2075%，R25=91.5094%，R50=97.3585%。无无效位姿。

证据：`output/cambridge_full_v417/StMarysChurch/full_test_evaluation/metrics.json`。这些不是新 PLY 的结果，也不是五场景结果。历史 seq13 曾多轮检查，不能宣传为从未触碰的盲测。

## 执行入口与完成判据

- `build_cambridge_complete_scene_map.py` → `full_map/MAP_COMPLETE.json`
- `build_cambridge_complete_atlas.py` → `full_atlas/ATLAS_COMPLETE.json`
- `prepare_cambridge_complete_runtime.py` → `runtime_base/RUNTIME_ASSETS_COMPLETE.json`
- `prepare_cambridge_complete_queries.py` → `runtime_base/READY.json`
- `run_cambridge_complete_scene.py` → 每 batch 的 frontend/backend `COMPLETE.json`
- `evaluate_cambridge_complete_scene.py` → `full_test_evaluation/endpoint_seal.json` 后再读取标签，输出 `metrics.json`

只有最后一项完整出现且覆盖官方清单，才算该场景完整实验结束。各级日志位于 `output/cambridge_full_v418`；当前应以实际完成标记和日志为准，不以本文推断五场景已完成。

## 实际接线检查与中间验证补充

ShopFacade 两个坐标头均通过原验证门槛：点坐标头相对 token 中心的保留集像素误差中位数改善 8.02%；表面头的 chart-UV 中位误差改善 6.21%。fine 校准的保留集像素误差中位数由 1.5200 降至 1.3755。这些是不同监督样本上的中间检查，不是最终定位误差，也不能将改善百分比相加。

查询准备首次运行时，两处格式检查主动拒绝了输入：分批平面清单缺少 `selected_names_in_order`；准备阶段的矩阵形式相机缓存不满足旧消费者的模型/尺寸/参数数组接口。均通过补全格式修复，没有修改几何或算法阈值。相机适配器以原生内参清单生成原 camera-only 格式；未使用 contributor 档案，其对应哈希字段显式留空并说明来源，不伪造 contributor 哈希。五场景全部 1918 张 test 的最终工作网格 K 和 radial k1 与已有相机缓存逐项完全一致。实际适配输出在 `camera_schema_gate`。

`training_pose_frame_audit.json` 只解码官方 train 名称对应的原生 COLMAP 位姿，再与官方 train 文本比较：各场景平移最大差异小于 0.000001m，旋转最大差异约 0.00011°，符合文本舍入差异；GreatCourt 仍仅缺前述一张训练图。此检查证明这两种训练位姿来源一致，不替代用户 PLY 上游重建来源审计。

## 用户确认的先验来源

用户已明确回复：五份高斯先验均“仅使用官方 train 图像”。该来源声明及五个 PLY 的 SHA256 存于 `output/cambridge_full_v418/user_prior_provenance.json`。此前关于先验来源待确认的说明由此补齐；这是用户明确确认，不冒充独立重建审计。StMary 历史查询曾反复检查的限制仍然成立。

## 后端接线与复现归档补充

新场景后端必须显式链接本次前端生成的目录，不能依赖旧 StMary 基座中碰巧已有同名目录；已修复这一目录发现逻辑。另补齐原 v405 验证器的 `decision_risk_v405_expanded/calibration.json`，使用原文件、原阈值。完整回放相关 3 项测试通过。修复及来源记录归档于 `reproducibility/source_corrections_003.tar.gz`，与 001 基础源码、002 格式修复及原通用资产包共同复现当前执行版本。

## 最终完成状态

五场景、1918 张官方 test、两个求解种子均已完成。GreatCourt 使用已记录的训练校准约束修订，其余四场景沿用原校准。完整指标见 `output/cambridge_full_v418/complete_suite_metrics.md`；完成标记 `SUITE_COMPLETE.json` 及五场景冻结端点哈希已逐项复核。最终结果不支持各场景均有高成功率的结论。
