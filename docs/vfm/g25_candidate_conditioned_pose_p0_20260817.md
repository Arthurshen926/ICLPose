# G25 candidate-conditioned pose：本轮 P0 落地与实测

## 决策

冻结 RADIO 物理区域召回前端。召回概率只定义 `q_ret`，不再被解释为
位姿测量。新增的 reference seam 定义独立的 `q_pose(. | basin)`：低维 pose
code、法向、相对深度和边界一致性只能把检索先验质量转移到同 child 的匹配
质量；任何没有被转移的质量都进入 explicit unmatched。该 seam 不含关键点、
硬对应、PnP 或绝对位姿回归。

当前 `q_pose` 核心刻意标记 `production_eligible=false`。本轮已补齐 32D query
readout 架构、低秩匿名视角 map-code 接口、稀疏 candidate transport reference、
leave-one-view-out 统计和能量地形训练损失；但它们尚未在真实 mapping train/dev
上产出并训练成冻结 artifact，也尚未通过独立 validation capture gate。数学接口
和合成梯度通过不等于模型已训练，更不等于真实位姿性能已经提高。

## 已完成的 P0

1. primitive compositing 后直接按物理 parent membership segment-sum；parent
   Top-L/tail 不再由 child Top-L 反推。重叠 membership 先在每个 primitive 内
   归一化，因此不会重复制造 alpha。
2. resident profile 拆成 projection/tile/raster、D2H、depth sort/composite、
   raw-token gather、child identity、feature、typed finalize、direct parent。
3. `_batch_token_remap` 对 global pixel hit 单调性和三数组 shape 硬断言。
4. pattern search 改为标准 poll：任意方向成功时移动但不缩其他轴；整个 poll
   失败才缩 active axes。增加耦合地形与多 basin 状态隔离反例。
5. atlas 新增独立预算的 wide-location 与 near-view nested queues；near quota
   不能被 wide 消耗，最终输出 location ID 和连续 2m/45deg 搜索域。
6. 新增 `candidate_conditioned_pose_attribution.py` reference kernel。固定分母
   保证 map support 消失、invalid 或 compatibility 下降只能减少 matched mass。
7. reference kernel 升级为 source-substochastic sparse transport。source state
   为 `(query token, retrieval child)`，target state 为 `(rendered token, map
   child)`；边只允许 exact child、connected support、physical adjacency 或同
   parent，并有独立 unmatched sink。每个 source 严格满足
   `matched + unmatched = q_ret`。
8. coarse/medium/fine 三阶段分别冻结不同 token radius、允许的物理关系和 depth
   schema。旧 exact same-token/same-child 核保留为 fine reference，不再承担
   2m/45deg capture。
9. 法向统一为 camera frame；double-sided 使用 unsigned compatibility；pose code、
   normal、depth、boundary 各有独立 validity/confidence；零向量自动 invalid；每
   token child ID 必须唯一。parent/support/adjacency hierarchy 从数组重算内容
   哈希，并拒绝重复、自环和非对称 physical-adjacency。
10. compatibility 改为相对 unknown floor 的非负加性 log-energy。任何模态失效
    都只会删除非负证据，不能通过删除一个负余弦项提高分数。最终仍使用固定
    query-only reliability 和固定分母。
11. 新增 `pose_transport_training.py`：continuous joint-error soft listwise、
    adaptive pairwise margin、随机/插值路径 monotonic loss、已知 optimizer drift
    negative、带 sink 的 attribution KL，以及 additive sufficient statistics 的
    leave-one-view-out 减法。
12. 新增 `trainable_pose_transport.py`：RADIO 1280D + ray grid 的低容量
    `1x1 + two 3x3` query head，输出32D pose code、camera normal、relative depth、
    boundary 和4类 confidence；map code 使用
    `normalize(mu + B phi(view,scale))`，rank 默认4。模型内容哈希覆盖配置和全部
    tensor；artifact 明确 `uses_pnp=false`、`uses_absolute_pose_regression=false`。

当前物理图尚未拥有与 parent 不同的显式 connected-support artifact。因此本轮
没有把 parent 数组改名伪装成 support。下一轮应从 primitive adjacency/overlap
建立稳定 support ID，再与 parent/child 一次 raster 独立归约。

## 真实 resident 计时

审计：
`output/g25_pose_handoff/resident_renderer_v2_direct_parent_timing/seq11_frame00001_pose3.json`

3 个真实 pose、约 50.9 万 primitive、64x36 token、4x coordinate supersampling：

| 阶段 | 正序时间 |
|---|---:|
| projection/tile/packed raster | 0.036 s |
| device to host | 0.010 s |
| CPU depth sort/composite | 1.723 s |
| raw-token gather | 0.094 s |
| child identity reduction | 3.155 s |
| feature reduction | 3.494 s |
| typed finalize | 1.257 s |
| direct parent reduction | 1.789 s |
| resident total | 11.609 s |

scalar 三 pose 总计 18.093 s，resident speedup 1.559x。child、parent、feature
及正逆 batch 的误差最大为 `5.96e-8`，parent/child rows exact；等价门通过，
速度门不通过。

结论比原 profile 更明确：GPU raster 本身只占约 0.3%，D2H 也不是主瓶颈；
CPU sort/composite 约 15%，四类 CPU reduction 合计约 83%。所以下一步必须
实现 packed hit 到 parent/support/child/feature/score 的 GPU fused path，而不是
只优化 projection 或只搬 child segment-sum。

### GPU compositor v3 实测

新审计：
`output/g25_pose_handoff/resident_renderer_v3_gpu_compositor/seq11_frame00001_pose3.json`

本轮已把410万 raw packed hits 的稳定 `(pixel,depth,primitive)` 排序、exclusive
transmittance 和 `T_after<=1e-4` inclusive stop 留在 GPU。只把 composited
survivors 回传 CPU。相同三 pose 的结果：

| 阶段 | v2 CPU compositor | v3 GPU compositor |
|---|---:|---:|
| projection/tile/raster | 0.036 s | 0.036 s |
| sort + composite | 1.723 s | 0.010 s |
| D2H | 0.010 s | 0.008 s |
| CPU post-composite reduction | 9.741 s | 8.599 s |
| resident total | 11.609 s | 8.755 s |

相对本轮 scalar 三 pose 15.001 s，resident batch 为1.713x；相对旧 resident 总时
间下降24.6%。child/parent rows完全相同，所有 numeric field 最大误差
`1.1921e-7 < 2e-6`，正逆 batch 稳定，等价门通过。生产速度门仍不通过，因为
CPU token remap、child/support/parent/feature/typed reduction 占8.599 s。下一步
必须搬整段，不能把本轮标成“GPU reducer完成”。

## 本轮验证状态

组合定向回归共67项通过，覆盖 renderer、GPU/NumPy compositor 等价、parent
direct reduction、soft pose energy、pattern search、atlas、稀疏 transport、LOO、
训练损失和32D readout。关键新增反例包括：

* target Top-L 与 query Top-L 不同；
* duplicate child、错误 normal frame、错误 depth schema；
* 四种 modality 任一消失不得提高 score；
* 负法向变 invalid 不得因删除负项获益；
* zero vector 不得成为“中性有效证据”；
* drift trajectory 中 score 提高而 pose error 恶化时，梯度必须压低错误终点；
* held view 是唯一观测时，LOO 输出 invalid，不能回填自己的 feature；
* 模型 tensor 篡改后，内容哈希重开必须失败。

本轮没有训练真实模型，也没有重新打开530-query test。因此科学结论是：P0/P1
的可训练接口已经闭合，GPU compositor 的数值与速度方向成立；真实 pose capture
改进仍为待验证，当前 V3 objective 的 KILL 判定不变。

## 下一实验的硬顺序

1. GPU fused reference-equivalent reducer；先做 Top8 batch，再扩 coarse64。
2. 建 32--64D pose field 的离线监督样本：GT pose、正负 SE(3) 扰动、soft
   parent/support/child、normal/depth/boundary；训练目标首先是 pose ranking。
3. 分别从 0.5m/5deg、1m/10deg、2m/20deg、2m/45deg 做 oracle basin
   capture。GT-visible/full-map 都不收敛则否决 pose field；oracle 收敛而当前
   attribution 不收敛才继续改 query readout。
4. 只有 P2/P3 通过后才跑 530-query 完整级联；不再用完整实验掩盖局部能量
   地形尚未成立的问题。

## 530-query wide/near 预注册诊断

按附件建议直接测试 `wide=48, near=16`，没有根据 test GT 调预算：

`output/g25_pose_handoff/visibility_atlas_child_layout4x4_joint_v3/nested_wide48_near16_top64_v3.json`

| queue | 1m/10deg | 2m/20deg | 2m/45deg |
|---|---:|---:|---:|
| wide 48 | 12.08% | 62.45% | 74.72% |
| near 16 | 17.92% | 58.68% | 60.19% |
| union | 18.11% | 67.74% | 79.06% |

它验证了分析中的 trade-off：near quota 相对 wide-only 恢复 `+6.03` 个百分点
的 1m/10deg acquisition，但 48 个 wide 名额不足以保持原 wide64 的 83.40%
2m/45deg；union 也只有 79.06%。因此双队列接口正确，但这个 48/16 配置不能
替代现有 wide64。后续预算只能在 train/dev 上冻结，不能查看 530 test 后继续
调成 56/8 或 64/16。当前合理做法是把 wide64 当召回基线，near queue 作为
额外受保护的 refinement seeds；若部署硬预算必须仍为64，则需由新的 q_pose
在渲染前压缩 location 内 orientations，而不是按 GT 调静态比例。

## Full-map oracle capture 与实际 pattern search

固定 query `seq13/frame00001.png`，完整 50.9 万 primitive，使用当前 retrieval
posterior 与 hierarchical V3 energy。GT 只用于构造诊断轨迹。

方向审计：
`output/g25_pose_handoff/pose_capture_direction_v1/seq13_frame00001_fullmap.json`

在两条预冻结耦合 SE(3) 方向及四档误差
`0.5m/5deg, 1m/10deg, 2m/20deg, 2m/45deg` 上，8/8 都满足
`score(start) < score(half) < score(GT)`。但最困难的 2m/45deg 方向从 start
到 halfway 只增加 `0.001983`；score 已接近 unknown floor，说明方向存在但搜索
信号很弱。这是必要条件通过，不是 capture 成功。

实际搜索：
`output/g25_pose_handoff/pattern_search_capture_v1/seq13_frame00001_2m45_two_basins_sweep3.json`

从两条 2m/45deg 起点运行修正后的六轴 pattern search，3 sweeps、74 pose、
254.10 秒：

| basin | initial | final | 1m/10deg capture | 判定 |
|---|---|---|---:|---|
| 0 | 1.949m / 45deg | 2.780m / 24.50deg | 否 | score 上升但 joint error 1.00→1.39 |
| 1 | 1.954m / 45deg | 0.996m / 18.33deg | 否 | joint error 1.00→0.498，但旋转未捕获 |

两者都有3次 accepted update，说明失败不是搜索没有移动。一个 basin 出现明确
objective drift，另一个只完成宽松改善；0/2 进入1m/10deg，0/2进入0.5m/5deg。

因此当前 V3 可以作为 coarse survivor/control energy，但不能承担最终位姿测量。
单纯加入更多 sweeps、缩小 trust radius 或把相同能量搬到 GPU，只会更快优化一个
与真实 SE(3) error 不完全一致的目标。新 q_pose 的 pose-rank loss、relative depth、
normal/boundary 以及联合平移--旋转方向是必要改动。另一方面，254 秒只评估两个
basin也证明 GPU fused reducer 必须与 pose-field 训练并行推进。

## 2026-08-19：map-disjoint sparse transport 与 full-token pose energy

本轮严格回归主线：冻结 3DGS 物理地图、RADIO 全局召回、直接多 basin 位姿测量。
没有 ALIKE、特征点对应、PnP、五折或高斯重训。数据是 `seq11` 单路线
map-disjoint 诊断；地图 contributor 排除 `seq11`，但现有 surface mapper 是
full-train 产物，因此仍不是最终未见测试集结论。

### 稀疏 child transport：真实训练否决

构建了 11 query × 8 natural candidates 的可重放数据集
`output/g25_pose_transport/seq11_map_disjoint_real8_v1/dataset.npz`，每 query 约
630--780 万条稀疏 edge。先修复 confidence square-root 在零点导数奇异导致的
NaN，并加入 loss、gradient、parameter 全链 finite gate。两种 30-epoch 训练均失败：

* minimal learned readout：留出 Spearman `-0.086`，pairwise `45.0%`，strict `0%`；
* frozen surface mapper shared projection：Spearman `-0.114`，pairwise `43.6%`，
  strict `0%`。

当前 sparse transport 丢失了决定 pose 的完整 token layout，不得作为主评分器。

### full-token × parent：受控 basin 测量通过

新增固定分母能量
`fixed_denominator_same_token_parent_gated_surface_radio_energy_v1`：每个 36×64
query token 同时要求冻结 RADIO surface appearance 与 pose-free physical parent
posterior 一致；任一通道 evidence/mass/validity 消失都不能提高分数。

11 query × 19 个对称 GT-relative 扰动上，静态 ranking Spearman `0.678`、
pairwise `82.1%`、selected strict `100%`。后 5 query 从固定约 `1m/10deg` 起点
做 6-sweep 搜索：strict/loose 均 `5/5`，中位终点 `0.413m/约0deg`，objective
drift `0/5`，365 poses、render `993.53s`。证据：
`output/g25_pose_transport/seq11_controlled_local19_v1/pattern_search_dev5_parent_product_sweep6_summary_v2.json`。
这证明局部 basin signal 存在，但 GT-relative 初始化使它只能称 oracle diagnostic。

### natural proposal：无保护 refinement 否决

同一后端从 retrieval 真实首候选开始，scorer 不读 GT。后 5 query 起点
strict/loose 为 `20%/80%`，搜索后为 `0%/40%`；tier improve `0%`、degrade
`60%`，objective drift `80%`，中位终点 `1.022m/1.85deg`。例如
`frame00010` 从 `0.625m/0.97deg` 漂到 `1.187m/1.85deg`，同时 score 上升。
证据：
`output/g25_pose_transport/seq11_map_disjoint_real8_v1/pattern_search_dev5_natural_parent_product_sweep6_summary_v2.json`。

进一步加入非负 contrastive hinge，避免 cosine=0 的无关表面仍产生正质量。它在
前 6 query 可把 Spearman/pairwise 提到 `0.508/70.8%`，但后 5 query 最好仅
`0.152/54.3%`，strict selection 仍 `20%`；阈值没有泛化，不继续后验调参。

### 当前边界与下一步

冻结保留：RADIO child/parent 全局召回、完整 token layout、固定分母 missingness、
physical parent soft gate、原始多 basin proposal。否决：当前 sparse learned
transport、full-token energy 全局 reranking、无保护连续优化。

下一后端必须是 candidate-conditioned 几何测量，而非再改外观权重：

1. 永久保留原始 basin；refined pose 只能新增 hypothesis，不能覆盖 seed。
2. 同一 render 加入 relative depth、camera-frame normal、boundary 和 occlusion
   transition；用受控 SE(3) 正负扰动训练 pose-rank/curvature，不做 absolute pose
   regression。
3. 先过 natural proposal 留出门：strict/loose 不下降、drift 接近零且部分 query
   真正升档；否则不进入 530-query test。
4. GPU 化只解决速度，不改变科学 gate；natural 5 query 已耗 `1110.81s`，最终
   需 fused token reducer 才能把 8--16 survivor refinement 压到部署量级。

本轮最终回归：81 tests passed；唯一 warning 是 PyTorch `scatter_reduce` beta API。

## 2026-08-19：完整 token phase 与 protected multiseed handoff

parent/child identity 能量仍主要回答“看见哪个区域”，对区域内部的连续位姿不够
敏感。新 phase 能量直接比较 query 与 candidate-rendered RADIO surface field 的
水平/垂直局部差分。每条可观测边使用非负 `(cos(delta_q,delta_r)+1)` evidence，
以完整网格边数作固定分母；水平和垂直分数取最小值，禁止单轴补偿另一轴。它不构造
点对应，也不解 PnP。

自然 Top8 静态评估的后 5 query：Spearman `0.852`、pairwise `86.4%`、GT anchor
Top1 `5/5`；selected strict 从 `20%` 提高到 `40%`，loose 从 `80%` 提高到
`100%`。这已经达到该 Top8 池自身可实现的 strict 上限。证据：
`output/g25_pose_transport/seq11_map_disjoint_real8_v1/fulltoken_conservative_phase.json`。

连续搜索分别验证了两种无 GT seed：

* retrieval 首候选：strict `20%→40%`，loose 保持 `80%`，20% query 升档、0%
  降档；其中 `frame00008` 从 `0.574m/0.99deg` 到 `0.304m/0.99deg`；
* phase Top1：初始 strict/loose `40%/100%`，搜索后仍为 `40%/100%`；说明
  reranking 有效，但单一 Top1 refinement 没有额外总体增益，且 basin 内仍可能
  score 上升而误差小幅变坏。

因此最终 handoff 不做单一 Top1。保留 retrieval 首 seed、phase Top1 seed 以及各自
refined hypothesis，形成 protected set-valued 输出。本 5-query 诊断的 hypothesis
union acquisition 为 strict `60%`、loose `100%`；refinement 永远不能删除输入
basin。证据：
`output/g25_pose_transport/seq11_map_disjoint_real8_v1/pattern_search_dev5_protected_multiseed_phase_summary.json`。

这仍只是 acquisition，不是 Top1/localization success：旧搜索分片没有保存 pose
matrix，当前 summary 无法重放 0.5m/5deg physical NMS；新 evaluator 已补保存
initial/final `pose_w2c`，下一批实验必须做真实物理去重。后续应扩展到 phase Top-K
独立 basins，每个 basin 只做小 trust-region refinement，再由独立校准器输出 pose
或 structural null。当前速度约每 query 两 seed 6 sweeps 数百秒，仍需 fused GPU
token reducer；但速度优化不能先于自然候选上的 set recall/去重门。

更新后的组合回归：83 tests passed；唯一 warning 仍是 PyTorch
`scatter_reduce` beta API。

### 两独立 basin 的 pose-bound 实际产物

随后对 phase Top8 做真实 `0.5m/5deg` SE(3) NMS。单纯增加静态 TopK 只能保持
strict `40%`、loose `100%`，说明收益不能来自重复堆候选。首 retrieval seed 与
phase Top1 在 3/5 query 上属于不同 physical basin；另外 2/5 则选择相对首 seed
最高分的独立 phase basin。

只补跑真正缺失的分支后：

* `frame00010` 独立 basin 从 `0.611m/1.21deg` refine 到
  `0.336m/0.81deg`，新增 strict capture；
* `frame00009` 从 `0.854m/1.58deg` 到 `0.610m/1.50deg`，保持 loose；
* `frame00008` 的首 seed pose-bound 重放再次得到 `0.304m/0.99deg` strict。

最终 builder 打开实际 initial/final `pose_w2c`，验证 refinement 的 initial pose 与
dataset seed 一致，按 phase score 排序后执行真实 physical NMS。输出每 query 平均
恰好 2 个独立 basin，strict acquisition `80%`、loose `100%`：
`output/g25_pose_transport/seq11_map_disjoint_real8_v1/protected_phase_pose_set_dev5_v1.json`。
这不再是只根据误差表做的 union，而是可重开的 set-valued pose artifact。

当前 GO 边界是 map-disjoint dev5 的 pose-set acquisition；仍不主张 Top1。下一步
需要在更大、冻结的数据上验证 Top2 physical-basin recall，并为集合内最终
pose/null 训练独立校准器。组合回归更新为85项通过。

## 2026-08-23：seq13 地图外 16-query 直接位姿后端复核

本轮没有 ALIKE、PnP、点对应、五折或高斯重建。地图、canonical field 与
surface mapper 都保持冻结；查询取标准 test 的 `seq13` 16 帧，故 query route
不属于十条 map-training routes。新的 pose-free pool 直接从冻结的纯 RADIO
posterior 与 child visibility atlas 重算，输入 run 的
`uses_alike/uses_pnp/uses_query_pose/uses_query_ground_truth` 均严格为 false；旧
candidate score 不进入后端。产物：

- `output/g25_pose_transport/seq13_map_disjoint_radio_atlas16_v1/candidate_pool.json`
- `output/g25_pose_transport/seq13_map_disjoint_radio_atlas16_v1/dataset.npz`
- `output/g25_pose_transport/seq13_map_disjoint_radio_atlas16_v1/fulltoken_conservative_phase.json`
- `output/g25_pose_transport/seq13_map_disjoint_radio_atlas16_v1/fulltoken_conservative_phase_shift_r1.json`
- `output/g25_pose_transport/seq13_map_disjoint_radio_atlas16_v1/pattern_oracle_best_q0_phase_sweep4.json`
- `output/g25_pose_transport/seq13_map_disjoint_radio_atlas16_v1/pattern_oracle_best_q2_phase_sweep4.json`

16 query × 16 stored poses 中 candidate 0 仅为诊断 GT anchor，实际有 15 个
pose-free candidates。候选集合 acquisition 为 strict `1/16=6.25%`、loose
`3/16=18.75%`、2m/20deg `10/16=62.5%`、2m/45deg `12/16=75%`。因此
front end 已能提供多数高容忍度 basin，但绝不是精位姿候选集。

同一 16-query 子集的纯物理区域召回仍强：parent Top64 exact/tolerant-0.5m
visible-mass recall 均值 `95.11%/96.02%`，parent token visible-mass coverage
`98.61%`；child 对应值为 `18.43%/35.48%` 与 token mass coverage `88.08%`。
这说明主要断点已从 parent global retrieval 移到 child identity compression、
pose-atlas orientation/location assignment 与最终 pose energy。

零位移 conservative phase 在 seq13 的 Top1 strict/loose 都为 `0%`，Spearman
仅 `0.172`、pairwise `56.0%`；物理去重 Top8 也只有 strict `6.25%`、loose
`18.75%`。GT anchor 却仍 16/16 排第一，说明能量有极窄正确峰，但从 coarse
basin 到该峰的排序地形不成立。

为检验投影平移，新增共同整数 shift 的 H/V conservative max：每个 shift 仍用
full-edge fixed denominator，越界为 unknown floor，所有 evidence 非负；对每个
shift 取 `min(H,V)` 后再 max，故 target evidence disappearing 不能提高任一
shift，也不能提高最终 max。半径1/2/3在 seq11 完全同分，冻结最小半径1后回放
seq13，只得到 Top1 strict/loose `6.25%/6.25%`、Spearman `0.195`。全局 shift
无法表达真实 SE(3) 的非刚性视差与遮挡。

最后从候选集中 GT 最近的 2m/45deg basin 启动 phase pattern search，仍得到两个
确定性反例：

- `0.469m/11.45deg -> 0.702m/21.63deg`，score `-0.1042 -> -0.0245`；
- `0.750m/13.23deg -> 0.750m/13.44deg`，score `-0.1531 -> -0.0280`。

两者均接受4次更新、objective drift=100%、strict/loose capture=0。因此当前
phase 后端正式 KILL；失败既不是纯召回缺少候选，也不是只差一个全局图像平移。
不得继续扩大 pattern sweeps 或把这个目标 GPU 化后宣称位姿改进。

下一主线应把 coarse basin ranking 与 local pose measurement 分离：前者保留完整
36x64 RADIO layout，以 candidate-conditioned soft physical keys 做有限形变容忍的
多模态 basin score；后者用冻结 renderer 生成 train-only SE(3) 扰动，学习
candidate-relative residual/energy，并以 coupled translation-rotation rank 与
局部负曲率反例作为硬门。输出始终保留物理去重的多 basin 与 null；只有 oracle
nearest basin 的局部 capture 通过后，才允许跑完整 test refinement。
