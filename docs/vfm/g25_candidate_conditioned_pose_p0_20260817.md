# G25 candidate-conditioned pose：本轮 P0 落地与实测

## 决策

冻结 RADIO 物理区域召回前端。召回概率只定义 `q_ret`，不再被解释为
位姿测量。新增的 reference seam 定义独立的 `q_pose(. | basin)`：低维 pose
code、法向、相对深度和边界一致性只能把检索先验质量转移到同 child 的匹配
质量；任何没有被转移的质量都进入 explicit unmatched。该 seam 不含关键点、
硬对应、PnP 或绝对位姿回归。

当前 `q_pose` 核心刻意标记 `production_eligible=false`。原因是五项实物仍未
齐备：冻结的 32--64D map pose field、query pose readout、真实 candidate
adapter、训练扰动/排名损失、独立验证集能量地形门禁。数学接口通过不等于
模型已训练。

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
