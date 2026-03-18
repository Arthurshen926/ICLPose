# ICLPose 下一步改进计划

## TL;DR
围绕三个核心方向展开论文级研究：(A) 端到端任务驱动特征学习，替代当前四段式解耦流程；(B) 架构范式反思，从 flow-based geometry 扩展到 scene coordinate regression 和 attention-based matching；(C) 可定位性先验学习，从重建质量信号中提取定位有效性信息。五张 4090 可支持 DDP 多卡训练和大规模消融。

## 方向 A：端到端任务驱动特征学习

### 现状问题
当前是四段串联：离线提特征 → 离线 AE 压缩 → 离线 3DGS 嵌入训练 → 在线位姿网络。
每一段优化目标不同（重建保真 vs 压缩保真 vs 嵌入保真 vs 定位精度），信息在接口处不可逆丢失。

### A1. 任务驱动特征压缩（中耦合，推荐先做）
- **思路**：保留离线 SD/DINO 提取，但把 AE 压缩器嵌入位姿训练图，用 pose loss 端到端微调压缩器后半层
- **实现**：
  1. 加载已训练好的 AE encoder 权重（`feature_compression/autoencoder.py` 的 `AutoencoderFlexible`）
  2. 在 `MSFlowPoseNet.__init__` 中替换 `ScaleDecoder`，改为 AE_encoder + 可选微调层
  3. 分阶段：先冻结 AE 跑 baseline → 解冻 encoder 后 2 层 → 全解冻
  4. 多尺度各自独立 AE，保持分块解耦（解决梯度博弈）
- **损失**：$L = L_{pose} + \lambda_1 L_{flow} + \lambda_2 L_{distill}$，其中 $L_{distill}$ 约束微调后特征不偏离原始分布太远
- **关键文件**：
  - `feature_compression/autoencoder.py` — AE 模型定义
  - `ic_models/ms_flow_pose_net.py` `ScaleDecoder` 类 — 替换目标
  - `scripts/train_ms_flow.py` — 训练循环修改
- **风险**：低。AE encoder 参数量小（~100K/尺度），不增加显存压力
- **验证**：Room0 + OldHospital 对比实验，观察 flow EPE 和 pose error 是否同步改善

### A2. 共享 Gaussian Latent + 多尺度解码（强耦合）
- **思路**：每个 Gaussian 存储一个紧凑 latent $z_i \in \mathbb{R}^{d}$（如 d=64），用可学习解码头 $D_s(z_i)$ 生成各尺度子特征
- **优势**：跨尺度一致性由共享 latent 保证；压缩天然嵌入；参数量可控
- **损失**：
  ```
  L = L_pose + λ_flow · L_flow + λ_recon · Σ_s L_recon^(s) + λ_compact · L_compact
  ```
  其中 $L_{compact}$ 可用信息瓶颈/VQ 约束 latent 容量
- **关键修改**：
  - `feature_3dgs/multiscale_gaussian_model.py` — 拆分 loc_feature 为 shared latent + per-scale decoder
  - `feature_3dgs/train_multiscale_embedding_v2.py` — 联合训练逻辑
- **风险**：中。需要仔细的 latent 维度搜索和训练稳定性调优
- **依赖**：A1 完成后再做，用 A1 验证端到端微调是否有收益

### A3. 多尺度分块独立优化策略
- **思路**：解决当前 loc_feature 224 维联合优化时的梯度博弈
- **方案**：
  1. Per-scale 独立学习率（coarse lr × 0.5, fine lr × 1.0）
  2. PCGrad 或 GradNorm 式梯度平衡
  3. 按尺度交替更新（奇数 iter 更新 fine，偶数 iter 更新 coarse+mid）
- **关键文件**：`feature_3dgs/train_multiscale_embedding_v2.py` 的 `optim_params` 构建
- **风险**：低。纯训练策略改动

## 方向 B：架构范式反思

### 现状问题
flow-based 方法的根本限制：(1) 局部 correlation 窗口 r=4 仅覆盖 ±4 像素；(2) flow→pose 的一阶线性化在大位移时不准确；(3) 3D 信息（深度）仅在最后求解阶段使用，未参与匹配过程。

### B1. Attention-based Global Matching（替代 correlation volume）
- **思路**：用 cross-attention 替代 coarse 全局 correlation 和 mid/fine 局部 correlation
- **参考**：GMFlow（全局 attention flow）、LoFTR（coarse attention + fine correlation）
- **实现**：
  1. Coarse 层：query/render 特征做 cross-attention（已有 `modules/transformer.py` 的 `MultiHeadAttention`）
  2. Fine 层：保留 GRU 迭代，但 correlation 输入改为 attention score map
  3. 混合方案：coarse 用 attention，fine 用 guided local correlation（不改动）
- **关键文件**：
  - `modules/transformer.py` — 已有 MultiHeadAttention 实现
  - `ic_models/ms_flow_pose_net.py` `global_correlation()` — 替换目标
- **显存估计**：coarse 7×10=70 tokens，attention 矩阵 70×70，显存可忽略
- **风险**：中。attention 在小分辨率 token 数上可能不如 all-pairs correlation

### B2. Scene Coordinate Regression（替代 flow 中间表示）
- **思路**：网络直接预测每像素的 3D 场景坐标（而非 2D 位移），然后用可微 EPnP/PnP 求解位姿
- **动机**：
  1. 避免 flow → pose 的一阶近似误差
  2. 自然利用 3D 信息（渲染深度直接参与预测）
  3. 场景坐标对重复纹理更鲁棒（3D 坐标唯一，flow 可能多解）
- **实现路线**：
  1. 渲染当前位姿下的 3D 坐标图（已有深度 + 内参，反投影即可）
  2. 网络预测当前坐标图的残差修正（类似 flow，但在 3D 空间）
  3. 2D-3D 对应 → 可微 EPnP/BPnPNet 求解
- **参考文献**：DSAC++（Brachmann 2021）、BPnPNet、ACE（Brachmann 2023）
- **关键文件**：
  - `scripts/eval_pnp_ransac.py` — 已有 PnP 验证脚本
  - `modules/geometry_solver.py` — 需新增可微 PnP solver
  - `modules/lie_algebra.py` — SE(3) 操作复用
- **风险**：高。需要实现可微 PnP 层；场景坐标预测的收敛性需要验证
- **建议**：作为 B1 之后的探索

### B3. FDA + 学习混合精修
- **思路**：利用已有 `modules/featuremetric.py` 的纯几何 FDA 方法作为后精修阶段
- **方案**：
  1. MSFlowPoseNet 输出粗位姿（现有方案）
  2. FDA 在高维原始特征空间（1408d）做 Gauss-Newton 精修（无需训练）
  3. 或训练一个轻量网络预测 FDA 的初始阻尼和迭代次数
- **依据**：FDA 已验证 5° 噪声下中位旋转 0.48°、<1° 比例 77.7%
- **关键文件**：`modules/featuremetric.py` — `FeaturemetricAligner.align()`
- **风险**：低。纯推理阶段叠加，不影响训练
- **注意**：FDA 需要原始高维特征，推理时需额外加载 SD+DINO 原始特征或在线提取

### B4. 深度感知匹配
- **思路**：将 3D 信息注入匹配过程（当前深度仅用于最后的 Jacobian 计算）
- **方案**：
  1. 深度编码为额外通道，与特征 concat 后做 correlation
  2. 或用深度构建 3D 位置编码（替代 2D PE），使 correlation 计算时"知道"两点的 3D 距离
  3. 深度一致性 mask：剔除渲染深度不一致区域的 correlation 结果
- **关键文件**：
  - `ic_models/ms_flow_pose_net.py` `PositionalEncoding2D` — 扩展为 3D PE
  - `modules/geometry_solver.py` `compute_image_jacobian()` — 深度已可用
- **风险**：低-中。3D PE 是纯增量改动

## 方向 C：可定位性先验学习

### 现状问题
当前置信度 `conf_fine` 是从 flow head 输出的标量/向量，仅反映"网络对 flow 预测的自信程度"，不反映"该区域本质上是否可定位"。

### C1. 可定位性评分网络（Localizability Prior）
- **思路**：学习一个先验函数 $p: \mathcal{F} \rightarrow [0,1]$，输入局部特征统计，输出可定位性分数
- **信号来源**：
  1. Correlation volume 的峰值锐度（单峰 vs 多峰 vs 平坦）
  2. 特征梯度幅值（高梯度 = 纹理丰富 = 可定位）
  3. 多尺度 flow 一致性（已有 `multiscale_consistency`，可提升为先验）
- **用途**：
  1. 几何求解器像素权重（替代/增强当前 confidence）
  2. 训练时样本权重（集中学习可定位区域）
  3. 3DGS 嵌入阶段：可定位性低的 Gaussian 降低学习率（替代盲目 densify）
- **实现**：
  1. 新增 `modules/localizability_head.py`
  2. 输入：correlation volume statistics + 特征图 + 深度
  3. 监督：用真实 flow 误差的反函数作为 soft label（flow 误差小的像素 → 高可定位性）
  4. 无需额外标注
- **关键文件**：
  - `ic_models/ms_flow_pose_net.py` — 注入 localizability 到 forward
  - `modules/geometry_solver.py` `diff_pose_solve()` — 用 localizability 调制权重
- **风险**：中。监督信号需要仔细设计

### C2. 特征质量引导的 Gaussian 管理
- **思路**：用定位任务的反馈信号指导 3DGS 的 densify/prune 决策
- **原则**：
  - 可修复信息缺失（采样不足、边界混叠）→ densify
  - 不可辨识区域（低纹理、重复纹理）→ 降权或 prune
- **判据**：C1 输出的 localizability score 在训练集上聚合，得到每个 Gaussian 的可定位性统计
- **风险**：中。需要 C1 先完成

## 执行路线图

### Phase 0：基础设施（1-2 周）
1. 搭建 5×4090 DDP 训练框架（当前 train_ms_flow.py 仅单卡）
2. 统一 Room0 + Stairs + OldHospital 的评估 benchmark 脚本
3. 建立消融实验的自动化对比表

### Phase 1：低风险高回报（2-4 周）
4. **A3**：多尺度分块独立学习率 + 梯度平衡（纯训练策略，1-2 天验证）
5. **A1**：任务驱动 AE 微调（中耦合，1-2 周完成实验）
6. **B3**：FDA 后精修叠加（推理阶段，几天验证）
7. **B4**：深度感知 3D 位置编码（增量改动，1 周）

### Phase 2：核心创新（1-2 月）
8. **C1**：可定位性先验网络设计 + 训练 + 接入 solver
9. **B1**：Attention-based coarse matching 替换 global correlation
10. **A2**：共享 Gaussian Latent 重设计（基于 Phase 1 结论决定是否推进）

### Phase 3：范式探索（2-3 月）
11. **B2**：Scene Coordinate Regression 原型
12. **C2**：特征质量引导 Gaussian 管理
13. 论文成型：选择最有效的 2-3 个方向组合，做完整消融和 SOTA 对比

## 关键决策点
- A1 完成后：如果端到端微调 AE 无收益 → 说明当前特征质量不是瓶颈，转向 B 方向
- B1 完成后：如果 attention matching 在 OldHospital 有 >0.5° 改善 → 加速 B2 探索
- C1 完成后：如果 localizability prior 能显著区分可定位/不可定位区域 → 论文核心贡献之一

## 排除范围
- 不做 RGB 3DGS 几何重训（计算代价太大，收益不明确）
- 不做 SD/DINO backbone 微调（冻结 backbone 是合理的工程选择）
- 不做分布式多节点训练（5×4090 单机 DDP 足够）
