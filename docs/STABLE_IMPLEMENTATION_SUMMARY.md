# ICLPose 稳定实现说明

> 本文档只总结当前已经稳定实现、并且能够支撑训练与推理主流程的部分。
> 不包含仍在探索中的 Transformer、FlowFeat 替代分支、TriPlane 扩展等实验性路线。

---

## 1. 项目目标

ICLPose 要解决的问题是：

- 输入一张查询图像及其初始位姿估计
- 在已有场景 3DGS 地图上渲染参考特征
- 通过多尺度光流匹配和几何求解，迭代优化相机 6-DOF 位姿

核心思想不是直接回归位姿，而是将定位拆成三步：

1. 用 3DGS 特征地图在当前位姿下渲染参考特征
2. 估计查询特征与参考特征之间的 dense flow
3. 用深度和 Image Jacobian 将 flow 转换为 SE(3) 位姿增量

因此，该方法本质上是一个“渲染驱动的迭代位姿精化框架”。

---

## 2. 稳定主线概览

当前稳定主线由 4 个模块组成：

1. 离线特征准备
2. 多尺度 3DGS 特征渲染
3. MSFlowPoseNet 多尺度光流网络
4. 可微几何求解与外循环迭代

整体流程如下：

```text
查询图像
  -> 离线提取多尺度特征
  -> 给定初始位姿 T_init
  -> 外循环 K 次:
       1) 在 T_curr 下渲染参考特征与深度
       2) MSFlowPoseNet 预测 coarse/mid/fine flow + confidence
       3) 用 fine flow + depth + Jacobian 求解 Δxi
       4) 用 SE(3) 指数映射更新 T_curr
  -> 输出最终位姿 T_final
```

稳定实现中，真正起决定作用的是：

- 多尺度 coarse-to-fine 匹配
- fine 阶段 RAFT-style 迭代精化
- confidence 加权几何求解
- 外循环 render -> match -> solve -> update

---

## 3. 离线阶段

### 3.1 查询特征提取

当前稳定版本使用 SD + DINOv2 多尺度特征。

典型尺度如下：

- coarse: 低分辨率全局语义特征
- mid: 中分辨率过渡特征
- fine_sd: 高分辨率 SD 特征
- fine_dino: 高分辨率 DINO patch 特征

在当前稳定训练配置中，实际常用的是 stride7 压缩特征版本：

- coarse: 32 通道
- mid: 64 通道
- fine_sd: 64 通道
- fine_dino: 64 通道

这些特征以 `.pt` 文件形式预先存储，训练时直接读取，不在主训练循环中实时提取。

### 3.2 3DGS 特征地图训练

除了原始的几何 3DGS 模型，还会为每个尺度单独训练一套 feature embedding：

- coarse.pth
- mid.pth
- fine_sd.pth
- fine_dino.pth

其核心做法是：

- 共享同一套 Gaussian 几何参数
- 仅学习每个 Gaussian 的特征向量 `_loc_feature`
- 在目标视角渲染特征图，并拟合离线提取的图像特征

这样，后续定位时就不需要渲染 RGB，而是直接渲染多尺度 feature map。

---

## 4. 数据输入与位姿定义

数据集主入口是 [data/dataset_v4.py](/root/ICLPose/data/dataset_v4.py)。

每个样本包含：

- 多尺度查询特征
- GT 位姿
- 加噪后的初始位姿
- depth 图

稳定实现中的关键约定：

- 原始轨迹文件存储的是 c2w
- 网络内部统一使用 w2c
- 训练时对 GT 位姿添加随机旋转和平移扰动，生成初始位姿

这样模型学到的不是“从零预测位姿”，而是“从一个粗初值迭代修正”。

这一定义非常重要，因为它决定了整个系统是 pose refinement，而不是 global relocalization from scratch。

---

## 5. 多尺度 3DGS 渲染

渲染入口是 [modules/multiscale_renderer.py](/root/ICLPose/modules/multiscale_renderer.py)。

### 5.1 渲染器做什么

MultiScaleRenderer 负责：

- 加载多个尺度的 feature-3DGS 模型
- 根据当前位姿渲染对应尺度的特征图
- 在最细尺度同时渲染深度图

稳定实现通常使用 4 个尺度模型：

- coarse
- mid
- fine_sd
- fine_dino

### 5.2 为什么渲染特征而不是渲染 RGB

因为定位阶段真正需要的是“对位姿敏感、可匹配的表示”，而不是外观图像本身。

渲染 feature map 有两个优势：

- 可以直接和查询特征做匹配
- 避免 RGB 光照和外观细节对定位带来的干扰

深度图则用于后续几何求解的 Jacobian 计算。

---

## 6. MSFlowPoseNet 主网络

核心模型是 [ic_models/ms_flow_pose_net.py](/root/ICLPose/ic_models/ms_flow_pose_net.py)。

稳定版本可以概括为：

- 共享 decoder 的多尺度特征编码
- coarse 全局相关性
- mid/fine 局部相关性
- fine 阶段 ConvGRU 迭代精化
- 用最终 fine flow 驱动几何求解

### 6.1 输入输出

输入：

- query_feats
- render_feats
- depth

输出：

- flow_coarse / flow_mid / flow_fine
- conf_coarse / conf_mid / conf_fine
- hidden_fine
- delta_xi

其中真正送入几何求解器的是 fine 分辨率的 flow 和 confidence。

### 6.2 Coarse 阶段

coarse 阶段的目标是解决大位移和全局对齐问题。

步骤：

1. query / render coarse feature 经过 decoder 降到统一维度
2. 构建全局 all-pairs correlation
3. 通过 flow head 预测初始 flow 与 confidence

因为 coarse 分辨率低，所以可以承受全局相关性计算。

### 6.3 Mid 阶段

mid 阶段的作用是将粗匹配结果细化到更合理的位置。

步骤：

1. 将 coarse flow 上采样到 mid 分辨率
2. 用当前 flow 引导 reference feature 的局部搜索
3. 只在局部窗口内构建相关性
4. 输出更细的 flow

这样复杂度从全局匹配显著下降，同时利用 coarse 结果缩小搜索范围。

### 6.4 Fine 阶段

fine 阶段是定位精度的关键。

其稳定实现有两个重点：

- fine_sd 与 fine_dino 双分支融合
- RAFT-style ConvGRU 多次迭代

每次迭代都执行：

1. 根据当前 flow 构建 guided local correlation
2. 编码 correlation 特征
3. 结合 hidden state 与上下文更新 ConvGRU
4. 预测 delta flow 与 confidence
5. 累积更新 flow

因此 fine 不是一次输出，而是反复修正。

---

## 7. 几何求解器

几何求解位于 [modules/geometry_solver.py](/root/ICLPose/modules/geometry_solver.py)。

### 7.1 输入

几何求解器使用：

- dense flow
- depth
- confidence
- 相机内参

### 7.2 核心原理

对于每个像素，深度给出了该像素对应的 3D 点位置；
给定相机运动的无穷小扰动 $\Delta \xi$，可以写出像素位移对相机运动的线性近似，也就是 Image Jacobian。

于是每个像素提供 2 个线性约束，整张图会形成一个过定约束系统：

$$J \Delta \xi \approx f$$

其中：

- $J$ 是由深度和内参得到的 Jacobian
- $f$ 是网络预测的光流
- $W$ 是由 confidence 构造的权重

最终解的是加权最小二乘：

$$\Delta \xi = (J^T W J + \lambda I)^{-1} J^T W f$$

这一步是整个系统里最强的几何归纳偏置之一：

- 网络只需要预测局部像素运动
- 6-DOF 位姿由显式几何求解得到

### 7.3 稳定版鲁棒性配置

稳定实现支持以下鲁棒机制：

- LM damping
- IRLS
- Huber / Geman-McClure / GNC-GM robust kernel
- directional confidence
- adaptive damping

但如果只说“当前最稳定、最核心的实现”，主流程依旧是：

- confidence 加权
- 可微最小二乘求解

IRLS 等机制属于在此基础上的鲁棒增强。

---

## 8. 外循环位姿精化

这是当前系统真正区别于早期失败版本的关键。

单次前向并不能保证定位成功，因为：

- 初始位姿有误差
- 错误位姿下渲染出来的参考特征也会偏移
- 一次匹配难以直接恢复到最终姿态

所以训练和推理都采用外循环：

1. 用当前位姿渲染参考特征
2. 预测 flow 与 confidence
3. 几何求解得到位姿增量
4. 更新位姿后重新渲染

这个循环使系统成为典型的 iterative refinement 框架。

稳定配置中常见设置：

- train outer_iters = 10
- val outer_iters = 20

验证阶段更多迭代通常会进一步提升精度。

---

## 9. 训练方式

训练入口是 [scripts/train_ms_flow.py](/root/ICLPose/scripts/train_ms_flow.py)。

### 9.1 两阶段训练思想

稳定实现延续了“flow 优先、pose 随后”的设计：

- Phase 1: 以 flow loss 为主
- Phase 2: 联合优化 flow loss 和 pose loss

这样做的原因是：

- 先把 dense correspondence 学稳
- 再用 pose supervision 做全局约束

### 9.2 主要损失

当前稳定损失包括：

- multiscale flow loss
- fine sequence loss
- pose loss
- confidence regularization

其中：

- coarse/mid/fine 有不同权重
- fine 使用 RAFT-style sequence supervision
- pose loss 采用 cosine rotation loss + L1 translation loss

### 9.3 为什么用 cosine rotation loss

早期 `acos` 形式在小角度区域梯度不稳定，容易导致训练和指标分析出现问题。

当前稳定版本改用：

- rotation loss = $1 - \cos(\theta)$

优点是：

- 在小角度附近更平滑
- 更适合高精度 pose refinement 场景

---

## 10. 当前稳定配置

如果只保留一条最稳定、最清楚的实现主线，建议以 [configs/exp143_room0_lownoise_ft.yaml](/root/ICLPose/configs/exp143_room0_lownoise_ft.yaml) 为代表。

该配置的主要特点：

- stride7 压缩特征
- coarse/mid/fine = 15×20 / 30×40 / 91×120
- fine_iters = 8
- mid_iters = 1
- irls_iters = 3
- low-noise fine-tuning
- outer loop 10/20
- cosine scheduler

它代表的是当前“稳定高精度 refine 版本”，而不是各种实验性新模块。

---

## 11. 当前建议视为稳定实现的范围

建议纳入“稳定实现”描述的内容：

- 多尺度 feature-3DGS 渲染
- DatasetV4 数据读取与 pose perturbation
- MSFlowPoseNet 主干
- fine 阶段 ConvGRU 迭代
- confidence 加权几何求解
- 外循环 render-match-solve-update
- cosine pose loss 与 multiscale flow supervision

建议不要放进“稳定实现”主文档的内容：

- Transformer refiner / transformer flow decoder
- FlowFeat 替代主特征分支
- TriPlane renderer
- 各类探索性 gating / attention / localizability 变体

这些内容更适合作为扩展功能或实验性分支单独说明。

---

## 12. 一句话总结

ICLPose 当前稳定实现可以概括为：

> 用 3DGS 渲染多尺度参考特征，用 coarse-to-fine dense flow 建立查询图像与渲染特征之间的像素级对应，再通过 confidence 加权的可微几何求解器恢复 SE(3) 位姿，并在外循环中不断重复这一过程，最终实现高精度位姿精化。