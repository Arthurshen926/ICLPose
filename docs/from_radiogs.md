# RADIO-GS 特征重建方法流程说明

本文总结 RADIO-GS 中“从教师特征训练出特征重建框架”的完整流程。这里的教师特征指 RADIO C-RADIOv4-H 提供的 1280 维空间特征，目标是把它们蒸馏进高斯场表示中，并在新视角下重建回接近教师空间的特征图，供深度、分割和文本 grounding 等下游任务继续使用。

## 1. 方法目标

RADIO-GS 的核心不是直接做像素重建，而是做“特征重建”。输入 RGB 图像后，先通过冻结的 RADIO 编码器提取高维教师特征，再训练一个高斯场特征表示，让它在任意视角渲染出来后，经过解码器恢复成和教师特征尽量一致的 1280 维表示。这样一来，场景的几何结构由 Gaussian Splatting 保存，语义与任务信息由特征场保存。

可以把整体目标概括为：

1. 用教师特征作为监督信号。
2. 用 3D 高斯场把多视角特征压进场景表示。
3. 用屏幕空间渲染得到新视角特征图。
4. 通过重建损失让渲染结果尽量回到教师特征空间。
5. 再把重建特征输入深度、分割、grounding 等头部做评估。

## 2. 整体流程

训练链路可以拆成五步：

1. 先训练或准备几何骨架。
2. 抽取每一帧的 RADIO 教师特征。
3. 用教师特征监督 feature field 训练。
4. 对新视角渲染结果做解码与辅助任务学习。
5. 在验证集上看特征相似度和下游任务表现。

更具体地说，流程是：

```text
RGB frame
  -> frozen RADIO encoder
  -> 1280d teacher feature
  -> Gaussian scene backbone + learnable feature field
  -> render compact feature map
  -> HCD codec / screen-space refiner / hybrid fusion
  -> reconstructed 1280d feature map
  -> depth / segmentation / grounding heads
```

## 3. 教师特征如何进入训练

教师特征来自预先抽取的 RADIO 输出。仓库中的 [`radio_gs/scripts/extract_radio_features.py`](../radio_gs/scripts/extract_radio_features.py) 负责把每帧特征保存成磁盘文件，训练脚本再从特征目录读取它们。

训练数据不仅包含特征，还会带上以下信息：

1. 相机位姿，用于把高斯场渲染回当前视角。
2. 深度图，用于几何约束和深度辅助任务。
3. 语义标签，用于 segmentation 和 grounding。
4. 可选 RGB 参考图，用于 screen-space refiner。

在 [`radio_gs/scripts/train_feature_field.py`](../radio_gs/scripts/train_feature_field.py) 里，`SimpleRadioDataset` 会把这些信息组装成一个 batch，然后送入训练循环。

## 4. 主干表示：高斯场特征重建框架

项目支持两类主干：

1. `explicit`：每个高斯点直接学习一个特征向量。
2. `hybrid`：每个高斯点既有轻量 latent code，又结合 3D hash grid 的 coarse 分支，最后做融合。

当前仓库的主线更偏向 hybrid，因为它更适合同时保留局部细节和场景级上下文。

### 4.1 Explicit 架构

显式架构比较直接：每个 Gaussian 绑定一个可学习特征，渲染时直接把这些特征 alpha-blend 成屏幕空间特征图，再送入 HCD codec 解码回 1280 维。它的优点是简单，缺点是表达容量有限，而且对复杂几何和语义结构的鲁棒性较弱。

### 4.2 Hybrid 架构

Hybrid 架构是当前重点。它在 [`radio_gs/models/hybrid_gaussian.py`](../radio_gs/models/hybrid_gaussian.py) 中实现，主要由三部分组成：

1. per-Gaussian latent code，提供 fine 细节。
2. 3D hash grid，按空间位置查询 coarse 特征。
3. 融合头，把 fine / coarse 两路信息合并成最终输出。

训练时，首先根据高斯场把 latent feature map 渲染到屏幕空间；然后通过深度反投影把每个像素对应的三维位置恢复出来，再查询 hash grid 得到 coarse feature；最后融合成最终的特征表示。这样做的好处是：

1. fine 分支对局部几何更敏感。
2. coarse 分支提供全局场景先验。
3. decoupled heads 还能把 geometry 和 semantic 信号拆开，让不同任务各自受益。

## 5. 屏幕空间渲染器做了什么

特征重建不是在体素空间里完成的，而是在新视角的屏幕空间里完成。这个部分由 [`radio_gs/rendering/feature_renderer.py`](../radio_gs/rendering/feature_renderer.py) 负责。

渲染器会从高斯模型读取以下信息：

1. 坐标。
2. 旋转。
3. 尺度。
4. 不透明度。
5. 特征向量。

然后通过 gsplat 把它们投影到当前视角，输出三张核心图：

1. `feature_map`：渲染得到的 compact 特征图。
2. `depth_map`：几何深度图。
3. `alpha_map`：可见性或覆盖度图。

如果开启 RGB 联合渲染，它还会同时输出 RGB，用于自监督的 refiner 或可视化。

## 6. 从 compact 特征到 1280d 的重建

重建不是直接把渲染出的低维特征当结果，而是经过一个压缩-解压框架。这个框架在 [`radio_gs/models/hcd_codec.py`](../radio_gs/models/hcd_codec.py) 中实现。

HCD codec 有两个阶段：

1. Encoder：把 1280d RADIO 特征压到较小 bottleneck，一般是 32 到 64 维。
2. Decoder：把 bottleneck 再恢复成 1280d。

它本质上是一个 1×1 卷积 MLP 系统，原因很明确：特征重建要求像素对齐，不能引入会破坏空间对齐的卷积核。也就是说，编码和解码都应该尽量只做通道混合，不改变空间坐标。

训练时有两种模式：

1. `latent` 模式：直接在低维空间对齐，codec 冻结，只训练场景特征本身。
2. `decoded` 模式：默认方式，先渲染 compact 特征，再经过 decoder 回到 1280d 后和教师特征对齐。

当前项目主线更常用 `decoded`，因为它能把特征场学习和最终任务监督统一到同一个语义空间里。

## 7. 训练监督是怎么组合的

训练逻辑集中在 [`radio_gs/scripts/train_feature_field.py`](../radio_gs/scripts/train_feature_field.py)。主损失不是单一项，而是一个多目标组合。

### 7.1 主重建损失

主损失由 [`radio_gs/losses/distillation_loss.py`](../radio_gs/losses/distillation_loss.py) 定义，包含两项最基本的重建约束：

1. L2 / Huber 损失。
2. Cosine similarity 损失。

其中 cosine 负责对齐方向，L2 负责对齐幅值；如果特征存在通道尺度问题，还可以加 channel-standardized loss，缓解“归一化后空间结构被抹掉”的情况。

### 7.2 紧凑空间损失

如果使用 decoded 模式，训练过程中还会监督渲染出的 compact 特征接近 codec 编码后的教师 compact 特征。这样可以让压缩空间和解码空间都保持稳定，不至于只在 1280d 上看起来对齐，但中间瓶颈已经偏移。

### 7.3 空间正则

项目还加了几个让特征图更平滑、更边界敏感的辅助项：

1. Total variation loss，抑制局部噪声。
2. gradient-weighted loss，让边缘保留更清楚。
3. depth-guided feature loss，利用几何深度约束特征平滑区域和边界区域。
4. geometric edge alignment loss，鼓励特征边界和几何边界一致。
5. boundary-aware loss，综合深度与 alpha 边界来约束重建特征。

这些项的意义是：特征重建不只是“像”，还要“结构对”。

### 7.4 任务头监督

在重建特征之上，还可以挂下游任务头：

1. DepthHead：从重建特征预测深度。
2. SegmentationHead：从重建特征预测语义分割。
3. GroundingHead / QueryGroundingAuxLoss：从 SigLIP2 对齐空间做文本 grounding。

这些头的作用不是替代主重建损失，而是逼迫重建特征保留任务可用性。也就是说，模型不只是要重建得像，更要“可用”。

### 7.5 SigLIP2 对齐

项目还引入两种 SigLIP2 相关监督：

1. spatial feature projection 对齐。
2. summary head 对齐。

它们分别对应 [`radio_gs/models/siglip_projection.py`](../radio_gs/models/siglip_projection.py) 中的两个映射：一个偏空间视觉 embedding，一个偏文本对齐空间。训练时，这两个头大多是冻结的，作为固定对齐目标，用来约束重建特征在开放词汇任务中的语义一致性。

## 8. 一次训练迭代的真实数据流

在 [`train_feature_field.py`](../radio_gs/scripts/train_feature_field.py) 里，一次迭代大致是：

1. 读取某个视角的教师特征、位姿、深度和语义。
2. 用高斯场渲染出 compact feature map。
3. 经过 FeatSharp3D 和可选的 screen-space refiner 做细节修正。
4. 如果是 hybrid，再把深度反投影成 3D 位置，走 hash grid 分支并融合。
5. 如果是 decoded 模式，把 compact 特征送进 HCD decoder，恢复成 1280d。
6. 计算主重建损失和所有启用的辅助损失。
7. 反向传播，更新高斯特征、hash grid、decoder、refiner 和可训练 head。

这条链路的关键在于：几何由高斯场保证，语义由特征蒸馏保证，任务可用性由多头辅助监督保证。

## 9. 为什么这个方法有效

这个框架之所以成立，原因主要有四点：

1. RADIO 教师特征本身已经包含了丰富的语义和空间结构，因此适合作为蒸馏目标。
2. Gaussian Splatting 天然适合可微渲染，能把三维场景和二维监督直接连起来。
3. HCD codec 把“高维教师空间”拆成“可学习的紧凑瓶颈”和“可恢复的输出空间”，训练更稳定。
4. 辅助任务不是附加噪声，而是在逼模型保留真正有用的表示能力。

## 10. 局限与风险

这套方法也有明显风险：

1. 如果教师特征抽取有索引错位，重建会学到错误监督，即使损失下降也不可信。
2. 如果几何骨架不干净，特征场会把几何误差一起编码进去。
3. 辅助头过多时，训练目标可能相互拉扯，导致主重建性能下降。
4. hybrid 分支对配置更敏感，尤其是 hash 分辨率、latent 维度和 decoder 宽度。

仓库 README 已明确指出：如果教师特征后来被修正，旧 checkpoint 只能算诊断结果，正式结论需要重新训练。

## 11. 配置里最值得关注的参数

在配置文件里，最核心的参数是：

1. `ply_path`：几何骨架来源。
2. `feature_dir`：教师特征目录。
3. `architecture`：`explicit` 还是 `hybrid`。
4. `train_mode`：`latent` 还是 `decoded`。
5. `bottleneck_dim` 或 `hybrid_latent_dim`：压缩容量。
6. `depth_loss_weight`、`seg_loss_weight`、`grounding_query_loss_weight`：辅助任务权重。
7. `use_refiner`、`self_guided`：是否启用屏幕空间修正。
8. `siglip_projection_weights`、`grounding_text_embeddings`：文本对齐资源。

例如 [`radio_gs/configs/replica_hybrid_v14_room_2_reextract.yaml`](../radio_gs/configs/replica_hybrid_v14_room_2_reextract.yaml) 和 [`radio_gs/configs/lerf_hybrid_v14_figurines_frozen_dh.yaml`](../radio_gs/configs/lerf_hybrid_v14_figurines_frozen_dh.yaml) 都体现了这一模式：先渲染 compact features，再做 decode，最后混合几何、深度、语义和 grounding 的监督。

## 12. 推荐阅读顺序

如果想快速理解代码，建议按这个顺序看：

1. [`radio_gs/scripts/train_feature_field.py`](../radio_gs/scripts/train_feature_field.py)
2. [`radio_gs/rendering/feature_renderer.py`](../radio_gs/rendering/feature_renderer.py)
3. [`radio_gs/models/hcd_codec.py`](../radio_gs/models/hcd_codec.py)
4. [`radio_gs/models/hybrid_gaussian.py`](../radio_gs/models/hybrid_gaussian.py)
5. [`radio_gs/losses/distillation_loss.py`](../radio_gs/losses/distillation_loss.py)
6. [`radio_gs/heads/depth_head.py`](../radio_gs/heads/depth_head.py)
7. [`radio_gs/heads/segmentation_head.py`](../radio_gs/heads/segmentation_head.py)
8. [`radio_gs/heads/grounding_head.py`](../radio_gs/heads/grounding_head.py)

## 13. 一句话总结

RADIO-GS 的特征重建本质上是一个“冻结教师特征 + 高斯场渲染 + 紧凑瓶颈压缩 + 1280d 解码 + 多任务对齐”的蒸馏框架。它不只是复原特征图本身，而是把教师特征中可用于深度、分割和文本 grounding 的信息尽可能保留在三维场景表示里。
