# 特征重建方法流程说明

> 目标：说明本项目如何从教师特征出发，训练出一套可用于定位的特征重建框架，并对当前实现和理想目标之间的差异做系统分析。

## 1. 先给结论

本项目的“特征重建”不是单纯把图像重建出来，而是让场景侧表示在空间上尽可能复现教师特征的结构、边界和语义分布。当前主线可以理解为三层串联：

1. 教师端用 RADIO 提供监督目标，生成双尺度特征。
2. 显式端用 2DGS/3DGS 作为可学习 latent carrier，在几何上承载特征。
3. 隐式端用 DCFF / HashGrid decoder 对显式载体做进一步重建、补全和细化。

可简写为：

`teacher RADIO -> teacher cache / visualization -> 2DGS explicit feature carrier -> DCFF implicit feature field -> qualitative evaluation`

## 2. 目标到底是什么

这里的“重建”对象不是 RGB，而是教师特征。教师特征通常分成两类：

- fine_geo：偏几何、边界更敏感的浅层特征。
- coarse_sem：偏语义、上下文更强的深层特征。

理想中的特征重建框架要满足几件事：

- 教师端输出稳定、可解释，且可被独立可视化。
- 场景端的表示可以同时承接几何和特征，而不是只学颜色。
- 训练能支持先几何后特征，避免一开始就把几何和特征耦合在一起。
- 评价时能把 teacher / 2DGS / DCFF 三层放到同一个坐标系里比较。

这也是为什么本项目会同时保留显式和隐式两条路线：显式 2DGS 负责局部结构和边界，隐式 DCFF 负责空间补全、平滑传播和解码能力。

## 3. 理想流程

### 3.1 第一阶段：抽取教师特征

入口是 [feature_extract/extract_radio_dual_features.py](../extract_radio_dual_features.py)。这一步做两件事：

- 从原始图像中提取 RADIO 双尺度特征。
- 把特征压缩并缓存成可重复使用的 teacher dataset。

当前输出目录会形成三类内容：

- `fine_geo/`：浅层、偏几何的特征缓存。
- `coarse_sem/`：深层、偏语义的特征缓存。
- `pca_params/`：PCA 参数，便于复现压缩过程。

如果要看原始 teacher 特征的结构，不要只看训练日志，应使用 [feature_extract/visualize_teacher_features.py](../visualize_teacher_features.py)。它会把原始 teacher、PCA teacher 和输入图像放到同一张面板里。

### 3.2 第二阶段：显式特征高斯

入口是 [feature_gaussian.train](../../feature_gaussian/train.py)。当前 pilot 使用的是 [feature_gaussian/configs/joint_radio_dual_oldhospital_pilot.yaml](../../feature_gaussian/configs/joint_radio_dual_oldhospital_pilot.yaml)。

这一步的核心思想是：

- 用已有几何 PLY 作为 warmstart。
- 冻结几何，只训练特征嵌入，先把 feature carrier 站稳。
- 再让 2DGS 在显式层面重建 fine_geo / coarse_sem。

这和“先把几何和外观训稳，再加特征”的目标一致。现在训练器已经支持：

- `init_ply`：从现成点云几何开始。
- `freeze_geometry`：只训练特征，不让几何继续漂移。
- 多尺度 feature cache：可直接吃 `fine_geo` / `coarse_sem`。

### 3.3 第三阶段：隐式特征场

入口是 [feature_field.train](../../feature_field/train.py)。当前 pilot 使用的是 [feature_field/configs/dcff_oldhospital_radio_dual_pilot.yaml](../configs/dcff_oldhospital_radio_dual_pilot.yaml)。

DCFF 的工作方式可以理解为：

- 先用 HybridGaussian 作为场景几何底座。
- 再用 SpatialHashGrid 提供空间查询与隐式建模能力。
- 最后用 fine decoder 和 coarse branch 去重建教师特征。

这一步适合做两类事情：

- 把显式 2DGS 没学稳的部分继续补上。
- 在多尺度上让 teacher 特征更平滑、更连续地落到 map 上。

### 3.4 第四阶段：评价和可视化

这个项目里，真正的 qualitative evaluation 不应该只看训练 loss，而应该同时看：

- teacher 特征图。
- 2DGS 的重建图与误差图。
- DCFF 的重建图与误差图。

当前对应入口分别是：

- [feature_gaussian/evaluate_joint.py](../../feature_gaussian/evaluate_joint.py)
- [feature_field/visualize_reconstruction.py](../visualize_reconstruction.py)

注意：`feature_field/evaluate.py` 更偏 smoke test，不是最终 qual 评估入口。

## 4. 当前实现怎么映射这条流程

| 层级 | 当前模块 | 作用 | 主要产物 |
|---|---|---|---|
| Teacher | [feature_extract/extract_radio_dual_features.py](../extract_radio_dual_features.py) + [feature_extract/visualize_teacher_features.py](../visualize_teacher_features.py) | 抽取 RADIO 双尺度 teacher，并提供可视化 | `feature_extract/output/features_radio_dual/...`、`teacher_visuals/...` |
| 显式 carrier | [feature_gaussian/legacy_3dgs/train_2dgs_joint_v3.py](../../feature_gaussian/legacy_3dgs/train_2dgs_joint_v3.py) | warmstart 2DGS、冻结几何、训练 per-scale 特征 | `point_cloud/iteration_*`、`features_best/` |
| 隐式 field | [feature_field/train_impl.py](../train_impl.py) | cached teacher / online teacher 下训练 DCFF | `checkpoints/best.pth`、`latest.pth` |
| 评价 | [feature_gaussian/evaluation/eval_joint_reconstruction.py](../../feature_gaussian/evaluation/eval_joint_reconstruction.py) + [feature_field/visualize_reconstruction.py](../visualize_reconstruction.py) | 输出重建面板和数值指标 | `evaluation_test/`、`reconstruction_visuals_pilot/` |

## 5. 本次 pilot 跑出来了什么

这次我按上面的主线实际跑了一遍 OldHospital：

### 5.1 Teacher 提取

- 抽取帧数：1084
- teacher 可视化已经生成：见 [feature_extract/output/teacher_visuals/OldHospital_pilot/contact_sheet.png](../../feature_extract/output/teacher_visuals/OldHospital_pilot/contact_sheet.png)
- PCA 保留率大致为：fine_geo 约 0.67，coarse_sem 约 0.80

这说明 teacher 端本身是稳定的，尤其 coarse_sem 更容易压缩，符合它更偏语义、更平滑的性质。

### 5.2 2DGS 显式特征重建

pilot 训练日志见 [feature_gaussian/output/joint_radio_dual_oldhospital_pilot/train.log](../../feature_gaussian/output/joint_radio_dual_oldhospital_pilot/train.log)。关键结果：

- best PSNR：15.55 dB
- test 上 4 个样本的平均 cosine：
  - fine_geo 约 0.295
  - coarse_sem 约 0.488

这说明显式 2DGS 已经能较稳定地承接 teacher 特征，且 coarse 分支明显比 fine 分支更容易对齐。

对应可视化在 [feature_gaussian/output/joint_radio_dual_oldhospital_pilot/evaluation_test/vis_iter000000.png](../../feature_gaussian/output/joint_radio_dual_oldhospital_pilot/evaluation_test/vis_iter000000.png)。

### 5.3 DCFF 隐式特征重建

pilot 训练日志见 [feature_field/output/dcff_oldhospital_radio_dual_pilot/train.log](../output/dcff_oldhospital_radio_dual_pilot/train.log)。关键结果：

- 最终 best total loss：1.8817
- best 出现在 iter 40
- iter 40 的 validation loss：1.8763

重建面板已输出到 [feature_field/output/reconstruction_visuals_pilot/reconstruction_oldhospital_radio_dual_pilot/test/contact_sheet.png](../output/reconstruction_visuals_pilot/reconstruction_oldhospital_radio_dual_pilot/test/contact_sheet.png)。

## 6. 当前实现和理想流程的差异

### 6.1 之前的主要偏差

1. 教师可视化缺口

   之前 teacher 抽取脚本只负责缓存，没有独立的 teacher qual 入口，导致很难判断“是 teacher 质量问题，还是 map 侧重建问题”。

2. 显式 2DGS 缺少 warmstart

   原来的 joint 2DGS 训练更像从头构建显式表示，不够贴近“先几何、后特征”的目标。

3. DCFF 训练与 runtime 不一致

   训练端和 evaluation 端在 fine decoder 结构上曾经不一致，导致 checkpoint 在重建时会被跳过部分权重。

4. best checkpoint 追踪不合理

   之前 best 追踪会被早期 fine-only 阶段的 loss 干扰，进入 coarse 阶段后不一定还能正确生成 best.pth。

### 6.2 现在已经补上的部分

- teacher extraction 已经能输出 raw 和 cached 相关可视化。
- 2DGS 已支持 `init_ply` 和 `freeze_geometry`，能做真正的 warmstart feature training。
- DCFF 已修复 fine decoder 类型透传问题，训练和 runtime 现在按同一个 decoder 配置走。
- DCFF 的 best 追踪改成了 coarse-aware，不会再被前期 fine loss 卡死。

### 6.3 还剩下的结构性差距

即便现在能跑通，这条线仍然不是单一端到端 trainer，而是分段式系统：

- teacher extraction 一段。
- 显式 2DGS 一段。
- 隐式 DCFF 一段。

也就是说，当前实现已经比原来更接近理想流程，但离“一个统一 trainer 同时优化 teacher supervision、几何、显式 latent 和隐式 decoder”还有一层整合空间。

## 7. 如何理解这条流程的训练逻辑

如果把它抽象成训练逻辑，可以记成下面这条线：

1. 先定义监督目标：teacher RADIO 特征。
2. 再选择可承载该目标的场景表示：2DGS / HybridGaussian。
3. 先让几何和外观稳住，再把特征 embedding 接进去。
4. 用 fine / coarse 两个尺度分别拟合不同层次的 teacher。
5. 最后用可视化和 cosine / L1 / PSNR 去确认“重建的是特征结构，而不是偶然的颜色相似”。

这个逻辑的关键点是：

- fine_geo 更看重局部结构和边界。
- coarse_sem 更看重连续语义和上下文。
- DCFF 的价值在于把显式 carrier 没学全的部分补齐。

## 8. 实际建议

- 新实验建议默认 batch_size 16，这次 pilot 也是按这个设置跑的。
- 如果你要判断 coarse 变差的原因，先看 teacher cache 和 train/runtime 一致性，再看损失权重，不要先怀疑 teacher 坏了。
- 如果你要做下一轮改进，优先顺序应该是：
  1. 保持 teacher cache 和 joint 2DGS / DCFF 的尺度命名一致。
  2. 继续把显式 warmstart 和 implicit decoder 的接口收紧。
  3. 再考虑把两段训练进一步合并成统一的联合优化流程。
