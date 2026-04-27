# 联合自适应压缩主线（128d，连续瓶颈优先）

## Summary
当前 Cambridge 主链路的 `64d` 是离线 PCA 和接口约束的结果，不是面向下游学习出来的最优维度。主线改为：用在线 RADIO teacher 替代缓存 PCA teacher，把 `1280d -> 128d` 压缩模块直接放进联合训练里，让压缩空间同时服务于混合特征场重建、学生特征学习和定位监督。  
实施顺序按你选定的方案执行：先做连续联合瓶颈并跑通主线，VQ/codebook 只作为第二阶段对照，不放进第一轮主线。

## Interfaces
主要改动集中在 `feature_field/dcff/radio_teacher.py`、`feature_field/train_impl.py`、`feature_extract/joint_radio.py`。

新增/调整配置接口：
- `teacher.mode: online_bottleneck`
- `teacher.target_dim: 128`
- `teacher.pca_init_dir: ...` 仅用于 warm start，不再作为最终监督空间
- `teacher.compress.hidden_dim: 256`
- `teacher.compress.sample_pixels: 1024`
- `teacher.compress.recon_cos_weight: 1.0`
- `teacher.compress.recon_l1_weight: 0.25`
- `loss.lambda_cross_scale_orth: 0.01`
- `loss.use_mask_aware_contrast: true`

checkpoint 新增：
- `compression_state`
- `compression_optimizer_state`

保留兼容：
- `teacher.mode: cached`
- `teacher.mode: online`

## Implementation Changes
### 1. Teacher 压缩模块
- 在 frozen RADIO 之后，为 fine 和 coarse 各自增加一套连续瓶颈。
- 编码器结构固定为：`base_proj(1280->128)` + `res_adapter(1280->256->128)`，输出相加后做归一化。
- `base_proj` 用现有 PCA 权重初始化；`res_adapter` 末层零初始化，保证起步接近当前 PCA/linear projection。
- 解码器结构固定为：`128->256->1280`，只作为训练期辅助分支，不作为最终下游表示。
- canonical teacher space 定义为压缩后的 `z_fine(128d)` 和 `z_coarse(128d)`；下游全部对齐到这个空间。
- teacher 自监督损失固定为 raw RADIO 重建：
  - `L_recon = 1.0 * cosine(dec(z), raw) + 0.25 * L1(dec(z), raw)`
  - 只在每张图每个尺度随机采样 `1024` 个空间位置计算，避免显存浪费。

### 2. Map / Hybrid Feature Field 联合训练
- Cambridge 主配置从 `cached + 64d` 切到 `online_bottleneck + 128d`。
- fine 分支从迭代 0 开始监督，coarse 分支按现有 warmup 机制延后开启。
- 高斯 latent、显式特征解码器、hash/grid、fine/coarse decoder 全部改为输出并对齐 `128d` teacher 空间。
- 显式高斯特征不再被当作最终语义表示；它只承担 geometry/latent warm start 作用。
- 几何训练固定两阶段：
  - 前 10% 迭代冻结 geometry，只训 latent、decoder、compressor
  - 后 90% 解冻 `xyz/scale/rotation/opacity`，学习率用当前配置的 `0.1x`
- 第一轮完整实验保持 densification 关闭，先排除几何扰动，把特征质量和条纹问题单独收敛。

### 3. Student / 定位联合训练
- student 双头输出从 `64d` 改为 `128d`。
- student teacher supervision 改为在线压缩后的 `z_fine / z_coarse`，不再读磁盘 PCA cache。
- map supervision 继续保留，但统一在同一个 `128d` 空间里做 student-map-teacher 三方对齐。
- 对比学习固定为跨分支对齐，不做同分支自对比：
  - `student_fine -> teacher_fine`
  - `student_coarse -> teacher_coarse`
  - `rendered_map_fine -> teacher_fine`
  - `rendered_map_coarse -> teacher_coarse`
- InfoNCE 规则固定：
  - positive：同一像素/同一射线对应位置
  - negatives：batch 内其他像素，且排除 `16px` 邻域
  - `temperature = 0.07`
  - `negatives/query = 512`
- 使用现有 STDLoc semantic mask 作为先验，但只用于 supervision，不作为网络输入：
  - coarse：同 mask 区域不作为负样本，跨区域负样本加权更高
  - fine：mask 边界和图像边缘像素加权更高，用来抑制条纹并增强几何细节
- 不在 v1 强推 coarse/fine “远离”；只保留轻量正交约束 `lambda_cross_scale_orth=0.01`，避免再次把两者一起训塌。

### 4. Phase-2 VQ / Codebook 对照
- 只有在连续瓶颈主线稳定后才做。
- 第一版只量化 `coarse` 分支，`fine` 保持连续。
- 不做 DF-3DGS 那种离线先压缩再训场景的流程；VQ 必须挂在当前联合训练图里。
- 目标是做研究对照，而不是替代第一轮主线。

## Test Plan
- 配置与梯度检查：
  - 日志必须显示 `teacher=online_bottleneck`、`feature_dim=128`
  - compressor 参数必须进入 optimizer
  - `cached` fallback 仍能正常跑
- 小规模过拟合：
  - 用 OldHospital 16-32 张训练图快速过拟合
  - 观察 teacher fine 条纹是否明显减弱
  - 观察 student fine/coarse 是否出现清晰差异
  - 观察 map decode 特征是否摆脱系统性条纹/塌缩
- 全量评测：
  - 与当前 `PCA-64 cached` 基线做同 split、同 geometry init 的 A/B
  - 记录 teacher recon cosine/L1、rendered-vs-teacher cosine/L1、student-vs-teacher cosine/L1
  - 跑现有 Cambridge 定位评测，保存定量结果
- 可视化产物：
  - teacher raw / compressed / reconstructed
  - map fine / coarse
  - student fine / coarse
  - PCA 可视化、通道 norm/std 统计
  - 全部保存到 `/root/ICLPose/result`

## Assumptions
- 已锁定默认路线：先连续联合瓶颈，后续再补 VQ 对照。
- 已锁定默认维度：`128d`。
- `64d` PCA 仅保留作 warm start 和兼容，不再作为主监督空间。
- 第一轮主线优先解决条纹、粗细分支塌缩、map 解码异常；不额外追求 coarse/fine 强分离。
- 训练时以 AMP 和尽量高的 batch 使用 4090 显存，最终 batch 以“不 OOM 且显存接近满载”为准，并写回保存的 config。
