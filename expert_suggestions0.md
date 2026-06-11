基于我对公开 `vfm` 分支核心代码的静态阅读，我认为你看到的现象**不是很奇怪**，反而是当前 formulation 下很容易出现的 **render-pose lock / self-fulfilling PnP**。我没有运行你的数据和训练权重，所以不能保证下面每一条都已在你的实验里触发；但有几处代码路径足以解释“GT render 很好、render pose 一扰动 final pose 几乎跟扰动一样”的现象。

## 1. 最核心原因：PnP 的 3D 点被当前 render pose 锁住了

你现在的关键路径大概是：

`query keypoint/feature` → 匹配到 `render keypoint/feature` → 在 `render_xy` 处采样 `rendered_depth` → 用 `render_pose_w2c` 反投影成 world XYZ → 用 `query_xy ↔ XYZ` 做 PnP。

代码里确实是这样构造的：`keypoint_feature_matches_to_pnp_matches()` 接收 `rendered_depth`、`camera` 和 `render_pose_w2c`，先在 `match.render_xy` 上采样 depth，然后调用 `backproject_depth_to_world(..., render_pose_w2c)`，最后生成 `QueryTo3DMatch(xy=match.query_xy, track_id=match.render_index, xyz=...)`。([GitHub][1])

这件事本身不一定错。**如果** matcher 能够把 query 点 (q_i) 匹配到 perturbed render 里同一个物理 3D 点的真实投影 (u_i)，那么 `render_depth(u_i) + render_pose` 反投影得到的 (X_i) 仍然可以是正确世界点，PnP 可以把位姿拉回 GT。

但现在的问题是，RADIO/VFM token 的匹配很可能没有提供这个“物理点对应的 render 投影残差”。一旦 matcher 输出的是“同屏幕位置/同语义区域/同 coarse cell 附近”的 `render_xy`，PnP 就会天然锁死在 render pose。

数学上写就是：

[
X_i = T_r^{-1}(K^{-1} u_i D_r(u_i))
]

PnP 解：

[
\min_T \sum_i | \pi(T X_i) - q_i |^2
]

如果匹配出来的 (u_i \approx q_i)，那么由构造可知：

[
\pi(T_r X_i) \approx u_i \approx q_i
]

也就是说，**render pose (T_r) 本身已经是一个低 reprojection error 解**。这时无论你用 PnP、RANSAC 还是从 render pose 初始化的 residual solver，都会倾向于返回接近 (T_r) 的结果。你看到的“最终误差 ≈ 扰动 render pose 误差”就是这个机制的直接表现。

GT render 时为什么看起来很好？因为 (T_r = T_{gt})，此时“同屏幕位置匹配”刚好也是正确的，所以这个退化被掩盖了。它验证的是“query 和 GT render 对齐时 VFM 能不能配上”，但没有验证“VFM 能不能从错误 render pose 中恢复正确的跨视角 2D residual”。

## 2. 代码里已经出现了这个现象的诊断痕迹

仓库 README 当前把主线定义为“从 frozen VFM dense tokens 学一个 compact/mapable/localizable subspace，用于 map-conditioned hypothesis verification”，并明确说这个 branch **不声称是新的 camera pose refinement solver**；固定 retrieval/matching/PnP/3DGS solver 可以作为 candidate generator 或 downstream handoff，但贡献是用 selected feature evidence 去 verify/reject hypotheses。([GitHub][2])

README 的 evaluation gates 也把 `rendered_pose` 定义为“围绕声明 init 的显式 rendered pose candidates”，并把 “Final Localization: fixed downstream solver handoff” 和 solver-free hypothesis verification 分开报告。([GitHub][2]) 这说明你现在把 single perturbed render 当作 reference 然后期望 PnP refine，其实已经偏离了 README 里更稳妥的定位：**VFM 更适合做候选位姿打分/拒绝，而不是单独承担 metric pose refinement。**

另外，你的 eval 代码里已经专门实现了 `_render_lock_diagnostics_from_rows()`，计算 `translation_error_m` 和 `render_translation_error_m` 的差，并统计 `locked_to_render_within_3cm_rate`。这基本就是你描述的症状。([GitHub][3])

## 3. 当前实现里最值得优先排查的 bug / 设计问题

### 3.1 `render_index` 被当成 `track_id`，但它不是稳定 3D landmark ID

`QueryTo3DMatch` 的核心字段包括 `xy`, `track_id`, `xyz`, `similarity`, `ratio` 等；后续还有 `deduplicate_pnp_matches()` 会按 `track_id` 去重。([GitHub][4])

但在 render-keypoint 路线里，`track_id=int(match.render_index)`。这个 `render_index` 本质是当前 render feature/keypoint/cell 的索引，不是跨 pose 稳定的 3D landmark ID。不同 render pose 下，同一个 index 对应的世界表面点会变；同一个物理点也不一定落在同一个 render index。([GitHub][1])

这会导致两个问题：

第一，PnP 的“3D track”语义不成立，去重、统计、confidence 聚合都会变得不可靠。

第二，你没有一个 pose-invariant 的 map anchor。render pose 一变，anchor 也跟着变，最终很容易变成“当前 render hypothesis 的自洽解释”。

### 3.2 匹配主要是 descriptor-space matching，几何约束很弱

`dual_softmax_keypoint_matches()` 对 query/render descriptors 做 normalize，然后用 `scores = qdesc @ rdesc.T`，再做 row/column softmax 和 mutual best selection。这个过程本身没有显式使用 epipolar、depth、visibility、pose residual 或固定 3D anchor 约束。([GitHub][1])

这对 GT-aligned render 没问题，因为正确匹配几乎是 identity。但一旦 render pose 有扰动，RADIO 这类 VFM 的语义不变性会让“同类区域、同结构区域、同屏幕附近区域”都很像，尤其在建筑、重复窗格、弱纹理、树、天空、墙面上。它给 PnP 的不是稳定几何 correspondences，而是语义/外观相似点。

### 3.3 local refinement 可能进一步强化“零残差”

`refine_render_keypoint_matches_by_local_correlation()` 是在 render 端局部搜索，让 render-side keypoint measurement 往局部 descriptor correlation 最大的位置移动。注释里也写的是 “Refine render-side keypoint measurements by local descriptor correlation”。([GitHub][1])

这类 refinement 对小误差有效，但如果 perturbed pose 导致真实对应点已经离开 local search window，它会在附近找一个最像的点；而 VFM 局部响应比较平滑，最后很可能仍然停在原始 render cell 附近。这样 PnP 看到的就是 (u_i \approx q_i) 或 (u_i) 没有正确 residual，位姿自然回到 render pose。

你代码里 `matcha_coarse_to_fine_keypoint_matches()` 默认 `fine_search_radius_px=8.0`。如果 focal 约 883 px，1° 旋转就大约是 (883 \times 0.01745 \approx 15.4) px 的屏幕位移，已经超过 8 px 搜索半径；你的默认 camera 字符串里也有 `883.0` 这个 focal 量级。([GitHub][5]) 这意味着即使扰动“看起来很小”，它在图像上可能已经超出 fine matching 的 basin。

### 3.4 residual solver 从 render pose 初始化，低残差时天然不会动

`solve_render_pose_delta_from_matches()` 明确从 `render_pose_w2c` copy 出初始 `pose`，用当前 `match.xyz` 和 `match.xy` 做 reprojection residual，然后只接受能降低 error 的 SE(3) update。([GitHub][6])

如果你的 matches 使得 render pose 已经有低 reprojection error，solver 不动是正确行为，不是优化器坏了。问题在于输入 correspondences 没有包含正确的 pose correction signal。

### 3.5 cache key 可能不含 render pose，扰动实验一定要查

`_load_or_render_rgb_depth_cache()` 在 `skip_existing` 且 cache 存在时直接返回 cached `rgb/depth/alpha`；而 `_render_token_cache_path(cache_dir, image_id, render_width, render_height)` 的 key 只有 `image_id` 和 render size。([GitHub][7])

如果你的 perturb 实验里 `record.image_id` 不变，但 render pose 改了，并且开了 `--skip_existing_render_rgb_depth` 或 `--skip_existing_render_tokens`，那就可能出现：

* depth 是旧 pose 的；
* RGB feature 是旧 pose 的；
* 但 PnP/backprojection 用的是新 render pose；
* 或 RGB/depth/feature 三者彼此不一致。

这会制造非常隐蔽的错误。建议你立刻把 render cache key 改成至少包含：

`image_id + render_width/height + renderer + camera intrinsics hash + render_pose_w2c hash + gaussian ply hash/mtime + radio_version + layer_name + selector/adaptor checkpoint hash`

在定位问题前，最好先关掉所有 `skip_existing_*` 并强制 dump 每个 query 的 render RGB/depth/pose matrix checksum。

### 3.6 adapter/sample 生成看起来主要是 GT-GT 对齐，不能训练 perturb residual

在 `build_render_rgb_keypoint_adapter_samples.py` 里，render RGB/depth 是用 `gt.pose_w2c` 渲染的，后面计算 `render_keypoint_reprojection_errors()` 时 `render_pose_w2c=gt.pose_w2c` 且 `query_pose_w2c=gt.pose_w2c`。([GitHub][7])

这类样本主要教模型“query 和 render 已经对齐时哪些 match 是好 match”。它并没有教模型：当 render pose 偏了 (1^\circ)、(3^\circ)、10cm、30cm 时，query 点应该对应到 render 图上的哪个 displaced pixel。换句话说，它学不到你现在最需要的 correction residual。

这点非常关键。GT-GT 训练/评估会让模型在零 baseline 上表现很好，但对 perturbed render 不一定有泛化能力。

### 3.7 depth/backprojection 的 convention 需要严格验证

`backproject_depth_to_world()` 假设 depth 是 metric camera z-depth：先 `cv2.undistortPoints`，再构造 `(x * depth, y * depth, depth)`，最后用 `pose_w2c` 逆变换到 world。([GitHub][1])

而 official 2DGS renderer 里调用的是 `gsplat.rasterization_2dgs(..., render_mode="RGB+ED")`，返回 rendered 第 4 个通道作为 depth；注释里称它是 expected depth。([GitHub][8]) 这不一定等价于一个干净的单表面 z-depth，尤其在透明/半透明、高斯混合、depth discontinuity、边界处会有 bias。它可能不是主因，因为 GT render 下你已经很好，但它会放大 perturbed pose 下的 PnP 噪声。

必须做一个 round-trip test：对每个 sampled `render_xy, depth`，反投影成 `X`，再用同一个 `render_pose_w2c` 和同一个 render camera 投影回去，误差应该接近 0。如果不是，说明 camera scale、depth convention 或坐标系有问题。

### 3.8 image/camera coordinate frame 也有风险

`bilinear_sample_feature_map()` 是按 `image_width/image_height` 把 image-coordinate keypoints 映射到 feature map 坐标。([GitHub][1]) 但 backprojection 用的是传入 `camera` 的 intrinsic，而不是 `image_width/image_height` 自动缩放 intrinsic。也就是说，如果 render RGB/depth 是 resize 后的尺寸，必须确保传给 backprojection 的 camera 已经是同一个 render canvas/scale 的 camera。

你的 eval 文件里有 `_render_canvas_camera_from_base()`，它会创建更大的 canvas camera 并平移 principal point，但不缩放 focal length。([GitHub][3]) 这类逻辑只要有一个地方 query feature、render feature、depth、camera、PnP camera 用了不同坐标系，就会出现 GT 情况勉强好、扰动后迅速坏的现象。

## 4. 我建议你先做的 6 个诊断实验

### 实验 A：identity-lock synthetic test

对 perturbed render，直接构造 synthetic matches：

* `query_xy = render_xy`
* `xyz = backproject(render_xy, render_depth, render_pose)`

然后跑 PnP / residual solver。预期结果应该几乎等于 perturbed render pose。如果成立，说明你的 PnP pipeline 本身会被“同屏幕位置匹配”锁死。这是验证主假设的最小实验。

### 实验 B：oracle render correspondence test

对每个真实 world point / anchor，用 perturbed render pose 投影得到 (u_r)，用 GT query pose 投影得到 (q)，构造 (q \leftrightarrow X) 或 (q \leftrightarrow backproject(D_r(u_r), T_r))。

如果这时 PnP 能回到 GT，说明几何/backprojection 基本没问题，主要问题是 matcher 给不出正确 (u_r)。

如果 oracle 仍然回不去，优先查 depth convention、camera scale、pose convention、render depth 是否是 z-depth。

### 实验 C：fixed-XYZ test

保留当前 matcher 给出的 `query_xy ↔ render_xy`，但不要用 perturbed render depth 反投影得到 XYZ。改用稳定 map landmark / Gaussian anchor / COLMAP track 的 canonical world XYZ。

如果 fixed XYZ 后明显改善，说明“render-depth-defined 3D anchor”是瓶颈。

如果仍不改善，说明 matcher 的 2D correspondence 本身不对。

### 实验 D：residual magnitude vs feature stride/search radius

对 GT world points，统计 perturbed pose 下：

[
\Delta u = \pi(T_{gt}X) - \pi(T_rX)
]

看它的 median / p90 / p95 是否超过 RADIO token stride、keypoint detector cell size、local search radius。若 1° already > 8 px，那么 fine stage 默认设置就是不可能稳定修正的。

### 实验 E：cache sanity

每个 query 保存：

* render pose hash；
* rendered RGB hash；
* rendered depth hash；
* render feature hash；
* cache path；
* 是否命中 cache；
* `render_pose_error`；
* `final_pose_error`。

如果同一个 cache path 被不同 pose 复用，先修 cache，不要继续调模型。

### 实验 F：render-lock metrics 作为主指标

你已有 `_render_lock_diagnostics_from_rows()`。把下面几个指标放到 summary 第一屏：

* `median_abs_pnp_minus_render_translation_error_m`
* `median_abs_pnp_minus_render_rotation_error_deg`
* `locked_to_render_within_3cm_rate`
* `median_final_minus_init_improvement`
* `pnp_residual_at_render_pose`
* `pnp_residual_at_final_pose`

如果 final residual 只比 render residual 小一点点，pose 不动就是合理的；问题在 matches。

## 5. 优化路线：先决定你到底要做 verifier 还是 refiner

### 路线一：如果主线是 VFM hypothesis verification

那就不要把单个 perturbed render 的 PnP refinement 当作核心指标。更合理的是：

1. 由 retrieval / pose lattice / prior 生成多个 candidate poses；
2. 对每个 candidate render RGB/VFM/depth/visibility；
3. 用 selected VFM evidence 给 candidate 打分；
4. 拒绝 false positives，保留 top-k；
5. handoff 给固定 solver，例如 local feature matcher、photometric/3DGS refinement、PnP with true landmarks。

这条路线和 README 当前叙事更一致：selected VFM 的贡献是“map-conditioned evidence”，不是 pose solver 本体。([GitHub][2])

### 路线二：如果你确实要做 pose refinement

那需要改 formulation，不能只靠当前 render-depth PnP。

优先改成：

**render pose 只用于 visibility / candidate selection，不用于定义 3D anchor 身份。**

更具体地说：

1. 建一个稳定 3D anchor map：每个 anchor 有 persistent `anchor_id / gaussian_id / track_id`、canonical world XYZ、normal、visibility stats、descriptor distribution、uncertainty。
2. render 时输出的不只是 RGB/depth，还要输出 `anchor_id_map`、`xyz_map`、`visibility/alpha/top_contributor/depth_variance`。
3. matcher 输出 `query_xy ↔ anchor_id` 或 `query_xy ↔ canonical XYZ`。
4. PnP 用 canonical XYZ，而不是“当前 render pose + 当前 render depth”临时生成的 XYZ。
5. render depth 只用于判断该 anchor 是否在当前 pose 可见、是否被遮挡、是否处在 depth boundary。

你的工具目录里已经有一些与 Gaussian contribution / anchor map 相关的脚本，例如 `build_ray_contributed_gaussian_vfm_field.py`、`build_stage_h2_gaussian_token_contribution_maps.py`、`build_stage_h2_raw_gaussian_anchor_map.py` 等，可以沿这个方向复用。([GitHub][9])

## 6. 对当前代码的具体改法建议

### 6.1 立刻修 cache key

把：

```python
_safe_image_stem(image_id) + f"_{render_width}x{render_height}.npz"
```

改成包含 pose hash：

```python
pose_hash = sha1(np.asarray(render_pose_w2c, np.float32).tobytes()).hexdigest()[:12]
camera_hash = sha1(np.asarray(camera.params, np.float32).tobytes() + f"{camera.width}x{camera.height}".encode()).hexdigest()[:8]
cache_name = f"{stem}_{render_width}x{render_height}_{renderer}_{pose_hash}_{camera_hash}_{layer_name}.npz"
```

RGB/depth cache 和 render feature cache 都要改。否则 perturb 实验很容易污染。

### 6.2 训练 perturb-aware adapter，而不是 GT-GT adapter

构造训练样本时，render pose 应该是：

[
T_r = T_{gt} \oplus \delta
]

query pose 仍是：

[
T_q = T_{gt}
]

label 不是“GT-GT 下 query/render keypoints 是否接近”，而是：

* 对固定 anchor (X)，render 投影 (u_r = \pi(T_r X))；
* query 投影 (q = \pi(T_{gt} X))；
* 训练 matcher 或 offset head 学 (u_r \rightarrow q) 的 correspondence / residual；
* 同时训练 inlier classifier 判断这个 anchor 是否可见、是否可靠、是否落在边界。

扰动分布要覆盖你的真实 init error，比如：

* rotation: 0.5°, 1°, 2°, 5°；
* translation: 2cm, 5cm, 10cm, 30cm；
* hard negatives: repeated windows、vegetation、sky、textureless walls、thin structures。

### 6.3 把 `render_index` 换成 persistent anchor id

如果你继续用 render feature grid，那么至少给每个 render pixel/cell 绑定一个 stable ID：

* top contributing Gaussian ID；
* COLMAP track ID；
* surfel ID；
* 2DGS disk/plane intersection anchor ID；
* 或 multiview aggregated VFM landmark ID。

`track_id` 应该是这个 stable ID，而不是 `render_index`。否则 PnP 去重和统计都没有正确语义。

### 6.4 增加“是否真的有 correction signal”的过滤

在 PnP 前统计：

[
|q_i - u_i|
]

如果绝大多数 match 的 `query_xy` 和 `render_xy` 差异接近 0，但 render pose 已经有明显扰动，那么这批 matches 不可能修正 pose。此时应该直接判定该 candidate “只能 verification，不能 refinement”，不要把它送进 PnP 期望 refinement。

### 6.5 让 VFM 做 coarse，metric refinement 用高分辨率局部特征

RADIO/VFM 的优势是语义稳定、跨域鲁棒、可做 map-conditioned evidence；它的弱点是 token 粗、局部几何不够尖锐。pose refinement 需要的是精确的 image residual。建议 pipeline 改成：

1. RADIO/VFM：候选 pose ranking、hard negative rejection、可见区域选择；
2. selected anchors：提供局部 maplet / mask / prior；
3. SuperPoint/ALIKED/LoFTR/DKM/MASt3R-like local matcher：给 pixel-level residual；
4. robust PnP / bundle adjustment / render-based optimizer：做最终 pose update。

如果坚持 pure VFM，则需要训练一个专门的 fine offset/refinement head，而不是直接用 frozen RADIO cosine similarity 做最终 metric matching。

## 7. 最可能的瓶颈排序

我按优先级排序如下：

1. **render-pose lock formulation**：只要 matcher 输出同屏幕/同 cell 附近的 `render_xy`，PnP 必然回到 render pose。
2. **GT-GT 训练/诊断掩盖问题**：GT render 下成功不代表有 perturb correction 能力。
3. **VFM descriptor 不提供足够精确的局部 residual**：尤其搜索半径小于扰动造成的像素位移时。
4. **cache key 不含 pose**：可能让 perturb 实验混入旧 render/depth/feature。
5. **`render_index` 不是 stable track id**：破坏 PnP match 的 landmark 语义。
6. **depth/camera convention**：expected depth、camera scale、canvas principal point 任一不一致都会放大误差。
7. **当前 README 主线与 pose-refinement 目标不一致**：如果论文主张是 hypothesis verification，就不要把 refinement failure 当成唯一瓶颈；如果目标改成 refinement，就必须换成 stable 3D anchors + perturb-aware correspondence learning。

## 8. 一个最小可行的下一版方案

我建议你下一版先别大改所有模型，做一个 minimal but decisive version：

1. 关闭 cache，或修 pose-aware cache。
2. 构造 perturb-aware eval：每个 query 渲染 5–20 个 pose candidates。
3. 对每个 render candidate 输出 `anchor_id_map + depth + alpha + top_contributor + uncertainty`。
4. 用 VFM 只做 candidate score 和 anchor reliability，不直接承担最终 residual。
5. 用 stable anchor XYZ 做 PnP。
6. 加一个小的 offset/inlier head，训练目标来自 perturbed render pose 与 GT query pose 的真实投影差。
7. summary 同时报告：

   * candidate ranking accuracy；
   * render-lock rate；
   * final-vs-init improvement；
   * oracle correspondence upper bound；
   * matcher upper bound；
   * depth/camera round-trip error。

这样你能很快判断瓶颈到底在 matcher、depth/camera、anchor representation，还是 VFM 本身不适合做 fine pose refinement。

我的判断是：你当前最主要的问题不是 PnP 算法本身，而是 **PnP 输入的 2D-3D correspondences 没有携带能把 perturbed render pose 拉回 GT 的几何残差信号**。先用上面的 oracle / fixed-XYZ / synthetic lock 三个实验把这一点证实，再决定是沿 README 的 verifier 路线推进，还是把系统重构成真正的 stable-anchor pose refiner。

[1]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/vfm/rendered_keypoint_matching.py "ICLPose/feature_extract/vfm/rendered_keypoint_matching.py at vfm · Arthurshen926/ICLPose · GitHub"
[2]: https://github.com/Arthurshen926/ICLPose/tree/vfm "GitHub - Arthurshen926/ICLPose at vfm · GitHub"
[3]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py "ICLPose/feature_extract/tools/vfm/eval_render_rgb_feature_keypoint_pose.py at vfm · Arthurshen926/ICLPose · GitHub"
[4]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/vfm/query_to_3d_matching.py "ICLPose/feature_extract/vfm/query_to_3d_matching.py at vfm · Arthurshen926/ICLPose · GitHub"
[5]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/vfm/matcha_coarse_to_fine.py "ICLPose/feature_extract/vfm/matcha_coarse_to_fine.py at vfm · Arthurshen926/ICLPose · GitHub"
[6]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/vfm/render_pose_residual_solver.py "ICLPose/feature_extract/vfm/render_pose_residual_solver.py at vfm · Arthurshen926/ICLPose · GitHub"
[7]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/tools/vfm/build_render_rgb_keypoint_adapter_samples.py "ICLPose/feature_extract/tools/vfm/build_render_rgb_keypoint_adapter_samples.py at vfm · Arthurshen926/ICLPose · GitHub"
[8]: https://github.com/Arthurshen926/ICLPose/blob/vfm/feature_extract/vfm/official_2dgs_renderer.py "ICLPose/feature_extract/vfm/official_2dgs_renderer.py at vfm · Arthurshen926/ICLPose · GitHub"
[9]: https://github.com/Arthurshen926/ICLPose/tree/vfm/feature_extract/tools/vfm "ICLPose/feature_extract/tools/vfm at vfm · Arthurshen926/ICLPose · GitHub"
