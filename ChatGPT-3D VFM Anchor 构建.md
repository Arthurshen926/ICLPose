# 3D VFM Anchor 构建

**User:** Anonymous  
**Created:** 6/2/2026 23:48:41  
**Updated:** 6/4/2026 15:22:58  
**Exported:** 6/4/2026 15:23:55  
**Link:** [https://chatgpt.com/c/6a1efb59-163c-83ea-ab8d-566649845b84](https://chatgpt.com/c/6a1efb59-163c-83ea-ab8d-566649845b84)  

## Response:
2026/6/4 15:06:17

我会把方案收敛到已重建 2DGS 的设定：重点设计 mapping 中的 token-to-disk/region 分配、anchor 融合与索引，定位只保留局部匹配和 PnP。

## Response:
2026/6/4 15:06:20

关键调整是：2DGS 已显式提供 oriented surface disk，所以不再推断 surface support；分配应基于 token footprint 与可见 disk 投影/面积重叠，并利用法向、切平面、disk 邻接做多视角 disk-group anchor。

## Response:
2026/6/4 15:11:49

> **构建简洁明了的回答**
>
> 我需要用中文给出最终答案，并确保适当引用文献，避免过度引用公式。在网上查询的事实需要引用，但设计性表述无需引用。对于2DGS、ULF-Loc和PnP的事实要引用。确保避免过度冗长，维持详细但简洁的方案描述，同时遵从用户不希望的表述方式。
>
> 已思考 9m 23s

## Response:
2026/6/4 15:15:41

是的，**对你这个任务，2DGS 通常比 3DGS 更合适**。原因不是 2DGS 渲染一定比 3DGS 更强，而是它的 primitive 更符合你要构建的 **VFM patch / region anchor**：

> **VFM token 是 2D region descriptor；2DGS primitive 是 3D 中的 oriented planar disk / surfel-like surface element。两者在“区域支撑”这个层面更一致。**

2DGS 的论文和官方实现都强调，它把 3D volume collapse 成 **2D oriented planar Gaussian disks** 来建模场景表面；相比之下，原始 3DGS 是 anisotropic 3D Gaussian splatting，更偏体积 / radiance primitive，虽然渲染高效，但不是天然 surface primitive。([arXiv](https://arxiv.org/html/2403.17888v3?utm_source=chatgpt.com))

所以你前面针对 3DGS 的 VFM assignment 思路需要调整，但不是推倒重来，而是从：

$$
\text{VFM token} \rightarrow \text{visible 3D Gaussian volume support}
$$

改成：

$$
\text{VFM token} \rightarrow \text{visible 2D Gaussian surface disk support}
$$

更具体地说：

> **3DGS 里要先把 Gaussian 解释成 surface；2DGS 里 primitive 已经是 surface，所以重点变成 token footprint 与 planar disk projected support 的可见重叠、纯净性和多视角一致性。**

下面我把一个完整的 **VFM-2DGS localization map** 方案重新整理出来，重点放在 **mapping**。定位阶段按照你的设定：**不考虑 coarse retrieval，不考虑 pose refinement，只做局部匹配 + PnP-RANSAC 初始定位**。

---

# 1. 总体结论：VFM-2DGS 应该怎么设计

我建议最终 map 不是：

```text
2D Gaussian disk + VFM feature
```

也不是：

```text
2D Gaussian disk center + descriptor
```

而是：

```text
VFM-2DGS Anchor =
    一组 2D Gaussian disks / disk fragments
    + surface support geometry
    + multi-view VFM token feature prototypes
    + observation quality
    + localization reliability score
```

也就是：

$$
A_k =
(\mathcal{D}_k,\rho_k,c_k,n_k,\Sigma_k,F_k,\mathcal{O}_k,Q_k)
$$

其中：

- $\mathcal{D}_k$：anchor 覆盖的一组 2DGS disks；
- $\rho_k$：每个 disk 在该 anchor 中的 support weight；
- $c_k$：anchor 的 3D surface support centroid；
- $n_k$：anchor 的平均 surface normal；
- $\Sigma_k$：anchor 的 3D / tangent-plane covariance；
- $F_k$：VFM feature，建议是 multi-prototype；
- $\mathcal{O}_k$：多视角 token observations；
- $Q_k$：用于定位的可靠性分数。

核心思想是：

> **单个 2DGS disk 是 surface primitive，但 VFM anchor 应该是 disk group / surface region。**

这是因为 VFM token 本身是 patch / region descriptor，DINOv2 这类 ViT feature 也常作为 patch-level dense feature 使用，而不是像 SuperPoint keypoint 那样对应高精度点状 proposal。([arXiv](https://arxiv.org/html/2304.07193v2?utm_source=chatgpt.com))

---

# 2. 2DGS 相比 3DGS 的优势与注意点

## 2.1 为什么 2DGS 更适合

2DGS 对这个任务更自然，主要有五点。

第一，**surface support 是显式的**。  
每个 2D Gaussian 是 3D 空间里的 oriented disk，有中心、切向轴、尺度、法向。你不需要像 3DGS 那样从 3D covariance 的最小特征向量去近似 surface normal。

第二，**token-to-support assignment 更清楚**。  
VFM token 有一个 2D footprint $\Omega_t$，2DGS disk 投影后也有一个 2D footprint $\Pi_v(d_i)$。两者可以直接做 projected support overlap，而不是看 Gaussian center 离 token center 近不近。

第三，**depth / normal purity 更容易计算**。  
因为 disk 是表面元，render buffer 里可以直接得到较可靠的 depth、normal、opacity、visible disk id。

第四，**更符合定位中的局部几何约束**。  
定位需要几何可定位性：法向稳定、表面支撑明确、跨视角一致。2DGS 本身就是为了改善 3DGS 的 surface reconstruction / geometry consistency 问题而提出的。([arXiv](https://arxiv.org/html/2403.17888v3?utm_source=chatgpt.com))

第五，**anchor 的 covariance 更有物理意义**。  
2DGS disk 的局部尺度在 tangent plane 上，正好可以构成 surface patch anchor 的不确定性，而不是一个体积椭球。

---

## 2.2 但不要把单个 2DGS disk 当成 VFM anchor

2DGS 更合适，但不意味着：

$$
\text{one 2D Gaussian} = \text{one VFM anchor}
$$

这仍然不对。原因是：

```text
VFM token 的支撑范围可能覆盖多个 disks；
一个大 disk 也可能跨越多个 token；
深层 VFM token 含有上下文；
同一个局部 surface 在多视角下可能对应不同 token footprint；
定位需要的是稳定 surface region，而不是孤立 disk。
```

因此正确粒度应该是：

```text
2DGS disk:
    rendering / surface primitive

VFM-2DGS anchor:
    localization / matching primitive
    = disk group / virtual disk cell group / local surface region
```

---

# 3. 相比 3DGS 版本，VFM 分配机制需要怎么调整？

你之前提到的 3DGS VFM 分配方法需要做这些调整。

## 3.1 不再需要“把 Gaussian 变薄”或估计 surface normal

3DGS 里常要检查：

$$
s_{\min} \ll s_{\text{mid}}, s_{\max}
$$

然后用最小尺度方向当 normal。但 2DGS 已经有 disk tangent plane 和 normal，所以这部分可以删掉。

2DGS disk 可以写成：

$$
d_i = (\mu_i, t_{i,1}, t_{i,2}, n_i, s_{i,1}, s_{i,2}, \alpha_i)
$$

其中：

$$
n_i = t_{i,1} \times t_{i,2}
$$

所以你的几何 support 可以直接由 disk 的 tangent basis 和 scale 给出。

---

## 3.2 assignment 从 center distance 改成 projected surface overlap

不要用：

$$
\|\pi(T_v \mu_i) - u_t\|
$$

作为主要分配依据。

应该用：

$$
\operatorname{Overlap}
\left(
\Pi_v(d_i),
\Omega_{v,t}
\right)
$$

其中：

- $\Pi_v(d_i)$：第 $v$ 个 view 下 2DGS disk 的 projected elliptical footprint；
- $\Omega_{v,t}$：VFM token 的 image footprint。

更进一步，不只是几何 overlap，还要乘以 visibility contribution：

$$
C_{v,i}(u) = T_{v,i}(u)\alpha_{v,i}(u)
$$

这里 $T_{v,i}(u)$ 是前方 accumulated transmittance，$\alpha_{v,i}(u)$ 是 disk $i$ 对 pixel $u$ 的 alpha contribution。2DGS / 3DGS 都可以 rasterize visibility-aware contribution；原始 3DGS 也正是依靠 visibility-aware splatting 实现高效渲染。([Repo Sam](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/?utm_source=chatgpt.com))

---

## 3.3 不再做 keypoint vote，而做 token-region responsibility

ULF-Loc 里 keypoint-consensus landmark sampling 的前提是 SuperPoint keypoints 是点状、高分辨率、带 detector score 的 localization proposals；它明确使用 Gaussian 与 2D keypoints 的多视角一致性来筛选可靠 landmarks。([arXiv](https://arxiv.org/html/2605.04730v1?utm_source=chatgpt.com))

VFM 没有这种 keypoint proposal，因此 VFM-2DGS 应该用：

```text
token footprint 覆盖了哪些 visible surface disks？
这些 disks 是否形成单一、连续、纯净的 surface support？
这个 support 在多视角下是否反复被相似 VFM token 观测到？
```

而不是：

```text
disk center 是否靠近 token center？
```

---

# 4. VFM-2DGS mapping：推荐完整流程

下面是我建议的 mapping pipeline。

```text
Input:
    reconstructed 2DGS model
    calibrated training images
    training camera poses
    camera intrinsics
    VFM feature extractor, e.g. DINOv2 mid-layer tokens

Output:
    VFM-2DGS anchor map:
        surface-region anchors
        feature prototypes
        visibility / quality metadata
        ANN feature index
        covisibility graph
```

---

# 5. Mapping Step 0：预处理 2DGS disks

每个 2DGS disk 有：

```text
Disk_i {
    id
    center μ_i ∈ R^3
    tangent axes t_i1, t_i2
    normal n_i
    scales s_i1, s_i2
    opacity α_i
    color / SH
}
```

首先做 disk-level 清洗：

```text
1. 去掉 opacity 太低的 disks
2. 去掉尺度异常大的 disks
3. 去掉尺度异常小且观测不稳定的 disks
4. 去掉 normal 与邻域明显不一致的 floating disks
5. 建立 disk adjacency graph
```

disk adjacency graph 后面很重要。两个 disks $d_i,d_j$ 可以连边，如果满足：

$$
\|\mu_i-\mu_j\| < \tau_d
$$

$$
n_i^\top n_j > \tau_n
$$

并且它们的 projected / tangent-plane support 有重叠或接近。

实际可以定义：

```text
adjacent(i, j) =
    centers close
    normals similar
    tangent-plane distance small
    scale-compatible
    often co-visible
```

---

# 6. Mapping Step 1：处理大 disk，必要时做 virtual subdivision

这是 2DGS 特有的一个重要点。

如果某些 2DGS disk 很大，它可能覆盖多个 VFM tokens。此时如果把整个 disk 分给某个 token，会污染 anchor support。

所以建议引入：

> **virtual 2DGS surface cells**

它们不改变 2DGS 渲染模型，只是为了 VFM mapping 和 localization 建一个更细的 surface support index。

对于大 disk $d_i$，在它的 tangent coordinates 上切成若干小 cell：

$$
x(a,b) = \mu_i + a t_{i,1} + b t_{i,2}
$$

其中：

$$
a \in [-s_{i,1}, s_{i,1}], \quad
b \in [-s_{i,2}, s_{i,2}]
$$

每个 virtual cell 继承：

```text
parent disk id
local center
normal
local tangent support
area weight
opacity / visibility metadata
```

什么时候需要切？

可以用一个简单规则：

```text
如果某个 disk 在训练视角中的 median projected area
    > 1.5 ~ 2.0 个 VFM token cell area
则做 virtual subdivision。
```

如果 disk 很小，则保持原样。

最终 mapping primitive 变成：

```text
SurfaceElement =
    original 2DGS disk
    or virtual cell split from a large disk
```

后面我统一称为 surface element $e_i$。它比“原始 disk”更适合和 VFM token 对齐。

---

# 7. Mapping Step 2：每个训练视角渲染 geometry / visibility buffers

对每个 training view $v$，用 2DGS renderer 额外输出：

```text
1. RGB image, optional
2. depth map D_v(u)
3. normal map N_v(u)
4. opacity / accumulated alpha A_v(u)
5. top-K visible surface element ids per pixel
6. per-element contribution C_{v,i}(u)
7. optional: boundary / depth discontinuity / normal discontinuity map
```

其中：

$$
C_{v,i}(u)
=
T_{v,i}(u)\alpha_{v,i}(u)
$$

表示 surface element $e_i$ 对 pixel $u$ 的可见贡献。

这一步是整个 VFM-2DGS mapping 的核心。你要避免做：

```text
token center backproject 到一个 3D 点
```

而要做：

```text
token footprint 内的 visible surface contribution 分布
```

---

# 8. Mapping Step 3：提取 VFM token，并定义 soft footprint

对训练图 $I_v$ 提取 VFM token：

$$
T_{v,t} = (\Omega_{v,t}, u_{v,t}, f_{v,t})
$$

其中：

- $\Omega_{v,t}$：token 的 image footprint；
- $u_{v,t}$：token center；
- $f_{v,t}$：VFM descriptor。

这里建议使用 **DINOv2 intermediate layer patch token** 作为 localization descriptor，而不是最语义化的最后层。DINOv2 本身支持 patch-level dense features，并且其 patch features 常用于 dense recognition / matching 类任务。([arXiv](https://arxiv.org/html/2304.07193v2?utm_source=chatgpt.com))

但是 $\Omega_{v,t}$ 不要简单看成一个 hard square。ViT token 的有效感受野不是严格方块，所以建议用 soft kernel：

$$
H_{v,t}(u)
=
\exp
\left(
-\frac{\|u-u_{v,t}\|^2}{2\sigma_t^2}
\right)
\cdot
\mathbf{1}[u \in \Omega_{v,t}^{+}]
$$

其中：

- $\Omega_{v,t}^{+}$：token cell 稍微扩张后的区域；
- $\sigma_t$：可以设为 $0.4 \sim 0.7$ 个 token stride。

这样比 hard box 更稳。

---

# 9. Mapping Step 4：计算 token purity

VFM token 经常跨边界。对于 localization，这类 token 要么丢弃，要么拆分，不要直接融合。

对每个 token 计算 purity：

$$
P_{v,t}
=
P_D \cdot P_N \cdot P_A \cdot P_B \cdot P_C
$$

各项可以这样定义。

## 9.1 Depth purity

$$
P_D
=
\exp
\left(
-\frac{\operatorname{Var}_{u \in \Omega_t}(D_v(u))}{\tau_D^2}
\right)
$$

depth variance 大，说明 token 可能跨前景 / 背景边界。

---

## 9.2 Normal purity

$$
P_N
=
\exp
\left(
-\frac{1-\|\frac{1}{|\Omega_t|}\sum_{u \in \Omega_t}N_v(u)\|}{\tau_N}
\right)
$$

normal 分散，说明 token 内部不是单一 surface patch。

---

## 9.3 Opacity purity

$$
P_A
=
\frac{1}{|\Omega_t|}
\sum_{u \in \Omega_t}
A_v(u)
$$

opacity 低的 token 往往对应空洞、边界、透明、反射或重建不稳定区域。

---

## 9.4 Boundary safety

用 depth edge / normal edge / image edge / SAM mask edge 都可以：

$$
P_B =
1 -
\operatorname{EdgeRatio}(\Omega_t)
$$

如果 token 内有明显边界，定位时应降权。

---

## 9.5 Component concentration

这个是 VFM-2DGS 特别有用的纯净性指标。

在 token footprint 里统计 visible surface elements，如果贡献主要集中在一个 connected component，则 token 纯净；如果分散在多个 disconnected components，则 token 跨边界。

$$
P_C
=
\max_m
\frac{
\sum_{i \in \mathcal{C}_m}
a_{v,t,i}
}{
\sum_i a_{v,t,i}
}
$$

其中 $\mathcal{C}_m$ 是 token 内 visible surface elements 的第 $m$ 个 connected component。

实际规则：

```text
P_C > 0.7:
    token 基本可用，取主 component

0.5 < P_C <= 0.7:
    token 可低权重使用，或尝试 split

P_C <= 0.5:
    丢弃，不用于 localization anchor
```

---

# 10. Mapping Step 5：token-to-surface-element responsibility

对 token $t$ 和 surface element $e_i$，计算贡献面积：

$$
a_{v,t,i}
=
\sum_{u \in \Omega_{v,t}^{+}}
H_{v,t}(u)
C_{v,i}(u)
$$

这个量表示：

> 在这个 token 的 soft footprint 内，surface element $e_i$ 实际贡献了多少可见 surface signal。

然后做双向归一化。

## 10.1 token 内归一化

$$
R_{i|t}^{v}
=
\frac{
a_{v,t,i}
}{
\sum_j a_{v,t,j} + \epsilon
}
$$

含义是：

> token 的可见 support 中，有多少来自 element $i$。

---

## 10.2 element 内归一化

$$
R_{t|i}^{v}
=
\frac{
a_{v,t,i}
}{
\sum_{u \in \Pi_v(e_i)}
C_{v,i}(u) + \epsilon
}
$$

含义是：

> element $i$ 的可见投影中，有多少被 token $t$ 覆盖。

---

## 10.3 最终 responsibility

$$
\rho_{v,t,i}
=
\left(R_{i|t}^{v}\right)^\lambda
\left(R_{t|i}^{v}\right)^{1-\lambda}
\cdot
P_{v,t}
\cdot
V_{v,i}
$$

其中：

- $\lambda$ 可以先设为 $0.5$；
- $V_{v,i}$ 是 visibility confidence；
- $P_{v,t}$ 是 token purity。

为什么需要双向项？

因为只用 $R_{i|t}$，一个大 token 会把 feature 分给太多 surface elements；只用 $R_{t|i}$，一个大 disk / element 可能被多个 tokens 反复污染。双向 responsibility 可以同时抑制这两类问题。

---

# 11. Mapping Step 6：把一个 token 的 support 切成 connected components

对一个 token，取：

$$
\mathcal{E}_{v,t}
=
\{e_i \mid \rho_{v,t,i} > \tau_\rho\}
$$

然后在 2DGS surface adjacency graph 上做 connected component：

$$
\mathcal{E}_{v,t}
\rightarrow
\mathcal{C}_{v,t}^{1},
\mathcal{C}_{v,t}^{2},
...
$$

这一步非常重要，因为 token footprint 可能覆盖多个 surface：

```text
椅子腿 + 地面
桌面边缘 + 背景
墙面 + 画框
窗户 + 外面
```

处理规则：

```text
只有一个 dominant component:
    生成一个 TokenSurfaceObservation

多个 component，但最大 component 占比 > 0.7:
    只保留最大 component，降低 confidence

多个 component 且都显著:
    丢弃该 token，不用于 localization mapping

如果有额外 mask / edge 信息:
    可以 split token 成多个 sub-observation，但 feature 权重要降低
```

生成一个 token-level 3D observation：

$$
O_{v,t}
=
(\mathcal{C}_{v,t},\rho_{v,t},f_{v,t},u_{v,t},\Omega_{v,t},q_{v,t})
$$

其中 observation quality：

$$
q_{v,t}
=
P_{v,t}
\cdot
\sum_{i \in \mathcal{C}_{v,t}}\rho_{v,t,i}
\cdot
Q_{\text{view}}
\cdot
Q_{\text{desc}}
$$

---

# 12. Mapping Step 7：计算 observation 的 3D 几何

每个 observation 是一个 surface element group。它的 3D center：

$$
c_O =
\frac{
\sum_{i \in \mathcal{C}}
\rho_i \mu_i
}{
\sum_{i \in \mathcal{C}}
\rho_i
}
$$

normal：

$$
n_O =
\operatorname{normalize}
\left(
\sum_{i \in \mathcal{C}}
\rho_i n_i
\right)
$$

covariance：

$$
\Sigma_O =
\frac{
\sum_{i \in \mathcal{C}}
\rho_i
\left[
(\mu_i-c_O)(\mu_i-c_O)^\top
+
R_i
\begin{bmatrix}
s_{i,1}^2 & 0 & 0 \\
0 & s_{i,2}^2 & 0 \\
0 & 0 & \epsilon
\end{bmatrix}
R_i^\top
\right]
}{
\sum_{i \in \mathcal{C}}
\rho_i
}
$$

这里：

$$
R_i = [t_{i,1}, t_{i,2}, n_i]
$$

因为 2DGS disk 是 planar disk，所以第三个方向上的方差 $\epsilon$ 应该很小。

这个 $\Sigma_O$ 后面有两个用途：

```text
1. 判断 anchor 尺度是否合适；
2. PnP-RANSAC 时给 reprojection threshold / inlier score 提供不确定性。
```

---

# 13. Mapping Step 8：多视角 observation association

现在你有很多 observations：

$$
O_{v,t}
$$

需要把来自不同训练视角、但对应同一块 surface region 的 observations 合并成 VFM-2DGS anchors。

一个 anchor：

$$
A_k =
(\mathcal{E}_k,\rho_k,c_k,n_k,\Sigma_k,F_k,\mathcal{O}_k)
$$

## 13.1 用 surface-element weighted IoU 做主关联

两个 support 的 weighted IoU：

$$
\operatorname{IoU}_E(O,A)
=
\frac{
\sum_i \min(\rho_{O,i},\rho_{A,i})
}{
\sum_i \max(\rho_{O,i},\rho_{A,i})+\epsilon
}
$$

这是 2DGS 版本最关键的 association signal。因为 2DGS disk / element id 是全局一致的，所以多视角 support association 比 3DGS 里更干净。

---

## 13.2 综合 association score

$$
S(O,A)
=
\alpha \operatorname{IoU}_E(O,A)
+
\beta \cos(f_O,F_A)
+
\gamma n_O^\top n_A
-
\eta \frac{\|c_O-c_A\|}{s_A}
$$

满足阈值则 merge，否则新建 anchor。

推荐优先级：

```text
surface-element IoU 最高优先
normal / center consistency 第二
feature similarity 第三
```

不要反过来。因为 VFM feature 可能有语义相似错配，例如多把椅子腿、多块墙砖、多扇窗户。

---

# 14. Mapping Step 9：anchor support fusion

当 observation $O$ merge 到 anchor $A$ 时，更新 surface support：

$$
\rho_{A,i}
\leftarrow
(1-\eta_s)\rho_{A,i}
+
\eta_s q_O \rho_{O,i}
$$

或者用累计权重：

$$
\rho_{A,i}
=
\frac{
\sum_{O \in \mathcal{O}_A}
q_O \rho_{O,i}
}{
\sum_{O \in \mathcal{O}_A}
q_O
}
$$

然后更新 anchor center：

$$
c_A =
\frac{
\sum_i \rho_{A,i}\mu_i
}{
\sum_i \rho_{A,i}
}
$$

normal：

$$
n_A =
\operatorname{normalize}
\left(
\sum_i \rho_{A,i}n_i
\right)
$$

covariance：

$$
\Sigma_A =
\frac{
\sum_i
\rho_{A,i}
\left[
(\mu_i-c_A)(\mu_i-c_A)^\top
+
\Sigma_i^{disk}
\right]
}{
\sum_i \rho_{A,i}
}
$$

surface area：

$$
\operatorname{area}(A)
=
\sum_i \rho_{A,i}\pi s_{i,1}s_{i,2}
$$

如果使用 virtual cells，则 area 用 cell area。

---

# 15. Mapping Step 10：VFM feature fusion

这里不要做简单平均，也不建议直接训练一个 high-dimensional feature-2DGS renderer。

ULF-Loc 对 3DGS localization 的一个重要观点是：直接把 descriptor 通过 alpha blending 融到 dense Gaussians 中容易引入 bias，所以它采用 geometry-weighted feature fusion 来得到 landmark descriptors。([arXiv](https://arxiv.org/html/2605.04730v1?utm_source=chatgpt.com))

对 VFM-2DGS，我建议：

```text
不把 feature 直接塞进每个 disk；
不优化 feature rendering loss；
而是在 anchor 层做 multi-view token observation fusion。
```

## 15.1 observation 权重

对 observation $O$，定义：

$$
w_O
=
q_O
\cdot
w_{\text{view}}
\cdot
w_{\text{area}}
\cdot
w_{\text{distinct}}
$$

其中 view weight 可以利用 2DGS normal：

$$
w_{\text{view}}
=
\left[
\max(0,n_O^\top d_{v,O})
\right]^\gamma
$$

这里 $d_{v,O}$ 是 surface 指向 camera 的 viewing direction。

但不要把 oblique view 全部丢掉。因为 query 可能也是斜视角，所以我更建议：

```text
feature_mean:
    用所有高质量 views 融合

view-bin prototypes:
    按 viewing direction 分 bin 存多个 prototype
```

---

## 15.2 feature mean

$$
F_A^{mean}
=
\operatorname{normalize}
\left(
\sum_{O \in \mathcal{O}_A}
w_O f_O
\right)
$$

---

## 15.3 feature prototypes

不要只存一个 feature。建议存 $K$ 个 prototypes：

```text
1. 按 view direction bin 分 prototypes
2. 或者对 observation features 做 online k-means
3. 或者两者结合
```

$$
F_A^{b}
=
\operatorname{normalize}
\left(
\sum_{O \in \mathcal{O}_A, \operatorname{bin}(O)=b}
w_O f_O
\right)
$$

local matching 时用：

$$
s(q,A)
=
\max_b
\cos(f_q,F_A^b)
$$

这样比单个 average descriptor 稳，因为 VFM token 本身会带有视角和上下文依赖。

---

# 16. Mapping Step 11：anchor quality scoring，用于筛选定位 anchors

不是所有 VFM-2DGS anchors 都适合定位。你需要选出 localizable anchors。

定义：

$$
Q_A
=
Q_{\text{view}}
\cdot
Q_{\text{purity}}
\cdot
Q_{\text{feature}}
\cdot
Q_{\text{distinct}}
\cdot
Q_{\text{geom}}
\cdot
Q_{\text{spread}}
$$

## 16.1 Multi-view support

$$
Q_{\text{view}}
=
\log(1+|\mathcal{O}_A|)
$$

至少要求：

```text
observation count >= 3
或 visible baseline 足够
```

---

## 16.2 Purity

$$
Q_{\text{purity}}
=
\frac{
\sum_{O \in \mathcal{O}_A}
w_O P_O
}{
\sum_{O \in \mathcal{O}_A}
w_O
}
$$

低 purity 的 anchor 不用于定位。

---

## 16.3 Feature stability

$$
Q_{\text{feature}}
=
\exp
\left(
-\frac{
\operatorname{Var}_{O\in\mathcal{O}_A}
[
\cos(f_O,F_A^{mean})
]
}{
\tau_f
}
\right)
$$

如果同一个 surface support 多视角 feature 很不一致，说明这个 anchor 对 VFM matching 不稳定。

---

## 16.4 Distinctiveness

$$
Q_{\text{distinct}}
=
1 -
\max_{B \neq A}
\cos(F_A^{mean},F_B^{mean})
$$

或者用 nearest-neighbor margin：

$$
Q_{\text{distinct}}
=
\cos(F_A,F_{nn1})
-
\cos(F_A,F_{nn2})
$$

重复墙面、重复窗户、重复椅子腿会被降权。

---

## 16.5 Geometry localizability

不适合定位的 anchor：

```text
太大的墙面区域
太小的孤立 disk
跨多个深度层的 region
normal variance 大
surface covariance 近似退化
只在一个方向可见
opacity 不稳定
动态物体上的 surface
```

几何质量可以写成：

$$
Q_{\text{geom}}
=
Q_{\text{area}}
\cdot
Q_{\text{normal}}
\cdot
Q_{\text{condition}}
$$

其中 area 需要在合理范围内：

```text
太小:
    center 不稳定，容易受 2DGS 噪声影响

太大:
    token descriptor 不是唯一局部区域，PnP 点代表性差
```

经验上，anchor 的 projected footprint 在常见训练视角里最好接近：

```text
0.5 ~ 3 个 VFM token cell
```

不要让一个 localization anchor 覆盖太大语义区域。

---

# 17. Mapping Step 12：spatial NMS 与 covisibility graph

选完 high-quality anchors 后，还要做 spatial NMS。

原因是 VFM token 是 patch grid，很多邻近 anchors 的 feature 会高度相似。如果全保留，在线 matching 会产生大量近重复候选。

做法：

```text
1. 按 Q_A 从高到低排序
2. 保留高分 anchor
3. 抑制附近且 support overlap 高的 anchors
4. 保证不同空间区域都有 anchors
5. 对大平面区域做上限限制
```

同时建立 covisibility graph：

$$
E_{ij}=1
$$

如果 anchors $A_i,A_j$ 经常在同一训练视角中被同时观测到。

边权可以是：

$$
w_{ij}
=
\frac{
|\mathcal{V}_i \cap \mathcal{V}_j|
}{
|\mathcal{V}_i \cup \mathcal{V}_j|
}
$$

这个 graph 在线定位时用于过滤空间不一致的 match set。

---

# 18. 最终 map 数据结构

建议保存三层。

## 18.1 Surface element layer

```text
SurfaceElement {
    id
    parent_2dgs_disk_id
    center: Vec3
    tangent1: Vec3
    tangent2: Vec3
    normal: Vec3
    scale1: float
    scale2: float
    area: float
    opacity_stats
    visibility_stats
    adjacency_ids
}
```

如果没有 virtual subdivision，SurfaceElement 就是原始 2DGS disk。

---

## 18.2 Token observation layer

```text
TokenSurfaceObservation {
    view_id
    token_id
    token_center_2d
    token_footprint
    vfm_feature
    surface_element_ids
    surface_element_weights
    center_3d
    normal_3d
    covariance_3d
    purity_score
    view_direction
    quality_score
}
```

这层可以不全部在线使用，但 mapping debug 非常有用。

---

## 18.3 VFM-2DGS anchor layer

```text
VFM2DGSAnchor {
    id

    # surface support
    surface_element_ids
    surface_element_weights
    center_3d
    normal_3d
    covariance_3d
    surface_area

    # VFM descriptors
    feature_mean
    feature_prototypes
    view_bin_features
    feature_variance

    # observations
    observation_ids
    observed_view_ids
    observation_count
    view_direction_stats

    # localization metadata
    quality_score
    purity_score
    distinctiveness_score
    stability_score
    covisibility_neighbors
}
```

在线 localization 主要用：

```text
center_3d
surface support
feature_prototypes
quality_score
covisibility_neighbors
```

---

# 19. Mapping 伪代码

```python
# Given:
#   2DGS disks
#   training images and poses
#   VFM extractor

surface_elements = preprocess_2dgs_disks(
    disks,
    split_large_disks=True,
    build_adjacency=True,
)

all_observations = []

for view in training_views:
    buffers = render_2dgs_buffers(
        view,
        surface_elements,
        outputs=[
            "depth",
            "normal",
            "opacity",
            "topk_element_ids",
            "element_contribution",
        ],
    )

    tokens = extract_vfm_tokens(view.image)

    for token in tokens:
        footprint = build_soft_token_footprint(token)

        purity = compute_token_purity(
            footprint,
            buffers.depth,
            buffers.normal,
            buffers.opacity,
            buffers.element_contribution,
        )

        if purity < PURITY_MIN:
            continue

        responsibilities = compute_token_element_responsibility(
            footprint,
            buffers.element_contribution,
            purity,
        )

        components = split_by_surface_connectivity(
            responsibilities,
            surface_elements.adjacency,
        )

        dominant = select_dominant_component(components)

        if dominant is None:
            continue

        observation = build_token_surface_observation(
            token_feature=token.feature,
            token_center=token.center,
            token_footprint=footprint,
            component=dominant,
            responsibilities=responsibilities,
            surface_elements=surface_elements,
            quality=purity,
        )

        all_observations.append(observation)

anchors = []

for obs in all_observations:
    candidate = find_best_anchor_by_surface_iou(
        obs,
        anchors,
    )

    if candidate is not None and association_score(obs, candidate) > ASSOC_THR:
        merge_observation_into_anchor(obs, candidate)
    else:
        anchors.append(create_new_anchor(obs))

anchors = compute_anchor_features_and_quality(anchors)

anchors = filter_low_quality_anchors(anchors)

anchors = spatial_nms_and_balance(anchors)

feature_index = build_ann_index(
    [anchor.feature_prototypes for anchor in anchors]
)

covisibility_graph = build_covisibility_graph(anchors)
```

---

# 20. 在线定位：不做 coarse retrieval，不做 refinement

按照你的设定，定位阶段只做：

```text
query VFM tokens
    ↓
direct local matching to VFM-2DGS anchors
    ↓
PnP-RANSAC
    ↓
pose output
```

这里默认你已经知道当前 query 属于这个局部 2DGS map，或者 map 本身不大。否则完全不做 coarse retrieval 时，大场景会有明显重复结构错配和计算量问题。

---

## 20.1 Query token extraction

输入 query image $I_q$，提取：

$$
Q_j = (\Omega_j,u_j,f_j)
$$

同样建议用 DINOv2 intermediate layer patch tokens。

可以先过滤 query tokens：

```text
1. 去掉图像边界 token
2. 去掉过于平坦 / 低纹理 / 低 saliency token
3. 去掉 feature 过于常见的 token
4. 保留 spatially spread 的 tokens
```

query 没有 depth 时，不能算 3D purity，但可以算：

```text
image edge crossing
VFM feature distinctiveness
token saliency
mask boundary safety, optional
```

---

## 20.2 Query token ↔ VFM-2DGS anchor matching

对每个 query token $Q_j$，在 anchor feature index 中检索 top-$K$ anchors：

$$
A_i \in \operatorname{TopK}
\left(
\max_b \cos(f_j,F_i^b)
\right)
$$

匹配分数：

$$
s_{j,i}^{feat}
=
\max_b \cos(f_j,F_i^b)
\cdot
Q_i
$$

其中 $Q_i$ 是 anchor localization quality。

过滤规则：

```text
1. top-1 similarity > τ_sim
2. top-1 / top-2 margin > τ_margin
3. anchor quality > τ_quality
4. one query token 最多保留 K 个 candidates
5. one anchor 不要被过多 query tokens 重复使用
```

此时得到候选匹配：

$$
m_{j,i}: u_j \leftrightarrow c_i
$$

注意：

$$
c_i
$$

只是 PnP solver 的代表点，不代表 anchor 本体退化成 3D point。

---

## 20.3 用 covisibility graph 做匹配集过滤

因为没有 coarse retrieval，feature-only matching 会有大量语义相似错配。

构造 match graph：

```text
node:
    candidate match m_{j,i}

edge:
    query tokens j,k 在图像上相近或属于同一局部区域
    且 anchors i,l 在 map covisibility graph 中相邻 / 共视
```

两个 matches：

$$
m_a=(Q_j,A_i), \quad m_b=(Q_k,A_l)
$$

兼容性：

$$
C(m_a,m_b)
=
\mathbf{1}
[A_i,A_l \text{ covisible}]
\cdot
\mathbf{1}
[\|u_j-u_k\| \text{ reasonable}]
$$

然后取最大 connected component 或 top graph cluster：

```text
candidate_matches = largest_consistent_component(match_graph)
```

这一步可以显著减少：

```text
墙面重复 patch
窗户重复 patch
椅子腿重复 patch
地面重复 patch
```

造成的错误匹配。

---

## 20.4 PnP-RANSAC 求位姿

把匹配转成 2D-3D correspondences：

$$
(u_j,c_i)
$$

求：

$$
T_{cw}
=
\arg\min_T
\sum_{(j,i)}
\rho
\left(
\operatorname{dist}
\left(
\pi(KT c_i),
\Omega_j
\right)^2
\right)
$$

这里推荐使用：

```text
P3P / EPnP inside RANSAC
```

RANSAC 采样时要加约束：

```text
1. 采样 anchors 空间上要分散
2. 不要全部来自同一平面
3. query tokens 在图像上也要分散
4. 不要多个样本来自高度重叠的 anchors
5. 优先采样 high-quality / high-distinctiveness anchors
```

---

## 20.5 RANSAC hypothesis scoring：必须用 projected support

虽然你不做 pose refinement，但 RANSAC 的 hypothesis verification 仍然应该用 support region，而不是只看 center reprojection。

对 pose hypothesis $T$，一个 match 的得分：

$$
S_{j,i}(T)
=
\lambda_p S_{\text{point}}
+
\lambda_r S_{\text{region}}
+
\lambda_f S_{\text{feature}}
+
\lambda_v S_{\text{visibility}}
$$

其中：

## point score

$$
S_{\text{point}}
=
\exp
\left(
-\frac{
\operatorname{dist}(\pi(KT c_i),\Omega_j)^2
}{
2\sigma_j^2
}
\right)
$$

注意这里是：

$$
\operatorname{dist}(\pi(KT c_i),\Omega_j)
$$

不是：

$$
\|\pi(KT c_i)-u_j\|
$$

因为 query observation 是 token region，不是 keypoint。

---

## region score

把 anchor 的 surface support 投影到 query image：

$$
\Pi_T(A_i)
=
\pi(KT\mathcal{D}_i)
$$

然后计算：

$$
S_{\text{region}}
=
\operatorname{IoU}
\left(
\Pi_T(A_i),
\Omega_j
\right)
$$

如果精确投影所有 disks 太慢，可以近似用：

```text
1. anchor projected ellipse
2. anchor covariance projected ellipse
3. anchor support bounding box
4. sampled surface elements
```

---

## feature score

$$
S_{\text{feature}}
=
\max_b \cos(f_j,F_i^b)
$$

---

## visibility score

根据 2DGS normal 和 camera direction 判断：

$$
S_{\text{visibility}}
=
\mathbf{1}
[
n_i^\top d_{T,i} > 0
]
\cdot
\operatorname{opacity\_confidence}(A_i)
$$

实际要注意 normal 朝向可能有符号约定问题。如果 2DGS normal 方向不稳定，可以先用：

$$
|n_i^\top d_{T,i}|
$$

再结合训练视角可见性统计。

---

## 总 inlier 判断

一个 match 是 inlier，如果：

```text
dist(projected center, token footprint) < τ_p
feature similarity > τ_f
projected support overlap > τ_r, if available
visibility valid
```

最终 pose 选：

$$
T^*
=
\arg\max_T
\sum_{(j,i)}
S_{j,i}(T)
$$

这个阶段仍然不是 pose refinement；只是比传统 PnP-RANSAC 更适合 VFM region anchor 的 hypothesis scoring。

---

# 21. 在线定位伪代码

```python
def localize_with_vfm_2dgs(query_image, K, anchor_map):
    query_tokens = extract_vfm_tokens(query_image)

    query_tokens = filter_query_tokens(
        query_tokens,
        min_saliency=True,
        remove_boundary_tokens=True,
        spatial_balance=True,
    )

    candidate_matches = []

    for q in query_tokens:
        anchors = anchor_map.feature_index.search(
            q.feature,
            top_k=K_ANCHORS_PER_TOKEN,
        )

        for a in anchors:
            sim = max_cosine(q.feature, a.feature_prototypes)
            score = sim * a.quality_score

            if sim > SIM_THR and feature_margin_ok(q, anchors):
                candidate_matches.append((q, a, score))

    candidate_matches = filter_by_covisibility_graph(
        candidate_matches,
        anchor_map.covisibility_graph,
    )

    best_pose = None
    best_score = -float("inf")
    best_inliers = None

    for _ in range(NUM_RANSAC_ITERS):
        sample = sample_spatially_diverse_matches(candidate_matches)

        pose = solve_pnp(
            object_points=[m.anchor.center_3d for m in sample],
            image_points=[m.query.center_2d for m in sample],
            K=K,
        )

        if pose is None:
            continue

        score, inliers = score_pose_with_region_support(
            pose,
            candidate_matches,
            K,
            use_projected_anchor_support=True,
        )

        if score > best_score:
            best_pose = pose
            best_score = score
            best_inliers = inliers

    return best_pose, best_inliers, best_score
```

---

# 22. 推荐初始参数

这些不是理论最优，但可以作为第一版系统的起点。

```text
Token footprint:
    soft window sigma = 0.4 ~ 0.7 token stride
    footprint expansion = 1.0 ~ 1.5 token cell

Token purity:
    P_min = 0.5 ~ 0.7
    dominant component ratio > 0.7

Responsibility:
    λ = 0.5
    ρ threshold = 0.02 ~ 0.05
    每个 token 最多保留 top 20 ~ 80 个 surface elements
    具体取决于 2DGS density

Anchor association:
    weighted surface IoU > 0.3
    normal similarity > 0.8
    feature cosine > 0.5, 仅作为辅助

Anchor quality:
    observation count >= 3
    feature variance 不超过全局中位数
    distinctiveness 排名前 30% ~ 50%
    projected area 约 0.5 ~ 3 个 VFM token cell

Online matching:
    每个 query token top-K anchors = 2 ~ 10
    cosine threshold 根据模型调，DINO 常需用 margin 而不是固定阈值
    ratio / margin test 必须开

PnP-RANSAC:
    minimal sample = 4
    reprojection threshold = 0.5 ~ 1.5 token stride
    inlier scoring 使用 dist-to-token-footprint
    不要使用严格 keypoint pixel threshold
```

最后一点很重要：**VFM token 的定位精度不可能按 SuperPoint keypoint 的 pixel-level threshold 来设**。它的 2D uncertainty 至少是 token stride 量级。

---

# 23. 这个方案和 ULF-Loc 的对应关系

可以这样理解：

| ULF-Loc / SuperPoint-3DGS | VFM-2DGS |
|---|---|
| SuperPoint keypoint | VFM patch token |
| keypoint detector score | token purity + feature distinctiveness |
| Gaussian center near keypoint | projected disk support overlaps token footprint |
| keypoint-consensus sampling | token-region-consensus sampling |
| single Gaussian landmark | 2DGS disk group / surface anchor |
| descriptor fusion on landmark | multi-view token prototype fusion on anchor |
| point reprojection error | token footprint distance + projected support overlap |

所以 VFM-2DGS 的核心不是“给 disk 投票”，而是：

$$
\boxed{
\text{VFM token 的区域观测}
\rightarrow
\text{可见 2DGS surface support}
\rightarrow
\text{多视角一致的 surface-region anchor}
}
$$

---

# 24. 我建议的最终 VFM-2DGS 系统结构

你现在已经重建好了 2DGS，我会这样落地：

```text
Mapping:
    1. 预处理 2DGS disks
    2. 必要时把大 disks 切成 virtual surface cells
    3. 对训练视角渲染 depth / normal / opacity / visible element contribution
    4. 提取 DINOv2 mid-layer VFM tokens
    5. 对每个 token 计算 purity
    6. 用 token footprint 与 visible 2DGS support 的双向 overlap 计算 responsibility
    7. 在 surface adjacency graph 上切 connected component
    8. 生成 TokenSurfaceObservation
    9. 多视角 observation association
    10. 融合成 VFM-2DGS surface-region anchors
    11. 计算 feature prototypes、quality、distinctiveness、covisibility graph
    12. 构建 ANN feature index

Localization:
    1. Query 图像提取 VFM tokens
    2. Query tokens 直接检索 VFM-2DGS anchors
    3. 用 feature margin + anchor quality + covisibility graph 过滤
    4. 用 anchor centroid + token center 做 PnP-RANSAC
    5. RANSAC scoring 使用 token footprint distance + projected support overlap
    6. 输出 pose
```

---

# 25. 最后给一个清晰判断

**2DGS 比 3DGS 更适合你这个 VFM anchor localization 任务。**

但正确用法不是：

$$
\text{2D Gaussian disk} + \text{VFM descriptor}
$$

而是：

$$
\boxed{
\text{VFM-2DGS anchor}
=
\text{multi-view consistent 2DGS surface-region support}
+
\text{VFM token feature prototypes}
}
$$

2DGS 已经解决了很多 3DGS 中“体积 Gaussian 如何解释成 surface patch”的问题；你需要重点解决的是：

```text
VFM token footprint 如何分配给 visible 2DGS surface support；
跨边界 token 如何过滤 / split；
多个训练视角下的 token observations 如何合并成稳定 surface-region anchors；
定位时如何用 region anchor 做 PnP hypothesis scoring。
```

一句话概括：

> **VFM-2DGS 的 mapping 核心应该是 token-region-to-surface-disk-group assignment，而不是 token-center-to-Gaussian-center voting。**

---

**Sources:**

- [2D Gaussian Splatting for Geometrically Accurate ...](https://arxiv.org/html/2403.17888v3?utm_source=chatgpt.com)
- [3D Gaussian Splatting for Real-Time Radiance Field ...](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/?utm_source=chatgpt.com)



---
Powered by [ChatGPT Exporter](https://www.chatgptexporter.com)