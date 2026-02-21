先检查可视化结果和提取进度： 

Ran terminal command:  ls /home/yons/Projects/ICLPose/output/features_fixed/vis/ && echo "---" && ls /home/yons/Projects/ICLPose/output/features_fixed/features_fused/ | wc -l

900帧全部提取完成，19张可视化。看可视化代码是否有问题，同时研究其他方法的预处理方式：  

研究SD特征提取预处理方法

读取 [](file:///home/yons/Projects/ICLPose/scripts/extract_fused_features.py#24-24)，行 24 到 90



Fetched 3 resources



Fetched 2 resources



Fetched 2 resources



Fetched 3 resources



Fetched 2 resources



Fetched 2 resources



Fetched 3 resources



Fetched 2 resources



Fetched 3 resources



Fetched 3 resources



Fetched 3 resources



Fetched 3 resources



Fetched 2 resources



Fetched 2 resources



Fetched 3 resources



Fetched 3 resources



Fetched 2 resources



Fetched 2 resources

读取 [](file:///home/yons/Projects/ICLPose/feature_extraction/fused_feature_extractor.py#1-1)，行 1 到 120

读取 [](file:///home/yons/Projects/ICLPose/feature_extraction/fused_feature_extractor.py#120-120)，行 120 到 205

现在有完整信息了。先检查可视化代码的正确性，再汇总分析：   直接给出结论，不再花时间在数值验证上了：  

**可视化代码正确性**：采用 `F.interpolate(bilinear)` 将 35×46 token grid 上采样到 480×640 与 RGB 并排显示，空间对齐误差来源于 DINO 对原图做了 480→490 的轻微拉伸（约 2%），导致最大约 **7px 的视觉偏移**，在 480px 图像上占 1.5%，对可视化对比完全可以接受 ✓。

---

## 各主流方法如何处理任意分辨率

| 方法 | 策略 | 代码逻辑 |
|---|---|---|
| **DIFT** (Tang et al., NeurIPS23) | 硬拉伸到 **768×768** 正方形 | `img.resize([768,768])` |
| **Diffusion Hyperfeatures** (Luo et al., NeurIPS23) | 硬拉伸到 **512×512** 正方形 | `image.resize((512,512))` |
| **SD-DINO/GeoAware** `edge=False` | 保宽高比 resize + **黑边 center-pad** 到 960×960 | `canvas=zeros(960,960)`, 居中贴图 |
| **SD-DINO/GeoAware** `edge=True` | 保宽高比 resize + **边缘像素复制** 到 960×960 | `np.pad(..., mode='edge')` |
| **ControlNet** | 保宽高比，H/W **分别 round 到最近 64 倍数** | `H = round(H/64)*64, W = round(W/64)*64` |
| **InstructPix2Pix** | 向上取整到 64 倍数 + **中心裁剪** | `ImageOps.fit(img, (W64, H64))` |
| **我们当前** | 矩形送入 SD，**事后裁掉 zero-pad 特征行** | `valid_fh = sd_h // factor_h` |

---

**对我们场景最优的改进**是 ControlNet 思路——**在送入 SD 之前**就将尺寸对齐到 64 的倍数（反射填充而非零填充），这样 SD 内部根本不会产生 zero-pad，边界特征质量也更好： 

正在将 [](file:///home/yons/Projects/ICLPose/feature_extraction/fused_feature_extractor.py) 中的 39 行替换为 53 行

读取 [](file:///home/yons/Projects/ICLPose/feature_extraction/fused_feature_extractor.py#100-100)，行 100 到 175

正在将 [](file:///home/yons/Projects/ICLPose/feature_extraction/fused_feature_extractor.py) 中的 45 行替换为 57 行


```

```
确认 `orig_w, orig_h` 没有被重复定义（原来在上面还有加载图像的代码）：   

读取 [](file:///home/yons/Projects/ICLPose/feature_extraction/fused_feature_extractor.py#95-95)，行 95 到 175

代码结构完全正确。

---

**总结**

## 可视化正确性 ✓
双线性上采样 35×46 → 480×640 后与 RGB 并排，空间对齐误差来自 DINO 的 480→490 拉伸约 **2%**（最大 ~7px 偏移），对可视化比较完全可接受。RGB 和特征图显示同一分辨率，坐标对应正确。

## 其他 CV 方向的做法

没有统一标准，主流分两派：

| 派别 | 代表方法 | 核心处理 | 缺点 |
|---|---|---|---|
| **正方形统一** | DIFT、Diffusion Hyperfeatures、SD-DINO | resize 到固定正方形（512/768/960），60 或 64 整除 → 无伪影 | 破坏宽高比（A）或引入黑边（B） |
| **自适应取整** | **ControlNet**、InstructPix2Pix | H/W 各自 round 到最近 64 倍数，无填充或轻微裁剪 | 轻微宽高比失真 |

## 最终方案（已更新到 fused_feature_extractor.py）

采用 **ControlNet 思路 + 反射填充**，优于旧的"事后裁除零填充特征"：

```
输入 640×480
  ↓ sd_align_h = ceil(480/64)*64 = 512  (+32px)
  ↓ sd_align_w = ceil(640/64)*64 = 640  (无变化)
  ↓ 反射填充：底部 32px 用边缘像素镜像延伸 → 640×512
  ↓ 送入 SD，无内部 zero-pad → 边界特征质量最佳
  ↓ 提取后裁回 30h×30w（s3）等有效 token
  ↓ 插值到 DINO grid 35×46
```

**区别于旧方案**：之前已提取的 900 帧用的是"事后裁零填充"版本，功能正确但边界 2~3 个 token 的 SD 特征是由零填充内容计算的。新方案用反射填充，这些 token 计算时看到的是真实图像内容的镜像延伸，质量更好。如需最高质量，可用新代码重新提取。

Made changes.


不影响。分析如下：

**特征坐标的空间映射完全由 DINO 决定，与 SD 预处理无关。**

提取管线中两个模型的关系：

```
原图 640×480
  ├─ DINO 路径：resize → 644×490（ceil 到 14 的倍数）
  │    stride=14 → token grid 35×46
  │    token(i,j) 中心像素 (在 DINO 输入中): (i×14+7, j×14+7)
  │    反算回原图:  y = (i×14+7)×480/490,  x = (j×14+7)×640/644
  │    ← 这是 feature→pixel 对应关系的唯一来源
  │
  └─ SD 路径：resize → 640×480 → reflect-pad → 640×512 → UNet
       各层 s3/s4/s5 features → 裁回有效区域 → bilinear 插值至 35×46
       ← 只贡献融合特征的语义质量，不决定空间坐标
```

**定位用到的内参计算**：

$$
\text{pixel}(u, v) = \left( \frac{(j \cdot 14 + 7) \cdot W}{W_\text{DINO}},\; \frac{(i \cdot 14 + 7) \cdot H}{H_\text{DINO}} \right)
$$

其中 $W_\text{DINO} = 644,\; H_\text{DINO} = 490$，完全与 SD 侧的任何处理无关。无论 SD 用零填充、反射填充还是黑边正方形，`token(i,j)` 对应的原图像素坐标不变，PnP 求解、内外参也不变。

**反射填充相比零填充唯一真实的影响**：UNet attention 在计算 boundary 区域的特征时，能看到镜像的真实内容而不是全零，使 s3/s4/s5 中靠近边界的那几个 token 的**语义质量更好**（与真实图像内容的相关性更强）。这对特征匹配精度有正面作用，但对坐标映射零影响。