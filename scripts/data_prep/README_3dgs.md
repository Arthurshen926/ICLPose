# 新场景数据准备指南（7-Scenes & Cambridge）

本文档描述如何为 **7-Scenes (stairs)** 和 **Cambridge Landmarks (OldHospital)** 
两个场景完成完整的数据准备流程，使 ICLPose 项目能直接在这两个数据集上训练。

**不改动任何已有数据加载代码**，所有工作通过格式转换和配置文件实现。

---

## 整体流程

```
原始数据集
    │
    ├─ Step 1: 格式转换（本脚本）
    │          ↓
    │   dataset/{stairs,cambridge_OldHospital}/{seq-XX或train,test}/
    │          rgb/*.png  +  traj_w_c.txt
    │
    ├─ Step 2: 3DGS 重建（2DGS, SIGGRAPH 2024 — 几何精度最优）
    │          ↓
    │   dataset/{scene}/gaussian_splatting/point_cloud/final/point_cloud.ply
    │
    ├─ Step 3: 特征嵌入（feature_3dgs）
    │          ↓
    │   output/feature_3dgs/{scene}/fused_feature/checkpoint_best.pth
    │
    └─ Step 4: ICLPose 训练（train_v3.py）
               使用 configs/exp030_stairs.yaml 或 exp031_oldhospital.yaml
```

---

## Step 1: 格式转换

### 7-Scenes / stairs

```bash
python scripts/data_prep/convert_7scenes.py \
    --src /mnt/pool1/sqy/7scenes/stairs \
    --dst /home/yons/Projects/ICLPose/dataset/stairs \
    --mode symlink   # 使用符号链接，不复制数据
```

转换后目录结构：
```
dataset/stairs/
├── seq-01/rgb/frame_000000.png → traj_w_c.txt  (TestSplit)
├── seq-02/rgb/frame_000000.png → traj_w_c.txt  (TrainSplit)
├── seq-03/rgb/...
├── seq-04/rgb/...  (TestSplit)
├── seq-05/rgb/...
├── seq-06/rgb/...
└── camera_intrinsics.txt  (fx=525.505, fy=525.505, cx=320, cy=240)
```

TrainSplit: seq-02, seq-03, seq-05, seq-06  
TestSplit:  seq-01, seq-04

### Cambridge Landmarks / OldHospital

```bash
python scripts/data_prep/convert_cambridge.py \
    --src /home/yons/Projects/ICLPose/dataset/OldHospital \
    --dst /home/yons/Projects/ICLPose/dataset/cambridge_OldHospital \
    --mode symlink
```

转换后目录结构：
```
dataset/cambridge_OldHospital/
├── train/rgb/seq01_frame00001.png → traj_w_c.txt
├── test/rgb/seq01_frame00001.png  → traj_w_c.txt
└── camera_intrinsics.txt  (fx=fy=1673.27, cx=960, cy=540)
```

---

## Step 2: 3DGS 重建（2DGS —— 几何精度最优）

### 方案选择

**推荐方案：[2DGS (2D Gaussian Splatting)](https://github.com/hbb1/2d-gaussian-splatting)**（Huang et al., SIGGRAPH 2024）

> **论文**: *2D Gaussian Splatting for Geometrically Accurate Radiance Fields*  
> Binbin Huang, Zehao Yu, Anpei Chen, Andreas Geiger, Shenghua Gao

选择理由：
1. **几何精度最优**：2DGS 使用 2D surfel（盘片）原语替代 3D 椭球，曲面重建质量显著优于 3DGS/3DGS-MCMC。在 DTU、T&T 等基准上 Chamfer Distance 大幅降低
2. **对定位至关重要**：本项目通过 `feature_3dgs` 渲染深度 → 反投影到 3D 建立 2D-3D 对应。**几何精度直接决定深度精度 → 决定 3D 点质量 → 决定定位精度**
3. **STDLoc 已验证**：参考文献 STDLoc 同时实现了 3DGS 和 2DGS，实验表明 2DGS 在定位任务中表现更优
4. **PLY 格式已适配**：2DGS PLY 与标准 3DGS 唯一区别是 `scale_*` 从 3 维变为 2 维。  
   我们的 `GaussianFeatureModel.load_ply()` **已自动处理**——检测到 2 个 scale 时自动 pad 第 3 维（`scale_2 = -30`，即 `exp(-30) ≈ 0` 的零厚度平面），gsplat 1.0.0 渲染验证通过
5. **CLI 接口与原版 3DGS 相同**：`python train.py -s <source> -m <model>` 完全一致

**为什么不选其他方案**：
- ~~Scaffold-GS~~：锚点+偏移结构，PLY 格式完全不兼容 `GaussianFeatureModel`
- ~~3DGS-MCMC~~：PSNR 更高，但几何质量不如 2DGS；定位任务中**几何精度 > 渲染质量**
- ~~Mip-Splatting~~：抗锯齿改进显著，但几何精度提升有限

**适配细节**（已实现，无需额外操作）：
```
GaussianFeatureModel.load_ply():
  if scales.shape[1] == 2:   # 检测到 2DGS PLY
      pad scale_2 = -30      # exp(-30) ≈ 0，等效于零厚度 surfel
      → scales [N, 3]         # gsplat 1.0.0 正常渲染
```

---

### 安装 2DGS

```bash
# ⚠️ 在独立 conda 环境安装，不影响当前训练环境！
conda create -n gs python=3.8 -y
conda activate gs

# 克隆仓库（含子模块）
git clone https://github.com/hbb1/2d-gaussian-splatting --recursive
cd 2d-gaussian-splatting

# 安装 PyTorch（选择与你的 CUDA 版本匹配的版本）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 安装依赖
pip install -r requirements.txt

# 编译 CUDA 子模块
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
```

---

### 7-Scenes / stairs：重建

7-Scenes 已包含 COLMAP 稀疏重建（`stairs/sparse/0/`），可直接作为输入：

```bash
cd 2d-gaussian-splatting

# ── 重建 stairs ──
python train.py \
    -s /mnt/pool1/sqy/7scenes/stairs \
    -m /home/yons/Projects/ICLPose/dataset/stairs/gaussian_splatting \
    --iterations 30000 \
    --resolution 1 \
    --eval
```

**2DGS 关键参数说明**：
- `--iterations 30000`：室内小场景 3 万步即可收敛
- `--resolution 1`：原分辨率（640×480）
- `--eval`：自动使用 COLMAP 的 train/test 分割
- 2DGS 无需 `--cap_max`，通过自适应密度控制管理 Gaussian 数量

训练完成后，点云位于：
```
dataset/stairs/gaussian_splatting/point_cloud/iteration_30000/point_cloud.ply
```

**软链接到 `final/`**（项目配置约定的路径）：
```bash
mkdir -p /home/yons/Projects/ICLPose/dataset/stairs/gaussian_splatting/point_cloud/final/
ln -s \
    /home/yons/Projects/ICLPose/dataset/stairs/gaussian_splatting/point_cloud/iteration_30000/point_cloud.ply \
    /home/yons/Projects/ICLPose/dataset/stairs/gaussian_splatting/point_cloud/final/point_cloud.ply
```

---

### Cambridge Landmarks / OldHospital：COLMAP + 重建

Cambridge 只提供 NVM 重建，需先转换为 COLMAP 格式，再训练 2DGS。

#### 方法 A：使用 nerfstudio 自动处理（推荐，最省事）

```bash
pip install nerfstudio

# NVM → COLMAP（自动特征提取+匹配+重建）
ns-process-data images \
    --data /home/yons/Projects/ICLPose/dataset/OldHospital \
    --output-dir /tmp/colmap_oldhospital \
    --matching-method vocab-tree \
    --sfm-tool colmap
```

#### 方法 B：手动 NVM → COLMAP 转换

```bash
# 使用 kapture-localization 转换
pip install kapture kapture-localization

# 转 NVM → kapture
kapture_import_nvm.py \
    -v /home/yons/Projects/ICLPose/dataset/OldHospital/reconstruction.nvm \
    -o /tmp/kapture_oldhospital

# 转 kapture → COLMAP
kapture_export_colmap.py \
    -v /tmp/kapture_oldhospital \
    -o /tmp/colmap_oldhospital
```

#### 运行 2DGS 重建

```bash
cd 2d-gaussian-splatting

python train.py \
    -s /tmp/colmap_oldhospital \
    -m /home/yons/Projects/ICLPose/dataset/cambridge_OldHospital/gaussian_splatting \
    --iterations 50000 \
    --resolution 2 \
    --eval
```

**室外大场景参数建议**：
- `--iterations 50000`：室外场景更复杂，迭代次数更多
- `--resolution 2`：降采样到 960×540，节省显存和训练时间

---

## Step 3: 特征嵌入

3DGS 重建完成后，用 `feature_3dgs` 模块将语义特征嵌入 Gaussian：

```bash
# 示例：为 stairs 嵌入特征（可在后台运行，不影响主训练进程）
cd /home/yons/Projects/ICLPose

CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. nohup python -m feature_3dgs.train_raw_embedding \
    --scale all \
    --ply_path dataset/stairs/gaussian_splatting/point_cloud/final/point_cloud.ply \
    --feature_dir output/features_multiscale/stairs \
    --traj_path dataset/stairs/seq-02/traj_w_c.txt \
    --output_dir output/feature_3dgs/stairs_raw \
    --num_iters 5000 --grad_accum 4 \
    > logs/feature_embed_stairs.log 2>&1 &
```

---

## Step 4: 计算场景边界并启动训练

```bash
# 计算场景边界（更新到 configs/exp030_stairs.yaml 的 scene.bound 字段）
python scripts/data_prep/compute_scene_bound.py \
    --ply dataset/stairs/gaussian_splatting/point_cloud/final/point_cloud.ply

# 开始训练
python train_v3.py --config configs/exp030_stairs.yaml
```

---

## 关键数据格式对比

| 字段 | Replica room_0 | 7-Scenes stairs | Cambridge OldHospital |
|------|---------------|-----------------|----------------------|
| RGB 格式 | `rgb/rgb_N.png` | `frame-NNNNNN.color.png` → `rgb/frame_N.png` | `seqN/frameN.png` → `rgb/seqNN_frameN.png` |
| 位姿格式 | `traj_w_c.txt` (16 floats/line) | 每帧独立 `pose.txt` → 合并 | `dataset_train.txt` (xyz+quat) → 转换 |
| 位姿含义 | C2W (camera-to-world) | C2W ✓ | C2W ✓ |
| 深度 | ✓ | ✓ | ✗ |
| 图像尺寸 | 640×480 | 640×480 | 1920×1080 |
| fx/fy | 320.0 | 525.505 | 1673.27 |
| cx/cy | 319.5 / 239.5 | 320.0 / 240.0 | 960.0 / 540.0 |
| 场景类型 | 室内小场景 | 室内小场景 | 室外大场景 |
| COLMAP 重建 | 已有 | 已有 (`sparse/0/`) | 需从 NVM 转换 |

---

## 注意事项

1. **不影响正在训练的主进程**：所有转换脚本只读原始数据并写入新目录，  
   `--mode symlink`（默认）不复制文件，速度极快。

2. **3DGS 重建建议在空闲 GPU 上进行**，与主训练进程隔离：  
   ```bash
   CUDA_VISIBLE_DEVICES=1 python gaussian-splatting/train.py ...
   ```

3. **Cambridge 室外场景特殊说明**：  
   - 图像分辨率 1920×1080，batch_size 建议降到 8-16  
   - 场景尺度大（约 50m 范围），`pose_noise_trans_m` 建议设为 0.5m  
   - `normalize_translation: true` 有助于训练稳定性
