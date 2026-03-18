# ICLPose 环境配置指南

> **硬件**: 6× RTX 4090 (sm_89, Ada Lovelace)
> **系统 CUDA**: 11.6 (不支持 sm_89, 编译需要 11.8)
> **最终环境**: Python 3.9 + PyTorch 2.0.1 + CUDA 11.8

---

## 安装顺序原则

```
conda 安装基础环境 (Python + nvcc)
    ↓
pip 安装 PyTorch (需要指定 cu118 index-url)
    ↓
pip 安装纯 Python 依赖
    ↓
源码编译 CUDA 扩展 (diff-gaussian-rasterization, simple-knn, tinycudann)
    ↓
源码编译 detectron2
    ↓
.pth 挂载 mask2former / ODISE
```

**为什么这个顺序？**
- conda 管理 Python 版本和 nvcc 编译器最可靠
- PyTorch 必须先装, 因为 CUDA 扩展编译依赖它
- 纯 Python 包顺序无关, 但建议在 CUDA 编译前装好 (减少依赖报错)
- CUDA 扩展必须在 PyTorch 之后, 因为用 `torch.utils.cpp_extension` 编译
- detectron2/mask2former/ODISE 放最后, 它们是 SD 特征提取的可选依赖

---

## Step 1: 创建 conda 环境

```bash
conda deactivate
conda env remove -n iclpose -y  # 如有旧环境
conda create -n iclpose python=3.9 -y
conda activate iclpose
```

## Step 2: 安装 CUDA 11.8 nvcc (编译用)

系统自带 CUDA 11.6 不支持 RTX 4090 的 sm_89 架构, 需要 11.8 的 nvcc:

```bash
conda install -c "nvidia/label/cuda-11.8.0" cuda-nvcc -y
# 验证:
$CONDA_PREFIX/bin/nvcc --version  # 应该显示 release 11.8
```

## Step 3: 安装 PyTorch

```bash
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
```

验证:
```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
# 期望: 2.0.1+cu118 11.8 True
```

## Step 4: 安装 pip 依赖

```bash
# 核心科学计算
pip install numpy==1.23.5 scipy==1.13.1 PyYAML tqdm Pillow==9.5.0 \
    matplotlib==3.9.4 scikit-learn==1.6.1 opencv-python==4.8.1.78 \
    tensorboard requests

# 深度学习工具
pip install einops==0.3.0 munch==4.0.0 kornia==0.6.12 timm==1.0.25 \
    torchmetrics==0.6.0 transformers==4.26.1 wandb

# 3D / 几何
pip install open3d==0.17.0 plyfile==1.1.3 trimesh==4.11.0

# 3DGS 渲染
pip install gsplat==1.4.0

# 编译工具
pip install ninja
```

> **注意**: 如果 `cv2` 或 `open3d` 报 `libGL.so.1` 缺失:
> ```bash
> apt-get install -y libgl1-mesa-glx libglib2.0-0
> ```

## Step 5: 编译安装 CUDA 扩展

**关键环境变量** (每个编译步骤都需要):
```bash
export CUDA_HOME=/usr/local/cuda-11.6          # headers + libs
export PATH=$CONDA_PREFIX/bin:$PATH            # nvcc 11.8
export TORCH_CUDA_ARCH_LIST="8.0+PTX"          # PTX 兼容 sm_89
```

> **为什么是 `8.0+PTX`？**
> - 系统 CUDA 11.6 的 headers 不认识 `compute_89`
> - `8.0+PTX` 编译出 sm_80 原生码 + PTX 中间码
> - PTX 在运行时 JIT 编译为 sm_89, 功能完全正常

### 5a. diff-gaussian-rasterization

```bash
git clone --depth 1 --recursive \
    https://github.com/graphdeco-inria/diff-gaussian-rasterization.git /tmp/diff-gaussian-rasterization
cd /tmp/diff-gaussian-rasterization
pip install . --no-build-isolation
cd -
```

项目里 import 的是 `diff_gauss`, 需要创建兼容包装:
```bash
SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
mkdir -p "$SITE_PKG/diff_gauss"
cat > "$SITE_PKG/diff_gauss/__init__.py" << 'EOF'
from diff_gaussian_rasterization import *
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
EOF
```

### 5b. simple-knn

```bash
git clone --depth 1 https://github.com/camenduru/simple-knn.git /tmp/simple-knn
cd /tmp/simple-knn
pip install . --no-build-isolation
cd -
```

### 5c. tiny-cuda-nn

```bash
TCNN_CUDA_ARCHITECTURES=80 \
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch \
    --no-build-isolation
```

> tinycudann 会提示 `built for lower compute capability (80) than system's (89)`, 这是性能警告, 功能正常。

## Step 6: 安装 detectron2 生态 (SD 特征提取, 可选)

### 6a. detectron2 依赖 + 源码编译

```bash
pip install fvcore iopath pycocotools

git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2_build
cd /tmp/detectron2_build
pip install . --no-build-isolation
cd -
```

### 6b. mask2former (.pth 挂载)

```bash
git clone --depth 1 https://github.com/facebookresearch/Mask2Former.git /opt/mask2former

SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
echo "/opt/mask2former" > "$SITE_PKG/mask2former.pth"
```

### 6c. ODISE (.pth 挂载)

```bash
git clone --depth 1 https://github.com/NVlabs/ODISE.git /opt/odise

SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
echo "/opt/odise" > "$SITE_PKG/odise.pth"

pip install open-clip-torch ftfy regex
```

## Step 7: 验证

```bash
cd /root/ICLPose
python scripts/verify_env.py
```

期望输出:
```
=== Result: 34 OK, 1 warnings, 0 failures ===
Environment is ready!
```

---

## 常见问题

### Q: `nvcc fatal: Unsupported gpu architecture 'compute_89'`
**A**: 编译时 nvcc 版本太低。确保 `$CONDA_PREFIX/bin/nvcc --version` 是 11.8, 且 `TORCH_CUDA_ARCH_LIST="8.0+PTX"`。

### Q: `cc1plus: fatal error: cuda_runtime.h: No such file or directory`
**A**: 需要设置 `CUDA_HOME=/usr/local/cuda-11.6` (含 headers), 同时 PATH 中 nvcc 11.8 优先。

### Q: `libGL.so.1: cannot open shared object file`
**A**: `apt-get install -y libgl1-mesa-glx libglib2.0-0`

### Q: gsplat 缺少 `rasterization_2dgs`
**A**: 需要 gsplat >= 1.4.0, 不是 1.0.0。

### Q: `diff_gauss` 找不到
**A**: 标准包名是 `diff_gaussian_rasterization`, 需要 Step 5a 中的兼容包装。

---

## 已安装包版本汇总

| 包 | 版本 | 安装方式 |
|----|------|----------|
| Python | 3.9.25 | conda |
| cuda-nvcc | 11.8.89 | conda (nvidia channel) |
| torch | 2.0.1+cu118 | pip (cu118 index) |
| torchvision | 0.15.2+cu118 | pip (cu118 index) |
| numpy | 1.23.5 | pip |
| scipy | 1.13.1 | pip |
| gsplat | 1.4.0 | pip |
| kornia | 0.6.12 | pip |
| timm | 1.0.25 | pip |
| open3d | 0.17.0 | pip |
| diff_gaussian_rasterization | 0.0.0 | 源码编译 (8.0+PTX) |
| simple_knn | 1.0.0 | 源码编译 (8.0+PTX) |
| tinycudann | 2.0 | 源码编译 (sm_80) |
| detectron2 | 0.6 | 源码编译 (8.0+PTX) |
| mask2former | - | .pth 挂载 (/opt/mask2former) |
| odise | - | .pth 挂载 (/opt/odise) |
