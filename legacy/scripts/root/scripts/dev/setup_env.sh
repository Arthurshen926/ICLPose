#!/bin/bash
# =============================================================================
# ICLPose 环境一键配置脚本
# 适配: 6× RTX 4090 (sm_89) + 系统 CUDA 11.6
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="iclpose"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  ICLPose 环境配置 (RTX 4090 适配)                ║"
echo "╚═══════════════════════════════════════════════════╝"
echo "  项目目录: $PROJECT_DIR"
echo ""

# ── Step 1: 删除旧环境 ──
echo "─── [1/8] 清理旧环境 ───"
conda deactivate 2>/dev/null || true
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "  删除旧 $ENV_NAME 环境..."
    conda env remove -n "$ENV_NAME" -y
    echo "  ✓ 已删除"
else
    echo "  无旧环境, 跳过"
fi

# ── Step 2: 创建新 conda 环境 + PyTorch ──
echo ""
echo "─── [2/8] 创建 conda 环境 + 安装 PyTorch ───"
conda create -n "$ENV_NAME" python=3.9 -y
echo "  ✓ conda env 创建完成"

# 从这里开始所有命令都在新环境中运行
# 使用 conda run 避免 activate 的 shell 问题
CONDA_RUN="conda run --no-banner -n $ENV_NAME"

echo "  安装 PyTorch 2.0.1 + CUDA 11.8..."
$CONDA_RUN pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118
echo "  ✓ PyTorch 安装完成"

# 验证 PyTorch
$CONDA_RUN python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')
print(f'  GPU OK:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
    print(f'  #GPUs:   {torch.cuda.device_count()}')
"

# ── Step 3: 安装核心 pip 依赖 ──
echo ""
echo "─── [3/8] 安装核心依赖 ───"
$CONDA_RUN pip install \
    numpy==1.23.5 \
    scipy==1.13.1 \
    pyyaml==6.0.2 \
    tqdm==4.67.1 \
    pillow==9.5.0 \
    matplotlib==3.9.4 \
    scikit-learn==1.6.1 \
    opencv-python==4.8.1.78 \
    tensorboard==2.14.0 \
    einops==0.3.0 \
    munch==4.0.0 \
    requests
echo "  ✓ 核心依赖完成"

# ── Step 4: 3D / 几何依赖 ──
echo ""
echo "─── [4/8] 安装 3D 几何依赖 ───"
$CONDA_RUN pip install \
    open3d==0.17.0 \
    plyfile==1.1.3 \
    trimesh==4.11.0
echo "  ✓ 3D 几何依赖完成"

# ── Step 5: 深度学习生态 ──
echo ""
echo "─── [5/8] 安装 DL 生态依赖 ───"
$CONDA_RUN pip install \
    kornia==0.6.12 \
    timm==0.6.11 \
    torchmetrics==0.6.0 \
    transformers==4.26.1 \
    wandb
echo "  ✓ DL 生态依赖完成"

# ── Step 6: gsplat (关键 - 需要 CUDA 编译) ──
echo ""
echo "─── [6/8] 安装 gsplat ───"
# gsplat 1.0.0 需要编译 CUDA 扩展
# 设置 CUDA arch 为 sm_89 (RTX 4090)
$CONDA_RUN bash -c "TORCH_CUDA_ARCH_LIST='8.9' pip install gsplat==1.0.0 --no-build-isolation" 2>&1 || {
    echo "  gsplat==1.0.0 安装失败, 尝试最新版本..."
    $CONDA_RUN bash -c "TORCH_CUDA_ARCH_LIST='8.9' pip install gsplat --no-build-isolation" 2>&1 || {
        echo "  ⚠ gsplat 编译安装失败, 尝试从源码..."
        $CONDA_RUN bash -c "TORCH_CUDA_ARCH_LIST='8.9' pip install git+https://github.com/nerfstudio-project/gsplat.git --no-build-isolation" 2>&1 || \
            echo "  ✗ gsplat 安装失败 - 需要手动处理"
    }
}
echo "  gsplat 安装步骤完成"

# ── Step 7: 自定义 CUDA 扩展 (diff-gauss, simple-knn, tiny-cuda-nn) ──
echo ""
echo "─── [7/8] 安装自定义 CUDA 扩展 ───"

# diff-gaussian-rasterization
echo "  [a] diff-gaussian-rasterization..."
if [ -d "$PROJECT_DIR/submodules/diff-gaussian-rasterization" ]; then
    cd "$PROJECT_DIR/submodules/diff-gaussian-rasterization"
    $CONDA_RUN pip install .
else
    # 尝试 pip 安装
    $CONDA_RUN pip install diff-gauss 2>/dev/null || {
        echo "  从 GitHub 安装 diff-gaussian-rasterization..."
        rm -rf /tmp/diff-gaussian-rasterization
        git clone --depth 1 --recursive https://github.com/graphdeco-inria/diff-gaussian-rasterization.git /tmp/diff-gaussian-rasterization
        cd /tmp/diff-gaussian-rasterization
        $CONDA_RUN bash -c "TORCH_CUDA_ARCH_LIST='8.9' pip install . --no-build-isolation" || \
            echo "  ⚠ diff-gaussian-rasterization 安装失败"
        cd "$PROJECT_DIR"
    }
fi

# simple-knn
echo "  [b] simple-knn..."
if [ -d "$PROJECT_DIR/submodules/simple-knn" ]; then
    cd "$PROJECT_DIR/submodules/simple-knn"
    $CONDA_RUN pip install .
else
    $CONDA_RUN pip install simple-knn 2>/dev/null || {
        echo "  从 GitHub 安装 simple-knn..."
        rm -rf /tmp/simple-knn
        git clone --depth 1 https://gitlab.inria.fr/bkerbl/simple-knn.git /tmp/simple-knn 2>/dev/null || \
            git clone --depth 1 https://github.com/camenduru/simple-knn.git /tmp/simple-knn
        cd /tmp/simple-knn
        $CONDA_RUN bash -c "TORCH_CUDA_ARCH_LIST='8.9' pip install . --no-build-isolation" || \
            echo "  ⚠ simple-knn 安装失败 (有 fallback, 非致命)"
        cd "$PROJECT_DIR"
    }
fi

# tiny-cuda-nn (仅 splatloc encoding 使用, 非核心训练必需)
echo "  [c] tiny-cuda-nn..."
$CONDA_RUN pip install ninja
$CONDA_RUN bash -c "TCNN_CUDA_ARCHITECTURES=89 pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch --no-build-isolation" 2>&1 || \
    echo "  ⚠ tiny-cuda-nn 安装失败 (仅 splatloc encoding 需要, 非核心)"

cd "$PROJECT_DIR"

# ── Step 8: detectron2 + mask2former + ODISE (SD 特征提取) ──
echo ""
echo "─── [8/8] 安装 detectron2 生态 (SD 特征提取) ───"

# fvcore + iopath (detectron2 依赖)
$CONDA_RUN pip install fvcore iopath pycocotools

# detectron2 预编译 wheel
echo "  [a] detectron2..."
$CONDA_RUN pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html 2>&1 || {
    echo "  预编译 wheel 不可用, 从源码编译..."
    rm -rf /tmp/detectron2_build
    git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2_build
    cd /tmp/detectron2_build
    $CONDA_RUN bash -c "TORCH_CUDA_ARCH_LIST='8.9' pip install . --no-build-isolation" || \
        echo "  ⚠ detectron2 安装失败"
    cd "$PROJECT_DIR"
}

# mask2former (通过 .pth 文件安装)
echo "  [b] mask2former..."
M2F_DIR="/opt/mask2former"
if [ ! -d "$M2F_DIR" ]; then
    git clone --depth 1 https://github.com/facebookresearch/Mask2Former.git "$M2F_DIR" 2>&1 | tail -3
fi
SITE_PKG=$($CONDA_RUN python -c "import site; print(site.getsitepackages()[0])")
echo "$M2F_DIR" > "$SITE_PKG/mask2former.pth"
echo "  ✓ mask2former 通过 .pth 安装"

# ODISE
echo "  [c] ODISE..."
ODISE_DIR="/opt/odise"
if [ ! -d "$ODISE_DIR" ]; then
    git clone --depth 1 https://github.com/NVlabs/ODISE.git "$ODISE_DIR" 2>&1 | tail -3
fi
echo "$ODISE_DIR" > "$SITE_PKG/odise.pth"
# ODISE 的额外依赖
$CONDA_RUN pip install open-clip-torch ftfy regex 2>/dev/null || true
echo "  ✓ ODISE 通过 .pth 安装"

# ══════════════════════ 验证 ══════════════════════
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  环境验证"
echo "═══════════════════════════════════════════════════════"

$CONDA_RUN python << 'PYEOF'
import sys, os
sys.path.insert(0, '.')
os.chdir(os.environ.get('PROJECT_DIR', '.'))

ok, warn, fail = 0, 0, 0

def check(name, stmt, critical=True):
    global ok, warn, fail
    try:
        exec(stmt, {})
        print(f'  ✓ {name}')
        ok += 1
    except Exception as e:
        short = str(e).split('\n')[0][:70]
        if critical:
            print(f'  ✗ {name}: {short}')
            fail += 1
        else:
            print(f'  ⚠ {name}: {short}')
            warn += 1

# Core
check("torch",        "import torch; assert torch.cuda.is_available()")
check("torchvision",  "import torchvision")
check("numpy",        "import numpy")
check("scipy",        "import scipy")
check("cv2",          "import cv2")
check("PIL",          "import PIL")
check("yaml",         "import yaml")
check("tqdm",         "import tqdm")
check("einops",       "import einops")
check("matplotlib",   "import matplotlib")
check("sklearn",      "import sklearn")
check("tensorboard",  "import tensorboard")

# 3D
check("gsplat",       "import gsplat")
check("plyfile",      "import plyfile")
check("open3d",       "import open3d")
check("trimesh",      "import trimesh")

# DL
check("kornia",       "import kornia")
check("timm",         "import timm")
check("munch",        "import munch")

# CUDA extensions
check("diff_gauss",   "import diff_gauss", critical=False)
check("simple_knn",   "from simple_knn._C import distCUDA2", critical=False)
check("tinycudann",   "import tinycudann", critical=False)

# ODISE ecosystem
check("detectron2",   "import detectron2", critical=False)
check("mask2former",  "import mask2former", critical=False)
check("odise",        "import odise", critical=False)

print(f'\n  结果: {ok} 通过, {warn} 警告(非关键), {fail} 失败')
if fail == 0:
    print('  ✓✓✓ 核心训练环境就绪! ✓✓✓')
else:
    print(f'  有 {fail} 个关键问题需要解决')
PYEOF

echo ""
echo "══════════════════════════════════════════════════════"
echo "  安装完成! 使用方式:"
echo "  conda activate $ENV_NAME"
echo "  python train.py --config configs/train_config.yaml"
echo "══════════════════════════════════════════════════════"
