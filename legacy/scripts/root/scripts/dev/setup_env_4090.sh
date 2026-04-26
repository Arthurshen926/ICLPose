#!/bin/bash
# =============================================================================
# ICLPose 一键环境配置 + 验证 (RTX 4090 / PyTorch 2.1 + CUDA 12.1)
# =============================================================================
# 用法:  bash scripts/setup_env_4090.sh
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
ENV_NAME="iclpose"

echo "╔══════════════════════════════════════════════╗"
echo "║  ICLPose 环境配置 (6×RTX 4090)              ║"
echo "║  项目目录: $PROJECT_DIR"
echo "╚══════════════════════════════════════════════╝"

# ── 0. 检测系统 ──
echo ""
echo "─── [0/8] 系统检测 ───"
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader 2>/dev/null || echo "  [WARN] nvidia-smi 不可用"
echo ""

# ── 1. 确保 conda 可用 ──
echo "─── [1/8] 检测 conda ───"
if ! command -v conda &>/dev/null; then
    echo "  conda 未找到, 自动安装 Miniconda..."
    cd /tmp
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p $HOME/miniconda3
    rm miniconda.sh
    eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
    conda init bash
    echo "  ✓ Miniconda 安装完成"
    echo "  请运行: source ~/.bashrc 然后重新执行本脚本"
    exit 0
fi
echo "  conda: $(conda --version)"

# ── 2. 创建 conda 环境 ──
echo ""
echo "─── [2/8] 创建 conda 环境: $ENV_NAME ───"
if conda env list 2>/dev/null | grep -q "^${ENV_NAME} \|/${ENV_NAME}\$"; then
    echo "  环境 $ENV_NAME 已存在, 跳过创建"
else
    conda create -n "$ENV_NAME" python=3.9 -y
    echo "  ✓ 环境创建完成"
fi

# 激活环境
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"
echo "  Python: $(python --version) @ $(which python)"

# ── 3. 安装 PyTorch 2.1 + CUDA 12.1 ──
echo ""
echo "─── [3/8] 安装 PyTorch 2.1 + CUDA 12.1 ───"
if python -c "import torch; assert torch.cuda.is_available(); print(f'已安装: PyTorch {torch.__version__}')" 2>/dev/null; then
    echo "  PyTorch 已就绪, 跳过"
else
    pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 \
        --index-url https://download.pytorch.org/whl/cu121
    echo "  ✓ PyTorch 安装完成"
fi

python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA: {torch.version.cuda}')
print(f'  GPU count: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'  GPU 0: {torch.cuda.get_device_name(0)}')
    x = torch.randn(100, device='cuda')
    print(f'  CUDA compute test: OK')
"

# ── 4. 安装核心依赖 ──
echo ""
echo "─── [4/8] 安装核心依赖 ───"
pip install -q \
    numpy==1.24.4 \
    scipy==1.13.1 \
    PyYAML==6.0.1 \
    tqdm==4.66.1 \
    pillow==10.2.0 \
    opencv-python-headless==4.8.1.78 \
    matplotlib==3.8.2 \
    tensorboard==2.15.1 \
    scikit-learn==1.3.2 \
    trimesh==4.0.8 \
    plyfile==1.0.3 \
    kornia==0.7.1 \
    einops==0.7.0 \
    timm==0.9.12 \
    lpips==0.1.4 \
    wandb \
    munch \
    natsort \
    imageio \
    loguru \
    h5py \
    pandas \
    seaborn

# open3d 单独安装 (有时需要特殊处理)
pip install -q open3d==0.18.0 2>/dev/null || pip install -q open3d 2>/dev/null || \
    echo "  [WARN] open3d 安装失败, 非关键依赖"
echo "  ✓ 核心依赖安装完成"

# ── 5. 安装 gsplat ──
echo ""
echo "─── [5/8] 安装 gsplat ───"
if python -c "import gsplat; print(f'已安装: gsplat {gsplat.__version__}')" 2>/dev/null; then
    echo "  gsplat 已就绪, 跳过"
else
    echo "  安装 gsplat (可能需要编译, 耗时数分钟)..."
    pip install gsplat==1.4.0 2>/dev/null || {
        echo "  pip 直接安装失败, 从源码编译..."
        cd /tmp
        rm -rf gsplat_build
        git clone --branch v1.4.0 --depth 1 https://github.com/nerfstudio-project/gsplat.git gsplat_build
        cd gsplat_build
        pip install .
        cd "$PROJECT_DIR"
    }
    echo "  ✓ gsplat 安装完成"
fi
python -c "import gsplat; print(f'  gsplat: {gsplat.__version__}')"

# ── 6. 安装 Detectron2 + ODISE (SD 特征提取依赖) ──
echo ""
echo "─── [6/8] 安装 Detectron2 + ODISE 生态 ───"
ODISE_OK=true

# 关闭 set -e, 防止 ODISE 生态安装失败导致整个脚本退出
set +e

# 先安装构建依赖 (detectron2 需要 ninja/fvcore/iopath 才能构建)
pip install -q ninja fvcore iopath pycocotools 2>/dev/null

# detectron2
if python -c "import detectron2" 2>/dev/null; then
    echo "  detectron2 已安装"
else
    echo "  安装 detectron2 (从源码, 需要数分钟)..."

    # 方法1: 使用 python setup.py develop (避免 PEP 517 构建隔离)
    rm -rf /tmp/detectron2_build
    git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2_build 2>/dev/null
    cd /tmp/detectron2_build
    python setup.py develop 2>&1 | tail -10
    cd "$PROJECT_DIR"

    if ! python -c "import detectron2" 2>/dev/null; then
        echo "  setup.py develop 失败, 尝试 pip --no-build-isolation..."
        cd /tmp/detectron2_build
        pip install --no-build-isolation -e . 2>&1 | tail -10
        cd "$PROJECT_DIR"
    fi

    if python -c "import detectron2" 2>/dev/null; then
        echo "  ✓ detectron2 安装成功"
    else
        echo "  [WARN] detectron2 安装失败"
        ODISE_OK=false
    fi
fi

# ODISE 依赖
pip install -q \
    omegaconf==2.3.0 \
    panopticapi \
    ftfy \
    regex \
    open-clip-torch==2.20.0 \
    transformers==4.36.2 \
    2>/dev/null || true

# mask2former — 没有 setup.py, 需要 clone 后加入 Python path
if ! python -c "import mask2former" 2>/dev/null; then
    echo "  安装 mask2former (clone + path)..."
    M2F_DIR="/opt/mask2former"
    rm -rf "$M2F_DIR"
    git clone --depth 1 https://github.com/facebookresearch/Mask2Former.git "$M2F_DIR" 2>/dev/null
    if [ -d "$M2F_DIR" ]; then
        SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "$M2F_DIR" > "$SITE_PKG/mask2former.pth"
        # 编译 mask2former C++ 扩展 (如果有)
        if [ -f "$M2F_DIR/mask2former/modeling/pixel_decoder/ops/setup.py" ]; then
            cd "$M2F_DIR/mask2former/modeling/pixel_decoder/ops"
            python setup.py build_ext --inplace 2>&1 | tail -5 || true
            cd "$PROJECT_DIR"
        fi
    fi
    python -c "import mask2former; print('  ✓ mask2former 可导入')" 2>/dev/null || {
        echo "  [WARN] mask2former 安装失败"
        ODISE_OK=false
    }
fi

# ODISE — 也需要 clone + setup.py develop
if ! python -c "import odise" 2>/dev/null; then
    echo "  安装 ODISE (clone + develop)..."
    ODISE_DIR="/opt/odise"
    rm -rf "$ODISE_DIR"
    git clone --depth 1 https://github.com/NVlabs/ODISE.git "$ODISE_DIR" 2>/dev/null
    if [ -d "$ODISE_DIR" ]; then
        cd "$ODISE_DIR"
        # 使用 python setup.py develop 以避免构建隔离问题
        if [ -f "setup.py" ]; then
            python setup.py develop 2>&1 | tail -10 || \
                pip install --no-build-isolation -e . 2>&1 | tail -10 || true
        elif [ -f "pyproject.toml" ]; then
            pip install --no-build-isolation -e . 2>&1 | tail -10 || true
        else
            # 无安装配置, 使用 .pth 方式
            SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
            echo "$ODISE_DIR" > "$SITE_PKG/odise.pth"
        fi
        cd "$PROJECT_DIR"
    fi
    python -c "import odise; print('  ✓ odise 可导入')" 2>/dev/null || {
        echo "  [WARN] ODISE 安装失败"
        ODISE_OK=false
    }
fi

# 恢复 set -e
set -e

if [ "$ODISE_OK" = true ]; then
    echo "  ✓ Detectron2 + ODISE 生态安装完成"
else
    echo "  ⚠ ODISE 生态部分安装失败 (SD 特征提取可能受影响)"
    echo "    可单独运行: bash scripts/install_odise_ecosystem.sh"
fi

# ── 7. 创建 output 目录 ──
echo ""
echo "─── [7/8] 创建目录结构 ───"
cd "$PROJECT_DIR"
mkdir -p output/{features_multiscale/room_0,features_multiscale/room_0_val}
mkdir -p output/{feature_3dgs/room_0_raw/{fine_sd,fine_dino,mid,coarse}}
mkdir -p output/{2dgs_models/OldHospital,logs}
mkdir -p output/{exp032_fresh/checkpoints,exp032_fresh/logs}
echo "  ✓ 目录结构已创建"

# ── 8. 全局验证 ──
echo ""
echo "─── [8/8] 全局验证 ───"
python -c "
import sys
modules = {
    'torch': 'torch',
    'torchvision': 'torchvision',
    'gsplat': 'gsplat',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'cv2': 'opencv',
    'open3d': 'open3d',
    'kornia': 'kornia',
    'timm': 'timm',
    'einops': 'einops',
    'plyfile': 'plyfile',
    'PIL': 'pillow',
    'yaml': 'PyYAML',
    'sklearn': 'scikit-learn',
    'tensorboard': 'tensorboard',
}
ok, fail = 0, 0
for imp, name in modules.items():
    try:
        m = __import__(imp)
        v = getattr(m, '__version__', 'OK')
        print(f'  ✓ {name:20s} {v}')
        ok += 1
    except:
        print(f'  ✗ {name:20s} NOT INSTALLED')
        fail += 1

# ODISE 生态
for imp, name in [('detectron2', 'detectron2'), ('mask2former', 'mask2former'), ('odise', 'odise')]:
    try:
        __import__(imp)
        print(f'  ✓ {name:20s} OK')
        ok += 1
    except:
        print(f'  ⚠ {name:20s} NOT INSTALLED (SD feature extraction may fail)')
        fail += 1

import torch
print(f'')
print(f'  PyTorch CUDA: {torch.version.cuda}')
print(f'  GPU count: {torch.cuda.device_count()}')
for i in range(min(torch.cuda.device_count(), 6)):
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)}')

print(f'')
print(f'  {ok} packages OK, {fail} failed')
"

# 验证数据集
echo ""
echo "  数据集检查:"
for d in room_0 OldHospital stairs; do
    if [ -d "dataset/$d" ]; then
        echo "  ✓ dataset/$d 存在"
    else
        echo "  ✗ dataset/$d 不存在!"
    fi
done

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  ✓ 环境配置完成!                                         ║"
echo "║                                                          ║"
echo "║  激活环境:  conda activate $ENV_NAME                     ║"
echo "║  运行流水线: bash scripts/run_pipeline.sh check          ║"
echo "║  或分步执行:                                             ║"
echo "║    1. bash scripts/run_pipeline.sh 2dgs   (OldHospital)  ║"
echo "║    2. bash scripts/run_pipeline.sh extract (特征提取)    ║"
echo "║    3. bash scripts/run_pipeline.sh embed   (嵌入训练)    ║"
echo "║    4. bash scripts/run_pipeline.sh pose    (位姿训练)    ║"
echo "╚════════════════════════════════════════════════════════════╝"
