#!/bin/bash
# =============================================================================
# 修复 setup_env_4090.sh 中失败的步骤并验证全流程
# =============================================================================
# 用法:  conda activate iclpose && bash scripts/fix_and_resume.sh
# =============================================================================
set +e  # 不退出, 逐一处理

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "╔══════════════════════════════════════════╗"
echo "║  ICLPose 修复 + 验证脚本                ║"
echo "╚══════════════════════════════════════════╝"

# ── 验证基础环境 ──
echo ""
echo "─── [1/6] 验证基础环境 ───"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')
import gsplat; print(f'  gsplat:  {gsplat.__version__}')
" || { echo "[FATAL] PyTorch/gsplat 不可用"; exit 1; }
echo "  ✓ 基础环境 OK"

# ── 修复 open3d ──
echo ""
echo "─── [2/6] 修复 open3d ───"
if python -c "import open3d" 2>/dev/null; then
    echo "  open3d 已安装, 跳过"
else
    echo "  安装 open3d..."
    pip install open3d==0.18.0 2>&1 | tail -5
    if ! python -c "import open3d" 2>/dev/null; then
        echo "  0.18.0 失败, 尝试最新版..."
        pip install open3d 2>&1 | tail -5
    fi
    python -c "import open3d; print(f'  ✓ open3d {open3d.__version__}')" 2>/dev/null || \
        echo "  ⚠ open3d 安装失败 (非关键依赖, 不影响训练)"
fi

# ── 安装 detectron2 ──
echo ""
echo "─── [3/6] 安装 detectron2 ───"
if python -c "import detectron2; print(f'  已安装: {detectron2.__version__}')" 2>/dev/null; then
    echo "  跳过"
else
    # 先确保构建依赖
    pip install -q ninja fvcore iopath pycocotools 2>/dev/null

    # 关键: 使用 python setup.py develop, 不走 pip 构建隔离
    echo "  clone + python setup.py develop (避免 pip 构建隔离找不到 torch)..."
    rm -rf /tmp/detectron2_build
    git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2_build
    cd /tmp/detectron2_build

    # 方法1: setup.py develop
    echo "  [尝试 1] python setup.py develop..."
    python setup.py develop 2>&1 | tail -15
    cd "$PROJECT_DIR"

    if ! python -c "import detectron2" 2>/dev/null; then
        # 方法2: pip --no-build-isolation
        echo "  [尝试 2] pip --no-build-isolation..."
        cd /tmp/detectron2_build
        pip install --no-build-isolation -e . 2>&1 | tail -15
        cd "$PROJECT_DIR"
    fi

    if ! python -c "import detectron2" 2>/dev/null; then
        # 方法3: 直接 build + install
        echo "  [尝试 3] python setup.py build install..."
        cd /tmp/detectron2_build
        python setup.py build install 2>&1 | tail -15
        cd "$PROJECT_DIR"
    fi

    if python -c "import detectron2; print(f'  ✓ detectron2 {detectron2.__version__}')" 2>/dev/null; then
        echo "  detectron2 安装成功!"
    else
        echo "  ✗✗✗ detectron2 安装失败 ✗✗✗"
        echo ""
        echo "  === 完整错误日志 (请复制给我) ==="
        cd /tmp/detectron2_build
        python setup.py develop 2>&1 | tail -50
        cd "$PROJECT_DIR"
        echo "  === 日志结束 ==="
        echo ""
        echo "  常见排查:"
        echo "    gcc --version  (需要 gcc 9-12)"
        echo "    nvcc --version (需要 CUDA toolkit)"
        echo "    python -c 'import torch; print(torch.version.cuda)'"
    fi
fi

# ── 安装 ODISE 周边 ──
echo ""
echo "─── [4/6] 安装 ODISE 周边依赖 ───"
pip install -q \
    omegaconf==2.3.0 \
    panopticapi \
    ftfy \
    regex \
    open-clip-torch==2.20.0 \
    transformers==4.36.2 \
    2>/dev/null
echo "  ✓ 周边依赖 OK"

# ── 安装 mask2former ──
echo ""
echo "─── [5/6] 安装 mask2former + ODISE ───"

# mask2former: 没有 setup.py, 用 .pth 文件添加到 Python path
if python -c "import mask2former" 2>/dev/null; then
    echo "  mask2former 已可导入"
else
    echo "  安装 mask2former (clone → .pth)..."
    M2F_DIR="/opt/mask2former"
    rm -rf "$M2F_DIR"
    git clone --depth 1 https://github.com/facebookresearch/Mask2Former.git "$M2F_DIR" 2>&1 | tail -3

    SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "$M2F_DIR" > "$SITE_PKG/mask2former.pth"
    echo "  添加 .pth: $SITE_PKG/mask2former.pth → $M2F_DIR"

    # 编译 MSDeformAttn C++ ops (如果存在)
    OPS_DIR="$M2F_DIR/mask2former/modeling/pixel_decoder/ops"
    if [ -f "$OPS_DIR/setup.py" ]; then
        echo "  编译 MSDeformAttn..."
        cd "$OPS_DIR"
        python setup.py build_ext --inplace 2>&1 | tail -5
        cd "$PROJECT_DIR"
    fi

    python -c "import mask2former; print('  ✓ mask2former 可导入')" 2>/dev/null || \
        echo "  ⚠ mask2former 导入失败"
fi

# ODISE
if python -c "import odise" 2>/dev/null; then
    echo "  ODISE 已可导入"
else
    echo "  安装 ODISE (clone → setup.py develop)..."
    ODISE_DIR="/opt/odise"
    rm -rf "$ODISE_DIR"
    git clone --depth 1 https://github.com/NVlabs/ODISE.git "$ODISE_DIR" 2>&1 | tail -3

    cd "$ODISE_DIR"
    if [ -f "setup.py" ]; then
        python setup.py develop 2>&1 | tail -10
    elif [ -f "pyproject.toml" ]; then
        pip install --no-build-isolation -e . 2>&1 | tail -10
    else
        SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "$ODISE_DIR" > "$SITE_PKG/odise.pth"
        echo "  使用 .pth 链接"
    fi
    cd "$PROJECT_DIR"

    python -c "import odise; print('  ✓ ODISE 可导入')" 2>/dev/null || \
        echo "  ⚠ ODISE 导入失败"
fi

# ── 全局验证 ──
echo ""
echo "─── [6/6] 全局验证 ───"
python << 'PYEOF'
import sys, os
sys.path.insert(0, '.')

# 核心包
modules = {
    'torch': 'torch', 'torchvision': 'torchvision', 'gsplat': 'gsplat',
    'numpy': 'numpy', 'scipy': 'scipy', 'cv2': 'opencv',
    'open3d': 'open3d', 'kornia': 'kornia', 'timm': 'timm',
    'einops': 'einops', 'plyfile': 'plyfile', 'PIL': 'pillow',
    'yaml': 'PyYAML', 'sklearn': 'scikit-learn', 'tensorboard': 'tensorboard',
}
ok, fail, warn = 0, 0, 0
for imp, name in modules.items():
    try:
        m = __import__(imp)
        v = getattr(m, '__version__', 'OK')
        print(f'  ✓ {name:20s} {v}')
        ok += 1
    except:
        if name == 'open3d':
            print(f'  ⚠ {name:20s} NOT INSTALLED (非关键)')
            warn += 1
        else:
            print(f'  ✗ {name:20s} MISSING')
            fail += 1

# ODISE 生态
print()
for imp, name in [('detectron2','detectron2'), ('mask2former','mask2former'), ('odise','odise')]:
    try:
        __import__(imp)
        print(f'  ✓ {name:20s} OK')
        ok += 1
    except Exception as e:
        print(f'  ⚠ {name:20s} MISSING → {str(e)[:50]}')
        warn += 1

# SD 特征提取器端到端测试
print()
try:
    from feature_extraction.extractor_sd import load_model
    print('  ✓ extractor_sd.load_model      可导入')
except Exception as e:
    print(f'  ⚠ extractor_sd.load_model      失败: {e}')

try:
    from feature_extraction.extractor_dino import ViTExtractor
    print('  ✓ extractor_dino.ViTExtractor   可导入')
except Exception as e:
    print(f'  ⚠ extractor_dino.ViTExtractor   失败: {e}')

# 训练组件
print()
try:
    from ic_models.ms_flow_pose_net import MSFlowPoseNet
    print('  ✓ MSFlowPoseNet                 可导入')
except Exception as e:
    print(f'  ✗ MSFlowPoseNet 导入失败:       {e}')
    fail += 1

try:
    from modules.multiscale_renderer import MultiScaleRenderer
    print('  ✓ MultiScaleRenderer            可导入')
except Exception as e:
    print(f'  ✗ MultiScaleRenderer:           {e}')
    fail += 1

try:
    from feature_3dgs.raw_gaussian_model import RawScaleGaussianModel
    print('  ✓ RawScaleGaussianModel         可导入')
except Exception as e:
    print(f'  ✗ RawScaleGaussianModel:        {e}')
    fail += 1

# GPU
import torch
print()
print(f'  GPUs: {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}')
x = torch.randn(100, device='cuda'); y = x @ x; print(f'  CUDA compute: OK')

# 数据集
print()
datasets = {'room_0': 'dataset/room_0/Sequence_1/rgb',
            'OldHospital': 'dataset/OldHospital/seq1',
            'stairs': 'dataset/stairs/seq-01'}
for name, path in datasets.items():
    if os.path.isdir(path):
        print(f'  ✓ {name:20s} 存在')
    else:
        print(f'  ✗ {name:20s} 不存在!')
        fail += 1

print(f'\n  总计: {ok} OK, {warn} 警告, {fail} 失败')
if fail == 0:
    print('  ✓✓✓ 环境就绪, 可以开始训练! ✓✓✓')
    sys.exit(0)
else:
    print(f'  ✗✗✗ 有 {fail} 个关键问题需要修复 ✗✗✗')
    sys.exit(1)
PYEOF

echo ""
echo "═══════════════════════════════════════════════════"
echo "  修复完成! 下一步:"
echo ""
echo "  # OldHospital 2DGS (不依赖 ODISE, 可直接启动):"
echo "  bash scripts/run_pipeline.sh 2dgs"
echo ""
echo "  # 如果 ODISE 生态也 OK, 可以跑 SD 特征提取:"
echo "  bash scripts/run_pipeline.sh extract"
echo ""
echo "  # 全流程:"
echo "  bash scripts/run_pipeline.sh all"
echo "═══════════════════════════════════════════════════"
