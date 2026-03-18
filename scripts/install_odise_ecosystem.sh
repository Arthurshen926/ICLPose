#!/bin/bash
# =============================================================================
# 安装 Detectron2 + Mask2Former + ODISE 生态
# (SD 特征提取的核心依赖)
# =============================================================================
# 用法:
#   conda activate iclpose
#   bash scripts/install_odise_ecosystem.sh
# =============================================================================
set +e  # 不要因单个命令失败就退出

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "╔══════════════════════════════════════════════╗"
echo "║  安装 ODISE 生态 (detectron2 + mask2former)  ║"
echo "╚══════════════════════════════════════════════╝"

# ── Step 0: 验证 PyTorch ──
echo ""
echo "─── [0] 验证 PyTorch ───"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.version.cuda}')
print(f'  GPUs:    {torch.cuda.device_count()}')
assert torch.cuda.is_available(), 'CUDA not available!'
" || { echo "[FATAL] PyTorch 不可用, 请先运行 setup_env_4090.sh"; exit 1; }

# ── Step 1: 构建前置依赖 ──
echo ""
echo "─── [1] 安装构建前置依赖 (ninja, fvcore, iopath) ───"
pip install ninja fvcore iopath pycocotools 2>&1 | tail -5
echo "  ✓ 构建依赖就绪"

# ── Step 2: 安装 detectron2 ──
echo ""
echo "─── [2] 安装 detectron2 ───"
if python -c "import detectron2; print(f'  已安装: detectron2 {detectron2.__version__}')" 2>/dev/null; then
    echo "  跳过"
else
    echo "  方法1: clone + python setup.py develop..."
    rm -rf /tmp/detectron2_build
    git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2_build
    cd /tmp/detectron2_build
    python setup.py develop 2>&1 | tail -20
    cd "$PROJECT_DIR"

    if ! python -c "import detectron2" 2>/dev/null; then
        echo ""
        echo "  方法1失败, 尝试方法2: pip --no-build-isolation..."
        cd /tmp/detectron2_build
        pip install --no-build-isolation -e . 2>&1 | tail -20
        cd "$PROJECT_DIR"
    fi

    if ! python -c "import detectron2" 2>/dev/null; then
        echo ""
        echo "  方法2也失败, 尝试方法3: 旧版 v0.6..."
        cd /tmp/detectron2_build
        git fetch --tags
        git checkout v0.6
        python setup.py develop 2>&1 | tail -20
        cd "$PROJECT_DIR"
    fi

    if python -c "import detectron2; print(f'  ✓ detectron2 {detectron2.__version__}')" 2>/dev/null; then
        echo "  detectron2 安装成功!"
    else
        echo "  ✗ detectron2 安装失败!"
        echo "  常见原因: gcc 版本, CUDA toolkit, 或 ninja 编译问题"
        exit 1
    fi
fi

# ── Step 3: 安装 ODISE 周边依赖 ──
echo ""
echo "─── [3] 安装 ODISE 周边依赖 ───"
pip install \
    omegaconf==2.3.0 \
    panopticapi \
    ftfy \
    regex \
    open-clip-torch==2.20.0 \
    transformers==4.36.2 \
    2>&1 | tail -5
echo "  ✓ 周边依赖安装完成"

# ── Step 4: 安装 mask2former ──
echo ""
echo "─── [4] 安装 mask2former ───"
if python -c "import mask2former" 2>/dev/null; then
    echo "  已安装, 跳过"
else
    echo "  Clone + path install mask2former..."
    M2F_DIR="/opt/mask2former"
    rm -rf "$M2F_DIR"
    git clone --depth 1 https://github.com/facebookresearch/Mask2Former.git "$M2F_DIR"

    if [ -d "$M2F_DIR" ]; then
        SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "$M2F_DIR" > "$SITE_PKG/mask2former.pth"

        # 编译 MSDeformAttn C++ 扩展
        if [ -f "$M2F_DIR/mask2former/modeling/pixel_decoder/ops/setup.py" ]; then
            echo "  编译 MSDeformAttn 扩展..."
            cd "$M2F_DIR/mask2former/modeling/pixel_decoder/ops"
            python setup.py build_ext --inplace 2>&1 | tail -5 || true
            cd "$PROJECT_DIR"
        fi
    fi

    python -c "import mask2former; print('  ✓ mask2former 可导入')" 2>/dev/null || \
        echo "  ⚠ mask2former 导入失败"
fi

# ── Step 5: 安装 ODISE ──
echo ""
echo "─── [5] 安装 ODISE ───"
if python -c "import odise" 2>/dev/null; then
    echo "  已安装, 跳过"
else
    echo "  Clone + develop ODISE..."
    ODISE_DIR="/opt/odise"
    rm -rf "$ODISE_DIR"
    git clone --depth 1 https://github.com/NVlabs/ODISE.git "$ODISE_DIR"

    if [ -d "$ODISE_DIR" ]; then
        cd "$ODISE_DIR"
        if [ -f "setup.py" ]; then
            python setup.py develop 2>&1 | tail -10 || \
                pip install --no-build-isolation -e . 2>&1 | tail -10 || true
        elif [ -f "pyproject.toml" ]; then
            pip install --no-build-isolation -e . 2>&1 | tail -10 || true
        else
            echo "  无 setup.py/pyproject.toml, 使用 .pth 链接..."
            SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
            echo "$ODISE_DIR" > "$SITE_PKG/odise.pth"
        fi
        cd "$PROJECT_DIR"
    fi

    python -c "import odise; print('  ✓ odise 可导入')" 2>/dev/null || \
        echo "  ⚠ odise 导入失败"
fi

# ── Step 6: 全量验证 ──
echo ""
echo "─── [6] 验证 ODISE 生态 ───"
python -c "
import sys
status = {}
for mod_name in ['detectron2', 'mask2former', 'odise', 'fvcore', 'pycocotools']:
    try:
        m = __import__(mod_name)
        v = getattr(m, '__version__', 'OK')
        status[mod_name] = ('✓', v)
    except Exception as e:
        status[mod_name] = ('✗', str(e)[:60])

for name, (s, v) in status.items():
    print(f'  {s} {name:20s} {v}')

# 测试 SD 特征提取器
print()
try:
    from feature_extraction.extractor_sd import load_model
    print('  ✓ extractor_sd.load_model 可导入')
except Exception as e:
    print(f'  ✗ extractor_sd 导入失败: {e}')

fails = sum(1 for s, _ in status.values() if s == '✗')
if fails == 0:
    print()
    print('  ✓✓✓ ODISE 生态全部就绪! SD 特征提取可用 ✓✓✓')
else:
    print()
    print(f'  ⚠ {fails} 个模块安装失败')
    sys.exit(1)
"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║  完成! 下一步: bash scripts/run_pipeline.sh check ║"
echo "╚══════════════════════════════════════════════════╝"
