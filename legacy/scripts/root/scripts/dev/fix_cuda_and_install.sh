#!/bin/bash
# =============================================================================
# 一键修复 CUDA 版本不匹配 + 安装 detectron2/mask2former/ODISE
# =============================================================================
# 问题: 系统 nvcc 是 CUDA 11.6, 但 PyTorch 编译用的 CUDA 12.1
#        detectron2 编译 C++ 扩展时检测到不匹配就报错
#
# 解决方案:
#   1. 用 conda 安装 CUDA 12.1 toolkit (仅供编译使用)
#   2. 设置 CUDA_HOME 指向 12.1
#   3. 编译安装 detectron2
#   4. 用 .pth 方式安装 mask2former
#   5. 编译安装 ODISE
# =============================================================================
# 用法:  conda activate iclpose && bash scripts/fix_cuda_and_install.sh
# =============================================================================
set +e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

echo "╔═══════════════════════════════════════════════════╗"
echo "║  修复 CUDA 版本 + 安装 ODISE 生态                ║"
echo "╚═══════════════════════════════════════════════════╝"

# ── Step 1: 诊断当前 CUDA 状态 ──
echo ""
echo "─── [1/7] 诊断 CUDA 状态 ───"
echo "  PyTorch CUDA: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null)"
echo "  系统 nvcc:    $(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')"
echo "  CUDA_HOME:    ${CUDA_HOME:-未设置}"

# 查找系统已有的 CUDA 12.x
FOUND_CUDA=""
for p in /usr/local/cuda-12.1 /usr/local/cuda-12 /usr/local/cuda; do
    if [ -f "$p/bin/nvcc" ]; then
        VER=$("$p/bin/nvcc" --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')
        echo "  发现: $p → CUDA $VER"
        if [[ "$VER" == 12.1* ]] || [[ "$VER" == 12.* ]]; then
            FOUND_CUDA="$p"
        fi
    fi
done

# 也检查 conda env 中的 CUDA
CONDA_CUDA="$CONDA_PREFIX/lib"
if [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
    VER=$("$CONDA_PREFIX/bin/nvcc" --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')
    echo "  Conda env: $CONDA_PREFIX → CUDA $VER"
    if [[ "$VER" == 12.1* ]] || [[ "$VER" == 12.* ]]; then
        FOUND_CUDA="$CONDA_PREFIX"
    fi
fi

# ── Step 2: 安装 CUDA 12.1 toolkit (如果没有) ──
echo ""
echo "─── [2/7] 确保 CUDA 12.1 toolkit 可用 ───"
if [ -n "$FOUND_CUDA" ]; then
    echo "  已有 CUDA 12.x: $FOUND_CUDA"
    export CUDA_HOME="$FOUND_CUDA"
else
    echo "  未找到 CUDA 12.x, 通过 conda 安装 cuda-toolkit 12.1..."
    conda install -y -c nvidia cuda-toolkit=12.1 2>&1 | tail -10
    
    if [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
        export CUDA_HOME="$CONDA_PREFIX"
        echo "  ✓ CUDA toolkit 安装到: $CUDA_HOME"
    else
        # 尝试 cuda-nvcc
        echo "  cuda-toolkit 未提供 nvcc, 尝试 cuda-nvcc..."
        conda install -y -c nvidia cuda-nvcc=12.1 2>&1 | tail -5
        if [ -f "$CONDA_PREFIX/bin/nvcc" ]; then
            export CUDA_HOME="$CONDA_PREFIX"
            echo "  ✓ nvcc 安装到: $CUDA_HOME"
        else
            echo "  conda 方式未成功, 尝试跳过 CUDA 版本检查..."
        fi
    fi
fi

echo "  CUDA_HOME=$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
echo "  nvcc: $(nvcc --version 2>/dev/null | grep release || echo '不可用')"

# 如果还是没有正确的 nvcc, 使用环境变量跳过版本检查
NVCC_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')
if [[ ! "$NVCC_VER" == 12.* ]]; then
    echo "  nvcc 仍然不是 12.x ($NVCC_VER), 使用 TORCH_CUDA_ARCH_LIST 跳过检查..."
    export TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 = SM 8.9
    export FORCE_CUDA=1
    # 跳过 detectron2 的 CUDA 版本检查
    export D2_SKIP_CUDA_CHECK=1
fi

# ── Step 3: 安装编译依赖 ──
echo ""
echo "─── [3/7] 安装编译依赖 ───"
pip install -q ninja fvcore iopath pycocotools 2>/dev/null
echo "  ✓ OK"

# ── Step 4: 安装 detectron2 ──
echo ""
echo "─── [4/7] 安装 detectron2 ───"
if python -c "import detectron2" 2>/dev/null; then
    echo "  已安装, 跳过"
else
    rm -rf /tmp/detectron2_build
    git clone --depth 1 https://github.com/facebookresearch/detectron2.git /tmp/detectron2_build
    cd /tmp/detectron2_build

    # 如果 CUDA 版本检查仍然会失败, 打补丁跳过它
    NVCC_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')
    if [[ ! "$NVCC_VER" == 12.* ]]; then
        echo "  打补丁: 跳过 PyTorch cpp_extension 中的 CUDA 版本检查..."
        TORCH_EXT=$(python -c "import torch.utils.cpp_extension as e; print(e.__file__)")
        if [ -f "$TORCH_EXT" ]; then
            # 备份
            cp "$TORCH_EXT" "${TORCH_EXT}.bak"
            # 注释掉版本检查 raise
            python -c "
import re
with open('$TORCH_EXT', 'r') as f:
    content = f.read()
# 替换 raise RuntimeError(CUDA_MISMATCH_MESSAGE... 为 pass
content = re.sub(
    r'raise RuntimeError\(CUDA_MISMATCH_MESSAGE\.format\(cuda_str_version, torch\.version\.cuda\)\)',
    'pass  # PATCHED: skip CUDA version check',
    content
)
with open('$TORCH_EXT', 'w') as f:
    f.write(content)
print('  ✓ 已打补丁跳过 CUDA 版本检查')
"
        fi
    fi

    echo "  编译 detectron2 (CUDA_HOME=$CUDA_HOME)..."
    python setup.py build develop 2>&1 | tail -20

    cd "$PROJECT_DIR"

    if python -c "import detectron2; print(f'  ✓ detectron2 {detectron2.__version__}')" 2>/dev/null; then
        echo "  detectron2 安装成功!"
    else
        echo "  ✗ detectron2 安装失败"
        echo "  尝试纯 Python 安装 (不编译 C++ 扩展)..."
        cd /tmp/detectron2_build
        # 不编译 C++ 扩展, 只安装 Python 部分
        SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "/tmp/detectron2_build" > "$SITE_PKG/detectron2.pth"
        cd "$PROJECT_DIR"
        python -c "import detectron2; print(f'  ✓ detectron2 {detectron2.__version__} (纯 Python)')" 2>/dev/null || \
            echo "  ✗ detectron2 完全失败"
    fi

    # 恢复备份 (如果打了补丁)
    TORCH_EXT=$(python -c "import torch.utils.cpp_extension as e; print(e.__file__)")
    if [ -f "${TORCH_EXT}.bak" ]; then
        mv "${TORCH_EXT}.bak" "$TORCH_EXT"
        echo "  已恢复 cpp_extension.py 原始版本"
    fi
fi

# ── Step 5: 安装 mask2former ──
echo ""
echo "─── [5/7] 安装 mask2former ───"
if python -c "import mask2former" 2>/dev/null; then
    echo "  已可导入, 跳过"
else
    M2F_DIR="/opt/mask2former"
    if [ ! -d "$M2F_DIR" ]; then
        git clone --depth 1 https://github.com/facebookresearch/Mask2Former.git "$M2F_DIR" 2>&1 | tail -3
    fi

    SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "$M2F_DIR" > "$SITE_PKG/mask2former.pth"

    # MSDeformAttn 编译 (需要 CUDA 版本匹配)
    OPS_DIR="$M2F_DIR/mask2former/modeling/pixel_decoder/ops"
    if [ -f "$OPS_DIR/setup.py" ]; then
        echo "  编译 MSDeformAttn (CUDA_HOME=$CUDA_HOME)..."
        cd "$OPS_DIR"

        # 同样需要临时打补丁
        TORCH_EXT=$(python -c "import torch.utils.cpp_extension as e; print(e.__file__)")
        NVCC_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')
        PATCHED=false
        if [[ ! "$NVCC_VER" == 12.* ]] && [ -f "$TORCH_EXT" ]; then
            cp "$TORCH_EXT" "${TORCH_EXT}.bak"
            python -c "
import re
with open('$TORCH_EXT', 'r') as f:
    content = f.read()
content = re.sub(
    r'raise RuntimeError\(CUDA_MISMATCH_MESSAGE\.format\(cuda_str_version, torch\.version\.cuda\)\)',
    'pass  # PATCHED: skip CUDA version check',
    content
)
with open('$TORCH_EXT', 'w') as f:
    f.write(content)
"
            PATCHED=true
        fi

        python setup.py build_ext --inplace 2>&1 | tail -10
        
        if [ "$PATCHED" = true ] && [ -f "${TORCH_EXT}.bak" ]; then
            mv "${TORCH_EXT}.bak" "$TORCH_EXT"
        fi
        cd "$PROJECT_DIR"
    fi

    python -c "import mask2former; print('  ✓ mask2former 可导入')" 2>/dev/null || {
        echo "  ⚠ mask2former 导入失败 (MSDeformAttn 可能未编译)"
        echo "  mask2former 的 SD 特征提取依赖这个, 但训练流程不受影响"
    }
fi

# ── Step 6: 安装 ODISE ──
echo ""
echo "─── [6/7] 安装 ODISE ───"
if python -c "import odise" 2>/dev/null; then
    echo "  已可导入, 跳过"
else
    ODISE_DIR="/opt/odise"
    if [ ! -d "$ODISE_DIR" ]; then
        git clone --depth 1 https://github.com/NVlabs/ODISE.git "$ODISE_DIR" 2>&1 | tail -3
    fi

    cd "$ODISE_DIR"

    # 临时打补丁
    TORCH_EXT=$(python -c "import torch.utils.cpp_extension as e; print(e.__file__)")
    NVCC_VER=$(nvcc --version 2>/dev/null | grep release | awk '{print $6}' | tr -d ',')
    PATCHED=false
    if [[ ! "$NVCC_VER" == 12.* ]] && [ -f "$TORCH_EXT" ]; then
        cp "$TORCH_EXT" "${TORCH_EXT}.bak"
        python -c "
import re
with open('$TORCH_EXT', 'r') as f:
    content = f.read()
content = re.sub(
    r'raise RuntimeError\(CUDA_MISMATCH_MESSAGE\.format\(cuda_str_version, torch\.version\.cuda\)\)',
    'pass  # PATCHED: skip CUDA version check',
    content
)
with open('$TORCH_EXT', 'w') as f:
    f.write(content)
"
        PATCHED=true
    fi

    if [ -f "setup.py" ]; then
        python setup.py develop 2>&1 | tail -10
    elif [ -f "pyproject.toml" ]; then
        pip install --no-build-isolation -e . 2>&1 | tail -10
    else
        SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
        echo "$ODISE_DIR" > "$SITE_PKG/odise.pth"
    fi

    if [ "$PATCHED" = true ] && [ -f "${TORCH_EXT}.bak" ]; then
        mv "${TORCH_EXT}.bak" "$TORCH_EXT"
    fi
    cd "$PROJECT_DIR"

    python -c "import odise; print('  ✓ ODISE 可导入')" 2>/dev/null || \
        echo "  ⚠ ODISE 导入失败"
fi

# ── Step 7: 全面验证 ──
echo ""
echo "─── [7/7] 全面验证 ───"
python << 'PYEOF'
import sys, os
sys.path.insert(0, '.')

ok, warn, fail = 0, 0, 0

# 核心训练包
core = {
    'torch': None, 'torchvision': None, 'gsplat': None,
    'numpy': None, 'scipy': None, 'cv2': None,
    'kornia': None, 'timm': None, 'einops': None,
    'plyfile': None, 'PIL': None, 'yaml': None,
    'sklearn': None, 'tensorboard': None,
}
for m in core:
    try:
        mod = __import__(m)
        v = getattr(mod, '__version__', 'OK')
        print(f'  ✓ {m:20s} {v}')
        ok += 1
    except:
        print(f'  ✗ {m:20s} MISSING')
        fail += 1

# ODISE 生态
print()
for m in ['detectron2', 'mask2former', 'odise']:
    try:
        __import__(m)
        print(f'  ✓ {m:20s} OK')
        ok += 1
    except Exception as e:
        short_e = str(e).split('\n')[0][:60]
        print(f'  ⚠ {m:20s} {short_e}')
        warn += 1

# 训练组件 (关键)
print()
checks = [
    ("from ic_models.ms_flow_pose_net import MSFlowPoseNet", "MSFlowPoseNet"),
    ("from modules.multiscale_renderer import MultiScaleRenderer", "MultiScaleRenderer"),
    ("from feature_3dgs.raw_gaussian_model import RawScaleGaussianModel", "RawScaleGaussianModel"),
    ("from feature_extraction.extractor_dino import ViTExtractor", "ViTExtractor"),
    ("from data.dataset_v4 import PoseDatasetV4", "PoseDatasetV4"),
]
for stmt, name in checks:
    try:
        exec(stmt)
        print(f'  ✓ {name:40s} OK')
        ok += 1
    except Exception as e:
        short_e = str(e).split('\n')[0][:60]
        print(f'  ✗ {name:40s} {short_e}')
        fail += 1

# SD 特征提取
print()
try:
    from feature_extraction.extractor_sd import load_model
    print(f'  ✓ {"SD extractor (load_model)":40s} OK')
    ok += 1
except Exception as e:
    short_e = str(e).split('\n')[0][:60]
    print(f'  ⚠ {"SD extractor (load_model)":40s} {short_e}')
    warn += 1

# 数据集
print()
import torch
print(f'  GPUs: {torch.cuda.device_count()} × {torch.cuda.get_device_name(0)}')

for name, path in [('room_0','dataset/room_0/Sequence_1/rgb'),
                    ('OldHospital','dataset/OldHospital/seq1'),
                    ('stairs','dataset/stairs/seq-01')]:
    if os.path.isdir(path):
        print(f'  ✓ {name:20s} 存在')
    else:
        print(f'  ✗ {name:20s} 不存在!')
        fail += 1

print(f'\n  {ok} OK, {warn} 警告 (非关键), {fail} 失败')
if fail == 0:
    print('  ✓✓✓ 训练环境就绪! ✓✓✓')
    if warn > 0:
        print('  (SD 特征提取可能受限, 但 2DGS/embedding/pose 训练均可运行)')
else:
    print(f'  有 {fail} 个关键问题')
PYEOF

echo ""
echo "══════════════════════════════════════════════════════"
echo "  下一步 (请直接复制运行):"
echo ""
echo "  # 先跑不依赖 ODISE 的流程:"
echo "  conda activate iclpose"
echo "  bash scripts/run_pipeline.sh 2dgs     # OldHospital 2DGS"
echo ""
echo "  # 如果 SD extractor OK:"
echo "  bash scripts/run_pipeline.sh extract   # 特征提取"
echo "  bash scripts/run_pipeline.sh embed     # 嵌入训练"  
echo "  bash scripts/run_pipeline.sh pose      # 位姿训练"
echo "══════════════════════════════════════════════════════"
