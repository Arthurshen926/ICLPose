#!/usr/bin/env bash
#
# 3DGS-MCMC 重建脚本（一键）
# ===========================
# 用法:
#   bash scripts/data_prep/run_3dgs_mcmc.sh <scene> [gpu_id]
#
# 示例:
#   bash scripts/data_prep/run_3dgs_mcmc.sh stairs 1
#   bash scripts/data_prep/run_3dgs_mcmc.sh oldhospital 1
#
# 前提:
#   1. 已安装 3DGS-MCMC 到独立 conda 环境（conda activate gs）
#   2. 7-Scenes stairs 已有 COLMAP sparse（stairs/sparse/0/）
#   3. Cambridge OldHospital 已完成 NVM → COLMAP 转换
#
# 注意: 不会影响正在运行的训练进程（使用独立 GPU 和 conda 环境）

set -euo pipefail

# ─── 配置 ───────────────────────────────────
ICLPOSE_ROOT="/home/yons/Projects/ICLPose"
GS_MCMC_DIR="${HOME}/Projects/3dgs-mcmc"   # 3DGS-MCMC 安装路径
CONDA_ENV="gs"                              # conda 环境名
GPU_ID="${2:-1}"                             # 默认 GPU 1（不干扰主训练）

SCENE="${1:?用法: $0 <stairs|oldhospital> [gpu_id]}"

# ─── 场景参数 ───────────────────────────────
case "$SCENE" in
  stairs)
    SOURCE_DIR="/mnt/pool1/sqy/7scenes/stairs"
    MODEL_DIR="${ICLPOSE_ROOT}/dataset/stairs/gaussian_splatting"
    ITERATIONS=30000
    CAP_MAX=500000       # 50 万 Gaussians（室内小场景）
    RESOLUTION=1         # 640x480 原分辨率
    ;;
  oldhospital)
    # ⚠️ 需要先完成 NVM→COLMAP 转换（见 README_3dgs.md）
    SOURCE_DIR="${ICLPOSE_ROOT}/dataset/cambridge_OldHospital/colmap"
    MODEL_DIR="${ICLPOSE_ROOT}/dataset/cambridge_OldHospital/gaussian_splatting"
    ITERATIONS=50000
    CAP_MAX=1000000      # 100 万 Gaussians（室外大场景）
    RESOLUTION=2         # 降采样到 960x540
    ;;
  *)
    echo "[ERROR] 未知场景: $SCENE (支持: stairs, oldhospital)"
    exit 1
    ;;
esac

# ─── 检查 ───────────────────────────────────
if [ ! -d "$SOURCE_DIR" ]; then
  echo "[ERROR] 数据源目录不存在: $SOURCE_DIR"
  exit 1
fi

if [ ! -d "$GS_MCMC_DIR" ]; then
  echo "[ERROR] 3DGS-MCMC 目录不存在: $GS_MCMC_DIR"
  echo ""
  echo "请先安装 3DGS-MCMC："
  echo "  git clone https://github.com/ubc-vision/3dgs-mcmc --recursive ${GS_MCMC_DIR}"
  echo "  conda activate ${CONDA_ENV}"
  echo "  cd ${GS_MCMC_DIR}"
  echo "  pip install -r requirements.txt"
  echo "  pip install submodules/diff-gaussian-rasterization submodules/simple-knn"
  exit 1
fi

# ─── 执行 ───────────────────────────────────
echo "========================================"
echo " 3DGS-MCMC 重建"
echo "========================================"
echo " 场景:      ${SCENE}"
echo " 数据源:    ${SOURCE_DIR}"
echo " 输出:      ${MODEL_DIR}"
echo " 迭代次数:  ${ITERATIONS}"
echo " 最大GS数:  ${CAP_MAX}"
echo " 分辨率:    1/${RESOLUTION}"
echo " GPU:       ${GPU_ID}"
echo "========================================"

mkdir -p "$(dirname "$MODEL_DIR")"

# 激活 conda 环境并运行
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

cd "$GS_MCMC_DIR"

CUDA_VISIBLE_DEVICES="$GPU_ID" python train.py \
    -s "$SOURCE_DIR" \
    -m "$MODEL_DIR" \
    --iterations "$ITERATIONS" \
    --cap_max "$CAP_MAX" \
    --resolution "$RESOLUTION" \
    --eval

echo ""
echo "[完成] 重建完毕！"
echo ""

# ─── 创建 final/ 软链接 ───────────────────
FINAL_DIR="${MODEL_DIR}/point_cloud/final"
ITER_PLY="${MODEL_DIR}/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"

if [ -f "$ITER_PLY" ]; then
    mkdir -p "$FINAL_DIR"
    if [ ! -e "${FINAL_DIR}/point_cloud.ply" ]; then
        ln -s "$ITER_PLY" "${FINAL_DIR}/point_cloud.ply"
        echo "[INFO] 已创建软链接: ${FINAL_DIR}/point_cloud.ply → ${ITER_PLY}"
    fi
    echo ""
    echo "[下一步] 计算场景边界并更新配置:"
    echo "  cd ${ICLPOSE_ROOT}"
    echo "  python scripts/data_prep/compute_scene_bound.py --ply ${FINAL_DIR}/point_cloud.ply"
    echo "  vim configs/exp030_stairs.yaml  # 更新 scene.bound"
else
    echo "[WARN] 未找到最终点云: ${ITER_PLY}"
    echo "       请检查训练日志并手动创建软链接"
fi
