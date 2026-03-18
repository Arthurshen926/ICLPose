#!/bin/bash
# =============================================================================
# ICLPose 全流水线脚本 — 从零 output 到训练就绪
# =============================================================================
# 用法:
#   conda activate iclpose
#   bash scripts/run_pipeline.sh [STAGE]
#
# STAGE 可选:
#   all        — 执行全部 (默认)
#   check      — 仅检查数据和环境
#   extract    — 仅提取特征
#   embed      — 仅训练 3DGS 特征嵌入
#   pose       — 仅训练位姿网络
#   2dgs       — 仅训练 OldHospital 2DGS
# =============================================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_DIR"

STAGE="${1:-all}"
TIMESTAMP=$(date "+%Y%m%d_%H%M%S")

# 日志
mkdir -p output/logs
LOG_FILE="output/logs/pipeline_${TIMESTAMP}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "============================================="
echo "  ICLPose 流水线 — Stage: $STAGE"
echo "  时间: $(date)"
echo "  项目: $PROJECT_DIR"
echo "============================================="

# =============================================================================
# 辅助函数
# =============================================================================
check_gpu_free() {
    # 返回第一个空闲 (<2GB使用) 的 GPU ID
    local gpu_id
    for gpu_id in 0 1 2 3 4 5; do
        local mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_id" 2>/dev/null | tr -d ' ')
        if [ -n "$mem" ] && [ "$mem" -lt 2000 ]; then
            echo "$gpu_id"
            return 0
        fi
    done
    echo "-1"
    return 1
}

wait_for_gpu() {
    echo "  等待空闲 GPU..."
    while true; do
        local gpu=$(check_gpu_free)
        if [ "$gpu" != "-1" ]; then
            echo "  GPU $gpu 可用"
            echo "$gpu"
            return 0
        fi
        sleep 30
    done
}

# =============================================================================
# Stage 0: 环境与数据检查
# =============================================================================
stage_check() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Stage 0: 环境与数据检查"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Python 环境
    echo ""
    echo "[检查] Python 环境"
    python -c "
import torch
print(f'  PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPUs: {torch.cuda.device_count()}')
for i in range(torch.cuda.device_count()):
    mem = torch.cuda.get_device_properties(i).total_mem / 1024**3
    print(f'    GPU {i}: {torch.cuda.get_device_name(i)} ({mem:.0f} GB)')
"

    # gsplat
    python -c "import gsplat; print(f'  gsplat: {gsplat.__version__}')" || \
        { echo "  [ERROR] gsplat 不可用!"; exit 1; }

    # 数据集检查
    echo ""
    echo "[检查] 数据集"

    # room_0
    if [ -d "dataset/room_0/Sequence_1/rgb" ]; then
        local n_train=$(ls dataset/room_0/Sequence_1/rgb/*.png 2>/dev/null | wc -l)
        local n_val=$(ls dataset/room_0/Sequence_2/rgb/*.png 2>/dev/null | wc -l)
        echo "  ✓ room_0: Seq1=${n_train} frames, Seq2=${n_val} frames"
    else
        echo "  ✗ room_0: Sequence_1/rgb 不存在!"
    fi

    if [ -f "dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply" ]; then
        echo "  ✓ room_0: point_cloud.ply 存在"
    else
        echo "  ✗ room_0: point_cloud.ply 不存在!"
    fi

    # OldHospital
    if [ -d "dataset/OldHospital/seq1" ]; then
        local n_train_oh=$(cat dataset/OldHospital/dataset_train.txt 2>/dev/null | wc -l)
        local n_test_oh=$(cat dataset/OldHospital/dataset_test.txt 2>/dev/null | wc -l)
        echo "  ✓ OldHospital: train=${n_train_oh}, test=${n_test_oh}"
    else
        echo "  ✗ OldHospital: seq 目录不存在!"
    fi

    if [ -f "dataset/OldHospital/masks.pkl" ]; then
        echo "  ✓ OldHospital: masks.pkl 存在"
    else
        echo "  ✗ OldHospital: masks.pkl 不存在!"
    fi

    if [ -f "dataset/OldHospital/sparse/0/points3D.ply" ]; then
        echo "  ✓ OldHospital: COLMAP sparse 存在"
    else
        echo "  ✗ OldHospital: COLMAP sparse 不存在!"
    fi

    # stairs
    if [ -d "dataset/stairs/seq-01" ]; then
        echo "  ✓ stairs: 数据存在"
    else
        echo "  ✗ stairs: 不存在!"
    fi

    echo ""
    echo "[检查] output/ 目录"
    mkdir -p output/{features_multiscale/room_0,feature_3dgs/room_0_raw,2dgs_models/OldHospital,logs}
    echo "  ✓ output/ 目录已创建"

    echo ""
    echo "━━━ Stage 0 完成 ━━━"
}

# =============================================================================
# Stage 1: 特征提取 (SD + DINO) — room_0
# =============================================================================
stage_extract() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Stage 1: 多尺度特征提取 (room_0)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Sequence_1 (训练集)
    local out_seq1="output/features_multiscale/room_0"
    if [ -d "$out_seq1/fine_sd" ] && [ "$(ls $out_seq1/fine_sd/*.pt 2>/dev/null | wc -l)" -gt 100 ]; then
        echo "  Sequence_1 特征已存在 ($(ls $out_seq1/fine_sd/*.pt | wc -l) files), 跳过"
    else
        echo "  提取 Sequence_1 特征..."
        echo "  (SD + DINO 模型首次加载可能需要下载, 请耐心等待)"
        CUDA_VISIBLE_DEVICES=0 python scripts/extract_features_v2.py \
            --input_dir dataset/room_0/Sequence_1/rgb \
            --output_dir "$out_seq1" \
            --device cuda

        echo ""
        echo "  重命名为 v1 格式 (dataset_v4 优先识别)"
        # v2 输出: sd_s3 → fine_sd, sd_s4 → mid, sd_s5 → coarse, dino → fine_dino
        cd "$out_seq1"
        [ -d "sd_s3" ] && [ ! -d "fine_sd" ]   && mv sd_s3   fine_sd
        [ -d "sd_s4" ] && [ ! -d "mid" ]        && mv sd_s4   mid
        [ -d "sd_s5" ] && [ ! -d "coarse" ]     && mv sd_s5   coarse
        [ -d "dino" ]  && [ ! -d "fine_dino" ]  && mv dino    fine_dino
        cd "$PROJECT_DIR"
    fi

    # Sequence_2 (验证集) — 可选，验证集也可用训练集拆分
    local out_seq2="output/features_multiscale/room_0_val"
    if [ -d "$out_seq2/fine_sd" ] && [ "$(ls $out_seq2/fine_sd/*.pt 2>/dev/null | wc -l)" -gt 50 ]; then
        echo "  Sequence_2 特征已存在, 跳过"
    else
        echo "  提取 Sequence_2 (验证集) 特征..."
        CUDA_VISIBLE_DEVICES=0 python scripts/extract_features_v2.py \
            --input_dir dataset/room_0/Sequence_2/rgb \
            --output_dir "$out_seq2" \
            --device cuda

        cd "$out_seq2"
        [ -d "sd_s3" ] && [ ! -d "fine_sd" ]   && mv sd_s3   fine_sd
        [ -d "sd_s4" ] && [ ! -d "mid" ]        && mv sd_s4   mid
        [ -d "sd_s5" ] && [ ! -d "coarse" ]     && mv sd_s5   coarse
        [ -d "dino" ]  && [ ! -d "fine_dino" ]  && mv dino    fine_dino
        cd "$PROJECT_DIR"
    fi

    echo ""
    echo "━━━ Stage 1 完成: 特征提取 ━━━"
}

# =============================================================================
# Stage 2: 3DGS 特征嵌入训练 — room_0 (4 尺度并行)
# =============================================================================
stage_embed() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Stage 2: 3DGS 特征嵌入训练 (room_0, 4尺度并行)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    local PLY="dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply"
    local FEAT_DIR="output/features_multiscale/room_0"
    local TRAJ="dataset/room_0/Sequence_1/traj_w_c.txt"
    local OUT_BASE="output/feature_3dgs/room_0_raw"

    if [ ! -f "$PLY" ]; then
        echo "  [ERROR] PLY 不存在: $PLY"
        return 1
    fi
    if [ ! -d "$FEAT_DIR/fine_sd" ]; then
        echo "  [ERROR] 特征目录不存在, 请先运行 stage_extract"
        return 1
    fi

    # 4 个尺度并行训练 (各占 ~4-8 GB, 4090 足够)
    echo "  在 GPU 0-3 上并行训练 4 个尺度..."

    SCALES=("fine_sd" "fine_dino" "mid" "coarse")
    PIDS=()

    for i in "${!SCALES[@]}"; do
        local scale="${SCALES[$i]}"
        local gpu_id=$i
        local out_dir="${OUT_BASE}/${scale}"

        if [ -f "${out_dir}/best_model.pth" ]; then
            echo "  ✓ ${scale} 已训练完成, 跳过"
            continue
        fi

        echo "  启动 ${scale} @ GPU ${gpu_id}..."
        mkdir -p "$out_dir"

        CUDA_VISIBLE_DEVICES=$gpu_id PYTHONPATH=. python -u -m feature_3dgs.train_raw_embedding \
            --scale "$scale" \
            --ply_path "$PLY" \
            --feature_dir "$FEAT_DIR" \
            --traj_path "$TRAJ" \
            --output_dir "$out_dir" \
            --num_iters 5000 \
            --grad_accum 4 \
            --precache_gpu \
            &> "output/logs/embed_${scale}_${TIMESTAMP}.log" &

        PIDS+=($!)
        echo "    PID: ${PIDS[-1]}"
    done

    # 等待所有完成
    if [ ${#PIDS[@]} -gt 0 ]; then
        echo ""
        echo "  等待 ${#PIDS[@]} 个嵌入训练完成..."
        for pid in "${PIDS[@]}"; do
            wait "$pid" || echo "  [WARN] PID $pid 退出码非零"
        done
        echo "  ✓ 所有尺度嵌入训练完成"
    fi

    # 验证
    echo ""
    echo "  验证嵌入模型:"
    for scale in "${SCALES[@]}"; do
        if [ -f "${OUT_BASE}/${scale}/best_model.pth" ]; then
            echo "  ✓ ${scale}: $(du -sh ${OUT_BASE}/${scale}/best_model.pth | cut -f1)"
        else
            echo "  ✗ ${scale}: best_model.pth 不存在!"
        fi
    done

    echo ""
    echo "━━━ Stage 2 完成: 3DGS 特征嵌入 ━━━"
}

# =============================================================================
# Stage 3: MSFlowPoseNet 位姿网络训练 — room_0 (exp032 复现)
# =============================================================================
stage_pose() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Stage 3: MSFlowPoseNet 训练 (room_0, exp032 复现)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # 检查前置条件
    local required_models=("fine_sd" "fine_dino" "mid" "coarse")
    for scale in "${required_models[@]}"; do
        if [ ! -f "output/feature_3dgs/room_0_raw/${scale}/best_model.pth" ]; then
            echo "  [ERROR] 缺少 ${scale} 嵌入模型, 请先运行 embed 阶段"
            return 1
        fi
    done

    echo "  ✓ 所有 3DGS 嵌入模型就绪"

    # 使用 exp032 配置
    # 从头训练使用 exp032_fresh (无 warmstart, 更高 lr, 更长 epochs)
    local CONFIG="configs/exp032_fresh.yaml"
    echo "  配置: $CONFIG"
    echo "  从头训练 (无 warmstart): epochs=80, lr=1e-4"

    # 选择空闲 GPU
    local gpu=$(check_gpu_free)
    if [ "$gpu" == "-1" ]; then
        gpu=4  # 默认用 GPU 4
    fi

    echo "  使用 GPU ${gpu}"

    CUDA_VISIBLE_DEVICES=$gpu python scripts/train_ms_flow.py \
        --config "$CONFIG" \
        2>&1 | tee "output/logs/pose_train_${TIMESTAMP}.log"

    echo ""
    echo "━━━ Stage 3 完成: 位姿网络训练 ━━━"
}

# =============================================================================
# Stage 4: OldHospital 2DGS 几何重建 (retrain38f 复现 + 新实验)
# =============================================================================
stage_2dgs() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Stage 4: OldHospital 2DGS 重建"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if [ ! -f "dataset/OldHospital/sparse/0/points3D.ply" ]; then
        echo "  [ERROR] OldHospital COLMAP sparse 数据不存在!"
        return 1
    fi

    # ── 实验 1: 复现 retrain38f (基线) ──
    local MODEL_DIR="output/2dgs_models/OldHospital/v3_retrain38f_repro"
    if [ -d "$MODEL_DIR/point_cloud/iteration_30000" ]; then
        echo "  retrain38f 复现已完成, 跳过"
    else
        echo "  启动 retrain38f 复现 (GPU 0)..."
        mkdir -p "$MODEL_DIR"

        CUDA_VISIBLE_DEVICES=0 nohup python -u feature_3dgs/train_2dgs_geometry.py \
            --source_dir dataset/OldHospital \
            --model_dir "$MODEL_DIR" \
            --iterations 30000 \
            --batch_size 4 \
            --use_mask \
            --random_background \
            --use_appearance \
            --wildgaussians \
            --wg_output_scale 0.3 \
            --lambda_dssim 0.2 \
            --lambda_normal 0 --lambda_dist 0 --lambda_scale 0 --lambda_depth 0 \
            --densify_until_iter 15000 \
            --densify_grad_threshold 0.0002 \
            --max_gaussians 200000 \
            --opacity_reset_interval 30001 \
            --appearance_lr_init 5e-4 \
            --gaussian_emb_lr 5e-3 \
            --image_emb_lr 1e-3 \
            --appearance_reg 0.01 \
            --grad_clip_max_norm 1.0 \
            --test_iterations 1000 3000 5000 10000 15000 20000 25000 30000 \
            --save_iterations 10000 15000 20000 30000 \
            --wg_test_opt_steps 0 \
            &> "output/logs/2dgs_retrain38f_repro_${TIMESTAMP}.log" &
        local PID_38F=$!
        echo "    PID: $PID_38F"
    fi

    # ── 实验 2: retrain39a — 更大 MLP + 更高 output_scale (GPU 1) ──
    local MODEL_DIR_39A="output/2dgs_models/OldHospital/v3_retrain39a"
    if [ -d "$MODEL_DIR_39A/point_cloud/iteration_30000" ]; then
        echo "  retrain39a 已完成, 跳过"
    else
        echo "  启动 retrain39a: 256×3 MLP, output_scale=0.5, embed=64 (GPU 1)..."
        mkdir -p "$MODEL_DIR_39A"

        CUDA_VISIBLE_DEVICES=1 nohup python -u feature_3dgs/train_2dgs_geometry.py \
            --source_dir dataset/OldHospital \
            --model_dir "$MODEL_DIR_39A" \
            --iterations 50000 \
            --batch_size 4 \
            --use_mask \
            --random_background \
            --use_appearance \
            --wildgaussians \
            --no_dino_uncertainty \
            --wg_output_scale 0.5 \
            --wg_hidden_dim 256 \
            --wg_n_hidden 3 \
            --wg_image_embed_dim 64 \
            --lambda_dssim 0.2 \
            --lambda_normal 0 --lambda_dist 0 --lambda_scale 0 --lambda_depth 0 \
            --densify_until_iter 15000 \
            --densify_grad_threshold 0.0002 \
            --max_gaussians 200000 \
            --opacity_reset_interval 50001 \
            --appearance_lr_init 5e-4 \
            --gaussian_emb_lr 5e-3 \
            --image_emb_lr 1e-3 \
            --appearance_reg 0.001 \
            --grad_clip_max_norm 1.0 \
            --test_iterations 1000 5000 10000 15000 20000 25000 30000 40000 50000 \
            --save_iterations 10000 20000 30000 50000 \
            --wg_test_opt_steps 0 \
            &> "output/logs/2dgs_retrain39a_${TIMESTAMP}.log" &
        local PID_39A=$!
        echo "    PID: $PID_39A"
    fi

    # ── 实验 3: retrain39b — output_scale=1.0, 极端校正 (GPU 2) ──
    local MODEL_DIR_39B="output/2dgs_models/OldHospital/v3_retrain39b"
    if [ -d "$MODEL_DIR_39B/point_cloud/iteration_30000" ]; then
        echo "  retrain39b 已完成, 跳过"
    else
        echo "  启动 retrain39b: 256×3, output_scale=1.0, reg=0 (GPU 2)..."
        mkdir -p "$MODEL_DIR_39B"

        CUDA_VISIBLE_DEVICES=2 nohup python -u feature_3dgs/train_2dgs_geometry.py \
            --source_dir dataset/OldHospital \
            --model_dir "$MODEL_DIR_39B" \
            --iterations 50000 \
            --batch_size 4 \
            --use_mask \
            --random_background \
            --use_appearance \
            --wildgaussians \
            --no_dino_uncertainty \
            --wg_output_scale 1.0 \
            --wg_hidden_dim 256 \
            --wg_n_hidden 3 \
            --wg_image_embed_dim 64 \
            --lambda_dssim 0.2 \
            --lambda_normal 0 --lambda_dist 0 --lambda_scale 0 --lambda_depth 0 \
            --densify_until_iter 20000 \
            --densify_grad_threshold 0.0002 \
            --max_gaussians 300000 \
            --opacity_reset_interval 50001 \
            --appearance_lr_init 5e-4 \
            --gaussian_emb_lr 5e-3 \
            --image_emb_lr 1e-3 \
            --appearance_reg 0.0 \
            --grad_clip_max_norm 1.0 \
            --test_iterations 1000 5000 10000 15000 20000 30000 40000 50000 \
            --save_iterations 10000 20000 30000 50000 \
            --wg_test_opt_steps 0 \
            &> "output/logs/2dgs_retrain39b_${TIMESTAMP}.log" &
        local PID_39B=$!
        echo "    PID: $PID_39B"
    fi

    echo ""
    echo "  2DGS 实验已在后台启动。"
    echo "  监控: tail -f output/logs/2dgs_retrain*.log"
    echo "  GPU状态: nvidia-smi"
    echo ""
    echo "━━━ Stage 4 启动完成 ━━━"
}

# =============================================================================
# Main
# =============================================================================
case "$STAGE" in
    check)
        stage_check
        ;;
    extract)
        stage_check
        stage_extract
        ;;
    embed)
        stage_check
        stage_embed
        ;;
    pose)
        stage_check
        stage_pose
        ;;
    2dgs)
        stage_check
        stage_2dgs
        ;;
    all)
        stage_check
        stage_2dgs       # 2DGS 不依赖特征提取, 最先启动 (后台)
        stage_extract    # 特征提取 (~30min, 前台)
        stage_embed      # 嵌入训练 (~20min, 并行)
        stage_pose       # 位姿训练 (前台, 最长)
        ;;
    *)
        echo "未知 STAGE: $STAGE"
        echo "用法: bash scripts/run_pipeline.sh [all|check|extract|embed|pose|2dgs]"
        exit 1
        ;;
esac

echo ""
echo "============================================="
echo "  流水线完成 — $(date)"
echo "  日志: $LOG_FILE"
echo "============================================="
