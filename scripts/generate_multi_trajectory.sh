#!/bin/bash
# 批量生成多组虚拟轨迹数据集
# 设计: 5组轨迹, 每组200帧 = 1000帧总计
# 这样可以覆盖场景的不同区域并提供足够的训练数据

set -e

PROJECT_DIR="/home/yons/Projects/ICLPose"
OUTPUT_BASE="$PROJECT_DIR/dataset/room_0/Synthetic_Multi"
REFERENCE_TRAJ="$PROJECT_DIR/dataset/room_0/Sequence_1/traj_tum.txt"
PLY_PATH="$PROJECT_DIR/dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply"
AE_MODEL="$PROJECT_DIR/dataset/room_0/Sequence_1/ae_models/ae_fused.pth"

# 参数
NUM_GROUPS=5
FRAMES_PER_GROUP=200
WAYPOINTS_PER_GROUP=15

echo "============================================================"
echo "批量生成多组虚拟轨迹数据集"
echo "============================================================"
echo "  轨迹组数: $NUM_GROUPS"
echo "  每组帧数: $FRAMES_PER_GROUP"
echo "  总帧数: $((NUM_GROUPS * FRAMES_PER_GROUP))"
echo "  输出目录: $OUTPUT_BASE"
echo ""

mkdir -p "$OUTPUT_BASE"

# 为每组生成数据
for i in $(seq 1 $NUM_GROUPS); do
    GROUP_DIR="$OUTPUT_BASE/traj_$i"
    SEED=$((42 + i * 10))
    
    echo ""
    echo "============================================================"
    echo "处理轨迹组 $i / $NUM_GROUPS (seed=$SEED)"
    echo "============================================================"
    
    # Step 1: 生成轨迹
    echo "[Step 1/4] 生成随机轨迹..."
    python "$PROJECT_DIR/scripts/generate_random_trajectory.py" \
        --reference_traj "$REFERENCE_TRAJ" \
        --output_file "$GROUP_DIR/traj_tum.txt" \
        --num_frames $FRAMES_PER_GROUP \
        --num_waypoints $WAYPOINTS_PER_GROUP \
        --seed $SEED \
        --visualize
    
    # Step 2: 渲染RGB
    echo "[Step 2/4] 3DGS渲染..."
    python "$PROJECT_DIR/scripts/render_3dgs.py" \
        --poses_file "$GROUP_DIR/traj_tum.txt" \
        --output_dir "$GROUP_DIR/rgb" \
        --ply_path "$PLY_PATH"
    
    # Step 3: 特征提取 (带可视化, 每组首帧)
    echo "[Step 3/4] 特征提取..."
    HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python "$PROJECT_DIR/scripts/extract_fused_features.py" \
        --input_dir "$GROUP_DIR/rgb" \
        --output_dir "$GROUP_DIR/features_raw" \
        --visualize --vis_interval $FRAMES_PER_GROUP
    
    # Step 4: 特征压缩
    echo "[Step 4/4] 特征压缩..."
    python "$PROJECT_DIR/scripts/compress_features.py" \
        --input_dir "$GROUP_DIR/features_raw" \
        --output_dir "$GROUP_DIR/fused_feat" \
        --model_path "$AE_MODEL" \
        --visualize --vis_interval $FRAMES_PER_GROUP
    
    echo "✓ 轨迹组 $i 完成!"
done

# 生成统计信息
echo ""
echo "============================================================"
echo "生成完成! 统计信息:"
echo "============================================================"
for i in $(seq 1 $NUM_GROUPS); do
    GROUP_DIR="$OUTPUT_BASE/traj_$i"
    N_RGB=$(ls "$GROUP_DIR/rgb/"*.png 2>/dev/null | wc -l)
    N_FEAT=$(ls "$GROUP_DIR/fused_feat/"*.npy 2>/dev/null | wc -l)
    echo "  traj_$i: RGB=$N_RGB, 特征=$N_FEAT"
done

echo ""
du -sh "$OUTPUT_BASE"/*
echo ""
echo "✓ 全部完成!"
