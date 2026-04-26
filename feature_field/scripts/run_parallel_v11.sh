#!/bin/bash
# Run 4 DCFF v11 experiments in parallel on 4 GPUs
# GPU 0: v11a (B=32, cs=0.1)
# GPU 1: v11b (B=32, cs=0.3, tv=0.05)
# GPU 3: v11c (B=24, cs=0.1)
# GPU 4: v11d (B=32, cs=0.1, higher LR)

set -e

EXP_A="dcff_oldhospital_v11a_joint_geo"
EXP_B="dcff_oldhospital_v11b_joint_geo"
EXP_C="dcff_oldhospital_v11c_joint_geo"
EXP_D="dcff_oldhospital_v11d_joint_geo"

CFG_A="feature_field/configs/dcff_oldhospital_v11a_joint_geo.yaml"
CFG_B="feature_field/configs/dcff_oldhospital_v11b_joint_geo.yaml"
CFG_C="feature_field/configs/dcff_oldhospital_v11c_joint_geo.yaml"
CFG_D="feature_field/configs/dcff_oldhospital_v11d_joint_geo.yaml"

LOG_A="feature_field/output/${EXP_A}_launch.log"
LOG_B="feature_field/output/${EXP_B}_launch.log"
LOG_C="feature_field/output/${EXP_C}_launch.log"
LOG_D="feature_field/output/${EXP_D}_launch.log"

echo "========================================"
echo "DCFF v11 Parallel Training"
echo "========================================"
echo "v11a: GPU0, B=32, channel_std=0.1, TV=0.03"
echo "v11b: GPU1, B=32, channel_std=0.3, TV=0.05"
echo "v11c: GPU3, B=24, channel_std=0.1, TV=0.03"
echo "v11d: GPU4, B=32, channel_std=0.1, TV=0.03, higher LR"
echo "========================================"

# Kill any existing DCFF processes
pkill -f "feature_field.train" 2>/dev/null || true
sleep 2

# Launch all 4 experiments
CUDA_VISIBLE_DEVICES=0 python -m feature_field.train --config $CFG_A > $LOG_A 2>&1 &
PID_A=$!
echo "Started v11a on GPU0 (PID $PID_A)"

CUDA_VISIBLE_DEVICES=1 python -m feature_field.train --config $CFG_B > $LOG_B 2>&1 &
PID_B=$!
echo "Started v11b on GPU1 (PID $PID_B)"

CUDA_VISIBLE_DEVICES=3 python -m feature_field.train --config $CFG_C > $LOG_C 2>&1 &
PID_C=$!
echo "Started v11c on GPU3 (PID $PID_C)"

CUDA_VISIBLE_DEVICES=4 python -m feature_field.train --config $CFG_D > $LOG_D 2>&1 &
PID_D=$!
echo "Started v11d on GPU4 (PID $PID_D)"

echo ""
echo "All 4 experiments launched. PIDs: $PID_A $PID_B $PID_C $PID_D"
echo "Logs: $LOG_A, $LOG_B, $LOG_C, $LOG_D"
echo ""

# Wait and monitor
sleep 30
for i in {1..60}; do
    sleep 60
    echo "=== Check $(date) ==="
    ps aux | grep -c "feature_field.train" | xargs echo "Running processes:"
    echo -n "GPU0: "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 0
    echo -n "GPU1: "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 1
    echo -n "GPU3: "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 3
    echo -n "GPU4: "; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -i 4
    echo "--- Latest logs ---"
    for log in $LOG_A $LOG_B $LOG_C $LOG_D; do
        echo "=== $(basename $log) ==="
        tail -2 $log 2>/dev/null || echo "(no output yet)"
    done
done

# Keep running
wait
