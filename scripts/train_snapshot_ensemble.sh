#!/bin/bash
# Snapshot Ensemble Training Configurations
# Uses GPU 3, cosine warm restarts with different schedules
set -e

GPU=5
BASE="python -m feature_retrieval.snapshot_ensemble_train --gpu ${GPU}"

echo "============================================================"
echo "Config A: T0=500, T_mult=1, 10 cycles, seed 314 (baseline)"
echo "  Total epochs: 5000, 10 snapshots"
echo "============================================================"
${BASE} \
    --seed 314 \
    --T0 500 --T_mult 1 --n_cycles 10 \
    --lr_max 0.001 --lr_min 1e-6 \
    --exp_name exp15a_snapshot_T500x10

echo ""
echo "============================================================"
echo "Config B: T0=200, T_mult=2, 4 cycles, seed 314 (increasing)"
echo "  Cycles: 200,400,800,1600 = 3000 total, 4 snapshots"
echo "============================================================"
${BASE} \
    --seed 314 \
    --T0 200 --T_mult 2 --n_cycles 4 \
    --lr_max 0.001 --lr_min 1e-6 \
    --exp_name exp15b_snapshot_T200xmult2

echo ""
echo "============================================================"
echo "Config C: T0=500, T_mult=1, 10 cycles, seed 123"
echo "  Same as A but different seed for comparison"
echo "============================================================"
${BASE} \
    --seed 123 \
    --T0 500 --T_mult 1 --n_cycles 10 \
    --lr_max 0.001 --lr_min 1e-6 \
    --exp_name exp15c_snapshot_T500x10_seed123

echo ""
echo "============================================================"
echo "Config D: T0=300, T_mult=1, 15 cycles, seed 314"
echo "  Shorter cycles, more snapshots (4500 total, 15 snapshots)"
echo "============================================================"
${BASE} \
    --seed 314 \
    --T0 300 --T_mult 1 --n_cycles 15 \
    --lr_max 0.001 --lr_min 1e-6 \
    --exp_name exp15d_snapshot_T300x15

echo ""
echo "============================================================"
echo "Config E: T0=500, T_mult=1, 10 cycles, lr_max=0.003"
echo "  Aggressive restarts for more basin diversity"
echo "============================================================"
${BASE} \
    --seed 314 \
    --T0 500 --T_mult 1 --n_cycles 10 \
    --lr_max 0.003 --lr_min 1e-6 \
    --exp_name exp15e_snapshot_T500x10_aggr

echo ""
echo "All snapshot ensemble configs complete!"
