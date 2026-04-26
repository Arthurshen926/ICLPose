#!/usr/bin/env bash
set -euo pipefail

SCENE_ROOT="${SCENE_ROOT:-/hy-tmp/Cambridge_stdloc/OldHospital}"
PROCESSED_DIR="${SCENE_ROOT}/processed"
RADIO_REPO="${RADIO_REPO:-feature_extract/checkpoints/RADIO}"
DEVICE="${DEVICE:-cuda}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6+PTX}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

RESULT_ROOT="${RESULT_ROOT:-/root/ICLPose/result}"
FEATURE_DIR="${RESULT_ROOT}/feature_extract/features_radio_dual/cambridge_oldhospital_processed"
TEACHER_VIS_DIR="${RESULT_ROOT}/feature_extract/teacher_visuals/cambridge_oldhospital_processed"
GAUSS_CFG="feature_gaussian/configs/joint_radio_dual_cambridge_oldhospital_processed_pilot.yaml"
FIELD_CFG="feature_field/configs/dcff_cambridge_oldhospital_processed_joint_pilot.yaml"
FIELD_RECON_CFG="feature_field/configs/reconstruction_cambridge_oldhospital_processed_joint_pilot.yaml"
STUDENT_CFG="feature_extract/configs/joint_radio_dcff_cambridge_oldhospital_processed_student_pilot.yaml"
GAUSS_EXP="joint_radio_dual_cambridge_oldhospital_processed_pilot"
FIELD_EXP="dcff_cambridge_oldhospital_processed_joint_pilot"

python -m feature_extract.extract_radio_dual_features \
  --source_dir "${PROCESSED_DIR}" \
  --output_dir "${FEATURE_DIR}" \
  --target_dim 64 \
  --radio_repo "${RADIO_REPO}" \
  --device "${DEVICE}"

python -m feature_extract.visualize_teacher_features \
  --source_dir "${PROCESSED_DIR}" \
  --feature_dir "${FEATURE_DIR}" \
  --output_dir "${TEACHER_VIS_DIR}" \
  --radio_repo "${RADIO_REPO}" \
  --device "${DEVICE}"

python -m feature_gaussian.train --config "${GAUSS_CFG}"

python -m feature_gaussian.evaluate \
  --config "${GAUSS_CFG}" \
  --split test \
  --camera_split test \
  --num_samples 4 \
  --max_metrics_cameras 64 \
  --output_dir "${RESULT_ROOT}/feature_gaussian/${GAUSS_EXP}/evaluation_test"

python -m feature_field.train --config "${FIELD_CFG}"

python -m feature_field.eval_dcff_metrics \
  --checkpoint "${RESULT_ROOT}/feature_field/${FIELD_EXP}/checkpoints/best.pth" \
  --source_dir "${SCENE_ROOT}" \
  --images_subdir processed \
  --feature_dir "${FEATURE_DIR}" \
  --camera_split test \
  --max_cameras 64 \
  --output_json "${RESULT_ROOT}/feature_field/${FIELD_EXP}/metrics_test.json"

python -m feature_field.visualize_reconstruction \
  --checkpoint "${RESULT_ROOT}/feature_field/${FIELD_EXP}/checkpoints/best.pth" \
  --source_dir "${SCENE_ROOT}" \
  --feature_dir "${FEATURE_DIR}" \
  --images_subdir processed \
  --output_dir "${RESULT_ROOT}/feature_field/${FIELD_EXP}/visualizations" \
  --camera_idx 0

python -m feature_extract.train --config "${STUDENT_CFG}"

echo "Teacher features: ${FEATURE_DIR}"
echo "Teacher visuals:  ${TEACHER_VIS_DIR}"
echo "Gaussian output:  ${RESULT_ROOT}/feature_gaussian/${GAUSS_EXP}"
echo "DCFF output:      ${RESULT_ROOT}/feature_field/${FIELD_EXP}"
echo "Student output:   ${RESULT_ROOT}/feature_extract/joint_radio_dcff_cambridge_oldhospital_processed_student_pilot"
