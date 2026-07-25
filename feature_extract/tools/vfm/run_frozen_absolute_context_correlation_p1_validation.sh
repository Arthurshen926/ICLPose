#!/usr/bin/env bash
# Run a deterministic partition of the frozen multiscale absolute-context P1.
# Usage: $0 <root> <gpu> <start-shard> [stride] [visual|support_channel_permutation_control|both]

set -euo pipefail

ROOT=${1:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
GPU=${2:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
START=${3:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
STRIDE=${4:-2}
VARIANT=${5:-both}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COLMAP_MODEL_DIR=${COLMAP_MODEL_DIR:-/hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/model_train}

case "$VARIANT" in
  visual)
    VARIANTS=(visual)
    ;;
  support_channel_permutation_control)
    VARIANTS=(support_channel_permutation_control)
    ;;
  both)
    VARIANTS=(visual support_channel_permutation_control)
    ;;
  *)
    printf 'unknown evidence variant: %s\n' "$VARIANT" >&2
    exit 2
    ;;
esac

COMMON=(
  --profile_set context_correlation_v2
  --detector_query_cache "$ROOT/s8_p19_disjoint_detector_global_all105_v1/detector_query_cache.npz"
  --proposals "$ROOT/s8_p19_disjoint_support_probe_v1/detector_support_reranked_proposals.npz"
  --candidate_artifact "$ROOT/s404a_p28_targetfree_q128_maplet_features_v1/features_inference_only.npz"
  --fixed_candidate_prior_overlay "$ROOT/s480_radio_multiscale_absolute_prior_overlay_v1/candidate_prior_overlay.npz"
  --mixed_verification_points_artifact "$ROOT/s1223_p1_mixed_multiscale_trainval_currentfit_v1/mixed_verification_points.npz"
  --maplet_support_index "$ROOT/s8_p19_disjoint_maplet_index_v1/mean_k32_candidate128_support8.npz"
  --support_geometry_index "$ROOT/s4_l67_multisequence_support_geometry_v1/support_observation_geometry.npz"
  --projected_landmark_bank "$ROOT/s8_p18_upstream_disjoint_mapper_s300_featureonly_v7/bank_mean/projected_observations_mean.npz"
  --colmap_model_dir "$COLMAP_MODEL_DIR"
  --radio_final_context_cache "$ROOT/s595_s1i_radio_final_grid16_pca64_train657_v1/radio_final_context_pca.npz"
  --radio_intermediate_context_cache "$ROOT/s601_s1j_radio_intermediate_grid16_pca64_train657_v1/radio_intermediate_image_context_pca.npz"
  --alike_spatial_context_cache "$ROOT/s602_s1j_alike_grid16_grid32_image_context_v1/alike_image_spatial_context.npz"
  --context_contract "$ROOT/s1199_p5_dual_head_raw_layout_context_contract_v5/contract.json"
  --hypothesis_batch_size 128
  --lookup_point_batch_size 16
  --lookup_anchor_batch_size 16
  --device cuda:0
)

for SHARD in $(seq "$START" "$STRIDE" 20); do
  SHARD_TAG=$(printf "%02d" "$SHARD")
  HYPOTHESIS="$ROOT/s436_v4_topology_repeat_shard${SHARD_TAG}of21_v1/grouped_hypotheses_inference_only_v1.npz"
  BASELINE="$ROOT/s846_s0_fixedposterior_top20_validation_shard${SHARD_TAG}of21_v1/independent_landmark_hypothesis_scores_v1.npz"
  for EVIDENCE_VARIANT in "${VARIANTS[@]}"; do
    if [[ "$EVIDENCE_VARIANT" == visual ]]; then
      OUTPUT_DIR="$ROOT/s1253_p1_context_correlation_visual_full_shard${SHARD_TAG}of21_v2"
    else
      OUTPUT_DIR="$ROOT/s1254_p1_context_correlation_control_full_shard${SHARD_TAG}of21_v2"
    fi
    OUTPUT="$OUTPUT_DIR/scores.npz"
    SIDECAR="$OUTPUT_DIR/scores_point_sidecar.npz"
    if [[ -f "$OUTPUT" && -f "$SIDECAR" ]]; then
      continue
    fi
    mkdir -p "$OUTPUT_DIR"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU" python \
      "$SCRIPT_DIR/score_frozen_absolute_phase_pose_evidence.py" \
      --hypothesis_artifact "$HYPOTHESIS" \
      --baseline_score_artifact "$BASELINE" \
      "${COMMON[@]}" \
      --evidence_variant "$EVIDENCE_VARIANT" \
      --output "$OUTPUT" \
      >"$OUTPUT_DIR/run.log" 2>&1
  done
done
