#!/usr/bin/env bash
# Score immutable S0 validation hypotheses with candidate-specific maplet P1.
# Usage: $0 <root> <gpu> <start-shard> [stride] [visual|control|both]

set -euo pipefail

ROOT=${1:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
GPU=${2:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
START=${3:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
STRIDE=${4:-2}
VARIANT=${5:-both}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COLMAP_MODEL_DIR=${COLMAP_MODEL_DIR:-/hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/model_train}
MAPLET_PROFILES=${MAPLET_PROFILES:-radio_final_wide,radio_intermediate_wide,alike_wide}
MAPLET_NEIGHBORS_PER_QUADRANT=${MAPLET_NEIGHBORS_PER_QUADRANT:-1}
MAPLET_VERIFICATION_POINT_COUNT=${MAPLET_VERIFICATION_POINT_COUNT:-64}
VISUAL_OUTPUT_PREFIX=${VISUAL_OUTPUT_PREFIX:-s1313_poseconditioned_maplet_visual_full}
CONTROL_OUTPUT_PREFIX=${CONTROL_OUTPUT_PREFIX:-s1314_poseconditioned_maplet_control_full}

case "$VARIANT" in
  visual)
    VARIANTS=(visual)
    ;;
  control)
    VARIANTS=(support_descriptor_permutation_control)
    ;;
  both)
    VARIANTS=(visual support_descriptor_permutation_control)
    ;;
  *)
    printf 'unknown evidence variant: %s\n' "$VARIANT" >&2
    exit 2
    ;;
esac

COMMON=(
  --detector-query-cache "$ROOT/s8_p19_disjoint_detector_global_all105_v1/detector_query_cache.npz"
  --proposals "$ROOT/s8_p19_disjoint_support_probe_v1/detector_support_reranked_proposals.npz"
  --candidate-artifact "$ROOT/s404a_p28_targetfree_q128_maplet_features_v1/features_inference_only.npz"
  --fixed-candidate-prior-overlay "$ROOT/s480_radio_multiscale_absolute_prior_overlay_v1/candidate_prior_overlay.npz"
  --fixed-candidate-support-view-overlay "$ROOT/s1291_candidate_maplet_support_view_overlay_v1/support_view_overlay.npz"
  --maplet-support-index "$ROOT/s8_p19_disjoint_maplet_index_v1/mean_k32_candidate128_support8.npz"
  --support-geometry-index "$ROOT/s4_l67_multisequence_support_geometry_v1/support_observation_geometry.npz"
  --projected-landmark-bank "$ROOT/s8_p18_upstream_disjoint_mapper_s300_featureonly_v7/bank_mean/projected_observations_mean.npz"
  --neighbor-topology-cache "$ROOT/s1055_s10_sparsemaplet_v2_topologycache_v1/sparse_maplet_neighbor_topology_cache.npz"
  --colmap-model-dir "$COLMAP_MODEL_DIR"
  --radio-final-context-cache "$ROOT/s775_currentv3_radio_final_grid4_grid16_pca64_maplet790_v1/radio_final_context_pca.npz"
  --radio-intermediate-context-cache "$ROOT/s923_s1_radio_intermediate_grid16_pca256_maplet790_v1/radio_intermediate_image_context_pca.npz"
  --alike-spatial-context-cache "$ROOT/s778_currentv3_alike_grid32_image_context_v1/alike_image_spatial_context.npz"
  --profiles "$MAPLET_PROFILES"
  --verification-point-count "$MAPLET_VERIFICATION_POINT_COUNT"
  --verification-grid-size 8
  --fit-exclusion-radius-px 16
  --neighbors-per-quadrant "$MAPLET_NEIGHBORS_PER_QUADRANT"
  --minimum-active-neighbors-per-quadrant 1
  --minimum-active-quadrants 3
  --hypothesis-batch-size 1024
  --template-chunk-size 4096
  --device cuda:0
)

for SHARD in $(seq "$START" "$STRIDE" 20); do
  SHARD_TAG=$(printf '%02d' "$SHARD")
  HYPOTHESIS="$ROOT/s436_v4_topology_repeat_shard${SHARD_TAG}of21_v1/grouped_hypotheses_inference_only_v1.npz"
  BASELINE="$ROOT/s846_s0_fixedposterior_top20_validation_shard${SHARD_TAG}of21_v1/independent_landmark_hypothesis_scores_v1.npz"
  for EVIDENCE_VARIANT in "${VARIANTS[@]}"; do
    if [[ "$EVIDENCE_VARIANT" == visual ]]; then
      OUTPUT_DIR="$ROOT/${VISUAL_OUTPUT_PREFIX}_shard${SHARD_TAG}of21_v1"
    else
      OUTPUT_DIR="$ROOT/${CONTROL_OUTPUT_PREFIX}_shard${SHARD_TAG}of21_v1"
    fi
    OUTPUT="$OUTPUT_DIR/scores.npz"
    if [[ -f "$OUTPUT" ]]; then
      continue
    fi
    mkdir -p "$OUTPUT_DIR"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU" python \
      "$SCRIPT_DIR/score_frozen_pose_conditioned_maplet_appearance.py" \
      --hypothesis-artifact "$HYPOTHESIS" \
      --baseline-score-artifact "$BASELINE" \
      "${COMMON[@]}" \
      --evidence-variant "$EVIDENCE_VARIANT" \
      --output "$OUTPUT" \
      >"$OUTPUT_DIR/run.log" 2>&1
  done
done
