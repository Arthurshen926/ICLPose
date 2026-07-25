#!/usr/bin/env bash
# Score a deterministic shard partition of fixed-maplet lifted LoFTR P1.
# Usage: $0 <root> <gpu> <start-shard> [stride] [visual|xyz_permutation_control|both]

set -euo pipefail

ROOT=${1:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
GPU=${2:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
START=${3:?"usage: $0 <root> <gpu> <start-shard> [stride] [variant]"}
STRIDE=${4:-2}
VARIANT=${5:-both}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CACHE_ROOT=${LOFTR_PAIR_CACHE_ROOT:-"$ROOT/s897_s1_loftr_anchor_full_global_frozenprovenance_v1"}
COLMAP_MODEL_DIR=${COLMAP_MODEL_DIR:-/hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/model_train}
IMAGE_ROOT=${IMAGE_ROOT:-/hy-tmp/Cambridge_stdloc/OldHospital}
LOFTR_CHECKPOINT=${LOFTR_CHECKPOINT:-/root/.cache/torch/hub/checkpoints/loftr_outdoor.ckpt}

case "$VARIANT" in
  visual)
    VARIANTS=(visual)
    ;;
  xyz_permutation_control)
    VARIANTS=(xyz_permutation_control)
    ;;
  both)
    VARIANTS=(visual xyz_permutation_control)
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
  --evidence-layout candidate_specific_support_view_v1
  --maplet-support-index "$ROOT/s8_p19_disjoint_maplet_index_v1/mean_k32_candidate128_support8.npz"
  --support-geometry-index "$ROOT/s4_l67_multisequence_support_geometry_v1/support_observation_geometry.npz"
  --projected-landmark-bank "$ROOT/s8_p18_upstream_disjoint_mapper_s300_featureonly_v7/bank_mean/projected_observations_mean.npz"
  --colmap-model-dir "$COLMAP_MODEL_DIR"
  --image-root "$IMAGE_ROOT"
  --loftr-checkpoint "$LOFTR_CHECKPOINT"
  --fixed-support-view-count 2
  --base-support-reliability 0.75
  --hypothesis-batch-size 512
  --device cuda:0
)

for SHARD in $(seq "$START" "$STRIDE" 20); do
  SHARD_TAG=$(printf '%02d' "$SHARD")
  HYPOTHESIS="$ROOT/s436_v4_topology_repeat_shard${SHARD_TAG}of21_v1/grouped_hypotheses_inference_only_v1.npz"
  BASELINE="$ROOT/s846_s0_fixedposterior_top20_validation_shard${SHARD_TAG}of21_v1/independent_landmark_hypothesis_scores_v1.npz"
  # The grouped-hypothesis source may pack several queries.  The immutable S0
  # baseline is the exact one-query frozen evaluation subset consumed by the
  # scorer, so it is the only valid owner for pair-cache resolution.
  QUERY_ID=$(PYTHONPATH=. python -c 'import numpy as np, sys; p=sys.argv[1]; z=np.load(p, allow_pickle=False); ids=np.unique(np.asarray(z["query_ids"]).astype(str)); z.close(); assert len(ids)==1, ids; print(ids[0])' "$BASELINE")
  QUERY_TOKEN=$(printf '%s' "$QUERY_ID" | tr '/.' '__')
  mapfile -t CACHE_CANDIDATES < <(find "$CACHE_ROOT" -type f -path '*/pair_caches/*' -name "*_${QUERY_TOKEN}_*.npz" -print | sort)
  if [[ ${#CACHE_CANDIDATES[@]} -ne 1 ]]; then
    printf 'expected exactly one LoFTR cache for %s, found %s\n' "$QUERY_ID" "${#CACHE_CANDIDATES[@]}" >&2
    printf '%s\n' "${CACHE_CANDIDATES[@]:-}" >&2
    exit 3
  fi
  for EVIDENCE_VARIANT in "${VARIANTS[@]}"; do
    if [[ "$EVIDENCE_VARIANT" == visual ]]; then
      OUTPUT_DIR="$ROOT/s1293_lifted_loftr_maptoq_candidateview_visual_v6_shard${SHARD_TAG}of21"
    else
      OUTPUT_DIR="$ROOT/s1294_lifted_loftr_maptoq_candidateview_control_v6_shard${SHARD_TAG}of21"
    fi
    OUTPUT="$OUTPUT_DIR/scores.npz"
    if [[ -f "$OUTPUT" ]]; then
      continue
    fi
    mkdir -p "$OUTPUT_DIR"
    PYTHONPATH=. CUDA_VISIBLE_DEVICES="$GPU" python \
      "$SCRIPT_DIR/score_frozen_lifted_loftr_map_to_query_pose_evidence.py" \
      --hypothesis-artifact "$HYPOTHESIS" \
      --baseline-score-artifact "$BASELINE" \
      --loftr-pair-cache "${CACHE_CANDIDATES[0]}" \
      "${COMMON[@]}" \
      --evidence-variant "$EVIDENCE_VARIANT" \
      --output "$OUTPUT" \
      >"$OUTPUT_DIR/run.log" 2>&1
  done
done
