#!/usr/bin/env bash
set -euo pipefail

SHARD_INDEX="${1:?usage: $0 SHARD_INDEX}"
OUTPUT_STAGE="${RELATION_OUTPUT_STAGE:-s433_v3_topology_peredge}"
ROOT="output/vfm/stage_r_matcha_joint/oldhospital/real_radio_joint_referenced_v1"
PROFILES='[{"name":"ap3p_raw_beta1_h64","minimal_set_sizes":[4],"hypotheses_per_limit":64,"candidate_probability_power":1.0,"candidate_uniform_mix":0.0,"local_optimization":false,"use_spatial_modes":false},{"name":"ap3p_local_beta1_h16","minimal_set_sizes":[4],"hypotheses_per_limit":16,"candidate_probability_power":1.0,"candidate_uniform_mix":0.0,"local_optimization":true,"use_spatial_modes":false},{"name":"m456_raw_beta1_h48","minimal_set_sizes":[4,5,6],"hypotheses_per_limit":48,"candidate_probability_power":1.0,"candidate_uniform_mix":0.0,"local_optimization":false,"use_spatial_modes":false}]'

PYTHONPATH=/root/ICLPose python -m feature_extract.tools.vfm.eval_pose_hypothesis_verification \
  --proposals "$ROOT/s8_p19_disjoint_support_probe_v1/detector_support_reranked_proposals.npz" \
  --candidate_artifact "$ROOT/s404a_p28_targetfree_q128_maplet_features_v1/features_inference_only.npz" \
  --score_artifact "$ROOT/s406_p28_targetfree_q128_ensemble_inference_v1/inference_scores.npz" \
  --score_keys ensemble__factorized_set_candidate_probability \
  --baseline_score_key ensemble__factorized_set_candidate_probability \
  --frozen_baseline_source_score_key ensemble__factorized_set_candidate_probability \
  --projected_landmark_bank "$ROOT/s8_p18_upstream_disjoint_mapper_s300_featureonly_v7/bank_mean/projected_observations_mean.npz" \
  --maplet_support_index "$ROOT/s8_p19_disjoint_maplet_index_v1/mean_k32_candidate128_support8.npz" \
  --colmap_model_dir /hy-tmp/Cambridge_stdloc/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/model_train \
  --split_json "$ROOT/s158_stage5_factorized_priorfree_s2_seed0_e2_b3072_v1/split.json" \
  --candidate_evidence "$ROOT/s408_p28_targetfree_candidate_evidence_top5_v1/candidate_evidence_v3.npz" \
  --candidate_spatial_likelihood_train "$ROOT/s414a_p28_targetfree_rgb_spatial_train_shard0of2_v1/candidate_spatial_likelihood_v7.npz,$ROOT/s414b_p28_targetfree_rgb_spatial_train_shard1of2_v1/candidate_spatial_likelihood_v7.npz" \
  --candidate_spatial_likelihood_validation "$ROOT/s411a_p28_targetfree_rgb_spatial_validation_v1/candidate_spatial_likelihood_v7.npz" \
  --candidate_spatial_likelihood_test "$ROOT/s411b_p28_targetfree_rgb_spatial_test_v1/candidate_spatial_likelihood_v7.npz" \
  --candidate_spatial_view_mixture_policy artifact_pose_view_posterior \
  --candidate_generation_spatial_view_mixture_policy artifact_pose_view_posterior \
  --candidate_spatial_log_evidence_weight 0 \
  --holdout_folds 6 \
  --immutable_baseline_pose_artifact "$ROOT/s324d_s43_immutable_selected_pose_exact_v1/selected_pose_inference_only_v1.npz" \
  --immutable_baseline_pose_evaluation_label grouped_candidate_pool__ensemble__set_candidate_probability \
  --enable_grouped_candidate_pnp --grouped_only \
  --enable_grouped_crossfit_likelihood_fallback \
  --enable_grouped_independent_shortlist_pool \
  --enable_grouped_latent_em \
  --grouped_null_score_key ensemble__factorized_set_dustbin_probability_DIAGNOSTIC_ONLY \
  --grouped_generation_mode grouped_prosac \
  --grouped_crossfit_mode token_spatial_track_maplet_purged \
  --grouped_crossfit_spatial_fold_policy cell_rotated_balanced \
  --grouped_rank_fold_count 2 \
  --grouped_sampling_temperatures 0.5,1,2 \
  --grouped_prosac_profiles_json "$PROFILES" \
  --grouped_prosac_verification_top_k 128 \
  --grouped_prosac_shortlist_evidence_mode base_coordinate_then_full_spatial \
  --grouped_prosac_spatial_rescore_top_k 256 \
  --grouped_prosac_shortlist_selection_mode profile_pose_diverse \
  --grouped_prosac_shortlist_diverse_count 64 \
  --grouped_prosac_shortlist_min_per_profile 4 \
  --grouped_latent_em_seed_count 256 \
  --candidate_relation_feature_neighbor_k 4 \
  --candidate_relation_feature_max_modes 2 \
  --export_train_grouped_hypotheses \
  --export_grouped_hypothesis_artifact \
  --query_shard_count 21 --query_shard_index "$SHARD_INDEX" \
  --development_cross_block_audit \
  --output_dir "$ROOT/${OUTPUT_STAGE}_shard$(printf '%02d' "$SHARD_INDEX")of21_v1"
