#!/usr/bin/env bash
# =============================================================================
# cleanup_output.sh — 迁移前激进清理脚本
# 策略：激进清理 + 保留摘要证据
# 使用方法：
#   bash cleanup_output.sh --dry-run   # 仅打印，不删除
#   bash cleanup_output.sh             # 实际执行
# =============================================================================
set -euo pipefail

OUTPUT="/root/ICLPose-loc/output"
ARCHIVE="$OUTPUT/_migration_archive"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
  echo "[DRY-RUN MODE] 只打印操作，不删除任何文件"
fi

# ----------------------------------------------------------
# 辅助函数
# ----------------------------------------------------------
safe_rm() {
  local target="$1"
  if [[ -e "$target" || -L "$target" ]]; then
    if $DRY_RUN; then
      echo "[DRY-RUN] rm -rf '$target'"
    else
      echo "[DELETE] $target"
      rm -rf "$target"
    fi
  fi
}

safe_rm_glob() {
  # 删除满足 glob 的文件（不扩展到不存在的文件）
  local pattern="$1"
  while IFS= read -r -d '' f; do
    safe_rm "$f"
  done < <(find "$OUTPUT" -path "$pattern" -print0 2>/dev/null)
}

keep_summaries_then_rm_dir() {
  # 从目录 $1 中把 summary/report/results/config 文件复制到归档，再删整目录
  local src="$1"
  local label="$2"
  local dest="$ARCHIVE/summaries/$label"
  if [[ ! -d "$src" ]]; then return; fi
  if ! $DRY_RUN; then
    mkdir -p "$dest"
    # 复制所有摘要文件
    find "$src" -maxdepth 3 \( \
      -name "summary.json" -o -name "summary.txt" \
      -o -name "results.json" -o -name "results.txt" \
      -o -name "report.md" -o -name "report.txt" \
      -o -name "config.yaml" -o -name "config.json" \
      -o -name "training_log.json" \
    \) -exec cp --parents {} "$dest/" \; 2>/dev/null || true
    echo "[ARCHIVE] $src → $dest"
  else
    echo "[DRY-RUN] archive summaries from '$src' → '$dest'"
    echo "[DRY-RUN] rm -rf '$src'"
    return
  fi
  rm -rf "$src"
  echo "[DELETE] $src"
}

echo "======================================================================="
echo "Phase A: 验证主线必保资产"
echo "======================================================================="
KEEP_ASSETS=(
  "$OUTPUT/feature_field/dcff_oldhospital_v15d_v14b_coarse_smooth_frozen/checkpoints/best.pth"
  "$OUTPUT/feature_extract/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4/checkpoints/best.pth"
  "$OUTPUT/feature_extract/joint_radio_dcff_oh_v5d_joint_fine_schedule/checkpoints/best.pth"
  "$OUTPUT/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall/checkpoints/best.pth"
  "$OUTPUT/feature_retrieval/pose_regression/exp29a_direct_both_seed123/init_top5_learned.npz"
  "$OUTPUT/feature_retrieval/pose_regression/best_ensemble_init_top5.npz"
)
ASSET_OK=true
for asset in "${KEEP_ASSETS[@]}"; do
  if [[ -f "$asset" ]]; then
    echo "[OK]   $asset"
  else
    echo "[WARN] 缺少主线资产: $asset"
    ASSET_OK=false
  fi
done
if ! $ASSET_OK; then
  echo "[ERROR] 主线资产不完整，中止清理！检查以上 WARN 行。"
  exit 1
fi
echo ""

# 创建归档目录
if ! $DRY_RUN; then
  mkdir -p "$ARCHIVE/summaries"
  echo "[CREATE] $ARCHIVE/summaries"
fi

echo "======================================================================="
echo "Phase B: 删除高冗余 / 可再生大体积内容"
echo "======================================================================="

# ---- B1: 所有 qual_real_init PNG 目录 ----
echo "--- B1: 删除 qual_real_init 目录 (PNG 可视化) ---"
# feature_retrieval 下每个 real_init_* 目录的 qual_real_init 子目录
for parent in "$OUTPUT/feature_retrieval"/real_init_* "$OUTPUT/feature_retrieval"/two_stage_smoke_eval*; do
  [[ -d "$parent/qual_real_init" ]] && safe_rm "$parent/qual_real_init"
done
# pose_refine 下各实验的 qual_real_init 子目录
for parent in "$OUTPUT/pose_refine"/concat_loc_*; do
  [[ -d "$parent/qual_real_init" ]] && safe_rm "$parent/qual_real_init"
done

# ---- B2: feature_extract 中可再生的 features_* 目录 ----
echo "--- B2: 删除 features_query_student_* 等可再生特征缓存 ---"
for d in \
  "$OUTPUT/feature_extract/features_query_student_v5g" \
  "$OUTPUT/feature_extract/features_query_student_v5l_pointwise_featsharp_full_fsm_v1_b4" \
  "$OUTPUT/feature_extract/features_query_student_v5l_pointwise_featsharp_full_fsm_v1_b4_lightcoarse" \
  "$OUTPUT/feature_extract/features_query_student_v5l_pointwise_featsharp_full_fsm_v1_b4_stacked" \
  "$OUTPUT/feature_extract/features_query_student_v5l_pointwise_featsharp_full_fsm_v1_geomcorr2e_smoke" \
  "$OUTPUT/feature_extract/features_query_student_v5l_pointwise_featsharp_full_fsm_v1_geomcorr_smoke" \
  "$OUTPUT/feature_extract/features_query_student_v5l_stablemap_geomcorr2e_smoke" \
  "$OUTPUT/feature_extract/features_query_student_v5l_stablemap_geomcorr2e_smoke_colmapid" \
  "$OUTPUT/feature_extract/features_multiscale_compressed" \
  "$OUTPUT/feature_extract/features_radio_dual" \
  "$OUTPUT/feature_extract/features_radio_dual_128" \
  "$OUTPUT/feature_extract/features_radio_dual_256" \
  "$OUTPUT/feature_extract/teacher_visuals" \
  "$OUTPUT/feature_extract/visualizations" \
  "$OUTPUT/feature_extract/scr_radio_oh_v1" \
  "$OUTPUT/feature_extract/scr_radio_oh_v2" \
  "$OUTPUT/feature_extract/logs" \
; do
  safe_rm "$d"
done

# ---- B3: feature_field vis_* 目录 ----
echo "--- B3: 删除 feature_field vis_* 可视化目录 ---"
while IFS= read -r -d '' d; do
  safe_rm "$d"
done < <(find "$OUTPUT/feature_field" -maxdepth 1 -type d -name "vis_*" -print0 2>/dev/null)
safe_rm "$OUTPUT/feature_field/eval_p1_final"
safe_rm "$OUTPUT/feature_field/postprocess_metrics"

# ---- B4: pose_refine 可视化和调试目录 ----
echo "--- B4: 删除 pose_refine 可视化目录 ---"
safe_rm "$OUTPUT/pose_refine/render_compare_eval"
safe_rm "$OUTPUT/pose_refine/pnp_refine_eval"
safe_rm "$OUTPUT/pose_refine/pipeline_eval"
safe_rm "$OUTPUT/pose_refine/logs"

# ---- B5: pipeline_eval ----
echo "--- B5: 删除 pipeline_eval 目录 ---"
safe_rm "$OUTPUT/pipeline_eval"

# ---- B6: 所有 *.launch.log / *.watch.log 文件 ----
echo "--- B6: 删除 *.launch.log / *.watch.log 文件 ---"
for logdir in \
  "$OUTPUT/feature_extract" "$OUTPUT/feature_field" \
  "$OUTPUT/pose_refine" "$OUTPUT/feature_retrieval" \
; do
  for f in "$logdir"/*.launch.log "$logdir"/*.watch.log; do
    [[ -f "$f" ]] && safe_rm "$f"
  done
done

# ---- B7: 旧版本训练日志 ----
echo "--- B7: 删除旧训练日志 ---"
for logdir in \
  "$OUTPUT/feature_extract" "$OUTPUT/feature_field" \
  "$OUTPUT/pose_refine" "$OUTPUT/feature_retrieval" \
; do
  for f in "$logdir"/*_train.log "$logdir"/*.train.log "$logdir"/*.log; do
    [[ -f "$f" ]] && safe_rm "$f"
  done
done
safe_rm "$OUTPUT/pose_refine/v2_train.log"
safe_rm "$OUTPUT/pose_refine/v3_lownoise_train.log"
safe_rm "$OUTPUT/feature_retrieval/radio_loc_oh_v3/radio_loc_oh_v3_train.log"
safe_rm "$OUTPUT/feature_retrieval/radio_loc_oh_v3_train.log"

echo ""
echo "======================================================================="
echo "Phase C: 删除已验证无效 / 历史实验分支"
echo "======================================================================="

# ---- C1: feature_retrieval real_init 无效分支 ----
echo "--- C1: real_init 无效 / 退化分支 ---"
# 保留最优 eval 主线: oi2_gi6_fixedmetric_wlsfull_eval (best canonical result)
# 保留: real_init_exp29a_top1 (base reference)
# 删除: hybridrot, coarsew2_wlsfull, flowdirect, centroid, smoke_top5, two_stage
REAL_INIT_DELETE=(
  "real_init_exp29a_top1_coarsefine_v1d_coarsew2_wlsfull"
  "real_init_exp29a_top1_coarsefine_v1e_lowresw2_wls"
  "real_init_exp29a_top1_coarsefine_v1e_lowresw2_wls_learnedinit"
  "real_init_exp29a_top1_coarsefine_v1e_lowresw2_wls_learnedinit_flowsmall"
  "real_init_exp29a_top1_coarsefine_v1e_lowresw2_wls_learnedinit_flowsmall_fixedmetric_eval"
  "real_init_exp29a_top1_coarsefine_v1e_lowresw2_wls_learnedinit_oi2_gi6"
  "real_init_exp29a_top1_coarsefine_v1e_lowresw2_wls_learnedinit_oi2_gi8"
  "real_init_exp29a_top1_coarsefine_v1_full16_learnedinit_flowsmall_hybridrot_e0probe"
  "real_init_exp29a_top1_coarsefine_v1_full16_learnedinit_flowsmall_hybridrot_e2probe"
  "real_init_exp29a_top1_coarsefine_v1_full16_learnedinit_flowsmall_hybridrot_fixedmetric_eval"
  "real_init_exp29a_top1_flowdirect5_eval"
  "real_init_exp29a_top1_flowdirect10_eval"
  "real_init_smoke_top5_centroid_baseline_eval"
  "real_init_smoke_top5_consensus_baseline_eval"
  "real_init_smoke_top5_consensus_centroid_baseline_eval"
  "real_init_smoke_top5_none_baseline_eval"
  "two_stage_smoke_eval"
  "two_stage_smoke_eval_learnedinit_oi2"
  "two_stage_smoke_eval_learnedinit_oi2_matched"
  "two_stage_smoke_eval_learnedinit_wrapper_default"
)
for name in "${REAL_INIT_DELETE[@]}"; do
  keep_summaries_then_rm_dir "$OUTPUT/feature_retrieval/$name" "feature_retrieval/$name"
done

# ---- C2: feature_retrieval real_init probe/smoke 目录 ----
echo "--- C2: real_init probe / smoke / e0probe 目录 ---"
for d in "$OUTPUT/feature_retrieval"/real_init_*; do
  [[ ! -d "$d" ]] && continue
  dname=$(basename "$d")
  case "$dname" in
    *_e0probe|*_e2probe|*_smokeprobe|*smoke*) 
      keep_summaries_then_rm_dir "$d" "feature_retrieval/$dname" ;;
  esac
done

# ---- C3: feature_retrieval 老版本训练/eval 目录 ----
echo "--- C3: radio_loc_oh / full_eval 历史目录 ---"
for name in \
  "radio_loc_oh_fm" "radio_loc_oh_reg" "radio_loc_oh_sanity" \
  "radio_loc_oh_v1" \
  "full_eval_best_ens_consensus" "full_eval_best_ens_none" \
  "full_eval_best_ens_rgbselect" \
  "full_eval_exp29a_direct_both" \
  "experiment_reports" \
; do
  keep_summaries_then_rm_dir "$OUTPUT/feature_retrieval/$name" "feature_retrieval/$name"
done

# ---- C4: pose_regression — 删除 exp29a 之外所有 exp* 目录 ----
echo "--- C4: pose_regression 历史 exp 目录（除 exp29a_direct_both_seed123）---"
for d in "$OUTPUT/feature_retrieval/pose_regression"/exp*; do
  [[ ! -d "$d" ]] && continue
  dname=$(basename "$d")
  if [[ "$dname" == "exp29a_direct_both_seed123" ]]; then
    echo "[KEEP]  $d"
    continue
  fi
  keep_summaries_then_rm_dir "$d" "pose_regression/$dname"
done

# ---- C5: feature_field — 删除旧版本 dcff_radio_oh_* 和 dcff_oldhospital v10-v14 ----
echo "--- C5: feature_field 历史版本 ---"
FIELD_DELETE_PATTERNS=(
  "dcff_radio_oh_v1" "dcff_radio_oh_v2" "dcff_radio_oh_v3" "dcff_radio_oh_v4"
  "dcff_radio_oh_v5" "dcff_radio_oh_v6" "dcff_radio_oh_v7" "dcff_radio_oh_v8"
  "dcff_radio_oh_v9a" "dcff_radio_oh_v9b" "dcff_radio_oh_v10" "dcff_radio_oh_v11"
  "dcff_radio_oh_v12"
  "dcff_oldhospital_radio_dual_pilot"
  "dcff_oldhospital_v10b_coarse_fix"
  "dcff_oldhospital_v10c_carrier_residual"
  "dcff_oldhospital_v10_retrain"
  "dcff_oldhospital_v11a_joint_geo"
  "dcff_oldhospital_v11b_joint_geo"
  "dcff_oldhospital_v11c_joint_geo"
  "dcff_oldhospital_v11d_joint_geo"
  "dcff_oldhospital_v11_joint_geo"
  "dcff_oldhospital_v12_fsm"
  "dcff_oldhospital_v13a_no_fsm"
  "dcff_oldhospital_v13b_fsm_b16"
  "dcff_oldhospital_v13b_fsm_b16_test"
  "dcff_oldhospital_v14b_spatial_only_frozen"
  "dcff_oldhospital_v14c_no_cross_attn_frozen"
  "dcff_oldhospital_v14d_binary_no_cross_attn_frozen"
  "dcff_oldhospital_v14_fsm_frozen"
)
for name in "${FIELD_DELETE_PATTERNS[@]}"; do
  keep_summaries_then_rm_dir "$OUTPUT/feature_field/$name" "feature_field/$name"
done
# 删除 v15b/v15c（几何正则实验，已被 v15d 取代）
keep_summaries_then_rm_dir "$OUTPUT/feature_field/dcff_oldhospital_v15b_geometry_reg_joint" "feature_field/dcff_oldhospital_v15b_geometry_reg_joint"
keep_summaries_then_rm_dir "$OUTPUT/feature_field/dcff_oldhospital_v15c_geometry_reg_coarse_smooth_joint" "feature_field/dcff_oldhospital_v15c_geometry_reg_coarse_smooth_joint"
# 删除 v16a（teachergeom 实验，已被 v15d 取代）
keep_summaries_then_rm_dir "$OUTPUT/feature_field/dcff_oldhospital_v16a_v15d_teachergeom_frozen" "feature_field/dcff_oldhospital_v16a_v15d_teachergeom_frozen"
# 删除 smoke eval
safe_rm "$OUTPUT/feature_field/_smoke_dcff_eval.json"

# ---- C6: feature_extract — 删除历史版本 ----
echo "--- C6: feature_extract 历史版本（保留 v5d + v5l_b4）---"
FE_DELETE=(
  "joint_radio_dcff_oh_v1"
  "joint_radio_dcff_oh_v2_retrieval"
  "joint_radio_dcff_oh_v3_retrieval_sim"
  "joint_radio_dcff_oh_v4b_map_fine_only"
  "joint_radio_dcff_oh_v4_map_consistency"
  "joint_radio_dcff_oh_v5a_joint_fine_pilot"
  "joint_radio_dcff_oh_v5b_joint_fine_anchor"
  "joint_radio_dcff_oh_v5c_joint_fine_balanced"
  "joint_radio_dcff_oh_v5e_no_l2_warmstart"
  "joint_radio_dcff_oh_v5f_maghead_warmstart"
  "joint_radio_dcff_oh_v5g_maghead_normloss_warmstart"
  "joint_radio_dcff_oh_v5l_pointwise_featsharp_full"
  "joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1"
  "joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_fineonly"
  "joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_lightcoarse"
  "joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4_strongcoarse"
  "joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_geomcorr2e_smoke"
  "joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_geomcorr_smoke"
  "joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_smoke"
  "joint_radio_dcff_oh_v5l_stablemap_geomcorr2e_smoke"
  "joint_radio_dcff_oh_v5m_pointwise_teacher_anchor_pilot"
)
for name in "${FE_DELETE[@]}"; do
  keep_summaries_then_rm_dir "$OUTPUT/feature_extract/$name" "feature_extract/$name"
done

# ---- C7: pose_refine — 删除历史版本（v1-v19, v20a-j, v20k 非主线变体, v21-v22）----
echo "--- C7: pose_refine 历史版本 ---"
# 主线保留: v20k_v5l_..._flowsmall (和可选回退: v20k_v5l_..._b4_bs4)
# 注: v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4 作为非coarsefine回退保留
POSE_KEEP_EXACT=(
  "concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall"
  "concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4"
)

for d in "$OUTPUT/pose_refine"/concat_loc_*; do
  [[ ! -d "$d" ]] && continue
  dname=$(basename "$d")
  # 跳过保留列表
  skip=false
  for keep in "${POSE_KEEP_EXACT[@]}"; do
    if [[ "$dname" == "$keep" ]]; then
      echo "[KEEP]  $d"
      skip=true
      break
    fi
  done
  $skip && continue
  # 删除所有其他 concat_loc 目录
  keep_summaries_then_rm_dir "$d" "pose_refine/$dname"
done

echo ""
echo "======================================================================="
echo "Phase D: 非主线 checkpoint 专项瘦身"
echo "======================================================================="

# ---- D1: feature_field 保留目录 — 只保留 best.pth ----
echo "--- D1: dcff_oldhospital_v15a checkpoint 瘦身 ---"
FIELD_KEEP_DIRS=(
  "$OUTPUT/feature_field/dcff_oldhospital_v15a_coarse_smooth_frozen"
  "$OUTPUT/feature_field/dcff_oldhospital_v15d_v14b_coarse_smooth_frozen"
)
for d in "${FIELD_KEEP_DIRS[@]}"; do
  if [[ -d "$d/checkpoints" ]]; then
    for f in "$d/checkpoints"/*.pth; do
      fname=$(basename "$f")
      [[ "$fname" != "best.pth" ]] && safe_rm "$f"
    done
  fi
done

# ---- D2: feature_extract 保留目录 — 只保留 best.pth ----
echo "--- D2: feature_extract checkpoint 瘦身 ---"
FE_KEEP_DIRS=(
  "$OUTPUT/feature_extract/joint_radio_dcff_oh_v5d_joint_fine_schedule"
  "$OUTPUT/feature_extract/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4"
)
for d in "${FE_KEEP_DIRS[@]}"; do
  if [[ -d "$d/checkpoints" ]]; then
    for f in "$d/checkpoints"/*.pth; do
      fname=$(basename "$f")
      [[ "$fname" != "best.pth" ]] && safe_rm "$f"
    done
  fi
done

# ---- D3: pose_refine 保留的回退模型 — 只保留 best.pth ----
echo "--- D3: pose_refine 回退模型 checkpoint 瘦身 ---"
POSE_FALLBACK="$OUTPUT/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4"
if [[ -d "$POSE_FALLBACK/checkpoints" ]]; then
  for f in "$POSE_FALLBACK/checkpoints"/*.pth; do
    fname=$(basename "$f")
    [[ "$fname" != "best.pth" ]] && safe_rm "$f"
  done
fi
# 主线 flowsmall checkpoint 也只保留 best.pth
POSE_MAIN="$OUTPUT/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall"
if [[ -d "$POSE_MAIN/checkpoints" ]]; then
  for f in "$POSE_MAIN/checkpoints"/*.pth; do
    fname=$(basename "$f")
    [[ "$fname" != "best.pth" ]] && safe_rm "$f"
  done
fi

# ---- D4: feature_retrieval radio_loc — 只保留 best.pth ----
echo "--- D4: radio_loc_oh 检索模型 checkpoint 瘦身 ---"
for d in \
  "$OUTPUT/feature_retrieval/radio_loc_oh_v1" \
  "$OUTPUT/feature_retrieval/radio_loc_oh_v3" \
; do
  if [[ -d "$d/checkpoints" ]]; then
    for f in "$d/checkpoints"/*.pth; do
      fname=$(basename "$f")
      [[ "$fname" != "best.pth" ]] && safe_rm "$f"
    done
  fi
done

echo ""
echo "======================================================================="
echo "Phase E: 生成迁移清单"
echo "======================================================================="

if ! $DRY_RUN; then
  MANIFEST="$ARCHIVE/KEEP_MANIFEST.txt"
  mkdir -p "$ARCHIVE"
  {
    echo "# KEEP_MANIFEST.txt — 主线必保资产"
    echo "# 生成时间: $(date)"
    echo ""
    echo "## 核心 Checkpoints"
    for asset in "${KEEP_ASSETS[@]}"; do
      if [[ -f "$asset" ]]; then
        echo "  FOUND  $asset"
      else
        echo "  MISS   $asset"
      fi
    done
    echo ""
    echo "## 主线实验目录"
    echo "  $OUTPUT/feature_field/dcff_oldhospital_v15d_v14b_coarse_smooth_frozen/"
    echo "  $OUTPUT/feature_extract/joint_radio_dcff_oh_v5l_pointwise_featsharp_full_fsm_v1_b4/"
    echo "  $OUTPUT/feature_extract/joint_radio_dcff_oh_v5d_joint_fine_schedule/"
    echo "  $OUTPUT/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4_coarsefine_v1_full16_learnedinit_flowsmall/"
    echo "  $OUTPUT/feature_retrieval/pose_regression/exp29a_direct_both_seed123/"
    echo ""
    echo "## 回退资产"
    echo "  $OUTPUT/pose_refine/concat_loc_oh_v20k_v5l_pointwise_featsharp_full_fsm_v1_b4_bs4/"
    echo "  $OUTPUT/feature_field/dcff_oldhospital_v15a_coarse_smooth_frozen/"
    echo "  $OUTPUT/feature_retrieval/radio_loc_oh_v3/"
  } > "$MANIFEST"
  echo "[CREATE] $MANIFEST"

  echo ""
  echo "最终磁盘用量（清理后）:"
  du -sh "$OUTPUT"/*/  2>/dev/null | sort -rh
fi

echo ""
echo "======================================================================="
echo "清理完成"
if $DRY_RUN; then
  echo "（DRY-RUN: 未删除任何文件）"
fi
echo "======================================================================="
