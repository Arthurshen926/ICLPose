#!/bin/bash
# Move large unused feature directories from local SSD to mechanical disk
# Creates symlinks back so configs still work
# Run with: nohup bash scripts/move_to_archive.sh > /root/ICLPose/output/move_archive.log 2>&1 &

# set -e removed: tar returns non-zero on NFS chown warnings (harmless)

SRC=/root/ICLPose/output
ARCHIVE=/mnt/pool/sqy/ICLPose_archive/output
mkdir -p "$ARCHIVE"

move_dir() {
    local name=$1
    local src_path="$SRC/$name"
    
    # Skip if already a symlink
    if [ -L "$src_path" ]; then
        echo "[SKIP] $name (already symlink)"
        return
    fi
    
    # Skip if doesn't exist
    if [ ! -d "$src_path" ]; then
        echo "[SKIP] $name (not found)"
        return
    fi
    
    local size=$(du -sh "$src_path" 2>/dev/null | cut -f1)
    echo "[MOVE] $name ($size) ..."
    
    # Use tar pipe for efficient cross-filesystem copy
    mkdir -p "$ARCHIVE/$name"
    (cd "$src_path" && tar cf - .) | (cd "$ARCHIVE/$name" && tar xf - --no-same-owner)
    
    # Verify at least some data arrived
    if [ "$(ls -A "$ARCHIVE/$name" 2>/dev/null)" ]; then
        rm -rf "$src_path"
    else
        echo "[ERROR] Archive appears empty, keeping local copy of $name"
        return
    fi
    
    # Create symlink
    ln -sfn "$ARCHIVE/$name" "$src_path"
    
    echo "[DONE] $name -> symlink"
    df -h / | tail -1
}

echo "=== Starting archive migration $(date) ==="
echo "Source: $SRC"
echo "Target: $ARCHIVE"
echo ""

# --- Feature extraction outputs (not used by active experiments) ---
# Active experiments (exp170-175) only use features_multiscale_stride7_compressed

# Old pre-stride7 features
move_dir "features_multiscale_compressed"
move_dir "features_multiscale_pca"
move_dir "features_multiscale_pca16"
move_dir "features_multiscale"

# PCA/rendered variants (used by old OldHospital experiments only)
move_dir "features_pca_nomean"
move_dir "features_pca_nomean_rendered"
move_dir "features_render_extracted"
move_dir "features_selected_pca"
move_dir "features_selected_pca_rendered"

# FlowFeat features (experimental, may be re-needed but archivable)
move_dir "features_flowfeat"
move_dir "features_flowfeat_v2"
move_dir "features_flowfeat_v2_pca"
move_dir "features_flowfeat_v2_pca_lr"

# --- feature_3dgs models (only room_0_v4_stride7_decoupled is active) ---
# Move old/unused 3DGS feature models
for d in $(ls -d "$SRC/feature_3dgs/"*/ 2>/dev/null); do
    name=$(basename "$d")
    # Keep the active one
    if [ "$name" = "room_0_v4_stride7_decoupled" ]; then
        echo "[KEEP] feature_3dgs/$name (active)"
        continue
    fi
    # Move the rest
    mkdir -p "$ARCHIVE/feature_3dgs"
    if [ -L "$d" ]; then
        echo "[SKIP] feature_3dgs/$name (already symlink)"
        continue
    fi
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "[MOVE] feature_3dgs/$name ($size) ..."
    mkdir -p "$ARCHIVE/feature_3dgs/$name"
    (cd "$d" && tar cf - .) | (cd "$ARCHIVE/feature_3dgs/$name" && tar xf - --no-same-owner)
    if [ "$(ls -A "$ARCHIVE/feature_3dgs/$name" 2>/dev/null)" ]; then
        rm -rf "$d"
        ln -sfn "$ARCHIVE/feature_3dgs/$name" "$d"
    else
        echo "[ERROR] feature_3dgs/$name archive empty, keeping local"
        continue
    fi
    echo "[DONE] feature_3dgs/$name -> symlink"
done

# --- Old experiment outputs (not exp170-175, exp143) ---
# Move experiments that are clearly old and not needed for active work
for d in $(ls -d "$SRC/exp0"*/ "$SRC/exp1"[0-3]*/ 2>/dev/null); do
    name=$(basename "$d")
    # Keep exp143 (warmstart source)
    if [ "$name" = "exp143_room0_lownoise_ft" ]; then
        echo "[KEEP] $name (warmstart source)"
        continue
    fi
    if [ -L "$d" ]; then
        echo "[SKIP] $name (already symlink)"
        continue
    fi
    mkdir -p "$ARCHIVE/experiments"
    size=$(du -sh "$d" 2>/dev/null | cut -f1)
    echo "[MOVE] $name ($size) ..."
    mkdir -p "$ARCHIVE/experiments/$name"
    (cd "$d" && tar cf - .) | (cd "$ARCHIVE/experiments/$name" && tar xf - --no-same-owner)
    if [ "$(ls -A "$ARCHIVE/experiments/$name" 2>/dev/null)" ]; then
        rm -rf "$d"
        ln -sfn "$ARCHIVE/experiments/$name" "$d"
    else
        echo "[ERROR] $name archive empty, keeping local"
        continue
    fi
    echo "[DONE] $name -> symlink"
done

# --- Dataset: move large datasets that aren't actively needed ---
DSRC=/root/ICLPose/dataset
DARCHIVE=/mnt/pool/sqy/ICLPose_archive/dataset

move_dataset() {
    local name=$1
    local src_path="$DSRC/$name"
    
    if [ -L "$src_path" ]; then
        echo "[SKIP] dataset/$name (already symlink)"
        return
    fi
    if [ ! -e "$src_path" ]; then
        echo "[SKIP] dataset/$name (not found)"
        return
    fi
    
    local size=$(du -sh "$src_path" 2>/dev/null | cut -f1)
    echo "[MOVE] dataset/$name ($size) ..."
    mkdir -p "$DARCHIVE"
    mkdir -p "$DARCHIVE/$name"
    (cd "$src_path" && tar cf - .) | (cd "$DARCHIVE/$name" && tar xf - --no-same-owner)
    if [ ! "$(ls -A "$DARCHIVE/$name" 2>/dev/null)" ]; then
        echo "[ERROR] dataset/$name archive empty, keeping local"
        return
    fi
    rm -rf "$src_path"
    ln -sfn "$DARCHIVE/$name" "$src_path"
    echo "[DONE] dataset/$name -> symlink"
    df -h / | tail -1
}

# OldHospital is 32G, splatloc-generated is 13G, Synthetic_Multi is 6.7G
# Active experiments only use room_0
move_dataset "OldHospital"
move_dataset "splatloc-generated-files"
move_dataset "Synthetic_Multi"
move_dataset "crescentbay"
move_dataset "train"
move_dataset "map_results"

echo ""
echo "=== Archive migration complete $(date) ==="
df -h / | tail -1
