#!/usr/bin/env bash
set -euo pipefail

repo=/root/ICLPose

gpu0() {
  "$repo/scripts/run_goal_maplet_g23_strict_map_fold.sh" fold0 0
  "$repo/scripts/run_goal_maplet_g23_strict_map_fold.sh" fold2 0
  "$repo/scripts/run_goal_maplet_g23_strict_map_fold.sh" fold4 0
}

gpu1() {
  "$repo/scripts/run_goal_maplet_g23_strict_map_fold.sh" fold1 1
  "$repo/scripts/run_goal_maplet_g23_strict_map_fold.sh" fold3 1
  "$repo/scripts/run_goal_maplet_g23_strict_map_fold.sh" final_alltrain 1
}

gpu0 &
pid0=$!
gpu1 &
pid1=$!
wait "$pid0"
wait "$pid1"
