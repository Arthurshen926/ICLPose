# Correctness-repaired frozen-input baseline (2026-09-09)

## Decision

Implements the first priority of external review attachment
`376dc43b-a41b-4678-9e99-fb37043caabc/pasted-text.txt`: establish what the corrected
software actually achieves before another optimization campaign. The full 438
query result is **96/330/398/409/412**, not the historical
**104/336/395/409/413**. This is a corrected research baseline, not a performance
promotion. Historical files are preserved. Probability-backend experiments remain
opt-in and are not a prerequisite for the next mainline improvement.

## Frozen scope and results

Replayed both historical coordinate branches for all queries: grouped PnP,
anonymous view/range geometry, MoGe3 plane/scale, uncertainty refinement, sparse
plane comparison, sparse-first full-map dense normal rendering, and the existing
AND consensus. Retrieval, trained weights, original correspondence populations,
thresholds and historical per-branch final association policy are unchanged.
Mandatory fixes are independent-token LM, positive depth, independent-token
group budgeting and valid zero-residual sorting. This is a bundle comparison;
it does not isolate the causal contribution of each fix.

| Split | Historical .1/.25/.5/1/2m joint hits | Corrected joint hits |
| --- | --- | --- |
| seq10 (88) | 21/71/85/87/88 | 24/72/85/87/88 |
| seq13 shard0 (88) | 21/65/79/82/82 | 19/63/79/82/82 |
| seq13 shard1 (88) | 19/65/75/79/80 | 15/64/77/79/80 |
| seq13 shard2 (87) | 22/68/79/81/82 | 20/66/79/81/82 |
| seq13 shard3 (87) | 21/67/77/80/81 | 18/65/78/80/80 |
| Total (438) | 104/336/395/409/413 | 96/330/398/409/412 |

Rotation thresholds remain 1/2/5/10/45 degrees. Median translation changes
.162267 -> .162160 m, which must not hide the strict-threshold losses. Strict
paired transitions are 8 gains and 16 losses (exact McNemar p=.1516); .25m gives
4 gains/10 losses; .5m gives 4 gains/1 loss; coarse gives 1 gain/2 losses.
There are now 26 coarse failures. Finite-pair mean translation delta is -1.999mm;
route-stratified 10-frame block interval is [-53.945,+43.984]mm. No convincing
overall precision improvement follows. Five splits represent TWO routes, not
five independent generalization tests. Both routes are reused development data.

The original 25-frame attribution is historical and must not be relabeled as
the new 26-frame attribution without rerunning that diagnostic.

## Reproduction and audit contract

Added branch replay, consensus finalization and aggregate audit commands.
`finalize_goal_maplet_corrected_baseline --replay_branches` is a single invocation
from frozen coordinate inventories to one final pose per query. It is NOT an
RGB-to-pose/map-building entrypoint. Contributor files are still supplied for
post-solve evaluation; constituent solvers freeze outputs before their evaluation.
The orchestration consumes only lineage fields of historical post-label audits
to identify the fixed baseline, not their errors to choose new parameters.

Historical file AND content hashes resolve the original artifacts. Map and atlas
hashes are checked. Protocol files record actual subprocess commands, Python,
NumPy, git HEAD and input hashes; stage logs and timing are retained. This is not
yet a containerized dependency lock or a full clean-checkout reproduction of map
construction and feature extraction.

Two near-identical filename families initially resolved to later ablation inputs
on seq10/shard3. Those V182 runs are retained as non-authoritative trials and
excluded from the aggregate. The correct historical identities were resolved
from the V114 hash ledger; the finalizer rejects a different historical branch.
Authoritative branches use V183 on seq10/shard3 and V182 on shard0/1/2; ALL final
consensus directories use V183. This was input-lineage correction, not selection
of the better result.

Historical MoGe settings are explicitly replayed, rather than silently relying
on CLI defaults: `sqrt_visible_fraction` and
`many_query_fragments_per_map_plane`. Final coordinate selection is read from
the historical branch; old missing fields mean `nearest_reprojection`.
Several recent experimental controls used other defaults and cannot be treated
as a replay of the historical full mainline, even when internally matched.

V184 independently repeats seq10 through the single-command path. Names, poses,
usability and selected branches are exactly equal to the staged V183 run.

Example, with NEW output directories (existing outputs are rejected):

```bash
B=output/g25_pose_transport/planar_map_rendered_ransac_v1/surface_coordinate_upgrade_v1
P=output/g25_pose_transport/planar_map_rendered_ransac_v1
python -m feature_extract.tools.vfm.finalize_goal_maplet_corrected_baseline \
  --replay_branches \
  --primary "$B/reproduce_seq10_primary" --alternate "$B/reproduce_seq10_alternate" \
  --output_dir "$B/reproduce_seq10_consensus" \
  --historical_audit "$B/pose_failure_attribution_seq10_v114.json" \
  --plane_uv_atlas "$B/stmarys_metric_plane_uv_radio_atlas_cell050_p4_learned64d_strict_v9.npz" \
  --planar_map "$P/stmarys_rendered_ransac_fused_planes_v1.npz" \
  --physical_map output/vfm/2dgs_surface/StMarysChurch/full_train/goal_maplet/physical_map_v4.npz \
  --query_plane_dir output/g25_pose_transport/planar_query_geometry/moge3_vitl_seq10_plane_regions_v1 \
  --query_contributors output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v6/contributors_alltrain_clean \
  --query_camera_inventory "$P/query_camera_only_seq10_v1.npz" \
  --moge3_query output/g25_pose_transport/planar_query_geometry/moge3_vitl_seq10_level9_fp32_v2
```

## Runtime/resource boundary

Across the authoritative branch jobs, recorded subprocess durations sum to
355.90s PnP, 41.14s view geometry, 38.74s MoGe and 32.96s uncertainty. Consensus
adds 13.28s sparse geometry, 6.35s scheduling, 288.48s two-arm dense rendering
and 6.31s selection. These include process startup, I/O and evaluation, and
some jobs ran concurrently. They are NOT end-to-end online latency, nor a speed
comparison with the prior run. RADIO/MoGe inference and correspondence generation
were frozen inputs, and GPU peak memory was not measured.

Atlas file is 21,743,646 bytes; the separately required dense physical map is
41,805,215 bytes. Neither is the entire deployment footprint: include model
weights, planar map, transient buffers and caches in a future full measurement.

## V185 shared coordinate bias diagnostic

Compared old local V162 and aligned local V157 on identical held mapping rows.
Added per-source-image SIGNED mean joint residual and exact squared-error
decomposition into mean-bias energy and centered residual energy. This avoids
letting opposite image biases cancel in a global average.

On 49 held source images / 25,147 candidate rows, mean per-image bias norm is
.146836 -> .141242 px; 27 images improve and 22 worsen. Thus an average coordinate
gain does not imply every image's systematic residual improves. This is a
descriptive mapping result, not an explanation of query-pose regression.
Banks currently lack plane IDs and camera Jacobians, and multiple anonymous
modes share tokens. Therefore this is NOT independent-token, per-plane or
weak-pose-direction analysis, and not the requested complete shared-pose test.

## Validation and next experiment boundary

1,091 goal-maplet tests pass with the existing scatter_reduce warning; relevant
new tests verify historical flag replay and signed-bias decomposition. Full-repo
pytorch3d collection limitation remains; no new all-repository pass claim.
Full data replay and single-command equivalence supply integration evidence.

Next priorities remain (1) geometry-complementary token sampling with exploratory
support and equal effective-hypothesis/time controls; (2) training candidates
generated by actual cross-plane retrieval and signed per-plane/shared-pose
coordinate diagnostics. Neither is claimed completed by this baseline work.
No new sampler thresholds, network weights or query-based selector were tuned.

Aggregate: `surface_coordinate_upgrade_v1/corrected_baseline_all438_v183.json`.
Shared bias: `surface_coordinate_upgrade_v1/mapping_shared_bias_v185.json`.
