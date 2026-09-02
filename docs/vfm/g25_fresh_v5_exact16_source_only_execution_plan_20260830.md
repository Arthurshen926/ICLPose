# G25 fresh-v5 exact-16 source-only execution plan (2026-08-30)

## Purpose and hard information barrier

This is the resumable command sheet for the fresh-v5 exact-16 comparison.  It
starts only after the new physical-isolation run has been sealed as a strict
`goal_maplet_disjoint_chart_upstream_authority_v2`, and after the held-free
model-neutral selector has sealed a new exact-16 plan against that authority.

The commands below consume only the fresh **source** MASt3R run, the isolated
source posed-COLMAP input, mapping cameras/images and source-only initializer
artifacts.  They must not inspect the fresh held images, cameras or pointmaps.
Held geometry is admitted only later by the full-gate input builder, after the
v3 topology and source-only bounded-submap authority have been frozen.

Old dense-local-v4 plans, initializers, v1/v2/v3 domains, alignments, atlases,
bounds and held rays are diagnostic-only.  No old-v4 artifact may be paired
with the fresh-v5 source or held run.

## Frozen paths and hash pins

Run from `/root/ICLPose`.  The namespace is deliberately new and must not be
reused if a command has partially written it.

```bash
G25_SOURCE_ROOT=/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_isolated/source_seq4_frames271_294_mast3r
G25_SOURCE_POSED=/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_isolated/inputs/source_posed_colmap
G25_FRESH_ROOT=/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_exact16_fresh_singlewriter_v2
G25_AUTHORITY=/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_isolated/disjoint_upstream_authority_v2.json
G25_PLAN=/root/ICLPose/output/g25_pose_transport/explicit_chart_atlas/dense_local_v5_isolated/source_seq4_fresh_exact16_overlap_aware_submap_plan_v3.npz
G25_MATCHA=/tmp/matcha-gaussians-official
G25_DAV2_INIT="$G25_FRESH_ROOT/initializers/dav2_source_seq4_24"
G25_MOGE3_INIT="$G25_FRESH_ROOT/initializers/moge3_source_seq4_24"
G25_DOMAIN_V1="$G25_FRESH_ROOT/authority/comparison_domain_exact16_optimizer_v1.npz"
G25_DOMAIN_V2="$G25_FRESH_ROOT/authority/comparison_domain_exact16_exact_topology_v2.npz"
G25_DOMAIN_V3="$G25_FRESH_ROOT/authority/comparison_domain_exact16_reference_safe_exact_topology_v3.npz"
G25_M1_ALIGN="$G25_FRESH_ROOT/alignment/m1_dav2_exact16_v1"
G25_M2_ALIGN="$G25_FRESH_ROOT/alignment/m2_moge3_exact16_v1"

# Fill only from independently validated, sealed manifests.  Never infer a
# content hash from a filename and never reuse any dense-local-v4 value.
G25_AUTHORITY_CONTENT=34b9acecf4fa7472bf39697827783ba900d22216d58901107750a1bdb7792a82
G25_PLAN_CONTENT=cd639963a640de6f4861fff56b74143497b954975276b72f73c4f17f787f7e99
G25_SOURCE_TREE=dc8f1c8743e622a3065be1e50e83c5cb51bb1813396ac550717d10dd1f07dd2f
G25_ALIGNMENT_CODE=<CURRENT_ALIGNMENT_CODE_INVENTORY_SHA256>
```

`G25_FRESH_ROOT` must have exactly one process owner.  Before launching,
confirm no initializer/alignment process targets it and that every leaf output
directory is absent.  A second writer, even if interrupted early, is a P0:
stop downstream work and switch to another never-used root rather than trying
to prove a partially shared directory clean after the fact.

Before any GPU command, validate all of the following from the new authority
and plan:

- authority artifact type is `goal_maplet_disjoint_chart_upstream_authority_v2`;
- source root, cameras bytes and tree hash equal `G25_SOURCE_ROOT` exactly;
- isolated source input root equals `G25_SOURCE_POSED` exactly;
- physical source/held input roots are disjoint;
- plan artifact type is `goal_maplet_overlap_aware_chart_submap_plan_v3`;
- plan authority/content/source-tree lineage equals the values above;
- official `selection_rank` inventory has exactly 16 unique source names;
- `selection_cardinality_frozen_before_held_geometry=true` and
  `held_geometry_used_for_selection=false`, with selector min=max=16.

The alignment code pin is computed from the same implementation that the
runner replays:

```bash
PYTHONPATH=. /usr/local/miniconda3/envs/matcha/bin/python -c "from pathlib import Path; from feature_extract.tools.vfm.run_goal_maplet_masked_chart_alignment_gate import _alignment_code_inventory; print(_alignment_code_inventory(Path('/tmp/matcha-gaussians-official'))[1])"
```

Record that output as `G25_ALIGNMENT_CODE` before either alignment starts.

## Stage 1: two source-only initializer arms

These jobs are independent and should run concurrently on separate GPUs.
Each builder refuses to reuse an existing output directory.

GPU 0, DAV2 metric-fit initialization:

```bash
PYTHONPATH=. /usr/local/miniconda3/envs/matcha/bin/python feature_extract/tools/vfm/build_goal_maplet_dav2_chart_initializers.py \
  --posed_colmap "$G25_SOURCE_POSED" \
  --mast3r_source_run "$G25_SOURCE_ROOT" \
  --disjoint_upstream_authority "$G25_AUTHORITY" \
  --matcha_repo "$G25_MATCHA" \
  --output_dir "$G25_DAV2_INIT" \
  --device cuda:0 \
  --encoder vitl
```

GPU 1, pose-free MoGe-3 initialization:

```bash
PYTHONPATH=. /tmp/moge3-env310/bin/python feature_extract/tools/vfm/build_goal_maplet_moge3_chart_initializers.py \
  --cameras "$G25_SOURCE_ROOT/cameras.json" \
  --output_dir "$G25_MOGE3_INIT" \
  --device cuda:1 \
  --model_id Ruicheng/moge-3-vitl \
  --resolution_level 9 \
  --refine_steps 3 \
  --camera_focal_canvas_width 512 \
  --routes seq4
```

Do not proceed unless both manifests contain 24 unique source rows, no query
or GT use, and every initializer file/content hash can be replayed.  DAV2 must
also bind the new authority and source tree; MoGe-3 must bind the exact new
source `cameras.json` bytes and every source image byte.

## Stage 2: one common optimizer domain and physical-safe topology

The v1 optimizer domain is built once from both completed initializer
inventories and the same exact-16 plan.  Both alignment arms must consume this
same v1 file/content.

```bash
PYTHONPATH=. python feature_extract/tools/vfm/build_goal_maplet_chart_comparison_domain.py \
  --cameras "$G25_SOURCE_ROOT/cameras.json" \
  --pointmaps_dir "$G25_SOURCE_ROOT/pointmaps" \
  --dav2_initializers "$G25_DAV2_INIT" \
  --moge3_initializers "$G25_MOGE3_INIT" \
  --disjoint_upstream_authority "$G25_AUTHORITY" \
  --frozen_submap_plan "$G25_PLAN" \
  --expected_plan_content_sha256 "$G25_PLAN_CONTENT" \
  --route seq4 \
  --output "$G25_DOMAIN_V1"
```

After validating the emitted v1 content hash, pin it explicitly:

```bash
G25_DOMAIN_V1_CONTENT=<FRESH_V5_OPTIMIZER_V1_CONTENT_SHA256>

PYTHONPATH=. python feature_extract/tools/vfm/seal_goal_maplet_chart_comparison_exact_topology.py \
  --comparison_domain "$G25_DOMAIN_V1" \
  --expected_comparison_domain_content_sha256 "$G25_DOMAIN_V1_CONTENT" \
  --frozen_submap_plan "$G25_PLAN" \
  --expected_plan_content_sha256 "$G25_PLAN_CONTENT" \
  --disjoint_upstream_authority "$G25_AUTHORITY" \
  --expected_disjoint_authority_content_sha256 "$G25_AUTHORITY_CONTENT" \
  --source_root "$G25_SOURCE_ROOT" \
  --expected_source_tree_sha256 "$G25_SOURCE_TREE" \
  --output "$G25_DOMAIN_V2"
```

After validating v2, pin its content hash and seal the source-reference-safe
v3 subset:

```bash
G25_DOMAIN_V2_CONTENT=<FRESH_V5_EXACT_TOPOLOGY_V2_CONTENT_SHA256>

PYTHONPATH=. python feature_extract/tools/vfm/seal_goal_maplet_chart_comparison_reference_safe_v3.py \
  --upstream_exact_topology_v2 "$G25_DOMAIN_V2" \
  --expected_upstream_v2_content_sha256 "$G25_DOMAIN_V2_CONTENT" \
  --optimizer_comparison_domain_v1 "$G25_DOMAIN_V1" \
  --expected_optimizer_v1_content_sha256 "$G25_DOMAIN_V1_CONTENT" \
  --disjoint_upstream_authority "$G25_AUTHORITY" \
  --expected_disjoint_authority_content_sha256 "$G25_AUTHORITY_CONTENT" \
  --source_root "$G25_SOURCE_ROOT" \
  --expected_source_tree_sha256 "$G25_SOURCE_TREE" \
  --output "$G25_DOMAIN_V3"
```

Formal full-submap evaluation uses only stride 4.  Require all 16 charts to
have nonempty stride-4 face inventories.  Stride-8 empties may be recorded for
diagnostics but must neither block stride 4 nor silently change the selected
chart inventory.  Record and externally pin the v3 file/content/arrays/exact-
topology hashes before atlas export.

## Stage 3: paired 16-chart alignment

Only v1 is an optimizer input.  v2/v3 are sealed evaluation topology and must
not alter either arm's optimizer pixels.  Run the two arms concurrently on
separate GPUs after freezing `G25_ALIGNMENT_CODE`.

GPU 0, M1/DAV2:

```bash
PYTHONPATH=. /usr/local/miniconda3/envs/matcha/bin/python feature_extract/tools/vfm/run_goal_maplet_masked_chart_alignment_gate.py \
  --initializer dav2 \
  --cameras "$G25_SOURCE_ROOT/cameras.json" \
  --pointmaps_dir "$G25_SOURCE_ROOT/pointmaps" \
  --dav2_initializers "$G25_DAV2_INIT" \
  --output_dir "$G25_M1_ALIGN" \
  --matcha_repo "$G25_MATCHA" \
  --disjoint_upstream_authority "$G25_AUTHORITY" \
  --comparison_domain "$G25_DOMAIN_V1" \
  --expected_comparison_domain_content_sha256 "$G25_DOMAIN_V1_CONTENT" \
  --frozen_submap_plan "$G25_PLAN" \
  --expected_plan_content_sha256 "$G25_PLAN_CONTENT" \
  --expected_alignment_code_inventory_sha256 "$G25_ALIGNMENT_CODE" \
  --route seq4 \
  --chart_count 16 \
  --iterations 1000 \
  --device cuda:0
```

GPU 1, M2/MoGe-3:

```bash
PYTHONPATH=. /usr/local/miniconda3/envs/matcha/bin/python feature_extract/tools/vfm/run_goal_maplet_masked_chart_alignment_gate.py \
  --initializer moge3 \
  --cameras "$G25_SOURCE_ROOT/cameras.json" \
  --pointmaps_dir "$G25_SOURCE_ROOT/pointmaps" \
  --moge3_initializers "$G25_MOGE3_INIT" \
  --output_dir "$G25_M2_ALIGN" \
  --matcha_repo "$G25_MATCHA" \
  --disjoint_upstream_authority "$G25_AUTHORITY" \
  --comparison_domain "$G25_DOMAIN_V1" \
  --expected_comparison_domain_content_sha256 "$G25_DOMAIN_V1_CONTENT" \
  --frozen_submap_plan "$G25_PLAN" \
  --expected_plan_content_sha256 "$G25_PLAN_CONTENT" \
  --expected_alignment_code_inventory_sha256 "$G25_ALIGNMENT_CODE" \
  --route seq4 \
  --chart_count 16 \
  --iterations 1000 \
  --device cuda:1
```

Both manifests must replay the same authority, official plan order, v1 domain
content and code inventory.  Their `initializer_file_sha256` lists must be in
official plan order.  `charts_data.npz` and output `cameras.json` must remain
bound by the manifest; a lexical internal permutation is permitted only when
all numeric inputs are permuted together and outputs are inverted back to the
official order.

## Stage 4: intentionally deferred atlas and held gate

Do not export an atlas against an old bound.  First build a new source-only
bounded-submap authority from this exact v3 topology and pin its canonical
AABB content.  Then export four stride-4 atlases with the strict v3 exporter:
M1 initial/aligned and M2 initial/aligned.  Each export must replay the arm's
alignment manifest, initializer manifest and rows, v3->v2->v1 lineage,
authority, plan and new bounds.  All four must have byte-identical chart
names/order, vertex offsets, pixel indices, UVs, face offsets and faces.

Only after those source artifacts and bounds are frozen may the fresh held
authority be consumed by the full-gate input builder.  This sequencing makes
it impossible for held geometry to influence selection, initialization,
alignment, topology or map bounds.

## Expected resources and wall time

Measured on the prior 24-source run:

| Stage | Prior observation | Conservative fresh-v5 budget |
|---|---:|---:|
| DAV2 initializer | 115.8 s, 2.85 GB peak CUDA, 11 MB output | 2--3 min, 3.5 GB GPU |
| MoGe-3 initializer | 17.3 s row-inference sum, 2.71 GB peak CUDA, 11 MB output | 1--2 min incl. load, 3.5 GB GPU |
| 3-chart alignment | 46 s/arm, 0.29 GB peak | 2.5--3.5 min/16-chart arm, 2 GB GPU |
| Domain seals | CPU, small NPZs | under 1 min, under 2 GB RAM |
| Two 16-chart alignments | projected from 3/8-chart measurements | run concurrently; 3--4 min wall |

After authority/plan are sealed, the expected critical path is roughly
5--7 minutes for both initializer and alignment arms, excluding source/held
MASt3R production, selector resealing, atlas/gate construction and model cache
downloads.  Reserve about 100 MB for the source-only initializer/domain/
alignment namespace; downstream dense RADIO UV fields are separate and much
larger.

## Resume checklist

1. Receive the new authority file/content/source-tree hashes from the fresh
   physical-isolation audit; do not derive them from the old chain.
2. Receive the newly rebuilt exact-16 plan v3 path/content and confirm its
   official 16-name order and held-free cardinality flags.
3. Freeze the current alignment code-inventory hash.
4. Launch DAV2 on GPU 0 and MoGe-3 on GPU 1.
5. Validate both initializer manifests, then build v1 and seal v2/v3.
6. Launch paired 1,000-iteration alignments on GPU 0/GPU 1.
7. Report file/content hashes and lineage replay before any held-side build.
