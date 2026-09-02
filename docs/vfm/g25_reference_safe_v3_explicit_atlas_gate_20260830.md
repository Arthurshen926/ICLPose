# G25 reference-safe v3 explicit-atlas export and full gate (2026-08-30)

## Verdict

The strict exporter/interface is **GO**, while the present three-chart
geometry is **KILL** as a localization map.  Those are intentionally separate
conclusions.

The previous exact-topology v2 seal was exact only across DAV2 and MoGe-3.  It
did not include the source-MASt3R reference surface in its physical edge
safety test.  Four of 1,085 stride-4 quads were unsafe, including a 6.49 m
unit-pixel edge.  This was a P0 because M0, M1 and M2 did not actually consume
one physically valid face domain.  All v2 atlas/full-gate outputs are therefore
diagnostic-only.

The replacement `goal_maplet_chart_comparison_domain_v3` intersects the v2
mask with source-reference edge safety and then repacks vertices/faces without
orphans.  The strict CLI accepts v3 only.  Library callers may opt into v2 only
with `allow_legacy_v2_diagnostic=True`; such an atlas is stamped
`full_submap_gate_eligible=false` and `source_reference_edge_safe=false`.

## Replayed contract

The exporter fails closed unless it can replay all of:

```text
externally pinned disjoint source/held authority v2
  -> exact frozen source chart order from the model-neutral plan
  -> externally pinned physical-safe comparison domain v3
     -> actual externally pinned exact-topology v2 bytes
        -> optimizer comparison-domain v1 file/content recorded by alignment
  -> v3 chart/valid byte-equality and face-mask subset of v2
  -> v3 packed vertices/faces reconstructed exactly from its final masks
  -> physical-safety flags, mask hashes and safety-config hash
  -> authority source-reference root, camera bytes and selected pointmap bytes
  -> arm alignment manifest + runner/code/charts/cameras hashes
  -> initializer run manifest + selected per-view file/content hashes
  -> source-only bounded-submap authority bound to this exact v3 file/content
```

Initial geometry is read directly from original initializer bytes: DAV2
`points_world`, or MoGe-3 `points_camera` transformed by the frozen source
camera.  Aligned geometry is read from alignment `charts_data`.  The initial
atlas is not reconstructed from aligned rays and prior depth.  Both states use
the v3 packed pixel indices, offsets, faces and UV exactly; no arm-specific
confidence, depth-range or edge deletion is permitted at export.

The exporter also accepts the new cardinality-frozen chart-plan v3 schema.  A
v3 plan is accepted here only when minimum=maximum=16, the actual selected
count is 16, selection was frozen before held geometry, and held geometry was
not used for selection.  The official `selection_rank` order remains the sole
chart order.

## Real operational-3 artifacts

Physical-safe comparison domain:

- path: `output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_reference_safe_exact_topology_v3.npz`;
- file/content: `59c97f68...` / `3e6acde0...`;
- stride-4 topology: **1,081 quads**, 1,583 packed vertices and
  2,162 triangles;
- four unsafe v2 quads removed; stride-8 remains 156 quads.

New v3-bound gate root:

`output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_full_gate_reference_safe_v3/`

The source-only AABB changed materially after removing the bad reference
faces: maximum x shrank from 19.12 m to 12.42 m.  Its canonical AABB hash is
`af1a021a...`; 31,244 held reference rays remain inside it.

All four atlases contain the exact same 3 chart names, vertex/face offsets,
1,583 UV vertices and 2,162 triangles:

| Atlas | File SHA-256 | Content SHA-256 |
|---|---|---|
| M1 DAV2 initial | `8ae70c95...` | `5769cc1a...` |
| M1 DAV2 aligned | `cc1b1e60...` | `9ace2d56...` |
| M2 MoGe-3 initial | `629b41bb...` | `5f0dec27...` |
| M2 MoGe-3 aligned | `02a502ae...` | `30c509d9...` |

Initial-to-aligned vertex displacement is large enough to make the distortion
test meaningful: median 3.42 m for M1 and 3.81 m for M2.

## Full-gate result

The 2,000-resample report passes the complete input contract and is genuinely
evaluated:

`output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_full_gate_reference_safe_v3/full_submap_geometry_gate_v1.json`

| Macro-view metric | M0 source control | M1 DAV2 | M2 MoGe-3 |
|---|---:|---:|---:|
| Rendered-ray recall | 0.0589 | 0.0608 | 0.0611 |
| Good-ray recall | 0.0573 | 0.0563 | 0.0567 |
| Joint depth+normal recall@20 | 0.0295 | 0.0285 | 0.0295 |
| Conditional AbsRel median | 0.0077 | 0.0099 | 0.0096 |

Relative non-inferiority versus the equally sparse M0 control passes, but that
does not rescue the method: M0, M1 and M2 all fail the absolute coverage floor
(0.20 good-ray and 0.10 joint recall).  Missing rays remain failures.

Geometry integrity also fails.  M1 has normalized per-chart area ratios up to
1.50.  Seam thickness P50/P90 is 0.296/0.821 m for M1 and 0.304/0.827 m for
M2; unsigned-normal P90 is about 35 degrees for both.  Consequently both
composite atlas-vs-M0 decisions are **KILL**, and MoGe-3 does not establish
strict dominance over DAV2.

The first-order bottleneck is now clearly inventory coverage: three charts do
not cover the held submap, and even M0 fails the same absolute floor.  The
correct next test is the separately frozen held-free exact-16 source plan,
followed by new initializers/domain/alignment and this unchanged v3 exporter
and gate.  Seam/area failures remain a second independent optimization issue;
they must not be hidden by deleting faces after alignment.

This gate is still a source-disjoint mapping-geometry diagnostic, not Cambridge
sensor depth ground truth or a query-pose result.  It uses no query image,
query pose/GT, ALIKE, PnP, Gaussian training or held geometry for selection.

## Verification

- v3 domain/exporter/legacy-atlas/runner/full-gate/input-builder combined
  targeted suite: 58 passed;
- v3 real exporter replay: four of four artifacts built successfully;
- cross-arm/state topology and UV equality: byte-exact;
- full gate: `input_contract=PASS`, `scientific_performance_conclusion=EVALUATED`;
- rehashed v3 face addition outside v2, v2 default use, v1 substitution,
  initializer substitution, and cross-authority bounds all fail closed.

## Exact exporter CLI (M1 aligned)

The other three artifacts use the same fixed domain/plan/authority/bounds;
only the alignment directory, initializer directory/hash, arm, state and
output name change.

```bash
PYTHONPATH=. python feature_extract/tools/vfm/build_goal_maplet_explicit_chart_atlas.py \
  --charts output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_alignment/m1_dav2_operational3_v3_orderfixed_final/charts_data.npz \
  --cameras output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_alignment/m1_dav2_operational3_v3_orderfixed_final/cameras.json \
  --alignment_manifest output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_alignment/m1_dav2_operational3_v3_orderfixed_final/manifest.json \
  --expected_alignment_manifest_content_sha256 d9cd0f41c7c97cb80a306b8661460a559986378b55155e4ebf254005db4b4bce \
  --initializer_artifacts output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/dav2_source_seq4_24 \
  --initializer_manifest output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/dav2_source_seq4_24/manifest.json \
  --expected_initializer_manifest_content_sha256 bb50067290e644e819f07040c48ba477b194ffdfdef78d5d2f20d7b33339feae \
  --comparison_domain_v3 output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_reference_safe_exact_topology_v3.npz \
  --expected_comparison_domain_content_sha256 3e6acde0777afa58f287054c18b73aeafdfe1a1e5d0d45caf1304865d8b94656 \
  --upstream_exact_topology_v2 output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_initializers/comparison_domain_operational3_exact_topology_v2.npz \
  --expected_upstream_exact_topology_v2_content_sha256 84ff5f39a111a918c60c3ebb239b1ce3095f28f9ec8f4693cc70f24a6a26d811 \
  --frozen_submap_plan output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_isolated/source_seq4_dense_overlap_aware_submap_plan_v2.npz \
  --expected_plan_content_sha256 3c507e10a28f9e1edfd5255d959bad06dac601b700b1b20974a5e504a533948b \
  --disjoint_upstream_authority output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_isolated/disjoint_upstream_authority_v2.json \
  --expected_disjoint_authority_content_sha256 bd6f7d7bd238b8d29e36dd1311b29cb0cf80c8f11bb1d7deb4280040c9d4f207 \
  --bounded_submap output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_full_gate_reference_safe_v3/bounded_submap_authority_v1.json \
  --expected_bounded_submap_content_sha256 af1a021a8606c9f6cf6d1ced1ad5de1089a6f3f2ff21956b834ef871ba88520e \
  --initializer_arm DAV2 \
  --geometry_state aligned \
  --stride 4 \
  --output output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_full_gate_reference_safe_v3/m1_dav2_aligned_atlas_reference_safe_v3.npz
```
