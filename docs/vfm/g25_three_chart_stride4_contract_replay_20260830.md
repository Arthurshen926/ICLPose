# G25 three-chart stride-4 contract replay (2026-08-30)

## Result

The complete three-chart diagnostic has been replayed from the frozen
optimizer v1 domain with the current stride-aware v2/v3 sealers, current
strict atlas exporter and current evaluator.  It is internally valid:
`input_contract=PASS` and `scientific_performance_conclusion=EVALUATED`.

The scientific result remains **KILL** for both chart-atlas arms.  Both arms
pass only relative non-inferiority against an equally sparse M0 source-surface
control.  M0, M1 and M2 all fail the absolute held-coverage floor, while M1
and M2 also fail geometry integrity.  The composite gate is therefore KILL.

This replay is deliberately isolated under:

`output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_stride4_contract_replay_20260830/`

No initializer or alignment was rerun.  The frozen old-v4 three-chart source,
initializer and order-fixed alignment bytes are reused only as inputs to this
diagnostic.  It is not mixed with the fresh-v5 exact-16 source/held chain.

## Replayed domain chain

The current v2 sealer was run directly from optimizer v1 content
`26d2ccf9...`, then the current physical-reference-safe v3 sealer was run from
that newly emitted v2.  All three actual NPZs are colocated so the evaluator
can replay bytes rather than trust metadata claims.

| Artifact | File SHA-256 | Content SHA-256 | Topology/arrays SHA-256 |
|---|---|---|---|
| Optimizer v1 | `a367eeba41979a4e64f1faf3aeaff7525049563ea9f5c067016aba9e207eaf08` | `26d2ccf9030ffe5813d25b29bc849aa6ab36f876db4d80a545de470e498c1a0b` | frozen optimizer arrays |
| New exact-topology v2 | `42c7e3a85d157cba0a505d01267aec2d1273b69cb20f9a05e92e72d10190876b` | `12c04d6a9df3e91b350008df5d8a8ab6eeb105ead654cfcc1012c750ddee978d` | `e1765ecdb7165af557d676b676ae750e63f9241b32088ccdbcd2a1277590c883` |
| New reference-safe v3 | `6ab5a08d3a348dd7c643623772b91d7bef85afca1a9d5af754fc41f9225448a9` | `41189d64f38773fc713bf164e858fdfa719181664cfc396d75edda856a7c9177` | arrays `8a5794ba...`; exact topology `4eae5062...` |

The v3 physical audit again removes four unsafe stride-4 quads from the 1,085
v2 quads, leaving 1,081.  The worst removed source-reference unit edge is
6.4867 m.  Stride 4 is the explicit full-gate topology; stride 8 remains a
diagnostic inventory and does not drive the formal result.

## Rebuilt source bound, M0 and held inventory

The input builder froze the source-only bound before opening the isolated held
run.  It used topology stride 4, 1.0 m margin, held confidence threshold 0.25
and temporal block size 3.

| Artifact | File SHA-256 | Content SHA-256 |
|---|---|---|
| Bounded authority JSON | `ba3417b382ba4b546c51e65d917ce38b2afba66a1a4bab0290f7ad73375bf2c4` | authority `d0908e1b...`; canonical AABB `af1a021a8606c9f6cf6d1ced1ad5de1089a6f3f2ff21956b834ef871ba88520e` |
| M0 source-reference surface | `77621bcf4d113f8fa57134fd6e52e3402616fa4b7d1766dccc24bf2f2b0b5d2f` | `a7b6fca0ec8d3562c9684e4e86f73602ed26b314a8cde909f97b09972dcd640d` |
| Held ray inventory | `bb396d25f4793242e330cdca562fd1d8e13e5214f259c330e2de63013d2babb8` | `9edb87c89c63c3ed23e9d5649e3e6e949b6258894eff5b1bc243747429ef69e0` |
| Input-build audit | `7582b8c847ee1d16fd4cc70f32daa0be9d25191ceda7fac10c11ab0cd96da0d0` | `3ec586c542dc5e1168b1cdbb971a5bd9cd5c8c745623bea57d5301f733c37a2e` |

The AABB is `[-1.83195, -38.58570, -3.67497]` to
`[12.42110, -3.14406, 22.00268]` m.  All 12 held views are preserved and
contain 31,244 valid reference rays in total.  Missing renders remain failures.
M0 is an equal-budget source-reference surface control, not 2DGS or sensor
depth ground truth.

## Four strict atlases

All atlases contain the same official-order three chart names, 1,583 packed
vertices and 2,162 triangles.  Their chart names/order, vertex offsets, packed
pixel indices, UVs, face offsets and faces are byte-identical; only geometry
and arm/state lineage differ.

| Atlas | File SHA-256 | Content SHA-256 |
|---|---|---|
| M1 DAV2 initial | `c6f3da27b28f5ead04abd2d2a174d14a32be265d6d6ac6d88333e16aa165f6fa` | `c3f37d4e98f439037b56a9dc825a89cf85423dd1a4bba45eaa7426bf61663a52` |
| M1 DAV2 aligned | `e2caf8b80cad85e5fd627a95ac2b89158b9c9b93e8a1aeb24544cc8228c686af` | `01846d16595c045f83b57b102f4306690066d61cfb108305ba5e37f4960f0d7c` |
| M2 MoGe-3 initial | `89af0daccb4141675960a97a3825556826a5066fb8248734cd6b53a64eb79777` | `185aef4365ab61cf76f6e1d0dd362c4f310ee69d721786e6bf56240a0352eb5f` |
| M2 MoGe-3 aligned | `9f497f5ba3eaa7e622577ffc02aeaa608de5403a8b4e0e8e49b08bc4a0d08172` | `1be6b3302ba606baf285334bb4617f79a01398e1f7bdc81b9fbc83f7a1138c59` |

The exporter replays the same new v3 -> new v2 -> optimizer-v1 chain for all
four outputs, plus exact alignment/initializer manifests and rows, source
camera bytes, authority, plan and new bound.

## 2,000-bootstrap gate

Final report:

`output/g25_pose_transport/explicit_chart_atlas/dense_local_v4_stride4_contract_replay_20260830/full_submap_geometry_gate_stride_contract_final.json`

- file SHA-256: `d01e8541793a3b44088bfd3cac509fa92080adf67cfa3f970a363b87635fdcd4`;
- content SHA-256: `7e177a2298ee1b0389395972919596f8915e81910a281d0b47a665291f5f7c84`;
- bootstrap: 2,000 resamples, seed 260830;
- evaluator core SHA-256: `d9e1977d1b64a0f3dcaa73f4718109a3a1b991d79eef4ff2983145ac3d865da9`;
- evaluator CLI SHA-256: `a17678479aa0bdd3a8010c5e26b88cb711fd452f6a6584d527a1c8df7575be8b`.

| Macro-view metric | M0 | M1 DAV2 | M2 MoGe-3 |
|---|---:|---:|---:|
| Rendered-ray recall | 0.05893 | 0.06077 | 0.06113 |
| Good-ray recall | 0.05734 | 0.05629 | 0.05672 |
| Joint depth+normal recall at 20 degrees | 0.02949 | 0.02852 | 0.02946 |
| Conditional AbsRel mean | 0.00773 | 0.00989 | 0.00962 |

The paired block-bootstrap relative non-inferiority decision is GO for M1 vs
M0, M2 vs M0 and M2 vs M1.  This does not establish strict dominance: M2 vs
M1 strict dominance is KILL because good-ray recall does not pass.

Absolute coverage is KILL for all three representations because mean good-ray
recall is below 0.20 and mean joint depth+normal recall at 20 degrees is below
0.10.  The sparse inventory, rather than initializer choice, is the common
first-order failure.

Geometry integrity is independently KILL:

- M1 normalized chart-area ratio reaches 1.496; M2 reaches 1.145;
- M1 seam thickness P50/P90 is 0.296/0.821 m and unsigned-normal P90 is
  34.86 degrees;
- M2 seam thickness P50/P90 is 0.304/0.827 m and unsigned-normal P90 is
  35.86 degrees.

M1 fails chart area plus all seam gates; M2 passes the chart-area ceiling but
still fails seam P50, seam P90 and seam normal.  Both composite decisions are
therefore KILL.  `production_eligible=false` remains mandatory because this
is a single-route mapping-geometry diagnostic, not query localization or
sensor-depth ground truth.

## Interpretation

The current stride contract, current v3 physical-safety chain and evaluator
are now fully replayed; no unresolved interface ambiguity can explain the
three-chart KILL.  The numerical result is intentionally unchanged from the
previous reference-safe run because the final physical stride-4 arrays are
the same, while the new metadata/lineage hashes prove the current contract was
actually consumed.

The next informative experiment remains the fresh-v5 exact-16 chain.  It must
use its new authority, selector plan, initializers, domains, alignments, bound,
held inventory and atlases end to end.  Reusing any artifact from this
three-chart diagnostic would invalidate that comparison.

