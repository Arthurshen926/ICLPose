# V8.1 Non-redundant Region/Surface Localization

## Decision

V8.1 is a diagnostic research branch. It is **not** promoted over the frozen
V8.0-B control. It fixes real probability and proposal bugs, but the final
Strict12 pose result remains metre-scale and does not meet the decimetre gate.

The multi-teacher student remains rejected: its held-out retrieval gain did
not improve strict pose. The runtime map continues to store one canonical
RADIO-derived feature type only.

## Runtime contract

```text
query RGB
  -> RADIO-final
  -> calibrated maplet posterior
  -> non-redundant local evidence groups
  -> query-independent virtual pose lattice
  -> maplet x image-cell inverted retrieval
  -> structured coarse re-ranking + SE(3) mode NMS
  -> footprint/graph surface score
  -> optional local coordinate refinement
```

The map/lattice stores no mapping RGB, mapping image ID/path, downstream
teacher embedding, SfM point/track, ALIKE descriptor, RADIO intermediate,
LoFTR match or point-correspondence PnP input.

## Bugs corrected

1. The V8 physical artifact contains 17,436 directed records but only 8,718
   unique undirected relations. V8.1 canonicalizes physical lookup and query
   edges, so every relation is scored once.
2. Fixed RADIO samples are not assumed independent. Overlapping and
   appearance-consistent supports are grouped by a medoid rule which prevents
   transitive facade-sized chains. Complete categorical mass is conserved.
3. A local VFM support is no longer forced to have the same centre and scale
   as a complete physical maplet. Unary evidence uses projected-footprint
   containment.
4. Maplet assignment has a global projected-area capacity term and a reverse
   maplet-to-query visibility term.
5. The virtual lattice is ranked before truncation with medium/long regional
   relations. Subdivision now applies SE(3) mode NMS; otherwise 63 variants of
   one wrong mode consume the entire Top-N.
6. New artifacts live in `mainline_v81` and use fail-closed SHA-256 lineage for
   the canonical feature bank, physical graph and anonymous pose statistics.

## Graph identifiability audit

The 807-node physical graph is not the limiting ambiguity. With quantized
node geometry alone, 74.72% of nodes have a unique signature. With one rooted
graph hop, all 807 nodes have distinct signatures; two and three hops remain
fully distinct. The failure is therefore in transferring query posterior
evidence to the correct graph configuration and continuous pose, not in an
intrinsically isomorphic physical graph.

## O1--O4 oracle results on Strict12

| Diagnostic | median translation | P90 translation | <=1m/10deg | <=50cm/5deg | <=30cm/3deg |
|---|---:|---:|---:|---:|---:|
| O1 oracle maplet support + identity | 0.50 m | 0.50 m | 91.67% | 75.00% | 33.33% |
| O2 current local supports + oracle identity | **0.30 m** | 0.50 m | **100%** | **91.67%** | **75.00%** |
| O3 current posterior + GT-centred dense local lattice | 0.49 m | 1.00 m | 83.33% | 50.00% | 41.67% |
| O3 coordinate refinement | 0.67 m | 1.38 m | 58.33% | 33.33% | 16.67% |

The important result is O2 being better than O1. Forcing 128 supports into
24--64 complete region boxes is not justified: the conservative grouping
produces 98--107 evidence instances (median 103), and the current local
supports retain useful within-maplet position. Large-region aggregation loses
that information.

Coordinate refinement makes O3 worse. The footprint score does not have a
well-behaved decimetre convergence basin and must not be used as Stage C.

## Virtual pose lattice

The query-independent artifact contains 558,848 poses:

- 4,366 anonymous feasible positions at 1 m spacing with 3 m trajectory
  expansion;
- 128 orientation prototypes;
- 43,426,901 sparse `maplet x 4x4 image cell` visibility keys;
- 19.4 MB compressed size.

The lattice itself has median oracle spacing 0.47 m / 3.83 degrees. Coarse
Top-8192 covers 10/12 queries within 1 m / 10 degrees. Structured re-ranking
puts 6/12 inside Top-16. Fixing subdivision mode collapse raises its Top-16
coverage from 3/12 to 7/12, but none of the 12 has a Top-16 mode inside
50 cm / 5 degrees.

The final graph score discards four of those seven valid coarse modes:

| Deployment result | median translation | P90 translation | median rotation | P90 rotation | <=1m/10deg |
|---|---:|---:|---:|---:|---:|
| V8.1 virtual lattice + graph | 1.72 m | 5.75 m | 5.09° | 13.36° | 25.00% |
| V8.1 plus coordinate refinement | 1.72 m | 5.06 m | 5.10° | 12.93° | 25.00% |

This does not exceed the published frozen V8.0-B control (1.68 m median,
5.72 m P90 translation). V8.1 is therefore not promoted.

## First-principles conclusion

The experiments separate the remaining failure into two parts:

1. identity/configuration proposal is incomplete but no longer the sole
   blocker: diverse graph candidates reach a usable coarse basin in 7/12;
2. a box-level region/maplet likelihood cannot identify a continuous location
   inside a broad maplet and cannot reliably rank/refine those valid modes.

The next justified module is not a larger lattice, another graph weight, or a
new stored embedding. It is continuous maplet-interior alignment: render
disconnected retrieved atlas regions at each diverse coarse pose, correlate
the single canonical RADIO feature with the query feature field, estimate a
continuous surface displacement/pose update, and measure its convergence basin
before end-to-end deployment. The existing O2 result is the gate: such a module
must preserve the local supports' within-maplet information and improve over
the 0.30 m oracle-identity median.

## Reproducible artifacts

- final merged conclusion:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v81/v81_strict12_final_conclusion_v1.json`
- rooted graph audit:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v81/graph_identifiability_v1.json`
- virtual lattice:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v81/virtual_pose_lattice_1m_r3_k128_c4_v1.npz`
- O1--O4 shards:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v81/oracles_strict12_shard{0,1}_v1.json`
- final NMS deployment shards:
  `output/vfm/2dgs_surface/StMarysChurch/full_train/mainline_v81/virtual_lattice_nms_strict12_shard{0,1}_v5.json`

Focused verification: 130 tests pass across V6 atlas/frame, V8 and V8.1.
