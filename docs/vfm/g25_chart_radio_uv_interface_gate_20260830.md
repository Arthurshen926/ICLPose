# G25 explicit-chart RADIO UV interface gate (2026-08-30)

## Verdict

The coordinate/fusion interface is a **GO for the next surface-family gate**,
but the current eight-chart artifact is **not a deployment map and not a pose
result**.

The gate establishes the following strict split:

1. `SourceViewChartRadioField` is an offline observation artifact. It retains
   every 36x64 RADIO-final token, source-view identity, mapping camera and the
   exact coordinate transform.
2. `CanonicalSurfaceFamilyLayout` is the only bridge from source vertices to
   physical family/node identity. In strict mode its family identity comes
   from the separately sealed `goal_maplet_canonical_surface_family_carrier_v1`;
   the RADIO adapter may form metric sampling nodes only *within* an existing
   carrier family and can never merge families. A chart/view ID is not
   accepted as a canonical identity.
3. `CanonicalChartRadioField` first balances observations within each
   `(physical node, source view)`, then fuses across views. It stores only
   anonymous feature/view-direction prototypes and hash-only source lineage.

No query image, query pose/GT, ALIKE/PnP/SfM point correspondence, Gaussian or
2DGS is read by this builder.

## Coordinate correction

The aligned chart UV is defined on the actual 256x144 ideal-pinhole chart
camera, while RADIO-final is extracted from the 1024x576 raw SIMPLE_RADIAL
image. Treating the two grids as the same silently introduces both a resize
pixel-phase error and radial displacement. The final v2 ray-based contract
measures radial displacement up to 10.73 source pixels in this eight-chart
sample (mean 1.30 px).

The frozen mapping is:

```text
complete raw 36x64 RADIO endpoint grid
  -> inverse SIMPLE_RADIAL
  -> normalized ideal camera ray
  -> actual 256x144 chart camera
  -> chart UV

chart vertex ideal UV
  -> chart camera ray
  -> raw ideal camera
  -> forward SIMPLE_RADIAL
  -> raw pixel endpoint
  -> bilinear RADIO endpoint-grid sample
```

The source artifact stores raw camera `(W,H,f,cx,cy,k1)` values and validation
replays every token-to-chart coordinate without consulting an external path.

## Real eight-chart CPU smoke

Inputs:

- MoGe-3 initialized, masked, 1000-step seq4 chart alignment;
- eight mapping/train keyframes only;
- complete 1280-D RADIO-final 36x64 grids;
- raw mapping-camera manifest and chart mapping camera poses.

Results:

| Metric | Result |
|---|---:|
| Complete token inventory | 18,432 / 18,432 (100%) |
| Complete token grid inside chart canvas | 98.19% (all tokens still retained) |
| Atlas vertices with coordinate-valid RADIO samples | 98.50% |
| Atlas vertices with positive geometry/view weight | 63.15% |
| Diagnostic canonical nodes | 4,511 |
| Observed canonical nodes | 62.47% |
| Multi-view nodes | 82 (1.82%) |
| Same-node cross-view RADIO cosine, median | 0.732 |
| Random observed-node RADIO cosine, median | 0.275 |
| Multi-view feature uncertainty, median | 0.066 |

The cosine separation is evidence that the coordinate-correct attachment and
view-balanced fusion preserve meaningful cross-view appearance identity. The
very small multi-view-node fraction is simultaneously a fail-closed warning:
the diagnostic mutual-nearest canonicalizer is highly fragmented and cannot
stand in for the planned overlap-aware surface-family atlas.

This smoke predates the strict v2 source/held authority. Its manifest records
`strict_disjoint_authority_verified=false`; it is valid only as an interface
diagnostic. The builder now has a fail-closed, three-artifact composition
hook. A strict replay requires:

1. `goal_maplet_disjoint_chart_upstream_authority_v2`;
2. a separately sealed `goal_maplet_overlap_aware_chart_submap_plan_v2` whose
   exact selected order equals the atlas chart order and whose authority/tree
   hashes replay;
3. a separately sealed `goal_maplet_canonical_surface_family_carrier_v1`
   bound to the exact atlas content hash.

The plan and family carrier are deliberately not conflated. The plan selects
the source-only operational chart inventory; the carrier defines anonymous
physical surface-family identity. Their exact hashes, schemas, selected-name
hash, alignment-runner contract and atlas hash are sealed into a composition
audit. Both the plan content hash and carrier content hash must also be pinned
externally on the builder command line; reading an artifact's own hash and
feeding it back to itself is not accepted as runner authority.

The current strict-v2 selector replay is scientifically blocked before RADIO:
its valid v2 plan selects **zero** operational charts. The RADIO builder
therefore rejects it via the model-neutral plan loader instead of silently
falling back to source-view identities. A non-empty operational v2 plan and a
carrier rebuilt on that exact selected atlas are required for the next real
strict smoke.

## Resource gate

The geometry atlas alone is about 0.2 MiB, but that number must never be quoted
as the complete map footprint:

| Artifact | Payload | Serialized size |
|---|---|---:|
| Source-view field (offline) | 18,432 x 1280 float32 + mappings | 52.3 MiB |
| Diagnostic family layout | geometry/assignments | 0.18 MiB |
| Canonical field | 4,511 x 1280 + anonymous prototypes | 26.2 MiB |

The single-process CPU smoke took 19.5 s and reached 1.14 GiB peak RSS. The
summary binds these resource values and the 308 KiB diagnostic visualization
by file hash and serialized byte count.

Therefore the deployment resource verdict is **BLOCK**. Feature compression
must happen *after* physical-family fusion, using a separately validated
64-128D or sparse multiresolution field. Its acceptance criteria must include
global retrieval and pose-sensitive correspondence fidelity, not only PCA
reconstruction error.

## Next strict sequence

1. Produce a non-empty operational v2 chart-submap plan from the strict
   source-only inventory; the current plan contains zero selected charts.
2. Rebuild the explicit atlas and independent canonical family carrier on the
   plan's exact ordered names.
3. Use the carrier-constrained RADIO adapter. It resolves patch-boundary
   ownership deterministically, preserves unassigned vertices as `-1/zero`
   rather than inventing evidence, and never merges carrier families.
4. Increase mapping keyframe overlap/coverage; the current eight route-spaced
   views give only 82 multi-view canonical nodes.
5. Replay this unchanged RADIO source interface on the production family
   assignment and require higher multi-view surface coverage without degrading
   same-node/random feature separation.
6. Only then build a post-fusion compact feature field and validate retrieval
   plus pose-sensitive matching fidelity.
7. Run query-to-chart correspondence/pose oracles after the map-side carrier
   passes these gates; this artifact itself makes no localization claim.

## Artifacts

- Source: `feature_extract/vfm/localization_goal_maplet/chart_radio_uv_field.py`
- Builder: `feature_extract/tools/vfm/build_goal_maplet_chart_radio_uv_field.py`
- Visualization: `feature_extract/tools/vfm/visualize_goal_maplet_chart_radio_uv_field.py`
- Tests: `tests/test_goal_maplet_chart_radio_uv_field.py`
- Real smoke: `output/g25_pose_transport/explicit_chart_atlas/chart_radio_uv_seq4_8chart_moge3_smoke_v1/`
