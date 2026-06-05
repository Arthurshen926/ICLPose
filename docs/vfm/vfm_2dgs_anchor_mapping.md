# VFM-2DGS Anchor Mapping

## Scope

This stage redesigns only the mapping step. It builds a persistent 3D VFM anchor map from raw RADIO VFM patch tokens and a trained 2DGS/3DGS-style Gaussian surface map. It does not run query matching or localization.

## Method

The previous Gaussian path effectively assigned a VFM token to one projected Gaussian center. That is mismatched to VFM patch tokens, whose descriptor represents a local image region. The new mapping uses:

- 2DGS Gaussian disks as surface elements.
- Token footprint to projected-disk overlap, not token-center to Gaussian-center ownership.
- Projection-depth support sets: all near-front surface elements whose projected disk overlaps a salient VFM token footprint.
- Soft responsibility weights from opacity, projected distance, projected disk radius, and optional view-angle weighting.
- Multi-view anchor fusion by weighted surface-support IoU, normal agreement, and center distance.

The output anchor stores:

- `center`, `normal`, `covariance`
- raw 1280D VFM feature
- feature variance across observations
- quality and purity diagnostics
- ragged support 2DGS element ids and weights
- observed reference view ids

## Implementation

Core module:

- `feature_extract/vfm/vfm_2dgs_mapping.py`

Tools:

- `feature_extract/tools/vfm/build_vfm_2dgs_anchor_map.py`
- `feature_extract/tools/vfm/fuse_vfm_2dgs_observation_banks.py`
- `feature_extract/tools/vfm/visualize_vfm_2dgs_anchor_map_camera.py`

Tests:

- `tests/test_vfm_2dgs_mapping.py`

## OldHospital Smoke

Command output:

- map: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/full_g504k_v32_projection_depth/anchor_map.npz`
- summary: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/full_g504k_v32_projection_depth/summary.json`
- camera-view visualizations: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/full_g504k_v32_projection_depth/vis_camera/`

Configuration:

- 504,352 source Gaussians
- 425,528 surface elements after opacity filtering
- 32 uniformly sampled reference views
- raw `radio_final` 1280D features
- token top fraction `0.005`
- token footprint radius `1.5` token-grid units
- projected disk radius cap `2.0` token-grid units
- projection-depth support mode

Result:

- token-surface observations: 1,279
- fused VFM-2DGS anchors: 238
- feature dim: 1280
- observation support count: mean `4.78`, median `2`, p90 `12`, max `37`
- anchor support count: mean `6.50`, median `3`, p90 `17.3`, max `41`
- observations per anchor: mean `2.74`, median `2`, max `15`
- feature variance: mean `0.0427`, median `0.0264`, p90 `0.0923`

Compared with the initial center-neighborhood assignment, projection-depth support avoids collapsing each token to a single Gaussian. In the same 120k-Gaussian/16-view smoke, anchor support mean increased from about `1.03` to `3.06`, and observation support mean increased from about `1.02` to `2.49`.

## Mapping Completeness Update

The first smoke was intentionally minimal. The next implementation pass added several pieces from `ChatGPT-3D VFM Anchor 构建.md`:

- spatially balanced token proposal with `token_selection_mode=grid_top`;
- approximate bidirectional responsibility using token-normalized support and element footprint coverage;
- depth / normal / opacity / component purity diagnostics;
- `min_purity` filtering;
- anchor stability from multi-view feature variance;
- anchor distinctiveness from nearest-neighbor feature similarity;
- camera-view visualization with proper camera-to-image scaling.

The following mapping components are now also implemented:

- virtual 2DGS surface cells for large disks, controlled by `--virtual_cell_max_scale`;
- persistent surface-element layer via `--surface_npz`;
- anchor support parent Gaussian ids, so virtual cells remain traceable to source 2DGS disks;
- token soft-footprint sampling via `--footprint_sample_grid` and `--footprint_sample_extent_px`;
- multi-prototype anchor descriptors via `--max_feature_prototypes` and `--prototype_min_cosine`.
- view-bin descriptors via `--view_bin_count`, using observation view directions;
- spatial anchor NMS via `--spatial_nms_radius` / `--max_anchors`;
- persistent covisibility graph via `--covisibility_min_score` / `--max_covisibility_neighbors`.
- flat descriptor index export via `--descriptor_index_npz`, including multi-prototype descriptors when enabled.
- FAISS inner-product descriptor sidecar via `--descriptor_faiss_index`.
- per-view projected contribution buffers via `--contribution_dir`;
- renderer alpha-compositing contribution buffers via `--contribution_renderer gsplat_2dgs`;
- persistent token observation layer via `--observation_bank_npz`.

OldHospital full-Gaussian / 32-view comparison:

| config | observations | anchors | anchor support mean | purity median | quality median |
| --- | ---: | ---: | ---: | ---: | ---: |
| top `0.005`, no purity | 1,279 | 238 | 6.50 | 1.000 | 0.0839 |
| grid-top `0.02`, purity diagnostic | 5,213 | 959 | 7.49 | 0.203 | 0.0030 |
| grid-top `0.02`, `min_purity=0.2` | 2,524 | 454 | 7.97 | 0.472 | 0.0130 |

The grid-balanced configuration increases map coverage by about 4x over the minimal smoke. The purity-filtered configuration removes many low-purity boundary/floating observations, but camera-view visualizations still show some anchors around sky/tree regions. That is expected for the current approximate projection-depth path: it does not yet use real renderer contribution buffers.

Current best mapping-only outputs:

- broad coverage map: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/full_g504k_v32_gridtop_purity/anchor_map.npz`
- purity-filtered map: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/full_g504k_v32_gridtop_purity02/anchor_map.npz`
- purity-filtered camera-view visualizations: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/full_g504k_v32_gridtop_purity02/vis_camera_scaled/`

Virtual-cell smoke:

- no-virtual control: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/no_virtual_g120k_v16_gridtop_purity02/`
- virtual-cell map: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/virtual_cells_g120k_v16_gridtop_purity02/`
- virtual-cell surface elements: 176,968 cells from 104,308 parent Gaussians; 109,797 cells are virtual splits.
- Compared with the no-virtual control, virtual cells reduced median feature variance from `0.0723` to `0.0604` and increased mean quality from `0.0361` to `0.0444` in the 120k-Gaussian / 16-view smoke.

Prototype smoke:

- output: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/virtual_cells_g120k_v8_prototypes_gridtop_purity02/anchor_map.npz`
- features shape: `(28, 1280)`
- feature prototypes shape: `(28, 4, 1280)`
- 11 / 28 anchors kept more than one prototype.

View-bin smoke:

- output: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/virtual_cells_g60k_v4_viewbins_contrib_index_gridtop_purity02/anchor_map.npz`
- anchors: `140`
- view-bin features shape: `(140, 4, 1280)`
- view-bin counts shape: `(140, 4)`
- non-empty anchor/bin pairs: `145`

Spatial NMS + covisibility smoke:

- output: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/virtual_cells_g120k_v8_nms_covis_gridtop_purity02/anchor_map.npz`
- NMS radius `0.10m`: pre/post anchors `24 -> 24`; covisibility edges `136`.
- covisibility degree: mean `5.67`, median `5.5`, max `8`.
- stronger NMS radius `0.50m`: pre/post anchors `24 -> 23`; covisibility edges `128`.
- NPZ round-trip stores `covisibility_offsets`, `covisibility_anchor_ids`, and `covisibility_scores`.

Descriptor index smoke:

- output: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/virtual_cells_g120k_v8_index_gridtop_purity02/descriptor_index.npz`
- source anchors: `24`
- descriptor rows after prototype expansion: `34`
- descriptor dimension: `1280`
- self-search top-1 cosine is `1.0`, confirming round-trip and cosine lookup semantics.

FAISS index smoke:

- output: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/faiss_smoke_g5k_v1/descriptor.faiss`
- metric: inner product over L2-normalized descriptors
- `ntotal`: `34`
- dimension: `1280`

Full layer smoke:

- output: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/virtual_cells_g60k_v4_full_layers_gridtop_purity02/`
- views: `4`
- contribution buffer files: `4`
- token observations: `162`
- observation feature dimension: `1280`
- observation support entries: `361`
- anchors: `140`
- descriptor rows: `148`

Renderer contribution smoke:

- output: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/gsplat2dgs_g20k_v2_full_layers_gridtop_purity02/`
- renderer: `gsplat_2dgs`
- device: `cuda`
- source views: `2`
- surface elements: `28,754`
- contribution buffer files: `2`
- token observations: `39`
- anchors: `34`
- descriptor rows: `35`
- covisibility edges: `272`
- first contribution buffer renderer field: `gsplat_2dgs`
- first contribution buffer support entries: `455`

The `gsplat_2dgs` contribution path renders the current surface-element layer as 2D Gaussian disks, obtains renderer intersections from `gsplat.rasterization_2dgs`, computes per-pixel 2DGS alpha values with the public PyTorch formula, performs front-to-back alpha compositing to get `T_i alpha_i`, and integrates those contributions over VFM token footprints.

Camera-view diagnostic:

- summary: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/gsplat2dgs_g20k_v2_full_layers_gridtop_purity02/vis_camera_summary.json`
- visualized views: `2`
- output PNGs:
  - `output/vfm/vfm_2dgs_anchor_maps/OldHospital/gsplat2dgs_g20k_v2_full_layers_gridtop_purity02/vis_camera/seq9__frame00001.png_vfm_2dgs_anchor_map.png`
  - `output/vfm/vfm_2dgs_anchor_maps/OldHospital/gsplat2dgs_g20k_v2_full_layers_gridtop_purity02/vis_camera/seq1__frame00013.png_vfm_2dgs_anchor_map.png`
- visible anchors: `29` and `26`
- visible support elements: `459` and `420`

## Current Limitations

- `projection_depth_soft` remains an approximate fallback. Use `--contribution_renderer gsplat_2dgs` for renderer alpha-composited token-to-surface support.
- The `gsplat_2dgs` path renders the current `SurfaceElementMap`. If virtual cells are enabled, those virtual cells are rendered as mapping-time 2DGS disks; this is the intended VFM surface-element layer, not a re-training of the original 2DGS model.
- Token purity still uses depth/normal/opacity statistics over the selected surface elements; depth comes from renderer metadata for `gsplat_2dgs`, but boundary maps are not yet exported as separate images.
- Component purity is only exact when a meaningful surface adjacency graph is enabled; the fast full-map smoke uses projection-depth support without adjacency.
- View-bin descriptors are based on observation view-direction azimuth; more robust hemisphere/elevation binning is still open.
- No localization or descriptor selection is included in this stage.

## Canonical Mapping-Only Protocol

The canonical mapping diagnostic now uses:

- raw `radio_final` 1280D tokens;
- `--canonical_vfm_2dgs`;
- `--contribution_renderer gsplat_2dgs`;
- `--support_mode surface_component`;
- automatic projected-token virtual-cell scale estimation;
- persistent `surface_elements.npz`, `observation_bank.npz`, and `descriptor_index.npz`;
- mapping-only evaluation with `feature_extract/tools/vfm/eval_vfm_2dgs_mapability.py`;
- camera-view visualization with surface-element support and token observation boxes.

Implementation fixes in this pass:

- 2DGS surface bases are normalized to a right-handed orthonormal frame before renderer export.
- Renderer denominator handling preserves negative denominators instead of clamping them positive.
- Surface adjacency now uses disk/cell support extent, not only center distance. This prevents `surface_component` mode from fragmenting a VFM token support into single-element components.
- Virtual-cell scale can be estimated from VFM token-grid projection via `--auto_virtual_cell_target_px`.
- Anchor fusion can use adjacency-dilated support IoU and parent-Gaussian support IoU. The canonical protocol enables one-hop dilated support association.
- `--canonical_token_supply broad_grid` increases map-supported token observations before quality filtering, instead of relying on a very sparse top-token proposal.
- Support-association caching avoids recomputing dilated support for every observation/anchor comparison.
- `fusion_mode=graph` clusters token observations with union-find instead of order-dependent greedy merging.
- Mapping-only evaluation can report surface-seed consensus when `surface_elements.npz` is provided.
- `token_selection_mode=surface` can generate token candidates directly from renderer contribution coverage.
- Surface token ranking can balance renderer contribution and VFM saliency through `surface_token_saliency_power`.

OldHospital canonical smoke:

| config | renderer | support mode | views | observations | anchors | obs support median/mean | anchor support median/mean | LOO R@1 | LOO R@5 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `canonical_g20k_v8_maponly` before adjacency fix | `gsplat_2dgs` | `surface_component` | 8 | 97 | 79 | 1 / 1.02 | 1 / 1.03 | 0.667 | 0.758 |
| `canonical_g20k_v8_adjfix_maponly` | `gsplat_2dgs` | `surface_component` | 8 | 163 | 123 | 2 / 2.88 | 2 / 3.02 | 0.653 | 0.847 |
| `canonical_g20k_v8_tolerant_maponly` | `gsplat_2dgs` | `surface_component + 1-hop fusion` | 8 | 163 | 122 | 2 / 2.88 | 2 / 3.04 | 0.653 | 0.847 |
| `canonical_g20k_v8_broad_tolerant_cached_maponly` | `gsplat_2dgs` | `surface_component + 1-hop fusion` | 8 | 336 | 201 | 2 / 3.99 | 2 / 3.77 | 0.659 | 0.858 |
| `canonical_g20k_v8_broad_graph_maponly` | `gsplat_2dgs` | `surface_component + graph fusion` | 8 | 336 | 199 | 2 / 3.99 | 2 / 3.75 | 0.692 | 0.863 |
| `canonical_g20k_v8_surface_graph_maponly` | `gsplat_2dgs` | `surface-token + graph fusion` | 8 | 279 | 104 | 14 / 21.1 | 14 / 21.1 | 0.632 | 0.846 |
| `canonical_g20k_v8_surface_balanced_graph_maponly` | `gsplat_2dgs` | `surface-token + saliency balanced + graph fusion` | 8 | 101 | 57 | 10 / 10.4 | 10 / 10.4 | 0.758 | 0.970 |
| `canonical_g20k_v8_dual_broad_surface_balanced_graph_maponly` | `gsplat_2dgs` | `broad + surface-balanced observation fusion` | 8 | 420 | 93 | 3 / 6.51 | 3 / 6.51 | 0.785 | 0.920 |
| `canonical_g20k_v8_dual_feature_agree06_graph_maponly` | `gsplat_2dgs` | `broad + surface-balanced + source feature-agreement fusion` | 8 | 420 | 92 | 3 / 6.38 | 3 / 6.38 | 0.790 | 0.920 |

The adjacency fix restores region-like surface support and improves held-out R@5 / coverage. However, the current canonical map is still not a proven replacement for SfM anchors: observation count per anchor remains low, with median `1`, so many anchors still lack true multi-view VFM consensus.

The broad token-supply run confirms that sparse token proposal is a major bottleneck:

- mean observations per view increased from `20.4` to `42.0`;
- anchors with at least three observations increased from `5.7%` to `15.4%`;
- held-out observation R@5 increased slightly from `0.847` to `0.858`;
- feature positive cosine increased from `0.775` to `0.823`.

This is positive mapping-only evidence, but the map still falls short of the target consensus level. A robust VFM-2DGS map should aim for many more `obs>=3` anchors under a larger representative view set.

Graph fusion gives a small but clean improvement over greedy fusion:

- R@1 increased from `0.659` to `0.692`;
- R@5 increased from `0.858` to `0.863`;
- `obs>=5` anchor ratio increased from `2.0%` to `2.5%`.

Surface-seed diagnostics on `canonical_g20k_v8_broad_graph_maponly`:

- observed surface elements: `745`;
- observed parent Gaussians: `665`;
- multi-view element seeds: `87`;
- multi-view parent seeds: `69`;
- multi-view element / parent anchor recall: `1.0 / 1.0`.

This means the latest fusion path is capturing the multi-view surface seeds that exist in the current observation bank. The remaining bottleneck is upstream: the observation bank still contains too few surface elements / parents that are seen by multiple selected VFM tokens across views.

Surface-driven token proposal changes the tradeoff:

- pure surface contribution supply finds many more multi-view seeds (`649` multi-view parents), but supports are large and noisy, lowering purity and R@1;
- a saliency-balanced surface supply (`surface_token_saliency_power=0.5`, `min_purity=0.3`, `min_component_concentration=0.7`) yields a high-confidence subset with R@5 `0.970`, but only `66` leave-one-out retrieval queries.

Current interpretation:

- `broad_graph` is the better broad-coverage map;
- `surface_balanced_graph` is the better high-confidence map;
- `dual_broad_surface_balanced_graph` is the first map that combines both sources and improves over broad coverage on both R@1 and R@5.

The dual-source map is built from existing observation banks, without rerunning the renderer:

```bash
PYTHONPATH=. python feature_extract/tools/vfm/fuse_vfm_2dgs_observation_banks.py \
  --surface_npz output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_broad_graph_maponly/surface_elements.npz \
  --observation_bank broad=output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_broad_graph_maponly/observation_bank.npz \
  --observation_bank surface_balanced=output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_surface_balanced_graph_maponly/observation_bank.npz \
  --fusion_mode graph \
  --min_surface_iou 0.2 \
  --min_dilated_surface_iou 0.15 \
  --support_iou_dilation_hops 1 \
  --min_observations 2 \
  --output_npz output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_dual_broad_surface_balanced_graph_maponly/anchor_map.npz \
  --merged_observation_bank_npz output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_dual_broad_surface_balanced_graph_maponly/observation_bank.npz \
  --summary_json output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_dual_broad_surface_balanced_graph_maponly/summary.json
```

Dual-source diagnostic:

- selected observations after dedupe: `332` from broad and `88` from surface-balanced, with `17` duplicates removed;
- global leave-one-out retrieval: `R@1=0.785`, `R@5=0.920`, `query_count=288`;
- broad-source query breakdown: `R@1=0.769`, `R@5=0.914`, `query_count=221`;
- surface-balanced query breakdown: `R@1=0.836`, `R@5=0.940`, `query_count=67`;
- multi-view parent seeds increased to `203`, with anchor recall `1.0`.

Dual-source camera-view visualization:

- summary: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_dual_broad_surface_balanced_graph_maponly/vis_camera_summary.json`
- output directory: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_dual_broad_surface_balanced_graph_maponly/vis_camera/`
- visualized views: `4`
- visible anchors per view: `81`, `51`, `69`, `79`
- visible support elements per view: `576`, `270`, `515`, `525`

This supports a two-tier mapping interpretation: broad observations provide coverage, while surface-balanced observations provide a cleaner high-confidence subset. Blind dual-source fusion improved retrieval but collapsed the map to fewer anchors (`93`) and raised feature variance, motivating source-aware fusion.

Source-aware fusion update:

- `source_merge_policy=same_source` reduces feature variance (`0.0871 -> 0.0805` mean), but drops R@1 (`0.785 -> 0.735`), so fully separating sources is too conservative.
- `source_merge_policy=feature_agree` allows cross-source fusion only when descriptor cosine exceeds `cross_source_min_feature_cosine`.
- A threshold sweep on OldHospital gives:

| policy | threshold | anchors | feature variance mean | LOO R@1 | LOO R@5 | query count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| blind dual | n/a | 93 | 0.0871 | 0.785 | 0.920 | 288 |
| same source | n/a | 91 | 0.0805 | 0.735 | 0.919 | 272 |
| feature agree | 0.6 | 92 | 0.0851 | 0.790 | 0.920 | 286 |
| feature agree | 0.7 | 90 | 0.0832 | 0.787 | 0.918 | 282 |
| feature agree | 0.8 | 88 | 0.0833 | 0.784 | 0.917 | 278 |
| feature agree | 0.9 | 87 | 0.0825 | 0.766 | 0.920 | 274 |

Current best mapping-only default is therefore `feature_agree` with threshold `0.6`: it slightly improves R@1 over blind dual while reducing mean feature variance and preserving R@5.

Feature-agreement source breakdown:

- broad queries: `R@1=0.773`, `R@5=0.914`, `query_count=220`;
- surface-balanced queries: `R@1=0.848`, `R@5=0.939`, `query_count=66`.

Feature-agreement camera-view visualization:

- summary: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_dual_feature_agree06_graph_maponly/vis_camera_summary.json`
- output directory: `output/vfm/vfm_2dgs_anchor_maps/OldHospital/canonical_g20k_v8_dual_feature_agree06_graph_maponly/vis_camera/`
- visualized views: `4`
- visible anchors per view: `80`, `50`, `69`, `78`
- visible support elements per view: `569`, `263`, `515`, `518`

## Next Mapping-Only Checks

Before re-entering localization, the next useful checks are:

- run the same mapping on ShopFacade once a ShopFacade Gaussian PLY is available in the workspace;
- compare the current feature-agreement fusion against a true two-layer descriptor index, where broad and surface-balanced anchors are searched separately then merged at retrieval/scoring time;
- sweep `token_top_fraction`, `footprint_radius_px`, and `max_projected_disk_radius_px`;
- compare exact renderer contribution support against projection-depth support;
- add optional virtual disk subdivision for very large 2DGS disks;
- visualize feature PCA colors in camera view for anchor support consistency.
