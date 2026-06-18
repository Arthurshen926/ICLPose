# RADIO MATCHA Adaptation Design

## Goal

Replace the current lightweight MATCHA-like training path with a RADIO-adapted MATCHA implementation. The implementation keeps RADIO as the frozen feature source and does not import MATCHA's DIFT or DINOv2 feature extractors.

## Scope

The default model is a dual RADIO feature model:

- fine branch: RADIO intermediate feature map
- coarse branch: RADIO final or configured intermediate feature map
- fusion: MATCHA-style bidirectional attention between fine and coarse tokens
- outputs: coarse descriptors, fine descriptors, heatmap logits, 65-bin RGB keypoint logits, 65-bin per-cell offset logits, and 64-bin pair fine logits

Single RADIO feature input remains a compatibility path, but the main training and cache defaults use `radio_dual`.

## Supervision

Positive correspondence supervision must come only from clean geometry:

- render/query depth maps
- render/query camera poses
- camera intrinsics
- 3DGS-rendered alpha/depth visibility checks
- bidirectional projection roundtrip checks

The current matcher output must not be used as positive supervision. Hard mined negatives may be used for confidence or ranking losses, but they are not descriptor positives.

Each geometry sample supervises all MATCHA heads in the same optimization step:

- dual-softmax descriptor loss on the same positive correspondence set
- query/render 65-bin offset classification
- render and optional query 64-bin fine coordinate classification
- dense heatmap/reliability loss from correspondence confidence
- optional RGB keypoint distillation as detector auxiliary only

## Architecture

`RadioDualAttentionFusionJointModel` is the main implementation. Its default fusion mode becomes `matcha_original`:

- `forward_fuse_feature()` returns distinct normalized coarse and fine descriptor maps.
- `forward_feature_map()` uses the fine descriptor map for geometric matching, offset prediction, heatmap prediction, and fine pair heads.
- Row-level adapter losses are bypassed for `radio_dual_attention` when full feature maps and cell indices are available.

The legacy residual adapter remains only for old checkpoints and row-only tests.

## Data Flow

1. Extract RADIO dual features for query and rendered candidate.
2. Render query depth from GT pose and render candidate RGB/depth/alpha from candidate pose using the 3DGS renderer.
3. Build `MatchaCoarseSupervision` from depth, pose, alpha, depth-edge, and roundtrip checks.
4. Build index-only joint samples that store feature maps plus geometry cell indices.
5. Train from full maps so all heads see the same correspondence batch.

## Testing

Tests must cover:

- `radio_dual_attention` total loss uses full-map descriptors rather than row adapter descriptors.
- default fusion mode is `matcha_original` for config and streaming CLI.
- geometry supervision metadata identifies depth/pose or 3DGS geometry sources and never matcher-derived positives.
- existing cache, streaming manifest, save/load, and projection paths remain compatible.
