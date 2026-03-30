# RADIO-GS: Foundation Feature Gaussian Splatting

Distill [RADIO](https://github.com/NVlabs/RADIO) (C-RADIOv4-H, 1280d) features into 3D Gaussian Splatting for multi-task novel view synthesis.

## Features

- **Two architectures**: Explicit per-Gaussian (64d) and Hybrid DCFF-style (16d latent + hash grid)
- **HCD Codec**: Hierarchical Compression-Decompression (1280d → 64d → 1280d, 20× compression)
- **FeatSharp-3D**: Multi-view consistency feature sharpening
- **Three downstream tasks**: Depth estimation, semantic segmentation, text grounding
- **RADIO adaptor compatibility**: Decoded features work with SigLIP2/SAM3 adaptors

## Quick Start

### 1. Extract RADIO features (offline, once per scene)
```bash
python radio_gs/scripts/extract_radio_features.py \
    --scene room_0 \
    --image_dir dataset/room_0/Sequence_1/rgb/ \
    --output_dir output/radio_features/room_0/ \
    --batch_size 4
```

### 2. Train feature field
```bash
# Architecture A: Explicit per-Gaussian
python radio_gs/scripts/train_feature_field.py \
    --config radio_gs/configs/replica_explicit.yaml

# Architecture B: Hybrid DCFF-style
python radio_gs/scripts/train_feature_field.py \
    --config radio_gs/configs/replica_hybrid.yaml
```

### 3. Evaluate downstream tasks
```bash
python radio_gs/scripts/eval_downstream.py \
    --config radio_gs/configs/replica_explicit.yaml \
    --checkpoint output/radio_gs/replica_explicit/checkpoints/best.pth \
    --tasks depth segmentation grounding feature_quality
```

### 4. Visualize features
```bash
python radio_gs/scripts/visualize_features.py \
    --config radio_gs/configs/replica_explicit.yaml \
    --checkpoint output/radio_gs/replica_explicit/checkpoints/best.pth \
    --num_views 20
```

## Architecture

```
Training Images → RADIO C-RADIOv4-H (frozen) → 1280d features (GT)
                                                       ↓ distillation
3DGS (frozen geometry) + Feature Embeddings → Render compact features (64d)
                                                       ↓
                            HCD Decoder → 1280d reconstructed features
                                                       ↓
                            Task Heads → Depth | Segmentation | Grounding
```

## Module Structure

```
radio_gs/
├── config.py                        # YAML config system
├── models/
│   ├── explicit_gaussian.py         # Arch A: per-Gaussian 64d features
│   ├── hybrid_gaussian.py           # Arch B: 16d latent + hash grid
│   ├── hcd_codec.py                 # Compression-decompression (1280d↔64d)
│   └── featsharp_3d.py              # Feature sharpening module
├── rendering/
│   ├── feature_renderer.py          # gsplat-based feature rendering
│   └── multiview_warp.py            # Depth-based multi-view warping
├── heads/
│   ├── depth_head.py                # Depth estimation (linear/MLP/DPT)
│   ├── segmentation_head.py         # Semantic segmentation
│   └── grounding_head.py            # Text grounding via SigLIP2
├── losses/
│   └── distillation_loss.py         # Feature distillation + consistency losses
├── scripts/
│   ├── extract_radio_features.py    # Offline feature extraction
│   ├── train_feature_field.py       # Main training script
│   ├── eval_downstream.py           # Multi-task evaluation
│   └── visualize_features.py        # PCA feature visualization
└── configs/
    ├── replica_explicit.yaml        # Replica + Explicit arch
    └── replica_hybrid.yaml          # Replica + Hybrid arch
```

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| RADIO over single teacher | Aggregates CLIP+DINOv2+SAM → multi-task from single distillation |
| Screen-space decode | O(H×W) not O(N_gaussians) — much faster |
| Dual-stream compression | Separate geometric (edges) and semantic (abstract) pathways |
| FeatSharp-3D | Counteracts alpha-blending smoothing in 3DGS |
