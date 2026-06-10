"""Lightweight MATCHA-style attention fusion for VFM maps.

This module keeps the existing numpy local-fusion helper for old experiments,
and also exposes a small torch implementation that mirrors the shape contract
of MATCHA's ``AttentionFusionNet``: coarse/fine feature fusion, decoder blocks,
65-bin keypoint logits, and a 64-bin fine matcher.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:  # Keep numpy-only callers importable when torch is unavailable.
    import torch
    from torch import nn
    from torch.nn import functional as F
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]


def _require_torch() -> None:
    if torch is None or nn is None or F is None:  # pragma: no cover
        raise RuntimeError("Torch is required for MATCHA-style attention fusion")


if nn is not None:

    class _BasicConvLayer(nn.Module):
        """Conv2d -> BatchNorm2d -> ReLU, matching MATCHA's BasicLayer."""

        def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, *, padding: int = 1) -> None:
            super().__init__()
            self.layer = nn.Sequential(
                nn.Conv2d(int(in_channels), int(out_channels), int(kernel_size), padding=int(padding), bias=False),
                nn.BatchNorm2d(int(out_channels), affine=False),
                nn.ReLU(inplace=True),
            )

        def forward(self, image: "torch.Tensor") -> "torch.Tensor":
            return self.layer(image)


    class MatchaLightKeypointHead(nn.Module):
        """MATCHA-style 65-bin detector over 8x8 RGB cells."""

        def __init__(self, window_size: int = 8, stem_channels: int = 16, hidden_channels: int = 64) -> None:
            super().__init__()
            self.window_size = int(window_size)
            self.keypoint_encoder = _BasicConvLayer(3, int(stem_channels), 3, padding=1)
            self.keypoint_head = nn.Sequential(
                _BasicConvLayer(int(stem_channels) * int(window_size) ** 2, int(hidden_channels), 1, padding=0),
                _BasicConvLayer(int(hidden_channels), int(hidden_channels), 1, padding=0),
                _BasicConvLayer(int(hidden_channels), int(hidden_channels), 1, padding=0),
                nn.Conv2d(int(hidden_channels), 65, 1),
            )

        def forward(self, image: "torch.Tensor") -> "torch.Tensor":
            if image.ndim != 4 or int(image.shape[1]) != 3:
                raise ValueError("image must have shape (B, 3, H, W)")
            window = int(self.window_size)
            if int(image.shape[2]) % window != 0 or int(image.shape[3]) % window != 0:
                raise ValueError("image height and width must be divisible by window_size")
            features = self.keypoint_encoder(image)
            batch, channels, height, width = features.shape
            features = (
                features.unfold(2, window, window)
                .unfold(3, window, window)
                .reshape(batch, channels, height // window, width // window, window**2)
            )
            features = features.permute(0, 1, 4, 2, 3).reshape(batch, channels * window**2, height // window, width // window)
            return self.keypoint_head(features)


    class MatchaLightFineMatcher(nn.Module):
        """Original-MATCHA-style 64-way coordinate classifier for a matched pair."""

        def __init__(self, descriptor_dim: int, hidden_dim: int = 512, output_bins: int = 64) -> None:
            super().__init__()
            hidden = int(hidden_dim)
            self.net = nn.Sequential(
                nn.Linear(int(descriptor_dim) * 2, hidden),
                nn.BatchNorm1d(hidden, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, hidden),
                nn.BatchNorm1d(hidden, affine=False),
                nn.ReLU(inplace=True),
                nn.Linear(hidden, int(output_bins)),
            )

        def forward(self, query_descriptors: "torch.Tensor", render_descriptors: "torch.Tensor") -> "torch.Tensor":
            if query_descriptors.shape != render_descriptors.shape:
                raise ValueError("query_descriptors and render_descriptors must have matching shape")
            return self.net(torch.cat([query_descriptors, render_descriptors], dim=-1))


    class MatchaLightAttentionBlock(nn.Module):
        """Small joint decoder block with self attention, cross attention, and MLPs."""

        def __init__(self, hidden_dim: int, num_heads: int = 4, mlp_ratio: int = 4) -> None:
            super().__init__()
            hidden = int(hidden_dim)
            heads = max(1, int(num_heads))
            while hidden % heads != 0 and heads > 1:
                heads -= 1
            self.fine_self_norm = nn.LayerNorm(hidden)
            self.coarse_self_norm = nn.LayerNorm(hidden)
            self.fine_cross_norm = nn.LayerNorm(hidden)
            self.coarse_cross_norm = nn.LayerNorm(hidden)
            self.fine_self = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.coarse_self = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.fine_cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.coarse_cross = nn.MultiheadAttention(hidden, heads, batch_first=True)
            self.fine_mlp = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * int(mlp_ratio)),
                nn.GELU(),
                nn.Linear(hidden * int(mlp_ratio), hidden),
            )
            self.coarse_mlp = nn.Sequential(
                nn.LayerNorm(hidden),
                nn.Linear(hidden, hidden * int(mlp_ratio)),
                nn.GELU(),
                nn.Linear(hidden * int(mlp_ratio), hidden),
            )

        def forward(self, fine_tokens: "torch.Tensor", coarse_tokens: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            fine_update, _ = self.fine_self(
                self.fine_self_norm(fine_tokens),
                self.fine_self_norm(fine_tokens),
                self.fine_self_norm(fine_tokens),
                need_weights=False,
            )
            coarse_update, _ = self.coarse_self(
                self.coarse_self_norm(coarse_tokens),
                self.coarse_self_norm(coarse_tokens),
                self.coarse_self_norm(coarse_tokens),
                need_weights=False,
            )
            fine_tokens = fine_tokens + fine_update
            coarse_tokens = coarse_tokens + coarse_update
            fine_update, _ = self.fine_cross(
                self.fine_cross_norm(fine_tokens),
                self.coarse_cross_norm(coarse_tokens),
                self.coarse_cross_norm(coarse_tokens),
                need_weights=False,
            )
            coarse_update, _ = self.coarse_cross(
                self.coarse_cross_norm(coarse_tokens),
                self.fine_cross_norm(fine_tokens),
                self.fine_cross_norm(fine_tokens),
                need_weights=False,
            )
            fine_tokens = fine_tokens + fine_update
            coarse_tokens = coarse_tokens + coarse_update
            fine_tokens = fine_tokens + self.fine_mlp(fine_tokens)
            coarse_tokens = coarse_tokens + self.coarse_mlp(coarse_tokens)
            return fine_tokens, coarse_tokens


    class MatchaLightDecoderRefinementBlock(nn.Module):
        """Convolutional refinement block for decoder context fusion."""

        def __init__(self, in_channels: int, hidden_channels: int) -> None:
            super().__init__()
            self.proj = _BasicConvLayer(int(in_channels), int(hidden_channels), 3, padding=1)
            self.refine = _BasicConvLayer(int(hidden_channels), int(hidden_channels), 3, padding=1)
            self.skip = (
                nn.Identity()
                if int(in_channels) == int(hidden_channels)
                else nn.Conv2d(int(in_channels), int(hidden_channels), 1)
            )

        def forward(self, tensor: "torch.Tensor") -> "torch.Tensor":
            return F.relu(self.skip(tensor) + self.refine(self.proj(tensor)), inplace=True)


    @dataclass(frozen=True)
    class AttentionFusionOutput:
        coarse_descriptors: "torch.Tensor"
        fine_descriptors: "torch.Tensor"
        heatmap_logits: "torch.Tensor"
        offset_logits: "torch.Tensor | None" = None
        keypoint_logits: "torch.Tensor | None" = None
        coarse_context: "torch.Tensor | None" = None
        fine_context: "torch.Tensor | None" = None

        @property
        def descriptor_map(self) -> "torch.Tensor":
            return self.fine_descriptors

        def as_matcha_tuple(self) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            return self.coarse_descriptors, self.fine_descriptors, self.heatmap_logits

        def as_joint_training_tuple(self) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            if self.offset_logits is None:
                raise ValueError("offset_logits are required for joint training tuple output")
            return self.fine_descriptors, self.heatmap_logits, self.offset_logits


    class AttentionFusionNet(nn.Module):
        """Lightweight adaptation of MATCHA's ``AttentionFusionNet``.

        ``forward_feature_map`` expects a channel-concatenated map in the
        existing project convention: ``[fine_feature, coarse_feature]``.
        """

        def __init__(
            self,
            *,
            coarse_input_dim: int,
            fine_input_dim: int,
            output_dim: int = 128,
            hidden_dim: int = 256,
            decoder_depth: int = 2,
            num_heads: int = 4,
            patch_size: int = 2,
            upsample_mode: str = "bilinear",
            refinement_blocks: int = 1,
            fine_matcher_hidden_dim: int | None = None,
        ) -> None:
            super().__init__()
            self.coarse_input_dim = int(coarse_input_dim)
            self.fine_input_dim = int(fine_input_dim)
            self.output_dim = int(output_dim)
            self.hidden_dim = int(hidden_dim)
            self.decoder_depth = int(decoder_depth)
            self.num_heads = int(num_heads)
            self.patch_size = int(patch_size)
            self.upsample_mode = str(upsample_mode)
            if self.upsample_mode not in {"bilinear", "pixel_shuffle"}:
                raise ValueError("upsample_mode must be 'bilinear' or 'pixel_shuffle'")
            if self.patch_size <= 0:
                raise ValueError("patch_size must be positive")
            if self.upsample_mode == "pixel_shuffle" and self.hidden_dim % (self.patch_size**2) != 0:
                raise ValueError("hidden_dim must be divisible by patch_size^2 for pixel_shuffle")
            context_channels = self.hidden_dim if self.upsample_mode == "bilinear" else self.hidden_dim // (self.patch_size**2)
            self.fine_proj = nn.Conv2d(self.fine_input_dim, self.hidden_dim, 1)
            self.coarse_proj = nn.Conv2d(self.coarse_input_dim, self.hidden_dim, 1)
            self.decoder_blocks = nn.ModuleList(
                [MatchaLightAttentionBlock(self.hidden_dim, self.num_heads) for _ in range(self.decoder_depth)]
            )
            self.fine_norm = nn.LayerNorm(self.hidden_dim)
            self.coarse_norm = nn.LayerNorm(self.hidden_dim)

            def fusion_stack(in_channels: int) -> nn.Sequential:
                hidden_channels = max(self.output_dim, self.hidden_dim)
                layers: list[nn.Module] = [
                    MatchaLightDecoderRefinementBlock(in_channels, hidden_channels)
                ]
                for _ in range(max(0, int(refinement_blocks) - 1)):
                    layers.append(MatchaLightDecoderRefinementBlock(hidden_channels, hidden_channels))
                layers.append(nn.Conv2d(hidden_channels, self.output_dim, 1))
                return nn.Sequential(*layers)

            self.fusion_c = fusion_stack(self.coarse_input_dim + context_channels)
            self.fusion_f = fusion_stack(self.fine_input_dim + context_channels)
            self.heatmap_head = nn.Sequential(
                _BasicConvLayer(self.output_dim, max(64, self.output_dim), 3, padding=1),
                _BasicConvLayer(max(64, self.output_dim), max(64, self.output_dim), 1, padding=0),
                nn.Conv2d(max(64, self.output_dim), 1, 1),
            )
            self.offset_head = nn.Sequential(
                _BasicConvLayer(self.output_dim, max(64, self.output_dim), 1, padding=0),
                nn.Conv2d(max(64, self.output_dim), 65, 1),
            )
            self.keypoint_head = MatchaLightKeypointHead()
            self.fine_matcher = MatchaLightFineMatcher(
                descriptor_dim=self.output_dim,
                hidden_dim=int(fine_matcher_hidden_dim or max(64, self.output_dim * 2)),
            )

        @property
        def input_dim(self) -> int:
            return int(self.fine_input_dim + self.coarse_input_dim)

        def _split_feature_map(self, feature_maps: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor"]:
            if feature_maps.ndim != 4 or int(feature_maps.shape[1]) != self.input_dim:
                raise ValueError("feature_maps must have shape (B, fine_input_dim + coarse_input_dim, H, W)")
            fine = feature_maps[:, : self.fine_input_dim]
            coarse = feature_maps[:, self.fine_input_dim :]
            return coarse, fine

        def _restore_context(self, tokens: "torch.Tensor", batch: int, height: int, width: int, out_height: int, out_width: int) -> "torch.Tensor":
            context = tokens.transpose(1, 2).reshape(batch, self.hidden_dim, height, width)
            if self.upsample_mode == "pixel_shuffle" and self.patch_size > 1:
                context = F.pixel_shuffle(context, self.patch_size)
                if int(context.shape[2]) >= out_height and int(context.shape[3]) >= out_width:
                    return context[:, :, :out_height, :out_width]
            if tuple(context.shape[-2:]) != (out_height, out_width):
                context = F.interpolate(context, size=(out_height, out_width), mode="bilinear", align_corners=False)
            return context

        def _attention_contexts(self, coarse: "torch.Tensor", fine: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            if fine.ndim != 4 or coarse.ndim != 4:
                raise ValueError("coarse and fine feature maps must have shape (B, C, H, W)")
            batch, _channels, height, width = fine.shape
            if tuple(coarse.shape[-2:]) != (height, width):
                coarse = F.interpolate(coarse, size=(height, width), mode="bilinear", align_corners=False)
            patch = int(self.patch_size)
            if patch > 1:
                fine_input = F.avg_pool2d(fine, kernel_size=patch, stride=patch, ceil_mode=True)
                coarse_input = F.avg_pool2d(coarse, kernel_size=patch, stride=patch, ceil_mode=True)
            else:
                fine_input = fine
                coarse_input = coarse
            _batch, _c, token_height, token_width = fine_input.shape
            fine_tokens = self.fine_proj(fine_input).flatten(2).transpose(1, 2)
            coarse_tokens = self.coarse_proj(coarse_input).flatten(2).transpose(1, 2)
            for block in self.decoder_blocks:
                fine_tokens, coarse_tokens = block(fine_tokens, coarse_tokens)
            fine_tokens = self.fine_norm(fine_tokens)
            coarse_tokens = self.coarse_norm(coarse_tokens)
            fine_context = self._restore_context(fine_tokens, batch, token_height, token_width, height, width)
            coarse_context = self._restore_context(coarse_tokens, batch, token_height, token_width, height, width)
            return coarse, fine, coarse_context, fine_context

        def _forward_output(
            self,
            coarse: "torch.Tensor",
            fine: "torch.Tensor",
            image: "torch.Tensor | None" = None,
        ) -> AttentionFusionOutput:
            coarse, fine, coarse_context, fine_context = self._attention_contexts(coarse, fine)
            coarse_desc = F.normalize(self.fusion_c(torch.cat([coarse, coarse_context], dim=1)), dim=1, eps=1e-8)
            fine_desc = F.normalize(self.fusion_f(torch.cat([fine, fine_context], dim=1)), dim=1, eps=1e-8)
            heatmap_logits = self.heatmap_head(fine_desc)
            offset_logits = self.offset_head(fine_desc)
            keypoint_logits = self.forward_rgb_keypoints(image) if image is not None else None
            return AttentionFusionOutput(
                coarse_descriptors=coarse_desc,
                fine_descriptors=fine_desc,
                heatmap_logits=heatmap_logits,
                offset_logits=offset_logits,
                keypoint_logits=keypoint_logits,
                coarse_context=coarse_context,
                fine_context=fine_context,
            )

        def forward_fuse_feature(self, feat_c: "torch.Tensor", feat_f: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            return self._forward_output(feat_c, feat_f).as_matcha_tuple()

        def forward_feature_map(self, feature_maps: "torch.Tensor") -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            coarse, fine = self._split_feature_map(feature_maps)
            return self._forward_output(coarse, fine).as_joint_training_tuple()

        def forward_rgb_keypoints(self, image: "torch.Tensor") -> "torch.Tensor":
            return self.keypoint_head(image)

        def pair_fine_logits(self, query_descriptors: "torch.Tensor", render_descriptors: "torch.Tensor") -> "torch.Tensor":
            return self.fine_matcher(query_descriptors, render_descriptors)

        def query_pair_fine_logits(self, query_descriptors: "torch.Tensor", render_descriptors: "torch.Tensor") -> "torch.Tensor":
            return self.fine_matcher(query_descriptors, render_descriptors)

        def forward(
            self,
            *,
            feature_maps: "torch.Tensor | None" = None,
            image: "torch.Tensor | None" = None,
            feat_c: "torch.Tensor | None" = None,
            feat_f: "torch.Tensor | None" = None,
        ) -> AttentionFusionOutput:
            if feature_maps is not None:
                coarse, fine = self._split_feature_map(feature_maps)
            elif feat_c is not None and feat_f is not None:
                coarse, fine = feat_c, feat_f
            else:
                raise ValueError("provide either feature_maps or both feat_c and feat_f")
            return self._forward_output(coarse, fine, image)

else:  # pragma: no cover
    MatchaLightKeypointHead = None  # type: ignore[assignment]
    MatchaLightFineMatcher = None  # type: ignore[assignment]
    MatchaLightAttentionBlock = None  # type: ignore[assignment]
    MatchaLightDecoderRefinementBlock = None  # type: ignore[assignment]
    AttentionFusionOutput = None  # type: ignore[assignment]
    AttentionFusionNet = None  # type: ignore[assignment]


def local_attention_fuse_feature_map(
    feature_map: np.ndarray,
    *,
    radius: int = 1,
    temperature: float = 5.0,
    alpha: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    """Fuse each cell with a local self-attention context window.

    The operation is intentionally candidate-independent: it only looks at a
    feature map's own local neighborhood, so it can be applied identically to
    query and rendered map features before descriptor adaptation.
    """

    if int(radius) < 0:
        raise ValueError("radius must be non-negative")
    if float(temperature) <= 0.0:
        raise ValueError("temperature must be positive")
    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    fmap = np.asarray(feature_map, dtype=np.float32)
    if fmap.ndim != 3:
        raise ValueError("feature_map must have shape (C, H, W)")
    if int(radius) == 0:
        flat = fmap.reshape(fmap.shape[0], -1)
        flat = flat / np.maximum(np.linalg.norm(flat, axis=0, keepdims=True), 1e-8)
        return flat.reshape(fmap.shape).astype(np.float32, copy=False)
    try:
        import torch
        from torch.nn import functional as F
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Torch is required for local attention feature fusion") from exc

    torch_device = torch.device(device if torch.cuda.is_available() or not str(device).startswith("cuda") else "cpu")
    x = torch.as_tensor(fmap[None], dtype=torch.float32, device=torch_device)
    channels, height, width = int(fmap.shape[0]), int(fmap.shape[1]), int(fmap.shape[2])
    x_norm = F.normalize(x, dim=1, eps=1e-8)
    kernel = 2 * int(radius) + 1
    padded = F.pad(x_norm, (int(radius), int(radius), int(radius), int(radius)), mode="replicate")
    patches = F.unfold(padded, kernel_size=kernel).reshape(1, channels, kernel * kernel, height * width)[0]
    centers = x_norm.reshape(channels, height * width)
    scores = torch.sum(patches * centers[:, None, :], dim=0) * float(temperature)
    weights = torch.softmax(scores, dim=0)
    context = torch.sum(patches * weights[None, :, :], dim=1)
    fused = (1.0 - float(alpha)) * centers + float(alpha) * context
    fused = F.normalize(fused, dim=0, eps=1e-8).reshape(channels, height, width)
    return fused.detach().cpu().numpy().astype(np.float32, copy=False)


def maybe_fuse_feature_map(
    feature_map: np.ndarray,
    *,
    mode: str = "none",
    radius: int = 1,
    temperature: float = 5.0,
    alpha: float = 0.5,
    device: str = "cpu",
) -> np.ndarray:
    mode = str(mode)
    if mode == "none":
        return feature_map
    if mode == "local_attention":
        return local_attention_fuse_feature_map(
            feature_map,
            radius=int(radius),
            temperature=float(temperature),
            alpha=float(alpha),
            device=str(device),
        )
    raise ValueError("feature fusion mode must be 'none' or 'local_attention'")
