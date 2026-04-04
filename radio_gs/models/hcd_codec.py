"""
Hierarchical Compression-Decompression (HCD) Codec for RADIO-GS.

Compresses 1280d RADIO (C-RADIOv4-H) spatial features into a compact
per-Gaussian representation (32–64d) via dual-stream encoding, and
reconstructs back to 1280d for downstream task heads.

All layers use 1×1 convolutions to preserve strict pixel alignment
with 3D Gaussian splatting rasterization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def _conv1x1_gnorm_gelu(in_ch: int, out_ch: int, num_groups: int = 32) -> nn.Sequential:
    """1×1 Conv → GroupNorm → GELU block."""
    groups = min(num_groups, out_ch)
    while out_ch % groups != 0:
        groups -= 1
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
        nn.GroupNorm(groups, out_ch),
        nn.GELU(),
    )


class HCDEncoder(nn.Module):
    """Dual-stream hierarchical encoder: 1280d RADIO features → compact bottleneck.

    Geometric stream captures low-level spatial structure, semantic stream
    captures high-level category information.  Both are concatenated to form
    the final bottleneck representation.

    Args:
        input_dim:      Input feature dimension (default 1280 for C-RADIOv4-H).
        bottleneck_dim: Output dimension. Must be even in dual-stream mode.
        dual_stream:    If True, use separate geometric + semantic streams
                        each producing bottleneck_dim // 2 channels.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        bottleneck_dim: int = 64,
        dual_stream: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim
        self.dual_stream = dual_stream

        if dual_stream:
            if bottleneck_dim % 2 != 0:
                raise ValueError(
                    f"bottleneck_dim must be even in dual-stream mode, got {bottleneck_dim}"
                )
            stream_dim = bottleneck_dim // 2
            self.geometric_stream = nn.Sequential(
                _conv1x1_gnorm_gelu(input_dim, 512),
                _conv1x1_gnorm_gelu(512, 256),
                nn.Conv2d(256, stream_dim, kernel_size=1),
            )
            self.semantic_stream = nn.Sequential(
                _conv1x1_gnorm_gelu(input_dim, 512),
                _conv1x1_gnorm_gelu(512, 256),
                nn.Conv2d(256, stream_dim, kernel_size=1),
            )
        else:
            self.stream = nn.Sequential(
                _conv1x1_gnorm_gelu(input_dim, 512),
                _conv1x1_gnorm_gelu(512, 256),
                nn.Conv2d(256, bottleneck_dim, kernel_size=1),
            )

        # Zero-initialized residual shortcut for optional PCA warm-start.
        # Adds no effect until init_from_pca() is called.
        self.pca_shortcut = nn.Conv2d(input_dim, bottleneck_dim, kernel_size=1, bias=False)
        nn.init.zeros_(self.pca_shortcut.weight)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[HCDEncoder] {input_dim}d → {bottleneck_dim}d "
              f"({'dual' if dual_stream else 'single'}-stream) | "
              f"{n_params / 1e6:.2f}M params")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode RADIO features to compact representation.

        Args:
            x: (B, input_dim, H, W) RADIO spatial features.

        Returns:
            (B, bottleneck_dim, H, W) compact features.
        """
        if self.dual_stream:
            z_geo = self.geometric_stream(x)
            z_sem = self.semantic_stream(x)
            return torch.cat([z_geo, z_sem], dim=1) + self.pca_shortcut(x)
        return self.stream(x) + self.pca_shortcut(x)


class HCDDecoder(nn.Module):
    """Multi-layer 1×1 Conv MLP decoder with residual shortcut.

    Reconstructs 1280d RADIO features from the compact bottleneck.
    A linear residual projection is added to the MLP output for
    gradient stability, followed by LayerNorm for adaptor compatibility.

    Args:
        bottleneck_dim: Input compact dimension.
        output_dim:     Reconstructed feature dimension (default 1280).
        symmetric:      If True, use a deeper decoder that mirrors the encoder
                        capacity (bottleneck → 256 → 512 → 512 → output_dim).
    """

    def __init__(
        self,
        bottleneck_dim: int = 64,
        output_dim: int = 1280,
        symmetric: bool = False,
    ) -> None:
        super().__init__()
        self.bottleneck_dim = bottleneck_dim
        self.output_dim = output_dim

        if symmetric:
            self.mlp = nn.Sequential(
                _conv1x1_gnorm_gelu(bottleneck_dim, 256),
                _conv1x1_gnorm_gelu(256, 512),
                _conv1x1_gnorm_gelu(512, 512),
                nn.Conv2d(512, output_dim, kernel_size=1),
            )
        else:
            self.mlp = nn.Sequential(
                _conv1x1_gnorm_gelu(bottleneck_dim, 256),
                _conv1x1_gnorm_gelu(256, 512),
                nn.Conv2d(512, output_dim, kernel_size=1),
            )
        self.residual = nn.Conv2d(bottleneck_dim, output_dim, kernel_size=1)
        self.norm = nn.GroupNorm(1, output_dim)  # equivalent to LayerNorm over C

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[HCDDecoder] {bottleneck_dim}d → {output_dim}d "
              f"({'symmetric' if symmetric else 'standard'}) | "
              f"{n_params / 1e6:.2f}M params")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode compact features back to full RADIO dimension.

        Args:
            z: (B, bottleneck_dim, H, W) compact features.

        Returns:
            (B, output_dim, H, W) reconstructed features.
        """
        return self.norm(self.mlp(z) + self.residual(z))


class HCDCodec(nn.Module):
    """Hierarchical Compression-Decompression codec for RADIO-GS.

    Wraps :class:`HCDEncoder` and :class:`HCDDecoder` into a single module
    with helpers for loss computation, PCA warm-start, and selective freezing.

    Args:
        input_dim:      RADIO feature dimension (1280).
        bottleneck_dim: Compact representation dimension (32–64).
        dual_stream:    Use dual geometric + semantic encoder streams.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        bottleneck_dim: int = 64,
        dual_stream: bool = True,
        symmetric_decoder: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        self.encoder = HCDEncoder(input_dim, bottleneck_dim, dual_stream)
        self.decoder = HCDDecoder(bottleneck_dim, input_dim, symmetric=symmetric_decoder)

        total = sum(p.numel() for p in self.parameters())
        print(f"[HCDCodec] total {total / 1e6:.2f}M params | "
              f"compression ratio {self.compression_ratio:.1f}×")

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Compress RADIO features.

        Args:
            x: (B, input_dim, H, W).

        Returns:
            (B, bottleneck_dim, H, W).
        """
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Reconstruct RADIO features from compact code.

        Args:
            z: (B, bottleneck_dim, H, W).

        Returns:
            (B, input_dim, H, W).
        """
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode → decode pass.

        Args:
            x: (B, input_dim, H, W).

        Returns:
            (B, input_dim, H, W) reconstructed features.
        """
        return self.decode(self.encode(x))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def compute_reconstruction_loss(
        self,
        x_original: torch.Tensor,
        x_reconstructed: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """L2 + cosine reconstruction loss.

        Args:
            x_original:      (B, C, H, W) ground-truth RADIO features.
            x_reconstructed: (B, C, H, W) decoded features.

        Returns:
            Dict with keys ``l2``, ``cosine``, and ``total``.
        """
        l2 = F.mse_loss(x_reconstructed, x_original)

        orig_flat = x_original.flatten(2)       # (B, C, H*W)
        recon_flat = x_reconstructed.flatten(2)
        cosine = 1.0 - F.cosine_similarity(orig_flat, recon_flat, dim=1).mean()

        return {"l2": l2, "cosine": cosine, "total": l2 + cosine}

    # ------------------------------------------------------------------
    # Freezing helpers
    # ------------------------------------------------------------------

    def freeze_encoder(self) -> None:
        """Freeze all encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False

    def freeze_decoder(self) -> None:
        """Freeze all decoder parameters."""
        for p in self.decoder.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------
    # PCA warm-start
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_from_pca(self, pca_components: torch.Tensor) -> None:
        """Warm-start encoder's residual PCA shortcut from pre-computed components.

        The shortcut directly projects input_dim → bottleneck_dim and is added
        to the learned MLP output.  Before this call the shortcut is zero.

        Args:
            pca_components: (bottleneck_dim, input_dim) PCA basis vectors.
        """
        if pca_components.shape != (self.bottleneck_dim, self.input_dim):
            raise ValueError(
                f"Expected PCA shape ({self.bottleneck_dim}, {self.input_dim}), "
                f"got {tuple(pca_components.shape)}"
            )

        self.encoder.pca_shortcut.weight.copy_(
            pca_components.unsqueeze(-1).unsqueeze(-1)
        )
        print(f"[HCDCodec] Initialized encoder PCA shortcut "
              f"({self.bottleneck_dim} components)")

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def compression_ratio(self) -> float:
        """Dimensionality compression ratio (input_dim / bottleneck_dim)."""
        return self.input_dim / self.bottleneck_dim
