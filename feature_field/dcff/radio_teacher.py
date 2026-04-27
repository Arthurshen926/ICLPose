from __future__ import annotations

"""
Online RADIO Teacher for DCFF v5.

Loads a frozen RADIO ViT-H/16, extracts dual-scale features
(shallow block for fine-geometric, final for coarse-semantic),
and applies learned linear projections to reduce dimensionality.

The projections are trained jointly with the 3DGS decoders,
acting as task-specific dimensionality reduction (replaces PCA).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from feature_extract.utils.radio_loader import load_radio_model


class AdaptiveRadioCompressor(nn.Module):
    """Task-adaptive 1280d RADIO channel compressor.

    The encoder starts from a PCA-compatible linear projection and adds a
    zero-initialized residual adapter.  The decoder is used only for the
    auxiliary raw-RADIO reconstruction loss during training.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        output_dim: int = 64,
        hidden_dim: int = 256,
        pca_state: dict | None = None,
        sample_pixels: int = 1024,
        recon_chunk_pixels: int = 4096,
        recon_cos_weight: float = 1.0,
        recon_l1_weight: float = 0.25,
        min_spatial_std: float = 0.0,
        std_weight: float = 0.0,
        decorrelation_weight: float = 0.0,
        raw_highpass_kernel: int = 0,
        adapter_highpass_kernel: int = 0,
        normalize_output: bool = True,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.sample_pixels = int(sample_pixels)
        self.recon_chunk_pixels = max(1, int(recon_chunk_pixels))
        self.recon_cos_weight = float(recon_cos_weight)
        self.recon_l1_weight = float(recon_l1_weight)
        self.min_spatial_std = float(min_spatial_std)
        self.std_weight = float(std_weight)
        self.decorrelation_weight = float(decorrelation_weight)
        self.raw_highpass_kernel = int(raw_highpass_kernel)
        self.adapter_highpass_kernel = int(adapter_highpass_kernel)
        self.normalize_output = bool(normalize_output)

        self.base_proj = nn.Conv2d(self.input_dim, self.output_dim, 1, bias=True)
        self.res_adapter = nn.Sequential(
            nn.Conv2d(self.input_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, self.output_dim, 1, bias=False),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(self.output_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, self.input_dim, 1),
        )

        self._init_base_projection(pca_state)
        nn.init.zeros_(self.res_adapter[-1].weight)

    def _init_base_projection(self, pca_state: dict | None):
        with torch.no_grad():
            nn.init.orthogonal_(self.base_proj.weight[:, :, 0, 0])
            nn.init.zeros_(self.base_proj.bias)

            if not pca_state:
                return

            components = pca_state.get("components")
            mean = pca_state.get("mean")
            if components is None or mean is None:
                return

            components = components.float()
            mean = mean.float()
            if components.ndim != 2 or components.shape[1] != self.input_dim:
                return
            if mean.numel() != self.input_dim:
                return

            d = min(self.output_dim, components.shape[0])
            self.base_proj.weight[:d, :, 0, 0].copy_(components[:d])
            self.base_proj.bias[:d].copy_(-(components[:d] @ mean))

    def forward(self, raw: torch.Tensor) -> torch.Tensor:
        base_raw = self._preprocess_raw(raw)
        if self.adapter_highpass_kernel > 1:
            adapter_raw = self._highpass_raw(raw, self.adapter_highpass_kernel)
        else:
            adapter_raw = base_raw
        z = self.base_proj(base_raw) + self.res_adapter(adapter_raw)
        if self.normalize_output:
            z = F.normalize(z, p=2, dim=1)
        return z

    def _preprocess_raw(self, raw: torch.Tensor) -> torch.Tensor:
        return self._highpass_raw(raw, self.raw_highpass_kernel)

    def _highpass_raw(self, raw: torch.Tensor, kernel: int) -> torch.Tensor:
        if kernel <= 1:
            return raw
        if kernel % 2 == 0:
            kernel += 1
        pad = kernel // 2
        low = F.avg_pool2d(F.pad(raw, (pad, pad, pad, pad), mode="replicate"), kernel, stride=1)
        return raw - low

    def reconstruction_loss(self, raw: torch.Tensor, z: torch.Tensor) -> dict:
        """Sampled raw-space reconstruction loss for the learned bottleneck."""
        raw = self._preprocess_raw(raw)
        total_pixels = raw.shape[0] * raw.shape[2] * raw.shape[3]
        k = min(max(1, self.sample_pixels * raw.shape[0]), total_pixels)

        raw_flat = raw.permute(0, 2, 3, 1).reshape(total_pixels, self.input_dim)
        z_flat = z.permute(0, 2, 3, 1).reshape(total_pixels, self.output_dim)

        if k < total_pixels:
            idx = torch.randperm(total_pixels, device=raw.device)[:k]
            raw_flat = raw_flat[idx]
            z_flat = z_flat[idx]

        cos_accum = raw_flat.new_tensor(0.0)
        l1_accum = raw_flat.new_tensor(0.0)
        for start in range(0, k, self.recon_chunk_pixels):
            end = min(start + self.recon_chunk_pixels, k)
            n = end - start
            raw_sample = raw_flat[start:end].t().reshape(1, self.input_dim, n, 1)
            z_sample = z_flat[start:end].t().reshape(1, self.output_dim, n, 1)
            recon = self.decoder(z_sample)
            weight = n / k
            cos_accum = cos_accum + weight * F.cosine_similarity(recon, raw_sample, dim=1).mean()
            l1_accum = l1_accum + weight * F.l1_loss(
                F.normalize(recon, p=2, dim=1),
                F.normalize(raw_sample, p=2, dim=1),
            )

        cos = 1.0 - cos_accum
        l1 = l1_accum
        std = self._spatial_std_loss(z)
        decorrelation = self._decorrelation_loss(z)
        total = (
            self.recon_cos_weight * cos
            + self.recon_l1_weight * l1
            + self.std_weight * std
            + self.decorrelation_weight * decorrelation
        )
        return {"cos": cos, "l1": l1, "std": std, "decorrelation": decorrelation, "total": total}

    def _spatial_std_loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.min_spatial_std <= 0 or self.std_weight <= 0:
            return z.new_tensor(0.0)
        z_flat = z.flatten(2)
        std = z_flat.float().std(dim=-1, unbiased=False)
        return F.relu(self.min_spatial_std - std).mean()

    def _decorrelation_loss(self, z: torch.Tensor) -> torch.Tensor:
        if self.decorrelation_weight <= 0:
            return z.new_tensor(0.0)
        b, c, _, _ = z.shape
        if c <= 1:
            return z.new_tensor(0.0)
        x = z.permute(0, 2, 3, 1).reshape(-1, c).float()
        x = x - x.mean(dim=0, keepdim=True)
        x = x / x.std(dim=0, keepdim=True, unbiased=False).clamp(min=1e-6)
        corr = (x.T @ x) / max(1, x.shape[0])
        off_diag = corr - torch.eye(c, device=z.device, dtype=corr.dtype)
        return off_diag.square().mean()


class OnlineRadioTeacher(nn.Module):
    """Online RADIO inference + learned linear projection.

    Dual-scale features:
      - fine:   shallow block features (geometric edges/textures)
      - coarse: final block features (abstract semantics)

    Projections are 1×1 Conv (per-pixel linear), optionally
    initialized from PCA components for stable warm-start.
    """

    def __init__(self, target_dim=128, shallow_block=10,
                 radio_repo='feature_extract/checkpoints/RADIO', pca_init_dir=None,
                 device='cuda', fine_dim=None, coarse_dim=None,
                 bottleneck=False, compress_hidden_dim=256,
                 sample_pixels=1024, recon_chunk_pixels=4096,
                 recon_cos_weight=1.0,
                 recon_l1_weight=0.25, min_spatial_std=0.0,
                 std_weight=0.0, decorrelation_weight=0.0,
                 fine_raw_highpass_kernel=0, coarse_raw_highpass_kernel=0,
                 fine_adapter_highpass_kernel=0, coarse_adapter_highpass_kernel=0,
                 normalize_output=True):
        super().__init__()
        self.target_dim = target_dim
        self.fine_dim = int(fine_dim if fine_dim is not None else target_dim)
        self.coarse_dim = int(coarse_dim if coarse_dim is not None else target_dim)
        self.bottleneck = bool(bottleneck)
        self.shallow_block = shallow_block
        self.device_str = device
        self.intermediate_aggregation = 'dense'
        self.intermediate_norm_alpha_scheme = 'post-alpha'

        # Load frozen RADIO model
        print(f"[RadioTeacher] Loading RADIO ViT-H/16...")
        self.radio = load_radio_model(version='c-radio_v4-h', radio_repo=radio_repo)
        self.radio.eval()
        for p in self.radio.parameters():
            p.requires_grad_(False)

        self.patch_size = self.radio.patch_size
        n_params = sum(p.numel() for p in self.radio.parameters()) / 1e6
        print(f"  RADIO: {n_params:.0f}M params, patch_size={self.patch_size}")

        if self.bottleneck:
            pca_fine = self._load_pca_state(pca_init_dir, 'fine_geo_pca.pt')
            pca_coarse = self._load_pca_state(pca_init_dir, 'coarse_sem_pca.pt')
            self.compress_fine = AdaptiveRadioCompressor(
                input_dim=1280,
                output_dim=self.fine_dim,
                hidden_dim=compress_hidden_dim,
                pca_state=pca_fine,
                sample_pixels=sample_pixels,
                recon_chunk_pixels=recon_chunk_pixels,
                recon_cos_weight=recon_cos_weight,
                recon_l1_weight=recon_l1_weight,
                min_spatial_std=min_spatial_std,
                std_weight=std_weight,
                decorrelation_weight=decorrelation_weight,
                raw_highpass_kernel=fine_raw_highpass_kernel,
                adapter_highpass_kernel=fine_adapter_highpass_kernel,
                normalize_output=normalize_output,
            )
            self.compress_coarse = AdaptiveRadioCompressor(
                input_dim=1280,
                output_dim=self.coarse_dim,
                hidden_dim=compress_hidden_dim,
                pca_state=pca_coarse,
                sample_pixels=sample_pixels,
                recon_chunk_pixels=recon_chunk_pixels,
                recon_cos_weight=recon_cos_weight,
                recon_l1_weight=recon_l1_weight,
                min_spatial_std=min_spatial_std,
                std_weight=std_weight,
                decorrelation_weight=decorrelation_weight,
                raw_highpass_kernel=coarse_raw_highpass_kernel,
                adapter_highpass_kernel=coarse_adapter_highpass_kernel,
                normalize_output=normalize_output,
            )
        else:
            # Learned linear projections (1×1 Conv = per-pixel linear)
            self.proj_fine = nn.Conv2d(1280, self.fine_dim, 1, bias=False)
            self.proj_coarse = nn.Conv2d(1280, self.coarse_dim, 1, bias=False)

            # Initialize from PCA components if available
            if pca_init_dir and os.path.isdir(pca_init_dir):
                self._init_from_pca(pca_init_dir)
            else:
                self._init_orthogonal()

        self._shallow_features = None
        self._use_forward_intermediates = hasattr(self.radio, 'forward_intermediates')
        if self._use_forward_intermediates:
            print(
                f"  Using RADIO forward_intermediates() for shallow block {self.shallow_block} "
                f"(aggregation={self.intermediate_aggregation})"
            )
        else:
            self._register_hook()

        if self.bottleneck:
            print(
                f"  Adaptive bottlenecks: fine 1280d→{self.fine_dim}d, "
                f"coarse 1280d→{self.coarse_dim}d"
            )
        else:
            print(
                f"  Projections: fine 1280d→{self.fine_dim}d, "
                f"coarse 1280d→{self.coarse_dim}d"
            )

    def _load_pca_state(self, pca_dir, filename):
        if not pca_dir or not os.path.isdir(pca_dir):
            return None
        path = os.path.join(pca_dir, filename)
        if not os.path.exists(path):
            print(f"  [RadioTeacher] PCA init missing: {path}")
            return None
        return torch.load(path, map_location='cpu')

    def _init_orthogonal(self):
        """Random orthogonal initialization for projections."""
        for proj in [self.proj_fine, self.proj_coarse]:
            nn.init.orthogonal_(proj.weight[:, :, 0, 0])

    def _init_from_pca(self, pca_dir):
        """Initialize projections from saved PCA components."""
        geo_path = os.path.join(pca_dir, 'fine_geo_pca.pt')
        sem_path = os.path.join(pca_dir, 'coarse_sem_pca.pt')

        if not os.path.exists(geo_path) or not os.path.exists(sem_path):
            print("  [RadioTeacher] PCA files not found, using orthogonal init")
            self._init_orthogonal()
            return

        geo_pca = torch.load(geo_path, map_location='cpu')
        sem_pca = torch.load(sem_path, map_location='cpu')

        geo_comp = geo_pca['components']  # [64, 1280]
        sem_comp = sem_pca['components']  # [64, 1280]

        d_pca = geo_comp.shape[0]  # 64
        d_fine = self.fine_dim
        d_coarse = self.coarse_dim

        with torch.no_grad():
            # First d_pca dims from PCA, rest orthogonal
            d_geo = min(d_pca, d_fine)
            d_sem = min(d_pca, d_coarse)
            self.proj_fine.weight[:d_geo, :, 0, 0] = geo_comp[:d_geo]
            self.proj_coarse.weight[:d_sem, :, 0, 0] = sem_comp[:d_sem]
            if d_fine > d_geo:
                nn.init.orthogonal_(self.proj_fine.weight[d_geo:, :, 0, 0])
            if d_coarse > d_sem:
                nn.init.orthogonal_(self.proj_coarse.weight[d_sem:, :, 0, 0])

        print(f"  [RadioTeacher] Initialized projections from PCA "
              f"(fine first {d_geo}/{d_fine}, coarse first {d_sem}/{d_coarse})")

    def _projection_device(self):
        return self.get_projection_params()[0].device

    def _register_hook(self):
        """Register forward hook on shallow transformer block."""
        blocks = None
        for attr_path in ['model.blocks', 'blocks']:
            obj = self.radio
            try:
                for part in attr_path.split('.'):
                    obj = getattr(obj, part)
                if hasattr(obj, '__len__') and len(obj) > self.shallow_block:
                    blocks = obj
                    break
            except AttributeError:
                continue

        if blocks is None:
            raise RuntimeError(
                "Cannot find transformer blocks in RADIO model. "
                "Shallow feature extraction will not work."
            )

        n_blocks = len(blocks)
        idx = min(self.shallow_block, n_blocks - 1)
        print(f"  Hooking shallow features at block {idx}/{n_blocks-1}")

        def hook_fn(module, input, output):
            if isinstance(output, tuple):
                self._shallow_features = output[0].detach()
            else:
                self._shallow_features = output.detach()

        blocks[idx].register_forward_hook(hook_fn)

    @torch.no_grad()
    def extract_raw(self, images):
        """Extract raw 1280d dual-scale features.

        Args:
            images: [B, 3, H, W] float32 in [0, 1], any resolution

        Returns:
            fine_raw:   [B, 1280, Hp, Wp] shallow block features
            coarse_raw: [B, 1280, Hp, Wp] final block features
        """
        B, _, H, W = images.shape
        nearest = self.radio.get_nearest_supported_resolution(H, W)
        tH, tW = nearest.height, nearest.width
        if tH != H or tW != W:
            images = F.interpolate(images, (tH, tW),
                                   mode='bilinear', align_corners=False)
        images = images.to(self._projection_device())

        Hp = tH // self.patch_size
        Wp = tW // self.patch_size

        self._shallow_features = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
            if self._use_forward_intermediates:
                final, intermediates = self.radio.forward_intermediates(
                    images,
                    indices=[self.shallow_block],
                    return_prefix_tokens=False,
                    norm=False,
                    stop_early=False,
                    output_fmt='NCHW',
                    intermediates_only=False,
                    aggregation=self.intermediate_aggregation,
                    norm_alpha_scheme=self.intermediate_norm_alpha_scheme,
                )
                deep_features = final.features if hasattr(final, 'features') else final[1]
                deep_feat = deep_features.float()
                if intermediates:
                    fine_feat = intermediates[0].float()
                else:
                    fine_feat = deep_feat
            else:
                summary, deep_feat = self.radio(images, feature_fmt='NCHW')

                deep_feat = deep_feat.float()  # [B, 1280, Hp, Wp]

                # Process shallow features from hook
                shallow = self._shallow_features.float()  # [B, N_tokens, D]
                if shallow.shape[1] == Hp * Wp + 1:
                    shallow = shallow[:, 1:]  # remove CLS
                elif shallow.shape[1] != Hp * Wp:
                    shallow = shallow[:, -Hp * Wp:]
                fine_feat = shallow.permute(0, 2, 1).reshape(B, -1, Hp, Wp)

        return fine_feat, deep_feat

    def project(self, fine_raw, coarse_raw, return_loss=False):
        """Apply learned projections. HAS GRADIENTS for training.

        Args:
            fine_raw:   [B, 1280, Hp, Wp]
            coarse_raw: [B, 1280, Hp, Wp]

        Returns:
            fine_proj:   [B, target_dim, Hp, Wp]
            coarse_proj: [B, target_dim, Hp, Wp]
        """
        if self.bottleneck:
            fine_proj = self.compress_fine(fine_raw)
            coarse_proj = self.compress_coarse(coarse_raw)
            if not return_loss:
                return fine_proj, coarse_proj
            fine_loss = self.compress_fine.reconstruction_loss(fine_raw, fine_proj)
            coarse_loss = self.compress_coarse.reconstruction_loss(coarse_raw, coarse_proj)
            total = fine_loss["total"] + coarse_loss["total"]
            losses = {
                "compress_fine_cos": fine_loss["cos"],
                "compress_fine_l1": fine_loss["l1"],
                "compress_fine_std": fine_loss["std"],
                "compress_fine_decorrelation": fine_loss["decorrelation"],
                "compress_coarse_cos": coarse_loss["cos"],
                "compress_coarse_l1": coarse_loss["l1"],
                "compress_coarse_std": coarse_loss["std"],
                "compress_coarse_decorrelation": coarse_loss["decorrelation"],
                "compress_total": total,
            }
            return fine_proj, coarse_proj, losses

        fine_proj = self.proj_fine(fine_raw)
        coarse_proj = self.proj_coarse(coarse_raw)
        if return_loss:
            return fine_proj, coarse_proj, {}
        return fine_proj, coarse_proj

    def get_projection_params(self):
        """Return projection parameters for optimizer."""
        if self.bottleneck:
            return list(self.compress_fine.parameters()) + list(self.compress_coarse.parameters())
        return list(self.proj_fine.parameters()) + list(self.proj_coarse.parameters())

    def get_projection_state(self):
        if self.bottleneck:
            return {
                "mode": "bottleneck",
                "fine": self.compress_fine.state_dict(),
                "coarse": self.compress_coarse.state_dict(),
            }
        return {
            "mode": "linear",
            "fine": self.proj_fine.state_dict(),
            "coarse": self.proj_coarse.state_dict(),
        }

    def load_projection_state(self, state):
        if not isinstance(state, dict):
            return
        if self.bottleneck:
            if "fine" in state:
                self.compress_fine.load_state_dict(state["fine"], strict=False)
            if "coarse" in state:
                self.compress_coarse.load_state_dict(state["coarse"], strict=False)
        else:
            if "fine" in state:
                self.proj_fine.load_state_dict(state["fine"], strict=False)
            if "coarse" in state:
                self.proj_coarse.load_state_dict(state["coarse"], strict=False)

    @property
    def feature_resolution(self):
        """Expected feature resolution — must be set via set_image_size()."""
        if hasattr(self, '_feat_res'):
            return self._feat_res
        # Default fallback for 1920×1080 input
        return 68, 120  # (H, W)

    def infer_radio_input_and_feature_size(self, img_h, img_w):
        """Infer the actual RADIO input size and token feature resolution.

        RADIO snaps arbitrary image sizes to a supported resolution inside
        ``extract_raw``.  Training/render code needs to use that same snapped
        token grid, otherwise high-res online targets are silently compared
        against a differently sized rendered feature map.
        """
        img_h = int(img_h)
        img_w = int(img_w)
        if hasattr(self.radio, 'get_nearest_supported_resolution'):
            nearest = self.radio.get_nearest_supported_resolution(img_h, img_w)
            radio_h = int(getattr(nearest, 'height', nearest[0] if isinstance(nearest, (tuple, list)) else img_h))
            radio_w = int(getattr(nearest, 'width', nearest[1] if isinstance(nearest, (tuple, list)) else img_w))
        else:
            radio_h = ((img_h + self.patch_size - 1) // self.patch_size) * self.patch_size
            radio_w = ((img_w + self.patch_size - 1) // self.patch_size) * self.patch_size
        feat_h = max(1, radio_h // self.patch_size)
        feat_w = max(1, radio_w // self.patch_size)
        return (radio_h, radio_w), (feat_h, feat_w)

    def set_image_size(self, img_h, img_w):
        """Set feature resolution based on actual image dimensions."""
        (radio_h, radio_w), (feat_h, feat_w) = self.infer_radio_input_and_feature_size(img_h, img_w)
        self._radio_input_size = (radio_h, radio_w)
        self._feat_res = (feat_h, feat_w)


class CachedFeatureTeacher:
    """Loads pre-extracted PCA features from disk.

    Compatible interface with OnlineRadioTeacher for the training loop.
    """

    def __init__(self, feature_dir, max_gpu_cache=64):
        self.feature_dir = feature_dir
        self._max_gpu = max_gpu_cache
        self._gpu_cache = {}
        self._gpu_order = []
        self._cpu_cache = {}

        geo_dir = os.path.join(feature_dir, 'fine_geo')
        sem_dir = os.path.join(feature_dir, 'coarse_sem')

        if not os.path.isdir(geo_dir) or not os.path.isdir(sem_dir):
            raise FileNotFoundError(
                f"Expected fine_geo/ and coarse_sem/ in {feature_dir}")

        self._geo_paths = {}
        self._sem_paths = {}
        self.frame_ids = set()

        for fn in sorted(os.listdir(geo_dir)):
            if fn.endswith('.pt') and fn.startswith('rgb_'):
                fid = int(fn.replace('.pt', '').split('_')[1])
                self.frame_ids.add(fid)
                self._geo_paths[fid] = os.path.join(geo_dir, fn)

        for fn in sorted(os.listdir(sem_dir)):
            if fn.endswith('.pt') and fn.startswith('rgb_'):
                fid = int(fn.replace('.pt', '').split('_')[1])
                self._sem_paths[fid] = os.path.join(sem_dir, fn)

        sample = torch.load(self._geo_paths[min(self.frame_ids)], map_location='cpu')
        self.feature_dim, self.feat_h, self.feat_w = sample.shape
        print(f"  [CachedTeacher] {len(self.frame_ids)} frames, "
              f"{self.feature_dim}d @ {self.feat_w}×{self.feat_h}")

    def get(self, fid):
        """Get (geo, sem) tensors on GPU for frame fid."""
        if fid in self._gpu_cache:
            return self._gpu_cache[fid]

        if fid not in self._cpu_cache:
            geo = torch.load(self._geo_paths[fid], map_location='cpu').float()
            sem = torch.load(self._sem_paths[fid], map_location='cpu').float()
            self._cpu_cache[fid] = (geo, sem)

        geo_cpu, sem_cpu = self._cpu_cache[fid]
        result = (geo_cpu.cuda(), sem_cpu.cuda())
        self._gpu_cache[fid] = result
        self._gpu_order.append(fid)

        while len(self._gpu_cache) > self._max_gpu:
            old = self._gpu_order.pop(0)
            if old in self._gpu_cache:
                del self._gpu_cache[old]

        return result
