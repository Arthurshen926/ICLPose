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


class OnlineRadioTeacher(nn.Module):
    """Online RADIO inference + learned linear projection.

    Dual-scale features:
      - fine:   shallow block features (geometric edges/textures)
      - coarse: final block features (abstract semantics)

    Projections are 1×1 Conv (per-pixel linear), optionally
    initialized from PCA components for stable warm-start.
    """

    def __init__(self, target_dim=128, shallow_block=10,
                 radio_repo='/root/RADIO', pca_init_dir=None,
                 device='cuda'):
        super().__init__()
        self.target_dim = target_dim
        self.shallow_block = shallow_block
        self.device_str = device

        # Load frozen RADIO model
        print(f"[RadioTeacher] Loading RADIO ViT-H/16...")
        self.radio = torch.hub.load(
            radio_repo, 'radio_model',
            version='c-radio_v4-h', source='local', skip_validation=True,
        )
        self.radio.eval()
        for p in self.radio.parameters():
            p.requires_grad_(False)

        self.patch_size = self.radio.patch_size
        n_params = sum(p.numel() for p in self.radio.parameters()) / 1e6
        print(f"  RADIO: {n_params:.0f}M params, patch_size={self.patch_size}")

        # Learned linear projections (1×1 Conv = per-pixel linear)
        self.proj_fine = nn.Conv2d(1280, target_dim, 1, bias=False)
        self.proj_coarse = nn.Conv2d(1280, target_dim, 1, bias=False)

        # Initialize from PCA components if available
        if pca_init_dir and os.path.isdir(pca_init_dir):
            self._init_from_pca(pca_init_dir)
        else:
            # Orthogonal initialization
            self._init_orthogonal()

        # Hook for shallow features
        self._shallow_features = None
        self._register_hook()

        print(f"  Projections: 1280d → {target_dim}d (fine + coarse)")

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
        d_target = self.target_dim

        with torch.no_grad():
            # First d_pca dims from PCA, rest orthogonal
            d = min(d_pca, d_target)
            self.proj_fine.weight[:d, :, 0, 0] = geo_comp[:d]
            self.proj_coarse.weight[:d, :, 0, 0] = sem_comp[:d]
            if d_target > d:
                nn.init.orthogonal_(self.proj_fine.weight[d:, :, 0, 0])
                nn.init.orthogonal_(self.proj_coarse.weight[d:, :, 0, 0])

        print(f"  [RadioTeacher] Initialized projections from PCA "
              f"({d_pca}d → {d_target}d, first {d} dims from PCA)")

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
        images = images.to(self.proj_fine.weight.device)

        Hp = tH // self.patch_size
        Wp = tW // self.patch_size

        self._shallow_features = None
        with torch.autocast('cuda', dtype=torch.bfloat16):
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

    def project(self, fine_raw, coarse_raw):
        """Apply learned projections. HAS GRADIENTS for training.

        Args:
            fine_raw:   [B, 1280, Hp, Wp]
            coarse_raw: [B, 1280, Hp, Wp]

        Returns:
            fine_proj:   [B, target_dim, Hp, Wp]
            coarse_proj: [B, target_dim, Hp, Wp]
        """
        fine_proj = self.proj_fine(fine_raw)
        coarse_proj = self.proj_coarse(coarse_raw)
        return fine_proj, coarse_proj

    def get_projection_params(self):
        """Return projection parameters for optimizer."""
        return list(self.proj_fine.parameters()) + list(self.proj_coarse.parameters())

    @property
    def feature_resolution(self):
        """Expected feature resolution — must be set via set_image_size()."""
        if hasattr(self, '_feat_res'):
            return self._feat_res
        # Default fallback for 1920×1080 input
        return 68, 120  # (H, W)

    def set_image_size(self, img_h, img_w):
        """Set feature resolution based on actual image dimensions."""
        # RADIO ViT-H/16 produces features at (H/16, W/16)
        # Images must be padded to multiple of 16
        feat_h = (img_h + 15) // 16
        feat_w = (img_w + 15) // 16
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
