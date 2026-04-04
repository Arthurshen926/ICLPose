# Source Generated with Decompyle++
# File: radio_teacher.cpython-39.pyc (Python 3.9)

'''
Online RADIO Teacher for DCFF v5.

Loads a frozen RADIO ViT-H/16, extracts dual-scale features
(shallow block for fine-geometric, final for coarse-semantic),
and applies learned linear projections to reduce dimensionality.

The projections are trained jointly with the 3DGS decoders,
acting as task-specific dimensionality reduction (replaces PCA).
'''
import os
import torch
from torch.nn import nn
import torch.nn.functional
F = functional
nn

class OnlineRadioTeacher(nn.Module):
    '''Online RADIO inference + learned linear projection.

    Dual-scale features:
      - fine:   shallow block features (geometric edges/textures)
      - coarse: final block features (abstract semantics)

    Projections are 1×1 Conv (per-pixel linear), optionally
    initialized from PCA components for stable warm-start.
    '''
    
    def __init__(self = None, target_dim = None, shallow_block = None, radio_repo = None, pca_init_dir = None, device = None):
        super().__init__()
        self.target_dim = target_dim
        self.shallow_block = shallow_block
        self.device_str = device
        print('[RadioTeacher] Loading RADIO ViT-H/16...')
        self.radio = torch.hub.load(radio_repo, 'radio_model', 'c-radio_v4-h', 'local', True, **('version', 'source', 'skip_validation'))
        self.radio.eval()
        for p in self.radio.parameters():
            p.requires_grad_(False)
        self.patch_size = self.radio.patch_size
        n_params = sum((lambda .0: for p in .0:
p.numel())(self.radio.parameters())) / 1e+06
        print(f'''  RADIO: {n_params:.0f}M params, patch_size={self.patch_size}''')
        self.proj_fine = nn.Conv2d(1280, target_dim, 1, False, **('bias',))
        self.proj_coarse = nn.Conv2d(1280, target_dim, 1, False, **('bias',))
        if pca_init_dir and os.path.isdir(pca_init_dir):
            self._init_from_pca(pca_init_dir)
        else:
            self._init_orthogonal()
        self._shallow_features = None
        self._register_hook()
        print(f'''  Projections: 1280d → {target_dim}d (fine + coarse)''')

    
    def _init_orthogonal(self):
        '''Random orthogonal initialization for projections.'''
        for proj in (self.proj_fine, self.proj_coarse):
            nn.init.orthogonal_(proj.weight[(:, :, 0, 0)])

    
    def _init_from_pca(self, pca_dir):
        '''Initialize projections from saved PCA components.'''
        geo_path = os.path.join(pca_dir, 'fine_geo_pca.pt')
        sem_path = os.path.join(pca_dir, 'coarse_sem_pca.pt')
        if not os.path.exists(geo_path) or os.path.exists(sem_path):
            print('  [RadioTeacher] PCA files not found, using orthogonal init')
            self._init_orthogonal()
            return None
        geo_pca = None.load(geo_path, 'cpu', **('map_location',))
        sem_pca = torch.load(sem_path, 'cpu', **('map_location',))
        geo_comp = geo_pca['components']
        sem_comp = sem_pca['components']
        d_pca = geo_comp.shape[0]
        d_target = self.target_dim
        with torch.no_grad():
            d = min(d_pca, d_target)
            self.proj_fine.weight[(:d, :, 0, 0)] = geo_comp[:d]
            self.proj_coarse.weight[(:d, :, 0, 0)] = sem_comp[:d]
            if d_target > d:
                nn.init.orthogonal_(self.proj_fine.weight[(d:, :, 0, 0)])
                nn.init.orthogonal_(self.proj_coarse.weight[(d:, :, 0, 0)])
            None(None, None, None)
    # WARNING: Decompyle incomplete

    
    def _register_hook(self):
        '''Register forward hook on shallow transformer block.'''
        blocks = None
    # WARNING: Decompyle incomplete

    
    def extract_raw(self, images):
        '''Extract raw 1280d dual-scale features.

        Args:
            images: [B, 3, H, W] float32 in [0, 1], any resolution

        Returns:
            fine_raw:   [B, 1280, Hp, Wp] shallow block features
            coarse_raw: [B, 1280, Hp, Wp] final block features
        '''
        (B, _, H, W) = images.shape
        nearest = self.radio.get_nearest_supported_resolution(H, W)
        tH = nearest.height
        tW = nearest.width
        if tH != H or tW != W:
            images = F.interpolate(images, (tH, tW), 'bilinear', False, **('mode', 'align_corners'))
        images = images.to(self.proj_fine.weight.device)
        Hp = tH // self.patch_size
        Wp = tW // self.patch_size
        self._shallow_features = None
        with torch.autocast('cuda', torch.bfloat16, **('dtype',)):
            (summary, deep_feat) = self.radio(images, 'NCHW', **('feature_fmt',))
            None(None, None, None)
    # WARNING: Decompyle incomplete

    extract_raw = torch.no_grad()(extract_raw)
    
    def project(self, fine_raw, coarse_raw):
        '''Apply learned projections. HAS GRADIENTS for training.

        Args:
            fine_raw:   [B, 1280, Hp, Wp]
            coarse_raw: [B, 1280, Hp, Wp]

        Returns:
            fine_proj:   [B, target_dim, Hp, Wp]
            coarse_proj: [B, target_dim, Hp, Wp]
        '''
        fine_proj = self.proj_fine(fine_raw)
        coarse_proj = self.proj_coarse(coarse_raw)
        return (fine_proj, coarse_proj)

    
    def get_projection_params(self):
        '''Return projection parameters for optimizer.'''
        return list(self.proj_fine.parameters()) + list(self.proj_coarse.parameters())

    
    def feature_resolution(self):
        '''Expected feature resolution — must be set via set_image_size().'''
        if hasattr(self, '_feat_res'):
            return self._feat_res

    feature_resolution = property(feature_resolution)
    
    def set_image_size(self, img_h, img_w):
        '''Set feature resolution based on actual image dimensions.'''
        feat_h = (img_h + 15) // 16
        feat_w = (img_w + 15) // 16
        self._feat_res = (feat_h, feat_w)

    __classcell__ = None


class CachedFeatureTeacher:
    '''Loads pre-extracted PCA features from disk.

    Compatible interface with OnlineRadioTeacher for the training loop.
    '''
    
    def __init__(self, feature_dir, max_gpu_cache = (64,)):
