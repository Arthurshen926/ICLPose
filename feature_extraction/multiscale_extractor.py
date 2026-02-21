"""
多尺度特征提取器 (Multi-Scale Feature Extractor)

提取三层特征金字塔 + DINO CLS Token:
  - Fine   : SD s3 (640d) + DINO Patch (768d) → concat → 1408d, 35×46
  - Mid    : SD s4 (1280d), 原生分辨率 ~15×20
  - Coarse : SD s5 (1280d), 原生分辨率 ~8×10
  - Global : DINO CLS Token (768d), 1D 向量

用途: 为 v2-iterative-routing 架构提供多尺度特征，
      后续由 PCA/AutoEncoder 分别降维后嵌入 3DGS。
"""
import os
import math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Optional, Union

os.environ.setdefault("HF_HOME", "/home/yons/.cache/huggingface")
os.environ.setdefault("TORCH_HOME", "/home/yons/.cache/torch")


@dataclass
class MultiScaleFeatures:
    """多尺度特征输出容器"""
    fine: torch.Tensor       # [1408, fH, fW]  SD s3 + DINO patch
    mid: torch.Tensor        # [1280, mH, mW]  SD s4
    coarse: torch.Tensor     # [1280, cH, cW]  SD s5
    cls_token: torch.Tensor  # [768]           DINO CLS token
    # 元数据
    fine_hw: tuple = None    # (fH, fW) - DINO patch grid, 即定位坐标系的基准
    mid_hw: tuple = None     # (mH, mW)
    coarse_hw: tuple = None  # (cH, cW)


class MultiScaleFeatureExtractor:
    """
    多尺度 DINO + Stable Diffusion 特征提取器

    与 FusedFeatureExtractor 的区别:
      - 不使用 AggregationNetwork 将 4 层特征揉成单一的 768d
      - 保留 SD 的 s3/s4/s5 各自独立输出
      - DINO 的 patch token 仅与 SD s3 拼接 (Fine 层)
      - 额外提取 DINO CLS token 用于全局检索
    """

    def __init__(self,
                 device: str = 'cuda',
                 sd_image_size: int = 480):
        self.device = device
        self.sd_image_size = sd_image_size
        self._load_models()
        torch.set_grad_enabled(False)

    def _load_models(self):
        print("[MultiScaleFeatureExtractor] 加载模型...")

        # SD 模型
        from .extractor_sd import load_model
        self.sd_model, self.sd_aug = load_model(
            diffusion_ver='v1-5',
            image_size=self.sd_image_size,
            num_timesteps=50,
            block_indices=[2, 5, 8, 11]
        )
        print("  ✓ Stable Diffusion 模型加载完成")

        # DINO 模型
        from .extractor_dino import ViTExtractor
        self.extractor_vit = ViTExtractor(
            'dinov2_vitb14', stride=14, device=self.device
        )
        print("  ✓ DINOv2 模型加载完成")
        print("[MultiScaleFeatureExtractor] 模型加载完成\n")

    def extract(self, image_input: Union[str, Path, Image.Image]) -> MultiScaleFeatures:
        """
        提取多尺度特征

        Args:
            image_input: 图像路径或 PIL.Image

        Returns:
            MultiScaleFeatures 包含 fine/mid/coarse/cls_token
        """
        # 加载图像
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert('RGB')
        else:
            img = image_input

        orig_w, orig_h = img.size

        # ── 计算 SD 输入尺寸 (短边对齐 sd_image_size) ──
        scale = self.sd_image_size / min(orig_w, orig_h)
        sd_w = int(round(orig_w * scale))
        sd_h = int(round(orig_h * scale))

        # ═══════════════════════════════════════════════════
        #  SD 特征提取 (反射填充, 参考 ControlNet)
        # ═══════════════════════════════════════════════════
        sd_align_h = int(math.ceil(sd_h / 64) * 64)
        sd_align_w = int(math.ceil(sd_w / 64) * 64)
        img_sd_base = img.resize((sd_w, sd_h), Image.Resampling.LANCZOS)

        if sd_align_h != sd_h or sd_align_w != sd_w:
            import torchvision.transforms.functional as TF
            img_t = TF.to_tensor(img_sd_base)
            pad_b = sd_align_h - sd_h
            pad_r = sd_align_w - sd_w
            img_t = F.pad(img_t.unsqueeze(0),
                          (0, pad_r, 0, pad_b), mode='reflect').squeeze(0)
            img_sd_input = Image.fromarray(
                (img_t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype('uint8')
            )
        else:
            img_sd_input = img_sd_base

        from .extractor_sd import process_features_and_mask
        feats_sd = process_features_and_mask(
            self.sd_model, self.sd_aug, img_sd_input,
            mask=False, raw=True
        )
        del feats_sd['s2']  # 不使用 s2

        # 裁回有效区域 (去掉反射填充对应的 feature 行/列)
        for key in ['s3', 's4', 's5']:
            if key not in feats_sd:
                continue
            _, _, fh, fw = feats_sd[key].shape
            factor_h = sd_align_h // fh
            factor_w = sd_align_w // fw
            valid_fh = sd_h // factor_h
            valid_fw = sd_w // factor_w
            if valid_fh < fh or valid_fw < fw:
                feats_sd[key] = feats_sd[key][:, :, :valid_fh, :valid_fw]

        # ═══════════════════════════════════════════════════
        #  DINO 特征提取 (Patch Tokens + CLS Token)
        # ═══════════════════════════════════════════════════
        dino_w = int(math.ceil(sd_w / 14) * 14)
        dino_h = int(math.ceil(sd_h / 14) * 14)
        img_dino_input = img.resize((dino_w, dino_h), Image.Resampling.BILINEAR)
        img_batch = self.extractor_vit.preprocess_pil(img_dino_input)

        tokens_h, tokens_w = dino_h // 14, dino_w // 14

        # Patch tokens (不含 CLS)
        feats_dino_patch = self.extractor_vit.extract_descriptors(
            img_batch.to(self.device), layer=11, facet='token',
            include_cls=False
        )
        # [B, 1, num_patches, 768] → [1, 768, tokens_h, tokens_w]
        feats_dino_patch = feats_dino_patch.permute(0, 1, 3, 2).reshape(
            1, -1, tokens_h, tokens_w
        )

        # CLS token: 使用 include_cls=True, 取第 0 个 token
        feats_dino_with_cls = self.extractor_vit.extract_descriptors(
            img_batch.to(self.device), layer=11, facet='token',
            include_cls=True
        )
        # [B, 1, num_patches+1, 768] → 取 token 0 即 CLS
        cls_token = feats_dino_with_cls[:, :, 0, :]  # [B, 1, 768]
        cls_token = cls_token.squeeze()               # [768]

        # ═══════════════════════════════════════════════════
        #  构建三层特征金字塔
        # ═══════════════════════════════════════════════════

        # --- Fine (SD s3 + DINO Patch) ---
        # 将 SD s3 上采样到 DINO 网格 (tokens_h × tokens_w)
        sd_s3_aligned = F.interpolate(
            feats_sd['s3'],
            size=(tokens_h, tokens_w),
            mode='bilinear',
            align_corners=False
        )
        # 通道拼接: [1, 640, H, W] + [1, 768, H, W] → [1, 1408, H, W]
        fine_feat = torch.cat([sd_s3_aligned, feats_dino_patch], dim=1)
        fine_feat = fine_feat.squeeze(0).cpu()  # [1408, tokens_h, tokens_w]

        # --- Mid (SD s4 原生分辨率) ---
        mid_feat = feats_sd['s4'].squeeze(0).cpu()  # [1280, mH, mW]

        # --- Coarse (SD s5 原生分辨率) ---
        coarse_feat = feats_sd['s5'].squeeze(0).cpu()  # [1280, cH, cW]

        # 归一化 CLS token
        cls_token = F.normalize(cls_token.float(), p=2, dim=-1).cpu()

        return MultiScaleFeatures(
            fine=fine_feat,
            mid=mid_feat,
            coarse=coarse_feat,
            cls_token=cls_token,
            fine_hw=(tokens_h, tokens_w),
            mid_hw=(mid_feat.shape[1], mid_feat.shape[2]),
            coarse_hw=(coarse_feat.shape[1], coarse_feat.shape[2]),
        )


def create_multiscale_extractor(device='cuda'):
    """便捷工厂函数"""
    return MultiScaleFeatureExtractor(device=device)
