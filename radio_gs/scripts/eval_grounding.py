"""Evaluate text grounding: RADIO features → SigLIP2 projection → cosine sim with text.

Pipeline:
  1. Load pre-computed SigLIP2 text embeddings (1536d) for object queries.
  2. Load standalone SigLIP2 feature projection (RADIO 1280d → 1536d).
  3. For each eval frame:
     a. Load GT RADIO 1280d features  →  project to SigLIP2 → heatmaps
     b. Render decoded 1280d features →  project to SigLIP2 → heatmaps
  4. Evaluate:
     - Heatmap correlation (GT vs rendered) per query
     - Zero-shot segmentation via argmax over queries → mIoU against GT semantic
     - Per-class grounding AP using GT semantic masks as ground truth
"""
import sys, argparse, cv2, torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from tqdm import tqdm
from timm.models.vision_transformer import Block

sys.path.insert(0, '.')
from radio_gs.config import load_config
from radio_gs.models.explicit_gaussian import ExplicitFeatureGaussian
from radio_gs.models.hcd_codec import HCDCodec
from radio_gs.models.featsharp_3d import FeatSharp3D
from radio_gs.models.screen_refiner import ScreenSpaceRefiner
from radio_gs.rendering.feature_renderer import FeatureFieldRenderer

device = torch.device("cuda")

# Standard Replica semantic class names (NYU-style) mapped to class IDs in room_0
REPLICA_CLASSES = {
    0: "undefined",
    11: "wall",
    12: "floor",
    13: "ceiling",
    20: "door",
    29: "table",
    31: "chair",
    40: "window",
    44: "picture",
    47: "cabinet",
    59: "cushion",
    60: "sofa",
    63: "bed",
    64: "curtain",
    65: "chest of drawers",
    76: "plant",
    80: "stool",
    92: "lamp",
    93: "shelf",
    97: "blanket",
    98: "mirror",
}


class SigLIP2FeatureProjection(nn.Module):
    """Standalone SigLIP2 feature projection from RADIO checkpoint."""

    def __init__(self):
        super().__init__()
        self.blocks = nn.Sequential(*[
            Block(1280, num_heads=16, init_values=1e-5)
            for _ in range(2)
        ])
        self.mlp_fc1 = nn.Linear(1280, 1520)
        self.mlp_final = nn.Sequential(
            nn.LayerNorm(1520),
            nn.GELU(),
            nn.Linear(1520, 1536),
        )

    def forward(self, x):
        """x: [B, N, 1280] -> [B, N, 1536]"""
        x = self.blocks(x)
        x = self.mlp_fc1(x)
        x = self.mlp_final(x)
        return x

    @classmethod
    def from_radio_checkpoint(cls, ckpt_path):
        chk = torch.load(ckpt_path, map_location="cpu")
        sd = chk["state_dict"]
        proj = cls()
        proj_sd = {}
        prefix = "_feature_projections.siglip2-g."
        for k, v in sd.items():
            if k.startswith(prefix):
                new_k = k[len(prefix):]
                if new_k.startswith("mlp.fc1"):
                    new_k = new_k.replace("mlp.fc1", "mlp_fc1")
                elif new_k.startswith("mlp.final"):
                    new_k = new_k.replace("mlp.final", "mlp_final")
                proj_sd[new_k] = v.float()
        proj.load_state_dict(proj_sd, strict=True)
        return proj


def load_model_and_render_pipeline(config_path, checkpoint_path):
    """Load trained RADIO-GS model and components."""
    config = load_config(config_path)

    model = ExplicitFeatureGaussian(latent_dim=getattr(config, "latent_dim", 64))
    ply_path = getattr(config, "ply_path", "")
    if ply_path:
        model.load_from_ply(ply_path)
    model = model.to(device).eval()

    codec = HCDCodec(
        input_dim=getattr(config, "radio_feature_dim", 1280),
        bottleneck_dim=getattr(config, "bottleneck_dim", 64),
    ).to(device).eval()

    fH = getattr(config, "feature_height", 30)
    fW = getattr(config, "feature_width", 40)
    renderer = FeatureFieldRenderer(
        image_height=fH, image_width=fW,
        fx=getattr(config, "fx", 320.0) * fW / getattr(config, "image_width", 640),
        fy=getattr(config, "fy", 320.0) * fH / getattr(config, "image_height", 480),
        cx=getattr(config, "cx", 319.5) * fW / getattr(config, "image_width", 640),
        cy=getattr(config, "cy", 239.5) * fH / getattr(config, "image_height", 480),
        max_channels_per_chunk=getattr(config, "max_channels_per_chunk", 32),
        use_2dgs=getattr(config, "use_2dgs", False),
    ).to(device)

    sharpener = FeatSharp3D(
        mode=getattr(config, "featsharp_mode", "analytical"),
        feature_dim=getattr(config, "latent_dim", 64),
        strength=getattr(config, "featsharp_strength", 0.3),
    ).to(device).eval()

    refiner = None
    rgb_guide = getattr(config, "refiner_rgb_guide", False)
    if getattr(config, "use_refiner", False):
        extra_ch = 3 if rgb_guide else 0
        refiner = ScreenSpaceRefiner(
            latent_dim=getattr(config, "latent_dim", 64),
            hidden_dim=getattr(config, "refiner_hidden_dim", 128),
            num_blocks=getattr(config, "refiner_num_blocks", 4),
            dropout=getattr(config, "refiner_dropout", 0.1),
            extra_channels=extra_ch,
        ).to(device).eval()

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    codec.load_state_dict(ckpt["codec_state_dict"], strict=False)
    if "sharpener_state_dict" in ckpt:
        sharpener.load_state_dict(ckpt["sharpener_state_dict"], strict=False)
    if refiner is not None and "refiner_state_dict" in ckpt:
        refiner.load_state_dict(ckpt["refiner_state_dict"], strict=False)

    return model, codec, renderer, sharpener, refiner, config


def render_1280d(model, codec, renderer, sharpener, refiner, viewmat,
                 rgb_guide=None):
    """Render a single frame's 1280d decoded features.
    
    Args:
        viewmat: [1, 4, 4] world-to-camera matrix.
        rgb_guide: [1, 3, H, W] RGB guide for refiner.
    """
    with torch.no_grad():
        result = renderer.render_features(model, viewmat.squeeze(0))
        latent = result["feature_map"].unsqueeze(0)  # [1, D, H, W]
        latent = sharpener(latent)
        if refiner is not None:
            latent = refiner(latent, guide=rgb_guide)
        decoded = codec.decode(latent)  # [1, 1280, H, W]
    return decoded


def project_to_siglip2(features_1280, proj_model):
    """Project [1, 1280, H, W] features to SigLIP2 space [1, 1536, H, W]."""
    B, C, H, W = features_1280.shape
    feat_flat = features_1280.reshape(B, C, H * W).permute(0, 2, 1)  # [B, HW, 1280]
    with torch.no_grad():
        siglip_feat = proj_model(feat_flat)  # [B, HW, 1536]
    siglip_feat = F.normalize(siglip_feat, dim=-1)
    return siglip_feat.permute(0, 2, 1).reshape(B, -1, H, W)  # [B, 1536, H, W]


def compute_heatmaps(visual_feat, text_emb):
    """Compute per-query cosine similarity heatmaps.

    Args:
        visual_feat: [1, D, H, W] normalized SigLIP2 visual features.
        text_emb: [N, D] normalized text embeddings.

    Returns:
        [N, H, W] similarity heatmaps.
    """
    _, D, H, W = visual_feat.shape
    vis_flat = visual_feat.squeeze(0).reshape(D, H * W)  # [D, HW]
    sim = text_emb @ vis_flat  # [N, HW]
    return sim.reshape(-1, H, W)


def load_semantic_gt(sem_dir, idx, target_size):
    """Load Replica semantic mask and resize to feature resolution."""
    path = Path(sem_dir) / f"semantic_class_{idx}.png"
    sem = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    sem = cv2.resize(sem, (target_size[1], target_size[0]),
                     interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(sem.astype(np.int64))


def evaluate_grounding(args):
    print("=" * 60)
    print("RADIO-GS Text Grounding Evaluation (SigLIP2)")
    print("=" * 60)

    # Load text embeddings
    text_data = torch.load(args.text_embeddings, map_location="cpu")
    all_queries = text_data["queries"]
    all_text_emb = text_data["embeddings"].to(device).half()  # [N_all, 1536]
    print(f"Loaded {len(all_queries)} text queries, dim={all_text_emb.shape[-1]}")

    # Filter to only queries that match Replica classes in this scene
    class_ids_in_scene = sorted(REPLICA_CLASSES.keys())
    class_names = [REPLICA_CLASSES[cid] for cid in class_ids_in_scene]

    # Map query names to indices in text_emb
    query_to_idx = {q: i for i, q in enumerate(all_queries)}
    active_queries = []
    active_text_emb = []
    active_class_ids = []
    for cid in class_ids_in_scene:
        name = REPLICA_CLASSES[cid]
        if name in query_to_idx:
            active_queries.append(name)
            active_text_emb.append(all_text_emb[query_to_idx[name]])
            active_class_ids.append(cid)
    active_text_emb = torch.stack(active_text_emb)  # [K, 1536]
    print(f"Active queries ({len(active_queries)}): {active_queries}")
    print(f"Active class IDs: {active_class_ids}")

    # Load SigLIP2 feature projection
    if args.projection_weights:
        proj = SigLIP2FeatureProjection()
        proj.load_state_dict(torch.load(args.projection_weights, map_location="cpu"))
    else:
        proj = SigLIP2FeatureProjection.from_radio_checkpoint(
            "/root/.cache/torch/hub/checkpoints/c-radio_v4-h_half.pth.tar"
        )
    proj = proj.to(device).half().eval()
    print("Loaded SigLIP2 feature projection")

    # Load RADIO-GS model
    model, codec, renderer, sharpener, refiner, config = \
        load_model_and_render_pipeline(args.config, args.checkpoint)
    fH = getattr(config, "feature_height", 30)
    fW = getattr(config, "feature_width", 40)
    rgb_guide_enabled = getattr(config, "refiner_rgb_guide", False)

    # Load poses (traj_w_c.txt: camera-to-world, need inverse for renderer)
    pose_file = args.pose_file
    if not pose_file:
        # Auto-detect from gt_features dir
        feat_parent = Path(args.gt_features).parent
        pose_file = str(feat_parent / "traj_w_c.txt")
    assert Path(pose_file).exists(), f"Pose file not found: {pose_file}"
    c2w = np.loadtxt(pose_file).reshape(-1, 4, 4).astype(np.float32)
    w2c = np.linalg.inv(c2w)
    n_frames = len(w2c)
    print(f"Loaded {n_frames} poses from {pose_file}")

    # Setup paths
    gt_feat_dir = Path(args.gt_features)
    sem_dir = Path(args.semantic_dir)
    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else None

    # Metrics accumulators
    heatmap_corrs = []      # per-frame mean correlation between GT and rendered heatmaps
    per_class_iou_gt = {q: [] for q in active_queries}
    per_class_iou_rend = {q: [] for q in active_queries}
    per_class_ap_gt = {q: [] for q in active_queries}
    per_class_ap_rend = {q: [] for q in active_queries}
    argmax_correct_gt = 0
    argmax_correct_rend = 0
    argmax_total = 0

    vis_frames = []  # Save some frames for visualization

    for frame_idx in tqdm(range(n_frames), desc="Evaluating"):
        # Load GT features
        gt_path = gt_feat_dir / f"rgb_{frame_idx}.pt"
        if not gt_path.exists():
            continue
        gt_feat = torch.load(gt_path, map_location=device).unsqueeze(0)  # [1, 1280, H, W]

        # Render decoded features
        viewmat = torch.from_numpy(w2c[frame_idx:frame_idx+1]).float().to(device)
        rgb_guide = None
        if refiner is not None and rgb_guide_enabled and rgb_dir is not None:
            img = cv2.imread(str(rgb_dir / f"rgb_{frame_idx}.png"))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (fW, fH))
            rgb_guide = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
        rend_feat = render_1280d(model, codec, renderer, sharpener, refiner,
                                 viewmat, rgb_guide)

        # Project to SigLIP2 space
        gt_siglip = project_to_siglip2(gt_feat.half(), proj)    # [1, 1536, H, W]
        rend_siglip = project_to_siglip2(rend_feat.half(), proj)  # [1, 1536, H, W]

        # Compute heatmaps
        gt_hm = compute_heatmaps(gt_siglip, active_text_emb)      # [K, H, W]
        rend_hm = compute_heatmaps(rend_siglip, active_text_emb)  # [K, H, W]

        # Heatmap correlation
        gt_flat = gt_hm.reshape(len(active_queries), -1).float()
        rend_flat = rend_hm.reshape(len(active_queries), -1).float()
        # Per-query Pearson correlation
        corrs = []
        for q in range(len(active_queries)):
            g = gt_flat[q] - gt_flat[q].mean()
            r = rend_flat[q] - rend_flat[q].mean()
            denom = g.norm() * r.norm()
            if denom > 0:
                corrs.append((g @ r / denom).item())
        if corrs:
            heatmap_corrs.append(np.mean(corrs))

        # Load semantic GT
        sem_gt = load_semantic_gt(sem_dir, frame_idx, (fH, fW)).to(device)

        # Per-class IoU and AP
        for qi, (qname, cid) in enumerate(zip(active_queries, active_class_ids)):
            mask_gt = (sem_gt == cid)
            if mask_gt.sum() == 0:
                continue  # Class not in this frame

            # GT grounding
            gt_sim = gt_hm[qi]
            thresh = gt_sim.median().item()
            pred_gt = gt_sim > thresh
            inter = (pred_gt & mask_gt).float().sum()
            union = (pred_gt | mask_gt).float().sum()
            iou = (inter / union).item() if union > 0 else 0.0
            per_class_iou_gt[qname].append(iou)

            # GT AP
            scores = gt_sim.flatten().float()
            labels = mask_gt.flatten().float()
            sorted_idx = scores.argsort(descending=True)
            labels_s = labels[sorted_idx]
            tp_cum = labels_s.cumsum(0)
            prec = tp_cum / torch.arange(1, len(labels_s) + 1, device=device, dtype=torch.float32)
            ap = (prec * labels_s).sum() / labels_s.sum()
            per_class_ap_gt[qname].append(ap.item())

            # Rendered grounding
            rend_sim = rend_hm[qi]
            thresh_r = rend_sim.median().item()
            pred_rend = rend_sim > thresh_r
            inter_r = (pred_rend & mask_gt).float().sum()
            union_r = (pred_rend | mask_gt).float().sum()
            iou_r = (inter_r / union_r).item() if union_r > 0 else 0.0
            per_class_iou_rend[qname].append(iou_r)

            # Rendered AP
            scores_r = rend_sim.flatten().float()
            sorted_idx_r = scores_r.argsort(descending=True)
            labels_sr = labels[sorted_idx_r]
            tp_cum_r = labels_sr.cumsum(0)
            prec_r = tp_cum_r / torch.arange(1, len(labels_sr) + 1, device=device, dtype=torch.float32)
            ap_r = (prec_r * labels_sr).sum() / labels_sr.sum()
            per_class_ap_rend[qname].append(ap_r.item())

        # Argmax segmentation (zero-shot)
        gt_argmax = gt_hm.argmax(dim=0)   # [H, W] index into active queries
        rend_argmax = rend_hm.argmax(dim=0)
        for qi, cid in enumerate(active_class_ids):
            mask = (sem_gt == cid)
            if mask.sum() == 0:
                continue
            argmax_total += mask.sum().item()
            argmax_correct_gt += (gt_argmax[mask] == qi).sum().item()
            argmax_correct_rend += (rend_argmax[mask] == qi).sum().item()

        # Save visualization for first few frames
        if len(vis_frames) < 5:
            vis_frames.append({
                'idx': frame_idx,
                'gt_hm': gt_hm.cpu(),
                'rend_hm': rend_hm.cpu(),
                'sem_gt': sem_gt.cpu(),
            })

    # Print results
    print("\n" + "=" * 60)
    print("TEXT GROUNDING RESULTS")
    print("=" * 60)

    # Heatmap correlation
    mean_corr = np.mean(heatmap_corrs) if heatmap_corrs else 0
    print(f"\nMean heatmap correlation (GT vs rendered): {mean_corr:.4f}")

    # Zero-shot argmax accuracy
    gt_acc = argmax_correct_gt / max(argmax_total, 1)
    rend_acc = argmax_correct_rend / max(argmax_total, 1)
    print(f"Zero-shot argmax accuracy  — GT: {gt_acc:.4f}, Rendered: {rend_acc:.4f}")

    # Per-class IoU
    print(f"\n{'Class':<20} {'GT mIoU':>8} {'Rend mIoU':>10} {'GT mAP':>8} {'Rend mAP':>10}")
    print("-" * 60)
    all_gt_iou, all_rend_iou = [], []
    all_gt_ap, all_rend_ap = [], []
    for qname in active_queries:
        if qname == "undefined":
            continue
        gt_iou = np.mean(per_class_iou_gt[qname]) if per_class_iou_gt[qname] else 0
        rend_iou = np.mean(per_class_iou_rend[qname]) if per_class_iou_rend[qname] else 0
        gt_ap = np.mean(per_class_ap_gt[qname]) if per_class_ap_gt[qname] else 0
        rend_ap = np.mean(per_class_ap_rend[qname]) if per_class_ap_rend[qname] else 0
        if per_class_iou_gt[qname]:
            all_gt_iou.append(gt_iou)
            all_rend_iou.append(rend_iou)
            all_gt_ap.append(gt_ap)
            all_rend_ap.append(rend_ap)
            print(f"{qname:<20} {gt_iou:>8.4f} {rend_iou:>10.4f} {gt_ap:>8.4f} {rend_ap:>10.4f}")
    print("-" * 60)
    print(f"{'Mean':<20} {np.mean(all_gt_iou):>8.4f} {np.mean(all_rend_iou):>10.4f} "
          f"{np.mean(all_gt_ap):>8.4f} {np.mean(all_rend_ap):>10.4f}")

    # Save visualizations
    if vis_frames and args.vis_dir:
        save_visualizations(vis_frames, active_queries, active_class_ids,
                            Path(args.vis_dir))

    return {
        "heatmap_correlation": mean_corr,
        "gt_argmax_acc": gt_acc,
        "rend_argmax_acc": rend_acc,
        "gt_mean_iou": np.mean(all_gt_iou) if all_gt_iou else 0,
        "rend_mean_iou": np.mean(all_rend_iou) if all_rend_iou else 0,
        "gt_mean_ap": np.mean(all_gt_ap) if all_gt_ap else 0,
        "rend_mean_ap": np.mean(all_rend_ap) if all_rend_ap else 0,
    }


def save_visualizations(vis_frames, queries, class_ids, out_dir):
    """Save heatmap comparison grids."""
    out_dir.mkdir(parents=True, exist_ok=True)

    for vf in vis_frames:
        idx = vf['idx']
        gt_hm = vf['gt_hm'].numpy()    # [K, H, W]
        rend_hm = vf['rend_hm'].numpy()
        sem_gt = vf['sem_gt'].numpy()

        # Pick top-5 classes by area in this frame
        class_areas = []
        for qi, cid in enumerate(class_ids):
            area = (sem_gt == cid).sum()
            if area > 0 and queries[qi] != "undefined":
                class_areas.append((qi, queries[qi], area))
        class_areas.sort(key=lambda x: -x[2])
        top5 = class_areas[:5]

        if not top5:
            continue

        rows = []
        for qi, qname, _ in top5:
            # Normalize heatmaps to [0,1]
            gt_h = gt_hm[qi]
            rend_h = rend_hm[qi]
            vmin = min(gt_h.min(), rend_h.min())
            vmax = max(gt_h.max(), rend_h.max())
            if vmax - vmin > 1e-6:
                gt_norm = (gt_h - vmin) / (vmax - vmin)
                rend_norm = (rend_h - vmin) / (vmax - vmin)
            else:
                gt_norm = np.zeros_like(gt_h)
                rend_norm = np.zeros_like(rend_h)

            gt_color = cv2.applyColorMap((gt_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
            rend_color = cv2.applyColorMap((rend_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

            # Semantic GT mask
            mask = (sem_gt == class_ids[qi]).astype(np.uint8) * 255
            mask_color = cv2.applyColorMap(mask, cv2.COLORMAP_BONE)

            # Label
            label_img = np.zeros_like(gt_color)
            cv2.putText(label_img, qname, (2, 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1)

            row = np.concatenate([label_img, mask_color, gt_color, rend_color], axis=1)
            rows.append(row)

        grid = np.concatenate(rows, axis=0)

        # Add column headers
        H, W_col = gt_hm.shape[1], gt_hm.shape[2]
        header = np.zeros((25, grid.shape[1], 3), dtype=np.uint8)
        for ci, name in enumerate(["Query", "GT Mask", "GT Heatmap", "Rendered Heatmap"]):
            cv2.putText(header, name, (ci * W_col + 2, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        grid = np.concatenate([header, grid], axis=0)

        cv2.imwrite(str(out_dir / f"grounding_frame_{idx}.png"), grid)

    print(f"Saved {len(vis_frames)} visualization frames to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Text grounding evaluation")
    parser.add_argument("--config", required=True, help="RADIO-GS config YAML")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint")
    parser.add_argument("--gt_features", required=True,
                        help="Dir with GT RADIO 1280d .pt files")
    parser.add_argument("--semantic_dir", required=True,
                        help="Dir with semantic_class_*.png GT masks")
    parser.add_argument("--pose_file", default="",
                        help="Pose file (override config)")
    parser.add_argument("--rgb_dir", default=None,
                        help="RGB dir for refiner guide")
    parser.add_argument("--text_embeddings",
                        default="output/radio_gs/siglip2_text_embeddings.pt",
                        help="Pre-computed SigLIP2 text embeddings")
    parser.add_argument("--projection_weights",
                        default="output/radio_gs/siglip2_feat_projection.pth",
                        help="SigLIP2 feature projection weights")
    parser.add_argument("--vis_dir", default=None,
                        help="Dir to save visualization grids")
    args = parser.parse_args()
    evaluate_grounding(args)


if __name__ == "__main__":
    main()
