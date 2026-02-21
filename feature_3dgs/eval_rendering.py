"""
训练后特征嵌入3DGS 渲染验证脚本
=================================
加载训练好的checkpoint，对多帧进行特征图渲染 + RGB渲染，
与GT特征和GT RGB做定量/定性对比。

用法:
    python -m feature_3dgs.eval_rendering \
        --ply_path dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply \
        --checkpoint output/feature_3dgs/room_0_new/best_model.pth \
        --feature_dir dataset/room_0/Sequence_1/features_compressed/fused \
        --traj_path dataset/room_0/Sequence_1/traj_w_c.txt \
        --rgb_dir dataset/room_0/Sequence_1/rgb \
        --output_dir output/feature_3dgs/room_0_new/eval
"""

import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer
from feature_3dgs.feature_dataset import FeatureEmbeddingDataset

try:
    import PIL.Image as Image
except ImportError:
    Image = None


# ──────────────────────────── 可视化工具 ────────────────────────────

def pca_colorize(feat_map: torch.Tensor, n_components: int = 3) -> torch.Tensor:
    """
    将 [D, H, W] 特征图 PCA 降到 3 维，归一化到 [0,1] 返回 [3, H, W]。
    """
    D, H, W = feat_map.shape
    flat = feat_map.reshape(D, -1).T  # [HW, D]
    flat = flat - flat.mean(0)
    U, S, V = torch.svd(flat)
    proj = U[:, :n_components]  # [HW, 3]
    for c in range(n_components):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)
    return proj.T.reshape(n_components, H, W)


def pca_colorize_joint(feat_a: torch.Tensor, feat_b: torch.Tensor, n_components: int = 3):
    """
    对两张特征图做 **联合 PCA**（共用同一组主成分），
    确保颜色空间一致，便于视觉对比。
    返回 (vis_a, vis_b) 各为 [3, H, W]。
    """
    D, H, W = feat_a.shape
    flat_a = feat_a.reshape(D, -1).T  # [HW, D]
    flat_b = feat_b.reshape(D, -1).T
    combined = torch.cat([flat_a, flat_b], dim=0)  # [2*HW, D]
    combined = combined - combined.mean(0)
    U, S, V = torch.svd(combined)
    proj = U[:, :n_components]
    for c in range(n_components):
        lo, hi = proj[:, c].min(), proj[:, c].max()
        proj[:, c] = (proj[:, c] - lo) / (hi - lo + 1e-8)
    n = H * W
    vis_a = proj[:n].T.reshape(n_components, H, W)
    vis_b = proj[n:].T.reshape(n_components, H, W)
    return vis_a, vis_b


def cosine_sim_map(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """计算 per-pixel cosine similarity [H, W]"""
    return F.cosine_similarity(pred, gt, dim=0)  # 沿通道维度


def tensor_to_pil(t: torch.Tensor) -> "Image.Image":
    """[C, H, W] float tensor → PIL Image"""
    arr = (t.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    return Image.fromarray(arr)


def heatmap_to_pil(val: torch.Tensor, vmin: float = 0.0, vmax: float = 1.0) -> "Image.Image":
    """[H, W] float → 伪彩色 heatmap PIL Image（不依赖 matplotlib）"""
    v = ((val.cpu().float() - vmin) / (vmax - vmin + 1e-8)).clamp(0, 1).numpy()
    # 简单 jet-like colormap
    r = np.clip(1.5 - np.abs(v * 4 - 3), 0, 1)
    g = np.clip(1.5 - np.abs(v * 4 - 2), 0, 1)
    b = np.clip(1.5 - np.abs(v * 4 - 1), 0, 1)
    rgb = np.stack([r, g, b], axis=-1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def make_comparison_strip(*images, target_height: int = None) -> "Image.Image":
    """将多张 PIL Image 水平拼接为一张"""
    if target_height is None:
        target_height = max(im.size[1] for im in images)
    resized = []
    for im in images:
        if im.size[1] != target_height:
            w = int(im.size[0] * target_height / im.size[1])
            im = im.resize((w, target_height), Image.BILINEAR)
        resized.append(im)
    total_w = sum(im.size[0] for im in resized)
    canvas = Image.new("RGB", (total_w, target_height))
    x = 0
    for im in resized:
        canvas.paste(im.convert("RGB"), (x, 0))
        x += im.size[0]
    return canvas


# ──────────────────────────── 评估主函数 ────────────────────────────

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ── 1. 加载模型 + checkpoint ──
    print("=== 加载模型 ===")
    model = GaussianFeatureModel(feature_dim=args.feature_dim)
    model.load_ply(args.ply_path)
    model = model.to(device)

    ckpt = torch.load(args.checkpoint, map_location=device)
    model._loc_feature.data.copy_(ckpt["loc_feature"])
    print(f"  checkpoint: {args.checkpoint}")
    print(f"  训练迭代: {ckpt.get('iteration', '?')},  loss: {ckpt.get('loss', '?')}")

    # ── 2. 加载数据集 ──
    intrinsics = {"fx": args.fx, "fy": args.fy, "cx": args.cx, "cy": args.cy}
    dataset = FeatureEmbeddingDataset(
        feature_dir=args.feature_dir,
        traj_path=args.traj_path,
        intrinsics=intrinsics,
        img_size=(args.img_height, args.img_width),
        normalize_features=True,
    )

    # 选取要评估的帧
    n_total = len(dataset)
    if args.eval_frames is not None:
        indices = [int(x) for x in args.eval_frames.split(",")]
    else:
        step = max(1, n_total // args.num_eval)
        indices = list(range(0, n_total, step))[: args.num_eval]
    print(f"  评估帧索引 ({len(indices)}帧): {indices}")

    rgb_dir = Path(args.rgb_dir) if args.rgb_dir else None

    # ── 3. 逐帧渲染 + 评估 ──
    all_l1, all_cos, all_psnr = [], [], []

    for rank, idx in enumerate(indices):
        sample = dataset[idx]
        frame_id = sample["frame_id"]
        gt_feat = sample["feature_map"].to(device)       # [D, fH, fW]
        pose = sample["pose"].to(device)                  # [4, 4]
        fH, fW = gt_feat.shape[1], gt_feat.shape[2]

        tag = f"frame_{frame_id:04d}"
        print(f"\n── [{rank+1}/{len(indices)}] {tag} ──")

        # 渲染特征图
        with torch.no_grad():
            result = FeatureRenderer.render_features(
                gaussian_model=model, viewmat=pose,
                fx=intrinsics["fx"], fy=intrinsics["fy"],
                cx=intrinsics["cx"], cy=intrinsics["cy"],
                img_height=args.img_height, img_width=args.img_width,
                feature_height=fH, feature_width=fW,
            )
        rendered_feat = result["feature_map"]  # [D, fH, fW]

        # ─ 定量指标 ─
        l1 = torch.abs(rendered_feat - gt_feat).mean().item()
        cos = F.cosine_similarity(rendered_feat, gt_feat, dim=0).mean().item()
        # 把特征当图像算 PSNR（值域 [-1,1]→[0,1] 近似）
        mse = ((rendered_feat - gt_feat) ** 2).mean().item()
        psnr = -10 * np.log10(mse + 1e-10)
        all_l1.append(l1)
        all_cos.append(cos)
        all_psnr.append(psnr)
        print(f"  L1={l1:.6f}  CosSim={cos:.4f}  PSNR={psnr:.2f}dB  visible={result['visible_mask'].sum().item()}")

        if Image is None:
            continue  # 没有 PIL 就跳过可视化

        # ─ 联合 PCA 可视化（同一主成分空间）─
        vis_rendered, vis_gt = pca_colorize_joint(
            rendered_feat.detach().cpu(), gt_feat.detach().cpu()
        )

        # ─ cosine similarity heatmap ─
        cos_map = cosine_sim_map(rendered_feat.detach().cpu(), gt_feat.detach().cpu())

        # ─ 保存单帧结果 ─
        im_rendered = tensor_to_pil(vis_rendered)
        im_gt = tensor_to_pil(vis_gt)
        im_cos = heatmap_to_pil(cos_map, vmin=-0.2, vmax=1.0)

        im_rendered.save(str(out / f"{tag}_feat_rendered.png"))
        im_gt.save(str(out / f"{tag}_feat_gt.png"))
        im_cos.save(str(out / f"{tag}_cosine_sim.png"))

        # ─ 渲染 RGB ─
        strip_parts = [im_gt, im_rendered, im_cos]
        with torch.no_grad():
            rgb_result = FeatureRenderer.render_rgb(
                gaussian_model=model, viewmat=pose,
                fx=intrinsics["fx"], fy=intrinsics["fy"],
                cx=intrinsics["cx"], cy=intrinsics["cy"],
                img_height=args.img_height, img_width=args.img_width,
            )
        rgb_rendered = rgb_result["rgb"]  # [3, H, W]
        im_rgb_rendered = tensor_to_pil(rgb_rendered)
        im_rgb_rendered.save(str(out / f"{tag}_rgb_rendered.png"))

        # ─ GT RGB（如有）─
        if rgb_dir:
            gt_rgb_path = rgb_dir / f"rgb_{frame_id}.png"
            if gt_rgb_path.exists():
                im_gt_rgb = Image.open(str(gt_rgb_path)).convert("RGB")
                im_gt_rgb.save(str(out / f"{tag}_rgb_gt.png"))
                strip_parts = [im_gt_rgb, im_rgb_rendered, im_gt, im_rendered, im_cos]

        # ─ 拼接对比条 ─
        strip = make_comparison_strip(*strip_parts, target_height=fH * 4)
        strip.save(str(out / f"{tag}_comparison.png"))
        print(f"  保存 → {out / tag}_*.png")

    # ── 4. 汇总 ──
    print("\n" + "=" * 60)
    print("评估汇总")
    print("=" * 60)
    print(f"  帧数:        {len(indices)}")
    print(f"  平均 L1:     {np.mean(all_l1):.6f}")
    print(f"  平均 CosSim: {np.mean(all_cos):.4f}")
    print(f"  平均 PSNR:   {np.mean(all_psnr):.2f} dB")

    # 保存指标
    metrics_path = out / "metrics.txt"
    with open(metrics_path, "w") as f:
        f.write("frame_id,L1,CosSim,PSNR\n")
        for i, idx in enumerate(indices):
            fid = dataset[idx]["frame_id"]
            f.write(f"{fid},{all_l1[i]:.6f},{all_cos[i]:.4f},{all_psnr[i]:.2f}\n")
        f.write(f"\nmean,{np.mean(all_l1):.6f},{np.mean(all_cos):.4f},{np.mean(all_psnr):.2f}\n")
    print(f"  指标已保存: {metrics_path}")
    print(f"  可视化目录: {out}")


# ──────────────────────────── CLI ────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained feature-embedded 3DGS rendering")

    p.add_argument("--ply_path", type=str, required=True, help="原始3DGS PLY路径")
    p.add_argument("--checkpoint", type=str, required=True, help="训练 checkpoint (.pth)")
    p.add_argument("--feature_dir", type=str, required=True, help="GT压缩特征目录")
    p.add_argument("--traj_path", type=str, required=True, help="位姿文件路径")
    p.add_argument("--rgb_dir", type=str, default=None, help="GT RGB图像目录（可选）")
    p.add_argument("--output_dir", type=str, default="output/feature_3dgs/eval", help="输出目录")

    p.add_argument("--feature_dim", type=int, default=256)
    p.add_argument("--fx", type=float, default=320.0)
    p.add_argument("--fy", type=float, default=320.0)
    p.add_argument("--cx", type=float, default=319.5)
    p.add_argument("--cy", type=float, default=239.5)
    p.add_argument("--img_height", type=int, default=480)
    p.add_argument("--img_width", type=int, default=640)

    p.add_argument("--num_eval", type=int, default=10, help="均匀采样评估帧数")
    p.add_argument("--eval_frames", type=str, default=None,
                   help="指定帧索引 (逗号分隔), 如 '0,5,10'，优先于 --num_eval")

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
