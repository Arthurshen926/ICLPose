#!/usr/bin/env python3#!/usr/bin/env python3









































































































































































    main()if __name__ == "__main__":    print(f"\nAll renders saved to {args_cli.out_dir}")                f.write(f"  {cam.image_name}: {p:.2f} dB\n")            for j, (cam, p) in enumerate(zip(selected, psnrs)):            f.write(f"Individual PSNRs:\n")            f.write(f"Avg PSNR: {avg_psnr:.2f} dB ({n} views)\n")            f.write(f"N_Gaussians: {gaussians.get_xyz.shape[0]:,}\n")            f.write(f"PLY: {ply_path}\n")            f.write(f"Scene: {name}\n")        with open(os.path.join(scene_dir, "summary.txt"), "w") as f:        # Save summary        print(f"  {name} avg PSNR: {avg_psnr:.2f} dB ({n} views)")        avg_psnr = np.mean(psnrs) if psnrs else 0                print(f"  [{i+1}/{n}] PSNR={psnr:.2f} dB  {cam.image_name}")                img.save(os.path.join(scene_dir, fname))                fname = f"{i:02d}_{cam.image_name.replace('/', '_')}_psnr{psnr:.1f}.png"                img = Image.fromarray(canvas)                canvas[:, w*3+6:w*4+6] = depth_color                canvas[:, w*2+4:w*3+4] = error_amp                canvas[:, w+2:w*2+2] = rendered_np                canvas[:, :w] = gt_np                canvas = np.zeros((h, w * 4 + 6, 3), dtype=np.uint8)                h, w = rendered_np.shape[:2]                # Create comparison: GT | Rendered | Error | Depth                    depth_color = np.zeros_like(rendered_np)                else:                    depth_color = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)                    import matplotlib.cm as cm                    matplotlib.use('Agg')                    import matplotlib                    # Turbo colormap                        d_norm = np.zeros_like(d)                    else:                        d_norm = np.clip((d - d_min) / max(d_max - d_min, 1e-6), 0, 1)                        d_min, d_max = np.percentile(d_valid, [2, 98])                    if len(d_valid) > 0:                    d_valid = d[d > 0]                    d = depth.squeeze().cpu().numpy()                if depth is not None:                # Depth visualization                error_amp = np.clip(error * 5, 0, 255).astype(np.uint8)                error = np.abs(rendered_np.astype(float) - gt_np.astype(float))                # Error map (amplified 5x)                gt_np = (gt.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)                rendered_np = (rendered.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)                # Convert to numpy                psnrs.append(psnr)                psnr = 10 * math.log10(1.0 / mse.item()) if mse > 0 else 100.0                mse = ((rendered - gt) ** 2).mean()                # Compute PSNR                gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)                gt = load_image_tensor(cam)                depth = render_pkg["depth"]  # [1, H, W]                rw, rh = render_pkg["width"], render_pkg["height"]                rendered = render_pkg["render"].clamp(0, 1)  # [3, H, W]                render_pkg = render_2dgs(gaussians, cam, bg_color, longest_edge=longest_edge)            with torch.no_grad():        for i, cam in enumerate(selected):        psnrs = []        os.makedirs(scene_dir, exist_ok=True)        scene_dir = os.path.join(args_cli.out_dir, name)        selected = [cams_to_use[i] for i in indices]        indices = np.linspace(0, len(cams_to_use) - 1, n, dtype=int)        n = min(n_samples, len(cams_to_use))        # Pick evenly-spaced samples        gaussians.load_ply(ply_path)        gaussians = GaussianModel2DGS(sh_degree=3)        # Load Gaussian model        cams_to_use = test_cams if test_cams else train_cams        train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene(source_dir)        # Load scene cameras        print(f"{'='*60}")        print(f"  Rendering {name}")        print(f"\n{'='*60}")            continue            print(f"[SKIP] {name}: source not found at {source_dir}")        if not os.path.exists(source_dir):            continue            print(f"[SKIP] {name}: PLY not found at {ply_path}")        if not os.path.exists(ply_path):        n_samples = scene_cfg["n_samples"]        longest_edge = scene_cfg["longest_edge"]        ply_path = scene_cfg["ply_path"]        source_dir = scene_cfg["source_dir"]        name = scene_cfg["name"]    for scene_cfg in SCENES:    bg_color = torch.zeros(3, device="cuda")    os.makedirs(args_cli.out_dir, exist_ok=True)    tmod.args = _FakeArgs()        sensor_depth_dir = None        init_from_depth = False    class _FakeArgs:    import feature_3dgs.train_2dgs_geometry as tmod    # Monkey-patch: load_scene references `args` globally; provide minimal stub    )        GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor, CameraData,    from feature_3dgs.train_2dgs_geometry import (    # Import after setting CUDA device    torch.cuda.set_device(0)    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)    args_cli = parser.parse_args()    parser.add_argument("--out_dir", default="output/geometry_eval_renders")    parser.add_argument("--gpu", type=int, default=0)    parser = argparse.ArgumentParser()def main():]    },        "n_samples": 6,        "longest_edge": 0,        "ply_path": "output/2dgs_models/room_0/v4_minN/point_cloud/iteration_30000/point_cloud.ply",        "source_dir": "dataset/room_0",        "name": "room_0",    {    },        "n_samples": 6,        "longest_edge": 0,        "ply_path": "output/2dgs_models/stairs/v5_minN/point_cloud/iteration_30000/point_cloud.ply",        "source_dir": "dataset/stairs",        "name": "stairs",    {    },        "n_samples": 6,        "longest_edge": 960,        "ply_path": "output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply",        "source_dir": "dataset/OldHospital",        "name": "OldHospital",    {SCENES = [# ── Scenes to evaluate ──────────────────────────────────────────────────from PIL import Imageimport numpy as npimport torch.nn.functional as Fimport torchimport mathimport argparsesys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))import sys, os"""    python scripts/render_geometry_eval.py --gpu 0Usage:"""Render visual samples from trained 2DGS geometry models."""Render visual samples from trained 2DGS geometry models."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import argparse, math
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

SCENES = [
    {"name": "OldHospital",
     "source_dir": "dataset/OldHospital",
     "ply_path": "output/2dgs_models/OldHospital/v7_depth/point_cloud/iteration_30000/point_cloud.ply",
     "longest_edge": 960, "n_samples": 6},
    {"name": "stairs",
     "source_dir": "dataset/stairs",
     "ply_path": "output/2dgs_models/stairs/v5_minN/point_cloud/iteration_30000/point_cloud.ply",
     "longest_edge": 0, "n_samples": 6},
    {"name": "room_0",
     "source_dir": "dataset/room_0",
     "ply_path": "output/2dgs_models/room_0/v4_minN/point_cloud/iteration_30000/point_cloud.ply",
     "longest_edge": 0, "n_samples": 6},
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--out_dir", default="output/geometry_eval_renders")
    args_cli = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args_cli.gpu)
    torch.cuda.set_device(0)

    from feature_3dgs.train_2dgs_geometry import (
        GaussianModel2DGS, load_scene, render_2dgs, load_image_tensor,
    )
    import feature_3dgs.train_2dgs_geometry as tmod
    class _FakeArgs:
        init_from_depth = False
        sensor_depth_dir = None
    tmod.args = _FakeArgs()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.cm as cm

    os.makedirs(args_cli.out_dir, exist_ok=True)
    bg_color = torch.zeros(3, device="cuda")

    for scene_cfg in SCENES:
        name = scene_cfg["name"]
        ply_path = scene_cfg["ply_path"]
        longest_edge = scene_cfg["longest_edge"]
        n_samples = scene_cfg["n_samples"]

        if not os.path.exists(ply_path):
            print(f"[SKIP] {name}: PLY not found at {ply_path}")
            continue

        print(f"\n{'='*60}")
        print(f"  Rendering {name}")
        print(f"{'='*60}")

        train_cams, test_cams, _, _, _ = load_scene(scene_cfg["source_dir"])
        cams_to_use = test_cams if test_cams else train_cams

        gaussians = GaussianModel2DGS(sh_degree=3)
        gaussians.load_ply(ply_path)

        n = min(n_samples, len(cams_to_use))
        indices = np.linspace(0, len(cams_to_use) - 1, n, dtype=int)
        selected = [cams_to_use[i] for i in indices]

        scene_dir = os.path.join(args_cli.out_dir, name)
        os.makedirs(scene_dir, exist_ok=True)

        psnrs = []
        for i, cam_obj in enumerate(selected):
            with torch.no_grad():
                render_pkg = render_2dgs(gaussians, cam_obj, bg_color, longest_edge=longest_edge)
                rendered = render_pkg["render"].clamp(0, 1)
                rw, rh = render_pkg["width"], render_pkg["height"]
                depth = render_pkg["depth"]

                gt = load_image_tensor(cam_obj)
                gt = F.interpolate(gt.unsqueeze(0), size=(rh, rw), mode="bilinear", align_corners=False).squeeze(0)

                mse = ((rendered - gt) ** 2).mean()
                psnr = 10 * math.log10(1.0 / mse.item()) if mse > 0 else 100.0
                psnrs.append(psnr)

                rendered_np = (rendered.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                gt_np = (gt.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                error_amp = np.clip(np.abs(rendered_np.astype(float) - gt_np.astype(float)) * 5, 0, 255).astype(np.uint8)

                if depth is not None:
                    d = depth.squeeze().cpu().numpy()
                    d_valid = d[d > 0]
                    if len(d_valid) > 0:
                        d_min, d_max = np.percentile(d_valid, [2, 98])
                        d_norm = np.clip((d - d_min) / max(d_max - d_min, 1e-6), 0, 1)
                    else:
                        d_norm = np.zeros_like(d)
                    depth_color = (cm.turbo(d_norm)[:, :, :3] * 255).astype(np.uint8)
                else:
                    depth_color = np.zeros_like(rendered_np)

                h, w = rendered_np.shape[:2]
                canvas = np.zeros((h, w * 4 + 6, 3), dtype=np.uint8)
                canvas[:, :w] = gt_np
                canvas[:, w+2:w*2+2] = rendered_np
                canvas[:, w*2+4:w*3+4] = error_amp
                canvas[:, w*3+6:w*4+6] = depth_color

                img = Image.fromarray(canvas)
                safe_name = cam_obj.image_name.replace('/', '_').replace('\\', '_')
                fname = f"{i:02d}_{safe_name}_psnr{psnr:.1f}.png"
                img.save(os.path.join(scene_dir, fname))
                print(f"  [{i+1}/{n}] PSNR={psnr:.2f} dB  {cam_obj.image_name}")

        avg_psnr = np.mean(psnrs) if psnrs else 0
        print(f"  {name} avg PSNR: {avg_psnr:.2f} dB ({n} views)")

        with open(os.path.join(scene_dir, "summary.txt"), "w") as f:
            f.write(f"Scene: {name}\nPLY: {ply_path}\n")
            f.write(f"N_Gaussians: {gaussians.get_xyz.shape[0]:,}\n")
            f.write(f"Avg PSNR: {avg_psnr:.2f} dB ({n} views)\n")
            for j, (c, p) in enumerate(zip(selected, psnrs)):
                f.write(f"  {c.image_name}: {p:.2f} dB\n")

    print(f"\nAll renders saved to {args_cli.out_dir}")

if __name__ == "__main__":
    main()
