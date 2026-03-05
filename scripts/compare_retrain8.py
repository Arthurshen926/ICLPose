"""Compare retrain7 vs retrain8 on test views (side-by-side with GT)."""
import torch, json, sys, os, math, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.visualize_2dgs_recon import load_ply_2dgs, cam_to_viewmat, render_2dgs
from PIL import Image
import torch.nn.functional as F

# Models to compare
MODELS = {
    "v3 baseline (30k)": ("output/2dgs_models/OldHospital/v3/point_cloud/iteration_30000/point_cloud.ply",
                          "output/2dgs_models/OldHospital/v3/cameras.json"),
    "retrain7 (15k, 5fixes)": ("output/2dgs_models/OldHospital/v3_retrain7/point_cloud/iteration_15000/point_cloud.ply",
                                "output/2dgs_models/OldHospital/v3_retrain7/cameras.json"),
    "retrain8 (30k, stable)": ("output/2dgs_models/OldHospital/v3_retrain8/point_cloud/iteration_30000/point_cloud.ply",
                                "output/2dgs_models/OldHospital/v3_retrain8/cameras.json"),
}

# Test views to render
TEST_VIEWS = [
    "seq4/frame00002.png",
    "seq4/frame00049.png",
    "seq8/frame00110.png",
    "seq8/frame00051.png",
]

RENDER_W, RENDER_H = 960, 540  # half resolution for display

def compute_psnr(img, gt):
    mse = F.mse_loss(img, gt)
    if mse > 0:
        return -10 * math.log10(mse.item())
    return 99.0

def main():
    device = "cuda"
    
    # Load all models
    loaded = {}
    for name, (ply_path, cam_path) in MODELS.items():
        print(f"Loading {name}...")
        model = load_ply_2dgs(ply_path)
        cams = json.load(open(cam_path))
        loaded[name] = (model, cams)
    
    # Load GT images
    gt_dir = "dataset/OldHospital"
    
    # Create comparison grid
    n_views = len(TEST_VIEWS)
    n_models = len(MODELS)
    cell_w, cell_h = RENDER_W, RENDER_H
    gap = 4
    
    # Grid: rows=views, cols=GT + models
    total_w = (1 + n_models) * cell_w + n_models * gap
    total_h = n_views * cell_h + (n_views - 1) * gap + 30 * n_views  # +30 for labels
    
    grid = np.ones((total_h, total_w, 3), dtype=np.uint8) * 255
    
    y_offset = 0
    for v_idx, view_name in enumerate(TEST_VIEWS):
        print(f"\n  Rendering {view_name}...")
        
        # GT image
        gt_path = os.path.join(gt_dir, view_name)
        gt_img = Image.open(gt_path).convert("RGB").resize((cell_w, cell_h), Image.LANCZOS)
        gt_np = np.array(gt_img)
        gt_tensor = torch.from_numpy(gt_np).float().permute(2, 0, 1).to(device) / 255.0
        
        # Label
        label_y = y_offset
        y_offset += 30
        
        # Place GT
        grid[y_offset:y_offset+cell_h, 0:cell_w] = gt_np
        
        # Render each model
        x_offset = cell_w
        results = []
        for m_idx, (name, (model, cams)) in enumerate(loaded.items()):
            x_offset += gap
            
            cam_info = [c for c in cams if view_name.replace(".png", "") in c["img_name"]]
            if not cam_info:
                cam_info = [c for c in cams if os.path.splitext(view_name)[0] in c["img_name"]]
            if not cam_info:
                print(f"    WARNING: cam not found for {view_name} in {name}")
                x_offset += cell_w
                continue
            
            cam = cam_info[0]
            fw, fh = cam["width"], cam["height"]
            fx, fy = cam["fx"], cam["fy"]
            
            # Scale K to render resolution
            sx, sy = cell_w / fw, cell_h / fh
            viewmat = torch.tensor(cam_to_viewmat(cam), device=device)
            K = torch.tensor([[fx*sx, 0, cell_w/2.0], [0, fy*sy, cell_h/2.0], [0, 0, 1]], device=device)
            
            with torch.no_grad():
                rgb, depth, alpha, normal = render_2dgs(model, viewmat, K, cell_w, cell_h)
            
            # rgb is [H, W, 3] from render_2dgs
            rendered = rgb.clamp(0, 1).cpu().numpy()
            rendered_u8 = (rendered * 255).astype(np.uint8)
            
            # PSNR: convert gt_tensor from [3,H,W] to [H,W,3] to match rgb
            gt_hwc = gt_tensor.permute(1, 2, 0).to(device)
            psnr = compute_psnr(rgb, gt_hwc)
            results.append((name, psnr))
            print(f"    {name}: {psnr:.2f} dB")
            
            grid[y_offset:y_offset+cell_h, x_offset:x_offset+cell_w] = rendered_u8
            x_offset += cell_w
        
        y_offset += cell_h + gap
    
    # Save
    out_path = "output/comparison_retrain8.png"
    Image.fromarray(grid).save(out_path)
    print(f"\nSaved comparison to {out_path}")

if __name__ == "__main__":
    main()
