#!/usr/bin/env python3
"""Deep diagnostic: why does the geometry solver produce near-zero translation?"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from ic_models.radio_pose_net import RadioPoseNet
from modules.geometry_solver import compute_image_jacobian
from modules.lie_algebra import se3_exp, compute_gt_flow
from scripts.train_radio_pose import RadioRenderer
import yaml

torch.set_grad_enabled(False)

# --- Load config & model ---
with open('configs/radio_pose_oh.yaml') as f:
    cfg = yaml.safe_load(f)

rcfg = cfg['renderer']
model = RadioPoseNet(
    feat_dim=64, coarse_hw=(17,30), fine_hw=(34,60),
    intrinsics={'fx': rcfg['fx'], 'fy': rcfg['fy'], 'cx': rcfg['cx'], 'cy': rcfg['cy']},
    img_hw=tuple(rcfg['img_hw']),
)
ckpt = torch.load('output/radio_pose_oh/checkpoints/best.pth', map_location='cpu')
model.load_state_dict(ckpt['model_state_dict'])
model.cuda().eval()
print(f"Loaded checkpoint epoch {ckpt.get('epoch', '?')}")

# --- Load feature + pose ---
feat_dir = cfg['data']['feature_dir']
traj_path = cfg['data']['traj_path']
poses_c2w = []
with open(traj_path) as f:
    for line in f:
        vals = list(map(float, line.strip().split()))
        if len(vals) == 16:
            poses_c2w.append(np.array(vals).reshape(4, 4))

# --- Renderer ---
renderer = RadioRenderer(
    ply_path=rcfg['ply_path'],
    feature_model_path=rcfg['feature_model_path'],
    device=torch.device('cuda'),
    img_hw=tuple(rcfg['img_hw']),
    fx=rcfg['fx'], fy=rcfg['fy'],
    cx=rcfg['cx'], cy=rcfg['cy'],
)

# --- Helper ---
def solve_damped(A, b, damp=0.001):
    D = torch.clamp(A.diag(), min=1e-6)
    return torch.linalg.solve(A + damp * torch.diag(D), b)

# --- Run diagnostic on multiple samples ---
sample_indices = [0, 100, 200, 500, 800]
torch.manual_seed(42)

for idx in sample_indices:
    if idx >= len(poses_c2w):
        continue
    feat = torch.load(f'{feat_dir}/fine_radio/rgb_{idx}_fine_radio_64x68x120.pt',
                      map_location='cuda').float()
    c2w = torch.from_numpy(poses_c2w[idx].astype(np.float32))
    w2c_gt = torch.linalg.inv(c2w).cuda()

    # Add noise
    noise_rot = 5.0 * np.pi / 180
    noise_trans = 0.12
    rot_noise = torch.randn(3) * noise_rot
    trans_noise = torch.randn(3) * noise_trans
    xi_noise = torch.cat([trans_noise, rot_noise])
    dT = se3_exp(xi_noise.unsqueeze(0).cuda())
    w2c_noisy = dT[0] @ w2c_gt

    # Render
    render_feat = renderer.render_features(w2c_noisy.unsqueeze(0), feat_hw=(68, 120))
    render_depth = renderer.render_depth(w2c_noisy.unsqueeze(0), depth_hw=(34, 60))

    # Forward
    result = model(feat.unsqueeze(0), render_feat, render_depth)

    flow_fine = result['flow_fine']
    conf_fine = result['conf_fine']
    intr = model.fine_intrinsics

    # GT flow using model's own method (correct convention)
    # model.compute_gt_flow(pose_init=noisy, pose_gt=gt, depth=at_noisy, resolution)
    gt_flow, gt_valid_mask = model.compute_gt_flow(
        w2c_noisy.unsqueeze(0), w2c_gt.unsqueeze(0), render_depth, (34, 60))
    gt_valid = gt_valid_mask

    # Jacobian
    Ju, Jv, valid = compute_image_jacobian(render_depth, intr)
    v = valid[0].bool()
    nvalid = v.sum().item()

    print(f"\n{'='*60}")
    print(f"Sample idx={idx}, valid_pixels={nvalid}/{34*60}")

    # Flow stats
    pmag = (flow_fine[0,0]**2 + flow_fine[0,1]**2).sqrt()
    gmag = (gt_flow[0,0]**2 + gt_flow[0,1]**2).sqrt()
    print(f"  Pred flow: mean_u={flow_fine[0,0].mean():.4f} mean_v={flow_fine[0,1].mean():.4f} "
          f"mag={pmag.mean():.4f}")
    print(f"  GT   flow: mean_u={gt_flow[0,0].mean():.4f} mean_v={gt_flow[0,1].mean():.4f} "
          f"mag={gmag.mean():.4f}")

    # Correlation
    pu = flow_fine[0,0].reshape(-1)
    pv = flow_fine[0,1].reshape(-1)
    gu = gt_flow[0,0].reshape(-1)
    gv = gt_flow[0,1].reshape(-1)
    cu = torch.corrcoef(torch.stack([pu, gu]))[0,1]
    cv = torch.corrcoef(torch.stack([pv, gv]))[0,1]
    print(f"  Correlation: u={cu:.4f} v={cv:.4f}")
    print(f"  Conf: mean={conf_fine.mean():.4f} std={conf_fine.std():.4f}")
    print(f"  Depth: min={render_depth.min():.2f} max={render_depth.max():.2f} mean={render_depth.mean():.2f}")
    print(f"  GT valid: {gt_valid.sum().item():.0f}/{34*60}")

    # Jacobian column magnitudes
    Ju_v = Ju[0, v].double()
    Jv_v = Jv[0, v].double()
    print(f"\n  Intrinsics: fx={intr['fx']:.2f} fy={intr['fy']:.2f}")
    labels = ['tx','ty','tz','wx','wy','wz']
    for i, label in enumerate(labels):
        mag = (Ju_v[:,i]**2 + Jv_v[:,i]**2).sqrt()
        print(f"    J_{label}: mag_mean={mag.mean():.4f} mag_max={mag.max():.4f}")

    # Build systems
    w_v = conf_fine.reshape(-1)[v].double()
    fu_p = flow_fine[0,0].reshape(-1)[v].double()
    fv_p = flow_fine[0,1].reshape(-1)[v].double()
    fu_g = gt_flow[0,0].reshape(-1)[v].double()
    fv_g = gt_flow[0,1].reshape(-1)[v].double()

    # Weighted
    JtWJ = (w_v.unsqueeze(1) * Ju_v).T @ Ju_v + (w_v.unsqueeze(1) * Jv_v).T @ Jv_v
    JtWr_p = (w_v * fu_p) @ Ju_v + (w_v * fv_p) @ Jv_v
    JtWr_g = (w_v * fu_g) @ Ju_v + (w_v * fv_g) @ Jv_v

    # Unweighted
    JtJ = Ju_v.T @ Ju_v + Jv_v.T @ Jv_v
    Jtr_p = fu_p @ Ju_v + fv_p @ Jv_v
    Jtr_g = fu_g @ Ju_v + fv_g @ Jv_v

    eigvals_w = torch.linalg.eigvalsh(JtWJ)
    eigvals_u = torch.linalg.eigvalsh(JtJ)

    print(f"\n  JtWJ diag: {[f'{x:.1f}' for x in JtWJ.diag().tolist()]}")
    print(f"  JtWJ eigvals: {[f'{x:.2f}' for x in eigvals_w.tolist()]}")
    print(f"  JtWJ cond: {eigvals_w[-1]/max(eigvals_w[0].item(), 1e-10):.2f}")

    print(f"  JtJ  diag: {[f'{x:.1f}' for x in JtJ.diag().tolist()]}")
    print(f"  JtJ  cond: {eigvals_u[-1]/max(eigvals_u[0].item(), 1e-10):.2f}")

    # Solve
    xi_wp = solve_damped(JtWJ, JtWr_p)
    xi_wg = solve_damped(JtWJ, JtWr_g)
    xi_up = solve_damped(JtJ, Jtr_p)
    xi_ug = solve_damped(JtJ, Jtr_g)

    print(f"\n  Solve results:")
    for name, xi in [('w_pred', xi_wp), ('w_gt', xi_wg),
                      ('u_pred', xi_up), ('u_gt', xi_ug)]:
        print(f"    {name:8s}: trans={xi[:3].norm():.6f}m "
              f"rot={xi[3:].norm()*180/np.pi:.4f}deg "
              f"[{' '.join(f'{x:.6f}' for x in xi.tolist())}]")

    # Model output
    mxi = result['delta_xi'][0]
    print(f"    model   : trans={mxi[:3].norm():.6f}m "
          f"rot={mxi[3:].norm()*180/np.pi:.4f}deg "
          f"[{' '.join(f'{x:.6f}' for x in mxi.tolist())}]")

    # Reference noise
    print(f"\n  GT noise: trans={trans_noise.norm():.6f}m  rot={rot_noise.norm()*180/np.pi:.4f}deg")
    print(f"  GT xi: [{' '.join(f'{x:.6f}' for x in xi_noise.tolist())}]")

    # Cross-correlation between pred flow and GT flow projected onto each J column
    # This shows how much of the flow signal aligns with each DOF
    print(f"\n  Flow projection onto Jacobian columns (pred vs gt):")
    for i, label in enumerate(labels):
        # Project flow onto this J column direction
        proj_p = (fu_p * Ju_v[:,i] + fv_p * Jv_v[:,i]).sum() / nvalid
        proj_g = (fu_g * Ju_v[:,i] + fv_g * Jv_v[:,i]).sum() / nvalid
        print(f"    {label}: pred={proj_p:.6f}  gt={proj_g:.6f}  ratio={proj_p/max(abs(proj_g.item()),1e-10):.3f}")
