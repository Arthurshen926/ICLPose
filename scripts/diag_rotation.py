#!/usr/bin/env python3
"""Diagnose rotation refinement failure in RadioPoseNet v11.

Tests:
1. Single iteration rotation quality
2. Multi-iteration cumulative error
3. Flow accuracy analysis (predicted vs GT)
4. Solver rotation output analysis with GT flow
"""
import sys, os, math, yaml, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ic_models.radio_pose_net import RadioPoseNet
from modules.lie_algebra import se3_exp, se3_log
from scripts.train_radio_pose import RadioRenderer, RadioPoseDataset

torch.set_grad_enabled(False)

# Load config
cfg = yaml.safe_load(open('configs/radio_pose_oh_v11.yaml'))
rc = cfg['renderer']
mc = cfg['model']
dc = cfg['data']

device = torch.device('cuda')

# Setup renderer
renderer = RadioRenderer(
    rc['ply_path'], rc['feature_model_path'], device,
    tuple(rc['img_hw']), rc['fx'], rc['fy'], rc['cx'], rc['cy']
)

# Build model
intrinsics = {k: rc[k] for k in ('fx', 'fy', 'cx', 'cy')}
model = RadioPoseNet(
    feat_dim=mc.get('feat_dim', 64),
    hidden_dim=mc.get('hidden_dim', 128),
    n_heads=mc.get('n_heads', 4),
    n_attn_layers=mc.get('n_attn_layers', 2),
    ffn_dim=mc.get('ffn_dim', 128),
    local_radius=mc.get('local_radius', 4),
    fine_iters=mc.get('fine_iters', 4),
    damping=mc.get('damping', 0.001),
    coarse_hw=tuple(mc.get('coarse_hw', [17, 30])),
    fine_hw=tuple(mc.get('fine_hw', [34, 60])),
    intrinsics=intrinsics,
    img_hw=tuple(rc.get('img_hw', [1080, 1920])),
    conf_floor=mc.get('conf_floor', 0.0),
    detach_conf_in_solver=mc.get('detach_conf_in_solver', False),
    solver_hw=tuple(mc['solver_hw']) if 'solver_hw' in mc else None,
    sequential_solve=mc.get('sequential_solve', False),
    solver_trans_scale=1.0,  # Force scale=1 to see raw solver output
).to(device).eval()

# Load checkpoint
ckpt_path = 'output/radio_pose_oh_v11/checkpoints/latest.pth'
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt['model_state_dict'], strict=False)
print(f"Loaded {ckpt_path}, epoch={ckpt.get('epoch', '?')}")

# Load val dataset
val_ds = RadioPoseDataset(
    dc['feature_dir'], dc['traj_path'],
    noise_rot_deg=dc['val_noise_rot_deg'],
    noise_trans_m=dc['val_noise_trans_m'],
    is_train=False,
)
val_loader = torch.utils.data.DataLoader(
    val_ds, batch_size=8, shuffle=False, num_workers=0,
)

def rot_error_deg(R1, R2):
    """Rotation error between two 3x3 matrices in degrees."""
    R_rel = R1 @ R2.T
    trace = R_rel[0,0] + R_rel[1,1] + R_rel[2,2]
    cos_a = ((trace - 1) / 2).clamp(-1, 1)
    return torch.acos(cos_a).item() * 180 / math.pi

def trans_error_cm(T1, T2):
    """Translation error between two 4x4 w2c matrices in cm."""
    c2w_1 = torch.linalg.inv(T1)
    c2w_2 = torch.linalg.inv(T2)
    return (c2w_1[:3, 3] - c2w_2[:3, 3]).norm().item() * 100

# ================================================================
# Test 1: Single vs multi-iteration rotation
# ================================================================
print("\n" + "=" * 70)
print("TEST 1: Rotation error per outer iteration")
print("=" * 70)

all_init_rot = []
all_1iter_rot = []
all_5iter_rot = []
all_init_trans = []
all_1iter_trans = []
all_5iter_trans = []
all_delta_xi = []

for batch_idx, batch in enumerate(val_loader):
    if batch_idx >= 5:  # 5 batches × 8 = 40 samples
        break

    query_feat = batch['query_feat'].to(device)
    pose_gt = batch['pose_gt'].to(device)
    pose_init = batch['initial_pose'].to(device)

    B = pose_gt.shape[0]

    # Init errors
    for b in range(B):
        c2w_init = torch.linalg.inv(pose_init[b])
        c2w_gt = torch.linalg.inv(pose_gt[b])
        all_init_rot.append(rot_error_deg(c2w_init[:3,:3], c2w_gt[:3,:3]))
        all_init_trans.append((c2w_init[:3,3] - c2w_gt[:3,3]).norm().item() * 100)

    # 1-iteration
    pose_cur = pose_init.clone()
    render_feat, depth = renderer.render_features(pose_cur, (68, 120)), renderer.render_depth(pose_cur, (68, 120))
    result = model(query_feat, render_feat, depth)
    delta_xi_1 = result['delta_xi']
    all_delta_xi.append(delta_xi_1.cpu())

    T_delta = se3_exp(delta_xi_1)
    pose_1iter = T_delta @ pose_cur
    for b in range(B):
        c2w_pred = torch.linalg.inv(pose_1iter[b])
        c2w_gt = torch.linalg.inv(pose_gt[b])
        all_1iter_rot.append(rot_error_deg(c2w_pred[:3,:3], c2w_gt[:3,:3]))
        all_1iter_trans.append((c2w_pred[:3,3] - c2w_gt[:3,3]).norm().item() * 100)

    # 5-iteration
    pose_cur = pose_init.clone()
    for it in range(5):
        render_feat = renderer.render_features(pose_cur, (68, 120))
        depth = renderer.render_depth(pose_cur, (68, 120))
        result = model(query_feat, render_feat, depth)
        if 'delta_xi' in result:
            T_delta = se3_exp(result['delta_xi'])
            pose_cur = T_delta @ pose_cur

    for b in range(B):
        c2w_pred = torch.linalg.inv(pose_cur[b])
        c2w_gt = torch.linalg.inv(pose_gt[b])
        all_5iter_rot.append(rot_error_deg(c2w_pred[:3,:3], c2w_gt[:3,:3]))
        all_5iter_trans.append((c2w_pred[:3,3] - c2w_gt[:3,3]).norm().item() * 100)

print(f"\n{'Metric':<20} {'Init':>10} {'1-iter':>10} {'5-iter':>10}")
print("-" * 52)
print(f"{'median rot (°)':<20} {np.median(all_init_rot):>10.3f} {np.median(all_1iter_rot):>10.3f} {np.median(all_5iter_rot):>10.3f}")
print(f"{'mean rot (°)':<20} {np.mean(all_init_rot):>10.3f} {np.mean(all_1iter_rot):>10.3f} {np.mean(all_5iter_rot):>10.3f}")
print(f"{'median trans (cm)':<20} {np.median(all_init_trans):>10.2f} {np.median(all_1iter_trans):>10.2f} {np.median(all_5iter_trans):>10.2f}")
print(f"{'mean trans (cm)':<20} {np.mean(all_init_trans):>10.2f} {np.mean(all_1iter_trans):>10.2f} {np.mean(all_5iter_trans):>10.2f}")

# ================================================================
# Test 2: Analyze delta_xi from first iteration
# ================================================================
xi_all = torch.cat(all_delta_xi, dim=0)  # (N, 6)
print(f"\n{'='*70}")
print(f"TEST 2: delta_xi statistics (1st iteration, N={xi_all.shape[0]})")
print(f"{'='*70}")
print(f"  trans norm:  mean={xi_all[:,:3].norm(dim=1).mean():.5f}m  "
      f"median={xi_all[:,:3].norm(dim=1).median():.5f}m")
print(f"  rot norm:    mean={xi_all[:,3:].norm(dim=1).mean()*180/math.pi:.3f}°  "
      f"median={xi_all[:,3:].norm(dim=1).median()*180/math.pi:.3f}°")
print(f"  trans [tx,ty,tz]:  {xi_all[:,:3].mean(dim=0).tolist()}")
print(f"  rot [wx,wy,wz]:   {xi_all[:,3:].mean(dim=0).tolist()}")

# ================================================================
# Test 3: Compare predicted flow vs GT flow
# ================================================================
print(f"\n{'='*70}")
print("TEST 3: Flow quality analysis (first 3 batches)")
print(f"{'='*70}")

batch = next(iter(val_loader))
query_feat = batch['query_feat'].to(device)
pose_gt = batch['pose_gt'].to(device)
pose_init = batch['initial_pose'].to(device)

render_feat = renderer.render_features(pose_init, (68, 120))
depth = renderer.render_depth(pose_init, (68, 120))
result = model(query_feat, render_feat, depth)

pred_flow = result['flow_fine']  # (B, 2, 34, 60) at fine resolution
conf = result['conf_fine']  # (B, 1, 34, 60)

# GT flow at fine resolution (34×60)
gt_flow = model.compute_gt_flow(pose_init, pose_gt, depth, (34, 60))

print(f"  Pred flow shape: {pred_flow.shape}, range: [{pred_flow.min():.2f}, {pred_flow.max():.2f}]")
print(f"  GT flow shape:   {gt_flow.shape}, range: [{gt_flow.min():.2f}, {gt_flow.max():.2f}]")
print(f"  Confidence:      range: [{conf.min():.3f}, {conf.max():.3f}], mean={conf.mean():.3f}")

# Per-sample flow error
for b in range(min(4, query_feat.shape[0])):
    diff = (pred_flow[b] - gt_flow[b])
    epe = diff.norm(dim=0).mean()  # End-point error
    pred_mag = pred_flow[b].norm(dim=0).mean()
    gt_mag = gt_flow[b].norm(dim=0).mean()
    # Correlation
    corr_u = torch.corrcoef(torch.stack([pred_flow[b,0].flatten(), gt_flow[b,0].flatten()]))[0,1]
    corr_v = torch.corrcoef(torch.stack([pred_flow[b,1].flatten(), gt_flow[b,1].flatten()]))[0,1]
    print(f"  Sample {b}: EPE={epe:.3f}px, pred_mag={pred_mag:.3f}, gt_mag={gt_mag:.3f}, "
          f"corr_u={corr_u:.3f}, corr_v={corr_v:.3f}")

# ================================================================
# Test 4: Solver with GT flow vs predicted flow
# ================================================================
print(f"\n{'='*70}")
print("TEST 4: Solver output with GT flow vs predicted flow")
print(f"{'='*70}")

from modules.geometry_solver import compute_image_jacobian, diff_pose_solve, diff_pose_solve_sequential

# Upscale flow to solver_hw
import torch.nn.functional as F
solver_hw = tuple(mc.get('solver_hw', [68, 120]))

# GT delta_xi
T_gt_delta = pose_gt.float() @ torch.linalg.inv(pose_init.float())
xi_gt = se3_log(T_gt_delta)

# Predicted flow at solver resolution
pred_flow_s = F.interpolate(pred_flow.float(), size=solver_hw, mode='bilinear', align_corners=False)
pred_flow_s[:, 0] *= solver_hw[1] / pred_flow.shape[-1]
pred_flow_s[:, 1] *= solver_hw[0] / pred_flow.shape[-2]

# GT flow at solver resolution
gt_flow_s = model.compute_gt_flow(pose_init, pose_gt, depth, solver_hw)

# Depth at solver resolution
depth_s = F.interpolate(depth.unsqueeze(1).float(), size=solver_hw, mode='bilinear', align_corners=False).squeeze(1)

# Intrinsics at solver resolution
s_intr = model._scale_intrinsics(solver_hw[0], solver_hw[1])

Ju, Jv, valid = compute_image_jacobian(depth_s, s_intr)

# Uniform confidence
uniform_conf = torch.ones(query_feat.shape[0], 1, solver_hw[0], solver_hw[1], device=device)

# Solve with GT flow + uniform conf
xi_gt_flow = diff_pose_solve_sequential(gt_flow_s, uniform_conf, Ju, Jv, valid, damping=0.001)

# Solve with pred flow + uniform conf
xi_pred_flow = diff_pose_solve_sequential(pred_flow_s, uniform_conf, Ju, Jv, valid, damping=0.001)

# Solve with pred flow + predicted conf
conf_s = F.interpolate(conf.float(), size=solver_hw, mode='bilinear', align_corners=False)
xi_pred_conf = diff_pose_solve_sequential(pred_flow_s, conf_s, Ju, Jv, valid, damping=0.001)

print(f"\n  {'Sample':<8} {'xi_gt rot(°)':<15} {'GTflow rot(°)':<15} {'Pred+uni rot(°)':<18} {'Pred+conf rot(°)':<18}")
print("  " + "-" * 75)
for b in range(min(4, query_feat.shape[0])):
    gt_r = xi_gt[b, 3:].norm().item() * 180 / math.pi
    gtf_r = xi_gt_flow[b, 3:].norm().item() * 180 / math.pi
    pu_r = xi_pred_flow[b, 3:].norm().item() * 180 / math.pi
    pc_r = xi_pred_conf[b, 3:].norm().item() * 180 / math.pi
    # Direction check: cosine similarity
    cos_gtf = F.cosine_similarity(xi_gt[b:b+1, 3:], xi_gt_flow[b:b+1, 3:])
    cos_pu = F.cosine_similarity(xi_gt[b:b+1, 3:], xi_pred_flow[b:b+1, 3:])
    print(f"  {b:<8} {gt_r:<15.4f} {gtf_r:<15.4f} {pu_r:<18.4f} {pc_r:<18.4f}  "
          f"cos(gt,gtflow)={cos_gtf.item():.3f} cos(gt,pred)={cos_pu.item():.3f}")

print(f"\n  {'Sample':<8} {'xi_gt trans(cm)':<18} {'GTflow trans(cm)':<18} {'Pred+uni trans(cm)':<21} {'Pred+conf trans(cm)':<21}")
print("  " + "-" * 90)
for b in range(min(4, query_feat.shape[0])):
    gt_t = xi_gt[b, :3].norm().item() * 100
    gtf_t = xi_gt_flow[b, :3].norm().item() * 100
    pu_t = xi_pred_flow[b, :3].norm().item() * 100
    pc_t = xi_pred_conf[b, :3].norm().item() * 100
    print(f"  {b:<8} {gt_t:<18.2f} {gtf_t:<18.2f} {pu_t:<21.2f} {pc_t:<21.2f}")

print("\nDone!")
