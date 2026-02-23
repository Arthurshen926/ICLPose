#!/usr/bin/env python3
"""诊断 NaN: 逐步检查 V3 管线的数值稳定性"""
import sys, os, torch, numpy as np
sys.path.insert(0, '/home/yons/Projects/ICLPose')
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

device = torch.device('cuda')

# 1. 加载一个特征，检查是否有 NaN/Inf
print("=== 1. 检查查询特征 ===")
for scale, fname in [
    ('coarse', 'rgb_0_coarse_1280x7x10.pt'),
    ('fine_sd', 'rgb_0_fine_sd_640x35x46.pt'),
    ('fine_dino', 'rgb_0_fine_dino_768x35x46.pt'),
]:
    f = torch.load(f'output/features_multiscale/room_0/{scale}/{fname}', map_location='cpu', weights_only=True)
    has_nan = f.isnan().any().item()
    has_inf = f.isinf().any().item()
    print(f"  {scale}: shape={f.shape}, range=[{f.min():.4f}, {f.max():.4f}], nan={has_nan}, inf={has_inf}")

# 2. 加载渲染器并渲染
print("\n=== 2. 测试渲染 ===")
from modules.multiscale_renderer import MultiScaleRenderer

renderer = MultiScaleRenderer(
    ply_path='dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'coarse': 'output/feature_3dgs/room_0_raw/coarse/best_model.pth',
        'fine_sd': 'output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device='cuda',
)

# 用第一帧的 GT 位姿渲染
poses_c2w = np.loadtxt('dataset/room_0/Sequence_1/traj_w_c.txt')
c2w = poses_c2w[0].reshape(4, 4)
R, t = c2w[:3, :3], c2w[:3, 3]
w2c = np.eye(4, dtype=np.float32)
w2c[:3, :3] = R.T
w2c[:3, 3] = -R.T @ t
pose_w2c = torch.from_numpy(w2c).float().to(device)

for name in renderer.models.keys():
    r = renderer.render_scale(name, pose_w2c)
    fm = r['feature_map']
    has_nan = fm.isnan().any().item()
    has_inf = fm.isinf().any().item()
    print(f"  {name}: rendered shape={fm.shape}, range=[{fm.min():.4f}, {fm.max():.4f}], nan={has_nan}, inf={has_inf}")
    if has_nan:
        nan_count = fm.isnan().sum().item()
        total = fm.numel()
        print(f"    *** NaN count: {nan_count}/{total} ({100*nan_count/total:.1f}%)")

# 3. 完整前向传播
print("\n=== 3. 单步前向传播 ===")
from ic_models.ic_pose_net_v3 import ICPoseNetV3

scale_configs = []
for name in renderer.models.keys():
    info = renderer.scale_info[name]
    scale_configs.append({'name': name, 'feat_dim': info['feat_dim'], 'resolution': info['resolution']})

model = ICPoseNetV3(scale_configs=scale_configs, hidden_dim=128, num_iters=1).to(device)

query_feats = {}
for scale, fname in [
    ('coarse', 'rgb_0_coarse_1280x7x10.pt'),
    ('fine_sd', 'rgb_0_fine_sd_640x35x46.pt'),
    ('fine_dino', 'rgb_0_fine_dino_768x35x46.pt'),
]:
    f = torch.load(f'output/features_multiscale/room_0/{scale}/{fname}', map_location='cpu', weights_only=True)
    query_feats[scale] = f.unsqueeze(0).to(device)

initial_pose = pose_w2c.unsqueeze(0)

# 前向 1 步
with torch.cuda.amp.autocast(enabled=False):
    preds = model(
        query_feats=query_feats,
        initial_pose=initial_pose,
        renderer=renderer,
        num_iters=1,
    )

for k, v in preds.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: shape={v.shape}, nan={v.isnan().any()}, range=[{v.min():.4f}, {v.max():.4f}]")
    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
        for i, vi in enumerate(v):
            print(f"  {k}[{i}]: shape={vi.shape}, nan={vi.isnan().any()}, range=[{vi.min():.4f}, {vi.max():.4f}]")
    elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], dict):
        for i, vi in enumerate(v):
            gates_str = ', '.join(f'{gn}={gv:.4f}' for gn, gv in vi.items())
            print(f"  {k}[{i}]: {gates_str}")

# 4. Sequence loss
print("\n=== 4. Loss 计算 ===")
from losses.sequence_loss import PoseOnlySequenceLoss, rotation_geodesic_loss
loss_fn = PoseOnlySequenceLoss(gamma=0.8, lambda_translation=1.0)
loss_dict = loss_fn(preds, initial_pose)  # GT = initial pose (near-zero error)
for k, v in loss_dict.items():
    if isinstance(v, torch.Tensor):
        print(f"  {k}: {v.item():.6f}, nan={v.isnan().item()}")

# 5. 带扰动的位姿
print("\n=== 5. 带扰动位姿前向传播 ===")
from data.dataset_v3 import perturb_pose
perturbed = perturb_pose(pose_w2c.cpu(), noise_rot_deg=15.0, noise_trans_m=0.5).unsqueeze(0).to(device)
print(f"  Perturbed pose valid: nan={perturbed.isnan().any()}")

preds2 = model(query_feats=query_feats, initial_pose=perturbed, renderer=renderer, num_iters=2)
final = preds2['final_pose']
print(f"  Final pose: nan={final.isnan().any()}, range=[{final.min():.4f}, {final.max():.4f}]")
for i, xi in enumerate(preds2['xi_list']):
    print(f"  xi[{i}]: {xi[0].tolist()}, nan={xi.isnan().any()}")

loss_dict2 = loss_fn(preds2, initial_pose)
print(f"  Loss: {loss_dict2['total_loss'].item():.6f}, nan={loss_dict2['total_loss'].isnan().item()}")

print("\n=== 诊断完成 ===")
