"""Quick 10-step test to verify PoseHead init fix."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from ic_models.ic_pose_net_v3 import ICPoseNetV3
from modules.multiscale_renderer import MultiScaleRenderer
from losses.sequence_loss import PoseOnlySequenceLoss
from data.dataset_v3 import PoseDatasetV3, collate_v3

device = torch.device('cuda:0')
base = '/home/yons/Projects/ICLPose'

renderer = MultiScaleRenderer(
    ply_path=base + '/dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply',
    scale_model_paths={
        'fine_sd': base + '/output/feature_3dgs/room_0_raw/fine_sd/best_model.pth',
        'fine_dino': base + '/output/feature_3dgs/room_0_raw/fine_dino/best_model.pth',
    },
    device=device,
)

scale_configs = [
    {'name': n, 'feat_dim': renderer.scale_info[n]['feat_dim'], 'resolution': renderer.scale_info[n]['resolution']}
    for n in ['fine_sd', 'fine_dino']
]

model = ICPoseNetV3(scale_configs=scale_configs, hidden_dim=128, output_resolution=(35, 46), num_iters=3).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
criterion = PoseOnlySequenceLoss(gamma=0.8, lambda_translation=0.5)

ds = PoseDatasetV3(
    feature_base_dir=base + '/output/features_multiscale/room_0',
    traj_path=base + '/dataset/room_0/Sequence_1/traj_w_c.txt',
    frame_indices=list(range(5)),
    scale_names=['fine_sd', 'fine_dino'],
    is_train=True, noise_rot_deg=15.0, noise_trans_m=0.3,
)
batch = collate_v3([ds[0]])
qf = {k: v.to(device) for k, v in batch['query_feats'].items()}
ip = batch['initial_pose'].to(device)
gt = batch['pose_gt'].to(device)

model.train()
print('--- Quick 10-step test (fixed PoseHead init) ---')
for step in range(10):
    optimizer.zero_grad()
    out = model(query_feats=qf, initial_pose=ip, renderer=renderer, num_iters=3)
    ld = criterion(out, gt)
    loss = ld['total_loss']
    if torch.isnan(loss):
        print(f'  Step {step}: NaN!')
        break
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    xi_max = out['xi_list'][-1].abs().max().item()
    rot_err = ld['final_rotation_error_deg'].item()
    trans_err = ld['final_translation_error_m'].item()
    print(f'  Step {step}: loss={loss.item():.4f} rot={rot_err:.2f} trans={trans_err:.3f}m gn={gn:.2f} xi={xi_max:.6f}')

print('Done!')
