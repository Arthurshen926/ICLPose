#!/usr/bin/env python3
"""测试更大的batch_size"""
import torch, time, sys
sys.path.insert(0, '/home/yons/Projects/ICLPose')
from modules.multiscale_renderer import MultiScaleRenderer
from ic_models.corr_pose_net import CorrPoseNet

device = torch.device('cuda:0')
print('Loading renderer...')
ply_path = 'dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply'
scale_paths = {'fine_dino': 'output/feature_3dgs/room_0_raw/fine_dino/best_model.pth'}
renderer = MultiScaleRenderer(ply_path=ply_path, scale_model_paths=scale_paths, device='cuda:0')
feat_dim = 768; fH, fW = 35, 46
INTRINSICS = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}
net = CorrPoseNet(feat_dim=feat_dim, enc_dim=128, hidden_dim=128, 
                  corr_radius=4, num_iters=3, damping=1e-3).to(device)
net.train()

for BS in [16, 32, 64]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    query = torch.randn(BS, feat_dim, fH, fW, device=device)
    depth = torch.ones(BS, fH, fW, device=device) * 1.5
    pose_gt = torch.eye(4, device=device).unsqueeze(0).expand(BS, -1, -1).clone()
    initial_pose = pose_gt.clone()
    initial_pose[:, :3, 3] += torch.randn(BS, 3, device=device) * 0.1
    try:
        results = net(query, initial_pose, depth, INTRINSICS, renderer, 'fine_dino')
        loss = sum(((p[:,:3,3]-pose_gt[:,:3,3])**2).sum() for p in results['poses'][1:])
        loss.backward(); net.zero_grad(); torch.cuda.synchronize()
        
        times = []
        for _ in range(3):
            torch.cuda.synchronize(); t0 = time.time()
            results = net(query, initial_pose, depth, INTRINSICS, renderer, 'fine_dino')
            loss = sum(((p[:,:3,3]-pose_gt[:,:3,3])**2).sum() for p in results['poses'][1:])
            loss.backward(); torch.cuda.synchronize(); times.append(time.time()-t0); net.zero_grad()
        
        mem = torch.cuda.max_memory_allocated()/1024**3
        avg = sum(times)/len(times)
        print(f'BS={BS}: total={avg*1000:.0f}ms, per_sample={avg*1000/BS:.0f}ms, mem={mem:.1f}GB, throughput={BS/avg:.1f}/s')
    except RuntimeError as e:
        mem = torch.cuda.max_memory_allocated()/1024**3
        print(f'BS={BS}: FAILED ({mem:.1f}GB) — {str(e)[:80]}')
        break
