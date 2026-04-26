#!/usr/bin/env python3
"""
Test: Replace coarse flow with soft-argmax of global correlation at inference time.
Instead of using the model's coarse GRU (which outputs ~zero), compute
soft-argmax of the global correlation volume to get an initial flow estimate.
"""
import argparse, math, os, re, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_models.pose_refiner import PoseRefiner, global_correlation, guided_local_correlation
from modules.lie_algebra import se3_exp
from modules.geometry_solver import compute_image_jacobian, diff_pose_solve_sequential
from feature_3dgs.gaussian_feature_model import GaussianFeatureModel
from feature_3dgs.feature_renderer import FeatureRenderer


class RadioRenderer:
    def __init__(s, ply_path, feature_model_path, device,
                 img_hw=(1080,1920), fx=1663.12, fy=1663.12, cx=960.0, cy=540.0):
        s.device=device; s.img_hw=img_hw; s.fx=fx; s.fy=fy; s.cx=cx; s.cy=cy
        ckpt=torch.load(feature_model_path, map_location='cpu', weights_only=True)
        s.gs_model=GaussianFeatureModel(feature_dim=ckpt['feature_dim'])
        s.gs_model.load_ply(ply_path)
        s.gs_model._loc_feature=nn.Parameter(ckpt['loc_feature'].to(device))
        s.gs_model=s.gs_model.to(device); s.gs_model.eval()
    def render_features(s, viewmat, feat_hw=(68,120)):
        if viewmat.dim()==2: viewmat=viewmat.unsqueeze(0)
        fH,fW=feat_hw
        r=FeatureRenderer.render_features_batch(s.gs_model,viewmat,
            fx=s.fx*fW/s.img_hw[1],fy=s.fy*fH/s.img_hw[0],
            cx=s.cx*fW/s.img_hw[1],cy=s.cy*fH/s.img_hw[0],
            img_height=fH,img_width=fW,
            norm_feat_before_render=True,norm_feat_after_render=True)
        return r['feature_map']
    def render_depth(s, viewmat, depth_hw=(68,120)):
        if viewmat.dim()==2: viewmat=viewmat.unsqueeze(0)
        dH,dW=depth_hw; depths=[]
        for i in range(viewmat.shape[0]):
            d=FeatureRenderer.render_depth(s.gs_model,viewmat[i],
                fx=s.fx*dW/s.img_hw[1],fy=s.fy*dH/s.img_hw[0],
                cx=s.cx*dW/s.img_hw[1],cy=s.cy*dH/s.img_hw[0],
                img_height=dH,img_width=dW)
            depths.append(d)
        return torch.stack(depths)


def add_noise_deterministic(pose_w2c, rot_deg, trans_m, seed):
    device=pose_w2c.device; rng=torch.Generator(); rng.manual_seed(seed)
    axis=torch.randn(3,generator=rng); axis=axis/(axis.norm()+1e-8)
    omega=axis*(rot_deg*math.pi/180.0)
    direction=torch.randn(3,generator=rng); direction=direction/(direction.norm()+1e-8)
    xi=torch.cat([direction*trans_m,omega]).to(device)
    return se3_exp(xi)@pose_w2c


def pose_error(pred_w2c, gt_w2c):
    p_c2w=torch.inverse(pred_w2c); g_c2w=torch.inverse(gt_w2c)
    pos_err=(p_c2w[:3,3]-g_c2w[:3,3]).norm().item()*100
    trace=(p_c2w[:3,:3]@g_c2w[:3,:3].T).diagonal().sum()
    cos_a=((trace-1)/2).clamp(-1,1)
    rot_err=torch.acos(cos_a).item()*180/math.pi
    return pos_err, rot_err


def soft_argmax_flow(corr, H_r, W_r, temperature=1.0):
    """
    Compute soft-argmax flow from global correlation volume.
    
    corr: (B, Hr*Wr, Hq, Wq)
    Returns: (B, 2, Hq, Wq) flow from query to reference
    """
    B, HrWr, Hq, Wq = corr.shape
    device = corr.device
    
    # Create reference coordinate grid
    ry = torch.arange(H_r, device=device, dtype=torch.float32)
    rx = torch.arange(W_r, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ry, rx, indexing='ij')
    coords_x = grid_x.reshape(-1)  # (Hr*Wr,)
    coords_y = grid_y.reshape(-1)  # (Hr*Wr,)
    
    # Create query coordinate grid
    qy = torch.arange(Hq, device=device, dtype=torch.float32)
    qx = torch.arange(Wq, device=device, dtype=torch.float32)
    qgrid_y, qgrid_x = torch.meshgrid(qy, qx, indexing='ij')
    
    # Softmax over reference positions for each query pixel
    weights = F.softmax(corr / temperature, dim=1)  # (B, Hr*Wr, Hq, Wq)
    
    # Weighted sum of reference coordinates
    # Expected ref x for each query pixel
    exp_x = (weights * coords_x.view(1, -1, 1, 1)).sum(dim=1)  # (B, Hq, Wq)
    exp_y = (weights * coords_y.view(1, -1, 1, 1)).sum(dim=1)  # (B, Hq, Wq)
    
    # Flow = expected ref position - query position
    flow_x = exp_x - qgrid_x.unsqueeze(0)
    flow_y = exp_y - qgrid_y.unsqueeze(0)
    
    return torch.stack([flow_x, flow_y], dim=1)


@torch.no_grad()
def evaluate(config, checkpoint, noise_degs, temperatures, max_iters, device='cuda'):
    rc=config['renderer']; mc=config.get('model',{})
    feat_hw=tuple(rc.get('feat_hw',[68,120]))
    intrinsics={'fx':rc['fx'],'fy':rc['fy'],'cx':rc['cx'],'cy':rc['cy']}
    model=PoseRefiner(
        in_dim=mc.get('in_dim',64), match_dim=mc.get('match_dim',64),
        hidden_dim=mc.get('hidden_dim',128), n_heads=mc.get('n_heads',4),
        n_attn_layers=mc.get('n_attn_layers',2), ffn_dim=mc.get('ffn_dim',128),
        local_radius=mc.get('local_radius',4), fine_iters=mc.get('fine_iters',8),
        damping=mc.get('damping',1e-3),
        coarse_hw=tuple(mc.get('coarse_hw',[17,30])),
        fine_hw=tuple(mc.get('fine_hw',[34,60])),
        solver_upsample=mc.get('solver_upsample',4),
        solver_hw=tuple(mc['solver_hw']) if 'solver_hw' in mc else None,
        intrinsics=intrinsics, img_hw=tuple(rc.get('img_hw',[1080,1920])),
        depth_normalize=mc.get('depth_normalize',True),
        sequential_solve=mc.get('sequential_solve',True),
        detach_conf=mc.get('detach_conf',True),
        conf_floor=mc.get('conf_floor',0.1),
        use_trans_head=mc.get('use_trans_head',False),
        trans_head_mode=mc.get('trans_head_mode','replace'),
        solver_trans_scale=mc.get('solver_trans_scale',0.0),
    ).to(device)
    ckpt_data=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(ckpt_data.get('model_state_dict',ckpt_data),strict=False)
    model.eval()
    print(f"Loaded: epoch={ckpt_data.get('epoch','?')}")
    
    COARSE_HW = model.COARSE_HW
    FINE_HW = model.FINE_HW
    print(f"Coarse: {COARSE_HW}, Fine: {FINE_HW}")

    renderer=RadioRenderer(rc['ply_path'],rc['feature_model_path'],device,
        tuple(rc.get('img_hw',[1080,1920])),rc['fx'],rc['fy'],rc['cx'],rc['cy'])
    dc=config['data']
    feature_dir=Path(dc['feature_dir'])/'fine_radio'
    pattern=re.compile(r'rgb_(\d+)_fine_radio_(\d+)x(\d+)x(\d+)\.pt')
    features={}
    for f in sorted(feature_dir.iterdir()):
        m=pattern.match(f.name)
        if m: features[int(m.group(1))]=f
    poses_c2w=[]
    with open(dc['traj_path']) as f:
        for line in f:
            vals=list(map(float,line.strip().split()))
            if len(vals)==16: poses_c2w.append(np.array(vals).reshape(4,4))
    test_idx_path=os.path.join(dc['feature_dir'],'test_indices.npy')
    test_indices=np.load(test_idx_path).tolist() if os.path.exists(test_idx_path) else sorted(features.keys())
    test_indices=[i for i in test_indices if i in features and i<len(poses_c2w)]
    print(f"Test: {len(test_indices)} frames")

    for noise_deg in noise_degs:
        noise_trans=noise_deg/8.0*0.25
        print(f"\n{'='*70}")
        print(f"  Noise {noise_deg}° / {noise_trans:.2f}m")
        print(f"{'='*70}")

        # Baseline: original model
        per_iter_rot_orig = {i: [] for i in range(max_iters+1)}
        for idx_i, frame_idx in enumerate(test_indices):
            feat=torch.load(str(features[frame_idx]),map_location=device,weights_only=True).float().unsqueeze(0)
            c2w=torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c=torch.linalg.inv(c2w).to(device)
            seed=frame_idx*1000+int(noise_deg*100)
            noisy_w2c=add_noise_deterministic(gt_w2c,noise_deg,noise_trans,seed)
            pose_cur=noisy_w2c.unsqueeze(0)
            _,rot_err=pose_error(pose_cur[0],gt_w2c)
            per_iter_rot_orig[0].append(rot_err)
            for it in range(1, max_iters+1):
                ref_feat=renderer.render_features(pose_cur,feat_hw)
                depth=renderer.render_depth(pose_cur,feat_hw)
                result=model(feat.to(device),ref_feat,depth)
                if 'delta_xi' in result:
                    pose_cur=se3_exp(result['delta_xi'])@pose_cur
                _,rot_err=pose_error(pose_cur[0],gt_w2c)
                per_iter_rot_orig[it].append(rot_err)
        
        print(f"\n  [Baseline - original model]")
        detail = ' → '.join(f'I{i}={np.median(per_iter_rot_orig[i]):.2f}°' for i in range(max_iters+1))
        print(f"    {detail}")

        # Test soft-argmax coarse flow initialization
        for temp in temperatures:
            per_iter_rot = {i: [] for i in range(max_iters+1)}
            
            for idx_i, frame_idx in enumerate(test_indices):
                feat=torch.load(str(features[frame_idx]),map_location=device,weights_only=True).float().unsqueeze(0)
                c2w=torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
                gt_w2c=torch.linalg.inv(c2w).to(device)
                seed=frame_idx*1000+int(noise_deg*100)
                noisy_w2c=add_noise_deterministic(gt_w2c,noise_deg,noise_trans,seed)
                pose_cur=noisy_w2c.unsqueeze(0)
                _,rot_err=pose_error(pose_cur[0],gt_w2c)
                per_iter_rot[0].append(rot_err)
                
                for it in range(1, max_iters+1):
                    ref_feat=renderer.render_features(pose_cur,feat_hw)
                    depth=renderer.render_depth(pose_cur,feat_hw)
                    
                    # Manually run the pipeline with soft-argmax coarse flow
                    q = F.normalize(feat.to(device), dim=1)
                    r = F.normalize(ref_feat, dim=1)
                    q_coarse = model.projection(F.interpolate(q, COARSE_HW, mode='bilinear', align_corners=False))
                    r_coarse = model.projection(F.interpolate(r, COARSE_HW, mode='bilinear', align_corners=False))
                    q_fine = model.projection(F.interpolate(q, FINE_HW, mode='bilinear', align_corners=False))
                    r_fine = model.projection(F.interpolate(r, FINE_HW, mode='bilinear', align_corners=False))
                    
                    q_enh, r_enh = model.cross_attn(q_coarse, r_coarse)
                    coarse_corr = global_correlation(q_enh, r_enh)
                    
                    # SOFT-ARGMAX instead of GRU coarse head
                    Hr, Wr = COARSE_HW
                    flow_c = soft_argmax_flow(coarse_corr, Hr, Wr, temperature=temp)
                    
                    # Upsample to fine
                    scale_x = FINE_HW[1] / COARSE_HW[1]
                    scale_y = FINE_HW[0] / COARSE_HW[0]
                    flow_f = F.interpolate(flow_c, FINE_HW, mode='bilinear', align_corners=False)
                    flow_f[:, 0] *= scale_x
                    flow_f[:, 1] *= scale_y
                    
                    # Use model's coarse head for hidden/conf (still need the hidden state)
                    h_coarse = model.context_net(q_coarse)
                    flow_c_dummy = torch.zeros(1, 2, *COARSE_HW, device=device)
                    conf_c = torch.ones(1, 1, *COARSE_HW, device=device) * 0.5
                    _, conf_c, h_coarse, _ = model.coarse_head(coarse_corr, h_coarse, flow_c_dummy, conf_c)
                    
                    conf_f = F.interpolate(conf_c, FINE_HW, mode='bilinear', align_corners=False)
                    h_fine_up = F.interpolate(h_coarse, FINE_HW, mode='bilinear', align_corners=False)
                    h_fine = model.fine_context(h_fine_up, q_fine)
                    
                    # Fine GRU iterations with soft-argmax-initialized flow
                    for _ in range(model.fine_iters):
                        fine_corr = guided_local_correlation(
                            q_fine, r_fine, flow_f, radius=model.local_radius)
                        _, conf_f, h_fine, flow_f = model.fine_head(
                            fine_corr, h_fine, flow_f, conf_f)
                    
                    # Solver
                    sH, sW = model.SOLVER_HW
                    with torch.cuda.amp.autocast(enabled=False):
                        flow_f32 = flow_f.float()
                        conf_f32 = conf_f.float()
                        depth_f32 = depth.float()
                        
                        if (sH, sW) != FINE_HW:
                            sx = sW / FINE_HW[1]; sy = sH / FINE_HW[0]
                            solve_flow = F.interpolate(flow_f32, (sH, sW), mode='bilinear', align_corners=False)
                            solve_flow[:, 0] *= sx; solve_flow[:, 1] *= sy
                            solve_conf = F.interpolate(conf_f32, (sH, sW), mode='bilinear', align_corners=False)
                        else:
                            solve_flow = flow_f32; solve_conf = conf_f32
                        
                        if depth_f32.ndim == 3: depth_f32 = depth_f32.unsqueeze(1)
                        depth_solve = F.interpolate(depth_f32, (sH, sW), mode='bilinear', align_corners=False).squeeze(1)
                        
                        solver_intrinsics = model._scale_intrinsics(sH, sW)
                        Ju, Jv, valid = compute_image_jacobian(depth_solve, solver_intrinsics)
                        conf_det = solve_conf.detach()
                        
                        if model.sequential_solve:
                            delta_xi = diff_pose_solve_sequential(solve_flow, conf_det, Ju, Jv, valid, damping=model.damping)
                        else:
                            from modules.geometry_solver import diff_pose_solve
                            delta_xi = diff_pose_solve(solve_flow, conf_det, Ju, Jv, valid, damping=model.damping)
                        
                        if model.solver_trans_scale != 1.0:
                            delta_xi = torch.cat([delta_xi[:, :3] * model.solver_trans_scale, delta_xi[:, 3:]], dim=1)
                    
                    pose_cur = se3_exp(delta_xi) @ pose_cur
                    _, rot_err = pose_error(pose_cur[0], gt_w2c)
                    per_iter_rot[it].append(rot_err)
            
            detail = ' → '.join(f'I{i}={np.median(per_iter_rot[i]):.2f}°' for i in range(max_iters+1))
            print(f"\n  [Soft-argmax T={temp}]")
            print(f"    {detail}")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--noise_deg',nargs='+',type=float,default=[8,5,3])
    parser.add_argument('--temperatures',nargs='+',type=float,default=[1.0, 5.0, 10.0, 20.0])
    parser.add_argument('--max_iters',type=int,default=5)
    args=parser.parse_args()
    with open(args.config) as f: config=yaml.safe_load(f)
    evaluate(config,args.checkpoint,args.noise_deg,args.temperatures,args.max_iters)
