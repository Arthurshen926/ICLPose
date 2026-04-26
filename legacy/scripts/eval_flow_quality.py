#!/usr/bin/env python3
"""
Diagnostic: measure flow prediction quality vs ground truth.
Reports EPE (end-point error) per iteration, and the theoretical
pose error if the flow were perfect.
"""
import argparse, math, os, re, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_models.pose_refiner import PoseRefiner
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


@torch.no_grad()
def evaluate(config, checkpoint, noise_degs, max_iters, device='cuda'):
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
    ckpt=torch.load(checkpoint,map_location=device,weights_only=False)
    model.load_state_dict(ckpt.get('model_state_dict',ckpt),strict=False)
    model.eval()
    print(f"Loaded: epoch={ckpt.get('epoch','?')}")
    
    # Get solver resolution
    solver_hw = model.SOLVER_HW
    fine_hw_model = model.FINE_HW
    print(f"Fine HW: {fine_hw_model}, Solver HW: {solver_hw}")

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

        # Collect per-iter stats
        epe_fine = {i: [] for i in range(1, max_iters+1)}  # EPE at fine resolution
        epe_coarse = {i: [] for i in range(1, max_iters+1)}  # EPE at coarse resolution
        epe_solver = {i: [] for i in range(1, max_iters+1)}  # EPE at solver resolution
        rot_pred = {i: [] for i in range(max_iters+1)}
        rot_gtflow = {i: [] for i in range(1, max_iters+1)}  # pose if GT flow used
        gt_flow_mag = {i: [] for i in range(1, max_iters+1)}  # GT flow magnitude 
        pred_flow_mag = {i: [] for i in range(1, max_iters+1)}  # pred flow magnitude
        coarse_flow_mag = {i: [] for i in range(1, max_iters+1)}

        for idx_i, frame_idx in enumerate(test_indices):
            feat=torch.load(str(features[frame_idx]),map_location=device,weights_only=True).float().unsqueeze(0)
            c2w=torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c=torch.linalg.inv(c2w).to(device)
            seed=frame_idx*1000+int(noise_deg*100)
            noisy_w2c=add_noise_deterministic(gt_w2c,noise_deg,noise_trans,seed)
            pose_cur=noisy_w2c.unsqueeze(0)
            
            _,rot_err=pose_error(pose_cur[0],gt_w2c)
            rot_pred[0].append(rot_err)

            for it in range(1, max_iters+1):
                ref_feat=renderer.render_features(pose_cur,feat_hw)
                depth=renderer.render_depth(pose_cur,feat_hw)
                result=model(feat.to(device),ref_feat,depth)
                
                # Get predicted flow at fine resolution
                pred_flow = result['flow_fine']  # (B,2,H,W)
                
                # Also get coarse flow 
                coarse_flow = result.get('flow_coarse', None)
                
                # Compute GT flow at fine resolution
                gt_flow, gt_mask = model.compute_gt_flow(
                    pose_cur, gt_w2c.unsqueeze(0), depth, fine_hw_model)
                
                # EPE at fine resolution
                diff = (pred_flow - gt_flow) * gt_mask
                epe = (diff**2).sum(dim=1, keepdim=True).sqrt()  # (B,1,H,W)
                valid_epe = epe[gt_mask.expand_as(epe) > 0]
                epe_fine[it].append(valid_epe.mean().item() if len(valid_epe) > 0 else 0)
                
                # Coarse flow EPE (upsample to fine and compare)
                if coarse_flow is not None:
                    cH, cW = coarse_flow.shape[2:]
                    fH, fW = fine_hw_model
                    coarse_up = F.interpolate(coarse_flow, (fH, fW), mode='bilinear', align_corners=False)
                    coarse_up[:, 0] *= fW / cW
                    coarse_up[:, 1] *= fH / cH
                    diff_c = (coarse_up - gt_flow) * gt_mask
                    epe_c = (diff_c**2).sum(dim=1, keepdim=True).sqrt()
                    valid_epe_c = epe_c[gt_mask.expand_as(epe_c) > 0]
                    epe_coarse[it].append(valid_epe_c.mean().item() if len(valid_epe_c) > 0 else 0)
                    cm = (coarse_up**2).sum(dim=1, keepdim=True).sqrt()
                    cv = cm[gt_mask.expand_as(cm) > 0]
                    coarse_flow_mag[it].append(cv.mean().item() if len(cv) > 0 else 0)
                else:
                    epe_coarse[it].append(0)
                    coarse_flow_mag[it].append(0)
                
                # Flow magnitudes
                gt_mag = (gt_flow**2).sum(dim=1, keepdim=True).sqrt()
                pred_mag = (pred_flow**2).sum(dim=1, keepdim=True).sqrt()
                gt_valid = gt_mag[gt_mask.expand_as(gt_mag) > 0]
                pred_valid = pred_mag[gt_mask.expand_as(pred_mag) > 0]
                gt_flow_mag[it].append(gt_valid.mean().item() if len(gt_valid) > 0 else 0)
                pred_flow_mag[it].append(pred_valid.mean().item() if len(pred_valid) > 0 else 0)
                
                # Use predicted flow → pose update
                if 'delta_xi' in result:
                    pose_cur_next = se3_exp(result['delta_xi']) @ pose_cur
                    _,rot_err=pose_error(pose_cur_next[0],gt_w2c)
                    rot_pred[it].append(rot_err)
                    
                    # Use GT flow → solve pose, see theoretical limit
                    # Upsample GT flow to solver resolution and run solver
                    sH, sW = solver_hw
                    gt_flow_up = F.interpolate(gt_flow, (sH, sW), mode='bilinear', align_corners=False)
                    gt_flow_up[:, 0] *= sW / fine_hw_model[1]
                    gt_flow_up[:, 1] *= sH / fine_hw_model[0]
                    gt_mask_up = F.interpolate(gt_mask, (sH, sW), mode='nearest')
                    
                    # Get depth at solver resolution  
                    depth_up = F.interpolate(depth.unsqueeze(1), (sH, sW), mode='bilinear', align_corners=False).squeeze(1)
                    
                    # Solver with GT flow
                    intr = model._scale_intrinsics(sH, sW)
                    Ju, Jv, valid = compute_image_jacobian(depth_up, intr)
                    # Use uniform confidence for GT flow
                    conf_gt = gt_mask_up  # (B, 1, sH, sW)
                    xi_gt = diff_pose_solve_sequential(
                        gt_flow_up, conf_gt, Ju, Jv, valid, damping=model.damping)
                    pose_gt_flow = se3_exp(xi_gt) @ pose_cur
                    _,rot_err_gt=pose_error(pose_gt_flow[0],gt_w2c)
                    rot_gtflow[it].append(rot_err_gt)
                    
                    pose_cur = pose_cur_next  # Update for next iter
                else:
                    rot_pred[it].append(rot_pred[it-1][-1])
                    rot_gtflow[it].append(rot_pred[it-1][-1])
                    # No update

        # Report
        print(f"\n  {'Iter':>4} | {'Med Rot':>8} | {'GT-Flow':>8} | {'EPE_f':>7} | {'EPE_c':>7} | {'GT |F|':>7} | {'Pred |F|':>8} | {'Coarse|F|':>10} | {'EPE/|F|':>7}")
        print(f"  {'-'*4}-+-{'-'*8}-+-{'-'*8}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*8}-+-{'-'*10}-+-{'-'*7}")
        print(f"  {'I0':>4} | {np.median(rot_pred[0]):>7.2f}° | {'':>8} | {'':>7} | {'':>7} | {'':>7} | {'':>8} | {'':>10} | {'':>7}")
        for it in range(1, max_iters+1):
            med_rot = np.median(rot_pred[it])
            med_gtflow = np.median(rot_gtflow[it])
            med_epe = np.median(epe_fine[it])
            med_epe_c = np.median(epe_coarse[it])
            med_gt_mag = np.median(gt_flow_mag[it])
            med_pred_mag = np.median(pred_flow_mag[it])
            med_coarse_mag = np.median(coarse_flow_mag[it])
            rel_epe = med_epe / med_gt_mag if med_gt_mag > 0.01 else float('inf')
            print(f"  I{it:>2}  | {med_rot:>7.2f}° | {med_gtflow:>7.2f}° | {med_epe:>6.2f}px | {med_epe_c:>6.2f}px | {med_gt_mag:>6.2f}px | {med_pred_mag:>7.2f}px | {med_coarse_mag:>9.2f}px | {rel_epe:>6.1%}")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--noise_deg',nargs='+',type=float,default=[8,5,3])
    parser.add_argument('--max_iters',type=int,default=3)
    args=parser.parse_args()
    with open(args.config) as f: config=yaml.safe_load(f)
    evaluate(config,args.checkpoint,args.noise_deg,args.max_iters)
