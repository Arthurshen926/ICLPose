#!/usr/bin/env python3
"""
Diagnostic: Check the quality of global correlation volume at coarse resolution.
Tests if argmax of correlation gives reasonable flow.
"""
import argparse, math, os, re, sys
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ic_models.pose_refiner import PoseRefiner, global_correlation
from modules.lie_algebra import se3_exp
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
def evaluate(config, checkpoint, noise_degs, device='cuda'):
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

        coarse_epe_list = []
        coarse_pred_mag_list = []
        argmax_epe_list = []
        argmax_mag_list = []
        gt_coarse_mag_list = []
        corr_max_val_list = []
        corr_gt_val_list = []

        for idx_i, frame_idx in enumerate(test_indices):
            feat=torch.load(str(features[frame_idx]),map_location=device,weights_only=True).float().unsqueeze(0)
            c2w=torch.from_numpy(poses_c2w[frame_idx].astype(np.float32))
            gt_w2c=torch.linalg.inv(c2w).to(device)
            seed=frame_idx*1000+int(noise_deg*100)
            noisy_w2c=add_noise_deterministic(gt_w2c,noise_deg,noise_trans,seed)
            pose_cur=noisy_w2c.unsqueeze(0)

            ref_feat=renderer.render_features(pose_cur,feat_hw)
            depth=renderer.render_depth(pose_cur,feat_hw)
            
            # Run projection to get coarse features
            q = F.normalize(feat.to(device), dim=1)
            r = F.normalize(ref_feat, dim=1)
            q_coarse = model.projection(F.interpolate(q, COARSE_HW, mode='bilinear', align_corners=False))
            r_coarse = model.projection(F.interpolate(r, COARSE_HW, mode='bilinear', align_corners=False))
            q_enhanced, r_enhanced = model.cross_attn(q_coarse, r_coarse)
            
            # Global correlation
            coarse_corr = global_correlation(q_enhanced, r_enhanced)  # (B, H_r*W_r, H_q, W_q)
            B, HrWr, Hq, Wq = coarse_corr.shape
            Hr, Wr = COARSE_HW
            
            # Get model's coarse flow prediction (run coarse head)
            h_coarse = model.context_net(q_coarse)
            flow_c = torch.zeros(B, 2, *COARSE_HW, device=device)
            conf_c = torch.ones(B, 1, *COARSE_HW, device=device) * 0.5
            _, conf_c, h_coarse, flow_c = model.coarse_head(coarse_corr, h_coarse, flow_c, conf_c)
            
            # GT coarse flow
            gt_flow_c, gt_mask_c = model.compute_gt_flow(pose_cur, gt_w2c.unsqueeze(0), depth, COARSE_HW)
            
            # Model's coarse flow EPE
            diff = (flow_c - gt_flow_c) * gt_mask_c
            epe = (diff**2).sum(dim=1, keepdim=True).sqrt()
            valid = gt_mask_c.expand_as(epe) > 0
            coarse_epe_list.append(epe[valid].mean().item() if valid.sum() > 0 else 0)
            
            # Coarse pred magnitude
            cm = (flow_c**2).sum(dim=1, keepdim=True).sqrt()
            coarse_pred_mag_list.append(cm[valid].mean().item() if valid.sum() > 0 else 0)
            
            # GT coarse flow magnitude
            gm = (gt_flow_c**2).sum(dim=1, keepdim=True).sqrt()
            gt_coarse_mag_list.append(gm[valid].mean().item() if valid.sum() > 0 else 0)
            
            # Argmax flow: for each query pixel, find the best-matching reference pixel
            # corr shape: (B, Hr*Wr, Hq, Wq) = correlation of each query pixel with all ref pixels
            max_idx = coarse_corr[0].reshape(HrWr, Hq, Wq).argmax(dim=0)  # (Hq, Wq)
            ref_y = (max_idx // Wr).float()  # (Hq, Wq)
            ref_x = (max_idx % Wr).float()   # (Hq, Wq)
            
            # Create coordinate grids for query
            qy = torch.arange(Hq, device=device, dtype=torch.float32).unsqueeze(1).expand(Hq, Wq)
            qx = torch.arange(Wq, device=device, dtype=torch.float32).unsqueeze(0).expand(Hq, Wq)
            
            argmax_flow_x = ref_x - qx  # displacement in x
            argmax_flow_y = ref_y - qy  # displacement in y
            argmax_flow = torch.stack([argmax_flow_x, argmax_flow_y], dim=0).unsqueeze(0)  # (1, 2, Hq, Wq)
            
            diff_am = (argmax_flow - gt_flow_c) * gt_mask_c
            epe_am = (diff_am**2).sum(dim=1, keepdim=True).sqrt()
            argmax_epe_list.append(epe_am[valid].mean().item() if valid.sum() > 0 else 0)
            
            am_mag = (argmax_flow**2).sum(dim=1, keepdim=True).sqrt()
            argmax_mag_list.append(am_mag[valid].mean().item() if valid.sum() > 0 else 0)
            
            # Correlation quality: max val vs val at GT location
            max_val = coarse_corr[0].reshape(HrWr, Hq, Wq).max(dim=0).values
            corr_max_val_list.append(max_val.mean().item())
            
            # Correlation at GT displacement
            gt_u = gt_flow_c[0, 0]  # (Hq, Wq) displacement in x
            gt_v = gt_flow_c[0, 1]  # displacement in y
            gt_ref_x = (qx + gt_u).clamp(0, Wr-1).long()
            gt_ref_y = (qy + gt_v).clamp(0, Hr-1).long()
            gt_ref_idx = gt_ref_y * Wr + gt_ref_x
            gt_val = coarse_corr[0].reshape(HrWr, Hq*Wq)
            gt_ref_idx_flat = gt_ref_idx.reshape(-1).clamp(0, HrWr-1)
            gt_corr_vals = gt_val[gt_ref_idx_flat, torch.arange(Hq*Wq, device=device)]
            corr_gt_val_list.append(gt_corr_vals.mean().item())

        print(f"\n  === Coarse Flow Analysis ===")
        print(f"  GT flow mag     : {np.median(gt_coarse_mag_list):.3f}px (mean {np.mean(gt_coarse_mag_list):.3f})")
        print(f"  Model coarse mag: {np.median(coarse_pred_mag_list):.3f}px (mean {np.mean(coarse_pred_mag_list):.3f})")
        print(f"  Model coarse EPE: {np.median(coarse_epe_list):.3f}px (mean {np.mean(coarse_epe_list):.3f})")
        print(f"  Argmax flow mag : {np.median(argmax_mag_list):.3f}px (mean {np.mean(argmax_mag_list):.3f})")
        print(f"  Argmax flow EPE : {np.median(argmax_epe_list):.3f}px (mean {np.mean(argmax_epe_list):.3f})")
        print(f"  Corr max val    : {np.median(corr_max_val_list):.4f}")
        print(f"  Corr at GT pos  : {np.median(corr_gt_val_list):.4f}")
        print(f"  Corr confidence : max/GT ratio = {np.median(corr_max_val_list)/max(np.median(corr_gt_val_list),1e-6):.3f}")


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--noise_deg',nargs='+',type=float,default=[8,5,3])
    args=parser.parse_args()
    with open(args.config) as f: config=yaml.safe_load(f)
    evaluate(config,args.checkpoint,args.noise_deg)
