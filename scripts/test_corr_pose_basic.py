#!/usr/bin/env python3
"""Quick test for CorrPoseNet components."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from ic_models.corr_pose_net import CorrPoseNet, local_correlation, diff_pose_solve

# Test network construction
net = CorrPoseNet(feat_dim=768, enc_dim=128, hidden_dim=128, corr_radius=4, num_iters=3)
print(net)
print()

# Test local_correlation
B, C, H, W = 2, 128, 35, 46
fmap1 = torch.randn(B, C, H, W)
fmap2 = torch.randn(B, C, H, W)
fmap1 = torch.nn.functional.normalize(fmap1, dim=1)
fmap2 = torch.nn.functional.normalize(fmap2, dim=1)
corr = local_correlation(fmap1, fmap2, radius=4)
print(f'Correlation shape: {corr.shape}')

# Test encode
feats = torch.randn(B, 768, H, W)
enc = net.encode(feats)
print(f'Encoded shape: {enc.shape}')

# Test context encoder
ctx = torch.tanh(net.context_encoder(feats))
print(f'Context shape: {ctx.shape}')

# Test corr_encoder
corr_feat = net.corr_encoder(corr)
print(f'Corr encoded shape: {corr_feat.shape}')

# Test GRU
hidden = net.gru(ctx, corr_feat)
print(f'GRU output shape: {hidden.shape}')

# Test heads
flow = net.flow_head(hidden)
conf = torch.sigmoid(net.conf_head(hidden))
print(f'Flow shape: {flow.shape}')
print(f'Conf shape: {conf.shape}')

# Test diff_pose_solve
from modules.featuremetric import compute_image_jacobian
depth = torch.ones(B, H, W) * 2.0
intrinsics = {'fx': 23.0, 'fy': 23.3, 'cx': 22.97, 'cy': 17.47}
Ju, Jv, valid = compute_image_jacobian(depth, intrinsics)
print(f'Ju shape: {Ju.shape}, Jv shape: {Jv.shape}, valid shape: {valid.shape}')

delta_xi = diff_pose_solve(flow, conf, Ju, Jv, valid, damping=1e-3)
print(f'delta_xi shape: {delta_xi.shape}')
print(f'delta_xi values: {delta_xi}')

# Test se3_exp
from modules.lie_algebra import se3_exp
delta_T = se3_exp(delta_xi)
print(f'delta_T shape: {delta_T.shape}')

# Test gradient flow - set non-zero flow head weights for full chain test
with torch.no_grad():
    net.flow_head[-1].weight.fill_(0.01)
    net.conf_head[-1].weight.fill_(0.01)

# Recompute with non-zero weights
flow2 = net.flow_head(hidden)
conf2 = torch.sigmoid(net.conf_head(hidden))
delta_xi2 = diff_pose_solve(flow2, conf2, Ju, Jv, valid, damping=1e-3)
print(f'delta_xi (non-zero weights): {delta_xi2[0, :3]}...')

target_xi = torch.randn(B, 6)
loss = (delta_xi2 - target_xi).norm()
loss.backward()
print(f'\nGradient flow (with non-zero weights):')
print(f'  flow_head last conv: {net.flow_head[-1].weight.grad.norm():.6f}')
print(f'  conf_head last conv: {net.conf_head[-1].weight.grad.norm():.6f}')
has_enc_grad = net.feat_encoder[0].weight.grad is not None
print(f'  feat_encoder has grad: {has_enc_grad}')
if has_enc_grad:
    print(f'  feat_encoder first conv: {net.feat_encoder[0].weight.grad.norm():.6f}')
print(f'  gru convz: {net.gru.convz.weight.grad.norm():.6f}')

print('\n=== All basic tests passed! ===')
