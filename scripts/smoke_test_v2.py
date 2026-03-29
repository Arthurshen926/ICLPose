"""Quick smoke test for TransformerPoseNetV2."""
import torch
import sys
sys.path.insert(0, '/root/ICLPose')

from ic_models.transformer_pose_net_v2 import TransformerPoseNetV2

model = TransformerPoseNetV2(
    d_model=128, feat_in_dim=64, n_heads=4, n_layers=6,
    ffn_dim=256, attn_hw=(35, 61), fine_hw=(69, 121),
    intrinsics={'fx': 1673.5, 'fy': 1673.5, 'cx': 960.0, 'cy': 540.0},
    img_hw=(1080, 1920),
).cuda()

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Parameters: {n_params/1e6:.3f}M')

# Forward pass test
B = 2
q = {'fine': torch.randn(B, 64, 69, 121).cuda()}
r = {'fine': torch.randn(B, 64, 69, 121).cuda()}
d = torch.rand(B, 69, 121).cuda() * 3 + 0.5

with torch.cuda.amp.autocast():
    out = model(q, r, d)

print(f'flow_fine: {out["flow_fine"].shape}')
print(f'conf_fine: {out["conf_fine"].shape}')
print(f'delta_xi: {out["delta_xi"].shape}')
print(f'flow_fine range: [{out["flow_fine"].min():.4f}, {out["flow_fine"].max():.4f}]')
print(f'conf_fine range: [{out["conf_fine"].min():.4f}, {out["conf_fine"].max():.4f}]')

mem = torch.cuda.max_memory_allocated() / 1024**3
print(f'Peak GPU memory: {mem:.2f} GB')
print('OK')
