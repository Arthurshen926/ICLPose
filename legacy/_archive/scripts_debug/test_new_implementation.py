"""测试新实现的模块"""
from modules.transformer import TransformerLayer, MultiHeadAttention
from modules.fusion_module import CrossModalFusionModule
from ic_models.ic_pose_net import ICPoseNet
import torch

# Test TransformerLayer
print('Testing TransformerLayer...')
layer = TransformerLayer(d_model=256, num_heads=8, dropout=0.1)
q = torch.randn(2, 10, 256)
k = torch.randn(2, 20, 256)
v = torch.randn(2, 20, 256)
output = layer(q, k, v)
print(f'TransformerLayer output shape: {output.shape}')

# Test with embeds
print('\nTesting with position embeds...')
q_embeds = torch.randn(2, 10, 256)
k_embeds = torch.randn(2, 20, 256)
output = layer(q, k, v, q_embeds=q_embeds, k_embeds=k_embeds)
print(f'Output shape with embeds: {output.shape}')

# Test FusionModule
print('\nTesting CrossModalFusionModule...')
fusion = CrossModalFusionModule(feature_dim=256, num_layers=8, num_heads=8)
query = torch.randn(2, 128, 256)
img = torch.randn(2, 512, 256)
pcd = torch.randn(2, 1024, 256)
query_list = fusion(query, img, pcd)
print(f'FusionModule output: {len(query_list)} layers')
print(f'Final query shape: {query_list[-1].shape}')

# Test ICPoseNet
print('\nTesting ICPoseNet...')
model = ICPoseNet(feature_dim=256, num_queries=128, fusion_layers=8, num_heads=8)
img_feats = torch.randn(2, 512, 256)
pcd_feats = torch.randn(2, 1024, 256)
img_pos = torch.randn(2, 512, 256)
pcd_pos = torch.randn(2, 1024, 256)
pose_matrix, pose_9d, rotation_6d, translation = model(img_feats, pcd_feats, img_pos, pcd_pos)
print(f'Pose matrix shape: {pose_matrix.shape}')
print(f'Pose 9D shape: {pose_9d.shape}')
print(f'Rotation 6D shape: {rotation_6d.shape}')
print(f'Translation shape: {translation.shape}')

print('\n✓ All tests passed!')
