#!/usr/bin/env python3
"""Quick test: dataset + model + render + split."""
import torch
from feature_3dgs.multiscale_dataset import MultiScaleFeatureDataset
from feature_3dgs.multiscale_gaussian_model import MultiScaleGaussianModel
from feature_3dgs.feature_renderer import FeatureRenderer

# 1. Dataset
ds = MultiScaleFeatureDataset(
    feature_dir='output/features_multiscale_compressed/room_0',
    traj_path='dataset/room_0/Sequence_1/traj_w_c.txt',
)
sample = ds[0]
print('Fine:', sample['fine_feat'].shape)
print('Mid:', sample['mid_feat'].shape)
print('Coarse:', sample['coarse_feat'].shape)

# 2. Model
model = MultiScaleGaussianModel()
model.load_ply('dataset/room_0/splatloc-test/data/room_0/point_cloud/final/point_cloud.ply')
model = model.cuda()

# 3. Render one frame
fine_H, fine_W = ds.fine_hw
scale_x = fine_W / 640
scale_y = fine_H / 480
fine_fx = 320.0 * scale_x
fine_fy = 320.0 * scale_y
fine_cx = 319.5 * scale_x
fine_cy = 239.5 * scale_y
pose = sample['pose'].cuda()

result = FeatureRenderer.render_features(
    gaussian_model=model, viewmat=pose,
    fx=fine_fx, fy=fine_fy, cx=fine_cx, cy=fine_cy,
    img_height=fine_H, img_width=fine_W,
    norm_feat_before_render=True, norm_feat_after_render=False,
)
print('Rendered:', result['feature_map'].shape)

split = MultiScaleGaussianModel.split_feature_map(result['feature_map'])
for k, v in split.items():
    print(f'  {k}: {v.shape}')
print('OK!')
