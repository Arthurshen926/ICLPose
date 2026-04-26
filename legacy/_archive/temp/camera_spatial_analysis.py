"""Analyze camera spatial overlap between train/test sequences."""
import sys
sys.path.insert(0, '.')
import numpy as np
from feature_3dgs.train_2dgs_geometry import load_scene

train_cams, test_cams, pcd_xyz, pcd_rgb, cameras_extent = load_scene("dataset/OldHospital")
all_cams = train_cams + test_cams

# Extract camera positions and sequence info
seq_data = {}
test_cams_set = set(id(c) for c in test_cams)
for cam in all_cams:
    seq = cam.image_name.split('/')[0]
    # Camera position in world coordinates: C = -R^T @ T
    R = cam.R  # Already transposed in our format (world-to-cam R^T)
    T = cam.T
    pos = -R @ T  # world position
    
    is_test = id(cam) in test_cams_set
    if seq not in seq_data:
        seq_data[seq] = {'train': [], 'test': [], 'positions': []}
    
    if is_test:
        seq_data[seq]['test'].append(pos)
    else:
        seq_data[seq]['train'].append(pos)
    seq_data[seq]['positions'].append(pos)

print("=" * 70)
print("Camera Spatial Analysis per Sequence")
print("=" * 70)

# Compute centroid and extent for each sequence
for seq in sorted(seq_data.keys()):
    d = seq_data[seq]
    all_pos = np.array(d['positions'])
    n_train = len(d['train'])
    n_test = len(d['test'])
    centroid = all_pos.mean(axis=0)
    spread = all_pos.std(axis=0)
    bbox_min = all_pos.min(axis=0)
    bbox_max = all_pos.max(axis=0)
    extent = np.linalg.norm(bbox_max - bbox_min)
    
    print(f"\n{seq}: {n_train} train, {n_test} test")
    print(f"  Centroid: [{centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f}]")
    print(f"  Extent:   {extent:.2f}m")
    print(f"  BBox:     [{bbox_min[0]:.1f},{bbox_min[1]:.1f},{bbox_min[2]:.1f}] → [{bbox_max[0]:.1f},{bbox_max[1]:.1f},{bbox_max[2]:.1f}]")

# Compute nearest-train-camera distance for each test camera
print("\n" + "=" * 70)
print("Test Camera → Nearest Training Camera Distance")
print("=" * 70)

train_positions = []
for cam in train_cams:
    R = cam.R
    T = cam.T
    pos = -R @ T
    train_positions.append(pos)
train_positions = np.array(train_positions)

test_distances = {}
for cam in test_cams:
    seq = cam.image_name.split('/')[0]
    R = cam.R  
    T = cam.T
    pos = -R @ T
    dists = np.linalg.norm(train_positions - pos[None], axis=1)
    min_dist = dists.min()
    if seq not in test_distances:
        test_distances[seq] = []
    test_distances[seq].append((cam.image_name, min_dist))

for seq in sorted(test_distances.keys()):
    dists_list = test_distances[seq]
    all_dists = [d for _, d in dists_list]
    print(f"\n{seq} ({len(dists_list)} test views):")
    print(f"  Min distance to train: {min(all_dists):.3f}m")
    print(f"  Max distance to train: {max(all_dists):.3f}m")  
    print(f"  Mean distance to train: {np.mean(all_dists):.3f}m")
    print(f"  Median distance: {np.median(all_dists):.3f}m")
    
    # Show worst 5 (farthest from any training view)
    dists_list.sort(key=lambda x: -x[1])
    print(f"  Farthest 5 from training:")
    for name, d in dists_list[:5]:
        print(f"    {name}: {d:.3f}m")
    print(f"  Closest 5 to training:")
    for name, d in dists_list[-5:]:
        print(f"    {name}: {d:.3f}m")
