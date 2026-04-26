import numpy as np
import os, struct

cameras_path = 'dataset/OldHospital/sparse/0/images.bin'
with open(cameras_path, 'rb') as f:
    n_images = struct.unpack('Q', f.read(8))[0]
    cam_centers = []
    for _ in range(n_images):
        image_id = struct.unpack('I', f.read(4))[0]
        qvec = struct.unpack('4d', f.read(32))
        tvec = np.array(struct.unpack('3d', f.read(24)))
        camera_id = struct.unpack('I', f.read(4))[0]
        name = b''
        while True:
            c = f.read(1)
            if c == b'\x00': break
            name += c
        n_2d = struct.unpack('Q', f.read(8))[0]
        f.read(n_2d * 24)
        w, x, y, z = qvec
        R = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
        ])
        C = -R.T @ tvec
        cam_centers.append(C)

    cam_centers = np.array(cam_centers)
    avg_center = cam_centers.mean(axis=0)
    dists = np.linalg.norm(cam_centers - avg_center, axis=1)
    max_dist = dists.max()
    cameras_extent = max_dist * 1.1
    print(f'N cameras: {n_images}')
    print(f'Avg center: {avg_center}')
    print(f'Max dist from avg: {max_dist:.2f}')
    print(f'cameras_extent (spatial_lr_scale): {cameras_extent:.2f}')
    print(f'Min dist: {dists.min():.2f}, Median: {np.median(dists):.2f}')
    print(f'Scene bbox: min={cam_centers.min(axis=0)} max={cam_centers.max(axis=0)}')
