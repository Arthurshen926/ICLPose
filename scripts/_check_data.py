import torch, numpy as np, os, cv2

f = torch.load('output/features_multiscale/room_0/coarse/rgb_0_coarse_1280x7x10.pt', map_location='cpu', weights_only=True)
print('coarse:', type(f), f.shape)
f2 = torch.load('output/features_multiscale/room_0/fine_dino/rgb_0_fine_dino_768x35x46.pt', map_location='cpu', weights_only=True)
print('fine_dino:', type(f2), f2.shape)
for fn in sorted(os.listdir('output/retrieval/room_0_netvlad')):
    if fn.endswith('.npy'):
        d = np.load(os.path.join('output/retrieval/room_0_netvlad', fn))
        print(f'{fn}: {d.shape}')
print('Seq1 poses:', np.loadtxt('dataset/room_0/Sequence_1/traj_w_c.txt').shape)
print('Seq2 poses:', np.loadtxt('dataset/room_0/Sequence_2/traj_w_c.txt').shape)
dfiles = sorted(os.listdir('dataset/room_0/Sequence_1/depth'))
print(f'Depth: {len(dfiles)} files')
d = cv2.imread(os.path.join('dataset/room_0/Sequence_1/depth', dfiles[0]), cv2.IMREAD_UNCHANGED)
print(f'Depth img: {d.shape} {d.dtype} range=[{d.min()},{d.max()}]')
