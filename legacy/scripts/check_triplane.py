"""Check triplane model training results."""
import torch
import os

for name in ['oldhospital_triplane_single_v1', 'oldhospital_triplane_single_v2', 'oldhospital_triplane_v3']:
    d = f'output/feature_3dgs/{name}'
    print(f"=== {name} ===")
    for fn in ['best_model.pth', 'final_model.pth']:
        fp = os.path.join(d, fn)
        if os.path.exists(fp):
            m = torch.load(fp, map_location='cpu')
            keys = [k for k in m if k not in ('model_state_dict', 'optimizer_state_dict')]
            info = {k: (float(m[k]) if isinstance(m[k], (int, float, torch.Tensor)) and (not isinstance(m[k], torch.Tensor) or m[k].numel() == 1) else m[k]) for k in keys}
            print(f"  {fn}: {info}")
    print()

# Compare with per-Gaussian baseline
pg = 'output/feature_3dgs/oldhospital_da3_perscale_v2/fine/best_model.pth'
if os.path.exists(pg):
    m = torch.load(pg, map_location='cpu')
    keys = [k for k in m if k not in ('model_state_dict', 'optimizer_state_dict')]
    info = {k: (float(m[k]) if isinstance(m[k], (int, float, torch.Tensor)) and (not isinstance(m[k], torch.Tensor) or m[k].numel() == 1) else m[k]) for k in keys}
    print(f"=== per-Gaussian baseline (fine) ===")
    print(f"  best_model.pth: {info}")
