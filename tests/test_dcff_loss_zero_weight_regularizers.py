import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from feature_field.dcff.losses import DCFFLoss


def test_zero_weight_geometry_regularizers_do_not_make_total_nan():
    loss_fn = DCFFLoss(lambda_normal=0.0, lambda_dist=0.0)
    rgb = torch.zeros(1, 3, 4, 4)
    render_result = {
        "rgb": rgb.clone(),
        "normals": torch.full((1, 3, 4, 4), float("inf")),
        "surf_normals": torch.ones(1, 4, 4, 3),
        "distort": torch.full((1, 4, 4, 1), float("inf")),
    }

    losses = loss_fn.compute(render_result=render_result, gt_rgb=rgb, phase=1)

    assert torch.isfinite(losses["total"])


if __name__ == "__main__":
    test_zero_weight_geometry_regularizers_do_not_make_total_nan()
    print("dcff zero-weight regularizer tests passed")
