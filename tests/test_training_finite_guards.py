import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from feature_field.train_impl import _all_grads_finite, _all_params_finite


def test_all_grads_finite_detects_nan_gradient():
    param = torch.nn.Parameter(torch.ones(2))
    param.grad = torch.tensor([0.0, float("nan")])

    assert not _all_grads_finite([param])


def test_all_params_finite_detects_inf_parameter():
    param = torch.nn.Parameter(torch.tensor([1.0, float("inf")]))

    assert not _all_params_finite([param])


if __name__ == "__main__":
    test_all_grads_finite_detects_nan_gradient()
    test_all_params_finite_detects_inf_parameter()
    print("training finite guard tests passed")
