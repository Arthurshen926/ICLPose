import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def test_infer_feature_dims_supports_asymmetric_config():
    from feature_field.utils.dcff_eval_targets import infer_feature_dims

    cfg = {
        "model": {
            "feature_dim": 96,
            "fine_feature_dim": 96,
            "coarse_feature_dim": 32,
        }
    }

    assert infer_feature_dims(cfg) == (96, 32)


def test_online_teacher_provider_loads_projection_state_and_returns_asymmetric_targets(tmp_path):
    from feature_field.utils.dcff_eval_targets import OnlineTeacherTargetProvider

    class FakeTeacher(torch.nn.Module):
        def __init__(self, target_dim, fine_dim, coarse_dim, bottleneck, **kwargs):
            super().__init__()
            self.fine_dim = fine_dim
            self.coarse_dim = coarse_dim
            self.bottleneck = bottleneck
            self.loaded_state = None
            self.image_size = None

        def set_image_size(self, img_h, img_w):
            self.image_size = (img_h, img_w)

        @property
        def feature_resolution(self):
            return 3, 5

        def load_projection_state(self, state):
            self.loaded_state = state

        def extract_raw(self, images):
            b = images.shape[0]
            return (
                torch.ones(b, 1280, 3, 5, device=images.device),
                torch.ones(b, 1280, 3, 5, device=images.device) * 2,
            )

        def project(self, fine_raw, coarse_raw, return_loss=False):
            b = fine_raw.shape[0]
            fine = torch.ones(b, self.fine_dim, 3, 5, device=fine_raw.device)
            coarse = torch.ones(b, self.coarse_dim, 3, 5, device=fine_raw.device) * 2
            if return_loss:
                return fine, coarse, {}
            return fine, coarse

    ckpt = {
        "projection_state": {"fine": {"sentinel": torch.tensor(1)}, "coarse": {}},
    }
    cfg = {
        "model": {
            "feature_dim": 96,
            "fine_feature_dim": 96,
            "coarse_feature_dim": 32,
        },
        "teacher": {
            "mode": "online_bottleneck",
            "input_longest_edge": 640,
        },
    }

    provider = OnlineTeacherTargetProvider(
        cfg=cfg,
        ckpt=ckpt,
        device="cpu",
        teacher_factory=FakeTeacher,
    )

    class Cam:
        uid = 7
        image_name = "seq0/frame00000.png"
        image = str(tmp_path / "frame.png")
        width = 10
        height = 6

    from PIL import Image

    Image.new("RGB", (10, 6)).save(Cam.image)

    fine, coarse, fids = provider.get_batch([Cam()])

    assert provider.teacher.loaded_state is ckpt["projection_state"]
    assert provider.feature_resolution == (3, 5)
    assert tuple(fine.shape) == (1, 96, 3, 5)
    assert tuple(coarse.shape) == (1, 32, 3, 5)
    assert fids == [None]


if __name__ == "__main__":
    import tempfile

    test_infer_feature_dims_supports_asymmetric_config()
    with tempfile.TemporaryDirectory() as d:
        test_online_teacher_provider_loads_projection_state_and_returns_asymmetric_targets(Path(d))
    print("dcff_eval_teacher_targets tests passed")
