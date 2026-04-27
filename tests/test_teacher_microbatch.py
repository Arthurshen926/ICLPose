import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


class FakeTeacher:
    def __init__(self):
        self.extract_sizes = []

    def extract_raw(self, images):
        self.extract_sizes.append(images.shape[0])
        b = images.shape[0]
        base = images[:, :1, :2, :2]
        return base.expand(b, 4, 2, 2), (base + 1).expand(b, 4, 2, 2)

    def project(self, fine_raw, coarse_raw, return_loss=False):
        fine = fine_raw[:, :2]
        coarse = coarse_raw[:, :3]
        if not return_loss:
            return fine, coarse
        loss = fine.mean() + coarse.mean()
        return fine, coarse, {"compress_total": loss, "aux": loss * 2}


def test_extract_project_teacher_batch_respects_microbatch_and_concatenates():
    from feature_field.train_impl import _extract_project_teacher_batch

    teacher = FakeTeacher()
    images = torch.arange(5 * 3 * 4 * 4, dtype=torch.float32).reshape(5, 3, 4, 4)

    fine, coarse, losses = _extract_project_teacher_batch(
        teacher=teacher,
        teacher_images=images,
        teacher_mode="online_bottleneck",
        micro_batch=2,
    )

    assert teacher.extract_sizes == [2, 2, 1]
    assert tuple(fine.shape) == (5, 2, 2, 2)
    assert tuple(coarse.shape) == (5, 3, 2, 2)
    assert set(losses) == {"compress_total", "aux"}
    assert losses["compress_total"].requires_grad is False


if __name__ == "__main__":
    test_extract_project_teacher_batch_respects_microbatch_and_concatenates()
    print("teacher_microbatch tests passed")
