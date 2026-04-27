import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import feature_field.train_impl as train_impl


def test_online_teacher_targets_are_prepared_before_render_graph():
    source = inspect.getsource(train_impl.train)
    teacher_pos = source.index("fine_proj, coarse_proj, compression_losses = _extract_project_teacher_batch")
    render_pos = source.index("render_result = renderer(")

    assert teacher_pos < render_pos


def test_warmstart_can_skip_coarse_fusion_state():
    source = inspect.getsource(train_impl.train)

    assert "warmstart_restore_coarse_fusion_state" in source
    assert "Skipped warm-start coarse fusion state" in source


if __name__ == "__main__":
    test_online_teacher_targets_are_prepared_before_render_graph()
    test_warmstart_can_skip_coarse_fusion_state()
    print("teacher_target_order tests passed")
