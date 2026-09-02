import numpy as np

from feature_extract.vfm.localization_goal_maplet.chart_pose_only_densification import (
    MappingCameraPose,
    PoseOnlyDensificationConfig,
    diagnose_held_camera_neighbors,
    select_source_camera_window,
)


def _trajectory(route, positions, *, global_offset=0):
    rows = []
    for index, position in enumerate(positions):
        rows.append(
            MappingCameraPose(
                name=f"{route}__frame{index + 1:05d}.png",
                global_lexical_index=global_offset + index,
                center_world=np.asarray(position, np.float64),
                forward_world=np.asarray([0.0, 0.0, 1.0]),
            ).validated()
        )
    return rows


def test_source_window_is_source_only_and_held_diagnostic_is_posthoc():
    increments = 1.0 + 0.01 * np.arange(24)
    x = np.concatenate(([0.0], np.cumsum(increments)))
    source = _trajectory("seq4", [[value, 0.0, 0.0] for value in x])
    config = PoseOnlyDensificationConfig(
        source_view_count=16,
        maximum_adjacent_baseline_m=1.5,
        maximum_adjacent_forward_angle_deg=5.0,
        minimum_path_length_m=15.0,
        held_neighbor_radius_m=2.0,
        held_neighbor_forward_angle_deg=20.0,
        held_target_view_count_per_route=4,
    )
    selected, candidates = select_source_camera_window(source, config=config)
    assert len(candidates) == 10
    rank_zero = next(row for row in candidates if row["candidate_rank_after_gate"] == 0)
    assert selected[0].name == rank_zero["start_name"]
    assert selected[-1].name == rank_zero["end_name"]

    held = _trajectory(
        "seq1",
        [[row.center_world[0], 0.5, 0.0] for row in selected],
        global_offset=100,
    )
    diagnostics, frozen = diagnose_held_camera_neighbors(
        selected,
        {"seq1": held},
        config=config,
    )
    assert diagnostics["seq1"]["eligible_camera_count"] == 16
    assert diagnostics["seq1"]["selected_camera_count"] == 4
    assert diagnostics["seq1"]["source_view_count_covered_by_selected"] > 0
    assert diagnostics["seq1"]["selection_used_by_source_window_ranker"] is False
    assert len(frozen) == 4

    # A completely different held trajectory changes only the posthoc report;
    # rerunning the source-only ranker returns the identical source window.
    far_held = _trajectory(
        "seq1",
        [[row.center_world[0], 100.0, 0.0] for row in selected],
        global_offset=100,
    )
    far_diagnostics, far_frozen = diagnose_held_camera_neighbors(
        selected,
        {"seq1": far_held},
        config=config,
    )
    selected_again, _ = select_source_camera_window(source, config=config)
    assert [row.name for row in selected_again] == [row.name for row in selected]
    assert far_diagnostics["seq1"]["eligible_camera_count"] == 0
    assert far_frozen == []


def test_source_window_records_rejected_candidates():
    positions = [[float(index), 0.0, 0.0] for index in range(20)]
    positions[19] = [30.0, 0.0, 0.0]
    source = _trajectory("seq4", positions)
    config = PoseOnlyDensificationConfig(
        source_view_count=16,
        maximum_adjacent_baseline_m=2.0,
        minimum_path_length_m=10.0,
    )
    selected, candidates = select_source_camera_window(source, config=config)
    assert len(selected) == 16
    assert len(candidates) == 5
    assert candidates[-1]["source_only_gate_pass"] is False
    assert "adjacent_baseline_above_limit" in candidates[-1]["gate_reasons"]
