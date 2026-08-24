from __future__ import annotations

from feature_extract.tools.vfm.evaluate_goal_maplet_controlled_6dof_basin_gate import (
    _direction_failures,
)


def _sign(correct: int = 12, complete: int = 6) -> dict[str, object]:
    return {
        "radial_pair_correct": correct,
        "radial_pair_count": 12,
        "radial_pair_accuracy": correct / 12.0,
        "complete_path_count": complete,
        "signed_path_count": 6,
        "complete_path_rate": complete / 6.0,
    }


def test_direction_gate_is_fail_closed_per_sign() -> None:
    directions = []
    for index in range(21):
        directions.append({
            "direction_id": index,
            "label": f"d{index}",
            "kind": "coordinate_axis" if index < 6 else "pair_coupling",
            "negative_sign": _sign(),
            "positive_sign": _sign(),
        })
    directions[5]["negative_sign"] = _sign(correct=1, complete=0)
    result = _direction_failures({
        "full_6dof_directional_capture": {
            "direction_count": 21,
            "per_direction": directions,
        }
    })
    assert result["all_42_signed_directions_strictly_monotonic"] is False
    assert result["imperfect_signed_direction_count"] == 1
    assert result["stable_zero_complete_signed_direction_count"] == 1
    assert result["stable_zero_complete_signed_directions"][0]["label"] == "d5"


def test_direction_gate_accepts_only_all_perfect_signed_rays() -> None:
    result = _direction_failures({
        "full_6dof_directional_capture": {
            "direction_count": 21,
            "per_direction": [
                {
                    "direction_id": index,
                    "label": f"d{index}",
                    "kind": "coordinate_axis" if index < 6 else "pair_coupling",
                    "negative_sign": _sign(),
                    "positive_sign": _sign(),
                }
                for index in range(21)
            ],
        }
    })
    assert result["all_42_signed_directions_strictly_monotonic"] is True
    assert result["imperfect_signed_direction_count"] == 0
