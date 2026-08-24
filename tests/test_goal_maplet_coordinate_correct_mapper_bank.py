import numpy as np
import pytest

from feature_extract.tools.vfm.build_goal_maplet_coordinate_correct_mapper_bank import (
    _contributor_filename_identity,
    _parent_observations,
    _select_capped_views,
)
from feature_extract.tools.vfm.seal_goal_maplet_surface_mapper_coordinate_lineage import (
    _verified_training_completion,
    _verified_lineage,
)
from feature_extract.vfm.localization_goal_maplet.retrieval_surface_metrics import (
    COORDINATE_CONTRACT,
)


def test_parent_observations_preserve_mass_and_raw_token_centroid() -> None:
    ids = np.asarray(
        [[[10], [10]], [[20], [10]]],
        dtype=np.int64,
    )
    weights = np.ones(ids.shape, dtype=np.float32)
    lookup = np.full((21,), -1, dtype=np.int32)
    lookup[10] = 0
    lookup[20] = 1
    parent, mass, xy, quality = _parent_observations(
        ids,
        weights,
        lookup,
        token_height=2,
        token_width=2,
        minimum_parent_mass_tokens=0.5,
    )
    assert parent.tolist() == [0, 1]
    np.testing.assert_allclose(mass, [3.0, 1.0])
    np.testing.assert_allclose(xy[0], [2.0 / 3.0, 1.0 / 3.0])
    np.testing.assert_allclose(xy[1], [0.0, 1.0])
    assert np.all((quality > 0.0) & (quality <= 1.0))


def test_contributor_route_is_recovered_without_opening_archive(tmp_path) -> None:
    held = tmp_path / "seq14__frame00001.png.npz"
    held.write_bytes(b"intentionally-not-an-npz")
    assert _contributor_filename_identity(held) == (
        "seq14", "seq14/frame00001.png",
    )
    with pytest.raises(ValueError, match="lacks route identity"):
        _contributor_filename_identity(tmp_path / "seq14.npz")


def test_capped_view_selection_is_deterministic_and_train_identifiable() -> None:
    rows = [
        {
            "parent_row": parent,
            "trajectory_id": route,
            "image_id": f"{route}/frame{index:05d}.png",
            "mass_tokens": mass,
        }
        for parent, route, index, mass in (
            (0, "seq1", 1, 1.0),
            (0, "seq1", 2, 3.0),
            (0, "seq2", 1, 2.0),
            (0, "seq9", 1, 4.0),
            (1, "seq1", 3, 5.0),
            (1, "seq9", 2, 6.0),
        )
    ]
    selected, retained = _select_capped_views(
        rows,
        training_trajectories={"seq1", "seq2"},
        maximum_views_per_parent_per_trajectory=1,
    )
    assert retained.tolist() == [0]
    assert [row["image_id"] for row in selected] == [
        "seq1/frame00002.png",
        "seq2/frame00001.png",
        "seq9/frame00001.png",
    ]


def test_mapper_lineage_seal_is_hash_bound_and_fails_closed() -> None:
    bank_hash = "a" * 64
    manifest_hash = "b" * 64
    metadata = {
        "supervision_coordinate_lineage": {
            "coordinate_correct": True,
            "coordinate_contract": COORDINATE_CONTRACT,
            "radio_final_manifest_file_sha256": manifest_hash,
            "strict_holdout_present": False,
            "route_allowlist_applied_before_opening_contributor_archives": True,
        }
    }
    lineage = _verified_lineage(
        bank_metadata=metadata,
        bank_file_sha256=bank_hash,
        manifest_file_sha256=manifest_hash,
    )
    assert lineage["surface_maplets_file_sha256"] == bank_hash
    assert lineage["radio_final_manifest_file_sha256"] == manifest_hash
    assert lineage["coordinate_correct"] is True
    with pytest.raises(ValueError, match="manifest hashes differ"):
        _verified_lineage(
            bank_metadata=metadata,
            bank_file_sha256=bank_hash,
            manifest_file_sha256="c" * 64,
        )


def test_mapper_seal_rejects_interrupted_training_summary(tmp_path) -> None:
    checkpoint = (tmp_path / "mapper.pt").resolve()
    metadata = {"best_epoch": 50, "best_validation": {"recall_at_5": 0.8}}
    base = {
        "stage": "train_surface_maplet_mapper",
        "output_checkpoint": str(checkpoint),
        "best_epoch": 50,
        "best_validation": {"recall_at_5": 0.8},
        "config": {"epochs": 120, "patience": 30},
        "history": [{"epoch": 0}, {"epoch": 50}, {"epoch": 75}],
    }
    with pytest.raises(ValueError, match="natural completion"):
        _verified_training_completion(
            base,
            checkpoint_path=checkpoint,
            checkpoint_metadata=metadata,
            expected_patience=30,
        )
    completed = {
        **base,
        "history": [{"epoch": 0}, {"epoch": 50}, {"epoch": 80}],
    }
    audit = _verified_training_completion(
        completed,
        checkpoint_path=checkpoint,
        checkpoint_metadata=metadata,
        expected_patience=30,
    )
    assert audit["natural_completion_verified"] is True
    assert audit["reached_early_stop_patience"] is True
