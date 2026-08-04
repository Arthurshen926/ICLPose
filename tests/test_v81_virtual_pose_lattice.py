import numpy as np

from feature_extract.vfm.localization_v81.virtual_pose_lattice import (
    VirtualPoseLattice,
    expand_virtual_pose_seeds,
)


def _lattice():
    # Two positions x one rotation, with one visible key per pose.
    return VirtualPoseLattice(
        positions=np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.float32),
        rotations_w2c=np.eye(3, dtype=np.float32)[None],
        key_offsets=np.asarray([0, 1, 2]),
        visibility_keys=np.asarray([0, 1], dtype=np.uint16),
        maplet_ids=np.asarray([7]),
        cell_grid_size=2,
        physical_graph_sha256="a" * 64,
        pose_vote_bank_sha256="b" * 64,
        metadata={},
    )


def test_virtual_lattice_contract_contains_no_images_or_embeddings(tmp_path):
    path = tmp_path / "lattice.npz"
    _lattice().save_npz(path)
    with np.load(path, allow_pickle=False) as data:
        names = " ".join(data.files).lower()
        assert "image" not in names
        assert "descriptor" not in names
        assert "embedding" not in names
    loaded = VirtualPoseLattice.load_npz(path)
    assert loaded.pose_count == 2
    assert loaded.metadata["stores_mapping_rgb"] is False


def test_virtual_lattice_pose_index_is_position_rotation_cartesian():
    lattice = _lattice()
    poses = lattice.poses_w2c(np.asarray([0, 1]))
    centers = -np.einsum("nji,nj->ni", poses[:, :3, :3], poses[:, :3, 3])
    assert np.allclose(centers, lattice.positions)


def test_subdivision_retains_original_seed_and_has_unique_modes():
    seed = np.eye(4)[None]
    expanded = expand_virtual_pose_seeds(seed)
    assert any(np.allclose(value, np.eye(4)) for value in expanded)
    flat = np.round(expanded[:, :3].reshape(expanded.shape[0], -1), decimals=5)
    assert np.unique(flat, axis=0).shape[0] == expanded.shape[0]
