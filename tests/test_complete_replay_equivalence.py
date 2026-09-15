import numpy as np
import pytest

from feature_extract.tools.vfm.replay_complete_frontend import compare


def test_replay_allows_metadata_path_changes_but_preserves_nan(tmp_path):
    old, new = tmp_path / 'old.npz', tmp_path / 'new.npz'
    for path, metadata in [(old, 'archive'), (new, 'fresh')]:
        np.savez(path, names=np.array(['query']), pose=np.array([np.nan, 1.0]),
                 metadata_json=np.array(metadata))
    assert all(compare(new, old).values())


def test_replay_rejects_a_dropped_output_field(tmp_path):
    old, new = tmp_path / 'old.npz', tmp_path / 'new.npz'
    np.savez(old, pose=np.ones(3), usable=np.array([True]))
    np.savez(new, pose=np.ones(3))
    with pytest.raises(ValueError, match='schema'):
        compare(new, old)


def test_replay_rejects_small_numerical_changes(tmp_path):
    old, new = tmp_path / 'old.npz', tmp_path / 'new.npz'
    np.savez(old, pose=np.array([1.0], dtype=np.float32))
    np.savez(new, pose=np.array([1.0000001], dtype=np.float32))
    with pytest.raises(ValueError, match='equivalence failed'):
        compare(new, old)
