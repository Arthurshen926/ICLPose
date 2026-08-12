import numpy as np
import pytest

from feature_extract.tools.vfm.train_surface_maplet_mapper import (
    _fixed_epoch_training_images,
)


def test_fixed_epoch_training_uses_every_non_strict_image():
    image_ids = np.asarray(
        ["seq1/frame1.png", "seq2/frame1.png", "seq3/frame1.png"],
        dtype=object,
    )
    assert _fixed_epoch_training_images(image_ids, (), ("seq3",)) == {
        "seq1/frame1.png",
        "seq2/frame1.png",
    }


def test_fixed_epoch_explicit_trajectories_are_exact():
    image_ids = np.asarray(
        ["seq1/frame1.png", "seq2/frame1.png"], dtype=object
    )
    assert _fixed_epoch_training_images(image_ids, ("seq2",), ()) == {
        "seq2/frame1.png"
    }
    with pytest.raises(ValueError, match="absent"):
        _fixed_epoch_training_images(image_ids, ("seq9",), ())
