import numpy as np
import pytest

from feature_extract.vfm.cambridge_pose_lattice import (
    camera_center_from_pose_w2c,
    pose_w2c_from_center_rotation,
)
from feature_extract.vfm.localization_v6.maplet_retrieval import (
    MapletRetrievalResult,
    QueryMapletGroup,
)
from feature_extract.vfm.localization_v7.pose_signature import (
    PoseSignature,
    PoseSignatureBank,
    pose_modes_from_signature_scores,
    pose_signature_from_retrieval,
    score_pose_signatures,
)


def _metadata(**updates):
    value = {
        "representation": "pose_aware_maplet_sufficient_statistics",
        "uses_mapping_pose_statistics": True,
        "stores_mapping_rgb": False,
        "stores_mapping_image_ids": False,
        "stores_mapping_image_paths": False,
        "stores_observation_descriptors": False,
        "uses_sfm_points": False,
        "uses_sfm_tracks": False,
        "uses_alike_descriptors": False,
        "uses_radio_intermediate": False,
    }
    value.update(updates)
    return value


def _signature(xy: float) -> PoseSignature:
    return PoseSignature(
        identity=np.asarray([1.0, 0.0]),
        layout_mean_xy=np.asarray([[xy, 0.5], [0.0, 0.0]]),
        layout_extent_xy=np.asarray([[0.1, 0.1], [0.0, 0.0]]),
        layout_variance_xy=np.zeros((2, 2)),
        layout_mass=np.asarray([1.0, 0.0]),
    )


def test_pose_signature_compresses_identity_and_region_geometry():
    retrieval = MapletRetrievalResult(
        groups=(
            QueryMapletGroup(
                query_region_xy=np.asarray([100.0, 25.0]),
                query_region_extent=np.asarray([20.0, 10.0]),
                maplet_ids=np.asarray([7]),
                probabilities=np.asarray([0.8]),
                null_probability=0.2,
                omitted_probability=0.0,
            ),
        ),
        ranked_maplet_ids=np.asarray([7]),
        evidence=np.asarray([3.0]),
    )
    signature = pose_signature_from_retrieval(
        retrieval, np.asarray([5, 7]), (200, 100)
    )
    np.testing.assert_allclose(signature.identity, [0.0, 1.0])
    np.testing.assert_allclose(signature.layout_mean_xy[1], [0.5, 0.25])
    np.testing.assert_allclose(signature.layout_extent_xy[1], [0.1, 0.1])
    np.testing.assert_allclose(signature.layout_mass, [0.0, 1.0])


def test_layout_breaks_an_identity_tie_without_descriptor_interaction():
    poses = np.stack(
        [
            pose_w2c_from_center_rotation([0.0, 0.0, 0.0], np.eye(3)),
            pose_w2c_from_center_rotation([1.0, 0.0, 0.0], np.eye(3)),
        ]
    )
    left = _signature(0.25)
    right = _signature(0.75)
    bank = PoseSignatureBank(
        maplet_ids=np.asarray([1, 2]),
        poses_w2c=poses,
        identity=np.stack([left.identity, right.identity]),
        layout_mean_xy=np.stack([left.layout_mean_xy, right.layout_mean_xy]),
        layout_extent_xy=np.stack(
            [left.layout_extent_xy, right.layout_extent_xy]
        ),
        layout_variance_xy=np.stack(
            [left.layout_variance_xy, right.layout_variance_xy]
        ),
        layout_mass=np.stack([left.layout_mass, right.layout_mass]),
        metadata=_metadata(),
    )
    identity_only, _ = score_pose_signatures(
        left, bank, layout_weight=0.0, extent_weight=0.0
    )
    np.testing.assert_allclose(identity_only[0], identity_only[1])
    full, _ = score_pose_signatures(left, bank)
    assert full[0] > full[1]


def test_pose_modes_interpolate_a_local_pose_cluster():
    centers = np.asarray([[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [4.0, 0.0, 0.0]])
    poses = np.stack(
        [pose_w2c_from_center_rotation(value, np.eye(3)) for value in centers]
    )
    template = _signature(0.5)
    bank = PoseSignatureBank(
        maplet_ids=np.asarray([1, 2]),
        poses_w2c=poses,
        identity=np.stack([template.identity] * 3),
        layout_mean_xy=np.stack([template.layout_mean_xy] * 3),
        layout_extent_xy=np.stack([template.layout_extent_xy] * 3),
        layout_variance_xy=np.stack([template.layout_variance_xy] * 3),
        layout_mass=np.stack([template.layout_mass] * 3),
        metadata=_metadata(),
    )
    modes = pose_modes_from_signature_scores(
        bank,
        np.asarray([1.0, 0.99, 0.5]),
        translation_radius_m=0.5,
        maximum_modes=2,
    )
    assert modes[0].source == "pose_signature_cluster_mean"
    center = camera_center_from_pose_w2c(modes[0].pose_w2c)
    assert 0.0 < center[0] < 0.2


def test_pose_signature_bank_rejects_mapping_image_identity_storage():
    template = _signature(0.5)
    with pytest.raises(ValueError, match="stores_mapping_image_ids"):
        PoseSignatureBank(
            maplet_ids=np.asarray([1, 2]),
            poses_w2c=np.eye(4)[None],
            identity=template.identity[None],
            layout_mean_xy=template.layout_mean_xy[None],
            layout_extent_xy=template.layout_extent_xy[None],
            layout_variance_xy=template.layout_variance_xy[None],
            layout_mass=template.layout_mass[None],
            metadata=_metadata(stores_mapping_image_ids=True),
        )
