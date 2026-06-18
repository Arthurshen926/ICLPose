import numpy as np

from feature_extract.tools.vfm.render_2dgs_virtual_tokens import render_virtual_pose_token_manifest
from feature_extract.vfm.cambridge_pose_lattice import parse_cambridge_pose_file
from feature_extract.vfm.colmap_tracks import ColmapCamera


class FakeExtractor:
    def __init__(self):
        self.calls = 0

    def extract(self, tensor):
        self.calls += 1
        import torch

        return {"local": torch.ones((2, 2, 2), dtype=torch.float32) * float(self.calls)}


def _pose_file(tmp_path):
    path = tmp_path / "virtual_poses.txt"
    path.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "virtual/a.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "virtual/b.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    return path


def test_render_virtual_pose_token_manifest_writes_images_tokens_and_metadata(tmp_path):
    records = parse_cambridge_pose_file(_pose_file(tmp_path))
    camera = ColmapCamera(camera_id=1, model_id=1, width=4, height=3, params=(2.0, 2.0, 2.0, 1.5))

    def render_fn(**_kwargs):
        rgb = np.full((3, 4, 3), 0.5, dtype=np.float32)
        depth = np.ones((3, 4), dtype=np.float32)
        alpha = np.full((3, 4), 0.75, dtype=np.float32)
        return rgb, depth, alpha

    manifest = render_virtual_pose_token_manifest(
        records=records,
        source=object(),
        camera=camera,
        width=4,
        height=3,
        extractor=FakeExtractor(),
        render_rgb_depth_fn=render_fn,
        image_output_root=tmp_path / "renders",
        token_output_root=tmp_path / "tokens",
        scene="OldHospital",
        split="virtual",
        layer_name="radio_final",
        model_name="c-radio_v4-h",
        storage_dtype="float16",
        renderer_name="official_2dgs",
        source_path="point_cloud.ply",
    )

    assert [record.image_id for record in manifest.records] == ["virtual/a.png", "virtual/b.png"]
    assert (tmp_path / "renders" / "virtual" / "a.png").exists()
    assert manifest.records[0].metadata["renderer"] == "official_2dgs"
    assert manifest.records[0].metadata["alpha_coverage"] == 1.0
    with np.load(manifest.records[0].token_path) as data:
        assert data["radio_final"].shape == (2, 2, 2)
        assert data["radio_final"].dtype == np.float16


def test_render_virtual_pose_token_manifest_filters_low_alpha_coverage(tmp_path):
    records = parse_cambridge_pose_file(_pose_file(tmp_path))
    camera = ColmapCamera(camera_id=1, model_id=1, width=4, height=3, params=(2.0, 2.0, 2.0, 1.5))

    def render_fn(**kwargs):
        image_id = kwargs["record"].image_id
        rgb = np.full((3, 4, 3), 0.5, dtype=np.float32)
        depth = np.ones((3, 4), dtype=np.float32)
        alpha = np.ones((3, 4), dtype=np.float32) if image_id.endswith("a.png") else np.zeros((3, 4), dtype=np.float32)
        return rgb, depth, alpha

    manifest = render_virtual_pose_token_manifest(
        records=records,
        source=object(),
        camera=camera,
        width=4,
        height=3,
        extractor=FakeExtractor(),
        render_rgb_depth_fn=render_fn,
        image_output_root=None,
        token_output_root=tmp_path / "tokens",
        scene="OldHospital",
        split="virtual",
        layer_name="radio_final",
        model_name="c-radio_v4-h",
        storage_dtype="float32",
        renderer_name="official_2dgs",
        source_path="point_cloud.ply",
        min_alpha_coverage=0.5,
    )

    assert [record.image_id for record in manifest.records] == ["virtual/a.png"]


def test_render_virtual_pose_token_manifest_can_uniformly_sample_pose_records(tmp_path):
    pose_path = tmp_path / "virtual_poses.txt"
    pose_path.write_text(
        "\n".join(
            [
                "Visual Landmark Dataset V1",
                "ImageFile, Camera Position [X Y Z W P Q R]",
                "",
                "virtual/a.png 0.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "virtual/b.png 1.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "virtual/c.png 2.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "virtual/d.png 3.0 0.0 0.0 1.0 0.0 0.0 0.0",
                "virtual/e.png 4.0 0.0 0.0 1.0 0.0 0.0 0.0",
            ]
        )
        + "\n"
    )
    records = parse_cambridge_pose_file(pose_path)
    camera = ColmapCamera(camera_id=1, model_id=1, width=4, height=3, params=(2.0, 2.0, 2.0, 1.5))

    def render_fn(**_kwargs):
        return (
            np.full((3, 4, 3), 0.5, dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
        )

    manifest = render_virtual_pose_token_manifest(
        records=records,
        source=object(),
        camera=camera,
        width=4,
        height=3,
        extractor=FakeExtractor(),
        render_rgb_depth_fn=render_fn,
        image_output_root=None,
        token_output_root=tmp_path / "tokens",
        scene="OldHospital",
        split="virtual",
        layer_name="radio_final",
        model_name="c-radio_v4-h",
        storage_dtype="float32",
        renderer_name="official_2dgs",
        source_path="point_cloud.ply",
        max_poses=3,
        view_selection="uniform",
    )

    assert [record.image_id for record in manifest.records] == ["virtual/a.png", "virtual/c.png", "virtual/e.png"]
