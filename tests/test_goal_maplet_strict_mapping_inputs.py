import json

from feature_extract.tools.vfm.build_goal_maplet_strict_mapping_inputs import build_inputs
from feature_extract.vfm.official_oof_protocol import ordered_id_sha256


def test_strict_mapping_inputs_exclude_held_route(tmp_path):
    token = tmp_path / "token.npz"
    token.write_bytes(b"token")
    ids = ["seq1/frame00001.png", "seq2/frame00001.png"]
    pose = tmp_path / "train.txt"
    pose.write_text(
        "Visual Landmark Dataset V1\n"
        "ImageFile, Camera Position [X Y Z W P Q R]\n\n"
        + "\n".join(f"{value} 0 0 0 1 0 0 0" for value in ids)
        + "\n"
    )
    manifest = tmp_path / "train.json"
    manifest.write_text(json.dumps({"records": [
        {
            "image_id": value, "token_path": str(token), "split": "train",
            "scene": "scene", "layers": [{"name": "radio_final", "model": "radio",
            "layer": "final", "channels": 4, "stride": 16}],
        } for value in ids
    ]}))
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({
        "official_train": {
            "count": 2, "trajectory_counts": {"seq1": 1, "seq2": 1},
            "image_ids_sha256": ordered_id_sha256(ids),
            "pose_file": str(pose), "token_manifest": str(manifest),
        },
        "development": {"folds": [{
            "fold_id": "fold0", "mapping_trajectories": ["seq1"],
            "held_query_trajectories": ["seq2"], "mapping_count": 1,
            "mapping_image_ids_sha256": ordered_id_sha256(ids[:1]),
        }]},
    }))
    out_manifest = tmp_path / "fold" / "manifest.json"
    out_pose = tmp_path / "fold" / "poses.txt"
    report = build_inputs(
        protocol_path=protocol, fold_id="fold0",
        output_manifest=out_manifest, output_pose_file=out_pose,
        output_json=tmp_path / "fold" / "inputs.json",
    )
    assert report["mapping_image_count"] == 1
    assert report["contains_held_query"] is False
    assert "seq2/" not in out_manifest.read_text()
    assert "seq2/" not in out_pose.read_text()
