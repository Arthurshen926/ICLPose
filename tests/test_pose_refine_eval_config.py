from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_extract.export_impl import build_localization_manifest
from feature_field.runtime import _load_gaussian_latent_compatible
from feature_extract.students.radio_query_student import LocalCorrDomainAdapter
from pose_refine.evaluate_impl import (
    resolve_eval_num_workers,
    resolve_eval_pose_update_scale,
)
from pose_refine.models.concat_pose_net import ConcatPoseNet
from pose_refine.runtime import build_concat_pose_model, load_external_local_corr_projector
from pose_refine.train_impl import (
    build_eval_sample_record,
    build_eval_sweep_record,
    resolve_best_metric_value,
)
from feature_field.utils.project_config import (
    apply_localization_manifest_overrides,
    load_localization_manifest,
    load_mainline_config,
    should_restore_pose_checkpoint_map_state,
)
from pose_refine.tools.diag_corr_split import parse_split_names
from pose_refine.tools.eval_pose_update_scale import parse_scales


def test_eval_num_workers_defaults_to_training_config():
    config = {"training": {"num_workers": 0}}

    assert resolve_eval_num_workers(None, config) == 0


def test_eval_num_workers_cli_overrides_config():
    config = {"training": {"num_workers": 0}}

    assert resolve_eval_num_workers(2, config) == 2


def test_eval_pose_update_scale_cli_overrides_model_config():
    config = {"model": {"pose_update_scale": 0.5}}

    assert resolve_eval_pose_update_scale(0.25, config) == 0.25


def test_eval_pose_update_scale_parser_preserves_order():
    assert parse_scales(["0", "0.25", "1.0"]) == [0.0, 0.25, 1.0]


def test_diag_corr_split_parser_expands_aliases():
    assert parse_split_names(["both"]) == ["train", "val"]
    assert parse_split_names(["test", "train"]) == ["val", "train"]


def test_mainline_config_supports_relative_base_config(tmp_path):
    base = tmp_path / "base.yaml"
    child = tmp_path / "child.yaml"
    base.write_text(
        "\n".join(
            [
                "exp_name: base_exp",
                "model:",
                "  full_wls: true",
                "  rot_mode: wls",
                "training:",
                "  epochs: 8",
                "  loss:",
                "    flow_weight: 40.0",
                "    rot_weight: 1.0",
            ]
        ),
        encoding="utf-8",
    )
    child.write_text(
        "\n".join(
            [
                "base_config: base.yaml",
                "exp_name: child_exp",
                "model:",
                "  rot_mode: hybrid",
                "training:",
                "  loss:",
                "    rot_weight: 120.0",
            ]
        ),
        encoding="utf-8",
    )

    cfg = load_mainline_config(str(child))

    assert cfg["exp_name"] == "child_exp"
    assert cfg["model"]["full_wls"] is True
    assert cfg["model"]["rot_mode"] == "hybrid"
    assert cfg["training"]["epochs"] == 8
    assert cfg["training"]["loss"]["flow_weight"] == 40.0
    assert cfg["training"]["loss"]["rot_weight"] == 120.0


def test_localization_manifest_overrides_pose_refine_feature_and_dcff_paths(tmp_path):
    feature_dir = tmp_path / "exported_query"
    feature_dir.mkdir()
    joint_ckpt = tmp_path / "joint.pth"
    joint_ckpt.write_bytes(b"joint checkpoint")
    val_init = tmp_path / "fixed_init.npz"
    val_init.write_bytes(b"fixed init cache")
    manifest_path = tmp_path / "localization_manifest.json"
    manifest_path.write_text(
        "\n".join(
            [
                "{",
                '  "schema_version": 1,',
                '  "dataset": {',
                f'    "feature_dir": "{feature_dir}",',
                f'    "val_init_poses_path": "{val_init}"',
                "  },",
                '  "dcff": {',
                f'    "joint_checkpoint": "{joint_ckpt}",',
                '    "joint_override_components": ["fine_decoder", "feat_sharp", "fsm"]',
                "  },",
                '  "export": {"fine_key": "fine_loc"},',
                '  "init_caches": {',
                f'    "val": {{"path": "{val_init}", "sha256": "unused"}}',
                "  }",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    cfg = {
        "dataset": {"feature_dir": "old_features"},
        "dcff": {"joint_checkpoint": "old.pth"},
        "model": {"feature_dim": 64},
    }
    loaded = load_localization_manifest(str(manifest_path))
    updated = apply_localization_manifest_overrides(cfg, loaded)

    assert updated["dataset"]["feature_dir"] == str(feature_dir)
    assert updated["dataset"]["val_init_poses_path"] == str(val_init)
    assert updated["dcff"]["joint_checkpoint"] == str(joint_ckpt)
    assert updated["dcff"]["joint_override_components"] == ["fine_decoder", "feat_sharp", "fsm"]
    assert updated["export"]["fine_key"] == "fine_loc"
    assert updated["localization_manifest"]["path"] == str(manifest_path)
    assert updated["localization_manifest"]["init_caches"]["val"]["sha256"] != "unused"


def test_joint_checkpoint_takes_precedence_over_pose_checkpoint_map_state():
    cfg = {"dcff": {"joint_checkpoint": "/tmp/joint.pth"}}

    assert should_restore_pose_checkpoint_map_state(cfg) is False


def test_pose_checkpoint_map_state_can_be_explicitly_restored():
    cfg = {
        "dcff": {
            "joint_checkpoint": "/tmp/joint.pth",
            "restore_pose_checkpoint_map_state": True,
        }
    }

    assert should_restore_pose_checkpoint_map_state(cfg) is True


def test_external_local_corr_projector_loads_from_manifest_source(tmp_path):
    source_config = tmp_path / "source.yaml"
    source_config.write_text(
        "\n".join(
            [
                "model:",
                "  local_corr_projector_enabled: true",
                "  local_corr_projector_domain_adapter: true",
                "  local_corr_projector_hidden_dim: 8",
                "  local_corr_projector_output_dim: 4",
                "  local_corr_projector_zero_init: true",
                "  local_corr_projector_l2_normalize: true",
                "map_supervision:",
                "  query_corr_temperature: 0.03",
                "  query_corr_wls_conf_mode: variance",
                "  query_corr_wls_conf_variance_scale: 0.5",
                "  query_corr_wls_pose_update_scale: 0.25",
            ]
        ),
        encoding="utf-8",
    )
    projector = LocalCorrDomainAdapter(
        feature_dim=4,
        hidden_dim=8,
        output_dim=4,
        zero_init=True,
        l2_normalize=True,
    )
    checkpoint = tmp_path / "source.pth"
    torch_state = {
        f"local_corr_projector.{key}": value
        for key, value in projector.state_dict().items()
    }
    import torch

    torch.save({"model_state_dict": torch_state}, checkpoint)
    model = ConcatPoseNet(feature_dim=4, use_corr_wls=True, full_wls=True, proj_mode="identity")

    loaded = load_external_local_corr_projector(
        model,
        {
            "model": {
                "external_local_corr_projector": {
                    "sync_corr_settings": True,
                    "sync_pose_update_scale": True,
                }
            },
            "localization_manifest": {
                "source": {
                    "config_path": str(source_config),
                    "checkpoint_path": str(checkpoint),
                }
            }
        },
        "cpu",
        printer=None,
    )

    assert loaded is True
    assert model.external_local_corr_projector is not None
    assert model.external_local_corr_projector_bypass_pose_proj is True
    assert model.corr_wls_temperature == 0.03
    assert model.corr_wls_conf_mode == "variance"
    assert model.corr_wls_conf_variance_scale == 0.5
    assert model.pose_update_scale == 0.25


def test_build_pose_model_records_projected_featuremetric_flags():
    model = build_concat_pose_model(
        {
            "feature_dim": 4,
            "direct_featuremetric_use_local_corr_projection": True,
            "feature_select_use_local_corr_projection": False,
        },
        "cpu",
    )

    assert model.direct_featuremetric_use_local_corr_projection is True
    assert model.feature_select_use_local_corr_projection is False


def test_export_builds_localization_manifest_with_hashes(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("exp_name: loc\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "features"
    output_dir.mkdir()
    val_init = tmp_path / "val_init.npz"
    val_init.write_bytes(b"val init")

    manifest = build_localization_manifest(
        config_path=str(config),
        checkpoint_path=str(checkpoint),
        output_dir=str(output_dir),
        cfg={"model": {"export_fine_key": "fine_loc"}},
        val_init_cache=str(val_init),
        joint_override_components=["fine_decoder", "feat_sharp", "fsm"],
    )

    assert manifest["dataset"]["feature_dir"] == str(output_dir.resolve())
    assert manifest["dcff"]["joint_checkpoint"] == str(checkpoint.resolve())
    assert manifest["dcff"]["joint_override_components"] == ["fine_decoder", "feat_sharp", "fsm"]
    assert manifest["export"]["fine_key"] == "fine_loc"
    assert manifest["source"]["config_sha256"]
    assert manifest["init_caches"]["val"]["sha256"]


def test_export_manifest_default_joint_overrides_include_gaussian_latent(tmp_path):
    config = tmp_path / "train.yaml"
    config.write_text("exp_name: loc\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "features"
    output_dir.mkdir()

    manifest = build_localization_manifest(
        config_path=str(config),
        checkpoint_path=str(checkpoint),
        output_dir=str(output_dir),
        cfg={},
    )

    assert "hash_grid_mlp" in manifest["dcff"]["joint_override_components"]
    assert "gaussian_latent" in manifest["dcff"]["joint_override_components"]


def test_export_manifest_includes_base_dcff_map_config(tmp_path):
    map_config = tmp_path / "map.yaml"
    map_checkpoint = tmp_path / "dcff_best.pth"
    map_ply = tmp_path / "dcff_best.ply"
    map_checkpoint.write_bytes(b"dcff checkpoint")
    map_ply.write_bytes(b"dcff ply")
    map_config.write_text(
        "\n".join(
            [
                "dcff:",
                f"  checkpoint: {map_checkpoint}",
                f"  ply_path: {map_ply}",
                "  feature_dim: 64",
                "  fine_feature_dim: 64",
                "  coarse_feature_dim: 64",
                "  latent_dim: 64",
                "  fine_latent_dim: 40",
                "  coarse_latent_dim: 24",
                "  render_width: 120",
                "  render_height: 68",
                "  coarse_smoothing_kernel: 1",
                "  fine_decoder_override:",
                "    type: residual_spatial",
                "    hidden_dim: 224",
                "    num_layers: 5",
                "    use_viewdirs: false",
            ]
        ),
        encoding="utf-8",
    )
    config = tmp_path / "train.yaml"
    config.write_text("exp_name: loc\n", encoding="utf-8")
    checkpoint = tmp_path / "best.pth"
    checkpoint.write_bytes(b"checkpoint")
    output_dir = tmp_path / "features"
    output_dir.mkdir()

    manifest = build_localization_manifest(
        config_path=str(config),
        checkpoint_path=str(checkpoint),
        output_dir=str(output_dir),
        cfg={"map_supervision": {"config_path": str(map_config)}},
    )

    assert manifest["dcff"]["checkpoint"] == str(map_checkpoint.resolve())
    assert manifest["dcff"]["ply_path"] == str(map_ply.resolve())
    assert manifest["dcff"]["fine_decoder_override"]["type"] == "residual_spatial"
    assert manifest["dcff"]["fine_decoder_override"]["hidden_dim"] == 224
    assert manifest["dcff"]["fine_latent_dim"] == 40


def test_joint_map_state_can_restore_gaussian_latent():
    class DummyGaussians:
        def __init__(self):
            self._latent = torch.zeros(2, 3)

    gaussians = DummyGaussians()

    loaded = _load_gaussian_latent_compatible(
        gaussians,
        torch.ones(2, 3),
        "joint gaussian_latent",
        printer=None,
    )

    assert loaded is True
    assert torch.equal(gaussians._latent, torch.ones(2, 3))


def test_pose_refine_best_metric_defaults_to_lower_translation():
    value, higher_is_better, label = resolve_best_metric_value(
        {"val_trans_median": 73.0, "val_rot_median": 0.64},
        "trans_median",
    )

    assert value == 73.0
    assert higher_is_better is False
    assert label == "val_trans_median"


def test_pose_refine_best_metric_can_maximize_joint_accuracy():
    value, higher_is_better, label = resolve_best_metric_value(
        {"val_joint_1deg_50mm": 17.6, "val_trans_median": 73.0},
        "joint_1deg_50mm",
    )

    assert value == 17.6
    assert higher_is_better is True
    assert label == "val_joint_1deg_50mm"


def test_eval_sweep_record_preserves_core_metrics():
    record = build_eval_sweep_record(
        outer_iters=3,
        gru_iters=2,
        metrics={
            "val_rot_median": 0.36,
            "val_trans_median": 73.6,
            "val_pct_1deg": 86.8,
            "val_joint_1deg_50mm": 20.9,
        },
        seed_count=1,
    )

    assert record["outer_iters"] == 3
    assert record["gru_iters"] == 2
    assert record["seed_count"] == 1
    assert record["rot_median_deg"] == 0.36
    assert record["trans_median_mm"] == 73.6
    assert record["pct_1deg"] == 86.8
    assert record["joint_1deg_50mm"] == 20.9


def test_eval_sample_record_preserves_identity_and_stage_errors():
    record = build_eval_sample_record(
        image_id=42,
        image_name="seq1/frame00042.png",
        outer_iters=5,
        gru_iters=2,
        seed=0,
        init_rot_deg=0.64,
        init_trans_mm=74.0,
        one_rot_deg=0.47,
        one_trans_mm=73.2,
        final_rot_deg=0.34,
        final_trans_mm=73.2,
        flow_epe_px=2.95,
    )

    assert record["image_id"] == 42
    assert record["image_name"] == "seq1/frame00042.png"
    assert record["outer_iters"] == 5
    assert record["gru_iters"] == 2
    assert record["seed"] == 0
    assert record["final_rot_deg"] == 0.34
    assert record["final_trans_mm"] == 73.2
    assert record["flow_epe_px"] == 2.95


if __name__ == "__main__":
    test_eval_num_workers_defaults_to_training_config()
    test_eval_num_workers_cli_overrides_config()
    test_eval_pose_update_scale_cli_overrides_model_config()
    test_eval_pose_update_scale_parser_preserves_order()
    test_diag_corr_split_parser_expands_aliases()
    test_eval_sweep_record_preserves_core_metrics()
    test_eval_sample_record_preserves_identity_and_stage_errors()
    print("pose refine eval config tests passed")
