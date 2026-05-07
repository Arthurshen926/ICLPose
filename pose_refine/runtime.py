from __future__ import annotations

from typing import Callable, Optional

import torch

from pose_refine.models.concat_pose_net import ConcatPoseNet
from pose_refine.utils.lie_algebra import se3_exp
from feature_field.utils.project_paths import resolve_checkpoint_path, resolve_repo_path


Printer = Callable[[str], None]
RenderBatchFn = Callable[[torch.Tensor, Optional[bool]], dict]


def build_concat_pose_model(model_cfg: dict, device: torch.device | str) -> ConcatPoseNet:
    model = ConcatPoseNet(
        feature_dim=model_cfg.get("feature_dim", 64),
        coarse_feature_dim=model_cfg.get("coarse_feature_dim", model_cfg.get("feature_dim", 64)),
        hidden_dim=model_cfg.get("hidden_dim", 256),
        irls_iters=model_cfg.get("irls_iters", 0),
        robust_kernel=model_cfg.get("robust_kernel", "huber"),
        use_coarse=model_cfg.get("use_coarse", False),
        use_gru=model_cfg.get("use_gru", False),
        use_corr_wls=model_cfg.get("use_corr_wls", False),
        gru_iters=model_cfg.get("gru_iters", 8),
        local_radius=model_cfg.get("local_radius", 4),
        proj_dim=model_cfg.get("proj_dim", 32),
        corr_wls_temperature=model_cfg.get("corr_wls_temperature", 0.04),
        corr_wls_conf_mode=model_cfg.get("corr_wls_conf_mode", "max"),
        corr_wls_conf_variance_scale=model_cfg.get("corr_wls_conf_variance_scale", 0.5),
        coarse_flow_init=model_cfg.get("coarse_flow_init", False),
        coarse_pool_factor=model_cfg.get("coarse_pool_factor", 4),
        use_multiscale_corr=model_cfg.get("use_multiscale_corr", False),
        multiscale_corr_pool_factors=model_cfg.get("multiscale_corr_pool_factors", [4, 2]),
        multiscale_corr_coarse_radius=model_cfg.get("multiscale_corr_coarse_radius", None),
        proj_mode=model_cfg.get("proj_mode", "separate"),
        full_wls=model_cfg.get("full_wls", True),
        use_coarse_in_fine_head=model_cfg.get("use_coarse_in_fine_head", True),
        flow_init=model_cfg.get("flow_init", "zero"),
        detach_trans=model_cfg.get("detach_trans", False),
        detach_wls_rot=model_cfg.get("detach_wls_rot", False),
        rot_mode=model_cfg.get("rot_mode", "wls"),
        use_cross_attention=model_cfg.get("use_cross_attention", False),
        cross_attn_heads=model_cfg.get("cross_attn_heads", 4),
        cross_attn_layers=model_cfg.get("cross_attn_layers", 1),
        cross_attn_downsample=model_cfg.get("cross_attn_downsample", 2),
        cross_attn_dropout=model_cfg.get("cross_attn_dropout", 0.1),
        use_two_stage_refine=model_cfg.get("use_two_stage_refine", False),
        coarse_only_first_iter=model_cfg.get("coarse_only_first_iter", True),
        coarse_stage_hidden_dim=model_cfg.get("coarse_stage_hidden_dim", model_cfg.get("hidden_dim", 256)),
        coarse_stage_use_transformer=model_cfg.get("coarse_stage_use_transformer", True),
        coarse_stage_heads=model_cfg.get("coarse_stage_heads", 4),
        coarse_stage_layers=model_cfg.get("coarse_stage_layers", 1),
        coarse_stage_downsample=model_cfg.get("coarse_stage_downsample", 2),
        coarse_stage_dropout=model_cfg.get("coarse_stage_dropout", 0.1),
        coarse_stage_pool_hw=model_cfg.get("coarse_stage_pool_hw", 4),
        coarse_stage_use_fsm=model_cfg.get("coarse_stage_use_fsm", True),
        pose_update_scale=model_cfg.get("pose_update_scale", 1.0),
        local_matcher_enabled=model_cfg.get("local_matcher_enabled", False),
        local_matcher_hidden_dim=model_cfg.get("local_matcher_hidden_dim", 64),
        local_matcher_zero_init=model_cfg.get("local_matcher_zero_init", True),
        local_matcher_residual_scale=model_cfg.get("local_matcher_residual_scale", 1.0),
        local_matcher_context_mode=model_cfg.get("local_matcher_context_mode", "basic"),
        checkpoint_guided_corr=model_cfg.get("checkpoint_guided_corr", False),
        local_flow_head_enabled=model_cfg.get("local_flow_head_enabled", False),
        local_flow_head_hidden_dim=model_cfg.get("local_flow_head_hidden_dim", 64),
        local_flow_head_zero_init=model_cfg.get("local_flow_head_zero_init", True),
        local_flow_head_max_flow=model_cfg.get("local_flow_head_max_flow", None),
        local_flow_head_base_flow_mode=model_cfg.get("local_flow_head_base_flow_mode", "none"),
        local_flow_head_base_temperature=model_cfg.get("local_flow_head_base_temperature", 0.05),
        local_flow_head_context_mode=model_cfg.get("local_flow_head_context_mode", "basic"),
        corr_wls_use_local_flow_head=model_cfg.get("corr_wls_use_local_flow_head", False),
        pose_update_trans_scale=model_cfg.get("pose_update_trans_scale", 1.0),
        pose_update_rot_scale=model_cfg.get("pose_update_rot_scale", 1.0),
        pose_update_trans_scale_after_first=model_cfg.get("pose_update_trans_scale_after_first", None),
        pose_update_rot_scale_after_first=model_cfg.get("pose_update_rot_scale_after_first", None),
    ).to(device)
    model.direct_featuremetric_use_local_corr_projection = bool(
        model_cfg.get("direct_featuremetric_use_local_corr_projection", False)
    )
    model.feature_select_use_local_corr_projection = bool(
        model_cfg.get("feature_select_use_local_corr_projection", True)
    )
    return model


def load_local_matcher_weights(
    model: ConcatPoseNet,
    checkpoint_path: str,
    device: torch.device | str,
    *,
    strict: bool = False,
) -> object:
    """Load ``local_matcher.*`` weights from a query-student or pose-refine checkpoint."""
    matcher = getattr(model, "local_matcher", None)
    if matcher is None:
        raise RuntimeError("Cannot load local matcher weights: model.local_matcher is disabled")
    resolved_ckpt = resolve_checkpoint_path(checkpoint_path, must_exist=True)
    assert resolved_ckpt is not None
    checkpoint = torch.load(resolved_ckpt, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    matcher_state = {
        key[len("local_matcher."):]: value
        for key, value in state.items()
        if key.startswith("local_matcher.")
    }
    if not matcher_state:
        raise RuntimeError(f"No local_matcher.* weights found in {resolved_ckpt}")
    return matcher.load_state_dict(matcher_state, strict=strict)


def load_local_flow_head_weights(
    model: ConcatPoseNet,
    checkpoint_path: str,
    device: torch.device | str,
    *,
    strict: bool = False,
) -> object:
    """Load ``local_flow_head.*`` weights from a query-student or pose-refine checkpoint."""
    flow_head = getattr(model, "local_flow_head", None)
    if flow_head is None:
        raise RuntimeError("Cannot load local flow head weights: model.local_flow_head is disabled")
    resolved_ckpt = resolve_checkpoint_path(checkpoint_path, must_exist=True)
    assert resolved_ckpt is not None
    checkpoint = torch.load(resolved_ckpt, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    flow_state = {
        key[len("local_flow_head.") :]: value
        for key, value in state.items()
        if key.startswith("local_flow_head.")
    }
    if not flow_state:
        raise RuntimeError(f"No local_flow_head.* weights found in {resolved_ckpt}")
    return flow_head.load_state_dict(flow_state, strict=strict)


def load_external_local_corr_projector(
    model: ConcatPoseNet,
    config: dict,
    device: torch.device | str,
    *,
    printer: Printer | None = print,
) -> bool:
    """Attach a query-student local-correlation projector to a pose refiner.

    The feature-extract stage can train a domain adapter/projector used for
    local query-map correlation. If final localization ignores that module, the
    evaluated correspondence space no longer matches the trained one.
    """
    model_cfg = config.get("model", {}) if isinstance(config, dict) else {}
    explicit = model_cfg.get("external_local_corr_projector", {})
    if explicit is None:
        explicit = {}
    if not isinstance(explicit, dict):
        explicit = {"enabled": bool(explicit)}
    if explicit.get("enabled") is False:
        return False

    manifest = config.get("localization_manifest", {}) if isinstance(config, dict) else {}
    source = manifest.get("source", {}) if isinstance(manifest, dict) else {}
    config_path = explicit.get("config_path") or source.get("config_path")
    checkpoint_path = explicit.get("checkpoint_path") or source.get("checkpoint_path")
    if not config_path or not checkpoint_path:
        return False

    # Local import avoids coupling pose_refine import time to feature_extract.
    from feature_extract.students.radio_query_student import (  # noqa: WPS433
        LocalCorrDomainAdapter,
        LocalCorrProjector,
    )
    from feature_extract.train_impl import load_config  # noqa: WPS433

    source_cfg = load_config(str(config_path))
    source_model_cfg = source_cfg.get("model", {})
    if not bool(source_model_cfg.get("local_corr_projector_enabled", False)):
        return False

    feature_dim = int(getattr(model, "feature_dim", model_cfg.get("feature_dim", 64)))
    hidden_dim = int(source_model_cfg.get("local_corr_projector_hidden_dim", 96))
    output_dim = source_model_cfg.get("local_corr_projector_output_dim")
    zero_init = bool(source_model_cfg.get("local_corr_projector_zero_init", True))
    l2_normalize = bool(source_model_cfg.get("local_corr_projector_l2_normalize", True))
    domain_adapter = bool(source_model_cfg.get("local_corr_projector_domain_adapter", False))

    projector_cls = LocalCorrDomainAdapter if domain_adapter else LocalCorrProjector
    projector = projector_cls(
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        zero_init=zero_init,
        l2_normalize=l2_normalize,
    ).to(device)

    resolved_ckpt = resolve_repo_path(str(checkpoint_path), must_exist=True, enforce_local=False)
    assert resolved_ckpt is not None
    checkpoint = torch.load(resolved_ckpt, map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint)
    projector_state = {
        key[len("local_corr_projector.") :]: value
        for key, value in state.items()
        if key.startswith("local_corr_projector.")
    }
    if not projector_state:
        return False
    projector.load_state_dict(projector_state, strict=True)
    projector.eval()
    for param in projector.parameters():
        param.requires_grad_(False)

    model.external_local_corr_projector = projector
    bypass = explicit.get("bypass_pose_proj", model_cfg.get("external_local_corr_projector_bypass_pose_proj", True))
    model.external_local_corr_projector_bypass_pose_proj = bool(bypass)
    synced = []
    map_cfg = source_cfg.get("map_supervision", {})
    if bool(explicit.get("sync_corr_settings", False)):
        if map_cfg.get("query_corr_temperature") is not None:
            model.corr_wls_temperature = float(map_cfg["query_corr_temperature"])
            synced.append(f"temp={model.corr_wls_temperature:g}")
        if map_cfg.get("query_corr_wls_conf_mode") is not None:
            model.corr_wls_conf_mode = str(map_cfg["query_corr_wls_conf_mode"] or "max")
            synced.append(f"conf={model.corr_wls_conf_mode}")
        if map_cfg.get("query_corr_wls_conf_variance_scale") is not None:
            model.corr_wls_conf_variance_scale = float(map_cfg["query_corr_wls_conf_variance_scale"])
            synced.append(f"var_scale={model.corr_wls_conf_variance_scale:g}")
        if (
            map_cfg.get("query_corr_radius") is not None
            and getattr(model, "local_matcher", None) is None
            and getattr(model, "local_flow_head", None) is None
        ):
            model.local_radius = int(map_cfg["query_corr_radius"])
            synced.append(f"radius={model.local_radius}")
    if bool(explicit.get("sync_pose_update_scale", False)) and map_cfg.get("query_corr_wls_pose_update_scale") is not None:
        model.pose_update_scale = float(map_cfg["query_corr_wls_pose_update_scale"])
        synced.append(f"update_scale={model.pose_update_scale:g}")
    if printer is not None:
        kind = "domain_adapter" if domain_adapter else "shared"
        bypass_label = "bypass" if model.external_local_corr_projector_bypass_pose_proj else "preproject"
        sync_label = f"; synced {', '.join(synced)}" if synced else ""
        printer(f"Loaded external local-corr projector ({kind}, {bypass_label}) from {resolved_ckpt}{sync_label}")
    return True


def apply_pose_delta(
    pose_w2c: torch.Tensor,
    delta_xi: torch.Tensor,
    scale: float = 1.0,
    trans_scale: float = 1.0,
    rot_scale: float = 1.0,
) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=False):
        scaled = delta_xi.float().clone()
        scaled[:, :3] = scaled[:, :3] * float(scale) * float(trans_scale)
        scaled[:, 3:] = scaled[:, 3:] * float(scale) * float(rot_scale)
        return torch.bmm(se3_exp(scaled), pose_w2c.float())


def _pose_update_component_scales(model: ConcatPoseNet, outer_iter: int) -> tuple[float, float]:
    trans_scale = float(getattr(model, "pose_update_trans_scale", 1.0))
    rot_scale = float(getattr(model, "pose_update_rot_scale", 1.0))
    if int(outer_iter) > 0:
        trans_after = getattr(model, "pose_update_trans_scale_after_first", None)
        rot_after = getattr(model, "pose_update_rot_scale_after_first", None)
        if trans_after is not None:
            trans_scale = float(trans_after)
        if rot_after is not None:
            rot_scale = float(rot_after)
    return trans_scale, rot_scale


def run_model_refine_iteration(
    model: ConcatPoseNet,
    render_batch_fn: RenderBatchFn,
    query_fine: torch.Tensor,
    query_coarse: torch.Tensor | None,
    pose_cur: torch.Tensor,
    render_intr: dict,
    *,
    outer_iter: int = 0,
    irls_iters: int | None = None,
    robust_kernel: str | None = None,
    run_coarse_stage: bool | None = None,
    autocast_enabled: bool = True,
    apply_fine_update: bool = True,
) -> dict:
    coarse_bundle = None
    coarse_pred = None
    pose_mid = pose_cur

    if run_coarse_stage is None:
        run_coarse_stage = model.should_run_coarse_stage(outer_iter)
    if run_coarse_stage and query_coarse is not None:
        coarse_bundle = render_batch_fn(pose_cur, True)
        rendered_coarse = coarse_bundle.get("coarse_features")
        if rendered_coarse is not None:
            with torch.cuda.amp.autocast(enabled=autocast_enabled):
                coarse_pred = model.forward_coarse_stage(
                    query_coarse,
                    rendered_coarse,
                    fsm_spatial_conf=coarse_bundle.get("fsm_spatial_conf"),
                )
            update_scale = float(getattr(model, "pose_update_scale", 1.0))
            trans_scale, rot_scale = _pose_update_component_scales(model, outer_iter)
            pose_mid = apply_pose_delta(
                pose_cur,
                coarse_pred["delta_xi"],
                scale=update_scale,
                trans_scale=trans_scale,
                rot_scale=rot_scale,
            )

    needs_fine_coarse_context = bool(
        query_coarse is not None
        and getattr(model, "use_coarse", False)
        and getattr(model, "use_coarse_in_fine_head", False)
    )
    fine_bundle = render_batch_fn(pose_mid, needs_fine_coarse_context)
    with torch.cuda.amp.autocast(enabled=autocast_enabled):
        fine_pred = model.forward_fine_stage(
            query_fine,
            fine_bundle["fine_features"],
            fine_bundle["depth"],
            intrinsics=render_intr,
            query_coarse=query_coarse,
            rendered_coarse=fine_bundle.get("coarse_features"),
            irls_iters=irls_iters,
            robust_kernel=robust_kernel,
        )
    pose_next = pose_mid
    if apply_fine_update and "delta_xi" in fine_pred:
        update_scale = float(getattr(model, "pose_update_scale", 1.0))
        trans_scale, rot_scale = _pose_update_component_scales(model, outer_iter)
        pose_next = apply_pose_delta(
            pose_mid,
            fine_pred["delta_xi"],
            scale=update_scale,
            trans_scale=trans_scale,
            rot_scale=rot_scale,
        )

    return {
        "pose_mid": pose_mid,
        "pose_next": pose_next,
        "coarse_bundle": coarse_bundle,
        "coarse_pred": coarse_pred,
        "fine_bundle": fine_bundle,
        "fine_pred": fine_pred,
        "ran_coarse_stage": coarse_pred is not None,
    }


def load_concat_pose_model(
    config: dict,
    checkpoint_path: str,
    device: torch.device | str,
    *,
    printer: Printer | None = print,
) -> tuple[ConcatPoseNet, int | str]:
    model_cfg = config.get("model", {})
    model = build_concat_pose_model(model_cfg, device)
    resolved_ckpt = resolve_checkpoint_path(checkpoint_path, must_exist=True)
    assert resolved_ckpt is not None
    checkpoint = torch.load(resolved_ckpt, map_location=device)
    local_matcher_init = model_cfg.get("local_matcher_init_checkpoint")
    local_flow_head_init = model_cfg.get("local_flow_head_init_checkpoint")
    if local_matcher_init or local_flow_head_init:
        incompatible = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        allowed_missing_prefixes = []
        if local_matcher_init:
            allowed_missing_prefixes.append("local_matcher.")
        if local_flow_head_init:
            allowed_missing_prefixes.append("local_flow_head.")
        unexpected = list(incompatible.unexpected_keys)
        missing = [
            key
            for key in incompatible.missing_keys
            if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        ]
        if missing or unexpected:
            parts = []
            if missing:
                parts.append(f"missing={missing}")
            if unexpected:
                parts.append(f"unexpected={unexpected}")
            raise RuntimeError("Pose checkpoint is incompatible after optional local-head init: " + "; ".join(parts))
        if local_matcher_init:
            load_local_matcher_weights(model, local_matcher_init, device)
            if printer is not None:
                printer(f"Loaded local matcher weights from {local_matcher_init}")
        if local_flow_head_init:
            load_local_flow_head_weights(model, local_flow_head_init, device)
            if printer is not None:
                printer(f"Loaded local flow head weights from {local_flow_head_init}")
    else:
        model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    epoch = checkpoint.get("epoch", "?")
    if printer is not None:
        printer(f"Loaded localization model from epoch {epoch}")
    return model, epoch


def load_concat_pose_checkpoint(
    checkpoint_path: str,
    device: torch.device | str,
) -> dict:
    resolved_ckpt = resolve_checkpoint_path(checkpoint_path, must_exist=True)
    assert resolved_ckpt is not None
    return torch.load(resolved_ckpt, map_location=device)
