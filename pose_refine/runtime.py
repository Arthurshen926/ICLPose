from __future__ import annotations

from typing import Callable, Optional

import torch

from pose_refine.models.concat_pose_net import ConcatPoseNet
from pose_refine.utils.lie_algebra import se3_exp
from feature_field.utils.project_paths import resolve_checkpoint_path


Printer = Callable[[str], None]
RenderBatchFn = Callable[[torch.Tensor, Optional[bool]], dict]


def build_concat_pose_model(model_cfg: dict, device: torch.device | str) -> ConcatPoseNet:
    return ConcatPoseNet(
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
        coarse_flow_init=model_cfg.get("coarse_flow_init", False),
        coarse_pool_factor=model_cfg.get("coarse_pool_factor", 4),
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
        pose_update_trans_scale=model_cfg.get("pose_update_trans_scale", 1.0),
        pose_update_rot_scale=model_cfg.get("pose_update_rot_scale", 1.0),
        pose_update_trans_scale_after_first=model_cfg.get("pose_update_trans_scale_after_first", None),
        pose_update_rot_scale_after_first=model_cfg.get("pose_update_rot_scale_after_first", None),
    ).to(device)


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
    model = build_concat_pose_model(config.get("model", {}), device)
    resolved_ckpt = resolve_checkpoint_path(checkpoint_path, must_exist=True)
    assert resolved_ckpt is not None
    checkpoint = torch.load(resolved_ckpt, map_location=device)
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
