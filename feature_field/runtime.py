from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from feature_field.dcff import DeferredCascadedRenderer, SpatialHashGrid
from feature_field.dcff.feature_selection import FeatureSelectionModule
from feature_field.dcff.hybrid_gaussian import HybridGaussianModel
from feature_field.utils.checkpoint_io import safe_torch_load
from feature_field.utils.project_paths import resolve_checkpoint_path, resolve_repo_path


Printer = Callable[[str], None]


def _print(printer: Printer | None, message: str) -> None:
    if printer is None:
        return
    printer(message)


@dataclass
class DCFFRuntime:
    gaussians: HybridGaussianModel
    hash_grid: SpatialHashGrid
    renderer: DeferredCascadedRenderer
    refiner: nn.Module
    feat_select: nn.Module | None
    render_height: int
    render_width: int
    finetune_decoder: bool
    finetune_fsm: bool


def _resolve_feature_hw(
    feature_hw: tuple[int, int] | None,
    render_h: int,
    render_w: int,
) -> tuple[int, int]:
    if feature_hw is None:
        return render_h, render_w
    return int(feature_hw[0]), int(feature_hw[1])


def _apply_dcff_postprocess(
    result: dict[str, torch.Tensor],
    feat_h: int,
    feat_w: int,
    *,
    feat_sharp: nn.Module | None = None,
    feat_select: nn.Module | None = None,
    use_coarse_for_fsm: bool = True,
    temperature: float = 0.5,
    hard: bool = False,
) -> dict[str, torch.Tensor]:
    result = dict(result)
    depth = result.get("depth")
    alpha = result.get("alpha")

    depth_feat = None
    alpha_feat = None
    if depth is not None:
        depth_feat = depth if depth.shape[-2:] == (feat_h, feat_w) else F.interpolate(
            depth,
            (feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        )
    if alpha is not None:
        alpha_feat = alpha if alpha.shape[-2:] == (feat_h, feat_w) else F.interpolate(
            alpha,
            (feat_h, feat_w),
            mode="bilinear",
            align_corners=False,
        )

    if feat_sharp is not None:
        result["fine_features"] = feat_sharp(
            result["fine_features"].float(),
            depth=depth_feat.float() if depth_feat is not None else None,
            alpha=alpha_feat.float() if alpha_feat is not None else None,
        )

    if feat_select is not None:
        orig_coarse = result.get("coarse_features")
        coarse_for_fsm = orig_coarse if use_coarse_for_fsm else None
        if coarse_for_fsm is None:
            coarse_for_fsm = result["fine_features"]
        elif coarse_for_fsm.shape[-2:] != (feat_h, feat_w):
            coarse_for_fsm = F.interpolate(
                coarse_for_fsm,
                (feat_h, feat_w),
                mode="bilinear",
                align_corners=False,
            )

        if alpha_feat is None:
            alpha_feat = torch.ones(
                result["fine_features"].shape[0],
                1,
                feat_h,
                feat_w,
                device=result["fine_features"].device,
                dtype=result["fine_features"].dtype,
            )
        if depth_feat is None:
            depth_feat = torch.zeros(
                result["fine_features"].shape[0],
                1,
                feat_h,
                feat_w,
                device=result["fine_features"].device,
                dtype=result["fine_features"].dtype,
            )

        fsm_result = feat_select(
            fine_features=result["fine_features"],
            coarse_features=coarse_for_fsm,
            alpha=alpha_feat.float(),
            depth=depth_feat.float(),
            temperature=temperature,
            hard=hard,
        )
        result["fine_features"] = fsm_result["fine_features"]
        if use_coarse_for_fsm and orig_coarse is not None:
            result["coarse_features"] = fsm_result["coarse_features"]
        result["fsm_spatial_conf"] = fsm_result["spatial_confidence"]
        result["fsm_channel_weights"] = fsm_result["channel_weights"]

    return result


class FeatSharp(nn.Module):
    """Lightweight learnable sharpening applied after rasterization."""

    def __init__(self, feature_dim: int, kernel_size: int = 3):
        super().__init__()
        self.sharpen = nn.Sequential(
            nn.Conv2d(
                feature_dim,
                feature_dim,
                kernel_size,
                padding=kernel_size // 2,
                groups=feature_dim,
            ),
            nn.Conv2d(feature_dim, feature_dim, 1),
        )
        for module in self.sharpen:
            if isinstance(module, nn.Conv2d):
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                if module.kernel_size == (1, 1):
                    nn.init.zeros_(module.weight)
                else:
                    nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                    module.weight.data *= 0.01

    def forward(self, x: torch.Tensor, depth: torch.Tensor | None = None, alpha: torch.Tensor | None = None) -> torch.Tensor:
        return x + self.sharpen(x)


class DepthGuidedRefiner(nn.Module):
    """Spatial refinement conditioned on depth and alpha."""

    def __init__(self, feature_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        input_dim = feature_dim + 2
        self.refiner = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, feature_dim, 1),
        )
        nn.init.zeros_(self.refiner[-1].weight)
        nn.init.zeros_(self.refiner[-1].bias)

    def forward(
        self,
        features: torch.Tensor,
        depth: torch.Tensor | None = None,
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if depth is None or alpha is None:
            return features
        fh, fw = features.shape[-2:]
        if depth.shape[-2:] != (fh, fw):
            depth = F.interpolate(depth, (fh, fw), mode="bilinear", align_corners=False)
        if alpha.shape[-2:] != (fh, fw):
            alpha = F.interpolate(alpha, (fh, fw), mode="bilinear", align_corners=False)
        depth_norm = depth / (depth.amax(dim=(-2, -1), keepdim=True) + 1e-6)
        x = torch.cat([features, depth_norm, alpha], dim=1)
        return features + self.refiner(x)


def intrinsics_to_K(intrinsics: dict[str, float], device: torch.device | str) -> torch.Tensor:
    return torch.tensor(
        [
            [intrinsics["fx"], 0, intrinsics["cx"]],
            [0, intrinsics["fy"], intrinsics["cy"]],
            [0, 0, 1],
        ],
        dtype=torch.float32,
        device=device,
    )


def _deep_update_dict(base: dict, override: dict | None) -> dict:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_state_dict_compatible(
    module: nn.Module,
    state_dict: dict | None,
    label: str,
    *,
    printer: Printer | None = print,
) -> bool:
    if not state_dict:
        return False
    try:
        module.load_state_dict(state_dict, strict=True)
        _print(printer, f"  Loaded {label}")
        return True
    except (RuntimeError, KeyError) as exc:
        _print(printer, f"  Skipping {label} (architecture changed): {exc}")
        return False


def _restore_gaussian_state(
    gaussians: HybridGaussianModel,
    checkpoint: dict,
    *,
    printer: Printer | None = print,
) -> bool:
    gs = checkpoint.get("gaussians_state")
    if not gs:
        return False
    try:
        with torch.no_grad():
            gaussians._xyz.data.copy_(gs["_xyz"].to(gaussians._xyz.device))
            gaussians._rotation.data.copy_(gs["_rotation"].to(gaussians._rotation.device))
            gaussians._scaling.data.copy_(gs["_scaling"].to(gaussians._scaling.device))
            gaussians._opacity.data.copy_(gs["_opacity"].to(gaussians._opacity.device))
            gaussians._features_dc.data.copy_(gs["_features_dc"].to(gaussians._features_dc.device))
            gaussians._features_rest.data.copy_(gs["_features_rest"].to(gaussians._features_rest.device))
            gaussians.active_sh_degree = int(gs.get("active_sh_degree", gaussians.active_sh_degree))
        _print(printer, "  Restored Gaussian geometry/state from checkpoint")
        return True
    except KeyError as exc:
        _print(printer, f"  WARNING: incomplete gaussians_state in checkpoint, skipping ({exc})")
        return False


def apply_localization_map_state(
    dcff_renderer: DeferredCascadedRenderer,
    feat_sharp: nn.Module,
    checkpoint_or_path: dict | str | Path,
    *,
    printer: Printer | None = print,
) -> list[str]:
    if isinstance(checkpoint_or_path, dict):
        checkpoint = checkpoint_or_path
    else:
        ckpt_path = resolve_checkpoint_path(str(checkpoint_or_path), must_exist=True)
        assert ckpt_path is not None
        checkpoint = safe_torch_load(
            ckpt_path,
            map_location=next(dcff_renderer.parameters()).device,
        )

    loaded: list[str] = []
    if _load_state_dict_compatible(
        dcff_renderer.fine_decoder,
        checkpoint.get("fine_decoder_state"),
        "localization fine_decoder",
        printer=printer,
    ):
        loaded.append("fine_decoder")
    if getattr(dcff_renderer, "coarse_carrier_fusion", None) is not None and _load_state_dict_compatible(
        dcff_renderer.coarse_carrier_fusion,
        checkpoint.get("coarse_fusion_state"),
        "localization coarse_fusion",
        printer=printer,
    ):
        loaded.append("coarse_fusion")
    if _load_state_dict_compatible(
        feat_sharp,
        checkpoint.get("feat_sharp_state"),
        "localization feat_sharp",
        printer=printer,
    ):
        loaded.append("feat_sharp")
    feat_select = getattr(dcff_renderer, "_feat_select", None)
    if feat_select is not None and _load_state_dict_compatible(
        feat_select,
        checkpoint.get("fsm_state"),
        "localization FSM",
        printer=printer,
    ):
        loaded.append("fsm")
    return loaded


def build_dcff_runtime(
    config: dict,
    device: torch.device | str,
    *,
    printer: Printer | None = print,
) -> DCFFRuntime:
    cfg_dcff = config.get("dcff", {})
    dcff_ckpt_path = resolve_checkpoint_path(cfg_dcff.get("checkpoint"), must_exist=True)
    cfg_ply_path = resolve_repo_path(cfg_dcff.get("ply_path"), must_exist=False)
    joint_ckpt_path = resolve_checkpoint_path(cfg_dcff.get("joint_checkpoint"))

    assert dcff_ckpt_path is not None
    checkpoint = safe_torch_load(dcff_ckpt_path, map_location=device)
    ckpt_ply_path = dcff_ckpt_path.with_suffix(".ply")
    if ckpt_ply_path.exists():
        ply_path = ckpt_ply_path
        _print(printer, f"  Using checkpoint geometry PLY: {ply_path}")
    elif cfg_ply_path is not None and cfg_ply_path.exists():
        ply_path = cfg_ply_path
        _print(printer, f"  Using configured geometry PLY: {ply_path}")
    else:
        raise FileNotFoundError(
            f"No geometry PLY found for DCFF runtime. "
            f"Tried checkpoint sibling {ckpt_ply_path} and configured path {cfg_dcff.get('ply_path')}"
        )

    latent_dim = cfg_dcff.get("latent_dim", 32)
    feature_dim = cfg_dcff.get("feature_dim", 64)

    gaussians = HybridGaussianModel(sh_degree=3, latent_dim=latent_dim)
    gaussians.load_ply(str(ply_path), freeze_geometry=True)
    _print(printer, f"  Loaded {gaussians.num_points:,} Gaussians from {ply_path}")
    _restore_gaussian_state(gaussians, checkpoint, printer=printer)
    gaussians.active_sh_degree = 3

    xyz = gaussians.get_xyz.detach().float()
    xyz_norm = torch.linalg.norm(xyz, dim=1)
    if xyz_norm.numel() == 0:
        scene_extent = 1.0
    else:
        scene_extent = float(torch.quantile(xyz_norm, 0.99).item()) * 1.2
    _print(printer, f"  Scene extent: {scene_extent:.2f}")

    dcff_config_path = dcff_ckpt_path.parent.parent / "config.yaml"
    dcff_cfg = {}
    if dcff_config_path.is_file():
        with open(dcff_config_path, "r", encoding="utf-8") as handle:
            dcff_cfg = yaml.safe_load(handle) or {}
        _print(printer, f"  Loaded DCFF config from {dcff_config_path}")

    fine_decoder_override = cfg_dcff.get("fine_decoder_override")
    if fine_decoder_override:
        dcff_cfg = _deep_update_dict(dcff_cfg, {"fine_decoder": fine_decoder_override})
        _print(printer, f"  Overriding fine_decoder config: {fine_decoder_override}")

    refiner_override = cfg_dcff.get("refiner_override")
    if refiner_override:
        dcff_cfg = _deep_update_dict(dcff_cfg, {"refiner": refiner_override})
        _print(printer, f"  Overriding refiner config: {refiner_override}")

    coarse_decoder_override = cfg_dcff.get("coarse_decoder_override") or cfg_dcff.get("coarse_decoder")
    if coarse_decoder_override:
        dcff_cfg = _deep_update_dict(dcff_cfg, {"coarse_decoder": coarse_decoder_override})
        _print(printer, f"  Overriding coarse_decoder config: {coarse_decoder_override}")

    hcfg = dcff_cfg.get("hash_grid", {})
    fcfg = dcff_cfg.get("fine_decoder", {})
    ccfg = dcff_cfg.get("coarse_decoder", {})
    mcfg = dcff_cfg.get("model", {})
    fsm_cfg = dcff_cfg.get("fsm", {})

    hash_grid = SpatialHashGrid(
        scene_extent=hcfg.get("scene_extent", scene_extent),
        feature_dim=feature_dim,
        input_mode=hcfg.get("input_mode", "implicit_scale"),
        latent_dim=latent_dim,
        n_levels=hcfg.get("n_levels", 16),
        n_features_per_level=hcfg.get("n_features_per_level", 2),
        log2_hashmap_size=hcfg.get("log2_hashmap_size", 19),
        base_resolution=hcfg.get("base_resolution", 16),
        max_resolution=hcfg.get("max_resolution", 2048),
        mlp_hidden=hcfg.get("mlp_hidden", 128),
        mlp_layers=hcfg.get("mlp_layers", 2),
        scale_pe_freqs=hcfg.get("scale_pe_freqs", 4),
        include_raw_scale=hcfg.get("include_raw_scale", True),
    ).to(device)

    renderer = DeferredCascadedRenderer(
        hash_grid=hash_grid,
        latent_dim=latent_dim,
        fine_feature_dim=feature_dim,
        coarse_feature_dim=feature_dim,
        fine_hidden_dim=fcfg.get("hidden_dim", 128),
        fine_num_layers=fcfg.get("num_layers", 3),
        fine_use_viewdirs=fcfg.get("use_viewdirs", False),
        fine_view_degree=fcfg.get("view_degree", 2),
        fine_decoder_type=fcfg.get("type", "mlp"),
        coarse_mode=ccfg.get("mode", "implicit_only"),
        coarse_carrier_hidden_dim=ccfg.get("carrier_hidden_dim"),
        coarse_gate_hidden_dim=ccfg.get("gate_hidden_dim"),
        coarse_smoothing_kernel=cfg_dcff.get("coarse_smoothing_kernel", mcfg.get("coarse_smoothing_kernel", 1)),
    ).to(device)

    feat_select = None
    if fsm_cfg.get("enable", False):
        feat_select = FeatureSelectionModule(
            feature_dim=feature_dim,
            hidden_dim=int(fsm_cfg.get("hidden_dim", 32)),
            num_heads=int(fsm_cfg.get("num_heads", 4)),
            channel_routing_mode=fsm_cfg.get("channel_routing_mode", "categorical"),
            use_channel_select=fsm_cfg.get("use_channel_select", True),
            use_cross_attn=fsm_cfg.get("use_cross_attn", True),
            use_spatial_conf=fsm_cfg.get("use_spatial_conf", True),
        ).to(device)
        n_fsm = sum(p.numel() for p in feat_select.parameters())
        _print(printer, f"  FSM enabled: {n_fsm:,} params")

    refiner_type = dcff_cfg.get("refiner", {}).get("type")
    refiner_state = checkpoint.get("feat_sharp_fine_state", {})
    if refiner_type == "depth_guided":
        hidden_dim = dcff_cfg.get("refiner", {}).get("hidden_dim", 128)
        refiner = DepthGuidedRefiner(feature_dim, hidden_dim=hidden_dim).to(device)
        _print(printer, f"  Using DepthGuidedRefiner (hidden={hidden_dim}) [from config]")
    elif refiner_type == "featsharp":
        refiner = FeatSharp(feature_dim).to(device)
        _print(printer, "  Using FeatSharp [from config]")
    elif (
        refiner_type is None
        and "refiner.0.weight" in refiner_state
        and refiner_state["refiner.0.weight"].shape[1] == feature_dim + 2
    ):
        hidden_dim = refiner_state["refiner.0.weight"].shape[0]
        refiner = DepthGuidedRefiner(feature_dim, hidden_dim=hidden_dim).to(device)
        _print(printer, f"  Using DepthGuidedRefiner (hidden={hidden_dim}) [auto-detected]")
    else:
        refiner = FeatSharp(feature_dim).to(device)
        _print(printer, "  Using FeatSharp")

    _print(printer, f"  Loading DCFF checkpoint: {dcff_ckpt_path}")
    hash_grid.load_state_dict(checkpoint["hash_grid_state"])
    _load_state_dict_compatible(
        renderer.fine_decoder,
        checkpoint.get("fine_decoder_state"),
        "fine_decoder weights",
        printer=printer,
    )
    if renderer.coarse_carrier_fusion is not None:
        _load_state_dict_compatible(
            renderer.coarse_carrier_fusion,
            checkpoint.get("coarse_fusion_state"),
            "coarse carrier fusion",
            printer=printer,
        )
    _load_state_dict_compatible(
        refiner,
        refiner_state,
        "feat_sharp weights",
        printer=printer,
    )
    if feat_select is not None:
        _load_state_dict_compatible(
            feat_select,
            checkpoint.get("fsm_state"),
            "FSM weights",
            printer=printer,
        )
    if "latent" in checkpoint:
        saved_latent = checkpoint["latent"].to(device)
        if saved_latent.shape == gaussians._latent.shape:
            with torch.no_grad():
                gaussians._latent.data.copy_(saved_latent)
            _print(printer, "  Restored latent embeddings from checkpoint")
        else:
            _print(
                printer,
                f"  WARNING: Latent shape mismatch: ckpt={saved_latent.shape} "
                f"vs model={gaussians._latent.shape}, skipping",
            )

    if joint_ckpt_path:
        joint_ckpt = resolve_checkpoint_path(joint_ckpt_path, must_exist=True)
        assert joint_ckpt is not None
        _print(printer, f"  Loading joint feature checkpoint: {joint_ckpt}")
        joint_state = safe_torch_load(joint_ckpt, map_location=device)
        joint_map_state = joint_state.get("map_renderer_state_dict") or {}
        # Optionally restrict which components to override from joint checkpoint
        allowed = cfg_dcff.get("joint_override_components")
        if allowed is not None:
            _print(printer, f"  Joint override restricted to: {allowed}")
        loaded_components: list[str] = []
        if "fine_decoder" in joint_map_state and (allowed is None or "fine_decoder" in allowed):
            if _load_state_dict_compatible(
                renderer.fine_decoder,
                joint_map_state["fine_decoder"],
                "joint fine_decoder",
                printer=printer,
            ):
                loaded_components.append("fine_decoder")
        if "feat_sharp" in joint_map_state and (allowed is None or "feat_sharp" in allowed):
            if _load_state_dict_compatible(
                refiner,
                joint_map_state["feat_sharp"],
                "joint feat_sharp",
                printer=printer,
            ):
                loaded_components.append("feat_sharp")
        if "hash_grid_mlp" in joint_map_state and (allowed is None or "hash_grid_mlp" in allowed):
            if _load_state_dict_compatible(
                renderer.hash_grid.mlp,
                joint_map_state["hash_grid_mlp"],
                "joint hash_grid_mlp",
                printer=printer,
            ):
                loaded_components.append("hash_grid_mlp")
        if feat_select is not None and "fsm" in joint_map_state and (allowed is None or "fsm" in allowed):
            if _load_state_dict_compatible(
                feat_select,
                joint_map_state["fsm"],
                "joint FSM",
                printer=printer,
            ):
                loaded_components.append("fsm")
        if loaded_components:
            _print(
                printer,
                f"  Overrode DCFF modules from joint checkpoint: {', '.join(loaded_components)}",
            )
        else:
            _print(
                printer,
                "  WARNING: joint checkpoint has no map_renderer_state_dict overrides; keeping base DCFF weights",
            )

    dcff_iter = checkpoint.get("iteration", "?")
    _print(printer, f"  DCFF loaded (iter {dcff_iter})")

    train_cfg = dcff_cfg.get("training", {})
    coarse_start_iter = int(train_cfg.get("coarse_start_iter", train_cfg.get("coarse_start", 0)))
    fsm_use_coarse_default = bool(
        feat_select is not None
        and isinstance(dcff_iter, (int, float))
        and dcff_iter >= coarse_start_iter
    )
    fsm_use_coarse = bool(cfg_dcff.get("fsm_use_coarse", fsm_use_coarse_default))

    gaussians._latent.requires_grad_(False)
    for param in hash_grid.parameters():
        param.requires_grad_(False)
    for param in renderer.parameters():
        param.requires_grad_(False)
    for param in refiner.parameters():
        param.requires_grad_(False)
    if feat_select is not None:
        for param in feat_select.parameters():
            param.requires_grad_(False)
    hash_grid.eval()
    renderer.eval()
    refiner.eval()
    if feat_select is not None:
        feat_select.eval()

    finetune_decoder = bool(cfg_dcff.get("finetune_decoder", False))
    finetune_fsm = bool(cfg_dcff.get("finetune_fsm", finetune_decoder and feat_select is not None))
    if finetune_decoder:
        for param in renderer.fine_decoder.parameters():
            param.requires_grad_(True)
        if renderer.coarse_carrier_fusion is not None:
            for param in renderer.coarse_carrier_fusion.parameters():
                param.requires_grad_(True)
        for param in refiner.parameters():
            param.requires_grad_(True)
        renderer.fine_decoder.train()
        if renderer.coarse_carrier_fusion is not None:
            renderer.coarse_carrier_fusion.train()
        refiner.train()
        n_decoder = sum(p.numel() for p in renderer.fine_decoder.parameters())
        n_coarse = sum(p.numel() for p in renderer.coarse_carrier_fusion.parameters()) if renderer.coarse_carrier_fusion is not None else 0
        n_refiner = sum(p.numel() for p in refiner.parameters())
        _print(
            printer,
            f"  Decoder fine-tuning enabled: {n_decoder + n_coarse + n_refiner:,} params unfrozen",
        )
    if feat_select is not None and finetune_fsm:
        for param in feat_select.parameters():
            param.requires_grad_(True)
        feat_select.train()
        n_fsm = sum(p.numel() for p in feat_select.parameters())
        _print(printer, f"  FSM fine-tuning enabled: {n_fsm:,} params unfrozen")

    setattr(renderer, "_feat_select", feat_select)
    setattr(renderer, "_fsm_use_coarse", fsm_use_coarse)

    render_width = int(cfg_dcff.get("render_width", 120))
    render_height = int(cfg_dcff.get("render_height", 68))

    _print(printer, f"  Render resolution: {render_width}×{render_height}")

    return DCFFRuntime(
        gaussians=gaussians,
        hash_grid=hash_grid,
        renderer=renderer,
        refiner=refiner,
        feat_select=feat_select,
        render_height=render_height,
        render_width=render_width,
        finetune_decoder=finetune_decoder,
        finetune_fsm=finetune_fsm,
    )


def build_dcff(
    config: dict,
    device: torch.device | str,
    *,
    printer: Printer | None = print,
) -> tuple[HybridGaussianModel, DeferredCascadedRenderer, nn.Module]:
    runtime = build_dcff_runtime(config, device, printer=printer)
    setattr(runtime.renderer, "_feat_select", runtime.feat_select)
    return runtime.gaussians, runtime.renderer, runtime.refiner


@torch.no_grad()
def render_at_pose(
    gaussians: HybridGaussianModel,
    dcff_renderer: DeferredCascadedRenderer,
    feat_sharp: nn.Module,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    render_h: int,
    render_w: int,
    feature_hw: tuple[int, int] | None = None,
    fsm_temperature: float = 0.5,
    fsm_hard: bool = False,
    render_coarse: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    bundle = render_feature_bundle_at_pose(
        gaussians,
        dcff_renderer,
        feat_sharp,
        pose_w2c,
        K,
        render_h,
        render_w,
        feature_hw=feature_hw,
        fsm_temperature=fsm_temperature,
        fsm_hard=fsm_hard,
        render_coarse=render_coarse,
    )
    fine_feat = bundle["fine_features"]
    depth = bundle["depth"]
    return fine_feat, depth


@torch.no_grad()
def render_feature_bundle_at_pose(
    gaussians: HybridGaussianModel,
    dcff_renderer: DeferredCascadedRenderer,
    feat_sharp: nn.Module,
    pose_w2c: torch.Tensor,
    K: torch.Tensor,
    render_h: int,
    render_w: int,
    feature_hw: tuple[int, int] | None = None,
    fsm_temperature: float = 0.5,
    fsm_hard: bool = False,
    render_coarse: bool | None = None,
) -> dict[str, torch.Tensor | None]:
    viewmat = pose_w2c.float()
    feat_h, feat_w = _resolve_feature_hw(feature_hw, render_h, render_w)
    use_coarse_for_fsm = bool(getattr(dcff_renderer, "_fsm_use_coarse", False))
    should_render_coarse = use_coarse_for_fsm if render_coarse is None else bool(render_coarse)
    result = dcff_renderer(
        gaussians,
        viewmat=viewmat,
        K=K,
        width=render_w,
        height=render_h,
        render_coarse=should_render_coarse,
        feature_height=feat_h,
        feature_width=feat_w,
    )
    result = _apply_dcff_postprocess(
        result,
        feat_h,
        feat_w,
        feat_sharp=feat_sharp,
        feat_select=getattr(dcff_renderer, "_feat_select", None),
        use_coarse_for_fsm=use_coarse_for_fsm,
        temperature=fsm_temperature,
        hard=fsm_hard,
    )
    return {
        "fine_features": result["fine_features"].float(),
        "coarse_features": result.get("coarse_features").float() if result.get("coarse_features") is not None else None,
        "depth": result["depth"],
        "fsm_spatial_conf": result.get("fsm_spatial_conf").float() if result.get("fsm_spatial_conf") is not None else None,
        "fsm_channel_weights": result.get("fsm_channel_weights").float() if result.get("fsm_channel_weights") is not None else None,
    }


@torch.no_grad()
def render_batch(
    gaussians: HybridGaussianModel,
    dcff_renderer: DeferredCascadedRenderer,
    feat_sharp: nn.Module,
    poses_w2c: torch.Tensor,
    K: torch.Tensor,
    render_h: int,
    render_w: int,
    feature_hw: tuple[int, int] | None = None,
    fsm_temperature: float = 0.5,
    fsm_hard: bool = False,
    render_coarse: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    fine_list = []
    depth_list = []
    for pose in poses_w2c:
        bundle_i = render_feature_bundle_at_pose(
            gaussians,
            dcff_renderer,
            feat_sharp,
            pose,
            K,
            render_h,
            render_w,
            feature_hw=feature_hw,
            fsm_temperature=fsm_temperature,
            fsm_hard=fsm_hard,
            render_coarse=render_coarse,
        )
        fine_list.append(bundle_i["fine_features"].squeeze(0))
        depth_list.append(bundle_i["depth"].squeeze(0).squeeze(0))
    return torch.stack(fine_list, dim=0), torch.stack(depth_list, dim=0)


@torch.no_grad()
def render_feature_bundle_batch(
    gaussians: HybridGaussianModel,
    dcff_renderer: DeferredCascadedRenderer,
    feat_sharp: nn.Module,
    poses_w2c: torch.Tensor,
    K: torch.Tensor,
    render_h: int,
    render_w: int,
    feature_hw: tuple[int, int] | None = None,
    fsm_temperature: float = 0.5,
    fsm_hard: bool = False,
    render_coarse: bool | None = None,
) -> dict[str, torch.Tensor | None]:
    fine_list = []
    depth_list = []
    coarse_list = []
    fsm_spatial_list = []
    fsm_channel_list = []
    all_have_coarse = True
    all_have_fsm_spatial = True
    all_have_fsm_channel = True

    for pose in poses_w2c:
        bundle_i = render_feature_bundle_at_pose(
            gaussians,
            dcff_renderer,
            feat_sharp,
            pose,
            K,
            render_h,
            render_w,
            feature_hw=feature_hw,
            fsm_temperature=fsm_temperature,
            fsm_hard=fsm_hard,
            render_coarse=render_coarse,
        )
        fine_list.append(bundle_i["fine_features"].squeeze(0))
        depth_list.append(bundle_i["depth"].squeeze(0).squeeze(0))
        if bundle_i["coarse_features"] is None:
            all_have_coarse = False
        else:
            coarse_list.append(bundle_i["coarse_features"].squeeze(0))
        if bundle_i["fsm_spatial_conf"] is None:
            all_have_fsm_spatial = False
        else:
            fsm_spatial_list.append(bundle_i["fsm_spatial_conf"].squeeze(0))
        if bundle_i["fsm_channel_weights"] is None:
            all_have_fsm_channel = False
        else:
            fsm_channel_list.append(bundle_i["fsm_channel_weights"].squeeze(0))

    return {
        "fine_features": torch.stack(fine_list, dim=0),
        "depth": torch.stack(depth_list, dim=0),
        "coarse_features": torch.stack(coarse_list, dim=0) if all_have_coarse else None,
        "fsm_spatial_conf": torch.stack(fsm_spatial_list, dim=0) if all_have_fsm_spatial else None,
        "fsm_channel_weights": torch.stack(fsm_channel_list, dim=0) if all_have_fsm_channel else None,
    }
