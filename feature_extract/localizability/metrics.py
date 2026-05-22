"""Ranking metrics for localization feature selection."""

from __future__ import annotations

import torch


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    ranks = torch.empty_like(values, dtype=torch.float32)
    ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
    return ranks


def _spearman_one(scores: torch.Tensor, utility: torch.Tensor) -> torch.Tensor:
    if scores.numel() < 2:
        return scores.new_zeros(())
    sr = _rankdata(scores.float())
    ur = _rankdata(utility.float())
    sr = sr - sr.mean()
    ur = ur - ur.mean()
    denom = sr.norm() * ur.norm()
    if float(denom.detach().cpu()) <= 1.0e-12:
        return scores.new_zeros(())
    return (sr * ur).sum() / denom


def _ndcg_one(scores: torch.Tensor, relevance: torch.Tensor, k: int) -> torch.Tensor:
    k = min(int(k), scores.numel())
    if k <= 0:
        return scores.new_zeros(())
    order = torch.argsort(scores, descending=True)[:k]
    ideal = torch.argsort(relevance, descending=True)[:k]
    discount = 1.0 / torch.log2(torch.arange(k, device=scores.device, dtype=torch.float32) + 2.0)
    dcg = (relevance[order].float() * discount).sum()
    idcg = (relevance[ideal].float() * discount).sum().clamp_min(1.0e-8)
    return dcg / idcg


def ranking_metrics(
    scores: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    basin_label: torch.Tensor | None = None,
    topk: tuple[int, ...] = (1, 5),
) -> dict[str, torch.Tensor]:
    if scores.shape != pose_cost_m.shape:
        raise ValueError("scores and pose_cost_m must have the same shape")
    valid = valid_mask.bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
    masked_scores = scores.float().masked_fill(~valid, torch.finfo(scores.dtype).min / 4.0)
    masked_cost = pose_cost_m.float().masked_fill(~valid, float("inf"))
    pred_idx = masked_scores.argmax(dim=1)
    oracle_idx = masked_cost.argmin(dim=1)
    row = torch.arange(scores.shape[0], device=scores.device)
    pred_cost = pose_cost_m.float()[row, pred_idx]
    oracle_cost = pose_cost_m.float()[row, oracle_idx]
    active = valid.any(dim=1)
    if not active.any():
        zero = scores.new_zeros(())
        return {"pred_cost_m": zero, "oracle_cost_m": zero, "oracle_gap_m": zero, "top1_acc": zero}

    metrics = {
        "pred_cost_m": pred_cost[active].mean(),
        "oracle_cost_m": oracle_cost[active].mean(),
        "oracle_gap_m": (pred_cost - oracle_cost)[active].mean(),
        "top1_acc": (pred_idx == oracle_idx).float()[active].mean(),
    }
    spearman_values = []
    ndcg_values = []
    for row_scores, row_cost, row_valid in zip(scores, pose_cost_m, valid):
        if not row_valid.any():
            continue
        rs = row_scores[row_valid]
        rel = -row_cost[row_valid]
        spearman_values.append(_spearman_one(rs, rel))
        ndcg_values.append(_ndcg_one(rs, rel - rel.min(), min(max(topk), rs.numel())))
    metrics["spearman"] = torch.stack(spearman_values).mean() if spearman_values else scores.new_zeros(())
    metrics["ndcg"] = torch.stack(ndcg_values).mean() if ndcg_values else scores.new_zeros(())

    if basin_label is not None:
        basin = basin_label.bool() & valid
        sorted_idx = torch.argsort(masked_scores, dim=1, descending=True)
        has_positive = basin.any(dim=1)
        for k in topk:
            kk = min(int(k), scores.shape[1])
            top_idx = sorted_idx[:, :kk]
            top_basin = basin.gather(1, top_idx).any(dim=1)
            denom = has_positive.float().sum().clamp_min(1.0)
            metrics[f"basin_recall@{k}"] = (top_basin & has_positive).float().sum() / denom
    return metrics


def risk_coverage_metrics(
    confidence: torch.Tensor,
    success: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    coverages: tuple[float, ...] = (0.25, 0.50, 0.75, 1.0),
) -> dict[str, torch.Tensor]:
    """Report failure risk as increasingly uncertain predictions are accepted.

    Higher ``confidence`` means a prediction is trusted earlier.  ``success`` is
    a binary final-outcome label, for example selected candidate in basin or
    downstream solver final pose within threshold.  Lower risk is better.
    """
    conf = confidence.float().reshape(-1)
    ok = success.to(device=conf.device).bool().reshape(-1)
    if conf.shape != ok.shape:
        raise ValueError("confidence and success must have the same number of elements")
    valid = (
        valid_mask.to(device=conf.device).bool().reshape(-1)
        if valid_mask is not None
        else torch.ones_like(ok, dtype=torch.bool)
    )
    if valid.shape != conf.shape:
        raise ValueError("valid_mask must match confidence shape")
    conf = conf[valid]
    ok = ok[valid]
    if conf.numel() == 0:
        zero = confidence.new_zeros(())
        out = {
            "num_predictions": zero,
            "success_rate": zero,
            "risk_coverage_auc": zero,
        }
        for coverage in coverages:
            pct = int(round(float(coverage) * 100.0))
            out[f"risk@{pct}"] = zero
            out[f"high_conf_false_accept@{pct}"] = zero
        return out

    order = torch.argsort(conf, descending=True)
    ok_sorted = ok[order].float()
    prefix_count = torch.arange(1, ok_sorted.numel() + 1, device=conf.device, dtype=torch.float32)
    prefix_success = torch.cumsum(ok_sorted, dim=0) / prefix_count
    prefix_risk = 1.0 - prefix_success
    out = {
        "num_predictions": conf.new_tensor(float(conf.numel())),
        "success_rate": ok.float().mean(),
        "risk_coverage_auc": prefix_risk.mean(),
    }
    for coverage in coverages:
        frac = max(0.0, min(1.0, float(coverage)))
        count = max(1, min(int(conf.numel()), int(torch.ceil(conf.new_tensor(frac * conf.numel())).item())))
        risk = prefix_risk[count - 1]
        pct = int(round(frac * 100.0))
        out[f"risk@{pct}"] = risk
        out[f"high_conf_false_accept@{pct}"] = risk
    return out


def ranking_row_diagnostics(
    scores: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    valid_mask: torch.Tensor | None = None,
    basin_label: torch.Tensor | None = None,
    sample_names: list[str] | tuple[str, ...] | None = None,
) -> list[dict]:
    """Return per-row selected/oracle diagnostics for failure analysis."""
    if scores.shape != pose_cost_m.shape:
        raise ValueError("scores and pose_cost_m must have the same shape")
    valid = valid_mask.bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
    masked_scores = scores.float().masked_fill(~valid, torch.finfo(scores.dtype).min / 4.0)
    masked_cost = pose_cost_m.float().masked_fill(~valid, float("inf"))
    pred_idx = masked_scores.argmax(dim=1)
    oracle_idx = masked_cost.argmin(dim=1)
    rows: list[dict] = []
    for batch_idx in range(scores.shape[0]):
        if not bool(valid[batch_idx].any().detach().cpu()):
            continue
        selected = int(pred_idx[batch_idx].detach().cpu())
        oracle = int(oracle_idx[batch_idx].detach().cpu())
        selected_cost = float(pose_cost_m[batch_idx, selected].detach().cpu())
        oracle_cost = float(pose_cost_m[batch_idx, oracle].detach().cpu())
        row = {
            "row": int(batch_idx),
            "sample_name": str(sample_names[batch_idx]) if sample_names is not None else str(batch_idx),
            "selected_idx": selected,
            "oracle_idx": oracle,
            "selected_cost_m": selected_cost,
            "oracle_cost_m": oracle_cost,
            "oracle_gap_m": selected_cost - oracle_cost,
            "selected_score": float(scores[batch_idx, selected].detach().cpu()),
            "oracle_score": float(scores[batch_idx, oracle].detach().cpu()),
            "top1": bool(selected == oracle),
        }
        if basin_label is not None:
            basin = basin_label.bool()
            row["selected_in_basin"] = bool(basin[batch_idx, selected].detach().cpu())
            row["oracle_in_basin"] = bool(basin[batch_idx, oracle].detach().cpu())
        rows.append(row)
    return rows


def candidate_score_table_rows(
    scores: torch.Tensor,
    pose_cost_m: torch.Tensor,
    *,
    trans_err_m: torch.Tensor | None = None,
    rot_err_deg: torch.Tensor | None = None,
    valid_mask: torch.Tensor | None = None,
    basin_label: torch.Tensor | None = None,
    sample_names: list[str] | tuple[str, ...] | None = None,
    extra_fields: dict[str, torch.Tensor] | None = None,
) -> list[dict]:
    """Return one JSON-serializable row per candidate for light calibrators."""
    if scores.shape != pose_cost_m.shape:
        raise ValueError("scores and pose_cost_m must have the same shape")
    if trans_err_m is not None and trans_err_m.shape != scores.shape:
        raise ValueError("trans_err_m must match scores shape")
    if rot_err_deg is not None and rot_err_deg.shape != scores.shape:
        raise ValueError("rot_err_deg must match scores shape")
    extras = extra_fields or {}
    for name, values in extras.items():
        if values.shape != scores.shape:
            raise ValueError(f"extra field {name!r} must match scores shape")
    valid = valid_mask.bool() if valid_mask is not None else torch.ones_like(scores, dtype=torch.bool)
    basin = basin_label.bool() if basin_label is not None else torch.zeros_like(scores, dtype=torch.bool)
    masked_cost = pose_cost_m.float().masked_fill(~valid, float("inf"))
    oracle_idx = masked_cost.argmin(dim=1)
    rows: list[dict] = []
    for batch_idx in range(scores.shape[0]):
        score_order = torch.argsort(scores[batch_idx].float(), descending=True)
        score_rank = torch.empty_like(score_order)
        score_rank[score_order] = torch.arange(score_order.numel(), device=score_order.device)
        for candidate_idx in range(scores.shape[1]):
            row = {
                "row": int(batch_idx),
                "sample_name": str(sample_names[batch_idx]) if sample_names is not None else str(batch_idx),
                "candidate_idx": int(candidate_idx),
                "score": float(scores[batch_idx, candidate_idx].detach().cpu()),
                "pose_cost_m": float(pose_cost_m[batch_idx, candidate_idx].detach().cpu()),
                "score_rank": int(score_rank[candidate_idx].detach().cpu()),
                "is_oracle": bool(candidate_idx == int(oracle_idx[batch_idx].detach().cpu())),
                "valid": bool(valid[batch_idx, candidate_idx].detach().cpu()),
                "in_basin": bool(basin[batch_idx, candidate_idx].detach().cpu()),
            }
            if trans_err_m is not None:
                row["trans_err_m"] = float(trans_err_m[batch_idx, candidate_idx].detach().cpu())
            if rot_err_deg is not None:
                row["rot_err_deg"] = float(rot_err_deg[batch_idx, candidate_idx].detach().cpu())
            for name, values in extras.items():
                row[name] = float(values[batch_idx, candidate_idx].detach().cpu())
            rows.append(row)
    return rows
