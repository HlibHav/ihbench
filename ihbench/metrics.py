"""Metrics aggregation for IHBench evaluation results.

Supports multi-epoch results for confidence interval computation. When results
contain an ``epoch`` field, per-epoch aggregates are computed and used to derive
mean +/- 95% CI across epochs.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Any


def _ci95(values: list[float]) -> tuple[float, float, float]:
    """Compute mean, lower, upper 95% CI from a list of per-epoch means.

    Uses t-distribution for small sample sizes.
    """
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    stderr = math.sqrt(variance / n)
    # t critical value for 95% CI (two-tailed) — approximate for small n
    t_crit = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447,
              8: 2.365, 9: 2.306, 10: 2.262}.get(n, 1.96)
    margin = t_crit * stderr
    return mean, mean - margin, mean + margin


def aggregate(results: list[dict]) -> dict:
    """Compute aggregated metrics over a list of result dicts."""
    if not results:
        return {"n": 0, "tf_win_rate": None, "rq_pass_rate": None, "rq_avg_score": None}
    n = len(results)
    tf_vals = [r["tf_win_score"] for r in results if r.get("tf_win_score") is not None]
    rq_passes = [int(r["rq_pass"]) for r in results if r.get("rq_pass") is not None]
    rq_scores = [r["rq_score"] for r in results if r.get("rq_score") is not None]
    return {
        "n": n,
        "tf_win_rate": round(sum(tf_vals) / len(tf_vals), 4) if tf_vals else None,
        "rq_pass_rate": round(sum(rq_passes) / len(rq_passes), 4) if rq_passes else None,
        "rq_avg_score": round(sum(rq_scores) / len(rq_scores), 4) if rq_scores else None,
    }


def aggregate_with_ci(results: list[dict]) -> dict:
    """Compute metrics with 95% confidence intervals across epochs.

    Results must contain an ``epoch`` field. Returns per-metric mean, CI lower,
    and CI upper bounds.
    """
    by_epoch: dict[int, list[dict]] = defaultdict(list)
    for r in results:
        by_epoch[r.get("epoch", 0)].append(r)

    n_epochs = len(by_epoch)
    if n_epochs <= 1:
        agg = aggregate(results)
        # No CI with single epoch
        for key in ("tf_win_rate", "rq_pass_rate", "rq_avg_score"):
            if agg[key] is not None:
                agg[f"{key}_ci_lower"] = agg[key]
                agg[f"{key}_ci_upper"] = agg[key]
        agg["n_epochs"] = n_epochs
        return agg

    # Compute per-epoch aggregates
    epoch_aggs = {e: aggregate(rs) for e, rs in sorted(by_epoch.items())}

    out: dict[str, Any] = {"n": sum(len(rs) for rs in by_epoch.values()), "n_epochs": n_epochs}
    for key in ("tf_win_rate", "rq_pass_rate", "rq_avg_score"):
        epoch_vals = [epoch_aggs[e][key] for e in sorted(by_epoch) if epoch_aggs[e][key] is not None]
        if epoch_vals:
            mean, lo, hi = _ci95(epoch_vals)
            out[key] = round(mean, 4)
            out[f"{key}_ci_lower"] = round(lo, 4)
            out[f"{key}_ci_upper"] = round(hi, 4)
        else:
            out[key] = None
            out[f"{key}_ci_lower"] = None
            out[f"{key}_ci_upper"] = None
    return out


def aggregate_per_type(results: list[dict]) -> dict[str, dict]:
    """Compute metrics broken down by interruption type."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["interruption_type"]].append(r)
    return {t: aggregate(rs) for t, rs in sorted(by_type.items())}


def aggregate_per_depth(
    results: list[dict],
    bins: list[tuple[int, int]] | None = None,
) -> dict[str, dict]:
    """Compute metrics broken down by conversation depth bins."""
    if bins is None:
        bins = [(0, 4), (5, 9), (10, 14), (15, 19)]

    by_bin: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        depth = r.get("turn_depth", 0)
        for lo, hi in bins:
            if lo <= depth <= hi:
                by_bin[f"{lo}-{hi}"].append(r)
                break
    return {b: aggregate(rs) for b, rs in sorted(by_bin.items())}


def compute_all_metrics(results: list[dict], with_ci: bool = False) -> dict[str, Any]:
    """Compute overall, per-type, and per-depth metrics.

    Args:
        results: List of result dicts. If ``with_ci`` is True, each must have an ``epoch`` field.
        with_ci: If True, compute 95% confidence intervals across epochs.
    """
    agg_fn = aggregate_with_ci if with_ci else aggregate

    # Per-type with CI
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_type[r["interruption_type"]].append(r)

    # Per-depth with CI
    bins = [(0, 4), (5, 9), (10, 14), (15, 19)]
    by_bin: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        depth = r.get("turn_depth", 0)
        for lo, hi in bins:
            if lo <= depth <= hi:
                by_bin[f"{lo}-{hi}"].append(r)
                break

    return {
        "overall": agg_fn(results),
        "per_type": {t: agg_fn(rs) for t, rs in sorted(by_type.items())},
        "per_depth": {b: agg_fn(rs) for b, rs in sorted(by_bin.items())},
    }


def _fmt_val(m: dict, key: str) -> str:
    """Format a metric value, with CI if available."""
    val = m.get(key)
    if val is None:
        return "N/A"
    lo = m.get(f"{key}_ci_lower")
    hi = m.get(f"{key}_ci_upper")
    if lo is not None and hi is not None and lo != hi:
        return f"{val:.3f} [{lo:.3f}, {hi:.3f}]"
    return f"{val:.3f}"


def format_report(results: list[dict], with_ci: bool = False) -> str:
    """Format a human-readable report of evaluation results."""
    metrics = compute_all_metrics(results, with_ci=with_ci)
    lines = []

    overall = metrics["overall"]
    lines.append("=== Overall ===")
    lines.append(f"  n={overall['n']}")
    if overall.get("n_epochs", 1) > 1:
        lines.append(f"  epochs={overall['n_epochs']}")
    for key in ("tf_win_rate", "rq_pass_rate", "rq_avg_score"):
        lines.append(f"  {key}: {_fmt_val(overall, key)}")

    lines.append("\n=== Per Type ===")
    for t, m in metrics["per_type"].items():
        tf = _fmt_val(m, "tf_win_rate")
        rq = _fmt_val(m, "rq_pass_rate")
        ra = _fmt_val(m, "rq_avg_score")
        lines.append(f"  {t:<15} n={m['n']:>4}  TF={tf}  RQ={rq}  RQavg={ra}")

    lines.append("\n=== Per Depth ===")
    for d, m in metrics["per_depth"].items():
        tf = _fmt_val(m, "tf_win_rate")
        rq = _fmt_val(m, "rq_pass_rate")
        ra = _fmt_val(m, "rq_avg_score")
        lines.append(f"  depth {d:<8} n={m['n']:>4}  TF={tf}  RQ={rq}  RQavg={ra}")

    return "\n".join(lines)
