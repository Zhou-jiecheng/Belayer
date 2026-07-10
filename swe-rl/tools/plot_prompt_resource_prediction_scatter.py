#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-cache")

import matplotlib.pyplot as plt

from analyze_prompt_memory_prediction_accuracy import analyze


def _repo_from_prompt(prompt: str) -> str:
    if prompt.startswith("getmoto__"):
        return "moto"
    if prompt.startswith("python__"):
        return "mypy"
    parts = prompt.split("__")
    return parts[0] if parts else "unknown"


def _finite_positive_pairs(rows: list[dict[str, Any]], pred_key: str, actual_key: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        pred = float(row.get(pred_key, 0.0) or 0.0)
        actual = float(row.get(actual_key, 0.0) or 0.0)
        if pred <= 0.0 or actual <= 0.0:
            continue
        if not (math.isfinite(pred) and math.isfinite(actual)):
            continue
        out.append(row)
    return out


def _scatter_one(
    ax: Any,
    rows: list[dict[str, Any]],
    *,
    pred_key: str,
    actual_key: str,
    scale: float,
    xlabel: str,
    ylabel: str,
    title: str,
    draw_factor_lines: bool,
) -> dict[str, float]:
    by_repo: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_repo.setdefault(str(row.get("repo") or _repo_from_prompt(str(row.get("prompt", "")))), []).append(row)

    colors = {
        "moto": "#1f77b4",
        "mypy": "#d62728",
        "unknown": "#7f7f7f",
    }
    all_x: list[float] = []
    all_y: list[float] = []
    for repo, repo_rows in sorted(by_repo.items()):
        xs = [float(row[pred_key]) / scale for row in repo_rows]
        ys = [float(row[actual_key]) / scale for row in repo_rows]
        all_x.extend(xs)
        all_y.extend(ys)
        ax.scatter(
            xs,
            ys,
            s=34,
            alpha=0.78,
            edgecolors="white",
            linewidths=0.45,
            label=f"{repo} (n={len(xs)})",
            color=colors.get(repo, colors["unknown"]),
        )

    if not all_x or not all_y:
        ax.set_title(title)
        return {"count": 0.0}

    max_value = max(max(all_x), max(all_y))
    min_value = min(min(all_x), min(all_y))
    lo = 0.0 if min_value >= 0.0 else min_value
    hi = max_value * 1.08 if max_value > 0.0 else 1.0
    ax.plot([lo, hi], [lo, hi], color="#222222", linewidth=1.2, linestyle="-", label="actual = predicted")
    if draw_factor_lines:
        for factor, color in ((1.5, "#999999"), (2.0, "#bbbbbb")):
            ax.plot([lo, hi], [lo * factor, hi * factor], color=color, linewidth=0.9, linestyle="--")
            ax.text(hi * 0.98, hi * factor * 0.98, f"{factor:g}x", color=color, ha="right", va="top", fontsize=8)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi if not draw_factor_lines else max(hi, max(all_y) * 1.08))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linewidth=0.45, alpha=0.28)
    ax.legend(frameon=False, fontsize=8, loc="best")

    ratios = [y / x for x, y in zip(all_x, all_y) if x > 0.0]
    within_20 = sum(abs(ratio - 1.0) <= 0.2 for ratio in ratios) / len(ratios)
    under = sum(ratio > 1.0 for ratio in ratios) / len(ratios)
    covered_16 = sum(ratio <= 1.6 for ratio in ratios) / len(ratios)
    return {
        "count": float(len(ratios)),
        "within_20": within_20,
        "underprediction_rate": under,
        "coverage_1_6": covered_16,
        "ratio_p50": sorted(ratios)[len(ratios) // 2],
        "ratio_max": max(ratios),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot prompt-level predicted vs actual memory and CPU scatter.")
    parser.add_argument("experiment_log")
    parser.add_argument("--output-prefix", default="")
    parser.add_argument("--exclude-cold-start", action="store_true", default=True)
    parser.add_argument("--include-cold-start", action="store_false", dest="exclude_cold_start")
    parser.add_argument("--min-valid-leases", type=int, default=1)
    parser.add_argument("--cold-start-memory-mib", type=float, default=184.3)
    parser.add_argument("--cold-start-cpu-percent", type=float, default=5.4)
    parser.add_argument("--cold-start-tolerance", type=float, default=0.05)
    parser.add_argument("--dpi", type=int, default=180)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    args.accuracy_factors = [1.1, 1.2, 1.5, 1.6, 2.0]
    result = analyze(args)
    rows = [
        row
        for row in result["rows"]
        if int(row.get("leases") or 0) >= max(0, int(args.min_valid_leases))
    ]
    memory_rows = _finite_positive_pairs(rows, "pred_memory_bytes", "actual_memory_bytes")
    cpu_rows = _finite_positive_pairs(rows, "pred_cpu_percent", "actual_cpu_percent")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), constrained_layout=True)
    mem_stats = _scatter_one(
        axes[0],
        memory_rows,
        pred_key="pred_memory_bytes",
        actual_key="actual_memory_bytes",
        scale=1024**2,
        xlabel="Predicted peak memory (MiB)",
        ylabel="Actual peak memory (MiB)",
        title="Memory prediction",
        draw_factor_lines=True,
    )
    cpu_stats = _scatter_one(
        axes[1],
        cpu_rows,
        pred_key="pred_cpu_percent",
        actual_key="actual_cpu_percent",
        scale=1.0,
        xlabel="Predicted average CPU (%)",
        ylabel="Actual average CPU (%)",
        title="CPU prediction",
        draw_factor_lines=False,
    )

    output_prefix = Path(args.output_prefix) if args.output_prefix else Path(args.experiment_log).with_suffix("")
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".prediction_scatter.png")
    pdf_path = output_prefix.with_suffix(".prediction_scatter.pdf")
    summary_path = output_prefix.with_suffix(".prediction_scatter.json")
    fig.savefig(png_path, dpi=max(72, int(args.dpi)))
    fig.savefig(pdf_path)
    plt.close(fig)

    summary = {
        "experiment_log": str(Path(args.experiment_log).resolve()),
        "exclude_cold_start": bool(args.exclude_cold_start),
        "min_valid_leases": int(args.min_valid_leases),
        "memory": mem_stats,
        "cpu": cpu_stats,
        "png": str(png_path.resolve()),
        "pdf": str(pdf_path.resolve()),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
