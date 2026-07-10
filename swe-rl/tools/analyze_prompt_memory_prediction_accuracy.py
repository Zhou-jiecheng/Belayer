#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


_UNITS = {
    "B": 1,
    "KiB": 1024,
    "MiB": 1024**2,
    "GiB": 1024**3,
    "TiB": 1024**4,
}

_ADMITTED_RE = re.compile(
    r"admitted prompt=(?P<prompt>\S+) .*? "
    r"predicted\(mem=(?P<mem>[^,]+),cpu=(?P<cpu>[0-9.]+)%,r=(?P<read>[^,]+),w=(?P<write>[^\)]+)\)"
)
_SUMMARY_RE = re.compile(
    r"prompt summary prompt=(?P<prompt>\S+) repo=(?P<repo>\S+) "
    r"peak_mem=(?P<mem>\S+) avg_cpu=(?P<cpu>[0-9.]+)% "
    r"disk\(r=(?P<read>[^,]+),w=(?P<write>[^\)]+)\).*?"
    r"samples=(?P<samples>\d+) cpu_samples=(?P<cpu_samples>\d+)(?: leases=(?P<leases>\d+))?"
)


def _parse_size(text: str) -> float:
    raw = str(text or "").strip()
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*([A-Za-z]+)?$", raw)
    if not match:
        return float(raw or 0.0)
    value = float(match.group(1))
    unit = match.group(2) or "B"
    return value * _UNITS.get(unit, 1)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * float(q)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _is_cold_start(
    row: dict[str, Any],
    *,
    memory_mib: float,
    cpu_percent: float,
    tolerance: float,
) -> bool:
    pred_mib = float(row["pred_memory_bytes"]) / 1024**2
    pred_cpu = float(row["pred_cpu_percent"])
    return abs(pred_mib - memory_mib) <= tolerance and abs(pred_cpu - cpu_percent) <= tolerance


def _parse_log(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    predictions: dict[str, dict[str, Any]] = {}
    actuals: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        match = _ADMITTED_RE.search(line)
        if match:
            prompt = match.group("prompt")
            predictions[prompt] = {
                "pred_memory_bytes": _parse_size(match.group("mem")),
                "pred_cpu_percent": float(match.group("cpu")),
                "pred_disk_read_bytes": _parse_size(match.group("read")),
                "pred_disk_write_bytes": _parse_size(match.group("write")),
            }
        match = _SUMMARY_RE.search(line)
        if match:
            prompt = match.group("prompt")
            actuals[prompt] = {
                "repo": match.group("repo"),
                "actual_memory_bytes": _parse_size(match.group("mem")),
                "actual_cpu_percent": float(match.group("cpu")),
                "actual_disk_read_bytes": _parse_size(match.group("read")),
                "actual_disk_write_bytes": _parse_size(match.group("write")),
                "samples": int(match.group("samples")),
                "cpu_samples": int(match.group("cpu_samples")),
                "leases": int(match.group("leases") or 0),
            }
    return predictions, actuals


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    log_path = Path(args.experiment_log)
    predictions, actuals = _parse_log(log_path)
    matched_prompts = sorted(set(predictions) & set(actuals))

    rows: list[dict[str, Any]] = []
    excluded_cold_start = 0
    for prompt in matched_prompts:
        row = {
            "prompt": prompt,
            **predictions[prompt],
            **actuals[prompt],
        }
        if args.exclude_cold_start and _is_cold_start(
            row,
            memory_mib=float(args.cold_start_memory_mib),
            cpu_percent=float(args.cold_start_cpu_percent),
            tolerance=float(args.cold_start_tolerance),
        ):
            excluded_cold_start += 1
            continue
        pred_mem = float(row["pred_memory_bytes"])
        actual_mem = float(row["actual_memory_bytes"])
        ratio = actual_mem / pred_mem if pred_mem > 0.0 else 0.0
        row["memory_actual_over_predicted"] = ratio
        row["memory_abs_relative_error"] = abs(actual_mem - pred_mem) / pred_mem if pred_mem > 0.0 else 0.0
        row["memory_underpredicted"] = actual_mem > pred_mem
        for factor in args.accuracy_factors:
            row[f"memory_within_{factor:g}x"] = abs(ratio - 1.0) <= (factor - 1.0)
            row[f"memory_covered_by_{factor:g}x"] = ratio <= factor
        rows.append(row)

    ratios = [float(row["memory_actual_over_predicted"]) for row in rows]
    abs_rel = [float(row["memory_abs_relative_error"]) for row in rows]

    summary = {
        "experiment_log": str(log_path.resolve()),
        "prediction_count": len(predictions),
        "actual_count": len(actuals),
        "matched_count": len(matched_prompts),
        "excluded_cold_start_count": excluded_cold_start,
        "evaluated_count": len(rows),
        "exclude_cold_start": bool(args.exclude_cold_start),
        "cold_start_memory_mib": float(args.cold_start_memory_mib),
        "cold_start_cpu_percent": float(args.cold_start_cpu_percent),
        "memory_underprediction_rate": _mean([1.0 if row["memory_underpredicted"] else 0.0 for row in rows]),
        "memory_actual_over_predicted": {
            "mean": _mean(ratios),
            "p50": _quantile(ratios, 0.50),
            "p90": _quantile(ratios, 0.90),
            "p95": _quantile(ratios, 0.95),
            "p99": _quantile(ratios, 0.99),
            "min": min(ratios) if ratios else 0.0,
            "max": max(ratios) if ratios else 0.0,
        },
        "memory_abs_relative_error": {
            "mean": _mean(abs_rel),
            "p50": _quantile(abs_rel, 0.50),
            "p90": _quantile(abs_rel, 0.90),
            "p95": _quantile(abs_rel, 0.95),
        },
        "memory_accuracy": {
            f"within_{factor:g}x": _mean(
                [1.0 if bool(row[f"memory_within_{factor:g}x"]) else 0.0 for row in rows]
            )
            for factor in args.accuracy_factors
        },
        "memory_coverage": {
            f"covered_by_{factor:g}x": _mean(
                [1.0 if bool(row[f"memory_covered_by_{factor:g}x"]) else 0.0 for row in rows]
            )
            for factor in args.accuracy_factors
        },
    }
    return {"summary": summary, "rows": rows}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze prompt-level memory prediction accuracy from replay logs.")
    parser.add_argument("experiment_log")
    parser.add_argument("--output-json", default="")
    parser.add_argument("--output-csv", default="")
    parser.add_argument("--exclude-cold-start", action="store_true", default=True)
    parser.add_argument("--include-cold-start", action="store_false", dest="exclude_cold_start")
    parser.add_argument("--cold-start-memory-mib", type=float, default=184.3)
    parser.add_argument("--cold-start-cpu-percent", type=float, default=5.4)
    parser.add_argument("--cold-start-tolerance", type=float, default=0.05)
    parser.add_argument(
        "--accuracy-factors",
        type=float,
        nargs="+",
        default=[1.1, 1.2, 1.5, 1.6, 2.0],
        help="Factors for symmetric accuracy and one-sided scheduler coverage.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = analyze(args)
    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    if args.output_csv:
        _write_csv(Path(args.output_csv), result["rows"])
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
