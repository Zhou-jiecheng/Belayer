#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


DEFAULT_EXPERIMENT_ROOT = Path(
    "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/"
    "export/checkpoint_policy_fault_experiment_20260521_094103"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _policy_sort_key(policy: str) -> tuple[int, str]:
    preferred = {
        "oracle-no-fault-no-checkpoint": 0,
        "never": 1,
        "always": 2,
        "every-3": 3,
        "adaptive-risk": 4,
    }
    return preferred.get(policy, 100), policy


def _is_successful_checkpoint_event(event: dict[str, Any]) -> bool:
    if not isinstance(event, dict) or event.get("skipped"):
        return False
    create_result = event.get("create_result") or {}
    return isinstance(create_result, dict) and bool(create_result.get("ok", False))


def _critical_interval(event: dict[str, Any]) -> tuple[float, float] | None:
    start = _as_float(event.get("create_call_start_ts"), -1.0)
    end = _as_float(event.get("create_call_end_ts"), -1.0)
    overlap_budget = max(0.0, _as_float(event.get("overlap_budget_sec"), 0.0))
    critical_start = start + overlap_budget
    if start < 0.0 or end <= critical_start:
        return None
    return critical_start, end


def _union_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals = sorted(intervals)
    total = 0.0
    cur_start, cur_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        total += cur_end - cur_start
        cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def _successful_events_from_reports(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for report in reports:
        for event in report.get("checkpoint_events", []) or []:
            if _is_successful_checkpoint_event(event):
                events.append(event)
    return events


def _summary_overhead(policy_dir: Path) -> dict[str, Any] | None:
    summary_path = policy_dir / "summary.json"
    if not summary_path.exists():
        return None
    payload = _load_json(summary_path)
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        return None
    overhead = summary.get("checkpoint_overhead", {})
    if not isinstance(overhead, dict):
        overhead = {}

    checkpoint_count = _as_int(
        overhead.get("checkpoint_count", summary.get("checkpoint_created", summary.get("checkpoint_attempts", 0)))
    )
    total_elapsed = _as_float(
        overhead.get("total_checkpoint_elapsed_sec", summary.get("checkpoint_total_elapsed_sec", 0.0))
    )
    total_overlapped = _as_float(
        overhead.get("total_overlapped_checkpoint_sec", summary.get("checkpoint_total_overlapped_sec", 0.0))
    )
    total_critical = _as_float(
        overhead.get(
            "total_critical_path_overhead_sec",
            summary.get("checkpoint_total_critical_path_overhead_sec", 0.0),
        )
    )
    mean_elapsed = _as_float(
        overhead.get("mean_checkpoint_elapsed_sec"),
        total_elapsed / checkpoint_count if checkpoint_count else 0.0,
    )
    overlap_fraction = _as_float(
        overhead.get("overlap_fraction"),
        total_overlapped / total_elapsed if total_elapsed > 0 else 0.0,
    )
    trajectory_count = _as_int(summary.get("trajectory_count", len(payload.get("reports", []) or [])))
    events = _successful_events_from_reports(payload.get("reports", []) or [])
    batch_exposed_wall_sec = _union_duration(
        [interval for event in events if (interval := _critical_interval(event)) is not None]
    )
    return {
        "policy": policy_dir.name,
        "trajectory_count": trajectory_count,
        "batch_wall_time_sec": _as_float(summary.get("batch_wall_time_sec", 0.0)),
        "mean_traj_wall_time_sec": _as_float(summary.get("mean_traj_wall_time_sec", 0.0)),
        "total_checkpoint_number": checkpoint_count,
        "average_checkpoint_cost_sec": mean_elapsed,
        "total_checkpoint_cost_sec": total_elapsed,
        "checkpoint_overlap_efficiency": overlap_fraction,
        "total_overlapped_checkpoint_sec": total_overlapped,
        # This is a sum over trajectories/events. With concurrent trajectories,
        # it is measured in trajectory-seconds and must not be compared directly
        # to batch_wall_time_sec.
        "aggregate_exposed_overhead_traj_sec": total_critical,
        "batch_exposed_checkpoint_wall_sec": batch_exposed_wall_sec,
        "average_exposed_overhead_per_checkpoint_sec": total_critical / checkpoint_count if checkpoint_count else 0.0,
        "average_exposed_overhead_per_trajectory_sec": total_critical / trajectory_count if trajectory_count else 0.0,
        "source": "summary.json",
    }


def _event_overhead(policy_dir: Path) -> dict[str, Any]:
    checkpoint_count = 0
    total_elapsed = 0.0
    total_overlapped = 0.0
    total_critical = 0.0
    trajectory_count = 0
    events: list[dict[str, Any]] = []
    for path in sorted((policy_dir / "per_traj").glob("*.json")):
        trajectory_count += 1
        payload = _load_json(path)
        for event in payload.get("checkpoint_events", []) or []:
            if not _is_successful_checkpoint_event(event):
                continue
            events.append(event)
            checkpoint_count += 1
            total_elapsed += _as_float(event.get("checkpoint_elapsed_sec", event.get("create_call_elapsed_sec", 0.0)))
            total_overlapped += _as_float(event.get("overlapped_checkpoint_sec", 0.0))
            total_critical += _as_float(event.get("critical_path_overhead_sec", 0.0))
    batch_exposed_wall_sec = _union_duration(
        [interval for event in events if (interval := _critical_interval(event)) is not None]
    )
    return {
        "policy": policy_dir.name,
        "trajectory_count": trajectory_count,
        "total_checkpoint_number": checkpoint_count,
        "average_checkpoint_cost_sec": total_elapsed / checkpoint_count if checkpoint_count else 0.0,
        "total_checkpoint_cost_sec": total_elapsed,
        "checkpoint_overlap_efficiency": total_overlapped / total_elapsed if total_elapsed > 0 else 0.0,
        "total_overlapped_checkpoint_sec": total_overlapped,
        "aggregate_exposed_overhead_traj_sec": total_critical,
        "batch_exposed_checkpoint_wall_sec": batch_exposed_wall_sec,
        "average_exposed_overhead_per_checkpoint_sec": total_critical / checkpoint_count if checkpoint_count else 0.0,
        "average_exposed_overhead_per_trajectory_sec": total_critical / trajectory_count if trajectory_count else 0.0,
        "source": "per_traj/*.json",
    }


def analyze_experiment(root: Path, *, prefer_summary: bool = True) -> list[dict[str, Any]]:
    if not root.exists():
        raise FileNotFoundError(root)
    rows: list[dict[str, Any]] = []
    for policy_dir in sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: _policy_sort_key(p.name)):
        row = _summary_overhead(policy_dir) if prefer_summary else None
        if row is None:
            row = _event_overhead(policy_dir)
        rows.append(row)
    oracle_wall_time = next(
        (
            _as_float(row.get("batch_wall_time_sec"), 0.0)
            for row in rows
            if row.get("policy") == "oracle-no-fault-no-checkpoint"
        ),
        0.0,
    )
    for row in rows:
        batch_wall_time = _as_float(row.get("batch_wall_time_sec"), 0.0)
        row["e2e_time_increase_vs_oracle"] = (
            (batch_wall_time - oracle_wall_time) / oracle_wall_time
            if oracle_wall_time > 0.0 and batch_wall_time > 0.0
            else 0.0
        )
    return rows


def _print_markdown(rows: list[dict[str, Any]]) -> None:
    headers = [
        "Policy",
        "Traj",
        "Total checkpoint number",
        "E2E increase vs oracle",
        "Total checkpoint cost (s)",
        "Average checkpoint cost (s)",
        "Checkpoint overlap efficiency",
        "Critical-path overhead (s)",
        "Exposed overhead / ckpt (s)",
        "Exposed overhead / traj (s)",
        "Batch exposed wall (s)",
    ]
    print("| " + " | ".join(headers) + " |")
    print("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        print(
            "| "
            + " | ".join(
                [
                    str(row["policy"]),
                    str(row["trajectory_count"]),
                    str(row["total_checkpoint_number"]),
                    _percent(float(row["e2e_time_increase_vs_oracle"])),
                    f"{row['total_checkpoint_cost_sec']:.3f}",
                    f"{row['average_checkpoint_cost_sec']:.3f}",
                    _percent(float(row["checkpoint_overlap_efficiency"])),
                    f"{row['aggregate_exposed_overhead_traj_sec']:.3f}",
                    f"{row['average_exposed_overhead_per_checkpoint_sec']:.3f}",
                    f"{row['average_exposed_overhead_per_trajectory_sec']:.3f}",
                    f"{row['batch_exposed_checkpoint_wall_sec']:.3f}",
                ]
            )
            + " |"
        )


def _print_csv(rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "policy",
        "trajectory_count",
        "total_checkpoint_number",
        "e2e_time_increase_vs_oracle",
        "average_checkpoint_cost_sec",
        "checkpoint_overlap_efficiency",
        "average_exposed_overhead_per_checkpoint_sec",
        "average_exposed_overhead_per_trajectory_sec",
        "total_checkpoint_cost_sec",
        "total_overlapped_checkpoint_sec",
        "aggregate_exposed_overhead_traj_sec",
        "batch_exposed_checkpoint_wall_sec",
        "batch_wall_time_sec",
        "mean_traj_wall_time_sec",
        "source",
    ]
    writer = csv.DictWriter(__import__("sys").stdout, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze checkpoint overhead for each policy in a SWE checkpoint fault experiment."
    )
    parser.add_argument("experiment_root", nargs="?", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--format", choices=["markdown", "csv", "json"], default="markdown")
    parser.add_argument(
        "--from-events",
        action="store_true",
        help="Recompute from per_traj checkpoint_events instead of using summary.json aggregates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = analyze_experiment(args.experiment_root, prefer_summary=not args.from_events)
    if args.format == "json":
        print(json.dumps(rows, indent=2, sort_keys=True))
    elif args.format == "csv":
        _print_csv(rows)
    else:
        _print_markdown(rows)


if __name__ == "__main__":
    main()
