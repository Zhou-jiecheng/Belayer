#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-9;]*m")
RUNNING_REQ_RE = re.compile(
    r"\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?: [^\]]+)?\].*?#running-req:\s*(?P<running>\d+)"
)


def parse_running_req_series(log_path: Path) -> tuple[list[datetime], list[int]]:
    timestamps: list[datetime] = []
    values: list[int] = []

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            line = ANSI_ESCAPE_RE.sub("", raw_line)
            if "SGLangEngine" not in line or "#running-req:" not in line:
                continue
            match = RUNNING_REQ_RE.search(line)
            if match is None:
                continue
            timestamps.append(datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S"))
            values.append(int(match.group("running")))

    return timestamps, values


def _percentile(values: list[int], percentile: float) -> float:
    if not values:
        raise ValueError("Cannot compute percentile of empty values")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * percentile
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def plot_series(timestamps: list[datetime], values: list[int], output_path: Path, title: str) -> None:
    avg_value = mean(values)
    p50_value = _percentile(values, 0.50)
    p90_value = _percentile(values, 0.90)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, values, linewidth=1.4, color="#1f77b4", label="#running-req")
    ax.axhline(avg_value, color="#d62728", linestyle="--", linewidth=1.2, label=f"AVG={avg_value:.2f}")
    ax.axhline(p50_value, color="#2ca02c", linestyle="--", linewidth=1.2, label=f"P50={p50_value:.2f}")
    ax.axhline(p90_value, color="#ff7f0e", linestyle="--", linewidth=1.2, label=f"P90={p90_value:.2f}")
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("#running-req")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")

    stat_text = f"AVG={avg_value:.2f}\nP50={p50_value:.2f}\nP90={p90_value:.2f}"
    ax.text(
        0.01,
        0.99,
        stat_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#bbbbbb"},
    )

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S"))
    fig.autofmt_xdate()
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot SGLangEngine #running-req time series from a swe-rl log file."
    )
    parser.add_argument("log_path", help="Path to the input log file")
    parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path. Defaults to <log_path>.running_req.png",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Defaults to the log file name.",
    )
    args = parser.parse_args()

    log_path = Path(args.log_path).expanduser().resolve()
    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else (log_path.parent / "figures" / f"{log_path.name}.running_req.png")
    )

    timestamps, values = parse_running_req_series(log_path)
    if not timestamps:
        raise RuntimeError(f"No SGLangEngine #running-req lines found in: {log_path}")

    title = args.title or f"SGLangEngine #running-req - {log_path.name}"
    plot_series(timestamps, values, output_path, title)
    print(f"Wrote plot to: {output_path}")
    print(f"Parsed points: {len(values)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
