#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


GIB = 1024 ** 3
STOP = False


@dataclass
class ResourceSample:
    ts: float
    hostname: str
    cpu_count: int
    cpu_used_percent: float
    cpu_available_percent: float
    cpu_total_scheduler_percent: float
    cpu_used_scheduler_percent: float
    cpu_available_scheduler_percent: float
    load1: float
    load5: float
    load15: float
    memory_total_bytes: int
    memory_available_bytes: int
    memory_free_bytes: int
    memory_used_bytes: int
    memory_used_percent: float
    memory_available_percent: float
    container_count: int
    status: str


def _handle_stop(signum: int, frame: Any) -> None:
    global STOP
    STOP = True


def _read_cpu_ticks() -> tuple[int, int]:
    with open("/proc/stat", "r", encoding="utf-8") as f:
        line = f.readline()
    parts = line.split()
    if not parts or parts[0] != "cpu":
        raise RuntimeError("failed to read aggregate cpu line from /proc/stat")
    values = [int(item) for item in parts[1:]]
    idle = values[3] + (values[4] if len(values) > 4 else 0)
    total = sum(values)
    return total, idle


def _read_meminfo_bytes() -> tuple[int, int, int]:
    fields: dict[str, int] = {}
    with open("/proc/meminfo", "r", encoding="utf-8") as f:
        for line in f:
            key, rest = line.split(":", 1)
            value = rest.strip().split()[0]
            fields[key] = int(value) * 1024
    total = int(fields.get("MemTotal", 0))
    available = int(fields.get("MemAvailable", fields.get("MemFree", 0)))
    free = int(fields.get("MemFree", 0))
    return total, available, free


def _loadavg() -> tuple[float, float, float]:
    try:
        return os.getloadavg()
    except Exception:
        return 0.0, 0.0, 0.0


def _docker_container_count() -> int:
    try:
        result = subprocess.run(
            ["docker", "ps", "-aq"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5.0,
        )
        if result.returncode != 0:
            return -1
        output = result.stdout.strip()
        if not output:
            return 0
        return len(output.splitlines())
    except Exception:
        return -1


def _format_bytes(value: int | float) -> str:
    v = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024.0 or unit == "TiB":
            return f"{v:.1f}{unit}" if unit != "B" else f"{int(v)}B"
        v /= 1024.0
    return f"{int(value)}B"


def _bar(value: float, *, width: int = 28) -> str:
    value = max(0.0, min(100.0, float(value)))
    filled = int(round(width * value / 100.0))
    return "#" * filled + "." * (width - filled)


def _status(cpu_used: float, mem_used: float, *, cpu_warn: float, cpu_crit: float, mem_warn: float, mem_crit: float) -> str:
    if cpu_used >= cpu_crit or mem_used >= mem_crit:
        return "critical"
    if cpu_used >= cpu_warn or mem_used >= mem_warn:
        return "warn"
    return "safe"


def _sample(prev_total: int, prev_idle: int, *, interval_elapsed: float, hostname: str, thresholds: dict[str, float]) -> tuple[ResourceSample, int, int]:
    total, idle = _read_cpu_ticks()
    dt_total = max(1, total - prev_total)
    dt_idle = max(0, idle - prev_idle)
    cpu_used = max(0.0, min(100.0, 100.0 * (1.0 - dt_idle / dt_total)))
    cpu_count = max(1, int(os.cpu_count() or 1))
    total_sched = float(cpu_count * 100.0)
    used_sched = total_sched * cpu_used / 100.0
    avail_sched = max(0.0, total_sched - used_sched)

    mem_total, mem_available, mem_free = _read_meminfo_bytes()
    mem_used = max(0, mem_total - mem_available)
    mem_used_percent = (100.0 * mem_used / mem_total) if mem_total > 0 else 0.0
    mem_avail_percent = (100.0 * mem_available / mem_total) if mem_total > 0 else 0.0
    load1, load5, load15 = _loadavg()
    container_count = _docker_container_count()
    status = _status(
        cpu_used,
        mem_used_percent,
        cpu_warn=thresholds["cpu_warn"],
        cpu_crit=thresholds["cpu_crit"],
        mem_warn=thresholds["memory_warn"],
        mem_crit=thresholds["memory_crit"],
    )
    return (
        ResourceSample(
            ts=time.time(),
            hostname=hostname,
            cpu_count=cpu_count,
            cpu_used_percent=cpu_used,
            cpu_available_percent=max(0.0, 100.0 - cpu_used),
            cpu_total_scheduler_percent=total_sched,
            cpu_used_scheduler_percent=used_sched,
            cpu_available_scheduler_percent=avail_sched,
            load1=float(load1),
            load5=float(load5),
            load15=float(load15),
            memory_total_bytes=mem_total,
            memory_available_bytes=mem_available,
            memory_free_bytes=mem_free,
            memory_used_bytes=mem_used,
            memory_used_percent=mem_used_percent,
            memory_available_percent=mem_avail_percent,
            container_count=container_count,
            status=status,
        ),
        total,
        idle,
    )


def _print_live(sample: ResourceSample, *, clear_line: bool) -> None:
    prefix = "\r\033[K" if clear_line else ""
    text = (
        f"{prefix}{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(sample.ts))} "
        f"host={sample.hostname} status={sample.status:<8} "
        f"cpu={sample.cpu_used_percent:5.1f}% [{_bar(sample.cpu_used_percent)}] "
        f"mem={sample.memory_used_percent:5.1f}% [{_bar(sample.memory_used_percent)}] "
        f"containers={sample.container_count:4d} "
        f"mem_avail={_format_bytes(sample.memory_available_bytes)} "
        f"load={sample.load1:.2f}/{sample.load5:.2f}/{sample.load15:.2f}"
    )
    print(text, end="" if clear_line else "\n", flush=True)


def _write_csv_header_if_needed(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(ResourceSample.__dataclass_fields__.keys()))
        writer.writeheader()


def _append_sample(sample: ResourceSample, *, jsonl_path: Path | None, csv_path: Path | None) -> None:
    payload = asdict(sample)
    if jsonl_path is not None:
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")
    if csv_path is not None:
        _write_csv_header_if_needed(csv_path)
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ResourceSample.__dataclass_fields__.keys()))
            writer.writerow(payload)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _svg_polyline(points: list[tuple[float, float]], *, width: int, height: int) -> str:
    if not points:
        return ""
    xs = [p[0] for p in points]
    min_x = min(xs)
    max_x = max(xs)
    span_x = max(1e-6, max_x - min_x)
    out = []
    for x, y in points:
        px = (x - min_x) / span_x * width
        py = height - (max(0.0, min(100.0, y)) / 100.0 * height)
        out.append(f"{px:.1f},{py:.1f}")
    return " ".join(out)


def _write_html(path: Path, samples: list[ResourceSample], *, cpu_warn: float, memory_warn: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width = 1100
    height = 260
    cpu_points = [(s.ts, s.cpu_used_percent) for s in samples]
    mem_points = [(s.ts, s.memory_used_percent) for s in samples]
    cpu_line = _svg_polyline(cpu_points, width=width, height=height)
    mem_line = _svg_polyline(mem_points, width=width, height=height)
    warn_cpu_y = height - (max(0.0, min(100.0, cpu_warn)) / 100.0 * height)
    warn_mem_y = height - (max(0.0, min(100.0, memory_warn)) / 100.0 * height)
    last = samples[-1] if samples else None
    rows = "\n".join(
        f"<tr><td>{time.strftime('%H:%M:%S', time.localtime(s.ts))}</td>"
        f"<td>{s.status}</td><td>{s.cpu_used_percent:.2f}</td>"
        f"<td>{s.memory_used_percent:.2f}</td><td>{_html_escape(_format_bytes(s.memory_available_bytes))}</td>"
        f"<td>{s.container_count}</td><td>{s.load1:.2f}</td></tr>"
        for s in samples[-120:]
    )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>SWE Exec Server Resources</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #17202a; }}
    .metric {{ display: inline-block; margin-right: 28px; }}
    svg {{ width: 100%; max-width: {width}px; height: {height + 40}px; border: 1px solid #c8d0d8; background: #fbfcfd; }}
    .cpu {{ fill: none; stroke: #2563eb; stroke-width: 2.5; }}
    .mem {{ fill: none; stroke: #dc2626; stroke-width: 2.5; }}
    .warn {{ stroke: #f59e0b; stroke-width: 1.2; stroke-dasharray: 6 5; }}
    table {{ border-collapse: collapse; margin-top: 18px; font-size: 13px; }}
    td, th {{ border: 1px solid #d8dee6; padding: 5px 8px; }}
    th {{ background: #eef2f7; }}
  </style>
</head>
<body>
  <h1>SWE Exec Server Resources</h1>
  <p>Generated at {time.strftime('%Y-%m-%d %H:%M:%S')} from local /proc samples.</p>
  <div>
    <span class="metric"><b>Host:</b> {_html_escape(last.hostname if last else '')}</span>
    <span class="metric"><b>Status:</b> {_html_escape(last.status if last else 'n/a')}</span>
    <span class="metric"><b>CPU:</b> {(last.cpu_used_percent if last else 0.0):.1f}%</span>
    <span class="metric"><b>Memory:</b> {(last.memory_used_percent if last else 0.0):.1f}%</span>
    <span class="metric"><b>Containers:</b> {(last.container_count if last else 0)}</span>
    <span class="metric"><b>Mem available:</b> {_html_escape(_format_bytes(last.memory_available_bytes) if last else '0B')}</span>
  </div>
  <svg viewBox="0 0 {width} {height + 40}" preserveAspectRatio="none">
    <line class="warn" x1="0" x2="{width}" y1="{warn_cpu_y:.1f}" y2="{warn_cpu_y:.1f}"></line>
    <line class="warn" x1="0" x2="{width}" y1="{warn_mem_y:.1f}" y2="{warn_mem_y:.1f}"></line>
    <polyline class="cpu" points="{cpu_line}"></polyline>
    <polyline class="mem" points="{mem_line}"></polyline>
    <text x="8" y="18" fill="#2563eb">CPU used %</text>
    <text x="8" y="36" fill="#dc2626">Memory used %</text>
  </svg>
  <table>
    <tr><th>Time</th><th>Status</th><th>CPU %</th><th>Memory %</th><th>Mem available</th><th>Containers</th><th>Load1</th></tr>
    {rows}
  </table>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor local SWE exec server host CPU/memory by reading /proc directly. "
            "This script does not query swe_env_pool_server or any HTTP endpoint."
        )
    )
    parser.add_argument("--interval-sec", type=float, default=1.0)
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until Ctrl-C/SIGTERM.")
    parser.add_argument("--jsonl", default="exec_server_resource_samples.jsonl")
    parser.add_argument("--csv", default="")
    parser.add_argument("--html", default="exec_server_resource_monitor.html")
    parser.add_argument("--html-update-sec", type=float, default=5.0)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--cpu-warn-percent", type=float, default=85.0)
    parser.add_argument("--cpu-critical-percent", type=float, default=95.0)
    parser.add_argument("--memory-warn-percent", type=float, default=85.0)
    parser.add_argument("--memory-critical-percent", type=float, default=95.0)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    interval = max(0.1, float(args.interval_sec))
    jsonl_path = Path(args.jsonl).expanduser() if args.jsonl else None
    csv_path = Path(args.csv).expanduser() if args.csv else None
    html_path = Path(args.html).expanduser() if args.html else None
    thresholds = {
        "cpu_warn": float(args.cpu_warn_percent),
        "cpu_crit": float(args.cpu_critical_percent),
        "memory_warn": float(args.memory_warn_percent),
        "memory_crit": float(args.memory_critical_percent),
    }

    hostname = socket.gethostname()
    prev_total, prev_idle = _read_cpu_ticks()
    samples: list[ResourceSample] = []
    started = time.time()
    last_html = 0.0

    if not args.no_live:
        print(
            "Monitoring local exec server resources from /proc. "
            "Press Ctrl-C to stop.",
            file=sys.stderr,
        )

    while not STOP:
        if args.duration_sec > 0 and time.time() - started >= float(args.duration_sec):
            break
        time.sleep(interval)
        sample, prev_total, prev_idle = _sample(
            prev_total,
            prev_idle,
            interval_elapsed=interval,
            hostname=hostname,
            thresholds=thresholds,
        )
        samples.append(sample)
        _append_sample(sample, jsonl_path=jsonl_path, csv_path=csv_path)
        if not args.no_live:
            _print_live(sample, clear_line=True)
        now = time.time()
        if html_path is not None and now - last_html >= max(0.5, float(args.html_update_sec)):
            _write_html(
                html_path,
                samples,
                cpu_warn=float(args.cpu_warn_percent),
                memory_warn=float(args.memory_warn_percent),
            )
            last_html = now

    if not args.no_live:
        print()
    if html_path is not None and samples:
        _write_html(
            html_path,
            samples,
            cpu_warn=float(args.cpu_warn_percent),
            memory_warn=float(args.memory_warn_percent),
        )
    if samples:
        peak_cpu = max(s.cpu_used_percent for s in samples)
        peak_mem = max(s.memory_used_percent for s in samples)
        peak_containers = max(s.container_count for s in samples)
        print(
            json.dumps(
                {
                    "hostname": hostname,
                    "sample_count": len(samples),
                    "duration_sec": max(0.0, samples[-1].ts - samples[0].ts),
                    "peak_cpu_used_percent": peak_cpu,
                    "peak_memory_used_percent": peak_mem,
                    "peak_container_count": peak_containers,
                    "final_status": samples[-1].status,
                    "jsonl": str(jsonl_path) if jsonl_path is not None else "",
                    "csv": str(csv_path) if csv_path is not None else "",
                    "html": str(html_path) if html_path is not None else "",
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
