#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _format_bytes(value: int | float) -> str:
    v = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024.0 or unit == "TiB":
            return f"{v:.1f}{unit}" if unit != "B" else f"{int(v)}B"
        v /= 1024.0
    return f"{int(value)}B"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(values)
    pos = max(0.0, min(1.0, q)) * (len(ordered) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


@dataclass
class ContainerAccumulator:
    container_id: str
    image: str = ""
    name: str = ""
    first_ts: float | None = None
    last_ts: float | None = None
    sample_count: int = 0
    memory_values: list[float] = field(default_factory=list)
    cpu_values: list[float] = field(default_factory=list)
    cpu_values_excluding_first: list[float] = field(default_factory=list)
    first_disk_read_bytes: int | None = None
    last_disk_read_bytes: int | None = None
    peak_disk_read_bytes: int = 0
    first_disk_write_bytes: int | None = None
    last_disk_write_bytes: int | None = None
    peak_disk_write_bytes: int = 0

    def add(self, row: dict[str, Any]) -> None:
        ts = float(row.get("ts", 0.0) or 0.0)
        memory = float(row.get("memory_usage_bytes", 0.0) or 0.0)
        cpu = float(row.get("cpu_percent", 0.0) or 0.0)
        disk_read = int(float(row.get("disk_read_bytes", 0.0) or 0.0))
        disk_write = int(float(row.get("disk_write_bytes", 0.0) or 0.0))
        image = str(row.get("image", "") or "")
        name = str(row.get("name", "") or "")
        if image:
            self.image = image
        if name:
            self.name = name
        if self.first_ts is None or ts < self.first_ts:
            self.first_ts = ts
            self.first_disk_read_bytes = disk_read
            self.first_disk_write_bytes = disk_write
        if self.last_ts is None or ts >= self.last_ts:
            self.last_ts = ts
            self.last_disk_read_bytes = disk_read
            self.last_disk_write_bytes = disk_write
        self.sample_count += 1
        self.memory_values.append(memory)
        self.cpu_values.append(cpu)
        if self.sample_count > 1:
            self.cpu_values_excluding_first.append(cpu)
        self.peak_disk_read_bytes = max(self.peak_disk_read_bytes, disk_read)
        self.peak_disk_write_bytes = max(self.peak_disk_write_bytes, disk_write)

    def to_row(self) -> dict[str, Any]:
        mem = self.memory_values
        cpu = self.cpu_values_excluding_first or self.cpu_values
        first_ts = float(self.first_ts or 0.0)
        last_ts = float(self.last_ts or first_ts)
        duration = max(0.0, last_ts - first_ts)
        read_delta = max(
            0,
            int((self.last_disk_read_bytes or 0) - (self.first_disk_read_bytes or 0)),
        )
        write_delta = max(
            0,
            int((self.last_disk_write_bytes or 0) - (self.first_disk_write_bytes or 0)),
        )
        return {
            "container_id": self.container_id,
            "container_short_id": self.container_id[:12],
            "image": self.image,
            "name": self.name,
            "sample_count": self.sample_count,
            "first_ts": first_ts,
            "last_ts": last_ts,
            "duration_sec": duration,
            "avg_cpu_percent": sum(cpu) / len(cpu) if cpu else 0.0,
            "p50_cpu_percent": _percentile(cpu, 0.50),
            "p95_cpu_percent": _percentile(cpu, 0.95),
            "peak_cpu_percent": max(cpu) if cpu else 0.0,
            "avg_memory_bytes": int(sum(mem) / len(mem)) if mem else 0,
            "p50_memory_bytes": int(_percentile(mem, 0.50)),
            "p95_memory_bytes": int(_percentile(mem, 0.95)),
            "peak_memory_bytes": int(max(mem)) if mem else 0,
            "min_memory_bytes": int(min(mem)) if mem else 0,
            "avg_memory_human": _format_bytes(sum(mem) / len(mem)) if mem else "0B",
            "p95_memory_human": _format_bytes(_percentile(mem, 0.95)),
            "peak_memory_human": _format_bytes(max(mem)) if mem else "0B",
            "disk_read_delta_bytes": read_delta,
            "disk_write_delta_bytes": write_delta,
            "peak_disk_read_bytes": self.peak_disk_read_bytes,
            "peak_disk_write_bytes": self.peak_disk_write_bytes,
            "disk_read_delta_human": _format_bytes(read_delta),
            "disk_write_delta_human": _format_bytes(write_delta),
        }


def _load_samples(path: Path) -> dict[str, ContainerAccumulator]:
    containers: dict[str, ContainerAccumulator] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON at line {line_no}: {exc}") from exc
            if row.get("kind") not in (None, "sample"):
                continue
            container_id = str(row.get("container_id", "") or "").strip()
            if not container_id:
                continue
            acc = containers.get(container_id)
            if acc is None:
                acc = ContainerAccumulator(container_id=container_id)
                containers[container_id] = acc
            acc.add(row)
    return containers


def _overall(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    return {
        "container_count": len(rows),
        "total_sample_count": sum(int(row["sample_count"]) for row in rows),
        "avg_container_cpu_percent": sum(float(row["avg_cpu_percent"]) for row in rows) / len(rows),
        "peak_container_cpu_percent": max(float(row["peak_cpu_percent"]) for row in rows),
        "avg_container_memory_bytes": int(sum(int(row["avg_memory_bytes"]) for row in rows) / len(rows)),
        "p95_container_peak_memory_bytes": int(
            _percentile([float(row["peak_memory_bytes"]) for row in rows], 0.95)
        ),
        "max_container_peak_memory_bytes": max(int(row["peak_memory_bytes"]) for row in rows),
        "total_disk_read_delta_bytes": sum(int(row["disk_read_delta_bytes"]) for row in rows),
        "total_disk_write_delta_bytes": sum(int(row["disk_write_delta_bytes"]) for row in rows),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize cgroup truth samples per container.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--csv-output", required=True)
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--overall-output", required=True)
    parser.add_argument("--sort-by", default="peak_memory_bytes")
    parser.add_argument("--descending", action="store_true", default=True)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    containers = _load_samples(Path(args.input).expanduser())
    rows = [acc.to_row() for acc in containers.values()]
    sort_key = str(args.sort_by)
    rows.sort(key=lambda row: row.get(sort_key, 0), reverse=bool(args.descending))

    csv_path = Path(args.csv_output).expanduser()
    json_path = Path(args.json_output).expanduser()
    overall_path = Path(args.overall_output).expanduser()
    _write_csv(csv_path, rows)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    overall = _overall(rows)
    overall_path.parent.mkdir(parents=True, exist_ok=True)
    overall_path.write_text(json.dumps(overall, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "container_count": len(rows),
                "csv_output": str(csv_path),
                "json_output": str(json_path),
                "overall_output": str(overall_path),
                **overall,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
