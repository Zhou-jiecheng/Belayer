#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


STOP = False
_CGROUP_PATH_CACHE: dict[str, str] = {}
_CONTAINER_ID_RE = re.compile(r"(?:docker-|cri-containerd-)?([0-9a-f]{64})(?:\.scope)?", re.IGNORECASE)
try:
    _PROC_CLK_TCK = max(1, int(os.sysconf("SC_CLK_TCK")))
except Exception:
    _PROC_CLK_TCK = 100

_CGROUP_MEMORY_ROOTS = [
    Path("/sys/fs/cgroup/memory"),
    Path("/sys/fs/cgroup/unified"),
    Path("/sys/fs/cgroup"),
]
_CGROUP_CPU_ROOTS = [
    Path("/sys/fs/cgroup/cpu,cpuacct"),
    Path("/sys/fs/cgroup/cpuacct"),
    Path("/sys/fs/cgroup/cpu"),
    Path("/sys/fs/cgroup/unified"),
    Path("/sys/fs/cgroup"),
]
_CGROUP_IO_ROOTS = [
    Path("/sys/fs/cgroup/blkio"),
    Path("/sys/fs/cgroup/unified"),
    Path("/sys/fs/cgroup"),
]
_CPU_SAMPLE_MIN_ELAPSED_SEC = max(
    0.0,
    float(os.getenv("SWE_EXEC_STATS_CPU_SAMPLE_MIN_ELAPSED_SEC", "0.5")),
)
_CPU_SAMPLE_MAX_PERCENT = max(
    1.0,
    float(os.getenv("SWE_EXEC_STATS_MAX_CONTAINER_CPU_PERCENT", "1000.0")),
)


@dataclass
class ContainerInfo:
    container_id: str
    image: str
    name: str


@dataclass
class ContainerStats:
    ts: float
    container_id: str
    image: str
    name: str
    memory_usage_bytes: int
    cpu_percent: float
    disk_read_bytes: int
    disk_write_bytes: int
    cpu_sample_valid: bool = False
    cpu_source: str = ""


@dataclass
class ContainerPeak:
    container_id: str
    image: str
    name: str
    first_seen_ts: float
    last_seen_ts: float
    sample_count: int = 0
    peak_memory_usage_bytes: int = 0
    peak_cpu_percent: float = 0.0
    peak_disk_read_bytes: int = 0
    peak_disk_write_bytes: int = 0


def _handle_stop(signum: int, frame: Any) -> None:
    global STOP
    STOP = True


def _format_bytes(value: int | float) -> str:
    v = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if v < 1024.0 or unit == "TiB":
            return f"{v:.1f}{unit}" if unit != "B" else f"{int(v)}B"
        v /= 1024.0
    return f"{int(value)}B"


_DECIMAL_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def _parse_size_to_bytes(text: str) -> int:
    raw = str(text or "").strip()
    if not raw:
        return 0
    first = raw.split("/", 1)[0].strip()
    compact = first.replace(" ", "")
    num = []
    unit = []
    for ch in compact:
        if ch.isdigit() or ch in ".-":
            num.append(ch)
        else:
            unit.append(ch)
    try:
        value = float("".join(num))
    except Exception:
        return 0
    unit_key = "".join(unit).strip().lower() or "b"
    return int(value * _DECIMAL_UNITS.get(unit_key, 1))


def _parse_percent(text: str) -> float:
    try:
        return float(str(text or "").strip().rstrip("%"))
    except Exception:
        return 0.0


def _run(cmd: list[str], *, timeout: float) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
    )


def _list_running_containers(timeout: float) -> list[ContainerInfo]:
    result = _run(
        ["docker", "ps", "--no-trunc", "--format", "{{json .}}"],
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "docker ps failed")

    out: list[ContainerInfo] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        container_id = str(payload.get("ID") or payload.get("Container") or "").strip()
        if not container_id:
            continue
        out.append(
            ContainerInfo(
                container_id=container_id,
                image=str(payload.get("Image") or ""),
                name=str(payload.get("Names") or payload.get("Name") or ""),
            )
        )
    return out


def _infer_container_id_from_path(path: str) -> str | None:
    matches = _CONTAINER_ID_RE.findall(path)
    if not matches:
        return None
    return matches[-1].lower()


def _list_running_containers_from_cgroups() -> list[ContainerInfo]:
    roots = [Path("/sys/fs/cgroup/unified"), Path("/sys/fs/cgroup"), Path("/sys/fs/cgroup/memory")]
    memory_names = {"memory.current", "memory.usage_in_bytes"}
    found: dict[str, ContainerInfo] = {}

    for root in roots:
        if not root.exists():
            continue
        root_depth = len(root.parts)
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                depth = len(Path(dirpath).parts) - root_depth
                if depth > 7:
                    dirnames[:] = []
                    continue
                if not memory_names.intersection(filenames):
                    continue
                container_id = _infer_container_id_from_path(dirpath)
                if not container_id:
                    continue
                found.setdefault(
                    container_id,
                    ContainerInfo(container_id=container_id, image="", name=""),
                )
        except OSError:
            continue

    return sorted(found.values(), key=lambda item: item.container_id)


def _match_stats_container_id(reported: str, requested: list[str], used: set[str]) -> str | None:
    key = str(reported or "").strip()
    if not key:
        return None
    for container_id in requested:
        if container_id in used:
            continue
        if container_id == key or container_id.startswith(key) or key.startswith(container_id):
            return container_id
    return None


def _docker_stats(container_infos: list[ContainerInfo], *, timeout: float) -> list[ContainerStats]:
    if not container_infos:
        return []
    ids = [item.container_id for item in container_infos]
    info_by_id = {item.container_id: item for item in container_infos}
    result = _run(
        ["docker", "stats", "--no-stream", "--format", "{{json .}}", *ids],
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "docker stats failed")

    now = time.time()
    out: list[ContainerStats] = []
    used: set[str] = set()
    pending_payloads: list[dict[str, Any]] = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        reported = payload.get("Container") or payload.get("ID") or payload.get("Name")
        container_id = _match_stats_container_id(str(reported or ""), ids, used)
        if container_id is None:
            pending_payloads.append(payload)
            continue
        info = info_by_id[container_id]
        block_io = str(payload.get("BlockIO") or payload.get("Block I/O") or "")
        read_txt, _, write_txt = block_io.partition("/")
        out.append(
            ContainerStats(
                ts=now,
                container_id=container_id,
                image=info.image,
                name=info.name,
                memory_usage_bytes=_parse_size_to_bytes(str(payload.get("MemUsage") or "")),
                cpu_percent=_parse_percent(str(payload.get("CPUPerc") or "")),
                disk_read_bytes=_parse_size_to_bytes(read_txt),
                disk_write_bytes=_parse_size_to_bytes(write_txt),
                cpu_sample_valid=True,
                cpu_source="docker_stats",
            )
        )
        used.add(container_id)

    remaining = [container_id for container_id in ids if container_id not in used]
    for container_id, payload in zip(remaining, pending_payloads):
        info = info_by_id[container_id]
        block_io = str(payload.get("BlockIO") or payload.get("Block I/O") or "")
        read_txt, _, write_txt = block_io.partition("/")
        out.append(
            ContainerStats(
                ts=now,
                container_id=container_id,
                image=info.image,
                name=info.name,
                memory_usage_bytes=_parse_size_to_bytes(str(payload.get("MemUsage") or "")),
                cpu_percent=_parse_percent(str(payload.get("CPUPerc") or "")),
                disk_read_bytes=_parse_size_to_bytes(read_txt),
                disk_write_bytes=_parse_size_to_bytes(write_txt),
                cpu_sample_valid=True,
                cpu_source="docker_stats",
            )
        )
    return out


def _read_int_file(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def _first_existing_file(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def _find_related_cgroup_file(
    anchor_path: Path,
    *,
    anchor_roots: list[Path],
    target_roots: list[Path],
    names: list[str],
) -> Path | None:
    anchor_dir = anchor_path.parent
    direct = _first_existing_file(anchor_dir, names)
    if direct is not None:
        return direct

    seen: set[Path] = set()
    for anchor_root in anchor_roots:
        try:
            relative = anchor_dir.relative_to(anchor_root)
        except ValueError:
            continue
        for target_root in target_roots:
            candidate_dir = target_root / relative
            if candidate_dir in seen:
                continue
            seen.add(candidate_dir)
            found = _first_existing_file(candidate_dir, names)
            if found is not None:
                return found
    return None


def _candidate_cgroup_dirs(root: Path, container_id: str) -> list[Path]:
    short_id = container_id[:12]
    names = [
        container_id,
        short_id,
        f"docker-{container_id}.scope",
        f"docker-{short_id}.scope",
    ]
    parents = ["", "docker", "system.slice", "machine.slice"]
    return [
        ((root / parent) if parent else root) / name
        for parent in parents
        for name in names
    ]


def _find_cgroup_file(container_id: str, *, cache_kind: str, roots: list[Path], names: list[str]) -> Path | None:
    cache_key = f"{cache_kind}:{container_id}"
    cached = _CGROUP_PATH_CACHE.get(cache_key)
    if cached:
        cached_path = Path(cached)
        if cached_path.exists():
            return cached_path

    for root in roots:
        if not root.exists():
            continue
        for directory in _candidate_cgroup_dirs(root, container_id):
            for name in names:
                path = directory / name
                if path.exists():
                    _CGROUP_PATH_CACHE[cache_key] = str(path)
                    return path

    short_id = container_id[:12]
    for root in roots:
        if not root.exists():
            continue
        root_depth = len(root.parts)
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                depth = len(Path(dirpath).parts) - root_depth
                if depth > 6:
                    dirnames[:] = []
                    continue
                if container_id not in dirpath and short_id not in dirpath:
                    continue
                for name in names:
                    if name in filenames:
                        path = Path(dirpath) / name
                        _CGROUP_PATH_CACHE[cache_key] = str(path)
                        return path
        except OSError:
            continue
    return None


def _read_cgroup_cpu_usage_ns(path: Path) -> int:
    if path.name == "cpu.stat":
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "usage_usec":
                return int(parts[1]) * 1000
        raise ValueError(f"{path} does not expose usage_usec")
    return _read_int_file(path)


def _read_proc_stat_cpu_ticks(pid: int) -> int | None:
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    if len(fields) <= 12:
        return None
    try:
        return int(fields[11]) + int(fields[12])
    except (TypeError, ValueError):
        return None


def _read_proc_cpu_usage_ns(cgroup_dir: Path) -> int | None:
    pids: set[int] = set()
    root_depth = len(cgroup_dir.parts)
    try:
        for dirpath, dirnames, filenames in os.walk(cgroup_dir):
            depth = len(Path(dirpath).parts) - root_depth
            if depth > 4:
                dirnames[:] = []
                continue
            for name in ("tasks", "cgroup.procs"):
                if name not in filenames:
                    continue
                path = Path(dirpath) / name
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            pids.add(int(line))
                        except ValueError:
                            continue
                except OSError:
                    continue
    except OSError:
        return None

    if not pids:
        return None

    ticks = 0
    any_read = False
    for pid in pids:
        value = _read_proc_stat_cpu_ticks(pid)
        if value is None:
            continue
        any_read = True
        ticks += value
    if not any_read:
        return None
    return int(ticks * 1_000_000_000 / _PROC_CLK_TCK)


def _cpu_percent_from_usage(
    container_id: str,
    *,
    source: str,
    usage_ns: int,
    now: float,
    cpu_prev: dict[str, tuple[int, float]],
) -> tuple[float, bool]:
    prev_key = f"{source}:{container_id}"
    prev = cpu_prev.get(prev_key)
    if prev is None:
        cpu_prev[prev_key] = (usage_ns, now)
        return 0.0, False
    prev_usage_ns, prev_ts = prev
    elapsed = max(1e-6, now - prev_ts)
    if elapsed < _CPU_SAMPLE_MIN_ELAPSED_SEC:
        return 0.0, False
    cpu_prev[prev_key] = (usage_ns, now)
    delta_ns = usage_ns - prev_usage_ns
    if delta_ns <= 0:
        return 0.0, False
    cpu_percent = max(0.0, delta_ns / elapsed / 1e9 * 100.0)
    if cpu_percent > _CPU_SAMPLE_MAX_PERCENT:
        return 0.0, False
    return cpu_percent, True


def _read_cgroup_io_bytes(path: Path) -> tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if path.name == "io.stat":
            for item in parts[1:]:
                key, sep, value = item.partition("=")
                if not sep:
                    continue
                if key == "rbytes":
                    read_bytes += int(value)
                elif key == "wbytes":
                    write_bytes += int(value)
        elif len(parts) >= 3:
            op = parts[-2].lower()
            value = int(parts[-1])
            if op == "read":
                read_bytes += value
            elif op == "write":
                write_bytes += value
    return read_bytes, write_bytes


def _cgroup_stats(
    container_infos: list[ContainerInfo],
    *,
    cpu_prev: dict[str, tuple[int, float]],
) -> list[ContainerStats]:
    now = time.time()
    out: list[ContainerStats] = []
    for info in container_infos:
        container_id = info.container_id
        memory_path = _find_cgroup_file(
            container_id,
            cache_kind="memory",
            roots=_CGROUP_MEMORY_ROOTS,
            names=["memory.current", "memory.usage_in_bytes"],
        )
        if memory_path is None:
            continue
        cpu_path = _find_related_cgroup_file(
            memory_path,
            anchor_roots=_CGROUP_MEMORY_ROOTS,
            target_roots=_CGROUP_CPU_ROOTS,
            names=["cpuacct.usage", "cpu.stat"],
        )
        if cpu_path is None:
            cpu_path = _find_cgroup_file(
                container_id,
                cache_kind="cpu",
                roots=_CGROUP_CPU_ROOTS,
                names=["cpuacct.usage", "cpu.stat"],
            )
        io_path = _find_related_cgroup_file(
            memory_path,
            anchor_roots=_CGROUP_MEMORY_ROOTS,
            target_roots=_CGROUP_IO_ROOTS,
            names=["io.stat", "blkio.throttle.io_service_bytes_recursive", "blkio.throttle.io_service_bytes"],
        )
        if io_path is None:
            io_path = _find_cgroup_file(
                container_id,
                cache_kind="io",
                roots=_CGROUP_IO_ROOTS,
                names=["io.stat", "blkio.throttle.io_service_bytes_recursive", "blkio.throttle.io_service_bytes"],
            )
        memory_bytes = _read_int_file(memory_path)
        cpu_percent = 0.0
        cpu_sample_valid = False
        cpu_source = ""
        if cpu_path is not None:
            try:
                usage_ns = _read_cgroup_cpu_usage_ns(cpu_path)
                cpu_source = f"cgroup:{cpu_path.name}"
                cpu_percent, cpu_sample_valid = _cpu_percent_from_usage(
                    container_id,
                    source=cpu_source,
                    usage_ns=usage_ns,
                    now=now,
                    cpu_prev=cpu_prev,
                )
            except Exception:
                cpu_source = ""
        if not cpu_sample_valid:
            proc_usage_ns = _read_proc_cpu_usage_ns(memory_path.parent)
            if proc_usage_ns is not None:
                proc_cpu_percent, proc_valid = _cpu_percent_from_usage(
                    container_id,
                    source="procfs",
                    usage_ns=proc_usage_ns,
                    now=now,
                    cpu_prev=cpu_prev,
                )
                if proc_valid or not cpu_source:
                    cpu_percent = proc_cpu_percent
                    cpu_sample_valid = proc_valid
                    cpu_source = "procfs"
        disk_read_bytes = 0
        disk_write_bytes = 0
        if io_path is not None:
            disk_read_bytes, disk_write_bytes = _read_cgroup_io_bytes(io_path)
        out.append(
            ContainerStats(
                ts=now,
                container_id=container_id,
                image=info.image,
                name=info.name,
                memory_usage_bytes=memory_bytes,
                cpu_percent=cpu_percent,
                disk_read_bytes=disk_read_bytes,
                disk_write_bytes=disk_write_bytes,
                cpu_sample_valid=cpu_sample_valid,
                cpu_source=cpu_source,
            )
        )
    return out


def _update_peak(peaks: dict[str, ContainerPeak], sample: ContainerStats) -> None:
    peak = peaks.get(sample.container_id)
    if peak is None:
        peak = ContainerPeak(
            container_id=sample.container_id,
            image=sample.image,
            name=sample.name,
            first_seen_ts=sample.ts,
            last_seen_ts=sample.ts,
        )
        peaks[sample.container_id] = peak
    peak.image = sample.image or peak.image
    peak.name = sample.name or peak.name
    peak.last_seen_ts = sample.ts
    peak.sample_count += 1
    peak.peak_memory_usage_bytes = max(peak.peak_memory_usage_bytes, sample.memory_usage_bytes)
    peak.peak_cpu_percent = max(peak.peak_cpu_percent, sample.cpu_percent)
    peak.peak_disk_read_bytes = max(peak.peak_disk_read_bytes, sample.disk_read_bytes)
    peak.peak_disk_write_bytes = max(peak.peak_disk_write_bytes, sample.disk_write_bytes)


def _post_json(url: str, payload: dict[str, Any], *, timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise RuntimeError(f"{url} returned non-dict payload")
    return data


def _query_exec_server_stats_batch(
    base_url: str,
    container_ids: list[str],
    *,
    timeout: float,
) -> dict[str, dict[str, Any]]:
    if not base_url or not container_ids:
        return {}
    url = base_url.rstrip("/") + "/container/stats_batch"
    payload = _post_json(url, {"container_ids": container_ids, "include_raw": False}, timeout=timeout)
    stats = payload.get("stats")
    if not isinstance(stats, dict):
        raise RuntimeError(f"invalid stats_batch payload: {payload!r}")
    return {str(k): v for k, v in stats.items() if isinstance(v, dict)}


def _ratio(reported: float, observed: float) -> float | None:
    if observed <= 0:
        return None
    return reported / observed


def _compare_peaks(
    observed: dict[str, ContainerPeak],
    reported: dict[str, dict[str, Any]],
    *,
    min_observed_memory_bytes: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for container_id, obs in observed.items():
        if obs.peak_memory_usage_bytes < min_observed_memory_bytes:
            continue
        item = reported.get(container_id)
        if not isinstance(item, dict) or not item.get("ok", False):
            rows.append(
                {
                    "container_id": container_id,
                    "image": obs.image,
                    "name": obs.name,
                    "reported_ok": False,
                    "observed_peak_memory_bytes": obs.peak_memory_usage_bytes,
                    "observed_peak_cpu_percent": obs.peak_cpu_percent,
                    "memory_peak_ratio": None,
                    "cpu_peak_ratio": None,
                    "underreported_memory_bytes": obs.peak_memory_usage_bytes,
                    "error": "" if item is None else str(item.get("error", "")),
                }
            )
            continue
        reported_mem = float(
            item.get("peak_memory_usage_bytes", item.get("memory_usage_bytes", 0.0)) or 0.0
        )
        reported_cpu = float(item.get("peak_cpu_percent", item.get("cpu_percent", 0.0)) or 0.0)
        mem_ratio = _ratio(reported_mem, float(obs.peak_memory_usage_bytes))
        cpu_ratio = _ratio(reported_cpu, float(obs.peak_cpu_percent))
        rows.append(
            {
                "container_id": container_id,
                "image": obs.image,
                "name": obs.name,
                "reported_ok": True,
                "observed_peak_memory_bytes": obs.peak_memory_usage_bytes,
                "observed_peak_cpu_percent": obs.peak_cpu_percent,
                "reported_peak_memory_bytes": int(reported_mem),
                "reported_peak_cpu_percent": reported_cpu,
                "memory_peak_ratio": mem_ratio,
                "cpu_peak_ratio": cpu_ratio,
                "underreported_memory_bytes": max(0, int(obs.peak_memory_usage_bytes - reported_mem)),
                "sample_count": obs.sample_count,
            }
        )

    valid_mem = [row["memory_peak_ratio"] for row in rows if row.get("memory_peak_ratio") is not None]
    under_mem = [
        row
        for row in rows
        if row.get("memory_peak_ratio") is None or float(row.get("memory_peak_ratio") or 0.0) < 0.9
    ]
    worst = min(valid_mem) if valid_mem else None
    avg = (sum(valid_mem) / len(valid_mem)) if valid_mem else None
    summary = {
        "container_count": len(rows),
        "reported_count": sum(1 for row in rows if row.get("reported_ok")),
        "memory_peak_ratio_avg": avg,
        "memory_peak_ratio_worst": worst,
        "underreported_container_count": len(under_mem),
        "min_observed_memory_bytes": min_observed_memory_bytes,
    }
    return rows, summary


def _append_jsonl(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True, sort_keys=True) + "\n")


def _print_live(
    *,
    ts: float,
    running_count: int,
    peak_count: int,
    total_mem: int,
    total_peak_mem: int,
    compare_summary: dict[str, Any] | None,
) -> None:
    if compare_summary:
        avg = compare_summary.get("memory_peak_ratio_avg")
        worst = compare_summary.get("memory_peak_ratio_worst")
        avg_txt = "n/a" if avg is None else f"{100.0 * float(avg):5.1f}%"
        worst_txt = "n/a" if worst is None else f"{100.0 * float(worst):5.1f}%"
        compare_txt = (
            f"accuracy(avg={avg_txt},worst={worst_txt},"
            f"under={compare_summary.get('underreported_container_count', 0)})"
        )
    else:
        compare_txt = "accuracy(n/a)"
    print(
        "\r\033[K"
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(ts))} "
        f"running={running_count:4d} tracked={peak_count:4d} "
        f"mem_now={_format_bytes(total_mem):>10} "
        f"mem_peak={_format_bytes(total_peak_mem):>10} "
        f"{compare_txt}",
        end="",
        flush=True,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Continuously sample running Docker container resources and compare "
            "exec-server cached stats against high-frequency local observations. "
            "The default backends discover containers from cgroups and read cgroup "
            "files, avoiding docker ps and docker stats."
        )
    )
    parser.add_argument("--interval-sec", type=float, default=0.2, help="Ground-truth sampling interval.")
    parser.add_argument("--stats-backend", choices=["cgroup", "docker"], default="cgroup")
    parser.add_argument("--list-backend", choices=["cgroup", "docker"], default="cgroup")
    parser.add_argument("--list-interval-sec", type=float, default=2.0, help="How often to refresh the container list.")
    parser.add_argument("--duration-sec", type=float, default=0.0, help="0 means run until Ctrl-C/SIGTERM.")
    parser.add_argument("--docker-timeout-sec", type=float, default=5.0)
    parser.add_argument("--exec-server-url", default="http://127.0.0.1:5000")
    parser.add_argument("--compare-interval-sec", type=float, default=5.0)
    parser.add_argument("--compare-timeout-sec", type=float, default=10.0)
    parser.add_argument("--no-compare", action="store_true")
    parser.add_argument("--samples-jsonl", default="container_resource_truth_samples.jsonl")
    parser.add_argument("--compare-jsonl", default="container_resource_accuracy_compare.jsonl")
    parser.add_argument("--summary-json", default="container_resource_accuracy_summary.json")
    parser.add_argument("--min-observed-memory-mib", type=float, default=32.0)
    parser.add_argument("--max-compare-containers", type=int, default=256)
    parser.add_argument("--no-live", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    signal.signal(signal.SIGINT, _handle_stop)
    signal.signal(signal.SIGTERM, _handle_stop)

    interval = max(0.05, float(args.interval_sec))
    compare_interval = max(0.1, float(args.compare_interval_sec))
    samples_path = Path(args.samples_jsonl).expanduser() if args.samples_jsonl else None
    compare_path = Path(args.compare_jsonl).expanduser() if args.compare_jsonl else None
    summary_path = Path(args.summary_json).expanduser() if args.summary_json else None
    min_observed_memory_bytes = int(max(0.0, float(args.min_observed_memory_mib)) * 1024 * 1024)

    started = time.time()
    last_compare_ts = 0.0
    last_list_ts = 0.0
    last_compare_summary: dict[str, Any] | None = None
    peaks: dict[str, ContainerPeak] = {}
    cpu_prev: dict[str, tuple[int, float]] = {}
    known_containers: list[ContainerInfo] = []
    recent_errors: deque[str] = deque(maxlen=20)
    total_samples = 0
    total_compare_rounds = 0

    if not args.no_live:
        print(
            f"Sampling container resources via {args.stats_backend}, listing via {args.list_backend}. "
            "Press Ctrl-C to stop.",
            file=sys.stderr,
        )

    while not STOP:
        now = time.time()
        if args.duration_sec > 0 and now - started >= float(args.duration_sec):
            break

        if not known_containers or now - last_list_ts >= max(0.2, float(args.list_interval_sec)):
            try:
                if args.list_backend == "docker":
                    known_containers = _list_running_containers(timeout=float(args.docker_timeout_sec))
                else:
                    known_containers = _list_running_containers_from_cgroups()
                last_list_ts = now
            except Exception as exc:
                recent_errors.append(f"list error: {exc}")

        containers = known_containers
        try:
            if args.stats_backend == "docker":
                samples = _docker_stats(containers, timeout=float(args.docker_timeout_sec)) if containers else []
            else:
                samples = _cgroup_stats(containers, cpu_prev=cpu_prev) if containers else []
        except Exception as exc:
            recent_errors.append(f"sample error: {exc}")
            samples = []

        total_mem = 0
        for sample in samples:
            total_samples += 1
            total_mem += sample.memory_usage_bytes
            _update_peak(peaks, sample)
            _append_jsonl(samples_path, {"kind": "sample", **asdict(sample)})

        compare_due = (
            not args.no_compare
            and args.exec_server_url
            and time.time() - last_compare_ts >= compare_interval
            and bool(peaks)
        )
        if compare_due:
            last_compare_ts = time.time()
            total_compare_rounds += 1
            ids = list(peaks.keys())[: max(1, int(args.max_compare_containers))]
            try:
                reported = _query_exec_server_stats_batch(
                    str(args.exec_server_url),
                    ids,
                    timeout=float(args.compare_timeout_sec),
                )
                rows, summary = _compare_peaks(
                    peaks,
                    reported,
                    min_observed_memory_bytes=min_observed_memory_bytes,
                )
                summary.update(
                    {
                        "kind": "compare_summary",
                        "ts": time.time(),
                        "tracked_container_count": len(peaks),
                        "running_container_count": len(containers),
                    }
                )
                last_compare_summary = summary
                _append_jsonl(compare_path, summary)
                for row in rows:
                    _append_jsonl(compare_path, {"kind": "compare_container", "ts": summary["ts"], **row})
            except (urllib.error.URLError, TimeoutError, RuntimeError, OSError) as exc:
                recent_errors.append(f"compare error: {exc}")

        total_peak_mem = sum(item.peak_memory_usage_bytes for item in peaks.values())
        if not args.no_live:
            _print_live(
                ts=time.time(),
                running_count=len(containers),
                peak_count=len(peaks),
                total_mem=total_mem,
                total_peak_mem=total_peak_mem,
                compare_summary=last_compare_summary,
            )

        time.sleep(interval)

    if not args.no_live:
        print()

    final_summary = {
        "kind": "final_summary",
        "started_at": started,
        "finished_at": time.time(),
        "duration_sec": max(0.0, time.time() - started),
        "stats_backend": str(args.stats_backend),
        "list_backend": str(args.list_backend),
        "tracked_container_count": len(peaks),
        "total_truth_samples": total_samples,
        "total_compare_rounds": total_compare_rounds,
        "total_observed_peak_memory_bytes": sum(item.peak_memory_usage_bytes for item in peaks.values()),
        "last_compare_summary": last_compare_summary,
        "recent_errors": list(recent_errors),
        "samples_jsonl": str(samples_path) if samples_path else "",
        "compare_jsonl": str(compare_path) if compare_path else "",
        "summary_json": str(summary_path) if summary_path else "",
    }
    if summary_path is not None:
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(final_summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(final_summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
