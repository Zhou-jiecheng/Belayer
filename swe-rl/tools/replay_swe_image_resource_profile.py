from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
SWE_RL_ROOT = REPO_ROOT / "swe-rl"
for path in (TOOLS_ROOT, SWE_RL_ROOT):
    sys.path.insert(0, str(path))

from replay_swe_traj_checkpoint import (  # noqa: E402
    ReplayEnvClient,
    _build_image_name,
    _collect_traj_paths,
    _default_instance_id,
    _json_default,
    _load_traj_steps,
    _load_yaml_config,
)


DEFAULT_SAMPLE_INTERVAL_SEC = 1.0
DEFAULT_MAX_CONCURRENCY = 1
DEFAULT_NEIGHBOR_COUNT = 5


def _default_output_root(trajectory_root: str) -> Path:
    root = Path(trajectory_root).resolve()
    if root.is_file():
        return root.parent / "replay_resource_profile"
    return root.parent / "replay_resource_profile"


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    return float(value)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def _describe(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "max": 0.0,
            "min": 0.0,
            "P50": 0.0,
            "P90": 0.0,
            "P95": 0.0,
            "P99": 0.0,
            "avg": 0.0,
        }
    return {
        "max": max(values),
        "min": min(values),
        "P50": _quantile(values, 0.50),
        "P90": _quantile(values, 0.90),
        "P95": _quantile(values, 0.95),
        "P99": _quantile(values, 0.99),
        "avg": mean(values),
    }


def _bytes_to_mb(value: float) -> float:
    return float(value) / (1024.0 * 1024.0)


def _slugify_image(image: str) -> str:
    tail = image.split(":")[-1] if ":" in image else image
    tail = re.sub(r"[^A-Za-z0-9._-]+", "_", tail).strip("._-")
    return tail or "unknown_image"


class ProfilingReplayEnvClient(ReplayEnvClient):
    def _get_blocking(self, path: str, timeout: float) -> dict[str, Any]:
        with urllib_request.urlopen(f"{self.base_url}/{path}", timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            snippet = raw[:200].replace("\n", "\\n")
            raise RuntimeError(
                f"GET {path} returned non-JSON payload: {snippet}"
            ) from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unexpected response payload for GET {path}: {parsed!r}")
        return parsed

    async def status(self) -> dict[str, Any]:
        return await asyncio.to_thread(self._get_blocking, "status", 30.0)

    async def lease_stats(self, lease_id: str) -> dict[str, Any]:
        return await self._post_with_retry(
            path="stats",
            payload={"lease_id": lease_id},
            op_name="lease_stats",
            timeout=30.0,
            http_max_retries=self.default_http_max_retries,
            app_max_retries=1,
        )


def _extract_container_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        raw = payload.get("container_id")
        if raw:
            return str(raw)
        for value in payload.values():
            resolved = _extract_container_id(value)
            if resolved:
                return resolved
    elif isinstance(payload, list):
        for item in payload:
            resolved = _extract_container_id(item)
            if resolved:
                return resolved
    return None


async def _resolve_container_id(
    env_client: ProfilingReplayEnvClient,
    lease: dict[str, Any],
) -> tuple[str | None, dict[str, Any] | None]:
    container_id = _extract_container_id(lease)
    if container_id:
        return container_id, None

    lease_id = str(lease.get("lease_id", "") or "")
    if not lease_id:
        return None, None

    status = await env_client.status()
    containers = status.get("containers", {})
    if not isinstance(containers, dict):
        return None, None
    for cid, info in containers.items():
        if isinstance(info, dict) and str(info.get("lease_id", "") or "") == lease_id:
            return str(cid), info
    return None, None


async def _sample_container_stats(
    env_client: ProfilingReplayEnvClient,
    lease_id: str,
    interval_sec: float,
    stop_event: asyncio.Event,
    samples: list[dict[str, Any]],
    errors: list[str],
) -> None:
    while True:
        started = time.time()
        try:
            payload = await env_client.lease_stats(lease_id)
            samples.append(
                {
                    "ts": _safe_float(payload.get("ts"), time.time()),
                    "memory_usage_bytes": int(payload.get("memory_usage_bytes", 0) or 0),
                    "cpu_percent": _safe_float(payload.get("cpu_percent")),
                    "disk_read_bytes": int(payload.get("disk_read_bytes", 0) or 0),
                    "disk_write_bytes": int(payload.get("disk_write_bytes", 0) or 0),
                }
            )
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

        if stop_event.is_set():
            break
        remaining = max(0.05, interval_sec - (time.time() - started))
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=remaining)
        except asyncio.TimeoutError:
            pass


def _build_run_profile(
    *,
    traj_path: Path,
    instance_id: str,
    image_name: str,
    lease: dict[str, Any],
    container_id: str | None,
    cwd: str,
    exec_timeout: int,
    simulate_llm_delay: bool,
    start_time: float,
    end_time: float,
    samples: list[dict[str, Any]],
    sample_errors: list[str],
    step_reports: list[dict[str, Any]],
    error_text: str | None,
) -> dict[str, Any]:
    mem_mb_values = [_bytes_to_mb(item["memory_usage_bytes"]) for item in samples]
    cpu_values = [_safe_float(item["cpu_percent"]) for item in samples]
    disk_read_values = [_bytes_to_mb(item["disk_read_bytes"]) for item in samples]
    disk_write_values = [_bytes_to_mb(item["disk_write_bytes"]) for item in samples]

    profile = {
        "mode": "traj_replay_resource_profile",
        "provider_name": "swe_exec_server",
        "trajectory": str(traj_path.resolve()),
        "instance_id": instance_id,
        "image": image_name,
        "lease_id": str(lease.get("lease_id", "") or ""),
        "container_id": container_id or "",
        "container_name": str(lease.get("name", "") or lease.get("container_name", "") or ""),
        "cwd": cwd,
        "exec_timeout": int(exec_timeout),
        "simulate_llm_delay": bool(simulate_llm_delay),
        "start_time": float(start_time),
        "end_time": float(end_time),
        "duration_sec": max(0.0, float(end_time - start_time)),
        "sample_count": len(samples),
        "cpu_percent": _describe(cpu_values),
        "mem_mb": _describe(mem_mb_values),
        "disk_read_mb": _describe(disk_read_values),
        "disk_write_mb": _describe(disk_write_values),
        "disk_used_mb": _describe([]),
        "disk_used_percent": _describe([]),
        "docker_error_count": len(sample_errors),
        "docker_last_error": sample_errors[-1] if sample_errors else "",
        "replay_ok": error_text is None,
        "replay_error": error_text,
        "step_count": len(step_reports),
        "steps": step_reports,
        "samples": samples,
    }
    return profile


async def _profile_one_trajectory(
    traj_path: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    traj_payload, steps = _load_traj_steps(str(traj_path), start_step=args.start_step, end_step=args.end_step)
    swe_config = _load_yaml_config(args.config_path)
    env_config = swe_config.get("environment", {}) if isinstance(swe_config, dict) else {}
    cwd = args.cwd or str(env_config.get("cwd", "/testbed"))
    exec_timeout = int(args.exec_timeout or int(env_config.get("timeout", 180)))
    instance_id = args.instance_id or _default_instance_id(traj_payload)
    image_name = _build_image_name(
        traj_payload,
        instance_id=instance_id,
        image_name=args.image_name,
        data_source=args.data_source,
    )

    env_client = ProfilingReplayEnvClient(base_url=args.base_url)
    lease = await env_client.allocate(image=image_name, instance_id=instance_id, cwd=cwd)
    lease_id = str(lease["lease_id"])
    container_id, status_info = await _resolve_container_id(env_client, lease)

    samples: list[dict[str, Any]] = []
    sample_errors: list[str] = []
    step_reports: list[dict[str, Any]] = []
    stop_event = asyncio.Event()
    sampler_task: asyncio.Task[Any] | None = None
    started_at = time.time()
    replay_error: str | None = None
    closed = False

    if lease_id:
        sampler_task = asyncio.create_task(
            _sample_container_stats(
                env_client,
                lease_id=lease_id,
                interval_sec=max(0.05, float(args.sample_interval_sec)),
                stop_event=stop_event,
                samples=samples,
                errors=sample_errors,
            )
        )

    try:
        for step in steps:
            await env_client.heartbeat(lease_id)
            llm_sleep_sec = step.llm_elapsed if args.simulate_llm_delay else 0.0
            if llm_sleep_sec > 0:
                await asyncio.sleep(llm_sleep_sec)
            exec_started_at = time.time()
            exec_result = await env_client.exec(
                lease_id=lease_id,
                command=step.action,
                cwd=cwd,
                timeout=exec_timeout,
            )
            exec_finished_at = time.time()
            output = str(exec_result.get("output", ""))
            step_reports.append(
                {
                    "step_idx": step.step_idx,
                    "action": step.action,
                    "expected_returncode": step.expected_returncode,
                    "actual_returncode": int(exec_result.get("returncode", -1)),
                    "simulated_llm_delay_sec": llm_sleep_sec,
                    "exec_elapsed_sec": max(0.0, exec_finished_at - exec_started_at),
                    "output_preview": output[:2000],
                }
            )
    except Exception as exc:
        replay_error = f"{type(exc).__name__}: {exc}"
    finally:
        stop_event.set()
        if sampler_task is not None:
            try:
                await sampler_task
            except Exception as exc:  # pragma: no cover
                sample_errors.append(f"sampler_failed: {type(exc).__name__}: {exc}")

        if lease_id:
            try:
                final_sample = await env_client.lease_stats(lease_id)
                samples.append(
                    {
                        "ts": _safe_float(final_sample.get("ts"), time.time()),
                        "memory_usage_bytes": int(final_sample.get("memory_usage_bytes", 0) or 0),
                        "cpu_percent": _safe_float(final_sample.get("cpu_percent")),
                        "disk_read_bytes": int(final_sample.get("disk_read_bytes", 0) or 0),
                        "disk_write_bytes": int(final_sample.get("disk_write_bytes", 0) or 0),
                    }
                )
            except Exception as exc:
                sample_errors.append(f"final_sample_failed: {type(exc).__name__}: {exc}")

        try:
            await env_client.close(lease_id)
            closed = True
        except Exception as exc:
            sample_errors.append(f"close_failed: {type(exc).__name__}: {exc}")

    ended_at = time.time()
    profile = _build_run_profile(
        traj_path=traj_path,
        instance_id=instance_id,
        image_name=image_name,
        lease=lease,
        container_id=container_id,
        cwd=cwd,
        exec_timeout=exec_timeout,
        simulate_llm_delay=bool(args.simulate_llm_delay),
        start_time=started_at,
        end_time=ended_at,
        samples=samples,
        sample_errors=sample_errors,
        step_reports=step_reports,
        error_text=replay_error,
    )
    profile["lease"] = lease
    profile["status_container_info"] = status_info
    profile["closed"] = closed
    return profile


def _feature_vector_from_profile(profile: dict[str, Any]) -> list[float]:
    return [
        _safe_float(profile.get("duration_sec")),
        _safe_float(profile.get("cpu_percent", {}).get("avg")),
        _safe_float(profile.get("cpu_percent", {}).get("max")),
        _safe_float(profile.get("mem_mb", {}).get("avg")),
        _safe_float(profile.get("mem_mb", {}).get("max")),
        _safe_float(profile.get("disk_read_mb", {}).get("max")),
        _safe_float(profile.get("disk_write_mb", {}).get("max")),
    ]


def _similarity_matrix(feature_rows: list[list[float]]) -> list[list[float]]:
    if not feature_rows:
        return []
    means = [mean(column) for column in zip(*feature_rows)]
    stds = []
    for idx, column in enumerate(zip(*feature_rows)):
        values = list(column)
        col_mean = means[idx]
        variance = mean([(value - col_mean) ** 2 for value in values]) if values else 0.0
        std = math.sqrt(variance)
        stds.append(std if std > 0 else 1.0)
    normalized = [
        [(row[idx] - means[idx]) / stds[idx] for idx in range(len(row))]
        for row in feature_rows
    ]
    matrix: list[list[float]] = []
    for left in normalized:
        row = []
        for right in normalized:
            distance = math.sqrt(sum((left[idx] - right[idx]) ** 2 for idx in range(len(left))))
            row.append(1.0 / (1.0 + distance))
        matrix.append(row)
    return matrix


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _aggregate_profiles_by_image(
    run_profiles: list[dict[str, Any]],
    output_root: Path,
    neighbor_count: int,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for profile in run_profiles:
        grouped[str(profile.get("image", "") or "unknown_image")].append(profile)

    image_summaries: list[dict[str, Any]] = []
    for image in sorted(grouped):
        profiles = grouped[image]
        duration_values = [_safe_float(item.get("duration_sec")) for item in profiles]
        cpu_avg_values = [_safe_float(item.get("cpu_percent", {}).get("avg")) for item in profiles]
        cpu_max_values = [_safe_float(item.get("cpu_percent", {}).get("max")) for item in profiles]
        mem_avg_values = [_safe_float(item.get("mem_mb", {}).get("avg")) for item in profiles]
        mem_max_values = [_safe_float(item.get("mem_mb", {}).get("max")) for item in profiles]
        disk_read_values = [_safe_float(item.get("disk_read_mb", {}).get("max")) for item in profiles]
        disk_write_values = [_safe_float(item.get("disk_write_mb", {}).get("max")) for item in profiles]
        feature_vector = [
            mean(duration_values) if duration_values else 0.0,
            mean(cpu_avg_values) if cpu_avg_values else 0.0,
            mean(cpu_max_values) if cpu_max_values else 0.0,
            mean(mem_avg_values) if mem_avg_values else 0.0,
            mean(mem_max_values) if mem_max_values else 0.0,
            mean(disk_read_values) if disk_read_values else 0.0,
            mean(disk_write_values) if disk_write_values else 0.0,
        ]
        image_summaries.append(
            {
                "image": image,
                "image_slug": _slugify_image(image),
                "run_count": len(profiles),
                "aggregate": {
                    "duration_sec": _describe(duration_values),
                    "cpu_avg_percent": _describe(cpu_avg_values),
                    "cpu_max_percent": _describe(cpu_max_values),
                    "mem_avg_mb": _describe(mem_avg_values),
                    "mem_max_mb": _describe(mem_max_values),
                    "disk_read_max_mb": _describe(disk_read_values),
                    "disk_write_max_mb": _describe(disk_write_values),
                },
                "feature_vector": feature_vector,
                "runs": [
                    {
                        "trajectory": item.get("trajectory"),
                        "instance_id": item.get("instance_id"),
                        "sample_count": item.get("sample_count"),
                        "duration_sec": item.get("duration_sec"),
                        "cpu_percent": item.get("cpu_percent"),
                        "mem_mb": item.get("mem_mb"),
                        "disk_read_mb": item.get("disk_read_mb"),
                        "disk_write_mb": item.get("disk_write_mb"),
                        "replay_ok": item.get("replay_ok"),
                        "profile_path": item.get("_profile_path"),
                    }
                    for item in profiles
                ],
            }
        )

    feature_rows = [item["feature_vector"] for item in image_summaries]
    sim = _similarity_matrix(feature_rows)
    top_pairs: list[dict[str, Any]] = []

    for idx, item in enumerate(image_summaries):
        neighbors = []
        for other_idx, other in enumerate(image_summaries):
            if idx == other_idx:
                continue
            neighbors.append(
                {
                    "image": other["image"],
                    "image_slug": other["image_slug"],
                    "similarity": sim[idx][other_idx],
                    "run_count": other["run_count"],
                }
            )
        neighbors.sort(key=lambda row: (-row["similarity"], row["image"]))
        item["nearest_neighbors"] = neighbors[:neighbor_count]
        image_path = output_root / "by_image" / f"{item['image_slug']}.json"
        item["image_profile_path"] = str(image_path.resolve())

    for left_idx in range(len(image_summaries)):
        for right_idx in range(left_idx + 1, len(image_summaries)):
            top_pairs.append(
                {
                    "image_left": image_summaries[left_idx]["image"],
                    "image_right": image_summaries[right_idx]["image"],
                    "similarity": sim[left_idx][right_idx],
                }
            )
    top_pairs.sort(key=lambda row: (-row["similarity"], row["image_left"], row["image_right"]))

    for item in image_summaries:
        image_path = output_root / "by_image" / f"{item['image_slug']}.json"
        _write_json(image_path, item)

    similarity_report = {
        "image_count": len(image_summaries),
        "feature_names": [
            "duration_sec",
            "cpu_avg_percent",
            "cpu_max_percent",
            "mem_avg_mb",
            "mem_max_mb",
            "disk_read_max_mb",
            "disk_write_max_mb",
        ],
        "top_similar_pairs": top_pairs[: max(20, neighbor_count * 4)],
        "images": [
            {
                "image": item["image"],
                "image_slug": item["image_slug"],
                "run_count": item["run_count"],
                "nearest_neighbors": item["nearest_neighbors"],
                "image_profile_path": item["image_profile_path"],
            }
            for item in image_summaries
        ],
    }
    _write_json(output_root / "image_similarity_summary.json", similarity_report)

    index = {
        "image_count": len(image_summaries),
        "images": [
            {
                "image": item["image"],
                "image_slug": item["image_slug"],
                "run_count": item["run_count"],
                "image_profile_path": item["image_profile_path"],
            }
            for item in image_summaries
        ],
    }
    _write_json(output_root / "by_image" / "index.json", index)

    return similarity_report


async def _profile_many(args: argparse.Namespace) -> dict[str, Any]:
    traj_paths = _collect_traj_paths(args.trajectory, limit=args.limit)
    output_root = Path(args.output_root) if args.output_root else _default_output_root(args.trajectory)
    per_run_dir = output_root / "per_run"
    per_run_dir.mkdir(parents=True, exist_ok=True)

    concurrency = max(1, int(args.max_concurrency))
    sem = asyncio.Semaphore(concurrency)
    reports: list[dict[str, Any]] = []

    async def _run_with_guard(traj_path: Path) -> dict[str, Any]:
        async with sem:
            started = time.time()
            try:
                profile = await _profile_one_trajectory(traj_path, args)
                profile["ok"] = profile.get("replay_ok", False)
            except Exception as exc:
                profile = {
                    "trajectory": str(traj_path.resolve()),
                    "ok": False,
                    "replay_ok": False,
                    "replay_error": f"{type(exc).__name__}: {exc}",
                    "image": "",
                    "instance_id": "",
                    "sample_count": 0,
                    "duration_sec": 0.0,
                }
            profile["wall_time_sec"] = time.time() - started
            return profile

    tasks = [asyncio.create_task(_run_with_guard(traj_path)) for traj_path in traj_paths]
    for task in asyncio.as_completed(tasks):
        profile = await task
        traj_name = Path(profile.get("trajectory", f"failed-{len(reports)}")).parent.name
        out_path = per_run_dir / f"{traj_name}.resource_profile.json"
        profile["_profile_path"] = str(out_path.resolve())
        _write_json(out_path, profile)
        reports.append(profile)

    successful_profiles = [item for item in reports if item.get("image")]
    similarity_report = _aggregate_profiles_by_image(
        successful_profiles,
        output_root=output_root,
        neighbor_count=max(1, int(args.neighbor_count)),
    )

    summary = {
        "mode": "replay_image_resource_profile",
        "trajectory_root": str(Path(args.trajectory).resolve()),
        "trajectory_count": len(traj_paths),
        "ok_count": sum(1 for item in reports if item.get("ok")),
        "failed_count": sum(1 for item in reports if not item.get("ok")),
        "max_concurrency": concurrency,
        "output_root": str(output_root.resolve()),
        "per_run_dir": str(per_run_dir.resolve()),
        "by_image_dir": str((output_root / "by_image").resolve()),
        "image_similarity_summary": str((output_root / "image_similarity_summary.json").resolve()),
        "top_similar_pairs": similarity_report.get("top_similar_pairs", [])[:10],
        "reports": [
            {
                "trajectory": item.get("trajectory"),
                "image": item.get("image"),
                "instance_id": item.get("instance_id"),
                "sample_count": item.get("sample_count"),
                "duration_sec": item.get("duration_sec"),
                "ok": item.get("ok"),
                "profile_path": item.get("_profile_path"),
            }
            for item in reports
        ],
    }
    _write_json(output_root / "batch_summary.json", summary)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Replay all SWE trajectories under a rollout directory, sample container resource usage, "
            "aggregate profiles by image, and analyze image similarity."
        )
    )
    parser.add_argument("trajectory", help="Path to a traj.json file or a rollout directory containing traj.json files")
    parser.add_argument("--base-url", default=os.getenv("SWE_ENV_SERVER_URL"), help="swe_env_pool_server base URL")
    parser.add_argument("--config-path", default=os.getenv("SWE_CONFIG_PATH"), help="Path to swebench.yaml")
    parser.add_argument("--data-source", default="swe-gym", help="Data source for docker image naming")
    parser.add_argument("--instance-id", default=None, help="Override instance_id from trajectory")
    parser.add_argument("--image-name", default=None, help="Override docker image name")
    parser.add_argument("--cwd", default=None, help="Working directory inside container")
    parser.add_argument("--exec-timeout", type=int, default=None, help="Per-command timeout in seconds")
    parser.add_argument(
        "--simulate-llm-delay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to sleep for each step's llm_elapsed before executing the environment command.",
    )
    parser.add_argument("--start-step", type=int, default=0, help="Replay from this step index")
    parser.add_argument("--end-step", type=int, default=None, help="Replay through this step index")
    parser.add_argument("--sample-interval-sec", type=float, default=DEFAULT_SAMPLE_INTERVAL_SEC, help="Container stats sampling interval in seconds")
    parser.add_argument("--max-concurrency", type=int, default=DEFAULT_MAX_CONCURRENCY, help="Maximum concurrent trajectory replays")
    parser.add_argument("--limit", type=int, default=None, help="Only replay the first N trajectories after sorting")
    parser.add_argument("--output-root", default=None, help="Directory for per-run profiles, per-image aggregates, and similarity reports")
    parser.add_argument("--output-json", default=None, help="Optional path to save the batch summary JSON")
    parser.add_argument("--neighbor-count", type=int, default=DEFAULT_NEIGHBOR_COUNT, help="Nearest similar images to record per image")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if not args.base_url:
        raise ValueError("--base-url is required")
    report = await _profile_many(args)
    text = json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
