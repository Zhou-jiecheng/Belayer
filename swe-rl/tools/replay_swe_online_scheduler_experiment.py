#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SWE_RL_ROOT = REPO_ROOT / "swe-rl"
SLIME_ROOT = REPO_ROOT / "slime"
for path in (SLIME_ROOT, SWE_RL_ROOT):
    sys.path.insert(0, str(path))

from generate_with_swe_remote import (  # noqa: E402
    TrajectoryReplayCompletionProvider,
    _get_diff_semaphore,
    _get_docker_create_limiter,
    _get_eval_semaphore,
    _get_online_env_docker_scheduler,
    _get_swe_semaphore,
    _get_sweagent_config,
    _run_agent_remote,
)
from online_env_docker_scheduler import OnlineEnvDockerScheduler  # noqa: E402
from swe_env_client import SweEnvClient  # noqa: E402
from swe_utils import get_docker_image_name  # noqa: E402


DEFAULT_TRAJECTORY_ROOT = (
    REPO_ROOT
    / "export"
    / "swe_rl_online_scheduler_adaptive_fast_restart_no_ckpt"
    / "swe_rollouts"
)


@dataclass(frozen=True)
class ReplaySpec:
    traj_path: Path
    meta_path: Path | None
    traj_label: str
    instance_id: str
    group_index: int
    sample_index: int
    sample_slot: int
    sequence_index: int
    instance: dict[str, Any]
    data_source: str


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _set_replay_seed(seed: int) -> None:
    random.seed(int(seed))
    os.environ.setdefault("SWE_REPLAY_SEED", str(int(seed)))
    os.environ.setdefault("SWE_SCHED_RANDOM_SEED", str(int(seed)))


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_int(value: Any, default: int = -1) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _optional_env_int(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or str(value).strip() == "":
        return None
    return int(value)


def _collect_traj_paths(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    return sorted(path for path in root.rglob("traj.json") if path.is_file())


def _parse_indices_from_name(name: str) -> tuple[int, int] | None:
    match = re.search(r"__g(?P<group>-?\d+)__i(?P<index>-?\d+)(?:__|$)", name)
    if not match:
        return None
    return int(match.group("group")), int(match.group("index"))


def _infer_source_n_samples_per_prompt(paths: list[Path], fallback: int) -> int:
    by_group: dict[int, set[int]] = defaultdict(set)
    for path in paths:
        indices = _parse_indices_from_name(path.parent.name)
        if indices is None:
            continue
        group_index, sample_index = indices
        if group_index >= 0 and sample_index >= 0:
            by_group[group_index].add(sample_index)

    fallback_n = max(1, int(fallback))
    if not by_group:
        return fallback_n

    count_guess = max((len(indices) for indices in by_group.values()), default=1)
    stride_guess = 0
    ordered_min_indices = sorted((group_index, min(indices)) for group_index, indices in by_group.items() if indices)
    for (prev_group, prev_min), (next_group, next_min) in zip(ordered_min_indices, ordered_min_indices[1:]):
        group_delta = next_group - prev_group
        sample_delta = next_min - prev_min
        if group_delta <= 0 or sample_delta <= 0:
            continue
        if sample_delta % group_delta == 0:
            stride = sample_delta // group_delta
            stride_guess = stride if stride_guess <= 0 else gcd(stride_guess, stride)

    inferred = max(1, count_guess)
    if stride_guess > 0:
        inferred = max(inferred, stride_guess)
    return inferred


def _fallback_problem_statement(traj_payload: dict[str, Any]) -> str:
    messages = traj_payload.get("messages", [])
    if isinstance(messages, list) and len(messages) > 1 and isinstance(messages[1], dict):
        content = str(messages[1].get("content", "") or "")
        return content or "Replay trajectory task"
    return "Replay trajectory task"


def _build_replay_spec(path: Path, *, sequence_index: int, n_samples_per_prompt: int) -> ReplaySpec:
    traj_payload = _load_json(path)
    info = traj_payload.get("info") if isinstance(traj_payload.get("info"), dict) else {}
    meta_path = path.parent / "meta.json"
    meta_payload = _load_json(meta_path) if meta_path.exists() else {}
    sample_metadata = meta_payload.get("sample_metadata") if isinstance(meta_payload.get("sample_metadata"), dict) else {}
    instance = sample_metadata.get("instance") if isinstance(sample_metadata.get("instance"), dict) else {}
    instance = dict(instance) if isinstance(instance, dict) else {}

    name_indices = _parse_indices_from_name(path.parent.name)
    group_index = _as_int(info.get("group_index"), -1)
    sample_index = _as_int(info.get("index"), -1)
    if group_index < 0 and name_indices is not None:
        group_index = name_indices[0]
    if sample_index < 0 and name_indices is not None:
        sample_index = name_indices[1]

    instance_id = str(info.get("instance_id") or instance.get("instance_id") or "").strip()
    if not instance_id:
        # Directory stems are normally <instance_id>__gN__iM__timestamp.
        marker = "__g"
        instance_id = path.parent.name.split(marker, 1)[0] if marker in path.parent.name else path.parent.name
    instance.setdefault("instance_id", instance_id)
    instance.setdefault("problem_statement", _fallback_problem_statement(traj_payload))
    instance.setdefault("eval_script", "")

    data_source = str(sample_metadata.get("data_source") or meta_payload.get("data_source") or "swe-gym")
    n = max(1, int(n_samples_per_prompt))
    if group_index >= 0 and sample_index >= 0:
        sample_slot = sample_index - group_index * n
        if sample_slot < 0:
            sample_slot = sample_index % n
    elif sample_index >= 0:
        sample_slot = sample_index % n
    else:
        sample_slot = sequence_index

    return ReplaySpec(
        traj_path=path,
        meta_path=meta_path if meta_path.exists() else None,
        traj_label=path.parent.name,
        instance_id=instance_id,
        group_index=group_index,
        sample_index=sample_index,
        sample_slot=sample_slot,
        sequence_index=sequence_index,
        instance=instance,
        data_source=data_source,
    )


def _order_specs(specs: list[ReplaySpec], mode: str) -> list[ReplaySpec]:
    if mode == "original":
        return list(specs)
    if mode == "sorted":
        return sorted(specs, key=lambda spec: str(spec.traj_path))
    if mode != "breadth-first":
        raise ValueError(f"unsupported reorder mode: {mode}")
    return sorted(
        specs,
        key=lambda spec: (
            spec.sample_slot,
            spec.group_index if spec.group_index >= 0 else 1 << 30,
            spec.sample_index if spec.sample_index >= 0 else 1 << 30,
            spec.sequence_index,
        ),
    )


def _batch_shape_filter(
    specs: list[ReplaySpec],
    *,
    rollout_batch_size: int,
    n_samples_per_prompt: int,
    enabled: bool,
) -> tuple[list[ReplaySpec], list[dict[str, Any]], dict[str, Any]]:
    target_groups = max(0, int(rollout_batch_size))
    target_samples = max(0, int(n_samples_per_prompt))
    target_count = target_groups * target_samples
    if not enabled:
        return (
            list(specs),
            [],
            {
                "enabled": False,
                "rollout_batch_size": target_groups,
                "n_samples_per_prompt": target_samples,
                "target_replay_count": target_count,
                "selected_group_indices": [],
                "filtered_out_count": 0,
                "selected_count_before_debug_limit": len(specs),
                "missing_count": max(0, target_count - len(specs)),
            },
        )

    group_indices = sorted({spec.group_index for spec in specs if spec.group_index >= 0})
    selected_groups = group_indices[:target_groups]
    selected_group_set = set(selected_groups)

    selected: list[ReplaySpec] = []
    filtered: list[dict[str, Any]] = []
    for spec in specs:
        reason = ""
        if spec.group_index < 0:
            reason = "missing_group_index"
        elif spec.group_index not in selected_group_set:
            reason = "outside_rollout_batch_size"
        elif spec.sample_slot < 0:
            reason = "missing_sample_slot"
        elif spec.sample_slot >= target_samples:
            reason = "outside_n_samples_per_prompt"

        if reason:
            filtered.append(
                {
                    "reason": reason,
                    "traj_label": spec.traj_label,
                    "traj_path": str(spec.traj_path.resolve()),
                    "instance_id": spec.instance_id,
                    "group_index": spec.group_index,
                    "index": spec.sample_index,
                    "sample_slot": spec.sample_slot,
                    "sequence_index": spec.sequence_index,
                    "data_source": spec.data_source,
                }
            )
        else:
            selected.append(spec)

    stats = {
        "enabled": True,
        "rollout_batch_size": target_groups,
        "n_samples_per_prompt": target_samples,
        "target_replay_count": target_count,
        "selected_group_indices": selected_groups,
        "available_group_count": len(group_indices),
        "selected_count_before_debug_limit": len(selected),
        "filtered_out_count": len(filtered),
        "missing_count": max(0, target_count - len(selected)),
    }
    return selected, filtered, stats


def _sample_slot_range_filter(
    specs: list[ReplaySpec],
    *,
    sample_slot_start: int | None,
    sample_slot_end: int | None,
) -> tuple[list[ReplaySpec], list[dict[str, Any]], dict[str, Any]]:
    if sample_slot_start is None and sample_slot_end is None:
        return (
            list(specs),
            [],
            {
                "enabled": False,
                "sample_slot_start": None,
                "sample_slot_end": None,
                "selected_count": len(specs),
                "filtered_out_count": 0,
            },
        )

    start = int(sample_slot_start) if sample_slot_start is not None else None
    end = int(sample_slot_end) if sample_slot_end is not None else None
    if start is not None and start < 0:
        raise ValueError(f"sample_slot_start must be >= 0, got {start}")
    if end is not None and end < 0:
        raise ValueError(f"sample_slot_end must be >= 0, got {end}")
    if start is not None and end is not None and end < start:
        raise ValueError(f"sample_slot_end must be >= sample_slot_start, got {end} < {start}")

    selected: list[ReplaySpec] = []
    filtered: list[dict[str, Any]] = []
    for spec in specs:
        reason = ""
        if spec.sample_slot < 0:
            reason = "missing_sample_slot"
        elif start is not None and spec.sample_slot < start:
            reason = "before_sample_slot_start"
        elif end is not None and spec.sample_slot >= end:
            reason = "after_sample_slot_end"

        if reason:
            filtered.append(
                {
                    "reason": reason,
                    "traj_label": spec.traj_label,
                    "traj_path": str(spec.traj_path.resolve()),
                    "instance_id": spec.instance_id,
                    "group_index": spec.group_index,
                    "index": spec.sample_index,
                    "sample_slot": spec.sample_slot,
                    "sequence_index": spec.sequence_index,
                    "data_source": spec.data_source,
                }
            )
        else:
            selected.append(spec)

    return (
        selected,
        filtered,
        {
            "enabled": True,
            "sample_slot_start": start,
            "sample_slot_end": end,
            "selected_count": len(selected),
            "filtered_out_count": len(filtered),
        },
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * float(q)) - 1))
    return ordered[idx]


def _summarize_values(values: list[float]) -> dict[str, float]:
    return {
        "count": float(len(values)),
        "sum": sum(values),
        "mean": statistics.mean(values) if values else 0.0,
        "p50": statistics.median(values) if values else 0.0,
        "p95": _percentile(values, 0.95),
        "max": max(values) if values else 0.0,
    }


def _snapshot_scheduler_env() -> dict[str, str]:
    prefixes = (
        "SWE_ENABLE_ONLINE_ENV_DOCKER_SCHEDULER",
        "SWE_SCHED_",
        "SWE_POOL_",
        "SWE_MAX_CONTAINERS_PER_NODE",
        "SWE_MAX_CONCURRENT_DOCKER_CREATE",
        "SWE_DOCKER_CREATE_MIN_INTERVAL_SEC",
        "SWE_RESOURCE_STATS_DIR",
        "SWE_REPO_RESOURCE_STATS_PATH",
        "SWE_ENV_SERVER_URL",
        "SWE_EXEC_SERVER_URLS",
    )
    return {
        key: value
        for key, value in sorted(os.environ.items())
        if any(key == prefix or key.startswith(prefix) for prefix in prefixes)
    }


async def _scheduler_state_sampler(
    scheduler: OnlineEnvDockerScheduler | None,
    *,
    interval_sec: float,
    stop_event: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    if scheduler is None or interval_sec <= 0.0:
        return
    while not stop_event.is_set():
        try:
            active_prompts = getattr(scheduler, "_active_prompts", {})
            pending = getattr(scheduler, "_pending", [])
            active_predicted = getattr(scheduler, "_active_predicted", None)
            budget = scheduler._effective_budget()
            samples.append(
                {
                    "ts": time.time(),
                    "active_prompt_count": len(active_prompts),
                    "pending_count": len(pending),
                    "active_predicted_memory_bytes": float(getattr(active_predicted, "memory_bytes", 0.0)),
                    "active_predicted_cpu_percent": float(getattr(active_predicted, "cpu_percent", 0.0)),
                    "active_predicted_disk_read_bytes": float(getattr(active_predicted, "disk_read_bytes", 0.0)),
                    "active_predicted_disk_write_bytes": float(getattr(active_predicted, "disk_write_bytes", 0.0)),
                    "budget_memory_bytes": float(getattr(budget, "memory_bytes", 0.0)),
                    "budget_cpu_percent": float(getattr(budget, "cpu_percent", 0.0)),
                    "budget_disk_read_bytes": float(getattr(budget, "disk_read_bytes", 0.0)),
                    "budget_disk_write_bytes": float(getattr(budget, "disk_write_bytes", 0.0)),
                }
            )
        except Exception as exc:
            samples.append({"ts": time.time(), "error": f"{type(exc).__name__}: {exc}"})
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval_sec)
        except asyncio.TimeoutError:
            pass


def _scheduler_budget_snapshot(scheduler: OnlineEnvDockerScheduler) -> dict[str, Any]:
    budget = scheduler._effective_budget()
    return {
        "ts": time.time(),
        "memory_bytes": float(getattr(budget, "memory_bytes", 0.0)),
        "cpu_percent": float(getattr(budget, "cpu_percent", 0.0)),
        "disk_read_bytes": float(getattr(budget, "disk_read_bytes", 0.0)),
        "disk_write_bytes": float(getattr(budget, "disk_write_bytes", 0.0)),
        "server_memory_available_bytes": float(getattr(scheduler, "_server_memory_available_bytes", 0.0) or 0.0),
        "server_cpu_available_percent": float(getattr(scheduler, "_server_cpu_available_percent", 0.0) or 0.0),
        "server_disk_read_available_bps": float(getattr(scheduler, "_server_disk_read_available_bps", 0.0) or 0.0),
        "server_disk_write_available_bps": float(getattr(scheduler, "_server_disk_write_available_bps", 0.0) or 0.0),
    }


def _scheduler_budget_ready(scheduler: OnlineEnvDockerScheduler) -> bool:
    budget = scheduler._effective_budget()
    config = scheduler.config
    if config.use_realtime_server_memory and float(getattr(budget, "memory_bytes", 0.0)) <= 0.0:
        return False
    if config.use_realtime_server_cpu and float(getattr(budget, "cpu_percent", 0.0)) <= 0.0:
        return False
    if config.use_realtime_server_disk:
        if float(getattr(budget, "disk_read_bytes", 0.0)) <= 0.0:
            return False
        if float(getattr(budget, "disk_write_bytes", 0.0)) <= 0.0:
            return False
    return True


async def _force_scheduler_resource_refresh(
    scheduler: OnlineEnvDockerScheduler,
    *,
    samples: list[dict[str, Any]],
) -> bool:
    refreshed = False
    error: str | None = None
    try:
        refreshed = bool(await scheduler._refresh_server_memory_budget_if_needed(force=True))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    snapshot = _scheduler_budget_snapshot(scheduler)
    snapshot["refreshed"] = refreshed
    snapshot["ready"] = _scheduler_budget_ready(scheduler)
    if error is not None:
        snapshot["error"] = error
    samples.append(snapshot)

    cond = getattr(scheduler, "_cond", None)
    if cond is not None:
        async with cond:
            if refreshed:
                scheduler._reset_admission_window_unlocked()
            cond.notify_all()
    return bool(snapshot["ready"])


async def _wait_for_scheduler_initial_budget(
    scheduler: OnlineEnvDockerScheduler | None,
    *,
    timeout_sec: float,
    interval_sec: float,
    allow_zero_budget: bool,
    samples: list[dict[str, Any]],
) -> None:
    if scheduler is None:
        return
    deadline = time.time() + max(0.0, float(timeout_sec))
    interval = max(0.1, float(interval_sec))
    while True:
        ready = await _force_scheduler_resource_refresh(scheduler, samples=samples)
        if ready:
            snapshot = samples[-1]
            print(
                "[scheduler-replay] scheduler budget ready "
                f"mem={snapshot['memory_bytes']:.0f} cpu={snapshot['cpu_percent']:.1f}% "
                f"disk_r={snapshot['disk_read_bytes']:.0f} disk_w={snapshot['disk_write_bytes']:.0f}",
                flush=True,
            )
            return
        if allow_zero_budget:
            print("[scheduler-replay] WARNING: continuing with zero scheduler budget", flush=True)
            return
        if time.time() >= deadline:
            raise RuntimeError(
                "scheduler realtime resource budget stayed zero. "
                f"Last snapshot: {samples[-1] if samples else {}}. "
                "Check SWE_ENV_SERVER_URL, swe_env_pool_server /status, and exec server health."
            )
        await asyncio.sleep(interval)


async def _scheduler_resource_refresh_loop(
    scheduler: OnlineEnvDockerScheduler | None,
    *,
    interval_sec: float,
    stop_event: asyncio.Event,
    samples: list[dict[str, Any]],
) -> None:
    if scheduler is None or interval_sec <= 0.0:
        return
    interval = max(0.1, float(interval_sec))
    while not stop_event.is_set():
        await _force_scheduler_resource_refresh(scheduler, samples=samples)
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def _run_one_spec(
    spec: ReplaySpec,
    args: argparse.Namespace,
    *,
    scheduler: OnlineEnvDockerScheduler | None,
) -> dict[str, Any]:
    started_at = time.time()
    env_client = SweEnvClient(base_url=args.base_url)
    sweagent_config = _get_sweagent_config()
    model_config = dict(sweagent_config.get("model", {}) if isinstance(sweagent_config, dict) else {})
    model_config["model_name"] = model_config.get("model_name") or "replay/trajectory"
    model_config.setdefault("model_kwargs", {})
    image_name = args.image_name or get_docker_image_name(spec.instance, spec.data_source)
    sample = SimpleNamespace(
        metadata={
            "instance": spec.instance,
            "data_source": spec.data_source,
        },
        group_index=spec.group_index,
        index=spec.sample_index,
        prompt=None,
    )

    admission_ticket = None
    lease_id: str | None = None
    eval_lease_id: str | None = None
    eval_slot_acquired = False
    swe_slot_acquired = False
    metrics: dict[str, float] = {}
    run_info: dict[str, Any] = {
        "messages": [],
        "step_debug": [],
        "reward": 0,
        "error": None,
        "git_patch": None,
        "patch_source": None,
        "exit_status": None,
        "n_steps": 0,
        "eval_result": None,
        "phase_events": [],
    }

    try:
        if scheduler is not None:
            t0 = time.time()
            admission_ticket = await scheduler.admit_prompt(
                sample=sample,
                image_name=image_name,
                rollout_batch_size=max(1, int(args.rollout_batch_size)),
            )
            metrics["admission_wait_sec"] = time.time() - t0
        else:
            metrics["admission_wait_sec"] = 0.0

        swe_semaphore = _get_swe_semaphore()
        t0 = time.time()
        await swe_semaphore.acquire()
        swe_slot_acquired = True
        metrics["semaphore_wait_sec"] = time.time() - t0

        docker_create_limiter = _get_docker_create_limiter()
        t0 = time.time()
        lease = await docker_create_limiter.allocate(
            env_client,
            image=image_name,
            instance_id=spec.instance_id,
        )
        metrics["allocate_wait_sec"] = time.time() - t0
        lease_id = str(lease["lease_id"])

        if scheduler is not None and admission_ticket is not None:
            await scheduler.attach_lease(prompt_id=admission_ticket.prompt_id, lease_id=lease_id)

        provider = TrajectoryReplayCompletionProvider.from_path(
            spec.traj_path,
            llm_delay_scale=args.llm_delay_scale,
            min_delay_sec=args.min_llm_delay_sec,
            max_delay_sec=args.max_llm_delay_sec,
            strict_action_match=args.strict_action_match,
        )
        t0 = time.time()
        run_info.update(
            await _run_agent_remote(
                env_client,
                lease_id,
                spec.instance,
                model_config.get("model_name", "replay/trajectory"),
                model_config,
                sweagent_config,
                args=SimpleNamespace(prm_enable=False),
                prm_agent=None,
                tokenizer=None,
                cm_max_input_tokens=None,
                policy="never",
                tail_model=None,
                fault_injection_armed=False,
                fault_injection_probability=0.0,
                docker_create_limiter=docker_create_limiter,
                image_name=image_name,
                replay_completion_provider=provider,
            )
        )
        metrics["agent_runtime_sec"] = time.time() - t0

        git_patch = run_info.get("git_patch")
        if git_patch and not args.skip_eval:
            try:
                if lease_id is not None:
                    await env_client.close(lease_id)
                    if scheduler is not None:
                        await scheduler.detach_lease(lease_id, reason="replay_agent_closed_before_eval")
                    lease_id = None

                eval_wait_started_at = time.time()
                eval_semaphore = _get_eval_semaphore()
                await eval_semaphore.acquire()
                eval_slot_acquired = True
                metrics["eval_slot_wait_sec"] = time.time() - eval_wait_started_at

                t0 = time.time()
                eval_lease = await docker_create_limiter.allocate(
                    env_client,
                    image=image_name,
                    instance_id=f"{spec.instance_id}__eval",
                )
                metrics["eval_allocate_wait_sec"] = time.time() - t0
                eval_lease_id = str(eval_lease["lease_id"])
                if scheduler is not None and admission_ticket is not None:
                    await scheduler.attach_lease(prompt_id=admission_ticket.prompt_id, lease_id=eval_lease_id)

                t0 = time.time()
                eval_timeout = float(args.eval_timeout)
                eval_wait_timeout = float(os.getenv("SWE_EVALUATE_WAIT_TIMEOUT_SEC", "0") or "0")
                if eval_wait_timeout <= 0:
                    eval_wait_timeout = eval_timeout + 60.0
                run_info["eval_result"] = await asyncio.wait_for(
                    env_client.evaluate(
                        lease_id=eval_lease_id,
                        patch=str(git_patch),
                        eval_script=str(spec.instance.get("eval_script", "") or ""),
                        timeout=eval_timeout,
                    ),
                    timeout=eval_wait_timeout,
                )
                metrics["eval_runtime_sec"] = time.time() - t0
                run_info["reward"] = int(bool(run_info["eval_result"].get("resolved", False)))
            except Exception as exc:
                run_info["error"] = run_info.get("error") or f"eval failed: {type(exc).__name__}: {exc}"
                metrics["eval_error"] = 1.0
        else:
            metrics["eval_skipped"] = 1.0

        ok = run_info.get("error") is None
        return {
            "ok": bool(ok),
            "traj_label": spec.traj_label,
            "traj_path": str(spec.traj_path.resolve()),
            "instance_id": spec.instance_id,
            "group_index": spec.group_index,
            "index": spec.sample_index,
            "sample_slot": spec.sample_slot,
            "image_name": image_name,
            "metrics": metrics,
            "run_info": run_info,
        }
    except Exception as exc:
        return {
            "ok": False,
            "traj_label": spec.traj_label,
            "traj_path": str(spec.traj_path.resolve()),
            "instance_id": spec.instance_id,
            "group_index": spec.group_index,
            "index": spec.sample_index,
            "sample_slot": spec.sample_slot,
            "image_name": image_name,
            "metrics": metrics,
            "error": f"{type(exc).__name__}: {exc}",
            "run_info": run_info,
        }
    finally:
        if eval_lease_id is not None:
            try:
                await env_client.close(eval_lease_id)
                if scheduler is not None:
                    await scheduler.detach_lease(eval_lease_id, reason="replay_eval_closed")
            except BaseException:
                pass
        if eval_slot_acquired:
            try:
                _get_eval_semaphore().release()
            except BaseException:
                pass
        if lease_id is not None:
            try:
                await env_client.close(lease_id)
                if scheduler is not None:
                    await scheduler.detach_lease(lease_id, reason="replay_agent_closed")
            except BaseException:
                pass
        if swe_slot_acquired:
            try:
                _get_swe_semaphore().release()
            except BaseException:
                pass
        if scheduler is not None and admission_ticket is not None:
            try:
                await scheduler.finish_prompt(admission_ticket.prompt_id)
            except Exception:
                pass
        metrics["total_wall_sec"] = time.time() - started_at


async def _run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    os.environ["SWE_CHECKPOINT_POLICY"] = "never"
    os.environ["SWE_FAULT_INJECTION_ENABLE"] = "0"
    os.environ["SWE_ENV_SERVER_URL"] = args.base_url

    traj_root = Path(args.trajectory_root)
    source_paths = _collect_traj_paths(traj_root)
    source_n_samples_per_prompt = max(
        1,
        int(
            args.source_n_samples_per_prompt
            if args.source_n_samples_per_prompt is not None
            else _infer_source_n_samples_per_prompt(source_paths, args.n_samples_per_prompt)
        ),
    )
    specs: list[ReplaySpec] = []
    skipped_specs: list[dict[str, Any]] = []
    for idx, path in enumerate(source_paths):
        try:
            specs.append(
                _build_replay_spec(path, sequence_index=idx, n_samples_per_prompt=source_n_samples_per_prompt)
            )
        except Exception as exc:
            skipped_specs.append(
                {
                    "traj_path": str(path.resolve()),
                    "sequence_index": idx,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
    batch_filtered_specs, batch_filtered_out, batch_shape_stats = _batch_shape_filter(
        specs,
        rollout_batch_size=args.rollout_batch_size,
        n_samples_per_prompt=args.n_samples_per_prompt,
        enabled=not bool(args.disable_batch_shape_filter),
    )
    slot_filtered_specs, sample_slot_filtered_out, sample_slot_stats = _sample_slot_range_filter(
        batch_filtered_specs,
        sample_slot_start=args.sample_slot_start,
        sample_slot_end=args.sample_slot_end,
    )
    ordered_specs = _order_specs(slot_filtered_specs, args.reorder_mode)
    pre_debug_limit_count = len(ordered_specs)
    if args.limit is not None:
        ordered_specs = ordered_specs[: max(0, int(args.limit))]
    debug_limit_applied = args.limit is not None and len(ordered_specs) < pre_debug_limit_count

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or (REPO_ROOT / "export" / f"swe_online_scheduler_replay_{ts}"))
    output_root.mkdir(parents=True, exist_ok=True)
    per_traj_dir = output_root / "per_traj"
    per_traj_dir.mkdir(parents=True, exist_ok=True)

    manifest = [
        {
            "order": order,
            "traj_label": spec.traj_label,
            "traj_path": str(spec.traj_path.resolve()),
            "instance_id": spec.instance_id,
            "group_index": spec.group_index,
            "index": spec.sample_index,
            "sample_slot": spec.sample_slot,
            "sequence_index": spec.sequence_index,
            "data_source": spec.data_source,
        }
        for order, spec in enumerate(ordered_specs)
    ]
    (output_root / "reordered_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_root / "skipped_trajectories.json").write_text(
        json.dumps(skipped_specs, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_root / "batch_shape_filtered_out.json").write_text(
        json.dumps(batch_filtered_out, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_root / "sample_slot_filtered_out.json").write_text(
        json.dumps(sample_slot_filtered_out, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    (output_root / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "scheduler_env": _snapshot_scheduler_env(),
                "source_n_samples_per_prompt": source_n_samples_per_prompt,
                "batch_shape": batch_shape_stats,
                "sample_slot_filter": sample_slot_stats,
                "trajectory_count_before_debug_limit": pre_debug_limit_count,
                "debug_limit_applied": debug_limit_applied,
                "trajectory_count": len(ordered_specs),
                "source_trajectory_count": len(source_paths),
                "valid_trajectory_count": len(specs),
                "skipped_trajectory_count": len(skipped_specs),
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )

    scheduler = _get_online_env_docker_scheduler()
    scheduler_resource_samples: list[dict[str, Any]] = []
    await _wait_for_scheduler_initial_budget(
        scheduler,
        timeout_sec=float(args.scheduler_resource_warmup_timeout_sec),
        interval_sec=float(args.scheduler_resource_refresh_sec),
        allow_zero_budget=bool(args.allow_zero_scheduler_budget),
        samples=scheduler_resource_samples,
    )
    stop_resource_refresher = asyncio.Event()
    resource_refresher_task = asyncio.create_task(
        _scheduler_resource_refresh_loop(
            scheduler,
            interval_sec=float(args.scheduler_resource_refresh_sec),
            stop_event=stop_resource_refresher,
            samples=scheduler_resource_samples,
        )
    )
    stop_sampler = asyncio.Event()
    scheduler_samples: list[dict[str, Any]] = []
    sampler_task = asyncio.create_task(
        _scheduler_state_sampler(
            scheduler,
            interval_sec=float(args.scheduler_sample_interval_sec),
            stop_event=stop_sampler,
            samples=scheduler_samples,
        )
    )

    sem = asyncio.Semaphore(max(1, int(args.max_inflight)))
    reports: list[dict[str, Any]] = []
    batch_started_at = time.time()

    async def _guard(spec: ReplaySpec) -> dict[str, Any]:
        async with sem:
            return await _run_one_spec(spec, args, scheduler=scheduler)

    tasks = [asyncio.create_task(_guard(spec)) for spec in ordered_specs]
    for task in asyncio.as_completed(tasks):
        report = await task
        reports.append(report)
        out_path = per_traj_dir / f"{report.get('traj_label', f'failed-{len(reports)}')}.json"
        out_path.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )

    stop_sampler.set()
    stop_resource_refresher.set()
    with contextlib_suppress_cancelled():
        await sampler_task
    with contextlib_suppress_cancelled():
        await resource_refresher_task
    if scheduler is not None:
        await scheduler.close()

    metric_names = [
        "admission_wait_sec",
        "semaphore_wait_sec",
        "allocate_wait_sec",
        "agent_runtime_sec",
        "eval_slot_wait_sec",
        "eval_allocate_wait_sec",
        "eval_runtime_sec",
        "total_wall_sec",
    ]
    metrics_summary = {
        name: _summarize_values(
            [
                float(report.get("metrics", {}).get(name, 0.0) or 0.0)
                for report in reports
                if name in report.get("metrics", {})
            ]
        )
        for name in metric_names
    }
    summary = {
        "mode": "swe_online_scheduler_replay",
        "trajectory_root": str(traj_root.resolve()),
        "output_root": str(output_root.resolve()),
        "source_trajectory_count": len(source_paths),
        "valid_trajectory_count": len(specs),
        "skipped_trajectory_count": len(skipped_specs),
        "source_n_samples_per_prompt": source_n_samples_per_prompt,
        "batch_shape": batch_shape_stats,
        "sample_slot_filter": sample_slot_stats,
        "trajectory_count_before_debug_limit": pre_debug_limit_count,
        "debug_limit": args.limit,
        "debug_limit_applied": debug_limit_applied,
        "trajectory_count": len(ordered_specs),
        "ok_count": sum(1 for report in reports if report.get("ok")),
        "failed_count": sum(1 for report in reports if not report.get("ok")),
        "batch_wall_time_sec": time.time() - batch_started_at,
        "reorder_mode": args.reorder_mode,
        "rollout_batch_size": args.rollout_batch_size,
        "n_samples_per_prompt": args.n_samples_per_prompt,
        "llm_delay_scale": args.llm_delay_scale,
        "skip_eval": bool(args.skip_eval),
        "metrics": metrics_summary,
        "eval_count": sum(1 for report in reports if report.get("run_info", {}).get("eval_result") is not None),
        "diff_fallback_count": sum(
            1 for report in reports if report.get("run_info", {}).get("patch_source") == "git_diff_fallback"
        ),
        "submission_patch_count": sum(
            1 for report in reports if report.get("run_info", {}).get("patch_source") == "submission"
        ),
        "scheduler_samples": {
            "count": len(scheduler_samples),
            "peak_active_prompt_count": max(
                [int(item.get("active_prompt_count", 0) or 0) for item in scheduler_samples] or [0]
            ),
            "peak_pending_count": max(
                [int(item.get("pending_count", 0) or 0) for item in scheduler_samples] or [0]
            ),
            "peak_active_predicted_memory_bytes": max(
                [float(item.get("active_predicted_memory_bytes", 0.0) or 0.0) for item in scheduler_samples] or [0.0]
            ),
            "peak_active_predicted_cpu_percent": max(
                [float(item.get("active_predicted_cpu_percent", 0.0) or 0.0) for item in scheduler_samples] or [0.0]
            ),
        },
        "scheduler_resource_refresh": {
            "count": len(scheduler_resource_samples),
            "last": scheduler_resource_samples[-1] if scheduler_resource_samples else None,
            "zero_budget_count": sum(
                1
                for item in scheduler_resource_samples
                if not bool(item.get("ready", False))
            ),
        },
        "scheduler_profile_entry_count": (
            len(scheduler.get_repo_resource_stats()) if scheduler is not None else 0
        ),
        "scheduler_env": _snapshot_scheduler_env(),
    }
    (output_root / "scheduler_state_samples.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, default=_json_default) + "\n" for item in scheduler_samples),
        encoding="utf-8",
    )
    (output_root / "scheduler_resource_refresh_samples.jsonl").write_text(
        "".join(
            json.dumps(item, ensure_ascii=False, default=_json_default) + "\n"
            for item in scheduler_resource_samples
        ),
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps({"summary": summary, "reports": reports}, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    return summary


class contextlib_suppress_cancelled:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: Any, exc: BaseException | None, tb: Any) -> bool:
        return exc_type is asyncio.CancelledError


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay saved SWE trajectories through the real online env docker scheduler."
    )
    parser.add_argument("trajectory_root", nargs="?", default=str(DEFAULT_TRAJECTORY_ROOT))
    parser.add_argument("--base-url", default=os.getenv("SWE_ENV_SERVER_URL", "http://127.0.0.1:18090"))
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Debug-only cap applied after rollout-batch-size/n-samples-per-prompt selection and reordering.",
    )
    parser.add_argument("--max-inflight", type=int, default=int(os.getenv("SWE_SCHED_INTERNAL_MAX_INFLIGHT", "4096")))
    parser.add_argument("--rollout-batch-size", type=int, default=int(os.getenv("ROLLOUT_BATCH_SIZE", "64")))
    parser.add_argument("--n-samples-per-prompt", type=int, default=int(os.getenv("N_SAMPLES_PER_PROMPT", "8")))
    parser.add_argument(
        "--sample-slot-start",
        type=int,
        default=_optional_env_int("SWE_REPLAY_SAMPLE_SLOT_START"),
        help="Inclusive sample_slot lower bound after batch-shape selection.",
    )
    parser.add_argument(
        "--sample-slot-end",
        type=int,
        default=_optional_env_int("SWE_REPLAY_SAMPLE_SLOT_END"),
        help="Exclusive sample_slot upper bound after batch-shape selection.",
    )
    parser.add_argument("--seed", type=int, default=int(os.getenv("SWE_REPLAY_SEED", "0")))
    parser.add_argument(
        "--source-n-samples-per-prompt",
        type=int,
        default=(
            int(os.environ["SWE_REPLAY_SOURCE_N_SAMPLES_PER_PROMPT"])
            if os.getenv("SWE_REPLAY_SOURCE_N_SAMPLES_PER_PROMPT")
            else None
        ),
        help="Samples per prompt used when the saved trajectories were generated. Defaults to inference from filenames.",
    )
    parser.add_argument(
        "--disable-batch-shape-filter",
        action="store_true",
        default=os.getenv("SWE_REPLAY_DISABLE_BATCH_SHAPE_FILTER", "0").lower() in {"1", "true", "yes", "on"},
        help="Replay every discovered trajectory before the optional debug --limit.",
    )
    parser.add_argument("--reorder-mode", choices=["breadth-first", "original", "sorted"], default="breadth-first")
    parser.add_argument("--llm-delay-scale", type=float, default=float(os.getenv("SWE_REPLAY_LLM_DELAY_SCALE", "1.0")))
    parser.add_argument("--min-llm-delay-sec", type=float, default=0.0)
    parser.add_argument("--max-llm-delay-sec", type=float, default=None)
    parser.add_argument("--strict-action-match", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-timeout", type=float, default=float(os.getenv("SWE_EVAL_TIMEOUT", "2400")))
    parser.add_argument("--image-name", default=None)
    parser.add_argument("--scheduler-sample-interval-sec", type=float, default=0.5)
    parser.add_argument(
        "--scheduler-resource-refresh-sec",
        type=float,
        default=float(os.getenv("SWE_REPLAY_SCHED_RESOURCE_REFRESH_SEC", "20.0")),
        help="Force-refresh scheduler realtime server resources at this interval.",
    )
    parser.add_argument(
        "--scheduler-resource-warmup-timeout-sec",
        type=float,
        default=float(os.getenv("SWE_REPLAY_SCHED_RESOURCE_WARMUP_TIMEOUT_SEC", "60")),
        help="Wait this long for a non-zero scheduler realtime budget before submitting work.",
    )
    parser.add_argument(
        "--allow-zero-scheduler-budget",
        action="store_true",
        help="Submit work even if realtime scheduler budget is still zero after the first refresh.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    _set_replay_seed(int(args.seed))
    asyncio.run(_run_experiment(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
