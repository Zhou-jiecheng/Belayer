from __future__ import annotations

import argparse
import asyncio
import bisect
import hashlib
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_ROOT = Path(__file__).resolve().parent
SWE_RL_ROOT = REPO_ROOT / "swe-rl"
for path in (TOOLS_ROOT, SWE_RL_ROOT):
    sys.path.insert(0, str(path))

from checkpoint_policy_runtime import (
    POLICIES,
    AdaptiveTailModel,
    DEFAULT_ADAPTIVE_BUDGET_SEC,
    DEFAULT_ADAPTIVE_DECISION_INTERVAL_SEC,
    DEFAULT_ADAPTIVE_FAILURE_PROB,
    DEFAULT_ADAPTIVE_TAIL_ROOT,
    DEFAULT_ADAPTIVE_MIN_DELTA_ENV_COST_SEC,
    DEFAULT_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS,
    adaptive_delta_env_cost_sec as _adaptive_delta_env_cost_sec,
    adaptive_delta_replay_cost_sec as _adaptive_delta_replay_cost_sec,
    adaptive_expected_benefit_sec as _adaptive_expected_benefit_sec,
    redo_replay_cost_sec as _redo_replay_cost_sec,
    should_probe_in_llm_bubble as _shared_should_probe_in_llm_bubble,
)
from replay_swe_traj_checkpoint import (
    ReplayEnvClient,
    ReplayOpError,
    ReplayStep,
    _deferred_batch_gc_result,
    _maybe_run_checkpoint_gc,
    _build_image_name,
    _collect_traj_paths,
    _default_instance_id,
    _is_checkpoint_busy_error,
    _json_default,
    _load_traj_steps,
    _load_yaml_config,
)
logger = logging.getLogger("replay_swe_checkpoint_fault_experiment")


@dataclass(frozen=True)
class ReplayTrajectory:
    source_path: Path
    logical_index: int
    cycle_index: int
    sequence_index: int

    @property
    def key(self) -> str:
        return f"{self.source_path.resolve()}::logical={self.logical_index}"

    @property
    def report_name(self) -> str:
        base_name = self.source_path.parent.name
        if self.cycle_index <= 0:
            return base_name
        return f"{base_name}__repeat{self.cycle_index}"


@dataclass
class InjectionTarget:
    traj_key: str
    traj_path: str
    traj_label: str
    inject_before_step_indices: list[int]


ADAPTIVE_MIN_DELTA_ENV_COST_SEC = DEFAULT_ADAPTIVE_MIN_DELTA_ENV_COST_SEC
ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS = DEFAULT_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS


def _should_probe_in_llm_bubble(
    *,
    current_step_idx: int,
    probe_attempted_in_bubble: bool,
    adaptive_checkpoint_submitted: bool,
    pending_checkpoints: list[dict[str, Any]],
    delta_env_cost_sec: float,
    steps_since_latest_ready_checkpoint: int,
    expected_benefit_sec: float,
    expected_overhead_sec: float,
) -> bool:
    return _shared_should_probe_in_llm_bubble(
        current_step_idx=current_step_idx,
        probe_attempted_in_bubble=probe_attempted_in_bubble,
        adaptive_checkpoint_submitted=adaptive_checkpoint_submitted,
        pending_checkpoints=pending_checkpoints,
        delta_env_cost_sec_value=delta_env_cost_sec,
        steps_since_latest_ready_checkpoint=steps_since_latest_ready_checkpoint,
        expected_benefit_sec=expected_benefit_sec,
        expected_overhead_sec=expected_overhead_sec,
        min_delta_env_cost_sec=ADAPTIVE_MIN_DELTA_ENV_COST_SEC,
        min_steps_between_checkpoints=ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS,
    )


def _load_adaptive_tail_model(root_path: str | None, budget_sec: float) -> AdaptiveTailModel:
    root = Path(root_path) if root_path else DEFAULT_ADAPTIVE_TAIL_ROOT
    if not root.exists():
        raise FileNotFoundError(f"adaptive tail root does not exist: {root}")
    waits: list[float] = []
    for traj_path in _collect_traj_paths(str(root)):
        _, steps = _load_traj_steps(str(traj_path))
        waits.extend(step.llm_elapsed for step in steps if step.llm_elapsed > 0.0)
    if not waits:
        raise ValueError(f"no positive llm_elapsed values found under adaptive tail root: {root}")
    return AdaptiveTailModel.from_waits(waits, budget_sec=budget_sec)


async def _wait_for_global_gc_drain(
    *,
    base_url: str,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    env_client = ReplayEnvClient(base_url=base_url)
    return await env_client.checkpoint_gc_drain(
        timeout_sec=timeout_sec,
        poll_interval_sec=poll_interval_sec,
    )


def _trajectory_injection_rng(seed: int, traj_path: Path) -> random.Random:
    key = f"{seed}:{traj_path.resolve()}".encode("utf-8")
    digest = hashlib.sha256(key).hexdigest()
    return random.Random(int(digest[:16], 16))


def _expand_traj_paths(
    traj_paths: list[Path],
    limit: int | None,
) -> list[ReplayTrajectory]:
    if limit is None:
        target_count = len(traj_paths)
    else:
        target_count = max(0, int(limit))
    if target_count == 0 or not traj_paths:
        return []

    expanded: list[ReplayTrajectory] = []
    source_count = len(traj_paths)
    for logical_index in range(target_count):
        sequence_index = logical_index % source_count
        cycle_index = logical_index // source_count
        expanded.append(
            ReplayTrajectory(
                source_path=traj_paths[sequence_index],
                logical_index=logical_index,
                cycle_index=cycle_index,
                sequence_index=sequence_index,
            )
        )
    return expanded


def _build_probability_injection_plan(
    replay_trajectories: list[ReplayTrajectory],
    seed: int,
    injection_probability: float,
) -> list[InjectionTarget]:
    if not 0.0 <= float(injection_probability) <= 1.0:
        raise ValueError(f"injection_probability must be within [0, 1], got {injection_probability}")

    targets: list[InjectionTarget] = []
    for replay_traj in replay_trajectories:
        _, steps = _load_traj_steps(str(replay_traj.source_path))
        if len(steps) < 2:
            continue
        rng = _trajectory_injection_rng(seed, replay_traj.source_path)
        inject_before_step_indices = [
            step_idx
            for step_idx in range(1, len(steps))
            if rng.random() < float(injection_probability)
        ]
        if inject_before_step_indices:
            targets.append(
                InjectionTarget(
                    traj_key=replay_traj.key,
                    traj_path=str(replay_traj.source_path.resolve()),
                    traj_label=replay_traj.report_name,
                    inject_before_step_indices=inject_before_step_indices,
                )
            )
    return targets


def _promote_latest_ready_checkpoint(
    *,
    latest_ready_checkpoint_id: str | None,
    latest_ready_checkpoint_step: int,
    latest_ready_resume_step_idx: int,
    latest_ready_protected_env_cost_sec: float,
    latest_ready_protected_llm_cost_sec: float,
    checkpoint_id: str,
    step_idx: int,
    resume_step_idx: int,
    ready_at: float,
    protected_env_cost_sec: float,
    protected_llm_cost_sec: float,
) -> tuple[str | None, int, int, float, float, bool]:
    current_best = (
        latest_ready_checkpoint_step,
        -1.0 if latest_ready_checkpoint_id is None else ready_at,
    )
    candidate = (step_idx, ready_at)
    if latest_ready_checkpoint_id is not None and candidate < current_best:
        return (
            latest_ready_checkpoint_id,
            latest_ready_checkpoint_step,
            latest_ready_resume_step_idx,
            latest_ready_protected_env_cost_sec,
            latest_ready_protected_llm_cost_sec,
            False,
        )
    return (
        checkpoint_id,
        step_idx,
        resume_step_idx,
        protected_env_cost_sec,
        protected_llm_cost_sec,
        True,
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = max(0, min(len(ordered) - 1, int(len(ordered) * float(q)) - 1))
    return ordered[idx]


def _int_or_zero(value: Any) -> int:
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _annotate_checkpoint_overhead(
    checkpoint_event: dict[str, Any],
    *,
    overlap_budget_sec: float,
    overlap_source: str,
) -> None:
    elapsed_sec = float(checkpoint_event.get("create_call_elapsed_sec", 0.0) or 0.0)
    overlap_budget = max(0.0, float(overlap_budget_sec or 0.0))
    overlapped_sec = min(elapsed_sec, overlap_budget)
    critical_path_sec = max(0.0, elapsed_sec - overlap_budget)
    create_result = checkpoint_event.get("create_result") or {}
    status_result = checkpoint_event.get("status_result") or {}
    size_bytes = _int_or_zero(
        create_result.get("size_bytes")
        if isinstance(create_result, dict)
        else None
    )
    if size_bytes <= 0 and isinstance(status_result, dict):
        size_bytes = _int_or_zero(status_result.get("size_bytes"))
    checkpoint_event.update(
        {
            "checkpoint_elapsed_sec": elapsed_sec,
            "checkpoint_size_bytes": size_bytes,
            "overlap_budget_sec": overlap_budget,
            "overlap_source": overlap_source,
            "overlapped_checkpoint_sec": overlapped_sec,
            "critical_path_overhead_sec": critical_path_sec,
        }
    )


def _successful_checkpoint_overhead_events(reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for report in reports:
        for event in report.get("checkpoint_events", []) or []:
            if not isinstance(event, dict) or event.get("skipped", False):
                continue
            create_result = event.get("create_result") or {}
            if not isinstance(create_result, dict) or not create_result.get("ok", False):
                continue
            events.append(event)
    return events


def _summarize_checkpoint_overhead_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    elapsed = [float(item.get("checkpoint_elapsed_sec", item.get("create_call_elapsed_sec", 0.0)) or 0.0) for item in events]
    critical = [float(item.get("critical_path_overhead_sec", 0.0) or 0.0) for item in events]
    overlapped = [float(item.get("overlapped_checkpoint_sec", 0.0) or 0.0) for item in events]
    sizes = [_int_or_zero(item.get("checkpoint_size_bytes")) for item in events]
    return {
        "checkpoint_count": len(events),
        "total_checkpoint_elapsed_sec": sum(elapsed),
        "mean_checkpoint_elapsed_sec": mean(elapsed) if elapsed else 0.0,
        "p50_checkpoint_elapsed_sec": median(elapsed) if elapsed else 0.0,
        "p95_checkpoint_elapsed_sec": _percentile(elapsed, 0.95),
        "total_critical_path_overhead_sec": sum(critical),
        "mean_critical_path_overhead_sec": mean(critical) if critical else 0.0,
        "p50_critical_path_overhead_sec": median(critical) if critical else 0.0,
        "p95_critical_path_overhead_sec": _percentile(critical, 0.95),
        "total_overlapped_checkpoint_sec": sum(overlapped),
        "overlap_fraction": (sum(overlapped) / sum(elapsed)) if sum(elapsed) > 0.0 else 0.0,
        "total_checkpoint_size_bytes": sum(sizes),
        "mean_checkpoint_size_bytes": mean(sizes) if sizes else 0.0,
        "p50_checkpoint_size_bytes": median(sizes) if sizes else 0.0,
        "p95_checkpoint_size_bytes": _percentile([float(value) for value in sizes], 0.95),
    }


def _new_checkpoint_event(
    checkpoint_events: list[dict[str, Any]],
    before_len: int,
) -> dict[str, Any] | None:
    if len(checkpoint_events) <= before_len:
        return None
    event = checkpoint_events[-1]
    return event if isinstance(event, dict) else None


def _checkpoint_event_elapsed_sec(event: dict[str, Any] | None) -> float:
    if not event:
        return 0.0
    try:
        return max(
            0.0,
            float(event.get("checkpoint_elapsed_sec", event.get("create_call_elapsed_sec", 0.0)) or 0.0),
        )
    except (TypeError, ValueError):
        return 0.0


def _record_llm_wait_credit(
    event: dict[str, Any] | None,
    *,
    step_idx: int,
    available_sec: float,
) -> float:
    elapsed_sec = _checkpoint_event_elapsed_sec(event)
    credited_sec = min(max(0.0, available_sec), elapsed_sec)
    if event is not None:
        event["llm_wait_credit_step_idx"] = step_idx
        event["llm_wait_credit_available_sec"] = max(0.0, available_sec)
        event["llm_wait_credit_applied_sec"] = credited_sec
    return credited_sec


async def _attempt_checkpoint_create(
    *,
    env_client: ReplayEnvClient,
    lease_id: str,
    checkpoint_events: list[dict[str, Any]],
    policy: str,
    cwd: str,
    checkpoint_step_idx: int,
    command_seq: int,
    resume_step_idx: int,
    parent_checkpoint_id: str | None,
    protected_env_cost_sec: float,
    protected_llm_cost_sec: float,
    overlap_budget_sec: float = 0.0,
    overlap_source: str = "none",
    report_fields: dict[str, Any] | None = None,
    traj_label: str | None = None,
) -> tuple[dict[str, Any] | None, bool]:
    create_call_start_ts = time.time()
    create_call_started_perf = time.perf_counter()
    checkpoint_event: dict[str, Any] = {
        "event": "checkpoint_create",
        "step_idx": checkpoint_step_idx,
        "policy": policy,
        "create_result": None,
        "status_result": None,
        "skipped": False,
        "skip_reason": None,
        "resume_step_idx": resume_step_idx,
        "resolved_inline": False,
        "create_call_start_ts": create_call_start_ts,
        "create_call_end_ts": None,
        "create_call_elapsed_sec": None,
        # Keep the legacy field for backward compatibility with older reports.
        "protected_cost_sec": protected_env_cost_sec,
        "protected_env_cost_sec": protected_env_cost_sec,
        "protected_llm_cost_sec": protected_llm_cost_sec,
    }
    if report_fields:
        checkpoint_event.update(report_fields)
    if traj_label:
        logger.info(
            "checkpoint create request traj=%s policy=%s step_idx=%s resume_step_idx=%s command_seq=%s parent_checkpoint_id=%s protected_env_cost_sec=%.3f protected_llm_cost_sec=%.3f report_fields=%s",
            traj_label,
            policy,
            checkpoint_step_idx,
            resume_step_idx,
            command_seq,
            parent_checkpoint_id,
            protected_env_cost_sec,
            protected_llm_cost_sec,
            json.dumps(report_fields, ensure_ascii=False, default=_json_default) if report_fields else "{}",
        )
    try:
        create_result = await env_client.checkpoint_create(
            lease_id,
            step_idx=checkpoint_step_idx,
            command_seq=command_seq,
            cwd=cwd,
            policy=policy,
            reason="policy_experiment",
            parent_checkpoint_id=parent_checkpoint_id,
        )
        create_call_end_ts = time.time()
        checkpoint_event["create_call_end_ts"] = create_call_end_ts
        checkpoint_event["create_call_elapsed_sec"] = time.perf_counter() - create_call_started_perf
        checkpoint_event["create_result"] = create_result
        checkpoint_event["status_result"] = {
            **create_result,
            "created_at": create_call_start_ts,
            "ready_at": create_result.get("ready_at", create_call_end_ts),
        }
        checkpoint_event["resolved_inline"] = True
        _annotate_checkpoint_overhead(
            checkpoint_event,
            overlap_budget_sec=overlap_budget_sec,
            overlap_source=overlap_source,
        )
        checkpoint_events.append(checkpoint_event)
        if traj_label:
            logger.info(
                "checkpoint create completed traj=%s policy=%s step_idx=%s checkpoint_id=%s status=%s ready_at=%s",
                traj_label,
                policy,
                checkpoint_step_idx,
                create_result.get("checkpoint_id"),
                create_result.get("status"),
                create_result.get("ready_at"),
            )
        return create_result, False
    except Exception as exc:
        checkpoint_event["create_call_end_ts"] = time.time()
        checkpoint_event["create_call_elapsed_sec"] = time.perf_counter() - create_call_started_perf
        _annotate_checkpoint_overhead(
            checkpoint_event,
            overlap_budget_sec=overlap_budget_sec,
            overlap_source=overlap_source,
        )
        if not _is_checkpoint_busy_error(exc):
            raise
        checkpoint_event["skipped"] = True
        checkpoint_event["skip_reason"] = "checkpoint_busy"
        if isinstance(exc, ReplayOpError):
            checkpoint_event["create_error_payload"] = exc.payload
        checkpoint_events.append(checkpoint_event)
        if traj_label:
            logger.info(
                "checkpoint create skipped busy traj=%s policy=%s step_idx=%s payload=%s",
                traj_label,
                policy,
                checkpoint_step_idx,
                json.dumps(getattr(exc, "payload", None), ensure_ascii=False, default=_json_default),
            )
        return None, True


async def _run_one_trajectory(
    replay_traj: ReplayTrajectory,
    policy: str,
    injection_target: InjectionTarget | None,
    args: argparse.Namespace,
    *,
    defer_gc_until_batch_end: bool = False,
) -> dict[str, Any]:
    traj_path = replay_traj.source_path
    traj_payload, steps = _load_traj_steps(str(traj_path))
    swe_config = _load_yaml_config(args.config_path)
    env_config = swe_config.get("environment", {}) if isinstance(swe_config, dict) else {}
    cwd = args.cwd or str(env_config.get("cwd", "/testbed"))
    exec_timeout = int(args.exec_timeout or int(env_config.get("timeout", 180)))
    instance_id = args.instance_id or _default_instance_id(traj_payload)
    traj_label = replay_traj.report_name
    image_name = _build_image_name(
        traj_payload,
        instance_id=instance_id,
        image_name=args.image_name,
        data_source=args.data_source,
    )

    env_client = ReplayEnvClient(base_url=args.base_url)
    tail_model: AdaptiveTailModel | None = getattr(args, "adaptive_tail_model", None)
    lease = await env_client.allocate(image=image_name, instance_id=instance_id, cwd=cwd)
    lease_id = str(lease["lease_id"])

    report: dict[str, Any] = {
        "traj_key": replay_traj.key,
        "traj_label": traj_label,
        "traj_path": str(traj_path.resolve()),
        "traj_repeat_index": replay_traj.cycle_index,
        "traj_sequence_index": replay_traj.sequence_index,
        "policy": policy,
        "instance_id": instance_id,
        "image_name": image_name,
        "lease": lease,
        "fault_injection_probability": float(args.injection_probability),
        "injection_target": None if injection_target is None else {
            "traj_key": injection_target.traj_key,
            "traj_label": injection_target.traj_label,
            "inject_before_step_indices": list(injection_target.inject_before_step_indices),
            "planned_injection_count": len(injection_target.inject_before_step_indices),
        },
        "steps": [],
        "checkpoint_events": [],
        "failure_events": [],
        "rerun_events": [],
        "gc_result": None,
        "closed": False,
    }

    current_step_idx = 0
    pending_injection_steps = set(
        [] if injection_target is None else injection_target.inject_before_step_indices
    )
    latest_ready_checkpoint_id: str | None = None
    latest_ready_checkpoint_step = -1
    latest_ready_resume_step_idx = 0
    latest_ready_protected_env_cost_sec = 0.0
    latest_ready_protected_llm_cost_sec = 0.0
    cumulative_env_replay_cost_sec = 0.0
    cumulative_llm_replay_cost_sec = 0.0
    checkpoint_attempts = 0
    checkpoint_created = 0
    checkpoint_busy_skips = 0
    probe_count = 0
    probe_busy_skips = 0
    rerun_from_checkpoint = 0
    rerun_from_base = 0
    wall_t0 = time.time()
    attempt_idx = 0
    llm_wait_credit_by_step: dict[int, float] = {}

    try:
        while current_step_idx < len(steps):
            step = steps[current_step_idx]
            await env_client.heartbeat(lease_id)
            llm_delay = step.llm_elapsed if args.simulate_llm_delay else 0.0
            llm_skipped_for_adaptive_rerun = False
            prepaid_llm_wait_sec = min(
                llm_delay,
                max(0.0, llm_wait_credit_by_step.pop(current_step_idx, 0.0)),
            )

            redo_replay_cost_sec = _redo_replay_cost_sec(
                cumulative_env_replay_cost_sec,
                latest_ready_protected_env_cost_sec,
                cumulative_llm_replay_cost_sec,
                latest_ready_protected_llm_cost_sec,
            )
            delta_replay_cost_sec = _adaptive_delta_replay_cost_sec(
                cumulative_env_replay_cost_sec,
                latest_ready_protected_env_cost_sec,
                cumulative_llm_replay_cost_sec,
                latest_ready_protected_llm_cost_sec,
            )
            steps_since_latest_ready_checkpoint = (
                current_step_idx
                if latest_ready_checkpoint_step < 0
                else max(0, current_step_idx - latest_ready_checkpoint_step)
            )
            waited_in_llm_sec = prepaid_llm_wait_sec
            adaptive_checkpoint_submitted = False
            probe_attempted_in_bubble = False
            if llm_delay > waited_in_llm_sec:
                if (
                    policy == "adaptive-risk"
                    and tail_model is not None
                    and redo_replay_cost_sec > 0.0
                    and current_step_idx < len(steps)
                ):
                    while waited_in_llm_sec < llm_delay:
                        expected_overhead_sec = tail_model.expected_exposed_overhead(waited_in_llm_sec)
                        conditional_tail_prob = tail_model.conditional_tail_probability(waited_in_llm_sec)
                        expected_benefit_sec = _adaptive_expected_benefit_sec(
                            args.adaptive_failure_prob,
                            conditional_tail_prob,
                            redo_replay_cost_sec,
                        )
                        should_probe = _should_probe_in_llm_bubble(
                            current_step_idx=current_step_idx,
                            probe_attempted_in_bubble=probe_attempted_in_bubble,
                            adaptive_checkpoint_submitted=adaptive_checkpoint_submitted,
                            pending_checkpoints=[],
                            delta_env_cost_sec=_adaptive_delta_env_cost_sec(
                                cumulative_env_replay_cost_sec,
                                latest_ready_protected_env_cost_sec,
                            ),
                            steps_since_latest_ready_checkpoint=steps_since_latest_ready_checkpoint,
                            expected_benefit_sec=expected_benefit_sec,
                            expected_overhead_sec=expected_overhead_sec,
                        )
                        delta_env_replay_cost_sec = _adaptive_delta_env_cost_sec(
                            cumulative_env_replay_cost_sec,
                            latest_ready_protected_env_cost_sec,
                        )
                        delta_llm_replay_cost_sec = max(
                            0.0,
                            float(cumulative_llm_replay_cost_sec) - float(latest_ready_protected_llm_cost_sec),
                        )
                        logger.info(
                            "adaptive decision traj=%s step_idx=%s waited_sec=%.3f llm_delay_sec=%.3f redo_replay_cost_sec=%.3f delta_replay_cost_sec=%.3f steps_since_latest_ready_checkpoint=%s expected_benefit_sec=%.3f expected_overhead_sec=%.3f conditional_tail_prob=%.6f pending_count=%s probe_attempted=%s submitted=%s should_probe=%s latest_ready_checkpoint_id=%s latest_ready_step=%s",
                            traj_label,
                            current_step_idx,
                            waited_in_llm_sec,
                            llm_delay,
                            redo_replay_cost_sec,
                            delta_replay_cost_sec,
                            steps_since_latest_ready_checkpoint,
                            expected_benefit_sec,
                            expected_overhead_sec,
                            conditional_tail_prob,
                            0,
                            probe_attempted_in_bubble,
                            adaptive_checkpoint_submitted,
                            should_probe,
                            latest_ready_checkpoint_id,
                            latest_ready_checkpoint_step,
                        )
                        if should_probe:
                            probe_attempted_in_bubble = True
                            checkpoint_attempts += 1
                            event_context = {
                                "decision_type": "adaptive_llm_wait",
                                "during_llm_wait_for_step_idx": current_step_idx,
                                "waited_before_checkpoint_sec": waited_in_llm_sec,
                                "expected_benefit_sec": expected_benefit_sec,
                                "expected_overhead_sec": expected_overhead_sec,
                                "redo_from_resume_step_idx": current_step_idx if latest_ready_checkpoint_step < 0 else latest_ready_checkpoint_step + 1,
                                "redo_until_step_idx": current_step_idx,
                                "redo_replay_cost_sec": redo_replay_cost_sec,
                                "redo_env_replay_cost_sec": delta_env_replay_cost_sec,
                                "redo_llm_replay_cost_sec": delta_llm_replay_cost_sec,
                                "delta_replay_cost_sec": delta_replay_cost_sec,
                                "delta_env_replay_cost_sec": delta_env_replay_cost_sec,
                                "delta_llm_replay_cost_sec": delta_llm_replay_cost_sec,
                                # Keep the legacy field for backward compatibility.
                                "delta_env_cost_sec": delta_replay_cost_sec,
                                "steps_since_latest_ready_checkpoint": steps_since_latest_ready_checkpoint,
                                "conditional_tail_probability": conditional_tail_prob,
                            }
                            checkpoint_event_count_before = len(report["checkpoint_events"])
                            create_result, busy = await _attempt_checkpoint_create(
                                env_client=env_client,
                                lease_id=lease_id,
                                checkpoint_events=report["checkpoint_events"],
                                policy=policy,
                                cwd=cwd,
                                checkpoint_step_idx=current_step_idx - 1,
                                command_seq=current_step_idx,
                                resume_step_idx=current_step_idx,
                                parent_checkpoint_id=latest_ready_checkpoint_id,
                                protected_env_cost_sec=cumulative_env_replay_cost_sec,
                                protected_llm_cost_sec=cumulative_llm_replay_cost_sec,
                                overlap_budget_sec=max(0.0, llm_delay - waited_in_llm_sec),
                                overlap_source="current_llm_wait_remaining",
                                report_fields=event_context,
                                traj_label=traj_label,
                            )
                            checkpoint_event = _new_checkpoint_event(
                                report["checkpoint_events"],
                                checkpoint_event_count_before,
                            )
                            waited_in_llm_sec += _record_llm_wait_credit(
                                checkpoint_event,
                                step_idx=current_step_idx,
                                available_sec=llm_delay - waited_in_llm_sec,
                            )
                            if create_result is not None:
                                checkpoint_created += 1
                                adaptive_checkpoint_submitted = True
                                latest_ready_checkpoint_id, latest_ready_checkpoint_step, latest_ready_resume_step_idx, latest_ready_protected_env_cost_sec, latest_ready_protected_llm_cost_sec, promoted = _promote_latest_ready_checkpoint(
                                    latest_ready_checkpoint_id=latest_ready_checkpoint_id,
                                    latest_ready_checkpoint_step=latest_ready_checkpoint_step,
                                    latest_ready_resume_step_idx=latest_ready_resume_step_idx,
                                    latest_ready_protected_env_cost_sec=latest_ready_protected_env_cost_sec,
                                    latest_ready_protected_llm_cost_sec=latest_ready_protected_llm_cost_sec,
                                    checkpoint_id=str(create_result["checkpoint_id"]),
                                    step_idx=int(create_result.get("step_idx", current_step_idx - 1)),
                                    resume_step_idx=current_step_idx,
                                    ready_at=float(create_result.get("ready_at", time.time()) or time.time()),
                                    protected_env_cost_sec=cumulative_env_replay_cost_sec,
                                    protected_llm_cost_sec=cumulative_llm_replay_cost_sec,
                                )
                                logger.info(
                                    "adaptive checkpoint ready traj=%s step_idx=%s checkpoint_id=%s promoted=%s latest_ready_step=%s resume_step_idx=%s",
                                    traj_label,
                                    current_step_idx,
                                    create_result.get("checkpoint_id"),
                                    promoted,
                                    latest_ready_checkpoint_step,
                                    latest_ready_resume_step_idx,
                                )
                            elif busy:
                                checkpoint_busy_skips += 1
                                logger.info(
                                    "adaptive checkpoint create hit busy traj=%s step_idx=%s",
                                    traj_label,
                                    current_step_idx,
                                )
                        sleep_chunk = min(args.adaptive_decision_interval_sec, llm_delay - waited_in_llm_sec)
                        if sleep_chunk > 0:
                            await asyncio.sleep(sleep_chunk)
                            waited_in_llm_sec += sleep_chunk
                else:
                    await asyncio.sleep(llm_delay - waited_in_llm_sec)
                    waited_in_llm_sec = llm_delay

            if current_step_idx in pending_injection_steps:
                pending_injection_steps.discard(current_step_idx)
                failure_event: dict[str, Any] = {
                    "inject_before_step_idx": current_step_idx,
                    "fault_injection_probability": float(args.injection_probability),
                    "latest_ready_checkpoint_id": latest_ready_checkpoint_id,
                    "latest_ready_checkpoint_step": latest_ready_checkpoint_step,
                    "latest_ready_resume_step_idx": latest_ready_resume_step_idx,
                    "recovery_mode": None,
                }
                if latest_ready_checkpoint_id is not None:
                    logger.info(
                        "fault injection rerun from checkpoint traj=%s inject_before_step_idx=%s checkpoint_id=%s resume_step_idx=%s",
                        traj_label,
                        current_step_idx,
                        latest_ready_checkpoint_id,
                        latest_ready_resume_step_idx,
                    )
                    rerun_t0 = time.time()
                    rerun_result = await env_client.rerun(
                        lease_id,
                        checkpoint_id=latest_ready_checkpoint_id,
                        cwd=cwd,
                        timeout=args.rerun_timeout,
                    )
                    rerun_from_checkpoint += 1
                    failure_event["recovery_mode"] = "checkpoint_rerun"
                    failure_event["rerun_result"] = rerun_result
                    failure_event["rerun_wall_time_sec"] = time.time() - rerun_t0
                    if policy == "adaptive-risk":
                        attempt_idx += 1
                    report["failure_events"].append(failure_event)
                    report["rerun_events"].append(failure_event)
                    cumulative_env_replay_cost_sec = latest_ready_protected_env_cost_sec
                    cumulative_llm_replay_cost_sec = latest_ready_protected_llm_cost_sec
                    llm_wait_credit_by_step.clear()
                    current_step_idx = latest_ready_resume_step_idx
                    continue

                logger.info(
                    "fault injection rerun from base traj=%s inject_before_step_idx=%s latest_ready_checkpoint_id=%s",
                    traj_label,
                    current_step_idx,
                    latest_ready_checkpoint_id,
                )
                await env_client.close(lease_id)
                rerun_from_base += 1
                lease = await env_client.allocate(image=image_name, instance_id=instance_id, cwd=cwd)
                lease_id = str(lease["lease_id"])
                failure_event["recovery_mode"] = "base_restart"
                failure_event["new_lease"] = lease
                if policy == "adaptive-risk":
                    attempt_idx += 1
                report["failure_events"].append(failure_event)
                report["rerun_events"].append(failure_event)
                latest_ready_checkpoint_id = None
                latest_ready_checkpoint_step = -1
                latest_ready_resume_step_idx = 0
                latest_ready_protected_env_cost_sec = 0.0
                latest_ready_protected_llm_cost_sec = 0.0
                cumulative_env_replay_cost_sec = 0.0
                cumulative_llm_replay_cost_sec = 0.0
                llm_wait_credit_by_step.clear()
                current_step_idx = 0
                continue

            exec_t0 = time.time()
            exec_result = await env_client.exec(lease_id=lease_id, command=step.action, cwd=cwd, timeout=exec_timeout)
            exec_elapsed = time.time() - exec_t0
            output = str(exec_result.get("output", ""))
            report["steps"].append(
                {
                    "step_idx": current_step_idx,
                    "action": step.action,
                    "actual_returncode": int(exec_result.get("returncode", -1)),
                    "simulated_llm_delay_sec": llm_delay,
                    "llm_waited_before_exec_sec": waited_in_llm_sec,
                    "llm_wait_credit_applied_sec": prepaid_llm_wait_sec,
                    "attempt_idx": attempt_idx,
                    "is_rerun_attempt": attempt_idx > 0,
                    "llm_skipped_for_adaptive_rerun": llm_skipped_for_adaptive_rerun,
                    "exec_elapsed_sec": exec_elapsed,
                    "output_preview": output[:400],
                }
            )
            cumulative_env_replay_cost_sec += exec_elapsed
            cumulative_llm_replay_cost_sec += float(step.llm_elapsed)

            should_attempt_checkpoint = policy == "always" or (
                policy == "every-3" and (current_step_idx + 1) % 3 == 0
            )
            if policy not in POLICIES:
                raise ValueError(f"Unsupported policy: {policy}")

            if current_step_idx < len(steps) - 1 and should_attempt_checkpoint:
                next_llm_overlap_budget_sec = (
                    float(steps[current_step_idx + 1].llm_elapsed)
                    if args.simulate_llm_delay
                    else 0.0
                )
                checkpoint_attempts += 1
                checkpoint_event_count_before = len(report["checkpoint_events"])
                create_result, busy = await _attempt_checkpoint_create(
                    env_client=env_client,
                    lease_id=lease_id,
                    checkpoint_events=report["checkpoint_events"],
                    policy=policy,
                    cwd=cwd,
                    checkpoint_step_idx=current_step_idx,
                    command_seq=current_step_idx + 1,
                    resume_step_idx=current_step_idx + 1,
                    parent_checkpoint_id=latest_ready_checkpoint_id,
                    protected_env_cost_sec=cumulative_env_replay_cost_sec,
                    protected_llm_cost_sec=cumulative_llm_replay_cost_sec,
                    overlap_budget_sec=next_llm_overlap_budget_sec,
                    overlap_source="next_llm_response",
                    traj_label=traj_label,
                )
                checkpoint_event = _new_checkpoint_event(
                    report["checkpoint_events"],
                    checkpoint_event_count_before,
                )
                credited_sec = _record_llm_wait_credit(
                    checkpoint_event,
                    step_idx=current_step_idx + 1,
                    available_sec=next_llm_overlap_budget_sec,
                )
                if credited_sec > 0.0:
                    llm_wait_credit_by_step[current_step_idx + 1] = (
                        llm_wait_credit_by_step.get(current_step_idx + 1, 0.0) + credited_sec
                    )
                if create_result is not None:
                    checkpoint_created += 1
                    latest_ready_checkpoint_id, latest_ready_checkpoint_step, latest_ready_resume_step_idx, latest_ready_protected_env_cost_sec, latest_ready_protected_llm_cost_sec, promoted = _promote_latest_ready_checkpoint(
                        latest_ready_checkpoint_id=latest_ready_checkpoint_id,
                        latest_ready_checkpoint_step=latest_ready_checkpoint_step,
                        latest_ready_resume_step_idx=latest_ready_resume_step_idx,
                        latest_ready_protected_env_cost_sec=latest_ready_protected_env_cost_sec,
                        latest_ready_protected_llm_cost_sec=latest_ready_protected_llm_cost_sec,
                        checkpoint_id=str(create_result["checkpoint_id"]),
                        step_idx=int(create_result.get("step_idx", current_step_idx)),
                        resume_step_idx=current_step_idx + 1,
                        ready_at=float(create_result.get("ready_at", time.time()) or time.time()),
                        protected_env_cost_sec=cumulative_env_replay_cost_sec,
                        protected_llm_cost_sec=cumulative_llm_replay_cost_sec,
                    )
                    logger.info(
                        "checkpoint ready traj=%s policy=%s step_idx=%s checkpoint_id=%s promoted=%s latest_ready_step=%s resume_step_idx=%s",
                        traj_label,
                        policy,
                        current_step_idx,
                        create_result.get("checkpoint_id"),
                        promoted,
                        latest_ready_checkpoint_step,
                        latest_ready_resume_step_idx,
                    )
                elif busy:
                    checkpoint_busy_skips += 1

            current_step_idx += 1

        if args.gc_keep_latest is not None and not defer_gc_until_batch_end:
            gc_payload = await _maybe_run_checkpoint_gc(
                env_client,
                lease_id,
                keep_latest=args.gc_keep_latest,
                dry_run=args.gc_dry_run,
                min_checkpoint_count=args.gc_min_checkpoint_count,
            )
            report["gc_result"] = gc_payload["gc_result"]
        elif args.gc_keep_latest is not None and defer_gc_until_batch_end:
            report["gc_result"] = _deferred_batch_gc_result(
                keep_latest=args.gc_keep_latest,
                dry_run=args.gc_dry_run,
                min_checkpoint_count=args.gc_min_checkpoint_count,
            )
    finally:
        try:
            await env_client.close(lease_id)
            report["closed"] = True
        except Exception as exc:
            report["close_error"] = str(exc)

    checkpoint_overhead_summary = _summarize_checkpoint_overhead_events(
        _successful_checkpoint_overhead_events([report])
    )
    total_llm_wait_credit_applied_sec = sum(
        float(event.get("llm_wait_credit_applied_sec", 0.0) or 0.0)
        for event in report["checkpoint_events"]
        if isinstance(event, dict)
    )
    report["metrics"] = {
        "checkpoint_attempts": checkpoint_attempts,
        "checkpoint_created": checkpoint_created,
        "checkpoint_busy_skips": checkpoint_busy_skips,
        "probe_count": probe_count,
        "probe_busy_skips": probe_busy_skips,
        "rerun_from_checkpoint": rerun_from_checkpoint,
        "rerun_from_base": rerun_from_base,
        "fault_injections_triggered": len(report["failure_events"]),
        "fault_injections_remaining": len(pending_injection_steps),
        "wall_time_sec": time.time() - wall_t0,
        "llm_wait_credit_applied_sec": total_llm_wait_credit_applied_sec,
        "checkpoint_overhead": checkpoint_overhead_summary,
    }
    return report


async def _run_policy(
    replay_trajectories: list[ReplayTrajectory],
    injections: dict[str, InjectionTarget],
    policy: str,
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, Any]:
    sem = asyncio.Semaphore(max(1, args.max_concurrency))
    reports: list[dict[str, Any]] = []
    started_at = time.time()
    batch_gc_result: dict[str, Any] | None = None

    async def _guard(replay_traj: ReplayTrajectory) -> dict[str, Any]:
        async with sem:
            try:
                injection = injections.get(replay_traj.key)
                report = await _run_one_trajectory(
                    replay_traj,
                    policy,
                    injection,
                    args,
                    defer_gc_until_batch_end=args.gc_keep_latest is not None,
                )
                report["ok"] = True
                return report
            except Exception as exc:
                return {
                    "traj_key": replay_traj.key,
                    "traj_label": replay_traj.report_name,
                    "traj_path": str(replay_traj.source_path.resolve()),
                    "policy": policy,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }

    tasks = [asyncio.create_task(_guard(replay_traj)) for replay_traj in replay_trajectories]
    for task in asyncio.as_completed(tasks):
        report = await task
        reports.append(report)
        per_traj_dir = out_dir / "per_traj"
        per_traj_dir.mkdir(parents=True, exist_ok=True)
        name = str(report.get("traj_label") or (Path(report["traj_path"]).parent.name if "traj_path" in report else f"failed-{len(reports)}"))
        (per_traj_dir / f"{name}.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
            encoding="utf-8",
        )

    if args.gc_keep_latest is not None:
        env_client = ReplayEnvClient(base_url=args.base_url)
        gc_payload = await _maybe_run_checkpoint_gc(
            env_client,
            None,
            keep_latest=args.gc_keep_latest,
            dry_run=args.gc_dry_run,
            min_checkpoint_count=args.gc_min_checkpoint_count,
        )
        batch_gc_result = gc_payload["gc_result"]

    wall_times = [float(item.get("metrics", {}).get("wall_time_sec", 0.0)) for item in reports if item.get("ok")]
    total_attempts = sum(int(item.get("metrics", {}).get("checkpoint_attempts", 0)) for item in reports if item.get("ok"))
    total_created = sum(int(item.get("metrics", {}).get("checkpoint_created", 0)) for item in reports if item.get("ok"))
    total_busy_skips = sum(int(item.get("metrics", {}).get("checkpoint_busy_skips", 0)) for item in reports if item.get("ok"))
    total_probe_count = sum(int(item.get("metrics", {}).get("probe_count", 0)) for item in reports if item.get("ok"))
    total_probe_busy = sum(int(item.get("metrics", {}).get("probe_busy_skips", 0)) for item in reports if item.get("ok"))
    total_rerun_ckpt = sum(int(item.get("metrics", {}).get("rerun_from_checkpoint", 0)) for item in reports if item.get("ok"))
    total_rerun_base = sum(int(item.get("metrics", {}).get("rerun_from_base", 0)) for item in reports if item.get("ok"))
    total_fault_injections = sum(int(item.get("metrics", {}).get("fault_injections_triggered", 0)) for item in reports if item.get("ok"))
    total_llm_wait_credit = sum(
        float(item.get("metrics", {}).get("llm_wait_credit_applied_sec", 0.0) or 0.0)
        for item in reports
        if item.get("ok")
    )
    total_gc_deleted = int((batch_gc_result or {}).get("deleted_count", 0))
    total_gc_reclaimed = int((batch_gc_result or {}).get("reclaimed_bytes", 0))
    planned_injection_count = sum(len(item.inject_before_step_indices) for item in injections.values())
    checkpoint_overhead = _summarize_checkpoint_overhead_events(
        _successful_checkpoint_overhead_events([item for item in reports if item.get("ok")])
    )

    summary = {
        "policy": policy,
        "trajectory_count": len(replay_trajectories),
        "ok_count": sum(1 for item in reports if item.get("ok")),
        "failed_count": sum(1 for item in reports if not item.get("ok")),
        "batch_wall_time_sec": time.time() - started_at,
        "mean_traj_wall_time_sec": mean(wall_times) if wall_times else 0.0,
        "p50_traj_wall_time_sec": median(wall_times) if wall_times else 0.0,
        "p95_traj_wall_time_sec": sorted(wall_times)[max(0, int(len(wall_times) * 0.95) - 1)] if wall_times else 0.0,
        "checkpoint_attempts": total_attempts,
        "checkpoint_created": total_created,
        "checkpoint_busy_skips": total_busy_skips,
        "probe_count": total_probe_count,
        "probe_busy_skips": total_probe_busy,
        "rerun_from_checkpoint": total_rerun_ckpt,
        "rerun_from_base": total_rerun_base,
        "fault_injection_probability": float(args.injection_probability),
        "planned_injection_count": planned_injection_count,
        "fault_injections_triggered": total_fault_injections,
        "gc_deleted_count": total_gc_deleted,
        "gc_reclaimed_bytes": total_gc_reclaimed,
        "injection_trajectory_count": len(injections),
        "checkpoint_overhead": checkpoint_overhead,
        "checkpoint_total_elapsed_sec": checkpoint_overhead["total_checkpoint_elapsed_sec"],
        "checkpoint_total_critical_path_overhead_sec": checkpoint_overhead["total_critical_path_overhead_sec"],
        "checkpoint_total_overlapped_sec": checkpoint_overhead["total_overlapped_checkpoint_sec"],
        "checkpoint_total_size_bytes": checkpoint_overhead["total_checkpoint_size_bytes"],
        "llm_wait_credit_applied_sec": total_llm_wait_credit,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {"summary": summary, "batch_gc_result": batch_gc_result, "reports": reports},
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    return {"summary": summary, "batch_gc_result": batch_gc_result, "reports": reports}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run real checkpoint policy experiments with probabilistic fault injection on SWE trajectory replays.")
    parser.add_argument("trajectory_root", help="Directory containing traj.json files")
    parser.add_argument("--base-url", default=os.getenv("SWE_ENV_SERVER_URL"), required=False)
    parser.add_argument("--config-path", default=os.getenv("SWE_CONFIG_PATH"))
    parser.add_argument("--data-source", default="swe-gym")
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--image-name", default=None)
    parser.add_argument("--cwd", default=None)
    parser.add_argument("--exec-timeout", type=int, default=None)
    parser.add_argument("--simulate-llm-delay", action="store_true")
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--injection-seed", type=int, default=20260407)
    parser.add_argument(
        "--injection-probability",
        type=float,
        default=0.01,
        help="Per-step fault injection probability applied to each logical replay step index >= 1.",
    )
    parser.add_argument("--rerun-timeout", type=int, default=120)
    parser.add_argument("--gc-keep-latest", type=int, default=0)
    parser.add_argument(
        "--gc-min-checkpoint-count",
        type=int,
        default=100,
        help="Only run global checkpoint GC when total checkpoint/list count reaches at least this many checkpoints. Set 0 to always GC.",
    )
    parser.add_argument("--gc-dry-run", action="store_true")
    parser.add_argument("--gc-drain-timeout-sec", type=float, default=1800.0)
    parser.add_argument("--gc-drain-poll-interval-sec", type=float, default=0.1)
    parser.add_argument("--adaptive-failure-prob", type=float, default=0.01)
    parser.add_argument("--adaptive-tail-root", default=str(DEFAULT_ADAPTIVE_TAIL_ROOT))
    parser.add_argument("--adaptive-checkpoint-budget-sec", type=float, default=DEFAULT_ADAPTIVE_BUDGET_SEC)
    parser.add_argument("--adaptive-decision-interval-sec", type=float, default=DEFAULT_ADAPTIVE_DECISION_INTERVAL_SEC)
    parser.add_argument("--adaptive-idle-prob", type=float, default=0.7)
    parser.add_argument("--adaptive-probe-wait-busy", type=float, default=1.0)
    parser.add_argument("--adaptive-threshold", type=float, default=1.0)
    parser.add_argument("--output-root", default=None)
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if not args.base_url:
        raise ValueError("--base-url is required")
    if "adaptive-risk" in args.policies:
        args.adaptive_tail_model = _load_adaptive_tail_model(
            args.adaptive_tail_root,
            budget_sec=args.adaptive_checkpoint_budget_sec,
        )
    else:
        args.adaptive_tail_model = None
    source_traj_paths = _collect_traj_paths(args.trajectory_root, limit=None)
    traj_paths = _expand_traj_paths(source_traj_paths, args.limit)
    injections = _build_probability_injection_plan(
        traj_paths,
        args.injection_seed,
        args.injection_probability,
    )
    injection_map = {item.traj_key: item for item in injections}

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_root = Path(args.output_root or (REPO_ROOT / "export" / f"checkpoint_policy_fault_experiment_{ts}"))
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "injection_plan.json").write_text(
        json.dumps([item.__dict__ for item in injections], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    policy_summaries = []
    for policy_idx, policy in enumerate(args.policies):
        policy_dir = output_root / policy
        policy_dir.mkdir(parents=True, exist_ok=True)
        policy_injections = {} if policy == "oracle-no-fault-no-checkpoint" else injection_map
        result = await _run_policy(traj_paths, policy_injections, policy, args, policy_dir)
        policy_summaries.append(result["summary"])
        if policy_idx < len(args.policies) - 1 and args.gc_keep_latest is not None:
            logger.info(
                "waiting for async checkpoint gc drain after policy=%s timeout_sec=%.3f poll_interval_sec=%.3f",
                policy,
                args.gc_drain_timeout_sec,
                args.gc_drain_poll_interval_sec,
            )
            drain_result = await _wait_for_global_gc_drain(
                base_url=args.base_url,
                timeout_sec=args.gc_drain_timeout_sec,
                poll_interval_sec=args.gc_drain_poll_interval_sec,
            )
            logger.info(
                "async checkpoint gc drain completed after policy=%s result=%s",
                policy,
                json.dumps(drain_result, ensure_ascii=False, default=_json_default),
            )

    final_payload = {
        "trajectory_root": str(Path(args.trajectory_root).resolve()),
        "source_trajectory_count": len(source_traj_paths),
        "trajectory_count": len(traj_paths),
        "max_concurrency": args.max_concurrency,
        "fault_injection": {
            "mode": "per_step_probability_plan",
            "probability": float(args.injection_probability),
            "seed": int(args.injection_seed),
            "planned_trajectory_count": len(injections),
            "planned_injection_count": sum(len(item.inject_before_step_indices) for item in injections),
        },
        "adaptive_tail_model": None
        if args.adaptive_tail_model is None
        else {
            "root": str(Path(args.adaptive_tail_root).resolve()),
            "budget_sec": args.adaptive_tail_model.budget_sec,
            "sample_count": args.adaptive_tail_model.count,
            "decision_interval_sec": args.adaptive_decision_interval_sec,
            "failure_prob": args.adaptive_failure_prob,
        },
        "policies": policy_summaries,
        "injection_plan": [item.__dict__ for item in injections],
    }
    print(json.dumps(final_payload, indent=2, ensure_ascii=False, default=_json_default))
    (output_root / "summary_all_policies.json").write_text(
        json.dumps(final_payload, indent=2, ensure_ascii=False, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
