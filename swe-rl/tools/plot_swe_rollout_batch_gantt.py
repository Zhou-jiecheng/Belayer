#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


POLICIES = (
    "oracle-no-fault-no-checkpoint",
    "adaptive-risk",
    "every-3",
    "never",
    "always",
)


@dataclass
class StepSpan:
    step_idx: int
    llm_start_epoch: float
    llm_end_epoch: float
    exec_start_epoch: float
    exec_end_epoch: float


@dataclass
class TrajectoryTimeline:
    label: str
    ok: bool
    llm_bars: list[tuple[float, float]]
    exec_bars: list[tuple[float, float]]
    inferred_history_llm_bars: list[tuple[float, float]]
    inferred_history_exec_bars: list[tuple[float, float]]
    replay_llm_bars: list[tuple[float, float]]
    replay_exec_bars: list[tuple[float, float]]
    llm_only_bars: list[tuple[float, float]]
    checkpoint_call_bars: list[tuple[float, float]]
    discarded_attempt_bars: list[tuple[float, float]]
    gap_bars: list[tuple[float, float]]
    checkpoint_ready_bars: list[tuple[float, float]]
    checkpoint_create_points: list[float]
    checkpoint_busy_points: list[float]
    replay_bars: list[tuple[float, float]]
    recovery_bands: list[tuple[float, float, str]]
    injection_points: list[float]
    recovery_points: list[float]
    first_start_sec: float
    last_end_sec: float
    llm_total_sec: float
    exec_total_sec: float
    llm_only_total_sec: float
    checkpoint_call_total_sec: float
    discarded_attempt_total_sec: float
    replay_total_sec: float
    gap_total_sec: float
    checkpoint_pending_total_sec: float
    recovery_total_sec: float
    max_llm_seg_sec: float
    max_llm_only_seg_sec: float
    max_checkpoint_call_seg_sec: float
    max_discarded_attempt_seg_sec: float
    max_replay_seg_sec: float
    max_gap_seg_sec: float
    max_checkpoint_pending_seg_sec: float
    max_recovery_seg_sec: float
    dominant_component: str
    dominant_component_sec: float


def _bars_total_sec(bars: list[tuple[float, float]]) -> float:
    return float(sum(width_sec for _, width_sec in bars))


def _bars_max_sec(bars: list[tuple[float, float]]) -> float:
    return max((float(width_sec) for _, width_sec in bars), default=0.0)


def _dominant_component(
    *,
    max_llm_seg_sec: float,
    max_llm_only_seg_sec: float,
    max_checkpoint_call_seg_sec: float,
    max_discarded_attempt_seg_sec: float,
    max_replay_seg_sec: float,
    max_gap_seg_sec: float,
    max_checkpoint_pending_seg_sec: float,
    max_recovery_seg_sec: float,
) -> tuple[str, float]:
    candidates = [
        ("llm", float(max_llm_seg_sec)),
        ("llm_only", float(max_llm_only_seg_sec)),
        ("checkpoint_call", float(max_checkpoint_call_seg_sec)),
        ("discarded_attempt", float(max_discarded_attempt_seg_sec)),
        ("replay_after_recovery", float(max_replay_seg_sec)),
        ("unaccounted_gap", float(max_gap_seg_sec)),
        ("checkpoint_pending", float(max_checkpoint_pending_seg_sec)),
        ("recovery", float(max_recovery_seg_sec)),
    ]
    return max(candidates, key=lambda item: item[1])


def _iter_traj_paths(policy_root: Path) -> list[Path]:
    return sorted(policy_root.glob("rollouts/*/traj.json"))


def _build_step_spans(report: dict[str, Any]) -> list[StepSpan]:
    spans: list[StepSpan] = []
    for step in report.get("step_debug", []):
        exec_start = float(step.get("start_ts", 0.0) or 0.0)
        exec_end = float(step.get("end_ts", exec_start) or exec_start)
        llm_wait = float(step.get("llm_waited_before_exec_sec", 0.0) or 0.0)
        llm_end = exec_start
        llm_start = llm_end - llm_wait
        spans.append(
            StepSpan(
                step_idx=int(step.get("step_idx", len(spans))),
                llm_start_epoch=llm_start,
                llm_end_epoch=llm_end,
                exec_start_epoch=exec_start,
                exec_end_epoch=exec_end,
            )
        )
    return spans


def _find_step_span(spans: list[StepSpan], step_idx: int) -> StepSpan | None:
    for span in spans:
        if span.step_idx == step_idx:
            return span
    return None


def _event_time_epoch(event: dict[str, Any], spans: list[StepSpan]) -> float | None:
    target_step_idx = event.get("during_llm_wait_for_step_idx")
    if target_step_idx is not None:
        span = _find_step_span(spans, int(target_step_idx))
        if span is not None:
            waited = float(event.get("waited_before_checkpoint_sec", 0.0) or 0.0)
            return min(span.llm_end_epoch, span.llm_start_epoch + waited)

    step_idx = event.get("step_idx")
    if step_idx is None:
        return None
    span = _find_step_span(spans, int(step_idx))
    if span is None:
        return None
    if event.get("decision_type") == "adaptive_llm_wait":
        waited = float(event.get("waited_before_checkpoint_sec", 0.0) or 0.0)
        return min(span.llm_end_epoch, span.llm_start_epoch + waited)
    return span.exec_end_epoch


def _checkpoint_event_epoch(event: dict[str, Any], spans: list[StepSpan]) -> float | None:
    create_call_start_ts = event.get("create_call_start_ts")
    if create_call_start_ts is not None:
        return float(create_call_start_ts)
    status_result = event.get("status_result")
    if isinstance(status_result, dict):
        created_at = status_result.get("created_at")
        if created_at is not None:
            return float(created_at)
    return _event_time_epoch(event, spans)


def _checkpoint_ready_status_result(event: dict[str, Any]) -> dict[str, Any] | None:
    status_result = event.get("status_result")
    if not isinstance(status_result, dict):
        return None
    created_at = status_result.get("created_at")
    ready_at = status_result.get("ready_at")
    if created_at is None or ready_at is None:
        return None
    if (
        event.get("event") == "checkpoint_ready"
        or bool(event.get("resolved_from_pending"))
        or status_result.get("status") == "ready"
    ):
        return status_result
    return None


def _checkpoint_call_fallback_bars(
    report: dict[str, Any],
    *,
    spans: list[StepSpan],
    batch_start_epoch: float,
    min_width_sec: float = 0.0,
) -> list[tuple[float, float]]:
    bars: list[tuple[float, float]] = []
    for checkpoint_event in report.get("checkpoint_events", []):
        if checkpoint_event.get("event") != "checkpoint_create":
            continue
        start_epoch = _event_time_epoch(checkpoint_event, spans)
        if start_epoch is None:
            continue
        create_result = checkpoint_event.get("create_result")
        if not isinstance(create_result, dict):
            continue

        end_epoch: float | None = None
        ready_at = create_result.get("ready_at")
        if ready_at is not None:
            end_epoch = float(ready_at)
        elif bool(create_result.get("timed_out")) and create_result.get("timeout_sec") is not None:
            end_epoch = start_epoch + float(create_result.get("timeout_sec"))

        if end_epoch is None:
            continue
        width_sec = max(0.0, end_epoch - start_epoch)
        if width_sec < float(min_width_sec):
            continue
        bars.append((start_epoch - batch_start_epoch, width_sec))
    return bars


def _checkpoint_call_event_bars(
    report: dict[str, Any],
    *,
    batch_start_epoch: float,
    min_width_sec: float = 0.0,
) -> list[tuple[float, float]]:
    bars: list[tuple[float, float]] = []
    for checkpoint_event in report.get("checkpoint_events", []):
        if checkpoint_event.get("event") != "checkpoint_create":
            continue
        start_ts = checkpoint_event.get("create_call_start_ts")
        end_ts = checkpoint_event.get("create_call_end_ts")
        elapsed_sec = checkpoint_event.get("create_call_elapsed_sec")
        if start_ts is None:
            continue
        start_epoch = float(start_ts)
        if end_ts is not None:
            end_epoch = max(start_epoch, float(end_ts))
        elif elapsed_sec is not None:
            end_epoch = start_epoch + max(0.0, float(elapsed_sec))
        else:
            continue
        width_sec = end_epoch - start_epoch
        if width_sec < float(min_width_sec):
            continue
        bars.append((start_epoch - batch_start_epoch, width_sec))
    return bars


def _phase_event_bars(
    report: dict[str, Any],
    *,
    batch_start_epoch: float,
    category: str | None = None,
    event: str | None = None,
    min_width_sec: float = 0.0,
) -> list[tuple[float, float]]:
    bars: list[tuple[float, float]] = []
    for phase in report.get("phase_events", []):
        if category is not None and str(phase.get("category", "")) != category:
            continue
        if event is not None and str(phase.get("event", "")) != event:
            continue
        start_ts = phase.get("start_ts")
        end_ts = phase.get("end_ts")
        if start_ts is None or end_ts is None:
            continue
        start_epoch = float(start_ts)
        end_epoch = max(start_epoch, float(end_ts))
        width_sec = end_epoch - start_epoch
        if width_sec < float(min_width_sec):
            continue
        bars.append((start_epoch - batch_start_epoch, width_sec))
    return bars


def _collect_batch_start_epoch(traj_paths: list[Path]) -> float:
    batch_start_epoch: float | None = None
    for traj_path in traj_paths:
        report = json.loads(traj_path.read_text())
        spans = _build_step_spans(report)
        if spans:
            start_epoch = min(span.llm_start_epoch for span in spans)
            if batch_start_epoch is None or start_epoch < batch_start_epoch:
                batch_start_epoch = start_epoch
        for event in report.get("checkpoint_events", []):
            create_call_start_ts = event.get("create_call_start_ts")
            if create_call_start_ts is not None:
                create_call_start_ts = float(create_call_start_ts)
                if batch_start_epoch is None or create_call_start_ts < batch_start_epoch:
                    batch_start_epoch = create_call_start_ts
            status_result = event.get("status_result")
            if isinstance(status_result, dict):
                created_at = status_result.get("created_at")
                if created_at is not None:
                    created_at = float(created_at)
                    if batch_start_epoch is None or created_at < batch_start_epoch:
                        batch_start_epoch = created_at
        for phase in report.get("phase_events", []):
            start_ts = phase.get("start_ts")
            if start_ts is None:
                continue
            start_epoch = float(start_ts)
            if batch_start_epoch is None or start_epoch < batch_start_epoch:
                batch_start_epoch = start_epoch
    if batch_start_epoch is None:
        raise RuntimeError("No timing information found in rollout trajectories.")
    return batch_start_epoch


def _build_timeline(traj_path: Path, batch_start_epoch: float) -> TrajectoryTimeline:
    report = json.loads(traj_path.read_text())
    spans = sorted(_build_step_spans(report), key=lambda item: (item.llm_start_epoch, item.step_idx))
    label = traj_path.parent.name
    ok = not bool(report.get("info", {}).get("error"))

    llm_bars = [
        (span.llm_start_epoch - batch_start_epoch, max(0.0, span.llm_end_epoch - span.llm_start_epoch))
        for span in spans
        if span.llm_end_epoch > span.llm_start_epoch
    ]
    exec_bars = [
        (span.exec_start_epoch - batch_start_epoch, max(0.0, span.exec_end_epoch - span.exec_start_epoch))
        for span in spans
        if span.exec_end_epoch > span.exec_start_epoch
    ]
    llm_only_bars = _phase_event_bars(
        report,
        batch_start_epoch=batch_start_epoch,
        category="llm_only",
        event="llm_only_turn",
        min_width_sec=0.01,
    )
    checkpoint_call_bars = _phase_event_bars(
        report,
        batch_start_epoch=batch_start_epoch,
        category="checkpoint",
        event="checkpoint_create_call",
        min_width_sec=0.01,
    )
    if not checkpoint_call_bars:
        checkpoint_call_bars = _checkpoint_call_event_bars(
            report,
            batch_start_epoch=batch_start_epoch,
            min_width_sec=0.01,
        )
    if not checkpoint_call_bars:
        checkpoint_call_bars = _checkpoint_call_fallback_bars(
            report,
            spans=spans,
            batch_start_epoch=batch_start_epoch,
            min_width_sec=0.01,
        )
    discarded_attempt_bars = _phase_event_bars(
        report,
        batch_start_epoch=batch_start_epoch,
        category="discarded_attempt",
        event="discarded_attempt_window",
        min_width_sec=0.01,
    )
    phase_events_present = bool(report.get("phase_events"))
    gap_bars: list[tuple[float, float]] = []
    if not phase_events_present:
        for prev_span, next_span in zip(spans, spans[1:]):
            gap_start_epoch = prev_span.exec_end_epoch
            gap_end_epoch = next_span.llm_start_epoch
            gap_width_sec = gap_end_epoch - gap_start_epoch
            if gap_width_sec > 0.05:
                gap_bars.append((gap_start_epoch - batch_start_epoch, gap_width_sec))

    checkpoint_ready_bars: list[tuple[float, float]] = []
    checkpoint_create_points: list[float] = []
    checkpoint_busy_points: list[float] = []
    for event in report.get("checkpoint_events", []):
        if event.get("event") == "checkpoint_create":
            event_epoch = _checkpoint_event_epoch(event, spans)
            if event_epoch is not None:
                checkpoint_create_points.append(event_epoch - batch_start_epoch)
        if event.get("skip_reason") in {"checkpoint_busy", "probe_busy"}:
            event_epoch = _checkpoint_event_epoch(event, spans)
            if event_epoch is not None:
                checkpoint_busy_points.append(event_epoch - batch_start_epoch)
        status_result = _checkpoint_ready_status_result(event)
        if status_result is None:
            continue
        created_at = status_result.get("created_at")
        ready_at = status_result.get("ready_at")
        if created_at is None or ready_at is None:
            continue
        start_sec = float(created_at) - batch_start_epoch
        duration_sec = max(0.0, float(ready_at) - float(created_at))
        checkpoint_ready_bars.append((start_sec, duration_sec))

    recovery_bands: list[tuple[float, float, str]] = []
    injection_points: list[float] = []
    recovery_points: list[float] = []
    recovery_phase_events = [
        phase
        for phase in report.get("phase_events", [])
        if str(phase.get("category", "")) == "recovery"
    ]
    if recovery_phase_events:
        for phase in recovery_phase_events:
            start_ts = phase.get("start_ts")
            end_ts = phase.get("end_ts")
            if start_ts is None or end_ts is None:
                continue
            recovery_start_sec = float(start_ts) - batch_start_epoch
            recovery_width_sec = max(0.0, float(end_ts) - float(start_ts))
            recovery_mode = str(phase.get("recovery_mode", "recovery"))
            recovery_bands.append((recovery_start_sec, recovery_width_sec, recovery_mode))
            injection_points.append(recovery_start_sec)
            recovery_points.append(recovery_start_sec + recovery_width_sec)
    else:
        for rerun_event in report.get("rerun_events", []):
            inject_before_step_idx = int(rerun_event.get("inject_before_step_idx", -1))
            span = _find_step_span(spans, inject_before_step_idx)
            if span is None:
                continue
            recovery_end_sec = span.exec_start_epoch - batch_start_epoch
            recovery_width_sec = float(rerun_event.get("rerun_wall_time_sec", 0.0) or 0.0)
            recovery_start_sec = max(0.0, recovery_end_sec - recovery_width_sec)
            recovery_mode = str(rerun_event.get("recovery_mode", "recovery"))
            latest_ready_step = rerun_event.get("latest_ready_checkpoint_step")
            if recovery_mode == "checkpoint_rerun" and latest_ready_step is not None:
                recovery_label = f"ckpt {latest_ready_step}->{inject_before_step_idx}"
            elif recovery_mode == "base_restart":
                recovery_label = f"base->{inject_before_step_idx}"
            else:
                recovery_label = recovery_mode
            recovery_bands.append((recovery_start_sec, recovery_width_sec, recovery_label))
            injection_points.append(recovery_start_sec)
            recovery_points.append(recovery_end_sec)

    reconstructible_rerun_events = list(report.get("rerun_events", []))
    if reconstructible_rerun_events:
        reconstructible_rerun_events = [reconstructible_rerun_events[-1]]
    reconstructible_recovery_phase_events = (
        recovery_phase_events[-len(reconstructible_rerun_events):]
        if reconstructible_rerun_events
        else []
    )

    inferred_history_llm_bars: list[tuple[float, float]] = []
    inferred_history_exec_bars: list[tuple[float, float]] = []
    for rerun_idx, rerun_event in enumerate(reconstructible_rerun_events):
        resume_step_idx = int(rerun_event.get("resume_step_idx", -1))
        inject_before_step_idx = int(rerun_event.get("inject_before_step_idx", -1))
        if resume_step_idx < 0 or inject_before_step_idx <= resume_step_idx:
            continue
        replay_spans = [
            span for span in spans
            if resume_step_idx <= int(span.step_idx) < inject_before_step_idx
        ]
        if not replay_spans:
            continue
        recovery_start_epoch: float | None = None
        if rerun_idx < len(reconstructible_recovery_phase_events):
            phase_start_ts = reconstructible_recovery_phase_events[rerun_idx].get("start_ts")
            if phase_start_ts is not None:
                recovery_start_epoch = float(phase_start_ts)
        if recovery_start_epoch is None:
            inject_span = _find_step_span(spans, inject_before_step_idx)
            if inject_span is None:
                continue
            rerun_width_sec = float(rerun_event.get("rerun_wall_time_sec", 0.0) or 0.0)
            recovery_start_epoch = float(inject_span.exec_start_epoch) - rerun_width_sec
        cursor_epoch = float(recovery_start_epoch)
        for span in reversed(replay_spans):
            exec_width_sec = max(0.0, float(span.exec_end_epoch) - float(span.exec_start_epoch))
            exec_end_epoch = cursor_epoch
            exec_start_epoch = exec_end_epoch - exec_width_sec
            llm_width_sec = max(0.0, float(span.llm_end_epoch) - float(span.llm_start_epoch))
            llm_end_epoch = exec_start_epoch
            llm_start_epoch = llm_end_epoch - llm_width_sec
            if llm_width_sec > 0.0:
                inferred_history_llm_bars.append((llm_start_epoch - batch_start_epoch, llm_width_sec))
            if exec_width_sec > 0.0:
                inferred_history_exec_bars.append((exec_start_epoch - batch_start_epoch, exec_width_sec))
            cursor_epoch = llm_start_epoch

    if not discarded_attempt_bars:
        rerun_events = reconstructible_rerun_events
        for rerun_idx, rerun_event in enumerate(rerun_events):
            resume_step_idx = int(rerun_event.get("resume_step_idx", -1))
            inject_before_step_idx = int(rerun_event.get("inject_before_step_idx", -1))
            if inject_before_step_idx < 0:
                continue
            if resume_step_idx <= 0:
                # Legacy base-restart traces (resume=0) do not preserve the
                # pre-fault attempt in step_debug. Estimate that discarded
                # window from the replay prefix before inject_before_step_idx.
                recovery_start_epoch: float | None = None
                if rerun_idx < len(reconstructible_recovery_phase_events):
                    phase_start_ts = reconstructible_recovery_phase_events[rerun_idx].get("start_ts")
                    if phase_start_ts is not None:
                        recovery_start_epoch = float(phase_start_ts)
                if recovery_start_epoch is None:
                    first_span = _find_step_span(spans, 0)
                    if first_span is None:
                        continue
                    rerun_width_sec = float(rerun_event.get("rerun_wall_time_sec", 0.0) or 0.0)
                    recovery_start_epoch = float(first_span.llm_start_epoch) - rerun_width_sec
                replay_prefix_spans = [
                    span for span in spans
                    if 0 <= int(span.step_idx) < inject_before_step_idx
                ]
                if not replay_prefix_spans:
                    continue
                estimated_discarded_width_sec = (
                    max(float(span.exec_end_epoch) for span in replay_prefix_spans)
                    - min(float(span.llm_start_epoch) for span in replay_prefix_spans)
                )
                if estimated_discarded_width_sec <= 0.05:
                    continue
                discarded_start_sec = (
                    recovery_start_epoch - estimated_discarded_width_sec - batch_start_epoch
                )
                discarded_width_sec = float(estimated_discarded_width_sec)
                if discarded_start_sec < 0.0:
                    discarded_width_sec += discarded_start_sec
                    discarded_start_sec = 0.0
                if discarded_width_sec > 0.05:
                    discarded_attempt_bars.append((discarded_start_sec, discarded_width_sec))
                continue
            prev_span = _find_step_span(spans, resume_step_idx - 1)
            if prev_span is None:
                continue
            recovery_start_epoch: float | None = None
            if rerun_idx < len(reconstructible_recovery_phase_events):
                phase_start_ts = reconstructible_recovery_phase_events[rerun_idx].get("start_ts")
                if phase_start_ts is not None:
                    recovery_start_epoch = float(phase_start_ts)
            if recovery_start_epoch is None:
                inject_span = _find_step_span(spans, inject_before_step_idx)
                if inject_span is None:
                    continue
                rerun_width_sec = float(rerun_event.get("rerun_wall_time_sec", 0.0) or 0.0)
                recovery_start_epoch = float(inject_span.exec_start_epoch) - rerun_width_sec
            discarded_start_epoch = float(prev_span.exec_end_epoch)
            discarded_width_sec = recovery_start_epoch - discarded_start_epoch
            if discarded_width_sec > 0.05:
                discarded_attempt_bars.append(
                    (
                        discarded_start_epoch - batch_start_epoch,
                        discarded_width_sec,
                    )
                )

    replay_bars: list[tuple[float, float]] = []
    replay_llm_bars: list[tuple[float, float]] = []
    replay_exec_bars: list[tuple[float, float]] = []
    for rerun_event in reconstructible_rerun_events:
        resume_step_idx = int(rerun_event.get("resume_step_idx", -1))
        inject_before_step_idx = int(rerun_event.get("inject_before_step_idx", -1))
        if resume_step_idx < 0 or inject_before_step_idx <= resume_step_idx:
            continue
        replay_spans = [
            span for span in spans
            if resume_step_idx <= int(span.step_idx) < inject_before_step_idx
        ]
        if not replay_spans:
            continue
        replay_start_epoch = min(float(span.llm_start_epoch) for span in replay_spans)
        replay_end_epoch = max(float(span.exec_end_epoch) for span in replay_spans)
        replay_width_sec = replay_end_epoch - replay_start_epoch
        if replay_width_sec > 0.05:
            replay_bars.append((replay_start_epoch - batch_start_epoch, replay_width_sec))
        for span in replay_spans:
            llm_width_sec = max(0.0, float(span.llm_end_epoch) - float(span.llm_start_epoch))
            if llm_width_sec > 0.0:
                replay_llm_bars.append((float(span.llm_start_epoch) - batch_start_epoch, llm_width_sec))
            exec_width_sec = max(0.0, float(span.exec_end_epoch) - float(span.exec_start_epoch))
            if exec_width_sec > 0.0:
                replay_exec_bars.append((float(span.exec_start_epoch) - batch_start_epoch, exec_width_sec))

    first_start_sec = min(
        [start for start, _ in llm_bars]
        + [start for start, _ in exec_bars]
        + [start for start, _ in llm_only_bars]
        + [start for start, _ in checkpoint_call_bars]
        + [start for start, _ in discarded_attempt_bars]
        + [start for start, _ in replay_bars]
        + [start for start, _ in gap_bars]
        + [0.0]
    )
    last_end_sec = max(
        [start + width for start, width in llm_bars]
        + [start + width for start, width in exec_bars]
        + [start + width for start, width in llm_only_bars]
        + [start + width for start, width in checkpoint_call_bars]
        + [start + width for start, width in discarded_attempt_bars]
        + [start + width for start, width in replay_bars]
        + [start + width for start, width in gap_bars]
        + [start + width for start, width in checkpoint_ready_bars]
        + [start + width for start, width, _ in recovery_bands]
        + [0.0]
    )
    llm_total_sec = _bars_total_sec(llm_bars)
    exec_total_sec = _bars_total_sec(exec_bars)
    llm_only_total_sec = _bars_total_sec(llm_only_bars)
    checkpoint_call_total_sec = _bars_total_sec(checkpoint_call_bars)
    discarded_attempt_total_sec = _bars_total_sec(discarded_attempt_bars)
    replay_total_sec = _bars_total_sec(replay_bars)
    gap_total_sec = _bars_total_sec(gap_bars)
    checkpoint_pending_total_sec = _bars_total_sec(checkpoint_ready_bars)
    recovery_total_sec = float(sum(width_sec for _, width_sec, _ in recovery_bands))
    max_llm_seg_sec = _bars_max_sec(llm_bars)
    max_llm_only_seg_sec = _bars_max_sec(llm_only_bars)
    max_checkpoint_call_seg_sec = _bars_max_sec(checkpoint_call_bars)
    max_discarded_attempt_seg_sec = _bars_max_sec(discarded_attempt_bars)
    max_replay_seg_sec = _bars_max_sec(replay_bars)
    max_gap_seg_sec = _bars_max_sec(gap_bars)
    max_checkpoint_pending_seg_sec = _bars_max_sec(checkpoint_ready_bars)
    max_recovery_seg_sec = max((float(width_sec) for _, width_sec, _ in recovery_bands), default=0.0)
    dominant_component, dominant_component_sec = _dominant_component(
        max_llm_seg_sec=max_llm_seg_sec,
        max_llm_only_seg_sec=max_llm_only_seg_sec,
        max_checkpoint_call_seg_sec=max_checkpoint_call_seg_sec,
        max_discarded_attempt_seg_sec=max_discarded_attempt_seg_sec,
        max_replay_seg_sec=max_replay_seg_sec,
        max_gap_seg_sec=max_gap_seg_sec,
        max_checkpoint_pending_seg_sec=max_checkpoint_pending_seg_sec,
        max_recovery_seg_sec=max_recovery_seg_sec,
    )

    return TrajectoryTimeline(
        label=label,
        ok=ok,
        llm_bars=llm_bars,
        exec_bars=exec_bars,
        inferred_history_llm_bars=inferred_history_llm_bars,
        inferred_history_exec_bars=inferred_history_exec_bars,
        replay_llm_bars=replay_llm_bars,
        replay_exec_bars=replay_exec_bars,
        llm_only_bars=llm_only_bars,
        checkpoint_call_bars=checkpoint_call_bars,
        discarded_attempt_bars=discarded_attempt_bars,
        gap_bars=gap_bars,
        checkpoint_ready_bars=checkpoint_ready_bars,
        checkpoint_create_points=checkpoint_create_points,
        checkpoint_busy_points=checkpoint_busy_points,
        replay_bars=replay_bars,
        recovery_bands=recovery_bands,
        injection_points=injection_points,
        recovery_points=recovery_points,
        first_start_sec=first_start_sec,
        last_end_sec=last_end_sec,
        llm_total_sec=llm_total_sec,
        exec_total_sec=exec_total_sec,
        llm_only_total_sec=llm_only_total_sec,
        checkpoint_call_total_sec=checkpoint_call_total_sec,
        discarded_attempt_total_sec=discarded_attempt_total_sec,
        replay_total_sec=replay_total_sec,
        gap_total_sec=gap_total_sec,
        checkpoint_pending_total_sec=checkpoint_pending_total_sec,
        recovery_total_sec=recovery_total_sec,
        max_llm_seg_sec=max_llm_seg_sec,
        max_llm_only_seg_sec=max_llm_only_seg_sec,
        max_checkpoint_call_seg_sec=max_checkpoint_call_seg_sec,
        max_discarded_attempt_seg_sec=max_discarded_attempt_seg_sec,
        max_replay_seg_sec=max_replay_seg_sec,
        max_gap_seg_sec=max_gap_seg_sec,
        max_checkpoint_pending_seg_sec=max_checkpoint_pending_seg_sec,
        max_recovery_seg_sec=max_recovery_seg_sec,
        dominant_component=dominant_component,
        dominant_component_sec=dominant_component_sec,
    )


def _plot_policy(
    batch_root: Path,
    policy: str,
    output_path: Path,
    title: str,
    *,
    show_gap_bars: bool,
    show_recovery_bands: bool,
    annotate_dominant_threshold_sec: float,
) -> Path:
    policy_root = batch_root / policy
    traj_paths = _iter_traj_paths(policy_root)
    if not traj_paths:
        raise RuntimeError(f"No traj.json files found under {policy_root}")
    batch_start_epoch = _collect_batch_start_epoch(traj_paths)
    timelines = [_build_timeline(traj_path, batch_start_epoch) for traj_path in traj_paths]
    timelines.sort(key=lambda item: (item.last_end_sec - item.first_start_sec, item.label))

    fig_height = max(12.0, 0.36 * len(timelines) + 2.4)
    fig, ax = plt.subplots(figsize=(20, fig_height))

    llm_color = "#4c78a8"
    exec_color = "#f58518"
    llm_only_color = "#9d755d"
    checkpoint_call_color = "#2ca02c"
    discarded_attempt_color = "#d62728"
    replay_color = "#8c564b"
    gap_color = "#c9c9c9"
    checkpoint_pending_color = "#72b7b2"
    checkpoint_create_color = "#54a24b"
    checkpoint_busy_color = "#7f7f7f"
    recovery_band_color = "#e45756"
    injection_color = "#b279a2"
    recovery_point_color = "#222222"

    y_ticks: list[float] = []
    y_labels: list[str] = []
    dominant_notes: list[tuple[float, float, str]] = []

    for row_idx, timeline in enumerate(timelines):
        y_base = row_idx
        y_ticks.append(y_base)
        y_labels.append(timeline.label)

        if timeline.inferred_history_llm_bars:
            ax.broken_barh(
                timeline.inferred_history_llm_bars,
                (y_base - 0.34, 0.26),
                facecolors=llm_color,
                edgecolors="none",
                alpha=0.42,
                zorder=2,
            )
        if timeline.inferred_history_exec_bars:
            ax.broken_barh(
                timeline.inferred_history_exec_bars,
                (y_base - 0.03, 0.26),
                facecolors=exec_color,
                edgecolors="none",
                alpha=0.46,
                zorder=2,
            )
        if timeline.llm_bars:
            ax.broken_barh(
                timeline.llm_bars,
                (y_base - 0.34, 0.26),
                facecolors=llm_color,
                edgecolors="none",
                alpha=0.88,
                zorder=3,
            )
        if timeline.exec_bars:
            ax.broken_barh(
                timeline.exec_bars,
                (y_base - 0.03, 0.26),
                facecolors=exec_color,
                edgecolors="none",
                alpha=0.88,
                zorder=3,
            )
        if timeline.llm_only_bars:
            ax.broken_barh(
                timeline.llm_only_bars,
                (y_base - 0.19, 0.08),
                facecolors=llm_only_color,
                edgecolors="none",
                alpha=0.9,
            )
        if timeline.checkpoint_call_bars:
            ax.broken_barh(
                timeline.checkpoint_call_bars,
                (y_base + 0.18, 0.07),
                facecolors=checkpoint_call_color,
                edgecolors="none",
                alpha=0.78,
            )
        if timeline.discarded_attempt_bars:
            ax.broken_barh(
                timeline.discarded_attempt_bars,
                (y_base - 0.47, 0.08),
                facecolors=discarded_attempt_color,
                edgecolors="none",
                alpha=0.82,
            )
        if timeline.replay_bars:
            ax.broken_barh(
                timeline.replay_bars,
                (y_base - 0.93, 0.18),
                facecolors=replay_color,
                edgecolors="none",
                alpha=0.10,
                zorder=1,
            )
        if show_gap_bars and timeline.gap_bars:
            ax.broken_barh(
                timeline.gap_bars,
                (y_base + 0.41, 0.05),
                facecolors=gap_color,
                edgecolors="none",
                alpha=0.65,
            )
        if timeline.checkpoint_ready_bars:
            ax.broken_barh(
                timeline.checkpoint_ready_bars,
                (y_base + 0.28, 0.10),
                facecolors=checkpoint_pending_color,
                edgecolors="none",
                alpha=0.55,
            )
        if timeline.checkpoint_create_points:
            ax.scatter(
                timeline.checkpoint_create_points,
                [y_base + 0.33] * len(timeline.checkpoint_create_points),
                marker="^",
                s=24,
                color=checkpoint_create_color,
                zorder=5,
            )
        if timeline.checkpoint_busy_points:
            ax.scatter(
                timeline.checkpoint_busy_points,
                [y_base + 0.33] * len(timeline.checkpoint_busy_points),
                marker="x",
                s=20,
                linewidths=1.0,
                color=checkpoint_busy_color,
                zorder=5,
            )
        if show_recovery_bands:
            for recovery_start, recovery_width, recovery_label in timeline.recovery_bands:
                ax.broken_barh(
                    [(recovery_start, recovery_width)],
                    (y_base - 0.56, 0.03),
                    facecolors="none",
                    edgecolors=recovery_band_color,
                    linewidth=1.0,
                    alpha=0.95,
                    zorder=5,
                )
                if recovery_width >= 6.0:
                    ax.text(
                        recovery_start + 0.2,
                        y_base - 0.57,
                        recovery_label,
                        fontsize=5,
                        color="#6b1e1e",
                        va="top",
                        ha="left",
                    )
        if timeline.injection_points:
            ax.scatter(
                timeline.injection_points,
                [y_base - 0.55] * len(timeline.injection_points),
                marker="v",
                s=22,
                color=injection_color,
                zorder=6,
            )
        if timeline.recovery_points:
            ax.scatter(
                timeline.recovery_points,
                [y_base - 0.55] * len(timeline.recovery_points),
                marker="D",
                s=16,
                color=recovery_point_color,
                zorder=6,
            )
        if timeline.dominant_component_sec >= float(annotate_dominant_threshold_sec):
            dominant_notes.append(
                (
                    y_base,
                    timeline.last_end_sec + 0.6,
                    f"{timeline.dominant_component}~{timeline.dominant_component_sec:.1f}s",
                )
            )

    max_x = max((timeline.last_end_sec for timeline in timelines), default=1.0)
    rerun_traj_count = sum(1 for timeline in timelines if timeline.recovery_bands)
    checkpoint_traj_count = sum(1 for timeline in timelines if timeline.checkpoint_ready_bars or timeline.checkpoint_create_points)
    llm_only_traj_count = sum(1 for timeline in timelines if timeline.llm_only_bars)
    checkpoint_call_traj_count = sum(1 for timeline in timelines if timeline.checkpoint_call_bars)
    discarded_attempt_traj_count = sum(1 for timeline in timelines if timeline.discarded_attempt_bars)
    replay_traj_count = sum(1 for timeline in timelines if timeline.replay_bars)
    gap_traj_count = sum(1 for timeline in timelines if timeline.gap_bars)
    total_llm_sec = sum(item.llm_total_sec for item in timelines)
    total_exec_sec = sum(item.exec_total_sec for item in timelines)
    total_llm_only_sec = sum(item.llm_only_total_sec for item in timelines)
    total_checkpoint_call_sec = sum(item.checkpoint_call_total_sec for item in timelines)
    total_discarded_attempt_sec = sum(item.discarded_attempt_total_sec for item in timelines)
    total_replay_sec = sum(item.replay_total_sec for item in timelines)
    total_gap_sec = sum(item.gap_total_sec for item in timelines)
    total_checkpoint_pending_sec = sum(item.checkpoint_pending_total_sec for item in timelines)
    total_recovery_sec = sum(item.recovery_total_sec for item in timelines)
    max_llm_seg = max((item.max_llm_seg_sec for item in timelines), default=0.0)
    max_llm_only_seg = max((item.max_llm_only_seg_sec for item in timelines), default=0.0)
    max_checkpoint_call_seg = max((item.max_checkpoint_call_seg_sec for item in timelines), default=0.0)
    max_discarded_attempt_seg = max((item.max_discarded_attempt_seg_sec for item in timelines), default=0.0)
    max_replay_seg = max((item.max_replay_seg_sec for item in timelines), default=0.0)
    max_gap_seg = max((item.max_gap_seg_sec for item in timelines), default=0.0)
    max_ckpt_seg = max((item.max_checkpoint_pending_seg_sec for item in timelines), default=0.0)
    max_recovery_seg = max((item.max_recovery_seg_sec for item in timelines), default=0.0)
    title_suffix = (
        f"trajectories={len(timelines)} | rerun_traj={rerun_traj_count} | "
        f"checkpoint_traj={checkpoint_traj_count} | llm_only_traj={llm_only_traj_count} | "
        f"ckpt_call_traj={checkpoint_call_traj_count} | discarded_attempt_traj={discarded_attempt_traj_count} | "
        f"replay_traj={replay_traj_count} | "
        f"unaccounted_traj={gap_traj_count}\n"
        f"ΣLLM={total_llm_sec:.1f}s ΣExec={total_exec_sec:.1f}s ΣLLMOnly={total_llm_only_sec:.1f}s "
        f"ΣCkptCall={total_checkpoint_call_sec:.1f}s ΣDiscardedAttempt={total_discarded_attempt_sec:.1f}s "
        f"ΣReplay={total_replay_sec:.1f}s ΣCkptPending={total_checkpoint_pending_sec:.1f}s "
        f"ΣRecovery={total_recovery_sec:.1f}s ΣUnaccounted={total_gap_sec:.1f}s | "
        f"MaxSeg(LLM/LLMOnly/CkptCall/Discarded/Replay/CkptPending/Recovery/Unacc)="
        f"{max_llm_seg:.1f}/{max_llm_only_seg:.1f}/{max_checkpoint_call_seg:.1f}/{max_discarded_attempt_seg:.1f}/{max_replay_seg:.1f}/{max_ckpt_seg:.1f}/{max_recovery_seg:.1f}/{max_gap_seg:.1f}s"
    )

    ax.set_title(title)
    fig.text(0.5, 0.965, title_suffix, ha="center", va="center", fontsize=10)
    ax.set_xlabel("Seconds Since Policy Batch Start")
    right_pad = max(2.0, max_x * (0.14 if dominant_notes else 0.03))
    ax.set_xlim(0.0, max_x + right_pad if max_x > 0 else 1.0)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=6)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)
    ax.invert_yaxis()
    for y_base, note_x, note in dominant_notes:
        ax.text(note_x, y_base, note, fontsize=6, color="#333333", va="center", ha="left")

    legend_items = [
        Patch(facecolor=llm_color, label="LLM wait"),
        Patch(facecolor=exec_color, label="Env exec"),
        Patch(facecolor=llm_only_color, label="LLM-only turn"),
        Patch(facecolor=checkpoint_call_color, label="Checkpoint create call"),
        Patch(facecolor=discarded_attempt_color, label="Discarded attempt (rolled back)"),
        Patch(facecolor=llm_color, alpha=0.42, label="Rolled-back LLM (inferred)"),
        Patch(facecolor=exec_color, alpha=0.46, label="Rolled-back exec (inferred)"),
        Patch(facecolor=replay_color, alpha=0.18, label="Replay window"),
        Patch(facecolor=gap_color, label="Unaccounted gap (legacy/incomplete)") if show_gap_bars else None,
        Patch(facecolor=checkpoint_pending_color, label="Checkpoint pending->ready"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=checkpoint_create_color, markersize=7, label="Checkpoint create"),
        Line2D([0], [0], marker="x", color=checkpoint_busy_color, linestyle="None", markersize=7, label="Checkpoint/probe busy"),
        Line2D([0], [0], color=recovery_band_color, linewidth=1.2, label="Rerun / recovery") if show_recovery_bands else None,
        Line2D([0], [0], marker="v", color="w", markerfacecolor=injection_color, markersize=7, label="Fault inject"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=recovery_point_color, markersize=6, label="Recovery complete"),
    ]
    legend_items = [item for item in legend_items if item is not None]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.10, 0.04, 0.995, 0.95))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    breakdown_payload = {
        "policy": policy,
        "trajectory_count": len(timelines),
        "rows": [
            {
                "label": item.label,
                "ok": item.ok,
                "llm_total_sec": item.llm_total_sec,
                "exec_total_sec": item.exec_total_sec,
                "llm_only_total_sec": item.llm_only_total_sec,
                "checkpoint_call_total_sec": item.checkpoint_call_total_sec,
                "discarded_attempt_total_sec": item.discarded_attempt_total_sec,
                "replay_total_sec": item.replay_total_sec,
                "gap_total_sec": item.gap_total_sec,
                "checkpoint_pending_total_sec": item.checkpoint_pending_total_sec,
                "recovery_total_sec": item.recovery_total_sec,
                "max_llm_seg_sec": item.max_llm_seg_sec,
                "max_llm_only_seg_sec": item.max_llm_only_seg_sec,
                "max_checkpoint_call_seg_sec": item.max_checkpoint_call_seg_sec,
                "max_discarded_attempt_seg_sec": item.max_discarded_attempt_seg_sec,
                "max_replay_seg_sec": item.max_replay_seg_sec,
                "max_gap_seg_sec": item.max_gap_seg_sec,
                "max_checkpoint_pending_seg_sec": item.max_checkpoint_pending_seg_sec,
                "max_recovery_seg_sec": item.max_recovery_seg_sec,
                "dominant_component": item.dominant_component,
                "dominant_component_sec": item.dominant_component_sec,
                "first_start_sec": item.first_start_sec,
                "last_end_sec": item.last_end_sec,
            }
            for item in timelines
        ],
    }
    breakdown_path = output_path.with_suffix(".breakdown.json")
    breakdown_path.write_text(json.dumps(breakdown_payload, ensure_ascii=True, indent=2))
    return breakdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-policy gantt charts for SWE rollout batch outputs.")
    parser.add_argument("batch_root", help="Path to rollout batch root, e.g. export/swe_rollout_checkpoint_batch_4policies_...")
    parser.add_argument(
        "--policy",
        action="append",
        dest="policies",
        help="Policy to plot. Can be repeated. Defaults to all available policies under the batch root.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Defaults to <batch_root>/figures",
    )
    parser.add_argument(
        "--hide-gap-bars",
        action="store_true",
        help="Hide unaccounted gap bars (legacy trajectories without explicit phase_events).",
    )
    parser.add_argument(
        "--hide-recovery-bands",
        action="store_true",
        help="Hide rerun/recovery background bands.",
    )
    parser.add_argument(
        "--annotate-dominant-threshold-sec",
        type=float,
        default=20.0,
        help="Annotate per-trajectory dominant long segment when >= threshold seconds.",
    )
    args = parser.parse_args()

    batch_root = Path(args.batch_root).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else batch_root / "figures"

    requested_policies = args.policies or [policy for policy in POLICIES if (batch_root / policy).exists()]
    if not requested_policies:
        raise RuntimeError(f"No policy directories found under {batch_root}")

    for policy in requested_policies:
        output_path = output_dir / f"{policy}.rollout.gantt.png"
        title = f"{policy} rollout trajectory gantt"
        breakdown_path = _plot_policy(
            batch_root,
            policy,
            output_path,
            title,
            show_gap_bars=not args.hide_gap_bars,
            show_recovery_bands=not args.hide_recovery_bands,
            annotate_dominant_threshold_sec=float(args.annotate_dominant_threshold_sec),
        )
        print(output_path)
        print(breakdown_path)


if __name__ == "__main__":
    main()
