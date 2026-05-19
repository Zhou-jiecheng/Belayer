#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import replace
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


LOG_TS_RE = re.compile(r"^\[(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) ")
ALLOC_RE = re.compile(r"Allocated lease=(?P<lease>\S+)")
CLOSE_RE = re.compile(r"Closed lease=(?P<lease>\S+)")


@dataclass
class StepSpan:
    order_idx: int
    attempt_idx: int
    is_rerun_attempt: bool
    step_idx: int
    llm_start_sec: float
    llm_end_sec: float
    exec_start_sec: float
    exec_end_sec: float


@dataclass
class TrajectoryTimeline:
    label: str
    ok: bool
    spans: list[StepSpan]
    first_step_llm_bars: list[tuple[float, float]]
    llm_initial_bars: list[tuple[float, float]]
    llm_rerun_bars: list[tuple[float, float]]
    exec_initial_bars: list[tuple[float, float]]
    exec_rerun_bars: list[tuple[float, float]]
    rerun_attempt_bands: list[tuple[float, float, int]]
    checkpoint_ready_bars: list[tuple[float, float]]
    checkpoint_busy_points: list[float]
    injection_points: list[float]
    recovery_points: list[float]
    close_points: list[float]
    first_start_sec: float
    first_exec_start_sec: float
    last_end_sec: float
    llm_initial_total_sec: float
    llm_rerun_total_sec: float
    exec_initial_total_sec: float
    exec_rerun_total_sec: float
    rerun_window_total_sec: float
    checkpoint_pending_total_sec: float
    max_llm_seg_sec: float
    max_checkpoint_pending_seg_sec: float
    max_rerun_window_seg_sec: float
    dominant_component: str
    dominant_component_sec: float


def _bars_total_sec(bars: list[tuple[float, float]]) -> float:
    return float(sum(width_sec for _, width_sec in bars))


def _bars_max_sec(bars: list[tuple[float, float]]) -> float:
    return max((float(width_sec) for _, width_sec in bars), default=0.0)


def _dominant_component(
    *,
    max_llm_seg_sec: float,
    max_checkpoint_pending_seg_sec: float,
    max_rerun_window_seg_sec: float,
) -> tuple[str, float]:
    candidates = [
        ("llm", float(max_llm_seg_sec)),
        ("checkpoint_pending", float(max_checkpoint_pending_seg_sec)),
        ("rerun_window", float(max_rerun_window_seg_sec)),
    ]
    return max(candidates, key=lambda item: item[1])


def _parse_log_timestamp(line: str) -> datetime | None:
    match = LOG_TS_RE.match(line)
    if match is None:
        return None
    return datetime.strptime(match.group("ts"), "%Y-%m-%d %H:%M:%S,%f")


def parse_pool_log(pool_log_path: Path) -> tuple[float, dict[str, float], dict[str, float]]:
    lease_starts_epoch: dict[str, float] = {}
    lease_closes_epoch: dict[str, float] = {}

    with pool_log_path.open("r", encoding="utf-8", errors="replace") as f:
        for raw_line in f:
            ts = _parse_log_timestamp(raw_line)
            if ts is None:
                continue
            alloc_match = ALLOC_RE.search(raw_line)
            if alloc_match is not None:
                lease_starts_epoch.setdefault(alloc_match.group("lease"), ts.timestamp())
                continue
            close_match = CLOSE_RE.search(raw_line)
            if close_match is not None:
                lease_closes_epoch[close_match.group("lease")] = ts.timestamp()

    if not lease_starts_epoch:
        raise RuntimeError(f"No lease allocation lines found in pool log: {pool_log_path}")

    batch_start_epoch = min(lease_starts_epoch.values())
    lease_starts_sec = {
        lease_id: epoch - batch_start_epoch
        for lease_id, epoch in lease_starts_epoch.items()
    }
    lease_closes_sec = {
        lease_id: epoch - batch_start_epoch
        for lease_id, epoch in lease_closes_epoch.items()
    }
    return batch_start_epoch, lease_starts_sec, lease_closes_sec


def segment_steps(steps: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    if not steps:
        return []
    segments: list[list[dict[str, Any]]] = [[steps[0]]]
    prev_step_idx = int(steps[0].get("step_idx", -1))
    for step in steps[1:]:
        step_idx = int(step.get("step_idx", -1))
        if step_idx <= prev_step_idx:
            segments.append([])
        segments[-1].append(step)
        prev_step_idx = step_idx
    return segments


def collect_attempt_lease_ids(report: dict[str, Any]) -> list[str]:
    lease_ids: list[str] = []
    initial_lease = str(report.get("lease", {}).get("lease_id", "") or "")
    if initial_lease:
        lease_ids.append(initial_lease)

    for rerun_event in report.get("rerun_events", []):
        new_lease_id = str(rerun_event.get("new_lease", {}).get("lease_id", "") or "")
        if new_lease_id:
            lease_ids.append(new_lease_id)
    return lease_ids


def build_step_spans(
    report: dict[str, Any],
    lease_starts_sec: dict[str, float],
) -> tuple[list[StepSpan], list[float], list[float]]:
    steps = report.get("steps", [])
    segments = segment_steps(steps)
    attempt_lease_ids = collect_attempt_lease_ids(report)

    spans: list[StepSpan] = []
    segment_starts: list[float] = []
    segment_ends: list[float] = []
    next_order_idx = 0
    fallback_anchor = 0.0

    for segment_idx, segment in enumerate(segments):
        lease_id = attempt_lease_ids[segment_idx] if segment_idx < len(attempt_lease_ids) else ""
        anchor = lease_starts_sec.get(lease_id)
        if anchor is None:
            anchor = fallback_anchor
        segment_starts.append(anchor)

        cursor = anchor
        for step in segment:
            llm_wait = float(
                step.get("llm_waited_before_exec_sec", step.get("simulated_llm_delay_sec", 0.0)) or 0.0
            )
            exec_elapsed = float(step.get("exec_elapsed_sec", 0.0) or 0.0)
            llm_start = cursor
            llm_end = llm_start + llm_wait
            exec_start = llm_end
            exec_end = exec_start + exec_elapsed
            spans.append(
                StepSpan(
                    order_idx=next_order_idx,
                    attempt_idx=int(step.get("attempt_idx", segment_idx)),
                    is_rerun_attempt=bool(step.get("is_rerun_attempt", segment_idx > 0)),
                    step_idx=int(step.get("step_idx", -1)),
                    llm_start_sec=llm_start,
                    llm_end_sec=llm_end,
                    exec_start_sec=exec_start,
                    exec_end_sec=exec_end,
                )
            )
            next_order_idx += 1
            cursor = exec_end

        segment_ends.append(cursor)
        fallback_anchor = cursor

    return spans, segment_starts, segment_ends


def _find_next_step_span(spans: list[StepSpan], step_idx: int, used_until: int) -> tuple[StepSpan | None, int]:
    for idx in range(used_until + 1, len(spans)):
        if spans[idx].step_idx == step_idx:
            return spans[idx], idx
    for idx, span in enumerate(spans):
        if span.step_idx == step_idx:
            return span, idx
    return None, used_until


def build_timeline(
    report_path: Path,
    batch_start_epoch: float,
    lease_starts_sec: dict[str, float],
    lease_closes_sec: dict[str, float],
) -> TrajectoryTimeline:
    report = json.loads(report_path.read_text())
    traj_label = report_path.stem
    ok = bool(report.get("ok", False))

    if "steps" not in report:
        lease_id = str(report.get("lease", {}).get("lease_id", "") or "")
        anchor = lease_starts_sec.get(lease_id, 0.0)
        close_sec = lease_closes_sec.get(lease_id, anchor)
        return TrajectoryTimeline(
            label=f"{traj_label} [failed]",
            ok=False,
            spans=[],
            first_step_llm_bars=[],
            llm_initial_bars=[],
            llm_rerun_bars=[],
            exec_initial_bars=[],
            exec_rerun_bars=[],
            rerun_attempt_bands=[],
            checkpoint_ready_bars=[],
            checkpoint_busy_points=[],
            injection_points=[],
            recovery_points=[],
            close_points=[close_sec] if lease_id in lease_closes_sec else [],
            first_start_sec=anchor,
            first_exec_start_sec=anchor,
            last_end_sec=close_sec,
            llm_initial_total_sec=0.0,
            llm_rerun_total_sec=0.0,
            exec_initial_total_sec=0.0,
            exec_rerun_total_sec=0.0,
            rerun_window_total_sec=0.0,
            checkpoint_pending_total_sec=0.0,
            max_llm_seg_sec=0.0,
            max_checkpoint_pending_seg_sec=0.0,
            max_rerun_window_seg_sec=0.0,
            dominant_component="none",
            dominant_component_sec=0.0,
        )

    spans, segment_starts, segment_ends = build_step_spans(report, lease_starts_sec)
    first_step_llm_bars = [
        (span.llm_start_sec, span.llm_end_sec - span.llm_start_sec)
        for span in spans
        if span.llm_end_sec > span.llm_start_sec and not span.is_rerun_attempt and span.order_idx == 0
    ]
    llm_initial_bars = [
        (span.llm_start_sec, span.llm_end_sec - span.llm_start_sec)
        for span in spans
        if span.llm_end_sec > span.llm_start_sec and not span.is_rerun_attempt and span.order_idx != 0
    ]
    llm_rerun_bars = [
        (span.llm_start_sec, span.llm_end_sec - span.llm_start_sec)
        for span in spans
        if span.llm_end_sec > span.llm_start_sec and span.is_rerun_attempt
    ]
    exec_initial_bars = [
        (span.exec_start_sec, span.exec_end_sec - span.exec_start_sec)
        for span in spans
        if span.exec_end_sec > span.exec_start_sec and not span.is_rerun_attempt
    ]
    exec_rerun_bars = [
        (span.exec_start_sec, span.exec_end_sec - span.exec_start_sec)
        for span in spans
        if span.exec_end_sec > span.exec_start_sec and span.is_rerun_attempt
    ]
    rerun_attempt_bands = [
        (segment_start, max(0.0, segment_end - segment_start), attempt_idx)
        for attempt_idx, (segment_start, segment_end) in enumerate(zip(segment_starts, segment_ends))
        if attempt_idx > 0 and segment_end > segment_start
    ]

    checkpoint_ready_bars: list[tuple[float, float]] = []
    checkpoint_busy_points: list[float] = []
    used_step_idx = -1
    for event in report.get("checkpoint_events", []):
        step_span, used_step_idx = _find_next_step_span(spans, int(event.get("step_idx", -1)), used_step_idx)
        status_result = event.get("status_result")
        if isinstance(status_result, dict) and status_result.get("status") == "ready":
            created_at = status_result.get("created_at")
            ready_at = status_result.get("ready_at")
            if created_at is not None and ready_at is not None:
                start_sec = float(created_at) - batch_start_epoch
                duration_sec = max(0.0, float(ready_at) - float(created_at))
                checkpoint_ready_bars.append((start_sec, duration_sec))
            continue

        if step_span is None:
            continue
        if event.get("decision_type") == "adaptive_llm_wait":
            waited = float(event.get("waited_before_checkpoint_sec", 0.0) or 0.0)
            event_sec = min(step_span.llm_end_sec, step_span.llm_start_sec + waited)
        else:
            event_sec = step_span.exec_end_sec

        if event.get("skip_reason") in {"checkpoint_busy", "probe_busy"}:
            checkpoint_busy_points.append(event_sec)

    injection_points: list[float] = []
    injection_target = report.get("injection_target")
    if isinstance(injection_target, dict) and "inject_before_step_idx" in injection_target:
        inject_step_idx = int(injection_target["inject_before_step_idx"])
        for span in spans:
            if span.step_idx == inject_step_idx:
                injection_points.append(span.llm_start_sec)
                break

    recovery_points: list[float] = []
    for rerun_event in report.get("rerun_events", []):
        new_lease_id = str(rerun_event.get("new_lease", {}).get("lease_id", "") or "")
        if new_lease_id and new_lease_id in lease_starts_sec:
            recovery_points.append(lease_starts_sec[new_lease_id])

    close_points: list[float] = []
    for lease_id in collect_attempt_lease_ids(report):
        if lease_id in lease_closes_sec:
            close_points.append(lease_closes_sec[lease_id])

    first_start_sec = min(segment_starts) if segment_starts else 0.0
    first_exec_start_sec = min((span.exec_start_sec for span in spans), default=first_start_sec)
    last_end_candidates = segment_ends + close_points
    last_end_sec = max(last_end_candidates) if last_end_candidates else first_start_sec
    llm_initial_total_sec = _bars_total_sec(first_step_llm_bars) + _bars_total_sec(llm_initial_bars)
    llm_rerun_total_sec = _bars_total_sec(llm_rerun_bars)
    exec_initial_total_sec = _bars_total_sec(exec_initial_bars)
    exec_rerun_total_sec = _bars_total_sec(exec_rerun_bars)
    rerun_window_total_sec = float(sum(width_sec for _, width_sec, _ in rerun_attempt_bands))
    checkpoint_pending_total_sec = _bars_total_sec(checkpoint_ready_bars)
    max_llm_seg_sec = max(_bars_max_sec(first_step_llm_bars), _bars_max_sec(llm_initial_bars), _bars_max_sec(llm_rerun_bars))
    max_checkpoint_pending_seg_sec = _bars_max_sec(checkpoint_ready_bars)
    max_rerun_window_seg_sec = max((float(width_sec) for _, width_sec, _ in rerun_attempt_bands), default=0.0)
    dominant_component, dominant_component_sec = _dominant_component(
        max_llm_seg_sec=max_llm_seg_sec,
        max_checkpoint_pending_seg_sec=max_checkpoint_pending_seg_sec,
        max_rerun_window_seg_sec=max_rerun_window_seg_sec,
    )

    label = traj_label if ok else f"{traj_label} [failed]"
    return TrajectoryTimeline(
        label=label,
        ok=ok,
        spans=spans,
        first_step_llm_bars=first_step_llm_bars,
        llm_initial_bars=llm_initial_bars,
        llm_rerun_bars=llm_rerun_bars,
        exec_initial_bars=exec_initial_bars,
        exec_rerun_bars=exec_rerun_bars,
        rerun_attempt_bands=rerun_attempt_bands,
        checkpoint_ready_bars=checkpoint_ready_bars,
        checkpoint_busy_points=checkpoint_busy_points,
        injection_points=injection_points,
        recovery_points=recovery_points,
        close_points=close_points,
        first_start_sec=first_start_sec,
        first_exec_start_sec=first_exec_start_sec,
        last_end_sec=last_end_sec,
        llm_initial_total_sec=llm_initial_total_sec,
        llm_rerun_total_sec=llm_rerun_total_sec,
        exec_initial_total_sec=exec_initial_total_sec,
        exec_rerun_total_sec=exec_rerun_total_sec,
        rerun_window_total_sec=rerun_window_total_sec,
        checkpoint_pending_total_sec=checkpoint_pending_total_sec,
        max_llm_seg_sec=max_llm_seg_sec,
        max_checkpoint_pending_seg_sec=max_checkpoint_pending_seg_sec,
        max_rerun_window_seg_sec=max_rerun_window_seg_sec,
        dominant_component=dominant_component,
        dominant_component_sec=dominant_component_sec,
    )


def sort_timelines(timelines: list[TrajectoryTimeline], sort_by: str) -> list[TrajectoryTimeline]:
    if sort_by == "duration":
        return sorted(timelines, key=lambda item: (item.last_end_sec - item.first_start_sec, item.label))
    if sort_by == "name":
        return sorted(timelines, key=lambda item: item.label)
    return sorted(timelines, key=lambda item: (item.first_start_sec, item.label))


def _shift_bars(bars: list[tuple[float, float]], delta_sec: float) -> list[tuple[float, float]]:
    return [(start_sec - delta_sec, width_sec) for start_sec, width_sec in bars]


def _shift_points(points: list[float], delta_sec: float) -> list[float]:
    return [point_sec - delta_sec for point_sec in points]


def align_timeline(timeline: TrajectoryTimeline, origin: str) -> TrajectoryTimeline:
    if origin == "lease":
        return timeline
    if origin != "first-exec":
        raise ValueError(f"Unsupported align origin: {origin}")

    delta_sec = timeline.first_exec_start_sec
    shifted_spans = [
        replace(
            span,
            llm_start_sec=span.llm_start_sec - delta_sec,
            llm_end_sec=span.llm_end_sec - delta_sec,
            exec_start_sec=span.exec_start_sec - delta_sec,
            exec_end_sec=span.exec_end_sec - delta_sec,
        )
        for span in timeline.spans
    ]
    return replace(
        timeline,
        spans=shifted_spans,
        first_step_llm_bars=_shift_bars(timeline.first_step_llm_bars, delta_sec),
        llm_initial_bars=_shift_bars(timeline.llm_initial_bars, delta_sec),
        llm_rerun_bars=_shift_bars(timeline.llm_rerun_bars, delta_sec),
        exec_initial_bars=_shift_bars(timeline.exec_initial_bars, delta_sec),
        exec_rerun_bars=_shift_bars(timeline.exec_rerun_bars, delta_sec),
        rerun_attempt_bands=[
            (start_sec - delta_sec, width_sec, attempt_idx)
            for start_sec, width_sec, attempt_idx in timeline.rerun_attempt_bands
        ],
        checkpoint_ready_bars=_shift_bars(timeline.checkpoint_ready_bars, delta_sec),
        checkpoint_busy_points=_shift_points(timeline.checkpoint_busy_points, delta_sec),
        injection_points=_shift_points(timeline.injection_points, delta_sec),
        recovery_points=_shift_points(timeline.recovery_points, delta_sec),
        close_points=_shift_points(timeline.close_points, delta_sec),
        first_start_sec=timeline.first_start_sec - delta_sec,
        first_exec_start_sec=0.0,
        last_end_sec=timeline.last_end_sec - delta_sec,
    )


def plot_timelines(
    experiment_root: Path,
    policy: str,
    timelines: list[TrajectoryTimeline],
    output_path: Path,
    title: str,
    *,
    align_origin: str = "lease",
    split_first_step_llm: bool = False,
    show_rerun_window: bool = True,
    annotate_dominant_threshold_sec: float = 20.0,
) -> None:
    timelines = [align_timeline(timeline, align_origin) for timeline in timelines]
    timelines = sort_timelines(timelines, sort_by="duration" if align_origin == "first-exec" else "start")
    figure_height = max(10.0, 0.42 * len(timelines) + 2.2)
    fig, ax = plt.subplots(figsize=(18, figure_height))

    first_step_llm_color = "#d9d9d9"
    llm_initial_color = "#4c78a8"
    llm_rerun_color = "#72b7b2"
    exec_initial_color = "#f58518"
    exec_rerun_color = "#e45756"
    rerun_band_color = "#f3efe3"
    ready_color = "#54a24b"
    busy_color = "#7f7f7f"
    inject_color = "#b279a2"
    recovery_color = "#222222"
    close_color = "#9d9d9d"

    y_ticks: list[float] = []
    y_labels: list[str] = []
    dominant_notes: list[tuple[float, float, str]] = []

    for row_idx, timeline in enumerate(timelines):
        y_base = row_idx
        y_ticks.append(y_base)
        y_labels.append(timeline.label)

        if show_rerun_window:
            for band_start, band_width, attempt_idx in timeline.rerun_attempt_bands:
                ax.broken_barh(
                    [(band_start, band_width)],
                    (y_base - 0.42, 0.90),
                    facecolors=rerun_band_color,
                    edgecolors="none",
                    alpha=0.45,
                    zorder=0,
                )
                ax.text(
                    band_start + 0.15,
                    y_base - 0.36,
                    f"rerun#{attempt_idx}",
                    fontsize=6,
                    color="#6b5a36",
                    va="top",
                    ha="left",
                    zorder=7,
                )

        if split_first_step_llm and timeline.first_step_llm_bars:
            ax.broken_barh(
                timeline.first_step_llm_bars,
                (y_base - 0.34, 0.28),
                facecolors=first_step_llm_color,
                edgecolors="none",
                alpha=0.95,
            )
        if timeline.llm_initial_bars:
            ax.broken_barh(
                timeline.llm_initial_bars,
                (y_base - 0.34, 0.28),
                facecolors=llm_initial_color,
                edgecolors="none",
                alpha=0.88,
            )
        if timeline.llm_rerun_bars:
            ax.broken_barh(
                timeline.llm_rerun_bars,
                (y_base - 0.34, 0.28),
                facecolors=llm_rerun_color,
                edgecolors="none",
                alpha=0.92,
            )
        if timeline.exec_initial_bars:
            ax.broken_barh(
                timeline.exec_initial_bars,
                (y_base - 0.03, 0.28),
                facecolors=exec_initial_color,
                edgecolors="none",
                alpha=0.88,
            )
        if timeline.exec_rerun_bars:
            ax.broken_barh(
                timeline.exec_rerun_bars,
                (y_base - 0.03, 0.28),
                facecolors=exec_rerun_color,
                edgecolors="none",
                alpha=0.88,
            )
        if timeline.checkpoint_ready_bars:
            ax.broken_barh(
                timeline.checkpoint_ready_bars,
                (y_base + 0.28, 0.10),
                facecolors=ready_color,
                edgecolors="none",
                alpha=0.95,
            )
        if timeline.checkpoint_busy_points:
            ax.scatter(
                timeline.checkpoint_busy_points,
                [y_base + 0.33] * len(timeline.checkpoint_busy_points),
                marker="x",
                s=26,
                linewidths=1.0,
                color=busy_color,
                zorder=5,
            )
        if timeline.injection_points:
            ax.scatter(
                timeline.injection_points,
                [y_base - 0.18] * len(timeline.injection_points),
                marker="v",
                s=36,
                color=inject_color,
                zorder=6,
            )
        if timeline.recovery_points:
            ax.scatter(
                timeline.recovery_points,
                [y_base - 0.18] * len(timeline.recovery_points),
                marker="D",
                s=26,
                color=recovery_color,
                zorder=6,
            )
        if timeline.close_points:
            ax.scatter(
                timeline.close_points,
                [y_base + 0.05] * len(timeline.close_points),
                marker="|",
                s=140,
                linewidths=1.2,
                color=close_color,
                zorder=5,
            )
        if timeline.dominant_component_sec >= float(annotate_dominant_threshold_sec):
            dominant_notes.append(
                (
                    y_base,
                    timeline.last_end_sec + 0.6,
                    f"{timeline.dominant_component}~{timeline.dominant_component_sec:.1f}s",
                )
            )

    ax.set_title(title)
    if align_origin == "first-exec":
        ax.set_xlabel("Seconds Since First Exec Start (Per Trajectory)")
    else:
        ax.set_xlabel("Seconds Since First Lease Allocation")
    ax.set_ylabel("Trajectory")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7)
    ax.invert_yaxis()
    ax.grid(True, axis="x", linestyle="--", alpha=0.25)
    min_x = min((timeline.first_start_sec for timeline in timelines), default=0.0)
    max_x = max((timeline.last_end_sec for timeline in timelines), default=1.0)
    right_pad = max(2.0, max_x * (0.14 if dominant_notes else 0.03))
    ax.set_xlim(min(0.0, min_x), max_x + right_pad)
    for y_base, note_x, note in dominant_notes:
        ax.text(note_x, y_base, note, fontsize=6, color="#333333", va="center", ha="left")

    legend_handles = [
        Patch(facecolor=first_step_llm_color, label="LLM wait before first exec")
        if split_first_step_llm
        else Patch(facecolor=llm_initial_color, label="LLM wait (initial)"),
        Patch(facecolor=llm_initial_color, label="LLM wait (initial, later steps)")
        if split_first_step_llm
        else None,
        Patch(facecolor=llm_rerun_color, label="LLM wait (rerun)"),
        Patch(facecolor=exec_initial_color, label="Exec (initial)"),
        Patch(facecolor=exec_rerun_color, label="Exec (rerun)"),
        Patch(facecolor=rerun_band_color, label="Rerun attempt window") if show_rerun_window else None,
        Patch(facecolor=ready_color, label="Checkpoint create->ready"),
        Line2D([0], [0], marker="x", color=busy_color, linestyle="None", label="Checkpoint busy"),
        Line2D([0], [0], marker="v", color=inject_color, linestyle="None", label="Fault injection"),
        Line2D([0], [0], marker="D", color=recovery_color, linestyle="None", label="Recovery / new lease"),
        Line2D([0], [0], marker="|", color=close_color, linestyle="None", label="Lease close"),
    ]
    legend_handles = [handle for handle in legend_handles if handle is not None]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=9, frameon=True)

    ok_count = sum(1 for item in timelines if item.ok)
    total_llm_initial = sum(item.llm_initial_total_sec for item in timelines)
    total_llm_rerun = sum(item.llm_rerun_total_sec for item in timelines)
    total_exec_initial = sum(item.exec_initial_total_sec for item in timelines)
    total_exec_rerun = sum(item.exec_rerun_total_sec for item in timelines)
    total_checkpoint_pending = sum(item.checkpoint_pending_total_sec for item in timelines)
    total_rerun_window = sum(item.rerun_window_total_sec for item in timelines)
    max_llm_seg = max((item.max_llm_seg_sec for item in timelines), default=0.0)
    max_ckpt_seg = max((item.max_checkpoint_pending_seg_sec for item in timelines), default=0.0)
    max_rerun_window_seg = max((item.max_rerun_window_seg_sec for item in timelines), default=0.0)
    stat_text = (
        f"Policy: {policy}\n"
        f"Trajectories: {len(timelines)}\n"
        f"OK: {ok_count}\n"
        f"Failed: {len(timelines) - ok_count}\n"
        f"ΣLLM(init/rerun): {total_llm_initial:.1f}s / {total_llm_rerun:.1f}s\n"
        f"ΣExec(init/rerun): {total_exec_initial:.1f}s / {total_exec_rerun:.1f}s\n"
        f"ΣCkptPending: {total_checkpoint_pending:.1f}s\n"
        f"ΣRerunWindow: {total_rerun_window:.1f}s\n"
        f"Max seg (LLM/Ckpt/RerunWin): {max_llm_seg:.1f}s / {max_ckpt_seg:.1f}s / {max_rerun_window_seg:.1f}s"
    )
    ax.text(
        0.01,
        0.99,
        stat_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "#bbbbbb"},
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_breakdown_json(
    output_path: Path,
    policy: str,
    align_origin: str,
    timelines: list[TrajectoryTimeline],
) -> Path:
    rows: list[dict[str, Any]] = []
    for item in timelines:
        rows.append(
            {
                "label": item.label,
                "ok": item.ok,
                "llm_initial_total_sec": item.llm_initial_total_sec,
                "llm_rerun_total_sec": item.llm_rerun_total_sec,
                "exec_initial_total_sec": item.exec_initial_total_sec,
                "exec_rerun_total_sec": item.exec_rerun_total_sec,
                "checkpoint_pending_total_sec": item.checkpoint_pending_total_sec,
                "rerun_window_total_sec": item.rerun_window_total_sec,
                "max_llm_seg_sec": item.max_llm_seg_sec,
                "max_checkpoint_pending_seg_sec": item.max_checkpoint_pending_seg_sec,
                "max_rerun_window_seg_sec": item.max_rerun_window_seg_sec,
                "dominant_component": item.dominant_component,
                "dominant_component_sec": item.dominant_component_sec,
                "first_start_sec": item.first_start_sec,
                "first_exec_start_sec": item.first_exec_start_sec,
                "last_end_sec": item.last_end_sec,
            }
        )
    payload = {
        "policy": policy,
        "align_origin": align_origin,
        "trajectory_count": len(timelines),
        "rows": rows,
    }
    breakdown_path = output_path.with_suffix(".breakdown.json")
    breakdown_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))
    return breakdown_path


def load_timelines(experiment_root: Path, policy: str) -> list[TrajectoryTimeline]:
    pool_log_path = experiment_root / "swe_env_pool_server.log"
    batch_start_epoch, lease_starts_sec, lease_closes_sec = parse_pool_log(pool_log_path)

    per_traj_dir = experiment_root / policy / "per_traj"
    if not per_traj_dir.exists():
        raise FileNotFoundError(f"Per-trajectory directory not found: {per_traj_dir}")

    timelines = [
        build_timeline(report_path, batch_start_epoch, lease_starts_sec, lease_closes_sec)
        for report_path in sorted(per_traj_dir.glob("*.json"))
    ]
    if not timelines:
        raise RuntimeError(f"No per-trajectory reports found in: {per_traj_dir}")
    return timelines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot one policy of a SWE checkpoint fault experiment as a trajectory Gantt chart."
    )
    parser.add_argument("experiment_root", help="Path to experiment output root")
    parser.add_argument(
        "--policy",
        required=True,
        choices=["oracle-no-fault-no-checkpoint", "never", "always", "adaptive-risk"],
        help="Policy subdirectory to visualize",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="Output PNG path. Defaults to <experiment_root>/figures/<policy>.gantt.png",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Defaults to '<policy> trajectory gantt'.",
    )
    parser.add_argument(
        "--align-origin",
        choices=["lease", "first-exec"],
        default="lease",
        help="Time origin for each trajectory. 'lease' preserves batch timing; 'first-exec' normalizes away the initial LLM wait.",
    )
    parser.add_argument(
        "--split-first-step-llm",
        action="store_true",
        help="Render the initial LLM wait before the first exec as a separate light-gray band.",
    )
    parser.add_argument(
        "--hide-rerun-window",
        action="store_true",
        help="Hide the wide rerun-attempt background bands to avoid confusing them with checkpoint cost.",
    )
    parser.add_argument(
        "--annotate-dominant-threshold-sec",
        type=float,
        default=20.0,
        help="Annotate per-trajectory dominant long segment when >= threshold seconds.",
    )
    args = parser.parse_args()

    experiment_root = Path(args.experiment_root).expanduser().resolve()
    if not experiment_root.exists():
        raise FileNotFoundError(f"Experiment root not found: {experiment_root}")

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else experiment_root / "figures" / f"{args.policy}.gantt.png"
    )

    timelines = load_timelines(experiment_root, args.policy)
    title = args.title or f"{args.policy} trajectory gantt"
    plot_timelines(
        experiment_root,
        args.policy,
        timelines,
        output_path,
        title,
        align_origin=args.align_origin,
        split_first_step_llm=args.split_first_step_llm,
        show_rerun_window=not args.hide_rerun_window,
        annotate_dominant_threshold_sec=float(args.annotate_dominant_threshold_sec),
    )
    breakdown_path = write_breakdown_json(
        output_path=output_path,
        policy=args.policy,
        align_origin=args.align_origin,
        timelines=[align_timeline(timeline, args.align_origin) for timeline in timelines],
    )

    print(f"Wrote plot to: {output_path}")
    print(f"Wrote breakdown: {breakdown_path}")
    print(f"Policy: {args.policy}")
    print(f"Trajectories: {len(timelines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
