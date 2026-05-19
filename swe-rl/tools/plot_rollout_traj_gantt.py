#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


@dataclass
class StepSpan:
    step_idx: int
    llm_start_sec: float
    llm_end_sec: float
    exec_start_sec: float
    exec_end_sec: float
    action: str


def build_step_spans(step_debug: list[dict[str, Any]]) -> list[StepSpan]:
    spans: list[StepSpan] = []
    cursor = 0.0
    for step in step_debug:
        llm_wait = float(step.get("llm_waited_before_exec_sec", step.get("llm_elapsed", 0.0)) or 0.0)
        exec_elapsed = float(step.get("exec_elapsed_sec", step.get("elapsed", 0.0)) or 0.0)
        llm_start = cursor
        llm_end = llm_start + llm_wait
        exec_start = llm_end
        exec_end = exec_start + exec_elapsed
        spans.append(
            StepSpan(
                step_idx=int(step.get("step_idx", len(spans))),
                llm_start_sec=llm_start,
                llm_end_sec=llm_end,
                exec_start_sec=exec_start,
                exec_end_sec=exec_end,
                action=str(step.get("action", "")),
            )
        )
        cursor = exec_end
    return spans


def find_step_span(spans: list[StepSpan], step_idx: int) -> StepSpan | None:
    for span in spans:
        if span.step_idx == step_idx:
            return span
    return None


def checkpoint_event_time_sec(event: dict[str, Any], spans: list[StepSpan]) -> float | None:
    target_step_idx = event.get("during_llm_wait_for_step_idx")
    if target_step_idx is not None:
        span = find_step_span(spans, int(target_step_idx))
        if span is not None:
            waited = float(event.get("waited_before_checkpoint_sec", 0.0) or 0.0)
            return min(span.llm_end_sec, span.llm_start_sec + waited)

    step_idx = event.get("step_idx")
    if step_idx is None:
        return None
    span = find_step_span(spans, int(step_idx))
    if span is None:
        return None
    if event.get("decision_type") == "adaptive_llm_wait":
        waited = float(event.get("waited_before_checkpoint_sec", 0.0) or 0.0)
        return min(span.llm_end_sec, span.llm_start_sec + waited)
    return span.exec_end_sec


def checkpoint_ready_status_result(event: dict[str, Any]) -> dict[str, Any] | None:
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


def checkpoint_call_bars_from_events(checkpoint_events: list[dict[str, Any]]) -> list[tuple[float, float]]:
    bars: list[tuple[float, float]] = []
    for event in checkpoint_events:
        if event.get("event") != "checkpoint_create":
            continue
        start_ts = event.get("create_call_start_ts")
        end_ts = event.get("create_call_end_ts")
        elapsed_sec = event.get("create_call_elapsed_sec")
        if start_ts is None:
            continue
        start_sec = float(start_ts)
        if end_ts is not None:
            end_sec = max(start_sec, float(end_ts))
        elif elapsed_sec is not None:
            end_sec = start_sec + max(0.0, float(elapsed_sec))
        else:
            continue
        bars.append((start_sec, end_sec - start_sec))
    return bars


def infer_trajectory_start_epoch(
    checkpoint_events: list[dict[str, Any]], spans: list[StepSpan]
) -> float | None:
    anchors: list[float] = []
    for event in checkpoint_events:
        create_call_start_ts = event.get("create_call_start_ts")
        event_sec = checkpoint_event_time_sec(event, spans)
        if create_call_start_ts is not None and event_sec is not None:
            anchors.append(float(create_call_start_ts) - event_sec)
        status_result = checkpoint_ready_status_result(event)
        if status_result is None:
            continue
        created_at = status_result.get("created_at")
        checkpoint_id = status_result.get("checkpoint_id")
        if created_at is None or not checkpoint_id:
            continue

        create_event = None
        for candidate in checkpoint_events:
            create_result = candidate.get("create_result")
            if (
                candidate.get("event") == "checkpoint_create"
                and isinstance(create_result, dict)
                and create_result.get("checkpoint_id") == checkpoint_id
            ):
                create_event = candidate
                break
        if create_event is None:
            continue
        ready_event_sec = checkpoint_event_time_sec(create_event, spans)
        if ready_event_sec is None:
            continue
        anchors.append(float(created_at) - ready_event_sec)
    if not anchors:
        return None
    return statistics.median(anchors)


def plot_traj(report_path: Path, output_path: Path, title: str) -> None:
    report = json.loads(report_path.read_text())
    spans = build_step_spans(report.get("step_debug", []))
    if not spans:
        raise RuntimeError(f"No step_debug found in {report_path}")

    checkpoint_events = list(report.get("checkpoint_events", []))
    failure_events = list(report.get("failure_events", []))
    rerun_events = list(report.get("rerun_events", []))
    start_epoch = infer_trajectory_start_epoch(checkpoint_events, spans)

    fig, ax = plt.subplots(figsize=(18, 6.8))

    lane_y = {
        "recovery": 8.0,
        "checkpoint": 18.0,
        "steps": 32.0,
    }
    lane_h = {
        "recovery": 6.0,
        "checkpoint": 8.0,
        "steps": 12.0,
    }

    llm_color = "#4c78a8"
    exec_color = "#f58518"
    checkpoint_pending_color = "#72b7b2"
    checkpoint_ready_color = "#54a24b"
    checkpoint_busy_color = "#7f7f7f"
    recovery_color = "#e45756"
    inject_color = "#b279a2"
    recovery_done_color = "#222222"

    for span in spans:
        llm_width = span.llm_end_sec - span.llm_start_sec
        exec_width = span.exec_end_sec - span.exec_start_sec
        if llm_width > 0:
            ax.broken_barh(
                [(span.llm_start_sec, llm_width)],
                (lane_y["steps"], lane_h["steps"]),
                facecolors=llm_color,
                edgecolors="none",
                alpha=0.85,
            )
        if exec_width > 0:
            ax.broken_barh(
                [(span.exec_start_sec, exec_width)],
                (lane_y["steps"], lane_h["steps"]),
                facecolors=exec_color,
                edgecolors="none",
                alpha=0.95,
            )
            ax.text(
                span.exec_start_sec + exec_width / 2.0,
                lane_y["steps"] + lane_h["steps"] / 2.0,
                str(span.step_idx),
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    checkpoint_create_points: list[float] = []
    checkpoint_busy_points: list[float] = []
    checkpoint_call_bars_abs = checkpoint_call_bars_from_events(checkpoint_events)
    for event in checkpoint_events:
        event_sec = checkpoint_event_time_sec(event, spans)
        if event_sec is not None and event.get("event") == "checkpoint_create":
            checkpoint_create_points.append(event_sec)
        if event.get("skip_reason") in {"checkpoint_busy", "probe_busy"} and event_sec is not None:
            checkpoint_busy_points.append(event_sec)
        status_result = checkpoint_ready_status_result(event)
        if status_result is None:
            continue
        created_at = status_result.get("created_at")
        ready_at = status_result.get("ready_at")
        if created_at is None or ready_at is None or start_epoch is None:
            continue
        pending_start = float(created_at) - start_epoch
        pending_width = max(0.0, float(ready_at) - float(created_at))
        if pending_width > 0:
            ax.broken_barh(
                [(pending_start, pending_width)],
                (lane_y["checkpoint"], lane_h["checkpoint"]),
                facecolors=checkpoint_pending_color,
                edgecolors="none",
                alpha=0.55,
            )
        ax.scatter(
            [float(ready_at) - start_epoch],
            [lane_y["checkpoint"] + lane_h["checkpoint"] / 2.0],
            marker="s",
            s=28,
            color=checkpoint_ready_color,
            zorder=5,
        )

    if checkpoint_create_points:
        ax.scatter(
            checkpoint_create_points,
            [lane_y["checkpoint"] + lane_h["checkpoint"] / 2.0] * len(checkpoint_create_points),
            marker="^",
            s=42,
            color=checkpoint_ready_color,
            zorder=5,
        )
    if checkpoint_busy_points:
        ax.scatter(
            checkpoint_busy_points,
            [lane_y["checkpoint"] + lane_h["checkpoint"] / 2.0] * len(checkpoint_busy_points),
            marker="x",
            s=28,
            color=checkpoint_busy_color,
            zorder=6,
        )
    if checkpoint_call_bars_abs and start_epoch is not None:
        checkpoint_call_bars = [
            (start_sec - start_epoch, width_sec)
            for start_sec, width_sec in checkpoint_call_bars_abs
            if start_sec >= start_epoch
        ]
        if checkpoint_call_bars:
            ax.broken_barh(
                checkpoint_call_bars,
                (lane_y["checkpoint"], lane_h["checkpoint"]),
                facecolors=checkpoint_ready_color,
                edgecolors="none",
                alpha=0.35,
            )

    recovery_windows: list[tuple[float, float, str]] = []
    injection_points: list[float] = []
    recovery_done_points: list[float] = []
    for rerun_event in rerun_events:
        inject_step_idx = int(rerun_event.get("inject_before_step_idx", -1))
        span = find_step_span(spans, inject_step_idx)
        rerun_wall_time_sec = float(rerun_event.get("rerun_wall_time_sec", 0.0) or 0.0)
        if span is None:
            continue
        recovery_end = span.exec_start_sec
        recovery_start = max(0.0, recovery_end - rerun_wall_time_sec)
        recovery_mode = str(rerun_event.get("recovery_mode", "recovery"))
        label = recovery_mode
        latest_ready_step = rerun_event.get("latest_ready_checkpoint_step")
        if recovery_mode == "checkpoint_rerun" and latest_ready_step is not None:
            label = f"checkpoint_rerun ({latest_ready_step} -> {inject_step_idx})"
        recovery_windows.append((recovery_start, max(0.0, recovery_end - recovery_start), label))
        injection_points.append(recovery_start)
        recovery_done_points.append(recovery_end)

    for start_sec, width_sec, label in recovery_windows:
        ax.broken_barh(
            [(start_sec, width_sec)],
            (lane_y["recovery"], lane_h["recovery"]),
            facecolors=recovery_color,
            edgecolors="none",
            alpha=0.45,
        )
        if width_sec > 0:
            ax.text(
                start_sec + width_sec / 2.0,
                lane_y["recovery"] + lane_h["recovery"] / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    if not injection_points:
        for failure_event in failure_events:
            inject_step_idx = int(failure_event.get("inject_before_step_idx", -1))
            span = find_step_span(spans, inject_step_idx)
            if span is not None:
                injection_points.append(span.exec_start_sec)

    if injection_points:
        ax.scatter(
            injection_points,
            [lane_y["recovery"] + lane_h["recovery"] / 2.0] * len(injection_points),
            marker="v",
            s=44,
            color=inject_color,
            zorder=6,
        )
    if recovery_done_points:
        ax.scatter(
            recovery_done_points,
            [lane_y["recovery"] + lane_h["recovery"] / 2.0] * len(recovery_done_points),
            marker="D",
            s=36,
            color=recovery_done_color,
            zorder=6,
        )

    y_ticks = [
        lane_y["recovery"] + lane_h["recovery"] / 2.0,
        lane_y["checkpoint"] + lane_h["checkpoint"] / 2.0,
        lane_y["steps"] + lane_h["steps"] / 2.0,
    ]
    y_labels = [
        "Recovery / rerun",
        "Checkpoint pending->ready",
        "LLM wait + exec",
    ]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)

    max_x = max(
        max(span.exec_end_sec for span in spans),
        max((start + width for start, width, _ in recovery_windows), default=0.0),
        max((start + width for start, width in []), default=0.0),
    )
    ax.set_xlim(0.0, max_x * 1.03 if max_x > 0 else 1.0)
    ax.set_xlabel("Seconds Since Trajectory Start")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", linewidth=0.5, alpha=0.35)

    step_summary = report.get("info", {})
    policy = step_summary.get("checkpoint_policy", "unknown")
    exit_status = step_summary.get("exit_status", "unknown")
    metrics = step_summary.get("checkpoint_metrics", {})
    subtitle = (
        f"policy={policy} | exit={exit_status} | steps={len(spans)} | "
        f"checkpoint_created={metrics.get('checkpoint_created', 0)} | "
        f"rerun_from_checkpoint={metrics.get('rerun_from_checkpoint', 0)} | "
        f"rerun_from_base={metrics.get('rerun_from_base', 0)}"
    )
    fig.text(0.5, 0.945, subtitle, ha="center", va="center", fontsize=10)

    legend_items = [
        Patch(facecolor=llm_color, label="LLM wait"),
        Patch(facecolor=exec_color, label="Exec"),
        Patch(facecolor=checkpoint_pending_color, label="Checkpoint pending window"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=checkpoint_ready_color, markersize=8, label="Checkpoint create"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=checkpoint_ready_color, markersize=7, label="Checkpoint ready"),
        Line2D([0], [0], marker="x", color=checkpoint_busy_color, linestyle="None", markersize=7, label="Checkpoint busy / probe busy"),
        Patch(facecolor=recovery_color, label="Recovery / checkpoint rerun"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor=inject_color, markersize=8, label="Fault injected"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=recovery_done_color, markersize=7, label="Recovery complete"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.03, 0.05, 0.99, 0.92))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot a single rollout trajectory gantt chart.")
    parser.add_argument("traj_json", help="Path to rollout traj.json")
    parser.add_argument(
        "--output",
        help="Output PNG path. Defaults to <traj_dir>/figures/<traj_dir>.gantt.png",
    )
    parser.add_argument(
        "--title",
        help="Plot title. Defaults to '<traj_dir> rollout gantt'",
    )
    args = parser.parse_args()

    report_path = Path(args.traj_json).resolve()
    traj_dir = report_path.parent
    default_output = traj_dir / "figures" / f"{traj_dir.name}.gantt.png"
    output_path = Path(args.output).resolve() if args.output else default_output
    title = args.title or f"{traj_dir.name} rollout gantt"
    plot_traj(report_path, output_path, title)
    print(output_path)


if __name__ == "__main__":
    main()
