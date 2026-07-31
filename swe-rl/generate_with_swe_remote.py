"""Custom generate/reward for SWE-Bench RL with REMOTE Docker containers.

Drop-in replacement for generate_with_swe.py. Instead of running
Mini-SWE-Agent with local Docker, this version uses SweEnvClient to
interact with remote Docker containers managed by swe_env_pool_server.

The agent logic (multi-turn bash interaction) is reimplemented here
using the same prompt templates from swebench.yaml, but executes
commands via HTTP instead of `docker exec`.

Usage in training script:
    --custom-generate-function-path generate_with_swe_remote.generate
    --custom-rm-path generate_with_swe_remote.reward_func
"""

import asyncio
import copy
import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

try:
    import yaml
except Exception:  # pragma: no cover - lightweight replay/import environments
    yaml = None

from loguru import logger

try:
    from slime.rollout.sglang_rollout import GenerateState
    from slime.utils.types import Sample
except Exception:  # pragma: no cover - replay tools do not need the training stack
    GenerateState = None

    class Sample:  # type: ignore[no-redef]
        class Status:
            ABORTED = "aborted"
            COMPLETED = "completed"

from checkpoint_policy_runtime import (
    POLICIES,
    AdaptiveTailModel,
    DEFAULT_ADAPTIVE_BUDGET_SEC,
    DEFAULT_ADAPTIVE_DECISION_INTERVAL_SEC,
    DEFAULT_ADAPTIVE_FAILURE_PROB,
    DEFAULT_ADAPTIVE_TAIL_ROOT,
    DEFAULT_ADAPTIVE_MIN_DELTA_ENV_COST_SEC,
    DEFAULT_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS,
    adaptive_delta_env_cost_sec,
    adaptive_delta_replay_cost_sec,
    adaptive_expected_benefit_sec,
    fault_injection_armed_for_policy,
    redo_replay_cost_sec,
    should_probe_in_llm_bubble,
)
from swe_env_client import SweEnvClient
from swe_utils import get_docker_image_name
from message_utils import get_response_ids_and_loss_mask_from_messages
from swe_context_manager import get_context_messages


SWEAGENT_CONFIG_PATH = os.getenv(
    "SWE_CONFIG_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "swebench.yaml"),
)


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _extract_assistant_turn_spans(loss_mask: list[int]) -> list[list[int]]:
    """Extract [start, end) spans of consecutive 1s from loss_mask.

    Each span corresponds to one assistant turn's generated tokens.
    """
    spans: list[list[int]] = []
    in_span = False
    start = 0
    for i, m in enumerate(loss_mask):
        if m == 1 and not in_span:
            start = i
            in_span = True
        elif m == 0 and in_span:
            spans.append([start, i])
            in_span = False
    if in_span:
        spans.append([start, len(loss_mask)])
    return spans


@lru_cache(maxsize=1)
def _get_swe_semaphore() -> asyncio.Semaphore:
    max_inflight = int(os.getenv("SWE_MAX_CONCURRENT", "8"))
    return asyncio.Semaphore(max(1, max_inflight))


@lru_cache(maxsize=1)
def _get_eval_semaphore() -> asyncio.Semaphore:
    max_eval = int(os.getenv("SWE_MAX_CONCURRENT_EVAL", os.getenv("SWE_MAX_CONCURRENT", "8")))
    logger.info("[SWE-R] Evaluation limiter enabled: SWE_MAX_CONCURRENT_EVAL={}", max_eval)
    return asyncio.Semaphore(max(1, max_eval))


@lru_cache(maxsize=1)
def _get_diff_semaphore() -> asyncio.Semaphore:
    max_diff = int(os.getenv("SWE_MAX_CONCURRENT_DIFF", "8"))
    logger.info("[SWE-R] Diff limiter enabled: SWE_MAX_CONCURRENT_DIFF={}", max_diff)
    return asyncio.Semaphore(max(1, max_diff))


class _DockerCreateLimiter:
    def __init__(self, max_concurrent: int, min_interval_sec: float):
        self._sem = asyncio.Semaphore(max(1, int(max_concurrent)))
        self._min_interval_sec = max(0.0, float(min_interval_sec))
        self._ts_lock = asyncio.Lock()
        self._last_create_ts = 0.0

    async def allocate(self, env_client: SweEnvClient, *, image: str, instance_id: str) -> dict:
        async with self._sem:
            if self._min_interval_sec > 0:
                async with self._ts_lock:
                    now = time.time()
                    wait_s = self._min_interval_sec - (now - self._last_create_ts)
                    if wait_s > 0:
                        await asyncio.sleep(wait_s)
                    self._last_create_ts = time.time()
            return await env_client.allocate(image=image, instance_id=instance_id)


@lru_cache(maxsize=1)
def _get_docker_create_limiter() -> _DockerCreateLimiter:
    max_concurrent = int(os.getenv("SWE_MAX_CONCURRENT_DOCKER_CREATE", "1"))
    min_interval_sec = float(os.getenv("SWE_DOCKER_CREATE_MIN_INTERVAL_SEC", "0.5"))

    limiter = _DockerCreateLimiter(
        max_concurrent=max_concurrent,
        min_interval_sec=min_interval_sec,
    )
    logger.info(
        "[SWE-R] Docker create limiter enabled: max_concurrent={}, min_interval_sec={}",
        max_concurrent,
        min_interval_sec,
    )
    return limiter


@lru_cache(maxsize=1)
def _get_sweagent_config() -> dict:
    config_path = os.getenv("SWE_CONFIG_PATH", SWEAGENT_CONFIG_PATH)
    for candidate in [config_path, Path(config_path)]:
        p = Path(candidate)
        if p.exists():
            if yaml is None:
                raise RuntimeError(
                    f"PyYAML is required to load SWE config: {p}. "
                    "Install pyyaml or run with the project Python environment."
                )
            return yaml.safe_load(p.read_text())
    raise FileNotFoundError(f"SWE config not found: {config_path}")


def _sanitize_filename(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in value)


@lru_cache(maxsize=1)
def _get_swe_save_dir() -> Path | None:
    save_dir = os.getenv("SWE_SAVE_TRAJ_DIR", "").strip()
    if not save_dir:
        return None
    path = Path(save_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _get_checkpoint_policy() -> str:
    policy = os.getenv("SWE_CHECKPOINT_POLICY", "never").strip().lower().replace("_", "-").replace(" ", "-") or "never"
    if policy not in POLICIES:
        raise ValueError(f"Unsupported SWE_CHECKPOINT_POLICY={policy!r}; expected one of {POLICIES}")
    return policy


def _get_fault_injection_probability() -> float:
    return max(0.0, float(os.getenv("SWE_FAULT_INJECTION_PROB", "0.003")))


def _is_checkpoint_busy_error(exc: Exception) -> bool:
    return "checkpoint_busy" in str(exc).lower()


def _checkpoint_result_is_busy(result: dict[str, Any] | None) -> bool:
    if not isinstance(result, dict):
        return False
    if bool(result.get("busy", False)):
        return True
    return str(result.get("error_code", "") or "").lower() == "checkpoint_busy"


def _checkpoint_result_is_failed(result: dict[str, Any] | None) -> bool:
    return isinstance(result, dict) and not bool(result.get("ok", False)) and not _checkpoint_result_is_busy(result)


def _load_rollout_adaptive_tail_waits(root: Path) -> list[float]:
    waits: list[float] = []
    for traj_path in root.glob("**/traj.json"):
        try:
            payload = json.loads(traj_path.read_text())
        except Exception:
            continue
        for step in payload.get("step_debug", []):
            try:
                value = float(step.get("llm_elapsed", 0.0) or 0.0)
            except (TypeError, ValueError):
                value = 0.0
            if value > 0.0:
                waits.append(value)
    return waits


@lru_cache(maxsize=1)
def _get_rollout_adaptive_tail_model() -> AdaptiveTailModel | None:
    if _get_checkpoint_policy() != "adaptive-risk":
        return None
    root = Path(os.getenv("SWE_ADAPTIVE_TAIL_ROOT", str(DEFAULT_ADAPTIVE_TAIL_ROOT))).expanduser()
    budget_sec = float(os.getenv("SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC", str(DEFAULT_ADAPTIVE_BUDGET_SEC)))
    if not root.exists():
        logger.warning("[SWE-R] Adaptive tail root does not exist: {}", root)
        return None
    waits = _load_rollout_adaptive_tail_waits(root)
    if not waits:
        logger.warning("[SWE-R] Adaptive tail root has no positive llm_elapsed values: {}", root)
        return None
    model = AdaptiveTailModel.from_waits(waits, budget_sec=budget_sec)
    logger.info(
        "[SWE-R] Adaptive tail model loaded: root={} samples={} budget_sec={}",
        root,
        model.count,
        budget_sec,
    )
    return model


def _append_phase_event(
    phase_events: list[dict[str, Any]],
    *,
    event: str,
    category: str,
    start_ts: float,
    end_ts: float,
    step_idx: int | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    start = float(start_ts)
    end = max(start, float(end_ts))
    payload: dict[str, Any] = {
        "event": event,
        "category": category,
        "start_ts": start,
        "end_ts": end,
        "elapsed_sec": max(0.0, end - start),
    }
    if step_idx is not None:
        payload["step_idx"] = int(step_idx)
    if extra:
        payload.update(extra)
    phase_events.append(payload)


def _save_rollout_artifacts(*, sample: Sample, iid: str, sampling_params: dict, run_info: dict):
    try:
        save_dir = _get_swe_save_dir()
        if save_dir is None:
            return
        ts_ns = time.time_ns()
        stem = (
            f"{_sanitize_filename(iid)}"
            f"__g{sample.group_index if sample.group_index is not None else 'na'}"
            f"__i{sample.index if sample.index is not None else 'na'}"
            f"__{ts_ns}"
        )
        run_dir = save_dir / stem
        run_dir.mkdir(parents=True, exist_ok=True)
        traj_payload = {
            "messages": run_info.get("messages", []),
            "step_debug": run_info.get("step_debug", []),
            "checkpoint_events": run_info.get("checkpoint_events", []),
            "failure_events": run_info.get("failure_events", []),
            "rerun_events": run_info.get("rerun_events", []),
            "phase_events": run_info.get("phase_events", []),
            "info": {
                "instance_id": iid,
                "exit_status": run_info.get("exit_status"),
                "error": run_info.get("error"),
                "steps": run_info.get("n_steps"),
                "patch_source": run_info.get("patch_source"),
                "reward": run_info.get("reward"),
                "eval_result": run_info.get("eval_result"),
                "group_index": sample.group_index,
                "index": sample.index,
                "checkpoint_policy": run_info.get("checkpoint_policy"),
                "injection_target": run_info.get("injection_target"),
                "checkpoint_metrics": run_info.get("checkpoint_metrics", {}),
            },
            "trajectory_format": "slime-mini-swe-remote-1",
        }
        (run_dir / "traj.json").write_text(json.dumps(traj_payload, ensure_ascii=True, indent=2, default=str))
        git_patch = run_info.get("git_patch")
        if isinstance(git_patch, str):
            (run_dir / "patch.diff").write_text(git_patch)
        meta_payload = {
            "instance_id": iid,
            "sampling_params": sampling_params,
            "sample_metadata": sample.metadata,
            "sample_prompt": sample.prompt,
            "group_index": sample.group_index,
            "index": sample.index,
        }
        (run_dir / "meta.json").write_text(json.dumps(meta_payload, ensure_ascii=True, indent=2, default=str))
        logger.info(f"[SWE-R] [{iid}] Saved rollout artifacts to {run_dir}")
    except Exception as e:
        logger.warning(f"[SWE-R] [{iid}] Failed to save rollout artifacts: {e}")


def _parse_bash_action(response_text: str) -> str | None:
    """Extract the bash command from a response containing ```bash ... ```."""
    pattern = r"```bash\s*\n(.*?)```"
    match = re.search(pattern, response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


class TrajectoryReplayCompletionProvider:
    """Replay assistant turns from a saved SWE trajectory while simulating LLM latency."""

    def __init__(
        self,
        trajectory_payload: dict[str, Any],
        *,
        llm_delay_scale: float = 1.0,
        min_delay_sec: float = 0.0,
        max_delay_sec: float | None = None,
        strict_action_match: bool = False,
        traj_label: str = "",
    ):
        self.trajectory_payload = trajectory_payload
        self.llm_delay_scale = max(0.0, float(llm_delay_scale))
        self.min_delay_sec = max(0.0, float(min_delay_sec))
        self.max_delay_sec = None if max_delay_sec is None else max(0.0, float(max_delay_sec))
        self.strict_action_match = bool(strict_action_match)
        self.traj_label = traj_label or str(
            trajectory_payload.get("info", {}).get("instance_id", "unknown")
            if isinstance(trajectory_payload.get("info"), dict)
            else "unknown"
        )
        self.assistant_turns = [
            str(item.get("content", "") or "")
            for item in trajectory_payload.get("messages", [])
            if isinstance(item, dict) and item.get("role") == "assistant"
        ]
        self.recorded_steps = [
            item
            for item in trajectory_payload.get("step_debug", [])
            if isinstance(item, dict)
        ]
        self.turn_idx = 0
        self.action_idx = 0
        self.action_mismatches: list[dict[str, Any]] = []

    @classmethod
    def from_path(
        cls,
        traj_path: str | Path,
        *,
        llm_delay_scale: float = 1.0,
        min_delay_sec: float = 0.0,
        max_delay_sec: float | None = None,
        strict_action_match: bool = False,
    ) -> "TrajectoryReplayCompletionProvider":
        path = Path(traj_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            payload,
            llm_delay_scale=llm_delay_scale,
            min_delay_sec=min_delay_sec,
            max_delay_sec=max_delay_sec,
            strict_action_match=strict_action_match,
            traj_label=path.parent.name,
        )

    def _delay_for_turn(self, assistant_text: str) -> float:
        parsed_action = _parse_bash_action(assistant_text)
        if parsed_action is None:
            return self.min_delay_sec

        if self.action_idx >= len(self.recorded_steps):
            return self.min_delay_sec

        step = self.recorded_steps[self.action_idx]
        expected_action = str(step.get("action", "") or "").strip()
        parsed_action_stripped = parsed_action.strip()
        if expected_action and expected_action != parsed_action_stripped:
            mismatch = {
                "turn_idx": self.turn_idx,
                "action_idx": self.action_idx,
                "expected_action": expected_action,
                "parsed_action": parsed_action_stripped,
            }
            self.action_mismatches.append(mismatch)
            message = (
                f"[SWE-R] [{self.traj_label}] replay action mismatch "
                f"turn={self.turn_idx} action_idx={self.action_idx}"
            )
            if self.strict_action_match:
                raise ValueError(f"{message}: {mismatch}")
            logger.warning("{} expected={!r} parsed={!r}", message, expected_action[:240], parsed_action_stripped[:240])

        try:
            raw_delay = float(step.get("llm_elapsed", 0.0) or 0.0)
        except (TypeError, ValueError):
            raw_delay = 0.0
        self.action_idx += 1
        delay = max(self.min_delay_sec, raw_delay * self.llm_delay_scale)
        if self.max_delay_sec is not None:
            delay = min(delay, self.max_delay_sec)
        return delay

    async def acompletion(self, *, model: str, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        if self.turn_idx >= len(self.assistant_turns):
            raise RuntimeError(
                f"trajectory replay exhausted assistant turns at turn={self.turn_idx} "
                f"traj={self.traj_label}"
            )
        assistant_text = self.assistant_turns[self.turn_idx]
        delay = self._delay_for_turn(assistant_text)
        self.turn_idx += 1
        if delay > 0.0:
            await asyncio.sleep(delay)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=assistant_text),
                )
            ]
        )

    def summary(self) -> dict[str, Any]:
        return {
            "traj_label": self.traj_label,
            "assistant_turn_count": len(self.assistant_turns),
            "assistant_turns_used": self.turn_idx,
            "recorded_action_count": len(self.recorded_steps),
            "recorded_actions_used": self.action_idx,
            "action_mismatch_count": len(self.action_mismatches),
            "action_mismatches": self.action_mismatches[:20],
            "llm_delay_scale": self.llm_delay_scale,
        }


def _extract_patch_from_submission(output: str) -> str:
    """Extract a clean git patch text from submit command output."""
    if not isinstance(output, str):
        return ""
    text = output.lstrip("\n")
    sentinel = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
    if text.startswith(sentinel):
        text = text[len(sentinel):].lstrip("\n")
    return text


def _is_valid_git_patch(patch_text: str) -> bool:
    """Lightweight patch validity check before remote evaluation."""
    if not isinstance(patch_text, str):
        return False
    text = patch_text.strip()
    if not text:
        return False
    if "diff --git " not in text:
        return False
    has_old = ("--- a/" in text) or ("--- /dev/null" in text)
    has_new = "+++ b/" in text
    return has_old and has_new


def _render_observation(config: dict, returncode: int, output: str) -> str:
    """Render the action_observation_template from swebench.yaml."""
    from jinja2 import Template
    template_str = config.get("agent", {}).get("action_observation_template", "")
    if not template_str:
        return f"<returncode>{returncode}</returncode>\n<output>\n{output}\n</output>"
    template = Template(template_str)
    return template.render(output={"returncode": returncode, "output": output})


async def _refresh_rollout_pending_checkpoints(
    *,
    env_client: SweEnvClient,
    lease_id: str,
    checkpoint_state: dict[str, Any],
    traj_label: str,
) -> None:
    pending = checkpoint_state["pending_checkpoints"]
    if not pending:
        return
    for item in pending:
        checkpoint_state["checkpoint_events"].append(
            {
                "event": "checkpoint_status_skipped",
                "checkpoint_id": item["checkpoint_id"],
                "step_idx": int(item["step_idx"]),
                "resume_step_idx": int(item["resume_step_idx"]),
                "skip_reason": "checkpoint_status_deprecated",
            }
        )
    checkpoint_state["pending_checkpoints"] = []


async def _attempt_rollout_checkpoint_create(
    *,
    env_client: SweEnvClient,
    lease_id: str,
    checkpoint_state: dict[str, Any],
    phase_events: list[dict[str, Any]] | None,
    policy: str,
    cwd: str,
    checkpoint_step_idx: int,
    command_seq: int,
    resume_step_idx: int,
    protected_env_cost_sec: float,
    protected_llm_cost_sec: float,
    traj_label: str,
    report_fields: dict[str, Any] | None = None,
) -> tuple[bool, bool]:
    event: dict[str, Any] = {
        "event": "checkpoint_create",
        "policy": policy,
        "step_idx": int(checkpoint_step_idx),
        "command_seq": int(command_seq),
        "resume_step_idx": int(resume_step_idx),
        "protected_cost_sec": float(protected_env_cost_sec),
        "protected_env_cost_sec": float(protected_env_cost_sec),
        "protected_llm_cost_sec": float(protected_llm_cost_sec),
    }
    if report_fields:
        event.update(report_fields)
    create_call_t0 = time.time()
    create_result = await env_client.checkpoint_create(
        lease_id,
        step_idx=int(checkpoint_step_idx),
        command_seq=int(command_seq),
        cwd=cwd,
        policy=policy,
        reason="rollout_runtime",
        parent_checkpoint_id=checkpoint_state["latest_ready_checkpoint_id"],
    )
    create_call_t1 = time.time()
    event["create_call_start_ts"] = float(create_call_t0)
    event["create_call_end_ts"] = float(create_call_t1)
    event["create_call_elapsed_sec"] = max(0.0, float(create_call_t1 - create_call_t0))
    event["create_result"] = create_result
    if phase_events is not None:
        _append_phase_event(
            phase_events,
            event="checkpoint_create_call",
            category="checkpoint",
            start_ts=create_call_t0,
            end_ts=create_call_t1,
            step_idx=int(checkpoint_step_idx),
            extra={
                "created": bool(create_result.get("ok", False)),
                "busy": bool(_checkpoint_result_is_busy(create_result)),
                "policy": policy,
                "command_seq": int(command_seq),
            },
        )
    if bool(create_result.get("ok", False)):
        ready_at = float(create_result.get("ready_at", time.time()) or time.time())
        checkpoint_id = str(create_result["checkpoint_id"])
        checkpoint_state["latest_ready_checkpoint_id"] = checkpoint_id
        checkpoint_state["latest_ready_checkpoint_step"] = int(
            create_result.get("step_idx", checkpoint_step_idx)
        )
        checkpoint_state["latest_ready_resume_step_idx"] = int(resume_step_idx)
        checkpoint_state["latest_ready_protected_env_cost_sec"] = float(protected_env_cost_sec)
        checkpoint_state["latest_ready_protected_llm_cost_sec"] = float(protected_llm_cost_sec)
        checkpoint_state["latest_ready_checkpoint_ready_at"] = ready_at
        checkpoint_state["checkpoint_events"].append(event)
        logger.info(
            "[SWE-R] [{}] checkpoint ready step={} checkpoint_id={}",
            traj_label,
            checkpoint_step_idx,
            checkpoint_id,
        )
        return True, False

    if _checkpoint_result_is_busy(create_result):
        event["skipped"] = True
        event["skip_reason"] = "checkpoint_busy"
        checkpoint_state["checkpoint_events"].append(event)
        logger.info("[SWE-R] [{}] checkpoint busy at step={}", traj_label, checkpoint_step_idx)
        return False, True

    event["skipped"] = True
    event["skip_reason"] = "checkpoint_create_failed"
    checkpoint_state["checkpoint_events"].append(event)
    logger.warning(
        "[SWE-R] [{}] checkpoint create failed at step={} result={}",
        traj_label,
        checkpoint_step_idx,
        create_result,
    )
    return False, False


async def _wait_for_llm_with_adaptive_checkpointing(
    *,
    llm_task: asyncio.Task,
    env_client: SweEnvClient,
    lease_id: str,
    checkpoint_state: dict[str, Any],
    phase_events: list[dict[str, Any]],
    current_step_idx: int,
    step_limit: int,
    cwd: str,
    policy: str,
    tail_model: AdaptiveTailModel | None,
    traj_label: str,
) -> tuple[Any, float]:
    started_at = time.time()
    decision_interval_sec = float(
        os.getenv("SWE_ADAPTIVE_DECISION_INTERVAL_SEC", str(DEFAULT_ADAPTIVE_DECISION_INTERVAL_SEC))
    )
    adaptive_failure_prob = float(
        os.getenv("SWE_ADAPTIVE_FAILURE_PROB", str(DEFAULT_ADAPTIVE_FAILURE_PROB))
    )
    min_delta_env_cost_sec = float(
        os.getenv("SWE_ADAPTIVE_MIN_DELTA_ENV_COST_SEC", str(DEFAULT_ADAPTIVE_MIN_DELTA_ENV_COST_SEC))
    )
    min_steps_between_checkpoints = int(
        os.getenv(
            "SWE_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS",
            str(DEFAULT_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS),
        )
    )
    probe_attempted_in_bubble = False
    adaptive_checkpoint_submitted = False
    fixed_policy_checkpoint_attempted = False

    while True:
        if llm_task.done():
            break
        await _refresh_rollout_pending_checkpoints(
            env_client=env_client,
            lease_id=lease_id,
            checkpoint_state=checkpoint_state,
            traj_label=traj_label,
        )
        waited_in_llm_sec = max(0.0, time.time() - started_at)
        if llm_task.done():
            break
        if (
            not fixed_policy_checkpoint_attempted
            and current_step_idx > 0
            and current_step_idx < step_limit
            and int(checkpoint_state.get("last_fixed_policy_checkpoint_attempt_step_idx", -1)) < current_step_idx - 1
            and (
                policy == "always"
                or (policy == "every-3" and current_step_idx % 3 == 0)
            )
        ):
            fixed_policy_checkpoint_attempted = True
            checkpoint_state["last_fixed_policy_checkpoint_attempt_step_idx"] = int(current_step_idx - 1)
            checkpoint_state["checkpoint_attempts"] += 1
            created, busy = await _attempt_rollout_checkpoint_create(
                env_client=env_client,
                lease_id=lease_id,
                checkpoint_state=checkpoint_state,
                phase_events=phase_events,
                policy=policy,
                cwd=cwd,
                checkpoint_step_idx=current_step_idx - 1,
                command_seq=current_step_idx,
                resume_step_idx=current_step_idx,
                protected_env_cost_sec=checkpoint_state["cumulative_env_replay_cost_sec"],
                protected_llm_cost_sec=checkpoint_state["cumulative_llm_replay_cost_sec"],
                traj_label=traj_label,
                report_fields={
                    "decision_type": "fixed_policy_llm_wait",
                    "during_llm_wait_for_step_idx": current_step_idx,
                    "waited_before_checkpoint_sec": waited_in_llm_sec,
                },
            )
            if created:
                checkpoint_state["checkpoint_created"] += 1
            elif busy:
                checkpoint_state["checkpoint_busy_skips"] += 1
            if llm_task.done():
                break
        if policy == "adaptive-risk" and tail_model is not None:
            delta_env_replay_cost = adaptive_delta_env_cost_sec(
                checkpoint_state["cumulative_env_replay_cost_sec"],
                checkpoint_state["latest_ready_protected_env_cost_sec"],
            )
            delta_llm_replay_cost = max(
                0.0,
                float(checkpoint_state["cumulative_llm_replay_cost_sec"])
                - float(checkpoint_state["latest_ready_protected_llm_cost_sec"]),
            )
            redo_cost = redo_replay_cost_sec(
                checkpoint_state["cumulative_env_replay_cost_sec"],
                checkpoint_state["latest_ready_protected_env_cost_sec"],
                checkpoint_state["cumulative_llm_replay_cost_sec"],
                checkpoint_state["latest_ready_protected_llm_cost_sec"],
            )
            delta_replay_cost = adaptive_delta_replay_cost_sec(
                checkpoint_state["cumulative_env_replay_cost_sec"],
                checkpoint_state["latest_ready_protected_env_cost_sec"],
                checkpoint_state["cumulative_llm_replay_cost_sec"],
                checkpoint_state["latest_ready_protected_llm_cost_sec"],
            )
            latest_step = int(checkpoint_state["latest_ready_checkpoint_step"])
            steps_since_latest_ready_checkpoint = (
                current_step_idx if latest_step < 0 else max(0, current_step_idx - latest_step)
            )
            expected_overhead_sec = tail_model.expected_visible_overhead(waited_in_llm_sec)
            checkpoint_cover_probability = tail_model.conditional_survival_probability(
                waited_in_llm_sec
            )
            expected_benefit_sec = adaptive_expected_benefit_sec(
                adaptive_failure_prob,
                redo_cost,
            )
            if llm_task.done():
                break
            should_probe = should_probe_in_llm_bubble(
                current_step_idx=current_step_idx,
                probe_attempted_in_bubble=probe_attempted_in_bubble,
                adaptive_checkpoint_submitted=adaptive_checkpoint_submitted,
                pending_checkpoints=checkpoint_state["pending_checkpoints"],
                delta_env_cost_sec_value=delta_env_replay_cost,
                steps_since_latest_ready_checkpoint=steps_since_latest_ready_checkpoint,
                expected_benefit_sec=expected_benefit_sec,
                expected_overhead_sec=expected_overhead_sec,
                min_delta_env_cost_sec=min_delta_env_cost_sec,
                min_steps_between_checkpoints=min_steps_between_checkpoints,
            )
            if llm_task.done():
                break
            if should_probe:
                probe_attempted_in_bubble = True
                checkpoint_state["checkpoint_attempts"] += 1
                checkpoint_state["probe_count"] += 1
                created, busy = await _attempt_rollout_checkpoint_create(
                    env_client=env_client,
                    lease_id=lease_id,
                    checkpoint_state=checkpoint_state,
                    phase_events=phase_events,
                    policy=policy,
                    cwd=cwd,
                    checkpoint_step_idx=current_step_idx - 1,
                    command_seq=current_step_idx,
                    resume_step_idx=current_step_idx,
                    protected_env_cost_sec=checkpoint_state["cumulative_env_replay_cost_sec"],
                    protected_llm_cost_sec=checkpoint_state["cumulative_llm_replay_cost_sec"],
                    traj_label=traj_label,
                    report_fields={
                        "decision_type": "adaptive_llm_wait",
                        "during_llm_wait_for_step_idx": current_step_idx,
                        "waited_before_checkpoint_sec": waited_in_llm_sec,
                        "expected_benefit_sec": expected_benefit_sec,
                        "expected_overhead_sec": expected_overhead_sec,
                        "adaptive_failure_probability": adaptive_failure_prob,
                        "checkpoint_duration_estimate_sec": tail_model.budget_sec,
                        "checkpoint_cover_probability": checkpoint_cover_probability,
                        # Retain the legacy diagnostic field. It no longer
                        # discounts expected_benefit_sec.
                        "conditional_tail_probability": checkpoint_cover_probability,
                        "redo_from_resume_step_idx": int(checkpoint_state["latest_ready_resume_step_idx"]),
                        "redo_until_step_idx": int(current_step_idx),
                        "redo_replay_cost_sec": redo_cost,
                        "redo_env_replay_cost_sec": delta_env_replay_cost,
                        "redo_llm_replay_cost_sec": delta_llm_replay_cost,
                        # Keep the legacy field for backward compatibility.
                        "delta_env_cost_sec": delta_replay_cost,
                        "delta_replay_cost_sec": delta_replay_cost,
                        "delta_env_replay_cost_sec": delta_env_replay_cost,
                        "delta_llm_replay_cost_sec": delta_llm_replay_cost,
                        # Standalone probe RPC is removed; checkpoint/create now
                        # performs the probe inline and returns checkpoint_busy
                        # when the preflight check fails.
                        "probe_mode": "inline_before_create",
                    },
                )
                if created:
                    checkpoint_state["checkpoint_created"] += 1
                    adaptive_checkpoint_submitted = True
                elif busy:
                    checkpoint_state["probe_busy_skips"] += 1
            if llm_task.done():
                break
        try:
            await asyncio.wait_for(asyncio.shield(llm_task), timeout=decision_interval_sec)
        except asyncio.TimeoutError:
            pass

    response = await llm_task
    return response, max(0.0, time.time() - started_at)


async def _replay_prior_actions(
    *,
    env_client: SweEnvClient,
    lease_id: str,
    replay_actions: list[dict[str, Any]],
    cwd: str,
    exec_timeout: int,
    traj_label: str,
) -> list[dict[str, Any]]:
    replayed: list[dict[str, Any]] = []
    for step in replay_actions:
        step_idx = int(step["step_idx"])
        action = str(step["action"])
        exec_t0 = time.time()
        exec_result = await env_client.exec(lease_id=lease_id, command=action, cwd=cwd, timeout=exec_timeout)
        replayed.append(
            {
                "step_idx": step_idx,
                "action": action,
                "returncode": int(exec_result.get("returncode", -1)),
                "exec_elapsed_sec": time.time() - exec_t0,
                "output_preview": str(exec_result.get("output", ""))[:400],
            }
        )
    logger.info("[SWE-R] [{}] recovery replayed {} prior actions", traj_label, len(replayed))
    return replayed


def _trim_messages_for_resume(messages: list[dict[str, Any]], resume_step_idx: int) -> list[dict[str, Any]]:
    keep_count = max(2, 2 + 2 * int(resume_step_idx))
    return list(messages[:keep_count])


def _trim_step_debug_for_resume(step_debug: list[dict[str, Any]], resume_step_idx: int) -> list[dict[str, Any]]:
    return [step for step in step_debug if int(step.get("step_idx", -1)) < int(resume_step_idx)]


def _trim_pending_prm_tasks_for_resume(
    prm_pending_tasks: list[tuple[int, asyncio.Task]],
    resume_step_idx: int,
) -> list[tuple[int, asyncio.Task]]:
    kept: list[tuple[int, asyncio.Task]] = []
    for step_idx, task in prm_pending_tasks:
        if int(step_idx) < int(resume_step_idx):
            kept.append((step_idx, task))
        else:
            task.cancel()
    return kept


async def _recover_from_injected_fault(
    *,
    env_client: SweEnvClient,
    lease_id: str,
    image_name: str,
    instance_id: str,
    current_step_idx: int,
    checkpoint_state: dict[str, Any],
    cwd: str,
    exec_timeout: int,
    traj_label: str,
    docker_create_limiter: _DockerCreateLimiter,
) -> dict[str, Any]:
    recovery_t0 = time.time()
    failure_event: dict[str, Any] = {
        "inject_before_step_idx": current_step_idx,
        "latest_ready_checkpoint_id": checkpoint_state["latest_ready_checkpoint_id"],
        "latest_ready_checkpoint_step": checkpoint_state["latest_ready_checkpoint_step"],
        "latest_ready_resume_step_idx": checkpoint_state["latest_ready_resume_step_idx"],
        "recovery_mode": None,
    }
    if checkpoint_state["latest_ready_checkpoint_id"] is not None:
        rerun_t0 = time.time()
        rerun_result = await env_client.rerun(
            lease_id,
            checkpoint_id=checkpoint_state["latest_ready_checkpoint_id"],
            cwd=cwd,
            timeout=10,
        )
        if bool(rerun_result.get("ok", False)):
            checkpoint_state["rerun_from_checkpoint"] += 1
            checkpoint_state["pending_checkpoints"] = []
            checkpoint_state["cumulative_env_replay_cost_sec"] = checkpoint_state["latest_ready_protected_env_cost_sec"]
            checkpoint_state["cumulative_llm_replay_cost_sec"] = checkpoint_state["latest_ready_protected_llm_cost_sec"]
            checkpoint_state["last_fixed_policy_checkpoint_attempt_step_idx"] = int(
                checkpoint_state["latest_ready_checkpoint_step"]
            )
            resume_step_idx = int(checkpoint_state["latest_ready_resume_step_idx"])
            failure_event["recovery_mode"] = "checkpoint_rerun"
            failure_event["rerun_result"] = rerun_result
            failure_event["rerun_wall_time_sec"] = time.time() - rerun_t0
            failure_event["resume_step_idx"] = resume_step_idx
            checkpoint_state["rerun_events"].append(failure_event)
            return {
                "lease_id": lease_id,
                "resume_step_idx": resume_step_idx,
            }
        failure_event["recovery_mode"] = "checkpoint_rerun_failed"
        failure_event["rerun_result"] = rerun_result
        failure_event["rerun_wall_time_sec"] = time.time() - rerun_t0
        logger.warning(
            "[SWE-R] [{}] checkpoint rerun failed for checkpoint_id={} result={}; falling back to base rerun",
            traj_label,
            checkpoint_state["latest_ready_checkpoint_id"],
            rerun_result,
        )

    await env_client.close(lease_id)
    new_lease = await docker_create_limiter.allocate(env_client, image=image_name, instance_id=instance_id)
    new_lease_id = str(new_lease["lease_id"])
    checkpoint_state["rerun_from_base"] += 1
    checkpoint_state["pending_checkpoints"] = []
    checkpoint_state["latest_ready_checkpoint_id"] = None
    checkpoint_state["latest_ready_checkpoint_step"] = -1
    checkpoint_state["latest_ready_resume_step_idx"] = 0
    checkpoint_state["latest_ready_protected_env_cost_sec"] = 0.0
    checkpoint_state["latest_ready_protected_llm_cost_sec"] = 0.0
    checkpoint_state["latest_ready_checkpoint_ready_at"] = -1.0
    checkpoint_state["cumulative_env_replay_cost_sec"] = 0.0
    checkpoint_state["cumulative_llm_replay_cost_sec"] = 0.0
    checkpoint_state["last_fixed_policy_checkpoint_attempt_step_idx"] = -1
    failure_event["recovery_mode"] = "base_restart"
    failure_event["new_lease"] = new_lease
    failure_event["rerun_wall_time_sec"] = time.time() - recovery_t0
    failure_event["resume_step_idx"] = 0
    checkpoint_state["rerun_events"].append(failure_event)
    logger.info(
        "[SWE-R] [{}] injected fault recovered via {} at step={}",
        traj_label,
        failure_event["recovery_mode"],
        current_step_idx,
    )
    return {
        "lease_id": new_lease_id,
        "resume_step_idx": 0,
    }


async def _run_agent_remote(
    env_client: SweEnvClient,
    lease_id: str,
    instance: dict,
    litellm_model_name: str,
    model_config: dict,
    sweagent_config: dict,
    *,
    args: object | None = None,
    prm_agent: object | None = None,
    tokenizer: object | None = None,
    cm_max_input_tokens: int | None = None,
    cm_head_ratio: float = 0.3,
    policy: str = "never",
    tail_model: AdaptiveTailModel | None = None,
    fault_injection_armed: bool = False,
    fault_injection_probability: float = 0.0,
    docker_create_limiter: _DockerCreateLimiter | None = None,
    image_name: str = "",
    replay_completion_provider: TrajectoryReplayCompletionProvider | None = None,
) -> dict:
    """Run the multi-turn agent loop using remote Docker execution."""
    acompletion = None
    if replay_completion_provider is None:
        from litellm import acompletion as litellm_acompletion

        acompletion = litellm_acompletion

    iid = instance.get("instance_id", "unknown")
    agent_config = sweagent_config.get("agent", {})
    env_config = sweagent_config.get("environment", {})
    cwd = env_config.get("cwd", "/testbed")
    step_limit = int(agent_config.get("step_limit", 20))
    exec_timeout = int(env_config.get("timeout", 180))
    openai_api_base = os.environ.get("OPENAI_BASE_URL", "").strip()
    openai_api_key = os.environ.get("OPENAI_API_KEY", "").strip() or "dummy"
    litellm_timeout = float(os.environ.get("SWE_LITELLM_TIMEOUT", "600"))

    if replay_completion_provider is None:
        logger.info(
            "[SWE-R] [{}] LiteLLM request config: model={}, api_base={}, api_key_source={}, timeout={}",
            iid,
            litellm_model_name,
            openai_api_base or "<unset>",
            "env" if os.environ.get("OPENAI_API_KEY", "").strip() else "dummy",
            litellm_timeout,
        )
    else:
        logger.info(
            "[SWE-R] [{}] trajectory replay completion enabled: {}",
            iid,
            replay_completion_provider.summary(),
        )

    system_template = agent_config.get("system_template", "You are a helpful assistant.")
    instance_template = agent_config.get("instance_template", "{{task}}")
    from jinja2 import Template
    instance_message = Template(instance_template).render(task=instance["problem_statement"])

    messages = [
        {"role": "system", "content": system_template},
        {"role": "user", "content": instance_message},
    ]

    step_debug = []
    phase_events: list[dict[str, Any]] = []
    git_patch = None
    patch_source = None
    exit_status = None
    error = None
    n_steps = 0
    prm_pending_tasks: list[tuple[int, asyncio.Task]] = []
    managed_contexts: list[list[dict]] = []
    assistant_texts: list[str] = []
    checkpoint_state: dict[str, Any] = {
        "pending_checkpoints": [],
        "latest_ready_checkpoint_id": None,
        "latest_ready_checkpoint_step": -1,
        "latest_ready_resume_step_idx": 0,
        "latest_ready_protected_env_cost_sec": 0.0,
        "latest_ready_protected_llm_cost_sec": 0.0,
        "latest_ready_checkpoint_ready_at": -1.0,
        "cumulative_env_replay_cost_sec": 0.0,
        "cumulative_llm_replay_cost_sec": 0.0,
        "checkpoint_events": [],
        "failure_events": [],
        "rerun_events": [],
        "checkpoint_attempts": 0,
        "checkpoint_created": 0,
        "checkpoint_busy_skips": 0,
        "probe_count": 0,
        "probe_busy_skips": 0,
        "last_fixed_policy_checkpoint_attempt_step_idx": -1,
        "rerun_from_checkpoint": 0,
        "rerun_from_base": 0,
    }
    cm_enabled = tokenizer is not None and cm_max_input_tokens is not None and cm_max_input_tokens > 0

    t0 = time.time()
    step_idx = 0
    while step_idx < step_limit:
        n_steps = max(n_steps, step_idx + 1)
        await env_client.heartbeat(lease_id)
        await _refresh_rollout_pending_checkpoints(
            env_client=env_client,
            lease_id=lease_id,
            checkpoint_state=checkpoint_state,
            traj_label=iid,
        )

        if cm_enabled:
            ctx_messages = get_context_messages(
                messages, tokenizer,
                max_input_tokens=cm_max_input_tokens,
                head_ratio=cm_head_ratio,
            )
        else:
            ctx_messages = messages
        llm_step_t0 = time.time()
        llm_waited_before_exec_sec = 0.0
        try:
            completion_kwargs = dict(model_config.get("model_kwargs", {}))
            if openai_api_base and "api_base" not in completion_kwargs:
                completion_kwargs["api_base"] = openai_api_base
            if "api_key" not in completion_kwargs:
                completion_kwargs["api_key"] = openai_api_key
            if "timeout" not in completion_kwargs:
                completion_kwargs["timeout"] = litellm_timeout
            if replay_completion_provider is None:
                assert acompletion is not None
                completion_coro = acompletion(
                    model=litellm_model_name,
                    messages=ctx_messages,
                    **completion_kwargs,
                )
            else:
                completion_coro = replay_completion_provider.acompletion(
                    model=litellm_model_name,
                    messages=ctx_messages,
                    **completion_kwargs,
                )
            llm_task = asyncio.create_task(completion_coro)
            resp, llm_waited_before_exec_sec = await _wait_for_llm_with_adaptive_checkpointing(
                llm_task=llm_task,
                env_client=env_client,
                lease_id=lease_id,
                checkpoint_state=checkpoint_state,
                phase_events=phase_events,
                current_step_idx=step_idx,
                step_limit=step_limit,
                cwd=cwd,
                policy=policy,
                tail_model=tail_model,
                traj_label=iid,
            )
            assistant_text = resp.choices[0].message.content or ""
        except Exception as e:
            error = f"LLM call failed at step {step_idx}: {e}"
            logger.error(f"[SWE-R] [{iid}] {error}")
            break
        llm_step_elapsed = time.time() - llm_step_t0
        managed_contexts.append(copy.deepcopy(ctx_messages))
        messages.append({"role": "assistant", "content": assistant_text})
        assistant_texts.append(assistant_text)

        bash_cmd = _parse_bash_action(assistant_text)
        if bash_cmd is None:
            observation = _render_observation(sweagent_config, -1, "No valid bash command found in response.")
            messages.append({"role": "user", "content": observation})
            llm_only_end_ts = time.time()
            _append_phase_event(
                phase_events,
                event="llm_only_turn",
                category="llm_only",
                start_ts=llm_step_t0,
                end_ts=llm_only_end_ts,
                step_idx=step_idx,
                extra={
                    "llm_elapsed_sec": float(llm_step_elapsed),
                    "reason": "no_bash_action_extracted",
                },
            )
            step_idx += 1
            continue

        is_submit = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in bash_cmd

        step_t0 = time.time()
        try:
            exec_result = await env_client.exec(
                lease_id=lease_id,
                command=bash_cmd,
                cwd=cwd,
                timeout=exec_timeout,
                fault_injection_armed=fault_injection_armed,
                fault_injection_probability=fault_injection_probability,
            ) # Environment fault inject, e.g. system error, docker exec failure, docker killed due to OOM or timeout.
            if bool(exec_result.get("fault_injected", False)):
                checkpoint_state["failure_events"].append(
                    {
                        "event": "fault_injection",
                        "step_idx": step_idx,
                        "action": bash_cmd,
                        "mode": "exec_server_probability",
                        "fault_type": exec_result.get("fault_type", "exec_server_random_kill"),
                        "error_code": exec_result.get("error_code"),
                        "fault_injection_probability": exec_result.get(
                            "fault_injection_probability",
                            fault_injection_probability,
                        ),
                        "container_usable": exec_result.get("container_usable", False),
                        "latest_ready_checkpoint_id": checkpoint_state["latest_ready_checkpoint_id"],
                        "latest_ready_checkpoint_step": checkpoint_state["latest_ready_checkpoint_step"],
                        "latest_ready_resume_step_idx": checkpoint_state["latest_ready_resume_step_idx"],
                        "exec_result": exec_result,
                    }
                )
                if docker_create_limiter is None:
                    raise RuntimeError("docker_create_limiter is required for rollout fault recovery")
                recovery_t0 = time.time()
                recovery_result = await _recover_from_injected_fault(
                    env_client=env_client,
                    lease_id=lease_id,
                    image_name=image_name,
                    instance_id=iid,
                    current_step_idx=step_idx,
                    checkpoint_state=checkpoint_state,
                    cwd=cwd,
                    exec_timeout=exec_timeout,
                    traj_label=iid,
                    docker_create_limiter=docker_create_limiter,
                )
                recovery_t1 = time.time()
                latest_rerun_event = checkpoint_state["rerun_events"][-1] if checkpoint_state["rerun_events"] else {}
                resume_step_idx = int(recovery_result.get("resume_step_idx", 0))
                discarded_steps = [
                    step for step in step_debug
                    if int(step.get("step_idx", -1)) >= resume_step_idx
                ]
                discarded_starts: list[float] = []
                for discarded_step in discarded_steps:
                    exec_start_ts = float(discarded_step.get("start_ts", 0.0) or 0.0)
                    llm_wait_sec = float(discarded_step.get("llm_waited_before_exec_sec", 0.0) or 0.0)
                    discarded_starts.append(max(0.0, exec_start_ts - llm_wait_sec))
                if int(step_idx) >= resume_step_idx:
                    discarded_starts.append(float(llm_step_t0))
                if discarded_starts:
                    _append_phase_event(
                        phase_events,
                        event="discarded_attempt_window",
                        category="discarded_attempt",
                        start_ts=min(discarded_starts),
                        end_ts=recovery_t0,
                        step_idx=step_idx,
                        extra={
                            "resume_step_idx": resume_step_idx,
                            "inject_before_step_idx": int(step_idx),
                            "discarded_recorded_steps": len(discarded_steps),
                        },
                    )
                _append_phase_event(
                    phase_events,
                    event="fault_recovery",
                    category="recovery",
                    start_ts=recovery_t0,
                    end_ts=recovery_t1,
                    step_idx=step_idx,
                    extra={
                        "recovery_mode": latest_rerun_event.get("recovery_mode", "recovery"),
                        "resume_step_idx": resume_step_idx,
                        "latest_ready_checkpoint_step": latest_rerun_event.get("latest_ready_checkpoint_step"),
                    },
                )
                lease_id = str(recovery_result["lease_id"])
                messages = _trim_messages_for_resume(messages, resume_step_idx)
                step_debug = _trim_step_debug_for_resume(step_debug, resume_step_idx)
                managed_contexts = managed_contexts[:resume_step_idx]
                assistant_texts = assistant_texts[:resume_step_idx]
                prm_pending_tasks = _trim_pending_prm_tasks_for_resume(prm_pending_tasks, resume_step_idx)
                await env_client.heartbeat(lease_id)
                step_idx = resume_step_idx
                continue
            returncode = exec_result.get("returncode", -1)
            output = exec_result.get("output", "")
        except Exception as e:
            returncode = -1
            output = f"Execution error: {e}"
            logger.error(f"[SWE-R] [{iid}] step {step_idx} exec error: {e}")

        step_t1 = time.time()
        step_elapsed = max(0.0, step_t1 - step_t0)
        step_debug.append({
            "step_idx": step_idx,
            "action": bash_cmd,
            "returncode": returncode,
            "output_len": len(output),
            "output_head": output[:2000],
            "output_tail": output[-2000:] if len(output) > 2000 else output,
            "start_ts": step_t0,
            "end_ts": step_t1,
            "elapsed": step_elapsed,
            "llm_elapsed": llm_step_elapsed,
            "llm_waited_before_exec_sec": llm_waited_before_exec_sec,
            "ok": returncode != -1,
        })
        checkpoint_state["cumulative_env_replay_cost_sec"] += float(step_debug[-1]["elapsed"])
        checkpoint_state["cumulative_llm_replay_cost_sec"] += float(llm_step_elapsed)

        # PRM: dispatch async scoring right after execution result is ready.
        skip_prm = is_submit and getattr(prm_agent, "skip_submit", True)
        if prm_agent is not None and args is not None and not skip_prm:
            prm_pending_tasks.append((
                step_idx,
                prm_agent.submit_step_judge(
                    args,
                    problem_statement=instance.get("problem_statement", ""),
                    step_debug=list(step_debug),
                    policy_response=assistant_text,
                    step_index=step_idx,
                ),
            ))

        if is_submit:
            exit_status = "submitted"
            candidate_patch = _extract_patch_from_submission(output)
            if _is_valid_git_patch(candidate_patch):
                git_patch = candidate_patch
                patch_source = "submission"
            break

        observation = _render_observation(sweagent_config, returncode, output)
        remaining = step_limit - (step_idx + 1)
        if remaining == 1:
            observation += "\nREMINDER: You only have 1 turn left. Please provide the final answer"
        elif remaining > 1:
            observation += f"\nREMINDER: You have {remaining} turns left to arrive at the solution."
        messages.append({"role": "user", "content": observation})
        step_idx += 1

    if git_patch is None:
        try:
            diff_wait_started_at = time.time()
            logger.info(f"[SWE-R] [{iid}] Waiting for diff slot...")
            async with _get_diff_semaphore():
                logger.info(
                    f"[SWE-R] [{iid}] Diff slot acquired ({time.time() - diff_wait_started_at:.1f}s)"
                )
                diff_result = await env_client.diff(lease_id=lease_id, cwd=cwd)
            fallback_patch = diff_result if isinstance(diff_result, str) else ""
            if _is_valid_git_patch(fallback_patch):
                git_patch = fallback_patch
                patch_source = "git_diff_fallback"
            if exit_status is None:
                exit_status = "max_steps"
            logger.info(f"[SWE-R] [{iid}] Diff slot released")
        except Exception as e:
            error = f"diff failed: {e}"

    # PRM: collect all pending results
    prm_step_scores: list[float] = []
    prm_step_details: list[dict] = []
    if prm_agent is not None and prm_pending_tasks:
        prm_step_scores, prm_step_details = await prm_agent.collect_step_results(prm_pending_tasks)

    logger.info(
        f"[SWE-R] [{iid}] Agent done: steps={n_steps}, exit={exit_status}, "
        f"patch={'yes' if git_patch else 'no'}, "
        f"prm_steps={len(prm_step_scores)}, elapsed={time.time()-t0:.1f}s"
    )
    logger.info(
        "[SWE-R] [{}] checkpoint summary: policy={} attempts={} created={} checkpoint_busy_skips={} probe_count={} probe_busy_skips={} rerun_from_checkpoint={} rerun_from_base={} latest_ready_checkpoint_id={} latest_ready_checkpoint_step={}",
        iid,
        policy,
        checkpoint_state["checkpoint_attempts"],
        checkpoint_state["checkpoint_created"],
        checkpoint_state["checkpoint_busy_skips"],
        checkpoint_state["probe_count"],
        checkpoint_state["probe_busy_skips"],
        checkpoint_state["rerun_from_checkpoint"],
        checkpoint_state["rerun_from_base"],
        checkpoint_state["latest_ready_checkpoint_id"],
        checkpoint_state["latest_ready_checkpoint_step"],
    )
    return {
        "messages": messages,
        "step_debug": step_debug,
        "git_patch": git_patch,
        "patch_source": patch_source,
        "exit_status": exit_status,
        "n_steps": n_steps,
        "error": error,
        "prm_step_scores": prm_step_scores,
        "prm_step_details": prm_step_details,
        "managed_contexts": managed_contexts,
        "assistant_texts": assistant_texts,
        "checkpoint_policy": policy,
        "injection_target": {
            "mode": "exec_server_probability",
            "armed": bool(fault_injection_armed),
            "fault_injection_probability": float(fault_injection_probability),
        },
        "checkpoint_events": checkpoint_state["checkpoint_events"],
        "failure_events": checkpoint_state["failure_events"],
        "rerun_events": checkpoint_state["rerun_events"],
        "phase_events": phase_events,
        "replay_completion": (
            replay_completion_provider.summary()
            if replay_completion_provider is not None
            else None
        ),
        "checkpoint_metrics": {
            "checkpoint_attempts": checkpoint_state["checkpoint_attempts"],
            "checkpoint_created": checkpoint_state["checkpoint_created"],
            "checkpoint_busy_skips": checkpoint_state["checkpoint_busy_skips"],
            "probe_count": checkpoint_state["probe_count"],
            "probe_busy_skips": checkpoint_state["probe_busy_skips"],
            "rerun_from_checkpoint": checkpoint_state["rerun_from_checkpoint"],
            "rerun_from_base": checkpoint_state["rerun_from_base"],
            "cumulative_env_replay_cost_sec": checkpoint_state["cumulative_env_replay_cost_sec"],
            "cumulative_llm_replay_cost_sec": checkpoint_state["cumulative_llm_replay_cost_sec"],
            "latest_ready_protected_env_cost_sec": checkpoint_state["latest_ready_protected_env_cost_sec"],
            "latest_ready_protected_llm_cost_sec": checkpoint_state["latest_ready_protected_llm_cost_sec"],
        },
    }


def _ensure_openai_base_url(args):
    """Set OPENAI_BASE_URL from the framework's auto-detected router address.

    When OPENAI_BASE_URL is 'auto' or unset, derive it from
    args.sglang_router_ip / args.sglang_router_port which are populated
    by _start_router() after the router is actually running.
    """
    current = os.environ.get("OPENAI_BASE_URL", "")
    if current and current != "auto":
        return
    router_ip = getattr(args, "sglang_router_ip", None)
    router_port = getattr(args, "sglang_router_port", None)
    if router_ip and router_port:
        url = f"http://{router_ip}:{router_port}/v1"
        os.environ["OPENAI_BASE_URL"] = url
        logger.info(f"[SWE-R] OPENAI_BASE_URL resolved to {url}")


async def generate(args, sample: Sample, sampling_params: dict) -> Sample | list[Sample]:
    """Called by slime via ``--custom-generate-function-path``.

    Each sample corresponds to a SWE-Bench instance executed inside a
    remote Docker container via swe_env_pool_server + swe_exec_server.
    """
    rollout_timeout = float(os.getenv("SWE_ROLLOUT_TIMEOUT", "2400"))
    iid = (
        sample.metadata.get("instance", {}).get("instance_id", "unknown")
        if isinstance(sample.metadata, dict)
        else "unknown"
    )
    return await _generate_impl(args, sample, sampling_params, rollout_timeout=rollout_timeout)


async def _generate_impl(
    args,
    sample: Sample,
    sampling_params: dict,
    rollout_timeout: float,
) -> Sample | list[Sample]:
    """Core implementation — runtime timeout starts after the agent container is ready."""
    if GenerateState is None:
        raise RuntimeError(
            "The slime training runtime is required for generate(); "
            "install the slime dependencies or use the replay runner entrypoint."
        )
    _ensure_openai_base_url(args)
    state = GenerateState(args)
    instance = sample.metadata.get("instance", {})
    data_source = sample.metadata.get("data_source", "swe-gym")
    iid = instance.get("instance_id", "unknown")
    checkpoint_policy = _get_checkpoint_policy()
    tail_model = _get_rollout_adaptive_tail_model()
    fault_injection_armed = fault_injection_armed_for_policy(
        checkpoint_policy,
        _env_flag("SWE_FAULT_INJECTION_ENABLE", False),
    )
    fault_injection_probability = _get_fault_injection_probability()

    # PRM agent initialization
    prm_agent = None
    if getattr(args, "prm_enable", False):
        from swe_prm import SweRewardAgent
        prm_agent = SweRewardAgent(
            max_history_steps=int(getattr(args, "swe_prm_max_history_steps", 8)),
            max_problem_len=int(getattr(args, "swe_prm_max_problem_len", 8000)),
            max_output_len=int(getattr(args, "swe_prm_max_output_len", 4000)),
            max_history_output_len=int(getattr(args, "swe_prm_max_history_output_len", 1000)),
            skip_submit=bool(getattr(args, "swe_prm_skip_submit", True)),
            tokenizer=state.tokenizer,
        )

    t_start = time.time()
    swe_env_url = os.getenv("SWE_ENV_SERVER_URL", "?")
    logger.info(
        "[SWE-R] ========== REMOTE ROLLOUT ENTERED ========== instance_id={} | SWE_ENV_SERVER={} | data_source={}",
        iid, swe_env_url, data_source,
    )
    logger.info(f"[SWE-R] [{iid}] Step 1/5: generate() called, data_source={data_source}")

    sweagent_config = _get_sweagent_config()
    step_limit = int(sweagent_config.get("agent", {}).get("step_limit", 20))
    image_name = get_docker_image_name(instance, data_source)

    model_config = sweagent_config.get("model", {})
    litellm_model_name = (
        model_config.get("model_name")
        or os.getenv("SWE_LITELLM_MODEL_NAME")
        or "openai/Qwen/Qwen3-8B"
    )
    model_config["model_name"] = litellm_model_name
    model_config.setdefault("model_kwargs", {}).update({
        "temperature": sampling_params.get("temperature", 1.0),
        "max_tokens": sampling_params.get("max_new_tokens", 4096),
    })

    env_client = SweEnvClient()
    swe_semaphore = _get_swe_semaphore()
    eval_semaphore = _get_eval_semaphore()
    docker_create_limiter = _get_docker_create_limiter()

    logger.info(
        "[SWE-R] [{}] rollout policy summary: checkpoint_policy={} fault_injection_enabled={} fault_injection_armed={} fault_injection_probability={} adaptive_budget_sec={} adaptive_failure_prob={} adaptive_decision_interval_sec={}",
        iid,
        checkpoint_policy,
        _env_flag("SWE_FAULT_INJECTION_ENABLE", False),
        fault_injection_armed,
        fault_injection_probability,
        os.getenv("SWE_ADAPTIVE_CHECKPOINT_BUDGET_SEC", str(DEFAULT_ADAPTIVE_BUDGET_SEC)),
        os.getenv("SWE_ADAPTIVE_FAILURE_PROB", str(DEFAULT_ADAPTIVE_FAILURE_PROB)),
        os.getenv("SWE_ADAPTIVE_DECISION_INTERVAL_SEC", str(DEFAULT_ADAPTIVE_DECISION_INTERVAL_SEC)),
    )

    logger.info(f"[SWE-R] [{iid}] Step 1/5: Waiting for semaphore...")
    await swe_semaphore.acquire()
    logger.info(f"[SWE-R] [{iid}] Step 1/5: Semaphore acquired ({time.time()-t_start:.1f}s)")

    lease_id = None
    eval_lease_id = None
    eval_slot_acquired = False
    run_info = {"messages": [], "step_debug": [], "reward": 0, "error": None,
                "git_patch": None, "patch_source": None, "exit_status": None, "n_steps": 0, "eval_result": None,
                "checkpoint_policy": checkpoint_policy,
                "phase_events": [],
                "injection_target": {
                    "mode": "exec_server_probability",
                    "armed": bool(fault_injection_armed),
                    "fault_injection_probability": float(fault_injection_probability),
                }}
    timed_out = False
    agent_runtime_timeout = float(os.getenv("SWE_AGENT_RUNTIME_TIMEOUT", str(rollout_timeout)))
    eval_runtime_timeout = float(os.getenv("SWE_EVAL_TIMEOUT", str(rollout_timeout)))
    eval_wait_timeout = float(os.getenv("SWE_EVALUATE_WAIT_TIMEOUT_SEC", str(eval_runtime_timeout + 60.0)))

    async def _run_after_semaphore() -> None:
        nonlocal lease_id, eval_lease_id, eval_slot_acquired, run_info
        logger.info(f"[SWE-R] [{iid}] Step 2/5: Allocating container for {image_name}")
        lease = await docker_create_limiter.allocate(env_client, image=image_name, instance_id=iid)
        lease_id = lease["lease_id"]
        logger.info(f"[SWE-R] [{iid}] Step 2/5: Container ready, lease={lease_id}")

        max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        max_new_tokens = int(model_config.get("model_kwargs", {}).get("max_tokens", 4096))
        cm_max_input_tokens = max(1, max_context_len - max_new_tokens) if max_context_len > 0 else None
        cm_head_ratio = float(getattr(args, "swe_cm_head_ratio", 0.3))

        logger.info(
            f"[SWE-R] [{iid}] Step 3/5: Running agent... "
            f"(cm_max_input_tokens={cm_max_input_tokens}, cm_head_ratio={cm_head_ratio})"
        )
        try:
            agent_result = await asyncio.wait_for(
                _run_agent_remote(
                    env_client, lease_id, instance, litellm_model_name, model_config, sweagent_config,
                    args=args, prm_agent=prm_agent,
                    tokenizer=state.tokenizer,
                    cm_max_input_tokens=cm_max_input_tokens,
                    cm_head_ratio=cm_head_ratio,
                    policy=checkpoint_policy,
                    tail_model=tail_model,
                    fault_injection_armed=fault_injection_armed,
                    fault_injection_probability=fault_injection_probability,
                    docker_create_limiter=docker_create_limiter,
                    image_name=image_name,
                ),
                timeout=agent_runtime_timeout,
            )
        except (asyncio.TimeoutError, TimeoutError):
            raise TimeoutError(f"agent_runtime_timeout_after_container_ready:{agent_runtime_timeout}")
        run_info.update(agent_result)

        git_patch = run_info.get("git_patch")
        if git_patch:
            try:
                await env_client.close(lease_id)
                logger.info(f"[SWE-R] [{iid}] Step 3/5: Closed agent container lease={lease_id}")
                lease_id = None
            except Exception:
                logger.exception(f"[SWE-R] [{iid}] Failed to close agent lease before eval")

            eval_wait_started_at = time.time()
            logger.info(f"[SWE-R] [{iid}] Step 4/5: Waiting for evaluation slot...")
            await eval_semaphore.acquire()
            eval_slot_acquired = True
            logger.info(
                f"[SWE-R] [{iid}] Step 4/5: Evaluation slot acquired ({time.time() - eval_wait_started_at:.1f}s)"
            )
            try:
                logger.info(f"[SWE-R] [{iid}] Step 4/5: Allocating fresh eval container...")
                eval_lease = await docker_create_limiter.allocate(
                    env_client, image=image_name, instance_id=f"{iid}__eval"
                )
                eval_lease_id = eval_lease["lease_id"]
                logger.info(f"[SWE-R] [{iid}] Step 4/5: Eval container ready, lease={eval_lease_id}")
                try:
                    eval_result = await asyncio.wait_for(
                        env_client.evaluate(
                            lease_id=eval_lease_id,
                            patch=git_patch,
                            eval_script=instance.get("eval_script", ""),
                            timeout=eval_runtime_timeout,
                        ),
                        timeout=eval_wait_timeout,
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    raise TimeoutError(f"eval_timeout_after_container_ready:{eval_runtime_timeout}")
                resolved = eval_result.get("resolved", False)
                run_info["reward"] = int(resolved)
                run_info["eval_result"] = eval_result
                logger.info(f"[SWE-R] [{iid}] Step 4/5: resolved={resolved}")
            except TimeoutError:
                raise
            except Exception as e:
                run_info["error"] = str(e)
                logger.error(f"[SWE-R] [{iid}] Step 4/5: Eval error: {e}")
            finally:
                if eval_lease_id is not None:
                    try:
                        await env_client.close(eval_lease_id)
                    except BaseException:
                        logger.warning(f"[SWE-R] [{iid}] Failed to close eval lease (may be cancelled)")
                    finally:
                        eval_lease_id = None
                if eval_slot_acquired:
                    eval_semaphore.release()
                    eval_slot_acquired = False
                    logger.info(f"[SWE-R] [{iid}] Step 4/5: Evaluation slot released")
        else:
            logger.warning(f"[SWE-R] [{iid}] Step 4/5: No patch, skipping eval")

    try:
        try:
            await _run_after_semaphore()
        except TimeoutError as e:
            timed_out = True
            run_info["error"] = str(e)
            logger.error(
                f"[SWE-R] [{iid}] ROLLOUT TIMEOUT after container ready: {run_info['error']}, aborting sample"
            )
        except Exception as e:
            run_info["error"] = str(e)
            logger.exception(f"[SWE-R] [{iid}] Error: {e}")
    finally:
        if eval_lease_id is not None:
            try:
                await env_client.close(eval_lease_id)
            except BaseException:
                logger.warning(f"[SWE-R] [{iid}] Failed to close eval lease (may be cancelled)")
        if eval_slot_acquired:
            eval_semaphore.release()
            eval_slot_acquired = False
            logger.warning(f"[SWE-R] [{iid}] Evaluation slot released from outer cleanup path")
        if lease_id is not None:
            try:
                await env_client.close(lease_id)
            except BaseException:
                logger.warning(f"[SWE-R] [{iid}] Failed to close lease (may be cancelled)")
        swe_semaphore.release()
        logger.info(f"[SWE-R] [{iid}] Semaphore released")

    if timed_out:
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    messages = run_info["messages"]
    reward = run_info["reward"]
    error = run_info["error"]
    managed_contexts = run_info.get("managed_contexts", [])
    assistant_texts = run_info.get("assistant_texts", [])

    _save_rollout_artifacts(sample=sample, iid=iid, sampling_params=sampling_params, run_info=run_info)

    if not messages:
        logger.warning(f"[SWE-R] [{iid}] Step 5/5: ABORTED — no messages (error={error})")
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    use_dynamic_history = getattr(args, "dynamic_history", False) and managed_contexts and assistant_texts

    outcome_reward = 1.0 if reward else -1.0
    prm_step_scores = run_info.get("prm_step_scores", [])
    prm_step_details = run_info.get("prm_step_details", [])

    # ------------------------------------------------------------------
    # Dynamic-history path: one training sample per step, each with
    # the managed context the model actually saw during rollout.
    # ------------------------------------------------------------------
    if use_dynamic_history:
        dynamic_samples: list[Sample] = []
        n_steps = min(len(managed_contexts), len(assistant_texts))

        for step_idx in range(n_steps):
            ctx_msgs = managed_contexts[step_idx]
            resp_text = assistant_texts[step_idx]

            prompt_ids = state.tokenizer.apply_chat_template(
                ctx_msgs, add_generation_prompt=True, tokenize=True,
            )
            resp_msgs = [{"role": "assistant", "content": resp_text}]
            response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
                resp_msgs, state.tokenizer, assistant_logprobs=None,
            )

            max_ctx = int(getattr(args, "rollout_max_context_len", 0) or 0)
            if max_ctx > 0:
                max_resp = max(1, max_ctx - len(prompt_ids))
                if len(response_ids) > max_resp:
                    response_ids = response_ids[:max_resp]
                    loss_mask = loss_mask[:max_resp]

            child = copy.deepcopy(sample)
            child.tokens = prompt_ids + response_ids
            child.response = resp_text
            child.response_length = len(response_ids)
            child.loss_mask = loss_mask
            child.rollout_log_probs = None
            child.status = Sample.Status.COMPLETED if response_ids else Sample.Status.ABORTED

            child.metadata = copy.deepcopy(sample.metadata or {})
            child.metadata["dynamic_step_index"] = step_idx
            child.metadata["dynamic_outcome_reward"] = float(outcome_reward)
            child.metadata["num_steps"] = n_steps

            step_reward = float(outcome_reward)
            prm_score = 0.0
            if step_idx < len(prm_step_scores):
                prm_score = float(prm_step_scores[step_idx])

            if getattr(args, "prm_enable", False):
                child.metadata["step_wise"] = {
                    "step_scores": [prm_score],
                    "step_indices": [step_idx],
                    "step_token_spans": [[0, len(response_ids)]],
                    "step_scores_with_outcome": [prm_score + step_reward],
                    "outcome_reward": step_reward,
                }
                child.reward = None
            else:
                child.reward = {"score": step_reward, "acc": float(reward)}

            if child.status == Sample.Status.ABORTED:
                child.reward = child.reward or {"score": 0.0, "acc": 0.0}
                child.remove_sample = True

            dynamic_samples.append(child)

        if not dynamic_samples:
            sample.status = Sample.Status.ABORTED
            sample.reward = {"score": 0.0, "acc": 0.0}
            sample.remove_sample = True
            return [sample]

        if getattr(args, "prm_enable", False) and prm_step_scores:
            for child in dynamic_samples:
                child.metadata["prm"] = {
                    "enabled": True,
                    "step_scores": prm_step_scores,
                    "step_mean_score": (
                        sum(prm_step_scores) / len(prm_step_scores)
                    ),
                    "step_details": prm_step_details,
                }

        elapsed = time.time() - t_start
        logger.info(
            f"[SWE-R] [{iid}] Step 5/5: DONE (dynamic_history) — "
            f"n_samples={len(dynamic_samples)}, outcome_reward={outcome_reward}, "
            f"prm_enabled={getattr(args, 'prm_enable', False)}, "
            f"total_elapsed={elapsed:.1f}s"
        )
        return dynamic_samples

    # ------------------------------------------------------------------
    # Default path: single training sample from full messages.
    # ------------------------------------------------------------------
    prompt_messages = messages[:2]
    response_messages = messages[2:]

    while response_messages and response_messages[-1]["role"] == "user":
        response_messages.pop()

    if not response_messages:
        logger.warning(f"[SWE-R] [{iid}] Step 5/5: ABORTED — no assistant messages")
        sample.status = Sample.Status.ABORTED
        sample.reward = {"score": 0.0, "acc": 0.0}
        sample.remove_sample = True
        return sample

    prompt_ids = state.tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, tokenize=True
    )
    response_ids, loss_mask, _ = get_response_ids_and_loss_mask_from_messages(
        response_messages, state.tokenizer, assistant_logprobs=None
    )

    max_context_len = getattr(args, "rollout_max_context_len", None)
    if max_context_len is not None:
        max_response_tokens = max(1, int(max_context_len) - len(prompt_ids))
    else:
        max_response_tokens = getattr(args, "rollout_max_response_len", 4096)
    if len(response_ids) > max_response_tokens:
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        sample.status = Sample.Status.TRUNCATED

    sample.tokens = prompt_ids + response_ids
    sample.response = "\n".join(m["content"] for m in response_messages if m["role"] == "assistant")
    sample.response_length = len(response_ids)
    sample.loss_mask = loss_mask
    sample.rollout_log_probs = None
    if sample.status == Sample.Status.PENDING:
        sample.status = Sample.Status.COMPLETED

    # PRM metadata
    if getattr(args, "prm_enable", False):
        sample.metadata = sample.metadata or {}
        sample.metadata["prm"] = {
            "enabled": True,
            "step_scores": prm_step_scores,
            "step_mean_score": (sum(prm_step_scores) / len(prm_step_scores)) if prm_step_scores else 0.0,
            "step_details": prm_step_details,
        }

        step_token_spans = _extract_assistant_turn_spans(loss_mask)
        n_aligned = min(len(prm_step_scores), len(step_token_spans))
        sample.metadata["step_wise"] = {
            "step_scores": prm_step_scores[:n_aligned],
            "step_indices": list(range(n_aligned)),
            "step_token_spans": step_token_spans[:n_aligned],
            "step_scores_with_outcome": [
                float(s) + outcome_reward for s in prm_step_scores[:n_aligned]
            ],
            "outcome_reward": outcome_reward,
        }
        sample.reward = None
    else:
        sample.reward = {"score": outcome_reward, "acc": float(reward)}

    elapsed = time.time() - t_start
    logger.info(
        f"[SWE-R] [{iid}] Step 5/5: DONE — status={sample.status.name}, "
        f"reward={sample.reward}, response_len={sample.response_length}, "
        f"prm_enabled={getattr(args, 'prm_enable', False)}, "
        f"total_elapsed={elapsed:.1f}s"
    )
    return sample


async def reward_func(args, sample: Sample | list[Sample], **kwargs):
    """Compute reward, integrating PRM step-wise scores when enabled."""

    prm_step_coef = float(getattr(args, "prm_step_coef", 1.0))

    def _get_reward(s: Sample) -> dict:
        if getattr(args, "prm_enable", False) and isinstance(s.metadata, dict):
            prm_meta = s.metadata.get("prm", {})
            step_wise_meta = s.metadata.get("step_wise", {})
            outcome_reward = step_wise_meta.get("outcome_reward", 0.0)
            prm_step_mean = float(prm_meta.get("step_mean_score", 0.0))
            final_score = outcome_reward + prm_step_coef * prm_step_mean
            return {
                "score": final_score,
                "acc": 1.0 if outcome_reward > 0 else 0.0,
                "outcome_reward": outcome_reward,
                "prm_step_mean": prm_step_mean,
                "prm_step_coef": prm_step_coef,
            }

        if isinstance(s.reward, dict):
            return s.reward
        acc = float(s.metadata.get("eval_score", 0.0)) if isinstance(s.metadata, dict) else 0.0
        return {"score": 1.0 if acc == 1.0 else -1.0, "acc": acc}

    if isinstance(sample, list):
        return [_get_reward(s) for s in sample]
    return _get_reward(sample)
