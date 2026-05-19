from __future__ import annotations

import bisect
import json
import os
import random
import time
from dataclasses import asdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


POLICIES = ("oracle-no-fault-no-checkpoint", "never", "always", "adaptive-risk", "every-3")
DEFAULT_ADAPTIVE_TAIL_ROOT = Path(__file__).resolve().parents[1] / "export" / "swe_rollouts_profile_20260325_083236"
DEFAULT_ADAPTIVE_BUDGET_SEC = 7.0
DEFAULT_ADAPTIVE_DECISION_INTERVAL_SEC = 1.0
DEFAULT_ADAPTIVE_FAILURE_PROB = 0.003
DEFAULT_ADAPTIVE_MIN_DELTA_ENV_COST_SEC = 0.1
DEFAULT_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS = 4
DEFAULT_FAULT_INJECTION_OFFSET_FROM_END_STEPS = 3
DEFAULT_FAULT_INJECTION_MAX_STEP_IDX = 8
DEFAULT_FAULT_INJECTION_SELECTION_MODE = "longest"


@dataclass
class AdaptiveTailModel:
    sorted_waits: list[float]
    prefix_sums: list[float]
    budget_sec: float

    @classmethod
    def from_waits(cls, waits: list[float], budget_sec: float) -> "AdaptiveTailModel":
        ordered = sorted(value for value in waits if value > 0.0)
        prefix_sums = [0.0]
        for value in ordered:
            prefix_sums.append(prefix_sums[-1] + value)
        return cls(sorted_waits=ordered, prefix_sums=prefix_sums, budget_sec=float(budget_sec))

    @property
    def count(self) -> int:
        return len(self.sorted_waits)

    def conditional_tail_probability(self, waited_sec: float, horizon_sec: float | None = None) -> float:
        if not self.sorted_waits:
            return 0.0
        horizon = self.budget_sec if horizon_sec is None else float(horizon_sec)
        lo = bisect.bisect_right(self.sorted_waits, waited_sec)
        survivors = len(self.sorted_waits) - lo
        if survivors <= 0:
            return 0.0
        hi = bisect.bisect_right(self.sorted_waits, waited_sec + horizon)
        tail_count = len(self.sorted_waits) - hi
        return tail_count / survivors

    def expected_exposed_overhead(self, waited_sec: float) -> float:
        if not self.sorted_waits:
            return self.budget_sec
        lo = bisect.bisect_right(self.sorted_waits, waited_sec)
        survivors = len(self.sorted_waits) - lo
        if survivors <= 0:
            return self.budget_sec
        hi = bisect.bisect_right(self.sorted_waits, waited_sec + self.budget_sec)
        partial_count = hi - lo
        if partial_count <= 0:
            return 0.0
        partial_sum = self.prefix_sums[hi] - self.prefix_sums[lo]
        exposed_total = partial_count * (waited_sec + self.budget_sec) - partial_sum
        return max(0.0, exposed_total / survivors)


@dataclass
class FaultInjectionTarget:
    sample_key: str
    inject_before_step_idx: int


def redo_env_replay_cost_sec(cumulative_env_replay_cost_sec: float, latest_ready_protected_env_cost_sec: float) -> float:
    return max(0.0, float(cumulative_env_replay_cost_sec) - float(latest_ready_protected_env_cost_sec))


def adaptive_delta_env_cost_sec(
    cumulative_env_replay_cost_sec: float,
    latest_ready_protected_env_cost_sec: float,
) -> float:
    return max(0.0, float(cumulative_env_replay_cost_sec) - float(latest_ready_protected_env_cost_sec))


def redo_replay_cost_sec(
    cumulative_env_replay_cost_sec: float,
    latest_ready_protected_env_cost_sec: float,
    cumulative_llm_replay_cost_sec: float,
    latest_ready_protected_llm_cost_sec: float,
) -> float:
    env_redo = redo_env_replay_cost_sec(
        cumulative_env_replay_cost_sec,
        latest_ready_protected_env_cost_sec,
    )
    llm_redo = max(0.0, float(cumulative_llm_replay_cost_sec) - float(latest_ready_protected_llm_cost_sec))
    return env_redo + llm_redo


def adaptive_delta_replay_cost_sec(
    cumulative_env_replay_cost_sec: float,
    latest_ready_protected_env_cost_sec: float,
    cumulative_llm_replay_cost_sec: float,
    latest_ready_protected_llm_cost_sec: float,
) -> float:
    env_delta = adaptive_delta_env_cost_sec(
        cumulative_env_replay_cost_sec,
        latest_ready_protected_env_cost_sec,
    )
    llm_delta = max(0.0, float(cumulative_llm_replay_cost_sec) - float(latest_ready_protected_llm_cost_sec))
    return env_delta + llm_delta


def adaptive_expected_benefit_sec(
    failure_prob: float,
    conditional_tail_prob: float,
    redo_replay_cost_sec_value: float,
) -> float:
    return max(0.0, float(failure_prob) * float(conditional_tail_prob) * float(redo_replay_cost_sec_value))


def should_probe_in_llm_bubble(
    *,
    current_step_idx: int,
    probe_attempted_in_bubble: bool,
    adaptive_checkpoint_submitted: bool,
    pending_checkpoints: list[dict],
    delta_env_cost_sec_value: float,
    steps_since_latest_ready_checkpoint: int,
    expected_benefit_sec: float,
    expected_overhead_sec: float,
    min_delta_env_cost_sec: float = DEFAULT_ADAPTIVE_MIN_DELTA_ENV_COST_SEC,
    min_steps_between_checkpoints: int = DEFAULT_ADAPTIVE_MIN_STEPS_BETWEEN_CHECKPOINTS,
) -> bool:
    if current_step_idx < 3:
        return False
    if probe_attempted_in_bubble or adaptive_checkpoint_submitted or pending_checkpoints:
        return False
    if delta_env_cost_sec_value < float(min_delta_env_cost_sec):
        return False
    if steps_since_latest_ready_checkpoint < int(min_steps_between_checkpoints):
        return False
    return expected_benefit_sec > expected_overhead_sec


class LongTrajectoryFaultPlanner:
    def __init__(
        self,
        *,
        plan_path: str | Path,
        reference_root: str | Path,
        expected_sample_count: int,
        injection_count: int,
        seed: int,
        selection_mode: str = DEFAULT_FAULT_INJECTION_SELECTION_MODE,
        offset_from_end_steps: int = DEFAULT_FAULT_INJECTION_OFFSET_FROM_END_STEPS,
        max_inject_before_step_idx: int = DEFAULT_FAULT_INJECTION_MAX_STEP_IDX,
        poll_interval_sec: float = 0.2,
        wait_timeout_sec: float = 120.0,
    ) -> None:
        self.plan_path = Path(plan_path)
        self.lock_path = self.plan_path.with_suffix(self.plan_path.suffix + ".lock")
        self.reference_root = Path(reference_root)
        self.expected_sample_count = max(0, int(expected_sample_count))
        self.injection_count = max(0, int(injection_count))
        self.seed = int(seed)
        self.selection_mode = str(selection_mode or DEFAULT_FAULT_INJECTION_SELECTION_MODE).strip().lower()
        if self.selection_mode not in {"longest", "random"}:
            raise ValueError(f"unsupported fault injection selection mode: {self.selection_mode}")
        self.offset_from_end_steps = max(1, int(offset_from_end_steps))
        self.max_inject_before_step_idx = max(1, int(max_inject_before_step_idx))
        self.poll_interval_sec = max(0.01, float(poll_interval_sec))
        self.wait_timeout_sec = max(0.0, float(wait_timeout_sec))

    def register_and_wait(self, *, sample_key: str, instance_id: str, step_limit: int) -> FaultInjectionTarget | None:
        deadline = time.time() + self.wait_timeout_sec
        self.register(sample_key=sample_key, instance_id=instance_id, step_limit=step_limit)
        while True:
            finalized, target = self.poll_target(
                sample_key=sample_key,
                force_finalize=time.time() >= deadline,
            )
            if finalized:
                return target
            time.sleep(self.poll_interval_sec)

    def register(self, *, sample_key: str, instance_id: str, step_limit: int) -> dict:
        return self._register_and_maybe_finalize(
            sample_key=sample_key,
            instance_id=instance_id,
            step_limit=step_limit,
            force_finalize=False,
        )

    def poll_target(self, *, sample_key: str, force_finalize: bool) -> tuple[bool, FaultInjectionTarget | None]:
        state = self._poll_state_and_maybe_finalize(force_finalize=force_finalize)
        selected = state.get("selected", {})
        if not state.get("finalized", False):
            return False, None
        target = selected.get(sample_key)
        if not isinstance(target, dict):
            return True, None
        return True, FaultInjectionTarget(
            sample_key=sample_key,
            inject_before_step_idx=int(target["inject_before_step_idx"]),
        )

    def _register_and_maybe_finalize(
        self,
        *,
        sample_key: str,
        instance_id: str,
        step_limit: int,
        force_finalize: bool,
    ) -> dict:
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            raise RuntimeError("fcntl is required for RandomFaultInjectionPlanner on this platform")
        with self.lock_path.open("a+", encoding="utf-8") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            state = self._load_state_unlocked()
            registered = state.setdefault("registered", {})
            reference_lengths = _load_reference_trajectory_lengths(self.reference_root)
            registered[sample_key] = {
                "instance_id": str(instance_id),
                "step_limit": int(step_limit),
                "reference_step_count": int(
                    reference_lengths["sample_keys"].get(
                        sample_key,
                        reference_lengths["instance_ids"].get(str(instance_id), 0),
                    )
                ),
            }
            should_finalize = False
            if not state.get("finalized", False):
                if self.expected_sample_count > 0 and len(registered) >= self.expected_sample_count:
                    should_finalize = True
                elif force_finalize and len(registered) >= self.injection_count:
                    should_finalize = True
            if should_finalize:
                state["selected"] = self._select_targets(registered)
                state["finalized"] = True
                state["finalized_at"] = time.time()
            self._write_state_unlocked(state)
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            return state

    def _poll_state_and_maybe_finalize(self, *, force_finalize: bool) -> dict:
        self.plan_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None:
            raise RuntimeError("fcntl is required for RandomFaultInjectionPlanner on this platform")
        with self.lock_path.open("a+", encoding="utf-8") as lock_fp:
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
            state = self._load_state_unlocked()
            registered = state.setdefault("registered", {})
            should_finalize = False
            if not state.get("finalized", False):
                if self.expected_sample_count > 0 and len(registered) >= self.expected_sample_count:
                    should_finalize = True
                elif force_finalize and len(registered) >= self.injection_count:
                    should_finalize = True
            if should_finalize:
                state["selected"] = self._select_targets(registered)
                state["finalized"] = True
                state["finalized_at"] = time.time()
                self._write_state_unlocked(state)
            fcntl.flock(lock_fp.fileno(), fcntl.LOCK_UN)
            return state

    def _load_state_unlocked(self) -> dict:
        if self.plan_path.exists():
            try:
                state = json.loads(self.plan_path.read_text(encoding="utf-8"))
                if isinstance(state, dict):
                    return state
            except Exception:
                pass
        return {
            "expected_sample_count": self.expected_sample_count,
            "injection_count": self.injection_count,
            "seed": self.seed,
            "selection_mode": self.selection_mode,
            "reference_root": str(self.reference_root),
            "offset_from_end_steps": self.offset_from_end_steps,
            "max_inject_before_step_idx": self.max_inject_before_step_idx,
            "registered": {},
            "selected": {},
            "finalized": False,
        }

    def _write_state_unlocked(self, state: dict) -> None:
        payload = json.dumps(state, ensure_ascii=True, indent=2, sort_keys=True)
        self.plan_path.write_text(payload, encoding="utf-8")

    def _select_targets(self, registered: dict[str, dict]) -> dict[str, dict]:
        eligible = [
            (
                str(sample_key),
                int(item.get("reference_step_count", 0) or 0),
                int(item.get("step_limit", 0) or 0),
            )
            for sample_key, item in registered.items()
            if int(item.get("step_limit", 0) or 0) >= 2
        ]
        if not eligible or self.injection_count <= 0:
            return {}
        rng = random.Random(self.seed)
        decorated = [
            (sample_key, reference_step_count, step_limit, rng.random())
            for sample_key, reference_step_count, step_limit in eligible
        ]
        if self.selection_mode == "random":
            chosen = sorted(
                decorated,
                key=lambda item: (item[3], item[0]),
            )[: min(self.injection_count, len(decorated))]
        else:
            chosen = sorted(
                decorated,
                key=lambda item: (-item[1], -item[2], item[3], item[0]),
            )[: min(self.injection_count, len(decorated))]
        selected: dict[str, dict] = {}
        for sample_key, reference_step_count, step_limit, _ in chosen:
            reference_step_count = max(2, int(reference_step_count) or step_limit)
            inject_before_step_idx = max(
                1,
                min(
                    step_limit - 1,
                    reference_step_count - self.offset_from_end_steps,
                    self.max_inject_before_step_idx,
                ),
            )
            target = FaultInjectionTarget(
                sample_key=sample_key,
                inject_before_step_idx=inject_before_step_idx,
            )
            selected[sample_key] = asdict(target)
        return selected


def _infer_sample_key_from_traj_payload(payload: dict, traj_path: Path) -> str | None:
    info = payload.get("info", {}) if isinstance(payload, dict) else {}
    instance_id = info.get("instance_id")
    if not instance_id:
        stem = traj_path.parent.name
        if "__g" in stem and "__i" in stem:
            return stem
        return None
    return rollout_sample_key(
        instance_id=str(instance_id),
        group_index=info.get("group_index"),
        sample_index=info.get("index"),
    )


def _infer_step_count_from_traj_payload(payload: dict) -> int:
    if isinstance(payload, dict):
        if isinstance(payload.get("step_debug"), list):
            return len(payload["step_debug"])
        if isinstance(payload.get("steps"), list):
            return len(payload["steps"])
    return 0


def _infer_instance_id_from_traj_payload(payload: dict, traj_path: Path) -> str | None:
    info = payload.get("info", {}) if isinstance(payload, dict) else {}
    instance_id = info.get("instance_id")
    if instance_id:
        return str(instance_id)
    stem = traj_path.parent.name
    if "__g" in stem:
        return stem.split("__g", 1)[0]
    return None


@lru_cache(maxsize=8)
def _load_reference_trajectory_lengths(reference_root: str | Path) -> dict[str, dict[str, int]]:
    root = Path(reference_root)
    sample_keys: dict[str, int] = {}
    instance_ids: dict[str, int] = {}
    if not root.exists():
        return {"sample_keys": sample_keys, "instance_ids": instance_ids}
    for traj_path in root.glob("**/traj.json"):
        try:
            payload = json.loads(traj_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        step_count = _infer_step_count_from_traj_payload(payload)
        if step_count <= 0:
            continue
        sample_key = _infer_sample_key_from_traj_payload(payload, traj_path)
        if sample_key:
            sample_keys[sample_key] = max(sample_keys.get(sample_key, 0), step_count)
        instance_id = _infer_instance_id_from_traj_payload(payload, traj_path)
        if instance_id:
            instance_ids[instance_id] = max(instance_ids.get(instance_id, 0), step_count)
    return {"sample_keys": sample_keys, "instance_ids": instance_ids}


def rollout_sample_key(*, instance_id: str, group_index: object, sample_index: object) -> str:
    group_part = "na" if group_index is None else str(group_index)
    sample_part = "na" if sample_index is None else str(sample_index)
    return f"{instance_id}__g{group_part}__i{sample_part}"


def fault_injection_enabled_for_policy(policy: str, enabled: bool) -> bool:
    return bool(enabled)


def fault_injection_armed_for_policy(policy: str, enabled: bool) -> bool:
    return bool(enabled) and policy != "oracle-no-fault-no-checkpoint"
