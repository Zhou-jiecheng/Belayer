from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import re
import shlex
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import dedent
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[2]
SWE_RL_ROOT = Path(__file__).resolve().parents[1]
SLIME_ROOT = REPO_ROOT / "slime"

for path in (SLIME_ROOT, SWE_RL_ROOT):
    sys.path.insert(0, str(path))

from swe_utils import get_docker_image_name


DEFAULT_TRAJECTORY_ROOT = (
    REPO_ROOT / "export" / "swe_rl_static_baseline_128_no_ckpt" / "swe_rollouts"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT / "export" / f"checkpoint_correctness_validation_{time.strftime('%Y%m%d_%H%M%S')}"
)
DEFAULT_PHASES = [
    "before_action",
    "mid_action",
    "after_action_before_observation",
    "after_observation_before_checkpoint",
    "before_commit",
    "after_commit_before_ready",
    "after_checkpoint_ready",
]
RANDOM_CHECKPOINT_FAULT_PHASES = ["before_commit", "after_commit_before_ready"]
DEFAULT_DATA_SOURCE = "SumanthRH/SWE-Gym-Subset"
NON_IDEMPOTENT_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("append_redirect", re.compile(r">>")),
    ("in_place_edit", re.compile(r"\bsed\s+-i\b")),
    ("destructive_delete", re.compile(r"\brm\s+(-[^\s]+\s+)?", re.IGNORECASE)),
    ("move_or_rename", re.compile(r"\bmv\b")),
    ("git_apply", re.compile(r"\bgit\s+apply\b")),
    ("package_install", re.compile(r"\bpip(?:3)?\s+install\b")),
    ("heredoc_write", re.compile(r"cat\s+<<['\"]?EOF['\"]?\s*>")),
    ("overwrite_redirect", re.compile(r"(^|[;&|]\s*|\s)(?:echo|printf|cat)\b.*?>\s*[^>]", re.DOTALL)),
]


@dataclass(frozen=True)
class ValidationStep:
    step_idx: int
    name: str
    command: str


class ValidationEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")

    def _post_blocking(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        req = urllib_request.Request(
            f"{self.base_url}/{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"unexpected response payload for {path}: {parsed!r}")
        return parsed

    async def _post(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        return await asyncio.to_thread(self._post_blocking, path, payload, timeout)

    async def allocate(self, image: str, *, instance_id: str, cwd: str) -> dict[str, Any]:
        return await self._post(
            "allocate",
            {"image": image, "instance_id": instance_id, "cwd": cwd},
            180.0,
        )

    async def exec(
        self,
        lease_id: str,
        command: str,
        *,
        cwd: str,
        timeout: int,
        fault_injection_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lease_id": lease_id,
            "command": command,
            "cwd": cwd,
            "timeout": int(timeout),
            "env": {},
        }
        if fault_injection_spec:
            payload["fault_injection_spec"] = dict(fault_injection_spec)
        return await self._post("exec", payload, float(timeout + 30))

    async def checkpoint_create(
        self,
        lease_id: str,
        *,
        step_idx: int,
        command_seq: int,
        cwd: str,
        policy: str,
        reason: str,
        parent_checkpoint_id: str | None,
        fault_injection_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lease_id": lease_id,
            "step_idx": int(step_idx),
            "command_seq": int(command_seq),
            "cwd": cwd,
            "policy": policy,
            "reason": reason,
        }
        if parent_checkpoint_id is not None:
            payload["parent_checkpoint_id"] = parent_checkpoint_id
        if fault_injection_spec:
            payload["fault_injection_spec"] = dict(fault_injection_spec)
        return await self._post("checkpoint/create", payload, 60.0)

    async def checkpoint_list(self, lease_id: str) -> dict[str, Any]:
        return await self._post("checkpoint/list", {"lease_id": lease_id}, 30.0)

    async def rerun(
        self,
        lease_id: str,
        *,
        checkpoint_id: str | None,
        cwd: str,
        timeout: int,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lease_id": lease_id,
            "cwd": cwd,
            "timeout": int(timeout),
        }
        if checkpoint_id is not None:
            payload["checkpoint_id"] = checkpoint_id
        return await self._post("rerun", payload, float(timeout + 30))

    async def close(self, lease_id: str) -> dict[str, Any]:
        return await self._post("close", {"lease_id": lease_id}, 30.0)

    async def inject_fail_stop(
        self,
        lease_id: str,
        *,
        tag: str,
        delay_sec: float = 0.0,
    ) -> dict[str, Any]:
        return await self._post(
            "fault/kill",
            {
                "lease_id": lease_id,
                "tag": tag,
                "delay_sec": float(delay_sec),
            },
            90.0,
        )


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _write_trial_csv(path: Path, trials: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "trial_id",
        "trial_mode",
        "fault_model",
        "phase",
        "inject_step",
        "interrupted_op",
        "random_fault_delay_sec",
        "random_fault_strategy",
        "checkpoint_fault_phase",
        "expected_prefix",
        "matched_oracle_prefix",
        "hash_match",
        "final_hash_match",
        "duplicate_effect",
        "lost_effect",
        "partial_effect",
        "ok",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for trial in trials:
            semantics = trial.get("semantic_checks", {}) if isinstance(trial, dict) else {}
            writer.writerow(
                {
                    "trial_id": trial.get("trial_id"),
                    "trial_mode": trial.get("trial_mode"),
                    "fault_model": trial.get("fault_model"),
                    "phase": trial.get("phase"),
                    "inject_step": trial.get("inject_step"),
                    "interrupted_op": (
                        trial.get("interrupted_op", {}).get("op")
                        if isinstance(trial.get("interrupted_op"), dict)
                        else None
                    ),
                    "random_fault_delay_sec": trial.get("random_fault_delay_sec"),
                    "random_fault_strategy": trial.get("random_fault_strategy"),
                    "checkpoint_fault_phase": trial.get("checkpoint_fault_phase"),
                    "expected_prefix": trial.get("expected_prefix"),
                    "matched_oracle_prefix": trial.get("matched_oracle_prefix"),
                    "hash_match": trial.get("hash_match"),
                    "final_hash_match": trial.get("final_hash_match"),
                    "duplicate_effect": semantics.get("duplicate_effect"),
                    "lost_effect": semantics.get("lost_effect"),
                    "partial_effect": semantics.get("partial_effect"),
                    "ok": trial.get("ok"),
                }
            )


def _collect_traj_paths(path_value: str, limit: int | None = None) -> list[Path]:
    root = Path(path_value)
    if root.is_file():
        return [root]
    if not root.exists():
        return []
    traj_paths = sorted(p for p in root.rglob("traj.json") if p.is_file())
    if limit is not None:
        traj_paths = traj_paths[: max(0, int(limit))]
    return traj_paths


def _load_traj_payload(traj_path: Path) -> dict[str, Any]:
    return json.loads(traj_path.read_text(encoding="utf-8"))


def _load_sidecar_meta(traj_path: Path) -> dict[str, Any]:
    meta_path = traj_path.with_name("meta.json")
    if not meta_path.exists():
        return {}
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _default_instance_id(traj_payload: dict[str, Any]) -> str:
    info = traj_payload.get("info", {})
    if isinstance(info, dict):
        value = str(info.get("instance_id", "")).strip()
        if value:
            return value
    raise ValueError("instance_id not found in trajectory info")


def _extract_recorded_image_name(
    traj_payload: dict[str, Any],
    sidecar_meta: dict[str, Any],
) -> str | None:
    candidate_dicts: list[dict[str, Any]] = [traj_payload]
    info = traj_payload.get("info")
    if isinstance(info, dict):
        candidate_dicts.append(info)
    candidate_dicts.append(sidecar_meta)
    sample_metadata = sidecar_meta.get("sample_metadata")
    if isinstance(sample_metadata, dict):
        candidate_dicts.append(sample_metadata)
        instance = sample_metadata.get("instance")
        if isinstance(instance, dict):
            candidate_dicts.append(instance)
    for candidate in candidate_dicts:
        for key in ("image_name", "docker_image", "container_image", "base_image"):
            value = candidate.get(key) if isinstance(candidate, dict) else None
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _extract_recorded_data_source(
    traj_payload: dict[str, Any],
    sidecar_meta: dict[str, Any],
    fallback_data_source: str,
) -> str:
    candidate_dicts: list[dict[str, Any]] = [traj_payload]
    info = traj_payload.get("info")
    if isinstance(info, dict):
        candidate_dicts.append(info)
    candidate_dicts.append(sidecar_meta)
    sample_metadata = sidecar_meta.get("sample_metadata")
    if isinstance(sample_metadata, dict):
        candidate_dicts.append(sample_metadata)
    for candidate in candidate_dicts:
        value = candidate.get("data_source") if isinstance(candidate, dict) else None
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback_data_source


def _resolve_image_name(
    *,
    image_name: str | None,
    instance_id: str | None,
    trajectory: str | None,
    trajectory_root: str | None,
    data_source: str,
) -> tuple[str, str | None, str | None, str]:
    if image_name:
        return image_name, instance_id, trajectory, data_source
    candidate_paths: list[Path] = []
    if trajectory:
        candidate_paths = [Path(trajectory)]
    elif trajectory_root:
        candidate_paths = _collect_traj_paths(trajectory_root, limit=1)
    if not candidate_paths:
        if not instance_id:
            raise ValueError("image_name is required when no trajectory or instance_id is provided")
        return get_docker_image_name({"instance_id": instance_id}, data_source), instance_id, None, data_source
    resolved_traj = candidate_paths[0]
    traj_payload = _load_traj_payload(resolved_traj)
    sidecar_meta = _load_sidecar_meta(resolved_traj)
    resolved_instance_id = instance_id or _default_instance_id(traj_payload)
    recorded_image_name = _extract_recorded_image_name(traj_payload, sidecar_meta)
    resolved_data_source = _extract_recorded_data_source(traj_payload, sidecar_meta, data_source)
    if recorded_image_name:
        return recorded_image_name, resolved_instance_id, str(resolved_traj), resolved_data_source
    resolved_image_name = get_docker_image_name({"instance_id": resolved_instance_id}, resolved_data_source)
    return resolved_image_name, resolved_instance_id, str(resolved_traj), resolved_data_source


def _extract_non_idempotent_tags(action: str) -> list[str]:
    return [name for name, pattern in NON_IDEMPOTENT_PATTERNS if pattern.search(action or "")]


def _scan_non_idempotent_actions(
    trajectory_root: str,
    *,
    trajectory_limit: int,
    candidate_limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for traj_path in _collect_traj_paths(trajectory_root, limit=trajectory_limit):
        payload = _load_traj_payload(traj_path)
        for step in payload.get("step_debug", []) or []:
            if not isinstance(step, dict):
                continue
            action = str(step.get("action", "") or "")
            tags = _extract_non_idempotent_tags(action)
            if not tags:
                continue
            candidates.append(
                {
                    "traj_path": str(traj_path),
                    "instance_id": payload.get("info", {}).get("instance_id"),
                    "step_idx": int(step.get("step_idx", -1)),
                    "tags": tags,
                    "action": action,
                }
            )
            if len(candidates) >= candidate_limit:
                return candidates
    return candidates


def _phase_expected_prefix(step_idx: int, phase: str) -> int:
    if phase == "after_checkpoint_ready":
        return int(step_idx)
    return max(0, int(step_idx) - 1)


def _remaining_start_step(step_idx: int, phase: str) -> int:
    if phase == "after_checkpoint_ready":
        return int(step_idx) + 1
    return int(step_idx)


def _build_systematic_trial_specs(
    *,
    phases: list[str],
    steps: list[ValidationStep],
) -> list[dict[str, Any]]:
    return [
        {
            "phase": phase,
            "inject_step": step.step_idx,
            "trial_mode": "systematic",
            "trial_id": f"{phase}__step_{step.step_idx:02d}",
            "mid_action_fault_delay_sec": None,
        }
        for phase in phases
        for step in steps
    ]


def _build_random_trial_specs(
    *,
    phases: list[str],
    steps: list[ValidationStep],
    count: int,
    seed: int,
    min_delay_sec: float,
    max_delay_sec: float,
    checkpoint_interrupt_probability: float = 0.0,
    checkpoint_interrupt_phases: list[str] | None = None,
    checkpoint_interrupt_delay_sec: float = 0.0,
) -> list[dict[str, Any]]:
    del phases
    rng = random.Random(int(seed))
    specs: list[dict[str, Any]] = []
    lo = max(0.0, float(min_delay_sec))
    hi = max(lo, float(max_delay_sec))
    checkpoint_probability = min(1.0, max(0.0, float(checkpoint_interrupt_probability)))
    checkpoint_phases = [
        str(phase)
        for phase in (checkpoint_interrupt_phases or RANDOM_CHECKPOINT_FAULT_PHASES)
        if str(phase) in RANDOM_CHECKPOINT_FAULT_PHASES
    ]
    if not checkpoint_phases:
        checkpoint_phases = list(RANDOM_CHECKPOINT_FAULT_PHASES)
    for idx in range(max(0, int(count))):
        delay_sec = rng.uniform(lo, hi)
        use_checkpoint_internal = bool(steps) and rng.random() < checkpoint_probability
        checkpoint_step = rng.choice(steps) if use_checkpoint_internal else None
        checkpoint_fault_phase = rng.choice(checkpoint_phases) if use_checkpoint_internal else None
        specs.append(
            {
                "phase": "random_wall_clock",
                "inject_step": checkpoint_step.step_idx if checkpoint_step else None,
                "trial_mode": "random",
                "trial_id": (
                    f"random_{idx:04d}__checkpoint_{checkpoint_fault_phase}__step_{checkpoint_step.step_idx:02d}"
                    if checkpoint_step is not None and checkpoint_fault_phase is not None
                    else f"random_{idx:04d}__wall_clock"
                ),
                "random_trial_index": idx,
                "random_seed": int(seed),
                "random_fault_delay_sec": (
                    max(0.0, float(checkpoint_interrupt_delay_sec))
                    if use_checkpoint_internal
                    else delay_sec
                ),
                "random_fault_strategy": "checkpoint_internal" if use_checkpoint_internal else "wall_clock",
                "checkpoint_fault_phase": checkpoint_fault_phase,
                "checkpoint_fault_delay_sec": max(0.0, float(checkpoint_interrupt_delay_sec)) if use_checkpoint_internal else None,
            }
        )
    return specs


def _synthetic_steps(root: str, sleep_sec: float) -> list[ValidationStep]:
    root_q = shlex.quote(root)
    sleep_s = f"{max(0.0, float(sleep_sec)):.3f}"
    steps = [
        ValidationStep(
            step_idx=1,
            name="append_and_counter",
            command=dedent(
                f"""
                set -euo pipefail
                ROOT={root_q}
                MARKERS="$ROOT/markers.log"
                HISTORY="$ROOT/history.log"
                COUNTER="$ROOT/counter.txt"
                printf 'BEGIN_STEP_1\\n' >> "$MARKERS"
                count="$(cat "$COUNTER")"
                printf '%s\\n' "$((count + 1))" > "$COUNTER"
                printf 'STEP_1_DONE\\n' >> "$HISTORY"
                sleep {sleep_s}
                printf 'END_STEP_1\\n' >> "$MARKERS"
                """
            ).strip(),
        ),
        ValidationStep(
            step_idx=2,
            name="rename_temp_config",
            command=dedent(
                f"""
                set -euo pipefail
                ROOT={root_q}
                MARKERS="$ROOT/markers.log"
                HISTORY="$ROOT/history.log"
                COUNTER="$ROOT/counter.txt"
                printf 'BEGIN_STEP_2\\n' >> "$MARKERS"
                printf '{{"version":2,"step":2}}\\n' > "$ROOT/config.tmp"
                sleep {sleep_s}
                mv "$ROOT/config.tmp" "$ROOT/config.json"
                count="$(cat "$COUNTER")"
                printf '%s\\n' "$((count + 1))" > "$COUNTER"
                printf 'STEP_2_DONE\\n' >> "$HISTORY"
                printf 'END_STEP_2\\n' >> "$MARKERS"
                """
            ).strip(),
        ),
        ValidationStep(
            step_idx=3,
            name="delete_important_file",
            command=dedent(
                f"""
                set -euo pipefail
                ROOT={root_q}
                MARKERS="$ROOT/markers.log"
                HISTORY="$ROOT/history.log"
                COUNTER="$ROOT/counter.txt"
                printf 'BEGIN_STEP_3\\n' >> "$MARKERS"
                rm -f "$ROOT/important_file.txt"
                sleep {sleep_s}
                count="$(cat "$COUNTER")"
                printf '%s\\n' "$((count + 1))" > "$COUNTER"
                printf 'STEP_3_DONE\\n' >> "$HISTORY"
                printf 'END_STEP_3\\n' >> "$MARKERS"
                """
            ).strip(),
        ),
        ValidationStep(
            step_idx=4,
            name="move_nested_payload",
            command=dedent(
                f"""
                set -euo pipefail
                ROOT={root_q}
                MARKERS="$ROOT/markers.log"
                HISTORY="$ROOT/history.log"
                COUNTER="$ROOT/counter.txt"
                printf 'BEGIN_STEP_4\\n' >> "$MARKERS"
                mkdir -p "$ROOT/nested"
                printf 'payload-step-4\\n' > "$ROOT/nested/data.txt"
                sleep {sleep_s}
                mv "$ROOT/nested/data.txt" "$ROOT/finalized.txt"
                rmdir "$ROOT/nested"
                count="$(cat "$COUNTER")"
                printf '%s\\n' "$((count + 1))" > "$COUNTER"
                printf 'STEP_4_DONE\\n' >> "$HISTORY"
                printf 'END_STEP_4\\n' >> "$MARKERS"
                """
            ).strip(),
        ),
    ]
    return steps


def _initialization_command(root: str) -> str:
    root_q = shlex.quote(root)
    return dedent(
        f"""
        set -euo pipefail
        ROOT={root_q}
        rm -rf "$ROOT"
        mkdir -p "$ROOT"
        printf '0\\n' > "$ROOT/counter.txt"
        : > "$ROOT/history.log"
        : > "$ROOT/markers.log"
        printf 'important payload\\n' > "$ROOT/important_file.txt"
        printf '{{"version":1,"status":"base"}}\\n' > "$ROOT/config.json"
        rm -f "$ROOT/config.tmp" "$ROOT/finalized.txt"
        rm -rf "$ROOT/nested"
        """
    ).strip()


def _state_report_command(root: str) -> str:
    script = dedent(
        f"""
        python3 - <<'PY'
        import hashlib
        import json
        import os
        import stat

        root = {root!r}

        def hash_file(path):
            h = hashlib.sha256()
            with open(path, "rb") as handle:
                while True:
                    block = handle.read(1024 * 1024)
                    if not block:
                        break
                    h.update(block)
            return h.hexdigest()

        def read_lines(path):
            if not os.path.exists(path):
                return []
            with open(path, "r", encoding="utf-8") as handle:
                return [line.rstrip("\\n") for line in handle]

        entries = []
        if os.path.exists(root):
            for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
                dirnames.sort()
                filenames.sort()
                rel_dir = os.path.relpath(dirpath, root)
                rel_dir = "." if rel_dir == "." else rel_dir
                st = os.lstat(dirpath)
                entries.append(("DIR", rel_dir, oct(stat.S_IMODE(st.st_mode))))
                for name in filenames:
                    path = os.path.join(dirpath, name)
                    rel = os.path.relpath(path, root)
                    st = os.lstat(path)
                    mode = oct(stat.S_IMODE(st.st_mode))
                    if os.path.islink(path):
                        entries.append(("LINK", rel, mode, os.readlink(path)))
                    elif os.path.isfile(path):
                        entries.append(("FILE", rel, mode, hash_file(path)))
                    else:
                        entries.append(("OTHER", rel, mode))

        digest = hashlib.sha256()
        for entry in entries:
            digest.update(repr(entry).encode("utf-8"))
            digest.update(b"\\n")

        counter_path = os.path.join(root, "counter.txt")
        config_json_path = os.path.join(root, "config.json")
        finalized_path = os.path.join(root, "finalized.txt")
        report = {{
            "root": root,
            "fs_hash": digest.hexdigest(),
            "entries": len(entries),
            "files": [entry[1] for entry in entries if entry[0] == "FILE"],
            "counter": int(open(counter_path, "r", encoding="utf-8").read().strip()) if os.path.exists(counter_path) else None,
            "history_lines": read_lines(os.path.join(root, "history.log")),
            "markers_lines": read_lines(os.path.join(root, "markers.log")),
            "important_exists": os.path.exists(os.path.join(root, "important_file.txt")),
            "config_tmp_exists": os.path.exists(os.path.join(root, "config.tmp")),
            "nested_exists": os.path.exists(os.path.join(root, "nested")),
            "finalized_exists": os.path.exists(finalized_path),
            "config_json_sha": hash_file(config_json_path) if os.path.exists(config_json_path) else None,
            "finalized_sha": hash_file(finalized_path) if os.path.exists(finalized_path) else None,
        }}
        print(json.dumps(report, sort_keys=True))
        PY
        """
    ).strip()
    return script


def _oracle_prefix_for_hash(oracle_states: dict[int, dict[str, Any]], fs_hash: str | None) -> int | None:
    for prefix, state in oracle_states.items():
        if state.get("fs_hash") == fs_hash:
            return int(prefix)
    return None


def _detect_semantic_anomalies(
    expected_state: dict[str, Any],
    actual_state: dict[str, Any],
    *,
    expected_prefix: int,
) -> dict[str, Any]:
    expected_history = Counter(expected_state.get("history_lines", []) or [])
    actual_history = Counter(actual_state.get("history_lines", []) or [])
    duplicate_tokens = sorted(
        token
        for token, count in actual_history.items()
        if count > expected_history.get(token, 0)
    )
    lost_tokens = sorted(
        token
        for token, count in expected_history.items()
        if actual_history.get(token, 0) < count
    )
    marker_lines = actual_state.get("markers_lines", []) or []
    marker_set = set(marker_lines)
    partial_markers: list[str] = []
    for line in marker_lines:
        match = re.fullmatch(r"BEGIN_STEP_(\d+)", str(line))
        if not match:
            continue
        step_idx = int(match.group(1))
        if step_idx > expected_prefix and f"END_STEP_{step_idx}" not in marker_set:
            partial_markers.append(line)
    lost_state_fields: list[str] = []
    for key in ("counter", "important_exists", "config_tmp_exists", "nested_exists", "finalized_exists", "config_json_sha"):
        if expected_state.get(key) != actual_state.get(key):
            lost_state_fields.append(key)
    partial_paths: list[str] = []
    if bool(actual_state.get("config_tmp_exists")) and not bool(expected_state.get("config_tmp_exists")):
        partial_paths.append("config.tmp")
    if bool(actual_state.get("nested_exists")) and not bool(expected_state.get("nested_exists")):
        partial_paths.append("nested/")
    return {
        "duplicate_effect": bool(duplicate_tokens),
        "lost_effect": bool(lost_tokens or lost_state_fields),
        "partial_effect": bool(partial_markers or partial_paths),
        "duplicate_tokens": duplicate_tokens,
        "lost_tokens": lost_tokens,
        "lost_state_fields": lost_state_fields,
        "partial_markers": partial_markers,
        "partial_paths": partial_paths,
    }


def _latest_ready_checkpoint(list_payload: dict[str, Any]) -> dict[str, Any] | None:
    checkpoints = list_payload.get("checkpoints", []) or []
    ready = [item for item in checkpoints if isinstance(item, dict) and item.get("status") == "ready"]
    if not ready:
        return None
    ready.sort(key=lambda item: (int(item.get("step_idx", -1) or -1), float(item.get("created_at", 0.0) or 0.0)))
    return dict(ready[-1])


async def _safe_close(env_client: ValidationEnvClient, lease_id: str | None) -> None:
    if not lease_id:
        return
    try:
        await env_client.close(lease_id)
    except Exception:
        pass


def _expect_ok(result: dict[str, Any], op_name: str) -> dict[str, Any]:
    if not result.get("ok", False):
        raise RuntimeError(f"{op_name} failed: {result}")
    return result


def _expect_exec_success(result: dict[str, Any], op_name: str) -> dict[str, Any]:
    _expect_ok(result, op_name)
    if int(result.get("returncode", -1)) != 0:
        raise RuntimeError(f"{op_name} returned non-zero exit code: {result}")
    return result


def _is_mid_action_fault_observed(result: dict[str, Any]) -> bool:
    if bool(result.get("fault_injected", False)):
        return True
    try:
        returncode = int(result.get("returncode", -999))
    except (TypeError, ValueError):
        returncode = -999
    output = str(result.get("output", "") or "").lower()
    if returncode in {-1, -9, 137, 143}:
        return True
    return "killed" in output or "container killed" in output or "no such container" in output


def _expect_checkpoint_ready(result: dict[str, Any], op_name: str) -> dict[str, Any]:
    _expect_ok(result, op_name)
    if str(result.get("status", "")) != "ready":
        raise RuntimeError(f"{op_name} did not return a ready checkpoint: {result}")
    return result


async def _collect_state_report(
    env_client: ValidationEnvClient,
    lease_id: str,
    *,
    cwd: str,
    root: str,
    timeout: int,
) -> dict[str, Any]:
    out = await env_client.exec(lease_id, _state_report_command(root), cwd=cwd, timeout=timeout)
    out = _expect_exec_success(out, "state_report")
    output = str(out.get("output", "") or "").strip()
    if not output:
        raise RuntimeError("state_report returned empty output")
    return json.loads(output)


async def _initialize_trial_workspace(
    env_client: ValidationEnvClient,
    lease_id: str,
    *,
    cwd: str,
    root: str,
    timeout: int,
) -> None:
    result = await env_client.exec(lease_id, _initialization_command(root), cwd=cwd, timeout=timeout)
    _expect_exec_success(result, "initialize_workspace")


async def _checkpoint_now(
    env_client: ValidationEnvClient,
    lease_id: str,
    *,
    step_idx: int,
    cwd: str,
    reason: str,
    parent_checkpoint_id: str | None,
    timeout: int,
    fault_injection_spec: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del timeout
    result = await env_client.checkpoint_create(
        lease_id,
        step_idx=step_idx,
        command_seq=step_idx,
        cwd=cwd,
        policy="correctness-validation",
        reason=reason,
        parent_checkpoint_id=parent_checkpoint_id,
        fault_injection_spec=fault_injection_spec,
    )
    return result


async def _run_oracle(
    env_client: ValidationEnvClient,
    *,
    image_name: str,
    instance_id: str,
    cwd: str,
    root: str,
    steps: list[ValidationStep],
    exec_timeout: int,
    rerun_timeout: int,
) -> dict[str, Any]:
    del rerun_timeout
    allocation = _expect_ok(await env_client.allocate(image_name, instance_id=instance_id, cwd=cwd), "allocate")
    lease_id = str(allocation["lease_id"])
    try:
        await _initialize_trial_workspace(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
        parent_checkpoint_id: str | None = None
        base_checkpoint = _expect_checkpoint_ready(
            await _checkpoint_now(
                env_client,
                lease_id,
                step_idx=0,
                cwd=cwd,
                reason="oracle_init",
                parent_checkpoint_id=None,
                timeout=exec_timeout,
            ),
            "oracle_checkpoint_step_0",
        )
        parent_checkpoint_id = str(base_checkpoint["checkpoint_id"])
        oracle_states: dict[int, dict[str, Any]] = {
            0: {
                **(await _collect_state_report(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)),
                "prefix": 0,
                "checkpoint_id": parent_checkpoint_id,
            }
        }
        checkpoints: list[dict[str, Any]] = [dict(base_checkpoint)]
        for step in steps:
            exec_out = await env_client.exec(lease_id, step.command, cwd=cwd, timeout=exec_timeout)
            _expect_exec_success(exec_out, f"oracle_exec_step_{step.step_idx}")
            checkpoint_out = _expect_checkpoint_ready(
                await _checkpoint_now(
                    env_client,
                    lease_id,
                    step_idx=step.step_idx,
                    cwd=cwd,
                    reason="oracle_step",
                    parent_checkpoint_id=parent_checkpoint_id,
                    timeout=exec_timeout,
                ),
                f"oracle_checkpoint_step_{step.step_idx}",
            )
            parent_checkpoint_id = str(checkpoint_out["checkpoint_id"])
            checkpoints.append(dict(checkpoint_out))
            oracle_states[step.step_idx] = {
                **(await _collect_state_report(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)),
                "prefix": step.step_idx,
                "step_name": step.name,
                "checkpoint_id": parent_checkpoint_id,
            }
        return {
            "image_name": image_name,
            "instance_id": instance_id,
            "cwd": cwd,
            "validation_root": root,
            "step_count": len(steps),
            "steps": [asdict(step) for step in steps],
            "states": [oracle_states[idx] for idx in sorted(oracle_states)],
            "checkpoints": checkpoints,
        }
    finally:
        await _safe_close(env_client, lease_id)


async def _seed_ready_prefix(
    env_client: ValidationEnvClient,
    lease_id: str,
    *,
    cwd: str,
    root: str,
    steps: list[ValidationStep],
    target_prefix: int,
    exec_timeout: int,
) -> str:
    await _initialize_trial_workspace(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
    base_checkpoint = _expect_checkpoint_ready(
        await _checkpoint_now(
            env_client,
            lease_id,
            step_idx=0,
            cwd=cwd,
            reason="trial_init",
            parent_checkpoint_id=None,
            timeout=exec_timeout,
        ),
        "trial_checkpoint_step_0",
    )
    parent_checkpoint_id = str(base_checkpoint["checkpoint_id"])
    for step in steps:
        if step.step_idx > target_prefix:
            break
        exec_out = await env_client.exec(lease_id, step.command, cwd=cwd, timeout=exec_timeout)
        _expect_exec_success(exec_out, f"seed_exec_step_{step.step_idx}")
        checkpoint_out = _expect_checkpoint_ready(
            await _checkpoint_now(
                env_client,
                lease_id,
                step_idx=step.step_idx,
                cwd=cwd,
                reason="trial_seed",
                parent_checkpoint_id=parent_checkpoint_id,
                timeout=exec_timeout,
            ),
            f"seed_checkpoint_step_{step.step_idx}",
        )
        parent_checkpoint_id = str(checkpoint_out["checkpoint_id"])
    return parent_checkpoint_id


async def _rerun_from_latest_ready(
    env_client: ValidationEnvClient,
    lease_id: str,
    *,
    cwd: str,
    rerun_timeout: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    checkpoint_list = _expect_ok(await env_client.checkpoint_list(lease_id), "checkpoint_list")
    latest_ready = _latest_ready_checkpoint(checkpoint_list)
    if latest_ready is None:
        raise RuntimeError(f"no ready checkpoint available for lease {lease_id}")
    rerun_result = _expect_ok(
        await env_client.rerun(
            lease_id,
            checkpoint_id=str(latest_ready["checkpoint_id"]),
            cwd=cwd,
            timeout=rerun_timeout,
        ),
        "rerun",
    )
    return latest_ready, rerun_result


async def _run_remaining_steps(
    env_client: ValidationEnvClient,
    lease_id: str,
    *,
    steps: list[ValidationStep],
    start_step_idx: int,
    cwd: str,
    exec_timeout: int,
) -> str | None:
    checkpoint_list = _expect_ok(await env_client.checkpoint_list(lease_id), "checkpoint_list_before_remaining")
    latest_ready = _latest_ready_checkpoint(checkpoint_list)
    parent_checkpoint_id = str(latest_ready["checkpoint_id"]) if latest_ready else None
    for step in steps:
        if step.step_idx < start_step_idx:
            continue
        exec_out = await env_client.exec(lease_id, step.command, cwd=cwd, timeout=exec_timeout)
        _expect_exec_success(exec_out, f"remaining_exec_step_{step.step_idx}")
        checkpoint_out = _expect_checkpoint_ready(
            await _checkpoint_now(
                env_client,
                lease_id,
                step_idx=step.step_idx,
                cwd=cwd,
                reason="trial_remaining",
                parent_checkpoint_id=parent_checkpoint_id,
                timeout=exec_timeout,
            ),
            f"remaining_checkpoint_step_{step.step_idx}",
        )
        parent_checkpoint_id = str(checkpoint_out["checkpoint_id"])
    return parent_checkpoint_id


async def _run_trial(
    env_client: ValidationEnvClient,
    *,
    image_name: str,
    instance_id: str,
    cwd: str,
    root: str,
    steps: list[ValidationStep],
    oracle_states: dict[int, dict[str, Any]],
    phase: str,
    inject_step: int,
    trial_id: str | None = None,
    trial_mode: str = "systematic",
    random_trial_index: int | None = None,
    random_seed: int | None = None,
    exec_timeout: int,
    rerun_timeout: int,
    mid_action_fault_delay_sec: float,
) -> dict[str, Any]:
    allocation = _expect_ok(await env_client.allocate(image_name, instance_id=instance_id, cwd=cwd), "allocate")
    lease_id = str(allocation["lease_id"])
    step = next(item for item in steps if item.step_idx == inject_step)
    expected_prefix = _phase_expected_prefix(inject_step, phase)
    remaining_start = _remaining_start_step(inject_step, phase)
    resolved_trial_id = trial_id or f"{phase}__step_{inject_step:02d}"
    checkpoint_out: dict[str, Any] | None = None
    exec_out: dict[str, Any] | None = None
    latest_before_recovery: dict[str, Any] | None = None
    try:
        seeded_checkpoint_id = await _seed_ready_prefix(
            env_client,
            lease_id,
            cwd=cwd,
            root=root,
            steps=steps,
            target_prefix=inject_step - 1,
            exec_timeout=exec_timeout,
        )
        if phase == "before_action":
            exec_out = _expect_ok(
                await env_client.exec(
                    lease_id,
                    step.command,
                    cwd=cwd,
                    timeout=exec_timeout,
                    fault_injection_spec={"phase": "before_action"},
                ),
                f"{resolved_trial_id}_exec",
            )
            if not bool(exec_out.get("fault_injected")):
                raise RuntimeError(f"{resolved_trial_id} expected exec fault injection: {exec_out}")
        elif phase == "mid_action":
            exec_out = _expect_ok(
                await env_client.exec(
                    lease_id,
                    step.command,
                    cwd=cwd,
                    timeout=exec_timeout,
                    fault_injection_spec={"phase": "mid_action", "delay_sec": float(mid_action_fault_delay_sec)},
                ),
                f"{resolved_trial_id}_exec",
            )
            if not _is_mid_action_fault_observed(exec_out):
                raise RuntimeError(f"{resolved_trial_id} expected mid-action fault injection: {exec_out}")
            if not bool(exec_out.get("fault_injected")):
                exec_out = {
                    **exec_out,
                    "fault_injected": True,
                    "fault_ack_inferred": True,
                    "fault_phase": "mid_action",
                    "fault_type": "exec_server_mid_action_kill_inferred",
                }
        elif phase in {"after_action_before_observation", "after_observation_before_checkpoint"}:
            exec_out = _expect_exec_success(
                await env_client.exec(lease_id, step.command, cwd=cwd, timeout=exec_timeout),
                f"{resolved_trial_id}_exec",
            )
        elif phase in {"before_commit", "after_commit_before_ready"}:
            exec_out = _expect_exec_success(
                await env_client.exec(lease_id, step.command, cwd=cwd, timeout=exec_timeout),
                f"{resolved_trial_id}_exec",
            )
            checkpoint_out = await _checkpoint_now(
                env_client,
                lease_id,
                step_idx=step.step_idx,
                cwd=cwd,
                reason=f"trial_fault_{phase}",
                parent_checkpoint_id=seeded_checkpoint_id,
                timeout=exec_timeout,
                fault_injection_spec={"phase": phase},
            )
            if checkpoint_out.get("ok", False) or not bool(checkpoint_out.get("fault_injected")):
                raise RuntimeError(f"{resolved_trial_id} expected checkpoint fault injection: {checkpoint_out}")
        elif phase == "after_checkpoint_ready":
            exec_out = _expect_exec_success(
                await env_client.exec(lease_id, step.command, cwd=cwd, timeout=exec_timeout),
                f"{resolved_trial_id}_exec",
            )
            checkpoint_out = _expect_checkpoint_ready(
                await _checkpoint_now(
                    env_client,
                    lease_id,
                    step_idx=step.step_idx,
                    cwd=cwd,
                    reason="trial_ready_recovery",
                    parent_checkpoint_id=seeded_checkpoint_id,
                    timeout=exec_timeout,
                ),
                f"{resolved_trial_id}_checkpoint",
            )
        else:
            raise ValueError(f"unsupported phase: {phase}")

        checkpoint_list_before_recovery = _expect_ok(
            await env_client.checkpoint_list(lease_id),
            f"{resolved_trial_id}_checkpoint_list",
        )
        latest_before_recovery = _latest_ready_checkpoint(checkpoint_list_before_recovery)
        recovered_checkpoint, rerun_out = await _rerun_from_latest_ready(
            env_client,
            lease_id,
            cwd=cwd,
            rerun_timeout=rerun_timeout,
        )
        recovered_state = await _collect_state_report(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
        matched_oracle_prefix = _oracle_prefix_for_hash(oracle_states, recovered_state.get("fs_hash"))
        expected_state = oracle_states[expected_prefix]
        semantic_checks = _detect_semantic_anomalies(
            expected_state,
            recovered_state,
            expected_prefix=expected_prefix,
        )
        await _run_remaining_steps(
            env_client,
            lease_id,
            steps=steps,
            start_step_idx=remaining_start,
            cwd=cwd,
            exec_timeout=exec_timeout,
        )
        final_state = await _collect_state_report(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
        final_expected_state = oracle_states[max(oracle_states)]
        return {
            "trial_id": resolved_trial_id,
            "trial_mode": trial_mode,
            "random_trial_index": random_trial_index,
            "random_seed": random_seed,
            "phase": phase,
            "inject_step": inject_step,
            "mid_action_fault_delay_sec": float(mid_action_fault_delay_sec) if phase == "mid_action" else None,
            "expected_prefix": expected_prefix,
            "matched_oracle_prefix": matched_oracle_prefix,
            "hash_match": recovered_state.get("fs_hash") == expected_state.get("fs_hash"),
            "final_hash_match": final_state.get("fs_hash") == final_expected_state.get("fs_hash"),
            "ok": True,
            "seeded_checkpoint_id": seeded_checkpoint_id,
            "latest_ready_before_recovery": latest_before_recovery,
            "recovered_checkpoint": recovered_checkpoint,
            "rerun_result": rerun_out,
            "exec_result": exec_out,
            "checkpoint_result": checkpoint_out,
            "expected_state": expected_state,
            "recovered_state": recovered_state,
            "final_state": final_state,
            "semantic_checks": semantic_checks,
        }
    finally:
        await _safe_close(env_client, lease_id)


def _exec_result_indicates_fail_stop(result: dict[str, Any]) -> bool:
    try:
        returncode = int(result.get("returncode", -999))
    except (TypeError, ValueError):
        returncode = -999
    output = str(result.get("output", "") or "").lower()
    return (
        returncode in {-1, -9, 125, 126, 137, 143}
        or "no such container" in output
        or "is not running" in output
        or "is paused" in output
        or "killed" in output
        or ("container" in output and "paused" not in output and "exec" in output)
    )


def _checkpoint_result_indicates_random_fault(result: dict[str, Any]) -> bool:
    error_text = " ".join(
        str(result.get(key, "") or "").lower()
        for key in ("error", "output", "stderr", "error_code")
    )
    return (
        bool(result.get("fault_injected", False))
        or "no such container" in error_text
        or "is not running" in error_text
        or "is paused" in error_text
        or "killed" in error_text
        or "failed to probe runtime env" in error_text
        or "failed to write runtime state" in error_text
        or "checkpoint_create_failed" in error_text
    )


async def _await_fault_task(task: asyncio.Task[dict[str, Any]], *, timeout_sec: float) -> dict[str, Any] | None:
    if task.done():
        return task.result()
    try:
        return await asyncio.wait_for(asyncio.shield(task), timeout=max(0.0, float(timeout_sec)))
    except asyncio.TimeoutError:
        return None


async def _run_wall_clock_random_trial(
    env_client: ValidationEnvClient,
    *,
    image_name: str,
    instance_id: str,
    cwd: str,
    root: str,
    steps: list[ValidationStep],
    oracle_states: dict[int, dict[str, Any]],
    trial_id: str,
    random_trial_index: int,
    random_seed: int,
    random_fault_delay_sec: float,
    exec_timeout: int,
    rerun_timeout: int,
) -> dict[str, Any]:
    allocation = _expect_ok(await env_client.allocate(image_name, instance_id=instance_id, cwd=cwd), "allocate")
    lease_id = str(allocation["lease_id"])
    latest_ready_checkpoint_id: str | None = None
    latest_ready_checkpoint_step = 0
    parent_checkpoint_id: str | None = None
    fault_task: asyncio.Task[dict[str, Any]] | None = None
    fault_event: dict[str, Any] | None = None
    interrupted_op: dict[str, Any] | None = None
    try:
        await _initialize_trial_workspace(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
        base_checkpoint = _expect_checkpoint_ready(
            await _checkpoint_now(
                env_client,
                lease_id,
                step_idx=0,
                cwd=cwd,
                reason="random_trial_init",
                parent_checkpoint_id=None,
                timeout=exec_timeout,
            ),
            f"{trial_id}_checkpoint_step_0",
        )
        latest_ready_checkpoint_id = str(base_checkpoint["checkpoint_id"])
        parent_checkpoint_id = latest_ready_checkpoint_id
        latest_ready_checkpoint_step = 0

        async def _delayed_kill() -> dict[str, Any]:
            await asyncio.sleep(max(0.0, float(random_fault_delay_sec)))
            return _expect_ok(
                await env_client.inject_fail_stop(
                    lease_id,
                    tag=trial_id,
                    delay_sec=0.0,
                ),
                f"{trial_id}_random_fault",
            )

        fault_task = asyncio.create_task(_delayed_kill())
        fault_timer_started_perf = time.perf_counter()

        async def _await_due_random_fault(timeout_sec: float) -> dict[str, Any] | None:
            if fault_task is None:
                return None
            elapsed = time.perf_counter() - fault_timer_started_perf
            remaining = max(0.0, float(random_fault_delay_sec) - elapsed)
            if remaining > 0.0:
                return None
            return await _await_fault_task(fault_task, timeout_sec=timeout_sec)

        for step in steps:
            if fault_task.done():
                fault_event = fault_task.result()
                break

            exec_out: dict[str, Any] | None = None
            try:
                exec_out = await env_client.exec(lease_id, step.command, cwd=cwd, timeout=exec_timeout)
            except Exception as exc:
                fault_event = await _await_fault_task(fault_task, timeout_sec=2.0)
                if fault_event is None:
                    raise
                interrupted_op = {
                    "op": "exec",
                    "step_idx": step.step_idx,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                break
            if int(exec_out.get("returncode", -1)) != 0:
                fault_event = await _await_fault_task(fault_task, timeout_sec=2.0)
                if fault_event is None and _exec_result_indicates_fail_stop(exec_out):
                    fault_event = await _await_due_random_fault(timeout_sec=5.0)
                if fault_event is None and not _exec_result_indicates_fail_stop(exec_out):
                    _expect_exec_success(exec_out, f"{trial_id}_exec_step_{step.step_idx}")
                if fault_event is None:
                    raise RuntimeError(f"{trial_id}_exec_step_{step.step_idx} failed before random fault was due: {exec_out}")
                interrupted_op = {
                    "op": "exec",
                    "step_idx": step.step_idx,
                    "exec_result": exec_out,
                }
                break

            fault_event = await _await_fault_task(fault_task, timeout_sec=0.0)
            if fault_event is not None:
                interrupted_op = {"op": "between_exec_and_checkpoint", "step_idx": step.step_idx}
                break

            checkpoint_out: dict[str, Any] | None = None
            try:
                checkpoint_out = await _checkpoint_now(
                    env_client,
                    lease_id,
                    step_idx=step.step_idx,
                    cwd=cwd,
                    reason="random_trial_step",
                    parent_checkpoint_id=parent_checkpoint_id,
                    timeout=exec_timeout,
                )
            except Exception as exc:
                fault_event = await _await_fault_task(fault_task, timeout_sec=2.0)
                if fault_event is None:
                    raise
                interrupted_op = {
                    "op": "checkpoint",
                    "step_idx": step.step_idx,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                break
            if checkpoint_out.get("ok", False) and str(checkpoint_out.get("status", "")) == "ready":
                parent_checkpoint_id = str(checkpoint_out["checkpoint_id"])
                latest_ready_checkpoint_id = parent_checkpoint_id
                latest_ready_checkpoint_step = int(checkpoint_out.get("step_idx", step.step_idx))
            else:
                fault_event = await _await_fault_task(fault_task, timeout_sec=2.0)
                if fault_event is None and _checkpoint_result_indicates_random_fault(checkpoint_out):
                    fault_event = await _await_due_random_fault(timeout_sec=5.0)
                if fault_event is None:
                    raise RuntimeError(f"{trial_id} checkpoint failed without injected fault: {checkpoint_out}")
                interrupted_op = {
                    "op": "checkpoint",
                    "step_idx": step.step_idx,
                    "checkpoint_result": checkpoint_out,
                }
                break

            fault_event = await _await_fault_task(fault_task, timeout_sec=0.0)
            if fault_event is not None:
                interrupted_op = {"op": "after_checkpoint_ready", "step_idx": step.step_idx}
                break

        if fault_event is None:
            if fault_task is None:
                raise RuntimeError("random fault task was not started")
            fault_event = await fault_task
            interrupted_op = {
                "op": "after_all_steps",
                "step_idx": max(step.step_idx for step in steps),
            }

        if latest_ready_checkpoint_id is None:
            raise RuntimeError(f"{trial_id} has no ready checkpoint after random fault")

        recovered_checkpoint, rerun_out = await _rerun_from_latest_ready(
            env_client,
            lease_id,
            cwd=cwd,
            rerun_timeout=rerun_timeout,
        )
        recovered_state = await _collect_state_report(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
        expected_prefix = latest_ready_checkpoint_step
        expected_state = oracle_states[expected_prefix]
        matched_oracle_prefix = _oracle_prefix_for_hash(oracle_states, recovered_state.get("fs_hash"))
        semantic_checks = _detect_semantic_anomalies(
            expected_state,
            recovered_state,
            expected_prefix=expected_prefix,
        )
        await _run_remaining_steps(
            env_client,
            lease_id,
            steps=steps,
            start_step_idx=expected_prefix + 1,
            cwd=cwd,
            exec_timeout=exec_timeout,
        )
        final_state = await _collect_state_report(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
        final_expected_state = oracle_states[max(oracle_states)]
        return {
            "trial_id": trial_id,
            "trial_mode": "random",
            "fault_model": "wall_clock_fail_stop",
            "random_trial_index": random_trial_index,
            "random_seed": random_seed,
            "random_fault_delay_sec": float(random_fault_delay_sec),
            "phase": "random_wall_clock",
            "inject_step": None if interrupted_op is None else interrupted_op.get("step_idx"),
            "interrupted_op": interrupted_op,
            "expected_prefix": expected_prefix,
            "matched_oracle_prefix": matched_oracle_prefix,
            "hash_match": recovered_state.get("fs_hash") == expected_state.get("fs_hash"),
            "final_hash_match": final_state.get("fs_hash") == final_expected_state.get("fs_hash"),
            "ok": True,
            "fault_event": fault_event,
            "latest_ready_checkpoint_id": latest_ready_checkpoint_id,
            "latest_ready_checkpoint_step": latest_ready_checkpoint_step,
            "recovered_checkpoint": recovered_checkpoint,
            "rerun_result": rerun_out,
            "expected_state": expected_state,
            "recovered_state": recovered_state,
            "final_state": final_state,
            "semantic_checks": semantic_checks,
        }
    finally:
        if fault_task is not None and not fault_task.done():
            fault_task.cancel()
        await _safe_close(env_client, lease_id)


async def _run_random_checkpoint_internal_trial(
    env_client: ValidationEnvClient,
    *,
    image_name: str,
    instance_id: str,
    cwd: str,
    root: str,
    steps: list[ValidationStep],
    oracle_states: dict[int, dict[str, Any]],
    trial_id: str,
    random_trial_index: int,
    random_seed: int,
    inject_step: int,
    checkpoint_fault_phase: str,
    checkpoint_fault_delay_sec: float,
    exec_timeout: int,
    rerun_timeout: int,
) -> dict[str, Any]:
    allocation = _expect_ok(await env_client.allocate(image_name, instance_id=instance_id, cwd=cwd), "allocate")
    lease_id = str(allocation["lease_id"])
    step = next(item for item in steps if item.step_idx == inject_step)
    expected_prefix = max(0, inject_step - 1)
    checkpoint_out: dict[str, Any] | None = None
    try:
        seeded_checkpoint_id = await _seed_ready_prefix(
            env_client,
            lease_id,
            cwd=cwd,
            root=root,
            steps=steps,
            target_prefix=expected_prefix,
            exec_timeout=exec_timeout,
        )
        exec_out = _expect_exec_success(
            await env_client.exec(lease_id, step.command, cwd=cwd, timeout=exec_timeout),
            f"{trial_id}_exec_step_{step.step_idx}",
        )
        checkpoint_out = await _checkpoint_now(
            env_client,
            lease_id,
            step_idx=step.step_idx,
            cwd=cwd,
            reason=f"checkpoint_interrupt_{checkpoint_fault_phase}",
            parent_checkpoint_id=seeded_checkpoint_id,
            timeout=exec_timeout,
            fault_injection_spec={
                "phase": checkpoint_fault_phase,
                "delay_sec": max(0.0, float(checkpoint_fault_delay_sec)),
                "tag": trial_id,
            },
        )
        if checkpoint_out.get("ok", False) or not bool(checkpoint_out.get("fault_injected", False)):
            raise RuntimeError(f"{trial_id} expected checkpoint interrupt fault: {checkpoint_out}")

        checkpoint_list_before_recovery = _expect_ok(
            await env_client.checkpoint_list(lease_id),
            f"{trial_id}_checkpoint_list",
        )
        latest_before_recovery = _latest_ready_checkpoint(checkpoint_list_before_recovery)
        recovered_checkpoint, rerun_out = await _rerun_from_latest_ready(
            env_client,
            lease_id,
            cwd=cwd,
            rerun_timeout=rerun_timeout,
        )
        recovered_state = await _collect_state_report(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
        expected_state = oracle_states[expected_prefix]
        matched_oracle_prefix = _oracle_prefix_for_hash(oracle_states, recovered_state.get("fs_hash"))
        semantic_checks = _detect_semantic_anomalies(
            expected_state,
            recovered_state,
            expected_prefix=expected_prefix,
        )
        await _run_remaining_steps(
            env_client,
            lease_id,
            steps=steps,
            start_step_idx=inject_step,
            cwd=cwd,
            exec_timeout=exec_timeout,
        )
        final_state = await _collect_state_report(env_client, lease_id, cwd=cwd, root=root, timeout=exec_timeout)
        final_expected_state = oracle_states[max(oracle_states)]
        return {
            "trial_id": trial_id,
            "trial_mode": "random",
            "fault_model": "wall_clock_fail_stop",
            "random_trial_index": random_trial_index,
            "random_seed": random_seed,
            "random_fault_strategy": "checkpoint_internal",
            "random_fault_delay_sec": max(0.0, float(checkpoint_fault_delay_sec)),
            "phase": "random_wall_clock",
            "checkpoint_fault_phase": checkpoint_fault_phase,
            "checkpoint_fault_delay_sec": max(0.0, float(checkpoint_fault_delay_sec)),
            "inject_step": inject_step,
            "interrupted_op": {
                "op": "checkpoint",
                "step_idx": inject_step,
                "checkpoint_fault_phase": checkpoint_fault_phase,
                "checkpoint_result": checkpoint_out,
            },
            "expected_prefix": expected_prefix,
            "matched_oracle_prefix": matched_oracle_prefix,
            "hash_match": recovered_state.get("fs_hash") == expected_state.get("fs_hash"),
            "final_hash_match": final_state.get("fs_hash") == final_expected_state.get("fs_hash"),
            "ok": True,
            "fault_event": checkpoint_out,
            "seeded_checkpoint_id": seeded_checkpoint_id,
            "latest_ready_before_recovery": latest_before_recovery,
            "recovered_checkpoint": recovered_checkpoint,
            "rerun_result": rerun_out,
            "exec_result": exec_out,
            "checkpoint_result": checkpoint_out,
            "expected_state": expected_state,
            "recovered_state": recovered_state,
            "final_state": final_state,
            "semantic_checks": semantic_checks,
        }
    finally:
        await _safe_close(env_client, lease_id)


def _build_summary(trials: list[dict[str, Any]]) -> dict[str, Any]:
    per_phase: dict[str, dict[str, Any]] = {}
    interrupted_ops: Counter[str] = Counter()
    for trial in trials:
        phase = str(trial.get("phase", "unknown"))
        interrupted_op = trial.get("interrupted_op")
        if isinstance(interrupted_op, dict):
            interrupted_ops[str(interrupted_op.get("op", "unknown"))] += 1
        bucket = per_phase.setdefault(
            phase,
            {
                "phase": phase,
                "trial_count": 0,
                "ok_count": 0,
                "hash_match_count": 0,
                "final_hash_match_count": 0,
                "duplicate_effect_count": 0,
                "lost_effect_count": 0,
                "partial_effect_count": 0,
                "error_count": 0,
            },
        )
        bucket["trial_count"] += 1
        if trial.get("ok", False):
            bucket["ok_count"] += 1
        if trial.get("hash_match", False):
            bucket["hash_match_count"] += 1
        if trial.get("final_hash_match", False):
            bucket["final_hash_match_count"] += 1
        semantics = trial.get("semantic_checks", {}) if isinstance(trial, dict) else {}
        if semantics.get("duplicate_effect", False):
            bucket["duplicate_effect_count"] += 1
        if semantics.get("lost_effect", False):
            bucket["lost_effect_count"] += 1
        if semantics.get("partial_effect", False):
            bucket["partial_effect_count"] += 1
        if trial.get("error"):
            bucket["error_count"] += 1
    for bucket in per_phase.values():
        total = max(1, int(bucket["trial_count"]))
        bucket["hash_match_rate"] = bucket["hash_match_count"] / total
        bucket["final_hash_match_rate"] = bucket["final_hash_match_count"] / total
    return {
        "trial_count": len(trials),
        "phase_count": len(per_phase),
        "systematic_trial_count": sum(1 for item in trials if item.get("trial_mode") == "systematic"),
        "random_trial_count": sum(1 for item in trials if item.get("trial_mode") == "random"),
        "phases": [per_phase[name] for name in sorted(per_phase)],
        "interrupted_ops": [
            {"op": op, "trial_count": count}
            for op, count in sorted(interrupted_ops.items())
        ],
    }


async def _main_async(args: argparse.Namespace) -> dict[str, Any]:
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    image_name, instance_id, resolved_traj, resolved_data_source = _resolve_image_name(
        image_name=args.image_name,
        instance_id=args.instance_id,
        trajectory=args.trajectory,
        trajectory_root=args.trajectory_root,
        data_source=args.data_source,
    )
    resolved_instance_id = instance_id or "correctness-validation"
    phases = list(args.phases or DEFAULT_PHASES)
    steps = _synthetic_steps(args.validation_root, args.action_sleep_sec)
    env_client = ValidationEnvClient(base_url=args.base_url)

    real_trace_candidates: list[dict[str, Any]] = []
    if args.trajectory_root and not args.skip_real_trace_scan:
        real_trace_candidates = _scan_non_idempotent_actions(
            args.trajectory_root,
            trajectory_limit=args.scan_trajectory_limit,
            candidate_limit=args.scan_candidate_limit,
        )
        _write_json(output_root / "real_trace_candidates.json", real_trace_candidates)

    oracle = await _run_oracle(
        env_client,
        image_name=image_name,
        instance_id=resolved_instance_id,
        cwd=args.cwd,
        root=args.validation_root,
        steps=steps,
        exec_timeout=args.exec_timeout,
        rerun_timeout=args.rerun_timeout,
    )
    _write_json(output_root / "oracle" / "oracle_states.json", oracle)
    oracle_states = {int(item["prefix"]): dict(item) for item in oracle["states"]}

    trial_specs: list[dict[str, Any]] = []
    if not args.random_only:
        trial_specs.extend(_build_systematic_trial_specs(phases=phases, steps=steps))
    trial_specs.extend(
        _build_random_trial_specs(
            phases=phases,
            steps=steps,
            count=args.random_trials,
            seed=args.random_seed,
            min_delay_sec=args.random_min_delay_sec,
            max_delay_sec=args.random_max_delay_sec,
            checkpoint_interrupt_probability=args.random_checkpoint_interrupt_probability,
            checkpoint_interrupt_phases=args.random_checkpoint_interrupt_phases,
            checkpoint_interrupt_delay_sec=args.random_checkpoint_interrupt_delay_sec,
        )
    )

    trials: list[dict[str, Any]] = []
    for spec in trial_specs:
        phase = str(spec["phase"])
        inject_step = int(spec["inject_step"]) if spec.get("inject_step") is not None else -1
        mid_action_delay = (
            float(spec["mid_action_fault_delay_sec"])
            if spec.get("mid_action_fault_delay_sec") is not None
            else float(args.mid_action_fault_delay_sec)
        )
        try:
            if phase == "random_wall_clock":
                if spec.get("random_fault_strategy") == "checkpoint_internal":
                    report = await _run_random_checkpoint_internal_trial(
                        env_client,
                        image_name=image_name,
                        instance_id=resolved_instance_id,
                        cwd=args.cwd,
                        root=args.validation_root,
                        steps=steps,
                        oracle_states=oracle_states,
                        trial_id=str(spec["trial_id"]),
                        random_trial_index=int(spec.get("random_trial_index", -1)),
                        random_seed=int(spec.get("random_seed", args.random_seed)),
                        inject_step=inject_step,
                        checkpoint_fault_phase=str(spec.get("checkpoint_fault_phase", RANDOM_CHECKPOINT_FAULT_PHASES[0])),
                        checkpoint_fault_delay_sec=float(spec.get("checkpoint_fault_delay_sec", 0.0) or 0.0),
                        exec_timeout=args.exec_timeout,
                        rerun_timeout=args.rerun_timeout,
                    )
                else:
                    report = await _run_wall_clock_random_trial(
                        env_client,
                        image_name=image_name,
                        instance_id=resolved_instance_id,
                        cwd=args.cwd,
                        root=args.validation_root,
                        steps=steps,
                        oracle_states=oracle_states,
                        trial_id=str(spec["trial_id"]),
                        random_trial_index=int(spec.get("random_trial_index", -1)),
                        random_seed=int(spec.get("random_seed", args.random_seed)),
                        random_fault_delay_sec=float(spec.get("random_fault_delay_sec", 0.0) or 0.0),
                        exec_timeout=args.exec_timeout,
                        rerun_timeout=args.rerun_timeout,
                    )
            else:
                report = await _run_trial(
                    env_client,
                    image_name=image_name,
                    instance_id=resolved_instance_id,
                    cwd=args.cwd,
                    root=args.validation_root,
                    steps=steps,
                    oracle_states=oracle_states,
                    phase=phase,
                    inject_step=inject_step,
                    trial_id=str(spec["trial_id"]),
                    trial_mode=str(spec["trial_mode"]),
                    random_trial_index=spec.get("random_trial_index"),
                    random_seed=spec.get("random_seed"),
                    exec_timeout=args.exec_timeout,
                    rerun_timeout=args.rerun_timeout,
                    mid_action_fault_delay_sec=mid_action_delay,
                )
        except Exception as exc:
            report = {
                "trial_id": str(spec["trial_id"]),
                "trial_mode": str(spec["trial_mode"]),
                "random_trial_index": spec.get("random_trial_index"),
                "random_seed": spec.get("random_seed"),
                "phase": phase,
                "inject_step": inject_step if spec.get("random_fault_strategy") == "checkpoint_internal" or phase != "random_wall_clock" else None,
                "random_fault_delay_sec": spec.get("random_fault_delay_sec"),
                "random_fault_strategy": spec.get("random_fault_strategy"),
                "checkpoint_fault_phase": spec.get("checkpoint_fault_phase"),
                "checkpoint_fault_delay_sec": spec.get("checkpoint_fault_delay_sec"),
                "mid_action_fault_delay_sec": mid_action_delay if phase == "mid_action" else None,
                "expected_prefix": (
                    max(0, inject_step - 1)
                    if spec.get("random_fault_strategy") == "checkpoint_internal"
                    else None
                    if phase == "random_wall_clock"
                    else _phase_expected_prefix(inject_step, phase)
                ),
                "matched_oracle_prefix": None,
                "hash_match": False,
                "final_hash_match": False,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "semantic_checks": {
                    "duplicate_effect": False,
                    "lost_effect": False,
                    "partial_effect": False,
                },
            }
        trials.append(report)
        _write_json(output_root / "trials" / f"{report['trial_id']}.json", report)

    summary = {
        "image_name": image_name,
        "instance_id": resolved_instance_id,
        "resolved_trajectory": resolved_traj,
        "resolved_data_source": resolved_data_source,
        "cwd": args.cwd,
        "validation_root": args.validation_root,
        "phases": phases,
        "random_trials_requested": int(args.random_trials),
        "random_seed": int(args.random_seed),
        "random_only": bool(args.random_only),
        "random_fault_model": "wall_clock_fail_stop",
        "random_checkpoint_interrupt_probability": float(args.random_checkpoint_interrupt_probability),
        "random_checkpoint_interrupt_phases": list(args.random_checkpoint_interrupt_phases),
        "random_checkpoint_interrupt_delay_sec": float(args.random_checkpoint_interrupt_delay_sec),
        "trial_specs": trial_specs,
        "synthetic_steps": [asdict(step) for step in steps],
        "real_trace_candidate_count": len(real_trace_candidates),
        "trial_summary": _build_summary(trials),
        "trial_ids": [trial["trial_id"] for trial in trials],
    }
    _write_json(output_root / "summary.json", summary)
    _write_trial_csv(output_root / "summary.csv", trials)
    return {
        "output_root": str(output_root),
        "oracle_path": str(output_root / "oracle" / "oracle_states.json"),
        "summary_path": str(output_root / "summary.json"),
        "summary_csv_path": str(output_root / "summary.csv"),
        "trial_count": len(trials),
        "real_trace_candidate_count": len(real_trace_candidates),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fault-injection correctness validation for SWE checkpoint recovery.")
    parser.add_argument("trajectory_root", nargs="?", default=str(DEFAULT_TRAJECTORY_ROOT))
    parser.add_argument("--trajectory", help="Optional specific traj.json for image inference.")
    parser.add_argument("--base-url", default=os.getenv("SWE_ENV_SERVER_URL", "http://127.0.0.1:18090"))
    parser.add_argument("--image-name", default=None)
    parser.add_argument("--instance-id", default=None)
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--cwd", default="/testbed")
    parser.add_argument("--validation-root", default="/testbed/.belayer_correctness")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--exec-timeout", type=int, default=180)
    parser.add_argument("--rerun-timeout", type=int, default=180)
    parser.add_argument("--action-sleep-sec", type=float, default=1.5)
    parser.add_argument("--mid-action-fault-delay-sec", type=float, default=0.2)
    parser.add_argument("--phases", nargs="+", default=list(DEFAULT_PHASES), choices=DEFAULT_PHASES)
    parser.add_argument("--random-trials", type=int, default=0)
    parser.add_argument("--random-seed", type=int, default=20260521)
    parser.add_argument("--random-only", action="store_true")
    parser.add_argument("--random-min-delay-sec", type=float, default=0.0)
    parser.add_argument("--random-max-delay-sec", type=float, default=8.0)
    parser.add_argument("--random-checkpoint-interrupt-probability", type=float, default=0.0)
    parser.add_argument(
        "--random-checkpoint-interrupt-phases",
        nargs="+",
        default=list(RANDOM_CHECKPOINT_FAULT_PHASES),
        choices=RANDOM_CHECKPOINT_FAULT_PHASES,
    )
    parser.add_argument("--random-checkpoint-interrupt-delay-sec", type=float, default=0.0)
    parser.add_argument("--scan-trajectory-limit", type=int, default=128)
    parser.add_argument("--scan-candidate-limit", type=int, default=64)
    parser.add_argument("--skip-real-trace-scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    result = asyncio.run(_main_async(args))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
