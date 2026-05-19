"""SWE Docker Exec Server — runs on each Volcengine ECS Docker node.

Wraps local `docker` CLI into HTTP endpoints so that the GPU training cluster
can create/exec/destroy SWE-Bench containers remotely.

Usage (on the ECS node):
    python3 swe_exec_server.py                       # default :5000
    python3 swe_exec_server.py --port 5000 --host 0.0.0.0

Endpoints:
    GET  /healthz           → liveness check
    GET  /images            → list locally available Docker images
    POST /container/create  → docker run -d ... sleep infinity
    POST /container/exec    → docker exec <cid> bash -lc <cmd>
    POST /container/diff    → git add -A && git diff --cached
    POST /container/destroy → docker rm -f <cid>
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import os
import queue
import random
import re
import shlex
import subprocess
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request

app = Flask(__name__)
logger = logging.getLogger("swe_exec_server")

_active_containers: dict[str, dict] = {}
_lock = threading.Lock()
_container_op_gates: dict[str, "_ContainerDockerGate"] = {}
_maintenance_cond = threading.Condition()
_maintenance_active = False
_foreground_docker_ops = 0
_commit_ops_cond = threading.Condition()
_active_commit_ops = 0
_checkpoint_cooldown_lock = threading.Lock()
_checkpoint_cooldown_until_ts = 0.0
_gc_tasks_lock = threading.Lock()
_gc_tasks_inflight: set[str] = set()
_cpu_stat_lock = threading.Lock()
_disk_stat_lock = threading.Lock()
_cpu_prev_total: int | None = None
_cpu_prev_idle: int | None = None
_cpu_prev_ts: float | None = None
_disk_prev_read_bytes: int | None = None
_disk_prev_write_bytes: int | None = None
_disk_prev_ts: float | None = None

_RUNTIME_STATE_SCHEMA_VERSION = 1
_RUNTIME_STATE_BASE_DIR = "/tmp/swe-runtime-checkpoints"
_RUNTIME_ENV_WHITELIST = ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX")


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


class _ContainerDockerGate:
    def __init__(self) -> None:
        self.cond = threading.Condition()
        self.active_execs = 0
        self.waiting_execs = 0
        self.exclusive_active = False


@dataclass
class _CheckpointCreateJob:
    op_id: str
    checkpoint_id: str
    container_id: str
    checkpoint_image: str
    record: dict[str, Any]
    runtime_env: dict[str, str]
    done_event: threading.Event
    enqueued_perf: float
    worker_started_perf: float | None = None
    worker_finished_perf: float | None = None
    result: dict[str, Any] | None = None
    error: Exception | None = None


def _get_container_op_gate(container_id: str) -> _ContainerDockerGate:
    with _lock:
        gate = _container_op_gates.get(container_id)
        if gate is None:
            gate = _ContainerDockerGate()
            _container_op_gates[container_id] = gate
        return gate


def _drop_container_op_gate(container_id: str) -> None:
    with _lock:
        _container_op_gates.pop(container_id, None)


def _container_is_active(container_id: str) -> bool:
    with _lock:
        return container_id in _active_containers


def _docker_container_is_running(container_id: str, timeout: int = 10) -> bool:
    inspect = _docker("inspect", "-f", "{{.State.Running}}", container_id, timeout=timeout)
    return inspect.returncode == 0 and inspect.stdout.strip().lower() == "true"


def _normalize_runtime_env(raw: dict[str, Any] | None) -> dict[str, str]:
    env: dict[str, str] = {}
    for key in _RUNTIME_ENV_WHITELIST:
        value = (raw or {}).get(key)
        if value is None:
            continue
        value_str = str(value)
        if value_str:
            env[key] = value_str
    return env


def _derive_python_runtime(runtime_env: dict[str, str]) -> dict[str, str]:
    virtual_env = runtime_env.get("VIRTUAL_ENV", "").strip()
    conda_prefix = runtime_env.get("CONDA_PREFIX", "").strip()
    python_executable = ""
    venv_activate = ""
    if virtual_env:
        python_executable = f"{virtual_env}/bin/python"
        venv_activate = f"{virtual_env}/bin/activate"
    elif conda_prefix:
        python_executable = f"{conda_prefix}/bin/python"
    return {
        "python_executable": python_executable,
        "venv_activate": venv_activate,
        "conda_env": conda_prefix,
    }


def _probe_runtime_env(container_id: str, cwd: str, runtime_env: dict[str, str]) -> dict[str, str]:
    command = (
        "python3 -c "
        "\"import json, os; "
        "keys=('PATH','PYTHONPATH','VIRTUAL_ENV','CONDA_PREFIX'); "
        "print(json.dumps({k: os.environ.get(k, '') for k in keys}))\""
    )
    env_args: list[str] = []
    for key, value in runtime_env.items():
        env_args.extend(["-e", f"{key}={value}"])
    result = _docker(
        "exec",
        "-w",
        cwd,
        *env_args,
        container_id,
        "bash",
        "-lc",
        command,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to probe runtime env for {container_id}")
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid runtime env payload for {container_id}")
    merged = _normalize_runtime_env(payload)
    merged.update(_normalize_runtime_env(runtime_env))
    return merged


def _runtime_checkpoint_dir(checkpoint_id: str) -> str:
    return f"{_RUNTIME_STATE_BASE_DIR}/{checkpoint_id}"


def _runtime_checkpoint_file(checkpoint_id: str) -> str:
    return f"{_runtime_checkpoint_dir(checkpoint_id)}/runtime.json"


def _build_runtime_state_payload(record: dict[str, Any], runtime_env: dict[str, str]) -> dict[str, Any]:
    raw_step_idx = record.get("step_idx", -1)
    raw_command_seq = record.get("command_seq", -1)
    return {
        "schema_version": _RUNTIME_STATE_SCHEMA_VERSION,
        "checkpoint_id": str(record["checkpoint_id"]),
        "parent_checkpoint_id": record.get("parent_checkpoint_id"),
        "lease_id": str(record.get("lease_id", "")),
        "instance_id": str(record.get("instance_id", "")),
        "step_idx": -1 if raw_step_idx is None else int(raw_step_idx),
        "command_seq": -1 if raw_command_seq is None else int(raw_command_seq),
        "workspace": {
            "repo_path": str(record.get("cwd", "/testbed")),
            "cwd": str(record.get("cwd", "/testbed")),
            "user": os.environ.get("USER", "root"),
            "home": os.environ.get("HOME", "/root"),
            "shell": "/bin/bash",
        },
        "env": dict(runtime_env),
        "python_runtime": _derive_python_runtime(runtime_env),
        "progress": {
            "phase": str(record.get("reason", "") or ""),
            "last_successful_action_id": (
                f"cmd-{int(record.get('command_seq', -1) or -1)}"
                if int(record.get("command_seq", -1) or -1) >= 0
                else ""
            ),
        },
    }


def _capture_runtime_state(container_id: str, checkpoint_id: str, payload: dict[str, Any]) -> None:
    runtime_dir = _runtime_checkpoint_dir(checkpoint_id)
    runtime_file = _runtime_checkpoint_file(checkpoint_id)
    command = f"mkdir -p {shlex.quote(runtime_dir)} && cat > {shlex.quote(runtime_file)}"
    result = _docker(
        "exec",
        "-i",
        container_id,
        "bash",
        "-lc",
        command,
        input_text=json.dumps(payload, ensure_ascii=False, indent=2),
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"failed to write runtime state for {checkpoint_id}")


def _load_runtime_state(container_id: str, checkpoint_id: str) -> dict[str, Any]:
    runtime_file = _runtime_checkpoint_file(checkpoint_id)
    result = _docker(
        "exec",
        container_id,
        "bash",
        "-lc",
        f"cat {shlex.quote(runtime_file)}",
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"missing runtime state for {checkpoint_id}")
    payload = json.loads(result.stdout or "{}")
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid runtime state payload for {checkpoint_id}")
    return payload


def _validate_runtime_restore(container_id: str, runtime_state: dict[str, Any]) -> None:
    workspace = runtime_state.get("workspace", {}) if isinstance(runtime_state, dict) else {}
    cwd = str(workspace.get("cwd", "") or "")
    repo_path = str(workspace.get("repo_path", "") or "")
    python_runtime = runtime_state.get("python_runtime", {}) if isinstance(runtime_state, dict) else {}
    python_executable = str(python_runtime.get("python_executable", "") or "")
    env_state = runtime_state.get("env", {}) if isinstance(runtime_state, dict) else {}
    virtual_env = str(env_state.get("VIRTUAL_ENV", "") or "")
    conda_prefix = str(env_state.get("CONDA_PREFIX", "") or "")

    checks: list[str] = []
    if cwd:
        checks.append(f"test -d {shlex.quote(cwd)}")
    if repo_path and repo_path != cwd:
        checks.append(f"test -d {shlex.quote(repo_path)}")
    if virtual_env:
        checks.append(f"test -d {shlex.quote(virtual_env)}")
    if conda_prefix:
        checks.append(f"test -d {shlex.quote(conda_prefix)}")
    if python_executable:
        checks.append(f"test -x {shlex.quote(python_executable)}")
        checks.append(f"{shlex.quote(python_executable)} -V >/dev/null")

    if not checks:
        return

    result = _docker(
        "exec",
        container_id,
        "bash",
        "-lc",
        " && ".join(checks),
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "runtime restore validation failed")


@contextlib.contextmanager
def _foreground_docker_section() -> Any:
    global _foreground_docker_ops
    with _maintenance_cond:
        while _maintenance_active:
            _maintenance_cond.wait()
        _foreground_docker_ops += 1
    try:
        yield
    finally:
        with _maintenance_cond:
            _foreground_docker_ops = max(0, _foreground_docker_ops - 1)
            if _foreground_docker_ops == 0:
                _maintenance_cond.notify_all()


@contextlib.contextmanager
def _commit_docker_section() -> Any:
    global _active_commit_ops
    commit_limit = max(1, int(_SERVER_CONFIG.checkpoint_max_inflight))
    with _foreground_docker_section():
        with _commit_ops_cond:
            while _active_commit_ops >= commit_limit:
                _commit_ops_cond.wait()
            _active_commit_ops += 1
        try:
            yield
        finally:
            with _commit_ops_cond:
                _active_commit_ops = max(0, _active_commit_ops - 1)
                _commit_ops_cond.notify_all()


@contextlib.contextmanager
def _maintenance_docker_section() -> Any:
    global _maintenance_active
    with _maintenance_cond:
        while _maintenance_active:
            _maintenance_cond.wait()
        _maintenance_active = True
        while _foreground_docker_ops > 0:
            _maintenance_cond.wait()
    try:
        yield
    finally:
        with _maintenance_cond:
            _maintenance_active = False
            _maintenance_cond.notify_all()


@contextlib.contextmanager
def _container_exec_section(container_id: str) -> Any:
    gate = _get_container_op_gate(container_id)
    with gate.cond:
        gate.waiting_execs += 1
        while gate.exclusive_active:
            gate.cond.wait()
        gate.waiting_execs -= 1
        gate.active_execs += 1
    try:
        yield
    finally:
        with gate.cond:
            gate.active_execs = max(0, gate.active_execs - 1)
            if gate.active_execs == 0:
                gate.cond.notify_all()


@contextlib.contextmanager
def _container_exclusive_section(container_id: str) -> Any:
    gate = _get_container_op_gate(container_id)
    with gate.cond:
        # Exec gets priority over commit/destroy on the same container.
        while gate.exclusive_active or gate.active_execs > 0 or gate.waiting_execs > 0:
            gate.cond.wait()
        gate.exclusive_active = True
    try:
        yield
    finally:
        with gate.cond:
            gate.exclusive_active = False
            gate.cond.notify_all()


@dataclass
class ExecServerConfig:
    use_container_pool: bool = False
    pool_max_size_per_image: int = 4
    pool_max_total_size: int = 0
    pool_default_cwd: str = "/testbed"
    pool_create_timeout_sec: int = 1200
    pool_health_check_timeout_sec: int = 10
    pool_prewarm_ratio: float = 0.8
    pool_prewarm_max_concurrency: int = 0
    pool_resource_stats_dir: str = ""
    checkpoint_enabled: bool = True
    checkpoint_dir: str = "/tmp/swe-checkpoints"
    checkpoint_create_timeout_sec: int = 6
    checkpoint_timeout_cooldown_sec: float = 60.0
    checkpoint_min_ready_latency_sec: float = 2.0
    checkpoint_max_inflight: int = 8
    checkpoint_probe_inspect_timeout_sec: float = 0.3
    exec_fault_injection_default_probability: float = 0.003
    config_path: str = ""

    @classmethod
    def load(cls) -> "ExecServerConfig":
        default_path = Path(__file__).with_name("container_pool_config.json")
        default_stats_dir = Path(__file__).resolve().parents[1] / "resource_stats"
        config_path = Path(os.getenv("SWE_EXEC_SERVER_CONFIG_PATH", str(default_path)))
        raw: dict[str, Any] = {}
        if config_path.exists():
            try:
                raw = json.loads(config_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load exec server config from %s: %s", config_path, exc)

        def _read_int(env_name: str, key: str, default: int) -> int:
            if env_name in os.environ:
                return int(os.environ[env_name])
            return int(raw.get(key, default))

        def _read_str(env_name: str, key: str, default: str) -> str:
            if env_name in os.environ:
                return os.environ[env_name]
            return str(raw.get(key, default))

        def _read_float(env_name: str, key: str, default: float) -> float:
            if env_name in os.environ:
                return float(os.environ[env_name])
            return float(raw.get(key, default))

        use_container_pool = _env_flag(
            "USE_CONTAINER_POOL",
            bool(raw.get("use_container_pool", cls.use_container_pool)),
        )
        return cls(
            use_container_pool=use_container_pool,
            pool_max_size_per_image=_read_int(
                "CONTAINER_POOL_MAX_SIZE_PER_IMAGE",
                "pool_max_size_per_image",
                cls.pool_max_size_per_image,
            ),
            pool_max_total_size=_read_int(
                "CONTAINER_POOL_MAX_TOTAL_SIZE",
                "pool_max_total_size",
                cls.pool_max_total_size,
            ),
            pool_default_cwd=_read_str(
                "CONTAINER_POOL_DEFAULT_CWD",
                "pool_default_cwd",
                cls.pool_default_cwd,
            ),
            pool_create_timeout_sec=_read_int(
                "CONTAINER_POOL_CREATE_TIMEOUT_SEC",
                "pool_create_timeout_sec",
                cls.pool_create_timeout_sec,
            ),
            pool_health_check_timeout_sec=_read_int(
                "CONTAINER_POOL_HEALTH_CHECK_TIMEOUT_SEC",
                "pool_health_check_timeout_sec",
                cls.pool_health_check_timeout_sec,
            ),
            pool_prewarm_ratio=_read_float(
                "CONTAINER_POOL_PREWARM_RATIO",
                "pool_prewarm_ratio",
                cls.pool_prewarm_ratio,
            ),
            pool_prewarm_max_concurrency=_read_int(
                "CONTAINER_POOL_PREWARM_MAX_CONCURRENCY",
                "pool_prewarm_max_concurrency",
                cls.pool_prewarm_max_concurrency,
            ),
            pool_resource_stats_dir=_read_str(
                "CONTAINER_POOL_RESOURCE_STATS_DIR",
                "pool_resource_stats_dir",
                str(default_stats_dir),
            ),
            checkpoint_enabled=_env_flag(
                "SWE_ENABLE_CHECKPOINT",
                bool(raw.get("checkpoint_enabled", cls.checkpoint_enabled)),
            ),
            checkpoint_dir=_read_str(
                "SWE_CHECKPOINT_DIR",
                "checkpoint_dir",
                cls.checkpoint_dir,
            ),
            checkpoint_create_timeout_sec=_read_int(
                "SWE_CHECKPOINT_CREATE_TIMEOUT_SEC",
                "checkpoint_create_timeout_sec",
                cls.checkpoint_create_timeout_sec,
            ),
            checkpoint_timeout_cooldown_sec=_read_float(
                "SWE_CHECKPOINT_TIMEOUT_COOLDOWN_SEC",
                "checkpoint_timeout_cooldown_sec",
                cls.checkpoint_timeout_cooldown_sec,
            ),
            checkpoint_min_ready_latency_sec=_read_float(
                "SWE_CHECKPOINT_MIN_READY_LATENCY_SEC",
                "checkpoint_min_ready_latency_sec",
                cls.checkpoint_min_ready_latency_sec,
            ),
            checkpoint_max_inflight=_read_int(
                "SWE_CHECKPOINT_MAX_INFLIGHT",
                "checkpoint_max_inflight",
                cls.checkpoint_max_inflight,
            ),
            checkpoint_probe_inspect_timeout_sec=_read_float(
                "SWE_CHECKPOINT_PROBE_INSPECT_TIMEOUT_SEC",
                "checkpoint_probe_inspect_timeout_sec",
                cls.checkpoint_probe_inspect_timeout_sec,
            ),
            exec_fault_injection_default_probability=_read_float(
                "SWE_EXEC_FAULT_INJECTION_DEFAULT_PROBABILITY",
                "exec_fault_injection_default_probability",
                cls.exec_fault_injection_default_probability,
            ),
            config_path=str(config_path),
        )


class CheckpointManager:
    def __init__(self, checkpoint_dir: str, enabled: bool, max_inflight: int = 1) -> None:
        self.enabled = enabled
        self.root = Path(checkpoint_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.metadata_path = self.root / "metadata.json"
        self.records_dir = self.root / "checkpoints"
        self.records_dir.mkdir(parents=True, exist_ok=True)
        self.max_inflight = max(1, int(max_inflight))
        self._lock = threading.RLock()
        self._inflight_lock = threading.Lock()
        self._inflight_slots = threading.BoundedSemaphore(self.max_inflight)
        self._checkpoints: dict[str, dict[str, Any]] = {}
        self._ops: dict[str, dict[str, Any]] = {}
        self._inflight = 0
        self._load()

    def _load(self) -> None:
        checkpoints: dict[str, dict[str, Any]] = {}
        if self.metadata_path.exists():
            try:
                payload = json.loads(self.metadata_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load checkpoint metadata from %s: %s", self.metadata_path, exc)
                payload = None
            if isinstance(payload, dict):
                checkpoints = {
                    str(item["checkpoint_id"]): item
                    for item in payload.get("checkpoints", [])
                    if isinstance(item, dict) and item.get("checkpoint_id")
                }
        for record_path in sorted(self.records_dir.glob("*.json")):
            try:
                payload = json.loads(record_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.warning("Failed to load checkpoint record from %s: %s", record_path, exc)
                continue
            if not isinstance(payload, dict) or not payload.get("checkpoint_id"):
                continue
            checkpoints[str(payload["checkpoint_id"])] = payload
        with self._lock:
            self._checkpoints = checkpoints
            self._ops = {}

    def _checkpoint_record_path(self, checkpoint_id: str) -> Path:
        safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(checkpoint_id))
        return self.records_dir / f"{safe}.json"

    def _persist_checkpoint_unlocked(self, record: dict[str, Any]) -> dict[str, Any]:
        started_perf = time.perf_counter()
        checkpoint_id = str(record["checkpoint_id"])
        record_path = self._checkpoint_record_path(checkpoint_id)
        tmp_path = record_path.with_suffix(".tmp")
        encoded = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
        tmp_path.write_text(encoded, encoding="utf-8")
        tmp_path.replace(record_path)
        return {
            "persist_sec": time.perf_counter() - started_perf,
            "metadata_bytes": len(encoded.encode("utf-8")),
            "checkpoint_count": len(self._checkpoints),
            "op_count": len(self._ops),
            "metadata_path": str(record_path),
        }

    def _persist_unlocked(self) -> dict[str, Any]:
        started_perf = time.perf_counter()
        total_bytes = 0
        for checkpoint_id, record in list(self._checkpoints.items()):
            persist_stats = self._persist_checkpoint_unlocked(record)
            total_bytes += int(persist_stats.get("metadata_bytes", 0) or 0)
        for record_path in self.records_dir.glob("*.json"):
            checkpoint_id = record_path.stem
            if checkpoint_id.startswith("swe-ckpt-") and checkpoint_id not in self._checkpoints:
                record_path.unlink(missing_ok=True)
        if self.metadata_path.exists():
            self.metadata_path.unlink(missing_ok=True)
        return {
            "persist_sec": time.perf_counter() - started_perf,
            "metadata_bytes": total_bytes,
            "checkpoint_count": len(self._checkpoints),
            "op_count": len(self._ops),
            "metadata_path": str(self.records_dir),
        }

    def inflight_count(self) -> int:
        with self._inflight_lock:
            return self._inflight

    def can_start_create(self) -> bool:
        with self._inflight_lock:
            return self._inflight < self.max_inflight

    def try_begin_create(self) -> bool:
        if not self._inflight_slots.acquire(blocking=False):
            return False
        with self._inflight_lock:
            self._inflight += 1
        return True

    def begin_create(self) -> None:
        self._inflight_slots.acquire()
        with self._inflight_lock:
            self._inflight += 1

    def end_create(self) -> None:
        should_release = False
        with self._inflight_lock:
            if self._inflight > 0:
                self._inflight -= 1
                should_release = True
        if should_release:
            self._inflight_slots.release()

    def create_checkpoint(
        self,
        *,
        lease_id: str,
        generation: int,
        container_id: str,
        instance_id: str,
        image: str,
        cwd: str,
        step_idx: int,
        command_seq: int,
        policy: str,
        reason: str,
        parent_checkpoint_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        record, op, _ = self.create_checkpoint_with_stats(
            lease_id=lease_id,
            generation=generation,
            container_id=container_id,
            instance_id=instance_id,
            image=image,
            cwd=cwd,
            step_idx=step_idx,
            command_seq=command_seq,
            policy=policy,
            reason=reason,
            parent_checkpoint_id=parent_checkpoint_id,
        )
        return record, op

    def create_checkpoint_with_stats(
        self,
        *,
        lease_id: str,
        generation: int,
        container_id: str,
        instance_id: str,
        image: str,
        cwd: str,
        step_idx: int,
        command_seq: int,
        policy: str,
        reason: str,
        parent_checkpoint_id: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        checkpoint_id = f"swe-ckpt-{uuid.uuid4().hex[:16]}"
        op_id = f"swe-ckpt-op-{uuid.uuid4().hex[:16]}"
        image_tag = _checkpoint_image_tag(checkpoint_id)
        now = time.time()
        record = {
            "checkpoint_id": checkpoint_id,
            "lease_id": lease_id,
            "generation": int(generation),
            "container_id": container_id,
            "instance_id": instance_id,
            "image": image,
            "cwd": cwd,
            "checkpoint_image": image_tag,
            "parent_checkpoint_id": parent_checkpoint_id,
            "step_idx": int(step_idx),
            "command_seq": int(command_seq),
            "policy": policy,
            "reason": reason,
            "status": "pending",
            "created_at": now,
            "ready_at": None,
            "failed_at": None,
            "last_used_at": None,
            "size_bytes": None,
            "error": None,
        }
        op = {
            "op_id": op_id,
            "type": "create",
            "checkpoint_id": checkpoint_id,
            "lease_id": lease_id,
            "container_id": container_id,
            "status": "running",
            "started_at": now,
            "finished_at": None,
            "error": None,
        }
        with self._lock:
            self._checkpoints[checkpoint_id] = record
            self._ops[op_id] = op
            persist_stats = self._persist_checkpoint_unlocked(record)
        return record, op, persist_stats

    def update_checkpoint(self, checkpoint_id: str, **updates: Any) -> dict[str, Any]:
        record, _ = self.update_checkpoint_with_stats(checkpoint_id, **updates)
        return record

    def update_checkpoint_with_stats(self, checkpoint_id: str, **updates: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            record = self._checkpoints[checkpoint_id]
            record.update(updates)
            persist_stats = self._persist_checkpoint_unlocked(record)
            return dict(record), persist_stats

    def update_op(self, op_id: str, **updates: Any) -> dict[str, Any]:
        record, _ = self.update_op_with_stats(op_id, **updates)
        return record

    def update_op_with_stats(self, op_id: str, **updates: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        with self._lock:
            record = self._ops[op_id]
            record.update(updates)
            if str(record.get("status", "")) in {"ready", "failed"}:
                self._ops.pop(op_id, None)
            return dict(record), {
                "persist_sec": 0.0,
                "metadata_bytes": 0,
                "checkpoint_count": len(self._checkpoints),
                "op_count": len(self._ops),
                "metadata_path": str(self.metadata_path),
            }

    def get_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._checkpoints.get(checkpoint_id)
            return dict(record) if record else None

    def list_checkpoints(self, lease_id: str | None = None) -> list[dict[str, Any]]:
        with self._lock:
            records = list(self._checkpoints.values())
        if lease_id is not None:
            records = [item for item in records if item.get("lease_id") == lease_id]
        records.sort(key=lambda item: (float(item.get("step_idx", -1)), float(item.get("created_at", 0.0))))
        return [dict(item) for item in records]

    def latest_ready_checkpoint(self, lease_id: str) -> dict[str, Any] | None:
        records = [
            item for item in self.list_checkpoints(lease_id=lease_id)
            if item.get("status") == "ready"
        ]
        if not records:
            return None
        return records[-1]

    def mark_used(self, checkpoint_id: str) -> None:
        self.update_checkpoint(checkpoint_id, last_used_at=time.time())

    def delete_checkpoint(self, checkpoint_id: str) -> dict[str, Any] | None:
        with self._lock:
            record = self._checkpoints.pop(checkpoint_id, None)
            if record is not None:
                self._checkpoint_record_path(checkpoint_id).unlink(missing_ok=True)
        return dict(record) if record else None

    def clear(self) -> None:
        with self._lock:
            self._checkpoints.clear()
            self._ops.clear()
            self._persist_unlocked()
        with self._inflight_lock:
            self._inflight = 0
            self._inflight_slots = threading.BoundedSemaphore(self.max_inflight)


@dataclass
class ContainerPoolMetrics:
    created_count: int = 0
    prewarmed_count: int = 0
    adopted_count: int = 0
    reused_count: int = 0
    destroy_count: int = 0
    unhealthy_discard_count: int = 0
    warm_miss_count: int = 0
    create_time_sec_total: float = 0.0

    def snapshot(self) -> dict[str, Any]:
        created_avg = self.create_time_sec_total / self.created_count if self.created_count else 0.0
        return {
            **asdict(self),
            "avg_create_time_sec": created_avg,
        }


def _docker(
    *args: str,
    timeout: int = 300,
    input_text: str | None = None,
) -> subprocess.CompletedProcess:
    docker_env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "HOME": os.environ.get("HOME", "/root"),
    }
    for key in (
        "USER",
        "LOGNAME",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "TMPDIR",
        "DOCKER_HOST",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "DOCKER_CONTEXT",
        "XDG_RUNTIME_DIR",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    ):
        value = os.environ.get(key)
        if value:
            docker_env[key] = value
    return subprocess.run(
        ["docker", *args],
        env=docker_env,
        input=input_text,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


_DECIMAL_UNITS = {
    "b": 1,
    "kb": 1000,
    "mb": 1000**2,
    "gb": 1000**3,
    "tb": 1000**4,
}
_BINARY_UNITS = {
    "b": 1,
    "kib": 1024,
    "mib": 1024**2,
    "gib": 1024**3,
    "tib": 1024**4,
}


def _read_meminfo_bytes() -> tuple[int, int, int]:
    """Return (total_bytes, available_bytes, free_bytes) from /proc/meminfo."""
    total_kib = 0
    available_kib = 0
    free_kib = 0
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    total_kib = int(line.split()[1])
                elif line.startswith("MemAvailable:"):
                    available_kib = int(line.split()[1])
                elif line.startswith("MemFree:"):
                    free_kib = int(line.split()[1])
    except Exception:
        return 0, 0, 0
    return total_kib * 1024, available_kib * 1024, free_kib * 1024


def _read_cpu_ticks() -> tuple[int, int]:
    """Return (total_ticks, idle_ticks) from /proc/stat."""
    try:
        with open("/proc/stat", "r", encoding="utf-8") as f:
            first = f.readline().strip()
        parts = first.split()
        if not parts or parts[0] != "cpu":
            return 0, 0
        values = [int(x) for x in parts[1:]]
        if len(values) < 4:
            return 0, 0
        idle = values[3] + (values[4] if len(values) > 4 else 0)  # idle + iowait
        total = sum(values)
        return total, idle
    except Exception:
        return 0, 0


def _read_disk_bytes_totals() -> tuple[int, int]:
    """Return (read_bytes_total, write_bytes_total) from /proc/diskstats."""
    # Use top-level block devices from /sys/block to avoid partition double-counting.
    try:
        base_devices = set(os.listdir("/sys/block"))
    except Exception:
        base_devices = set()

    read_sectors = 0
    write_sectors = 0
    try:
        with open("/proc/diskstats", "r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) < 10:
                    continue
                dev = parts[2]
                if base_devices and dev not in base_devices:
                    continue
                # Fields: ... sectors_read at idx 5, sectors_written at idx 9.
                read_sectors += int(parts[5])
                write_sectors += int(parts[9])
    except Exception:
        return 0, 0

    # Linux sectors are 512 bytes for /proc/diskstats accounting.
    return read_sectors * 512, write_sectors * 512


def _cpu_usage_percent_total() -> float:
    """Return current total CPU usage percent in [0, 100] across all cores."""
    global _cpu_prev_total, _cpu_prev_idle, _cpu_prev_ts
    total, idle = _read_cpu_ticks()
    now = time.time()
    if total <= 0:
        return 0.0

    with _cpu_stat_lock:
        if _cpu_prev_total is None or _cpu_prev_idle is None:
            _cpu_prev_total = total
            _cpu_prev_idle = idle
            _cpu_prev_ts = now
            return 0.0

        dt_total = total - _cpu_prev_total
        dt_idle = idle - _cpu_prev_idle
        _cpu_prev_total = total
        _cpu_prev_idle = idle
        _cpu_prev_ts = now

    if dt_total <= 0:
        return 0.0
    busy = max(0, dt_total - dt_idle)
    usage = 100.0 * (busy / dt_total)
    return max(0.0, min(100.0, usage))


def _disk_bps_used() -> tuple[float, float]:
    """Return (read_bps, write_bps) from /proc/diskstats deltas."""
    global _disk_prev_read_bytes, _disk_prev_write_bytes, _disk_prev_ts
    read_bytes, write_bytes = _read_disk_bytes_totals()
    now = time.time()
    if read_bytes < 0 or write_bytes < 0:
        return 0.0, 0.0

    with _disk_stat_lock:
        if _disk_prev_read_bytes is None or _disk_prev_write_bytes is None or _disk_prev_ts is None:
            _disk_prev_read_bytes = read_bytes
            _disk_prev_write_bytes = write_bytes
            _disk_prev_ts = now
            return 0.0, 0.0

        dt = now - _disk_prev_ts
        dr = read_bytes - _disk_prev_read_bytes
        dw = write_bytes - _disk_prev_write_bytes

        _disk_prev_read_bytes = read_bytes
        _disk_prev_write_bytes = write_bytes
        _disk_prev_ts = now

    if dt <= 0:
        return 0.0, 0.0
    return max(0.0, dr / dt), max(0.0, dw / dt)


def _parse_size_to_bytes(text: str) -> int:
    if not text:
        return 0
    value = text.strip().replace(",", "")
    match = re.match(r"^([0-9]*\.?[0-9]+)\s*([A-Za-z]+)?$", value)
    if match is None:
        return 0
    number = float(match.group(1))
    unit = (match.group(2) or "B").lower()
    if unit in _BINARY_UNITS:
        return int(number * _BINARY_UNITS[unit])
    if unit in _DECIMAL_UNITS:
        return int(number * _DECIMAL_UNITS[unit])
    return int(number)


def _parse_percent(text: str) -> float:
    if not text:
        return 0.0
    cleaned = text.strip().rstrip("%")
    try:
        return float(cleaned)
    except Exception:
        return 0.0


def _is_valid_git_patch(patch_text: str) -> bool:
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


def _docker_create_container(image: str, cwd: str, timeout: int, *, container_name: str | None = None) -> dict[str, Any]:
    container_name = container_name or f"swe-{uuid.uuid4().hex[:12]}"
    started_at = time.time()
    proxy_args: list[str] = []
    for env_name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "no_proxy",
        "all_proxy",
    ):
        env_value = os.getenv(env_name, "").strip()
        if env_value:
            proxy_args.extend(["-e", f"{env_name}={env_value}"])
    r = _docker(
        "run", "-d",
        "--name", container_name,
        "--network", "host",
        *proxy_args,
        "-w", cwd,
        "--pids-limit", "256",
        "--memory", "4g",
        image,
        "sleep", "infinity",
        timeout=timeout,
    )
    elapsed = time.time() - started_at
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or f"docker run failed for {image}")
    return {
        "container_id": r.stdout.strip(),
        "name": container_name,
        "create_time_sec": elapsed,
    }


def _docker_destroy_container(container_id: str, timeout: int = 300) -> None:
    result = _docker("rm", "-f", container_id, timeout=timeout)
    if result.returncode == 0:
        return
    error_text = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
    if "no such container" in error_text.lower():
        logger.info("Container %s already absent during destroy", str(container_id)[:12])
        return
    raise RuntimeError(error_text or f"docker rm -f failed for {container_id}")


def _checkpoint_image_tag(checkpoint_id: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_.-]+", "-", checkpoint_id)
    return f"sweckpt:{safe}"


def _checkpoint_sort_key(item: dict[str, Any]) -> tuple[int, float]:
    raw_step_idx = item.get("step_idx", -1)
    return (
        -1 if raw_step_idx is None else int(raw_step_idx),
        float(item.get("created_at", 0.0) or 0.0),
    )


def _checkpoint_gc_plan(
    records: list[dict[str, Any]],
    *,
    keep_latest: int,
    active_checkpoint_images: set[str] | None = None,
) -> tuple[list[dict[str, Any]], list[str]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record.get("lease_id", ""))].append(record)

    active_images = set(active_checkpoint_images or set())
    deletions: list[dict[str, Any]] = []
    kept_ids: set[str] = set()

    for items in grouped.values():
        by_id = {
            str(item.get("checkpoint_id", "")): item
            for item in items
            if item.get("checkpoint_id")
        }
        protected_ids: set[str] = set()

        ready = [item for item in items if item.get("status") == "ready"]
        ready.sort(key=_checkpoint_sort_key)
        if keep_latest > 0:
            protected_ids.update(
                str(item["checkpoint_id"])
                for item in ready[-keep_latest:]
                if item.get("checkpoint_id")
            )

        for item in items:
            checkpoint_id = str(item.get("checkpoint_id", ""))
            if not checkpoint_id:
                continue
            if item.get("status") == "pending":
                protected_ids.add(checkpoint_id)
            checkpoint_image = str(item.get("checkpoint_image", ""))
            if checkpoint_image and checkpoint_image in active_images:
                protected_ids.add(checkpoint_id)

        stack = list(protected_ids)
        while stack:
            checkpoint_id = stack.pop()
            parent_checkpoint_id = str(by_id.get(checkpoint_id, {}).get("parent_checkpoint_id") or "")
            if parent_checkpoint_id and parent_checkpoint_id in by_id and parent_checkpoint_id not in protected_ids:
                protected_ids.add(parent_checkpoint_id)
                stack.append(parent_checkpoint_id)

        kept_ids.update(protected_ids)

        ready_delete_candidates = [
            item
            for item in ready
            if str(item.get("checkpoint_id", "")) not in protected_ids
        ]
        ready_delete_candidates.sort(key=_checkpoint_sort_key, reverse=True)
        deletions.extend(ready_delete_candidates)

        failed = [item for item in items if item.get("status") == "failed"]
        failed.sort(key=_checkpoint_sort_key, reverse=True)
        deletions.extend(failed)

    seen: set[str] = set()
    unique_deletions: list[dict[str, Any]] = []
    for item in deletions:
        checkpoint_id = str(item.get("checkpoint_id", ""))
        if checkpoint_id and checkpoint_id not in seen:
            unique_deletions.append(item)
            seen.add(checkpoint_id)

    return unique_deletions, sorted(kept_ids)


def _docker_image_size_bytes(image: str, timeout: int = 30) -> int | None:
    r = _docker("image", "inspect", "-f", "{{.Size}}", image, timeout=timeout)
    if r.returncode != 0:
        return None
    try:
        return int(r.stdout.strip())
    except Exception:
        return None


def _docker_list_running_containers(timeout: int = 30) -> list[dict[str, Any]]:
    r = _docker("ps", "--no-trunc", "--format", "{{json .}}", timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() or "docker ps failed")

    containers: list[dict[str, Any]] = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except Exception as exc:
            logger.warning("Failed to parse docker ps output line %r: %s", line, exc)
            continue
        if isinstance(payload, dict):
            containers.append(payload)
    return containers


def _docker_container_is_healthy(container_id: str, cwd: str, timeout: int) -> bool:
    inspect = _docker("inspect", "-f", "{{.State.Running}}", container_id, timeout=timeout)
    if inspect.returncode != 0 or inspect.stdout.strip().lower() != "true":
        return False
    probe = _docker(
        "exec", "-w", cwd, container_id,
        "bash", "-lc", "pwd >/dev/null && test -d .",
        timeout=timeout,
    )
    return probe.returncode == 0


def _load_prewarm_images(resource_stats_dir: str) -> list[str]:
    base = Path(resource_stats_dir)
    if not base.exists():
        logger.warning("Container pool resource_stats dir does not exist: %s", base)
        return []

    images: list[str] = []
    for path in sorted(base.rglob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to parse resource stats file %s: %s", path, exc)
            continue
        if not isinstance(payload, dict):
            continue
        image = str(payload.get("image", "")).strip()
        if not image:
            continue
        images.append(image)
    logger.info(
        "Loaded %d prewarm image entries (%d unique) from %s",
        len(images),
        len(set(images)),
        base,
    )
    return images


class ContainerPool:
    def __init__(
        self,
        *,
        config: ExecServerConfig,
        create_container_fn,
        destroy_container_fn,
        health_check_fn,
        active_count_fn,
        active_container_ids_fn,
        prewarm_images: list[str] | None = None,
    ) -> None:
        self.config = config
        self._create_container_fn = create_container_fn
        self._destroy_container_fn = destroy_container_fn
        self._health_check_fn = health_check_fn
        self._active_count_fn = active_count_fn
        self._active_container_ids_fn = active_container_ids_fn
        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._idle_by_image: dict[str, deque[str]] = defaultdict(deque)
        self._idle_meta: dict[str, dict[str, Any]] = {}
        self._pending_create_count = 0
        self._pending_by_image: dict[str, int] = defaultdict(int)
        self._prewarm_queue: deque[str] = deque()
        self._stop_event = threading.Event()
        self._prewarm_thread: threading.Thread | None = None
        self._prewarm_images = prewarm_images or _load_prewarm_images(self.config.pool_resource_stats_dir)
        self._prewarm_rr_index = 0
        self._metrics = ContainerPoolMetrics()

    @property
    def enabled(self) -> bool:
        return self.config.use_container_pool

    def _target_total_count(self) -> int:
        base = self.config.pool_prewarm_max_concurrency
        if base <= 0:
            base = int(os.getenv("SWE_MAX_CONTAINERS_PER_NODE", "0") or 0)
        if base <= 0 and self.config.pool_max_total_size > 0:
            base = self.config.pool_max_total_size
        if base <= 0:
            return 0
        return max(0, int(base * self.config.pool_prewarm_ratio))

    def _effective_total_cap(self) -> int:
        target = self._target_total_count()
        if self.config.pool_max_total_size > 0 and target > 0:
            return min(self.config.pool_max_total_size, target)
        if self.config.pool_max_total_size > 0:
            return self.config.pool_max_total_size
        return target

    def _idle_and_pending_for_image_unlocked(self, image: str) -> int:
        return len(self._idle_by_image.get(image, ())) + self._pending_by_image.get(image, 0)

    def _total_known_containers_unlocked(self) -> int:
        return len(self._idle_meta) + self._pending_create_count + int(self._active_count_fn())

    def _image_has_capacity_unlocked(self, image: str) -> bool:
        return self._idle_and_pending_for_image_unlocked(image) < max(0, self.config.pool_max_size_per_image)

    def _active_container_ids(self) -> set[str]:
        return {str(container_id) for container_id in self._active_container_ids_fn()}

    @staticmethod
    def _looks_like_prewarmed_container(container: dict[str, Any]) -> bool:
        command = str(container.get("Command", "")).lower()
        return "sleep infinity" in command or "tail -f /dev/null" in command

    def _sync_existing_prewarmed_containers(self) -> int:
        if not self.enabled or not self._prewarm_images:
            return 0

        try:
            containers = _docker_list_running_containers(timeout=self.config.pool_health_check_timeout_sec)
        except Exception as exc:
            logger.warning("Failed to scan existing containers for prewarm adoption: %s", exc)
            return 0

        allowed_images = set(self._prewarm_images)
        active_ids = self._active_container_ids()
        adopted = 0

        for container in containers:
            container_id = str(container.get("ID", "")).strip()
            image = str(container.get("Image", "")).strip()
            name = str(container.get("Names", "")).strip()
            if not container_id or not image or image not in allowed_images:
                continue
            if not self._looks_like_prewarmed_container(container):
                continue
            with self._lock:
                cap = self._effective_total_cap()
                if cap > 0 and self._total_known_containers_unlocked() >= cap:
                    break
                if container_id in self._idle_meta or container_id in active_ids:
                    continue
                if not self._image_has_capacity_unlocked(image):
                    continue
            if not self._health_check_fn(
                container_id,
                self.config.pool_default_cwd,
                self.config.pool_health_check_timeout_sec,
            ):
                continue
            with self._cond:
                cap = self._effective_total_cap()
                active_ids = self._active_container_ids()
                if cap > 0 and self._total_known_containers_unlocked() >= cap:
                    break
                if container_id in self._idle_meta or container_id in active_ids:
                    continue
                if not self._image_has_capacity_unlocked(image):
                    continue
                self._idle_by_image[image].append(container_id)
                self._idle_meta[container_id] = {
                    "name": name,
                    "image": image,
                    "cwd": self.config.pool_default_cwd,
                    "pooled_at": time.time(),
                    "adopted": True,
                }
                self._metrics.adopted_count += 1
                adopted += 1
                logger.info("Adopted existing container %s image=%s into warm pool", container_id[:12], image)
                self._cond.notify_all()

        return adopted

    def _pick_next_image_unlocked(self, preferred_image: str | None = None) -> str | None:
        candidates = self._prewarm_images
        if not candidates:
            return None
        if preferred_image and preferred_image in candidates and self._image_has_capacity_unlocked(preferred_image):
            return preferred_image
        for _ in range(len(candidates)):
            image = candidates[self._prewarm_rr_index % len(candidates)]
            self._prewarm_rr_index += 1
            if self._image_has_capacity_unlocked(image):
                return image
        return None

    def _enqueue_prewarm_unlocked(self, preferred_image: str | None = None) -> bool:
        cap = self._effective_total_cap()
        if cap > 0 and self._total_known_containers_unlocked() >= cap:
            return False
        image = self._pick_next_image_unlocked(preferred_image)
        if not image:
            return False
        self._prewarm_queue.append(image)
        self._pending_create_count += 1
        self._pending_by_image[image] += 1
        self._cond.notify_all()
        return True

    def start(self) -> None:
        if not self.enabled:
            return
        adopted = self._sync_existing_prewarmed_containers()
        with self._lock:
            if self._prewarm_thread is not None:
                return
            self._prewarm_thread = threading.Thread(
                target=self._prewarm_loop,
                name="container-pool-prewarm",
                daemon=True,
            )
            self._prewarm_thread.start()
            while self._enqueue_prewarm_unlocked():
                pass
        if adopted:
            logger.info("Adopted %d existing containers before starting warmup loop", adopted)

    def warmup(self, block: bool = False, timeout: float | None = None) -> None:
        self.start()
        if not block or not self.enabled:
            return
        deadline = None if timeout is None else time.time() + timeout
        while True:
            self._sync_existing_prewarmed_containers()
            with self._cond:
                cap = self._effective_total_cap()
                if cap <= 0:
                    return
                if (
                    len(self._idle_meta) + int(self._active_count_fn()) >= cap
                    and self._pending_create_count == 0
                ):
                    return
                remaining = None if deadline is None else max(0.0, deadline - time.time())
                if remaining == 0.0:
                    return
                self._cond.wait(timeout=remaining)

    def status(self) -> dict[str, Any]:
        with self._lock:
            idle_by_image = {image: len(queue) for image, queue in self._idle_by_image.items() if queue}
            return {
                "enabled": self.enabled,
                "mode": "strict_warm_one_shot" if self.enabled else "disabled",
                "prewarm_create_mode": "serial",
                "prewarm_worker_count": 1 if self.enabled else 0,
                "idle_containers": len(self._idle_meta),
                "pending_creates": self._pending_create_count,
                "idle_by_image": idle_by_image,
                "max_size_per_image": self.config.pool_max_size_per_image,
                "max_total_size": self._effective_total_cap(),
                "prewarm_target_total": self._target_total_count(),
                "prewarm_image_count": len(self._prewarm_images),
                "metrics": self._metrics.snapshot(),
            }

    def _pop_idle_candidate(self, image: str) -> tuple[str, dict[str, Any]] | None:
        with self._lock:
            queue = self._idle_by_image.get(image)
            if not queue:
                return None
            container_id = queue.popleft()
            meta = self._idle_meta.pop(container_id, None)
            if not queue:
                self._idle_by_image.pop(image, None)
            if meta is None:
                return None
            return container_id, meta

    def _prewarm_loop(self) -> None:
        # Intentionally use a single worker thread so docker create happens
        # serially during prewarming and does not overload the docker daemon.
        while not self._stop_event.is_set():
            with self._cond:
                while not self._prewarm_queue and not self._stop_event.is_set():
                    self._cond.wait(timeout=1.0)
                    while self._enqueue_prewarm_unlocked():
                        pass
                if self._stop_event.is_set():
                    return
                image = self._prewarm_queue.popleft()

            created: dict[str, Any] | None = None
            create_error = ""
            try:
                created = self._create_container_fn(
                    image=image,
                    cwd=self.config.pool_default_cwd,
                    timeout=self.config.pool_create_timeout_sec,
                )
            except Exception as exc:
                create_error = str(exc)

            with self._cond:
                self._pending_create_count = max(0, self._pending_create_count - 1)
                self._pending_by_image[image] = max(0, self._pending_by_image.get(image, 0) - 1)
                if self._pending_by_image.get(image, 0) == 0:
                    self._pending_by_image.pop(image, None)

                if created is not None:
                    container_id = created["container_id"]
                    self._idle_by_image[image].append(container_id)
                    self._idle_meta[container_id] = {
                        "name": created["name"],
                        "image": image,
                        "cwd": self.config.pool_default_cwd,
                        "pooled_at": time.time(),
                    }
                    self._metrics.prewarmed_count += 1
                    self._metrics.create_time_sec_total += float(created["create_time_sec"])
                    logger.info(
                        "Prewarmed container %s image=%s create_time=%.3fs",
                        container_id[:12],
                        image,
                        float(created["create_time_sec"]),
                    )
                else:
                    logger.warning("Prewarm create failed image=%s error=%s", image, create_error)
                while self._enqueue_prewarm_unlocked():
                    pass
                self._cond.notify_all()

    def acquire(self, *, image: str, cwd: str, timeout: int) -> dict[str, Any]:
        if self.enabled:
            while True:
                candidate = self._pop_idle_candidate(image)
                if candidate is None:
                    break
                container_id, meta = candidate
                if self._health_check_fn(container_id, meta.get("cwd", cwd), self.config.pool_health_check_timeout_sec):
                    with self._lock:
                        self._metrics.reused_count += 1
                        while self._enqueue_prewarm_unlocked(preferred_image=image):
                            break
                    logger.info(
                        "Using prewarmed container %s for image=%s",
                        container_id[:12],
                        image,
                    )
                    return {
                        "container_id": container_id,
                        "name": meta.get("name", ""),
                        "pooled": True,
                        "acquisition": "prewarmed",
                        "create_time_sec": 0.0,
                        "reset_time_sec": 0.0,
                    }
                with self._lock:
                    self._metrics.unhealthy_discard_count += 1
                logger.warning("Discarding unhealthy pooled container %s image=%s", container_id[:12], image)
                self._destroy_container_fn(container_id)

        with self._lock:
            self._metrics.warm_miss_count += 1
            while self._enqueue_prewarm_unlocked(preferred_image=image):
                break
        created = self._create_container_fn(image=image, cwd=cwd, timeout=timeout)
        with self._lock:
            self._metrics.created_count += 1
            self._metrics.create_time_sec_total += float(created["create_time_sec"])
        logger.info(
            "Created fresh container %s for image=%s in %.3fs",
            created["container_id"][:12],
            image,
            float(created["create_time_sec"]),
        )
        return {
            **created,
            "pooled": False,
            "acquisition": "created",
            "reset_time_sec": 0.0,
        }

    def release(self, *, container_id: str, image: str, name: str, cwd: str) -> dict[str, Any]:
        self._destroy_container_fn(container_id)
        with self._cond:
            self._metrics.destroy_count += 1
            if self.enabled:
                while self._enqueue_prewarm_unlocked(preferred_image=image):
                    break
                self._cond.notify_all()
        return {
            "pooled": False,
            "destroyed": True,
            "reason": "strict_warm_one_shot" if self.enabled else "pool_disabled",
        }


_SERVER_CONFIG = ExecServerConfig.load()
_CHECKPOINTS = CheckpointManager(
    checkpoint_dir=_SERVER_CONFIG.checkpoint_dir,
    enabled=_SERVER_CONFIG.checkpoint_enabled,
    max_inflight=_SERVER_CONFIG.checkpoint_max_inflight,
)
_checkpoint_create_dispatcher_lock = threading.Lock()
_checkpoint_create_dispatcher_queue: queue.Queue[_CheckpointCreateJob] | None = None
_checkpoint_create_dispatcher_threads: list[threading.Thread] = []


def _checkpoint_create_dispatcher_worker() -> None:
    while True:
        job = _checkpoint_create_dispatcher_queue.get()
        job.worker_started_perf = time.perf_counter()
        try:
            job.result = _checkpoint_create_worker(
                job.op_id,
                job.checkpoint_id,
                job.container_id,
                job.checkpoint_image,
                job.record,
                job.runtime_env,
            )
        except Exception as exc:  # pragma: no cover - defensive, worker already catches internally
            job.error = exc
        finally:
            job.worker_finished_perf = time.perf_counter()
            job.done_event.set()
            _checkpoint_create_dispatcher_queue.task_done()


def _ensure_checkpoint_create_dispatcher_started() -> None:
    global _checkpoint_create_dispatcher_queue
    with _checkpoint_create_dispatcher_lock:
        if _checkpoint_create_dispatcher_queue is not None:
            return
        _checkpoint_create_dispatcher_queue = queue.Queue()
        worker_count = max(1, int(_SERVER_CONFIG.checkpoint_max_inflight))
        for idx in range(worker_count):
            thread = threading.Thread(
                target=_checkpoint_create_dispatcher_worker,
                name=f"checkpoint-create-worker-{idx}",
                daemon=True,
            )
            thread.start()
            _checkpoint_create_dispatcher_threads.append(thread)
        logger.info(
            "Started checkpoint create dispatcher worker_count=%s",
            worker_count,
        )


def _dispatch_checkpoint_create_and_wait(
    *,
    op_id: str,
    checkpoint_id: str,
    container_id: str,
    checkpoint_image: str,
    record: dict[str, Any],
    runtime_env: dict[str, str],
) -> tuple[dict[str, Any] | None, float, float]:
    _ensure_checkpoint_create_dispatcher_started()
    enqueued_perf = time.perf_counter()
    job = _CheckpointCreateJob(
        op_id=op_id,
        checkpoint_id=checkpoint_id,
        container_id=container_id,
        checkpoint_image=checkpoint_image,
        record=dict(record),
        runtime_env=dict(runtime_env),
        done_event=threading.Event(),
        enqueued_perf=enqueued_perf,
    )
    _checkpoint_create_dispatcher_queue.put(job)
    job.done_event.wait()
    if job.error is not None:
        raise job.error
    queue_wait_sec = 0.0
    if job.worker_started_perf is not None:
        queue_wait_sec = max(0.0, job.worker_started_perf - enqueued_perf)
    worker_exec_sec = 0.0
    if job.worker_started_perf is not None and job.worker_finished_perf is not None:
        worker_exec_sec = max(0.0, job.worker_finished_perf - job.worker_started_perf)
    return job.result, queue_wait_sec, worker_exec_sec


def _active_container_count_snapshot() -> int:
    with _lock:
        return len(_active_containers)


def _active_container_ids_snapshot() -> list[str]:
    with _lock:
        return list(_active_containers.keys())


_CONTAINER_POOL = ContainerPool(
    config=_SERVER_CONFIG,
    create_container_fn=_docker_create_container,
    destroy_container_fn=_docker_destroy_container,
    health_check_fn=_docker_container_is_healthy,
    active_count_fn=_active_container_count_snapshot,
    active_container_ids_fn=_active_container_ids_snapshot,
)


def _activate_checkpoint_cooldown(reason: str) -> float:
    cooldown_sec = max(0.0, float(_SERVER_CONFIG.checkpoint_timeout_cooldown_sec))
    until_ts = time.time() + cooldown_sec
    with _checkpoint_cooldown_lock:
        global _checkpoint_cooldown_until_ts
        _checkpoint_cooldown_until_ts = max(_checkpoint_cooldown_until_ts, until_ts)
    logger.warning(
        "Checkpoint cooldown activated for %.1fs until %.3f reason=%s",
        cooldown_sec,
        until_ts,
        reason,
    )
    return cooldown_sec


def _checkpoint_cooldown_remaining_sec() -> float:
    with _checkpoint_cooldown_lock:
        remaining = _checkpoint_cooldown_until_ts - time.time()
    return max(0.0, remaining)


def _checkpoint_probe_state(container_id: str | None = None) -> dict[str, Any]:
    cooldown_remaining_sec = _checkpoint_cooldown_remaining_sec()
    inflight = _CHECKPOINTS.inflight_count()
    busy = cooldown_remaining_sec > 0.0 or inflight >= _SERVER_CONFIG.checkpoint_max_inflight
    if cooldown_remaining_sec > 0.0:
        reason = "checkpoint_cooldown_active"
    else:
        reason = "checkpoint_inflight_limit" if busy else "idle"
    metrics: dict[str, Any] = {
        "inflight_checkpoints": inflight,
        "max_inflight_checkpoints": _SERVER_CONFIG.checkpoint_max_inflight,
        "cooldown_remaining_sec": cooldown_remaining_sec,
    }
    inspect_timeout_sec = float(_SERVER_CONFIG.checkpoint_probe_inspect_timeout_sec)
    if not busy and container_id:
        started_at = time.time()
        inspect_error = ""
        inspect_returncode: int | None = None
        size_rw_bytes: int | None = None
        try:
            probe = _docker(
                "inspect",
                "--size",
                "-f",
                "{{.SizeRw}}",
                container_id,
                timeout=inspect_timeout_sec,
            )
            inspect_returncode = probe.returncode
            if probe.returncode == 0:
                try:
                    size_rw_bytes = int((probe.stdout or "").strip())
                except ValueError:
                    size_rw_bytes = None
            else:
                inspect_error = (probe.stderr or probe.stdout or "").strip()
        except subprocess.TimeoutExpired:
            inspect_error = f"docker inspect timeout>{inspect_timeout_sec:.3f}s"
        inspect_latency_sec = time.time() - started_at
        metrics.update(
            {
                "inspect_latency_sec": inspect_latency_sec,
                "inspect_timeout_sec": inspect_timeout_sec,
                "inspect_returncode": inspect_returncode,
                "inspect_size_rw_bytes": size_rw_bytes,
                "inspect_error": inspect_error,
            }
        )
        if inspect_error or inspect_latency_sec > inspect_timeout_sec:
            busy = True
            reason = "checkpoint_probe_inspect_slow"
    return {
        "busy": busy,
        "probe_wait_sec": cooldown_remaining_sec if cooldown_remaining_sec > 0.0 else (1.0 if busy else 0.0),
        "reason": reason,
        "metrics": metrics,
    }


def _checkpoint_create_worker(
    op_id: str,
    checkpoint_id: str,
    container_id: str,
    checkpoint_image: str,
    record: dict[str, Any],
    runtime_env: dict[str, str],
) -> dict[str, Any] | None:
    total_started_perf = time.perf_counter()
    exclusive_wait_sec = 0.0
    commit_section_wait_sec = 0.0
    docker_commit_sec = 0.0
    image_size_sec = 0.0
    ready_delay_sec = 0.0
    checkpoint_persist_sec = 0.0
    op_persist_sec = 0.0
    metadata_bytes = 0
    checkpoint_count = 0
    op_count = 0
    build_payload_sec = 0.0
    runtime_probe_sec = 0.0
    runtime_payload_build_sec = 0.0
    runtime_state_write_sec = 0.0
    post_commit_finalize_sec = 0.0
    try:
        started_at = time.time()
        logger.info(
            "checkpoint worker start checkpoint_id=%s container_id=%s image=%s",
            checkpoint_id,
            container_id,
            checkpoint_image,
        )
        exclusive_wait_started_perf = time.perf_counter()
        with _container_exclusive_section(container_id):
            exclusive_wait_sec = time.perf_counter() - exclusive_wait_started_perf
            logger.info(
                "checkpoint worker acquired exclusive gate checkpoint_id=%s container_id=%s wait_exclusive_sec=%.3f",
                checkpoint_id,
                container_id,
                exclusive_wait_sec,
            )
            if not _container_is_active(container_id) or not _docker_container_is_running(container_id, timeout=10):
                finished_at = time.time()
                error = f"container no longer active before checkpoint commit: {container_id}"
                logger.warning(
                    "checkpoint worker aborted before commit checkpoint_id=%s container_id=%s wait_exclusive_sec=%.3f error=%s",
                    checkpoint_id,
                    container_id,
                    exclusive_wait_sec,
                    error,
                )
                _, checkpoint_stats = _CHECKPOINTS.update_checkpoint_with_stats(
                    checkpoint_id,
                    status="failed",
                    failed_at=finished_at,
                    error=error,
                )
                _, op_stats = _CHECKPOINTS.update_op_with_stats(
                    op_id,
                    status="failed",
                    finished_at=finished_at,
                    error=error,
                )
                checkpoint_persist_sec = float(checkpoint_stats.get("persist_sec", 0.0) or 0.0)
                op_persist_sec = float(op_stats.get("persist_sec", 0.0) or 0.0)
                metadata_bytes = int(op_stats.get("metadata_bytes", checkpoint_stats.get("metadata_bytes", 0)) or 0)
                checkpoint_count = int(op_stats.get("checkpoint_count", checkpoint_stats.get("checkpoint_count", 0)) or 0)
                op_count = int(op_stats.get("op_count", checkpoint_stats.get("op_count", 0)) or 0)
                logger.warning(
                    "checkpoint worker aborted persisted failure checkpoint_id=%s container_id=%s checkpoint_persist_sec=%.3f op_persist_sec=%.3f metadata_bytes=%s checkpoint_count=%s op_count=%s",
                    checkpoint_id,
                    container_id,
                    checkpoint_persist_sec,
                    op_persist_sec,
                    metadata_bytes,
                    checkpoint_count,
                    op_count,
                )
                final_record = dict(_CHECKPOINTS.get_checkpoint(checkpoint_id) or {})
                final_record["build_payload_sec"] = build_payload_sec
                return final_record
            effective_cwd = str(record.get("cwd", "/testbed"))
            build_payload_started_perf = time.perf_counter()
            runtime_probe_started_perf = time.perf_counter()
            probed_runtime_env = _probe_runtime_env(container_id, effective_cwd, runtime_env)
            runtime_probe_sec = time.perf_counter() - runtime_probe_started_perf
            runtime_payload_build_started_perf = time.perf_counter()
            runtime_state_payload = _build_runtime_state_payload(record, probed_runtime_env)
            runtime_payload_build_sec = time.perf_counter() - runtime_payload_build_started_perf
            runtime_state_write_started_perf = time.perf_counter()
            _capture_runtime_state(container_id, checkpoint_id, runtime_state_payload)
            runtime_state_write_sec = time.perf_counter() - runtime_state_write_started_perf
            build_payload_sec = time.perf_counter() - build_payload_started_perf
            commit_section_wait_started_perf = time.perf_counter()
            with _commit_docker_section():
                commit_section_wait_sec = time.perf_counter() - commit_section_wait_started_perf
                logger.info(
                    "checkpoint worker entered commit section checkpoint_id=%s container_id=%s wait_exclusive_sec=%.3f wait_commit_section_sec=%.3f",
                    checkpoint_id,
                    container_id,
                    exclusive_wait_sec,
                    commit_section_wait_sec,
                )
                docker_commit_started_perf = time.perf_counter()
                result = _docker(
                    "commit",
                    container_id,
                    checkpoint_image,
                    timeout=_SERVER_CONFIG.checkpoint_create_timeout_sec,
                )
                docker_commit_sec = time.perf_counter() - docker_commit_started_perf
                logger.info(
                    "checkpoint worker docker commit finished checkpoint_id=%s container_id=%s returncode=%s docker_commit_sec=%.3f stderr=%r",
                    checkpoint_id,
                    container_id,
                    result.returncode,
                    docker_commit_sec,
                    (result.stderr or "").strip()[:400],
                )
        finished_at = time.time()
        if result.returncode != 0:
            error = result.stderr.strip() or f"docker commit failed for {container_id}"
            logger.warning(
                "checkpoint worker failed checkpoint_id=%s container_id=%s wait_exclusive_sec=%.3f wait_commit_section_sec=%.3f runtime_probe_sec=%.3f runtime_payload_build_sec=%.3f runtime_state_write_sec=%.3f build_payload_sec=%.3f docker_commit_sec=%.3f post_commit_finalize_sec=%.3f total_worker_sec=%.3f error=%s",
                checkpoint_id,
                container_id,
                exclusive_wait_sec,
                commit_section_wait_sec,
                runtime_probe_sec,
                runtime_payload_build_sec,
                runtime_state_write_sec,
                build_payload_sec,
                docker_commit_sec,
                post_commit_finalize_sec,
                time.perf_counter() - total_started_perf,
                error,
            )
            _, checkpoint_stats = _CHECKPOINTS.update_checkpoint_with_stats(
                checkpoint_id,
                status="failed",
                failed_at=finished_at,
                error=error,
            )
            _, op_stats = _CHECKPOINTS.update_op_with_stats(
                op_id,
                status="failed",
                finished_at=finished_at,
                error=error,
            )
            checkpoint_persist_sec = float(checkpoint_stats.get("persist_sec", 0.0) or 0.0)
            op_persist_sec = float(op_stats.get("persist_sec", 0.0) or 0.0)
            metadata_bytes = int(op_stats.get("metadata_bytes", checkpoint_stats.get("metadata_bytes", 0)) or 0)
            checkpoint_count = int(op_stats.get("checkpoint_count", checkpoint_stats.get("checkpoint_count", 0)) or 0)
            op_count = int(op_stats.get("op_count", checkpoint_stats.get("op_count", 0)) or 0)
            final_record = dict(_CHECKPOINTS.get_checkpoint(checkpoint_id) or {})
            final_record["build_payload_sec"] = build_payload_sec
            return final_record

        raw_create_latency_sec = finished_at - started_at
        ready_at = finished_at
        post_commit_finalize_started_perf = time.perf_counter()
        _, checkpoint_stats = _CHECKPOINTS.update_checkpoint_with_stats(
            checkpoint_id,
            status="ready",
            ready_at=ready_at,
            size_bytes=None,
            error=None,
            raw_create_latency_sec=raw_create_latency_sec,
            ready_latency_sec=raw_create_latency_sec,
            ready_delay_sec=0.0,
        )
        _, op_stats = _CHECKPOINTS.update_op_with_stats(
            op_id,
            status="ready",
            finished_at=ready_at,
            error=None,
        )
        checkpoint_persist_sec = float(checkpoint_stats.get("persist_sec", 0.0) or 0.0)
        op_persist_sec = float(op_stats.get("persist_sec", 0.0) or 0.0)
        metadata_bytes = int(op_stats.get("metadata_bytes", checkpoint_stats.get("metadata_bytes", 0)) or 0)
        checkpoint_count = int(op_stats.get("checkpoint_count", checkpoint_stats.get("checkpoint_count", 0)) or 0)
        op_count = int(op_stats.get("op_count", checkpoint_stats.get("op_count", 0)) or 0)
        post_commit_finalize_sec = time.perf_counter() - post_commit_finalize_started_perf
        final_record = dict(_CHECKPOINTS.get_checkpoint(checkpoint_id) or {})
        final_record["build_payload_sec"] = build_payload_sec
        final_record["runtime_probe_sec"] = runtime_probe_sec
        final_record["runtime_payload_build_sec"] = runtime_payload_build_sec
        final_record["runtime_state_write_sec"] = runtime_state_write_sec
        final_record["post_commit_finalize_sec"] = post_commit_finalize_sec
        total_worker_sec = time.perf_counter() - total_started_perf
        known_worker_sec = (
            exclusive_wait_sec
            + commit_section_wait_sec
            + runtime_probe_sec
            + runtime_payload_build_sec
            + runtime_state_write_sec
            + docker_commit_sec
            + post_commit_finalize_sec
        )
        unaccounted_worker_sec = max(0.0, total_worker_sec - known_worker_sec)
        logger.info(
            "checkpoint worker ready checkpoint_id=%s container_id=%s wait_exclusive_sec=%.3f wait_commit_section_sec=%.3f runtime_probe_sec=%.3f runtime_payload_build_sec=%.3f runtime_state_write_sec=%.3f build_payload_sec=%.3f docker_commit_sec=%.3f post_commit_finalize_sec=%.3f image_size_sec=%.3f ready_delay_sec=%.3f checkpoint_persist_sec=%.3f op_persist_sec=%.3f metadata_bytes=%s checkpoint_count=%s op_count=%s known_worker_sec=%.3f unaccounted_worker_sec=%.3f total_worker_sec=%.3f size_bytes=%s",
            checkpoint_id,
            container_id,
            exclusive_wait_sec,
            commit_section_wait_sec,
            runtime_probe_sec,
            runtime_payload_build_sec,
            runtime_state_write_sec,
            build_payload_sec,
            docker_commit_sec,
            post_commit_finalize_sec,
            image_size_sec,
            ready_delay_sec,
            checkpoint_persist_sec,
            op_persist_sec,
            metadata_bytes,
            checkpoint_count,
            op_count,
            known_worker_sec,
            unaccounted_worker_sec,
            total_worker_sec,
            None,
        )
        return final_record
    except Exception as exc:
        finished_at = time.time()
        error = str(exc)
        if isinstance(exc, subprocess.TimeoutExpired):
            cooldown_sec = _activate_checkpoint_cooldown(
                f"docker commit timeout checkpoint_id={checkpoint_id} container_id={container_id}"
            )
            error = (
                f"{error}; checkpoint cooldown activated for {cooldown_sec:.1f}s"
            )
        logger.exception(
            "checkpoint worker exception checkpoint_id=%s container_id=%s wait_exclusive_sec=%.3f wait_commit_section_sec=%.3f runtime_probe_sec=%.3f runtime_payload_build_sec=%.3f runtime_state_write_sec=%.3f build_payload_sec=%.3f docker_commit_sec=%.3f post_commit_finalize_sec=%.3f image_size_sec=%.3f total_worker_sec=%.3f",
            checkpoint_id,
            container_id,
            exclusive_wait_sec,
            commit_section_wait_sec,
            runtime_probe_sec,
            runtime_payload_build_sec,
            runtime_state_write_sec,
            build_payload_sec,
            docker_commit_sec,
            post_commit_finalize_sec,
            image_size_sec,
            time.perf_counter() - total_started_perf,
        )
        _, checkpoint_stats = _CHECKPOINTS.update_checkpoint_with_stats(
            checkpoint_id,
            status="failed",
            failed_at=finished_at,
            error=error,
        )
        _, op_stats = _CHECKPOINTS.update_op_with_stats(
            op_id,
            status="failed",
            finished_at=finished_at,
            error=error,
        )
        checkpoint_persist_sec = float(checkpoint_stats.get("persist_sec", 0.0) or 0.0)
        op_persist_sec = float(op_stats.get("persist_sec", 0.0) or 0.0)
        metadata_bytes = int(op_stats.get("metadata_bytes", checkpoint_stats.get("metadata_bytes", 0)) or 0)
        checkpoint_count = int(op_stats.get("checkpoint_count", checkpoint_stats.get("checkpoint_count", 0)) or 0)
        op_count = int(op_stats.get("op_count", checkpoint_stats.get("op_count", 0)) or 0)
        logger.warning(
            "checkpoint worker exception persisted failure checkpoint_id=%s container_id=%s checkpoint_persist_sec=%.3f op_persist_sec=%.3f metadata_bytes=%s checkpoint_count=%s op_count=%s",
            checkpoint_id,
            container_id,
            checkpoint_persist_sec,
            op_persist_sec,
            metadata_bytes,
            checkpoint_count,
            op_count,
        )
        final_record = dict(_CHECKPOINTS.get_checkpoint(checkpoint_id) or {})
        final_record["build_payload_sec"] = build_payload_sec
        return final_record
    finally:
        _CHECKPOINTS.end_create()


def _checkpoint_gc_task_key(lease_id: str | None, checkpoint_ids: list[str]) -> str:
    lease_part = lease_id or "__global__"
    return f"{lease_part}:{','.join(sorted(checkpoint_ids))}"


def _gc_tasks_inflight_count() -> int:
    with _gc_tasks_lock:
        return len(_gc_tasks_inflight)


def _checkpoint_gc_worker(task_key: str, checkpoint_items: list[dict[str, Any]]) -> None:
    try:
        with _maintenance_docker_section():
            for item in checkpoint_items:
                image_name = str(item.get("checkpoint_image", ""))
                if image_name:
                    _docker("image", "rm", "-f", image_name, timeout=60)
                _CHECKPOINTS.delete_checkpoint(str(item["checkpoint_id"]))
    except Exception:
        logger.exception("Asynchronous checkpoint GC worker failed for task %s", task_key)
    finally:
        with _gc_tasks_lock:
            _gc_tasks_inflight.discard(task_key)


# ── Health ────────────────────────────────────────────────────────────

@app.get("/healthz")
def healthz():
    r = _docker("info", "--format", "{{.ContainersRunning}}", timeout=10)
    running = r.stdout.strip() if r.returncode == 0 else "?"
    return jsonify({"ok": True, "running_containers": running})


@app.get("/images")
def list_images():
    r = _docker("images", "--format", "{{.Repository}}:{{.Tag}}", timeout=30)
    if r.returncode != 0:
        return jsonify({"ok": False, "error": r.stderr}), 500
    images = [l.strip() for l in r.stdout.strip().split("\n") if l.strip()]
    return jsonify({"ok": True, "images": images, "count": len(images)})


@app.get("/status")
def status():
    with _lock:
        active = {cid: info for cid, info in _active_containers.items()}
    return jsonify(
        {
            "ok": True,
            "active_containers": len(active),
            "containers": active,
            "container_pool": _CONTAINER_POOL.status(),
            "config": asdict(_SERVER_CONFIG),
        }
    )


@app.get("/host_stats")
def host_stats():
    total_bytes, available_bytes, free_bytes = _read_meminfo_bytes()
    cpu_count = max(1, int(os.cpu_count() or 1))
    cpu_total_percent = float(cpu_count * 100)
    cpu_used_percent = cpu_total_percent * (_cpu_usage_percent_total() / 100.0)
    cpu_available_percent = max(0.0, cpu_total_percent - cpu_used_percent)

    disk_read_total_bps = float(os.getenv("SWE_EXEC_DISK_READ_TOTAL_BPS", str(2 * 1024**3)))
    disk_write_total_bps = float(os.getenv("SWE_EXEC_DISK_WRITE_TOTAL_BPS", str(2 * 1024**3)))
    disk_read_used_bps, disk_write_used_bps = _disk_bps_used()
    disk_read_available_bps = max(0.0, disk_read_total_bps - disk_read_used_bps)
    disk_write_available_bps = max(0.0, disk_write_total_bps - disk_write_used_bps)

    return jsonify(
        {
            "ok": True,
            "memory_total_bytes": total_bytes,
            "memory_available_bytes": available_bytes,
            "memory_free_bytes": free_bytes,
            "cpu_total_percent": cpu_total_percent,
            "cpu_used_percent": cpu_used_percent,
            "cpu_available_percent": cpu_available_percent,
            "disk_read_total_bytes_per_sec": disk_read_total_bps,
            "disk_read_used_bytes_per_sec": disk_read_used_bps,
            "disk_read_available_bytes_per_sec": disk_read_available_bps,
            "disk_write_total_bytes_per_sec": disk_write_total_bps,
            "disk_write_used_bytes_per_sec": disk_write_used_bps,
            "disk_write_available_bytes_per_sec": disk_write_available_bps,
            "ts": time.time(),
        }
    )


# ── Container lifecycle ───────────────────────────────────────────────

@app.post("/container/create")
def container_create():
    """Create a detached container from a SWE-Bench image.

    Request JSON:
        image (str):   full image name, e.g. docker.io/xingyaoww/sweb.eval.x86_64.django_s_django-12345:latest
        cwd (str):     working directory inside container, default "/testbed"
        timeout (int): docker run timeout in seconds, default 1200
    """
    data = request.get_json(force=True) or {}
    image = data.get("image")
    if not image:
        return jsonify({"ok": False, "error": "image is required"}), 400

    cwd = data.get("cwd", "/testbed")
    timeout = int(data.get("timeout", 1200)) # default 20 minutes to allow for high concurrency docker setup time
    try:
        acquired = _CONTAINER_POOL.acquire(image=image, cwd=cwd, timeout=timeout)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500

    container_id = acquired["container_id"]
    with _lock:
        _active_containers[container_id] = {
            "name": acquired["name"],
            "image": image,
            "cwd": cwd,
            "runtime_env": {},
            "created_at": time.time(),
            "pooled": acquired["pooled"],
            "acquisition": acquired["acquisition"],
            "create_time_sec": acquired["create_time_sec"],
            "reset_time_sec": acquired["reset_time_sec"],
        }
    logger.info(
        "Acquired container %s (%s) from %s via %s create_time=%.3fs reset_time=%.3fs",
        container_id[:12],
        acquired["name"],
        image,
        acquired["acquisition"],
        float(acquired["create_time_sec"]),
        float(acquired["reset_time_sec"]),
    )
    return jsonify({"ok": True, **acquired})


def _maybe_inject_exec_fault(
    container_id: str,
    *,
    armed: bool,
    probability: float,
) -> dict[str, Any] | None:
    if not armed or probability <= 0.0:
        return None
    if random.random() >= probability:
        return None

    fault_event = {
        "fault_injected": True,
        "fault_type": "exec_server_random_kill",
        "error_code": "fault_injected_container_killed",
        "container_usable": False,
        "fault_injection_probability": float(probability),
        "container_id": str(container_id),
    }
    with _lock:
        active_info = _active_containers.get(str(container_id))
        if active_info is not None:
            active_info["faulted"] = True
            active_info["fault_injected_at"] = time.time()
            active_info["last_fault_type"] = fault_event["fault_type"]
    try:
        with _container_exclusive_section(str(container_id)):
            with _foreground_docker_section():
                if active_info is None:
                    _docker_destroy_container(str(container_id), timeout=30)
                    fault_event["destroyed"] = True
                    fault_event["destroy_reason"] = "inactive_faulted_container"
                else:
                    release = _CONTAINER_POOL.release(
                        container_id=str(container_id),
                        image=str(active_info.get("image", "")),
                        name=str(active_info.get("name", "")),
                        cwd=str(active_info.get("cwd", _SERVER_CONFIG.pool_default_cwd)),
                    )
                    fault_event["destroyed"] = bool(release.get("destroyed", False))
                    fault_event["destroy_reason"] = str(release.get("reason", "fault_injected_fail_stop"))
        with _lock:
            _active_containers.pop(str(container_id), None)
        _drop_container_op_gate(str(container_id))
    except Exception as exc:
        fault_event["destroy_error"] = str(exc)
        logger.warning("fault injection destroy failed for container %s: %s", container_id, exc)
    logger.warning(
        "Injected exec fault into container %s probability=%.6f destroyed=%s reason=%s",
        str(container_id)[:12],
        float(probability),
        fault_event.get("destroyed", False),
        fault_event.get("destroy_reason", ""),
    )
    return fault_event


@app.post("/container/exec")
def container_exec():
    """Execute a bash command inside a running container.

    Request JSON:
        container_id (str): container ID or name
        command (str):      bash command to execute
        cwd (str):          working directory, default "/testbed"
        timeout (int):      execution timeout in seconds, default 180
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    command = data.get("command")
    if not container_id or not command:
        return jsonify({"ok": False, "error": "container_id and command are required"}), 400

    cwd = data.get("cwd", "/testbed")
    timeout = int(data.get("timeout", 180))
    fault_injection_armed = bool(data.get("fault_injection_armed", False))
    fault_injection_probability = float(
        data.get(
            "fault_injection_probability",
            _SERVER_CONFIG.exec_fault_injection_default_probability,
        )
        or 0.0
    )

    with _lock:
        active_info = _active_containers.get(str(container_id), {})
        persisted_env = dict(active_info.get("runtime_env", {}))

    merged_env = dict(persisted_env)
    for key, value in (data.get("env", {}) or {}).items():
        value_str = str(value).strip()
        if value_str:
            merged_env[str(key)] = value_str

    env_args = []
    for key, value in merged_env.items():
        env_args.extend(["-e", f"{key}={value}"])

    injected_fault = _maybe_inject_exec_fault(
        str(container_id),
        armed=fault_injection_armed,
        probability=fault_injection_probability,
    )
    if injected_fault is not None:
        return jsonify(
            {
                "ok": True,
                "returncode": -1,
                "output": "Injected environment fault: container killed before exec.",
                **injected_fault,
            }
        )

    try:
        with _container_exec_section(container_id):
            with _foreground_docker_section():
                r = _docker(
                    "exec", "-w", cwd, *env_args, container_id,
                    "bash", "-lc", command,
                    timeout=timeout,
                )
        output = r.stdout + r.stderr
        return jsonify({
            "ok": True,
            "returncode": r.returncode,
            "output": output,
            "fault_injected": False,
            "container_usable": True,
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "ok": True,
            "returncode": -1,
            "output": f"Command timed out after {timeout}s",
            "fault_injected": False,
            "container_usable": True,
        })


@app.post("/container/diff")
def container_diff():
    """Get the git patch from the container's working directory.

    Request JSON:
        container_id (str): container ID or name
        cwd (str):          working directory, default "/testbed"
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400

    cwd = data.get("cwd", "/testbed")
    with _container_exec_section(container_id):
        with _foreground_docker_section():
            r = _docker(
                "exec", "-w", cwd, container_id,
                "bash", "-lc", "git add -A && git diff --cached",
                timeout=60,
            )
    return jsonify({
        "ok": True,
        "patch": r.stdout,
        "returncode": r.returncode,
        "error": r.stderr if r.returncode != 0 else "",
    })


@app.post("/container/destroy")
def container_destroy():
    """Stop and remove a container.

    Request JSON:
        container_id (str): container ID or name
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400

    with _lock:
        active_info = _active_containers.get(container_id)

    if active_info is None:
        with _container_exclusive_section(container_id):
            with _foreground_docker_section():
                _docker_destroy_container(container_id, timeout=300)
        _drop_container_op_gate(container_id)
        logger.info("Destroyed unknown container %s outside pool tracking", container_id[:12])
        return jsonify({"ok": True, "destroyed": True, "pooled": False})

    with _container_exclusive_section(container_id):
        with _foreground_docker_section():
            release = _CONTAINER_POOL.release(
                container_id=container_id,
                image=active_info.get("image", ""),
                name=active_info.get("name", ""),
                cwd=active_info.get("cwd", _SERVER_CONFIG.pool_default_cwd),
            )
    with _lock:
        _active_containers.pop(container_id, None)
    _drop_container_op_gate(container_id)
    logger.info(
        "Released container %s pooled=%s destroyed=%s reason=%s",
        container_id[:12],
        release.get("pooled"),
        release.get("destroyed"),
        release.get("reason", ""),
    )
    return jsonify({"ok": True, **release})


@app.post("/container/checkpoint/probe")
def container_checkpoint_probe():
    return jsonify({
        "ok": False,
        "busy": True,
        "error": "checkpoint probe is deprecated; checkpoint/create performs probe inline",
        "error_code": "checkpoint_busy",
        "reason": "checkpoint_probe_deprecated",
        "retryable": True,
        "probe_wait_sec": 1.0,
        "retry_after_sec": 1.0,
    })


@app.post("/container/checkpoint/create")
def container_checkpoint_create():
    request_started_perf = time.perf_counter()
    probe_sec = 0.0
    probe_inspect_sec = 0.0
    create_persist_sec = 0.0
    create_metadata_bytes = 0
    create_checkpoint_count = 0
    create_op_count = 0
    build_payload_sec = 0.0
    runtime_probe_sec = 0.0
    runtime_payload_build_sec = 0.0
    runtime_state_write_sec = 0.0
    post_commit_finalize_sec = 0.0
    worker_blocking_sec = 0.0
    dispatcher_queue_wait_sec = 0.0
    dispatcher_worker_exec_sec = 0.0
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400
    lease_id = str(data.get("lease_id", ""))
    step_idx = int(data.get("step_idx", -1))
    command_seq = int(data.get("command_seq", -1))
    policy = str(data.get("policy", ""))
    reason = str(data.get("reason", "manual"))
    logger.info(
        "checkpoint create request received lease_id=%s container_id=%s step_idx=%s command_seq=%s policy=%s reason=%s inflight_checkpoints=%s",
        lease_id,
        container_id,
        step_idx,
        command_seq,
        policy,
        reason,
        _CHECKPOINTS.inflight_count(),
    )
    if not _SERVER_CONFIG.checkpoint_enabled:
        return jsonify({"ok": False, "error": "checkpoint is disabled", "error_code": "checkpoint_disabled"}), 400
    cooldown_remaining_sec = _checkpoint_cooldown_remaining_sec()
    if cooldown_remaining_sec > 0.0:
        logger.warning(
            "Rejecting checkpoint request during cooldown lease_id=%s container_id=%s step_idx=%s command_seq=%s policy=%s cooldown_remaining_sec=%.3f handler_sec=%.3f",
            lease_id,
            container_id,
            step_idx,
            command_seq,
            policy,
            cooldown_remaining_sec,
            time.perf_counter() - request_started_perf,
        )
        return jsonify({
            "ok": False,
            "error": "checkpoint system is cooling down after a commit timeout",
            "error_code": "checkpoint_busy",
            "retryable": True,
            "busy": True,
            "reason": "checkpoint_cooldown_active",
            "probe_wait_sec": cooldown_remaining_sec,
            "retry_after_sec": cooldown_remaining_sec,
            "metrics": {
                "cooldown_remaining_sec": cooldown_remaining_sec,
                "inflight_checkpoints": _CHECKPOINTS.inflight_count(),
                "max_inflight_checkpoints": _SERVER_CONFIG.checkpoint_max_inflight,
            },
        })

    with _lock:
        active_info = _active_containers.get(container_id)
    if active_info is None:
        logger.warning(
            "checkpoint create request failed unknown container lease_id=%s container_id=%s step_idx=%s command_seq=%s policy=%s handler_sec=%.3f",
            lease_id,
            container_id,
            step_idx,
            command_seq,
            policy,
            time.perf_counter() - request_started_perf,
        )
        return jsonify({"ok": False, "error": f"unknown container_id: {container_id}", "error_code": "unknown_container_id"}), 404
    if not _CHECKPOINTS.try_begin_create():
        logger.info(
            "checkpoint create request rejected by inflight gate lease_id=%s container_id=%s step_idx=%s command_seq=%s policy=%s handler_sec=%.3f inflight_checkpoints=%s max_inflight=%s",
            lease_id,
            container_id,
            step_idx,
            command_seq,
            policy,
            time.perf_counter() - request_started_perf,
            _CHECKPOINTS.inflight_count(),
            _SERVER_CONFIG.checkpoint_max_inflight,
        )
        return jsonify({
            "ok": False,
            "error": "checkpoint system is busy",
            "error_code": "checkpoint_busy",
            "retryable": True,
            "busy": True,
            "reason": "checkpoint_inflight_limit",
            "probe_wait_sec": 1.0,
            "retry_after_sec": 1.0,
            "metrics": {
                "inflight_checkpoints": _CHECKPOINTS.inflight_count(),
                "max_inflight_checkpoints": _SERVER_CONFIG.checkpoint_max_inflight,
                "cooldown_remaining_sec": 0.0,
            },
        })

    record: dict[str, Any] | None = None
    op: dict[str, Any] | None = None
    final_record: dict[str, Any] | None = None
    try:
        record, op, create_stats = _CHECKPOINTS.create_checkpoint_with_stats(
            lease_id=lease_id,
            generation=int(data.get("generation", 0)),
            container_id=container_id,
            instance_id=str(data.get("instance_id", "")),
            image=str(active_info.get("image", "")),
            cwd=str(data.get("cwd", active_info.get("cwd", "/testbed"))),
            step_idx=step_idx,
            command_seq=command_seq,
            policy=policy,
            reason=reason,
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
        )
        create_persist_sec = float(create_stats.get("persist_sec", 0.0) or 0.0)
        create_metadata_bytes = int(create_stats.get("metadata_bytes", 0) or 0)
        create_checkpoint_count = int(create_stats.get("checkpoint_count", 0) or 0)
        create_op_count = int(create_stats.get("op_count", 0) or 0)
        persisted_runtime_env = _normalize_runtime_env(
            active_info.get("runtime_env", {}) if isinstance(active_info, dict) else {}
        )
        request_runtime_env = _normalize_runtime_env(
            data.get("env", {}) if isinstance(data.get("env"), dict) else {}
        )
        runtime_env = dict(persisted_runtime_env)
        runtime_env.update(request_runtime_env)
        worker_started_perf = time.perf_counter()
        final_record, dispatcher_queue_wait_sec, dispatcher_worker_exec_sec = _dispatch_checkpoint_create_and_wait(
            op_id=op["op_id"],
            checkpoint_id=record["checkpoint_id"],
            container_id=container_id,
            checkpoint_image=record["checkpoint_image"],
            record=record,
            runtime_env=runtime_env,
        )
        worker_blocking_sec = time.perf_counter() - worker_started_perf
        build_payload_sec = float(final_record.get("build_payload_sec", 0.0) or 0.0) if final_record else 0.0
        runtime_probe_sec = float(final_record.get("runtime_probe_sec", 0.0) or 0.0) if final_record else 0.0
        runtime_payload_build_sec = float(final_record.get("runtime_payload_build_sec", 0.0) or 0.0) if final_record else 0.0
        runtime_state_write_sec = float(final_record.get("runtime_state_write_sec", 0.0) or 0.0) if final_record else 0.0
        post_commit_finalize_sec = float(final_record.get("post_commit_finalize_sec", 0.0) or 0.0) if final_record else 0.0
    except Exception as exc:
        finished_at = time.time()
        if record is not None:
            _CHECKPOINTS.update_checkpoint(
                record["checkpoint_id"],
                status="failed",
                failed_at=finished_at,
                error=str(exc),
            )
        if op is not None:
            _CHECKPOINTS.update_op(
                op["op_id"],
                status="failed",
                finished_at=finished_at,
                error=str(exc),
            )
        _CHECKPOINTS.end_create()
        logger.exception(
            "checkpoint create request failed lease_id=%s container_id=%s step_idx=%s command_seq=%s policy=%s handler_sec=%.3f",
            lease_id,
            container_id,
            step_idx,
            command_seq,
            policy,
            time.perf_counter() - request_started_perf,
        )
        return jsonify({"ok": False, "error": str(exc), "error_code": "checkpoint_create_failed", "retryable": True})
    if not final_record:
        logger.warning(
            "checkpoint create request lost final record lease_id=%s container_id=%s step_idx=%s command_seq=%s policy=%s checkpoint_id=%s op_id=%s handler_sec=%.3f",
            lease_id,
            container_id,
            step_idx,
            command_seq,
            policy,
            record["checkpoint_id"] if record is not None else None,
            op["op_id"] if op is not None else None,
            time.perf_counter() - request_started_perf,
        )
        return jsonify({
            "ok": False,
            "error": "checkpoint_create lost final record",
            "error_code": "checkpoint_create_missing_record",
            "retryable": True,
        })
    if final_record.get("status") != "ready":
        logger.warning(
            "checkpoint create request completed non-ready lease_id=%s container_id=%s step_idx=%s command_seq=%s policy=%s checkpoint_id=%s op_id=%s status=%s handler_sec=%.3f probe_sec=%.3f probe_inspect_sec=%.3f create_persist_sec=%.3f create_metadata_bytes=%s checkpoint_count=%s op_count=%s runtime_probe_sec=%.3f runtime_payload_build_sec=%.3f runtime_state_write_sec=%.3f build_payload_sec=%.3f post_commit_finalize_sec=%.3f worker_blocking_sec=%.3f dispatcher_queue_wait_sec=%.3f dispatcher_worker_exec_sec=%.3f raw_create_latency_sec=%s ready_latency_sec=%s error=%s",
            lease_id,
            container_id,
            step_idx,
            command_seq,
            policy,
            final_record.get("checkpoint_id"),
            op["op_id"] if op is not None else None,
            final_record.get("status"),
            time.perf_counter() - request_started_perf,
            probe_sec,
            probe_inspect_sec,
            create_persist_sec,
            create_metadata_bytes,
            create_checkpoint_count,
            create_op_count,
            runtime_probe_sec,
            runtime_payload_build_sec,
            runtime_state_write_sec,
            build_payload_sec,
            post_commit_finalize_sec,
            worker_blocking_sec,
            dispatcher_queue_wait_sec,
            dispatcher_worker_exec_sec,
            final_record.get("raw_create_latency_sec"),
            final_record.get("ready_latency_sec"),
            final_record.get("error"),
        )
        return jsonify({
            "ok": False,
            "error": str(final_record.get("error") or "checkpoint_create_failed"),
            "error_code": "checkpoint_create_failed",
            "retryable": True,
            "checkpoint_id": final_record.get("checkpoint_id"),
            "status": final_record.get("status"),
        })
    handler_sec = time.perf_counter() - request_started_perf
    logger.info(
        "checkpoint create request completed lease_id=%s container_id=%s step_idx=%s command_seq=%s policy=%s checkpoint_id=%s op_id=%s handler_sec=%.3f probe_sec=%.3f probe_inspect_sec=%.3f create_persist_sec=%.3f create_metadata_bytes=%s checkpoint_count=%s op_count=%s runtime_probe_sec=%.3f runtime_payload_build_sec=%.3f runtime_state_write_sec=%.3f build_payload_sec=%.3f post_commit_finalize_sec=%.3f worker_blocking_sec=%.3f dispatcher_queue_wait_sec=%.3f dispatcher_worker_exec_sec=%.3f raw_create_latency_sec=%s ready_latency_sec=%s size_bytes=%s",
        lease_id,
        container_id,
        step_idx,
        command_seq,
        policy,
        final_record.get("checkpoint_id"),
        op["op_id"] if op is not None else None,
        handler_sec,
        probe_sec,
        probe_inspect_sec,
        create_persist_sec,
        create_metadata_bytes,
        create_checkpoint_count,
        create_op_count,
        runtime_probe_sec,
        runtime_payload_build_sec,
        runtime_state_write_sec,
        build_payload_sec,
        post_commit_finalize_sec,
        worker_blocking_sec,
        dispatcher_queue_wait_sec,
        dispatcher_worker_exec_sec,
        final_record.get("raw_create_latency_sec"),
        final_record.get("ready_latency_sec"),
        final_record.get("size_bytes"),
    )
    return jsonify({
        "ok": True,
        "checkpoint_id": final_record["checkpoint_id"],
        "op_id": op["op_id"],
        "status": final_record["status"],
        "checkpoint_image": final_record["checkpoint_image"],
        "step_idx": final_record["step_idx"],
        "ready_at": final_record["ready_at"],
        "size_bytes": final_record["size_bytes"],
    })


@app.post("/container/checkpoint/status")
def container_checkpoint_status():
    return jsonify({
        "ok": False,
        "error": "checkpoint_status is deprecated; checkpoint_create is synchronous",
        "error_code": "checkpoint_status_deprecated",
        "retryable": False,
    })


@app.post("/container/checkpoint/list")
def container_checkpoint_list():
    data = request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    checkpoints = _CHECKPOINTS.list_checkpoints(lease_id=str(lease_id) if lease_id is not None else None)
    return jsonify({"ok": True, "checkpoints": checkpoints})


@app.post("/container/checkpoint/delete")
def container_checkpoint_delete():
    data = request.get_json(force=True) or {}
    checkpoint_id = data.get("checkpoint_id")
    if not checkpoint_id:
        return jsonify({"ok": False, "error": "checkpoint_id is required"}), 400
    record = _CHECKPOINTS.get_checkpoint(str(checkpoint_id))
    if record is None:
        return jsonify({"ok": False, "error": f"unknown checkpoint_id: {checkpoint_id}", "error_code": "checkpoint_not_found"}), 404
    if record.get("status") == "pending":
        return jsonify({"ok": False, "error": "checkpoint is still pending", "error_code": "checkpoint_not_ready", "retryable": True}), 409
    image_name = str(record.get("checkpoint_image", ""))
    reclaimed = int(record.get("size_bytes") or 0)
    if image_name:
        with _maintenance_docker_section():
            _docker("image", "rm", "-f", image_name, timeout=60)
    _CHECKPOINTS.delete_checkpoint(str(checkpoint_id))
    return jsonify({"ok": True, "deleted": True, "reclaimed_bytes": reclaimed})


@app.post("/container/checkpoint/gc")
def container_checkpoint_gc():
    data = request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    keep_latest = max(0, int(data.get("keep_latest", 1)))
    dry_run = bool(data.get("dry_run", False))
    records = _CHECKPOINTS.list_checkpoints(lease_id=str(lease_id) if lease_id else None)
    with _lock:
        active_checkpoint_images = {
            str(info.get("image", ""))
            for info in _active_containers.values()
            if str(info.get("image", "")).startswith("sweckpt:")
        }
    unique_deletions, kept_checkpoint_ids = _checkpoint_gc_plan(
        records,
        keep_latest=keep_latest,
        active_checkpoint_images=active_checkpoint_images,
    )

    reclaimed = int(sum(int(item.get("size_bytes") or 0) for item in unique_deletions))
    if not dry_run:
        task_key = _checkpoint_gc_task_key(
            str(lease_id) if lease_id else None,
            [str(item["checkpoint_id"]) for item in unique_deletions],
        )
        should_enqueue = False
        with _gc_tasks_lock:
            if task_key not in _gc_tasks_inflight:
                _gc_tasks_inflight.add(task_key)
                should_enqueue = True
        if should_enqueue and unique_deletions:
            worker = threading.Thread(
                target=_checkpoint_gc_worker,
                args=(task_key, [dict(item) for item in unique_deletions]),
                name=f"checkpoint-gc-{len(unique_deletions)}",
                daemon=True,
            )
            worker.start()
        elif not unique_deletions:
            with _gc_tasks_lock:
                _gc_tasks_inflight.discard(task_key)
    return jsonify({
        "ok": True,
        "deleted_count": len(unique_deletions),
        "deleted_checkpoint_ids": [item["checkpoint_id"] for item in unique_deletions],
        "kept_checkpoint_ids": kept_checkpoint_ids,
        "reclaimed_bytes": reclaimed,
        "dry_run": dry_run,
        "queued": not dry_run,
    })


@app.post("/container/checkpoint/gc/drain")
def container_checkpoint_gc_drain():
    data = request.get_json(force=True) or {}
    timeout_sec = max(0.0, float(data.get("timeout_sec", 600.0) or 0.0))
    poll_interval_sec = max(0.01, float(data.get("poll_interval_sec", 0.1) or 0.1))
    started = time.monotonic()
    initial_inflight_count = _gc_tasks_inflight_count()

    while True:
        remaining_inflight_count = _gc_tasks_inflight_count()
        waited_sec = time.monotonic() - started
        if remaining_inflight_count <= 0:
            return jsonify(
                {
                    "ok": True,
                    "drained": True,
                    "timed_out": False,
                    "waited_sec": waited_sec,
                    "initial_inflight_count": initial_inflight_count,
                    "remaining_inflight_count": 0,
                }
            )
        if waited_sec >= timeout_sec:
            return (
                jsonify(
                    {
                        "ok": False,
                        "drained": False,
                        "timed_out": True,
                        "error": "checkpoint gc drain timed out",
                        "error_code": "checkpoint_gc_drain_timeout",
                        "retryable": True,
                        "waited_sec": waited_sec,
                        "initial_inflight_count": initial_inflight_count,
                        "remaining_inflight_count": remaining_inflight_count,
                    }
                ),
                409,
            )
        time.sleep(min(poll_interval_sec, max(0.0, timeout_sec - waited_sec)))


@app.post("/container/rerun")
def container_rerun():
    data = request.get_json(force=True) or {}
    checkpoint_id = data.get("checkpoint_id")
    old_container_id = data.get("old_container_id")
    if not checkpoint_id:
        return jsonify({"ok": False, "error": "checkpoint_id is required"}), 400
    record = _CHECKPOINTS.get_checkpoint(str(checkpoint_id))
    if record is None:
        return jsonify({"ok": False, "error": f"unknown checkpoint_id: {checkpoint_id}", "error_code": "checkpoint_not_found"}), 404
    if record.get("status") != "ready":
        return jsonify({"ok": False, "error": "checkpoint not ready", "error_code": "checkpoint_not_ready", "retryable": True}), 409

    image = str(record.get("checkpoint_image", ""))
    cwd = str(data.get("cwd", record.get("cwd", "/testbed")))
    timeout = int(data.get("timeout", 120))
    try:
        with _foreground_docker_section():
            created = _docker_create_container(image=image, cwd=cwd, timeout=timeout)
        new_container_id = created["container_id"]
        with _foreground_docker_section():
            runtime_state = _load_runtime_state(new_container_id, str(checkpoint_id))
            _validate_runtime_restore(new_container_id, runtime_state)
        restored_workspace = runtime_state.get("workspace", {}) if isinstance(runtime_state, dict) else {}
        restored_cwd = str(restored_workspace.get("cwd", cwd) or cwd)
        restored_runtime_env = _normalize_runtime_env(
            runtime_state.get("env", {}) if isinstance(runtime_state, dict) else {}
        )
        old_info: dict[str, Any] | None = None
        if old_container_id:
            with _lock:
                snapshot = _active_containers.get(str(old_container_id))
                if isinstance(snapshot, dict):
                    old_info = dict(snapshot)
        if old_container_id:
            with _container_exclusive_section(str(old_container_id)):
                with _foreground_docker_section():
                    _docker_destroy_container(str(old_container_id), timeout=300)
            _drop_container_op_gate(str(old_container_id))
        with _lock:
            if old_container_id:
                _active_containers.pop(str(old_container_id), None)
            _active_containers[new_container_id] = {
                **(old_info or {}),
                "name": created["name"],
                "image": image,
                "cwd": restored_cwd,
                "runtime_env": restored_runtime_env,
                "runtime_state": runtime_state,
                "created_at": time.time(),
                "pooled": False,
                "acquisition": "checkpoint_rerun",
                "create_time_sec": created["create_time_sec"],
                "reset_time_sec": 0.0,
            }
        _CHECKPOINTS.mark_used(str(checkpoint_id))
        return jsonify({
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "checkpoint_image": image,
            "new_container_id": new_container_id,
            "create_time_sec": created["create_time_sec"],
        })
    except Exception as exc:
        if "new_container_id" in locals():
            try:
                with _container_exclusive_section(str(new_container_id)):
                    with _foreground_docker_section():
                        _docker_destroy_container(str(new_container_id), timeout=300)
                _drop_container_op_gate(str(new_container_id))
            except Exception:
                logger.exception("Failed to destroy rerun container after restore error: %s", new_container_id)
        return jsonify({"ok": False, "error": str(exc), "error_code": "rerun_failed", "retryable": True}), 500


@app.post("/container/stats")
def container_stats():
    """Read one-shot docker stats for a running container.

    Returns:
        {
            ok: bool,
            memory_usage_bytes: int,
            cpu_percent: float,
            disk_read_bytes: int,
            disk_write_bytes: int,
        }
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400

    r = _docker("stats", "--no-stream", "--format", "{{json .}}", container_id, timeout=20)
    if r.returncode != 0:
        return jsonify({"ok": False, "error": r.stderr.strip() or "docker stats failed"}), 500

    line = r.stdout.strip().splitlines()
    if not line:
        return jsonify({"ok": False, "error": "empty docker stats output"}), 500

    try:
        payload = json.loads(line[-1])
    except Exception as exc:
        return jsonify({"ok": False, "error": f"failed to parse docker stats json: {exc}"}), 500

    mem_usage = (payload.get("MemUsage") or "0B / 0B").split("/", 1)[0].strip()
    cpu_percent = _parse_percent(payload.get("CPUPerc", "0%"))
    block_io = payload.get("BlockIO") or "0B / 0B"
    if "/" in block_io:
        read_txt, write_txt = [x.strip() for x in block_io.split("/", 1)]
    else:
        read_txt, write_txt = block_io.strip(), "0B"

    out = {
        "ok": True,
        "memory_usage_bytes": _parse_size_to_bytes(mem_usage),
        "cpu_percent": cpu_percent,
        "disk_read_bytes": _parse_size_to_bytes(read_txt),
        "disk_write_bytes": _parse_size_to_bytes(write_txt),
        "raw": payload,
        "ts": time.time(),
    }
    return jsonify(out)


# ── Evaluation (run test script inside a fresh container) ─────────────

@app.post("/container/evaluate")
def container_evaluate():
    """Apply a git patch and run the eval script inside the container.

    This replicates what swe_utils.evaluate_trajectory() does locally,
    but executed on the remote Docker node.

    Request JSON:
        container_id (str): container ID
        patch (str):        git diff patch to apply
        eval_script (str):  bash script to run for evaluation
        cwd (str):          working directory, default "/testbed"
        timeout (int):      eval timeout in seconds, default 3600
    """
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    patch = data.get("patch", "")
    eval_script = data.get("eval_script", "")
    cwd = data.get("cwd", "/testbed")
    timeout = int(data.get("timeout", 3600))

    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400
    if not _is_valid_git_patch(patch):
        return jsonify({
            "ok": True,
            "resolved": False,
            "error": "patch validation failed: expected unified git diff with diff --git headers",
        })

    # Reset the container to HEAD before applying the patch.  The model has
    # already modified files in this same container, so without the reset
    # `git apply` would see the *already-changed* working tree and fail to find
    # the original context lines it needs to apply the diff.
    apply_cmd = "git reset --hard HEAD && git clean -fd && git apply -"
    r_apply = _docker(
        "exec", "-i", "-w", cwd, container_id,
        "bash", "-lc", apply_cmd,
        input_text=patch,
        timeout=60,
    )
    if r_apply.returncode != 0:
        return jsonify({
            "ok": True,
            "resolved": False,
            "error": f"git apply failed: {r_apply.stderr}",
        })

    eval_cmd = "bash -s --"
    try:
        r_eval = _docker(
            "exec", "-i", "-w", cwd, container_id,
            "bash", "-lc", eval_cmd,
            input_text=eval_script,
            timeout=timeout,
        )
        resolved = r_eval.returncode == 0
        return jsonify({
            "ok": True,
            "resolved": resolved,
            "returncode": r_eval.returncode,
            "output": (r_eval.stdout + r_eval.stderr)[-2000:],
        })
    except subprocess.TimeoutExpired:
        return jsonify({
            "ok": True,
            "resolved": False,
            "error": f"Evaluation timed out after {timeout}s",
        })


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="SWE Docker Exec Server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s %(levelname)s %(name)s] %(message)s",
    )
    _CONTAINER_POOL.warmup(block=False)
    logger.info(
        "Starting SWE exec server on %s:%s pool_enabled=%s config=%s per_image=%s total_cap=%s prewarm_target=%s prewarm_ratio=%.2f create_timeout=%ss prewarm_mode=serial stats_dir=%s",
        args.host,
        args.port,
        _SERVER_CONFIG.use_container_pool,
        _SERVER_CONFIG.config_path,
        _SERVER_CONFIG.pool_max_size_per_image,
        _CONTAINER_POOL._effective_total_cap(),
        _CONTAINER_POOL._target_total_count(),
        _SERVER_CONFIG.pool_prewarm_ratio,
        _SERVER_CONFIG.pool_create_timeout_sec,
        _SERVER_CONFIG.pool_resource_stats_dir,
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
