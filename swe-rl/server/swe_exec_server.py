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
import importlib
import json
import logging
import os
import queue
import random
import re
import shlex
import shutil
import subprocess
import sys
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
_stats_cache: dict[str, dict[str, Any]] = {}
_stats_cache_lock = threading.RLock()
_stats_sampler_lock = threading.Lock()
_stats_sampler_thread: threading.Thread | None = None
_stats_sampler_stop = threading.Event()
_action_stats_sampler_lock = threading.Lock()
_action_stats_sampler_thread: threading.Thread | None = None
_action_stats_wakeup = threading.Event()
_action_stats_containers: defaultdict[str, int] = defaultdict(int)
_cgroup_cache_lock = threading.RLock()
_cgroup_path_cache: dict[str, str] = {}
_cgroup_cpu_prev: dict[str, tuple[int, float]] = {}
try:
    _PROC_CLK_TCK = max(1, int(os.sysconf("SC_CLK_TCK")))
except Exception:
    _PROC_CLK_TCK = 100

_CGROUP_MEMORY_ROOTS = [
    Path("/sys/fs/cgroup/memory"),
    Path("/sys/fs/cgroup/unified"),
    Path("/sys/fs/cgroup"),
]
_CGROUP_CPU_ROOTS = [
    Path("/sys/fs/cgroup/cpu,cpuacct"),
    Path("/sys/fs/cgroup/cpuacct"),
    Path("/sys/fs/cgroup/cpu"),
    Path("/sys/fs/cgroup/unified"),
    Path("/sys/fs/cgroup"),
]
_CGROUP_IO_ROOTS = [
    Path("/sys/fs/cgroup/blkio"),
    Path("/sys/fs/cgroup/unified"),
    Path("/sys/fs/cgroup"),
]
_CPU_SAMPLE_MIN_ELAPSED_SEC = max(
    0.0,
    float(os.getenv("SWE_EXEC_STATS_CPU_SAMPLE_MIN_ELAPSED_SEC", "0.5")),
)
_CPU_SAMPLE_MAX_PERCENT = max(
    1.0,
    float(os.getenv("SWE_EXEC_STATS_MAX_CONTAINER_CPU_PERCENT", "1000.0")),
)

_RUNTIME_STATE_SCHEMA_VERSION = 1
_RUNTIME_STATE_BASE_DIR = "/tmp/swe-runtime-checkpoints"
_RUNTIME_ENV_WHITELIST = ("PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX")
_EXEC_FAULT_PHASES = frozenset({"before_action", "mid_action"})
_CHECKPOINT_FAULT_PHASES = frozenset({"before_commit", "after_commit_before_ready"})


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
    fault_injection_spec: dict[str, Any] | None
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


def _normalize_fault_injection_spec(
    raw: dict[str, Any] | None,
    *,
    allowed_phases: set[str] | frozenset[str],
) -> dict[str, Any] | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError("fault_injection_spec must be a JSON object")
    phase = str(raw.get("phase", "") or "").strip()
    if not phase:
        raise ValueError("fault_injection_spec.phase is required")
    if phase not in allowed_phases:
        allowed = ", ".join(sorted(allowed_phases))
        raise ValueError(f"unsupported fault injection phase: {phase} (allowed: {allowed})")
    try:
        delay_sec = max(0.0, float(raw.get("delay_sec", 0.0) or 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("fault_injection_spec.delay_sec must be numeric") from exc
    tag = str(raw.get("tag", "") or "").strip()
    return {
        "phase": phase,
        "delay_sec": delay_sec,
        "tag": tag,
    }


def _build_fault_event(
    *,
    container_id: str,
    fault_type: str,
    fault_phase: str,
    delay_sec: float = 0.0,
    tag: str = "",
) -> dict[str, Any]:
    event = {
        "fault_injected": True,
        "fault_type": fault_type,
        "fault_phase": fault_phase,
        "error_code": "fault_injected_container_killed",
        "container_usable": False,
        "container_id": str(container_id),
    }
    if delay_sec > 0.0:
        event["fault_delay_sec"] = float(delay_sec)
    if tag:
        event["fault_tag"] = tag
    return event


def _mark_container_faulted(container_id: str, *, fault_type: str) -> dict[str, Any] | None:
    with _lock:
        active_info = _active_containers.get(str(container_id))
        if active_info is not None:
            active_info["faulted"] = True
            active_info["fault_injected_at"] = time.time()
            active_info["last_fault_type"] = fault_type
            return dict(active_info)
    return None


def _release_container_for_fault(
    container_id: str,
    *,
    active_info: dict[str, Any] | None,
    remove_tracking: bool,
    drop_gate: bool,
) -> dict[str, Any]:
    outcome: dict[str, Any] = {}
    try:
        # Fault injection must interrupt foreground docker operations instead of
        # queueing behind them; otherwise a "random" fail-stop only fires after
        # exec/commit finishes, which does not validate in-flight recovery.
        if active_info is None:
            _docker_destroy_container(str(container_id), timeout=30)
            outcome["destroyed"] = True
            outcome["destroy_reason"] = "inactive_faulted_container"
        else:
            release = _CONTAINER_POOL.release(
                container_id=str(container_id),
                image=str(active_info.get("image", "")),
                name=str(active_info.get("name", "")),
                cwd=str(active_info.get("cwd", _SERVER_CONFIG.pool_default_cwd)),
            )
            outcome["destroyed"] = bool(release.get("destroyed", False))
            outcome["destroy_reason"] = str(release.get("reason", "fault_injected_fail_stop"))
    except Exception as exc:  # pragma: no cover - best-effort cleanup
        outcome["destroyed"] = False
        outcome["destroy_error"] = str(exc)
    if remove_tracking:
        with _lock:
            _active_containers.pop(str(container_id), None)
    if drop_gate:
        _drop_container_op_gate(str(container_id))
    return outcome


def _inject_fail_stop_fault(
    container_id: str,
    *,
    fault_type: str,
    fault_phase: str,
    delay_sec: float = 0.0,
    tag: str = "",
    remove_tracking: bool,
    drop_gate: bool,
) -> dict[str, Any]:
    active_info = _mark_container_faulted(
        str(container_id),
        fault_type=fault_type,
    )
    event = _build_fault_event(
        container_id=str(container_id),
        fault_type=fault_type,
        fault_phase=fault_phase,
        delay_sec=delay_sec,
        tag=tag,
    )
    event.update(
        _release_container_for_fault(
            str(container_id),
            active_info=active_info,
            remove_tracking=remove_tracking,
            drop_gate=drop_gate,
        )
    )
    logger.warning(
        "Injected explicit fault into container %s phase=%s type=%s destroyed=%s reason=%s",
        str(container_id)[:12],
        fault_phase,
        fault_type,
        event.get("destroyed", False),
        event.get("destroy_reason", ""),
    )
    return event


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
    checkpoint_backend: str = "full"
    checkpoint_create_timeout_sec: int = 15
    full_checkpoint_project_root: str = ""
    full_checkpoint_state_root: str = ""
    full_checkpoint_docker_root: str = ""
    full_checkpoint_runtime_staging_root: str = "/dev/shm/docker-full-checkpoint"
    full_checkpoint_criu_timeout_sec: int = 120
    checkpoint_timeout_cooldown_sec: float = 60.0
    checkpoint_min_ready_latency_sec: float = 2.0
    checkpoint_max_inflight: int = 12
    checkpoint_probe_inspect_timeout_sec: float = 0.3
    exec_fault_injection_default_probability: float = 0.003
    stats_sampler_enabled: bool = True
    stats_backend: str = "cgroup"
    stats_sampler_interval_sec: float = 1.0
    stats_cache_ttl_sec: float = 5.0
    stats_command_timeout_sec: int = 20
    stats_batch_max_containers: int = 256
    action_stats_sampler_enabled: bool = True
    action_stats_interval_sec: float = 0.5
    action_stats_command_timeout_sec: int = 5
    action_stats_batch_max_containers: int = 64
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
            checkpoint_backend=_read_str(
                "SWE_CHECKPOINT_BACKEND",
                "checkpoint_backend",
                cls.checkpoint_backend,
            ),
            checkpoint_create_timeout_sec=_read_int(
                "SWE_CHECKPOINT_CREATE_TIMEOUT_SEC",
                "checkpoint_create_timeout_sec",
                cls.checkpoint_create_timeout_sec,
            ),
            full_checkpoint_project_root=_read_str(
                "SWE_FULL_CHECKPOINT_PROJECT_ROOT",
                "full_checkpoint_project_root",
                cls.full_checkpoint_project_root,
            ),
            full_checkpoint_state_root=_read_str(
                "SWE_FULL_CHECKPOINT_STATE_ROOT",
                "full_checkpoint_state_root",
                cls.full_checkpoint_state_root,
            ),
            full_checkpoint_docker_root=_read_str(
                "SWE_FULL_CHECKPOINT_DOCKER_ROOT",
                "full_checkpoint_docker_root",
                cls.full_checkpoint_docker_root,
            ),
            full_checkpoint_runtime_staging_root=_read_str(
                "SWE_FULL_CHECKPOINT_RUNTIME_STAGING_ROOT",
                "full_checkpoint_runtime_staging_root",
                cls.full_checkpoint_runtime_staging_root,
            ),
            full_checkpoint_criu_timeout_sec=_read_int(
                "SWE_FULL_CHECKPOINT_CRIU_TIMEOUT_SEC",
                "full_checkpoint_criu_timeout_sec",
                cls.full_checkpoint_criu_timeout_sec,
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
            stats_sampler_enabled=_env_flag(
                "SWE_EXEC_STATS_SAMPLER_ENABLE",
                bool(raw.get("stats_sampler_enabled", cls.stats_sampler_enabled)),
            ),
            stats_backend=_read_str(
                "SWE_EXEC_STATS_BACKEND",
                "stats_backend",
                cls.stats_backend,
            ),
            stats_sampler_interval_sec=_read_float(
                "SWE_EXEC_STATS_SAMPLER_INTERVAL_SEC",
                "stats_sampler_interval_sec",
                cls.stats_sampler_interval_sec,
            ),
            stats_cache_ttl_sec=_read_float(
                "SWE_EXEC_STATS_CACHE_TTL_SEC",
                "stats_cache_ttl_sec",
                cls.stats_cache_ttl_sec,
            ),
            stats_command_timeout_sec=_read_int(
                "SWE_EXEC_STATS_COMMAND_TIMEOUT_SEC",
                "stats_command_timeout_sec",
                cls.stats_command_timeout_sec,
            ),
            stats_batch_max_containers=_read_int(
                "SWE_EXEC_STATS_BATCH_MAX_CONTAINERS",
                "stats_batch_max_containers",
                cls.stats_batch_max_containers,
            ),
            action_stats_sampler_enabled=_env_flag(
                "SWE_EXEC_ACTION_STATS_SAMPLER_ENABLE",
                bool(raw.get("action_stats_sampler_enabled", cls.action_stats_sampler_enabled)),
            ),
            action_stats_interval_sec=_read_float(
                "SWE_EXEC_ACTION_STATS_INTERVAL_SEC",
                "action_stats_interval_sec",
                cls.action_stats_interval_sec,
            ),
            action_stats_command_timeout_sec=_read_int(
                "SWE_EXEC_ACTION_STATS_COMMAND_TIMEOUT_SEC",
                "action_stats_command_timeout_sec",
                cls.action_stats_command_timeout_sec,
            ),
            action_stats_batch_max_containers=_read_int(
                "SWE_EXEC_ACTION_STATS_BATCH_MAX_CONTAINERS",
                "action_stats_batch_max_containers",
                cls.action_stats_batch_max_containers,
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
        checkpoint_backend: str = "legacy",
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
            checkpoint_backend=checkpoint_backend,
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
        checkpoint_backend: str = "legacy",
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        checkpoint_id = f"swe-ckpt-{uuid.uuid4().hex[:16]}"
        op_id = f"swe-ckpt-op-{uuid.uuid4().hex[:16]}"
        image_tag = _checkpoint_image_tag(checkpoint_id)
        checkpoint_backend = _normalize_checkpoint_backend(
            checkpoint_backend, default="legacy"
        )
        now = time.time()
        record = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_backend": checkpoint_backend,
            "lease_id": lease_id,
            "generation": int(generation),
            "container_id": container_id,
            "instance_id": instance_id,
            "image": image,
            "cwd": cwd,
            "checkpoint_image": image_tag if checkpoint_backend == "legacy" else None,
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
        records.sort(key=_checkpoint_sort_key)
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


def _parse_docker_stats_payload(payload: dict[str, Any], *, include_raw: bool = False) -> dict[str, Any]:
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
        "avg_cpu_percent": cpu_percent,
        "cpu_sample_valid": True,
        "cpu_sample_elapsed_sec": 1.0,
        "cpu_source": "docker_stats",
        "disk_read_bytes": _parse_size_to_bytes(read_txt),
        "disk_write_bytes": _parse_size_to_bytes(write_txt),
        "ts": time.time(),
    }
    if include_raw:
        out["raw"] = payload
    return out


def _match_stats_container_id(reported: str, requested: list[str], used: set[str]) -> str | None:
    key = str(reported or "").strip()
    if not key:
        return None
    for container_id in requested:
        if container_id in used:
            continue
        if container_id == key or container_id.startswith(key) or key.startswith(container_id):
            return container_id
    return None


def _docker_stats_for_containers(
    container_ids: list[str],
    *,
    timeout: int,
    include_raw: bool = False,
) -> dict[str, dict[str, Any]]:
    requested = [str(cid).strip() for cid in container_ids if str(cid).strip()]
    if not requested:
        return {}
    r = _docker("stats", "--no-stream", "--format", "{{json .}}", *requested, timeout=timeout)
    if r.returncode != 0:
        if len(requested) <= 1:
            raise RuntimeError(r.stderr.strip() or "docker stats failed")
        out: dict[str, dict[str, Any]] = {}
        for container_id in requested:
            try:
                out.update(
                    _docker_stats_for_containers(
                        [container_id],
                        timeout=timeout,
                        include_raw=include_raw,
                    )
                )
            except Exception as exc:
                out[container_id] = {
                    "ok": False,
                    "error": str(exc),
                    "container_id": container_id,
                    "ts": time.time(),
                }
        return out

    lines = [line for line in r.stdout.strip().splitlines() if line.strip()]
    out: dict[str, dict[str, Any]] = {}
    used: set[str] = set()
    pending_payloads: list[dict[str, Any]] = []
    for line in lines:
        try:
            payload = json.loads(line)
        except Exception as exc:
            logger.warning("Failed to parse docker stats json line: %s", exc)
            continue
        reported = payload.get("Container") or payload.get("ID") or payload.get("Name")
        container_id = _match_stats_container_id(str(reported or ""), requested, used)
        if container_id is None:
            pending_payloads.append(payload)
            continue
        out[container_id] = _parse_docker_stats_payload(payload, include_raw=include_raw)
        used.add(container_id)

    remaining = [cid for cid in requested if cid not in used]
    for container_id, payload in zip(remaining, pending_payloads):
        out[container_id] = _parse_docker_stats_payload(payload, include_raw=include_raw)
        used.add(container_id)
    return out


def _read_int_file(path: Path) -> int:
    return int(path.read_text(encoding="utf-8").strip())


def _first_existing_file(directory: Path, names: list[str]) -> Path | None:
    for name in names:
        path = directory / name
        if path.exists():
            return path
    return None


def _find_related_cgroup_file(
    anchor_path: Path,
    *,
    anchor_roots: list[Path],
    target_roots: list[Path],
    names: list[str],
) -> Path | None:
    anchor_dir = anchor_path.parent
    direct = _first_existing_file(anchor_dir, names)
    if direct is not None:
        return direct

    seen: set[Path] = set()
    for anchor_root in anchor_roots:
        try:
            relative = anchor_dir.relative_to(anchor_root)
        except ValueError:
            continue
        for target_root in target_roots:
            candidate_dir = target_root / relative
            if candidate_dir in seen:
                continue
            seen.add(candidate_dir)
            found = _first_existing_file(candidate_dir, names)
            if found is not None:
                return found
    return None


def _candidate_cgroup_dirs(root: Path, container_id: str) -> list[Path]:
    short_id = container_id[:12]
    names = [
        container_id,
        short_id,
        f"docker-{container_id}.scope",
        f"docker-{short_id}.scope",
    ]
    parents = ["", "docker", "system.slice", "machine.slice"]
    candidates: list[Path] = []
    for parent in parents:
        base = root / parent if parent else root
        for name in names:
            candidates.append(base / name)
    return candidates


def _find_cgroup_file(container_id: str, *, cache_kind: str, roots: list[Path], names: list[str]) -> Path | None:
    cache_key = f"{cache_kind}:{container_id}"
    with _cgroup_cache_lock:
        cached = _cgroup_path_cache.get(cache_key)
    if cached:
        cached_path = Path(cached)
        if cached_path.exists():
            return cached_path

    for root in roots:
        if not root.exists():
            continue
        for directory in _candidate_cgroup_dirs(root, container_id):
            for name in names:
                path = directory / name
                if path.exists():
                    with _cgroup_cache_lock:
                        _cgroup_path_cache[cache_key] = str(path)
                    return path

    short_id = container_id[:12]
    for root in roots:
        if not root.exists():
            continue
        root_depth = len(root.parts)
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                depth = len(Path(dirpath).parts) - root_depth
                if depth > 6:
                    dirnames[:] = []
                    continue
                if container_id not in dirpath and short_id not in dirpath:
                    continue
                for name in names:
                    if name in filenames:
                        path = Path(dirpath) / name
                        with _cgroup_cache_lock:
                            _cgroup_path_cache[cache_key] = str(path)
                        return path
        except OSError:
            continue
    return None


def _read_cgroup_cpu_usage_ns(path: Path) -> int:
    if path.name == "cpu.stat":
        for line in path.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "usage_usec":
                return int(parts[1]) * 1000
        raise ValueError(f"{path} does not expose usage_usec")
    return _read_int_file(path)


def _read_proc_stat_cpu_ticks(pid: int) -> int | None:
    try:
        text = (Path("/proc") / str(pid) / "stat").read_text(encoding="utf-8")
    except OSError:
        return None
    end = text.rfind(")")
    if end < 0:
        return None
    fields = text[end + 2 :].split()
    if len(fields) <= 12:
        return None
    try:
        return int(fields[11]) + int(fields[12])
    except (TypeError, ValueError):
        return None


def _read_proc_cpu_usage_ns(cgroup_dir: Path) -> int | None:
    pids: set[int] = set()
    root_depth = len(cgroup_dir.parts)
    try:
        walker = os.walk(cgroup_dir)
        for dirpath, dirnames, filenames in walker:
            depth = len(Path(dirpath).parts) - root_depth
            if depth > 4:
                dirnames[:] = []
                continue
            for name in ("tasks", "cgroup.procs"):
                if name not in filenames:
                    continue
                path = Path(dirpath) / name
                try:
                    for line in path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            pids.add(int(line))
                        except ValueError:
                            continue
                except OSError:
                    continue
    except OSError:
        return None

    if not pids:
        return None

    ticks = 0
    any_read = False
    for pid in pids:
        value = _read_proc_stat_cpu_ticks(pid)
        if value is None:
            continue
        any_read = True
        ticks += value
    if not any_read:
        return None
    return int(ticks * 1_000_000_000 / _PROC_CLK_TCK)


def _cpu_percent_from_usage(
    container_id: str,
    *,
    source: str,
    usage_ns: int,
    now: float,
) -> tuple[float, bool, float]:
    prev_key = f"{source}:{container_id}"
    with _cgroup_cache_lock:
        prev = _cgroup_cpu_prev.get(prev_key)
        if prev is None:
            _cgroup_cpu_prev[prev_key] = (usage_ns, now)
            return 0.0, False, 0.0
        prev_usage_ns, prev_ts = prev
        elapsed = max(1e-6, now - prev_ts)
        if elapsed < _CPU_SAMPLE_MIN_ELAPSED_SEC:
            return 0.0, False, elapsed
        _cgroup_cpu_prev[prev_key] = (usage_ns, now)

    delta_ns = usage_ns - prev_usage_ns
    if delta_ns <= 0:
        return 0.0, False, elapsed
    cpu_percent = max(0.0, delta_ns / elapsed / 1e9 * 100.0)
    if cpu_percent > _CPU_SAMPLE_MAX_PERCENT:
        logger.debug(
            "Ignoring implausible container CPU sample container=%s source=%s cpu=%.2f%% elapsed=%.3fs",
            container_id[:12],
            source,
            cpu_percent,
            elapsed,
        )
        return 0.0, False, elapsed
    return cpu_percent, True, elapsed


def _read_cgroup_io_bytes(path: Path) -> tuple[int, int]:
    read_bytes = 0
    write_bytes = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        if path.name == "io.stat":
            for item in parts[1:]:
                key, sep, value = item.partition("=")
                if not sep:
                    continue
                if key == "rbytes":
                    read_bytes += int(value)
                elif key == "wbytes":
                    write_bytes += int(value)
        elif len(parts) >= 3:
            op = parts[-2].lower()
            value = int(parts[-1])
            if op == "read":
                read_bytes += value
            elif op == "write":
                write_bytes += value
    return read_bytes, write_bytes


def _cgroup_stats_for_container(container_id: str, *, include_raw: bool = False) -> dict[str, Any]:
    container_id = str(container_id).strip()
    now = time.time()
    memory_path = _find_cgroup_file(
        container_id,
        cache_kind="memory",
        roots=_CGROUP_MEMORY_ROOTS,
        names=["memory.current", "memory.usage_in_bytes"],
    )
    if memory_path is None:
        return {
            "ok": False,
            "error": "cgroup memory file not found",
            "container_id": container_id,
            "ts": now,
            "stats_backend": "cgroup",
        }

    cpu_path = _find_related_cgroup_file(
        memory_path,
        anchor_roots=_CGROUP_MEMORY_ROOTS,
        target_roots=_CGROUP_CPU_ROOTS,
        names=["cpuacct.usage", "cpu.stat"],
    )
    if cpu_path is None:
        cpu_path = _find_cgroup_file(
            container_id,
            cache_kind="cpu",
            roots=_CGROUP_CPU_ROOTS,
            names=["cpuacct.usage", "cpu.stat"],
        )
    io_path = _find_related_cgroup_file(
        memory_path,
        anchor_roots=_CGROUP_MEMORY_ROOTS,
        target_roots=_CGROUP_IO_ROOTS,
        names=["io.stat", "blkio.throttle.io_service_bytes_recursive", "blkio.throttle.io_service_bytes"],
    )
    if io_path is None:
        io_path = _find_cgroup_file(
            container_id,
            cache_kind="io",
            roots=_CGROUP_IO_ROOTS,
            names=["io.stat", "blkio.throttle.io_service_bytes_recursive", "blkio.throttle.io_service_bytes"],
        )

    memory_bytes = _read_int_file(memory_path)
    cpu_percent = 0.0
    cpu_sample_valid = False
    cpu_sample_elapsed_sec = 0.0
    cpu_source = ""
    if cpu_path is not None:
        try:
            usage_ns = _read_cgroup_cpu_usage_ns(cpu_path)
            cpu_source = f"cgroup:{cpu_path.name}"
            cpu_percent, cpu_sample_valid, cpu_sample_elapsed_sec = _cpu_percent_from_usage(
                container_id,
                source=cpu_source,
                usage_ns=usage_ns,
                now=now,
            )
        except Exception:
            cpu_source = ""

    if not cpu_sample_valid:
        proc_usage_ns = _read_proc_cpu_usage_ns(memory_path.parent)
        if proc_usage_ns is not None:
            proc_cpu_percent, proc_valid, proc_elapsed = _cpu_percent_from_usage(
                container_id,
                source="procfs",
                usage_ns=proc_usage_ns,
                now=now,
            )
            if proc_valid or not cpu_source:
                cpu_percent = proc_cpu_percent
                cpu_sample_valid = proc_valid
                cpu_sample_elapsed_sec = proc_elapsed
                cpu_source = "procfs"

    disk_read_bytes = 0
    disk_write_bytes = 0
    if io_path is not None:
        disk_read_bytes, disk_write_bytes = _read_cgroup_io_bytes(io_path)

    out = {
        "ok": True,
        "container_id": container_id,
        "memory_usage_bytes": int(memory_bytes),
        "cpu_percent": float(cpu_percent),
        "avg_cpu_percent": float(cpu_percent) if cpu_sample_valid else 0.0,
        "cpu_sample_valid": bool(cpu_sample_valid),
        "cpu_sample_elapsed_sec": float(cpu_sample_elapsed_sec) if cpu_sample_valid else 0.0,
        "cpu_source": cpu_source,
        "disk_read_bytes": int(disk_read_bytes),
        "disk_write_bytes": int(disk_write_bytes),
        "ts": now,
        "stats_backend": "cgroup",
    }
    if include_raw:
        out["raw"] = {
            "memory_path": str(memory_path),
            "cpu_path": str(cpu_path) if cpu_path is not None else "",
            "cpu_source": cpu_source,
            "cpu_sample_valid": bool(cpu_sample_valid),
            "cpu_sample_elapsed_sec": float(cpu_sample_elapsed_sec) if cpu_sample_valid else 0.0,
            "io_path": str(io_path) if io_path is not None else "",
        }
    return out


def _cgroup_stats_for_containers(
    container_ids: list[str],
    *,
    include_raw: bool = False,
) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for container_id in [str(cid).strip() for cid in container_ids if str(cid).strip()]:
        try:
            out[container_id] = _cgroup_stats_for_container(container_id, include_raw=include_raw)
        except Exception as exc:
            out[container_id] = {
                "ok": False,
                "error": str(exc),
                "container_id": container_id,
                "ts": time.time(),
                "stats_backend": "cgroup",
            }
    return out


def _resource_stats_for_containers(
    container_ids: list[str],
    *,
    timeout: int,
    include_raw: bool = False,
) -> dict[str, dict[str, Any]]:
    backend = str(_SERVER_CONFIG.stats_backend or "cgroup").strip().lower()
    if backend == "docker":
        return _docker_stats_for_containers(container_ids, timeout=timeout, include_raw=include_raw)

    cgroup_stats = _cgroup_stats_for_containers(container_ids, include_raw=include_raw)
    if backend != "auto":
        return cgroup_stats

    missing = [cid for cid, item in cgroup_stats.items() if not item.get("ok", False)]
    if not missing:
        return cgroup_stats
    try:
        docker_stats = _docker_stats_for_containers(missing, timeout=timeout, include_raw=include_raw)
        cgroup_stats.update(docker_stats)
    except Exception:
        pass
    return cgroup_stats


def _get_cached_container_stats(container_id: str, *, max_age_sec: float) -> dict[str, Any] | None:
    now = time.time()
    with _stats_cache_lock:
        cached = _stats_cache.get(container_id)
        if not isinstance(cached, dict):
            return None
        ts = float(cached.get("ts", 0.0) or 0.0)
        if now - ts > max(0.0, float(max_age_sec)):
            return None
        return dict(cached)


def _peak_number(existing: dict[str, Any], fresh: dict[str, Any], peak_key: str, value_key: str) -> float:
    values: list[float] = []
    for payload in (existing, fresh):
        for key in (peak_key, value_key):
            try:
                values.append(float(payload.get(key, 0.0) or 0.0))
            except (TypeError, ValueError):
                pass
    return max(values) if values else 0.0


def _merge_cached_container_stats(existing: dict[str, Any] | None, fresh: dict[str, Any]) -> dict[str, Any]:
    merged = dict(fresh)
    fresh_cpu_valid = bool(fresh.get("cpu_sample_valid", False))
    fresh_cpu_percent = float(fresh.get("cpu_percent", 0.0) or 0.0)
    fresh_cpu_elapsed = float(fresh.get("cpu_sample_elapsed_sec", 0.0) or 0.0)
    if fresh_cpu_valid and fresh_cpu_elapsed <= 0.0:
        fresh_cpu_elapsed = 1.0

    if not isinstance(existing, dict) or not existing.get("ok", False):
        merged.setdefault("peak_memory_usage_bytes", merged.get("memory_usage_bytes", 0))
        merged.setdefault("peak_cpu_percent", fresh_cpu_percent if fresh_cpu_valid else 0.0)
        merged["avg_cpu_percent"] = fresh_cpu_percent if fresh_cpu_valid else 0.0
        merged["cpu_percent"] = merged["avg_cpu_percent"]
        merged.setdefault("peak_disk_read_bytes", merged.get("disk_read_bytes", 0))
        merged.setdefault("peak_disk_write_bytes", merged.get("disk_write_bytes", 0))
        merged.setdefault("cache_sample_count", 1)
        merged.setdefault("cpu_sample_count", 1 if fresh_cpu_valid else 0)
        merged["cpu_sample_elapsed_total_sec"] = fresh_cpu_elapsed if fresh_cpu_valid else 0.0
        merged["cpu_percent_weighted_sum"] = (
            fresh_cpu_percent * fresh_cpu_elapsed if fresh_cpu_valid else 0.0
        )
        merged["cpu_sample_valid"] = fresh_cpu_valid
        merged.setdefault("cache_window_start_ts", merged.get("ts", time.time()))
        return merged

    existing_cpu_count = int(existing.get("cpu_sample_count", 0) or 0)
    if existing_cpu_count <= 0 and bool(existing.get("cpu_sample_valid", False)):
        existing_cpu_count = 1
    cpu_sample_count = existing_cpu_count + (1 if fresh_cpu_valid else 0)
    existing_cpu_elapsed = float(existing.get("cpu_sample_elapsed_total_sec", 0.0) or 0.0)
    if existing_cpu_elapsed <= 0.0 and existing_cpu_count > 0:
        existing_cpu_elapsed = float(existing_cpu_count)
    existing_avg_cpu = float(
        existing.get("avg_cpu_percent", existing.get("cpu_percent", 0.0)) or 0.0
    )
    existing_weighted_sum = float(
        existing.get("cpu_percent_weighted_sum", existing_avg_cpu * existing_cpu_elapsed) or 0.0
    )
    fresh_weighted_sum = fresh_cpu_percent * fresh_cpu_elapsed if fresh_cpu_valid else 0.0
    cpu_elapsed_total = existing_cpu_elapsed + (fresh_cpu_elapsed if fresh_cpu_valid else 0.0)
    cpu_weighted_sum = existing_weighted_sum + fresh_weighted_sum
    avg_cpu_percent = cpu_weighted_sum / cpu_elapsed_total if cpu_elapsed_total > 0.0 else 0.0

    merged["peak_memory_usage_bytes"] = int(
        _peak_number(existing, fresh, "peak_memory_usage_bytes", "memory_usage_bytes")
    )
    if fresh_cpu_valid:
        merged["peak_cpu_percent"] = _peak_number(existing, fresh, "peak_cpu_percent", "cpu_percent")
    else:
        try:
            merged["peak_cpu_percent"] = float(existing.get("peak_cpu_percent", 0.0) or 0.0)
        except (TypeError, ValueError):
            merged["peak_cpu_percent"] = 0.0
    merged["avg_cpu_percent"] = avg_cpu_percent
    merged["cpu_percent"] = avg_cpu_percent
    merged["peak_disk_read_bytes"] = int(
        _peak_number(existing, fresh, "peak_disk_read_bytes", "disk_read_bytes")
    )
    merged["peak_disk_write_bytes"] = int(
        _peak_number(existing, fresh, "peak_disk_write_bytes", "disk_write_bytes")
    )
    merged["cache_sample_count"] = int(existing.get("cache_sample_count", 1) or 1) + 1
    merged["cpu_sample_count"] = cpu_sample_count
    merged["cpu_sample_elapsed_total_sec"] = cpu_elapsed_total
    merged["cpu_percent_weighted_sum"] = cpu_weighted_sum
    merged["cpu_sample_valid"] = cpu_sample_count > 0
    if not fresh_cpu_valid and existing.get("cpu_source"):
        merged["cpu_source"] = existing.get("cpu_source")
    merged["cache_window_start_ts"] = min(
        float(existing.get("cache_window_start_ts", existing.get("ts", time.time())) or time.time()),
        float(fresh.get("ts", time.time()) or time.time()),
    )
    return merged


def _put_cached_container_stats(stats: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged_stats: dict[str, dict[str, Any]] = {}
    if not stats:
        return merged_stats
    with _stats_cache_lock:
        for container_id, payload in stats.items():
            if isinstance(payload, dict) and payload.get("ok", False):
                merged = _merge_cached_container_stats(
                    _stats_cache.get(container_id),
                    payload,
                )
                _stats_cache[container_id] = merged
                merged_stats[container_id] = dict(merged)
    return merged_stats


def _stats_sampler_loop() -> None:
    interval = max(0.2, float(_SERVER_CONFIG.stats_sampler_interval_sec))
    timeout = max(1, int(_SERVER_CONFIG.stats_command_timeout_sec))
    batch_size = max(1, int(_SERVER_CONFIG.stats_batch_max_containers))
    while not _stats_sampler_stop.is_set():
        with _lock:
            active_ids = list(_active_containers.keys())
        active_set = set(active_ids)
        with _stats_cache_lock:
            for cached_id in list(_stats_cache.keys()):
                if cached_id not in active_set:
                    _stats_cache.pop(cached_id, None)
        for idx in range(0, len(active_ids), batch_size):
            if _stats_sampler_stop.is_set():
                break
            chunk = active_ids[idx : idx + batch_size]
            try:
                _put_cached_container_stats(
                    _resource_stats_for_containers(chunk, timeout=timeout, include_raw=False)
                )
            except Exception as exc:
                logger.debug("background docker stats sample failed: %s", exc)
        _stats_sampler_stop.wait(interval)


def _ensure_stats_sampler_started() -> None:
    global _stats_sampler_thread
    if not _SERVER_CONFIG.stats_sampler_enabled:
        return
    with _stats_sampler_lock:
        if _stats_sampler_thread is not None and _stats_sampler_thread.is_alive():
            return
        _stats_sampler_stop.clear()
        _stats_sampler_thread = threading.Thread(
            target=_stats_sampler_loop,
            name="swe-exec-stats-sampler",
            daemon=True,
        )
        _stats_sampler_thread.start()


def _action_stats_sampler_loop() -> None:
    while True:
        interval = max(0.05, float(_SERVER_CONFIG.action_stats_interval_sec))
        timeout = max(1, int(_SERVER_CONFIG.action_stats_command_timeout_sec))
        batch_size = max(1, int(_SERVER_CONFIG.action_stats_batch_max_containers))
        with _action_stats_sampler_lock:
            active_ids = [
                container_id
                for container_id, ref_count in _action_stats_containers.items()
                if ref_count > 0
            ]

        if not active_ids:
            _action_stats_wakeup.wait(interval)
            _action_stats_wakeup.clear()
            continue

        for idx in range(0, len(active_ids), batch_size):
            chunk = active_ids[idx : idx + batch_size]
            try:
                _put_cached_container_stats(
                    _resource_stats_for_containers(chunk, timeout=timeout, include_raw=False)
                )
            except Exception as exc:
                logger.debug("action docker stats sample failed: %s", exc)

        _action_stats_wakeup.wait(interval)
        _action_stats_wakeup.clear()


def _ensure_action_stats_sampler_started() -> None:
    global _action_stats_sampler_thread
    if not _SERVER_CONFIG.action_stats_sampler_enabled:
        return
    with _action_stats_sampler_lock:
        if _action_stats_sampler_thread is not None and _action_stats_sampler_thread.is_alive():
            return
        _action_stats_sampler_thread = threading.Thread(
            target=_action_stats_sampler_loop,
            name="swe-exec-action-stats-sampler",
            daemon=True,
        )
        _action_stats_sampler_thread.start()


@contextlib.contextmanager
def _action_stats_sampling_section(container_id: str) -> Any:
    if not _SERVER_CONFIG.action_stats_sampler_enabled:
        yield
        return

    container_id = str(container_id)
    _ensure_action_stats_sampler_started()
    with _action_stats_sampler_lock:
        _action_stats_containers[container_id] += 1
        _action_stats_wakeup.set()
    try:
        yield
    finally:
        # Capture one final point and merge it with any peaks collected while the
        # command was running. The final point is usually cheap and helps for
        # commands that leave memory resident briefly after exit.
        try:
            _put_cached_container_stats(
                _resource_stats_for_containers(
                    [container_id],
                    timeout=max(1, int(_SERVER_CONFIG.action_stats_command_timeout_sec)),
                    include_raw=False,
                )
            )
        except Exception as exc:
            logger.debug("final action docker stats sample failed for %s: %s", container_id[:12], exc)
        with _action_stats_sampler_lock:
            remaining = max(0, int(_action_stats_containers.get(container_id, 0)) - 1)
            if remaining > 0:
                _action_stats_containers[container_id] = remaining
            else:
                _action_stats_containers.pop(container_id, None)
            _action_stats_wakeup.set()


def _container_stats_cached_or_direct(container_id: str, *, include_raw: bool = False) -> dict[str, Any]:
    _ensure_stats_sampler_started()
    cached = _get_cached_container_stats(
        container_id,
        max_age_sec=max(0.0, float(_SERVER_CONFIG.stats_cache_ttl_sec)),
    )
    if cached is not None:
        if include_raw:
            cached = dict(cached)
        return cached
    stats = _resource_stats_for_containers(
        [container_id],
        timeout=max(1, int(_SERVER_CONFIG.stats_command_timeout_sec)),
        include_raw=include_raw,
    )
    merged = _put_cached_container_stats(stats)
    return merged.get(container_id) or stats.get(container_id, {"ok": False, "error": "empty docker stats output"})


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


def _docker_proxy_env_args() -> list[str]:
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
    return proxy_args


def _docker_container_create_args(
    *, image: str, cwd: str, container_name: str
) -> list[str]:
    return [
        "--name", container_name,
        "--network", "host",
        *_docker_proxy_env_args(),
        "-w", cwd,
        "--pids-limit", "256",
        "--memory", "4g",
        image,
        "sleep", "infinity",
    ]


def _docker_create_container(image: str, cwd: str, timeout: int, *, container_name: str | None = None) -> dict[str, Any]:
    container_name = container_name or f"swe-{uuid.uuid4().hex[:12]}"
    started_at = time.time()
    r = _docker(
        "run", "-d",
        *_docker_container_create_args(image=image, cwd=cwd, container_name=container_name),
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


@dataclass(frozen=True)
class _FullCheckpointApi:
    checkpoint_options: Any
    resume_options: Any
    full_checkpoint: Any
    full_resume: Any


_full_checkpoint_api_lock = threading.Lock()
_full_checkpoint_api_cache: _FullCheckpointApi | None = None
_full_checkpoint_docker_root_lock = threading.Lock()
_full_checkpoint_docker_root_cache: Path | None = None


def _normalize_checkpoint_backend(value: Any, *, default: str = "full") -> str:
    backend = str(value or default).strip().lower().replace("-", "_")
    aliases = {
        "full": "full",
        "full_checkpoint": "full",
        "legacy": "legacy",
        "commit": "legacy",
        "docker_commit": "legacy",
    }
    normalized = aliases.get(backend)
    if normalized is None:
        raise ValueError(
            f"unsupported checkpoint backend {value!r}; expected 'full' or 'legacy'"
        )
    return normalized


def _record_checkpoint_backend(record: dict[str, Any]) -> str:
    # Records created before the full-checkpoint integration have no backend
    # marker and must keep using their docker-commit image.
    return _normalize_checkpoint_backend(record.get("checkpoint_backend"), default="legacy")


def _full_checkpoint_state_root(record: dict[str, Any] | None = None) -> Path:
    recorded = str((record or {}).get("full_checkpoint_state_root", "") or "").strip()
    configured = str(_SERVER_CONFIG.full_checkpoint_state_root or "").strip()
    return Path(recorded or configured or (Path(_SERVER_CONFIG.checkpoint_dir) / "full-checkpoint-state"))


def _full_checkpoint_project_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = str(_SERVER_CONFIG.full_checkpoint_project_root or "").strip()
    if configured:
        candidates.append(Path(configured))
    server_path = Path(__file__).resolve()
    if len(server_path.parents) > 2:
        candidates.append(server_path.parents[2] / "docker-full-checkpoint")
    if len(server_path.parents) > 3:
        candidates.append(server_path.parents[3] / "docker-full-checkpoint")
    candidates.extend(
        [
            Path.cwd() / "docker-full-checkpoint",
            Path("/opt/docker-full-checkpoint"),
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def _load_full_checkpoint_api() -> _FullCheckpointApi:
    global _full_checkpoint_api_cache
    with _full_checkpoint_api_lock:
        if _full_checkpoint_api_cache is not None:
            return _full_checkpoint_api_cache

        import_error: Exception | None = None
        for source_root in [None, *[path / "src" for path in _full_checkpoint_project_candidates()]]:
            if source_root is not None:
                if not (source_root / "full_checkpoint").is_dir():
                    continue
                source_text = str(source_root)
                if source_text not in sys.path:
                    sys.path.insert(0, source_text)
            try:
                checkpoint_module = importlib.import_module("full_checkpoint.checkpoint")
                resume_module = importlib.import_module("full_checkpoint.resume")
                _full_checkpoint_api_cache = _FullCheckpointApi(
                    checkpoint_options=checkpoint_module.CheckpointOptions,
                    resume_options=resume_module.ResumeOptions,
                    full_checkpoint=checkpoint_module.full_checkpoint,
                    full_resume=resume_module.full_resume,
                )
                return _full_checkpoint_api_cache
            except (ImportError, AttributeError) as exc:
                import_error = exc

        searched = ", ".join(str(path) for path in _full_checkpoint_project_candidates())
        raise RuntimeError(
            "docker-full-checkpoint is unavailable; install the package or set "
            f"SWE_FULL_CHECKPOINT_PROJECT_ROOT (searched: {searched})"
        ) from import_error


def _full_checkpoint_docker_root() -> Path:
    global _full_checkpoint_docker_root_cache
    configured = str(_SERVER_CONFIG.full_checkpoint_docker_root or "").strip()
    if configured:
        return Path(configured)
    with _full_checkpoint_docker_root_lock:
        if _full_checkpoint_docker_root_cache is not None:
            return _full_checkpoint_docker_root_cache
        result = _docker("info", "--format", "{{.DockerRootDir}}", timeout=10)
        if result.returncode != 0 or not result.stdout.strip():
            raise RuntimeError(
                result.stderr.strip() or "docker info did not report DockerRootDir"
            )
        _full_checkpoint_docker_root_cache = Path(result.stdout.strip())
        return _full_checkpoint_docker_root_cache


def _full_checkpoint_artifact_dir(checkpoint_id: str, record: dict[str, Any] | None = None) -> Path:
    safe_id = re.sub(r"[^a-zA-Z0-9_.-]+", "-", str(checkpoint_id))
    return _full_checkpoint_state_root(record) / "checkpoints" / safe_id


def _directory_size_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for root, _, files in os.walk(path, followlinks=False):
        for filename in files:
            try:
                total += (Path(root) / filename).lstat().st_size
            except FileNotFoundError:
                continue
    return total


def _remove_installed_docker_checkpoint(
    *, docker_root: Path, container_id: str, checkpoint_id: str
) -> None:
    if not container_id:
        return
    result = _docker(
        "checkpoint", "rm", container_id, checkpoint_id, timeout=60
    )
    if result.returncode == 0:
        return
    installed = docker_root / "containers" / container_id / "checkpoints" / checkpoint_id
    if installed.exists():
        # Fallback for daemon versions that refuse checkpoint rm after start.
        # This server already requires root access for docker-full-checkpoint.
        shutil.rmtree(installed)


def _delete_checkpoint_artifacts(record: dict[str, Any]) -> None:
    if _record_checkpoint_backend(record) == "legacy":
        image_name = str(record.get("checkpoint_image", "") or "")
        if image_name:
            _docker("image", "rm", "-f", image_name, timeout=60)
        return
    artifact_dir = _full_checkpoint_artifact_dir(str(record["checkpoint_id"]), record)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)


def _checkpoint_sort_key(item: dict[str, Any]) -> tuple[int, int, float]:
    raw_step_idx = item.get("step_idx", -1)
    return (
        int(item.get("generation", 0) or 0),
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
                job.fault_injection_spec,
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
    fault_injection_spec: dict[str, Any] | None = None,
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
        fault_injection_spec=dict(fault_injection_spec) if fault_injection_spec else None,
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


@contextlib.contextmanager
def _full_checkpoint_source_guard(
    *,
    container_id: str,
    checkpoint_id: str,
    docker_root: Path,
    recovery_state: dict[str, Any],
) -> Any:
    """Keep the container gate held while recovering a stopped source."""
    with _container_exclusive_section(container_id):
        try:
            yield
        except Exception:
            if _container_is_active(container_id):
                try:
                    if not _docker_container_is_running(container_id, timeout=10):
                        recovery_state["attempted"] = True
                        result = _docker("start", container_id, timeout=120)
                        if result.returncode != 0:
                            raise RuntimeError(
                                result.stderr.strip()
                                or f"docker start failed for {container_id}"
                            )
                        recovery_state["succeeded"] = _docker_container_is_running(
                            container_id, timeout=10
                        )
                        recovery_state["mode"] = "plain_restart"
                except Exception:
                    logger.exception(
                        "Failed to recover source after full checkpoint error "
                        "checkpoint_id=%s container_id=%s",
                        checkpoint_id,
                        container_id,
                    )
                if recovery_state.get("attempted"):
                    try:
                        _remove_installed_docker_checkpoint(
                            docker_root=docker_root,
                            container_id=container_id,
                            checkpoint_id=checkpoint_id,
                        )
                    except Exception:
                        logger.warning(
                            "Failed to clean installed checkpoint after source recovery: %s",
                            checkpoint_id,
                            exc_info=True,
                        )
            raise


def _full_checkpoint_create_worker(
    *,
    op_id: str,
    checkpoint_id: str,
    container_id: str,
    record: dict[str, Any],
    fault_injection_spec: dict[str, Any] | None,
) -> dict[str, Any]:
    """Create a durable CRIU+upperdir checkpoint and transparently resume source."""
    started_at = time.time()
    started_perf = time.perf_counter()
    source_resumed = False
    source_recovery: dict[str, Any] = {
        "attempted": False,
        "succeeded": False,
        "mode": "none",
    }
    fault_spec = dict(fault_injection_spec or {})
    state_root = _full_checkpoint_state_root(record)
    state_root.mkdir(parents=True, exist_ok=True)
    docker_root = _full_checkpoint_docker_root()
    runtime_staging = str(_SERVER_CONFIG.full_checkpoint_runtime_staging_root or "").strip()
    api = _load_full_checkpoint_api()
    manifest: dict[str, Any] = {}
    resume_result: dict[str, Any] = {}
    full_checkpoint_sec = 0.0
    source_resume_sec = 0.0
    exclusive_wait_started = time.perf_counter()

    try:
        with _full_checkpoint_source_guard(
            container_id=container_id,
            checkpoint_id=checkpoint_id,
            docker_root=docker_root,
            recovery_state=source_recovery,
        ):
            exclusive_wait_sec = time.perf_counter() - exclusive_wait_started
            if not _container_is_active(container_id) or not _docker_container_is_running(container_id, timeout=10):
                raise RuntimeError(
                    f"container no longer active before full checkpoint: {container_id}"
                )

            if fault_spec.get("phase") == "before_commit":
                fault_delay_sec = float(fault_spec.get("delay_sec", 0.0) or 0.0)
                if fault_delay_sec > 0.0:
                    time.sleep(fault_delay_sec)
                fault_event = _inject_fail_stop_fault(
                    container_id,
                    fault_type="checkpoint_explicit_before_full_checkpoint_kill",
                    fault_phase="before_commit",
                    delay_sec=fault_delay_sec,
                    tag=str(fault_spec.get("tag", "") or ""),
                    remove_tracking=True,
                    drop_gate=True,
                )
                _CHECKPOINTS.update_checkpoint(checkpoint_id, **fault_event)
                raise RuntimeError("injected checkpoint fault before full checkpoint")

            with _commit_docker_section():
                checkpoint_started = time.perf_counter()
                manifest = api.full_checkpoint(
                    container_id,
                    options=api.checkpoint_options(
                        checkpoint_id=checkpoint_id,
                        state_root=state_root,
                        docker_root=docker_root,
                        require_criu=True,
                        leave_running=False,
                        criu_timeout_sec=max(
                            1, int(_SERVER_CONFIG.full_checkpoint_criu_timeout_sec)
                        ),
                        docker_managed=True,
                        runtime_staging_root=Path(runtime_staging) if runtime_staging else None,
                    ),
                )
                full_checkpoint_sec = time.perf_counter() - checkpoint_started
                if fault_spec.get("phase") == "after_commit_before_ready":
                    fault_delay_sec = float(fault_spec.get("delay_sec", 0.0) or 0.0)
                    if fault_delay_sec > 0.0:
                        time.sleep(fault_delay_sec)
                    fault_event = _inject_fail_stop_fault(
                        container_id,
                        fault_type="checkpoint_explicit_after_full_checkpoint_before_ready_kill",
                        fault_phase="after_commit_before_ready",
                        delay_sec=fault_delay_sec,
                        tag=str(fault_spec.get("tag", "") or ""),
                        remove_tracking=True,
                        drop_gate=True,
                    )
                    _CHECKPOINTS.update_checkpoint(checkpoint_id, **fault_event)
                    raise RuntimeError(
                        "injected checkpoint fault after full checkpoint before ready"
                    )

                # Docker-managed checkpoint stops the source. Restore into the
                # same Docker container before releasing the exclusive gate so
                # rollout observes one coherent pause and keeps the same ID.
                resume_started = time.perf_counter()
                resume_result = api.full_resume(
                    checkpoint_id,
                    options=api.resume_options(
                        state_root=state_root,
                        container_id=container_id,
                        keep_failed=True,
                        criu_timeout_sec=max(
                            1, int(_SERVER_CONFIG.full_checkpoint_criu_timeout_sec)
                        ),
                        docker_managed=True,
                        docker_root=docker_root,
                    ),
                )
                source_resume_sec = time.perf_counter() - resume_started
                source_docker_id = str(resume_result.get("docker_container_id", "") or "")
                if not bool(resume_result.get("docker_exec_supported", False)):
                    raise RuntimeError("full resume did not preserve Docker exec support")
                if source_docker_id and not (
                    source_docker_id.startswith(container_id) or container_id.startswith(source_docker_id)
                ):
                    raise RuntimeError(
                        "in-place full resume returned a different Docker container: "
                        f"expected={container_id} actual={source_docker_id}"
                    )
                if not _docker_container_is_running(container_id, timeout=10):
                    raise RuntimeError(
                        f"source container is not running after in-place full resume: {container_id}"
                    )
                source_resumed = True
                try:
                    _remove_installed_docker_checkpoint(
                        docker_root=docker_root,
                        container_id=source_docker_id or container_id,
                        checkpoint_id=checkpoint_id,
                    )
                except Exception:
                    logger.warning(
                        "Full checkpoint is ready but installed-copy cleanup failed: %s",
                        checkpoint_id,
                        exc_info=True,
                    )

        ready_at = time.time()
        artifact_dir = _full_checkpoint_artifact_dir(checkpoint_id, record)
        artifact_size = _directory_size_bytes(artifact_dir)
        total_sec = time.perf_counter() - started_perf
        updated, _ = _CHECKPOINTS.update_checkpoint_with_stats(
            checkpoint_id,
            checkpoint_backend="full",
            checkpoint_image=None,
            status="ready",
            ready_at=ready_at,
            failed_at=None,
            error=None,
            full_checkpoint_state_root=str(state_root),
            full_checkpoint_artifact_dir=str(artifact_dir),
            full_checkpoint_manifest=str(artifact_dir / "manifest.json"),
            full_checkpoint_timings_sec=dict(manifest.get("timings_sec", {}) or {}),
            full_checkpoint_create_sec=full_checkpoint_sec,
            full_checkpoint_source_resume_sec=source_resume_sec,
            raw_create_latency_sec=ready_at - started_at,
            ready_latency_sec=ready_at - started_at,
            ready_delay_sec=0.0,
            size_bytes=artifact_size,
            source_resumed=True,
            container_usable=True,
            state_continuity=True,
            source_recovery_mode="full_resume",
        )
        _CHECKPOINTS.update_op(
            op_id,
            status="ready",
            finished_at=ready_at,
            error=None,
        )
        updated.update(
            {
                "build_payload_sec": 0.0,
                "runtime_probe_sec": 0.0,
                "runtime_payload_build_sec": 0.0,
                "runtime_state_write_sec": 0.0,
                "post_commit_finalize_sec": max(
                    0.0, total_sec - full_checkpoint_sec - source_resume_sec
                ),
                "full_checkpoint_total_sec": total_sec,
                "exclusive_wait_sec": exclusive_wait_sec,
            }
        )
        logger.info(
            "full checkpoint ready checkpoint_id=%s container_id=%s "
            "checkpoint_sec=%.3f source_resume_sec=%.3f total_sec=%.3f size_bytes=%s",
            checkpoint_id,
            container_id,
            full_checkpoint_sec,
            source_resume_sec,
            total_sec,
            artifact_size,
        )
        return updated
    except Exception:
        try:
            _CHECKPOINTS.update_checkpoint(
                checkpoint_id,
                source_resumed=source_resumed,
                source_recovery_attempted=source_recovery["attempted"],
                source_recovery_succeeded=source_recovery["succeeded"],
                source_recovery_mode=source_recovery["mode"],
                state_continuity=not source_recovery["attempted"],
                container_usable=(
                    not source_recovery["attempted"]
                    and _container_is_active(container_id)
                    and _docker_container_is_running(container_id, timeout=10)
                ),
                full_checkpoint_create_sec=full_checkpoint_sec,
                full_checkpoint_source_resume_sec=source_resume_sec,
            )
        except Exception:
            logger.exception("Failed to persist full checkpoint recovery state: %s", checkpoint_id)
        raise


def _checkpoint_create_worker(
    op_id: str,
    checkpoint_id: str,
    container_id: str,
    checkpoint_image: str,
    record: dict[str, Any],
    runtime_env: dict[str, str] | None = None,
    fault_injection_spec: dict[str, Any] | None = None,
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
    fault_spec = dict(fault_injection_spec or {})
    runtime_env = dict(runtime_env or {})
    record = dict(record or {})
    if not record.get("checkpoint_id"):
        record = dict(_CHECKPOINTS.get_checkpoint(checkpoint_id) or record)
    try:
        if _record_checkpoint_backend(record) == "full":
            return _full_checkpoint_create_worker(
                op_id=op_id,
                checkpoint_id=checkpoint_id,
                container_id=container_id,
                record=record,
                fault_injection_spec=fault_injection_spec,
            )
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
            if fault_spec.get("phase") == "before_commit":
                fault_delay_sec = float(fault_spec.get("delay_sec", 0.0) or 0.0)
                if fault_delay_sec > 0.0:
                    time.sleep(fault_delay_sec)
                fault_event = _inject_fail_stop_fault(
                    container_id,
                    fault_type="checkpoint_explicit_before_commit_kill",
                    fault_phase="before_commit",
                    delay_sec=fault_delay_sec,
                    tag=str(fault_spec.get("tag", "") or ""),
                    remove_tracking=True,
                    drop_gate=True,
                )
                finished_at = time.time()
                error = "injected checkpoint fault before docker commit"
                _, checkpoint_stats = _CHECKPOINTS.update_checkpoint_with_stats(
                    checkpoint_id,
                    status="failed",
                    failed_at=finished_at,
                    error=error,
                    **fault_event,
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
                final_record["runtime_probe_sec"] = runtime_probe_sec
                final_record["runtime_payload_build_sec"] = runtime_payload_build_sec
                final_record["runtime_state_write_sec"] = runtime_state_write_sec
                return final_record
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

        if fault_spec.get("phase") == "after_commit_before_ready":
            fault_delay_sec = float(fault_spec.get("delay_sec", 0.0) or 0.0)
            if fault_delay_sec > 0.0:
                time.sleep(fault_delay_sec)
            fault_event = _inject_fail_stop_fault(
                container_id,
                fault_type="checkpoint_explicit_after_commit_before_ready_kill",
                fault_phase="after_commit_before_ready",
                delay_sec=fault_delay_sec,
                tag=str(fault_spec.get("tag", "") or ""),
                remove_tracking=True,
                drop_gate=True,
            )
            error = "injected checkpoint fault after docker commit before ready pointer update"
            _, checkpoint_stats = _CHECKPOINTS.update_checkpoint_with_stats(
                checkpoint_id,
                status="failed",
                failed_at=finished_at,
                error=error,
                **fault_event,
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
            final_record["runtime_probe_sec"] = runtime_probe_sec
            final_record["runtime_payload_build_sec"] = runtime_payload_build_sec
            final_record["runtime_state_write_sec"] = runtime_state_write_sec
            return final_record

        raw_create_latency_sec = finished_at - started_at
        image_size_started_perf = time.perf_counter()
        size_bytes = _docker_image_size_bytes(checkpoint_image)
        image_size_sec = time.perf_counter() - image_size_started_perf
        ready_delay_sec = max(
            0.0,
            float(_SERVER_CONFIG.checkpoint_min_ready_latency_sec)
            - raw_create_latency_sec,
        )
        if ready_delay_sec > 0.0:
            time.sleep(ready_delay_sec)
            ready_at = time.time()
        else:
            ready_at = finished_at
        post_commit_finalize_started_perf = time.perf_counter()
        _, checkpoint_stats = _CHECKPOINTS.update_checkpoint_with_stats(
            checkpoint_id,
            status="ready",
            ready_at=ready_at,
            size_bytes=size_bytes,
            error=None,
            raw_create_latency_sec=raw_create_latency_sec,
            ready_latency_sec=ready_at - started_at,
            ready_delay_sec=ready_delay_sec,
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
            size_bytes,
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
                _delete_checkpoint_artifacts(item)
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
    checkpoint_health: dict[str, Any] = {
        "enabled": _SERVER_CONFIG.checkpoint_enabled,
        "backend": _normalize_checkpoint_backend(
            _SERVER_CONFIG.checkpoint_backend, default="full"
        ),
    }
    if checkpoint_health["backend"] == "full":
        try:
            _load_full_checkpoint_api()
            docker_root = _full_checkpoint_docker_root()
            checkpoint_health.update(
                {
                    "api_available": True,
                    "docker_root": str(docker_root),
                    "docker_root_readable": os.access(
                        docker_root / "containers", os.R_OK | os.X_OK
                    ),
                    "state_root": str(_full_checkpoint_state_root()),
                }
            )
        except Exception as exc:
            checkpoint_health.update({"api_available": False, "error": str(exc)})
    return jsonify({
        "ok": True,
        "running_containers": running,
        "checkpoint": checkpoint_health,
    })


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
    try:
        fault_injection_spec = _normalize_fault_injection_spec(
            data.get("fault_injection_spec"),
            allowed_phases=_EXEC_FAULT_PHASES,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc), "error_code": "invalid_fault_injection_spec"}), 400

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

    if fault_injection_spec is not None and fault_injection_spec.get("phase") == "before_action":
        event = _inject_fail_stop_fault(
            str(container_id),
            fault_type="exec_server_explicit_before_action_kill",
            fault_phase="before_action",
            delay_sec=float(fault_injection_spec.get("delay_sec", 0.0) or 0.0),
            tag=str(fault_injection_spec.get("tag", "") or ""),
            remove_tracking=True,
            drop_gate=True,
        )
        return jsonify(
            {
                "ok": True,
                "returncode": -1,
                "output": "Injected environment fault: container killed before exec.",
                **event,
            }
        )

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

    mid_action_cancel_event: threading.Event | None = None
    mid_action_thread: threading.Thread | None = None
    mid_action_state: dict[str, Any] = {"event": None}
    if fault_injection_spec is not None and fault_injection_spec.get("phase") == "mid_action":
        fault_delay_sec = float(fault_injection_spec.get("delay_sec", 0.0) or 0.0)
        active_info_snapshot = dict(active_info) if active_info else None
        mid_action_cancel_event = threading.Event()

        def _mid_action_fault_worker() -> None:
            if mid_action_cancel_event is not None and mid_action_cancel_event.wait(timeout=fault_delay_sec):
                return
            mid_action_state["event"] = _inject_fail_stop_fault(
                str(container_id),
                fault_type="exec_server_explicit_mid_action_kill",
                fault_phase="mid_action",
                delay_sec=fault_delay_sec,
                tag=str(fault_injection_spec.get("tag", "") or ""),
                remove_tracking=False,
                drop_gate=False,
            )

        mid_action_thread = threading.Thread(
            target=_mid_action_fault_worker,
            name=f"exec-mid-action-fault-{str(container_id)[:12]}",
            daemon=True,
        )
        mid_action_thread.start()

    try:
        with _container_exec_section(container_id):
            with _action_stats_sampling_section(str(container_id)):
                with _foreground_docker_section():
                    r = _docker(
                        "exec", "-w", cwd, *env_args, container_id,
                        "bash", "-lc", command,
                        timeout=timeout,
                    )
        if mid_action_cancel_event is not None:
            mid_action_cancel_event.set()
        if mid_action_thread is not None:
            mid_action_thread.join(timeout=0.1)
        if mid_action_state["event"] is not None:
            with _lock:
                _active_containers.pop(str(container_id), None)
            _drop_container_op_gate(str(container_id))
            return jsonify(
                {
                    "ok": True,
                    "returncode": -1,
                    "output": "Injected environment fault: container killed during exec.",
                    **mid_action_state["event"],
                }
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
        if mid_action_cancel_event is not None:
            mid_action_cancel_event.set()
        if mid_action_thread is not None:
            mid_action_thread.join(timeout=0.1)
        if mid_action_state["event"] is not None:
            with _lock:
                _active_containers.pop(str(container_id), None)
            _drop_container_op_gate(str(container_id))
            return jsonify(
                {
                    "ok": True,
                    "returncode": -1,
                    "output": "Injected environment fault: container killed during exec.",
                    **mid_action_state["event"],
                }
            )
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
        with _action_stats_sampling_section(str(container_id)):
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


@app.post("/container/fault/kill")
def container_fault_kill():
    """Inject a fail-stop fault into a live container without closing its lease."""
    data = request.get_json(force=True) or {}
    container_id = data.get("container_id")
    if not container_id:
        return jsonify({"ok": False, "error": "container_id is required"}), 400
    delay_sec = float(data.get("delay_sec", 0.0) or 0.0)
    tag = str(data.get("tag", "") or "")
    event = _inject_fail_stop_fault(
        str(container_id),
        fault_type="exec_server_random_wall_clock_kill",
        fault_phase="random_wall_clock",
        delay_sec=delay_sec,
        tag=tag,
        remove_tracking=False,
        drop_gate=False,
    )
    return jsonify({"ok": True, **event})


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
    parent_checkpoint_id = str(data.get("parent_checkpoint_id") or "") or None
    try:
        checkpoint_backend = _normalize_checkpoint_backend(
            data.get("checkpoint_backend"),
            default=_SERVER_CONFIG.checkpoint_backend,
        )
    except ValueError as exc:
        return jsonify({
            "ok": False,
            "error": str(exc),
            "error_code": "invalid_checkpoint_backend",
        }), 400
    try:
        fault_injection_spec = _normalize_fault_injection_spec(
            data.get("fault_injection_spec"),
            allowed_phases=_CHECKPOINT_FAULT_PHASES,
        )
    except ValueError as exc:
        return jsonify({"ok": False, "error": str(exc), "error_code": "invalid_fault_injection_spec"}), 400
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
    if parent_checkpoint_id:
        parent = _CHECKPOINTS.get_checkpoint(parent_checkpoint_id)
        if parent is None:
            return jsonify({
                "ok": False,
                "error": f"unknown parent checkpoint: {parent_checkpoint_id}",
                "error_code": "checkpoint_parent_not_found",
                "retryable": False,
            }), 404
        if parent.get("status") != "ready" or str(parent.get("lease_id", "")) != lease_id:
            return jsonify({
                "ok": False,
                "error": "parent checkpoint must be ready and belong to the same lease",
                "error_code": "checkpoint_parent_invalid",
                "retryable": False,
            }), 409
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
            parent_checkpoint_id=parent_checkpoint_id,
            checkpoint_backend=checkpoint_backend,
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
        record = _CHECKPOINTS.update_checkpoint(
            record["checkpoint_id"],
            runtime_env=runtime_env,
            full_checkpoint_state_root=(
                str(_full_checkpoint_state_root(record)) if checkpoint_backend == "full" else None
            ),
        )
        worker_started_perf = time.perf_counter()
        final_record, dispatcher_queue_wait_sec, dispatcher_worker_exec_sec = _dispatch_checkpoint_create_and_wait(
            op_id=op["op_id"],
            checkpoint_id=record["checkpoint_id"],
            container_id=container_id,
            checkpoint_image=str(record.get("checkpoint_image") or ""),
            record=record,
            runtime_env=runtime_env,
            fault_injection_spec=fault_injection_spec,
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
            "fault_injected": bool(final_record.get("fault_injected", False)),
            "fault_type": final_record.get("fault_type"),
            "fault_phase": final_record.get("fault_phase"),
            "checkpoint_image": final_record.get("checkpoint_image"),
            "checkpoint_backend": final_record.get("checkpoint_backend"),
            "container_usable": final_record.get("container_usable"),
            "state_continuity": final_record.get("state_continuity"),
            "source_recovery_mode": final_record.get("source_recovery_mode"),
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
        "checkpoint_backend": final_record.get("checkpoint_backend", "legacy"),
        "checkpoint_image": final_record.get("checkpoint_image"),
        "container_id": final_record.get("container_id"),
        "generation": final_record.get("generation", 0),
        "step_idx": final_record["step_idx"],
        "ready_at": final_record["ready_at"],
        "size_bytes": final_record["size_bytes"],
        "full_checkpoint_timings_sec": final_record.get("full_checkpoint_timings_sec"),
        "full_checkpoint_create_sec": final_record.get("full_checkpoint_create_sec"),
        "full_checkpoint_source_resume_sec": final_record.get("full_checkpoint_source_resume_sec"),
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
    reclaimed = int(record.get("size_bytes") or 0)
    with _maintenance_docker_section():
        _delete_checkpoint_artifacts(record)
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


def _full_checkpoint_rerun(
    *,
    record: dict[str, Any],
    old_container_id: str | None,
    cwd: str,
    timeout: int,
) -> dict[str, Any]:
    checkpoint_id = str(record["checkpoint_id"])
    image = str(record.get("image", "") or "")
    if not image:
        raise RuntimeError(f"full checkpoint record has no source image: {checkpoint_id}")
    target_name = re.sub(
        r"[^a-zA-Z0-9_.-]+",
        "-",
        f"swe-resume-{checkpoint_id[-12:]}-{uuid.uuid4().hex[:8]}",
    )
    state_root = _full_checkpoint_state_root(record)
    docker_root = _full_checkpoint_docker_root()
    api = _load_full_checkpoint_api()
    new_container_id = ""
    resume_sec = 0.0
    runtime_env = _normalize_runtime_env(
        record.get("runtime_env", {}) if isinstance(record.get("runtime_env"), dict) else {}
    )
    runtime_state = {
        "schema_version": _RUNTIME_STATE_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "workspace": {"cwd": cwd, "repo_path": cwd},
        "env": runtime_env,
        "python_runtime": _derive_python_runtime(runtime_env),
    }

    try:
        with _foreground_docker_section():
            # docker-full-checkpoint clones the source image, command, env,
            # workdir, network mode, memory and pids limits before restore.
            resume_started = time.perf_counter()
            resumed = api.full_resume(
                checkpoint_id,
                options=api.resume_options(
                    state_root=state_root,
                    container_id=target_name,
                    keep_failed=True,
                    criu_timeout_sec=max(
                        1, int(_SERVER_CONFIG.full_checkpoint_criu_timeout_sec)
                    ),
                    docker_managed=True,
                    docker_root=docker_root,
                ),
            )
            resume_sec = time.perf_counter() - resume_started
            new_container_id = str(resumed.get("docker_container_id", "") or "")
            if not new_container_id:
                raise RuntimeError("full resume did not return docker_container_id")
            if not bool(resumed.get("docker_exec_supported", False)):
                raise RuntimeError("full resume target does not support docker exec")
            if not _docker_container_is_running(new_container_id, timeout=10):
                raise RuntimeError(
                    f"full resume target is not running: {new_container_id}"
                )
            _validate_runtime_restore(new_container_id, runtime_state)
            try:
                _remove_installed_docker_checkpoint(
                    docker_root=docker_root,
                    container_id=new_container_id,
                    checkpoint_id=checkpoint_id,
                )
            except Exception:
                logger.warning(
                    "Restored container is usable but installed-copy cleanup failed: %s",
                    checkpoint_id,
                    exc_info=True,
                )

        old_info: dict[str, Any] | None = None
        if old_container_id:
            with _lock:
                snapshot = _active_containers.get(old_container_id)
                if isinstance(snapshot, dict):
                    old_info = dict(snapshot)
            with _container_exclusive_section(old_container_id):
                with _foreground_docker_section():
                    _docker_destroy_container(old_container_id, timeout=300)
            _drop_container_op_gate(old_container_id)

        with _lock:
            if old_container_id:
                _active_containers.pop(old_container_id, None)
            _active_containers[new_container_id] = {
                **(old_info or {}),
                "name": target_name,
                "image": image,
                "cwd": cwd,
                "runtime_env": runtime_env,
                "runtime_state": runtime_state,
                "created_at": time.time(),
                "pooled": False,
                "acquisition": "full_checkpoint_rerun",
                "create_time_sec": resume_sec,
                "reset_time_sec": 0.0,
            }
        _CHECKPOINTS.mark_used(checkpoint_id)
        return {
            "ok": True,
            "checkpoint_id": checkpoint_id,
            "checkpoint_backend": "full",
            "checkpoint_image": None,
            "restored_image": image,
            "new_container_id": new_container_id,
            "target_name": target_name,
            "full_resume_sec": resume_sec,
            "create_time_sec": resume_sec,
        }
    except Exception:
        try:
            with _foreground_docker_section():
                _docker_destroy_container(target_name, timeout=300)
            if new_container_id:
                _drop_container_op_gate(new_container_id)
        except Exception:
            logger.exception(
                "Failed to destroy full-resume target after restore error: %s", target_name
            )
        raise


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

    request_lease_id = str(data.get("lease_id", "") or "")
    record_lease_id = str(record.get("lease_id", "") or "")
    if request_lease_id and record_lease_id and request_lease_id != record_lease_id:
        return jsonify({
            "ok": False,
            "error": "checkpoint belongs to another lease",
            "error_code": "checkpoint_lease_mismatch",
            "retryable": False,
        }), 409

    image = str(record.get("checkpoint_image", ""))
    cwd = str(data.get("cwd", record.get("cwd", "/testbed")))
    timeout = int(data.get("timeout", 120))
    if _record_checkpoint_backend(record) == "full":
        try:
            return jsonify(
                _full_checkpoint_rerun(
                    record=record,
                    old_container_id=str(old_container_id) if old_container_id else None,
                    cwd=cwd,
                    timeout=timeout,
                )
            )
        except Exception as exc:
            logger.exception(
                "Full checkpoint rerun failed checkpoint_id=%s old_container_id=%s",
                checkpoint_id,
                old_container_id,
            )
            return jsonify({
                "ok": False,
                "error": str(exc),
                "error_code": "full_rerun_failed",
                "retryable": True,
                "checkpoint_backend": "full",
            }), 500
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
    """Read docker stats for a running container.

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

    try:
        out = _container_stats_cached_or_direct(str(container_id), include_raw=bool(data.get("include_raw", True)))
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    if not out.get("ok", False):
        return jsonify(out), 500
    return jsonify(out)


@app.post("/container/stats_batch")
def container_stats_batch():
    """Read docker stats for many running containers using cache + one batch call."""
    data = request.get_json(force=True) or {}
    raw_container_ids = data.get("container_ids")
    if not isinstance(raw_container_ids, list) or not raw_container_ids:
        return jsonify({"ok": False, "error": "container_ids must be a non-empty list"}), 400

    max_items = max(1, int(_SERVER_CONFIG.stats_batch_max_containers))
    requested: list[str] = []
    seen: set[str] = set()
    for item in raw_container_ids:
        container_id = str(item or "").strip()
        if not container_id or container_id in seen:
            continue
        requested.append(container_id)
        seen.add(container_id)

    if not requested:
        return jsonify({"ok": False, "error": "container_ids contains no valid container id"}), 400

    _ensure_stats_sampler_started()
    include_raw = bool(data.get("include_raw", False))
    ttl = max(0.0, float(_SERVER_CONFIG.stats_cache_ttl_sec))
    stats: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for container_id in requested:
        cached = _get_cached_container_stats(container_id, max_age_sec=ttl)
        if cached is None:
            missing.append(container_id)
        else:
            stats[container_id] = cached

    if missing:
        for idx in range(0, len(missing), max_items):
            chunk = missing[idx : idx + max_items]
            try:
                fresh = _resource_stats_for_containers(
                    chunk,
                    timeout=max(1, int(_SERVER_CONFIG.stats_command_timeout_sec)),
                    include_raw=include_raw,
                )
                merged = _put_cached_container_stats(fresh)
                stats.update(merged or fresh)
            except Exception as exc:
                for container_id in chunk:
                    stats[container_id] = {
                        "ok": False,
                        "error": str(exc),
                        "container_id": container_id,
                        "ts": time.time(),
                    }

    response_stats: dict[str, dict[str, Any]] = {}
    now = time.time()
    for container_id in requested:
        item = dict(stats.get(container_id) or {})
        if not item:
            item = {"ok": False, "error": "empty docker stats output", "ts": now}
        item.setdefault("container_id", container_id)
        response_stats[container_id] = item

    return jsonify({"ok": True, "stats": response_stats, "requested": len(requested), "ts": now})


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
    with _action_stats_sampling_section(str(container_id)):
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
        with _action_stats_sampling_section(str(container_id)):
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
        "Starting SWE exec server on %s:%s pool_enabled=%s config=%s per_image=%s total_cap=%s prewarm_target=%s prewarm_ratio=%.2f create_timeout=%ss prewarm_mode=serial stats_backend=%s action_stats=%s action_interval=%.3fs stats_dir=%s",
        args.host,
        args.port,
        _SERVER_CONFIG.use_container_pool,
        _SERVER_CONFIG.config_path,
        _SERVER_CONFIG.pool_max_size_per_image,
        _CONTAINER_POOL._effective_total_cap(),
        _CONTAINER_POOL._target_total_count(),
        _SERVER_CONFIG.pool_prewarm_ratio,
        _SERVER_CONFIG.pool_create_timeout_sec,
        _SERVER_CONFIG.stats_backend,
        _SERVER_CONFIG.action_stats_sampler_enabled,
        _SERVER_CONFIG.action_stats_interval_sec,
        _SERVER_CONFIG.pool_resource_stats_dir,
    )
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
