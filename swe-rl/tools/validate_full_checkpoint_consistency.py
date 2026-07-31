#!/usr/bin/env python3
"""Validate full checkpoint consistency with real Docker fail-stop faults.

The experiment compares every recovered execution with a no-fault golden run.
The container init process owns both deterministic in-memory state and a
non-idempotent filesystem workload.  The primary campaign chooses wall-clock
fault times before execution and independently of operation phases.  Phase
hooks only timestamp the execution so phase coverage can be classified after
the fault.  Optional directed boundary tests remain available as diagnostics.

This validator intentionally excludes pre-existing ``docker exec -d``
processes and live TCP connections.  Synchronous ``docker exec`` is used only
as a request/inspection control plane; no exec process is alive at a
checkpoint boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import random
import shutil
import signal
import struct
import subprocess
import sys
import textwrap
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


WORKSPACE_ROOT = Path(__file__).resolve().parents[3]
FULL_CHECKPOINT_ROOT = WORKSPACE_ROOT / "docker-full-checkpoint"
FULL_CHECKPOINT_SRC = FULL_CHECKPOINT_ROOT / "src"
if str(FULL_CHECKPOINT_SRC) not in sys.path:
    sys.path.insert(0, str(FULL_CHECKPOINT_SRC))

from full_checkpoint.checkpoint import CheckpointOptions, full_checkpoint
from full_checkpoint.resume import ResumeOptions, full_resume
from full_checkpoint.store import CheckpointStore


CHECKPOINT_PHASES = [
    "checkpoint_before_discovery",
    "checkpoint_after_discovery",
    "checkpoint_before_preflight",
    "checkpoint_after_preflight",
    "checkpoint_after_manifest_pending",
    "checkpoint_before_metadata_persist",
    "checkpoint_after_metadata_persist",
    "checkpoint_before_runtime_dump",
    "checkpoint_after_runtime_dump",
    "checkpoint_after_runtime_persist_started",
    "checkpoint_before_rootfs_snapshot",
    "checkpoint_after_rootfs_snapshot",
    "checkpoint_before_runtime_persist_wait",
    "checkpoint_after_runtime_persist_wait",
    "checkpoint_before_manifest_ready",
    "checkpoint_after_manifest_ready",
]

RESUME_PHASES = [
    "resume_before_manifest_load",
    "resume_after_manifest_load",
    "resume_before_compatibility_check",
    "resume_after_compatibility_check",
    "resume_before_capability_check",
    "resume_after_capability_check",
    "resume_before_target_create",
    "resume_after_target_create",
    "resume_before_target_identity_check",
    "resume_after_target_identity_check",
    "resume_before_checkpoint_install",
    "resume_after_checkpoint_install",
    "resume_before_upperdir_restore",
    "resume_after_upperdir_restore",
    "resume_before_runtime_restore",
    "resume_after_runtime_restore",
    "resume_after_target_running",
]

ACTION_PHASES = [
    "action_before_runtime_mutation",
    "action_after_runtime_mutation",
    "action_after_effect_append",
    "action_after_counter_temp_write",
    "action_after_counter_rename",
    "action_after_payload_write",
    "action_after_namespace_mutation",
    "action_after_history_commit",
    "action_before_reply",
]

DIFFERENTIAL_RUNTIME_KEYS = [
    "logical_step",
    "prng_state",
    "accumulator",
    "heap_sha256",
    "history",
    "effects_fd_offset",
    "state_chain",
]

RUNTIME_ARTIFACT_MAGIC = b"FC-RUNTIME-V1"
RUNTIME_ARTIFACT_SCHEMA = 1
RUNTIME_ARTIFACT_HEADER = struct.Struct(">16sI9Q")
RUNTIME_STATE_CHAIN_BYTES = 32


WORKLOAD_SOURCE = textwrap.dedent(
    r"""
    import hashlib
    import json
    import os
    import pathlib
    import signal
    import socket
    import stat
    import struct
    import time
    import uuid

    root = pathlib.Path("/consistency")
    root.mkdir(parents=True, exist_ok=True)
    socket_name = "\0" + os.environ["FULL_CONSISTENCY_SOCKET"]
    seed = int(os.environ.get("FULL_CONSISTENCY_SEED", "20260721"))
    heap_mb = int(os.environ.get("FULL_CONSISTENCY_HEAP_MB", "16"))
    payload_mb = int(os.environ.get("FULL_CONSISTENCY_PAYLOAD_MB", "16"))
    deterministic_epoch_ns = 1700000000 * 1000000000

    incarnation_uuid = uuid.uuid4().hex
    heap = bytearray(heap_mb * 1024 * 1024)
    for offset in range(0, len(heap), 4096):
        heap[offset] = ((offset // 4096) + seed) % 251

    payload_path = root / "payload.bin"
    pattern = hashlib.sha256(f"payload:{seed}".encode()).digest() * 4096
    with payload_path.open("wb") as handle:
        remaining = payload_mb * 1024 * 1024
        while remaining:
            chunk = pattern[: min(remaining, len(pattern))]
            handle.write(chunk)
            remaining -= len(chunk)
        handle.flush()
        os.fsync(handle.fileno())

    (root / "counter.txt").write_text("0\n", encoding="utf-8")
    (root / "delete-me.txt").write_text("delete-at-step-3\n", encoding="utf-8")
    executable = root / "executable.sh"
    executable.write_text("#!/bin/sh\necho deterministic\n", encoding="utf-8")
    executable.chmod(0o700)
    nested = root / "nested"
    nested.mkdir(exist_ok=True)
    (nested / "initial.txt").write_text("initial\n", encoding="utf-8")

    effects = (root / "effects.log").open("a+", buffering=1, encoding="utf-8")
    effects.seek(0, os.SEEK_END)
    logical_step = 0
    prng_state = seed & ((1 << 64) - 1)
    accumulator = (seed ^ 0x9E3779B97F4A7C15) & ((1 << 64) - 1)
    history = []
    request_count = 0
    active_action_step = 0
    state_chain = hashlib.sha256(f"runtime-init:{seed}".encode()).digest()
    filesystem_chain_path = root / "fs-chain.bin"
    initial_filesystem_chain = hashlib.sha256(
        f"filesystem-init:{seed}".encode()
    ).digest()
    with filesystem_chain_path.open("wb") as handle:
        handle.write(initial_filesystem_chain)
        handle.flush()
        os.fsync(handle.fileno())


    def emit_phase(phase, requested):
        print(
            json.dumps(
                {
                    "event": "action_phase",
                    "phase": phase,
                    "step": active_action_step,
                    "time_ns": time.time_ns(),
                }
            ),
            flush=True,
        )
        if requested == phase:
            print(
                json.dumps(
                    {
                        "event": "action_fault_fired",
                        "phase": phase,
                        "step": active_action_step,
                        "time_ns": time.time_ns(),
                    }
                ),
                flush=True,
            )
            # A process inside a PID namespace cannot reliably deliver a
            # fatal signal to that namespace's init task, even to itself.
            # _exit terminates PID1 immediately without Python cleanup and
            # gives Docker the same stopped-container fail-stop boundary.
            os._exit(137)


    def runtime_state():
        return {
            "logical_step": logical_step,
            "prng_state": prng_state,
            "accumulator": accumulator,
            "heap_sha256": hashlib.sha256(heap).hexdigest(),
            "history": list(history),
            "effects_fd_offset": effects.tell(),
            "state_chain": state_chain.hex(),
            "incarnation_uuid": incarnation_uuid,
            "pid": os.getpid(),
            "heap_bytes": len(heap),
            "request_count": request_count,
        }


    def write_counter(step, effect):
        tmp = root / "counter.txt.tmp"
        with tmp.open("w", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": step, "effect": effect}, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return tmp


    def write_filesystem_chain(value):
        tmp = root / "fs-chain.bin.tmp"
        with tmp.open("wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, filesystem_chain_path)
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


    def mutate_namespace(step, effect):
        if step == 1:
            (nested / "move-source.txt").write_text(effect + "\n", encoding="utf-8")
        elif step == 2:
            os.replace(nested / "move-source.txt", nested / "move-target.txt")
        elif step == 3:
            (root / "delete-me.txt").unlink(missing_ok=True)
        elif step == 4:
            link = root / "latest-counter"
            link.unlink(missing_ok=True)
            link.symlink_to("counter.txt")
        elif step == 5:
            executable.chmod(0o751)
            (root / "mode-updated.txt").write_text("0751\n", encoding="utf-8")
        elif step == 6:
            final_dir = root / "final-dir"
            final_dir.mkdir(exist_ok=True)
            os.replace(nested / "initial.txt", final_dir / "initial-moved.txt")
        else:
            (root / f"step-{step:03d}.txt").write_text(effect + "\n", encoding="utf-8")


    def normalize_tree_times(step):
        timestamp_ns = deterministic_epoch_ns + step * 1000000000
        paths = sorted(root.rglob("*"), key=lambda item: item.as_posix())
        for path in paths:
            os.utime(
                path,
                ns=(timestamp_ns, timestamp_ns),
                follow_symlinks=False,
            )
        os.utime(root, ns=(timestamp_ns, timestamp_ns))


    def apply_step(step, token, fail_phase):
        global logical_step, prng_state, accumulator, active_action_step, state_chain
        if step != logical_step + 1:
            raise RuntimeError(f"non-sequential step: expected={logical_step + 1} actual={step}")

        active_action_step = step
        previous_filesystem_chain = filesystem_chain_path.read_bytes()
        if len(previous_filesystem_chain) != 32:
            raise RuntimeError(
                f"invalid filesystem chain length: {len(previous_filesystem_chain)}"
            )
        emit_phase("action_before_runtime_mutation", fail_phase)
        filesystem_word = int.from_bytes(previous_filesystem_chain[:8], "big")
        prng_state = (
            prng_state * 6364136223846793005
            + 1442695040888963407
            + step
            + filesystem_word
        ) & ((1 << 64) - 1)
        accumulator = (
            accumulator * 2862933555777941757
            + prng_state
            + step
            + int.from_bytes(previous_filesystem_chain[8:16], "big")
        ) & ((1 << 64) - 1)
        for index in range(96):
            offset = (prng_state + index * 104729 + step * 8191) % len(heap)
            heap[offset] = (
                heap[offset]
                + step
                + index
                + (prng_state & 0xFF)
                + previous_filesystem_chain[index % 32]
            ) & 0xFF
        logical_step = step
        emit_phase("action_after_runtime_mutation", fail_phase)

        effect = hashlib.sha256(
            b"effect-v1\0"
            + state_chain
            + previous_filesystem_chain
            + f"{seed}:{step}:{token}:{prng_state}:{accumulator}".encode()
        ).hexdigest()
        event = json.dumps(
            {"effect": effect, "step": step, "token": token},
            sort_keys=True,
            separators=(",", ":"),
        )
        effects.write(event + "\n")
        effects.flush()
        os.fsync(effects.fileno())
        emit_phase("action_after_effect_append", fail_phase)

        counter_tmp = write_counter(step, effect)
        emit_phase("action_after_counter_temp_write", fail_phase)
        os.replace(counter_tmp, root / "counter.txt")
        emit_phase("action_after_counter_rename", fail_phase)

        payload_offset = (step * 1048573 + (prng_state & 0xFFFF)) % (payload_mb * 1024 * 1024 - 64)
        payload_bytes = hashlib.sha512(event.encode()).digest()
        with payload_path.open("r+b") as handle:
            handle.seek(payload_offset)
            handle.write(payload_bytes)
            handle.flush()
            os.fsync(handle.fileno())
        emit_phase("action_after_payload_write", fail_phase)

        mutate_namespace(step, effect)
        emit_phase("action_after_namespace_mutation", fail_phase)

        heap_sha256 = hashlib.sha256(heap).digest()
        state_chain = hashlib.sha256(
            b"runtime-chain-v1\0"
            + state_chain
            + previous_filesystem_chain
            + struct.pack(">QQQ", step, prng_state, accumulator)
            + heap_sha256
            + bytes.fromhex(effect)
        ).digest()
        filesystem_chain = hashlib.sha256(
            b"filesystem-chain-v1\0"
            + previous_filesystem_chain
            + state_chain
            + struct.pack(">Q", step)
            + bytes.fromhex(effect)
        ).digest()
        write_filesystem_chain(filesystem_chain)
        history.append(
            {
                "accumulator": accumulator,
                "effect": effect,
                "filesystem_chain": filesystem_chain.hex(),
                "heap_sha256": heap_sha256.hex(),
                "prng_state": prng_state,
                "state_chain": state_chain.hex(),
                "step": step,
                "token": token,
            }
        )
        normalize_tree_times(step)
        emit_phase("action_after_history_commit", fail_phase)
        emit_phase("action_before_reply", fail_phase)
        state = runtime_state()
        active_action_step = 0
        return state


    def dump_runtime(connection, expected_steps):
        if logical_step != expected_steps:
            raise RuntimeError(
                f"runtime dump step mismatch: expected={expected_steps} actual={logical_step}"
            )
        if active_action_step != 0:
            raise RuntimeError(f"runtime dump while action {active_action_step} is active")
        effects.flush()
        effects_fd_offset = os.lseek(effects.fileno(), 0, os.SEEK_CUR)
        history_bytes = json.dumps(
            history,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        header = struct.pack(
            ">16sI9Q",
            b"FC-RUNTIME-V1",
            1,
            seed,
            expected_steps,
            logical_step,
            prng_state,
            accumulator,
            effects_fd_offset,
            active_action_step,
            len(history_bytes),
            len(heap),
        )
        connection.sendall(header)
        connection.sendall(state_chain)
        connection.sendall(history_bytes)
        connection.sendall(heap)


    normalize_tree_times(0)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(socket_name)
    server.listen(8)
    print(json.dumps({"event": "ready", "incarnation_uuid": incarnation_uuid}), flush=True)

    while True:
        connection, _ = server.accept()
        with connection:
            data = b""
            while not data.endswith(b"\n"):
                chunk = connection.recv(65536)
                if not chunk:
                    break
                data += chunk
            try:
                request = json.loads(data.decode())
                request_count += 1
                operation = request.get("op")
                if operation == "inspect":
                    response = {"ok": True, "runtime": runtime_state()}
                elif operation == "apply":
                    state = apply_step(
                        int(request["step"]),
                        str(request["token"]),
                        str(request.get("fail_phase") or ""),
                    )
                    response = {"ok": True, "runtime": state}
                elif operation == "dump_runtime":
                    dump_runtime(connection, int(request["expected_steps"]))
                    continue
                elif operation == "calibrate_runtime_fault":
                    heap[-1] ^= 1
                    response = {"ok": True}
                else:
                    raise RuntimeError(f"unknown operation: {operation!r}")
            except BaseException as exc:
                response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            connection.sendall(json.dumps(response, sort_keys=True).encode() + b"\n")
    """
).strip()


RUNTIME_DUMP_CLIENT_SOURCE = textwrap.dedent(
    r"""
    import json
    import os
    import socket
    import sys
    import time

    address = "\0" + os.environ["FULL_CONSISTENCY_SOCKET"]
    expected_steps = int(sys.argv[1])
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + 10.0
    while True:
        try:
            sock.connect(address)
            break
        except (ConnectionRefusedError, FileNotFoundError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    sock.sendall(
        json.dumps(
            {"expected_steps": expected_steps, "op": "dump_runtime"},
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    while True:
        chunk = sock.recv(1024 * 1024)
        if not chunk:
            break
        sys.stdout.buffer.write(chunk)
    sys.stdout.buffer.flush()
    """
).strip()


RPC_CLIENT_SOURCE = textwrap.dedent(
    r"""
    import json
    import os
    import socket
    import sys
    import time

    address = "\0" + os.environ["FULL_CONSISTENCY_SOCKET"]
    request = json.loads(sys.argv[1])
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    deadline = time.monotonic() + 10.0
    while True:
        try:
            sock.connect(address)
            break
        except (ConnectionRefusedError, FileNotFoundError):
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)
    sock.sendall(json.dumps(request, sort_keys=True).encode() + b"\n")
    data = b""
    while not data.endswith(b"\n"):
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
    if data:
        print(data.decode(), end="")
    """
).strip()


FS_ORACLE_SOURCE = textwrap.dedent(
    r"""
    import hashlib
    import json
    import os
    import pathlib
    import stat

    root = pathlib.Path("/consistency")
    entries = []
    paths = [root, *sorted(root.rglob("*"), key=lambda item: str(item.relative_to(root)))]
    for path in paths:
        info = path.lstat()
        rel = "." if path == root else str(path.relative_to(root))
        item = {"mode": stat.S_IMODE(info.st_mode), "path": rel}
        if stat.S_ISDIR(info.st_mode):
            item["type"] = "dir"
        elif stat.S_ISLNK(info.st_mode):
            item.update({"type": "symlink", "target": os.readlink(path)})
        elif stat.S_ISREG(info.st_mode):
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            item.update({"type": "file", "size": info.st_size, "sha256": digest.hexdigest()})
        else:
            item["type"] = "other"
        xattrs = {}
        try:
            for name in sorted(os.listxattr(path, follow_symlinks=False)):
                value = os.getxattr(path, name, follow_symlinks=False)
                xattrs[name] = value.hex()
        except OSError:
            pass
        if xattrs:
            item["xattrs"] = xattrs
        entries.append(item)
    print(json.dumps(entries, sort_keys=True, separators=(",", ":")))
    """
).strip()


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    step: int
    snapshot: dict[str, Any]


@dataclass
class WorkerResult:
    exit_code: int
    duration_sec: float
    events: list[dict[str, Any]]
    fault_observed: bool
    manifest: dict[str, Any] | None = None

    @property
    def phases(self) -> list[str]:
        return [str(item["phase"]) for item in self.events if item.get("event") == "phase"]


class ExperimentError(RuntimeError):
    pass


def _json_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_runtime_artifact(
    path: Path,
    *,
    expected_seed: int,
    expected_steps: int,
    expected_heap_bytes: int,
) -> dict[str, Any]:
    """Validate the self-describing final runtime transcript and full heap."""
    size = path.stat().st_size
    minimum_size = RUNTIME_ARTIFACT_HEADER.size + RUNTIME_STATE_CHAIN_BYTES
    if size < minimum_size:
        raise ExperimentError(f"runtime artifact is truncated: size={size}")
    with path.open("rb") as handle:
        header_bytes = handle.read(RUNTIME_ARTIFACT_HEADER.size)
        (
            magic,
            schema,
            seed,
            declared_steps,
            logical_step,
            prng_state,
            accumulator,
            effects_fd_offset,
            active_action_step,
            history_length,
            heap_length,
        ) = RUNTIME_ARTIFACT_HEADER.unpack(header_bytes)
        if magic.rstrip(b"\0") != RUNTIME_ARTIFACT_MAGIC:
            raise ExperimentError(f"invalid runtime artifact magic: {magic!r}")
        if schema != RUNTIME_ARTIFACT_SCHEMA:
            raise ExperimentError(f"unsupported runtime artifact schema: {schema}")
        if seed != expected_seed:
            raise ExperimentError(f"runtime seed mismatch: expected={expected_seed} actual={seed}")
        if declared_steps != expected_steps or logical_step != expected_steps:
            raise ExperimentError(
                "runtime progress mismatch: "
                f"expected={expected_steps} declared={declared_steps} logical={logical_step}"
            )
        if active_action_step != 0:
            raise ExperimentError(f"runtime artifact captured active step {active_action_step}")
        if heap_length != expected_heap_bytes:
            raise ExperimentError(
                f"runtime heap length mismatch: expected={expected_heap_bytes} actual={heap_length}"
            )
        expected_size = minimum_size + history_length + heap_length
        if size != expected_size:
            raise ExperimentError(
                f"runtime artifact size mismatch: expected={expected_size} actual={size}"
            )
        state_chain = handle.read(RUNTIME_STATE_CHAIN_BYTES)
        history_bytes = handle.read(history_length)
        heap_bytes = handle.read(heap_length)
        if handle.read(1):
            raise ExperimentError("runtime artifact contains trailing bytes")

    try:
        history = json.loads(history_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"runtime history is not canonical JSON: {exc}") from exc
    if not isinstance(history, list) or len(history) != expected_steps:
        raise ExperimentError(
            f"runtime history length mismatch: expected={expected_steps} actual={len(history) if isinstance(history, list) else 'non-list'}"
        )
    if json.dumps(history, sort_keys=True, separators=(",", ":")).encode() != history_bytes:
        raise ExperimentError("runtime history encoding is not canonical")

    prior_runtime_chain = hashlib.sha256(
        f"runtime-init:{expected_seed}".encode()
    ).digest()
    prior_filesystem_chain = hashlib.sha256(
        f"filesystem-init:{expected_seed}".encode()
    ).digest()
    final_heap_digest = hashlib.sha256(heap_bytes).digest()
    for expected_step, entry in enumerate(history, 1):
        if not isinstance(entry, dict):
            raise ExperimentError(f"runtime history entry {expected_step} is not an object")
        if entry.get("step") != expected_step:
            raise ExperimentError(
                f"runtime transcript is missing, duplicated, or reordered at step {expected_step}: {entry.get('step')!r}"
            )
        expected_token = f"token-{expected_step:03d}"
        if entry.get("token") != expected_token:
            raise ExperimentError(
                f"runtime transcript token mismatch at step {expected_step}: {entry.get('token')!r}"
            )
        try:
            entry_prng = int(entry["prng_state"])
            entry_accumulator = int(entry["accumulator"])
            entry_heap_digest = bytes.fromhex(str(entry["heap_sha256"]))
            entry_effect = bytes.fromhex(str(entry["effect"]))
            entry_runtime_chain = bytes.fromhex(str(entry["state_chain"]))
            entry_filesystem_chain = bytes.fromhex(str(entry["filesystem_chain"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ExperimentError(
                f"runtime transcript entry {expected_step} has invalid fields: {exc}"
            ) from exc
        if any(
            len(value) != 32
            for value in (
                entry_heap_digest,
                entry_effect,
                entry_runtime_chain,
                entry_filesystem_chain,
            )
        ):
            raise ExperimentError(
                f"runtime transcript entry {expected_step} has a non-SHA-256 field"
            )
        expected_effect = hashlib.sha256(
            b"effect-v1\0"
            + prior_runtime_chain
            + prior_filesystem_chain
            + (
                f"{expected_seed}:{expected_step}:{expected_token}:"
                f"{entry_prng}:{entry_accumulator}"
            ).encode()
        ).digest()
        if entry_effect != expected_effect:
            raise ExperimentError(f"runtime effect chain mismatch at step {expected_step}")
        expected_runtime_chain = hashlib.sha256(
            b"runtime-chain-v1\0"
            + prior_runtime_chain
            + prior_filesystem_chain
            + struct.pack(">QQQ", expected_step, entry_prng, entry_accumulator)
            + entry_heap_digest
            + entry_effect
        ).digest()
        if entry_runtime_chain != expected_runtime_chain:
            raise ExperimentError(f"runtime state chain mismatch at step {expected_step}")
        expected_filesystem_chain = hashlib.sha256(
            b"filesystem-chain-v1\0"
            + prior_filesystem_chain
            + entry_runtime_chain
            + struct.pack(">Q", expected_step)
            + entry_effect
        ).digest()
        if entry_filesystem_chain != expected_filesystem_chain:
            raise ExperimentError(f"filesystem chain mismatch at step {expected_step}")
        prior_runtime_chain = entry_runtime_chain
        prior_filesystem_chain = entry_filesystem_chain

    last = history[-1] if history else None
    if state_chain != prior_runtime_chain:
        raise ExperimentError("runtime artifact state chain differs from transcript tail")
    if last is not None:
        if prng_state != int(last["prng_state"]):
            raise ExperimentError("runtime PRNG differs from transcript tail")
        if accumulator != int(last["accumulator"]):
            raise ExperimentError("runtime accumulator differs from transcript tail")
        if final_heap_digest.hex() != str(last["heap_sha256"]):
            raise ExperimentError("full heap differs from the transcript's final heap digest")
    return {
        "active_action_step": active_action_step,
        "artifact_sha256": _file_sha256(path),
        "bytes": size,
        "effects_fd_offset": effects_fd_offset,
        "heap_bytes": heap_length,
        "history_entries": len(history),
        "logical_step": logical_step,
        "schema": schema,
        "state_chain": state_chain.hex(),
        "transcript_valid": True,
    }


def _run(
    command: list[str],
    *,
    timeout: float = 180.0,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    if check and result.returncode != 0:
        raise ExperimentError(
            f"command failed rc={result.returncode}: {command!r}\n"
            f"stdout={result.stdout[-2000:]}\nstderr={result.stderr[-2000:]}"
        )
    return result


def _append_event(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, sort_keys=True, default=str) + "\n").encode()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, encoded)
        os.fsync(fd)
    finally:
        os.close(fd)


class _PhaseInjector:
    def __init__(self, *, log_path: Path, fault: dict[str, Any] | None) -> None:
        self.log_path = log_path
        self.fault = dict(fault or {})
        self.armed = False

    def __call__(self, phase: str, context: dict[str, object]) -> None:
        _append_event(
            self.log_path,
            {"context": context, "event": "phase", "phase": phase, "time_ns": time.time_ns()},
        )
        exact_phase = str(self.fault.get("exact_phase") or "")
        arm_phase = str(self.fault.get("arm_phase") or "")
        if exact_phase and phase == exact_phase:
            self._fire(phase=phase, strategy="exact_boundary")
        if arm_phase and phase == arm_phase and not self.armed:
            self.armed = True
            delay_sec = max(0.0, float(self.fault.get("delay_sec") or 0.0))
            thread = threading.Thread(
                target=self._fire_after_delay,
                args=(phase, delay_sec),
                name=f"fault-{phase}",
                daemon=True,
            )
            thread.start()

    def _fire_after_delay(self, phase: str, delay_sec: float) -> None:
        time.sleep(delay_sec)
        self._fire(phase=phase, strategy="timed_inside", delay_sec=delay_sec)

    def _fire(self, *, phase: str, strategy: str, delay_sec: float = 0.0) -> None:
        _append_event(
            self.log_path,
            {
                "delay_sec": delay_sec,
                "event": "fault_fired",
                "phase": phase,
                "strategy": strategy,
                "time_ns": time.time_ns(),
            },
        )
        try:
            os.killpg(os.getpgrp(), signal.SIGKILL)
        except OSError:
            os.kill(os.getpid(), signal.SIGKILL)


def _become_worker_session(log_path: Path) -> None:
    try:
        os.setsid()
    except OSError:
        pass
    _append_event(log_path, {"event": "worker_started", "pid": os.getpid(), "time_ns": time.time_ns()})


def _checkpoint_worker(config: dict[str, Any]) -> None:
    log_path = Path(config["log_path"])
    _become_worker_session(log_path)
    hook = _PhaseInjector(log_path=log_path, fault=config.get("fault"))
    try:
        full_checkpoint(
            str(config["container"]),
            options=CheckpointOptions(
                checkpoint_id=str(config["checkpoint_id"]),
                state_root=Path(config["state_root"]),
                docker_root=Path(config["docker_root"]),
                require_criu=True,
                docker_managed=True,
                leave_running=False,
                criu_timeout_sec=int(config["criu_timeout_sec"]),
                runtime_staging_root=Path(config["runtime_staging_root"]),
                phase_hook=hook,
            ),
        )
        _append_event(log_path, {"event": "worker_completed", "time_ns": time.time_ns()})
    except BaseException as exc:
        _append_event(
            log_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "event": "worker_error",
                "traceback": traceback.format_exc(limit=20),
                "time_ns": time.time_ns(),
            },
        )
        raise SystemExit(1) from exc


def _resume_worker(config: dict[str, Any]) -> None:
    log_path = Path(config["log_path"])
    _become_worker_session(log_path)
    hook = _PhaseInjector(log_path=log_path, fault=config.get("fault"))
    try:
        full_resume(
            str(config["checkpoint_id"]),
            options=ResumeOptions(
                state_root=Path(config["state_root"]),
                container_id=str(config["target"]),
                docker_root=Path(config["docker_root"]),
                docker_managed=True,
                keep_failed=True,
                criu_timeout_sec=int(config["criu_timeout_sec"]),
                phase_hook=hook,
            ),
        )
        _append_event(log_path, {"event": "worker_completed", "time_ns": time.time_ns()})
    except BaseException as exc:
        _append_event(
            log_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "event": "worker_error",
                "traceback": traceback.format_exc(limit=20),
                "time_ns": time.time_ns(),
            },
        )
        raise SystemExit(1) from exc


def _read_events(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _sequence_event(log_path: Path, phase: str, *, operation: str, **payload: Any) -> None:
    _append_event(
        log_path,
        {
            "event": "sequence_phase",
            "operation": operation,
            "phase": phase,
            "time_ns": time.time_ns(),
            **payload,
        },
    )


def _sequence_apply(container: str, step: int) -> None:
    result = _run(
        [
            "docker",
            "exec",
            container,
            "python3",
            "-c",
            RPC_CLIENT_SOURCE,
            json.dumps(
                {
                    "fail_phase": "",
                    "op": "apply",
                    "step": step,
                    "token": f"token-{step:03d}",
                },
                sort_keys=True,
            ),
        ],
        timeout=60,
        check=False,
    )
    if result.returncode != 0:
        raise ExperimentError(
            f"sequence action failed step={step} rc={result.returncode}: "
            f"{result.stdout[-1000:]} {result.stderr[-1000:]}"
        )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict) or not payload.get("ok", False):
        raise ExperimentError(f"sequence action returned failure step={step}: {payload!r}")


def _wallclock_sequence_worker(config: dict[str, Any]) -> None:
    """Run one complete fault-prone sequence; this worker never chooses a fault."""
    log_path = Path(config["log_path"])
    _become_worker_session(log_path)
    source = str(config["source"])
    state_root = Path(config["state_root"])
    docker_root = Path(config["docker_root"])
    checkpoint_step = int(config["checkpoint_step"])
    final_step = int(config["final_step"])
    candidate_id = str(config["candidate_id"])
    current_operation = {"value": "sequence_start"}

    def observe(phase: str, context: dict[str, object]) -> None:
        _append_event(
            log_path,
            {
                "context": context,
                "event": "phase",
                "operation": current_operation["value"],
                "phase": phase,
                "time_ns": time.time_ns(),
            },
        )

    def resume(checkpoint_id: str, operation: str) -> None:
        current_operation["value"] = operation
        _sequence_event(
            log_path,
            "sequence_before_resume",
            operation=operation,
            checkpoint_id=checkpoint_id,
        )
        full_resume(
            checkpoint_id,
            options=ResumeOptions(
                state_root=state_root,
                container_id=source,
                docker_root=docker_root,
                docker_managed=True,
                keep_failed=True,
                criu_timeout_sec=int(config["criu_timeout_sec"]),
                phase_hook=observe,
            ),
        )
        _run(
            ["docker", "checkpoint", "rm", source, checkpoint_id],
            timeout=60,
            check=False,
        )
        _sequence_event(
            log_path,
            "sequence_after_resume",
            operation=operation,
            checkpoint_id=checkpoint_id,
        )

    def action(step: int) -> None:
        operation = f"action_step_{step}"
        current_operation["value"] = operation
        _sequence_event(log_path, "sequence_before_action", operation=operation, step=step)
        _sequence_apply(source, step)
        _sequence_event(log_path, "sequence_after_action", operation=operation, step=step)

    try:
        _sequence_event(log_path, "sequence_start", operation="sequence_start")
        initial_checkpoint_id = str(config.get("initial_checkpoint_id") or "")
        if initial_checkpoint_id:
            resume(initial_checkpoint_id, "initial_restore")

        action(checkpoint_step)

        current_operation["value"] = f"checkpoint_step_{checkpoint_step}"
        _sequence_event(
            log_path,
            "sequence_before_checkpoint",
            operation=current_operation["value"],
            checkpoint_id=candidate_id,
            step=checkpoint_step,
        )
        full_checkpoint(
            source,
            options=CheckpointOptions(
                checkpoint_id=candidate_id,
                state_root=state_root,
                docker_root=docker_root,
                require_criu=True,
                docker_managed=True,
                leave_running=False,
                criu_timeout_sec=int(config["criu_timeout_sec"]),
                runtime_staging_root=Path(config["runtime_staging_root"]),
                phase_hook=observe,
            ),
        )
        _sequence_event(
            log_path,
            "sequence_after_checkpoint",
            operation=current_operation["value"],
            checkpoint_id=candidate_id,
            step=checkpoint_step,
        )

        resume(candidate_id, "source_resume")
        for step in range(checkpoint_step + 1, final_step + 1):
            action(step)
        _sequence_event(log_path, "sequence_complete", operation="sequence_complete")
    except BaseException as exc:
        _append_event(
            log_path,
            {
                "error": f"{type(exc).__name__}: {exc}",
                "event": "worker_error",
                "operation": current_operation["value"],
                "time_ns": time.time_ns(),
                "traceback": traceback.format_exc(limit=30),
            },
        )
        raise SystemExit(1) from exc


CHECKPOINT_STAGE_BY_PHASE = {
    "checkpoint_before_discovery": "checkpoint.discovery",
    "checkpoint_after_discovery": "checkpoint.preflight",
    "checkpoint_before_preflight": "checkpoint.preflight",
    "checkpoint_after_preflight": "checkpoint.manifest_create",
    "checkpoint_after_manifest_pending": "checkpoint.metadata_persist",
    "checkpoint_before_metadata_persist": "checkpoint.metadata_persist",
    "checkpoint_after_metadata_persist": "checkpoint.runtime_dump",
    "checkpoint_before_runtime_dump": "checkpoint.runtime_dump",
    "checkpoint_after_runtime_dump": "checkpoint.runtime_persist_start",
    "checkpoint_after_runtime_persist_started": "checkpoint.rootfs_snapshot",
    "checkpoint_before_rootfs_snapshot": "checkpoint.rootfs_snapshot",
    "checkpoint_after_rootfs_snapshot": "checkpoint.runtime_persist_wait",
    "checkpoint_before_runtime_persist_wait": "checkpoint.runtime_persist_wait",
    "checkpoint_after_runtime_persist_wait": "checkpoint.manifest_publish",
    "checkpoint_before_manifest_ready": "checkpoint.manifest_publish",
    "checkpoint_after_manifest_ready": "checkpoint.post_ready",
}

RESUME_STAGE_BY_PHASE = {
    "resume_before_manifest_load": "resume.manifest_load",
    "resume_after_manifest_load": "resume.compatibility_check",
    "resume_before_compatibility_check": "resume.compatibility_check",
    "resume_after_compatibility_check": "resume.capability_check",
    "resume_before_capability_check": "resume.capability_check",
    "resume_after_capability_check": "resume.target_create",
    "resume_before_target_create": "resume.target_create",
    "resume_after_target_create": "resume.identity_check",
    "resume_before_target_identity_check": "resume.identity_check",
    "resume_after_target_identity_check": "resume.checkpoint_install",
    "resume_before_checkpoint_install": "resume.checkpoint_install",
    "resume_after_checkpoint_install": "resume.upperdir_restore",
    "resume_before_upperdir_restore": "resume.upperdir_restore",
    "resume_after_upperdir_restore": "resume.runtime_restore",
    "resume_before_runtime_restore": "resume.runtime_restore",
    "resume_after_runtime_restore": "resume.post_restore",
    "resume_after_target_running": "resume.post_restore",
}

ACTION_STAGE_BY_PHASE = {
    "action_before_runtime_mutation": "action.runtime_mutation",
    "action_after_runtime_mutation": "action.effect_append",
    "action_after_effect_append": "action.counter_temp_write",
    "action_after_counter_temp_write": "action.counter_rename",
    "action_after_counter_rename": "action.payload_write",
    "action_after_payload_write": "action.namespace_mutation",
    "action_after_namespace_mutation": "action.history_commit",
    "action_after_history_commit": "action.reply",
    "action_before_reply": "action.reply",
}


def _wallclock_stage_for_event(event: dict[str, Any]) -> str | None:
    phase = str(event.get("phase") or "")
    if phase in CHECKPOINT_STAGE_BY_PHASE:
        return CHECKPOINT_STAGE_BY_PHASE[phase]
    if phase in RESUME_STAGE_BY_PHASE:
        return RESUME_STAGE_BY_PHASE[phase]
    if phase in ACTION_STAGE_BY_PHASE:
        return ACTION_STAGE_BY_PHASE[phase]
    sequence_stages = {
        "sequence_start": "sequence.dispatch",
        "sequence_before_resume": "resume.dispatch",
        "sequence_after_resume": "between_operations",
        "sequence_before_action": "action.dispatch",
        "sequence_after_action": "between_operations",
        "sequence_before_checkpoint": "checkpoint.dispatch",
        "sequence_after_checkpoint": "between_operations",
        "sequence_complete": "after_sequence_complete",
    }
    return sequence_stages.get(phase)


def _classify_wallclock_fault(
    events: list[dict[str, Any]],
    *,
    fault_time_ns: int,
) -> dict[str, Any]:
    prior = [
        event
        for event in events
        if int(event.get("time_ns", 0) or 0) <= fault_time_ns
        and event.get("event") not in {"fault_requested", "fault_signal_sent"}
        and _wallclock_stage_for_event(event) is not None
    ]
    if not prior:
        return {"stage": "before_first_trace_event", "last_event": None}
    last = max(prior, key=lambda event: int(event.get("time_ns", 0) or 0))
    return {
        "stage": _wallclock_stage_for_event(last),
        "last_event": last,
    }


class FullConsistencyExperiment:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_id = args.run_id or f"{int(time.time())}-{uuid.uuid4().hex[:6]}"
        self.prefix = f"fccons-{self.run_id}".lower().replace("_", "-")
        self.output_root = args.output_root.resolve()
        self.state_root = self.output_root / "state"
        self.runtime_staging_root = self.output_root / "runtime-staging"
        self.worker_log_root = self.output_root / "worker-events"
        self.oracle_root = self.output_root / "oracle-artifacts"
        self.runtime_observation_root = self.oracle_root / "runtime-artifacts"
        self.golden_runtime_observation = self.oracle_root / "golden-runtime.bin"
        self.output_path = self.output_root / "result.json"
        self.docker_root = self._docker_root()
        self.store = CheckpointStore(self.state_root)
        self.container_names: set[str] = set()
        self.sequence = 0
        self.trials: list[dict[str, Any]] = []
        self.golden: dict[str, Any] = {}
        self.golden_container = ""
        self.shared_checkpoint: CheckpointRef | None = None
        self.observed_checkpoint_phases: list[str] = []
        self.observed_resume_phases: list[str] = []
        self.wallclock_expected_stages: set[str] = set()
        self.wallclock_control_durations: dict[str, float] = {}
        self.oracle_sequence = 0
        self.oracle_calibration: dict[str, Any] = {}

    def run(self) -> dict[str, Any]:
        self._prepare()
        started = time.perf_counter()
        try:
            self.golden = self._run_golden()
            self.oracle_calibration = self._run_oracle_calibration()
            shared, checkpoint_phases, resume_phases = self._build_shared_control()
            self.shared_checkpoint = shared
            self.observed_checkpoint_phases = checkpoint_phases
            self.observed_resume_phases = resume_phases

            if self.args.wall_clock_trials > 0:
                self._run_wallclock_campaign(self.args.wall_clock_trials)

            if not self.args.wall_clock_only:
                self._run_base_restart_trials()
                self._run_action_trials()
                if not self.args.skip_checkpoint_faults:
                    self._run_checkpoint_boundary_trials(checkpoint_phases)
                    if not self.args.skip_timed_faults:
                        self._run_checkpoint_timed_trials()
                if not self.args.skip_resume_faults:
                    self._run_resume_boundary_trials(resume_phases)
                    if not self.args.skip_timed_faults:
                        self._run_resume_timed_trials()
        finally:
            self._cleanup_containers()

        result = self._build_result(duration_sec=time.perf_counter() - started)
        self._write_result(result)
        if not self.args.keep_artifacts:
            shutil.rmtree(self.state_root, ignore_errors=True)
            shutil.rmtree(self.runtime_staging_root, ignore_errors=True)
        return result

    def _prepare(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.runtime_staging_root.mkdir(parents=True, exist_ok=True)
        self.worker_log_root.mkdir(parents=True, exist_ok=True)
        self.runtime_observation_root.mkdir(parents=True, exist_ok=True)
        if not (FULL_CHECKPOINT_ROOT / "pyproject.toml").is_file():
            raise ExperimentError(f"docker-full-checkpoint project not found: {FULL_CHECKPOINT_ROOT}")
        for command in ("cmp", "rsync", "sudo"):
            if shutil.which(command) is None:
                raise ExperimentError(f"required consistency comparator is missing: {command}")
        sudo_probe = _run(["sudo", "-n", "true"], check=False, timeout=30)
        if sudo_probe.returncode != 0:
            raise ExperimentError(
                "non-interactive sudo is required to compare live container roots with rsync"
            )
        containers_dir = self.docker_root / "containers"
        try:
            if not containers_dir.is_dir():
                raise ExperimentError(f"Docker root is not readable: {containers_dir}; run as root")
            next(containers_dir.iterdir(), None)
        except PermissionError as exc:
            raise ExperimentError(f"Docker root is not readable: {containers_dir}; run as root") from exc
        image = _run(["docker", "image", "inspect", self.args.image], check=False)
        if image.returncode != 0:
            if self.args.no_pull:
                raise ExperimentError(f"required image is missing: {self.args.image}")
            _run(["docker", "pull", self.args.image], timeout=600)
        experimental = _run(
            ["docker", "info", "--format", "{{json .ExperimentalBuild}}"]
        ).stdout.strip()
        if experimental != "true":
            raise ExperimentError("Docker experimental checkpoint support is not enabled")

    def _docker_root(self) -> Path:
        value = _run(["docker", "info", "--format", "{{.DockerRootDir}}"] ).stdout.strip()
        if not value:
            raise ExperimentError("Docker did not report DockerRootDir")
        return Path(value)

    def _next_name(self, label: str) -> str:
        self.sequence += 1
        clean = "".join(char if char.isalnum() or char in "_.-" else "-" for char in label.lower())
        return f"{self.prefix}-{self.sequence:04d}-{clean}"[:120]

    def _create_container(self, label: str) -> str:
        name = self._next_name(label)
        command = [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            # Docker/CRIU cannot currently re-open Docker's short-lived
            # ``--network none`` namespace after the source task is gone
            # (the restore path resolves it as /proc/0/ns/net).  Host network
            # mode keeps the namespace stable; this workload still creates no
            # TCP/UDP sockets and validates only its Unix control socket.
            "--network",
            "host",
            "--memory",
            self.args.container_memory,
            "--pids-limit",
            "256",
            "--stop-timeout",
            "1",
            "--env",
            f"FULL_CONSISTENCY_SOCKET={name}",
            "--env",
            f"FULL_CONSISTENCY_SEED={self.args.seed}",
            "--env",
            f"FULL_CONSISTENCY_HEAP_MB={self.args.heap_mb}",
            "--env",
            f"FULL_CONSISTENCY_PAYLOAD_MB={self.args.payload_mb}",
            self.args.image,
            "python3",
            "-u",
            "-c",
            WORKLOAD_SOURCE,
        ]
        _run(command, timeout=300)
        self.container_names.add(name)
        deadline = time.monotonic() + self.args.startup_timeout_sec
        last_error = ""
        while time.monotonic() < deadline:
            try:
                state = self._inspect_runtime(name)
                if int(state.get("logical_step", -1)) == 0:
                    return name
            except Exception as exc:  # container startup races are expected
                last_error = str(exc)
            time.sleep(0.1)
        logs = _run(["docker", "logs", name], check=False).stdout
        raise ExperimentError(f"workload did not become ready: {name}: {last_error}\n{logs[-3000:]}")

    def _rpc(
        self,
        container: str,
        request: dict[str, Any],
        *,
        check: bool = True,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        result = _run(
            [
                "docker",
                "exec",
                container,
                "python3",
                "-c",
                RPC_CLIENT_SOURCE,
                json.dumps(request, sort_keys=True),
            ],
            timeout=timeout,
            check=False,
        )
        if result.returncode != 0:
            if check:
                raise ExperimentError(
                    f"RPC failed container={container} rc={result.returncode}: "
                    f"{result.stdout[-1000:]} {result.stderr[-1000:]}"
                )
            return {
                "ok": False,
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
            }
        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            if check:
                raise ExperimentError(f"RPC returned invalid JSON: {result.stdout!r}") from exc
            return {"ok": False, "stdout": result.stdout, "stderr": result.stderr}
        if not isinstance(payload, dict) or (check and not payload.get("ok", False)):
            raise ExperimentError(f"RPC returned failure: {payload!r}")
        return payload

    def _inspect_runtime(self, container: str) -> dict[str, Any]:
        payload = self._rpc(container, {"op": "inspect"})
        runtime = payload.get("runtime")
        if not isinstance(runtime, dict):
            raise ExperimentError(f"runtime state is missing: {payload!r}")
        return runtime

    def _apply_step(self, container: str, step: int, *, fail_phase: str = "") -> dict[str, Any]:
        return self._rpc(
            container,
            {
                "fail_phase": fail_phase,
                "op": "apply",
                "step": step,
                "token": f"token-{step:03d}",
            },
            check=not bool(fail_phase),
            timeout=60.0,
        )

    def _fs_state(self, container: str) -> list[dict[str, Any]]:
        result = _run(
            ["docker", "exec", container, "python3", "-c", FS_ORACLE_SOURCE],
            timeout=120,
        )
        payload = json.loads(result.stdout)
        if not isinstance(payload, list):
            raise ExperimentError("filesystem oracle did not return a list")
        return payload

    def _snapshot(self, container: str) -> dict[str, Any]:
        runtime = self._inspect_runtime(container)
        semantic = {key: runtime[key] for key in DIFFERENTIAL_RUNTIME_KEYS}
        filesystem = self._fs_state(container)
        continuity = {**semantic, "incarnation_uuid": runtime["incarnation_uuid"]}
        canonical = {"filesystem": filesystem, "runtime": semantic}
        effects_entry = next(
            (item for item in filesystem if item.get("path") == "effects.log"),
            {},
        )
        if int(effects_entry.get("size", -1)) != int(semantic["effects_fd_offset"]):
            raise ExperimentError(
                "open effects FD offset does not match filesystem size: "
                f"offset={semantic['effects_fd_offset']} file={effects_entry.get('size')}"
            )
        return {
            "canonical": canonical,
            "continuity_runtime": continuity,
            "fingerprint": _json_hash(canonical),
        }

    def _capture_runtime_artifact(
        self,
        container: str,
        path: Path,
        *,
        validate: bool = True,
    ) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            "docker",
            "exec",
            container,
            "python3",
            "-c",
            RUNTIME_DUMP_CLIENT_SOURCE,
            str(self.args.steps),
        ]
        with path.open("wb") as output:
            result = subprocess.run(
                command,
                stdout=output,
                stderr=subprocess.PIPE,
                timeout=120,
                check=False,
            )
        if result.returncode != 0:
            raise ExperimentError(
                f"runtime artifact capture failed rc={result.returncode}: "
                f"{result.stderr.decode(errors='replace')[-2000:]}"
            )
        if validate:
            return validate_runtime_artifact(
                path,
                expected_seed=self.args.seed,
                expected_steps=self.args.steps,
                expected_heap_bytes=self.args.heap_mb * 1024 * 1024,
            )
        return {
            "artifact_sha256": _file_sha256(path),
            "bytes": path.stat().st_size,
        }

    def _container_host_pid(self, container: str) -> int:
        result = _run(
            ["docker", "inspect", "--format", "{{.State.Pid}}", container],
            timeout=30,
        )
        try:
            pid = int(result.stdout.strip())
        except ValueError as exc:
            raise ExperimentError(
                f"Docker returned an invalid host PID for {container}: {result.stdout!r}"
            ) from exc
        if pid <= 0:
            raise ExperimentError(f"container is not running for oracle comparison: {container}")
        return pid

    def _runtime_differential_oracle(
        self,
        container: str,
        snapshot: dict[str, Any],
        *,
        label: str,
    ) -> dict[str, Any]:
        self.oracle_sequence += 1
        observation = self.runtime_observation_root / (
            f"{self.oracle_sequence:04d}-{label}-runtime.bin"
        )
        validation = self._capture_runtime_artifact(container, observation)
        if int(validation["effects_fd_offset"]) != int(
            snapshot["canonical"]["runtime"]["effects_fd_offset"]
        ):
            raise ExperimentError(
                "runtime artifact FD offset differs from the independent snapshot: "
                f"artifact={validation['effects_fd_offset']} "
                f"snapshot={snapshot['canonical']['runtime']['effects_fd_offset']}"
            )
        result = _run(
            ["cmp", "--silent", str(self.golden_runtime_observation), str(observation)],
            check=False,
            timeout=30,
        )
        if result.returncode not in {0, 1}:
            raise ExperimentError(
                f"cmp failed rc={result.returncode}: {result.stderr[-1000:]}"
            )
        matched = result.returncode == 0
        artifact_path: str | None = str(observation)
        if matched and not self.args.keep_artifacts:
            observation.unlink(missing_ok=True)
            artifact_path = None
        return {
            "artifact_bytes": validation["bytes"],
            "artifact_sha256": validation["artifact_sha256"],
            "comparator": "cmp --silent",
            "golden_observation": str(self.golden_runtime_observation),
            "match": matched,
            "method": "golden-run differential testing",
            "recovered_observation": artifact_path,
            "validation": validation,
        }

    def _filesystem_rsync_oracle(self, container: str) -> dict[str, Any]:
        if not self.golden_container:
            raise ExperimentError("golden container is unavailable for rsync comparison")
        golden_pid = self._container_host_pid(self.golden_container)
        recovered_pid = self._container_host_pid(container)
        command = [
            "sudo",
            "-n",
            "rsync",
            "--archive",
            "--hard-links",
            "--acls",
            "--xattrs",
            "--checksum",
            "--dry-run",
            "--delete",
            "--itemize-changes",
            "--numeric-ids",
            "--out-format=%i %n%L",
            f"/proc/{golden_pid}/root/consistency/",
            f"/proc/{recovered_pid}/root/consistency/",
        ]
        result = _run(command, check=False, timeout=180)
        if result.returncode != 0:
            raise ExperimentError(
                f"rsync comparison failed rc={result.returncode}: {result.stderr[-2000:]}"
            )
        differences = [line for line in result.stdout.splitlines() if line.strip()]
        return {
            "command": command,
            "comparator": "rsync 3.x checksum dry-run",
            "differences": differences,
            "match": not differences,
        }

    def _final_snapshot(self, container: str, *, label: str) -> dict[str, Any]:
        snapshot = self._snapshot(container)
        snapshot["oracles"] = {
            "filesystem": self._filesystem_rsync_oracle(container),
            "runtime": self._runtime_differential_oracle(
                container,
                snapshot,
                label=label,
            ),
        }
        return snapshot

    def _run_steps(self, container: str, start: int, end: int | None = None) -> None:
        last = self.args.steps if end is None else end
        for step in range(start, last + 1):
            result = self._apply_step(container, step)
            if not result.get("ok", False):
                raise ExperimentError(f"step {step} failed: {result}")

    def _run_golden(self) -> dict[str, Any]:
        container = self._create_container("golden")
        try:
            self._run_steps(container, 1)
            snapshot = self._snapshot(container)
            self.golden_container = container
            runtime_artifact = self._capture_runtime_artifact(
                container,
                self.golden_runtime_observation,
            )
            if int(runtime_artifact["effects_fd_offset"]) != int(
                snapshot["canonical"]["runtime"]["effects_fd_offset"]
            ):
                raise ExperimentError("golden runtime artifact FD offset mismatch")
            return {
                "container": container,
                "runtime_artifact": runtime_artifact,
                "snapshot": snapshot,
            }
        except BaseException:
            self._remove_container(container)
            self.golden_container = ""
            raise

    def _run_oracle_calibration(self) -> dict[str, Any]:
        """Prove that each final-state oracle rejects a controlled mutation."""
        container = self._create_container("oracle-calibration")
        mutated_runtime = self.oracle_root / "calibration-mutated-runtime.bin"
        try:
            self._run_steps(container, 1)
            snapshot = self._snapshot(container)
            baseline_filesystem = self._filesystem_rsync_oracle(container)
            baseline_runtime = self._runtime_differential_oracle(
                container,
                snapshot,
                label="oracle-calibration-baseline",
            )
            if not baseline_filesystem["match"] or not baseline_runtime["match"]:
                raise ExperimentError("oracle calibration baseline differs from golden")

            mutation = self._rpc(container, {"op": "calibrate_runtime_fault"})
            if not mutation.get("ok", False):
                raise ExperimentError("controlled runtime mutation was not applied")
            raw_metadata = self._capture_runtime_artifact(
                container,
                mutated_runtime,
                validate=False,
            )
            runtime_cmp = _run(
                [
                    "cmp",
                    "--silent",
                    str(self.golden_runtime_observation),
                    str(mutated_runtime),
                ],
                check=False,
                timeout=30,
            )
            if runtime_cmp.returncode != 1:
                raise ExperimentError(
                    f"runtime comparator failed calibration: rc={runtime_cmp.returncode}"
                )
            try:
                validate_runtime_artifact(
                    mutated_runtime,
                    expected_seed=self.args.seed,
                    expected_steps=self.args.steps,
                    expected_heap_bytes=self.args.heap_mb * 1024 * 1024,
                )
            except ExperimentError as exc:
                runtime_validation_error = str(exc)
            else:
                raise ExperimentError(
                    "runtime transcript validator accepted a controlled heap mutation"
                )

            filesystem_mutation = textwrap.dedent(
                r"""
                import os
                import pathlib
                root = pathlib.Path('/consistency')
                (root / 'counter.txt').write_text('calibration-mutation\n', encoding='utf-8')
                (root / 'calibration-extra.txt').write_text('extra\n', encoding='utf-8')
                os.chmod(root / 'executable.sh', 0o700)
                link = root / 'latest-counter'
                link.unlink(missing_ok=True)
                link.symlink_to('payload.bin')
                """
            ).strip()
            _run(
                ["docker", "exec", container, "python3", "-c", filesystem_mutation],
                timeout=60,
            )
            mutated_filesystem = self._filesystem_rsync_oracle(container)
            if mutated_filesystem["match"] or not mutated_filesystem["differences"]:
                raise ExperimentError("filesystem comparator accepted controlled mutations")
            result = {
                "baseline_filesystem_match": True,
                "baseline_runtime_match": True,
                "filesystem_mutation_detected": True,
                "filesystem_mutation_differences": mutated_filesystem["differences"],
                "runtime_mutation_artifact": raw_metadata,
                "runtime_mutation_detected_by_cmp": True,
                "runtime_mutation_detected_by_transcript_validator": True,
                "runtime_validation_error": runtime_validation_error,
            }
            if not self.args.keep_artifacts:
                mutated_runtime.unlink(missing_ok=True)
            return result
        finally:
            self._remove_container(container)

    def _build_shared_control(self) -> tuple[CheckpointRef, list[str], list[str]]:
        source = self._create_container("shared-source")
        checkpoint_id = self._next_name("shared-step-2")
        self._run_steps(source, 1, 2)
        expected = self._snapshot(source)
        checkpoint_result = self._run_checkpoint_worker(
            container=source,
            checkpoint_id=checkpoint_id,
            label="shared-create",
        )
        if checkpoint_result.exit_code != 0 or not checkpoint_result.manifest:
            raise ExperimentError(f"shared checkpoint failed: {checkpoint_result}")
        if checkpoint_result.manifest.get("status") != "ready":
            raise ExperimentError(f"shared checkpoint is not ready: {checkpoint_result.manifest}")
        self._assert_phase_sequence(
            operation="checkpoint",
            expected=CHECKPOINT_PHASES,
            observed=checkpoint_result.phases,
        )

        inplace_result = self._run_resume_worker(
            checkpoint_id=checkpoint_id,
            target=source,
            label="shared-inplace-resume",
        )
        if inplace_result.exit_code != 0:
            raise ExperimentError(f"source in-place resume failed: {inplace_result.events}")
        self._cleanup_installed_checkpoint(source, checkpoint_id)
        inplace_snapshot = self._snapshot(source)
        if inplace_snapshot["continuity_runtime"] != expected["continuity_runtime"]:
            raise ExperimentError("source in-place resume changed runtime state")
        if inplace_snapshot["canonical"]["filesystem"] != expected["canonical"]["filesystem"]:
            raise ExperimentError("source in-place resume changed filesystem state")
        self._remove_container(source)

        control_target = self._next_name("shared-control-target")
        resume_result = self._run_resume_worker(
            checkpoint_id=checkpoint_id,
            target=control_target,
            label="shared-new-target-resume",
        )
        self.container_names.add(control_target)
        if resume_result.exit_code != 0:
            raise ExperimentError(f"new-target control resume failed: {resume_result.events}")
        self._assert_phase_sequence(
            operation="resume",
            expected=RESUME_PHASES,
            observed=resume_result.phases,
        )
        self._cleanup_installed_checkpoint(control_target, checkpoint_id)
        restored = self._snapshot(control_target)
        if restored["continuity_runtime"] != expected["continuity_runtime"]:
            raise ExperimentError("new-target resume changed checkpoint runtime state")
        if restored["canonical"]["filesystem"] != expected["canonical"]["filesystem"]:
            raise ExperimentError("new-target resume changed checkpoint filesystem state")
        self._run_steps(control_target, 3)
        final = self._final_snapshot(control_target, label="shared-control")
        if not all(oracle["match"] for oracle in final["oracles"].values()):
            raise ExperimentError("no-fault checkpoint/resume control differs from golden")
        self._remove_container(control_target)

        self.trials.append(
            {
                "consistent": True,
                "consistency_oracles": final["oracles"],
                "fault_observed": False,
                "filesystem_rsync_match": True,
                "final_fingerprint": final["fingerprint"],
                "kind": "control",
                "phase": "no_fault_full_checkpoint_resume",
                "recovery_mode": "checkpoint",
                "runtime_differential_match": True,
                "status": "pass",
            }
        )
        return (
            CheckpointRef(checkpoint_id=checkpoint_id, step=2, snapshot=expected),
            checkpoint_result.phases,
            resume_result.phases,
        )

    def _run_wallclock_campaign(self, total_trials: int) -> None:
        """Inject at preselected times; phase traces are not consulted until afterward."""
        expected_snapshots = self._wallclock_candidate_snapshots()
        controls: dict[str, dict[str, Any]] = {}
        for variant in ("shared", "base"):
            trial = self._record_trial_safe(
                kind="wall_clock_control",
                phase=f"wall_clock_control_{variant}",
                callback=lambda variant=variant: self._execute_wallclock_sequence(
                    variant=variant,
                    delay_sec=None,
                    candidate_snapshot=expected_snapshots[variant],
                ),
            )
            if trial.get("status") != "pass":
                raise ExperimentError(f"wall-clock control failed variant={variant}: {trial}")
            controls[variant] = trial
            self.wallclock_control_durations[variant] = float(
                trial["sequence_duration_sec"]
            )
            self.wallclock_expected_stages.update(
                str(stage) for stage in trial.get("trace_stages", [])
            )

        rng = random.Random(self.args.wall_clock_seed)
        base_trials = int(round(total_trials * self.args.wall_clock_base_fraction))
        if total_trials >= 2:
            base_trials = min(total_trials - 1, max(1, base_trials))
        else:
            base_trials = 0
        shared_trials = total_trials - base_trials
        specs: list[tuple[str, int, float]] = []
        for variant, count in (("shared", shared_trials), ("base", base_trials)):
            duration = self.wallclock_control_durations[variant]
            delays = self._stratified_wallclock_delays(
                count=count,
                duration_sec=duration,
                rng=rng,
            )
            specs.extend((variant, index, delay) for index, delay in enumerate(delays, 1))
        rng.shuffle(specs)

        for campaign_index, (variant, stratum_index, delay_sec) in enumerate(specs, 1):
            self._record_trial_safe(
                kind="wall_clock_fail_stop",
                phase=f"wall_clock_sample_{campaign_index:04d}",
                callback=lambda variant=variant, stratum_index=stratum_index, delay_sec=delay_sec: self._execute_wallclock_sequence(
                    variant=variant,
                    delay_sec=delay_sec,
                    candidate_snapshot=expected_snapshots[variant],
                    stratum_index=stratum_index,
                ),
            )

    @staticmethod
    def _stratified_wallclock_delays(
        *,
        count: int,
        duration_sec: float,
        rng: random.Random,
    ) -> list[float]:
        if count <= 0:
            return []
        width = max(0.0, float(duration_sec)) / count
        delays = [(index + rng.random()) * width for index in range(count)]
        rng.shuffle(delays)
        return delays

    def _wallclock_candidate_snapshots(self) -> dict[str, dict[str, Any]]:
        assert self.shared_checkpoint is not None
        base = self._create_container("wall-clock-expected-step-1")
        try:
            self._apply_step(base, 1)
            step_one = self._snapshot(base)
        finally:
            self._remove_container(base)

        shared, _ = self._recover(
            [self.shared_checkpoint],
            label="wall-clock-expected-step-3",
        )
        try:
            self._apply_step(shared, 3)
            step_three = self._snapshot(shared)
        finally:
            self._remove_container(shared)
        return {"base": step_one, "shared": step_three}

    def _execute_wallclock_sequence(
        self,
        *,
        variant: str,
        delay_sec: float | None,
        candidate_snapshot: dict[str, Any],
        stratum_index: int | None = None,
    ) -> dict[str, Any]:
        assert self.shared_checkpoint is not None
        if variant not in {"base", "shared"}:
            raise ValueError(f"unknown wall-clock variant: {variant}")

        checkpoint_step = 1 if variant == "base" else 3
        source = self._create_container("wall-clock-base-source") if variant == "base" else self._next_name(
            "wall-clock-shared-source"
        )
        self.container_names.add(source)
        trial_candidate_snapshot = candidate_snapshot
        if variant == "base":
            # The semantic step-1 state is deterministic, but incarnation UUID
            # is intentionally random per fresh container. Capture this trial's
            # own UUID before arming the independent timer so a ready candidate
            # is checked against its real process incarnation rather than the
            # separate step-1 reference container.
            initial_runtime = self._inspect_runtime(source)
            trial_candidate_snapshot = json.loads(json.dumps(candidate_snapshot))
            trial_candidate_snapshot["continuity_runtime"]["incarnation_uuid"] = str(
                initial_runtime["incarnation_uuid"]
            )
        candidate_id = self._next_name(f"wall-clock-{variant}-step-{checkpoint_step}")
        log_path = self.worker_log_root / f"{self._next_name(f'wall-clock-{variant}')}.jsonl"
        config = {
            "candidate_id": candidate_id,
            "checkpoint_step": checkpoint_step,
            "criu_timeout_sec": self.args.criu_timeout_sec,
            "docker_root": str(self.docker_root),
            "final_step": self.args.steps,
            "initial_checkpoint_id": (
                self.shared_checkpoint.checkpoint_id if variant == "shared" else ""
            ),
            "log_path": str(log_path),
            "runtime_staging_root": str(self.runtime_staging_root),
            "source": source,
            "state_root": str(self.state_root),
        }
        context = multiprocessing.get_context("fork")
        process = context.Process(
            target=_wallclock_sequence_worker,
            args=(config,),
            daemon=False,
        )
        started = time.perf_counter()
        process.start()

        fault_observed = False
        fault_time_ns: int | None = None
        if delay_sec is None:
            process.join(self.args.worker_timeout_sec)
        else:
            sampled_delay = max(0.0, float(delay_sec))
            process.join(sampled_delay)
            coordinator_running_at_fault = process.is_alive()
            remaining = sampled_delay - (time.perf_counter() - started)
            if remaining > 0.0:
                time.sleep(remaining)
            if process.exitcode not in {None, 0}:
                raise ExperimentError(
                    "wall-clock sequence failed before its independently sampled fault time"
                )
            fault_observed = True
            fault_time_ns = time.time_ns()
            _append_event(
                log_path,
                {
                    "coordinator_running": coordinator_running_at_fault,
                    "delay_sec": float(delay_sec),
                    "event": "fault_requested",
                    "strategy": "independent_wall_clock",
                    "time_ns": fault_time_ns,
                },
            )
            if coordinator_running_at_fault:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except OSError:
                    process.kill()
            _run(
                ["docker", "kill", "--signal", "KILL", source],
                timeout=60,
                check=False,
            )
            _append_event(
                log_path,
                {
                    "coordinator_running": coordinator_running_at_fault,
                    "event": "fault_signal_sent",
                    "strategy": "independent_wall_clock",
                    "time_ns": time.time_ns(),
                },
            )
            if coordinator_running_at_fault:
                process.join(30.0)

        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            process.join(10.0)
            raise ExperimentError("wall-clock sequence exceeded worker timeout")

        sequence_duration = time.perf_counter() - started
        events = _read_events(log_path)
        events.extend(self._container_action_events(source))
        events.sort(key=lambda item: int(item.get("time_ns", 0) or 0))
        trace_stages = sorted(
            {
                stage
                for event in events
                if (stage := _wallclock_stage_for_event(event)) is not None
            }
        )
        worker_error = next(
            (
                str(event.get("error"))
                for event in reversed(events)
                if event.get("event") == "worker_error"
            ),
            None,
        )
        manifest = self._manifest_snapshot(candidate_id)

        if delay_sec is None:
            try:
                if process.exitcode != 0:
                    raise ExperimentError(
                        f"wall-clock control worker failed variant={variant}: {worker_error}"
                    )
                self._cleanup_installed_checkpoint(source, candidate_id)
                final = self._final_snapshot(
                    source,
                    label=f"wall-clock-control-{variant}",
                )
                consistent = all(
                    oracle["match"] for oracle in final["oracles"].values()
                )
                return {
                    "candidate_manifest": manifest,
                    "consistent": consistent,
                    "consistency_oracles": final["oracles"],
                    "fault_observed": False,
                    "filesystem_rsync_match": bool(
                        final["oracles"]["filesystem"]["match"]
                    ),
                    "final_fingerprint": final["fingerprint"],
                    "golden_fingerprint": self.golden["snapshot"]["fingerprint"],
                    "injection_strategy": "none_control",
                    "recovery_mode": "not_needed",
                    "runtime_differential_match": bool(
                        final["oracles"]["runtime"]["match"]
                    ),
                    "sequence_duration_sec": sequence_duration,
                    "status": "pass" if consistent else "final_state_mismatch",
                    "trace_stages": trace_stages,
                    "variant": variant,
                    "worker_exit_code": process.exitcode,
                }
            finally:
                self._remove_container(source)
                self._remove_checkpoint(candidate_id)

        assert fault_time_ns is not None
        classification = _classify_wallclock_fault(events, fault_time_ns=fault_time_ns)
        self._remove_container(source)
        candidate = self._ready_ref(
            candidate_id,
            step=checkpoint_step,
            snapshot=trial_candidate_snapshot,
        )
        candidates: list[CheckpointRef] = []
        if candidate is not None:
            candidates.append(candidate)
        if variant == "shared":
            candidates.append(self.shared_checkpoint)
        recovered, recovery = self._recover(
            candidates,
            label=f"wall-clock-{variant}-recovery",
        )
        try:
            self._run_steps(recovered, recovery["restored_step"] + 1)
            final = self._final_snapshot(
                recovered,
                label=f"wall-clock-{variant}-{stratum_index or 0:04d}",
            )
            return self._finalize_trial(
                final,
                recovery,
                fault_observed=True,
                candidate_manifest=manifest,
                coordinator_running_at_fault=coordinator_running_at_fault,
                fault_time_ns=fault_time_ns,
                injection_strategy="independent_wall_clock",
                observed_stage=classification["stage"],
                preceding_trace_event=classification["last_event"],
                sampled_delay_sec=delay_sec,
                sequence_duration_sec=sequence_duration,
                stratum_index=stratum_index,
                trace_stages=trace_stages,
                variant=variant,
                worker_exit_code=process.exitcode,
            )
        finally:
            self._remove_container(recovered)
            self._remove_checkpoint(candidate_id)

    def _container_action_events(self, container: str) -> list[dict[str, Any]]:
        result = _run(["docker", "logs", container], timeout=60, check=False)
        events: list[dict[str, Any]] = []
        for line in f"{result.stdout}\n{result.stderr}".splitlines():
            start = line.find("{")
            if start < 0:
                continue
            try:
                payload = json.loads(line[start:])
            except json.JSONDecodeError:
                continue
            if (
                isinstance(payload, dict)
                and payload.get("event") in {"action_phase", "action_fault_fired"}
                and int(payload.get("time_ns", 0) or 0) > 0
            ):
                events.append(payload)
        return events

    def _assert_phase_sequence(self, *, operation: str, expected: list[str], observed: list[str]) -> None:
        if observed != expected:
            raise ExperimentError(
                f"{operation} phase sequence mismatch:\nexpected={expected}\nobserved={observed}"
            )

    def _run_base_restart_trials(self) -> None:
        self._record_trial_safe(
            kind="base_fallback",
            phase="before_first_action",
            callback=self._trial_base_before_first_action,
        )
        self._record_trial_safe(
            kind="base_fallback",
            phase="checkpoint_after_runtime_dump_without_ancestor",
            callback=self._trial_base_checkpoint_internal,
        )

    def _trial_base_before_first_action(self) -> dict[str, Any]:
        container = self._create_container("base-before-action")
        self._remove_container(container)
        recovered, recovery = self._recover([], label="base-before-action")
        try:
            self._run_steps(recovered, 1)
            final = self._final_snapshot(recovered, label="base-before-action")
            return self._finalize_trial(final, recovery, fault_observed=True)
        finally:
            self._remove_container(recovered)

    def _trial_base_checkpoint_internal(self) -> dict[str, Any]:
        source = self._create_container("base-checkpoint-source")
        candidate_id = self._next_name("base-candidate-step-1")
        self._apply_step(source, 1)
        expected = self._snapshot(source)
        worker = self._run_checkpoint_worker(
            container=source,
            checkpoint_id=candidate_id,
            label="base-checkpoint-fault",
            fault={"exact_phase": "checkpoint_after_runtime_dump"},
        )
        self._remove_container(source)
        candidate = self._ready_ref(candidate_id, step=1, snapshot=expected)
        candidates = [candidate] if candidate else []
        recovered, recovery = self._recover(candidates, label="base-checkpoint-recovery")
        try:
            self._run_steps(recovered, recovery["restored_step"] + 1)
            final = self._final_snapshot(recovered, label="base-checkpoint-internal")
            return self._finalize_trial(
                final,
                recovery,
                fault_observed=worker.fault_observed,
                worker=self._worker_summary(worker),
            )
        finally:
            self._remove_container(recovered)
            self._remove_checkpoint(candidate_id)

    def _run_action_trials(self) -> None:
        for phase in ["before_action", *ACTION_PHASES, "after_action_before_checkpoint"]:
            self._record_trial_safe(
                kind="action_fail_stop",
                phase=phase,
                callback=lambda phase=phase: self._trial_action_phase(phase),
            )

    def _trial_action_phase(self, phase: str) -> dict[str, Any]:
        assert self.shared_checkpoint is not None
        source, _ = self._recover([self.shared_checkpoint], label=f"action-{phase}-source")
        fault_observed = False
        action_result: dict[str, Any] | None = None
        if phase == "before_action":
            self._remove_container(source)
            fault_observed = True
        elif phase == "after_action_before_checkpoint":
            self._apply_step(source, 3)
            self._remove_container(source)
            fault_observed = True
        else:
            action_result = self._apply_step(source, 3, fail_phase=phase)
            deadline = time.monotonic() + 10.0
            while time.monotonic() < deadline and self._container_running(source):
                time.sleep(0.05)
            logs = _run(["docker", "logs", source], check=False).stdout
            fault_observed = (not self._container_running(source)) and (
                f'"phase": "{phase}"' in logs or f'"phase":"{phase}"' in logs
            )
            self._remove_container(source)

        recovered, recovery = self._recover([self.shared_checkpoint], label=f"action-{phase}-recovery")
        try:
            self._run_steps(recovered, 3)
            final = self._final_snapshot(recovered, label=f"action-{phase}")
            return self._finalize_trial(
                final,
                recovery,
                fault_observed=fault_observed,
                action_result=action_result,
            )
        finally:
            self._remove_container(recovered)

    def _run_checkpoint_boundary_trials(self, phases: Iterable[str]) -> None:
        for phase in phases:
            self._record_trial_safe(
                kind="checkpoint_boundary_fail_stop",
                phase=phase,
                callback=lambda phase=phase: self._trial_checkpoint_fault(
                    phase=phase,
                    fault={"exact_phase": phase},
                    strategy="exact_boundary",
                ),
            )

    def _run_checkpoint_timed_trials(self) -> None:
        specs = [
            ("runtime_dump_inside", "checkpoint_before_runtime_dump", 0.05),
            ("rootfs_snapshot_inside", "checkpoint_before_rootfs_snapshot", 0.002),
        ]
        for label, arm_phase, delay_sec in specs:
            self._record_trial_safe(
                kind="checkpoint_timed_fail_stop",
                phase=label,
                callback=lambda label=label, arm_phase=arm_phase, delay_sec=delay_sec: self._trial_checkpoint_fault(
                    phase=label,
                    fault={"arm_phase": arm_phase, "delay_sec": delay_sec},
                    strategy="timed_inside",
                ),
            )
        self._record_trial_safe(
            kind="checkpoint_timed_fail_stop",
            phase="runtime_persist_overlap_window",
            callback=lambda: self._trial_checkpoint_fault(
                phase="runtime_persist_overlap_window",
                fault={
                    "arm_phase": "checkpoint_after_runtime_persist_started",
                    "delay_sec": 0.001,
                    "window_end_phase": "checkpoint_after_runtime_persist_wait",
                    "window_start_phase": "checkpoint_after_runtime_persist_started",
                },
                strategy="timed_inside_async_window",
            ),
        )

    def _trial_checkpoint_fault(
        self,
        *,
        phase: str,
        fault: dict[str, Any],
        strategy: str,
    ) -> dict[str, Any]:
        assert self.shared_checkpoint is not None
        source, _ = self._recover([self.shared_checkpoint], label=f"checkpoint-{phase}-source")
        candidate_id = self._next_name(f"candidate-{phase}")
        self._apply_step(source, 3)
        candidate_snapshot = self._snapshot(source)
        worker = self._run_checkpoint_worker(
            container=source,
            checkpoint_id=candidate_id,
            label=f"checkpoint-{phase}",
            fault=fault,
        )
        self._remove_container(source)
        candidate = self._ready_ref(candidate_id, step=3, snapshot=candidate_snapshot)
        candidates = [item for item in (candidate, self.shared_checkpoint) if item is not None]
        recovered, recovery = self._recover(candidates, label=f"checkpoint-{phase}-recovery")
        try:
            self._run_steps(recovered, recovery["restored_step"] + 1)
            final = self._final_snapshot(recovered, label=f"checkpoint-{phase}")
            phase_hit = self._fault_hit_requested_window(worker, fault)
            return self._finalize_trial(
                final,
                recovery,
                fault_observed=worker.fault_observed and phase_hit,
                candidate_manifest=worker.manifest,
                injection_strategy=strategy,
                requested_fault=fault,
                worker=self._worker_summary(worker),
            )
        finally:
            self._remove_container(recovered)
            self._remove_checkpoint(candidate_id)

    def _run_resume_boundary_trials(self, phases: Iterable[str]) -> None:
        for phase in phases:
            self._record_trial_safe(
                kind="resume_boundary_fail_stop",
                phase=phase,
                callback=lambda phase=phase: self._trial_resume_fault(
                    phase=phase,
                    fault={"exact_phase": phase},
                    strategy="exact_boundary",
                ),
            )

    def _run_resume_timed_trials(self) -> None:
        specs = [
            ("target_create_inside", "resume_before_target_create", 0.01),
            ("upperdir_restore_inside", "resume_before_upperdir_restore", 0.002),
            ("runtime_restore_inside", "resume_before_runtime_restore", 0.05),
        ]
        for label, arm_phase, delay_sec in specs:
            self._record_trial_safe(
                kind="resume_timed_fail_stop",
                phase=label,
                callback=lambda label=label, arm_phase=arm_phase, delay_sec=delay_sec: self._trial_resume_fault(
                    phase=label,
                    fault={"arm_phase": arm_phase, "delay_sec": delay_sec},
                    strategy="timed_inside",
                ),
            )

    def _trial_resume_fault(
        self,
        *,
        phase: str,
        fault: dict[str, Any],
        strategy: str,
    ) -> dict[str, Any]:
        assert self.shared_checkpoint is not None
        interrupted_target = self._next_name(f"resume-interrupted-{phase}")
        self.container_names.add(interrupted_target)
        worker = self._run_resume_worker(
            checkpoint_id=self.shared_checkpoint.checkpoint_id,
            target=interrupted_target,
            label=f"resume-{phase}",
            fault=fault,
        )
        self._remove_container(interrupted_target)
        recovered, recovery = self._recover(
            [self.shared_checkpoint],
            label=f"resume-{phase}-retry",
        )
        try:
            self._run_steps(recovered, recovery["restored_step"] + 1)
            final = self._final_snapshot(recovered, label=f"resume-{phase}")
            phase_hit = self._fault_hit_requested_window(worker, fault)
            return self._finalize_trial(
                final,
                recovery,
                fault_observed=worker.fault_observed and phase_hit,
                injection_strategy=strategy,
                requested_fault=fault,
                worker=self._worker_summary(worker),
            )
        finally:
            self._remove_container(recovered)

    def _recover(
        self,
        candidates: list[CheckpointRef],
        *,
        label: str,
    ) -> tuple[str, dict[str, Any]]:
        attempts: list[dict[str, Any]] = []
        ordered = sorted(candidates, key=lambda item: item.step, reverse=True)
        for candidate in ordered:
            target = self._next_name(f"{label}-step-{candidate.step}")
            self.container_names.add(target)
            worker = self._run_resume_worker(
                checkpoint_id=candidate.checkpoint_id,
                target=target,
                label=f"{label}-{candidate.checkpoint_id}",
            )
            attempt: dict[str, Any] = {
                "checkpoint_id": candidate.checkpoint_id,
                "step": candidate.step,
                "worker": self._worker_summary(worker),
            }
            if worker.exit_code != 0:
                attempt["outcome"] = "restore_error"
                attempts.append(attempt)
                self._remove_container(target)
                continue
            self._cleanup_installed_checkpoint(target, candidate.checkpoint_id)
            try:
                actual = self._snapshot(target)
            except Exception as exc:
                attempt.update({"outcome": "inspection_error", "error": str(exc)})
                attempts.append(attempt)
                self._remove_container(target)
                continue
            runtime_match = (
                actual["continuity_runtime"] == candidate.snapshot["continuity_runtime"]
            )
            filesystem_match = (
                actual["canonical"]["filesystem"]
                == candidate.snapshot["canonical"]["filesystem"]
            )
            attempt.update(
                {
                    "filesystem_match": filesystem_match,
                    "outcome": "restored" if runtime_match and filesystem_match else "snapshot_mismatch",
                    "runtime_match": runtime_match,
                }
            )
            attempts.append(attempt)
            if runtime_match and filesystem_match:
                return target, {
                    "attempts": attempts,
                    "checkpoint_id": candidate.checkpoint_id,
                    "mode": "checkpoint",
                    "restored_step": candidate.step,
                }
            self._remove_container(target)

        container = self._create_container(f"{label}-base-restart")
        return container, {
            "attempts": attempts,
            "checkpoint_id": None,
            "mode": "base_restart",
            "restored_step": 0,
        }

    def _run_checkpoint_worker(
        self,
        *,
        container: str,
        checkpoint_id: str,
        label: str,
        fault: dict[str, Any] | None = None,
    ) -> WorkerResult:
        config = {
            "checkpoint_id": checkpoint_id,
            "container": container,
            "criu_timeout_sec": self.args.criu_timeout_sec,
            "docker_root": str(self.docker_root),
            "fault": fault,
            "runtime_staging_root": str(self.runtime_staging_root),
            "state_root": str(self.state_root),
        }
        result = self._run_isolated_worker(
            target=_checkpoint_worker,
            config=config,
            label=label,
        )
        result.manifest = self._manifest_snapshot(checkpoint_id)
        return result

    def _run_resume_worker(
        self,
        *,
        checkpoint_id: str,
        target: str,
        label: str,
        fault: dict[str, Any] | None = None,
    ) -> WorkerResult:
        config = {
            "checkpoint_id": checkpoint_id,
            "criu_timeout_sec": self.args.criu_timeout_sec,
            "docker_root": str(self.docker_root),
            "fault": fault,
            "state_root": str(self.state_root),
            "target": target,
        }
        return self._run_isolated_worker(target=_resume_worker, config=config, label=label)

    def _run_isolated_worker(
        self,
        *,
        target: Any,
        config: dict[str, Any],
        label: str,
    ) -> WorkerResult:
        log_path = self.worker_log_root / f"{self._next_name(label)}.jsonl"
        config = {**config, "log_path": str(log_path)}
        context = multiprocessing.get_context("fork")
        process = context.Process(target=target, args=(config,), daemon=False)
        started = time.perf_counter()
        process.start()
        process.join(self.args.worker_timeout_sec)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                process.kill()
            process.join(10.0)
            _append_event(log_path, {"event": "parent_timeout", "time_ns": time.time_ns()})
        duration = time.perf_counter() - started
        events = _read_events(log_path)
        fault_observed = any(item.get("event") == "fault_fired" for item in events)
        return WorkerResult(
            exit_code=int(process.exitcode if process.exitcode is not None else -999),
            duration_sec=duration,
            events=events,
            fault_observed=fault_observed,
        )

    def _manifest_snapshot(self, checkpoint_id: str) -> dict[str, Any] | None:
        try:
            return self.store.load_manifest(checkpoint_id).to_dict()
        except Exception:
            return None

    def _ready_ref(
        self,
        checkpoint_id: str,
        *,
        step: int,
        snapshot: dict[str, Any],
    ) -> CheckpointRef | None:
        manifest = self._manifest_snapshot(checkpoint_id)
        if manifest and manifest.get("status") == "ready":
            return CheckpointRef(checkpoint_id=checkpoint_id, step=step, snapshot=snapshot)
        return None

    def _fault_hit_requested_window(self, worker: WorkerResult, fault: dict[str, Any]) -> bool:
        if not worker.fault_observed:
            return False
        exact = str(fault.get("exact_phase") or "")
        if exact:
            return exact in worker.phases
        window_start = str(fault.get("window_start_phase") or "")
        window_end = str(fault.get("window_end_phase") or "")
        if window_start and window_end:
            fired_index = next(
                (
                    index
                    for index, event in enumerate(worker.events)
                    if event.get("event") == "fault_fired"
                ),
                None,
            )
            if fired_index is None:
                return False
            preceding = {
                str(event["phase"])
                for event in worker.events[:fired_index]
                if event.get("event") == "phase"
            }
            return window_start in preceding and window_end not in preceding
        arm = str(fault.get("arm_phase") or "")
        fired_index = next(
            (index for index, event in enumerate(worker.events) if event.get("event") == "fault_fired"),
            None,
        )
        if not arm or fired_index is None:
            return False
        preceding = [
            str(event["phase"])
            for event in worker.events[:fired_index]
            if event.get("event") == "phase"
        ]
        return bool(preceding) and preceding[-1] == arm

    def _finalize_trial(
        self,
        final: dict[str, Any],
        recovery: dict[str, Any],
        *,
        fault_observed: bool,
        **details: Any,
    ) -> dict[str, Any]:
        oracles = final.get("oracles")
        if not isinstance(oracles, dict):
            raise ExperimentError("final snapshot is missing consistency oracle results")
        filesystem = oracles.get("filesystem", {})
        runtime = oracles.get("runtime", {})
        filesystem_match = bool(filesystem.get("match", False))
        runtime_match = bool(runtime.get("match", False))
        consistent = filesystem_match and runtime_match
        if not fault_observed:
            status = "coverage_incomplete"
        elif not consistent:
            status = "final_state_mismatch"
        else:
            status = "pass"
        return {
            "consistent": consistent,
            "consistency_oracles": oracles,
            "fault_observed": fault_observed,
            "filesystem_rsync_match": filesystem_match,
            "final_fingerprint": final["fingerprint"],
            "golden_fingerprint": self.golden["snapshot"]["fingerprint"],
            "recovery": recovery,
            "recovery_mode": recovery["mode"],
            "runtime_differential_match": runtime_match,
            "status": status,
            **details,
        }

    def _record_trial_safe(self, *, kind: str, phase: str, callback: Any) -> dict[str, Any]:
        started = time.perf_counter()
        print(f"[full-consistency] start kind={kind} phase={phase}", flush=True)
        try:
            payload = callback()
        except BaseException as exc:
            payload = {
                "error": f"{type(exc).__name__}: {exc}",
                "fault_observed": False,
                "status": "execution_error",
                "traceback": traceback.format_exc(limit=30),
            }
        trial = {
            "duration_sec": time.perf_counter() - started,
            "kind": kind,
            "phase": phase,
            **payload,
        }
        self.trials.append(trial)
        print(
            f"[full-consistency] done kind={kind} phase={phase} "
            f"status={trial['status']} recovery={trial.get('recovery_mode', '-')}",
            flush=True,
        )
        self._write_result(self._build_result(duration_sec=None, partial=True))
        return trial

    def _worker_summary(self, worker: WorkerResult) -> dict[str, Any]:
        error = next(
            (item.get("error") for item in reversed(worker.events) if item.get("event") == "worker_error"),
            None,
        )
        fired = next(
            (item for item in worker.events if item.get("event") == "fault_fired"),
            None,
        )
        return {
            "duration_sec": worker.duration_sec,
            "error": error,
            "exit_code": worker.exit_code,
            "fault_event": fired,
            "phases": worker.phases,
        }

    def _container_running(self, container: str) -> bool:
        result = _run(
            ["docker", "inspect", "--format", "{{json .State.Running}}", container],
            check=False,
            timeout=30,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _container_exists(self, container: str) -> bool:
        return _run(["docker", "inspect", container], check=False, timeout=30).returncode == 0

    def _remove_container(self, container: str) -> None:
        if not container:
            return
        for _ in range(20):
            result = _run(["docker", "rm", "-f", container], check=False, timeout=60)
            if result.returncode == 0 or not self._container_exists(container):
                self.container_names.discard(container)
                return
            time.sleep(0.1)
        self.container_names.add(container)

    def _cleanup_containers(self) -> None:
        for name in sorted(self.container_names):
            self._remove_container(name)

    def _cleanup_installed_checkpoint(self, container: str, checkpoint_id: str) -> None:
        _run(["docker", "checkpoint", "rm", container, checkpoint_id], check=False, timeout=60)
        inspect = _run(
            ["docker", "inspect", "--format", "{{.Id}}", container],
            check=False,
            timeout=30,
        )
        container_id = inspect.stdout.strip()
        if container_id:
            installed = self.docker_root / "containers" / container_id / "checkpoints" / checkpoint_id
            shutil.rmtree(installed, ignore_errors=True)

    def _remove_checkpoint(self, checkpoint_id: str) -> None:
        shutil.rmtree(self.store.checkpoint_dir(checkpoint_id), ignore_errors=True)
        for path in self.runtime_staging_root.glob(f"{checkpoint_id}-runtime-*"):
            shutil.rmtree(path, ignore_errors=True)

    def _build_result(self, *, duration_sec: float | None, partial: bool = False) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for trial in self.trials:
            status = str(trial.get("status", "unknown"))
            counts[status] = counts.get(status, 0) + 1
        boundary_expected = set(CHECKPOINT_PHASES + RESUME_PHASES + ACTION_PHASES)
        boundary_covered = {
            str(trial["phase"])
            for trial in self.trials
            if trial.get("fault_observed")
            and trial.get("phase") in boundary_expected
            and trial.get("status") in {"pass", "final_state_mismatch"}
        }
        wallclock_trials = [
            trial for trial in self.trials if trial.get("kind") == "wall_clock_fail_stop"
        ]
        wallclock_covered = {
            str(trial["observed_stage"])
            for trial in wallclock_trials
            if trial.get("fault_observed") and trial.get("observed_stage")
        }
        wallclock_stage_counts: dict[str, int] = {}
        for trial in wallclock_trials:
            stage = str(trial.get("observed_stage") or "fault_not_observed")
            wallclock_stage_counts[stage] = wallclock_stage_counts.get(stage, 0) + 1
        return {
            "config": {
                "heap_mb": self.args.heap_mb,
                "image": self.args.image,
                "payload_mb": self.args.payload_mb,
                "seed": self.args.seed,
                "steps": self.args.steps,
                "wall_clock_base_fraction": self.args.wall_clock_base_fraction,
                "wall_clock_seed": self.args.wall_clock_seed,
                "wall_clock_trials": self.args.wall_clock_trials,
            },
            "consistency_method": {
                "filesystem": {
                    "comparator": "rsync --archive --hard-links --acls --xattrs --checksum --dry-run --delete --itemize-changes --numeric-ids",
                    "pass_condition": "rsync exits zero and reports no itemized differences",
                },
                "runtime": {
                    "comparator": "cmp --silent",
                    "method": "golden-run differential testing",
                    "observation": (
                        "self-validating binary artifact containing scalar state, the real "
                        "effects.log FD offset, cross-coupled state chain, canonical exactly-once "
                        "action transcript, and the complete application heap"
                    ),
                    "pass_condition": (
                        "the artifact passes transcript validation and is byte-identical "
                        "to the fault-free golden artifact"
                    ),
                },
                "trial_pass_condition": "fault observed and both independent oracles pass",
            },
            "coverage": {
                "boundary_covered": sorted(boundary_covered),
                "boundary_expected": sorted(boundary_expected),
                "boundary_missing": sorted(boundary_expected - boundary_covered),
                "checkpoint_observed_no_fault": self.observed_checkpoint_phases,
                "resume_observed_no_fault": self.observed_resume_phases,
                "wall_clock": {
                    "control_duration_sec": dict(self.wallclock_control_durations),
                    "covered_stages": sorted(wallclock_covered),
                    "expected_stages": sorted(self.wallclock_expected_stages),
                    "missing_stages": sorted(
                        self.wallclock_expected_stages - wallclock_covered
                    ),
                    "stage_sample_counts": dict(sorted(wallclock_stage_counts.items())),
                },
            },
            "duration_sec": duration_sec,
            "diagnostic_golden_fingerprint": self.golden.get("snapshot", {}).get("fingerprint"),
            "golden_runtime_artifact": self.golden.get("runtime_artifact"),
            "oracle_calibration": self.oracle_calibration,
            "partial": partial,
            "run_id": self.run_id,
            "scope": {
                "included": [
                    "complete deterministic application runtime artifact compared to a golden run by differential testing and cmp",
                    "writable filesystem tree compared to a golden run by rsync checksum dry-run",
                    "cross-coupled runtime/filesystem hash chains and an exactly-once action transcript",
                    "phase-independent wall-clock fail-stop injection with post-hoc stage classification",
                    "optional directed checkpoint and restore boundary diagnostics",
                    "latest-ready, earlier-ready, and base-restart recovery policy",
                ],
                "excluded": [
                    "kernel-internal state not exposed by the deterministic application",
                    "pre-existing docker exec -d processes",
                    "live TCP connections",
                    "mounted volumes",
                    "host or VM power loss and storage durability",
                ],
            },
            "status_counts": counts,
            "trials": self.trials,
        }

    def _write_result(self, payload: dict[str, Any]) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        tmp = self.output_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        os.replace(tmp, self.output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=WORKSPACE_ROOT / "export" / f"full_checkpoint_consistency_{time.strftime('%Y%m%d_%H%M%S')}",
    )
    parser.add_argument("--run-id", default="")
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--no-pull", action="store_true")
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--heap-mb", type=int, default=16)
    parser.add_argument("--payload-mb", type=int, default=16)
    parser.add_argument("--container-memory", default="1g")
    parser.add_argument("--startup-timeout-sec", type=float, default=30.0)
    parser.add_argument("--criu-timeout-sec", type=int, default=120)
    parser.add_argument("--worker-timeout-sec", type=float, default=240.0)
    parser.add_argument(
        "--wall-clock-trials",
        type=int,
        default=0,
        help="number of phase-independent, stratified wall-clock fault trials",
    )
    parser.add_argument("--wall-clock-seed", type=int, default=20260721)
    parser.add_argument(
        "--wall-clock-base-fraction",
        type=float,
        default=0.25,
        help="fraction of wall-clock trials that start without a ready checkpoint",
    )
    parser.add_argument(
        "--wall-clock-only",
        action="store_true",
        help="run wall-clock campaign without directed phase-triggered diagnostics",
    )
    parser.add_argument("--skip-checkpoint-faults", action="store_true")
    parser.add_argument("--skip-resume-faults", action="store_true")
    parser.add_argument("--skip-timed-faults", action="store_true")
    parser.add_argument("--keep-artifacts", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero when any trial is not pass (control included)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.steps < 6:
        raise SystemExit("--steps must be at least 6 to exercise all filesystem mutations")
    if args.wall_clock_trials < 0:
        raise SystemExit("--wall-clock-trials must be non-negative")
    if not 0.0 <= args.wall_clock_base_fraction <= 1.0:
        raise SystemExit("--wall-clock-base-fraction must be between 0 and 1")
    if args.wall_clock_only and args.wall_clock_trials == 0:
        raise SystemExit("--wall-clock-only requires --wall-clock-trials > 0")
    experiment = FullConsistencyExperiment(args)
    try:
        result = experiment.run()
    except BaseException as exc:
        experiment._cleanup_containers()  # pylint: disable=protected-access
        fatal = {
            "error": f"{type(exc).__name__}: {exc}",
            "status": "fatal_setup_or_control_error",
            "traceback": traceback.format_exc(limit=40),
        }
        experiment._write_result(fatal)  # pylint: disable=protected-access
        print(f"[full-consistency] FATAL: {fatal['error']}", file=sys.stderr)
        print(f"[full-consistency] result={experiment.output_path}", file=sys.stderr)
        return 2

    counts = result["status_counts"]
    print(f"[full-consistency] status_counts={json.dumps(counts, sort_keys=True)}")
    print(f"[full-consistency] result={experiment.output_path}")
    if args.strict and any(trial.get("status") != "pass" for trial in result["trials"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
