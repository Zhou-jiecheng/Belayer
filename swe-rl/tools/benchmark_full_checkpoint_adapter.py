#!/usr/bin/env python3
"""Real-Docker correctness and performance gate for Belayer checkpoint backends.

Run as root because docker-full-checkpoint reads Docker's metadata and overlay
directories directly. The script uses the real Flask endpoints in
``swe_exec_server.py`` and cleans every container/checkpoint that it creates.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import textwrap
import time
import types
import uuid
from pathlib import Path
from typing import Any


WORKLOAD_SOURCE = textwrap.dedent(
    r"""
    import hashlib
    import json
    import os
    import time
    import uuid

    root = "/workspace/belayer-full-workload"
    os.makedirs(root, exist_ok=True)
    boot_uuid = uuid.uuid4().hex
    secret = os.urandom(64)
    secret_sha256 = hashlib.sha256(secret).hexdigest()
    memory_mb = int(os.environ.get("BELAYER_MEMORY_MB", "16"))
    touched = bytearray(memory_mb * 1024 * 1024)
    for offset in range(0, len(touched), 4096):
        touched[offset] = (offset // 4096) % 251
    counter = 0
    accumulator = int.from_bytes(secret[:8], "little")
    events = open(os.path.join(root, "events.log"), "a", buffering=1)
    while True:
        counter += 1
        accumulator = (accumulator * 6364136223846793005 + counter) & ((1 << 64) - 1)
        events.write(f"{counter}:{accumulator}\n")
        events.flush()
        state = {
            "boot_uuid": boot_uuid,
            "secret_sha256": secret_sha256,
            "counter": counter,
            "accumulator": accumulator,
            "pid": os.getpid(),
            "memory_mb": memory_mb,
            "events_offset": events.tell(),
            "wall_time": time.time(),
        }
        tmp = os.path.join(root, "state.json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(state, handle, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, os.path.join(root, "state.json"))
        time.sleep(0.05)
    """
).strip()


def _run(args: list[str], *, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
    if check and result.returncode != 0:
        raise RuntimeError(
            f"command failed rc={result.returncode}: {' '.join(args)}\n"
            f"stdout={result.stdout[-1000:]}\nstderr={result.stderr[-1000:]}"
        )
    return result


def _docker_root() -> Path:
    return Path(_run(["docker", "info", "--format", "{{.DockerRootDir}}"]).stdout.strip())


def _install_flask_stub_if_missing() -> None:
    try:
        import flask  # noqa: F401

        return
    except ModuleNotFoundError:
        pass

    module = types.ModuleType("flask")
    request_state = types.SimpleNamespace(payload={})
    request_state.get_json = lambda force=False: request_state.payload

    class _Response:
        def __init__(self, payload: Any, status_code: int = 200):
            self._payload = payload
            self.status_code = status_code

        def get_json(self):
            return self._payload

    class _TestClient:
        def __init__(self, app):
            self.app = app

        def post(self, path: str, json: dict[str, Any] | None = None):
            request_state.payload = json or {}
            result = self.app.routes[("POST", path)]()
            if isinstance(result, tuple):
                payload, status_code = result
            else:
                payload, status_code = result, 200
            return _Response(payload, int(status_code))

    class _Flask:
        def __init__(self, _name: str):
            self.routes: dict[tuple[str, str], Any] = {}

        def get(self, path: str):
            return lambda func: self._register("GET", path, func)

        def post(self, path: str):
            return lambda func: self._register("POST", path, func)

        def _register(self, method: str, path: str, func):
            self.routes[(method, path)] = func
            return func

        def test_client(self):
            return _TestClient(self)

        def run(self, *args, **kwargs):
            del args, kwargs

    module.Flask = _Flask
    module.jsonify = lambda payload=None, **kwargs: payload if payload is not None else kwargs
    module.request = request_state
    sys.modules["flask"] = module


def _prepare_server(args: argparse.Namespace, state_root: Path):
    project_root = Path(__file__).resolve().parents[3]
    belayer_root = Path(__file__).resolve().parents[2]
    full_checkpoint_root = belayer_root / "docker-full-checkpoint"
    if not (full_checkpoint_root / "pyproject.toml").is_file():
        full_checkpoint_root = project_root / "docker-full-checkpoint"
    os.environ["SWE_CHECKPOINT_BACKEND"] = "full"
    os.environ["SWE_CHECKPOINT_DIR"] = str(state_root / "belayer")
    os.environ["SWE_FULL_CHECKPOINT_STATE_ROOT"] = str(state_root / "full-state")
    os.environ["SWE_FULL_CHECKPOINT_PROJECT_ROOT"] = str(full_checkpoint_root)
    os.environ["SWE_FULL_CHECKPOINT_DOCKER_ROOT"] = str(_docker_root())
    os.environ["SWE_FULL_CHECKPOINT_RUNTIME_STAGING_ROOT"] = str(args.runtime_staging_root)
    os.environ["SWE_CHECKPOINT_MAX_INFLIGHT"] = "1"
    os.environ["SWE_CHECKPOINT_MIN_READY_LATENCY_SEC"] = "0"
    server_root = project_root / "Belayer" / "swe-rl" / "server"
    sys.path.insert(0, str(server_root))
    _install_flask_stub_if_missing()
    return importlib.import_module("swe_exec_server")


def _post(client, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = client.post(path, json=payload)
    body = response.get_json()
    if not isinstance(body, dict):
        raise RuntimeError(f"non-object response from {path}: {body!r}")
    if not body.get("ok", False):
        raise RuntimeError(f"request failed path={path} status={response.status_code}: {body}")
    return body


def _docker_exec(container_id: str, *command: str, timeout: int = 120) -> str:
    return _run(["docker", "exec", container_id, *command], timeout=timeout).stdout


def _read_state(container_id: str) -> dict[str, Any] | None:
    result = _run(
        [
            "docker",
            "exec",
            container_id,
            "python3",
            "-c",
            "import pathlib; print(pathlib.Path('/workspace/belayer-full-workload/state.json').read_text())",
        ],
        timeout=30,
        check=False,
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _wait_state(
    container_id: str,
    *,
    timeout_sec: float = 30.0,
    minimum_counter: int = 1,
    boot_uuid: str | None = None,
    minimum_wall_time: float = 0.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_sec
    last: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        last = _read_state(container_id)
        if (
            last
            and int(last.get("counter", 0)) >= minimum_counter
            and (boot_uuid is None or last.get("boot_uuid") == boot_uuid)
            and float(last.get("wall_time", 0.0) or 0.0) >= minimum_wall_time
        ):
            return last
        time.sleep(0.1)
    raise AssertionError(
        f"workload did not advance container={container_id[:12]} "
        f"minimum_counter={minimum_counter} boot_uuid={boot_uuid} last={last}"
    )


def _create_workload_container(server, *, image: str, memory_mb: int) -> str:
    name = f"belayer-full-bench-{uuid.uuid4().hex[:12]}"
    result = _run(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "--network",
            "host",
            "--memory",
            "4g",
            "--pids-limit",
            "256",
            "--workdir",
            "/workspace",
            "--env",
            f"BELAYER_MEMORY_MB={memory_mb}",
            image,
            "python3",
            "-u",
            "-c",
            WORKLOAD_SOURCE,
        ],
        timeout=300,
    )
    container_id = result.stdout.strip()
    with server._lock:  # pylint: disable=protected-access
        server._active_containers[container_id] = {  # pylint: disable=protected-access
            "name": name,
            "image": image,
            "cwd": "/workspace",
            "runtime_env": {},
            "created_at": time.time(),
            "pooled": False,
            "acquisition": "full_checkpoint_benchmark",
            "create_time_sec": 0.0,
            "reset_time_sec": 0.0,
        }
    return container_id


def _prepare_upperdir(container_id: str, *, upper_mb: int, token: str) -> None:
    _docker_exec(
        container_id,
        "python3",
        "-c",
        (
            "from pathlib import Path; "
            "import os; "
            "root=Path('/workspace/belayer-full-workload'); root.mkdir(parents=True, exist_ok=True); "
            f"root.joinpath('token.txt').write_text({token!r}); "
            f"root.joinpath('upper.bin').write_bytes(b'x' * ({upper_mb} * 1024 * 1024)); "
            "root.joinpath('executable.sh').write_text('#!/bin/sh\\nexit 0\\n'); "
            "root.joinpath('executable.sh').chmod(0o751); "
            "link=root.joinpath('token-link'); link.unlink(missing_ok=True); link.symlink_to('token.txt'); "
            "base=Path('/etc/debian_version'); base.unlink() if base.exists() else None"
        ),
        timeout=300,
    )


def _filesystem_state(container_id: str) -> dict[str, Any]:
    script = (
        "import hashlib,json,os,stat; from pathlib import Path; "
        "root=Path('/workspace/belayer-full-workload'); blob=root/'upper.bin'; "
        "print(json.dumps({"
        "'upper_sha256':hashlib.sha256(blob.read_bytes()).hexdigest(),"
        "'upper_size':blob.stat().st_size,"
        "'token':(root/'token.txt').read_text(),"
        "'symlink_target':os.readlink(root/'token-link'),"
        "'executable_mode':stat.S_IMODE((root/'executable.sh').stat().st_mode),"
        "'deleted_base_absent':not Path('/etc/debian_version').exists()"
        "},sort_keys=True))"
    )
    payload = json.loads(_docker_exec(container_id, "python3", "-c", script))
    if not isinstance(payload, dict):
        raise AssertionError(f"invalid filesystem state: {payload!r}")
    return payload


def _resource_limits(container_id: str) -> dict[str, int]:
    raw = _run(
        ["docker", "inspect", "--format", "{{json .HostConfig}}", container_id],
        timeout=30,
    ).stdout
    host_config = json.loads(raw)
    return {
        "memory_bytes": int(host_config.get("Memory") or 0),
        "pids_limit": int(host_config.get("PidsLimit") or 0),
    }


def _installed_checkpoints(container_id: str) -> list[str]:
    result = _run(
        ["docker", "checkpoint", "ls", "--format", "{{.Name}}", container_id],
        timeout=30,
        check=False,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.999999)))
    return ordered[index]


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"iterations": len(rows)}
    for key in ("checkpoint_wall_sec", "rerun_wall_sec", "size_bytes"):
        values = [float(row[key]) for row in rows]
        summary[key] = {
            "min": min(values),
            "p50": statistics.median(values),
            "p95": _percentile(values, 0.95),
            "max": max(values),
            "mean": statistics.fmean(values),
        }
    return summary


def _one_iteration(
    server,
    client,
    *,
    backend: str,
    index: int,
    image: str,
    memory_mb: int,
    upper_mb: int,
) -> dict[str, Any]:
    lease_id = f"bench-{backend}-{index}-{uuid.uuid4().hex[:8]}"
    token = f"token-{uuid.uuid4().hex}"
    checkpoint_id = ""
    known_containers: set[str] = set()
    artifact_dir = ""
    source_id = ""
    target_id = ""
    try:
        source_id = _create_workload_container(
            server, image=image, memory_mb=memory_mb
        )
        known_containers.add(source_id)
        _wait_state(source_id, minimum_counter=2)
        _prepare_upperdir(source_id, upper_mb=upper_mb, token=token)
        before = _wait_state(source_id, minimum_counter=3)
        filesystem_before = _filesystem_state(source_id)

        checkpoint_started = time.perf_counter()
        checkpoint = _post(
            client,
            "/container/checkpoint/create",
            {
                "lease_id": lease_id,
                "container_id": source_id,
                "generation": 0,
                "instance_id": lease_id,
                "cwd": "/workspace",
                "step_idx": 1,
                "command_seq": 1,
                "policy": "benchmark",
                "reason": "adapter_validation",
                "checkpoint_backend": backend,
            },
        )
        checkpoint_wall = time.perf_counter() - checkpoint_started
        checkpoint_finished_wall = time.time()
        checkpoint_id = str(checkpoint["checkpoint_id"])
        record = server._CHECKPOINTS.get_checkpoint(checkpoint_id)  # pylint: disable=protected-access
        artifact_dir = str((record or {}).get("full_checkpoint_artifact_dir", "") or "")

        source_after: dict[str, Any] | None = None
        source_gap_sec: float | None = None
        if backend == "full":
            if str(checkpoint.get("container_id")) != source_id:
                raise AssertionError("transparent source resume changed the Docker container ID")
            source_after = _wait_state(
                source_id,
                minimum_counter=int(before["counter"]) + 1,
                boot_uuid=str(before["boot_uuid"]),
                minimum_wall_time=checkpoint_finished_wall,
            )
            if source_after["secret_sha256"] != before["secret_sha256"]:
                raise AssertionError("in-place resume changed the in-memory secret")
            if _filesystem_state(source_id) != filesystem_before:
                raise AssertionError("in-place resume changed the checkpointed filesystem")
            source_gap_sec = float(source_after["wall_time"]) - float(before["wall_time"])

        # Simulate the fail-stop that Belayer's rerun path must recover from.
        _run(["docker", "rm", "-f", source_id], timeout=120)
        rerun_started = time.perf_counter()
        rerun = _post(
            client,
            "/container/rerun",
            {
                "lease_id": lease_id,
                "checkpoint_id": checkpoint_id,
                "old_container_id": source_id,
                "cwd": "/workspace",
                "timeout": 300,
            },
        )
        rerun_wall = time.perf_counter() - rerun_started
        rerun_finished_wall = time.time()
        target_id = str(rerun["new_container_id"])
        known_containers.add(target_id)

        full_correctness: dict[str, Any] | None = None
        if backend == "full":
            restored = _wait_state(
                target_id,
                minimum_counter=int(before["counter"]) + 1,
                boot_uuid=str(before["boot_uuid"]),
                minimum_wall_time=rerun_finished_wall,
            )
            filesystem_restored = _filesystem_state(target_id)
            limits = _resource_limits(target_id)
            installed = _installed_checkpoints(target_id)
            if target_id == source_id:
                raise AssertionError("fault recovery did not switch to a new Docker ID")
            if restored["secret_sha256"] != before["secret_sha256"]:
                raise AssertionError("fault recovery changed the in-memory secret")
            if filesystem_restored != filesystem_before:
                raise AssertionError(
                    "fault recovery changed filesystem state: "
                    f"before={filesystem_before} restored={filesystem_restored}"
                )
            if limits != {"memory_bytes": 4 * 1024**3, "pids_limit": 256}:
                raise AssertionError(f"restored resource limits changed: {limits}")
            if checkpoint_id in installed:
                raise AssertionError(
                    f"installed Docker checkpoint copy leaked in target: {installed}"
                )
            manifest = json.loads(
                (Path(artifact_dir) / "manifest.json").read_text(encoding="utf-8")
            )
            if manifest.get("status") != "ready":
                raise AssertionError(f"full manifest is not ready: {manifest.get('status')}")
            full_correctness = {
                "boot_uuid_preserved": restored["boot_uuid"] == before["boot_uuid"],
                "secret_preserved": restored["secret_sha256"] == before["secret_sha256"],
                "counter_advanced": int(restored["counter"]) > int(before["counter"]),
                "filesystem_preserved": filesystem_restored == filesystem_before,
                "upper_sha256": filesystem_restored["upper_sha256"],
                "upper_size": filesystem_restored["upper_size"],
                "token_preserved": filesystem_restored["token"] == token,
                "symlink_preserved": filesystem_restored["symlink_target"] == "token.txt",
                "mode_preserved": filesystem_restored["executable_mode"] == 0o751,
                "whiteout_preserved": filesystem_restored["deleted_base_absent"],
                "resource_limits": limits,
                "installed_checkpoint_cleanup": checkpoint_id not in installed,
                "manifest_status": manifest.get("status"),
            }

        row = {
            "backend": backend,
            "iteration": index,
            "checkpoint_id": checkpoint_id,
            "source_container_id": source_id,
            "target_container_id": target_id,
            "checkpoint_wall_sec": checkpoint_wall,
            "rerun_wall_sec": rerun_wall,
            "size_bytes": int(checkpoint.get("size_bytes") or 0),
            "full_checkpoint_create_sec": checkpoint.get("full_checkpoint_create_sec"),
            "full_checkpoint_source_resume_sec": checkpoint.get(
                "full_checkpoint_source_resume_sec"
            ),
            "full_checkpoint_timings_sec": checkpoint.get("full_checkpoint_timings_sec"),
            "source_visible_gap_sec": source_gap_sec,
            "correctness": full_correctness,
        }
        return row
    finally:
        for container in sorted(known_containers | ({target_id, source_id} - {""})):
            _run(["docker", "rm", "-f", container], timeout=120, check=False)
            with server._lock:  # pylint: disable=protected-access
                server._active_containers.pop(container, None)  # pylint: disable=protected-access
        if checkpoint_id:
            try:
                response = client.post(
                    "/container/checkpoint/delete", json={"checkpoint_id": checkpoint_id}
                )
                body = response.get_json() or {}
                if not body.get("ok", False):
                    raise RuntimeError(f"checkpoint cleanup failed: {body}")
            except Exception as exc:
                print(f"WARNING: failed to clean checkpoint {checkpoint_id}: {exc}", file=sys.stderr)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default="python:3.12-slim")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--memory-mb", type=int, default=16)
    parser.add_argument("--upper-mb", type=int, default=8)
    parser.add_argument(
        "--backends",
        default="full,legacy",
        help="Comma-separated subset of full,legacy",
    )
    parser.add_argument(
        "--state-root",
        type=Path,
        default=Path("/var/tmp/belayer-full-checkpoint-adapter"),
    )
    parser.add_argument(
        "--runtime-staging-root",
        type=Path,
        default=Path("/dev/shm/docker-full-checkpoint"),
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--keep-state", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if os.geteuid() != 0:
        print("ERROR: run this validation with sudo/root", file=sys.stderr)
        return 2
    backends = [item.strip() for item in args.backends.split(",") if item.strip()]
    if not backends or any(item not in {"full", "legacy"} for item in backends):
        raise ValueError("--backends must contain full and/or legacy")
    if _run(["docker", "image", "inspect", args.image], check=False).returncode != 0:
        raise RuntimeError(f"test image is not available locally: {args.image}")

    run_root = args.state_root / f"run-{int(time.time())}-{os.getpid()}"
    run_root.mkdir(parents=True, exist_ok=False)
    try:
        server = _prepare_server(args, run_root)
    except Exception:
        if not args.keep_state:
            shutil.rmtree(run_root, ignore_errors=True)
        raise
    client = server.app.test_client()
    rows: list[dict[str, Any]] = []
    started_at = time.time()
    try:
        for backend in backends:
            for index in range(max(1, args.iterations)):
                row = _one_iteration(
                    server,
                    client,
                    backend=backend,
                    index=index,
                    image=args.image,
                    memory_mb=max(0, args.memory_mb),
                    upper_mb=max(0, args.upper_mb),
                )
                rows.append(row)
                print(json.dumps(row, sort_keys=True))

        summaries = {
            backend: _summarize([row for row in rows if row["backend"] == backend])
            for backend in backends
        }
        payload = {
            "ok": True,
            "started_at": started_at,
            "finished_at": time.time(),
            "host": {
                "docker_root": str(_docker_root()),
                "image": args.image,
                "memory_mb": args.memory_mb,
                "upper_mb": args.upper_mb,
                "runtime_staging_root": str(args.runtime_staging_root),
            },
            "rows": rows,
            "summary": summaries,
        }
        output = args.output or (
            args.state_root / f"result-{int(started_at)}-{os.getpid()}.json"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"ok": True, "output": str(output), "summary": summaries}, indent=2))
        return 0
    finally:
        if not args.keep_state:
            # Preserve an explicitly requested output outside run_root, then
            # remove all checkpoint metadata/artifacts generated by this run.
            if not args.output or run_root not in args.output.parents:
                shutil.rmtree(run_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
