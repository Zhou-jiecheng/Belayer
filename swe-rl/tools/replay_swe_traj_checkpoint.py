from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[2]
SWE_RL_ROOT = Path(__file__).resolve().parents[1]
SLIME_ROOT = REPO_ROOT / "slime"

for path in (SLIME_ROOT, SWE_RL_ROOT):
    sys.path.insert(0, str(path))

from swe_utils import get_docker_image_name

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


DEFAULT_CONFIG_PATH = SWE_RL_ROOT / "swebench.yaml"
DEFAULT_GC_MIN_CHECKPOINT_COUNT = 100


class ReplayOpError(RuntimeError):
    def __init__(self, op_name: str, payload: dict[str, Any]):
        self.op_name = op_name
        self.payload = payload
        super().__init__(f"{op_name} failed: {payload}")


class ReplayEnvClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.default_http_max_retries = int(os.getenv("SWE_ENV_HTTP_MAX_RETRIES", "3"))
        self.allocate_http_max_retries = int(os.getenv("SWE_ALLOCATE_HTTP_MAX_RETRIES", "1"))
        self.default_app_max_retries = int(os.getenv("SWE_ENV_APP_MAX_RETRIES", "3"))
        self.allocate_app_max_retries = int(os.getenv("SWE_ALLOCATE_APP_MAX_RETRIES", "60"))
        self.app_error_retry_delay_sec = float(os.getenv("SWE_ENV_APP_RETRY_DELAY_SEC", "1.0"))
        self.app_error_retry_max_delay_sec = float(os.getenv("SWE_ENV_APP_RETRY_MAX_DELAY_SEC", "5.0"))
        self.exec_paused_max_retries = int(os.getenv("SWE_EXEC_PAUSED_MAX_RETRIES", "30"))
        self.exec_paused_retry_delay_sec = float(os.getenv("SWE_EXEC_PAUSED_RETRY_DELAY_SEC", "1.0"))

    @staticmethod
    def _is_retryable_app_error(out: dict[str, Any]) -> bool:
        if out.get("ok", False):
            return False
        if bool(out.get("retryable", False)):
            return True
        error_text = str(out.get("error", "") or "").lower()
        non_retryable_markers = (
            "unknown lease_id",
            "lease_id required",
            "image is required",
            "command required",
            "invalid lease_id",
            "not initialized",
            "checkpoint_id is required",
        )
        return not any(marker in error_text for marker in non_retryable_markers)

    @staticmethod
    def _is_paused_exec_result(out: dict[str, Any]) -> bool:
        if not out.get("ok", False):
            return False
        output = str(out.get("output", "") or "").lower()
        return (
            "is paused" in output
            and "unpause the container before exec" in output
        )

    def _post_blocking(self, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
        req = urllib_request.Request(
            f"{self.base_url}/{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read().decode("utf-8")
        except urllib_error.HTTPError as exc:
            raw = exc.read().decode("utf-8", errors="replace")
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise RuntimeError(f"Unexpected response payload for {path}: {parsed!r}")
        return parsed

    async def _post_with_retry(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        op_name: str,
        timeout: float,
        http_max_retries: int,
        app_max_retries: int,
    ) -> dict[str, Any]:
        last_error = ""
        for http_attempt in range(1, max(1, http_max_retries) + 1):
            try:
                out = await asyncio.to_thread(self._post_blocking, path, payload, timeout)
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                if http_attempt >= max(1, http_max_retries):
                    raise RuntimeError(f"{op_name} HTTP failed after {http_attempt} attempt(s): {last_error}") from exc
                await asyncio.sleep(min(float(http_attempt), 3.0))
                continue

            if out.get("ok", False):
                return out

            if not self._is_retryable_app_error(out):
                raise ReplayOpError(op_name, out)

            last_out = out
            for app_attempt in range(1, max(1, app_max_retries)):
                delay = min(
                    self.app_error_retry_delay_sec * (2 ** (app_attempt - 1)),
                    self.app_error_retry_max_delay_sec,
                )
                retry_after = last_out.get("retry_after_sec")
                if retry_after is not None:
                    try:
                        delay = max(delay, float(retry_after))
                    except (TypeError, ValueError):
                        pass
                await asyncio.sleep(delay)
                last_out = await asyncio.to_thread(self._post_blocking, path, payload, timeout)
                if last_out.get("ok", False):
                    return last_out
                if not self._is_retryable_app_error(last_out):
                    raise ReplayOpError(op_name, last_out)
            raise ReplayOpError(op_name, last_out)

        raise RuntimeError(f"{op_name} failed: {last_error or 'unknown error'}")

    async def allocate(self, image: str, instance_id: str, cwd: str) -> dict[str, Any]:
        return await self._post_with_retry(
            path="allocate",
            payload={"image": image, "instance_id": instance_id, "cwd": cwd},
            op_name="allocate",
            timeout=180.0,
            http_max_retries=self.allocate_http_max_retries,
            app_max_retries=self.allocate_app_max_retries,
        )

    async def heartbeat(self, lease_id: str) -> dict[str, Any]:
        return await self._post_with_retry(
            path="heartbeat",
            payload={"lease_id": lease_id},
            op_name="heartbeat",
            timeout=30.0,
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )

    async def exec(self, lease_id: str, command: str, cwd: str, timeout: int) -> dict[str, Any]:
        payload = {"lease_id": lease_id, "command": command, "cwd": cwd, "timeout": timeout, "env": {}}
        last_result: dict[str, Any] | None = None
        for attempt in range(max(1, self.exec_paused_max_retries) + 1):
            out = await self._post_with_retry(
                path="exec",
                payload=payload,
                op_name="exec",
                timeout=float(timeout + 30),
                http_max_retries=self.default_http_max_retries,
                app_max_retries=self.default_app_max_retries,
            )
            last_result = out
            if not self._is_paused_exec_result(out):
                return out
            if attempt >= max(1, self.exec_paused_max_retries):
                return out
            await asyncio.sleep(max(0.05, self.exec_paused_retry_delay_sec))
        return last_result or {}

    async def close(self, lease_id: str) -> dict[str, Any]:
        return await self._post_with_retry(
            path="close",
            payload={"lease_id": lease_id},
            op_name="close",
            timeout=30.0,
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )

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
        return await self._post_with_retry(
            path="checkpoint/create",
            payload=payload,
            op_name="checkpoint_create",
            timeout=30.0,
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )

    async def checkpoint_probe(self, lease_id: str) -> dict[str, Any]:
        return await self._post_with_retry(
            path="checkpoint/probe",
            payload={"lease_id": lease_id},
            op_name="checkpoint_probe",
            timeout=30.0,
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )

    async def checkpoint_status(self, lease_id: str, checkpoint_id: str) -> dict[str, Any]:
        return await self._post_with_retry(
            path="checkpoint/status",
            payload={"lease_id": lease_id, "checkpoint_id": checkpoint_id},
            op_name="checkpoint_status",
            timeout=30.0,
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )

    async def checkpoint_list(self, lease_id: str | None = None) -> dict[str, Any]:
        return await self._post_with_retry(
            path="checkpoint/list",
            payload={"lease_id": lease_id},
            op_name="checkpoint_list",
            timeout=30.0,
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )

    async def checkpoint_gc(self, lease_id: str | None, keep_latest: int, dry_run: bool) -> dict[str, Any]:
        return await self._post_with_retry(
            path="checkpoint/gc",
            payload={"lease_id": lease_id, "keep_latest": int(keep_latest), "dry_run": bool(dry_run)},
            op_name="checkpoint_gc",
            timeout=120.0,
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )

    async def checkpoint_gc_drain(self, timeout_sec: float = 600.0, poll_interval_sec: float = 0.1) -> dict[str, Any]:
        return await self._post_with_retry(
            path="checkpoint/gc/drain",
            payload={
                "timeout_sec": float(timeout_sec),
                "poll_interval_sec": float(poll_interval_sec),
            },
            op_name="checkpoint_gc_drain",
            timeout=float(timeout_sec + 30.0),
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )

    async def rerun(self, lease_id: str, checkpoint_id: str, cwd: str, timeout: int) -> dict[str, Any]:
        return await self._post_with_retry(
            path="rerun",
            payload={"lease_id": lease_id, "checkpoint_id": checkpoint_id, "cwd": cwd, "timeout": int(timeout)},
            op_name="rerun",
            timeout=float(timeout + 30),
            http_max_retries=self.default_http_max_retries,
            app_max_retries=self.default_app_max_retries,
        )


@dataclass
class ReplayStep:
    step_idx: int
    action: str
    expected_returncode: int
    expected_output_head: str
    expected_output_tail: str
    llm_elapsed: float

    @classmethod
    def from_step_debug(cls, payload: dict[str, Any]) -> "ReplayStep":
        return cls(
            step_idx=int(payload.get("step_idx", -1)),
            action=str(payload.get("action", "")),
            expected_returncode=int(payload.get("returncode", -1)),
            expected_output_head=str(payload.get("output_head", "")),
            expected_output_tail=str(payload.get("output_tail", "")),
            llm_elapsed=float(payload.get("llm_elapsed", 0.0) or 0.0),
        )


def _load_yaml_config(config_path: str | None) -> dict[str, Any]:
    target = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if yaml is None or not target.exists():
        return {}
    return yaml.safe_load(target.read_text(encoding="utf-8")) or {}


def _load_traj_steps(traj_path: str, start_step: int = 0, end_step: int | None = None) -> tuple[dict[str, Any], list[ReplayStep]]:
    payload = json.loads(Path(traj_path).read_text(encoding="utf-8"))
    raw_steps = payload.get("step_debug", [])
    if not isinstance(raw_steps, list):
        raise ValueError(f"trajectory step_debug must be a list, got {type(raw_steps).__name__}")

    steps = [ReplayStep.from_step_debug(item) for item in raw_steps if isinstance(item, dict)]
    steps = [step for step in steps if step.step_idx >= start_step]
    if end_step is not None:
        steps = [step for step in steps if step.step_idx <= end_step]
    return payload, steps


def _collect_traj_paths(path_value: str, limit: int | None = None) -> list[Path]:
    root = Path(path_value)
    if root.is_file():
        return [root]
    if not root.exists():
        raise FileNotFoundError(f"trajectory path does not exist: {root}")
    traj_paths = sorted(p for p in root.rglob("traj.json") if p.is_file())
    if limit is not None:
        traj_paths = traj_paths[: max(0, int(limit))]
    return traj_paths


def _output_matches(expected: str, actual: str, *, is_head: bool) -> bool:
    if not expected:
        return True
    if actual == expected:
        return True
    if is_head:
        return actual.startswith(expected)
    return actual.endswith(expected)


def _default_instance_id(traj_payload: dict[str, Any]) -> str:
    info = traj_payload.get("info", {})
    if isinstance(info, dict):
        value = str(info.get("instance_id", "")).strip()
        if value:
            return value
    raise ValueError("instance_id not found in trajectory info; please pass --instance-id")


def _build_image_name(
    traj_payload: dict[str, Any],
    *,
    instance_id: str | None,
    image_name: str | None,
    data_source: str,
) -> str:
    if image_name:
        return image_name
    resolved_instance_id = instance_id or _default_instance_id(traj_payload)
    return get_docker_image_name({"instance_id": resolved_instance_id}, data_source)


async def _wait_checkpoint_ready(
    env_client,
    lease_id: str,
    checkpoint_id: str,
    *,
    timeout_sec: float,
    poll_interval_sec: float,
) -> dict[str, Any]:
    deadline = time.time() + max(0.1, timeout_sec)
    last_status: dict[str, Any] | None = None
    while time.time() < deadline:
        last_status = await env_client.checkpoint_status(lease_id, checkpoint_id)
        status = str(last_status.get("status", ""))
        if status == "ready":
            return last_status
        if status == "failed":
            raise RuntimeError(f"checkpoint {checkpoint_id} failed: {last_status.get('error', '')}")
        await asyncio.sleep(max(0.05, poll_interval_sec))
    raise TimeoutError(f"checkpoint {checkpoint_id} did not become ready within {timeout_sec}s: {last_status}")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _is_checkpoint_busy_error(exc: Exception) -> bool:
    return isinstance(exc, ReplayOpError) and str(exc.payload.get("error_code", "")) == "checkpoint_busy"


async def _maybe_run_checkpoint_gc(
    env_client: ReplayEnvClient,
    lease_id: str | None,
    *,
    keep_latest: int,
    dry_run: bool,
    min_checkpoint_count: int,
) -> dict[str, Any]:
    checkpoint_list_before_gc = await env_client.checkpoint_list(None)
    checkpoint_count = len(checkpoint_list_before_gc.get("checkpoints", []) or [])

    if min_checkpoint_count > 0 and checkpoint_count < min_checkpoint_count:
        return {
            "checkpoint_list_before_gc": checkpoint_list_before_gc,
            "gc_result": {
                "ok": True,
                "skipped": True,
                "skip_reason": "below_threshold",
                "checkpoint_count": checkpoint_count,
                "gc_min_checkpoint_count": int(min_checkpoint_count),
                "gc_scope": "global",
                "keep_latest": int(keep_latest),
                "deleted_count": 0,
                "deleted_checkpoint_ids": [],
                "reclaimed_bytes": 0,
                "dry_run": bool(dry_run),
                "queued": False,
            },
            "checkpoint_list_after_gc": None,
        }

    gc_result = await env_client.checkpoint_gc(
        None,
        keep_latest=keep_latest,
        dry_run=dry_run,
    )
    gc_result["skipped"] = False
    gc_result["checkpoint_count"] = checkpoint_count
    gc_result["gc_min_checkpoint_count"] = int(min_checkpoint_count)
    gc_result["gc_scope"] = "global"
    gc_result["keep_latest"] = int(keep_latest)
    checkpoint_list_after_gc = await env_client.checkpoint_list(None)
    return {
        "checkpoint_list_before_gc": checkpoint_list_before_gc,
        "gc_result": gc_result,
        "checkpoint_list_after_gc": checkpoint_list_after_gc,
    }


def _deferred_batch_gc_result(*, keep_latest: int, dry_run: bool, min_checkpoint_count: int) -> dict[str, Any]:
    return {
        "ok": True,
        "skipped": True,
        "skip_reason": "deferred_to_batch_end",
        "checkpoint_count": None,
        "gc_min_checkpoint_count": int(min_checkpoint_count),
        "gc_scope": "global",
        "keep_latest": int(keep_latest),
        "deleted_count": 0,
        "deleted_checkpoint_ids": [],
        "reclaimed_bytes": 0,
        "dry_run": bool(dry_run),
        "queued": False,
    }


async def _replay(args: argparse.Namespace, *, defer_gc_until_batch_end: bool = False) -> dict[str, Any]:
    traj_payload, steps = _load_traj_steps(args.trajectory, start_step=args.start_step, end_step=args.end_step)
    if args.print_commands:
        for step in steps:
            print(f"[{step.step_idx}] {step.action}")
        return {"printed_commands": True, "step_count": len(steps)}

    swe_config = _load_yaml_config(args.config_path)
    env_config = swe_config.get("environment", {}) if isinstance(swe_config, dict) else {}
    cwd = args.cwd or str(env_config.get("cwd", "/testbed"))
    exec_timeout = int(args.exec_timeout or int(env_config.get("timeout", 180)))
    instance_id = args.instance_id or _default_instance_id(traj_payload)
    image_name = _build_image_name(
        traj_payload,
        instance_id=instance_id,
        image_name=args.image_name,
        data_source=args.data_source,
    )

    env_client = ReplayEnvClient(base_url=args.base_url)
    checkpoint_after_steps = set(args.checkpoint_after_step or [])

    lease = await env_client.allocate(image=image_name, instance_id=instance_id, cwd=cwd)
    lease_id = str(lease["lease_id"])

    report: dict[str, Any] = {
        "trajectory": str(Path(args.trajectory).resolve()),
        "instance_id": instance_id,
        "image_name": image_name,
        "cwd": cwd,
        "exec_timeout": exec_timeout,
        "lease": lease,
        "steps": [],
        "checkpoint_events": [],
        "rerun_event": None,
        "checkpoint_list_before_gc": None,
        "gc_result": None,
        "checkpoint_list_after_gc": None,
        "closed": False,
    }

    checkpoint_ids_by_step: dict[int, str] = {}
    latest_ready_checkpoint_id: str | None = None

    try:
        for step in steps:
            await env_client.heartbeat(lease_id)
            llm_sleep_sec = step.llm_elapsed if args.simulate_llm_delay else 0.0
            if llm_sleep_sec > 0:
                await asyncio.sleep(llm_sleep_sec)
            exec_result = await env_client.exec(
                lease_id=lease_id,
                command=step.action,
                cwd=cwd,
                timeout=exec_timeout,
            )
            output = str(exec_result.get("output", ""))
            actual_returncode = int(exec_result.get("returncode", -1))
            head_match = _output_matches(step.expected_output_head, output, is_head=True)
            tail_match = _output_matches(step.expected_output_tail, output, is_head=False)
            report["steps"].append(
                {
                    "step_idx": step.step_idx,
                    "action": step.action,
                    "expected_returncode": step.expected_returncode,
                    "actual_returncode": actual_returncode,
                    "simulated_llm_delay_sec": llm_sleep_sec,
                    "returncode_match": actual_returncode == step.expected_returncode,
                    "output_head_match": head_match,
                    "output_tail_match": tail_match,
                    "output_preview": output[:2000],
                }
            )

            checkpoint_id_for_step: str | None = None
            if step.step_idx in checkpoint_after_steps:
                checkpoint_event: dict[str, Any] = {
                    "step_idx": step.step_idx,
                    "create_result": None,
                    "ready_result": None,
                    "skipped": False,
                    "skip_reason": None,
                }
                try:
                    checkpoint_create = await env_client.checkpoint_create(
                        lease_id,
                        step_idx=step.step_idx,
                        command_seq=step.step_idx + 1,
                        cwd=cwd,
                        policy=args.checkpoint_policy,
                        reason=args.checkpoint_reason,
                        parent_checkpoint_id=latest_ready_checkpoint_id,
                    )
                    checkpoint_id_for_step = str(checkpoint_create["checkpoint_id"])
                    checkpoint_ids_by_step[step.step_idx] = checkpoint_id_for_step
                    checkpoint_event["create_result"] = checkpoint_create
                    if args.wait_checkpoint_ready or args.rerun_after_step == step.step_idx:
                        ready_result = await _wait_checkpoint_ready(
                            env_client,
                            lease_id,
                            checkpoint_id_for_step,
                            timeout_sec=args.wait_checkpoint_ready_timeout,
                            poll_interval_sec=args.checkpoint_poll_interval,
                        )
                        latest_ready_checkpoint_id = checkpoint_id_for_step
                        checkpoint_event["ready_result"] = ready_result
                except Exception as exc:
                    if not _is_checkpoint_busy_error(exc):
                        raise
                    checkpoint_event["skipped"] = True
                    checkpoint_event["skip_reason"] = "checkpoint_busy"
                    checkpoint_event["create_error"] = str(exc)
                    if isinstance(exc, ReplayOpError):
                        checkpoint_event["create_error_payload"] = exc.payload
                report["checkpoint_events"].append(checkpoint_event)

            if args.rerun_after_step == step.step_idx:
                target_checkpoint_id = checkpoint_id_for_step or checkpoint_ids_by_step.get(step.step_idx) or latest_ready_checkpoint_id
                if not target_checkpoint_id:
                    report["rerun_event"] = {
                        "after_step_idx": step.step_idx,
                        "checkpoint_id": None,
                        "checkpoint_ready_result": None,
                        "rerun_result": None,
                        "skipped": True,
                        "skip_reason": "no_ready_checkpoint",
                    }
                    continue
                if target_checkpoint_id != latest_ready_checkpoint_id:
                    ready_result = await _wait_checkpoint_ready(
                        env_client,
                        lease_id,
                        target_checkpoint_id,
                        timeout_sec=args.wait_checkpoint_ready_timeout,
                        poll_interval_sec=args.checkpoint_poll_interval,
                    )
                    latest_ready_checkpoint_id = target_checkpoint_id
                else:
                    ready_result = None
                rerun_result = await env_client.rerun(
                    lease_id,
                    checkpoint_id=target_checkpoint_id,
                    cwd=cwd,
                    timeout=args.rerun_timeout,
                )
                report["rerun_event"] = {
                    "after_step_idx": step.step_idx,
                    "checkpoint_id": target_checkpoint_id,
                    "checkpoint_ready_result": ready_result,
                    "rerun_result": rerun_result,
                    "skipped": False,
                    "skip_reason": None,
                }

        if args.gc_keep_latest is not None and not defer_gc_until_batch_end:
            gc_payload = await _maybe_run_checkpoint_gc(
                env_client,
                lease_id,
                keep_latest=args.gc_keep_latest,
                dry_run=args.gc_dry_run,
                min_checkpoint_count=args.gc_min_checkpoint_count,
            )
            report["checkpoint_list_before_gc"] = gc_payload["checkpoint_list_before_gc"]
            report["gc_result"] = gc_payload["gc_result"]
            report["checkpoint_list_after_gc"] = gc_payload["checkpoint_list_after_gc"]
        elif args.gc_keep_latest is not None and defer_gc_until_batch_end:
            report["gc_result"] = _deferred_batch_gc_result(
                keep_latest=args.gc_keep_latest,
                dry_run=args.gc_dry_run,
                min_checkpoint_count=args.gc_min_checkpoint_count,
            )
    finally:
        if not args.keep_lease_open:
            try:
                await env_client.close(lease_id)
                report["closed"] = True
            except Exception as exc:
                report["close_error"] = str(exc)

    return report


async def _replay_one_path(
    traj_path: Path,
    args: argparse.Namespace,
    *,
    defer_gc_until_batch_end: bool = False,
) -> dict[str, Any]:
    local_args = argparse.Namespace(**vars(args))
    local_args.trajectory = str(traj_path)
    report = await _replay(local_args, defer_gc_until_batch_end=defer_gc_until_batch_end)
    report["traj_path"] = str(traj_path.resolve())
    return report


async def _replay_many(args: argparse.Namespace) -> dict[str, Any]:
    traj_paths = _collect_traj_paths(args.trajectory, limit=args.limit)
    if args.print_commands:
        reports = []
        for traj_path in traj_paths:
            local_args = argparse.Namespace(**vars(args))
            local_args.trajectory = str(traj_path)
            report = await _replay(local_args)
            reports.append({"traj_path": str(traj_path.resolve()), **report})
        return {
            "mode": "print_commands",
            "trajectory_count": len(traj_paths),
            "reports": reports,
        }

    concurrency = max(1, int(args.max_concurrency))
    sem = asyncio.Semaphore(concurrency)
    reports: list[dict[str, Any]] = []
    batch_gc_result: dict[str, Any] | None = None

    async def _run_with_guard(traj_path: Path) -> dict[str, Any]:
        async with sem:
            started_at = time.time()
            try:
                report = await _replay_one_path(traj_path, args, defer_gc_until_batch_end=args.gc_keep_latest is not None)
                report["ok"] = True
            except Exception as exc:
                report = {
                    "traj_path": str(traj_path.resolve()),
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            report["wall_time_sec"] = time.time() - started_at
            return report

    tasks = [asyncio.create_task(_run_with_guard(traj_path)) for traj_path in traj_paths]
    for task in asyncio.as_completed(tasks):
        report = await task
        reports.append(report)
        if args.output_dir:
            out_dir = Path(args.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            traj_name = Path(report["traj_path"]).parent.name if "traj_path" in report else f"failed-{len(reports)}"
            out_path = out_dir / f"{traj_name}.json"
            out_path.write_text(
                json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n",
                encoding="utf-8",
            )

    if args.gc_keep_latest is not None:
        env_client = ReplayEnvClient(base_url=args.base_url)
        gc_payload = await _maybe_run_checkpoint_gc(
            env_client,
            None,
            keep_latest=args.gc_keep_latest,
            dry_run=args.gc_dry_run,
            min_checkpoint_count=args.gc_min_checkpoint_count,
        )
        batch_gc_result = gc_payload["gc_result"]

    ok_count = sum(1 for item in reports if item.get("ok"))
    failed_count = len(reports) - ok_count
    return {
        "mode": "batch_replay",
        "trajectory_root": str(Path(args.trajectory).resolve()),
        "trajectory_count": len(traj_paths),
        "ok_count": ok_count,
        "failed_count": failed_count,
        "max_concurrency": concurrency,
        "batch_gc_result": batch_gc_result,
        "reports": reports,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay a SWE trajectory command sequence and exercise checkpoint/rerun/gc against swe_env_pool_server.",
    )
    parser.add_argument("trajectory", help="Path to a traj.json file or a rollout directory containing traj.json files")
    parser.add_argument("--base-url", default=os.getenv("SWE_ENV_SERVER_URL"), help="swe_env_pool_server base URL")
    parser.add_argument("--config-path", default=os.getenv("SWE_CONFIG_PATH"), help="Path to swebench.yaml")
    parser.add_argument("--data-source", default="swe-gym", help="Data source for docker image naming")
    parser.add_argument("--instance-id", default=None, help="Override instance_id from trajectory")
    parser.add_argument("--image-name", default=None, help="Override docker image name")
    parser.add_argument("--cwd", default=None, help="Working directory inside container")
    parser.add_argument("--exec-timeout", type=int, default=None, help="Per-command timeout in seconds")
    parser.add_argument(
        "--simulate-llm-delay",
        action="store_true",
        help="Sleep for each step's llm_elapsed before executing the environment command.",
    )
    parser.add_argument("--start-step", type=int, default=0, help="Replay from this step index")
    parser.add_argument("--end-step", type=int, default=None, help="Replay through this step index")
    parser.add_argument(
        "--checkpoint-after-step",
        type=int,
        action="append",
        default=[],
        help="Create a checkpoint after this step index. May be repeated.",
    )
    parser.add_argument("--checkpoint-policy", default="manual-replay", help="Checkpoint policy label")
    parser.add_argument("--checkpoint-reason", default="traj_replay", help="Checkpoint reason label")
    parser.add_argument(
        "--wait-checkpoint-ready",
        action="store_true",
        help="Wait until each created checkpoint becomes ready before continuing.",
    )
    parser.add_argument(
        "--wait-checkpoint-ready-timeout",
        type=float,
        default=300.0,
        help="Timeout in seconds when waiting for checkpoint ready.",
    )
    parser.add_argument(
        "--checkpoint-poll-interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds for checkpoint status.",
    )
    parser.add_argument(
        "--rerun-after-step",
        type=int,
        default=None,
        help="Call rerun after this step finishes. Usually pair with --checkpoint-after-step.",
    )
    parser.add_argument("--rerun-timeout", type=int, default=120, help="rerun timeout in seconds")
    parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum concurrent trajectory replays in batch mode")
    parser.add_argument("--limit", type=int, default=None, help="Only replay the first N trajectories after sorting")
    parser.add_argument(
        "--gc-keep-latest",
        type=int,
        default=None,
        help="If set, run global checkpoint GC after replay, keeping this many ready checkpoints per GC plan.",
    )
    parser.add_argument(
        "--gc-min-checkpoint-count",
        type=int,
        default=DEFAULT_GC_MIN_CHECKPOINT_COUNT,
        help="Only run global checkpoint GC when total checkpoint/list count reaches at least this many checkpoints. Set 0 to always GC.",
    )
    parser.add_argument("--gc-dry-run", action="store_true", help="Run checkpoint GC in dry-run mode")
    parser.add_argument("--keep-lease-open", action="store_true", help="Do not close the lease at the end")
    parser.add_argument("--print-commands", action="store_true", help="Print replay commands and exit")
    parser.add_argument("--output-json", default=None, help="Write replay report to this JSON file")
    parser.add_argument("--output-dir", default=None, help="Write one JSON report per trajectory in batch mode")
    return parser


async def _main_async(args: argparse.Namespace) -> int:
    if not args.base_url and not args.print_commands:
        raise ValueError("--base-url is required unless --print-commands is used")
    traj_input = Path(args.trajectory)
    if traj_input.is_dir():
        report = await _replay_many(args)
    else:
        report = await _replay(args)
    text = json.dumps(report, indent=2, ensure_ascii=False, default=_json_default)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
