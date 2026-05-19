"""Async HTTP client for swe_env_pool_server.

Used by generate_with_swe_remote.py (inside the RolloutManager) to interact with
remote Docker containers via the pool server.  Modeled after gui/env_client.py.
"""

from __future__ import annotations

import asyncio
import os
import random
import time
from typing import Any

try:
    from loguru import logger
except Exception:  # pragma: no cover
    import logging

    logger = logging.getLogger("swe.env_client")

from slime.utils.http_utils import post


class SweEnvClient:
    def __init__(self, base_url: str | None = None):
        self.base_url = (base_url or os.getenv("SWE_ENV_SERVER_URL", "http://localhost:18090")).rstrip("/")
        self.default_max_retries = int(os.getenv("SWE_ENV_HTTP_MAX_RETRIES", "10"))
        self.allocate_http_max_retries = int(os.getenv("SWE_ALLOCATE_HTTP_MAX_RETRIES", "1"))
        self.evaluate_max_retries = int(os.getenv("SWE_EVALUATE_MAX_RETRIES", "3"))
        self.app_error_max_retries = int(os.getenv("SWE_ENV_APP_MAX_RETRIES", "3"))
        self.allocate_app_max_retries = int(os.getenv("SWE_ALLOCATE_APP_MAX_RETRIES", "360"))
        self.checkpoint_timeout_sec = float(os.getenv("SWE_CHECKPOINT_TIMEOUT_SEC", "10"))
        self.app_error_retry_delay_sec = float(os.getenv("SWE_ENV_APP_RETRY_DELAY_SEC", "1.0"))
        self.app_error_retry_jitter_sec = float(os.getenv("SWE_ENV_APP_RETRY_JITTER_SEC", "0.2"))
        self.app_error_retry_max_delay_sec = float(os.getenv("SWE_ENV_APP_RETRY_MAX_DELAY_SEC", "5.0"))

    async def _post_with_retry(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        op_name: str,
        http_max_retries: int,
        app_max_retries: int | None = None,
    ) -> dict[str, Any]:
        max_attempts = max(1, int(app_max_retries or self.app_error_max_retries))
        last_out: dict[str, Any] | None = None
        for attempt in range(1, max_attempts + 1):
            out = await post(
                f"{self.base_url}/{path}",
                payload,
                max_retries=http_max_retries,
            )
            last_out = out if isinstance(out, dict) else {"ok": False, "error": str(out)}
            if last_out.get("ok", False):
                return last_out

            if attempt >= max_attempts or not self._is_retryable_app_error(last_out):
                break

            sleep_s = self.app_error_retry_delay_sec * (2 ** (attempt - 1))
            sleep_s = min(sleep_s, max(0.0, self.app_error_retry_max_delay_sec))
            retry_after = last_out.get("retry_after_sec")
            if retry_after is not None:
                try:
                    sleep_s = max(sleep_s, float(retry_after))
                except (TypeError, ValueError):
                    pass
            sleep_s += random.uniform(0, max(0.0, self.app_error_retry_jitter_sec))
            logger.warning(
                f"[SWE-CLIENT] {op_name} failed (attempt {attempt}/{max_attempts}), "
                f"retrying in {sleep_s:.2f}s: {last_out}"
            )
            await asyncio.sleep(sleep_s)

        raise RuntimeError(f"SWE {op_name} failed after {max_attempts} attempt(s): {last_out}")

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
        )
        return not any(marker in error_text for marker in non_retryable_markers)

    @staticmethod
    def _is_timeout_error(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
            return True
        text = str(exc).lower()
        return "timed out" in text or "timeout" in text

    async def _post_checkpoint_once(
        self,
        *,
        path: str,
        payload: dict[str, Any],
        op_name: str,
    ) -> dict[str, Any]:
        started_perf = time.perf_counter()
        lease_id = str(payload.get("lease_id", ""))
        step_idx = int(payload.get("step_idx", -1))
        command_seq = int(payload.get("command_seq", -1))
        policy = str(payload.get("policy", ""))
        checkpoint_id = payload.get("checkpoint_id")
        try:
            out = await asyncio.wait_for(
                post(
                    f"{self.base_url}/{path}",
                    payload,
                    max_retries=1,
                ),
                timeout=self.checkpoint_timeout_sec,
            )
        except Exception as exc:
            timed_out = self._is_timeout_error(exc)
            elapsed_sec = time.perf_counter() - started_perf
            if op_name == "checkpoint_create":
                logger.warning(
                    "[SWE-CLIENT] {} failed without retry lease_id={} step_idx={} command_seq={} policy={} timeout_sec={:.3f} elapsed_sec={:.3f} timed_out={} error={}",
                    op_name,
                    lease_id,
                    step_idx,
                    command_seq,
                    policy,
                    self.checkpoint_timeout_sec,
                    elapsed_sec,
                    timed_out,
                    exc,
                )
            else:
                logger.warning(
                    "[SWE-CLIENT] {} failed without retry (timeout={}s): {}",
                    op_name,
                    self.checkpoint_timeout_sec,
                    exc,
                )
            return {
                "ok": False,
                "error": str(exc),
                "timed_out": timed_out,
                "timeout_sec": self.checkpoint_timeout_sec,
                "retryable": False,
                "error_code": f"{op_name}_{'timeout' if timed_out else 'failed'}",
            }

        if isinstance(out, dict):
            if op_name == "checkpoint_create":
                elapsed_sec = time.perf_counter() - started_perf
                logger.info(
                    "[SWE-CLIENT] {} completed lease_id={} step_idx={} command_seq={} policy={} elapsed_sec={:.3f} ok={} status={} busy={} error_code={} checkpoint_id={} op_id={}",
                    op_name,
                    lease_id,
                    step_idx,
                    command_seq,
                    policy,
                    elapsed_sec,
                    bool(out.get("ok", False)),
                    out.get("status"),
                    bool(out.get("busy", False)),
                    out.get("error_code"),
                    out.get("checkpoint_id", checkpoint_id),
                    out.get("op_id"),
                )
            return out

        elapsed_sec = time.perf_counter() - started_perf
        logger.warning(
            "[SWE-CLIENT] {} returned non-dict response without retry after {:.3f}s: {!r}",
            op_name,
            elapsed_sec,
            out,
        )
        return {
            "ok": False,
            "error": f"non_dict_response:{out!r}",
            "timed_out": False,
            "timeout_sec": self.checkpoint_timeout_sec,
            "retryable": False,
            "error_code": f"{op_name}_invalid_response",
        }

    async def allocate(self, image: str, instance_id: str = "", cwd: str = "/testbed") -> dict[str, Any]:
        return await self._post_with_retry(
            path="allocate",
            payload={"image": image, "instance_id": instance_id, "cwd": cwd},
            op_name="allocate",
            http_max_retries=self.allocate_http_max_retries,
            app_max_retries=self.allocate_app_max_retries,
        )

    async def heartbeat(self, lease_id: str) -> None:
        await self._post_with_retry(
            path="heartbeat",
            payload={"lease_id": lease_id},
            op_name="heartbeat",
            http_max_retries=self.default_max_retries,
        )

    async def exec(self, lease_id: str, command: str, cwd: str = "/testbed",
                   timeout: int = 180, env: dict | None = None,
                   fault_injection_armed: bool = False,
                   fault_injection_probability: float | None = None) -> dict[str, Any]:
        """Execute a command in the container. Returns {ok, returncode, output}."""
        payload: dict[str, Any] = {
            "lease_id": lease_id,
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
            "env": env or {},
            "fault_injection_armed": bool(fault_injection_armed),
        }
        if fault_injection_probability is not None:
            payload["fault_injection_probability"] = float(fault_injection_probability)
        return await self._post_with_retry(
            path="exec",
            payload=payload,
            op_name="exec",
            http_max_retries=self.default_max_retries,
        )

    async def diff(self, lease_id: str, cwd: str = "/testbed") -> str:
        """Get git diff from the container. Returns the patch string."""
        out = await self._post_with_retry(
            path="diff",
            payload={"lease_id": lease_id, "cwd": cwd},
            op_name="diff",
            http_max_retries=self.default_max_retries,
        )
        return out.get("patch", "")

    async def evaluate(self, lease_id: str, patch: str, eval_script: str,
                       cwd: str = "/testbed", timeout: int = 3600) -> dict[str, Any]:
        """Apply patch + run eval script. Returns {ok, resolved, ...}."""
        return await self._post_with_retry(
            path="evaluate",
            payload={
                "lease_id": lease_id,
                "patch": patch,
                "eval_script": eval_script,
                "cwd": cwd,
                "timeout": timeout,
            },
            op_name="evaluate",
            http_max_retries=self.evaluate_max_retries,
            app_max_retries=self.evaluate_max_retries,
        )

    async def close(self, lease_id: str) -> None:
        await self._post_with_retry(
            path="close",
            payload={"lease_id": lease_id},
            op_name="close",
            http_max_retries=self.default_max_retries,
        )

    async def stats(self, lease_id: str) -> dict[str, Any]:
        return await self._post_with_retry(
            path="stats",
            payload={"lease_id": lease_id},
            op_name="stats",
            http_max_retries=1,
            app_max_retries=1,
        )

    async def checkpoint_probe(self, lease_id: str) -> dict[str, Any]:
        return await self._post_checkpoint_once(
            path="checkpoint/probe",
            payload={"lease_id": lease_id},
            op_name="checkpoint_probe",
        )

    async def checkpoint_create(
        self,
        lease_id: str,
        *,
        step_idx: int = -1,
        command_seq: int = -1,
        cwd: str | None = None,
        env: dict[str, Any] | None = None,
        policy: str = "",
        reason: str = "manual",
        parent_checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "lease_id": lease_id,
            "step_idx": int(step_idx),
            "command_seq": int(command_seq),
            "policy": policy,
            "reason": reason,
        }
        if cwd is not None:
            payload["cwd"] = cwd
        if env:
            payload["env"] = env
        if parent_checkpoint_id is not None:
            payload["parent_checkpoint_id"] = parent_checkpoint_id
        return await self._post_checkpoint_once(
            path="checkpoint/create",
            payload=payload,
            op_name="checkpoint_create",
        )

    async def checkpoint_status(self, lease_id: str, checkpoint_id: str) -> dict[str, Any]:
        return {
            "ok": False,
            "error": "checkpoint_status is deprecated; checkpoint_create is synchronous",
            "error_code": "checkpoint_status_deprecated",
            "retryable": False,
        }

    async def checkpoint_list(self, lease_id: str) -> dict[str, Any]:
        return await self._post_checkpoint_once(
            path="checkpoint/list",
            payload={"lease_id": lease_id},
            op_name="checkpoint_list",
        )

    async def checkpoint_delete(self, lease_id: str, checkpoint_id: str) -> dict[str, Any]:
        return await self._post_checkpoint_once(
            path="checkpoint/delete",
            payload={"lease_id": lease_id, "checkpoint_id": checkpoint_id},
            op_name="checkpoint_delete",
        )

    async def checkpoint_gc(
        self,
        lease_id: str | None = None,
        *,
        keep_latest: int = 1,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "keep_latest": int(keep_latest),
            "dry_run": bool(dry_run),
        }
        if lease_id is not None:
            payload["lease_id"] = lease_id
        return await self._post_checkpoint_once(
            path="checkpoint/gc",
            payload=payload,
            op_name="checkpoint_gc",
        )

    async def rerun(
        self,
        lease_id: str,
        *,
        checkpoint_id: str | None = None,
        cwd: str | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {"lease_id": lease_id, "timeout": int(timeout)}
        if checkpoint_id is not None:
            payload["checkpoint_id"] = checkpoint_id
        if cwd is not None:
            payload["cwd"] = cwd
        return await self._post_checkpoint_once(
            path="rerun",
            payload=payload,
            op_name="rerun",
        )
