"""SWE environment pool server — runs on GPU Head Node.

Manages remote ECS Docker nodes: lease allocation, container lifecycle,
command execution, evaluation.  Modeled after gui/env_pool_server.py.

Each "lease" corresponds to one SWE-Bench instance running inside a Docker
container on one of the remote ECS nodes.  The pool server proxies all
requests to the appropriate swe_exec_server running on the ECS node.

Usage:
    python3 -m swe_env_pool_server \
        --port 18090 \
        --exec-server-urls http://10.0.0.10:5000,http://10.0.0.11:5000
"""

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import requests
from flask import Flask, jsonify, request as flask_request

logger = logging.getLogger("swe.env_pool_server")
app = Flask(__name__)
CHECKPOINT_PROBE_TIMEOUT_SEC = 5


def _post_exec(url: str, payload: dict, timeout: int = 300) -> dict:
    resp = requests.post(url, json=payload, timeout=timeout)
    try:
        body = resp.json()
    except ValueError:
        resp.raise_for_status()
        raise RuntimeError(f"Exec server returned a non-JSON response: {resp.text[:200]}")
    if isinstance(body, dict):
        return body
    if resp.status_code >= 400:
        resp.raise_for_status()
    raise RuntimeError(f"Exec server returned an unexpected JSON payload: {body!r}")


def _get_exec(url: str, timeout: int = 30) -> dict:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


# ── Data structures ───────────────────────────────────────────────────

@dataclass
class ExecNode:
    url: str
    active_containers: int = 0
    max_containers: int = 16
    healthy: bool = True
    last_health_check: float = 0.0
    consecutive_health_check_failures: int = 0


@dataclass
class Lease:
    lease_id: str
    node_url: str
    container_id: str
    image: str
    instance_id: str
    cwd: str = "/testbed"
    base_image: str | None = None
    current_image: str | None = None
    generation: int = 0
    latest_ready_checkpoint_id: str | None = None
    latest_checkpoint_step_idx: int = -1
    rerun_count: int = 0
    last_rerun_at: float | None = None
    created_at: float = field(default_factory=time.time)
    last_heartbeat: float = field(default_factory=time.time)


class SweEnvPool:
    def __init__(
        self,
        exec_server_urls: list[str],
        max_containers_per_node: int = 16,
        max_total_leases: int = 0,
        max_concurrent_allocates: int = 1,
        allocate_min_interval_sec: float = 0.5,
        create_timeout_sec: float = 120.0,
        health_check_timeout_sec: float = 30.0,
        health_check_failure_threshold: int = 3,
    ):
        self.nodes = [
            ExecNode(url=url.rstrip("/"), max_containers=max_containers_per_node)
            for url in exec_server_urls
        ]
        self._leases: dict[str, Lease] = {}
        self._lock = threading.RLock()
        self.max_total_leases = max(0, int(max_total_leases))
        self.max_concurrent_allocates = max(1, int(max_concurrent_allocates))
        self.allocate_min_interval_sec = max(0.0, float(allocate_min_interval_sec))
        self.create_timeout_sec = max(1.0, float(create_timeout_sec))
        self.health_check_timeout_sec = max(1.0, float(health_check_timeout_sec))
        self.health_check_failure_threshold = max(1, int(health_check_failure_threshold))
        self._pending_allocations = 0
        self._allocate_sem = threading.BoundedSemaphore(self.max_concurrent_allocates)
        self._allocate_interval_lock = threading.Lock()
        self._last_allocate_ts = 0.0

    def _pick_and_reserve_node(self) -> ExecNode:
        """Atomically pick the least-loaded healthy node and increment its counter.

        The counter is incremented optimistically BEFORE the container is
        actually created, so concurrent callers see the reservation and spread
        across nodes.  The caller MUST call ``_unreserve_node(node)`` if the
        subsequent container creation fails.
        """
        with self._lock:
            candidates = [
                n
                for n in self.nodes
                if n.healthy and (n.max_containers <= 0 or n.active_containers < n.max_containers)
            ]
            if not candidates:
                raise RuntimeError(
                    f"All exec nodes are full or unhealthy. "
                    f"nodes={[(n.url, n.active_containers, n.healthy) for n in self.nodes]}"
                )
            node = min(candidates, key=lambda n: n.active_containers)
            node.active_containers += 1
            return node

    def _unreserve_node(self, node: ExecNode) -> None:
        with self._lock:
            node.active_containers = max(0, node.active_containers - 1)

    def _find_node(self, url: str) -> ExecNode | None:
        for n in self.nodes:
            if n.url == url:
                return n
        return None

    def _checkpoint_gc_on_node(
        self,
        node_url: str,
        *,
        lease_id: str | None = None,
        keep_latest: int = 1,
        dry_run: bool = False,
        timeout: int = 120,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "keep_latest": int(keep_latest),
            "dry_run": bool(dry_run),
        }
        if lease_id is not None:
            payload["lease_id"] = lease_id
        return _post_exec(f"{node_url}/container/checkpoint/gc", payload, timeout=timeout)

    def _checkpoint_gc_drain_on_node(
        self,
        node_url: str,
        *,
        timeout_sec: float = 600.0,
        poll_interval_sec: float = 0.1,
    ) -> dict[str, Any]:
        timeout = max(1, int(float(timeout_sec) + 30.0))
        return _post_exec(
            f"{node_url}/container/checkpoint/gc/drain",
            {
                "timeout_sec": float(timeout_sec),
                "poll_interval_sec": float(poll_interval_sec),
            },
            timeout=timeout,
        )

    def _refresh_latest_ready_checkpoint(self, lease: Lease) -> None:
        result = _post_exec(
            f"{lease.node_url}/container/checkpoint/list",
            {"lease_id": lease.lease_id},
            timeout=30,
        )
        checkpoints = result.get("checkpoints", []) if result.get("ok", False) else []
        ready = [
            item
            for item in checkpoints
            if isinstance(item, dict) and item.get("status") == "ready" and item.get("checkpoint_id")
        ]
        if not ready:
            lease.latest_ready_checkpoint_id = None
            lease.latest_checkpoint_step_idx = -1
            return
        latest = max(
            ready,
            key=lambda item: (
                -1 if item.get("step_idx", -1) is None else int(item.get("step_idx", -1)),
                float(item.get("created_at", 0.0) or 0.0),
            ),
        )
        lease.latest_ready_checkpoint_id = str(latest["checkpoint_id"])
        raw_latest_step_idx = latest.get("step_idx", -1)
        lease.latest_checkpoint_step_idx = -1 if raw_latest_step_idx is None else int(raw_latest_step_idx)

    def _reserve_total_slot(self) -> None:
        if self.max_total_leases <= 0:
            return
        with self._lock:
            inflight = len(self._leases) + self._pending_allocations
            if inflight >= self.max_total_leases:
                raise RuntimeError(
                    f"Global lease limit reached ({inflight}>={self.max_total_leases}). "
                    f"Please lower rollout concurrency or increase SWE_POOL_MAX_TOTAL_LEASES."
                )
            self._pending_allocations += 1

    def _release_total_slot(self) -> None:
        if self.max_total_leases <= 0:
            return
        with self._lock:
            self._pending_allocations = max(0, self._pending_allocations - 1)

    def _throttle_allocate_rate(self) -> None:
        if self.allocate_min_interval_sec <= 0:
            return
        with self._allocate_interval_lock:
            now = time.time()
            wait_s = self.allocate_min_interval_sec - (now - self._last_allocate_ts)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_allocate_ts = time.time()

    def allocate(self, image: str, instance_id: str, cwd: str = "/testbed") -> dict[str, Any]:
        with self._allocate_sem:
            self._throttle_allocate_rate()
            self._reserve_total_slot()
            slot_reserved = True
            node = None
            try:
                node = self._pick_and_reserve_node()
                result = _post_exec(
                    f"{node.url}/container/create",
                    {"image": image, "cwd": cwd},
                    timeout=self.create_timeout_sec,
                )
                if not result.get("ok"):
                    raise RuntimeError(f"Container create failed on {node.url}: {result}")

                container_id = result["container_id"]
                lease_id = f"swe-lease-{uuid.uuid4().hex[:16]}"
                lease = Lease(
                    lease_id=lease_id,
                    node_url=node.url,
                    container_id=container_id,
                    image=image,
                    instance_id=instance_id,
                    cwd=cwd,
                    base_image=image,
                    current_image=image,
                )
                with self._lock:
                    self._leases[lease_id] = lease
                self._release_total_slot()
                slot_reserved = False
                logger.info("Allocated lease=%s node=%s cid=%s image=%s", lease_id, node.url, container_id[:12], image)
                return {"lease_id": lease_id, "container_id": container_id, "node_url": node.url}
            except Exception:
                if node is not None:
                    self._unreserve_node(node)
                raise
            finally:
                if slot_reserved:
                    self._release_total_slot()

    def _get_lease(self, lease_id: str) -> Lease:
        with self._lock:
            lease = self._leases.get(lease_id)
        if lease is None:
            raise KeyError(f"Unknown lease_id: {lease_id}")
        return lease

    def heartbeat(self, lease_id: str) -> None:
        lease = self._get_lease(lease_id)
        lease.last_heartbeat = time.time()

    def exec(self, lease_id: str, command: str, cwd: str = "/testbed",
             timeout: int = 180, env: dict | None = None,
             fault_injection_armed: bool = False,
             fault_injection_probability: float | None = None,
             fault_injection_spec: dict[str, Any] | None = None) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        lease.last_heartbeat = time.time()
        payload: dict[str, Any] = {
            "container_id": lease.container_id,
            "command": command,
            "cwd": cwd,
            "timeout": timeout,
            "env": env or {},
            "fault_injection_armed": bool(fault_injection_armed),
        }
        if fault_injection_probability is not None:
            payload["fault_injection_probability"] = float(fault_injection_probability)
        if fault_injection_spec:
            payload["fault_injection_spec"] = dict(fault_injection_spec)
        return _post_exec(f"{lease.node_url}/container/exec", payload, timeout=timeout + 30)

    def diff(self, lease_id: str, cwd: str = "/testbed") -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        return _post_exec(f"{lease.node_url}/container/diff", {
            "container_id": lease.container_id,
            "cwd": cwd,
        }, timeout=60)

    def evaluate(self, lease_id: str, patch: str, eval_script: str,
                 cwd: str = "/testbed", timeout: int = 3600) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        return _post_exec(f"{lease.node_url}/container/evaluate", {
            "container_id": lease.container_id,
            "patch": patch,
            "eval_script": eval_script,
            "cwd": cwd,
            "timeout": timeout,
        }, timeout=timeout + 60)

    def close(self, lease_id: str) -> None:
        with self._lock:
            lease = self._leases.get(lease_id)
        if lease is None:
            return
        _post_exec(f"{lease.node_url}/container/destroy", {
            "container_id": lease.container_id,
        }, timeout=300)
        with self._lock:
            removed = self._leases.pop(lease_id, None)
        if removed is None:
            return
        node = self._find_node(removed.node_url)
        if node is not None:
            self._unreserve_node(node)
        logger.info("Closed lease=%s cid=%s", lease_id, removed.container_id[:12])

    def inject_fail_stop(self, lease_id: str, *, tag: str = "", delay_sec: float = 0.0) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        lease.last_heartbeat = time.time()
        return _post_exec(
            f"{lease.node_url}/container/fault/kill",
            {
                "container_id": lease.container_id,
                "tag": tag,
                "delay_sec": float(delay_sec),
            },
            timeout=60,
        )

    def checkpoint_probe(self, lease_id: str) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        lease.last_heartbeat = time.time()
        return {
            "ok": False,
            "busy": True,
            "error": "checkpoint probe is deprecated; checkpoint/create performs probe inline",
            "error_code": "checkpoint_busy",
            "reason": "checkpoint_probe_deprecated",
            "retryable": True,
            "probe_wait_sec": 1.0,
            "retry_after_sec": 1.0,
        }

    def checkpoint_create(
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
        fault_injection_spec: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        lease.last_heartbeat = time.time()
        effective_cwd = cwd or lease.cwd
        payload = {
            "lease_id": lease.lease_id,
            "container_id": lease.container_id,
            "generation": lease.generation,
            "instance_id": lease.instance_id,
            "cwd": effective_cwd,
            "step_idx": int(step_idx),
            "command_seq": int(command_seq),
            "policy": policy,
            "reason": reason,
            "parent_checkpoint_id": parent_checkpoint_id,
        }
        if env:
            payload["env"] = env
        if fault_injection_spec:
            payload["fault_injection_spec"] = dict(fault_injection_spec)
        started_perf = time.perf_counter()
        try:
            result = _post_exec(f"{lease.node_url}/container/checkpoint/create", payload, timeout=30)
        except Exception as exc:
            logger.warning(
                "checkpoint_create forward failed lease_id=%s container_id=%s node_url=%s step_idx=%s command_seq=%s policy=%s elapsed_sec=%.3f error=%s",
                lease.lease_id,
                lease.container_id,
                lease.node_url,
                int(step_idx),
                int(command_seq),
                policy,
                time.perf_counter() - started_perf,
                exc,
            )
            raise
        logger.info(
            "checkpoint_create forward completed lease_id=%s container_id=%s node_url=%s step_idx=%s command_seq=%s policy=%s elapsed_sec=%.3f ok=%s status=%s busy=%s error_code=%s checkpoint_id=%s op_id=%s",
            lease.lease_id,
            lease.container_id,
            lease.node_url,
            int(step_idx),
            int(command_seq),
            policy,
            time.perf_counter() - started_perf,
            bool(result.get("ok", False)),
            result.get("status"),
            bool(result.get("busy", False)),
            result.get("error_code"),
            result.get("checkpoint_id"),
            result.get("op_id"),
        )
        return result

    def checkpoint_status(self, lease_id: str, checkpoint_id: str) -> dict[str, Any]:
        self._get_lease(lease_id)
        return {
            "ok": False,
            "error": "checkpoint_status is deprecated; checkpoint_create is synchronous",
            "error_code": "checkpoint_status_deprecated",
            "retryable": False,
        }

    def checkpoint_list(self, lease_id: str | None = None) -> dict[str, Any]:
        if lease_id is not None:
            lease = self._get_lease(lease_id)
            return _post_exec(
                f"{lease.node_url}/container/checkpoint/list",
                {"lease_id": lease.lease_id},
                timeout=30,
            )

        per_node: list[dict[str, Any]] = []
        checkpoints: list[dict[str, Any]] = []
        ok = True
        for node in self.nodes:
            try:
                result = _post_exec(
                    f"{node.url}/container/checkpoint/list",
                    {},
                    timeout=30,
                )
            except Exception as exc:
                ok = False
                per_node.append({"node_url": node.url, "ok": False, "error": str(exc)})
                continue
            node_checkpoints = [dict(item, node_url=node.url) for item in (result.get("checkpoints", []) or [])]
            checkpoints.extend(node_checkpoints)
            per_node.append(
                {
                    "node_url": node.url,
                    "ok": bool(result.get("ok", False)),
                    "checkpoint_count": len(node_checkpoints),
                }
            )
            if not result.get("ok", False):
                ok = False
        return {"ok": ok, "checkpoints": checkpoints, "per_node": per_node}

    def checkpoint_delete(self, lease_id: str, checkpoint_id: str) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        result = _post_exec(
            f"{lease.node_url}/container/checkpoint/delete",
            {"checkpoint_id": checkpoint_id},
            timeout=60,
        )
        if result.get("ok", False):
            self._refresh_latest_ready_checkpoint(lease)
        return result

    def checkpoint_gc(
        self,
        lease_id: str | None = None,
        *,
        keep_latest: int = 1,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        if lease_id is not None:
            lease = self._get_lease(lease_id)
            result = self._checkpoint_gc_on_node(
                lease.node_url,
                lease_id=lease.lease_id,
                keep_latest=keep_latest,
                dry_run=dry_run,
                timeout=120,
            )
            if result.get("ok", False):
                self._refresh_latest_ready_checkpoint(lease)
            return result

        per_node: list[dict[str, Any]] = []
        deleted_checkpoint_ids: list[str] = []
        deleted_count = 0
        reclaimed_bytes = 0
        ok = True
        for node in self.nodes:
            try:
                result = self._checkpoint_gc_on_node(
                    node.url,
                    keep_latest=keep_latest,
                    dry_run=dry_run,
                    timeout=120,
                )
            except Exception as exc:
                ok = False
                per_node.append({"node_url": node.url, "ok": False, "error": str(exc)})
                continue
            per_node.append({"node_url": node.url, **result})
            if not result.get("ok", False):
                ok = False
                continue
            deleted_count += int(result.get("deleted_count", 0) or 0)
            reclaimed_bytes += int(result.get("reclaimed_bytes", 0) or 0)
            deleted_checkpoint_ids.extend(result.get("deleted_checkpoint_ids", []) or [])
        return {
            "ok": ok,
            "scope": "global",
            "deleted_count": deleted_count,
            "deleted_checkpoint_ids": deleted_checkpoint_ids,
            "reclaimed_bytes": reclaimed_bytes,
            "dry_run": bool(dry_run),
            "nodes": per_node,
        }

    def checkpoint_gc_drain(
        self,
        *,
        timeout_sec: float = 600.0,
        poll_interval_sec: float = 0.1,
    ) -> dict[str, Any]:
        per_node: list[dict[str, Any]] = []
        ok = True
        drained = True
        timed_out = False
        waited_sec = 0.0
        for node in self.nodes:
            try:
                result = self._checkpoint_gc_drain_on_node(
                    node.url,
                    timeout_sec=timeout_sec,
                    poll_interval_sec=poll_interval_sec,
                )
            except Exception as exc:
                ok = False
                drained = False
                per_node.append({"node_url": node.url, "ok": False, "error": str(exc)})
                continue
            per_node.append({"node_url": node.url, **result})
            ok = ok and bool(result.get("ok", False))
            drained = drained and bool(result.get("drained", False))
            timed_out = timed_out or bool(result.get("timed_out", False))
            waited_sec = max(waited_sec, float(result.get("waited_sec", 0.0) or 0.0))
        return {
            "ok": ok,
            "scope": "global",
            "drained": drained,
            "timed_out": timed_out,
            "waited_sec": waited_sec,
            "nodes": per_node,
        }

    def rerun(
        self,
        lease_id: str,
        *,
        checkpoint_id: str | None = None,
        cwd: str | None = None,
        timeout: int = 120,
    ) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        target_checkpoint_id = checkpoint_id or lease.latest_ready_checkpoint_id
        if not target_checkpoint_id:
            raise RuntimeError(f"No ready checkpoint available for lease {lease_id}")
        effective_cwd = cwd or lease.cwd
        result = _post_exec(
            f"{lease.node_url}/container/rerun",
            {
                "checkpoint_id": target_checkpoint_id,
                "old_container_id": lease.container_id,
                "cwd": effective_cwd,
                "timeout": int(timeout),
            },
            timeout=int(timeout) + 30,
        )
        if result.get("ok", False):
            lease.container_id = str(result["new_container_id"])
            lease.cwd = effective_cwd
            lease.generation += 1
            lease.rerun_count += 1
            lease.last_rerun_at = time.time()
            checkpoint_image = str(result.get("checkpoint_image", "")).strip()
            if lease.base_image is None:
                lease.base_image = lease.image
            if checkpoint_image:
                lease.image = checkpoint_image
                lease.current_image = checkpoint_image
        return result

    def stats(self, lease_id: str) -> dict[str, Any]:
        lease = self._get_lease(lease_id)
        result = _post_exec(
            f"{lease.node_url}/container/stats",
            {"container_id": lease.container_id},
            timeout=30,
        )
        if not result.get("ok", False):
            raise RuntimeError(f"Stats failed for lease={lease_id}: {result}")
        return {
            "ok": True,
            "lease_id": lease_id,
            "container_id": lease.container_id,
            "node_url": lease.node_url,
            "image": lease.image,
            "base_image": lease.base_image,
            "current_image": lease.current_image or lease.image,
            "instance_id": lease.instance_id,
            **result,
        }

    def stats_batch(self, lease_ids: list[str]) -> dict[str, Any]:
        requested = [str(item).strip() for item in lease_ids if str(item).strip()]
        if not requested:
            return {"ok": True, "stats": {}}

        stats: dict[str, dict[str, Any]] = {}
        grouped: dict[str, list[tuple[str, Lease]]] = {}
        with self._lock:
            for lease_id in requested:
                lease = self._leases.get(lease_id)
                if lease is None:
                    stats[lease_id] = {
                        "ok": False,
                        "error": f"Unknown lease_id: {lease_id}",
                        "error_code": "unknown_lease_id",
                        "lease_id": lease_id,
                    }
                    continue
                grouped.setdefault(lease.node_url, []).append((lease_id, lease))

        def _decorate(lease_id: str, lease: Lease, item: dict[str, Any]) -> dict[str, Any]:
            payload = dict(item)
            payload.setdefault("ok", False)
            payload.update(
                {
                    "lease_id": lease_id,
                    "container_id": lease.container_id,
                    "node_url": lease.node_url,
                    "image": lease.image,
                    "base_image": lease.base_image,
                    "current_image": lease.current_image or lease.image,
                    "instance_id": lease.instance_id,
                }
            )
            return payload

        for node_url, leases in grouped.items():
            try:
                batch_result = _post_exec(
                    f"{node_url}/container/stats_batch",
                    {"container_ids": [lease.container_id for _, lease in leases]},
                    timeout=30,
                )
                if not batch_result.get("ok", False):
                    raise RuntimeError(f"batch stats failed: {batch_result}")
                by_container = batch_result.get("stats")
                if not isinstance(by_container, dict):
                    raise RuntimeError(f"batch stats returned invalid payload: {batch_result}")
                for lease_id, lease in leases:
                    item = by_container.get(lease.container_id)
                    if not isinstance(item, dict):
                        item = {"ok": False, "error": "missing container stats in batch response"}
                    stats[lease_id] = _decorate(lease_id, lease, item)
                continue
            except Exception as batch_exc:
                logger.debug("stats_batch failed for node=%s, falling back to per-lease stats: %s", node_url, batch_exc)

            for lease_id, lease in leases:
                try:
                    item = _post_exec(
                        f"{lease.node_url}/container/stats",
                        {"container_id": lease.container_id},
                        timeout=30,
                    )
                except Exception as exc:
                    item = {"ok": False, "error": str(exc), "error_code": "stats_unavailable"}
                stats[lease_id] = _decorate(lease_id, lease, item)

        return {"ok": True, "stats": stats}

    def status(self) -> dict[str, Any]:
        with self._lock:
            base = {
                "total_leases": len(self._leases),
                "pending_allocations": self._pending_allocations,
                "max_total_leases": self.max_total_leases,
                "max_concurrent_allocates": self.max_concurrent_allocates,
                "allocate_min_interval_sec": self.allocate_min_interval_sec,
                "create_timeout_sec": self.create_timeout_sec,
            }
            node_snapshots = [
                {
                    "url": n.url,
                    "active_containers": n.active_containers,
                    "max_containers": n.max_containers,
                    "healthy": n.healthy,
                }
                for n in self.nodes
            ]
            lease_snapshots = list(self._leases.values())

        # Query node host stats outside lock to avoid blocking other operations.
        cluster_total_bytes = 0
        cluster_available_bytes = 0
        cluster_free_bytes = 0
        cluster_cpu_total_percent = 0.0
        cluster_cpu_available_percent = 0.0
        cluster_disk_read_total_bps = 0.0
        cluster_disk_read_available_bps = 0.0
        cluster_disk_write_total_bps = 0.0
        cluster_disk_write_available_bps = 0.0
        nodes: list[dict[str, Any]] = []
        for node in node_snapshots:
            memory_total_bytes = 0
            memory_available_bytes = 0
            memory_free_bytes = 0
            cpu_total_percent = 0.0
            cpu_available_percent = 0.0
            disk_read_total_bps = 0.0
            disk_read_available_bps = 0.0
            disk_write_total_bps = 0.0
            disk_write_available_bps = 0.0
            if node["healthy"]:
                try:
                    host = _get_exec(f"{node['url']}/host_stats", timeout=5)
                    if host.get("ok", False):
                        memory_total_bytes = int(host.get("memory_total_bytes", 0) or 0)
                        memory_available_bytes = int(host.get("memory_available_bytes", 0) or 0)
                        memory_free_bytes = int(host.get("memory_free_bytes", 0) or 0)
                        cpu_total_percent = float(host.get("cpu_total_percent", 0.0) or 0.0)
                        cpu_available_percent = float(host.get("cpu_available_percent", 0.0) or 0.0)
                        disk_read_total_bps = float(host.get("disk_read_total_bytes_per_sec", 0.0) or 0.0)
                        disk_read_available_bps = float(host.get("disk_read_available_bytes_per_sec", 0.0) or 0.0)
                        disk_write_total_bps = float(host.get("disk_write_total_bytes_per_sec", 0.0) or 0.0)
                        disk_write_available_bps = float(host.get("disk_write_available_bytes_per_sec", 0.0) or 0.0)
                except Exception:
                    # Keep status endpoint robust; health check pipeline handles liveness.
                    pass

            node_out = {
                **node,
                "memory_total_bytes": memory_total_bytes,
                "memory_available_bytes": memory_available_bytes,
                "memory_free_bytes": memory_free_bytes,
                "cpu_total_percent": cpu_total_percent,
                "cpu_available_percent": cpu_available_percent,
                "disk_read_total_bytes_per_sec": disk_read_total_bps,
                "disk_read_available_bytes_per_sec": disk_read_available_bps,
                "disk_write_total_bytes_per_sec": disk_write_total_bps,
                "disk_write_available_bytes_per_sec": disk_write_available_bps,
            }
            nodes.append(node_out)
            cluster_total_bytes += max(0, memory_total_bytes)
            cluster_available_bytes += max(0, memory_available_bytes)
            cluster_free_bytes += max(0, memory_free_bytes)
            cluster_cpu_total_percent += max(0.0, cpu_total_percent)
            cluster_cpu_available_percent += max(0.0, cpu_available_percent)
            cluster_disk_read_total_bps += max(0.0, disk_read_total_bps)
            cluster_disk_read_available_bps += max(0.0, disk_read_available_bps)
            cluster_disk_write_total_bps += max(0.0, disk_write_total_bps)
            cluster_disk_write_available_bps += max(0.0, disk_write_available_bps)

        return {
            **base,
            "cluster_memory_total_bytes": cluster_total_bytes,
            "cluster_memory_available_bytes": cluster_available_bytes,
            "cluster_memory_free_bytes": cluster_free_bytes,
            "cluster_cpu_total_percent": cluster_cpu_total_percent,
            "cluster_cpu_available_percent": cluster_cpu_available_percent,
            "cluster_disk_read_total_bytes_per_sec": cluster_disk_read_total_bps,
            "cluster_disk_read_available_bytes_per_sec": cluster_disk_read_available_bps,
            "cluster_disk_write_total_bytes_per_sec": cluster_disk_write_total_bps,
            "cluster_disk_write_available_bytes_per_sec": cluster_disk_write_available_bps,
            "nodes": nodes,
            "leases": [
                {
                    "lease_id": lease.lease_id,
                    "node_url": lease.node_url,
                    "container_id": lease.container_id,
                    "image": lease.image,
                    "base_image": lease.base_image,
                    "current_image": lease.current_image or lease.image,
                    "instance_id": lease.instance_id,
                    "cwd": lease.cwd,
                    "generation": lease.generation,
                    "latest_ready_checkpoint_id": lease.latest_ready_checkpoint_id,
                    "latest_checkpoint_step_idx": lease.latest_checkpoint_step_idx,
                    "rerun_count": lease.rerun_count,
                    "created_at": lease.created_at,
                    "last_heartbeat": lease.last_heartbeat,
                    "last_rerun_at": lease.last_rerun_at,
                }
                for lease in lease_snapshots
            ],
        }

    def health_check(self) -> None:
        for node in self.nodes:
            ok = False
            try:
                r = _get_exec(f"{node.url}/healthz", timeout=self.health_check_timeout_sec)
                ok = bool(r.get("ok", False))
            except Exception:
                ok = False
            if ok:
                node.healthy = True
                node.consecutive_health_check_failures = 0
            else:
                node.consecutive_health_check_failures += 1
                if node.consecutive_health_check_failures >= self.health_check_failure_threshold:
                    node.healthy = False
            node.last_health_check = time.time()


# ── Flask routes ──────────────────────────────────────────────────────

POOL: SweEnvPool | None = None


@app.get("/healthz")
def healthz():
    return jsonify({"ok": True})


@app.get("/status")
def status():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    return jsonify({"ok": True, "pool": POOL.status()})


@app.post("/allocate")
def allocate():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    image = data.get("image")
    instance_id = data.get("instance_id", "")
    if not image:
        return jsonify({"ok": False, "error": "image is required"}), 400
    logger.info("[SWE-POOL] Allocate request: image=%s instance_id=%s", image, instance_id)
    try:
        result = POOL.allocate(image=image, instance_id=instance_id, cwd=data.get("cwd", "/testbed"))
        return jsonify({"ok": True, **result})
    except RuntimeError as e:
        # Capacity pressure is expected under high concurrency. Return app-level
        # retryable error instead of HTTP 500 to avoid transport-layer hard fail.
        err = str(e)
        err_lower = err.lower()
        if (
            "global lease limit reached" in err_lower
            or "no healthy exec nodes available" in err_lower
        ):
            return jsonify(
                {
                    "ok": False,
                    "error": err,
                    "error_code": "capacity_limited",
                    "retryable": True,
                    "retry_after_sec": 1.0,
                }
            )
        return jsonify({"ok": False, "error": err}), 500
    except requests.exceptions.Timeout as e:
        return jsonify(
            {
                "ok": False,
                "error": str(e),
                "error_code": "create_timeout",
                "retryable": True,
                "retry_after_sec": 2.0,
            }
        )
    except requests.exceptions.RequestException as e:
        err = str(e)
        err_lower = err.lower()
        if any(marker in err_lower for marker in ("timed out", "connection aborted", "connection reset", "remote disconnected")):
            return jsonify(
                {
                    "ok": False,
                    "error": err,
                    "error_code": "create_transport_error",
                    "retryable": True,
                    "retry_after_sec": 2.0,
                }
            )
        return jsonify({"ok": False, "error": err}), 500
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/heartbeat")
def heartbeat():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        POOL.heartbeat(lease_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/exec")
def exec_cmd():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    command = data.get("command")
    if not lease_id or not command:
        return jsonify({"ok": False, "error": "lease_id and command required"}), 400
    try:
        result = POOL.exec(
            lease_id=lease_id,
            command=command,
            cwd=data.get("cwd", "/testbed"),
            timeout=int(data.get("timeout", 180)),
            env=data.get("env"),
            fault_injection_armed=bool(data.get("fault_injection_armed", False)),
            fault_injection_probability=data.get("fault_injection_probability"),
            fault_injection_spec=data.get("fault_injection_spec"),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/diff")
def diff():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.diff(lease_id=lease_id, cwd=data.get("cwd", "/testbed"))
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/evaluate")
def evaluate():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    patch = data.get("patch", "")
    eval_script = data.get("eval_script", "")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.evaluate(
            lease_id=lease_id,
            patch=patch,
            eval_script=eval_script,
            cwd=data.get("cwd", "/testbed"),
            timeout=int(data.get("timeout", 3600)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/close")
def close():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        POOL.close(lease_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/fault/kill")
def fault_kill():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.inject_fail_stop(
            lease_id=str(lease_id),
            tag=str(data.get("tag", "") or ""),
            delay_sec=float(data.get("delay_sec", 0.0) or 0.0),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/stats")
def stats():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.stats(lease_id=lease_id)
        return jsonify(result)
    except KeyError as e:
        # Unknown lease is a normal race when sampler checks stats while a lease
        # is being detached/closed; return app-level error to avoid HTTP retries.
        return jsonify({"ok": False, "error": str(e), "error_code": "unknown_lease_id", "lease_id": lease_id})
    except Exception as e:
        return jsonify(
            {
                "ok": False,
                "error": str(e),
                "error_code": "stats_unavailable",
                "retryable": False,
                "lease_id": lease_id,
            }
        )


@app.post("/stats_batch")
def stats_batch():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_ids = data.get("lease_ids")
    if not isinstance(lease_ids, list) or not lease_ids:
        return jsonify({"ok": False, "error": "lease_ids must be a non-empty list"}), 400
    try:
        return jsonify(POOL.stats_batch([str(item) for item in lease_ids]))
    except Exception as e:
        return jsonify(
            {
                "ok": False,
                "error": str(e),
                "error_code": "stats_batch_unavailable",
                "retryable": False,
            }
        )


@app.post("/checkpoint/probe")
def checkpoint_probe():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.checkpoint_probe(lease_id=lease_id)
        return jsonify(result)
    except requests.exceptions.Timeout:
        return jsonify(
            {
                "ok": True,
                "busy": True,
                "reason": "checkpoint_probe_timeout",
                "probe_timeout_sec": CHECKPOINT_PROBE_TIMEOUT_SEC,
                "probe_wait_sec": float(CHECKPOINT_PROBE_TIMEOUT_SEC),
                "retry_after_sec": 1.0,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/checkpoint/create")
def checkpoint_create():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.checkpoint_create(
            lease_id=lease_id,
            step_idx=int(data.get("step_idx", -1)),
            command_seq=int(data.get("command_seq", -1)),
            cwd=data.get("cwd"),
            env=data.get("env"),
            policy=str(data.get("policy", "")),
            reason=str(data.get("reason", "manual")),
            parent_checkpoint_id=data.get("parent_checkpoint_id"),
            fault_injection_spec=data.get("fault_injection_spec"),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/checkpoint/status")
def checkpoint_status():
    return jsonify(
        {
            "ok": False,
            "error": "checkpoint_status is deprecated; checkpoint_create is synchronous",
            "error_code": "checkpoint_status_deprecated",
            "retryable": False,
        }
    )


@app.post("/checkpoint/list")
def checkpoint_list():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    try:
        result = POOL.checkpoint_list(lease_id=str(lease_id) if lease_id else None)
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/checkpoint/delete")
def checkpoint_delete():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    checkpoint_id = data.get("checkpoint_id")
    if not lease_id or not checkpoint_id:
        return jsonify({"ok": False, "error": "lease_id and checkpoint_id required"}), 400
    try:
        result = POOL.checkpoint_delete(lease_id=lease_id, checkpoint_id=str(checkpoint_id))
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/checkpoint/gc")
def checkpoint_gc():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    try:
        result = POOL.checkpoint_gc(
            lease_id=str(lease_id) if lease_id else None,
            keep_latest=int(data.get("keep_latest", 1)),
            dry_run=bool(data.get("dry_run", False)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/checkpoint/gc/drain")
def checkpoint_gc_drain():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    try:
        result = POOL.checkpoint_gc_drain(
            timeout_sec=float(data.get("timeout_sec", 600.0)),
            poll_interval_sec=float(data.get("poll_interval_sec", 0.1)),
        )
        return jsonify(result), (200 if result.get("ok", False) else 409)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.post("/rerun")
def rerun():
    if POOL is None:
        return jsonify({"ok": False, "error": "Pool not initialized"}), 500
    data = flask_request.get_json(force=True) or {}
    lease_id = data.get("lease_id")
    if not lease_id:
        return jsonify({"ok": False, "error": "lease_id required"}), 400
    try:
        result = POOL.rerun(
            lease_id=lease_id,
            checkpoint_id=data.get("checkpoint_id"),
            cwd=data.get("cwd"),
            timeout=int(data.get("timeout", 120)),
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


# ── Main ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    sched_enabled = os.getenv("SWE_ENABLE_ONLINE_ENV_DOCKER_SCHEDULER", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    max_total_leases_env = os.getenv("SWE_POOL_MAX_TOTAL_LEASES")
    if max_total_leases_env is not None:
        default_max_total_leases = int(max_total_leases_env)
    else:
        # Scheduler mode: avoid global hard cap by default.
        # Non-scheduler mode: keep legacy behavior tied to SWE_MAX_CONCURRENT.
        default_max_total_leases = 0 if sched_enabled else int(os.getenv("SWE_MAX_CONCURRENT", "0"))
    default_max_concurrent_allocates = int(
        os.getenv("SWE_POOL_MAX_CONCURRENT_ALLOCATES", "16" if sched_enabled else "1")
    )
    default_allocate_min_interval_sec = float(
        os.getenv("SWE_POOL_ALLOCATE_MIN_INTERVAL_SEC", "0.05" if sched_enabled else "0.5")
    )

    parser = argparse.ArgumentParser(description="SWE environment pool server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=int(os.getenv("SWE_ENV_SERVER_PORT", "18090")))
    parser.add_argument(
        "--exec-server-urls",
        type=str,
        default=os.getenv("SWE_EXEC_SERVER_URLS", "http://localhost:5000"),
        help="Comma-separated swe_exec_server URLs",
    )
    parser.add_argument(
        "--max-containers-per-node",
        type=int,
        default=int(os.getenv("SWE_MAX_CONTAINERS_PER_NODE", "16")),
    )
    parser.add_argument(
        "--max-total-leases",
        type=int,
        default=default_max_total_leases,
    )
    parser.add_argument(
        "--max-concurrent-allocates",
        type=int,
        default=default_max_concurrent_allocates,
    )
    parser.add_argument(
        "--allocate-min-interval-sec",
        type=float,
        default=default_allocate_min_interval_sec,
    )
    parser.add_argument(
        "--create-timeout-sec",
        type=float,
        default=float(os.getenv("SWE_POOL_CREATE_TIMEOUT_SEC", "120")),
    )
    parser.add_argument(
        "--health-check-timeout-sec",
        type=float,
        default=float(os.getenv("SWE_POOL_HEALTH_CHECK_TIMEOUT_SEC", "30")),
    )
    parser.add_argument(
        "--health-check-failure-threshold",
        type=int,
        default=int(os.getenv("SWE_POOL_HEALTH_CHECK_FAILURE_THRESHOLD", "3")),
    )
    return parser.parse_args()


def _periodic_health_check(pool: "SweEnvPool", interval: int = 30) -> None:
    """Background thread: re-check all exec nodes every `interval` seconds."""
    while True:
        time.sleep(interval)
        try:
            pool.health_check()
            healthy = sum(1 for n in pool.nodes if n.healthy)
            logger.info("[SWE-POOL] Periodic health check: %d/%d nodes healthy", healthy, len(pool.nodes))
        except Exception as exc:
            logger.warning("[SWE-POOL] Periodic health check error: %s", exc)


def main() -> None:
    global POOL
    args = parse_args()
    sched_enabled = os.getenv("SWE_ENABLE_ONLINE_ENV_DOCKER_SCHEDULER", "").strip().lower() in {
        "1", "true", "yes", "on",
    }
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s %(levelname)s %(name)s] %(message)s")

    # In online scheduler mode, per-node hard cap is intentionally disabled.
    # Concurrency is fully decided by the scheduler algorithm.
    effective_max_containers_per_node = args.max_containers_per_node
    if sched_enabled and args.max_containers_per_node > 0:
        logger.info(
            "Online scheduler enabled: ignoring --max-containers-per-node=%d; using unlimited per-node containers",
            args.max_containers_per_node,
        )
        effective_max_containers_per_node = 0

    urls = [u.strip() for u in args.exec_server_urls.split(",") if u.strip()]
    POOL = SweEnvPool(
        exec_server_urls=urls,
        max_containers_per_node=effective_max_containers_per_node,
        max_total_leases=args.max_total_leases,
        max_concurrent_allocates=args.max_concurrent_allocates,
        allocate_min_interval_sec=args.allocate_min_interval_sec,
        create_timeout_sec=args.create_timeout_sec,
        health_check_timeout_sec=args.health_check_timeout_sec,
        health_check_failure_threshold=args.health_check_failure_threshold,
    )
    POOL.health_check()

    healthy = sum(1 for n in POOL.nodes if n.healthy)
    max_containers_text = (
        "unlimited" if effective_max_containers_per_node <= 0 else str(effective_max_containers_per_node)
    )
    logger.info(
        "SWE env pool: %d/%d nodes healthy, max %s containers/node, max_total_leases=%d, "
        "max_concurrent_allocates=%d, allocate_min_interval_sec=%.2f, create_timeout_sec=%.1f, "
        "health_check_timeout_sec=%.1f, health_check_failure_threshold=%d, listening on %s:%s",
        healthy,
        len(POOL.nodes),
        max_containers_text,
        args.max_total_leases,
        args.max_concurrent_allocates,
        args.allocate_min_interval_sec,
        args.create_timeout_sec,
        args.health_check_timeout_sec,
        args.health_check_failure_threshold,
        args.host,
        args.port,
    )

    hc_thread = threading.Thread(target=_periodic_health_check, args=(POOL,), daemon=True)
    hc_thread.start()
    logger.info("[SWE-POOL] Periodic health check thread started (interval=30s)")

    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
