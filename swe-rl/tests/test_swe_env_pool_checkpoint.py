from __future__ import annotations

import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import requests
import sys

fake_flask = ModuleType("flask")


class _FakeFlaskApp:
    def __init__(self, name: str):
        self.name = name

    def get(self, _route: str):
        def decorator(func):
            return func

        return decorator

    def post(self, _route: str):
        def decorator(func):
            return func

        return decorator

    def run(self, *args, **kwargs):
        return None


fake_flask.Flask = _FakeFlaskApp
fake_flask.jsonify = lambda payload=None, **kwargs: payload if payload is not None else kwargs
fake_flask.request = type("_FakeRequest", (), {"get_json": staticmethod(lambda force=False: {})})()
sys.modules.setdefault("flask", fake_flask)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import swe_env_pool_server as pool_server  # noqa: E402


def make_pool(health_check_failure_threshold: int = 3) -> pool_server.SweEnvPool:
    return pool_server.SweEnvPool(
        exec_server_urls=["http://node-1:5000"],
        max_containers_per_node=4,
        max_total_leases=0,
        max_concurrent_allocates=1,
        allocate_min_interval_sec=0.0,
        create_timeout_sec=30.0,
        health_check_failure_threshold=health_check_failure_threshold,
    )


class TestSweEnvPoolCheckpoint(unittest.TestCase):
    def test_health_check_requires_consecutive_failures_before_marking_unhealthy(self):
        pool = make_pool(health_check_failure_threshold=3)

        with patch.object(pool_server, "_get_exec", return_value={"ok": True}):
            pool.health_check()
        self.assertTrue(pool.nodes[0].healthy)
        self.assertEqual(pool.nodes[0].consecutive_health_check_failures, 0)

        with patch.object(pool_server, "_get_exec", side_effect=requests.RequestException("transient")):
            pool.health_check()
        self.assertTrue(pool.nodes[0].healthy)
        self.assertEqual(pool.nodes[0].consecutive_health_check_failures, 1)

        with patch.object(pool_server, "_get_exec", side_effect=requests.RequestException("transient")):
            pool.health_check()
        self.assertTrue(pool.nodes[0].healthy)
        self.assertEqual(pool.nodes[0].consecutive_health_check_failures, 2)

        with patch.object(pool_server, "_get_exec", side_effect=requests.RequestException("transient")):
            pool.health_check()
        self.assertFalse(pool.nodes[0].healthy)
        self.assertEqual(pool.nodes[0].consecutive_health_check_failures, 3)

        with patch.object(pool_server, "_get_exec", return_value={"ok": True}):
            pool.health_check()
        self.assertTrue(pool.nodes[0].healthy)
        self.assertEqual(pool.nodes[0].consecutive_health_check_failures, 0)

    def test_initial_health_check_failure_marks_node_unhealthy_immediately(self):
        pool = make_pool(health_check_failure_threshold=3)

        with patch.object(pool_server, "_get_exec", side_effect=requests.RequestException("boot failure")):
            pool.health_check()

        self.assertTrue(pool.nodes[0].healthy)
        self.assertEqual(pool.nodes[0].consecutive_health_check_failures, 1)

    def test_checkpoint_probe_timeout_returns_busy_payload(self):
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            cwd="/workspace",
            generation=2,
        )
        fake_pool = make_pool()
        fake_pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        fake_request = type("_Req", (), {"get_json": staticmethod(lambda force=False: {"lease_id": "lease-1"})})()

        with patch.object(pool_server, "POOL", fake_pool), \
             patch.object(pool_server, "flask_request", fake_request), \
             patch.object(fake_pool, "checkpoint_probe", side_effect=requests.exceptions.Timeout("probe timeout")):
            out = pool_server.checkpoint_probe()

        self.assertTrue(out["ok"])
        self.assertTrue(out["busy"])
        self.assertEqual(out["reason"], "checkpoint_probe_timeout")
        self.assertEqual(out["probe_timeout_sec"], pool_server.CHECKPOINT_PROBE_TIMEOUT_SEC)

    def test_post_exec_returns_json_payload_on_http_error(self):
        class _FakeResponse:
            status_code = 409
            text = '{"ok": false}'

            def json(self):
                return {
                    "ok": False,
                    "error_code": "checkpoint_not_ready",
                    "retryable": True,
                }

            def raise_for_status(self):
                raise requests.HTTPError("should not be raised")

        with patch.object(pool_server.requests, "post", return_value=_FakeResponse()):
            out = pool_server._post_exec("http://node-1:5000/container/rerun", {"checkpoint_id": "ckpt-1"})

        self.assertEqual(out["error_code"], "checkpoint_not_ready")
        self.assertTrue(out["retryable"])

    def test_checkpoint_create_forwards_lease_metadata(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            cwd="/workspace",
            generation=2,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        seen: list[tuple[str, dict, int]] = []

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            seen.append((url, payload, timeout))
            return {"ok": True, "checkpoint_id": "ckpt-1"}

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_create(
                "lease-1",
                step_idx=5,
                command_seq=8,
                policy="adaptive-risk",
                reason="after_exec",
            )

        self.assertTrue(out["ok"])
        self.assertEqual(seen[0][0], "http://node-1:5000/container/checkpoint/create")
        self.assertEqual(
            seen[0][1],
            {
                "lease_id": "lease-1",
                "container_id": "cid-1",
                "generation": 2,
                "instance_id": "inst-1",
                "cwd": "/workspace",
                "step_idx": 5,
                "command_seq": 8,
                "policy": "adaptive-risk",
                "reason": "after_exec",
                "parent_checkpoint_id": None,
            },
        )

    def test_checkpoint_create_forwards_runtime_env_when_present(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            cwd="/workspace",
            generation=2,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        seen: list[tuple[str, dict, int]] = []

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            seen.append((url, payload, timeout))
            return {"ok": True, "checkpoint_id": "ckpt-1"}

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_create(
                "lease-1",
                cwd="/workspace",
                env={"PATH": "/opt/venv/bin:/usr/bin", "VIRTUAL_ENV": "/opt/venv"},
            )

        self.assertTrue(out["ok"])
        self.assertEqual(
            seen[0][1]["env"],
            {"PATH": "/opt/venv/bin:/usr/bin", "VIRTUAL_ENV": "/opt/venv"},
        )

    def test_exec_forwards_fault_injection_spec_when_present(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            cwd="/workspace",
            generation=2,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        seen: list[tuple[str, dict, int]] = []

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            seen.append((url, payload, timeout))
            return {"ok": True, "returncode": -1}

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.exec(
                "lease-1",
                "sleep 1",
                fault_injection_spec={"phase": "mid_action", "delay_sec": 0.5},
            )

        self.assertTrue(out["ok"])
        self.assertEqual(
            seen[0][1]["fault_injection_spec"],
            {"phase": "mid_action", "delay_sec": 0.5},
        )

    def test_inject_fail_stop_forwards_to_current_container_without_closing_lease(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            cwd="/workspace",
            generation=2,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        seen: list[tuple[str, dict, int]] = []

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            seen.append((url, payload, timeout))
            return {"ok": True, "fault_injected": True}

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.inject_fail_stop("lease-1", tag="trial-1", delay_sec=0.25)

        self.assertTrue(out["ok"])
        self.assertTrue(out["fault_injected"])
        self.assertIn("lease-1", pool._leases)  # pylint: disable=protected-access
        self.assertEqual(seen[0][0], "http://node-1:5000/container/fault/kill")
        self.assertEqual(seen[0][1], {"container_id": "cid-1", "tag": "trial-1", "delay_sec": 0.25})

    def test_checkpoint_create_forwards_fault_injection_spec_when_present(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            cwd="/workspace",
            generation=2,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        seen: list[tuple[str, dict, int]] = []

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            seen.append((url, payload, timeout))
            return {"ok": True, "checkpoint_id": "ckpt-1"}

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_create(
                "lease-1",
                fault_injection_spec={"phase": "after_commit_before_ready"},
            )

        self.assertTrue(out["ok"])
        self.assertEqual(
            seen[0][1]["fault_injection_spec"],
            {"phase": "after_commit_before_ready"},
        )

    def test_synchronous_checkpoint_create_updates_latest_ready_pointer(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            generation=2,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access

        def fake_post(_url: str, _payload: dict, timeout: int = 300) -> dict:
            del timeout
            return {
                "ok": True,
                "status": "ready",
                "checkpoint_id": "ckpt-full",
                "checkpoint_backend": "full",
                "container_id": "cid-1",
                "generation": 2,
                "step_idx": 9,
            }

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_create("lease-1", step_idx=9)

        self.assertTrue(out["ok"])
        self.assertEqual(lease.latest_ready_checkpoint_id, "ckpt-full")
        self.assertEqual(lease.latest_checkpoint_step_idx, 9)

    def test_full_rerun_keeps_restored_base_image_when_checkpoint_image_is_none(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-old",
            image="img:base",
            instance_id="inst-1",
            latest_ready_checkpoint_id="ckpt-full",
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access

        def fake_post(_url: str, _payload: dict, timeout: int = 300) -> dict:
            del timeout
            return {
                "ok": True,
                "checkpoint_id": "ckpt-full",
                "checkpoint_backend": "full",
                "checkpoint_image": None,
                "restored_image": "img:base",
                "new_container_id": "cid-new",
            }

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.rerun("lease-1")

        self.assertTrue(out["ok"])
        self.assertEqual(lease.image, "img:base")
        self.assertEqual(lease.current_image, "img:base")
        self.assertNotEqual(lease.image, "None")

    def test_checkpoint_status_is_deprecated(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            generation=3,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access

        out = pool.checkpoint_status("lease-1", "ckpt-7")

        self.assertFalse(out["ok"])
        self.assertEqual(out["error_code"], "checkpoint_status_deprecated")
        self.assertIsNone(lease.latest_ready_checkpoint_id)

    def test_rerun_updates_lease_state(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-old",
            image="img:base",
            instance_id="inst-1",
            cwd="/workspace",
            latest_ready_checkpoint_id="ckpt-9",
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        seen: list[tuple[str, dict, int]] = []

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            seen.append((url, payload, timeout))
            return {
                "ok": True,
                "checkpoint_id": "ckpt-9",
                "checkpoint_image": "sweckpt:ckpt-9",
                "new_container_id": "cid-new",
            }

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.rerun("lease-1", timeout=99)

        self.assertTrue(out["ok"])
        self.assertEqual(seen[0][0], "http://node-1:5000/container/rerun")
        self.assertEqual(
            seen[0][1],
            {
                "checkpoint_id": "ckpt-9",
                "lease_id": "lease-1",
                "old_container_id": "cid-old",
                "cwd": "/workspace",
                "timeout": 99,
            },
        )
        self.assertEqual(lease.container_id, "cid-new")
        self.assertEqual(lease.generation, 1)
        self.assertEqual(lease.rerun_count, 1)
        self.assertEqual(lease.image, "sweckpt:ckpt-9")
        self.assertEqual(lease.base_image, "img:base")
        self.assertEqual(lease.current_image, "sweckpt:ckpt-9")

    def test_checkpoint_gc_clears_deleted_latest_checkpoint(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            latest_ready_checkpoint_id="ckpt-2",
            latest_checkpoint_step_idx=9,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            if url == "http://node-1:5000/container/checkpoint/gc":
                self.assertEqual(payload, {"lease_id": "lease-1", "keep_latest": 0, "dry_run": False})
                return {"ok": True, "deleted_checkpoint_ids": ["ckpt-2"]}
            if url == "http://node-1:5000/container/checkpoint/list":
                return {"ok": True, "checkpoints": []}
            raise AssertionError(url)

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_gc("lease-1", keep_latest=0, dry_run=False)

        self.assertTrue(out["ok"])
        self.assertIsNone(lease.latest_ready_checkpoint_id)
        self.assertEqual(lease.latest_checkpoint_step_idx, -1)

    def test_checkpoint_delete_refreshes_previous_ready_checkpoint(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
            latest_ready_checkpoint_id="ckpt-2",
            latest_checkpoint_step_idx=9,
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        seen: list[tuple[str, dict, int]] = []

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            seen.append((url, payload, timeout))
            if url.endswith("/container/checkpoint/delete"):
                return {"ok": True, "deleted": True}
            if url.endswith("/container/checkpoint/list"):
                return {
                    "ok": True,
                    "checkpoints": [
                        {"checkpoint_id": "ckpt-1", "status": "ready", "step_idx": 3, "created_at": 1.0},
                    ],
                }
            raise AssertionError(url)

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_delete("lease-1", "ckpt-2")

        self.assertTrue(out["ok"])
        self.assertEqual(lease.latest_ready_checkpoint_id, "ckpt-1")
        self.assertEqual(lease.latest_checkpoint_step_idx, 3)
        self.assertEqual(len(seen), 2)

    def test_checkpoint_gc_without_lease_aggregates_all_nodes(self):
        pool = pool_server.SweEnvPool(
            exec_server_urls=["http://node-1:5000", "http://node-2:5000"],
            max_containers_per_node=4,
            max_total_leases=0,
            max_concurrent_allocates=1,
            allocate_min_interval_sec=0.0,
            create_timeout_sec=30.0,
        )

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            self.assertNotIn("lease_id", payload)
            if url.startswith("http://node-1:5000"):
                return {"ok": True, "deleted_count": 1, "deleted_checkpoint_ids": ["ckpt-1"], "reclaimed_bytes": 10}
            if url.startswith("http://node-2:5000"):
                return {"ok": True, "deleted_count": 2, "deleted_checkpoint_ids": ["ckpt-2", "ckpt-3"], "reclaimed_bytes": 20}
            raise AssertionError(url)

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_gc(keep_latest=0, dry_run=True)

        self.assertTrue(out["ok"])
        self.assertEqual(out["scope"], "global")
        self.assertEqual(out["deleted_count"], 3)
        self.assertEqual(out["reclaimed_bytes"], 30)
        self.assertEqual(out["deleted_checkpoint_ids"], ["ckpt-1", "ckpt-2", "ckpt-3"])

    def test_checkpoint_list_without_lease_aggregates_all_nodes(self):
        pool = pool_server.SweEnvPool(
            exec_server_urls=["http://node-1:5000", "http://node-2:5000"],
            max_containers_per_node=4,
            max_total_leases=0,
            max_concurrent_allocates=1,
            allocate_min_interval_sec=0.0,
            create_timeout_sec=30.0,
        )

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            self.assertEqual(payload, {})
            if url == "http://node-1:5000/container/checkpoint/list":
                return {"ok": True, "checkpoints": [{"checkpoint_id": "ckpt-1"}]}
            if url == "http://node-2:5000/container/checkpoint/list":
                return {"ok": True, "checkpoints": [{"checkpoint_id": "ckpt-2"}, {"checkpoint_id": "ckpt-3"}]}
            raise AssertionError(url)

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_list()

        self.assertTrue(out["ok"])
        self.assertEqual([item["node_url"] for item in out["per_node"]], ["http://node-1:5000", "http://node-2:5000"])
        self.assertEqual([item["checkpoint_count"] for item in out["per_node"]], [1, 2])
        self.assertEqual([item["checkpoint_id"] for item in out["checkpoints"]], ["ckpt-1", "ckpt-2", "ckpt-3"])
        self.assertEqual(
            [item["node_url"] for item in out["checkpoints"]],
            ["http://node-1:5000", "http://node-2:5000", "http://node-2:5000"],
        )

    def test_checkpoint_gc_drain_without_lease_aggregates_all_nodes(self):
        pool = pool_server.SweEnvPool(
            exec_server_urls=["http://node-1:5000", "http://node-2:5000"],
            max_containers_per_node=4,
            max_total_leases=0,
            max_concurrent_allocates=1,
            allocate_min_interval_sec=0.0,
            create_timeout_sec=30.0,
        )

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            self.assertEqual(payload, {"timeout_sec": 12.0, "poll_interval_sec": 0.25})
            if url == "http://node-1:5000/container/checkpoint/gc/drain":
                return {"ok": True, "drained": True, "timed_out": False, "waited_sec": 1.5}
            if url == "http://node-2:5000/container/checkpoint/gc/drain":
                return {"ok": True, "drained": True, "timed_out": False, "waited_sec": 2.0}
            raise AssertionError(url)

        with patch.object(pool_server, "_post_exec", new=fake_post):
            out = pool.checkpoint_gc_drain(timeout_sec=12.0, poll_interval_sec=0.25)

        self.assertTrue(out["ok"])
        self.assertTrue(out["drained"])
        self.assertFalse(out["timed_out"])
        self.assertEqual(out["waited_sec"], 2.0)
        self.assertEqual([item["node_url"] for item in out["nodes"]], ["http://node-1:5000", "http://node-2:5000"])

    def test_close_only_destroys_container_for_closed_lease(self):
        pool = make_pool()
        lease = pool_server.Lease(
            lease_id="lease-1",
            node_url="http://node-1:5000",
            container_id="cid-1",
            image="img:base",
            instance_id="inst-1",
        )
        pool._leases[lease.lease_id] = lease  # pylint: disable=protected-access
        seen: list[tuple[str, dict, int]] = []

        def fake_post(url: str, payload: dict, timeout: int = 300) -> dict:
            seen.append((url, payload, timeout))
            if url.endswith("/container/destroy"):
                return {"ok": True}
            raise AssertionError(url)

        with patch.object(pool_server, "_post_exec", new=fake_post):
            pool.close("lease-1")

        self.assertNotIn("lease-1", pool._leases)  # pylint: disable=protected-access
        self.assertEqual(
            seen,
            [
                ("http://node-1:5000/container/destroy", {"container_id": "cid-1"}, 300),
            ],
        )


if __name__ == "__main__":
    unittest.main()
