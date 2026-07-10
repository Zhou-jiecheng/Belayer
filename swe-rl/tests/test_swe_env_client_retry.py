from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch
import types

import sys

# `swe_env_client` imports `slime.utils.http_utils`, so add both roots.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "slime"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

if "httpx" not in sys.modules:
    # Minimal stub for import-time dependencies inside slime.utils.http_utils.
    sys.modules["httpx"] = types.SimpleNamespace(
        AsyncClient=object,
        Limits=object,
        Timeout=object,
        HTTPStatusError=Exception,
    )

from swe_env_client import SweEnvClient  # noqa: E402


class TestSweEnvClientRetry(unittest.IsolatedAsyncioTestCase):
    async def test_retry_on_retryable_app_error_then_success(self):
        attempts = {"n": 0}

        async def fake_post(url, payload, max_retries=60):
            attempts["n"] += 1
            if attempts["n"] < 3:
                return {"ok": False, "error": "All exec nodes are full or unhealthy."}
            return {"ok": True, "lease_id": "lease-1"}

        client = SweEnvClient(base_url="http://fake")
        client.app_error_max_retries = 4
        client.app_error_retry_delay_sec = 0.0
        client.app_error_retry_jitter_sec = 0.0

        with patch("swe_env_client.post", new=fake_post):
            out = await client.allocate(image="img", instance_id="iid")

        self.assertTrue(out["ok"])
        self.assertEqual(attempts["n"], 3)

    async def test_do_not_retry_on_non_retryable_error(self):
        attempts = {"n": 0}

        async def fake_post(url, payload, max_retries=60):
            attempts["n"] += 1
            return {"ok": False, "error": "Unknown lease_id: bad-id"}

        client = SweEnvClient(base_url="http://fake")
        client.app_error_max_retries = 5
        client.app_error_retry_delay_sec = 0.0
        client.app_error_retry_jitter_sec = 0.0

        with patch("swe_env_client.post", new=fake_post):
            with self.assertRaises(RuntimeError):
                await client.heartbeat("bad-id")

        self.assertEqual(attempts["n"], 1)

    async def test_allocate_sends_cwd(self):
        seen: list[tuple[str, dict]] = []

        async def fake_post(url, payload, max_retries=60):
            seen.append((url, payload))
            return {"ok": True, "lease_id": "lease-1"}

        client = SweEnvClient(base_url="http://fake")
        with patch("swe_env_client.post", new=fake_post):
            out = await client.allocate(image="img", instance_id="iid", cwd="/workspace")

        self.assertTrue(out["ok"])
        self.assertEqual(seen[0][0], "http://fake/allocate")
        self.assertEqual(seen[0][1]["cwd"], "/workspace")

    async def test_checkpoint_create_payload(self):
        seen: list[tuple[str, dict]] = []

        async def fake_post(url, payload, max_retries=60):
            seen.append((url, payload))
            return {"ok": True, "checkpoint_id": "ckpt-1"}

        client = SweEnvClient(base_url="http://fake")
        with patch("swe_env_client.post", new=fake_post):
            out = await client.checkpoint_create(
                "lease-1",
                step_idx=3,
                command_seq=4,
                cwd="/testbed",
                policy="adaptive-risk",
                reason="after_exec",
                parent_checkpoint_id="ckpt-0",
            )

        self.assertTrue(out["ok"])
        self.assertEqual(seen[0][0], "http://fake/checkpoint/create")
        self.assertEqual(
            seen[0][1],
            {
                "lease_id": "lease-1",
                "step_idx": 3,
                "command_seq": 4,
                "cwd": "/testbed",
                "policy": "adaptive-risk",
                "reason": "after_exec",
                "parent_checkpoint_id": "ckpt-0",
            },
        )

    async def test_checkpoint_create_payload_with_env(self):
        seen: list[tuple[str, dict]] = []

        async def fake_post(url, payload, max_retries=60):
            seen.append((url, payload))
            return {"ok": True, "checkpoint_id": "ckpt-1"}

        client = SweEnvClient(base_url="http://fake")
        with patch("swe_env_client.post", new=fake_post):
            out = await client.checkpoint_create(
                "lease-1",
                cwd="/testbed",
                env={"PATH": "/opt/venv/bin:/usr/bin", "VIRTUAL_ENV": "/opt/venv"},
            )

        self.assertTrue(out["ok"])
        self.assertEqual(
            seen[0][1],
            {
                "lease_id": "lease-1",
                "step_idx": -1,
                "command_seq": -1,
                "cwd": "/testbed",
                "reason": "manual",
                "policy": "",
                "env": {"PATH": "/opt/venv/bin:/usr/bin", "VIRTUAL_ENV": "/opt/venv"},
            },
        )

    async def test_checkpoint_create_payload_with_fault_injection_spec(self):
        seen: list[tuple[str, dict]] = []

        async def fake_post(url, payload, max_retries=60):
            seen.append((url, payload))
            return {"ok": True, "checkpoint_id": "ckpt-1"}

        client = SweEnvClient(base_url="http://fake")
        with patch("swe_env_client.post", new=fake_post):
            out = await client.checkpoint_create(
                "lease-1",
                fault_injection_spec={"phase": "before_commit", "delay_sec": 0.25},
            )

        self.assertTrue(out["ok"])
        self.assertEqual(
            seen[0][1]["fault_injection_spec"],
            {"phase": "before_commit", "delay_sec": 0.25},
        )

    async def test_exec_payload_with_fault_injection_spec(self):
        seen: list[tuple[str, dict]] = []

        async def fake_post(url, payload, max_retries=60):
            seen.append((url, payload))
            return {"ok": True, "returncode": -1}

        client = SweEnvClient(base_url="http://fake")
        with patch("swe_env_client.post", new=fake_post):
            out = await client.exec(
                "lease-1",
                "sleep 1",
                fault_injection_spec={"phase": "mid_action", "delay_sec": 0.0},
            )

        self.assertTrue(out["ok"])
        self.assertEqual(
            seen[0][1]["fault_injection_spec"],
            {"phase": "mid_action", "delay_sec": 0.0},
        )

    async def test_checkpoint_probe_does_not_retry_busy_payload(self):
        attempts = {"n": 0}

        async def fake_post(url, payload, max_retries=60):
            attempts["n"] += 1
            return {"ok": False, "error": "checkpoint system is busy", "error_code": "checkpoint_busy", "retryable": True}

        client = SweEnvClient(base_url="http://fake")
        client.app_error_retry_delay_sec = 0.0
        client.app_error_retry_jitter_sec = 0.0

        with patch("swe_env_client.post", new=fake_post):
            out = await client.checkpoint_probe("lease-1")

        self.assertTrue(out["ok"])
        self.assertTrue(out["busy"])
        self.assertEqual(attempts["n"], 1)

    async def test_checkpoint_probe_timeout_maps_to_busy_without_retry(self):
        attempts = {"n": 0}

        async def fake_post(url, payload, max_retries=60):
            attempts["n"] += 1
            raise TimeoutError("probe timed out")

        client = SweEnvClient(base_url="http://fake")
        client.checkpoint_probe_http_max_retries = 1

        with patch("swe_env_client.post", new=fake_post):
            out = await client.checkpoint_probe("lease-1")

        self.assertTrue(out["ok"])
        self.assertTrue(out["busy"])
        self.assertEqual(out["reason"], "checkpoint_probe_timeout")
        self.assertEqual(attempts["n"], 1)

    async def test_rerun_payload_without_optional_fields(self):
        seen: list[tuple[str, dict]] = []

        async def fake_post(url, payload, max_retries=60):
            seen.append((url, payload))
            return {"ok": True, "new_container_id": "cid-2"}

        client = SweEnvClient(base_url="http://fake")
        with patch("swe_env_client.post", new=fake_post):
            out = await client.rerun("lease-1", timeout=222)

        self.assertTrue(out["ok"])
        self.assertEqual(seen[0][0], "http://fake/rerun")
        self.assertEqual(seen[0][1], {"lease_id": "lease-1", "timeout": 222})


if __name__ == "__main__":
    unittest.main()
