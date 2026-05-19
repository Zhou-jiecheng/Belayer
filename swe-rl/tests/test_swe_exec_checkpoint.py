from __future__ import annotations

import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import ModuleType
from unittest import mock

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

import swe_exec_server as exec_server  # noqa: E402


class TestSweExecCheckpoint(unittest.TestCase):
    def setUp(self):
        super().setUp()
        exec_server._container_op_gates.clear()  # pylint: disable=protected-access
        exec_server._maintenance_active = False  # pylint: disable=protected-access
        exec_server._foreground_docker_ops = 0  # pylint: disable=protected-access
        exec_server._gc_tasks_inflight.clear()  # pylint: disable=protected-access

    def test_try_begin_create_enforces_max_inflight_atomically(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = exec_server.CheckpointManager(tmpdir, enabled=True, max_inflight=1)

            self.assertTrue(manager.try_begin_create())
            self.assertEqual(manager.inflight_count(), 1)
            self.assertFalse(manager.try_begin_create())

            manager.end_create()

            self.assertEqual(manager.inflight_count(), 0)
            self.assertTrue(manager.try_begin_create())

    def test_checkpoint_probe_state_reports_inflight_busy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = exec_server.CheckpointManager(tmpdir, enabled=True, max_inflight=1)
            completed = exec_server.subprocess.CompletedProcess(
                args=["docker", "inspect"],
                returncode=0,
                stdout="123\n",
                stderr="",
            )
            config = exec_server.ExecServerConfig(
                checkpoint_enabled=True,
                checkpoint_dir=tmpdir,
                checkpoint_max_inflight=1,
                checkpoint_probe_inspect_timeout_sec=1.0,
            )
            with mock.patch.object(exec_server, "_CHECKPOINTS", manager), mock.patch.object(
                exec_server, "_SERVER_CONFIG", config
            ), mock.patch.object(
                exec_server, "_docker", return_value=completed
            ):
                manager.begin_create()
                state = exec_server._checkpoint_probe_state("cid-1")
                self.assertTrue(state["busy"])
                self.assertEqual(state["reason"], "checkpoint_inflight_limit")
                self.assertEqual(state["metrics"]["inflight_checkpoints"], 1)

    def test_checkpoint_probe_state_uses_fast_inspect(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = exec_server.CheckpointManager(tmpdir, enabled=True, max_inflight=1)
            completed = exec_server.subprocess.CompletedProcess(
                args=["docker", "inspect"],
                returncode=0,
                stdout="123\n",
                stderr="",
            )
            config = exec_server.ExecServerConfig(
                checkpoint_enabled=True,
                checkpoint_dir=tmpdir,
                checkpoint_max_inflight=1,
                checkpoint_probe_inspect_timeout_sec=1.0,
            )
            with mock.patch.object(exec_server, "_CHECKPOINTS", manager), mock.patch.object(
                exec_server, "_SERVER_CONFIG", config
            ), mock.patch.object(exec_server, "_docker", return_value=completed):
                state = exec_server._checkpoint_probe_state("cid-1")
                self.assertFalse(state["busy"])
                self.assertEqual(state["reason"], "idle")
                self.assertEqual(state["metrics"]["inspect_size_rw_bytes"], 123)
                self.assertEqual(state["metrics"]["inspect_returncode"], 0)

    def test_checkpoint_probe_state_marks_slow_inspect_busy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = exec_server.CheckpointManager(tmpdir, enabled=True, max_inflight=1)
            config = exec_server.ExecServerConfig(
                checkpoint_enabled=True,
                checkpoint_dir=tmpdir,
                checkpoint_max_inflight=1,
                checkpoint_probe_inspect_timeout_sec=1.0,
            )
            with mock.patch.object(exec_server, "_CHECKPOINTS", manager), mock.patch.object(
                exec_server, "_SERVER_CONFIG", config
            ), mock.patch.object(
                exec_server,
                "_docker",
                side_effect=exec_server.subprocess.TimeoutExpired(cmd=["docker", "inspect"], timeout=1.0),
            ):
                state = exec_server._checkpoint_probe_state("cid-1")
                self.assertTrue(state["busy"])
                self.assertEqual(state["reason"], "checkpoint_probe_inspect_slow")
                self.assertIn("timeout", state["metrics"]["inspect_error"])

    def test_maybe_inject_exec_fault_returns_none_when_unarmed(self):
        with mock.patch.object(exec_server.random, "random", return_value=0.0):
            event = exec_server._maybe_inject_exec_fault(  # pylint: disable=protected-access
                "cid-1",
                armed=False,
                probability=0.003,
            )
        self.assertIsNone(event)

    def test_maybe_inject_exec_fault_destroys_container_and_clears_tracking(self):
        with mock.patch.object(exec_server._CONTAINER_POOL, "release", return_value={  # pylint: disable=protected-access
            "pooled": False,
            "destroyed": True,
            "reason": "pool_disabled",
        }) as mock_release, mock.patch.object(exec_server, "_drop_container_op_gate") as mock_drop_gate:
            with exec_server._lock:  # pylint: disable=protected-access
                exec_server._active_containers["cid-1"] = {
                    "image": "img",
                    "name": "name",
                    "cwd": "/testbed",
                }
            event = exec_server._maybe_inject_exec_fault(  # pylint: disable=protected-access
                "cid-1",
                armed=True,
                probability=1.0,
            )

        self.assertIsNotNone(event)
        self.assertTrue(event["fault_injected"])
        self.assertEqual(event["fault_type"], "exec_server_random_kill")
        self.assertTrue(event["destroyed"])
        mock_release.assert_called_once()
        mock_drop_gate.assert_called_once_with("cid-1")
        with exec_server._lock:  # pylint: disable=protected-access
            self.assertNotIn("cid-1", exec_server._active_containers)

    def test_checkpoint_gc_plan_keeps_ancestors_of_retained_checkpoint(self):
        records = [
            {
                "checkpoint_id": "ckpt-1",
                "lease_id": "lease-1",
                "status": "ready",
                "step_idx": 1,
                "created_at": 1.0,
                "checkpoint_image": "sweckpt:ckpt-1",
                "parent_checkpoint_id": None,
            },
            {
                "checkpoint_id": "ckpt-2",
                "lease_id": "lease-1",
                "status": "ready",
                "step_idx": 2,
                "created_at": 2.0,
                "checkpoint_image": "sweckpt:ckpt-2",
                "parent_checkpoint_id": "ckpt-1",
            },
        ]

        deletions, kept_ids = exec_server._checkpoint_gc_plan(records, keep_latest=1, active_checkpoint_images=set())

        self.assertEqual([item["checkpoint_id"] for item in deletions], [])
        self.assertEqual(kept_ids, ["ckpt-1", "ckpt-2"])

    def test_checkpoint_gc_plan_deletes_children_before_parents(self):
        records = [
            {
                "checkpoint_id": "ckpt-1",
                "lease_id": "lease-1",
                "status": "ready",
                "step_idx": 1,
                "created_at": 1.0,
                "checkpoint_image": "sweckpt:ckpt-1",
                "parent_checkpoint_id": None,
            },
            {
                "checkpoint_id": "ckpt-2",
                "lease_id": "lease-1",
                "status": "ready",
                "step_idx": 2,
                "created_at": 2.0,
                "checkpoint_image": "sweckpt:ckpt-2",
                "parent_checkpoint_id": "ckpt-1",
            },
            {
                "checkpoint_id": "ckpt-3",
                "lease_id": "lease-1",
                "status": "ready",
                "step_idx": 3,
                "created_at": 3.0,
                "checkpoint_image": "sweckpt:ckpt-3",
                "parent_checkpoint_id": "ckpt-2",
            },
        ]

        deletions, kept_ids = exec_server._checkpoint_gc_plan(records, keep_latest=0, active_checkpoint_images=set())

        self.assertEqual([item["checkpoint_id"] for item in deletions], ["ckpt-3", "ckpt-2", "ckpt-1"])
        self.assertEqual(kept_ids, [])

    def test_checkpoint_gc_plan_keeps_active_checkpoint_image_and_ancestors(self):
        records = [
            {
                "checkpoint_id": "ckpt-1",
                "lease_id": "lease-1",
                "status": "ready",
                "step_idx": 1,
                "created_at": 1.0,
                "checkpoint_image": "sweckpt:ckpt-1",
                "parent_checkpoint_id": None,
            },
            {
                "checkpoint_id": "ckpt-2",
                "lease_id": "lease-1",
                "status": "ready",
                "step_idx": 2,
                "created_at": 2.0,
                "checkpoint_image": "sweckpt:ckpt-2",
                "parent_checkpoint_id": "ckpt-1",
            },
        ]

        deletions, kept_ids = exec_server._checkpoint_gc_plan(
            records,
            keep_latest=0,
            active_checkpoint_images={"sweckpt:ckpt-2"},
        )

        self.assertEqual([item["checkpoint_id"] for item in deletions], [])
        self.assertEqual(kept_ids, ["ckpt-1", "ckpt-2"])

    def test_build_runtime_state_payload_keeps_minimal_whitelisted_env(self):
        record = {
            "checkpoint_id": "ckpt-1",
            "parent_checkpoint_id": "ckpt-0",
            "lease_id": "lease-1",
            "instance_id": "inst-1",
            "step_idx": 5,
            "command_seq": 8,
            "cwd": "/workspace/repo",
            "reason": "after_exec",
        }

        runtime_env = exec_server._normalize_runtime_env(  # pylint: disable=protected-access
            {
                "PATH": "/opt/venv/bin:/usr/bin:/bin",
                "PYTHONPATH": "/workspace/repo",
                "VIRTUAL_ENV": "/opt/venv",
                "IGNORED": "value",
            }
        )
        payload = exec_server._build_runtime_state_payload(  # pylint: disable=protected-access
            record,
            runtime_env,
        )

        self.assertEqual(payload["checkpoint_id"], "ckpt-1")
        self.assertEqual(payload["workspace"]["cwd"], "/workspace/repo")
        self.assertEqual(payload["env"]["PATH"], "/opt/venv/bin:/usr/bin:/bin")
        self.assertEqual(payload["env"]["PYTHONPATH"], "/workspace/repo")
        self.assertEqual(payload["env"]["VIRTUAL_ENV"], "/opt/venv")
        self.assertNotIn("IGNORED", payload["env"])
        self.assertEqual(payload["python_runtime"]["python_executable"], "/opt/venv/bin/python")
        self.assertEqual(payload["progress"]["last_successful_action_id"], "cmd-8")

    def test_probe_runtime_env_merges_container_and_explicit_env(self):
        docker_result = exec_server.subprocess.CompletedProcess(
            args=["docker", "exec"],
            returncode=0,
            stdout='{"PATH":"/usr/bin:/bin","PYTHONPATH":"","VIRTUAL_ENV":"","CONDA_PREFIX":""}\n',
            stderr="",
        )

        with mock.patch.object(exec_server, "_docker", return_value=docker_result) as docker_mock:
            runtime_env = exec_server._probe_runtime_env(  # pylint: disable=protected-access
                "cid-1",
                "/workspace",
                {"VIRTUAL_ENV": "/opt/venv"},
            )

        self.assertEqual(runtime_env["PATH"], "/usr/bin:/bin")
        self.assertEqual(runtime_env["VIRTUAL_ENV"], "/opt/venv")
        docker_mock.assert_called_once()

    def test_checkpoint_worker_skips_commit_for_inactive_container(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = exec_server.CheckpointManager(tmpdir, enabled=True, max_inflight=1)
            self.assertTrue(manager.try_begin_create())
            record, op = manager.create_checkpoint(
                lease_id="lease-1",
                generation=0,
                container_id="cid-dead",
                instance_id="inst-1",
                image="img:base",
                cwd="/testbed",
                step_idx=1,
                command_seq=1,
                policy="manual",
                reason="test",
                parent_checkpoint_id=None,
            )
            with mock.patch.object(exec_server, "_CHECKPOINTS", manager), mock.patch.object(
                exec_server, "_container_is_active", return_value=False
            ), mock.patch.object(exec_server, "_docker") as docker_mock:
                exec_server._checkpoint_create_worker(  # pylint: disable=protected-access
                    op["op_id"],
                    record["checkpoint_id"],
                    "cid-dead",
                    record["checkpoint_image"],
                    {},
                )

            updated = manager.get_checkpoint(record["checkpoint_id"])
            self.assertEqual(updated["status"], "failed")
            self.assertIn("no longer active", updated["error"])
            self.assertEqual(manager.inflight_count(), 0)
            docker_mock.assert_not_called()

    def test_checkpoint_worker_enforces_min_ready_latency(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = exec_server.CheckpointManager(tmpdir, enabled=True, max_inflight=1)
            self.assertTrue(manager.try_begin_create())
            record, op = manager.create_checkpoint(
                lease_id="lease-1",
                generation=0,
                container_id="cid-live",
                instance_id="inst-1",
                image="img:base",
                cwd="/testbed",
                step_idx=1,
                command_seq=1,
                policy="manual",
                reason="test",
                parent_checkpoint_id=None,
            )
            config = exec_server.ExecServerConfig(
                checkpoint_enabled=True,
                checkpoint_dir=tmpdir,
                checkpoint_create_timeout_sec=300,
                checkpoint_min_ready_latency_sec=2.0,
                checkpoint_max_inflight=1,
            )
            docker_result = exec_server.subprocess.CompletedProcess(
                args=["docker", "commit"],
                returncode=0,
                stdout="",
                stderr="",
            )
            probe_result = exec_server.subprocess.CompletedProcess(
                args=["docker", "exec"],
                returncode=0,
                stdout='{"PATH":"/usr/bin:/bin","PYTHONPATH":"","VIRTUAL_ENV":"","CONDA_PREFIX":""}\n',
                stderr="",
            )
            with mock.patch.object(exec_server, "_CHECKPOINTS", manager), mock.patch.object(
                exec_server, "_SERVER_CONFIG", config
            ), mock.patch.object(exec_server, "_container_is_active", return_value=True), mock.patch.object(
                exec_server, "_docker_container_is_running", return_value=True
            ), mock.patch.object(exec_server, "_docker", side_effect=[probe_result, docker_result, docker_result]), mock.patch.object(
                exec_server, "_docker_image_size_bytes", return_value=123
            ), mock.patch.object(
                exec_server.time, "sleep"
            ) as sleep_mock, mock.patch.object(
                exec_server.time, "time", side_effect=[100.0, 100.2, 102.0]
            ):
                exec_server._checkpoint_create_worker(  # pylint: disable=protected-access
                    op["op_id"],
                    record["checkpoint_id"],
                    "cid-live",
                    record["checkpoint_image"],
                    {},
                )

            updated = manager.get_checkpoint(record["checkpoint_id"])
            self.assertEqual(updated["status"], "ready")
            self.assertAlmostEqual(updated["raw_create_latency_sec"], 0.2)
            self.assertAlmostEqual(updated["ready_delay_sec"], 1.8)
            self.assertAlmostEqual(updated["ready_latency_sec"], 2.0)
            self.assertEqual(updated["size_bytes"], 123)
            self.assertEqual(manager.inflight_count(), 0)
            sleep_mock.assert_called_once()
            self.assertAlmostEqual(sleep_mock.call_args.args[0], 1.8)

    def test_checkpoint_gc_returns_after_queueing_background_work(self):
        started: list[tuple] = []

        class _FakeThread:
            def __init__(self, target=None, args=(), name=None, daemon=None):
                self.target = target
                self.args = args
                self.name = name
                self.daemon = daemon

            def start(self):
                started.append((self.target, self.args, self.name, self.daemon))

        records = [
            {
                "checkpoint_id": "ckpt-1",
                "lease_id": "lease-1",
                "status": "ready",
                "step_idx": 1,
                "created_at": 1.0,
                "checkpoint_image": "sweckpt:ckpt-1",
                "parent_checkpoint_id": None,
                "size_bytes": 123,
            }
        ]

        with mock.patch.object(
            exec_server.request,
            "get_json",
            return_value={"lease_id": "lease-1", "keep_latest": 0, "dry_run": False},
        ), mock.patch.object(exec_server._CHECKPOINTS, "list_checkpoints", return_value=records), mock.patch.object(
            exec_server,
            "_checkpoint_gc_plan",
            return_value=(records, []),
        ), mock.patch.object(exec_server.threading, "Thread", _FakeThread), mock.patch.dict(
            exec_server._active_containers, {}, clear=True
        ), mock.patch.object(exec_server, "_docker") as docker_mock:
            out = exec_server.container_checkpoint_gc()

        self.assertTrue(out["ok"])
        self.assertTrue(out["queued"])
        self.assertEqual(out["deleted_count"], 1)
        self.assertEqual(len(started), 1)
        docker_mock.assert_not_called()

    def test_checkpoint_gc_drain_waits_until_background_tasks_finish(self):
        exec_server._gc_tasks_inflight.update({"task-1"})  # pylint: disable=protected-access

        def fake_sleep(_secs: float) -> None:
            exec_server._gc_tasks_inflight.clear()  # pylint: disable=protected-access

        with mock.patch.object(
            exec_server.request,
            "get_json",
            return_value={"timeout_sec": 5.0, "poll_interval_sec": 0.1},
        ), mock.patch.object(exec_server.time, "sleep", side_effect=fake_sleep):
            out = exec_server.container_checkpoint_gc_drain()

        self.assertTrue(out["ok"])
        self.assertTrue(out["drained"])
        self.assertFalse(out["timed_out"])
        self.assertEqual(out["initial_inflight_count"], 1)
        self.assertEqual(out["remaining_inflight_count"], 0)

    def test_maintenance_section_waits_for_foreground_ops(self):
        events: list[str] = []
        foreground_entered = threading.Event()
        allow_foreground_exit = threading.Event()
        maintenance_done = threading.Event()

        def foreground() -> None:
            with exec_server._foreground_docker_section():
                events.append("foreground_enter")
                foreground_entered.set()
                allow_foreground_exit.wait(timeout=2.0)
                events.append("foreground_exit")

        def maintenance() -> None:
            with exec_server._maintenance_docker_section():
                events.append("maintenance_enter")
            maintenance_done.set()

        fg_thread = threading.Thread(target=foreground)
        mt_thread = threading.Thread(target=maintenance)
        fg_thread.start()
        self.assertTrue(foreground_entered.wait(timeout=1.0))
        mt_thread.start()
        self.assertFalse(maintenance_done.wait(timeout=0.2))
        allow_foreground_exit.set()
        fg_thread.join(timeout=1.0)
        mt_thread.join(timeout=1.0)
        self.assertEqual(events, ["foreground_enter", "foreground_exit", "maintenance_enter"])

    def test_foreground_section_does_not_wait_for_commit(self):
        events: list[str] = []
        commit_entered = threading.Event()
        allow_commit_exit = threading.Event()
        foreground_done = threading.Event()

        def commit() -> None:
            with exec_server._commit_docker_section():
                events.append("commit_enter")
                commit_entered.set()
                allow_commit_exit.wait(timeout=2.0)
                events.append("commit_exit")

        def foreground() -> None:
            with exec_server._foreground_docker_section():
                events.append("foreground_enter")
            foreground_done.set()

        commit_thread = threading.Thread(target=commit)
        fg_thread = threading.Thread(target=foreground)
        commit_thread.start()
        self.assertTrue(commit_entered.wait(timeout=1.0))
        fg_thread.start()
        self.assertTrue(foreground_done.wait(timeout=0.2))
        allow_commit_exit.set()
        commit_thread.join(timeout=1.0)
        fg_thread.join(timeout=1.0)
        self.assertEqual(events[0], "commit_enter")
        self.assertIn("foreground_enter", events)
        self.assertEqual(events[-1], "commit_exit")

    def test_commit_section_does_not_wait_for_foreground(self):
        events: list[str] = []
        foreground_entered = threading.Event()
        allow_foreground_exit = threading.Event()
        commit_done = threading.Event()

        def foreground() -> None:
            with exec_server._foreground_docker_section():
                events.append("foreground_enter")
                foreground_entered.set()
                allow_foreground_exit.wait(timeout=2.0)
                events.append("foreground_exit")

        def commit() -> None:
            with exec_server._commit_docker_section():
                events.append("commit_enter")
            commit_done.set()

        fg_thread = threading.Thread(target=foreground)
        commit_thread = threading.Thread(target=commit)
        fg_thread.start()
        self.assertTrue(foreground_entered.wait(timeout=1.0))
        commit_thread.start()
        self.assertTrue(commit_done.wait(timeout=0.2))
        allow_foreground_exit.set()
        fg_thread.join(timeout=1.0)
        commit_thread.join(timeout=1.0)
        self.assertEqual(events[0], "foreground_enter")
        self.assertIn("commit_enter", events)
        self.assertEqual(events[-1], "foreground_exit")

    def test_maintenance_section_waits_for_commit(self):
        events: list[str] = []
        commit_entered = threading.Event()
        allow_commit_exit = threading.Event()
        maintenance_done = threading.Event()

        def commit() -> None:
            with exec_server._commit_docker_section():
                events.append("commit_enter")
                commit_entered.set()
                allow_commit_exit.wait(timeout=2.0)
                events.append("commit_exit")

        def maintenance() -> None:
            with exec_server._maintenance_docker_section():
                events.append("maintenance_enter")
            maintenance_done.set()

        commit_thread = threading.Thread(target=commit)
        maintenance_thread = threading.Thread(target=maintenance)
        commit_thread.start()
        self.assertTrue(commit_entered.wait(timeout=1.0))
        maintenance_thread.start()
        self.assertFalse(maintenance_done.wait(timeout=0.2))
        allow_commit_exit.set()
        commit_thread.join(timeout=1.0)
        maintenance_thread.join(timeout=1.0)
        self.assertEqual(events, ["commit_enter", "commit_exit", "maintenance_enter"])

    def test_container_exclusive_section_serializes_same_container(self):
        events: list[str] = []
        first_entered = threading.Event()
        allow_first_exit = threading.Event()

        def first() -> None:
            with exec_server._container_exclusive_section("cid-1"):  # pylint: disable=protected-access
                events.append("first_enter")
                first_entered.set()
                allow_first_exit.wait(timeout=2.0)
                events.append("first_exit")

        def second() -> None:
            with exec_server._container_exclusive_section("cid-1"):  # pylint: disable=protected-access
                events.append("second_enter")
                events.append("second_exit")

        first_thread = threading.Thread(target=first)
        second_thread = threading.Thread(target=second)
        first_thread.start()
        self.assertTrue(first_entered.wait(timeout=1.0))
        second_thread.start()
        self.assertEqual(events, ["first_enter"])
        allow_first_exit.set()
        first_thread.join(timeout=1.0)
        second_thread.join(timeout=1.0)
        self.assertEqual(events, ["first_enter", "first_exit", "second_enter", "second_exit"])


if __name__ == "__main__":
    unittest.main()
