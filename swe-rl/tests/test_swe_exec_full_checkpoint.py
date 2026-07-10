from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock


fake_flask = ModuleType("flask")


class _FakeFlaskApp:
    def __init__(self, name: str):
        self.name = name

    def get(self, _route: str):
        return lambda func: func

    def post(self, _route: str):
        return lambda func: func

    def run(self, *args, **kwargs):
        return None


fake_flask.Flask = _FakeFlaskApp
fake_flask.jsonify = lambda payload=None, **kwargs: payload if payload is not None else kwargs
fake_flask.request = type("_FakeRequest", (), {"get_json": staticmethod(lambda force=False: {})})()
sys.modules.setdefault("flask", fake_flask)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))

import swe_exec_server as exec_server  # noqa: E402


class _Options:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


class TestSweExecFullCheckpoint(unittest.TestCase):
    def setUp(self):
        super().setUp()
        with exec_server._lock:  # pylint: disable=protected-access
            exec_server._active_containers.clear()  # pylint: disable=protected-access
            exec_server._container_op_gates.clear()  # pylint: disable=protected-access

    @staticmethod
    def _new_full_record(manager: exec_server.CheckpointManager):
        manager.begin_create()
        record, op = manager.create_checkpoint(
            lease_id="lease-1",
            generation=2,
            container_id="cid-source",
            instance_id="inst-1",
            image="image:base",
            cwd="/testbed",
            step_idx=7,
            command_seq=8,
            policy="test",
            reason="test",
            parent_checkpoint_id=None,
        )
        record = manager.update_checkpoint(
            record["checkpoint_id"],
            checkpoint_backend="full",
            checkpoint_image=None,
            runtime_env={"PATH": "/usr/bin:/bin"},
        )
        return record, op

    def test_backend_defaults_full_but_history_without_marker_is_legacy(self):
        self.assertEqual(exec_server.ExecServerConfig().checkpoint_backend, "full")
        self.assertEqual(exec_server._record_checkpoint_backend({}), "legacy")
        self.assertEqual(
            exec_server._normalize_checkpoint_backend("docker-commit"), "legacy"
        )

    def test_full_worker_checkpoints_and_resumes_same_source_before_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = exec_server.CheckpointManager(tmpdir, enabled=True, max_inflight=1)
            record, op = self._new_full_record(manager)
            state_root = Path(tmpdir) / "full-state"
            record = manager.update_checkpoint(
                record["checkpoint_id"], full_checkpoint_state_root=str(state_root)
            )
            calls: list[tuple[str, str]] = []

            def fake_checkpoint(container_id, *, options):
                calls.append(("checkpoint", container_id))
                artifact = Path(options.state_root) / "checkpoints" / options.checkpoint_id
                artifact.mkdir(parents=True)
                (artifact / "manifest.json").write_text(
                    json.dumps({"status": "ready"}), encoding="utf-8"
                )
                (artifact / "payload.bin").write_bytes(b"checkpoint-payload")
                return {
                    "checkpoint_id": options.checkpoint_id,
                    "status": "ready",
                    "timings_sec": {"docker_checkpoint": 0.4, "total": 0.6},
                }

            def fake_resume(checkpoint_id, *, options):
                calls.append(("resume", str(options.container_id)))
                self.assertEqual(checkpoint_id, record["checkpoint_id"])
                return {
                    "ok": True,
                    "docker_container_id": "cid-source",
                    "docker_exec_supported": True,
                }

            api = exec_server._FullCheckpointApi(  # pylint: disable=protected-access
                checkpoint_options=_Options,
                resume_options=_Options,
                full_checkpoint=fake_checkpoint,
                full_resume=fake_resume,
            )
            with mock.patch.object(exec_server, "_CHECKPOINTS", manager), mock.patch.object(
                exec_server, "_load_full_checkpoint_api", return_value=api
            ), mock.patch.object(
                exec_server, "_full_checkpoint_docker_root", return_value=Path("/docker")
            ), mock.patch.object(
                exec_server, "_container_is_active", return_value=True
            ), mock.patch.object(
                exec_server, "_docker_container_is_running", return_value=True
            ), mock.patch.object(exec_server, "_remove_installed_docker_checkpoint"):
                result = exec_server._checkpoint_create_worker(  # pylint: disable=protected-access
                    op["op_id"],
                    record["checkpoint_id"],
                    "cid-source",
                    "",
                    record,
                )

            self.assertEqual(calls, [("checkpoint", "cid-source"), ("resume", "cid-source")])
            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["checkpoint_backend"], "full")
            self.assertTrue(result["source_resumed"])
            self.assertTrue(result["state_continuity"])
            self.assertGreater(result["size_bytes"], 0)
            self.assertEqual(manager.inflight_count(), 0)

    def test_full_worker_recovers_stopped_source_while_gate_is_held(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            manager = exec_server.CheckpointManager(tmpdir, enabled=True, max_inflight=1)
            record, op = self._new_full_record(manager)
            record = manager.update_checkpoint(
                record["checkpoint_id"],
                full_checkpoint_state_root=str(Path(tmpdir) / "full-state"),
            )
            running = {"value": True}
            gate_was_held = {"value": False}

            def failing_checkpoint(_container_id, *, options):
                del options
                running["value"] = False
                raise RuntimeError("snapshot persist failed after source stopped")

            def fake_docker(*args, **kwargs):
                del kwargs
                if args[:2] == ("start", "cid-source"):
                    gate = exec_server._get_container_op_gate("cid-source")  # pylint: disable=protected-access
                    gate_was_held["value"] = gate.exclusive_active
                    running["value"] = True
                return exec_server.subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

            api = exec_server._FullCheckpointApi(  # pylint: disable=protected-access
                checkpoint_options=_Options,
                resume_options=_Options,
                full_checkpoint=failing_checkpoint,
                full_resume=mock.Mock(),
            )
            with mock.patch.object(exec_server, "_CHECKPOINTS", manager), mock.patch.object(
                exec_server, "_load_full_checkpoint_api", return_value=api
            ), mock.patch.object(
                exec_server, "_full_checkpoint_docker_root", return_value=Path("/docker")
            ), mock.patch.object(
                exec_server, "_container_is_active", return_value=True
            ), mock.patch.object(
                exec_server,
                "_docker_container_is_running",
                side_effect=lambda *_args, **_kwargs: running["value"],
            ), mock.patch.object(exec_server, "_docker", side_effect=fake_docker), mock.patch.object(
                exec_server, "_remove_installed_docker_checkpoint"
            ):
                result = exec_server._checkpoint_create_worker(  # pylint: disable=protected-access
                    op["op_id"],
                    record["checkpoint_id"],
                    "cid-source",
                    "",
                    record,
                )

            self.assertEqual(result["status"], "failed")
            self.assertTrue(gate_was_held["value"])
            self.assertEqual(result["source_recovery_mode"], "plain_restart")
            self.assertFalse(result["state_continuity"])
            self.assertFalse(result["container_usable"])
            self.assertEqual(manager.inflight_count(), 0)

    def test_full_rerun_uses_long_id_then_destroys_old_container(self):
        record = {
            "checkpoint_id": "swe-ckpt-1234",
            "checkpoint_backend": "full",
            "status": "ready",
            "image": "image:base",
            "cwd": "/testbed",
            "runtime_env": {"PATH": "/usr/bin:/bin"},
            "full_checkpoint_state_root": "/state",
        }
        api = exec_server._FullCheckpointApi(  # pylint: disable=protected-access
            checkpoint_options=_Options,
            resume_options=_Options,
            full_checkpoint=mock.Mock(),
            full_resume=lambda _checkpoint_id, *, options: {
                "ok": True,
                "container_id": options.container_id,
                "docker_container_id": "cid-new-long",
                "docker_exec_supported": True,
            },
        )
        checkpoints = mock.Mock()
        with exec_server._lock:  # pylint: disable=protected-access
            exec_server._active_containers["cid-old"] = {  # pylint: disable=protected-access
                "image": "image:base",
                "name": "old",
                "cwd": "/testbed",
            }
        with mock.patch.object(exec_server, "_CHECKPOINTS", checkpoints), mock.patch.object(
            exec_server, "_load_full_checkpoint_api", return_value=api
        ), mock.patch.object(
            exec_server, "_full_checkpoint_docker_root", return_value=Path("/docker")
        ), mock.patch.object(
            exec_server, "_docker_container_is_running", return_value=True
        ), mock.patch.object(exec_server, "_validate_runtime_restore") as validate_mock, mock.patch.object(
            exec_server, "_remove_installed_docker_checkpoint"
        ), mock.patch.object(exec_server, "_docker_destroy_container") as destroy_mock:
            result = exec_server._full_checkpoint_rerun(  # pylint: disable=protected-access
                record=record,
                old_container_id="cid-old",
                cwd="/testbed",
                timeout=300,
            )

        self.assertEqual(result["new_container_id"], "cid-new-long")
        self.assertIsNone(result["checkpoint_image"])
        validate_mock.assert_called_once()
        destroy_mock.assert_called_once_with("cid-old", timeout=300)
        checkpoints.mark_used.assert_called_once_with("swe-ckpt-1234")
        with exec_server._lock:  # pylint: disable=protected-access
            self.assertNotIn("cid-old", exec_server._active_containers)  # pylint: disable=protected-access
            self.assertEqual(
                exec_server._active_containers["cid-new-long"]["image"],  # pylint: disable=protected-access
                "image:base",
            )

    def test_full_delete_removes_artifact_without_deleting_image(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_id = "swe-ckpt-delete"
            artifact = Path(tmpdir) / "checkpoints" / checkpoint_id
            artifact.mkdir(parents=True)
            (artifact / "manifest.json").write_text("{}", encoding="utf-8")
            record = {
                "checkpoint_id": checkpoint_id,
                "checkpoint_backend": "full",
                "full_checkpoint_state_root": tmpdir,
                "checkpoint_image": None,
            }
            with mock.patch.object(exec_server, "_docker") as docker_mock:
                exec_server._delete_checkpoint_artifacts(record)  # pylint: disable=protected-access
            self.assertFalse(artifact.exists())
            docker_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
