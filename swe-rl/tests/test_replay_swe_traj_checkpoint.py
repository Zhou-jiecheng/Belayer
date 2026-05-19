from __future__ import annotations

import json
import tempfile
import unittest
import asyncio
import types
from unittest import mock
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import replay_swe_traj_checkpoint as replay_tool  # noqa: E402


class TestReplaySweTrajCheckpoint(unittest.TestCase):
    def test_load_traj_steps_filters_range(self):
        payload = {
            "info": {"instance_id": "demo__repo-1"},
            "step_debug": [
                {"step_idx": 0, "action": "echo a", "returncode": 0, "output_head": "a", "output_tail": "a", "llm_elapsed": 1.2},
                {"step_idx": 1, "action": "echo b", "returncode": 0, "output_head": "b", "output_tail": "b", "llm_elapsed": 2.3},
                {"step_idx": 2, "action": "echo c", "returncode": 0, "output_head": "c", "output_tail": "c", "llm_elapsed": 3.4},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_path = Path(tmpdir) / "traj.json"
            traj_path.write_text(json.dumps(payload), encoding="utf-8")

            _, steps = replay_tool._load_traj_steps(str(traj_path), start_step=1, end_step=2)

        self.assertEqual([step.step_idx for step in steps], [1, 2])
        self.assertEqual(steps[0].action, "echo b")
        self.assertAlmostEqual(steps[0].llm_elapsed, 2.3)

    def test_output_matches_prefix_and_suffix(self):
        actual = "line1\nline2\nline3\n"
        self.assertTrue(replay_tool._output_matches("line1\nline2\n", actual, is_head=True))
        self.assertTrue(replay_tool._output_matches("line2\nline3\n", actual, is_head=False))
        self.assertFalse(replay_tool._output_matches("lineX", actual, is_head=True))

    def test_collect_traj_paths_from_directory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "a").mkdir()
            (root / "b").mkdir()
            (root / "a" / "traj.json").write_text("{}", encoding="utf-8")
            (root / "b" / "traj.json").write_text("{}", encoding="utf-8")

            paths = replay_tool._collect_traj_paths(str(root))

        self.assertEqual(len(paths), 2)
        self.assertTrue(all(path.name == "traj.json" for path in paths))

    def test_checkpoint_busy_error_detection(self):
        exc = replay_tool.ReplayOpError(
            "checkpoint_create",
            {"ok": False, "error_code": "checkpoint_busy", "retryable": True},
        )
        self.assertTrue(replay_tool._is_checkpoint_busy_error(exc))

    def test_exec_retries_when_container_is_paused(self):
        client = replay_tool.ReplayEnvClient(base_url="http://127.0.0.1:18090")
        client.exec_paused_max_retries = 2
        client.exec_paused_retry_delay_sec = 0.0

        responses = [
            {
                "ok": True,
                "returncode": 1,
                "output": "Error response from daemon: Container abc is paused, unpause the container before exec\n",
            },
            {
                "ok": True,
                "returncode": 0,
                "output": "done\n",
            },
        ]
        calls: list[tuple[str, str]] = []

        async def fake_post_with_retry(**kwargs):
            calls.append((kwargs["path"], kwargs["op_name"]))
            return responses.pop(0)

        client._post_with_retry = fake_post_with_retry  # type: ignore[method-assign]

        result = asyncio.run(client.exec("lease-1", "echo done", "/testbed", 30))

        self.assertEqual(result["returncode"], 0)
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0], ("exec", "exec"))

    def test_maybe_run_checkpoint_gc_skips_below_threshold(self):
        class FakeClient:
            def __init__(self):
                self.gc_calls: list[tuple[str | None, int, bool]] = []
                self.list_calls: list[str | None] = []

            async def checkpoint_list(self, lease_id: str | None = None):
                self.list_calls.append(lease_id)
                return {
                    "ok": True,
                    "checkpoints": [
                        {"checkpoint_id": "ckpt-1"},
                        {"checkpoint_id": "ckpt-2"},
                    ],
                }

            async def checkpoint_gc(self, lease_id: str | None, keep_latest: int, dry_run: bool):
                self.gc_calls.append((lease_id, keep_latest, dry_run))
                return {"ok": True}

        client = FakeClient()

        out = asyncio.run(
            replay_tool._maybe_run_checkpoint_gc(
                client,
                "lease-1",
                keep_latest=0,
                dry_run=False,
                min_checkpoint_count=100,
            )
        )

        self.assertEqual(client.gc_calls, [])
        self.assertEqual(client.list_calls, [None])
        self.assertEqual(len(out["checkpoint_list_before_gc"]["checkpoints"]), 2)
        self.assertIsNone(out["checkpoint_list_after_gc"])
        self.assertTrue(out["gc_result"]["skipped"])
        self.assertEqual(out["gc_result"]["skip_reason"], "below_threshold")
        self.assertEqual(out["gc_result"]["checkpoint_count"], 2)
        self.assertEqual(out["gc_result"]["gc_min_checkpoint_count"], 100)
        self.assertEqual(out["gc_result"]["gc_scope"], "global")
        self.assertEqual(out["gc_result"]["deleted_count"], 0)

    def test_maybe_run_checkpoint_gc_runs_at_threshold(self):
        class FakeClient:
            def __init__(self):
                self.gc_calls: list[tuple[str | None, int, bool]] = []
                self.list_calls: list[str | None] = []

            async def checkpoint_list(self, lease_id: str | None = None):
                self.list_calls.append(lease_id)
                if len(self.list_calls) == 1:
                    return {
                        "ok": True,
                        "checkpoints": [
                            {"checkpoint_id": "ckpt-1"},
                            {"checkpoint_id": "ckpt-2"},
                            {"checkpoint_id": "ckpt-3"},
                        ],
                    }
                return {
                    "ok": True,
                    "checkpoints": [{"checkpoint_id": "ckpt-3"}],
                }

            async def checkpoint_gc(self, lease_id: str | None, keep_latest: int, dry_run: bool):
                self.gc_calls.append((lease_id, keep_latest, dry_run))
                return {
                    "ok": True,
                    "deleted_count": 2,
                    "deleted_checkpoint_ids": ["ckpt-2", "ckpt-1"],
                    "reclaimed_bytes": 42,
                    "queued": True,
                    "dry_run": dry_run,
                }

        client = FakeClient()

        out = asyncio.run(
            replay_tool._maybe_run_checkpoint_gc(
                client,
                "lease-1",
                keep_latest=0,
                dry_run=False,
                min_checkpoint_count=3,
            )
        )

        self.assertEqual(client.list_calls, [None, None])
        self.assertEqual(client.gc_calls, [(None, 0, False)])
        self.assertEqual(len(out["checkpoint_list_before_gc"]["checkpoints"]), 3)
        self.assertEqual(len(out["checkpoint_list_after_gc"]["checkpoints"]), 1)
        self.assertFalse(out["gc_result"]["skipped"])
        self.assertEqual(out["gc_result"]["checkpoint_count"], 3)
        self.assertEqual(out["gc_result"]["gc_min_checkpoint_count"], 3)
        self.assertEqual(out["gc_result"]["gc_scope"], "global")
        self.assertEqual(out["gc_result"]["deleted_count"], 2)

    def test_replay_many_runs_global_gc_once_after_batch(self):
        async def fake_replay_one_path(traj_path, args, *, defer_gc_until_batch_end=False):
            return {
                "traj_path": str(traj_path.resolve()),
                "ok": True,
                "gc_result": {"skip_reason": "deferred_to_batch_end"} if defer_gc_until_batch_end else None,
            }

        gc_calls = []

        async def fake_maybe_run_checkpoint_gc(env_client, lease_id, *, keep_latest, dry_run, min_checkpoint_count):
            gc_calls.append((lease_id, keep_latest, dry_run, min_checkpoint_count))
            return {
                "checkpoint_list_before_gc": {"ok": True, "checkpoints": [{"checkpoint_id": "ckpt-1"}]},
                "gc_result": {"ok": True, "deleted_count": 1, "reclaimed_bytes": 7},
                "checkpoint_list_after_gc": {"ok": True, "checkpoints": []},
            }

        class FakeClient:
            def __init__(self, base_url):
                self.base_url = base_url

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ["a", "b"]:
                traj_dir = root / name
                traj_dir.mkdir()
                (traj_dir / "traj.json").write_text("{}", encoding="utf-8")
            args = types.SimpleNamespace(
                trajectory=str(root),
                limit=None,
                print_commands=False,
                max_concurrency=2,
                output_dir=None,
                gc_keep_latest=0,
                gc_dry_run=False,
                gc_min_checkpoint_count=100,
                base_url="http://127.0.0.1:5000",
            )
            with mock.patch.object(replay_tool, "_replay_one_path", new=fake_replay_one_path), mock.patch.object(
                replay_tool, "_maybe_run_checkpoint_gc", new=fake_maybe_run_checkpoint_gc
            ), mock.patch.object(replay_tool, "ReplayEnvClient", new=FakeClient):
                out = asyncio.run(replay_tool._replay_many(args))

        self.assertEqual(len(gc_calls), 1)
        self.assertEqual(gc_calls[0], (None, 0, False, 100))
        self.assertEqual(out["batch_gc_result"]["deleted_count"], 1)
        self.assertTrue(all(item["gc_result"]["skip_reason"] == "deferred_to_batch_end" for item in out["reports"]))


if __name__ == "__main__":
    unittest.main()
