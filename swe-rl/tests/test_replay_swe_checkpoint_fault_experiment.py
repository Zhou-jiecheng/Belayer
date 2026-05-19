from __future__ import annotations

import unittest
import asyncio
import tempfile
import types
from unittest import mock
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import replay_swe_checkpoint_fault_experiment as fault_tool  # noqa: E402


class TestReplaySweCheckpointFaultExperiment(unittest.TestCase):
    def test_parser_defaults_to_thresholded_gc(self):
        parser = fault_tool._build_parser()
        args = parser.parse_args(["/tmp/trajs"])
        self.assertEqual(args.gc_keep_latest, 0)
        self.assertEqual(args.gc_min_checkpoint_count, 100)
        self.assertEqual(args.adaptive_checkpoint_budget_sec, fault_tool.DEFAULT_ADAPTIVE_BUDGET_SEC)

    def test_adaptive_tail_model_conditional_tail(self):
        model = fault_tool.AdaptiveTailModel.from_waits([5.0, 8.0, 12.0], budget_sec=7.0)
        self.assertAlmostEqual(model.conditional_tail_probability(0.0), 2.0 / 3.0)
        self.assertAlmostEqual(model.conditional_tail_probability(5.0), 0.0)

    def test_adaptive_tail_model_expected_overhead(self):
        model = fault_tool.AdaptiveTailModel.from_waits([5.0, 8.0, 12.0], budget_sec=7.0)
        self.assertAlmostEqual(model.expected_exposed_overhead(0.0), 2.0 / 3.0)
        self.assertAlmostEqual(model.expected_exposed_overhead(5.0), 2.0)
        self.assertAlmostEqual(model.expected_exposed_overhead(12.0), 7.0)

    def test_redo_cost_includes_llm_and_exec_cost(self):
        self.assertAlmostEqual(fault_tool._redo_replay_cost_sec(12.5, 4.0, 7.0, 2.5), 13.0)
        self.assertAlmostEqual(fault_tool._redo_replay_cost_sec(3.0, 4.0, 1.0, 2.0), 0.0)

    def test_adaptive_expected_benefit_includes_conditional_tail_probability(self):
        self.assertAlmostEqual(
            fault_tool._adaptive_expected_benefit_sec(0.05, 0.8, 10.0),
            0.4,
        )
        self.assertAlmostEqual(
            fault_tool._adaptive_expected_benefit_sec(0.05, 0.0, 10.0),
            0.0,
        )

    def test_latest_ready_from_records_preserves_step_zero(self):
        checkpoint_id, step_idx = fault_tool._latest_ready_from_records(
            [
                {
                    "checkpoint_id": "ckpt-0",
                    "step_idx": 0,
                    "status_result": {"status": "ready", "step_idx": 0, "ready_at": 10.0},
                }
            ]
        )
        self.assertEqual(checkpoint_id, "ckpt-0")
        self.assertEqual(step_idx, 0)

    def test_should_probe_in_llm_bubble_allows_only_one_probe(self):
        self.assertTrue(
            fault_tool._should_probe_in_llm_bubble(
                current_step_idx=3,
                probe_attempted_in_bubble=False,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[],
                delta_env_cost_sec=2.1,
                steps_since_latest_ready_checkpoint=4,
                expected_benefit_sec=1.2,
                expected_overhead_sec=0.8,
            )
        )

    def test_select_injections_longest_picks_longest_trajectories(self):
        traj_paths = [Path("/tmp/a/traj.json"), Path("/tmp/b/traj.json"), Path("/tmp/c/traj.json")]
        step_counts = {
            str(traj_paths[0]): 5,
            str(traj_paths[1]): 12,
            str(traj_paths[2]): 9,
        }

        def fake_load_traj_steps(path: str):
            count = step_counts[path]
            steps = [types.SimpleNamespace(llm_elapsed=1.0) for _ in range(count)]
            return "inst", steps

        with mock.patch.object(fault_tool, "_load_traj_steps", side_effect=fake_load_traj_steps):
            out = fault_tool._select_injections(
                traj_paths,
                seed=123,
                injection_count=2,
                selection_mode="longest",
            )

        self.assertEqual(
            [item.traj_path for item in out],
            [str(traj_paths[1].resolve()), str(traj_paths[2].resolve())],
        )

    def test_run_policy_runs_global_gc_once_after_batch(self):
        async def fake_run_one_trajectory(traj_path, policy, injection_target, args, *, defer_gc_until_batch_end=False):
            return {
                "traj_path": str(traj_path.resolve()),
                "policy": policy,
                "ok": True,
                "gc_result": {"skip_reason": "deferred_to_batch_end"} if defer_gc_until_batch_end else None,
                "metrics": {
                    "wall_time_sec": 1.0,
                    "checkpoint_attempts": 0,
                    "checkpoint_created": 0,
                    "checkpoint_busy_skips": 0,
                    "probe_count": 0,
                    "probe_busy_skips": 0,
                    "rerun_from_checkpoint": 0,
                    "rerun_from_base": 0,
                },
            }

        gc_calls = []

        async def fake_maybe_run_checkpoint_gc(env_client, lease_id, *, keep_latest, dry_run, min_checkpoint_count):
            gc_calls.append((lease_id, keep_latest, dry_run, min_checkpoint_count))
            return {
                "checkpoint_list_before_gc": {"ok": True, "checkpoints": [{"checkpoint_id": "ckpt-1"}]},
                "gc_result": {"ok": True, "deleted_count": 2, "reclaimed_bytes": 11},
                "checkpoint_list_after_gc": {"ok": True, "checkpoints": []},
            }

        class FakeClient:
            def __init__(self, base_url):
                self.base_url = base_url

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            traj_paths = []
            for name in ["a", "b"]:
                traj_dir = root / name
                traj_dir.mkdir()
                traj_path = traj_dir / "traj.json"
                traj_path.write_text("{}", encoding="utf-8")
                traj_paths.append(traj_path)
            out_dir = root / "out"
            args = types.SimpleNamespace(
                max_concurrency=2,
                gc_keep_latest=0,
                gc_dry_run=False,
                gc_min_checkpoint_count=100,
                base_url="http://127.0.0.1:5000",
            )
            with mock.patch.object(fault_tool, "_run_one_trajectory", new=fake_run_one_trajectory), mock.patch.object(
                fault_tool, "_maybe_run_checkpoint_gc", new=fake_maybe_run_checkpoint_gc
            ), mock.patch.object(fault_tool, "ReplayEnvClient", new=FakeClient):
                out = asyncio.run(fault_tool._run_policy(traj_paths, {}, "always", args, out_dir))

        self.assertEqual(len(gc_calls), 1)
        self.assertEqual(gc_calls[0], (None, 0, False, 100))
        self.assertEqual(out["batch_gc_result"]["deleted_count"], 2)
        self.assertEqual(out["summary"]["gc_deleted_count"], 2)
        self.assertTrue(all(item["gc_result"]["skip_reason"] == "deferred_to_batch_end" for item in out["reports"]))
        self.assertFalse(
            fault_tool._should_probe_in_llm_bubble(
                current_step_idx=3,
                probe_attempted_in_bubble=True,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[],
                delta_env_cost_sec=2.1,
                steps_since_latest_ready_checkpoint=3,
                expected_benefit_sec=1.2,
                expected_overhead_sec=0.8,
            )
        )
        self.assertFalse(
            fault_tool._should_probe_in_llm_bubble(
                current_step_idx=3,
                probe_attempted_in_bubble=False,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[{"checkpoint_id": "ckpt-1"}],
                delta_env_cost_sec=2.1,
                steps_since_latest_ready_checkpoint=3,
                expected_benefit_sec=1.2,
                expected_overhead_sec=0.8,
            )
        )
        self.assertFalse(
            fault_tool._should_probe_in_llm_bubble(
                current_step_idx=3,
                probe_attempted_in_bubble=False,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[],
                delta_env_cost_sec=0.05,
                steps_since_latest_ready_checkpoint=3,
                expected_benefit_sec=1.2,
                expected_overhead_sec=0.8,
            )
        )
        self.assertFalse(
            fault_tool._should_probe_in_llm_bubble(
                current_step_idx=3,
                probe_attempted_in_bubble=False,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[],
                delta_env_cost_sec=2.1,
                steps_since_latest_ready_checkpoint=3,
                expected_benefit_sec=1.2,
                expected_overhead_sec=1.3,
            )
        )
        self.assertFalse(
            fault_tool._should_probe_in_llm_bubble(
                current_step_idx=2,
                probe_attempted_in_bubble=False,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[],
                delta_env_cost_sec=2.1,
                steps_since_latest_ready_checkpoint=3,
                expected_benefit_sec=1.2,
                expected_overhead_sec=0.8,
            )
        )

    def test_main_async_waits_for_gc_drain_between_policies(self):
        async def fake_run_policy(traj_paths, injections, policy, args, out_dir):
            return {"summary": {"policy": policy, "trajectory_count": len(traj_paths)}}

        drain_calls = []

        async def fake_wait_for_global_gc_drain(*, base_url, timeout_sec, poll_interval_sec):
            drain_calls.append((base_url, timeout_sec, poll_interval_sec))
            return {"ok": True, "drained": True}

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            traj_dir = root / "traj-a"
            traj_dir.mkdir()
            (traj_dir / "traj.json").write_text("{}", encoding="utf-8")
            out_root = root / "out"
            args = types.SimpleNamespace(
                base_url="http://127.0.0.1:18090",
                policies=["always", "adaptive-risk"],
                adaptive_tail_root=str(root),
                adaptive_checkpoint_budget_sec=fault_tool.DEFAULT_ADAPTIVE_BUDGET_SEC,
                adaptive_decision_interval_sec=1.0,
                adaptive_failure_prob=0.01,
                max_concurrency=1,
                trajectory_root=str(root),
                limit=1,
                injection_seed=20260407,
                injection_count=0,
                injection_selection="random",
                output_root=str(out_root),
                gc_keep_latest=0,
                gc_dry_run=False,
                gc_min_checkpoint_count=100,
                gc_drain_timeout_sec=12.0,
                gc_drain_poll_interval_sec=0.25,
            )
            with mock.patch.object(fault_tool, "_collect_traj_paths", return_value=[traj_dir / "traj.json"]), mock.patch.object(
                fault_tool, "_select_injections", return_value=[]
            ), mock.patch.object(fault_tool, "_run_policy", new=fake_run_policy), mock.patch.object(
                fault_tool, "_wait_for_global_gc_drain", new=fake_wait_for_global_gc_drain
            ), mock.patch.object(
                fault_tool,
                "_load_adaptive_tail_model",
                return_value=types.SimpleNamespace(
                    budget_sec=fault_tool.DEFAULT_ADAPTIVE_BUDGET_SEC,
                    count=1,
                ),
            ):
                rc = asyncio.run(fault_tool._main_async(args))

        self.assertEqual(rc, 0)
        self.assertEqual(drain_calls, [("http://127.0.0.1:18090", 12.0, 0.25)])


if __name__ == "__main__":
    unittest.main()
