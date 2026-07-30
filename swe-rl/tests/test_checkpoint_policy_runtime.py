from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent
SWE_RL_DIR = TESTS_DIR.parent
sys.path.insert(0, str(SWE_RL_DIR))

from checkpoint_policy_runtime import (  # noqa: E402
    LongTrajectoryFaultPlanner,
    adaptive_delta_replay_cost_sec,
    adaptive_expected_benefit_sec,
    fault_injection_armed_for_policy,
    fault_injection_enabled_for_policy,
    redo_replay_cost_sec,
    rollout_sample_key,
    should_probe_in_llm_bubble,
)


class CheckpointPolicyRuntimeTests(unittest.TestCase):
    def test_oracle_participates_in_fault_planning_but_not_fault_injection(self) -> None:
        self.assertTrue(fault_injection_enabled_for_policy("oracle-no-fault-no-checkpoint", True))
        self.assertFalse(fault_injection_armed_for_policy("oracle-no-fault-no-checkpoint", True))
        self.assertTrue(fault_injection_armed_for_policy("adaptive-risk", True))

    def test_rollout_sample_key(self) -> None:
        self.assertEqual(
            rollout_sample_key(instance_id="iid", group_index=3, sample_index=9),
            "iid__g3__i9",
        )

    def test_should_probe_in_llm_bubble_respects_warmup(self) -> None:
        self.assertFalse(
            should_probe_in_llm_bubble(
                current_step_idx=2,
                probe_attempted_in_bubble=False,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[],
                delta_env_cost_sec_value=10.0,
                steps_since_latest_ready_checkpoint=10,
                expected_benefit_sec=5.0,
                expected_overhead_sec=1.0,
            )
        )
        self.assertTrue(
            should_probe_in_llm_bubble(
                current_step_idx=3,
                probe_attempted_in_bubble=False,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[],
                delta_env_cost_sec_value=10.0,
                steps_since_latest_ready_checkpoint=10,
                expected_benefit_sec=5.0,
                expected_overhead_sec=1.0,
            )
        )

    def test_redo_replay_cost_includes_llm_and_env(self) -> None:
        self.assertAlmostEqual(
            redo_replay_cost_sec(
                cumulative_env_replay_cost_sec=12.5,
                latest_ready_protected_env_cost_sec=4.0,
                cumulative_llm_replay_cost_sec=7.0,
                latest_ready_protected_llm_cost_sec=2.5,
            ),
            13.0,
        )
        self.assertAlmostEqual(
            adaptive_delta_replay_cost_sec(
                cumulative_env_replay_cost_sec=12.5,
                latest_ready_protected_env_cost_sec=4.0,
                cumulative_llm_replay_cost_sec=7.0,
                latest_ready_protected_llm_cost_sec=2.5,
            ),
            13.0,
        )

    def test_adaptive_expected_benefit_uses_full_regeneration_cost(self) -> None:
        self.assertAlmostEqual(adaptive_expected_benefit_sec(0.05, 10.0), 0.5)

    def test_should_probe_when_benefit_equals_visible_overhead(self) -> None:
        self.assertTrue(
            should_probe_in_llm_bubble(
                current_step_idx=3,
                probe_attempted_in_bubble=False,
                adaptive_checkpoint_submitted=False,
                pending_checkpoints=[],
                delta_env_cost_sec_value=10.0,
                steps_since_latest_ready_checkpoint=10,
                expected_benefit_sec=1.0,
                expected_overhead_sec=1.0,
            )
        )

    def test_long_trajectory_fault_planner_selects_longest_registered_samples(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            ref = tmp / "reference"
            ref.mkdir()
            for instance_id, group_index, sample_index, step_count in (
                ("iid-long-a", 0, 0, 20),
                ("iid-long-b", 0, 1, 18),
                ("iid-short", 0, 2, 7),
            ):
                run_dir = ref / rollout_sample_key(
                    instance_id=instance_id,
                    group_index=group_index,
                    sample_index=sample_index,
                )
                run_dir.mkdir()
                payload = {
                    "step_debug": [{"step_idx": i} for i in range(step_count)],
                    "info": {
                        "instance_id": instance_id,
                        "group_index": group_index,
                        "index": sample_index,
                    },
                }
                (run_dir / "traj.json").write_text(json.dumps(payload), encoding="utf-8")

            planner = LongTrajectoryFaultPlanner(
                plan_path=tmp / "fault_plan.json",
                reference_root=ref,
                expected_sample_count=3,
                injection_count=2,
                seed=7,
                offset_from_end_steps=3,
                max_inject_before_step_idx=8,
                poll_interval_sec=0.01,
                wait_timeout_sec=10.0,
            )

            state = planner._register_and_maybe_finalize(
                sample_key=rollout_sample_key(instance_id="iid-short", group_index=0, sample_index=2),
                instance_id="iid-short",
                step_limit=20,
                force_finalize=False,
            )
            self.assertFalse(state["finalized"])
            state = planner._register_and_maybe_finalize(
                sample_key=rollout_sample_key(instance_id="iid-long-a", group_index=0, sample_index=0),
                instance_id="iid-long-a",
                step_limit=20,
                force_finalize=False,
            )
            self.assertFalse(state["finalized"])
            state = planner._register_and_maybe_finalize(
                sample_key=rollout_sample_key(instance_id="iid-long-b", group_index=0, sample_index=1),
                instance_id="iid-long-b",
                step_limit=20,
                force_finalize=False,
            )
            self.assertTrue(state["finalized"])

            planner2 = LongTrajectoryFaultPlanner(
                plan_path=tmp / "fault_plan.json",
                reference_root=ref,
                expected_sample_count=3,
                injection_count=2,
                seed=7,
                offset_from_end_steps=3,
                max_inject_before_step_idx=8,
                poll_interval_sec=0.01,
                wait_timeout_sec=0.01,
            )
            target_a = planner2.register_and_wait(
                sample_key=rollout_sample_key(instance_id="iid-long-a", group_index=0, sample_index=0),
                instance_id="iid-long-a",
                step_limit=20,
            )
            target_b = planner2.register_and_wait(
                sample_key=rollout_sample_key(instance_id="iid-long-b", group_index=0, sample_index=1),
                instance_id="iid-long-b",
                step_limit=20,
            )
            self.assertIsNotNone(target_a)
            self.assertIsNotNone(target_b)
            self.assertEqual(target_a.inject_before_step_idx, 8)
            self.assertEqual(target_b.inject_before_step_idx, 8)

    def test_long_trajectory_fault_planner_random_mode_is_seeded(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            planner = LongTrajectoryFaultPlanner(
                plan_path=tmp / "fault_plan.json",
                reference_root=tmp / "reference",
                expected_sample_count=4,
                injection_count=2,
                seed=7,
                selection_mode="random",
                offset_from_end_steps=3,
                max_inject_before_step_idx=8,
            )
            selected_a = planner._select_targets(
                {
                    "a": {"reference_step_count": 20, "step_limit": 20},
                    "b": {"reference_step_count": 19, "step_limit": 20},
                    "c": {"reference_step_count": 18, "step_limit": 20},
                    "d": {"reference_step_count": 17, "step_limit": 20},
                }
            )
            selected_b = planner._select_targets(
                {
                    "a": {"reference_step_count": 20, "step_limit": 20},
                    "b": {"reference_step_count": 19, "step_limit": 20},
                    "c": {"reference_step_count": 18, "step_limit": 20},
                    "d": {"reference_step_count": 17, "step_limit": 20},
                }
            )
            self.assertEqual(selected_a, selected_b)
            self.assertEqual(len(selected_a), 2)
            self.assertNotEqual(set(selected_a.keys()), {"a", "b"})


if __name__ == "__main__":
    unittest.main()
