from __future__ import annotations

import json
import tempfile
import types
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

if "httpx" not in sys.modules:
    sys.modules["httpx"] = types.SimpleNamespace(
        AsyncClient=object,
        Limits=object,
        Timeout=object,
        HTTPStatusError=Exception,
    )

import validate_swe_checkpoint_correctness as correctness_tool  # noqa: E402


class TestValidateSweCheckpointCorrectness(unittest.TestCase):
    def test_phase_expected_prefix(self):
        self.assertEqual(correctness_tool._phase_expected_prefix(3, "before_action"), 2)
        self.assertEqual(correctness_tool._phase_expected_prefix(3, "mid_action"), 2)
        self.assertEqual(correctness_tool._phase_expected_prefix(3, "before_commit"), 2)
        self.assertEqual(correctness_tool._phase_expected_prefix(3, "after_checkpoint_ready"), 3)

    def test_detect_semantic_anomalies(self):
        expected = {
            "counter": 2,
            "history_lines": ["STEP_1_DONE", "STEP_2_DONE"],
            "markers_lines": ["BEGIN_STEP_1", "END_STEP_1", "BEGIN_STEP_2", "END_STEP_2"],
            "important_exists": True,
            "config_tmp_exists": False,
            "nested_exists": False,
            "finalized_exists": False,
            "config_json_sha": "cfg-a",
        }
        actual = {
            "counter": 1,
            "history_lines": ["STEP_1_DONE", "STEP_2_DONE", "STEP_2_DONE"],
            "markers_lines": ["BEGIN_STEP_1", "END_STEP_1", "BEGIN_STEP_3"],
            "important_exists": False,
            "config_tmp_exists": True,
            "nested_exists": True,
            "finalized_exists": False,
            "config_json_sha": "cfg-b",
        }

        out = correctness_tool._detect_semantic_anomalies(expected, actual, expected_prefix=2)

        self.assertTrue(out["duplicate_effect"])
        self.assertTrue(out["lost_effect"])
        self.assertTrue(out["partial_effect"])
        self.assertIn("STEP_2_DONE", out["duplicate_tokens"])
        self.assertIn("BEGIN_STEP_3", out["partial_markers"])
        self.assertIn("config.tmp", out["partial_paths"])

    def test_scan_non_idempotent_actions(self):
        payload = {
            "info": {"instance_id": "demo__repo-1"},
            "step_debug": [
                {"step_idx": 0, "action": "echo hello"},
                {"step_idx": 1, "action": "sed -i 's/a/b/' file.py"},
                {"step_idx": 2, "action": "echo 'done' >> history.log"},
                {"step_idx": 3, "action": "rm important.txt"},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "traj-1"
            traj_dir.mkdir(parents=True)
            (traj_dir / "traj.json").write_text(json.dumps(payload), encoding="utf-8")

            candidates = correctness_tool._scan_non_idempotent_actions(
                tmpdir,
                trajectory_limit=10,
                candidate_limit=10,
            )

        self.assertEqual(len(candidates), 3)
        self.assertEqual(candidates[0]["step_idx"], 1)
        self.assertIn("in_place_edit", candidates[0]["tags"])
        self.assertIn("append_redirect", candidates[1]["tags"])
        self.assertIn("destructive_delete", candidates[2]["tags"])

    def test_resolve_image_name_uses_sidecar_meta_data_source(self):
        traj_payload = {
            "info": {"instance_id": "demo__repo-1"},
            "step_debug": [],
        }
        meta_payload = {
            "sample_metadata": {
                "data_source": "SumanthRH/SWE-Gym-Subset",
                "instance": {"instance_id": "demo__repo-1"},
            }
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_dir = Path(tmpdir) / "traj-1"
            traj_dir.mkdir(parents=True)
            traj_path = traj_dir / "traj.json"
            traj_path.write_text(json.dumps(traj_payload), encoding="utf-8")
            (traj_dir / "meta.json").write_text(json.dumps(meta_payload), encoding="utf-8")

            image_name, instance_id, resolved_traj, resolved_data_source = correctness_tool._resolve_image_name(
                image_name=None,
                instance_id=None,
                trajectory=str(traj_path),
                trajectory_root=None,
                data_source="princeton-nlp/SWE-bench_Lite",
            )

        self.assertEqual(instance_id, "demo__repo-1")
        self.assertEqual(resolved_traj, str(traj_path))
        self.assertEqual(resolved_data_source, "SumanthRH/SWE-Gym-Subset")
        self.assertIn("swe-rl:demo_s_repo-1", image_name)

    def test_mid_action_fault_observed_accepts_kill_return_code_without_ack(self):
        self.assertTrue(
            correctness_tool._is_mid_action_fault_observed(
                {
                    "ok": True,
                    "returncode": 137,
                    "output": "",
                    "fault_injected": False,
                    "container_usable": True,
                }
            )
        )
        self.assertFalse(
            correctness_tool._is_mid_action_fault_observed(
                {
                    "ok": True,
                    "returncode": 0,
                    "output": "normal completion",
                    "fault_injected": False,
                }
            )
        )

    def test_random_fault_helpers_accept_paused_exec_and_checkpoint_probe_failures(self):
        self.assertTrue(
            correctness_tool._exec_result_indicates_fail_stop(
                {
                    "ok": True,
                    "returncode": 1,
                    "output": "Error response from daemon: Container abc is paused, unpause the container before exec",
                }
            )
        )
        self.assertTrue(
            correctness_tool._checkpoint_result_indicates_random_fault(
                {
                    "ok": False,
                    "error_code": "checkpoint_create_failed",
                    "error": "failed to probe runtime env for abc",
                }
            )
        )
        self.assertTrue(
            correctness_tool._checkpoint_result_indicates_random_fault(
                {
                    "ok": False,
                    "error_code": "checkpoint_create_failed",
                    "error": "failed to write runtime state for swe-ckpt-1",
                }
            )
        )

    def test_random_trial_specs_are_seeded_and_record_metadata(self):
        steps = [
            correctness_tool.ValidationStep(step_idx=1, name="a", command="true"),
            correctness_tool.ValidationStep(step_idx=2, name="b", command="true"),
        ]

        first = correctness_tool._build_random_trial_specs(
            phases=["before_action", "mid_action"],
            steps=steps,
            count=5,
            seed=7,
            min_delay_sec=0.1,
            max_delay_sec=0.2,
        )
        second = correctness_tool._build_random_trial_specs(
            phases=["before_action", "mid_action"],
            steps=steps,
            count=5,
            seed=7,
            min_delay_sec=0.1,
            max_delay_sec=0.2,
        )

        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)
        self.assertTrue(all(item["trial_mode"] == "random" for item in first))
        self.assertTrue(all(item["phase"] == "random_wall_clock" for item in first))
        self.assertTrue(all(item["inject_step"] is None for item in first))
        self.assertEqual([item["random_trial_index"] for item in first], list(range(5)))
        self.assertTrue(all(str(item["trial_id"]).startswith("random_") for item in first))
        for item in first:
            self.assertGreaterEqual(item["random_fault_delay_sec"], 0.1)
            self.assertLessEqual(item["random_fault_delay_sec"], 0.2)

    def test_random_trial_specs_can_sample_checkpoint_internal_faults(self):
        steps = [
            correctness_tool.ValidationStep(step_idx=1, name="a", command="true"),
            correctness_tool.ValidationStep(step_idx=2, name="b", command="true"),
        ]

        specs = correctness_tool._build_random_trial_specs(
            phases=["before_action", "mid_action"],
            steps=steps,
            count=4,
            seed=11,
            min_delay_sec=0.1,
            max_delay_sec=0.2,
            checkpoint_interrupt_probability=1.0,
            checkpoint_interrupt_phases=["before_commit"],
            checkpoint_interrupt_delay_sec=0.03,
        )

        self.assertEqual(len(specs), 4)
        self.assertTrue(all(item["trial_mode"] == "random" for item in specs))
        self.assertTrue(all(item["phase"] == "random_wall_clock" for item in specs))
        self.assertTrue(all(item["random_fault_strategy"] == "checkpoint_internal" for item in specs))
        self.assertTrue(all(item["checkpoint_fault_phase"] == "before_commit" for item in specs))
        self.assertTrue(all(item["checkpoint_fault_delay_sec"] == 0.03 for item in specs))
        self.assertTrue(all(item["inject_step"] in {1, 2} for item in specs))

    def test_systematic_trial_specs_cover_phase_step_cross_product(self):
        steps = [
            correctness_tool.ValidationStep(step_idx=1, name="a", command="true"),
            correctness_tool.ValidationStep(step_idx=2, name="b", command="true"),
        ]

        specs = correctness_tool._build_systematic_trial_specs(
            phases=["before_action", "after_checkpoint_ready"],
            steps=steps,
        )

        self.assertEqual(len(specs), 4)
        self.assertEqual(specs[0]["trial_id"], "before_action__step_01")
        self.assertEqual(specs[-1]["trial_id"], "after_checkpoint_ready__step_02")
        self.assertTrue(all(item["trial_mode"] == "systematic" for item in specs))


if __name__ == "__main__":
    unittest.main()
