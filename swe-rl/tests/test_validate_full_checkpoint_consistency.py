from __future__ import annotations

import hashlib
import json
import sys
import random
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import validate_full_checkpoint_consistency as consistency_tool  # noqa: E402


class TestValidateFullCheckpointConsistency(unittest.TestCase):
    @staticmethod
    def _runtime_artifact(path: Path, *, recorded_step: int = 1, mutate_heap: bool = False):
        seed = 17
        token = "token-001"
        prng_state = 23
        accumulator = 29
        heap = bytearray(b"complete-heap-state")
        heap_digest = hashlib.sha256(heap).digest()
        prior_runtime = hashlib.sha256(f"runtime-init:{seed}".encode()).digest()
        prior_filesystem = hashlib.sha256(f"filesystem-init:{seed}".encode()).digest()
        effect = hashlib.sha256(
            b"effect-v1\0"
            + prior_runtime
            + prior_filesystem
            + f"{seed}:1:{token}:{prng_state}:{accumulator}".encode()
        ).digest()
        state_chain = hashlib.sha256(
            b"runtime-chain-v1\0"
            + prior_runtime
            + prior_filesystem
            + struct.pack(">QQQ", 1, prng_state, accumulator)
            + heap_digest
            + effect
        ).digest()
        filesystem_chain = hashlib.sha256(
            b"filesystem-chain-v1\0"
            + prior_filesystem
            + state_chain
            + struct.pack(">Q", 1)
            + effect
        ).digest()
        history = [
            {
                "accumulator": accumulator,
                "effect": effect.hex(),
                "filesystem_chain": filesystem_chain.hex(),
                "heap_sha256": heap_digest.hex(),
                "prng_state": prng_state,
                "state_chain": state_chain.hex(),
                "step": recorded_step,
                "token": token,
            }
        ]
        history_bytes = json.dumps(
            history, sort_keys=True, separators=(",", ":")
        ).encode()
        if mutate_heap:
            heap[-1] ^= 1
        header = consistency_tool.RUNTIME_ARTIFACT_HEADER.pack(
            consistency_tool.RUNTIME_ARTIFACT_MAGIC,
            consistency_tool.RUNTIME_ARTIFACT_SCHEMA,
            seed,
            1,
            1,
            prng_state,
            accumulator,
            101,
            0,
            len(history_bytes),
            len(heap),
        )
        path.write_bytes(header + state_chain + history_bytes + heap)
        return seed, len(heap)

    def _worker(self, events):
        return consistency_tool.WorkerResult(
            exit_code=-9,
            duration_sec=0.1,
            events=events,
            fault_observed=any(item.get("event") == "fault_fired" for item in events),
        )

    def test_phase_lists_are_ordered_and_unique(self):
        for phases in (
            consistency_tool.ACTION_PHASES,
            consistency_tool.CHECKPOINT_PHASES,
            consistency_tool.RESUME_PHASES,
        ):
            self.assertEqual(len(phases), len(set(phases)))
            self.assertTrue(phases)

    def test_exact_boundary_fault_requires_requested_phase(self):
        worker = self._worker(
            [
                {"event": "phase", "phase": "checkpoint_before_runtime_dump"},
                {"event": "fault_fired", "phase": "checkpoint_before_runtime_dump"},
            ]
        )
        experiment = object.__new__(consistency_tool.FullConsistencyExperiment)

        self.assertTrue(
            experiment._fault_hit_requested_window(  # pylint: disable=protected-access
                worker,
                {"exact_phase": "checkpoint_before_runtime_dump"},
            )
        )
        self.assertFalse(
            experiment._fault_hit_requested_window(  # pylint: disable=protected-access
                worker,
                {"exact_phase": "checkpoint_after_runtime_dump"},
            )
        )

    def test_timed_fault_is_incomplete_if_operation_already_crossed_after_boundary(self):
        experiment = object.__new__(consistency_tool.FullConsistencyExperiment)
        inside = self._worker(
            [
                {"event": "phase", "phase": "resume_before_runtime_restore"},
                {"event": "fault_fired", "phase": "resume_before_runtime_restore"},
            ]
        )
        too_late = self._worker(
            [
                {"event": "phase", "phase": "resume_before_runtime_restore"},
                {"event": "phase", "phase": "resume_after_runtime_restore"},
                {"event": "fault_fired", "phase": "resume_before_runtime_restore"},
            ]
        )
        spec = {"arm_phase": "resume_before_runtime_restore", "delay_sec": 0.05}

        self.assertTrue(
            experiment._fault_hit_requested_window(inside, spec)  # pylint: disable=protected-access
        )
        self.assertFalse(
            experiment._fault_hit_requested_window(too_late, spec)  # pylint: disable=protected-access
        )

    def test_async_window_allows_intermediate_phases_but_not_window_end(self):
        experiment = object.__new__(consistency_tool.FullConsistencyExperiment)
        inside = self._worker(
            [
                {"event": "phase", "phase": "checkpoint_after_runtime_persist_started"},
                {"event": "phase", "phase": "checkpoint_before_rootfs_snapshot"},
                {
                    "event": "fault_fired",
                    "phase": "checkpoint_after_runtime_persist_started",
                },
            ]
        )
        too_late = self._worker(
            [
                {"event": "phase", "phase": "checkpoint_after_runtime_persist_started"},
                {"event": "phase", "phase": "checkpoint_after_runtime_persist_wait"},
                {
                    "event": "fault_fired",
                    "phase": "checkpoint_after_runtime_persist_started",
                },
            ]
        )
        spec = {
            "window_start_phase": "checkpoint_after_runtime_persist_started",
            "window_end_phase": "checkpoint_after_runtime_persist_wait",
        }

        self.assertTrue(
            experiment._fault_hit_requested_window(inside, spec)  # pylint: disable=protected-access
        )
        self.assertFalse(
            experiment._fault_hit_requested_window(too_late, spec)  # pylint: disable=protected-access
        )

    def test_final_status_separates_coverage_from_consistency(self):
        experiment = object.__new__(consistency_tool.FullConsistencyExperiment)
        experiment.golden = {"snapshot": {"fingerprint": "golden"}}
        recovery = {"mode": "base_restart"}

        def final(filesystem_match, runtime_match):
            return {
                "fingerprint": "diagnostic-only",
                "oracles": {
                    "filesystem": {"match": filesystem_match},
                    "runtime": {"match": runtime_match},
                },
            }

        missed = experiment._finalize_trial(  # pylint: disable=protected-access
            final(True, True),
            recovery,
            fault_observed=False,
        )
        mismatch = experiment._finalize_trial(  # pylint: disable=protected-access
            final(False, True),
            recovery,
            fault_observed=True,
        )
        passed = experiment._finalize_trial(  # pylint: disable=protected-access
            final(True, True),
            recovery,
            fault_observed=True,
        )

        self.assertEqual(missed["status"], "coverage_incomplete")
        self.assertEqual(mismatch["status"], "final_state_mismatch")
        self.assertEqual(passed["status"], "pass")
        self.assertFalse(mismatch["filesystem_rsync_match"])
        self.assertTrue(mismatch["runtime_differential_match"])

    def test_runtime_artifact_validates_complete_heap_and_action_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Path(tmp) / "runtime.bin"
            seed, heap_bytes = self._runtime_artifact(artifact)

            validation = consistency_tool.validate_runtime_artifact(
                artifact,
                expected_seed=seed,
                expected_steps=1,
                expected_heap_bytes=heap_bytes,
            )

            self.assertTrue(validation["transcript_valid"])
            self.assertEqual(validation["heap_bytes"], heap_bytes)
            self.assertEqual(validation["history_entries"], 1)

    def test_runtime_artifact_rejects_heap_mutation_and_non_exactly_once_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heap_mutation = root / "heap-mutation.bin"
            seed, heap_bytes = self._runtime_artifact(heap_mutation, mutate_heap=True)
            with self.assertRaisesRegex(
                consistency_tool.ExperimentError,
                "full heap differs",
            ):
                consistency_tool.validate_runtime_artifact(
                    heap_mutation,
                    expected_seed=seed,
                    expected_steps=1,
                    expected_heap_bytes=heap_bytes,
                )

            reordered = root / "reordered.bin"
            seed, heap_bytes = self._runtime_artifact(reordered, recorded_step=2)
            with self.assertRaisesRegex(
                consistency_tool.ExperimentError,
                "missing, duplicated, or reordered",
            ):
                consistency_tool.validate_runtime_artifact(
                    reordered,
                    expected_seed=seed,
                    expected_steps=1,
                    expected_heap_bytes=heap_bytes,
                )

    def test_external_comparators_detect_injected_differences(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            golden_tree = root / "golden"
            recovered_tree = root / "recovered"
            golden_tree.mkdir()
            recovered_tree.mkdir()
            (golden_tree / "state.bin").write_bytes(b"golden-state")
            (recovered_tree / "state.bin").write_bytes(b"changed-state")
            (recovered_tree / "extra.txt").write_text("extra\n", encoding="utf-8")

            rsync = subprocess.run(
                [
                    "rsync",
                    "--archive",
                    "--hard-links",
                    "--acls",
                    "--xattrs",
                    "--checksum",
                    "--dry-run",
                    "--delete",
                    "--itemize-changes",
                    f"{golden_tree}/",
                    f"{recovered_tree}/",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(rsync.returncode, 0, rsync.stderr)
            self.assertIn("state.bin", rsync.stdout)
            self.assertIn("extra.txt", rsync.stdout)

            golden_runtime = root / "golden-runtime"
            recovered_runtime = root / "recovered-runtime"
            golden_runtime.write_bytes(b"runtime-a")
            recovered_runtime.write_bytes(b"runtime-b")
            compared = subprocess.run(
                ["cmp", "--silent", str(golden_runtime), str(recovered_runtime)],
                check=False,
            )
            self.assertEqual(compared.returncode, 1)

    def test_wallclock_fault_is_classified_only_from_preceding_trace(self):
        events = [
            {
                "event": "phase",
                "phase": "checkpoint_before_runtime_dump",
                "time_ns": 100,
            },
            {
                "event": "phase",
                "phase": "checkpoint_after_runtime_dump",
                "time_ns": 300,
            },
        ]

        classified = consistency_tool._classify_wallclock_fault(  # pylint: disable=protected-access
            events,
            fault_time_ns=200,
        )

        self.assertEqual(classified["stage"], "checkpoint.runtime_dump")
        self.assertEqual(
            classified["last_event"]["phase"],
            "checkpoint_before_runtime_dump",
        )

    def test_stratified_delays_are_phase_independent_and_cover_time_strata(self):
        delays = consistency_tool.FullConsistencyExperiment._stratified_wallclock_delays(  # pylint: disable=protected-access
            count=4,
            duration_sec=8.0,
            rng=random.Random(7),
        )

        self.assertEqual(len(delays), 4)
        self.assertEqual(
            sorted(int(delay // 2.0) for delay in delays),
            [0, 1, 2, 3],
        )


if __name__ == "__main__":
    unittest.main()
