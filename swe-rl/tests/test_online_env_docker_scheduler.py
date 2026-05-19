from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from online_env_docker_scheduler import (  # noqa: E402
    OnlineEnvDockerScheduler,
    PromptResourceSummary,
    SchedulerConfig,
    extract_repo_key,
)


class DummySample:
    def __init__(self, repo: str, group_index: int, sample_index: int):
        self.group_index = group_index
        self.index = sample_index
        self.metadata = {
            "instance": {
                "repo": f"org/{repo}",
                "instance_id": f"org__{repo}-{sample_index}",
            }
        }


class FakeEnvClient:
    def __init__(self, stats_by_lease: dict[str, list[dict]] | None = None, always_fail: bool = False):
        self.stats_by_lease = stats_by_lease or {}
        self.always_fail = always_fail

    async def stats(self, lease_id: str) -> dict:
        if self.always_fail:
            raise RuntimeError("stats unavailable")

        seq = self.stats_by_lease.get(lease_id, [])
        if seq:
            return seq.pop(0)
        return {
            "ok": True,
            "memory_usage_bytes": 0,
            "cpu_percent": 0.0,
            "disk_read_bytes": 0,
            "disk_write_bytes": 0,
        }


def make_config(tmp_path: Path, **kwargs) -> SchedulerConfig:
    base = dict(
        enabled=True,
        sampling_interval_sec=0.02,
        scheduler_safety_margin=1.0,
        cpu_oversell_ratio=2.0,
        max_unknown_repo_concurrency=2,
        cold_start_memory_multiplier=1.0,
        cold_start_cpu_multiplier=1.0,
        startup_max_active_prompts=0,
        startup_cap_duration_sec=0.0,
        memory_budget_bytes=8,
        cpu_budget_percent=800.0,
        disk_read_budget_bytes=10_000,
        disk_write_budget_bytes=10_000,
        default_memory_bytes=2,
        default_cpu_percent=100.0,
        default_disk_read_bytes=100,
        default_disk_write_bytes=100,
        profile_json_path=str(tmp_path / "repo_resource_stats.json"),
    )
    base.update(kwargs)
    return SchedulerConfig(**base)


class TestOnlineEnvDockerScheduler(unittest.TestCase):
    def test_extract_repo_key_priority_and_fallbacks(self):
        self.assertEqual(extract_repo_key(sample_metadata={"instance": {"repo": "pallets/sphinx"}}), "sphinx")
        self.assertEqual(extract_repo_key(instance_id="pallets__sphinx-12345"), "sphinx")
        self.assertEqual(
            extract_repo_key(
                image_name="docker.io/swebench/sweb.eval.x86_64.pallets_s_sphinx-12345:latest"
            ),
            "sphinx",
        )

    def test_repo_profile_rolling_update(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(env_client=FakeEnvClient(), config=make_config(Path(td)))
            scheduler.record_prompt_summary(
                PromptResourceSummary(
                    prompt_id="p1",
                    repo="sphinx",
                    data_key="item-1",
                    sample_count=3,
                    peak_memory_bytes=10,
                    avg_cpu_percent=40.0,
                    disk_read_bytes=100,
                    disk_write_bytes=50,
                    started_at=0.0,
                    finished_at=1.0,
                )
            )
            scheduler.record_prompt_summary(
                PromptResourceSummary(
                    prompt_id="p2",
                    repo="sphinx",
                    data_key="item-2",
                    sample_count=4,
                    peak_memory_bytes=14,
                    avg_cpu_percent=60.0,
                    disk_read_bytes=300,
                    disk_write_bytes=150,
                    started_at=1.0,
                    finished_at=2.0,
                )
            )

            stats = scheduler.get_repo_resource_stats()
            self.assertEqual(set(stats.keys()), {"item-1", "item-2"})
            self.assertEqual(stats["item-1"]["sample_count"], 1)
            self.assertEqual(stats["item-2"]["sample_count"], 1)
            self.assertEqual(stats["item-1"]["peak_memory_bytes"], 10)
            self.assertEqual(stats["item-2"]["peak_memory_bytes"], 14)

    def test_plan_prompt_order_handles_unknown_repo(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(
                env_client=FakeEnvClient(),
                config=make_config(Path(td), memory_budget_bytes=6),
            )
            scheduler.record_prompt_summary(
                PromptResourceSummary(
                    prompt_id="h1",
                    repo="heavy",
                    data_key="heavy",
                    sample_count=3,
                    peak_memory_bytes=5,
                    avg_cpu_percent=120.0,
                    disk_read_bytes=300,
                    disk_write_bytes=200,
                    started_at=0.0,
                    finished_at=1.0,
                )
            )

            planned = scheduler.plan_prompt_order(
                [
                    {"repo": "unknown", "group_index": 0},
                    {"repo": "heavy", "group_index": 1},
                    {"repo": "unknown", "group_index": 2},
                ]
            )
            self.assertEqual(len(planned), 3)
            self.assertEqual({p["group_index"] for p in planned}, {0, 1, 2})

    def test_unknown_repo_prediction_uses_cold_start_multiplier(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(
                env_client=FakeEnvClient(),
                config=make_config(
                    Path(td),
                    default_memory_bytes=10,
                    default_cpu_percent=20.0,
                    cold_start_memory_multiplier=3.0,
                    cold_start_cpu_multiplier=2.0,
                ),
            )
            predicted = scheduler.predict_resources_for_repo("unknown")
            self.assertEqual(predicted.memory_bytes, 30)
            self.assertAlmostEqual(predicted.cpu_percent, 40.0, places=6)


class TestOnlineEnvDockerSchedulerAsync(unittest.IsolatedAsyncioTestCase):
    async def test_startup_active_prompt_cap_limits_initial_burst(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(
                env_client=FakeEnvClient(),
                config=make_config(
                    Path(td),
                    memory_budget_bytes=100,
                    cpu_budget_percent=10_000.0,
                    max_unknown_repo_concurrency=8,
                    startup_max_active_prompts=1,
                    startup_cap_duration_sec=3600.0,
                ),
            )

            async def submit(repo: str, idx: int):
                return await scheduler.admit_prompt(
                    sample=DummySample(repo=repo, group_index=0, sample_index=idx),
                    image_name=f"docker.io/swebench/sweb.eval.x86_64.org_s_{repo}-{idx}:latest",
                    rollout_batch_size=8,
                )

            t1 = asyncio.create_task(submit("a", 1))
            t2 = asyncio.create_task(submit("b", 2))
            await asyncio.sleep(0.05)
            self.assertTrue(t1.done())
            self.assertFalse(t2.done())

            tk1 = await t1
            await scheduler.finish_prompt(tk1.prompt_id)
            tk2 = await asyncio.wait_for(t2, timeout=1.0)
            await scheduler.finish_prompt(tk2.prompt_id)

    async def test_unknown_repo_concurrency_is_limited_during_cold_start(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(
                env_client=FakeEnvClient(),
                config=make_config(
                    Path(td),
                    memory_budget_bytes=100,
                    cpu_budget_percent=10_000.0,
                    max_unknown_repo_concurrency=1,
                ),
            )

            async def submit(idx: int):
                return await scheduler.admit_prompt(
                    sample=DummySample(repo=f"u{idx}", group_index=0, sample_index=idx),
                    image_name=f"docker.io/swebench/sweb.eval.x86_64.org_s_u{idx}-{idx}:latest",
                    rollout_batch_size=8,
                )

            t1 = asyncio.create_task(submit(1))
            t2 = asyncio.create_task(submit(2))
            await asyncio.sleep(0.05)
            self.assertTrue(t1.done())
            self.assertFalse(t2.done())

            tk1 = await t1
            await scheduler.finish_prompt(tk1.prompt_id)
            tk2 = await asyncio.wait_for(t2, timeout=1.0)
            await scheduler.finish_prompt(tk2.prompt_id)

    async def test_cpu_budget_allows_200_percent_oversell(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(
                env_client=FakeEnvClient(),
                config=make_config(
                    Path(td),
                    memory_budget_bytes=100,
                    cpu_budget_percent=100.0,
                    cpu_oversell_ratio=2.0,
                    default_memory_bytes=1,
                    default_cpu_percent=100.0,
                    default_disk_read_bytes=1,
                    default_disk_write_bytes=1,
                ),
            )

            async def submit(repo: str, idx: int):
                return await scheduler.admit_prompt(
                    sample=DummySample(repo=repo, group_index=0, sample_index=idx),
                    image_name=f"docker.io/swebench/sweb.eval.x86_64.org_s_{repo}-{idx}:latest",
                    rollout_batch_size=8,
                )

            t1 = asyncio.create_task(submit("a", 1))
            t2 = asyncio.create_task(submit("b", 2))
            await asyncio.sleep(0.05)
            self.assertTrue(t1.done() and t2.done())
            self.assertLessEqual(scheduler._active_predicted.cpu_percent, 200.0)

            tk1 = await t1
            tk2 = await t2
            await scheduler.finish_prompt(tk1.prompt_id)
            await scheduler.finish_prompt(tk2.prompt_id)

    async def test_memory_budget_allows_200_percent_oversell(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(
                env_client=FakeEnvClient(),
                config=make_config(
                    Path(td),
                    memory_budget_bytes=2,
                    memory_oversell_ratio=2.0,
                    cpu_budget_percent=1000.0,
                    default_memory_bytes=2,
                    default_cpu_percent=1.0,
                    default_disk_read_bytes=1,
                    default_disk_write_bytes=1,
                ),
            )

            async def submit(repo: str, idx: int):
                return await scheduler.admit_prompt(
                    sample=DummySample(repo=repo, group_index=0, sample_index=idx),
                    image_name=f"docker.io/swebench/sweb.eval.x86_64.org_s_{repo}-{idx}:latest",
                    rollout_batch_size=8,
                )

            t1 = asyncio.create_task(submit("a", 1))
            t2 = asyncio.create_task(submit("b", 2))
            await asyncio.sleep(0.05)
            self.assertTrue(t1.done() and t2.done())
            self.assertLessEqual(scheduler._active_predicted.memory_bytes, 4)

            tk1 = await t1
            tk2 = await t2
            await scheduler.finish_prompt(tk1.prompt_id)
            await scheduler.finish_prompt(tk2.prompt_id)

    async def test_greedy_admission_respects_budget_with_mixed_tasks(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(
                env_client=FakeEnvClient(),
                config=make_config(Path(td), memory_budget_bytes=6),
            )

            scheduler.record_prompt_summary(
                PromptResourceSummary(
                    prompt_id="h0",
                    repo="heavy",
                    data_key="org__heavy-0",
                    sample_count=3,
                    peak_memory_bytes=5,
                    avg_cpu_percent=100.0,
                    disk_read_bytes=100,
                    disk_write_bytes=100,
                    started_at=0.0,
                    finished_at=1.0,
                )
            )
            scheduler.record_prompt_summary(
                PromptResourceSummary(
                    prompt_id="l0",
                    repo="light",
                    data_key="org__light-1",
                    sample_count=3,
                    peak_memory_bytes=1,
                    avg_cpu_percent=100.0,
                    disk_read_bytes=100,
                    disk_write_bytes=100,
                    started_at=0.0,
                    finished_at=1.0,
                )
            )

            async def submit(repo: str, idx: int):
                return await scheduler.admit_prompt(
                    sample=DummySample(repo=repo, group_index=0, sample_index=idx),
                    image_name=f"docker.io/swebench/sweb.eval.x86_64.org_s_{repo}-{idx}:latest",
                    rollout_batch_size=8,
                )

            t_heavy = asyncio.create_task(submit("heavy", 0))
            t_light1 = asyncio.create_task(submit("light", 1))
            t_light2 = asyncio.create_task(submit("light", 2))

            await asyncio.sleep(0.05)
            done_cnt = sum(1 for t in [t_heavy, t_light1, t_light2] if t.done())
            self.assertEqual(done_cnt, 2)
            self.assertLessEqual(scheduler._active_predicted.memory_bytes, 6)

            ticket_heavy = await t_heavy
            ticket_light1 = await t_light1

            await scheduler.finish_prompt(ticket_heavy.prompt_id)
            await asyncio.wait_for(t_light2, timeout=1.0)

            await scheduler.finish_prompt(ticket_light1.prompt_id)
            ticket_light2 = t_light2.result()
            await scheduler.finish_prompt(ticket_light2.prompt_id)

    async def test_stats_missing_fallback_to_prediction(self):
        with tempfile.TemporaryDirectory() as td:
            scheduler = OnlineEnvDockerScheduler(
                env_client=FakeEnvClient(always_fail=True),
                config=make_config(Path(td), memory_budget_bytes=4),
            )

            ticket = await scheduler.admit_prompt(
                sample=DummySample(repo="unknown", group_index=0, sample_index=0),
                image_name="docker.io/swebench/sweb.eval.x86_64.psf_s_sphinx-1:latest",
                rollout_batch_size=8,
            )
            await scheduler.attach_lease(prompt_id=ticket.prompt_id, lease_id="lease-fail")

            await asyncio.sleep(0.09)
            summary = await scheduler.finish_prompt(ticket.prompt_id)
            assert summary is not None
            self.assertEqual(summary.peak_memory_bytes, scheduler.config.default_memory_bytes)
            self.assertAlmostEqual(summary.avg_cpu_percent, scheduler.config.default_cpu_percent, places=6)

    async def test_integration_round1_collect_round2_reorder(self):
        with tempfile.TemporaryDirectory() as td:
            env_client = FakeEnvClient(
                stats_by_lease={
                    "lease-heavy": [
                        {
                            "memory_usage_bytes": 5,
                            "cpu_percent": 300.0,
                            "disk_read_bytes": 800,
                            "disk_write_bytes": 200,
                        },
                        {
                            "memory_usage_bytes": 4,
                            "cpu_percent": 100.0,
                            "disk_read_bytes": 900,
                            "disk_write_bytes": 240,
                        },
                    ],
                    "lease-light": [
                        {
                            "memory_usage_bytes": 1,
                            "cpu_percent": 30.0,
                            "disk_read_bytes": 120,
                            "disk_write_bytes": 40,
                        },
                    ],
                }
            )
            scheduler = OnlineEnvDockerScheduler(
                env_client=env_client,
                config=make_config(Path(td), memory_budget_bytes=6),
            )

            heavy = await scheduler.admit_prompt(
                sample=DummySample(repo="heavy", group_index=0, sample_index=0),
                image_name="docker.io/swebench/sweb.eval.x86_64.org_s_heavy-0:latest",
                rollout_batch_size=8,
            )
            light = await scheduler.admit_prompt(
                sample=DummySample(repo="light", group_index=1, sample_index=1),
                image_name="docker.io/swebench/sweb.eval.x86_64.org_s_light-1:latest",
                rollout_batch_size=8,
            )
            await scheduler.attach_lease(prompt_id=heavy.prompt_id, lease_id="lease-heavy")
            await scheduler.attach_lease(prompt_id=light.prompt_id, lease_id="lease-light")

            await asyncio.sleep(0.08)
            await scheduler.finish_prompt(heavy.prompt_id)
            await scheduler.finish_prompt(light.prompt_id)

            stats = scheduler.get_repo_resource_stats()
            self.assertEqual(stats["org__heavy-0"]["peak_memory_bytes"], 5)
            self.assertEqual(stats["org__light-1"]["peak_memory_bytes"], 1)

            plan = scheduler.plan_prompt_order(
                [
                    {"repo": "heavy", "group_index": 8},
                    {"repo": "light", "group_index": 9},
                    {"repo": "light", "group_index": 10},
                ]
            )
            self.assertEqual(plan[0]["repo"], "heavy")


if __name__ == "__main__":
    unittest.main()
