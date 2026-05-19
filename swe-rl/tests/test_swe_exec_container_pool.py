from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path
from types import ModuleType

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

import swe_exec_server as exec_server  # noqa: E402


class FakeDockerOps:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.created = 0
        self.destroyed: list[str] = []
        self.unhealthy_ids: set[str] = set()

    def create(self, *, image: str, cwd: str, timeout: int) -> dict:
        with self._lock:
            self.created += 1
            index = self.created
        return {
            "container_id": f"cid-{index}",
            "name": f"ctr-{index}",
            "create_time_sec": 0.01 * index,
        }

    def destroy(self, container_id: str) -> None:
        with self._lock:
            self.destroyed.append(container_id)

    def healthy(self, container_id: str, cwd: str, timeout: int) -> bool:
        return container_id not in self.unhealthy_ids


def make_pool(
    fake: FakeDockerOps,
    *,
    prewarm_images: list[str] | None = None,
    active_count_fn=None,
    **overrides,
) -> exec_server.ContainerPool:
    config = exec_server.ExecServerConfig(
        use_container_pool=True,
        pool_max_size_per_image=2,
        pool_max_total_size=0,
        pool_default_cwd="/testbed",
        pool_create_timeout_sec=30,
        pool_health_check_timeout_sec=10,
        pool_prewarm_ratio=0.8,
        pool_prewarm_max_concurrency=4,
        pool_resource_stats_dir="unused",
        config_path="test",
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return exec_server.ContainerPool(
        config=config,
        create_container_fn=fake.create,
        destroy_container_fn=fake.destroy,
        health_check_fn=fake.healthy,
        active_count_fn=active_count_fn or (lambda: 0),
        prewarm_images=prewarm_images or ["img:a", "img:b", "img:c"],
    )


class TestContainerPool(unittest.TestCase):
    def test_warmup_creates_target_count(self):
        fake = FakeDockerOps()
        pool = make_pool(fake, pool_max_size_per_image=4, pool_prewarm_max_concurrency=5, pool_prewarm_ratio=0.8)

        pool.warmup(block=True, timeout=1.0)

        status = pool.status()
        self.assertEqual(status["prewarm_target_total"], 4)
        self.assertEqual(status["idle_containers"], 4)
        self.assertEqual(status["metrics"]["prewarmed_count"], 4)

    def test_acquire_uses_prewarmed_container(self):
        fake = FakeDockerOps()
        pool = make_pool(fake, prewarm_images=["img:a"], pool_max_size_per_image=4)
        pool.warmup(block=True, timeout=1.0)

        acquired = pool.acquire(image="img:a", cwd="/testbed", timeout=30)

        self.assertEqual(acquired["acquisition"], "prewarmed")
        self.assertTrue(acquired["pooled"])
        self.assertEqual(pool.status()["metrics"]["reused_count"], 1)

    def test_acquire_creates_when_warm_pool_empty(self):
        fake = FakeDockerOps()
        pool = make_pool(fake, prewarm_images=["img:a"], pool_max_size_per_image=1)
        pool.warmup(block=True, timeout=1.0)

        first = pool.acquire(image="img:a", cwd="/testbed", timeout=30)
        second = pool.acquire(image="img:a", cwd="/testbed", timeout=30)

        self.assertEqual(first["acquisition"], "prewarmed")
        self.assertEqual(second["acquisition"], "created")
        self.assertEqual(pool.status()["metrics"]["warm_miss_count"], 1)

    def test_release_always_destroys_and_refills(self):
        fake = FakeDockerOps()
        pool = make_pool(fake, prewarm_images=["img:a"], pool_max_size_per_image=4)
        pool.warmup(block=True, timeout=1.0)

        acquired = pool.acquire(image="img:a", cwd="/testbed", timeout=30)
        released = pool.release(
            container_id=acquired["container_id"],
            image="img:a",
            name=acquired["name"],
            cwd="/testbed",
        )
        pool.warmup(block=True, timeout=1.0)

        self.assertEqual(released["reason"], "strict_warm_one_shot")
        self.assertIn(acquired["container_id"], fake.destroyed)
        self.assertGreaterEqual(pool.status()["idle_containers"], 1)

    def test_unhealthy_warm_container_is_discarded(self):
        fake = FakeDockerOps()
        pool = make_pool(
            fake,
            prewarm_images=["img:a"],
            pool_max_size_per_image=1,
            pool_prewarm_max_concurrency=1,
            pool_prewarm_ratio=1.0,
        )
        pool.warmup(block=True, timeout=1.0)
        with pool._lock:  # pylint: disable=protected-access
            idle_id = next(iter(pool._idle_meta.keys()))  # pylint: disable=protected-access
        fake.unhealthy_ids.add(idle_id)

        acquired = pool.acquire(image="img:a", cwd="/testbed", timeout=30)

        self.assertEqual(acquired["acquisition"], "created")
        self.assertIn(idle_id, fake.destroyed)
        self.assertEqual(pool.status()["metrics"]["unhealthy_discard_count"], 1)

    def test_disabled_pool_falls_back_to_cold_create(self):
        fake = FakeDockerOps()
        pool = make_pool(fake, use_container_pool=False)

        acquired = pool.acquire(image="img:a", cwd="/testbed", timeout=30)
        released = pool.release(
            container_id=acquired["container_id"],
            image="img:a",
            name=acquired["name"],
            cwd="/testbed",
        )

        self.assertFalse(pool.enabled)
        self.assertEqual(acquired["acquisition"], "created")
        self.assertEqual(released["reason"], "pool_disabled")

    def test_round_robin_prewarm_distribution(self):
        fake = FakeDockerOps()
        pool = make_pool(
            fake,
            prewarm_images=["img:a", "img:b", "img:c"],
            pool_max_size_per_image=2,
            pool_prewarm_max_concurrency=6,
            pool_prewarm_ratio=1.0,
        )

        pool.warmup(block=True, timeout=1.0)
        status = pool.status()

        self.assertEqual(status["idle_by_image"].get("img:a", 0), 2)
        self.assertEqual(status["idle_by_image"].get("img:b", 0), 2)
        self.assertEqual(status["idle_by_image"].get("img:c", 0), 2)

    def test_concurrent_acquire_release_smoke(self):
        fake = FakeDockerOps()
        active_count = 0
        active_lock = threading.Lock()

        def get_active_count() -> int:
            with active_lock:
                return active_count

        pool = make_pool(
            fake,
            prewarm_images=["img:a", "img:b"],
            pool_max_size_per_image=4,
            pool_prewarm_max_concurrency=8,
            pool_prewarm_ratio=1.0,
            active_count_fn=get_active_count,
        )
        pool.warmup(block=True, timeout=1.0)
        errors: list[Exception] = []

        def worker(image: str, rounds: int) -> None:
            nonlocal active_count
            try:
                for _ in range(rounds):
                    acquired = pool.acquire(image=image, cwd="/testbed", timeout=30)
                    with active_lock:
                        active_count += 1
                    time.sleep(0.001)
                    with active_lock:
                        active_count -= 1
                    pool.release(
                        container_id=acquired["container_id"],
                        image=image,
                        name=acquired["name"],
                        cwd="/testbed",
                    )
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

        threads = [
            threading.Thread(target=worker, args=(f"img:{'a' if idx % 2 == 0 else 'b'}", 6))
            for idx in range(6)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        pool.warmup(block=True, timeout=1.0)

        self.assertEqual(errors, [])
        status = pool.status()
        self.assertLessEqual(status["idle_containers"], status["max_total_size"] or status["prewarm_target_total"])


if __name__ == "__main__":
    unittest.main()
