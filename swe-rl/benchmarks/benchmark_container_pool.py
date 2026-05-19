from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import swe_exec_server as exec_server  # noqa: E402


def benchmark_cold(image: str, cwd: str, iterations: int, timeout: int) -> dict[str, float]:
    times: list[float] = []
    for _ in range(iterations):
        started_at = time.perf_counter()
        created = exec_server._docker_create_container(image=image, cwd=cwd, timeout=timeout)
        exec_server._docker_destroy_container(created["container_id"], timeout=timeout)
        times.append(time.perf_counter() - started_at)
    return {
        "iterations": iterations,
        "avg_sec": statistics.mean(times),
        "median_sec": statistics.median(times),
        "min_sec": min(times),
        "max_sec": max(times),
    }


def benchmark_prewarmed(image: str, cwd: str, iterations: int, timeout: int, prewarm_count: int) -> dict[str, float]:
    config = exec_server.ExecServerConfig(
        use_container_pool=True,
        pool_max_size_per_image=prewarm_count,
        pool_max_total_size=prewarm_count,
        pool_default_cwd=cwd,
        pool_create_timeout_sec=timeout,
        pool_health_check_timeout_sec=10,
        pool_prewarm_ratio=1.0,
        pool_prewarm_max_concurrency=prewarm_count,
        pool_resource_stats_dir="benchmark",
        config_path="benchmark",
    )
    pool = exec_server.ContainerPool(
        config=config,
        create_container_fn=exec_server._docker_create_container,
        destroy_container_fn=exec_server._docker_destroy_container,
        health_check_fn=exec_server._docker_container_is_healthy,
        active_count_fn=lambda: 0,
        prewarm_images=[image],
    )
    pool.warmup(block=True, timeout=timeout)

    times: list[float] = []
    misses_before = pool.status()["metrics"]["warm_miss_count"]
    for _ in range(iterations):
        started_at = time.perf_counter()
        acquired = pool.acquire(image=image, cwd=cwd, timeout=timeout)
        pool.release(
            container_id=acquired["container_id"],
            image=image,
            name=acquired["name"],
            cwd=cwd,
        )
        pool.warmup(block=True, timeout=timeout)
        times.append(time.perf_counter() - started_at)

    status = pool.status()
    for container_id in list(pool._idle_meta.keys()):  # pylint: disable=protected-access
        exec_server._docker_destroy_container(container_id, timeout=timeout)

    return {
        "iterations": iterations,
        "avg_sec": statistics.mean(times),
        "median_sec": statistics.median(times),
        "min_sec": min(times),
        "max_sec": max(times),
        "warm_hits": status["metrics"]["reused_count"],
        "warm_misses": status["metrics"]["warm_miss_count"] - misses_before,
        "prewarmed_count": status["metrics"]["prewarmed_count"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark cold docker create/destroy versus startup prewarmed one-shot containers")
    parser.add_argument("--image", required=True, help="Docker image to benchmark")
    parser.add_argument("--cwd", default="/testbed", help="Working directory inside the container")
    parser.add_argument("--iterations", type=int, default=5, help="Benchmark rounds per mode")
    parser.add_argument("--timeout", type=int, default=1200, help="Docker create/destroy timeout in seconds")
    parser.add_argument("--prewarm-count", type=int, default=4, help="Number of strict-warm containers to precreate")
    args = parser.parse_args()

    cold = benchmark_cold(args.image, args.cwd, args.iterations, args.timeout)
    prewarmed = benchmark_prewarmed(args.image, args.cwd, args.iterations, args.timeout, args.prewarm_count)
    speedup = cold["avg_sec"] / prewarmed["avg_sec"] if prewarmed["avg_sec"] > 0 else 0.0

    print("Cold create/destroy:")
    print(cold)
    print("Startup prewarmed one-shot:")
    print(prewarmed)
    print({"speedup_x": speedup})


if __name__ == "__main__":
    main()
