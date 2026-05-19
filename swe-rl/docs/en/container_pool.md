# Container Pooling

The exec server now supports a strict warm-container pool:

- containers are precreated at startup
- each warmed container is used at most once
- release always destroys the used container
- the pool refills asynchronously in the background
- prewarm creation is serialized through a single worker thread

This avoids reusing mutated containers while still moving most `docker run`
latency off the request critical path.

## Toggle

The exec server reads [`container_pool_config.json`](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/server/container_pool_config.json)
at startup.

Example:

```json
{
  "use_container_pool": true,
  "pool_max_size_per_image": 4,
  "pool_max_total_size": 0,
  "pool_default_cwd": "/testbed",
  "pool_create_timeout_sec": 1200,
  "pool_health_check_timeout_sec": 10,
  "pool_prewarm_ratio": 0.8,
  "pool_prewarm_max_concurrency": 128,
  "pool_resource_stats_dir": "/path/to/swe-rl/resource_stats"
}
```

Environment overrides are also supported:

```bash
export USE_CONTAINER_POOL=1
export CONTAINER_POOL_PREWARM_MAX_CONCURRENCY=128
export CONTAINER_POOL_PREWARM_RATIO=0.8
export CONTAINER_POOL_RESOURCE_STATS_DIR=/path/to/swe-rl/resource_stats
```

## Prewarm Source

The prewarm image list is discovered from the JSON files under
`pool_resource_stats_dir`. Each file should contain an `image` field. The exec
server deduplicates those image names and fills the warm pool in round-robin
order.

## Usage Example

Start the exec server:

```bash
python3 swe-rl/server/swe_exec_server.py --host 0.0.0.0 --port 5000
```

Create and destroy a container twice. The first requests should come from the
prewarmed pool if the target image has already been warmed:

```bash
curl -s http://127.0.0.1:5000/container/create \
  -H 'Content-Type: application/json' \
  -d '{"image":"your-image:latest","cwd":"/testbed","timeout":1200}'

curl -s http://127.0.0.1:5000/container/destroy \
  -H 'Content-Type: application/json' \
  -d '{"container_id":"<container_id_from_previous_step>"}'
```

Inspect pool status:

```bash
curl -s http://127.0.0.1:5000/status | python3 -m json.tool
```

Useful fields:

- `container_pool.idle_containers`
- `container_pool.pending_creates`
- `container_pool.prewarm_target_total`
- `container_pool.metrics.prewarmed_count`
- `container_pool.metrics.reused_count`
- `container_pool.metrics.warm_miss_count`

## Benchmark

Run the benchmark script directly on an exec server node:

```bash
python3 swe-rl/benchmarks/benchmark_container_pool.py \
  --image your-image:latest \
  --cwd /testbed \
  --iterations 10 \
  --prewarm-count 4
```

It prints the cold `create + destroy` timing, the startup-prewarmed timing, and
the overall speedup ratio.
