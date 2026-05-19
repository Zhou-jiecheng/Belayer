# Fast Restart

`slime` now supports shadow-worker-based fast restart for local SGLang rollout engines.

## Behavior

- The normal rollout engine still starts first and registers with the router.
- When `--sglang-enable-fast-restart` is set and fault tolerance is enabled, `slime` also starts a shadow worker in `skeleton_worker` mode.
- If health monitoring detects that the active worker is gone, `slime` first asks every node in the affected engine group to promote its shadow worker.
- If that handover succeeds, the existing Ray actors stay alive and routing switches to the promoted worker.
- If handover is unavailable or fails, `slime` falls back to the existing full engine restart path.

## Required resources

Fast restart depends on the same external shared-memory services used in the legacy setup:

- checkpoint-engine parameter servers for `load_format=weight_deamon`
- KV cache socket paths exposed through `SGLANG_KV_CACHE_SOCKET_PATH`

You can configure the mapping with:

- `--sglang-shadow-worker-kv-cache-socket-path`
- `--sglang-shadow-worker-weight-server-base-port`
- `--sglang-shadow-worker-min-gpu-id`

If these resources are missing, `slime` logs the reason and keeps using the normal restart behavior.
