# SWE Runtime Memory Recovery Plan

## Background

`/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/server/swe_exec_server.py` already supports filesystem checkpointing by:

- creating a checkpoint image with `docker commit`
- recording checkpoint lineage in `CheckpointManager`
- starting a new container from the checkpoint image in `/container/rerun`

`/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/server/swe_env_pool_server.py` already:

- tracks the latest ready checkpoint per lease
- exposes `checkpoint_create`, `checkpoint_status`, `checkpoint_gc`
- updates lease state after rerun from a checkpoint image

What is still missing is a contract for recovering SWE runtime state that is not preserved by `docker commit`, especially:

- current shell/session context
- activated Python environment
- task progress metadata needed to safely resume after rerun

This document defines a phased plan for adding that runtime-memory recovery path without changing the existing checkpoint image model.

## Goal

After a checkpoint rerun, the new container should resume from the latest safe SWE execution state with:

- filesystem state restored by the checkpoint image
- runtime state reconstructed from explicit metadata
- correctness validated before the agent continues

The design target is task-internal recovery for SWE-style workloads. It does not try to solve:

- task-to-task contamination
- arbitrary in-memory object snapshots
- CRIU-based process memory restore

## Assumptions

- Filesystem state is the primary source of truth and is already captured by `docker commit`.
- We only support restart-based recovery, not same-process restore.
- Critical runtime state must be made explicit and serializable.
- Recovery must work even if the old container is destroyed immediately after rerun.

## SWE Runtime State Model

The recovery design should treat SWE runtime state as four layers.

### 1. Filesystem-backed state

Already handled by checkpoint image:

- repository working tree
- generated files
- downloaded artifacts
- caches written to disk
- local databases or sqlite files

### 2. Shell/session state

Must be reconstructed explicitly:

- `cwd`
- selected shell
- whitelisted environment variables
- activated venv/conda state
- sourced setup scripts
- shell options that affect later commands

### 3. Service state

For SWE, this is out of scope for the first implementation.

If service restore is ever needed later, it should be restarted declaratively rather than restored in-memory:

- local web servers
- test servers
- language servers
- helper daemons started by the task

### 4. Resume control state

Must be persisted to know where to continue:

- lease id / checkpoint id / generation
- step index
- command sequence
- last successfully completed action
- recovery policy / reason

## Proposed Runtime Checkpoint Contract

At each safe checkpoint, in addition to creating the checkpoint image, write a task-scoped metadata bundle inside the container.

Suggested location:

`/tmp/swe-runtime-checkpoints/<checkpoint_id>/`

Suggested files:

- `runtime.json`
- optional `validation.json` if validation grows beyond a few built-in checks

This metadata must be included in the committed image so the rerun container can read it immediately after boot.

## `runtime.json` Minimum Schema

The first version should be intentionally minimal and only capture restart-safe SWE state that is required for correctness.

```json
{
  "schema_version": 1,
  "checkpoint_id": "ckpt-123",
  "parent_checkpoint_id": "ckpt-122",
  "lease_id": "swe-lease-...",
  "instance_id": "django__django-12345",
  "step_idx": 12,
  "command_seq": 18,
  "workspace": {
    "repo_path": "/testbed",
    "cwd": "/testbed/tests",
    "user": "root",
    "home": "/root",
    "shell": "/bin/bash"
  },
  "env": {
    "PATH": "...",
    "PYTHONPATH": "...",
    "VIRTUAL_ENV": "...",
    "CONDA_PREFIX": ""
  },
  "python_runtime": {
    "python_executable": "/opt/venv/bin/python",
    "venv_activate": "/opt/venv/bin/activate",
    "conda_env": ""
  },
  "progress": {
    "phase": "editing",
    "last_successful_action_id": "cmd-18"
  }
}
```

## Minimal Restore Validation

Validation is required so restart does not silently continue in the wrong state. For the first version, validation should stay lightweight and built into the rerun restore helper instead of introducing a large external schema.

Suggested checks:

- `cwd` exists and is enterable
- `python_executable` exists and runs
- `VIRTUAL_ENV` or `CONDA_PREFIX` matches the runtime metadata when present
- repo root exists
- optional: `git rev-parse --show-toplevel` succeeds under the restored `cwd`

## Safe Checkpoint Boundaries

Runtime metadata must only be emitted at safe boundaries.

Allowed:

- after a command finishes
- after filesystem writes are flushed
- after required services are in ready state
- after step bookkeeping is updated

Not allowed:

- while a shell command is still running
- while a background service is being spawned
- while the agent is between command dispatch and command completion
- while `cwd` / env activation is mid-transition

## Restore Flow

The restore path should remain aligned with the current rerun flow.

### Stage 1. Current behavior remains

1. `checkpoint/create` creates a checkpoint image via `docker commit`
2. `/container/rerun` starts a new container from `checkpoint_image`
3. old container is destroyed

### Stage 2. Add runtime rehydrate hook

After `new_container_id` is created and before returning success:

1. locate runtime metadata inside the new container
2. restore `cwd`
3. restore whitelisted env activation
4. run minimal restore validation
5. only then return rerun success

If any step fails:

- mark rerun as failed
- destroy the new container
- preserve the checkpoint image
- return a retryable restore error

## Integration Points in Current Code

### `swe_exec_server.py`

Add runtime checkpoint responsibilities in these places:

- checkpoint creation request path:
  before or immediately around `_checkpoint_create_worker`, emit runtime metadata for the target checkpoint id
- rerun path:
  inside `/container/rerun`, after `_docker_create_container(...)` and before success response, call a new restore helper

Suggested helpers:

- `_runtime_checkpoint_dir(checkpoint_id: str) -> str`
- `_capture_runtime_state(container_id: str, checkpoint_id: str, cwd: str, ...)`
- `_restore_runtime_state(container_id: str, checkpoint_id: str, cwd: str, ...)`
- `_validate_runtime_restore(container_id: str, checkpoint_id: str, ...)`

### `swe_env_pool_server.py`

No protocol redesign is needed first. The pool can keep its current role. It only needs to surface restore failures returned by exec server and keep lease state unchanged when rerun fails.

## Capture Strategy

The first implementation should avoid trying to infer arbitrary shell memory.

Instead, define a strict contract:

- only whitelisted env vars are recoverable
- no background service restore in the first version
- no generic shell replay in the first version
- if a task depends on hidden REPL-only memory, that state is unsupported

This keeps the system deterministic and testable.

## Recovery Semantics

After restore, the system guarantees:

- same filesystem snapshot
- same declared `cwd`
- same declared Python environment activation
- same validated repo entry conditions

It does not guarantee:

- same PID values
- same in-memory Python objects
- same interactive shell local variables
- same partially executing subprocess state
- same background service liveness unless service restore is added later

## Failure Handling Policy

Capture failures:

- checkpoint image can still be created
- checkpoint status should be marked `failed` if minimal required runtime metadata is missing or invalid

Restore failures:

- rerun returns error code such as `runtime_restore_failed`
- new container is destroyed
- old checkpoint remains available
- caller may retry rerun from the same checkpoint or an earlier one

Validation failures:

- treat as restore failure, not agent failure
- do not continue rollout on an unvalidated container

## Testing Plan

### Unit tests

Add tests for:

- runtime metadata schema generation
- restore helper rebuilding `cwd` and env whitelist
- validation failure propagation
- rerun failure leaves lease untouched

### Integration tests

Add container-level tests for:

- restore of `cwd`
- restore of `VIRTUAL_ENV`
- restore of whitelisted env vars
- validation of repo entry conditions

### Regression tests

Ensure existing checkpoint/rerun tests still pass:

- checkpoint lineage
- lease image update
- checkpoint GC
- checkpoint busy handling

## Rollout Plan

### Phase 0. Documentation and schema

- define the minimal `runtime.json` schema
- define env var whitelist
- define supported restore guarantees

### Phase 1. Minimal capture/restore

- capture `cwd`
- capture Python activation info
- capture env whitelist
- capture step/progress metadata
- restore and validate them during rerun

### Phase 2. Resume metadata and replay

- optionally add explicit `pre_resume_cmds` if later SWE traces show they are needed
- resume from the last safe agent step

### Phase 3. Hardening

- metrics for capture/restore latency
- restore failure reason taxonomy
- checkpoint compatibility versioning
- richer tests on real SWE workloads

## Open Questions

- Which env vars are truly required for SWE correctness and should be whitelisted?
- Should runtime metadata live only inside the checkpoint image, or also be mirrored on the host for debugging?
- Should restore run synchronously in `/container/rerun`, or return a pending state and let the pool poll?
- Do we need a strict opt-in policy so only explicitly declared restoreable state is used?

## Recommendation

Start with a narrow, restart-safe design:

- filesystem rollback stays on `docker commit`
- runtime recovery starts with `cwd + env whitelist + python activation + step metadata`
- rerun success is gated on explicit validation

This keeps the existing checkpoint architecture intact while adding the minimum runtime-memory recovery contract needed for SWE correctness.


  1. Checkpoint 前采集 runtime state

  - env whitelist 探测: swe_exec_server.py:132
  - runtime.json payload 构造: swe_exec_server.py:171
  - worker 在 docker commit 前把探测到的 env 填回 payload: swe_exec_server.py:1566
  - checkpoint create endpoint 组装 payload 并起 worker: swe_exec_server.py:2013

  这条链路是这次实现的主体，改动量大约在 100 行级别，主要都集中在 swe_exec_server.py。

  2. Rerun 后恢复并校验 runtime state

  - restore 校验: swe_exec_server.py:236
  - rerun 后读取 runtime.json 并校验: swe_exec_server.py:2212

  这部分改动不多，大约几十行，逻辑很直接：load -> validate -> 挂回 new container metadata。

  3. 协议透传

  - exec 正常路径，已经回到无额外 cwd 跟踪版本: swe_exec_server.py:1834
  - pool exec 也不再更新 lease.cwd: swe_env_pool_server.py:283
  - pool checkpoint_create(..., env=...): swe_env_pool_server.py:342
  - client checkpoint_create(..., env=...): swe_env_client.py:178

  这部分是小改动，主要是把最小 runtime env 透传能力保留下来，方便后续真实调用点按需使用。