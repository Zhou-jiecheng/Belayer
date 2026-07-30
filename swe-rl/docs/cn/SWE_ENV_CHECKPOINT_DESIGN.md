# SWE Env Checkpoint 设计文档

本文档描述在 `swe-rl` 现有远程 Docker 执行架构上，为 SWE agent rollout 增加环境 checkpoint、错误后 rerun、以及 checkpoint garbage collection 的设计。

目标读者：
- `swe_exec_server.py` 维护者
- `swe_env_pool_server.py` 维护者
- rollout / env client 维护者

本文档对应的现有架构可参考：
- [SWE_REMOTE_DOCKER.md](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/docs/cn/SWE_REMOTE_DOCKER.md)

## 1. 背景

当前 `swe-rl` 的 SWE rollout 采用如下链路：

1. RolloutManager 通过 [swe_env_client.py](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/swe_env_client.py) 调用 pool server。
2. [swe_env_pool_server.py](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/server/swe_env_pool_server.py) 管理 lease，并将请求转发到某个 Docker node。
3. [swe_exec_server.py](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/server/swe_exec_server.py) 在 Docker node 上执行 `docker run / exec / rm / diff / evaluate`。

当前问题：

- 一条 trajectory 中，agent 执行的命令会逐步污染容器状态。
- 如果容器发生 infra 级错误，例如：
  - OOM 被 kill
  - docker exec 超时且容器不可恢复
  - runtime error 导致容器状态不可信
  - node / daemon 层异常
- 当前保守恢复方式通常只能整条 trajectory 重跑，代价高。

同时，日志分析已经表明：

- LLM response 时间通常显著长于 `docker commit` 的空闲态开销。
- checkpoint 可以与下一步 LLM response overlap。
- 在高并发下，commit 开销具有系统态依赖，空闲时快、繁忙时慢。

因此需要在 rollout 中引入：

- 异步 checkpoint
- 错误后的快速 rerun
- checkpoint 生命周期管理和 GC

## 2. 目标

### 2.1 核心目标

1. 在不改变 agent 语义的前提下，为容器状态提供可恢复点。
2. 当容器进入“不可信状态”时，从最近一个 ready checkpoint 快速恢复，而不是整条 trajectory 重跑。
3. checkpoint 的 wall-clock 开销尽可能与下一次 LLM response overlap。
4. 支持 rollout 侧策略控制：
   - `never`
   - `always`
   - `adaptive-risk`
5. 支持 global 的 checkpoint GC。

### 2.2 非目标

1. 本阶段不做跨节点 checkpoint 迁移。
2. 本阶段不做 shared registry push/pull 式的 checkpoint 分发。
3. 本阶段不做 CRIU / process-level snapshot。
4. 本阶段不尝试恢复“某条命令执行到一半”的中间态，只在 step 边界 checkpoint。
5. 本阶段不保证 node 整体宕机后的 checkpoint 可用性。Phase 1 checkpoint 是 node-local 的。

## 3. 设计原则

### 3.1 保守恢复

只要环境侧发生 infra 级错误，就认为当前容器可能已污染，直接丢弃当前容器，从最近一个 ready checkpoint 重建。

### 3.2 lease 稳定，container 可替换

对 rollout 侧，`lease_id` 应尽量稳定。
错误恢复后，允许 `lease_id` 不变，但 `container_id` 更新。

### 3.3 checkpoint 异步化

checkpoint 创建默认异步，允许和下一步 LLM response overlap。
因此 checkpoint 需要显式区分：

- `pending`
- `ready`
- `failed`

只有 `ready` checkpoint 才能用于 rerun。

### 3.4 策略层和执行层解耦

- 执行层只负责 probe / create / restore / gc。
- 是否 probe、何时 create，由 rollout 侧策略决定。

### 3.5 不依赖未来信息

`adaptive-risk` 的在线实现不能依赖“剩余多少步”“未来多少秒”这类 trajectory 未来信息。
策略输入只能来自当前 rollout 已知状态和 server 可观测状态。

## 4. 高层方案

### 4.1 checkpoint 载体

Phase 1 采用 `docker commit` 生成 node-local image：

- 优点：
  - 实现简单
  - 与现有 `docker` CLI 架构兼容
  - 恢复路径清晰：`docker run <checkpoint-image>`
- 缺点：
  - image 是 node-local 的
  - 高并发下 daemon / 磁盘可能繁忙
  - 需要额外 GC

### 4.2 checkpoint 生命周期

一个 checkpoint 的生命周期：

1. rollout 决定“尝试 checkpoint”
2. 可选 probe 系统忙闲状态
3. exec server 创建异步 checkpoint op
4. op 完成后 checkpoint 进入 `ready`
5. 后续错误发生时，rerun 从最新 `ready checkpoint` 启动新容器
6. checkpoint 被显式删除、lease 关闭、TTL 到期、或被 GC 淘汰

### 4.3 恢复语义

错误恢复后的执行语义：

- 回退到最近一个 `ready checkpoint`
- 重新创建容器
- 继续执行 checkpoint 之后的 step
- checkpoint 之前的 LLM 上下文被视为保留
- checkpoint 之后到失败点之间的 env + LLM 成本需要重付

## 5. 术语

### 5.1 Lease

pool server 分配给 rollout 的逻辑容器句柄，由 `lease_id` 标识。

### 5.2 Generation

同一个 `lease_id` 下，container 被 rerun 替换后的代次。
每次 rerun 成功后，`generation += 1`。

### 5.3 Checkpoint

某个 lease 在某个 step 边界上创建的可恢复点，由 `checkpoint_id` 标识。

### 5.4 Checkpoint Op

异步 checkpoint 创建任务，由 `op_id` 标识。

### 5.5 Latest Ready Checkpoint

当前 lease 下最新一个 `status=ready` 的 checkpoint。

## 6. 组件和改动范围

### 6.1 需要改动的文件

- [swe_exec_server.py](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/server/swe_exec_server.py)
- [swe_env_pool_server.py](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/server/swe_env_pool_server.py)
- [swe_env_client.py](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/swe_env_client.py)
- [generate_with_swe_remote.py](/mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/robust_rl/OpenClaw-RL/swe-rl/generate_with_swe_remote.py)

### 6.2 建议新增测试

- `tests/test_swe_exec_checkpoint.py`
- `tests/test_swe_env_pool_checkpoint.py`
- `tests/test_swe_env_client_checkpoint.py`
- `tests/test_swe_checkpoint_gc.py`
- `tests/test_swe_checkpoint_rerun_flow.py`

## 7. 数据模型

### 7.1 Exec server 侧元数据

建议在 exec server 本地维护 checkpoint metadata store。

建议路径：

- 元数据目录：`/var/lib/swe-checkpoints/`
- 元数据文件：`/var/lib/swe-checkpoints/metadata.json`
- 可配置环境变量：`SWE_CHECKPOINT_DIR`

### 7.2 CheckpointRecord

建议字段：

```json
{
  "checkpoint_id": "swe-ckpt-...",
  "lease_id": "swe-lease-...",
  "generation": 3,
  "container_id": "docker-container-id",
  "node_url": "http://10.0.0.12:5000",
  "instance_id": "getmoto__moto-4860",
  "image": "docker.io/swebench/...",
  "checkpoint_image": "sweckpt:swe-lease-...-step-7",
  "parent_checkpoint_id": "swe-ckpt-prev",
  "step_idx": 7,
  "command_seq": 8,
  "policy": "adaptive-risk",
  "reason": "post_step_boundary",
  "status": "pending",
  "created_at": 1774432000.123,
  "ready_at": null,
  "failed_at": null,
  "last_used_at": null,
  "size_bytes": null,
  "labels": {
    "lease_id": "swe-lease-...",
    "instance_id": "getmoto__moto-4860"
  },
  "error": null
}
```

### 7.3 CheckpointOpRecord

```json
{
  "op_id": "swe-ckpt-op-...",
  "type": "create",
  "checkpoint_id": "swe-ckpt-...",
  "lease_id": "swe-lease-...",
  "container_id": "docker-container-id",
  "status": "running",
  "started_at": 1774432000.123,
  "finished_at": null,
  "error": null
}
```

### 7.4 Lease 扩展字段

pool server 内部 `Lease` 建议增加：

```python
generation: int = 0
latest_ready_checkpoint_id: str | None = None
latest_checkpoint_step_idx: int = -1
checkpoint_policy: str | None = None
last_rerun_at: float | None = None
rerun_count: int = 0
```

## 8. 错误分类和恢复触发

### 8.1 不触发 rerun 的情况

以下情况不属于 infra 级错误，不应自动 rerun：

- 命令返回码非 0，但容器健康
- 测试失败
- patch apply 失败
- agent 输出无效命令

这些属于任务语义的一部分，应交给 agent 继续推理。

### 8.2 触发 rerun 的情况

以下情况应触发“容器不可信”判断：

- `docker exec` 调用失败，且重试后仍失败
- `docker inspect` 显示容器已退出
- `OOMKilled=true`
- `stats` / 健康检查显示容器不健康
- pool server / exec server 返回明确的 infra error code

### 8.3 恢复策略

默认恢复策略：

1. 查询 `latest_ready_checkpoint`
2. 若存在，则从该 checkpoint rerun
3. 若不存在，则根据配置：
   - `fallback_to_base_image=true`：从原始 image 重新起容器并从 step 0 rerun
   - 否则报错并终止该 trajectory

## 9. 接口设计

接口分三层：

1. exec server：实际 docker 操作
2. pool server：lease 语义和多节点转发
3. client：rollout 侧调用封装

### 9.1 swe_exec_server 新增接口

#### 9.1.1 `POST /container/checkpoint/probe`

用途：
- 查询当前 checkpoint 系统是否繁忙
- 为 `adaptive-risk` 提供忙闲探测

请求：

```json
{
  "container_id": "cid",
  "lease_id": "swe-lease-...",
  "step_idx": 7,
  "timeout": 5
}
```

响应：

```json
{
  "ok": true,
  "busy": false,
  "probe_wait_sec": 0.0,
  "reason": "idle",
  "metrics": {
    "inflight_checkpoints": 0,
    "disk_busy_ratio": 0.12,
    "recent_commit_p95_sec": 3.4
  }
}
```

繁忙时：

```json
{
  "ok": true,
  "busy": true,
  "probe_wait_sec": 1.0,
  "reason": "daemon_or_disk_busy",
  "retry_after_sec": 1.0,
  "metrics": {
    "inflight_checkpoints": 4,
    "disk_busy_ratio": 0.93,
    "recent_commit_p95_sec": 14.7
  }
}
```

说明：

- Phase 1 实现里，`busy` 判定可以先基于简单 heuristic：
  - in-flight commit 数
  - 近窗口 commit latency
  - 磁盘 busy ratio / iowait
- `probe_wait_sec` 语义是“为了拿到 busy 结论实际等待了多久”。

#### 9.1.2 `POST /container/checkpoint/create`

用途：
- 异步创建 checkpoint

请求：

```json
{
  "container_id": "cid",
  "lease_id": "swe-lease-...",
  "generation": 2,
  "instance_id": "getmoto__moto-4860",
  "step_idx": 7,
  "command_seq": 8,
  "policy": "adaptive-risk",
  "reason": "post_step_boundary",
  "parent_checkpoint_id": "swe-ckpt-prev",
  "async": true,
  "labels": {
    "agent_run_id": "run-123"
  }
}
```

响应：

```json
{
  "ok": true,
  "checkpoint_id": "swe-ckpt-...",
  "op_id": "swe-ckpt-op-...",
  "status": "pending",
  "checkpoint_image": "sweckpt:swe-lease-...-g2-s7"
}
```

说明：

- 默认异步。
- `status=pending` 不代表可恢复。
- `ready` 之前若容器报错，仍只能回退到更早的 ready checkpoint。

#### 9.1.3 `POST /container/checkpoint/status`

请求：

```json
{
  "checkpoint_id": "swe-ckpt-..."
}
```

响应：

```json
{
  "ok": true,
  "checkpoint_id": "swe-ckpt-...",
  "status": "ready",
  "ready_at": 1774432003.221,
  "size_bytes": 314572800
}
```

#### 9.1.4 `POST /container/checkpoint/list`

请求：

```json
{
  "lease_id": "swe-lease-..."
}
```

响应：

```json
{
  "ok": true,
  "checkpoints": [
    {
      "checkpoint_id": "swe-ckpt-1",
      "step_idx": 3,
      "status": "ready",
      "created_at": 1.0,
      "ready_at": 4.0
    }
  ]
}
```

#### 9.1.5 `POST /container/rerun`

用途：
- 销毁旧容器
- 从 checkpoint image 重新启动新容器

请求：

```json
{
  "old_container_id": "cid-old",
  "checkpoint_id": "swe-ckpt-...",
  "cwd": "/testbed",
  "timeout": 120
}
```

响应：

```json
{
  "ok": true,
  "new_container_id": "cid-new",
  "checkpoint_id": "swe-ckpt-...",
  "checkpoint_image": "sweckpt:swe-lease-...-g2-s7"
}
```

说明：

- 该接口必须验证 checkpoint 是否 `ready`。
- 若 old container 已不存在，也应允许恢复，只要 checkpoint image 仍在。

#### 9.1.6 `POST /container/checkpoint/delete`

请求：

```json
{
  "checkpoint_id": "swe-ckpt-..."
}
```

响应：

```json
{
  "ok": true,
  "deleted": true,
  "reclaimed_bytes": 314572800
}
```

#### 9.1.7 `POST /container/checkpoint/gc`

用途：
- 本节点执行 checkpoint GC，注意GC触发时，仅保留每条trajectory最新的checkpoint版本

请求：

```json
{
  "scope": "lease",
  "lease_id": "swe-lease-...",
  "keep_latest": 1,
  "ttl_sec": 3600,
  "max_bytes": 21474836480,
  "dry_run": false
}
```

响应：

```json
{
  "ok": true,
  "deleted_count": 3,
  "deleted_checkpoint_ids": ["swe-ckpt-a", "swe-ckpt-b"],
  "reclaimed_bytes": 1073741824,
  "skipped_in_use": 1
}
```

### 9.2 swe_env_pool_server 新增接口

pool server 保持 lease 抽象，对 rollout 不暴露 container_id。

#### 9.2.1 `POST /checkpoint/probe`

请求：

```json
{
  "lease_id": "swe-lease-...",
  "step_idx": 7
}
```

行为：

- 根据 `lease_id` 找到 node / container
- 转发到 exec server `/container/checkpoint/probe`

#### 9.2.2 `POST /checkpoint/create`

请求：

```json
{
  "lease_id": "swe-lease-...",
  "step_idx": 7,
  "command_seq": 8,
  "policy": "adaptive-risk",
  "reason": "post_step_boundary",
  "async": true
}
```

行为：

- 转发到 exec server `/container/checkpoint/create`
- 成功后在 lease 元数据里登记 `latest_pending_checkpoint`

#### 9.2.3 `POST /checkpoint/status`

请求：

```json
{
  "lease_id": "swe-lease-...",
  "checkpoint_id": "swe-ckpt-..."
}
```

行为：

- 转发状态查询
- 若 checkpoint 进入 `ready`，同步更新 `lease.latest_ready_checkpoint_id`

#### 9.2.4 `POST /checkpoint/list`

请求：

```json
{
  "lease_id": "swe-lease-..."
}
```

#### 9.2.5 `POST /rerun`

这是 rollout 侧最关键的恢复接口。

请求：

```json
{
  "lease_id": "swe-lease-...",
  "checkpoint_id": "swe-ckpt-...",
  "reason": "container_oom",
  "fallback_to_latest_ready": true,
  "fallback_to_base_image": false,
  "timeout": 180
}
```

行为：

1. 找到 lease
2. 确定恢复点：
   - 指定 `checkpoint_id`
   - 否则用 `lease.latest_ready_checkpoint_id`
3. 调用 exec server `/container/rerun`
4. 更新 lease：
   - `container_id = new_container_id`
   - `generation += 1`
   - `rerun_count += 1`
5. 返回新的 lease 状态

响应：

```json
{
  "ok": true,
  "lease_id": "swe-lease-...",
  "generation": 3,
  "container_id": "cid-new",
  "checkpoint_id": "swe-ckpt-...",
  "rerun_count": 2
}
```

#### 9.2.6 `POST /checkpoint/delete`

删除某个 lease 下的某个 checkpoint。

#### 9.2.7 `POST /checkpoint/gc`

pool server 负责聚合多节点 GC。

请求：

```json
{
  "scope": "global",
  "lease_id": null,
  "keep_latest": 1,
  "ttl_sec": 3600,
  "max_bytes_per_node": 21474836480,
  "dry_run": false
}
```

行为：

- `scope=lease`：只清某个 lease
- `scope=node`：清某个节点
- `scope=global`：所有节点聚合执行

### 9.3 SweEnvClient 扩展接口

建议新增方法：

```python
async def checkpoint_probe(self, lease_id: str, step_idx: int) -> dict: ...
async def checkpoint_create(self, lease_id: str, step_idx: int, command_seq: int, policy: str, reason: str) -> dict: ...
async def checkpoint_status(self, lease_id: str, checkpoint_id: str) -> dict: ...
async def checkpoint_list(self, lease_id: str) -> dict: ...
async def rerun(self, lease_id: str, checkpoint_id: str | None = None, reason: str = "") -> dict: ...
async def checkpoint_delete(self, lease_id: str, checkpoint_id: str) -> dict: ...
async def checkpoint_gc(self, scope: str, **kwargs) -> dict: ...
```

## 10. Checkpoint 创建语义

### 10.1 创建时机

只允许在 step 边界创建 checkpoint。

推荐时序：

1. step `i` 的 `exec()` 返回
2. rollout 记录 step 结果
3. 策略判断是否发起 probe / checkpoint
4. 若发起 checkpoint，则异步创建
5. 同时开始 LLM 推理 step `i+1`

### 10.2 overlap 语义

checkpoint 和 probe 的 wall-clock 成本都允许与下一次 LLM response overlap。

因此系统需要记录两个时间口径：

- raw checkpoint time
- visible checkpoint time

raw 用于资源统计，visible 用于 wall-clock 估计。

### 10.3 ready 语义

`checkpoint_id` 只有在异步 op 完成后才变成 `ready`。

在此之前：

- 不能用于 rerun
- 不应更新 `latest_ready_checkpoint_id`

## 11. Adaptive-risk 在线策略设计

### 11.1 输入

仅使用当前可观测输入：

- `costs_since_checkpoint`
- `failure_prob`
- `idle_prob` 或 server 估计出的 idle likelihood
- `probe_wait_busy`

### 11.2 决策

在线判据：

```text
expected_immediate_replay_loss = costs_since_checkpoint * failure_prob
expected_probe_cost = P(busy) * probe_wait_busy

if expected_immediate_replay_loss >= adaptive_threshold * expected_probe_cost:
    进行 probe
else:
    跳过
```

### 11.3 probe 后动作

- 如果 busy：跳过 checkpoint
- 如果 idle：提交异步 checkpoint create

### 11.4 为什么不看未来

这个策略不依赖：

- trajectory 剩余步数
- 未来 LLM 分布
- 未来 env cost

因此符合在线实现约束。

## 12. Rerun 设计

### 12.1 设计目标

错误后，rollout 不需要重新 allocate 新 lease。
而是在原 lease 上“热替换 container”。

### 12.2 rerun 状态机

```text
active(generation=n)
  -> infra_error_detected
  -> select latest ready checkpoint
  -> rerun start
  -> old container destroy
  -> new container create from checkpoint image
  -> lease.container_id swap
  -> generation=n+1
  -> active
```

### 12.3 generation 作用

避免并发请求打到旧容器：

- 每次 exec/checkpoint/rerun 可以带 `expected_generation`
- 若 pool server 发现 generation 不匹配，则返回错误

这个机制可以在 Phase 2 加入，Phase 1 可以先不强制。

### 12.4 rerun 触发路径

建议 rollout 侧在以下情况调用 `rerun()`：

- `exec()` 明显 infra failure
- `stats()` 返回容器不健康
- pool server 返回 `retryable=true` 且错误码属于 infra

### 12.5 rerun 后上下文

rerun 后 rollout 需要：

- 保留 conversation 历史
- 保留 step_debug 历史
- 记录一次 `rerun_event`
- 从 checkpoint 之后继续执行

## 13. Garbage Collection 设计

### 13.1 GC 目标

避免 checkpoint image 无限增长：

- 占满磁盘
- 拖慢 docker daemon
- 拖慢后续 checkpoint / create / ps / inspect

### 13.2 GC 触发器

建议支持四类触发：

1. lease close 时的同步 GC
2. 周期性后台 GC
3. 超额阈值触发 GC
4. 管理员手动 GC

### 13.3 GC 策略

建议按优先级删除：

1. `failed` checkpoint
2. orphaned checkpoint
3. 非 latest-ready 的老 checkpoint
4. TTL 超期 checkpoint
5. 超出 `keep_latest_k` 的历史 checkpoint

### 13.4 GC 保护规则

以下 checkpoint 默认不能删：

- 正在 `pending create`
- 当前 lease 的 `latest_ready_checkpoint`
- 最近一次成功 rerun 使用过且仍可能回滚的 checkpoint

### 13.5 GC 指标

每次 GC 应返回：

- 删除数
- 回收字节数
- 跳过数
- 原因分类

## 14. 错误码设计

建议统一错误码，便于 client 做重试/降级：

| error_code | 含义 | retryable |
|-----------|------|-----------|
| `unknown_lease_id` | lease 不存在 | 否 |
| `checkpoint_not_found` | checkpoint 不存在 | 否 |
| `checkpoint_not_ready` | checkpoint 还未 ready | 是 |
| `checkpoint_op_failed` | checkpoint 创建失败 | 视情况 |
| `checkpoint_busy` | 当前系统繁忙，不建议 commit | 是 |
| `checkpoint_probe_failed` | probe 失败 | 是 |
| `rerun_failed` | rerun 失败 | 是 |
| `container_unhealthy` | 容器健康检查失败 | 是 |
| `lease_generation_mismatch` | 代次不一致 | 是 |
| `checkpoint_gc_failed` | GC 执行失败 | 是 |
| `node_checkpoint_unavailable` | checkpoint 所在节点不可用 | 否 |

## 15. 观测与指标

### 15.1 Exec server 指标

建议新增：

- `checkpoint_create_count`
- `checkpoint_create_fail_count`
- `checkpoint_create_latency_sec`
- `checkpoint_ready_count`
- `checkpoint_probe_count`
- `checkpoint_probe_busy_count`
- `checkpoint_gc_delete_count`
- `checkpoint_gc_reclaimed_bytes`
- `checkpoint_disk_usage_bytes`
- `checkpoint_inflight_count`
- `rerun_count`
- `rerun_latency_sec`

### 15.2 Pool server 指标

- `lease_rerun_count`
- `lease_checkpoint_ready_count`
- `lease_checkpoint_pending_count`
- `rerun_fail_count`
- `recover_from_latest_ready_count`
- `recover_fallback_to_base_count`

### 15.3 Rollout 侧日志

每个 step 建议记录：

```json
{
  "step_idx": 7,
  "checkpoint_probe": {
    "attempted": true,
    "busy": false,
    "probe_wait_sec": 0.0
  },
  "checkpoint_create": {
    "attempted": true,
    "checkpoint_id": "swe-ckpt-...",
    "status": "pending"
  },
  "rerun": null
}
```

## 16. 配置项建议

### 16.1 Exec server

建议新增环境变量：

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SWE_ENABLE_CHECKPOINT` | `0` | 总开关 |
| `SWE_CHECKPOINT_DIR` | `/var/lib/swe-checkpoints` | checkpoint 元数据目录 |
| `SWE_CHECKPOINT_MAX_INFLIGHT` | `1` | 同时进行的 checkpoint 数 |
| `SWE_CHECKPOINT_GC_INTERVAL_SEC` | `300` | 周期 GC 间隔 |
| `SWE_CHECKPOINT_KEEP_LATEST` | `1` | 每 lease 保留的 ready checkpoint 数 |
| `SWE_CHECKPOINT_TTL_SEC` | `86400` | checkpoint TTL |
| `SWE_CHECKPOINT_MAX_BYTES` | `21474836480` | 节点 checkpoint 预算 |

### 16.2 Rollout

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `SWE_CHECKPOINT_POLICY` | `never` | `never/always/adaptive-risk` |
| `SWE_CHECKPOINT_FAILURE_PROB_ESTIMATE` | `0.01` | adaptive-risk 用 |
| `SWE_CHECKPOINT_PROBE_BUSY_WAIT_SEC` | `1.0` | adaptive-risk 用 |
| `SWE_CHECKPOINT_IDLE_PROB_ESTIMATE` | `0.7` | adaptive-risk 用 |
| `SWE_RERUN_ON_INFRA_ERROR` | `1` | 是否启用自动 rerun |
| `SWE_RERUN_FALLBACK_TO_BASE_IMAGE` | `0` | 无 checkpoint 时是否从 base image 重新开始 |

## 17. 向后兼容

要求：

1. 未开启 `SWE_ENABLE_CHECKPOINT` 时，现有接口和行为不变。
2. `SweEnvClient` 新增方法不能破坏现有调用。
3. 旧 rollout 代码无需修改也能继续运行。

## 18. 安全与资源控制

### 18.1 并发控制

必须限制并发 checkpoint 数，避免把 daemon 打死。

建议：

- exec server 内部一个 checkpoint worker queue
- 可配置 max inflight

### 18.2 磁盘配额

checkpoint image 会快速膨胀。
必须配合：

- 全局字节预算
- per-lease 保留上限
- 强制 GC

### 18.3 节点隔离

Phase 1 checkpoint 仅对生成它的 node 有效。

如果 pool server 把 rerun 调度到别的 node，会失败。
因此：

- rerun 必须优先固定到原 node
- 原 node 不可用时只能 fallback 或失败

## 19. 分阶段实现计划

### Phase 0: 元数据和骨架

目标：

- 在 exec server 引入 checkpoint metadata store
- 在 pool server / client 中引入空接口骨架

改动：

- `swe_exec_server.py`
  - 增加 metadata store
  - 增加 checkpoint worker 基础设施
- `swe_env_pool_server.py`
  - 扩展 `Lease`
- `swe_env_client.py`
  - 新增 checkpoint/rerun client 方法

交付物：

- 新接口返回 mock / no-op 结果
- 单元测试通过

### Phase 1: 最小可用 checkpoint + rerun

目标：

- 实现 `probe/create/status/list/rerun/delete`
- rollout 手动调用 rerun 路径可用

改动：

- exec server 真正执行 `docker commit`
- rerun 从 checkpoint image 重建容器
- pool server 维持 lease 不变、container 置换

验收：

- 手工注入容器 kill，可从 latest ready checkpoint 恢复

### Phase 2: rollout 集成 adaptive-risk

目标：

- 在 rollout step 边界接入策略
- 支持 `never / always / adaptive-risk`

改动：

- `generate_with_swe_remote.py`
  - step 完成后调用 probe/create

验收：

- 训练日志里能看到 checkpoint/rerun 事件
- 人工注入错误时 trajectory 可以续跑

### Phase 3: GC 和预算控制

目标：

- 加入后台 GC
- 加入预算和 TTL

改动：

- exec server 后台 GC 线程
- pool server global GC 聚合接口

验收：

- 压测下 checkpoint 磁盘空间稳定

### Phase 4: 观测与压测

目标：

- 指标完善
- 高并发下验证 daemon / disk 不失控

验收：

- 在 64/128 worker 压测下，checkpoint/rerun 行为稳定

## 20. 建议的实现顺序

推荐严格按下面顺序施工：

1. exec server metadata store
2. exec server `/container/checkpoint/create/status/list/delete`
3. exec server `/container/rerun`
4. pool server 转发接口
5. client 封装
6. rollout 手工触发 checkpoint/rerun
7. adaptive-risk 接入
8. GC
9. metrics 和 fault injection tests

## 21. 关键开放问题

下面这些问题建议在开始 coding 前确认：

1. `docker commit` 是否需要串行化到单 worker，还是允许小并发？
2. `probe` 的 busy 判据 Phase 1 是用简单阈值，还是直接接系统指标？
3. `rerun` 是否必须保留原 `lease_id`，还是允许返回新 `lease_id`？
4. 当 node 不可用时，是否允许 fallback 到 base image 重新开始？
5. `latest_ready_checkpoint` 是保留 1 个还是保留最近 2 个更稳妥？

## 22. 本设计的结论

本设计推荐的 Phase 1 路径是：

1. 基于现有 `docker commit` 做 node-local checkpoint
2. checkpoint 创建异步化
3. pool server 保持 lease 稳定，rerun 时只替换 container
4. rollout 侧只在 step 边界触发 checkpoint
5. `adaptive-risk` 只基于当前未保护代价和 server 忙闲 probe 做在线决策
6. 必须同时上线 checkpoint GC

这条路径和 `swe-rl` 当前的 `exec server -> pool server -> client -> rollout` 分层最兼容，工程风险最低，也足够支持后续策略迭代。
