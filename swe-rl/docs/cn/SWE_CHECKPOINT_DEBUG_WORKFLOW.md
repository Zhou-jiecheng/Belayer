# SWE Checkpoint Debug Workflow

本文档整理当前 `checkpoint / rerun / gc` 基础调试流程，以及带随机故障注入的 `fault experiment` 流程。文档只描述当前仓库里已经存在的脚本和输出，不引入新的实验框架。

相关脚本：

- `swe-rl/scripts/run_swe_checkpoint_replay_debug.sh`
- `swe-rl/tools/replay_swe_traj_checkpoint.py`
- `swe-rl/scripts/run_swe_checkpoint_policy_fault_experiment.sh`
- `swe-rl/tools/replay_swe_checkpoint_fault_experiment.py`

## 1. 总体目标

当前 checkpoint 调试分两层：

1. 基础功能调试
   验证 `checkpoint/create -> status -> rerun -> gc -> close` 是否正常。
2. 故障恢复实验
   在并发 replay 中随机注入环境错误，比较不同 checkpoint 策略的收益和额外开销。

建议顺序：

1. 先跑基础 replay debug，确认功能链路是通的。
2. 再跑并发 replay，确认 `docker error`、`checkpoint_busy`、`gc` 行为。
3. 最后再跑 fault experiment，比较策略。

## 2. 当前实现语义

当前 server 端的 checkpoint 不是 CRIU 式 container checkpoint，而是 image-based checkpoint：

- `checkpoint/create`
  底层调用 `docker commit <container_id> <checkpoint_image>`
- `rerun`
  底层用 checkpoint image 新起一个容器
- `gc`
  底层调用 `docker image rm -f <checkpoint_image>`

因此当前 debug 的重点是 Docker daemon 行为，而不是内核级 checkpoint/restore。

## 3. 环境准备

默认 Python：

```bash
/mnt/shared-storage-user/ailab-sys/zhoujiecheng/miniconda/envs/osworld/bin/python
```

默认 trajectory 根目录：

```bash
export/swe_rollouts_profile_20260325_093408
```

默认本地 pool server：

- host: `127.0.0.1`
- port: `18090`

当前脚本里的默认远端 exec server：

```bash
http://100.101.233.34:5000
```

两个 launcher 都会先起本地 `swe_env_pool_server`，再用它去访问远端 `swe_exec_server`。

## 4. 基础 Replay Debug Workflow

### 4.1 目标

用于验证：

- trajectory 命令是否能重放
- checkpoint 是否能创建并进入 `ready`
- rerun 是否能从 ready checkpoint 恢复
- gc 是否能删除 lease 下的 checkpoint
- close 时 lease 是否能正常回收

### 4.2 启动方式

脚本入口：

```bash
swe-rl/scripts/run_swe_checkpoint_replay_debug.sh
```

默认行为：

- 起本地 `swe_env_pool_server`
- 批量读取 `TRAJ_ROOT` 下的 `traj.json`
- 默认并发 `REPLAY_MAX_CONCURRENCY=32`
- 使用 `step_debug[*].action` 重放环境命令
- 如果设置了 `SIMULATE_LLM_DELAY=1`，会在每个 step 前 sleep `step_debug[*].llm_elapsed`

### 4.3 常用模式

只重放，不做 checkpoint：

```bash
REPLAY_LIMIT=8 \
REPLAY_MAX_CONCURRENCY=8 \
SIMULATE_LLM_DELAY=1 \
swe-rl/scripts/run_swe_checkpoint_replay_debug.sh
```

在指定 step 后做 checkpoint，并等待 ready：

```bash
CHECKPOINT_AFTER_STEP=1 \
WAIT_CHECKPOINT_READY=1 \
REPLAY_LIMIT=8 \
REPLAY_MAX_CONCURRENCY=8 \
SIMULATE_LLM_DELAY=1 \
swe-rl/scripts/run_swe_checkpoint_replay_debug.sh
```

加上 rerun：

```bash
CHECKPOINT_AFTER_STEP=1 \
WAIT_CHECKPOINT_READY=1 \
RERUN_AFTER_STEP=1 \
REPLAY_LIMIT=8 \
REPLAY_MAX_CONCURRENCY=8 \
SIMULATE_LLM_DELAY=1 \
swe-rl/scripts/run_swe_checkpoint_replay_debug.sh
```

加上 gc：

```bash
CHECKPOINT_AFTER_STEP=1 \
WAIT_CHECKPOINT_READY=1 \
GC_KEEP_LATEST=0 \
REPLAY_LIMIT=8 \
REPLAY_MAX_CONCURRENCY=8 \
SIMULATE_LLM_DELAY=1 \
swe-rl/scripts/run_swe_checkpoint_replay_debug.sh
```

### 4.4 关键参数

`replay_swe_traj_checkpoint.py` 支持的核心参数：

- `--checkpoint-after-step`
  在指定 step 后创建 checkpoint，可重复传入
- `--wait-checkpoint-ready`
  创建后轮询 `/checkpoint/status` 直到 ready
- `--rerun-after-step`
  在指定 step 之后触发一次 rerun
- `--gc-keep-latest`
  在 replay 末尾调用 lease-scoped GC
- `--gc-dry-run`
  只计算 GC，不实际删 image
- `--keep-lease-open`
  不调用 `close`
- `--simulate-llm-delay`
  用 trajectory 中的 `llm_elapsed` 模拟 LLM bubble

### 4.5 主要输出

默认输出目录：

```bash
export/checkpoint_replay_debug_<timestamp>/
```

关键文件：

- `summary.json`
  整批 replay 的汇总结果
- `per_traj/*.json`
  每条 trajectory 的独立报告
- `swe_env_pool_server.log`
  pool server 侧日志
- `replay.log`
  launcher 标准输出

### 4.6 每条 trajectory 报告重点字段

- `steps`
  每个环境命令的回放情况
- `checkpoint_events`
  checkpoint create / status / skip 信息
- `rerun_event`
  rerun 结果
- `checkpoint_list_before_gc`
- `gc_result`
- `checkpoint_list_after_gc`
- `closed`
- `wall_time_sec`

### 4.7 Debug 关注点

基础 replay debug 主要看这些现象：

- `checkpoint_busy`
  当前 exec server 的 inflight checkpoint admission 满了
- `paused, unpause the container before exec`
  通常是 `docker commit` 与前台 `docker exec` 干扰
- `exec attach failed`
- `broken pipe`
- `shim disconnected`
  这些通常是 Docker daemon / containerd 控制面冲突

排查顺序建议：

1. 先看 `summary.json` 的 `ok_count / failed_count`
2. 再看 `per_traj/*.json` 中失败样本
3. 再对照 `swe_env_pool_server.log`
4. 最后看远端 `swe_exec_server` 日志

## 5. Fault Experiment Workflow

### 5.1 目标

用于验证：

- 并发 replay 中随机环境错误注入后的恢复行为
- 不同 checkpoint 策略在真实 replay 上的收益和成本
- 从最近 ready checkpoint rerun 与从 base restart 的差异

### 5.2 策略

当前脚本支持：

- `oracle-no-fault-no-checkpoint`
  无错误、无 checkpoint 的下界 baseline
- `never`
  从不做 checkpoint，失败时只能从头开始
- `always`
  每个合适位置都尝试做 checkpoint
- `adaptive-risk`
  在 LLM bubble 内基于条件 tail 和期望收益判断是否发起 checkpoint

### 5.3 启动方式

脚本入口：

```bash
swe-rl/scripts/run_swe_checkpoint_policy_fault_experiment.sh
```

默认行为：

- 起本地 `swe_env_pool_server`
- 读取 `TRAJ_ROOT` 下 trajectory
- 默认 `limit=32`
- 默认并发 `32`
- 默认随机选择 `2` 条 trajectory 注入环境错误
- 输出每个策略一套单独目录

### 5.4 常用命令

默认四策略实验：

```bash
swe-rl/scripts/run_swe_checkpoint_policy_fault_experiment.sh
```

固定小规模实验：

```bash
EXPERIMENT_LIMIT=8 \
EXPERIMENT_MAX_CONCURRENCY=8 \
EXPERIMENT_INJECTION_COUNT=2 \
EXPERIMENT_INJECTION_SEED=20260407 \
SIMULATE_LLM_DELAY=1 \
swe-rl/scripts/run_swe_checkpoint_policy_fault_experiment.sh
```

只跑部分策略：

```bash
POLICIES="never always adaptive-risk" \
EXPERIMENT_LIMIT=16 \
EXPERIMENT_MAX_CONCURRENCY=16 \
swe-rl/scripts/run_swe_checkpoint_policy_fault_experiment.sh
```

### 5.5 输出目录

默认输出目录：

```bash
export/checkpoint_policy_fault_experiment_<timestamp>/
```

关键文件：

- `injection_plan.json`
  本轮随机注入计划
- `<policy>/summary.json`
  单个策略的完整结果
- `<policy>/per_traj/*.json`
  每条 trajectory 报告
- `summary_all_policies.json`
  多策略总汇总
- `swe_env_pool_server.log`
  pool server 日志
- `experiment.log`
  launcher 标准输出

### 5.6 单策略 summary 的核心指标

- `trajectory_count`
- `ok_count`
- `failed_count`
- `batch_wall_time_sec`
  整批实验 wall time
- `mean_traj_wall_time_sec`
- `p50_traj_wall_time_sec`
- `p95_traj_wall_time_sec`
- `checkpoint_attempts`
- `checkpoint_created`
- `checkpoint_busy_skips`
- `probe_count`
- `probe_busy_skips`
- `rerun_from_checkpoint`
- `rerun_from_base`
- `gc_deleted_count`
- `gc_reclaimed_bytes`

### 5.7 每条 trajectory 的恢复语义

故障注入之后：

1. 如果在注入点之前已经有 `ready checkpoint`
   就执行 `rerun`
2. 如果没有 ready checkpoint
   就关闭当前 lease，重新 allocate，从头 replay

因此单条报告里重点看：

- `failure_events`
- `rerun_events`
- `latest_ready_checkpoint_step`
- `metrics.rerun_from_checkpoint`
- `metrics.rerun_from_base`

## 6. Adaptive-risk 当前语义

当前 `adaptive-risk` 的设计重点：

- 在 LLM bubble 内做决策，不是在 step 末尾统一决策
- `C` 统计最近 ready checkpoint 之后已完成的 LLM response 和 environment action 重生成成本
- `p_fail` 使用配置的 step-level failure probability
- checkpoint duration `c` 使用当前配置/预测的 checkpoint 时间
- 使用经验条件生存函数 `q(x;u)=P(T>x+u | T>x)` 估计 LLM bubble 剩余时间
- 期望收益为 `B=p_fail*C`，不再乘以“整个 checkpoint 被 bubble 覆盖”的概率下界
- 可见开销为 `O=∫_0^c [1-q(x;u)]du`；经验分布实现对该积分做精确求值
- `one probe per bubble`

也就是：

1. 当前 step 进入 LLM wait
2. 根据已经等待的时间 `x`
3. 计算期望收益 `B=p_fail*C`
4. 计算期望可见开销 `O=∫_0^c [1-q(x;u)]du`
5. 如果 `B>=O`，则在该 bubble 内最多 probe 一次

## 7. Docker Error Debug Workflow

如果当前问题是 Docker 层错误，建议按这个顺序排查：

1. 先用 replay debug 复现
   这是最短链路，最容易把 `checkpoint/create`、`gc`、`rerun` 路径单独打出来。
2. 再看 pool log
   关注：
   - `POST /exec`
   - `POST /checkpoint/create`
   - `POST /checkpoint/gc`
   - `POST /rerun`
3. 再看 exec server 日志
   重点关键字：
   - `exec attach failed`
   - `broken pipe`
   - `shim disconnected`
   - `is paused, unpause the container before exec`
4. 再看 checkpoint image 状态
   如果出现大量 `<none>:<none>`，通常说明 GC 只是 untag，没有真的释放 layer。

## 8. 推荐 Debug 顺序

### 8.1 验证基础功能

先跑：

```bash
REPLAY_LIMIT=4 \
REPLAY_MAX_CONCURRENCY=1 \
CHECKPOINT_AFTER_STEP=1 \
WAIT_CHECKPOINT_READY=1 \
RERUN_AFTER_STEP=1 \
GC_KEEP_LATEST=0 \
SIMULATE_LLM_DELAY=1 \
swe-rl/scripts/run_swe_checkpoint_replay_debug.sh
```

目标：

- create / ready / rerun / gc / close 全链路正常

### 8.2 验证并发冲突

再跑：

```bash
REPLAY_LIMIT=32 \
REPLAY_MAX_CONCURRENCY=32 \
CHECKPOINT_AFTER_STEP=1 \
WAIT_CHECKPOINT_READY=1 \
GC_KEEP_LATEST=0 \
SIMULATE_LLM_DELAY=1 \
swe-rl/scripts/run_swe_checkpoint_replay_debug.sh
```

目标：

- 观察 `checkpoint_busy`
- 观察 `docker error`
- 观察 `gc` 是否稳定

### 8.3 验证策略实验

最后跑：

```bash
EXPERIMENT_LIMIT=32 \
EXPERIMENT_MAX_CONCURRENCY=32 \
EXPERIMENT_INJECTION_COUNT=2 \
SIMULATE_LLM_DELAY=1 \
swe-rl/scripts/run_swe_checkpoint_policy_fault_experiment.sh
```

目标：

- 比较 `oracle / never / always / adaptive-risk`
- 同时比较 `batch wall time` 和 `per-traj wall time`

## 9. 当前文档覆盖范围

本文档覆盖的是：

- 现有 replay debug workflow
- 现有 fault experiment workflow
- 当前输出结构
- 当前 debug 入口

本文档不覆盖：

- 新的 GC 策略设计
- 新的 checkpoint admission 策略
- 新的 adaptive-risk 数学模型扩展
