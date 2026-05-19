# Reshard Weights - 快速参考

## 一行总结

根据TP/PP策略自动切分模型权重，实现分布式部署。

## 基本用法

```python
from checkpoint_engine.persistent_ps import reshard_weights

# 切分权重
shards = reshard_weights(
    weights,      # dict[str, torch.Tensor]
    tp=4,         # Tensor Parallel size
    pp=2,         # Pipeline Parallel size
    num_layers=32 # 可选，会自动检测
)

# 结果: 8个shard (4 TP × 2 PP)
# shard[pp_rank * tp + tp_rank] = 该rank的权重
```

## 切分规则速查表

| 层类型 | 示例参数名 | TP切分维度 | PP切分 |
|--------|-----------|-----------|--------|
| **Column Parallel** | `qkv_proj.weight` | dim 0 (输出) | 按层 |
| | `gate_up_proj.weight` | dim 0 | 按层 |
| | `embed_tokens.weight` | dim 0 (vocab) | 复制 |
| **Row Parallel** | `o_proj.weight` | dim 1 (输入) | 按层 |
| | `down_proj.weight` | dim 1 | 按层 |
| **No Shard** | `*.norm.weight` | 复制 | 复制/按层 |
| | `*.bias` | 复制 | 按层 |

## 常用配置

### 单机8卡

```python
# 纯TP切分
shards = reshard_weights(weights, tp=8, pp=1)
```

### 4机×8卡 (32卡)

```python
# TP=8, PP=4
shards = reshard_weights(weights, tp=8, pp=4)
```

### 内存受限

```python
# 最大化切分
shards = reshard_weights(weights, tp=16, pp=8)  # 128个shard
```

## 维度变化示例

```python
# 原始权重
q_proj: (4096, 4096)
o_proj: (4096, 4096)

# TP=4 切分后
q_proj: (1024, 4096)  # Column: dim 0缩小4倍
o_proj: (4096, 1024)  # Row: dim 1缩小4倍
```

## 与Parameter Server集成

```python
# 1. 加载+切分
weights = load_tensors_from_checkpoint("model.safetensors")
shards = reshard_weights(weights, tp=4, pp=2)

# 2. 每个GPU启动server
for rank in range(8):
    server = PersistentParameterServer(
        tensors=shards[rank],
        device_id=rank,
        zmq_port=5555 + rank,
    )
    server.start()
```

## 验证切分正确性

```python
# 测试脚本
python examples/test_reshard_weights.py

# 手动验证
original = weights["model.layers.0.q_proj.weight"]
shards = reshard_weights(weights, tp=4, pp=1)

# 重建并比较
reconstructed = torch.cat([
    shards[i]["model.layers.0.q_proj.weight"] for i in range(4)
], dim=0)
assert torch.allclose(reconstructed, original)
```

## 常见问题

**Q: 维度不能被TP整除怎么办？**
A: 函数会自动退化为复制模式，或调整TP大小。

**Q: 支持哪些模型？**
A: Llama, Qwen, Mistral, Yi, DeepSeek等主流架构。

**Q: 如何添加自定义切分模式？**
A: 修改 `_shard_weight_for_tp()` 中的匹配模式列表。

## 性能提示

- **TP优先**: 单机多卡优先用TP (通信快)
- **PP补充**: 多机场景使用PP (减少跨机通信)
- **避免过度切分**: TP × PP不要超过GPU数量

## 完整文档

详见: `docs/RESHARD_WEIGHTS.md`
