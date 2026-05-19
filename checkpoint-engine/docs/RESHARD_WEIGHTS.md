# Reshard Weights 功能文档

## 概述

`reshard_weights` 函数实现了模型权重的张量并行(TP)和流水线并行(PP)切分，遵循 sglang/vllm 的切分策略。

## 函数签名

```python
def reshard_weights(
    weights: dict[str, torch.Tensor], 
    tp: int, 
    pp: int,
    num_layers: int | None = None,
) -> list[dict[str, torch.Tensor]]
```

### 参数

- `weights`: 模型权重字典，键为参数名，值为tensor
- `tp`: Tensor Parallel size (张量并行度)
- `pp`: Pipeline Parallel size (流水线并行度)  
- `num_layers`: transformer层数量 (可选，会自动检测)

### 返回值

返回 `tp * pp` 个权重字典的列表，每个字典对应一个shard的权重。

**Shard索引计算**：`shard_idx = pp_rank * tp + tp_rank`

## 切分策略

### 1. Tensor Parallel (TP) 切分

TP沿着模型的隐藏维度切分，减少单GPU的内存占用和计算量。

#### Column Parallel (列并行)

沿**输出维度**(通常是 dim 0)切分：

```python
# 匹配的层名称模式
- qkv_proj.weight      # Attention QKV (fused)
- q_proj.weight        # Query projection
- k_proj.weight        # Key projection
- v_proj.weight        # Value projection
- gate_proj.weight     # MLP gate
- up_proj.weight       # MLP up
- gate_up_proj.weight  # MLP gate+up (fused)
```

**示例**：
```python
# 原始权重: (4096, 4096)
# TP=4 切分后每个rank: (1024, 4096)
```

#### Row Parallel (行并行)

沿**输入维度**(通常是 dim 1)切分：

```python
# 匹配的层名称模式
- o_proj.weight        # Attention output
- down_proj.weight     # MLP down projection
```

**示例**：
```python
# 原始权重: (4096, 4096)
# TP=4 切分后每个rank: (4096, 1024)
```

#### Vocab Parallel (词表并行)

沿**词表维度**(通常是 dim 0)切分：

```python
# 匹配的层名称模式
- embed_tokens.weight  # Embedding层
- lm_head.weight       # Language model head
```

#### 不切分的层

以下类型的层在TP维度上**复制**到所有rank：

```python
- Layer Norm (input_layernorm, post_attention_layernorm, norm)
- Bias项 (大部分情况下)
- RMSNorm
```

### 2. Pipeline Parallel (PP) 切分

PP沿着模型的**层维度**切分，将不同的transformer层分配给不同的GPU。

#### 层分配策略

```python
# 假设有32层，PP=4
# PP rank 0: layers 0-7
# PP rank 1: layers 8-15
# PP rank 2: layers 16-23
# PP rank 3: layers 24-31
```

#### 非层权重的处理

以下权重在所有PP rank上**复制**：

- Embedding (`embed_tokens.weight`)
- LM Head (`lm_head.weight`)
- Final Layer Norm (`model.norm.weight`)

## 使用示例

### 示例 1: 仅 Tensor Parallel

```python
from checkpoint_engine.persistent_ps import reshard_weights
import torch

# 加载模型权重
weights = {
    "model.layers.0.self_attn.q_proj.weight": torch.randn(4096, 4096),
    "model.layers.0.self_attn.o_proj.weight": torch.randn(4096, 4096),
    # ... 更多权重
}

# TP=4, PP=1
shards = reshard_weights(weights, tp=4, pp=1, num_layers=32)

# 结果: 4个shard
# shards[0] = TP rank 0的权重
# shards[1] = TP rank 1的权重
# shards[2] = TP rank 2的权重
# shards[3] = TP rank 3的权重

# 每个shard的 q_proj 维度: (1024, 4096) - 输出维度被切分
# 每个shard的 o_proj 维度: (4096, 1024) - 输入维度被切分
```

### 示例 2: 仅 Pipeline Parallel

```python
# PP=4, TP=1
shards = reshard_weights(weights, tp=1, pp=4, num_layers=32)

# 结果: 4个shard
# shards[0] = PP rank 0的权重 (layers 0-7)
# shards[1] = PP rank 1的权重 (layers 8-15)
# shards[2] = PP rank 2的权重 (layers 16-23)
# shards[3] = PP rank 3的权重 (layers 24-31)

# 每个shard只包含分配给它的层
# 但embed_tokens和lm_head在所有shard中都存在
```

### 示例 3: TP + PP 组合

```python
# TP=2, PP=2
shards = reshard_weights(weights, tp=2, pp=2, num_layers=32)

# 结果: 4个shard
# shard[0] = PP rank 0, TP rank 0 (layers 0-15, TP切分)
# shard[1] = PP rank 0, TP rank 1 (layers 0-15, TP切分)
# shard[2] = PP rank 1, TP rank 0 (layers 16-31, TP切分)
# shard[3] = PP rank 1, TP rank 1 (layers 16-31, TP切分)

# 索引计算: shard_idx = pp_rank * tp + tp_rank
```

### 示例 4: 与 Parameter Server 集成

```python
from checkpoint_engine.persistent_ps import (
    load_tensors_from_checkpoint,
    reshard_weights,
    PersistentParameterServer,
)

# 1. 从checkpoint加载
weights = load_tensors_from_checkpoint(
    "model.safetensors",
    device_id=0
)

# 2. 切分权重
tp_size = 4
pp_size = 2
shards = reshard_weights(weights, tp=tp_size, pp=pp_size, num_layers=32)

# 3. 为每个rank启动parameter server
for pp_rank in range(pp_size):
    for tp_rank in range(tp_size):
        shard_idx = pp_rank * tp_size + tp_rank
        device_id = shard_idx  # 假设每个GPU一个shard
        
        server = PersistentParameterServer(
            tensors=shards[shard_idx],
            device_id=device_id,
            zmq_port=5555 + shard_idx,
        )
        server.start()
```

## 内存优化效果

### 不同配置的内存占用对比

假设原始模型: 7B参数，约28GB (FP32)

| 配置 | Shards | 每Shard内存 | 内存减少 |
|------|--------|------------|---------|
| TP=1, PP=1 | 1 | 28.0 GB | 1.0x |
| TP=2, PP=1 | 2 | 14.0 GB | 2.0x |
| TP=4, PP=1 | 4 | 7.0 GB | 4.0x |
| TP=8, PP=1 | 8 | 3.5 GB | 8.0x |
| TP=1, PP=2 | 2 | ~14.0 GB | ~2.0x |
| TP=2, PP=2 | 4 | ~7.0 GB | ~4.0x |

**注意**：PP的内存减少略少于TP，因为embedding和lm_head需要复制。

## 切分模式识别

函数通过**参数名称匹配**来识别切分模式：

```python
# Column Parallel 检测
if "qkv_proj" in name or "gate_up_proj" in name:
    # 沿dim 0切分
    
# Row Parallel 检测
if "o_proj" in name or "down_proj" in name:
    # 沿dim 1切分
    
# 不切分检测
if "norm" in name or ".bias" in name:
    # 复制到所有TP rank
```

## 支持的模型架构

基于命名模式，支持以下模型：

### ✅ 完全支持
- **Llama** (llama, llama2, llama3)
- **Qwen** (qwen, qwen2)
- **Mistral**
- **Yi**
- **DeepSeek**

### ✅ 部分支持
- **GPT-2** (需要映射: `c_attn` → `qkv_proj`, `c_proj` → `o_proj`)
- **BERT** (需要映射命名)

### 自定义模型

对于自定义模型，确保参数名包含以下关键字：

- Column Parallel: `qkv_proj`, `gate_proj`, `up_proj`
- Row Parallel: `o_proj`, `down_proj`
- No Shard: `norm`, `layernorm`

## 验证和测试

### 运行测试

```bash
python examples/test_reshard_weights.py
```

### 测试内容

1. ✅ 无切分 (TP=1, PP=1)
2. ✅ 仅TP切分 (验证列并行/行并行)
3. ✅ 仅PP切分 (验证层分配)
4. ✅ TP+PP组合
5. ✅ Fused层支持 (QKV, Gate+Up)
6. ✅ 内存节省演示

### 验证正确性

```python
# 验证切分后的维度
original = weights["model.layers.0.q_proj.weight"]  # (4096, 4096)
sharded = reshard_weights(weights, tp=4, pp=1)

# TP rank 0的切分
assert sharded[0]["model.layers.0.q_proj.weight"].shape == (1024, 4096)

# 验证所有shard加起来等于原始
reconstructed = torch.cat([
    sharded[i]["model.layers.0.q_proj.weight"] 
    for i in range(4)
], dim=0)
assert torch.allclose(reconstructed, original)
```

## 性能考虑

### TP vs PP 选择

| 维度 | Tensor Parallel | Pipeline Parallel |
|------|----------------|-------------------|
| 通信量 | 高 (每层都需要all-reduce) | 低 (只在边界传递) |
| 负载均衡 | 好 | 可能不均 |
| 内存效率 | 高 (精确切分) | 中 (需要复制embedding) |
| 适用场景 | 单机多卡 | 多机多卡 |

### 最佳实践

1. **单机多卡** (8 GPUs)
   ```python
   # 推荐: TP=8, PP=1
   shards = reshard_weights(weights, tp=8, pp=1)
   ```

2. **多机多卡** (32 GPUs, 4机×8卡)
   ```python
   # 推荐: TP=8, PP=4
   shards = reshard_weights(weights, tp=8, pp=4)
   ```

3. **超大模型** (175B+)
   ```python
   # TP=8, PP=16 (128 GPUs)
   shards = reshard_weights(weights, tp=8, pp=16)
   ```

## 故障排查

### 问题1: 维度不能被TP整除

```
WARNING: Dimension 0 of xxx (4095) is not divisible by TP size 4. Replicating instead.
```

**解决**：调整TP大小或padding权重维度。

### 问题2: 层数检测失败

```
WARNING: Could not detect number of layers, assuming no layer sharding
```

**解决**：手动指定 `num_layers` 参数。

### 问题3: 未知的参数名

如果某些参数名不匹配已知模式，它们会被**复制**到所有shard。

**解决**：在 `_shard_weight_for_tp()` 中添加新的匹配模式。

## 扩展功能

### 添加自定义切分模式

```python
def _shard_weight_for_tp(name: str, tensor: torch.Tensor, tp_size: int):
    # 添加自定义模式
    custom_column_patterns = [
        "my_model.custom_proj.weight",
    ]
    
    for pattern in custom_column_patterns:
        if pattern in name:
            # 自定义切分逻辑
            ...
```

### 支持新的并行策略

可以扩展以支持：
- Expert Parallel (MoE模型)
- Sequence Parallel
- Context Parallel

## 总结

`reshard_weights` 提供了完整的TP/PP切分功能：

- ✅ 自动识别切分模式
- ✅ 支持主流LLM架构
- ✅ 内存高效
- ✅ 与Parameter Server无缝集成
- ✅ 完整的测试覆盖

这使得大模型可以轻松分布到多个GPU上，实现高效的分布式训练和推理。
