# 从文件系统加载Checkpoint - 使用指南

## 新增功能

Parameter Server 现在支持直接从文件系统加载checkpoint，无需手动加载tensor字典。

### 支持的文件格式

1. **SafeTensors** (推荐)
   - 单个 `.safetensors` 文件
   - 分片模型 (目录包含 `model.safetensors.index.json`)
   
2. **PyTorch**
   - `.pt` 文件
   - `.pth` 文件
   - `.bin` 文件
   - 训练checkpoint (包含 `model` 或 `state_dict` 键)

3. **目录**
   - 包含多个 `.safetensors` 文件的目录
   - 包含多个 `.pt`/`.pth` 文件的目录

## 使用方法

### 方式1: 命令行启动（推荐）

```bash
# 从单个文件加载
python -m checkpoint_engine.persistent_ps \
    --checkpoint /path/to/model.safetensors \
    --device-id 0 \
    --port 5555

# 从目录加载（自动加载所有checkpoint文件）
python -m checkpoint_engine.persistent_ps \
    --checkpoint /path/to/model_dir \
    --device-id 0 \
    --port 5555

# 从HuggingFace下载的模型目录加载
python -m checkpoint_engine.persistent_ps \
    --checkpoint ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b/snapshots/xxx \
    --device-id 0 \
    --port 5555
```

### 方式2: 代码中使用

```python
from checkpoint_engine.persistent_ps import PersistentParameterServer

# 方式A: 从checkpoint文件加载
server = PersistentParameterServer(
    checkpoint_path="/path/to/model.safetensors",
    device_id=0,
    zmq_port=5555,
)

# 方式B: 从checkpoint目录加载
server = PersistentParameterServer(
    checkpoint_path="/path/to/model_dir",
    device_id=0,
    zmq_port=5555,
)

# 方式C: 仍然支持直接传入tensors字典（向后兼容）
tensors = {"weight": torch.randn(100, 100)}
server = PersistentParameterServer(
    tensors=tensors,
    device_id=0,
    zmq_port=5555,
)

server.start(daemon=False)
```

## 完整示例

### 示例1: 从SafeTensors加载

```python
import torch
from safetensors.torch import save_file
from checkpoint_engine.persistent_ps import (
    PersistentParameterServer,
    ParameterServerClient,
)

# 1. 创建并保存checkpoint
tensors = {
    "model.layers.0.weight": torch.randn(1024, 1024),
    "model.layers.0.bias": torch.randn(1024),
    "model.layers.1.weight": torch.randn(1024, 512),
    "model.layers.1.bias": torch.randn(512),
}
save_file(tensors, "model.safetensors")

# 2. 启动server（自动加载）
server = PersistentParameterServer(
    checkpoint_path="model.safetensors",
    device_id=0,
    zmq_port=5555,
)
server.start(daemon=True)

# 3. 客户端访问
client = ParameterServerClient(zmq_port=5555)
print(client.list_tensors())
# ['model.layers.0.weight', 'model.layers.0.bias', 'model.layers.1.weight', 'model.layers.1.bias']

weight = client.get_tensor("model.layers.0.weight")
print(weight.shape)  # torch.Size([1024, 1024])
```

### 示例2: 从PyTorch Checkpoint加载

```python
# 1. 创建训练checkpoint
checkpoint = {
    "model": {
        "fc1.weight": torch.randn(512, 256),
        "fc1.bias": torch.randn(512),
        "fc2.weight": torch.randn(256, 10),
        "fc2.bias": torch.randn(10),
    },
    "optimizer_state_dict": {},  # 会被忽略
    "epoch": 100,                 # 会被忽略
}
torch.save(checkpoint, "training_checkpoint.pt")

# 2. 启动server（自动提取model部分）
server = PersistentParameterServer(
    checkpoint_path="training_checkpoint.pt",
    device_id=0,
)
server.start()

# 3. 客户端会看到model中的所有tensor
client = ParameterServerClient()
print(client.list_tensors())
# ['fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias']
```

### 示例3: 加载分片模型

```python
# 假设模型目录结构：
# model_dir/
#   ├── model.safetensors.index.json
#   ├── model-00001-of-00003.safetensors
#   ├── model-00002-of-00003.safetensors
#   └── model-00003-of-00003.safetensors

# 直接指定目录即可，会自动读取index.json
server = PersistentParameterServer(
    checkpoint_path="model_dir",
    device_id=0,
)
server.start()

# 所有分片中的tensor都会被加载
```

## 使用load_tensors_from_checkpoint函数

如果只想加载tensor而不启动server：

```python
from checkpoint_engine.persistent_ps import load_tensors_from_checkpoint

# 加载checkpoint到GPU
tensors = load_tensors_from_checkpoint(
    checkpoint_path="/path/to/checkpoint",
    device_id=0,
)

# tensors是一个字典: {name: tensor}
for name, tensor in tensors.items():
    print(f"{name}: {tensor.shape}, {tensor.device}")
```

## 实际应用场景

### 场景1: LLM推理服务

```bash
# 1. 下载模型（使用huggingface-cli或直接指定路径）
# 假设模型在 ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b/snapshots/xxx

# 2. 启动parameter server
python -m checkpoint_engine.persistent_ps \
    --checkpoint ~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b/snapshots/xxx \
    --device-id 0 \
    --port 5555

# 3. 多个推理worker连接
python inference_worker.py --ps-port 5555 &
python inference_worker.py --ps-port 5555 &
python inference_worker.py --ps-port 5555 &
```

在 `inference_worker.py` 中：
```python
from checkpoint_engine.persistent_ps import ParameterServerClient

client = ParameterServerClient(zmq_port=5555, device_id=0)

# 获取所有模型参数
model_params = {}
for name in client.list_tensors():
    model_params[name] = client.get_tensor(name)

# 加载到模型
model.load_state_dict(model_params)

# 开始推理
```

### 场景2: 模型热更新

```python
# 训练循环
for epoch in range(num_epochs):
    train_one_epoch(model)
    
    # 每个epoch保存checkpoint
    torch.save({
        "model": model.state_dict(),
        "epoch": epoch,
    }, f"checkpoint_epoch_{epoch}.pt")

# 更新parameter server (重启)
# 旧server shutdown
# 新server启动加载新checkpoint
server = PersistentParameterServer(
    checkpoint_path="checkpoint_epoch_100.pt",
    device_id=0,
)
server.start()

# Worker自动获取新参数
```

## 性能优化

### 1. 使用SafeTensors (推荐)

SafeTensors比PyTorch格式加载更快且更安全：

```python
# 转换PyTorch checkpoint到SafeTensors
import torch
from safetensors.torch import save_file

state_dict = torch.load("model.pt")
if "model" in state_dict:
    state_dict = state_dict["model"]

save_file(state_dict, "model.safetensors")
```

### 2. 预加载到GPU

```python
# tensors会直接加载到指定GPU
server = PersistentParameterServer(
    checkpoint_path="model.safetensors",
    device_id=0,  # 直接加载到GPU 0
)
```

### 3. 分片加载大模型

对于超大模型，使用分片checkpoint：

```python
# 自动处理分片，无需特殊配置
server = PersistentParameterServer(
    checkpoint_path="llama-70b-sharded",  # 包含多个分片文件的目录
    device_id=0,
)
```

## 测试

运行测试脚本验证功能：

```bash
python examples/test_load_from_checkpoint.py
```

测试包括：
- ✅ 加载 .pt/.pth 文件
- ✅ 加载 .safetensors 文件
- ✅ 加载训练checkpoint (带model key)
- ✅ 加载目录中的多个文件
- ✅ 加载分片模型
- ✅ 向后兼容性 (直接传tensors)

## API参考

### load_tensors_from_checkpoint

```python
load_tensors_from_checkpoint(
    checkpoint_path: str,
    device_id: int = 0,
) -> dict[str, torch.Tensor]
```

**参数:**
- `checkpoint_path`: checkpoint文件或目录路径
- `device_id`: 加载到的GPU设备ID

**返回:**
- 字典: `{tensor_name: tensor}`

**支持格式:**
- `.safetensors` 单文件
- `.pt`/`.pth`/`.bin` 单文件
- 包含 `model.safetensors.index.json` 的目录（分片模型）
- 包含多个checkpoint文件的目录

### PersistentParameterServer.__init__

```python
PersistentParameterServer(
    tensors: dict[str, torch.Tensor] | None = None,
    checkpoint_path: str | None = None,
    device_id: int = 0,
    zmq_port: int = 5555,
    zmq_host: str = "127.0.0.1",
)
```

**参数:**
- `tensors`: tensor字典（可选，与checkpoint_path二选一）
- `checkpoint_path`: checkpoint路径（可选，与tensors二选一）
- `device_id`: GPU设备ID
- `zmq_port`: ZMQ端口
- `zmq_host`: ZMQ主机地址

**注意:** `tensors` 和 `checkpoint_path` 必须提供其中一个

## 常见问题

**Q: 支持哪些checkpoint格式？**  
A: SafeTensors (.safetensors), PyTorch (.pt, .pth, .bin), 以及包含这些文件的目录。

**Q: 如何加载HuggingFace模型？**  
A: 直接指定模型缓存目录，会自动处理分片和index文件。

**Q: 加载速度如何？**  
A: SafeTensors格式最快。PyTorch格式稍慢但也支持。建议使用SafeTensors。

**Q: 内存占用如何？**  
A: Tensor直接加载到GPU，不会在CPU和GPU间重复占用内存。

**Q: 是否支持CPU-only模式？**  
A: 支持，代码会自动检测设备类型。

## 总结

新的checkpoint加载功能让Parameter Server使用更加便捷：

- ✅ **无需手动加载** - 直接指定checkpoint路径
- ✅ **多种格式** - 支持SafeTensors和PyTorch格式
- ✅ **自动分片** - 自动处理大模型分片
- ✅ **向后兼容** - 仍支持直接传入tensors字典
- ✅ **高性能** - 直接加载到GPU，零拷贝访问
