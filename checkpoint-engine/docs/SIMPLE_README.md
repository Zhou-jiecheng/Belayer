# Persistent Parameter Server - 简化版

持久化参数服务器，用于在GPU显存中保存tensor并通过IPC实现零拷贝访问。

## 核心设计

- **Server初始化时加载所有tensor** - 所有tensor在server启动时就准备好
- **Client只能读取** - 客户端只能获取IPC handle，不能注册新tensor
- **零拷贝访问** - 通过IPC handle实现跨进程零拷贝共享

## 快速开始

### 1. 准备checkpoint文件

```python
import torch

# 创建包含tensor的字典
checkpoint = {
    "model.weight": torch.randn(1000, 1000),
    "model.bias": torch.randn(1000),
}

# 保存到文件
torch.save(checkpoint, "model.pt")
```

### 2. 启动Parameter Server

#### 方式A: 命令行启动

```bash
python -m checkpoint_engine.persistent_ps \
    --checkpoint model.pt \
    --device-id 0 \
    --port 5555
```

#### 方式B: 代码启动

```python
from checkpoint_engine.persistent_ps import PersistentParameterServer
import torch

# 加载或创建tensors
tensors = torch.load("model.pt")
# 或者直接创建
# tensors = {
#     "weights": torch.randn(1000, 1000),
#     "bias": torch.randn(1000),
# }

# 启动server
server = PersistentParameterServer(
    tensors=tensors,
    device_id=0,
    zmq_port=5555,
)

server.start(daemon=False)  # 阻塞运行
```

### 3. 客户端访问Tensor（零拷贝）

```python
from checkpoint_engine.persistent_ps import ParameterServerClient

# 连接到server
client = ParameterServerClient(zmq_port=5555, device_id=0)

# 查看可用的tensor
print(client.list_tensors())
# 输出: ['model.weight', 'model.bias']

# 获取tensor（零拷贝！）
weights = client.get_tensor("model.weight")
bias = client.get_tensor("model.bias")

# 直接使用tensor
output = torch.matmul(input, weights) + bias

# 关闭连接
client.close()
```

## API 文档

### PersistentParameterServer

服务器在初始化时加载所有tensor，并提供IPC访问接口。

```python
server = PersistentParameterServer(
    tensors: dict[str, torch.Tensor],  # 必需：要服务的tensor字典
    device_id: int = 0,                # GPU设备ID
    zmq_port: int = 5555,              # ZMQ端口
    zmq_host: str = "127.0.0.1",       # 绑定地址
)

server.start(daemon=False)  # 启动服务器
server.stop()               # 停止服务器
server.cleanup()            # 清理资源
```

### ParameterServerClient

客户端只能读取tensor，不能修改server的tensor池。

```python
client = ParameterServerClient(
    zmq_host: str = "127.0.0.1",  # Server地址
    zmq_port: int = 5555,         # Server端口
    device_id: int | None = None, # 本地GPU设备ID
)

# 列出所有tensor
tensors = client.list_tensors() -> list[str]

# 获取tensor信息
info = client.get_tensor_info(name: str) -> dict | None

# 获取tensor（零拷贝）
tensor = client.get_tensor(name: str) -> torch.Tensor | None

# 关闭连接
client.close()

# 关闭服务器
client.shutdown_server()
```

## 完整示例

```python
# === 服务器端 ===
import torch
from checkpoint_engine.persistent_ps import PersistentParameterServer

# 1. 准备tensors
checkpoint = torch.load("model_checkpoint.pt")
# checkpoint = {
#     "layer1.weight": tensor1,
#     "layer1.bias": tensor2,
#     ...
# }

# 2. 启动server
server = PersistentParameterServer(
    tensors=checkpoint,
    device_id=0,
    zmq_port=5555,
)
server.start(daemon=False)

# === 客户端（业务进程）===
from checkpoint_engine.persistent_ps import ParameterServerClient

# 1. 连接server
client = ParameterServerClient(zmq_port=5555, device_id=0)

# 2. 获取tensor
weight = client.get_tensor("layer1.weight")
bias = client.get_tensor("layer1.bias")

# 3. 使用tensor（零拷贝，直接访问GPU内存）
output = model_forward(input, weight, bias)

# 4. 关闭
client.close()
```

## 使用场景

### 1. 模型推理服务

```python
# Server: 加载模型参数
model_params = torch.load("model.pt")
server = PersistentParameterServer(tensors=model_params, device_id=0)
server.start()

# Client 1, 2, 3...: 多个推理进程共享同一份参数
client = ParameterServerClient(zmq_port=5555)
weights = {name: client.get_tensor(name) for name in client.list_tensors()}
model.load_state_dict(weights)
```

### 2. 大模型加载

```python
# Server: 一次性加载大模型到GPU
large_model = torch.load("70B_model.pt")  # 很大的模型
server = PersistentParameterServer(tensors=large_model, device_id=0)
server.start()

# Client: 快速获取参数，无需重复加载
client = ParameterServerClient(zmq_port=5555)
weights = client.get_tensor("transformer.layers.0.weight")
```

### 3. 热更新

虽然客户端不能注册新tensor，但可以通过重启server实现热更新：

```bash
# 1. 训练完成，保存新checkpoint
torch.save(new_model, "model_v2.pt")

# 2. 重启server（graceful restart）
# 旧server shutdown
# 新server启动加载model_v2.pt

# 3. 客户端自动连接到新server，获取新参数
```

## 运行示例

```bash
# 简单示例
python examples/simple_example.py

# 多进程示例
python examples/persistent_ps_example.py

# 测试
python tests/test_persistent_ps.py
```

## 技术实现

### 零拷贝原理

```python
# Server端：生成IPC handle
from torch.multiprocessing.reductions import reduce_tensor
ipc_handle = reduce_tensor(tensor)  # (func, args)

# Client端：重建tensor引用
func, args = ipc_handle
tensor = func(*args)  # 指向相同的GPU内存
```

### 进程通信

- 使用ZMQ的REQ-REP模式
- Server处理3种命令：
  - `list`: 列出所有tensor
  - `get_info`: 获取tensor元信息
  - `get_ipc_handle`: 获取IPC handle（核心）
  - `shutdown`: 关闭服务器

## 优势

| 特性 | 传统方式 | Parameter Server |
|------|---------|------------------|
| 显存占用 | N个进程 × 模型大小 | 1 × 模型大小 |
| 加载时间 | 每个进程都要加载 | 只加载一次 |
| 数据拷贝 | 每个进程一份副本 | 零拷贝共享 |
| 进程启动 | 慢（需加载模型） | 快（只获取handle） |

## 限制

1. **同机器限制**: IPC只支持同一台机器的进程间通信
2. **只读访问**: 客户端只能读取，不能注册新tensor
3. **设备兼容**: 需要相同类型的设备（GPU/NPU）
4. **无并发控制**: 多进程同时修改tensor需要自行同步

## 依赖

```
torch >= 1.9.0
pyzmq
loguru
```

## FAQ

**Q: 如何更新server中的tensor？**  
A: 重启server并加载新的checkpoint。

**Q: 多个进程同时修改tensor安全吗？**  
A: 当前无锁机制，建议一写多读模式。

**Q: 支持跨机器吗？**  
A: 不支持。IPC机制只能在同一台机器内使用。

**Q: Server崩溃了怎么办？**  
A: Tensor会丢失，需要重启server重新加载。

## License

参考原项目LICENSE。
