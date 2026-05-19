# Persistent Parameter Server - 简化版实现总结

## 变更说明

根据需求，已将实现简化为：

### ✅ 核心变更

1. **Server在初始化时加载tensor**
   - 构造函数接收 `tensors: dict[str, torch.Tensor]` 参数
   - 所有tensor在server启动时就加载到GPU显存
   - 移除了 `register_tensor()` 和 `unregister_tensor()` 方法

2. **Client只能读取**
   - 移除了 `register_tensor()` 方法
   - 移除了 `unregister_tensor()` 方法  
   - 只保留了 `get_tensor()`, `get_tensor_info()`, `list_tensors()` 方法

3. **Server只提供handle接口**
   - 主要命令：`get_ipc_handle` - 获取tensor的IPC handle
   - 辅助命令：`list`, `get_info` - 查询信息
   - 控制命令：`shutdown` - 关闭服务器

## 新的API

### PersistentParameterServer

```python
# 旧版本（已废弃）
server = PersistentParameterServer(device_id=0, zmq_port=5555)
server.register_tensor("weights", tensor)  # ❌ 移除

# 新版本
tensors = {"weights": tensor, "bias": bias}
server = PersistentParameterServer(
    tensors=tensors,  # ✅ 初始化时传入所有tensor
    device_id=0,
    zmq_port=5555,
)
server.start()
```

### ParameterServerClient

```python
client = ParameterServerClient(zmq_port=5555, device_id=0)

# ❌ 移除的方法
client.register_tensor("name", tensor)
client.unregister_tensor("name")

# ✅ 保留的方法（只读）
client.list_tensors()           # 列出所有tensor
client.get_tensor_info("name")  # 获取tensor信息
client.get_tensor("name")       # 获取tensor（零拷贝）
client.shutdown_server()        # 关闭服务器
```

## 命令行启动

```bash
# 新增 --checkpoint 参数（必需）
python -m checkpoint_engine.persistent_ps \
    --checkpoint model.pt \
    --device-id 0 \
    --port 5555

# checkpoint文件必须是包含tensor字典的.pt或.pth文件
```

## 使用流程

```
┌─────────────────────────────────────────┐
│ 1. 准备checkpoint                        │
│    tensors = {"w": ..., "b": ...}       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. 启动Server（加载所有tensor）         │
│    server = PersistentParameterServer(  │
│        tensors=tensors,                 │
│        device_id=0, port=5555           │
│    )                                    │
│    server.start()                       │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. Client连接并读取                     │
│    client = ParameterServerClient(...)  │
│    w = client.get_tensor("w") # 零拷贝  │
│    b = client.get_tensor("b") # 零拷贝  │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. Client退出，tensor仍在GPU显存        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 5. 新Client连接，继续读取               │
│    client2 = ParameterServerClient(...) │
│    w = client2.get_tensor("w")          │
└─────────────────────────────────────────┘
```

## 代码示例

### 示例1: 基本使用

```python
import torch
from checkpoint_engine.persistent_ps import (
    PersistentParameterServer,
    ParameterServerClient,
)

# === Server端 ===
# 准备tensors
checkpoint = {
    "layer1.weight": torch.randn(1024, 1024),
    "layer1.bias": torch.randn(1024),
    "layer2.weight": torch.randn(1024, 512),
    "layer2.bias": torch.randn(512),
}

# 启动server
server = PersistentParameterServer(
    tensors=checkpoint,
    device_id=0,
    zmq_port=5555,
)
server.start(daemon=False)  # 阻塞运行

# === Client端（业务进程）===
client = ParameterServerClient(zmq_port=5555, device_id=0)

# 列出所有tensor
print(client.list_tensors())
# ['layer1.weight', 'layer1.bias', 'layer2.weight', 'layer2.bias']

# 获取tensor（零拷贝）
w1 = client.get_tensor("layer1.weight")
b1 = client.get_tensor("layer1.bias")

# 使用tensor
output = torch.matmul(input, w1.T) + b1

client.close()
```

### 示例2: 从文件加载

```python
# 1. 保存checkpoint
checkpoint = {"weights": torch.randn(1000, 1000)}
torch.save(checkpoint, "model.pt")

# 2. 启动server（命令行）
# python -m checkpoint_engine.persistent_ps --checkpoint model.pt --port 5555

# 3. 客户端访问
client = ParameterServerClient(zmq_port=5555)
weights = client.get_tensor("weights")
```

### 示例3: 多进程共享

```python
# Server进程
def server_process():
    tensors = torch.load("large_model.pt")
    server = PersistentParameterServer(tensors=tensors, device_id=0)
    server.start(daemon=False)

# Worker进程1, 2, 3...
def worker_process(worker_id):
    client = ParameterServerClient(zmq_port=5555, device_id=0)
    
    # 所有worker共享同一份GPU内存中的tensor
    weights = client.get_tensor("model.weight")
    
    # 执行推理
    result = inference(data, weights)
    
    client.close()

# 启动多进程
import multiprocessing as mp
mp.Process(target=server_process).start()
for i in range(4):
    mp.Process(target=worker_process, args=(i,)).start()
```

## 文件更新清单

### 核心文件
- ✅ `checkpoint_engine/persistent_ps.py` - 简化实现
  - 修改 `PersistentParameterServer.__init__()` - 接收tensors参数
  - 移除 `register_tensor()` 和 `unregister_tensor()` 方法
  - 移除 `ParameterServerClient.register_tensor()` 方法
  - 修改 `_handle_request()` - 移除register/unregister命令
  - 修改 `run_server_daemon()` - 接收tensors参数
  - 修改 `__main__` - 添加--checkpoint参数

### 示例文件
- ✅ `examples/persistent_ps_example.py` - 更新为只读模式
- ✅ `examples/simple_example.py` - 新增简单示例

### 测试文件  
- ✅ `tests/test_persistent_ps.py` - 更新测试逻辑

### 文档文件
- ✅ `docs/SIMPLE_README.md` - 新版文档（推荐阅读）
- ⚠️ `docs/PERSISTENT_PS_README.md` - 旧版文档（已过时）
- ⚠️ `docs/QUICKSTART.md` - 旧版快速开始（已过时）

## 运行测试

```bash
# 1. 测试基本功能
cd /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/checkpoint-engine
python tests/test_persistent_ps.py

# 2. 运行简单示例
python examples/simple_example.py

# 3. 运行多进程示例
python examples/persistent_ps_example.py
```

## 对比总结

| 特性 | 旧版本 | 新版本（简化版） |
|------|--------|------------------|
| Server初始化 | 空的tensor池 | 必须传入所有tensor |
| Client注册 | ✅ 支持 | ❌ 不支持 |
| Client读取 | ✅ 支持 | ✅ 支持 |
| Client注销 | ✅ 支持 | ❌ 不支持 |
| 启动方式 | 代码启动 | 代码或命令行 |
| Checkpoint | 可选 | 必需（命令行） |
| 适用场景 | 动态注册 | 静态配置 |

## 优势

✅ **简化架构** - Server职责单一，只提供读取接口  
✅ **明确语义** - Server启动时就确定所有tensor，不再变化  
✅ **更安全** - Client无法修改server的tensor池  
✅ **更清晰** - 适合模型服务、推理等静态场景  

## 推荐使用方式

```bash
# 1. 训练完成后保存checkpoint
python train.py --output model.pt

# 2. 启动parameter server
python -m checkpoint_engine.persistent_ps \
    --checkpoint model.pt \
    --device-id 0 \
    --port 5555

# 3. 启动多个推理进程
python inference.py --ps-port 5555 &
python inference.py --ps-port 5555 &
python inference.py --ps-port 5555 &
```

在`inference.py`中：
```python
client = ParameterServerClient(zmq_port=5555)
model_params = {name: client.get_tensor(name) 
                for name in client.list_tensors()}
model.load_state_dict(model_params)
```

这样所有推理进程共享同一份GPU内存中的模型参数，节省显存并加快启动速度！
