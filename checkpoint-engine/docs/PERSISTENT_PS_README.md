# Persistent Parameter Server

一个持久化的参数服务器实现，用于在GPU显存中保存tensor，并通过IPC（进程间通信）实现零拷贝的tensor共享。

## 核心特性

1. **持久化存储**: Parameter Server作为常驻进程，即使业务进程退出，tensor仍然保留在GPU显存中
2. **零拷贝访问**: 通过IPC handle共享tensor，避免数据拷贝
3. **跨进程共享**: 多个业务进程可以同时访问同一个tensor
4. **简单易用**: 提供简洁的客户端API

## 架构设计

```
┌─────────────────────────────────────────────────┐
│         Parameter Server (常驻进程)              │
│  ┌───────────────────────────────────────┐     │
│  │      GPU Memory                        │     │
│  │  ┌─────────┐  ┌─────────┐  ┌────────┐│     │
│  │  │Tensor 1 │  │Tensor 2 │  │Tensor 3││     │
│  │  └─────────┘  └─────────┘  └────────┘│     │
│  └───────────────────────────────────────┘     │
│              ↑                                  │
│              │ IPC Handle                       │
│              ↓                                  │
│       ZMQ Server (REP)                          │
└──────────────┬──────────────────────────────────┘
               │
               │ TCP/ZMQ
               │
┌──────────────┴──────────────────────────────────┐
│         Business Process 1                      │
│  ┌─────────────────────────────┐               │
│  │  ParameterServerClient      │               │
│  │  - register_tensor()        │               │
│  │  - get_tensor() [IPC]       │               │
│  │  - unregister_tensor()      │               │
│  └─────────────────────────────┘               │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│         Business Process 2 (新启动)              │
│  ┌─────────────────────────────┐               │
│  │  ParameterServerClient      │               │
│  │  - get_tensor() [IPC]       │  ← 零拷贝访问  │
│  │  - 直接读写GPU tensor        │               │
│  └─────────────────────────────┘               │
└─────────────────────────────────────────────────┘
```

## 使用方法

### 1. 启动Parameter Server

有两种方式启动服务器：

#### 方式1: 作为独立进程运行

```bash
python -m checkpoint_engine.persistent_ps --device-id 0 --port 5555 --host 127.0.0.1
```

参数说明：
- `--device-id`: 使用的GPU设备ID（默认0）
- `--port`: ZMQ通信端口（默认5555）
- `--host`: 绑定的主机地址（默认127.0.0.1）

#### 方式2: 在代码中启动

```python
from checkpoint_engine.persistent_ps import PersistentParameterServer

server = PersistentParameterServer(device_id=0, zmq_port=5555)
server.start(daemon=False)  # daemon=False表示非后台线程

# 服务器会持续运行，直到收到shutdown命令
```

### 2. 业务进程：注册Tensor

```python
from checkpoint_engine.persistent_ps import ParameterServerClient
import torch

# 连接到parameter server
client = ParameterServerClient(zmq_port=5555, device_id=0)

# 创建tensor
weights = torch.randn(1000, 1000, dtype=torch.float32)
bias = torch.zeros(1000, dtype=torch.float32)

# 注册到parameter server
client.register_tensor("model.weights", weights)
client.register_tensor("model.bias", bias)

# 查看所有tensor
print(client.list_tensors())  # ['model.weights', 'model.bias']

# 获取tensor信息
info = client.get_tensor_info("model.weights")
print(info)  # {'name': ..., 'shape': ..., 'dtype': ..., 'device': ...}

client.close()
```

### 3. 业务进程退出后，新进程获取Tensor（零拷贝）

```python
from checkpoint_engine.persistent_ps import ParameterServerClient

# 重新连接（即使之前的进程已退出）
client = ParameterServerClient(zmq_port=5555, device_id=0)

# 通过IPC获取tensor（零拷贝）
weights = client.get_tensor("model.weights")
bias = client.get_tensor("model.bias")

# 直接使用tensor进行计算
# 这些tensor实际上指向parameter server中的GPU内存
output = torch.matmul(input, weights) + bias

# 也可以直接修改tensor（会影响parameter server中的数据）
weights[0, 0] = 999.0

client.close()
```

### 4. 关闭Parameter Server

```python
client = ParameterServerClient(zmq_port=5555)
client.shutdown_server()
client.close()
```

## API 文档

### PersistentParameterServer

**初始化参数：**
- `device_id`: GPU设备ID
- `zmq_port`: ZMQ通信端口
- `zmq_host`: 绑定的主机地址

**方法：**
- `start(daemon=True)`: 启动服务器
- `stop()`: 停止服务器
- `cleanup()`: 清理资源
- `register_tensor(name, tensor)`: 注册tensor
- `unregister_tensor(name)`: 注销tensor
- `get_tensor_info(name)`: 获取tensor信息
- `list_tensors()`: 列出所有tensor名称

### ParameterServerClient

**初始化参数：**
- `zmq_host`: Parameter Server的主机地址
- `zmq_port`: Parameter Server的端口
- `device_id`: 本地GPU设备ID（用于重建IPC tensor）

**方法：**
- `register_tensor(name, tensor)`: 注册tensor到服务器
- `unregister_tensor(name)`: 从服务器注销tensor
- `get_tensor(name)`: 通过IPC获取tensor（零拷贝）
- `get_tensor_info(name)`: 获取tensor元信息
- `list_tensors()`: 列出所有tensor
- `shutdown_server()`: 关闭服务器
- `close()`: 关闭客户端连接

## 运行示例

仓库提供了完整的示例代码：

```bash
# 运行多进程示例
python examples/persistent_ps_example.py

# 运行单进程示例
python examples/persistent_ps_example.py --sequential
```

示例演示了：
1. 启动parameter server
2. 业务进程1注册tensor
3. 业务进程1退出
4. 业务进程2通过IPC获取tensor（零拷贝）
5. 业务进程2修改tensor
6. 业务进程3验证修改（证明是零拷贝共享）

## 技术细节

### IPC (Inter-Process Communication)

使用PyTorch的`torch.multiprocessing.reductions.reduce_tensor`功能，将tensor转换为可序列化的IPC handle。其他进程可以通过这个handle重建对同一块GPU内存的引用，实现零拷贝共享。

### ZMQ通信

使用ZMQ的REQ-REP模式进行客户端和服务器的通信：
- 客户端发送请求（register, get_ipc_handle, list等）
- 服务器处理请求并返回响应
- 对于tensor数据，使用pickle序列化IPC handle

### 内存管理

- Parameter Server持有tensor的引用，防止被垃圾回收
- 当tensor被注销时，调用`gc.collect()`和`empty_cache()`释放GPU内存
- 客户端进程退出不影响Parameter Server中的tensor

## 使用场景

1. **模型热更新**: 训练进程更新参数后，推理进程无需重启即可获取新参数
2. **多进程推理**: 多个推理进程共享同一份模型参数，节省显存
3. **检查点管理**: 将checkpoint加载到parameter server，多个进程可以快速访问
4. **分布式训练**: 作为参数服务器，在多GPU/多节点间共享参数

## 注意事项

1. **进程间内存共享**: IPC handle只能在同一台机器的不同进程间使用
2. **设备兼容性**: 客户端和服务器需要使用相同类型的设备（GPU/NPU）
3. **并发安全**: 当前实现未加锁，多个进程同时修改tensor可能导致竞争条件
4. **网络通信**: 默认使用本地TCP连接，跨机器使用需要修改host地址

## 依赖项

- PyTorch >= 1.9.0
- pyzmq
- loguru
- pydantic

## License

参考原项目的LICENSE文件。
