# Persistent Parameter Server - 快速开始

## 安装

确保已安装依赖：

```bash
pip install torch pyzmq loguru pydantic
```

## 5分钟快速体验

### 第一步：启动Parameter Server

在终端1中运行：

```bash
cd /mnt/shared-storage-user/ailab-sys/zhoujiecheng/projs/checkpoint-engine
python -m checkpoint_engine.persistent_ps --device-id 0 --port 5555
```

你会看到类似输出：
```
Initialized PersistentParameterServer on cuda:0
Parameter server listening on tcp://127.0.0.1:5555
```

### 第二步：业务进程注册Tensor

在终端2中运行Python：

```python
from checkpoint_engine.persistent_ps import ParameterServerClient
import torch

# 连接到服务器
client = ParameterServerClient(zmq_port=5555, device_id=0)

# 创建并注册tensor
model_weights = torch.randn(1000, 1000, dtype=torch.float32)
client.register_tensor("my_model_weights", model_weights)

print(f"Registered tensors: {client.list_tensors()}")
# 输出: Registered tensors: ['my_model_weights']

client.close()
```

**重要：此时退出Python进程，但tensor仍然保留在GPU显存中！**

### 第三步：新进程访问Tensor（零拷贝）

在终端2中重新启动Python（模拟新的业务进程）：

```python
from checkpoint_engine.persistent_ps import ParameterServerClient

# 重新连接
client = ParameterServerClient(zmq_port=5555, device_id=0)

# 查看可用的tensor
print(f"Available tensors: {client.list_tensors()}")
# 输出: Available tensors: ['my_model_weights']

# 通过IPC获取tensor（零拷贝！）
weights = client.get_tensor("my_model_weights")
print(f"Retrieved tensor: shape={weights.shape}, device={weights.device}")
# 输出: Retrieved tensor: shape=torch.Size([1000, 1000]), device=cuda:0

# 可以直接使用这个tensor进行计算
# 注意：这个tensor指向parameter server中的GPU内存，是零拷贝的
result = torch.sum(weights)
print(f"Sum of weights: {result.item()}")

client.close()
```

### 第四步：关闭服务器

```python
from checkpoint_engine.persistent_ps import ParameterServerClient

client = ParameterServerClient(zmq_port=5555)
client.shutdown_server()
client.close()
```

或者在终端1中按 `Ctrl+C`。

## 运行完整示例

运行提供的完整示例：

```bash
# 多进程示例（推荐）
python examples/persistent_ps_example.py

# 单进程示例
python examples/persistent_ps_example.py --sequential
```

## 运行测试

```bash
python tests/test_persistent_ps.py
```

测试会验证：
- ✓ Tensor注册
- ✓ Tensor列表
- ✓ Tensor信息查询
- ✓ IPC获取Tensor
- ✓ 零拷贝验证
- ✓ Tensor注销
- ✓ 多Tensor管理

## 核心概念

### 零拷贝 (Zero-Copy)

传统方式：
```
业务进程A → 创建Tensor → 存储在显存A
业务进程B → 从A复制数据 → 存储在显存B  ❌ 浪费显存和时间
```

使用Parameter Server：
```
Parameter Server → Tensor存储在显存
业务进程A → 通过IPC Handle访问 → 直接读写显存  ✓ 零拷贝
业务进程B → 通过IPC Handle访问 → 直接读写显存  ✓ 零拷贝
```

### 持久化存储

- **传统方式**：进程退出 → Tensor被释放 → 下次需要重新加载
- **Parameter Server方式**：进程退出 → Tensor仍在显存 → 新进程直接获取

## 典型使用场景

### 场景1：模型热更新

```python
# 训练进程
client = ParameterServerClient(zmq_port=5555)
for epoch in range(num_epochs):
    train_one_epoch(model)
    # 更新parameter server中的权重
    client.register_tensor("model_weights", model.state_dict())

# 推理进程（无需重启）
client = ParameterServerClient(zmq_port=5555)
while True:
    # 获取最新权重
    weights = client.get_tensor("model_weights")
    model.load_state_dict(weights)
    inference(model, data)
```

### 场景2：多进程推理

```python
# Parameter Server中加载一份模型
client = ParameterServerClient(zmq_port=5555)
client.register_tensor("shared_model", model_weights)

# 启动多个推理进程，共享同一份权重
# 进程1, 2, 3... 都可以获取相同的tensor
weights = client.get_tensor("shared_model")
```

### 场景3：Checkpoint管理

```python
# 加载checkpoint到parameter server
checkpoint = torch.load("model.pt")
client.register_tensor("checkpoint", checkpoint)

# 多个进程可以快速访问checkpoint，无需重复加载
weights = client.get_tensor("checkpoint")
```

## 下一步

- 查看完整文档：`docs/PERSISTENT_PS_README.md`
- 阅读示例代码：`examples/persistent_ps_example.py`
- 查看API文档：`checkpoint_engine/persistent_ps.py`

## 常见问题

**Q: 能跨机器使用吗？**
A: 当前IPC机制只支持同一机器内的进程间共享。跨机器需要使用网络传输（非零拷贝）。

**Q: 多个进程同时修改tensor安全吗？**
A: 当前实现未加锁，需要业务层面保证同步。建议一个进程写，多个进程读。

**Q: 支持NPU吗？**
A: 支持！代码兼容NPU设备，会自动检测设备类型。

**Q: 显存占用多少？**
A: Parameter Server只占用注册的tensor的显存，不会额外复制数据。
