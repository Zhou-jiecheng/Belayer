# Parameter Server 快速参考

## 一分钟上手

```python
# === Server: 启动 ===
from checkpoint_engine.persistent_ps import PersistentParameterServer
import torch

tensors = {"weights": torch.randn(1000, 1000)}
server = PersistentParameterServer(tensors=tensors, device_id=0, zmq_port=5555)
server.start(daemon=False)

# === Client: 使用 ===
from checkpoint_engine.persistent_ps import ParameterServerClient

client = ParameterServerClient(zmq_port=5555, device_id=0)
weights = client.get_tensor("weights")  # 零拷贝！
client.close()
```

## 命令速查

```bash
# 从checkpoint启动
python -m checkpoint_engine.persistent_ps --checkpoint model.pt --port 5555

# 运行示例
python examples/simple_example.py
python examples/persistent_ps_example.py

# 运行测试
python tests/test_persistent_ps.py
```

## API速查

### Server API

```python
PersistentParameterServer(
    tensors: dict[str, torch.Tensor],  # 必需
    device_id: int = 0,
    zmq_port: int = 5555,
)

server.start(daemon=False)  # 启动
server.stop()               # 停止  
server.cleanup()            # 清理
```

### Client API

```python
ParameterServerClient(
    zmq_host: str = "127.0.0.1",
    zmq_port: int = 5555,
    device_id: int | None = None,
)

client.list_tensors() -> list[str]                    # 列出
client.get_tensor_info(name) -> dict | None           # 信息
client.get_tensor(name) -> torch.Tensor | None        # 获取（零拷贝）
client.shutdown_server()                              # 关闭
client.close()                                        # 断开
```

## 常用模式

### 模式1: 模型推理

```python
# 1. Server加载模型
model = torch.load("model.pt")
server = PersistentParameterServer(tensors=model, device_id=0)
server.start()

# 2. Worker获取参数
client = ParameterServerClient(zmq_port=5555)
params = {k: client.get_tensor(k) for k in client.list_tensors()}
```

### 模式2: 多进程

```python
import multiprocessing as mp

# Server进程
def server_proc():
    server = PersistentParameterServer(tensors=..., device_id=0)
    server.start(daemon=False)

# Worker进程
def worker_proc(i):
    client = ParameterServerClient(zmq_port=5555)
    tensor = client.get_tensor("weights")
    # 使用tensor...
    client.close()

# 启动
mp.Process(target=server_proc).start()
for i in range(4):
    mp.Process(target=worker_proc, args=(i,)).start()
```

### 模式3: 命令行

```bash
# Terminal 1: 启动server
python -m checkpoint_engine.persistent_ps \
    --checkpoint model.pt \
    --device-id 0 \
    --port 5555

# Terminal 2, 3, 4...: 运行worker
python worker.py --ps-port 5555
```

## 关键概念

| 概念 | 说明 |
|------|------|
| 零拷贝 | 通过IPC handle共享GPU内存，无数据拷贝 |
| 持久化 | Server不退出，tensor常驻GPU显存 |
| 只读 | Client只能读取，不能注册新tensor |
| 初始化 | Server启动时就加载所有tensor |

## 注意事项

⚠️ **只支持同机器** - IPC仅限同一台机器的进程  
⚠️ **只读访问** - Client不能注册tensor  
⚠️ **无并发锁** - 多进程写需自行同步  
⚠️ **Server崩溃** - Tensor会丢失，需重启  

## 性能对比

```
传统方式（3个进程）:
- 显存: 3 × 4GB = 12GB
- 加载: 3 × 10s = 30s

Parameter Server:
- 显存: 1 × 4GB = 4GB
- 加载: 1 × 10s + 3 × 0.1s = 10.3s
- 节省: 66% 显存，66% 时间
```

## 文档链接

- 📖 完整文档: `docs/SIMPLE_README.md`
- 🚀 快速开始: `docs/SIMPLIFIED_VERSION_SUMMARY.md`
- 💻 示例代码: `examples/simple_example.py`
- ✅ 测试: `tests/test_persistent_ps.py`

## 故障排查

```python
# 问题: 连接失败
# 解决: 检查端口和server是否启动
client = ParameterServerClient(zmq_port=5555)
print(client.list_tensors())  # 测试连接

# 问题: Tensor不存在
# 解决: 检查名称和server中的tensor
print(client.list_tensors())  # 查看所有tensor

# 问题: 设备错误
# 解决: 确保client和server使用相同的device_id
client = ParameterServerClient(zmq_port=5555, device_id=0)
```

## 最佳实践

✅ Server在独立进程中运行  
✅ 使用合适的端口避免冲突  
✅ Client使用后及时close()  
✅ 优雅关闭server（shutdown_server）  
✅ 错误处理和日志记录  

---

**快速帮助**: 查看 `docs/SIMPLE_README.md` 获取详细文档
