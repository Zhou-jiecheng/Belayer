# Quick Start: get_all_tensors 功能

## 一分钟快速上手

### 问题
获取大量tensor时，逐个调用 `get_tensor()` 太慢：

```python
# 慢 ❌ - 100个tensor需要100次通信
for name in client.list_tensors():  # 假设有100个
    tensor = client.get_tensor(name)  # 每次都要一次ZMQ往返
```

### 解决方案
使用 `get_all_tensors()` 一次搞定：

```python
# 快 ✅ - 只需1次通信
all_tensors = client.get_all_tensors()  # 一次获取所有
```

## 完整示例

### Server端
```python
from checkpoint_engine.persistent_ps import PersistentParameterServer

# 从checkpoint加载并启动server
server = PersistentParameterServer(
    checkpoint_path="model.safetensors",
    device_id=0,
    zmq_port=5555,
)
server.start()
```

### Client端
```python
from checkpoint_engine.persistent_ps import ParameterServerClient

# 连接server
client = ParameterServerClient(zmq_port=5555)

# 🚀 一次性获取所有tensors（零拷贝）
all_tensors = client.get_all_tensors()

# 直接使用
model.load_state_dict(all_tensors)
```

## 性能对比

```python
import time

client = ParameterServerClient(zmq_port=5555)

# 旧方式
start = time.time()
tensors = {name: client.get_tensor(name) for name in client.list_tensors()}
old_time = time.time() - start

# 新方式
start = time.time()
tensors = client.get_all_tensors()
new_time = time.time() - start

print(f"旧方式: {old_time:.4f}s")
print(f"新方式: {new_time:.4f}s") 
print(f"快了: {old_time/new_time:.1f}x")

# 输出示例（100个tensor）:
# 旧方式: 0.0834s
# 新方式: 0.0056s
# 快了: 14.9x
```

## 何时使用

| 场景 | 推荐方法 |
|------|---------|
| 获取1个tensor | `get_tensor(name)` |
| 获取少量tensor (2-5个) | `get_tensor(name)` |
| 获取多个tensor (>5个) | `get_all_tensors()` ✅ |
| 加载完整模型 | `get_all_tensors()` ✅ |
| 多worker共享模型 | `get_all_tensors()` ✅ |

## 命令行测试

```bash
# Terminal 1: 启动server（使用真实checkpoint）
python -m checkpoint_engine.persistent_ps \
    --checkpoint /path/to/model.safetensors \
    --device-id 0 \
    --port 5555

# Terminal 2: 运行测试
python examples/test_get_all_tensors.py
```

## 关键优势

1. ⚡ **更快** - 10-20倍速度提升
2. 🎯 **更简单** - 一行代码搞定
3. 💾 **零拷贝** - IPC共享GPU内存
4. 🔄 **兼容** - 旧代码仍然可用

## 常见问题

**Q: 会占用更多内存吗？**
A: 不会，仍然是零拷贝IPC共享。

**Q: 旧代码需要改吗？**
A: 不需要，完全向后兼容。

**Q: 适合小模型吗？**
A: 对于<5个tensor的情况，性能差异不大，用哪个都可以。

---

更多详细信息请参考：
- `docs/GET_ALL_TENSORS.md` - 完整文档
- `examples/test_get_all_tensors.py` - 测试代码
