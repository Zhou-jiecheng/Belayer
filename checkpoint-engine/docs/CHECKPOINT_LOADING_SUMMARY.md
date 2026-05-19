# 新功能：从文件系统加载Checkpoint

## 概述

已成功实现从文件系统直接加载模型checkpoint的功能，参考了 `loader.py` 中的 `_get_weights_iterator` 方法。

## 核心改动

### 1. 新增函数

```python
# 从SafeTensors加载
def _load_tensors_from_safetensors(file_path: str) -> Generator

# 从PyTorch文件加载
def _load_tensors_from_pytorch(file_path: str) -> Generator

# 主加载函数（支持多种格式）
def load_tensors_from_checkpoint(
    checkpoint_path: str,
    device_id: int = 0,
) -> dict[str, torch.Tensor]
```

### 2. 修改PersistentParameterServer

```python
# 旧版本
def __init__(self, tensors: dict[str, torch.Tensor], ...):
    pass

# 新版本（向后兼容）
def __init__(
    self,
    tensors: dict[str, torch.Tensor] | None = None,
    checkpoint_path: str | None = None,  # 新增参数
    ...
):
    # 如果提供checkpoint_path，自动加载
    if checkpoint_path is not None:
        tensors = load_tensors_from_checkpoint(checkpoint_path, device_id)
```

## 支持的格式

### 1. SafeTensors (推荐)

```python
# 单个文件
server = PersistentParameterServer(
    checkpoint_path="model.safetensors",
    device_id=0,
)

# 分片模型（自动读取index.json）
server = PersistentParameterServer(
    checkpoint_path="model_dir/",  # 包含 model.safetensors.index.json
    device_id=0,
)
```

### 2. PyTorch格式

```python
# .pt/.pth/.bin 文件
server = PersistentParameterServer(
    checkpoint_path="model.pt",
    device_id=0,
)

# 训练checkpoint（自动提取'model'或'state_dict'）
checkpoint = {
    "model": {...},      # 会被加载
    "optimizer": {...},  # 会被忽略
    "epoch": 100,        # 会被忽略
}
```

### 3. 目录

```python
# 包含多个checkpoint文件的目录
server = PersistentParameterServer(
    checkpoint_path="checkpoint_dir/",
    device_id=0,
)
```

## 使用方式

### 命令行（推荐）

```bash
# 从文件加载
python -m checkpoint_engine.persistent_ps \
    --checkpoint model.safetensors \
    --device-id 0 \
    --port 5555

# 从目录加载
python -m checkpoint_engine.persistent_ps \
    --checkpoint /path/to/model_dir \
    --device-id 0 \
    --port 5555

# 从HuggingFace缓存加载
python -m checkpoint_engine.persistent_ps \
    --checkpoint ~/.cache/huggingface/hub/models--xxx/snapshots/xxx \
    --device-id 0
```

### 代码中使用

```python
from checkpoint_engine.persistent_ps import PersistentParameterServer

# 方式1: 新功能 - 从checkpoint加载
server = PersistentParameterServer(
    checkpoint_path="model.safetensors",
    device_id=0,
)

# 方式2: 旧方式 - 手动传入tensors (仍然支持)
tensors = {"weight": torch.randn(100, 100)}
server = PersistentParameterServer(
    tensors=tensors,
    device_id=0,
)

server.start()
```

## 技术实现

### 参考sglang的实现

参考了 `loader.py` 中的以下设计：

1. **SafeTensors加载**
   ```python
   # 参考: loader.py 的 safetensors_weights_iterator
   with safe_open(file_path, framework="pt", device="cpu") as f:
       for name in f.keys():
           tensor = f.get_tensor(name)
           yield name, tensor
   ```

2. **PyTorch加载**
   ```python
   # 参考: loader.py 的 pt_weights_iterator
   state_dict = torch.load(file_path, map_location="cpu")
   # 处理训练checkpoint格式
   if "model" in state_dict:
       state_dict = state_dict["model"]
   ```

3. **分片模型支持**
   ```python
   # 参考: loader.py 的 index.json 处理逻辑
   with open("model.safetensors.index.json") as f:
       index = json.load(f)
   weight_files = set(index["weight_map"].values())
   ```

### 实现细节

```python
def load_tensors_from_checkpoint(checkpoint_path, device_id):
    tensors = {}
    device = torch.device(f"cuda:{device_id}")
    
    if os.path.isfile(checkpoint_path):
        # 单文件处理
        if checkpoint_path.endswith(".safetensors"):
            for name, tensor in _load_tensors_from_safetensors(checkpoint_path):
                tensors[name] = tensor.to(device)
        elif checkpoint_path.endswith((".pt", ".pth", ".bin")):
            for name, tensor in _load_tensors_from_pytorch(checkpoint_path):
                tensors[name] = tensor.to(device)
                
    elif os.path.isdir(checkpoint_path):
        # 目录处理
        # 1. 检查是否有index.json (分片模型)
        # 2. 否则加载目录下所有checkpoint文件
        ...
    
    return tensors
```

## 测试

创建了完整的测试脚本：

```bash
python examples/test_load_from_checkpoint.py
```

测试覆盖：
- ✅ 加载单个 .pt 文件
- ✅ 加载单个 .safetensors 文件
- ✅ 加载训练checkpoint (包含 'model' key)
- ✅ 加载目录中的多个文件
- ✅ 加载分片模型（index.json）
- ✅ Server直接使用checkpoint_path初始化
- ✅ 向后兼容性（仍支持传入tensors字典）

## 实际应用

### 场景1: 加载HuggingFace模型

```python
# 模型已下载到缓存
model_path = "~/.cache/huggingface/hub/models--meta-llama--Llama-2-7b/snapshots/xxx"

server = PersistentParameterServer(
    checkpoint_path=model_path,
    device_id=0,
)
server.start()

# 多个worker共享模型参数
client = ParameterServerClient(zmq_port=5555)
model_params = {name: client.get_tensor(name) for name in client.list_tensors()}
```

### 场景2: 快速启动推理服务

```bash
# 一行命令启动parameter server
python -m checkpoint_engine.persistent_ps --checkpoint model.safetensors --port 5555

# 启动多个推理worker
python worker.py --ps-port 5555 &
python worker.py --ps-port 5555 &
```

## 优势

| 特性 | 旧方式 | 新方式 |
|------|--------|--------|
| 加载方式 | 手动 `torch.load()` | 自动加载 |
| 代码量 | 需要额外代码 | 一行搞定 |
| 格式支持 | 需要手动处理 | 自动识别 |
| 分片模型 | 需要手动合并 | 自动处理 |
| 训练checkpoint | 需要手动提取 | 自动提取 |

## 文件更新

- ✅ `checkpoint_engine/persistent_ps.py` - 核心实现
- ✅ `examples/test_load_from_checkpoint.py` - 测试脚本
- ✅ `docs/LOAD_FROM_CHECKPOINT.md` - 详细文档

## 依赖

新增依赖：
```python
from safetensors.torch import safe_open  # 用于加载safetensors
import json  # 用于解析index.json
import glob  # 用于文件匹配
```

所有依赖都是标准库或已有依赖，无需额外安装。

## 总结

成功实现了从文件系统加载checkpoint的功能：

✅ 参考了 `loader.py` 的实现方式  
✅ 支持多种checkpoint格式  
✅ 自动处理分片模型  
✅ 保持向后兼容性  
✅ 提供完整测试和文档  

现在使用Parameter Server更加方便，只需一行命令即可从checkpoint启动服务！
