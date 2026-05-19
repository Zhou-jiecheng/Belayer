"""
Persistent Parameter Server for Zero-Copy Tensor Sharing

This module implements a parameter server that keeps tensors resident in GPU memory
and allows business processes to access them via IPC handles without copying data.
"""

import gc
import glob
import os
import threading
from collections.abc import Callable
from typing import Any, Generator
import json
import re

import torch
import zmq
from loguru import logger
from safetensors.torch import safe_open

from checkpoint_engine.device_utils import DeviceManager, npu_generate_uuid


def _load_tensors_from_safetensors(
    file_path: str,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Load tensors from a safetensors file."""
    with safe_open(file_path, framework="pt", device="cpu") as f:
        for name in f.keys():
            tensor = f.get_tensor(name)
            yield name, tensor


def _load_tensors_from_pytorch(
    file_path: str,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    """Load tensors from a PyTorch .pt or .pth file."""
    state_dict = torch.load(file_path, map_location="cpu")
    
    # Handle different checkpoint formats
    if isinstance(state_dict, dict):
        # Check if it's a training checkpoint with 'model' or 'state_dict' key
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        
        # Yield tensors from the state dict
        for name, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                yield name, tensor
    else:
        raise ValueError(f"Unsupported checkpoint format in {file_path}")


def load_tensors_from_checkpoint(
    checkpoint_path: str,
) -> dict[str, torch.Tensor]:
    """
    Load tensors from checkpoint files (safetensors or PyTorch format).
    
    Args:
        checkpoint_path: Path to checkpoint file or directory containing checkpoint files
        device_id: GPU device ID to load tensors to
        
    Returns:
        Dictionary mapping tensor names to tensors
        
    Supports:
        - Single .safetensors file
        - Single .pt/.pth file
        - Directory with multiple .safetensors files
        - Directory with model.safetensors.index.json (sharded models)
    """
    tensors = {}
    device = "cpu"
    
    if os.path.isfile(checkpoint_path):
        # Single file
        logger.info(f"Loading checkpoint from file: {checkpoint_path}")
        
        if checkpoint_path.endswith(".safetensors"):
            for name, tensor in _load_tensors_from_safetensors(checkpoint_path):
                tensors[name] = tensor.to(device)
                logger.debug(f"Loaded tensor '{name}' with shape {tensor.shape}")
        elif checkpoint_path.endswith((".pt", ".pth", ".bin")):
            for name, tensor in _load_tensors_from_pytorch(checkpoint_path):
                tensors[name] = tensor.to(device)
                logger.debug(f"Loaded tensor '{name}' with shape {tensor.shape}")
        else:
            raise ValueError(f"Unsupported file format: {checkpoint_path}")
            
    elif os.path.isdir(checkpoint_path):
        # Directory with checkpoint files
        logger.info(f"Loading checkpoint from directory: {checkpoint_path}")
        
        # Check for index file (sharded safetensors)
        index_file = os.path.join(checkpoint_path, "model.safetensors.index.json")
        if os.path.exists(index_file):
            import json
            with open(index_file, "r") as f:
                index = json.load(f)
            
            # Get unique weight files
            weight_map = index.get("weight_map", {})
            weight_files = set(weight_map.values())
            
            logger.info(f"Found {len(weight_files)} sharded safetensors files")
            for weight_file in sorted(weight_files):
                file_path = os.path.join(checkpoint_path, weight_file)
                logger.info(f"Loading from {weight_file}")
                for name, tensor in _load_tensors_from_safetensors(file_path):
                    tensors[name] = tensor.to(device)
                    logger.debug(f"Loaded tensor '{name}' with shape {tensor.shape}")
        else:
            # Load all safetensors or pytorch files in directory
            safetensors_files = glob.glob(os.path.join(checkpoint_path, "*.safetensors"))
            pytorch_files = glob.glob(os.path.join(checkpoint_path, "*.pt")) + \
                          glob.glob(os.path.join(checkpoint_path, "*.pth")) + \
                          glob.glob(os.path.join(checkpoint_path, "*.bin"))
            
            if safetensors_files:
                logger.info(f"Found {len(safetensors_files)} safetensors files")
                for file_path in sorted(safetensors_files):
                    logger.info(f"Loading from {os.path.basename(file_path)}")
                    for name, tensor in _load_tensors_from_safetensors(file_path):
                        tensors[name] = tensor.to(device)
                        logger.debug(f"Loaded tensor '{name}' with shape {tensor.shape}")
            elif pytorch_files:
                logger.info(f"Found {len(pytorch_files)} PyTorch files")
                for file_path in sorted(pytorch_files):
                    logger.info(f"Loading from {os.path.basename(file_path)}")
                    for name, tensor in _load_tensors_from_pytorch(file_path):
                        tensors[name] = tensor.to(device)
                        logger.debug(f"Loaded tensor '{name}' with shape {tensor.shape}")
            else:
                raise ValueError(f"No checkpoint files found in {checkpoint_path}")
    else:
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    logger.info(f"Loaded {len(tensors)} tensors from checkpoint")
    return tensors


def _find_model_config_path(checkpoint_path: str) -> str | None:
    """Best-effort locate config.json for the given checkpoint path."""
    explicit_path = os.environ.get("SGLANG_WEIGHT_DAEMON_CONFIG_PATH")
    if explicit_path:
        if os.path.exists(explicit_path):
            return explicit_path
        logger.warning(
            "SGLANG_WEIGHT_DAEMON_CONFIG_PATH is set but does not exist: %s",
            explicit_path,
        )

    start_dir = checkpoint_path if os.path.isdir(checkpoint_path) else os.path.dirname(checkpoint_path)
    current = os.path.abspath(start_dir)
    for _ in range(6):
        config_path = os.path.join(current, "config.json")
        if os.path.exists(config_path):
            return config_path
        parent = os.path.dirname(current)
        if parent == current:
            break
        current = parent
    return None


def reshard_weights(
    weights: dict[str, torch.Tensor],
    tp: int,
    pp: int,
    model_config: dict | None = None,
) -> list[dict[str, torch.Tensor]]:
    """
    Reshard model weights for Tensor Parallel (TP) and Pipeline Parallel (PP).
    
    Args:
        weights: Dictionary of parameter name -> tensor
        tp: Tensor parallel size
        pp: Pipeline parallel size  
        model_config: Model configuration dict with keys:
            - num_layers: Number of transformer layers
            - num_attention_heads: Total number of attention heads
            - num_key_value_heads: Total number of KV heads (for GQA)
            - hidden_size: Hidden dimension size
            - head_dim: Per-head hidden dimension size
            
    Returns:
        List of weight dicts, one per shard. Index = pp_rank * tp + tp_rank
        
    Reference:
        Implements sharding strategy from sglang/vllm:
        - QKVParallelLinear: Handles Q/K/V with different head counts (GQA)
        - MergedColumnParallelLinear: Handles fused gate/up projections
        - ColumnParallelLinear: Shard output dim
        - RowParallelLinear: Shard input dim
    """
    # Normalize checkpoint names to sglang model parameter names first.
    # This is required for direct weight_daemon replacement (no model.load_weights mapper).
    weights = _normalize_weight_names(weights, model_config)

    detected_config = _detect_model_config(weights)
    configured_decoder = _extract_decoder_config(model_config)

    # Prefer explicit config from checkpoint json, but fall back to tensor detection.
    num_layers = configured_decoder.get(
        "num_hidden_layers",
        detected_config.get("num_hidden_layers", detected_config.get("num_layers", 0)),
    )
    num_heads = configured_decoder.get(
        "num_attention_heads", detected_config.get("num_attention_heads", 0)
    )
    num_kv_heads = configured_decoder.get(
        "num_key_value_heads", detected_config.get("num_key_value_heads", num_heads)
    )
    hidden_size = configured_decoder.get(
        "hidden_size", detected_config.get("hidden_size", 0)
    )
    head_dim = configured_decoder.get(
        "head_dim", detected_config.get("head_dim", 0)
    )
    
    logger.info(
        f"Resharding weights: TP={tp}, PP={pp}, "
        f"num_layers={num_layers}, num_heads={num_heads}, num_kv_heads={num_kv_heads}, head_dim={head_dim}"
    )
    
    # pack qkv and gate/up projections (needed even for TP=1, PP=1)
    pack_layers(weights)
    
    if tp == 1 and pp == 1:
        return [weights]
    
    # Initialize shards
    total_shards = tp * pp
    sharded_weights = [{} for _ in range(total_shards)]
    
    # Determine PP layer boundaries
    pp_boundaries = _compute_pp_boundaries(num_layers, pp)

    # Process each weight
    for name, tensor in weights.items():
        # Determine PP rank assignment
        layer_idx = get_layer_id(name)
        if layer_idx is not None:
            # Layer-specific weight
            target_pp_ranks = []
            for pp_rank, (start, end) in enumerate(pp_boundaries):
                if start <= layer_idx < end:
                    target_pp_ranks = [pp_rank]
                    break
        else:
            # Non-layer weight (embedding, head, norm) - replicate to all PP ranks
            target_pp_ranks = list(range(pp))
        
        if not target_pp_ranks:
            logger.warning(f"Skipping weight {name}: not assigned to any PP rank")
            continue
        
        # Shard for TP based on weight type
        tp_shards = _shard_for_tp(
            name=name,
            tensor=tensor,
            tp_size=tp,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
        )
        
        # Assign to shards
        for pp_rank in target_pp_ranks:
            for tp_rank in range(tp):
                shard_idx = pp_rank * tp + tp_rank
                sharded_weights[shard_idx][name] = tp_shards[tp_rank]
    
    # Log statistics
    for idx, shard in enumerate(sharded_weights):
        pp_rank = idx // tp
        tp_rank = idx % tp
        total_size = sum(t.numel() * t.element_size() for t in shard.values())
        logger.info(
            f"Shard[PP={pp_rank},TP={tp_rank}]: {len(shard)} tensors, "
            f"{total_size / 1024**3:.2f} GB"
        )
    
    return sharded_weights


def _is_qwen3_vl_family(model_config: dict | None) -> bool:
    if not isinstance(model_config, dict):
        return False
    model_type = str(model_config.get("model_type", "")).lower()
    architectures = model_config.get("architectures", [])
    if not isinstance(architectures, list):
        architectures = [architectures]
    arch_str = " ".join(str(x).lower() for x in architectures)
    return (
        "qwen3_vl" in model_type
        or "qwen3vl" in model_type
        or "qwen3_vl" in arch_str
        or "qwen3vl" in arch_str
    )


def _normalize_single_weight_name(name: str, model_config: dict | None) -> str:
    new_name = name

    # Qwen/Qwen3 checkpoint variants may omit "model." prefix.
    if new_name.startswith(("layers.", "embed_tokens.", "norm.")):
        new_name = f"model.{new_name}"

    # Qwen3-VL checkpoints may use HF-side names that differ from sglang parameter names.
    if _is_qwen3_vl_family(model_config):
        if new_name.startswith("model.language_model.model."):
            new_name = new_name.replace("model.language_model.model.", "model.", 1)
        elif new_name.startswith("model.language_model."):
            new_name = new_name.replace("model.language_model.", "model.", 1)
        elif new_name.startswith("language_model.model."):
            new_name = new_name.replace("language_model.model.", "model.", 1)
        elif new_name.startswith("language_model.lm_head."):
            new_name = new_name.replace("language_model.lm_head.", "lm_head.", 1)

        if new_name.startswith("model.visual."):
            new_name = new_name.replace("model.visual.", "visual.", 1)

        if ".attn.qkv." in new_name:
            new_name = new_name.replace(".attn.qkv.", ".attn.qkv_proj.")

    return new_name


def _normalize_weight_names(
    weights: dict[str, torch.Tensor], model_config: dict | None
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}

    for name, tensor in weights.items():
        mapped = _normalize_single_weight_name(name, model_config)
        if mapped in normalized and normalized[mapped].shape != tensor.shape:
            logger.warning(
                "Weight name collision after normalization: '%s' and another tensor map to '%s'. Keeping the latter.",
                name,
                mapped,
            )
        normalized[mapped] = tensor

    return normalized


def _detect_model_config(weights: dict[str, torch.Tensor]) -> dict:
    """Auto-detect model configuration from weights."""
    config = {}
    
    # Detect number of layers
    layer_indices = set()
    for name in weights.keys():
        idx = get_layer_id(name)
        if idx is not None:
            layer_indices.add(idx)
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    config["num_layers"] = num_layers
    config["num_hidden_layers"] = num_layers
    
    # Detect hidden size from embedding
    for name, tensor in weights.items():
        if "embed_tokens.weight" in name or "wte.weight" in name:
            config["hidden_size"] = tensor.shape[1]
            break
    
    # Try to detect num_heads from qkv_proj or q_proj
    for name, tensor in weights.items():
        if ".layers.0." in name and "qkv_proj.weight" in name:
            # Fused QKV: shape is (hidden * 3, hidden) or (hidden * (1+2*kv_ratio), hidden)
            output_dim = tensor.shape[0]
            hidden = config.get("hidden_size", 0)
            if hidden > 0:
                total_heads_dim = output_dim
                # Assume equal Q/K/V for now, will be refined if we find separate projections
                config["num_attention_heads"] = total_heads_dim // hidden
            break
        elif ".layers.0." in name and "q_proj.weight" in name:
            # Separate Q projection
            hidden = config.get("hidden_size", 0)
            if hidden > 0:
                config["num_attention_heads"] = tensor.shape[0] // hidden
            break
    
    # Detect num_kv_heads
    for name, tensor in weights.items():
        if ".layers.0." in name and "k_proj.weight" in name:
            hidden = config.get("hidden_size", 0)
            if hidden > 0:
                config["num_key_value_heads"] = tensor.shape[0] // hidden
            break

    # Detect head_dim when grouped attention uses a dedicated per-head size.
    for name, tensor in weights.items():
        if ".layers.0." in name and "q_proj.weight" in name and "num_attention_heads" in config:
            num_heads = config["num_attention_heads"]
            if num_heads > 0:
                config["head_dim"] = tensor.shape[0] // num_heads
            break
        if ".layers.0." in name and "k_proj.weight" in name and "num_key_value_heads" in config:
            num_kv_heads = config["num_key_value_heads"]
            if num_kv_heads > 0:
                config["head_dim"] = tensor.shape[0] // num_kv_heads
            break
    
    # Default: num_kv_heads = num_heads (MHA)
    if "num_key_value_heads" not in config and "num_attention_heads" in config:
        config["num_key_value_heads"] = config["num_attention_heads"]
    
    logger.info(f"Auto-detected config: {config}")
    return config


def _extract_decoder_config(model_config: dict | None) -> dict:
    """Extract text-decoder fields from model config (supports nested multimodal configs)."""
    if not isinstance(model_config, dict):
        return {}

    candidates = [model_config]
    for key in ("text_config", "language_config", "llm_config", "model_config"):
        nested = model_config.get(key)
        if isinstance(nested, dict):
            candidates.append(nested)

    resolved = {}
    mapping = {
        "num_hidden_layers": "num_hidden_layers",
        "num_layers": "num_hidden_layers",
        "hidden_size": "hidden_size",
        "num_attention_heads": "num_attention_heads",
        "num_key_value_heads": "num_key_value_heads",
        "head_dim": "head_dim",
    }
    for candidate in candidates:
        for src_key, dst_key in mapping.items():
            if dst_key not in resolved and src_key in candidate:
                resolved[dst_key] = candidate[src_key]

    return resolved


def get_layer_id(weight_name):
    # example weight name: model.layers.10.self_attn.qkv_proj.weight
    match = re.search(r"layers\.(\d+)\.", weight_name)
    if match:
        return int(match.group(1))
    return None


def _get_layer_prefix(weight_name: str, layer_id: int) -> str:
    marker = f"layers.{layer_id}."
    idx = weight_name.find(marker)
    if idx == -1:
        return "model."
    return weight_name[:idx]

def _compute_pp_boundaries(num_layers: int, pp_size: int) -> list[tuple[int, int]]:
    """Compute layer boundaries for each PP rank."""
    if pp_size == 1 or num_layers == 0:
        return [(0, num_layers)] * pp_size
    
    layers_per_rank = num_layers // pp_size
    boundaries = []
    for rank in range(pp_size):
        start = rank * layers_per_rank
        end = (rank + 1) * layers_per_rank if rank < pp_size - 1 else num_layers
        boundaries.append((start, end))
    
    return boundaries

def pack_layers(tensors: dict[str, torch.Tensor]) -> None:
    """
    Pack qkv_proj.weight from q_proj.weight, k_proj.weight, and v_proj.weight.
    
    Modifies the tensors dict in-place.
    """
    layer_to_tensors = {}
    for k, v in tensors.items():
        layer_id = get_layer_id(k)
        if layer_id is not None:
            prefix = _get_layer_prefix(k, layer_id)
            layer_key = (prefix, layer_id)
            if layer_key not in layer_to_tensors:
                layer_to_tensors[layer_key] = []
            layer_to_tensors[layer_key].append((k, v))

    for (prefix, layer_id), layer_tensors in layer_to_tensors.items():
        if prefix and not prefix.endswith("."):
            prefix = f"{prefix}."
        q_weight = None
        k_weight = None
        v_weight = None
        q_bias = None
        k_bias = None
        v_bias = None
        gate_proj = None
        up_proj = None
        gate_proj_bias = None
        up_proj_bias = None
        
        for name, tensor in layer_tensors:
            if "q_proj.weight" in name:
                q_weight = tensor
                del tensors[name]
            elif "k_proj.weight" in name:
                k_weight = tensor
                del tensors[name]
            elif "v_proj.weight" in name:
                v_weight = tensor
                del tensors[name]
            elif "q_proj.bias" in name:
                q_bias = tensor
                del tensors[name]
            elif "k_proj.bias" in name:
                k_bias = tensor
                del tensors[name]
            elif "v_proj.bias" in name:
                v_bias = tensor
                del tensors[name]
            elif "gate_proj.weight" in name:
                gate_proj = tensor
                del tensors[name]
            elif "up_proj.weight" in name:
                up_proj = tensor
                del tensors[name]
            elif "gate_proj.bias" in name:
                gate_proj_bias = tensor
                del tensors[name]
            elif "up_proj.bias" in name:
                up_proj_bias = tensor
                del tensors[name]

        if q_weight is not None and k_weight is not None and v_weight is not None:
            packed_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            packed_name = f"{prefix}layers.{layer_id}.self_attn.qkv_proj.weight"
            tensors[packed_name] = packed_weight
            logger.info(f"Packed {packed_name} from q_proj, k_proj, v_proj")
        
        if gate_proj is not None and up_proj is not None:
            packed_weight = torch.cat([gate_proj, up_proj], dim=0)
            packed_name = f"{prefix}layers.{layer_id}.mlp.gate_up_proj.weight"
            tensors[packed_name] = packed_weight
            logger.info(f"Packed {packed_name} from gate_proj, up_proj")
        
        if q_bias is not None and k_bias is not None and v_bias is not None:
            packed_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
            packed_name = f"{prefix}layers.{layer_id}.self_attn.qkv_proj.bias"
            tensors[packed_name] = packed_bias
            logger.info(f"Packed {packed_name} from q_proj.bias, k_proj.bias, v_proj.bias")
        
        if gate_proj_bias is not None and up_proj_bias is not None:
            packed_bias = torch.cat([gate_proj_bias, up_proj_bias], dim=0)
            packed_name = f"{prefix}layers.{layer_id}.mlp.gate_up_proj.bias"
            tensors[packed_name] = packed_bias
            logger.info(f"Packed {packed_name} from gate_proj.bias, up_proj.bias")
        

def _shard_for_tp(
    name: str,
    tensor: torch.Tensor,
    tp_size: int,
    num_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    head_dim: int,
) -> list[torch.Tensor]:
    """
    Shard a single weight tensor for TP based on its type.
    
    Reference sglang implementation:
    - QKVParallelLinear: Q gets num_heads shards, K/V get num_kv_heads shards
    - MergedColumnParallelLinear: Each sub-matrix sharded independently
    - ColumnParallelLinear: Shard dim 0 (output)
    - RowParallelLinear: Shard dim 1 (input)
    """
    if tp_size == 1:
        return [tensor]
    
    # Check for no-shard patterns (norms, some biases)
    if _should_not_shard(name):
        return [tensor.clone() for _ in range(tp_size)]
    
    # Handle QKV fused projection (most complex case)
    if "qkv_proj.weight" in name:
        return _shard_qkv_fused(tensor, tp_size, num_heads, num_kv_heads, hidden_size, head_dim)
    
    if "qkv_proj.bias" in name:
        return _shard_qkv_fused(tensor, tp_size, num_heads, num_kv_heads, hidden_size, head_dim)

    # Handle gate_up fused projection
    if "gate_up_proj.weight" in name:
        return _shard_gate_up_fused(tensor, tp_size)
    
    # Handle separate Q/K/V projections
    if any(p in name for p in ["q_proj.weight", "k_proj.weight", "v_proj.weight"]):
        return _shard_column_parallel(tensor, tp_size, dim=0)
    
    # Handle MLP and attention output (row parallel)
    if any(
        p in name
        for p in [
            "o_proj.weight",
            "down_proj.weight",
            "attn.proj.weight",
            "linear_fc2.weight",
        ]
    ):
        return _shard_row_parallel(tensor, tp_size, dim=1)
    
    # Handle MLP gate/up (column parallel)
    if any(
        p in name for p in ["gate_proj.weight", "up_proj.weight", "linear_fc1.weight"]
    ):
        return _shard_column_parallel(tensor, tp_size, dim=0)
    
    # Handle embedding and lm_head (vocab parallel)
    if any(p in name for p in ["embed_tokens.weight", "lm_head.weight", "wte.weight"]):
        return _shard_column_parallel(tensor, tp_size, dim=0)
    
    # Handle biases for column-parallel layers
    if ".bias" in name:
        # Check if this is a bias for a column-parallel layer
        weight_name = name.replace(".bias", ".weight")
        if any(
            p in weight_name
            for p in [
                "q_proj",
                "k_proj",
                "v_proj",
                "gate_proj",
                "up_proj",
                "gate_up_proj",
                "linear_fc1",
            ]
        ):
            return _shard_column_parallel(tensor, tp_size, dim=0)
        else:
            # Replicate biases for row-parallel layers
            return [tensor.clone() for _ in range(tp_size)]
    
    # Default: replicate
    logger.debug(f"Replicating {name} (no specific sharding rule)")
    return [tensor.clone() for _ in range(tp_size)]


def _should_not_shard(name: str) -> bool:
    """Check if weight should not be sharded."""
    no_shard_patterns = [
        "layernorm", "layer_norm", "ln_", "norm.weight",
        ".norm.bias", "input_layernorm", "post_attention_layernorm",
    ]
    name_lower = name.lower()
    return any(pattern in name_lower for pattern in no_shard_patterns)


def _shard_qkv_fused(
    tensor: torch.Tensor,
    tp_size: int,
    num_heads: int,
    num_kv_heads: int,
    hidden_size: int,
    head_dim: int,
) -> list[torch.Tensor]:
    """
    Shard fused QKV projection following QKVParallelLinear logic.
    
    Reference: sglang/vllm QKVParallelLinear
    - Q: num_heads * head_dim
    - K: num_kv_heads * head_dim  
    - V: num_kv_heads * head_dim
    
    Each component is sharded independently based on its head count.
    """
    if (hidden_size == 0 and head_dim == 0) or num_heads == 0:
        logger.warning("Cannot shard QKV: missing model config, replicating")
        return [tensor.clone() for _ in range(tp_size)]
    
    head_dim = head_dim or (hidden_size // num_heads)
    q_size = num_heads * head_dim
    k_size = num_kv_heads * head_dim
    v_size = num_kv_heads * head_dim
    
    expected_size = q_size + k_size + v_size
    actual_size = tensor.shape[0]
    
    if actual_size != expected_size:
        logger.warning(
            f"QKV size mismatch: expected {expected_size}, got {actual_size}. "
            f"Using simple split instead."
        )
        return _shard_column_parallel(tensor, tp_size, dim=0)
    
    if q_size % tp_size != 0:
        logger.warning(
            "Q size %s is not divisible by TP=%s for fused QKV. Falling back to simple split.",
            q_size,
            tp_size,
        )
        return _shard_column_parallel(tensor, tp_size, dim=0)

    # Split into Q, K, V
    q_tensor = tensor[:q_size]
    k_tensor = tensor[q_size:q_size + k_size]
    v_tensor = tensor[q_size + k_size:]
    
    # Shard each component
    # Q: divide by tp_size
    q_shard_size = q_size // tp_size
    q_shards = [q_tensor[i * q_shard_size:(i + 1) * q_shard_size] for i in range(tp_size)]
    
    # K and V: handle GQA (grouped-query attention)
    if tp_size <= num_kv_heads:
        if num_kv_heads % tp_size != 0:
            logger.warning(
                "num_kv_heads=%s is not divisible by TP=%s. Falling back to simple split.",
                num_kv_heads,
                tp_size,
            )
            return _shard_column_parallel(tensor, tp_size, dim=0)
        k_shard_size = head_dim * (num_kv_heads // tp_size)
        v_shard_size = head_dim * (num_kv_heads // tp_size)
        k_shards = [k_tensor[i * k_shard_size:(i + 1) * k_shard_size] for i in range(tp_size)]
        v_shards = [v_tensor[i * v_shard_size:(i + 1) * v_shard_size] for i in range(tp_size)]
    else:
        # num_kv_heads < tp_size: replicate KV heads
        if tp_size % num_kv_heads != 0:
            logger.warning(
                "TP=%s is not divisible by num_kv_heads=%s. Falling back to simple split.",
                tp_size,
                num_kv_heads,
            )
            return _shard_column_parallel(tensor, tp_size, dim=0)
        replicas_per_head = tp_size // num_kv_heads
        k_shards = []
        v_shards = []
        for i in range(tp_size):
            kv_head_idx = i // replicas_per_head
            k_start = kv_head_idx * head_dim
            k_shards.append(k_tensor[k_start:k_start + head_dim])
            v_shards.append(v_tensor[k_start:k_start + head_dim])
    
    # Concatenate Q, K, V for each TP rank
    result = []
    for q, k, v in zip(q_shards, k_shards, v_shards):
        combined = torch.cat([q, k, v], dim=0).contiguous()
        result.append(combined)
    
    return result


def _shard_gate_up_fused(tensor: torch.Tensor, tp_size: int) -> list[torch.Tensor]:
    """
    Shard fused gate+up projection following MergedColumnParallelLinear logic.
    
    Reference: sglang/vllm MergedColumnParallelLinear
    Gate and Up are two independent matrices concatenated along dim 0.
    Each should be sharded independently.
    """
    # Assume gate and up have equal size (typical for most models)
    total_size = tensor.shape[0]
    if total_size % 2 != 0:
        logger.warning(f"gate_up_proj size {total_size} not divisible by 2, using simple split")
        return _shard_column_parallel(tensor, tp_size, dim=0)
    
    gate_size = total_size // 2
    up_size = total_size // 2
    
    # Split into gate and up
    gate_tensor = tensor[:gate_size]
    up_tensor = tensor[gate_size:]
    
    # Shard each independently
    gate_shard_size = gate_size // tp_size
    up_shard_size = up_size // tp_size
    
    result = []
    for i in range(tp_size):
        gate_shard = gate_tensor[i * gate_shard_size:(i + 1) * gate_shard_size]
        up_shard = up_tensor[i * up_shard_size:(i + 1) * up_shard_size]
        combined = torch.cat([gate_shard, up_shard], dim=0).contiguous()
        result.append(combined)
    
    return result


def _shard_column_parallel(tensor: torch.Tensor, tp_size: int, dim: int = 0) -> list[torch.Tensor]:
    """Shard tensor along specified dimension (default dim 0 for column parallel)."""
    if tensor.shape[dim] % tp_size != 0:
        logger.warning(
            f"Dimension {dim} size {tensor.shape[dim]} not divisible by TP={tp_size}, "
            f"replicating instead"
        )
        return [tensor.clone() for _ in range(tp_size)]
    
    shard_size = tensor.shape[dim] // tp_size
    shards = torch.split(tensor, shard_size, dim=dim)
    return [shard.contiguous() for shard in shards]


def _shard_row_parallel(tensor: torch.Tensor, tp_size: int, dim: int = 1) -> list[torch.Tensor]:
    """Shard tensor along specified dimension (default dim 1 for row parallel)."""
    if tensor.shape[dim] % tp_size != 0:
        logger.warning(
            f"Dimension {dim} size {tensor.shape[dim]} not divisible by TP={tp_size}, "
            f"replicating instead"
        )
        return [tensor.clone() for _ in range(tp_size)]
    
    shard_size = tensor.shape[dim] // tp_size
    shards = torch.split(tensor, shard_size, dim=dim)
    return [shard.contiguous() for shard in shards]


class PersistentParameterServer:
    """
    A persistent parameter server that stores tensors in GPU memory and provides
    IPC access to client processes.
    
    The server runs as a daemon process and keeps tensors in GPU memory even when
    business processes exit. Business processes can reconnect and retrieve tensor
    handles for zero-copy access.
    
    Note: Server is read-only for clients. Tensors must be initialized at server startup.
    """
    
    def __init__(
        self,
        tensors: dict[str, torch.Tensor] | None = None,
        checkpoint_path: str | None = None,
        device_ids: list[int] = [0],
        zmq_port: int = 5555,
        zmq_host: str = "127.0.0.1",
        tp: int = 1,
        pp: int = 1,
    ):
        """
        Initialize the persistent parameter server with tensors.
        
        Args:
            tensors: Dictionary of tensor name -> tensor to store in GPU memory.
                     If None, must provide checkpoint_path.
            checkpoint_path: Path to checkpoint file or directory to load tensors from.
                           Supports .safetensors, .pt, .pth files and sharded checkpoints.
            device_ids: List of GPU device IDs to use for storing tensors
            zmq_port: Port for ZMQ communication
            zmq_host: Host address for ZMQ binding
            
        Note: Either tensors or checkpoint_path must be provided.
        """
        if tensors is None and checkpoint_path is None:
            raise ValueError("Either 'tensors' or 'checkpoint_path' must be provided")
        self.device_ids = device_ids
        self.weights_list: list[dict[str, torch.Tensor]] = []
        
        # Load tensors from checkpoint if path is provided
        if checkpoint_path is not None:
            #TODO: fuse load tensors and resharding to reduce cpu tensor move cost
            weights = load_tensors_from_checkpoint(checkpoint_path)
            config = None
            config_file = _find_model_config_path(checkpoint_path)
            if config_file is not None:
                with open(config_file, "r") as f:
                    config = json.load(f)
                logger.info("Using model config: %s", config_file)

            if config is None:
                logger.warning(
                    "config.json not found for checkpoint '%s'. Falling back to tensor-shape detection.",
                    checkpoint_path,
                )
            self.weights_list = reshard_weights(weights, tp=tp, pp=pp, model_config=config)
        else:
            if len(device_ids) == 1:
                self.weights_list = [tensors]
            else:
                # Replicate in the multi-device raw-tensor case.
                self.weights_list = [
                    {k: v.clone() for k, v in tensors.items()}
                    for _ in range(len(device_ids))
                ]

        assert len(self.weights_list) == len(device_ids), "Number of device_ids must match number of shards"

        # Storage for tensors - move all tensors to GPUs if needed
        for i in range(len(self.weights_list)):
            device = torch.device(f"cuda:{device_ids[i]}" if torch.cuda.is_available() else "cpu")
            for name, tensor in self.weights_list[i].items():
                self.weights_list[i][name] = tensor.to(device)

        # Pre-compute IPC handles for all tensors to avoid repeated serialization
        # {device id: tensor handles}
        self.tensor_handles: list[dict[str, tuple[Callable, tuple]]] = [{} for _ in range(len(self.weights_list))]
        self._prepare_all_handles()
        
        # ZMQ context and socket
        self.zmq_ctx = zmq.Context()
        self.zmq_host = zmq_host
        self.zmq_port = zmq_port
        self.zmq_addr = f"tcp://{zmq_host}:{zmq_port}"
        self.socket: zmq.Socket | None = None
        
        # Control flags
        self.running = False
        self.server_thread: threading.Thread | None = None
        
        logger.info(
            f"Initialized PersistentParameterServer on with TP={tp}, PP={pp}"
        )
    
    def _prepare_all_handles(self):
        """
        Pre-compute IPC handles for all tensors to avoid repeated serialization.
        This is called during initialization.
        """
        from torch.multiprocessing.reductions import reduce_tensor
        
        logger.info(f"Preparing IPC handles for tensors")

        for i in range(len(self.weights_list)):
            tensors = self.weights_list[i]
            tensors_handle = {}
            for name, tensor in tensors.items():
                try:
                    ipc_handle = reduce_tensor(tensor)
                    tensors_handle[name] = ipc_handle
                    logger.debug(f"Prepared IPC handle for tensor '{name}'")
                except Exception as e:
                    logger.error(f"Failed to prepare IPC handle for '{name}': {e}")
                    raise

            self.tensor_handles[i] = tensors_handle

        logger.info(f"Successfully prepared {len(self.tensor_handles)} IPC handles")
    
    def get_all_handles(self, rank: int = 0) -> dict[str, tuple[Callable, tuple]]:
        """
        Get all pre-computed IPC handles for a specific rank.
        
        Args:
            rank: The rank index (0-based) to get handles for.
                  For TP and PP, rank = pp_rank * tp_size + tp_rank
        
        Returns:
            Dictionary mapping tensor name to IPC handle (function, args)
        """
        if rank < 0 or rank >= len(self.tensor_handles):
            raise ValueError(
                f"Invalid rank {rank}. Must be in range [0, {len(self.tensor_handles)})"
            )
        return self.tensor_handles[rank].copy()
    
    def get_tensor_info(self, name: str, rank: int = 0) -> dict[str, Any] | None:
        """
        Get information about a tensor without retrieving it.
        
        Args:
            name: Name of the tensor
            rank: The rank index (0-based) to get tensor info from
            
        Returns:
            Dictionary with tensor metadata or None if not found
        """
        if rank < 0 or rank >= len(self.weights_list):
            return None
            
        if name not in self.weights_list[rank]:
            return None
        
        tensor = self.weights_list[rank][name]
        return {
            "name": name,
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "nbytes": tensor.nbytes,
        }
    
    def list_tensors(self, rank: int = 0) -> list[str]:
        """
        List all registered tensor names for a specific rank.
        
        Args:
            rank: The rank index (0-based) to list tensors from
        
        Returns:
            List of tensor names
        """
        if rank < 0 or rank >= len(self.weights_list):
            return []
        return list(self.weights_list[rank].keys())
    
    def _get_ipc_handle(self, name: str, rank: int = 0) -> tuple[Callable, tuple] | None:
        """
        Get IPC handle for a tensor that can be shared with other processes.
        
        Args:
            name: Name of the tensor
            rank: The rank index (0-based) to get handle from
            
        Returns:
            IPC handle tuple (function, args) or None if not found
        """
        if rank < 0 or rank >= len(self.tensor_handles):
            logger.error(f"Invalid rank {rank}")
            return None
            
        if name not in self.tensor_handles[rank]:
            logger.error(f"Tensor '{name}' not found in rank {rank}")
            return None
        
        return self.tensor_handles[rank][name]
    
    def _handle_request(self, request: dict) -> dict:
        """
        Handle a client request.
        
        Args:
            request: Request dictionary with 'command' and optional parameters
            
        Returns:
            Response dictionary
        """
        command = request.get("command")
        
        if command == "get_ipc_handle":
            # Client wants to get IPC handle for a tensor
            name = request.get("name")
            rank = request.get("rank", 0)
            if name is None:
                return {"status": "error", "message": "Missing name"}
            
            ipc_handle = self._get_ipc_handle(name, rank)
            if ipc_handle is None:
                return {"status": "error", "message": f"Tensor '{name}' not found in rank {rank}"}
            
            return {
                "status": "success",
                "ipc_handle": ipc_handle,
                "info": self.get_tensor_info(name, rank),
            }
        
        elif command == "get_info":
            name = request.get("name")
            rank = request.get("rank", 0)
            if name is None:
                return {"status": "error", "message": "Missing name"}
            
            info = self.get_tensor_info(name, rank)
            if info is None:
                return {"status": "error", "message": f"Tensor '{name}' not found"}
            
            return {"status": "success", "info": info}
        
        elif command == "list":
            rank = request.get("rank", 0)
            return {"status": "success", "tensors": self.list_tensors(rank)}
        
        elif command == "get_all_handles":
            # Client wants to get all IPC handles at once for a specific rank
            rank = request.get("rank", 0)
            try:
                all_handles = self.get_all_handles(rank)
                # Also include tensor info for all tensors
                all_info = {name: self.get_tensor_info(name, rank) for name in all_handles.keys()}
                
                return {
                    "status": "success",
                    "handles": all_handles,
                    "info": all_info,
                    "rank": rank,
                }
            except Exception as e:
                logger.error(f"Failed to get all handles for rank {rank}: {e}")
                return {"status": "error", "message": str(e)}
        
        elif command == "shutdown":
            return {"status": "success", "message": "Shutting down server"}
        
        else:
            return {"status": "error", "message": f"Unknown command: {command}"}
    
    def _server_loop(self):
        """Main server loop that handles client requests."""
        self.socket = self.zmq_ctx.socket(zmq.REP)
        self.socket.bind(self.zmq_addr)
        logger.info(f"Parameter server listening on {self.zmq_addr}")
        
        while self.running:
            try:
                # Wait for request with timeout
                if self.socket.poll(timeout=1000):  # 1 second timeout
                    request = self.socket.recv_pyobj()
                    logger.debug(f"Received request: {request.get('command')}")
                    
                    response = self._handle_request(request)
                    self.socket.send_pyobj(response)
                    
                    # Check for shutdown command
                    if request.get("command") == "shutdown":
                        logger.info("Shutdown command received")
                        self.running = False
                        break
            except zmq.ZMQError as e:
                if self.running:  # Only log if we're still supposed to be running
                    logger.error(f"ZMQ error in server loop: {e}")
            except Exception as e:
                logger.error(f"Error in server loop: {e}")
                # Send error response if possible
                try:
                    self.socket.send_pyobj({
                        "status": "error",
                        "message": str(e)
                    })
                except Exception:
                    pass
        
        # Cleanup
        if self.socket:
            self.socket.close()
        logger.info("Parameter server stopped")
    
    def start(self, daemon: bool = True):
        """
        Start the parameter server in a separate thread.
        
        Args:
            daemon: If True, run as daemon thread
        """
        if self.running:
            logger.warning("Server is already running")
            return
        
        self.running = True
        self.server_thread = threading.Thread(target=self._server_loop, daemon=daemon)
        self.server_thread.start()
        logger.info("Parameter server started")
    
    def stop(self):
        """Stop the parameter server."""
        if not self.running:
            logger.warning("Server is not running")
            return
        
        self.running = False
        if self.server_thread:
            self.server_thread.join(timeout=5)
        
        logger.info("Parameter server stopped")
    
    def cleanup(self):
        """Clean up resources."""
        self.stop()
        
        # Clear all tensors
        for weights in self.weights_list:
            weights.clear()
        self.weights_list.clear()
        gc.collect()
        # Close ZMQ context
        self.zmq_ctx.term()
        
        logger.info("Parameter server cleaned up")


class ParameterServerClient:
    """
    Client for connecting to a persistent parameter server.
    
    Business processes use this client to retrieve IPC handles and access 
    tensors with zero-copy. Client can only read tensors, not modify the server.
    """
    
    def __init__(
        self,
        zmq_host: str = "127.0.0.1",
        zmq_port: int = 5555,
        device_id: int | None = None,
    ):
        """
        Initialize the parameter server client.
        
        Args:
            zmq_host: Host address of the parameter server
            zmq_port: Port of the parameter server
            device_id: Local GPU device ID (for rebuilding IPC tensors)
        """
        self.zmq_ctx = zmq.Context()
        self.zmq_addr = f"tcp://{zmq_host}:{zmq_port}"
        self.socket = self.zmq_ctx.socket(zmq.REQ)
        self.socket.connect(self.zmq_addr)
        self.device_id = device_id
        
        logger.info(f"Connected to parameter server at {self.zmq_addr}")
    
    def _send_request(self, request: dict) -> dict:
        """Send a request to the server and get response."""
        self.socket.send_pyobj(request)
        response = self.socket.recv_pyobj()
        return response
    
    def get_tensor(self, name: str, rank: int = 0) -> torch.Tensor | None:
        """
        Get a tensor from the parameter server via IPC (zero-copy).
        
        Args:
            name: Name of the tensor to retrieve
            rank: The rank index (0-based) to get tensor from
            
        Returns:
            The tensor or None if not found
        """
        response = self._send_request({
            "command": "get_ipc_handle",
            "name": name,
            "rank": rank,
        })
        
        if response.get("status") != "success":
            logger.error(f"Failed to get tensor: {response.get('message')}")
            return None
        
        ipc_handle = response.get("ipc_handle")
        if ipc_handle is None:
            return None
        
        # Rebuild tensor from IPC handle
        try:
            func, args = ipc_handle
            list_args = list(args)
            
            # Update device ID if specified
            if self.device_id is not None and len(list_args) > 6:
                list_args[6] = self.device_id
            
            tensor = func(*list_args)
            return tensor
        except Exception as e:
            logger.error(f"Failed to rebuild tensor from IPC handle: {e}")
            return None
    
    def get_tensor_info(self, name: str, rank: int = 0) -> dict | None:
        """
        Get information about a tensor without retrieving it.
        
        Args:
            name: Name of the tensor
            rank: The rank index (0-based) to get tensor info from
            
        Returns:
            Dictionary with tensor metadata or None if not found
        """
        response = self._send_request({
            "command": "get_info",
            "name": name,
            "rank": rank,
        })
        
        if response.get("status") == "success":
            return response.get("info")
        return None
    
    def list_tensors(self, rank: int = 0) -> list[str]:
        """
        List all tensors in the parameter server for a specific rank.
        
        Args:
            rank: The rank index (0-based) to list tensors from
        
        Returns:
            List of tensor names
        """
        response = self._send_request({
            "command": "list",
            "rank": rank,
        })
        
        if response.get("status") == "success":
            return response.get("tensors", [])
        return []
    
    def get_all_tensors(self, rank: int = 0) -> dict[str, torch.Tensor]:
        """
        Get all tensors for a specific rank from the parameter server via IPC (zero-copy) in a single request.
        This is more efficient than calling get_tensor() multiple times.
        
        Args:
            rank: The rank index (0-based) to get tensors for.
                  For TP and PP, rank = pp_rank * tp_size + tp_rank
        
        Returns:
            Dictionary mapping tensor name to tensor
        """
        response = self._send_request({
            "command": "get_all_handles",
            "rank": rank,
        })
        
        if response.get("status") != "success":
            logger.error(f"Failed to get all tensors: {response.get('message')}")
            return {}
        
        handles = response.get("handles", {})
        if not handles:
            logger.warning("No tensor handles returned from server")
            return {}
        
        # Rebuild all tensors from IPC handles
        tensors = {}
        for name, ipc_handle in handles.items():
            try:
                func, args = ipc_handle
                list_args = list(args)
                
                # Update device ID if specified
                if self.device_id is not None and len(list_args) > 6:
                    list_args[6] = self.device_id
                
                tensor = func(*list_args)
                tensors[name] = tensor
                # logger.info(f"Rebuilt tensor '{tensor.device}' from IPC handle")
            except Exception as e:
                logger.error(f"Failed to rebuild tensor '{name}' from IPC handle: {e}")
                continue
        
        logger.info(f"Successfully retrieved {len(tensors)} tensors via IPC in one request")
        return tensors
    
    def shutdown_server(self) -> bool:
        """
        Send shutdown command to the server.
        
        Returns:
            True if successful
        """
        response = self._send_request({
            "command": "shutdown",
        })
        
        return response.get("status") == "success"
    
    def close(self):
        """Close the client connection."""
        self.socket.close()
        self.zmq_ctx.term()
        logger.info("Client connection closed")

if __name__ == "__main__":
    checkpoint_path = "/mnt/shared-storage-user/ailab-sys/zhoujiecheng/aliyun_data/models/Qwen-1.5B-Distilled"
    weights = load_tensors_from_checkpoint(checkpoint_path, 0)
    config_file = os.path.join(checkpoint_path, "config.json")
    with open(config_file, "r") as f:
        config = json.load(f)
    weights_list = reshard_weights(weights, tp=4, pp=1, model_config=config)
