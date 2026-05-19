"""
Test the reshard_weights function for TP and PP sharding.

This demonstrates how model weights are sharded across different
tensor parallel and pipeline parallel ranks.
"""

import torch
from loguru import logger

from checkpoint_engine.persistent_ps import reshard_weights


def create_mock_llama_weights(num_layers: int = 4, hidden_size: int = 512, vocab_size: int = 1000):
    """Create mock weights for a Llama-style model."""
    intermediate_size = hidden_size * 4
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    weights = {}
    
    # Embedding
    weights["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
    
    # Transformer layers
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        
        # Attention
        weights[f"{prefix}.self_attn.q_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.self_attn.k_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.self_attn.v_proj.weight"] = torch.randn(hidden_size, hidden_size)
        weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        
        # MLP
        weights[f"{prefix}.mlp.gate_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        weights[f"{prefix}.mlp.up_proj.weight"] = torch.randn(intermediate_size, hidden_size)
        weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)
        
        # Layer norms
        weights[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)
    
    # Final norm and head
    weights["model.norm.weight"] = torch.randn(hidden_size)
    weights["lm_head.weight"] = torch.randn(vocab_size, hidden_size)
    
    return weights


def test_no_sharding():
    """Test with TP=1, PP=1 (no sharding)."""
    print("\n" + "=" * 80)
    print("Test 1: No Sharding (TP=1, PP=1)")
    print("=" * 80)
    
    weights = create_mock_llama_weights(num_layers=4)
    print(f"Created {len(weights)} weight tensors")
    
    sharded = reshard_weights(weights, tp=1, pp=1)
    
    assert len(sharded) == 1, "Should have 1 shard"
    assert len(sharded[0]) == len(weights), "Shard should have all weights"
    
    print("✅ Test passed: No sharding works correctly")


def test_tp_only():
    """Test with TP=4, PP=1 (tensor parallel only)."""
    print("\n" + "=" * 80)
    print("Test 2: Tensor Parallel Only (TP=4, PP=1)")
    print("=" * 80)
    
    weights = create_mock_llama_weights(num_layers=4, hidden_size=512)
    print(f"Created {len(weights)} weight tensors")
    
    tp_size = 4
    sharded = reshard_weights(weights, tp=tp_size, pp=1, num_layers=4)
    
    assert len(sharded) == tp_size, f"Should have {tp_size} shards"
    
    # Check column parallel sharding (qkv, gate, up)
    layer0_q = "model.layers.0.self_attn.q_proj.weight"
    original_shape = weights[layer0_q].shape  # (512, 512)
    
    for tp_rank in range(tp_size):
        shard_shape = sharded[tp_rank][layer0_q].shape
        print(f"TP rank {tp_rank}: {layer0_q} shape = {shard_shape}")
        assert shard_shape[0] == original_shape[0] // tp_size, \
            f"Column parallel should shard dim 0"
        assert shard_shape[1] == original_shape[1], \
            f"Column parallel should keep dim 1"
    
    # Check row parallel sharding (o_proj, down_proj)
    layer0_o = "model.layers.0.self_attn.o_proj.weight"
    original_shape = weights[layer0_o].shape  # (512, 512)
    
    for tp_rank in range(tp_size):
        shard_shape = sharded[tp_rank][layer0_o].shape
        print(f"TP rank {tp_rank}: {layer0_o} shape = {shard_shape}")
        assert shard_shape[0] == original_shape[0], \
            f"Row parallel should keep dim 0"
        assert shard_shape[1] == original_shape[1] // tp_size, \
            f"Row parallel should shard dim 1"
    
    # Check layer norm (should be replicated)
    layer0_ln = "model.layers.0.input_layernorm.weight"
    original_shape = weights[layer0_ln].shape
    
    for tp_rank in range(tp_size):
        shard_shape = sharded[tp_rank][layer0_ln].shape
        assert shard_shape == original_shape, \
            f"Layer norm should be replicated (not sharded)"
    
    print("✅ Test passed: Tensor parallel sharding works correctly")


def test_pp_only():
    """Test with TP=1, PP=2 (pipeline parallel only)."""
    print("\n" + "=" * 80)
    print("Test 3: Pipeline Parallel Only (TP=1, PP=2)")
    print("=" * 80)
    
    num_layers = 4
    weights = create_mock_llama_weights(num_layers=num_layers)
    print(f"Created {len(weights)} weight tensors for {num_layers} layers")
    
    pp_size = 2
    sharded = reshard_weights(weights, tp=1, pp=pp_size, num_layers=num_layers)
    
    assert len(sharded) == pp_size, f"Should have {pp_size} shards"
    
    # PP rank 0 should have layers 0-1
    # PP rank 1 should have layers 2-3
    
    for pp_rank in range(pp_size):
        print(f"\nPP rank {pp_rank} weights:")
        layer_weights = [name for name in sharded[pp_rank].keys() if ".layers." in name]
        
        # Extract layer indices
        layer_indices = set()
        for name in layer_weights:
            parts = name.split(".layers.")
            if len(parts) > 1:
                layer_idx = int(parts[1].split(".")[0])
                layer_indices.add(layer_idx)
        
        print(f"  Contains layers: {sorted(layer_indices)}")
        
        expected_start = pp_rank * (num_layers // pp_size)
        expected_end = (pp_rank + 1) * (num_layers // pp_size)
        expected_layers = set(range(expected_start, expected_end))
        
        assert layer_indices == expected_layers, \
            f"PP rank {pp_rank} should have layers {expected_layers}, got {layer_indices}"
        
        # Check that embedding and lm_head are replicated
        assert "model.embed_tokens.weight" in sharded[pp_rank], \
            "Embedding should be replicated to all PP ranks"
        assert "lm_head.weight" in sharded[pp_rank], \
            "LM head should be replicated to all PP ranks"
    
    print("✅ Test passed: Pipeline parallel sharding works correctly")


def test_tp_and_pp():
    """Test with TP=2, PP=2 (both tensor and pipeline parallel)."""
    print("\n" + "=" * 80)
    print("Test 4: Tensor + Pipeline Parallel (TP=2, PP=2)")
    print("=" * 80)
    
    num_layers = 4
    hidden_size = 512
    weights = create_mock_llama_weights(num_layers=num_layers, hidden_size=hidden_size)
    print(f"Created {len(weights)} weight tensors")
    
    tp_size = 2
    pp_size = 2
    sharded = reshard_weights(weights, tp=tp_size, pp=pp_size, num_layers=num_layers)
    
    total_shards = tp_size * pp_size
    assert len(sharded) == total_shards, f"Should have {total_shards} shards"
    
    # Check each combination
    for pp_rank in range(pp_size):
        for tp_rank in range(tp_size):
            shard_idx = pp_rank * tp_size + tp_rank
            print(f"\nShard [PP={pp_rank}, TP={tp_rank}] (index={shard_idx}):")
            
            # Check layer assignment (PP)
            layer_weights = [name for name in sharded[shard_idx].keys() if ".layers." in name]
            layer_indices = set()
            for name in layer_weights:
                parts = name.split(".layers.")
                if len(parts) > 1:
                    layer_idx = int(parts[1].split(".")[0])
                    layer_indices.add(layer_idx)
            
            print(f"  Layers: {sorted(layer_indices)}")
            
            # Check TP sharding
            if layer_indices:
                test_layer = min(layer_indices)
                q_proj_name = f"model.layers.{test_layer}.self_attn.q_proj.weight"
                shard_shape = sharded[shard_idx][q_proj_name].shape
                original_shape = weights[q_proj_name].shape
                
                print(f"  {q_proj_name}: {shard_shape} (original: {original_shape})")
                assert shard_shape[0] == original_shape[0] // tp_size, \
                    "Should be TP-sharded on dim 0"
    
    print("✅ Test passed: Combined TP+PP sharding works correctly")


def test_qwen_style_fused_layers():
    """Test with Qwen-style fused qkv and gate_up layers."""
    print("\n" + "=" * 80)
    print("Test 5: Fused Layers (QKV, Gate+Up)")
    print("=" * 80)
    
    hidden_size = 512
    intermediate_size = hidden_size * 4
    
    weights = {
        "model.layers.0.self_attn.qkv_proj.weight": torch.randn(hidden_size * 3, hidden_size),
        "model.layers.0.mlp.gate_up_proj.weight": torch.randn(intermediate_size * 2, hidden_size),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(hidden_size, hidden_size),
        "model.layers.0.mlp.down_proj.weight": torch.randn(hidden_size, intermediate_size),
    }
    
    tp_size = 4
    sharded = reshard_weights(weights, tp=tp_size, pp=1, num_layers=1)
    
    # Check QKV sharding
    qkv_original = weights["model.layers.0.self_attn.qkv_proj.weight"]
    for tp_rank in range(tp_size):
        qkv_shard = sharded[tp_rank]["model.layers.0.self_attn.qkv_proj.weight"]
        print(f"TP rank {tp_rank}: QKV shape = {qkv_shard.shape}")
        assert qkv_shard.shape[0] == qkv_original.shape[0] // tp_size
        assert qkv_shard.shape[1] == qkv_original.shape[1]
    
    # Check gate_up sharding
    gate_up_original = weights["model.layers.0.mlp.gate_up_proj.weight"]
    for tp_rank in range(tp_size):
        gate_up_shard = sharded[tp_rank]["model.layers.0.mlp.gate_up_proj.weight"]
        print(f"TP rank {tp_rank}: Gate+Up shape = {gate_up_shard.shape}")
        assert gate_up_shard.shape[0] == gate_up_original.shape[0] // tp_size
        assert gate_up_shard.shape[1] == gate_up_original.shape[1]
    
    print("✅ Test passed: Fused layer sharding works correctly")


def test_memory_savings():
    """Demonstrate memory savings from sharding."""
    print("\n" + "=" * 80)
    print("Test 6: Memory Savings Demonstration")
    print("=" * 80)
    
    num_layers = 32  # Typical LLM size
    hidden_size = 4096
    vocab_size = 32000
    
    weights = create_mock_llama_weights(
        num_layers=num_layers,
        hidden_size=hidden_size,
        vocab_size=vocab_size
    )
    
    # Calculate original size
    original_size = sum(t.numel() * t.element_size() for t in weights.values())
    print(f"\nOriginal model size: {original_size / 1024**3:.2f} GB")
    print(f"Number of tensors: {len(weights)}")
    
    # Test different sharding configs
    configs = [
        (1, 1),
        (2, 1),
        (4, 1),
        (8, 1),
        (1, 2),
        (1, 4),
        (2, 2),
        (4, 2),
    ]
    
    print("\n" + "-" * 60)
    print(f"{'Config':<15} {'Shards':<10} {'Size/Shard':<20} {'Reduction':<15}")
    print("-" * 60)
    
    for tp, pp in configs:
        sharded = reshard_weights(weights, tp=tp, pp=pp, num_layers=num_layers)
        
        # Calculate size of first shard (all shards should be similar)
        shard_size = sum(t.numel() * t.element_size() for t in sharded[0].values())
        reduction = original_size / shard_size if shard_size > 0 else 0
        
        print(f"TP={tp}, PP={pp:<8} {len(sharded):<10} {shard_size/1024**3:.2f} GB{'':<13} {reduction:.2f}x")
    
    print("-" * 60)
    print("✅ Memory savings demonstrated")


if __name__ == "__main__":
    # Run all tests
    test_no_sharding()
    test_tp_only()
    test_pp_only()
    test_tp_and_pp()
    test_qwen_style_fused_layers()
    test_memory_savings()
    
    print("\n" + "=" * 80)
    print("All tests passed! ✅")
    print("=" * 80)
