import torch

from checkpoint_engine.persistent_ps import reshard_weights


def _qwen3_vl_config():
    return {
        "model_type": "qwen3_vl",
        "text_config": {
            "num_hidden_layers": 1,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "hidden_size": 16,
        },
    }


def test_qwen3_prefixless_names_are_normalized_and_packed():
    model_config = {
        "model_type": "qwen3",
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 16,
    }
    weights = {
        "layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "layers.0.self_attn.k_proj.weight": torch.randn(4, 16),
        "layers.0.self_attn.v_proj.weight": torch.randn(4, 16),
    }

    sharded = reshard_weights(weights, tp=1, pp=1, model_config=model_config)[0]
    assert "model.layers.0.self_attn.qkv_proj.weight" in sharded
    assert "layers.0.self_attn.q_proj.weight" not in sharded


def test_qwen3_vl_name_mapping_for_language_and_vision():
    weights = {
        "model.language_model.layers.0.self_attn.q_proj.weight": torch.randn(16, 16),
        "model.language_model.layers.0.self_attn.k_proj.weight": torch.randn(4, 16),
        "model.language_model.layers.0.self_attn.v_proj.weight": torch.randn(4, 16),
        "model.visual.blocks.0.attn.qkv.weight": torch.randn(24, 8),
    }

    sharded = reshard_weights(weights, tp=1, pp=1, model_config=_qwen3_vl_config())[0]
    assert "model.layers.0.self_attn.qkv_proj.weight" in sharded
    assert "visual.blocks.0.attn.qkv_proj.weight" in sharded


def test_qwen3_vl_name_mapping_with_model_language_model_model_prefix():
    weights = {
        "model.language_model.model.layers.0.self_attn.q_proj.weight": torch.randn(
            16, 16
        ),
        "model.language_model.model.layers.0.self_attn.k_proj.weight": torch.randn(
            4, 16
        ),
        "model.language_model.model.layers.0.self_attn.v_proj.weight": torch.randn(
            4, 16
        ),
    }

    sharded = reshard_weights(weights, tp=1, pp=1, model_config=_qwen3_vl_config())[0]
    assert "model.layers.0.self_attn.qkv_proj.weight" in sharded


def test_qwen3_qkv_gqa_tp_sharding():
    weights = {
        "model.layers.0.self_attn.qkv_proj.weight": torch.randn(24, 16),
        "model.layers.0.self_attn.o_proj.weight": torch.randn(16, 16),
    }
    model_config = {
        "model_type": "qwen3",
        "num_hidden_layers": 1,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "hidden_size": 16,
    }

    sharded = reshard_weights(weights, tp=2, pp=1, model_config=model_config)
    assert sharded[0]["model.layers.0.self_attn.qkv_proj.weight"].shape == (12, 16)
    assert sharded[1]["model.layers.0.self_attn.qkv_proj.weight"].shape == (12, 16)
    assert sharded[0]["model.layers.0.self_attn.o_proj.weight"].shape == (16, 8)
    assert sharded[1]["model.layers.0.self_attn.o_proj.weight"].shape == (16, 8)


def test_qwen3_vl_vision_tp_sharding_patterns():
    weights = {
        "visual.blocks.0.mlp.linear_fc1.weight": torch.randn(16, 8),
        "visual.blocks.0.mlp.linear_fc2.weight": torch.randn(8, 16),
        "visual.blocks.0.attn.proj.weight": torch.randn(8, 8),
    }

    sharded = reshard_weights(weights, tp=2, pp=1, model_config=_qwen3_vl_config())
    assert sharded[0]["visual.blocks.0.mlp.linear_fc1.weight"].shape == (8, 8)
    assert sharded[1]["visual.blocks.0.mlp.linear_fc1.weight"].shape == (8, 8)

    assert sharded[0]["visual.blocks.0.mlp.linear_fc2.weight"].shape == (8, 8)
    assert sharded[1]["visual.blocks.0.mlp.linear_fc2.weight"].shape == (8, 8)

    assert sharded[0]["visual.blocks.0.attn.proj.weight"].shape == (8, 4)
    assert sharded[1]["visual.blocks.0.attn.proj.weight"].shape == (8, 4)
