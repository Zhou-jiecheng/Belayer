"""
Complete example: Load checkpoint, reshard weights, and serve via Parameter Server.

This demonstrates the full workflow:
1. Load model weights from checkpoint
2. Reshard weights for TP/PP distribution
3. Launch parameter servers on each GPU
4. Clients connect and retrieve weights
"""

import argparse
import time
from pathlib import Path

import torch
from loguru import logger

from checkpoint_engine.persistent_ps import (
    load_tensors_from_checkpoint,
    reshard_weights,
    PersistentParameterServer,
    ParameterServerClient,
)


def create_mock_checkpoint(save_path: str, num_layers: int = 4):
    """Create a mock checkpoint for testing."""
    logger.info(f"Creating mock checkpoint: {save_path}")
    
    hidden_size = 512
    vocab_size = 1000
    intermediate_size = hidden_size * 4
    
    weights = {}
    
    # Embedding
    weights["model.embed_tokens.weight"] = torch.randn(vocab_size, hidden_size)
    
    # Layers
    for layer_idx in range(num_layers):
        prefix = f"model.layers.{layer_idx}"
        
        # Attention (Qwen style - fused QKV)
        weights[f"{prefix}.self_attn.qkv_proj.weight"] = torch.randn(hidden_size * 3, hidden_size)
        weights[f"{prefix}.self_attn.qkv_proj.bias"] = torch.randn(hidden_size * 3)
        weights[f"{prefix}.self_attn.o_proj.weight"] = torch.randn(hidden_size, hidden_size)
        
        # MLP (fused gate+up)
        weights[f"{prefix}.mlp.gate_up_proj.weight"] = torch.randn(intermediate_size * 2, hidden_size)
        weights[f"{prefix}.mlp.down_proj.weight"] = torch.randn(hidden_size, intermediate_size)
        
        # Norms
        weights[f"{prefix}.input_layernorm.weight"] = torch.randn(hidden_size)
        weights[f"{prefix}.post_attention_layernorm.weight"] = torch.randn(hidden_size)
    
    # Final norm and head
    weights["model.norm.weight"] = torch.randn(hidden_size)
    weights["lm_head.weight"] = torch.randn(vocab_size, hidden_size)
    
    # Save as PyTorch checkpoint
    torch.save(weights, save_path)
    logger.info(f"Saved {len(weights)} tensors to {save_path}")
    
    return weights


def server_process(
    checkpoint_path: str,
    tp_size: int,
    pp_size: int,
    rank: int,
    num_layers: int,
):
    """
    Server process for a specific rank.
    
    Args:
        checkpoint_path: Path to model checkpoint
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        rank: Global rank (0 to tp_size*pp_size-1)
        num_layers: Number of transformer layers
    """
    pp_rank = rank // tp_size
    tp_rank = rank % tp_size
    
    logger.info(f"Starting server for rank {rank} (PP={pp_rank}, TP={tp_rank})")
    
    # Load checkpoint (only rank 0 loads, others can wait)
    if rank == 0:
        logger.info(f"Rank 0: Loading checkpoint from {checkpoint_path}")
        weights = load_tensors_from_checkpoint(checkpoint_path, device_id=0)
        
        logger.info(f"Rank 0: Resharding weights (TP={tp_size}, PP={pp_size})")
        shards = reshard_weights(weights, tp=tp_size, pp=pp_size, num_layers=num_layers)
        
        # In real implementation, broadcast shards to other ranks
        # For this demo, we just return shard 0
        shard_weights = shards[rank]
    else:
        # In real implementation, receive shard from rank 0
        # For this demo, we skip
        logger.info(f"Rank {rank}: Waiting for shard (mock)")
        return
    
    # Start parameter server
    port = 5555 + rank
    device_id = rank % torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    logger.info(f"Rank {rank}: Starting parameter server on port {port}")
    server = PersistentParameterServer(
        tensors=shard_weights,
        device_id=device_id,
        zmq_port=port,
    )
    
    server.start(daemon=False)
    
    # Keep server running
    try:
        if server.server_thread:
            server.server_thread.join()
    except KeyboardInterrupt:
        logger.info(f"Rank {rank}: Shutting down")
        server.cleanup()


def client_demo(num_servers: int, base_port: int = 5555):
    """
    Demo client that connects to all servers and retrieves weights.
    
    Args:
        num_servers: Number of parameter servers running
        base_port: Base port number (servers on base_port + rank)
    """
    logger.info(f"Client: Connecting to {num_servers} servers")
    
    # Connect to all servers
    clients = []
    for rank in range(num_servers):
        port = base_port + rank
        logger.info(f"Connecting to server on port {port}")
        client = ParameterServerClient(zmq_port=port)
        clients.append(client)
        time.sleep(0.1)
    
    # Method 1: Get all tensors from each server (efficient)
    logger.info("\n" + "=" * 80)
    logger.info("Method 1: Batch retrieval (get_all_tensors)")
    logger.info("=" * 80)
    
    all_shards = []
    for rank, client in enumerate(clients):
        start = time.time()
        shard_weights = client.get_all_tensors()
        elapsed = time.time() - start
        
        logger.info(
            f"Rank {rank}: Retrieved {len(shard_weights)} tensors "
            f"in {elapsed:.4f}s"
        )
        all_shards.append(shard_weights)
    
    # Method 2: Get tensors individually (for comparison)
    logger.info("\n" + "=" * 80)
    logger.info("Method 2: Individual retrieval (get_tensor)")
    logger.info("=" * 80)
    
    for rank, client in enumerate(clients):
        tensor_names = client.list_tensors()
        logger.info(f"Rank {rank}: {len(tensor_names)} tensors available")
        
        if len(tensor_names) > 0:
            # Get first tensor as example
            first_name = tensor_names[0]
            start = time.time()
            tensor = client.get_tensor(first_name)
            elapsed = time.time() - start
            
            if tensor is not None:
                logger.info(
                    f"  {first_name}: shape={tensor.shape}, "
                    f"retrieved in {elapsed:.4f}s"
                )
    
    # Demonstrate zero-copy verification
    logger.info("\n" + "=" * 80)
    logger.info("Zero-copy verification")
    logger.info("=" * 80)
    
    for rank, client in enumerate(clients):
        tensor_names = client.list_tensors()
        if len(tensor_names) == 0:
            continue
        
        name = tensor_names[0]
        
        # Get same tensor twice
        tensor1 = client.get_tensor(name)
        tensor2 = client.get_tensor(name)
        
        if tensor1 is not None and tensor2 is not None:
            same_memory = tensor1.data_ptr() == tensor2.data_ptr()
            logger.info(
                f"Rank {rank}, {name}: "
                f"data_ptr match = {same_memory} "
                f"(ptr={hex(tensor1.data_ptr())})"
            )
    
    # Cleanup
    logger.info("\n" + "=" * 80)
    logger.info("Shutting down servers")
    logger.info("=" * 80)
    
    for rank, client in enumerate(clients):
        logger.info(f"Shutting down server {rank}")
        client.shutdown_server()
        client.close()


def full_workflow_demo():
    """Complete workflow demonstration."""
    print("\n" + "=" * 80)
    print("Complete Workflow Demo: Checkpoint → Reshard → Parameter Server")
    print("=" * 80)
    
    # Configuration
    num_layers = 4
    tp_size = 2
    pp_size = 2
    total_ranks = tp_size * pp_size
    
    # Step 1: Create mock checkpoint
    checkpoint_path = "/tmp/mock_model.pt"
    logger.info(f"\nStep 1: Creating mock checkpoint ({num_layers} layers)")
    weights = create_mock_checkpoint(checkpoint_path, num_layers=num_layers)
    
    original_size = sum(t.numel() * t.element_size() for t in weights.values())
    logger.info(f"Original model size: {original_size / 1024**2:.2f} MB")
    
    # Step 2: Load checkpoint
    logger.info(f"\nStep 2: Loading checkpoint")
    loaded_weights = load_tensors_from_checkpoint(checkpoint_path, device_id=0)
    logger.info(f"Loaded {len(loaded_weights)} tensors")
    
    # Step 3: Reshard weights
    logger.info(f"\nStep 3: Resharding (TP={tp_size}, PP={pp_size})")
    shards = reshard_weights(
        loaded_weights,
        tp=tp_size,
        pp=pp_size,
        num_layers=num_layers
    )
    
    logger.info(f"Created {len(shards)} shards")
    for idx, shard in enumerate(shards):
        pp_rank = idx // tp_size
        tp_rank = idx % tp_size
        shard_size = sum(t.numel() * t.element_size() for t in shard.values())
        logger.info(
            f"  Shard {idx} [PP={pp_rank}, TP={tp_rank}]: "
            f"{len(shard)} tensors, {shard_size / 1024**2:.2f} MB"
        )
    
    # Step 4: Start parameter servers (simplified - only rank 0)
    logger.info(f"\nStep 4: Starting parameter servers")
    servers = []
    
    for rank in range(min(total_ranks, 1)):  # Only start rank 0 for demo
        port = 5555 + rank
        logger.info(f"Starting server for rank {rank} on port {port}")
        
        server = PersistentParameterServer(
            tensors=shards[rank],
            device_id=0,  # Use GPU 0 for demo
            zmq_port=port,
        )
        server.start()
        servers.append(server)
        time.sleep(0.5)
    
    # Step 5: Client demo
    logger.info(f"\nStep 5: Client connecting and retrieving weights")
    time.sleep(1)
    
    client = ParameterServerClient(zmq_port=5555)
    
    # List tensors
    tensor_names = client.list_tensors()
    logger.info(f"Client: Server has {len(tensor_names)} tensors")
    
    # Get all tensors (efficient)
    start = time.time()
    all_tensors = client.get_all_tensors()
    batch_time = time.time() - start
    logger.info(f"Batch retrieval: {len(all_tensors)} tensors in {batch_time:.4f}s")
    
    # Get tensors individually (for comparison)
    start = time.time()
    individual_tensors = {}
    for name in tensor_names:
        individual_tensors[name] = client.get_tensor(name)
    individual_time = time.time() - start
    logger.info(f"Individual retrieval: {len(individual_tensors)} tensors in {individual_time:.4f}s")
    
    if individual_time > 0:
        speedup = individual_time / batch_time
        logger.info(f"Batch method is {speedup:.2f}x faster!")
    
    # Step 6: Cleanup
    logger.info(f"\nStep 6: Cleanup")
    client.shutdown_server()
    client.close()
    
    for server in servers:
        server.cleanup()
    
    # Clean up checkpoint file
    Path(checkpoint_path).unlink(missing_ok=True)
    
    print("\n" + "=" * 80)
    print("✅ Complete workflow demo finished successfully!")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Server Complete Demo")
    parser.add_argument(
        "--mode",
        choices=["full", "server", "client"],
        default="full",
        help="Demo mode",
    )
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--tp", type=int, default=2, help="Tensor parallel size")
    parser.add_argument("--pp", type=int, default=2, help="Pipeline parallel size")
    parser.add_argument("--rank", type=int, default=0, help="Rank for server mode")
    parser.add_argument("--num-layers", type=int, default=4, help="Number of layers")
    
    args = parser.parse_args()
    
    if args.mode == "full":
        # Run complete workflow demo
        full_workflow_demo()
    
    elif args.mode == "server":
        # Run as server for specific rank
        if args.checkpoint is None:
            print("Error: --checkpoint required for server mode")
            exit(1)
        
        server_process(
            checkpoint_path=args.checkpoint,
            tp_size=args.tp,
            pp_size=args.pp,
            rank=args.rank,
            num_layers=args.num_layers,
        )
    
    elif args.mode == "client":
        # Run as client
        num_servers = args.tp * args.pp
        client_demo(num_servers=num_servers)
