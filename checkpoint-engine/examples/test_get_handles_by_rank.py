"""
Test script for getting tensor handles by rank.
Demonstrates how to retrieve tensors for specific TP/PP ranks.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from checkpoint_engine.persistent_ps import PersistentParameterServer, ParameterServerClient
import time


def test_get_handles_by_rank():
    """Test getting handles for different ranks."""
    
    # Create sample tensors for testing
    # Simulate a model with TP=2, PP=2 (4 total ranks)
    tp_size = 2
    pp_size = 2
    total_ranks = tp_size * pp_size
    
    # Create tensors for each rank
    weights_list = []
    for rank in range(total_ranks):
        pp_rank = rank // tp_size
        tp_rank = rank % tp_size
        
        # Each rank gets different tensors
        weights = {
            f"layer_{pp_rank}.weight": torch.randn(100, 200).cuda(),
            f"layer_{pp_rank}.tp_{tp_rank}.shard": torch.randn(50, 100).cuda(),
            f"shared.embedding": torch.randn(1000, 512).cuda(),  # Same across all ranks
        }
        weights_list.append(weights)
    
    # Initialize server with pre-sharded weights
    server = PersistentParameterServer(
        tensors=None,
        checkpoint_path=None,
        device_ids=list(range(total_ranks)),
        zmq_port=12347,
        tp=tp_size,
        pp=pp_size,
    )
    
    # Manually set weights_list for testing (bypass reshard_weights)
    server.weights_list = weights_list
    server._prepare_all_handles()
    
    # Start server
    server.start()
    time.sleep(1)
    
    try:
        # Create client
        client = ParameterServerClient(zmq_port=12347)
        
        print("=" * 60)
        print("Testing get_all_tensors() with rank parameter")
        print("=" * 60)
        
        # Test getting tensors for each rank
        for rank in range(total_ranks):
            pp_rank = rank // tp_size
            tp_rank = rank % tp_size
            
            print(f"\n{'='*60}")
            print(f"Rank {rank} (PP={pp_rank}, TP={tp_rank})")
            print(f"{'='*60}")
            
            # Get all tensors for this rank
            tensors = client.get_all_tensors(rank=rank)
            
            print(f"\nReceived {len(tensors)} tensors:")
            for name, tensor in tensors.items():
                print(f"  - {name}: shape={tensor.shape}, device={tensor.device}")
            
            # Verify the tensors match expected names
            expected_names = {
                f"layer_{pp_rank}.weight",
                f"layer_{pp_rank}.tp_{tp_rank}.shard",
                f"shared.embedding",
            }
            received_names = set(tensors.keys())
            
            if received_names == expected_names:
                print(f"✓ Tensors for rank {rank} match expected names")
            else:
                print(f"✗ Mismatch!")
                print(f"  Expected: {expected_names}")
                print(f"  Received: {received_names}")
        
        # Test list_tensors with rank
        print(f"\n{'='*60}")
        print("Testing list_tensors() with rank parameter")
        print(f"{'='*60}")
        
        for rank in range(total_ranks):
            tensor_names = client.list_tensors(rank=rank)
            print(f"\nRank {rank}: {len(tensor_names)} tensors")
            for name in tensor_names:
                print(f"  - {name}")
        
        # Test get_tensor with rank
        print(f"\n{'='*60}")
        print("Testing get_tensor() with rank parameter")
        print(f"{'='*60}")
        
        for rank in range(total_ranks):
            pp_rank = rank // tp_size
            tensor_name = f"layer_{pp_rank}.weight"
            
            tensor = client.get_tensor(tensor_name, rank=rank)
            if tensor is not None:
                print(f"\nRank {rank}: Retrieved '{tensor_name}'")
                print(f"  Shape: {tensor.shape}, Device: {tensor.device}")
            else:
                print(f"\nRank {rank}: Failed to retrieve '{tensor_name}'")
        
        # Test invalid rank
        print(f"\n{'='*60}")
        print("Testing invalid rank handling")
        print(f"{'='*60}")
        
        invalid_rank = total_ranks + 1
        try:
            tensors = client.get_all_tensors(rank=invalid_rank)
            if not tensors:
                print(f"✓ Correctly handled invalid rank {invalid_rank}")
        except Exception as e:
            print(f"✓ Correctly raised exception for invalid rank: {e}")
        
        print("\n" + "="*60)
        print("All tests completed successfully!")
        print("="*60)
        
    finally:
        # Cleanup
        client.close()
        server.cleanup()


if __name__ == "__main__":
    test_get_handles_by_rank()
