"""
Example usage of the Persistent Parameter Server.

This example demonstrates:
1. Starting a parameter server with pre-initialized tensors
2. Business processes retrieve tensors via IPC (zero-copy)
3. Business process exits
4. New business process retrieves the same tensors
"""

import time
import multiprocessing as mp

import torch
from loguru import logger

from checkpoint_engine.persistent_ps import (
    PersistentParameterServer,
    ParameterServerClient,
)


def server_process(port: int = 5555):
    """Run the parameter server with pre-initialized tensors."""
    logger.info("Starting parameter server...")
    
    # Initialize tensors (in real use case, load from checkpoint)
    tensors = {
        "weights": torch.randn(1000, 1000, dtype=torch.float32),
        "indices": torch.arange(0, 100, dtype=torch.int64),
        "mask": torch.ones(500, 500, dtype=torch.float16),
    }
    
    server = PersistentParameterServer(
        tensors=tensors,
        device_id=0,
        zmq_port=port,
    )
    
    server.start(daemon=False)
    
    # Keep server running
    if server.server_thread:
        server.server_thread.join()


def business_process_1(port: int = 5555):
    """First business process: retrieves and uses tensors."""
    logger.info("Business Process 1: Starting...")
    time.sleep(1)  # Wait for server to start
    
    # Connect to parameter server
    client = ParameterServerClient(zmq_port=port, device_id=0)
    
    # List all available tensors
    tensors = client.list_tensors()
    logger.info(f"Available tensors: {tensors}")
    
    # Get tensor info
    for name in tensors:
        info = client.get_tensor_info(name)
        logger.info(f"Tensor '{name}': {info}")
    
    # Retrieve tensors via IPC (zero-copy)
    logger.info("Retrieving tensors via IPC (zero-copy)...")
    
    weights = client.get_tensor("weights")
    indices = client.get_tensor("indices")
    mask = client.get_tensor("mask")
    
    if weights is not None:
        logger.info(f"Retrieved 'weights': shape={weights.shape}, dtype={weights.dtype}")
        logger.info(f"First few values: {weights[0, :5]}")
        
        # Modify tensor in place to demonstrate zero-copy
        logger.info("Modifying 'weights' tensor in place...")
        weights[0, 0] = 999.0
        logger.info(f"Modified value: weights[0, 0] = {weights[0, 0]}")
    
    if indices is not None:
        logger.info(f"Retrieved 'indices': shape={indices.shape}, dtype={indices.dtype}")
        logger.info(f"First few values: {indices[:10]}")
    
    if mask is not None:
        logger.info(f"Retrieved 'mask': shape={mask.shape}, dtype={mask.dtype}")
    
    client.close()
    logger.info("Business Process 1: Completed and exiting...")



def business_process_2(port: int = 5555):
    """Second business process: verifies zero-copy by reading modified tensor."""
    logger.info("Business Process 2: Starting...")
    time.sleep(3)  # Wait for business process 1 to finish
    
    # Connect to parameter server
    client = ParameterServerClient(zmq_port=port, device_id=0)
    
    # List available tensors
    tensors = client.list_tensors()
    logger.info(f"Available tensors: {tensors}")
    
    # Retrieve weights tensor to verify modification
    logger.info("Verifying tensor modification from previous process...")
    weights = client.get_tensor("weights")
    
    if weights is not None:
        logger.info(f"Verification: weights[0, 0] = {weights[0, 0]}")
        if weights[0, 0].item() == 999.0:
            logger.info("✓ Modification verified! Zero-copy sharing works!")
        else:
            logger.warning("✗ Modification not found. Something went wrong.")
        
        # Further modify to demonstrate persistent state
        weights[0, 1] = 888.0
        logger.info(f"Modified weights[0, 1] = {weights[0, 1]}")
    
    client.close()
    logger.info("Business Process 2: Completed")



def business_process_3(port: int = 5555):
    """Third business process: verifies both modifications."""
    logger.info("Business Process 3: Starting...")
    time.sleep(5)  # Wait for business process 2 to finish
    
    # Connect to parameter server
    client = ParameterServerClient(zmq_port=port, device_id=0)
    
    # Retrieve tensor again
    logger.info("Verifying both tensor modifications...")
    weights = client.get_tensor("weights")
    
    if weights is not None:
        logger.info(f"Verification: weights[0, 0] = {weights[0, 0]}")
        logger.info(f"Verification: weights[0, 1] = {weights[0, 1]}")
        
        if weights[0, 0].item() == 999.0 and weights[0, 1].item() == 888.0:
            logger.info("✓ Both modifications verified! Tensor state is persistent!")
        else:
            logger.warning("✗ Modifications not found. Something went wrong.")
    
    # Shutdown the server
    logger.info("Shutting down server...")
    client.shutdown_server()
    client.close()
    
    logger.info("Business Process 3: Completed")

def launch_server():

    logger.info("=" * 60)
    logger.info("Sequential Example: Server and Client in same process")
    logger.info("=" * 60)
    
    # Create some test tensors
    tensors = {
        "test_weight": torch.randn(10000, 1000),
        "test_activate": torch.randn(1000, 5000),
    }
    
    # Start server in a thread
    server = PersistentParameterServer(tensors=tensors, device_id=0, zmq_port=5556)
    server.start(daemon=False)
    time.sleep(0.5)  # Wait for server to start
    
    # Keep server running
    if server.server_thread:
        server.server_thread.join()

def launch_server_checkpoints(checkpoint_path: str = None, tp: int =2, port: int =5556):
    logger.info("=" * 60)
    logger.info("Sequential Example: Server and Client in same process")
    logger.info("=" * 60)
    
    # Start server in a thread
    start_time = time.time()
    server = PersistentParameterServer(checkpoint_path=checkpoint_path, device_ids=[i for i in range(tp)], zmq_port=port, tp=tp, pp=1)
    logger.info(f"Loaded checkpoint in {time.time() - start_time:.2f} seconds")
    server.start(daemon=False)
    time.sleep(0.5)  # Wait for server to start
    
    # Keep server running
    if server.server_thread:
        server.server_thread.join()

def launch_client_checkpoints():
    # Create client
    client = ParameterServerClient(zmq_port=5556, device_id=0)
    
    # List tensors
    logger.info(f"Available tensors: {client.list_tensors(0)}")
    cost_list = []
    param_list = client.list_tensors(0)

    # Method 1: Individual retrieval
    logger.info("\n--- Method 1: Individual Retrieval ---")
    for i in range(1):
        start_time = time.time()
        for name in param_list:
            retrieved_weight = client.get_tensor(name, 0)
            logger.info(f"tensor {name} first element: {retrieved_weight[0]}")
            retrieved_weight.zero_()  # in-place operation to test zero-copy
        elapsed = time.time() - start_time
        if i > 5:  # warm up
            cost_list.append(elapsed)
        logger.info(f"Run {i+1}: Retrieved {len(param_list)} tensors in {elapsed:.4f}s")
    
    if len(cost_list) > 0:
        avg_individual = sum(cost_list) / len(cost_list)
        logger.info(f"Average time (individual): {avg_individual:.4f}s")
        
    # Method 2: Batch retrieval
    logger.info("\n--- Method 2: Batch Retrieval (get_all_tensors) ---")
    cost_list_batch = []
    for i in range(1):
        start_time = time.time()
        all_tensors = client.get_all_tensors(0)
        for name, tensor in all_tensors.items():
            logger.info(f"tensor {name} first element: {tensor[0]}")  # in-place operation to test zero-copy
        elapsed = time.time() - start_time
        if i > 5:
            # warm up
            cost_list_batch.append(elapsed)
        logger.info(f"Run {i+1}: Retrieved {len(all_tensors)} tensors in {elapsed:.4f}s")
    if len(cost_list_batch) > 0:
        avg_batch = sum(cost_list_batch) / len(cost_list_batch)
        logger.info(f"Average time (batch): {avg_batch:.4f}s")

    # Performance comparison
    if len(cost_list) > 0 and len(cost_list_batch) > 0:
        logger.info("\n--- Performance Comparison ---")
        logger.info(f"Individual retrieval: {avg_individual:.4f}s")
        logger.info(f"Batch retrieval:      {avg_batch:.4f}s")
        speedup = avg_individual / avg_batch if avg_batch > 0 else 0
        logger.info(f"Speedup:              {speedup:.2f}x")
    
    client.close()

def launch_client():
    # Create client
    client = ParameterServerClient(zmq_port=5556, device_id=0)
    
    # List tensors
    logger.info(f"Available tensors: {client.list_tensors()}")
    cost_list = []
    # Retrieve via IPC
    for i in range(1):
        start_time = time.time()
        retrieved_weight = client.get_tensor("test_weight")
        retrieved_activate = client.get_tensor("test_activate")
        cost_list.append(time.time() - start_time)
        if retrieved_weight is not None and retrieved_activate is not None:
            logger.info(f"Retrieved tensor shape: {retrieved_weight.shape}")
            logger.info(f"Retrieved tensor shape: {retrieved_activate.shape}")
            logger.info("✓ Sequential example successful!")
        res = torch.matmul(retrieved_weight, retrieved_activate)
        print(res.shape)
        # set test_weight to zero
        retrieved_weight.zero_()
        logger.info("Set 'test_weight' tensor to zero in place.")
        # Cleanup
        time.sleep(2)
    logger.info(f"Average IPC retrieval time: {sum(cost_list)/len(cost_list):.6f} seconds")
    client.close()

def run_sequential_example():
    """Run a sequential example in the same process."""
    logger.info("=" * 60)
    logger.info("Sequential Example: Server and Client in same process")
    logger.info("=" * 60)
    
    # Create some test tensors
    tensors = {
        "test_tensor": torch.randn(100, 100),
        "test_vector": torch.arange(0, 50, dtype=torch.float32),
    }
    
    # Start server in a thread
    server = PersistentParameterServer(tensors=tensors, device_id=0, zmq_port=5556)
    server.start(daemon=True)
    time.sleep(0.5)  # Wait for server to start
    
    # Create client
    client = ParameterServerClient(zmq_port=5556, device_id=0)
    
    # List tensors
    logger.info(f"Available tensors: {client.list_tensors()}")
    
    # Retrieve via IPC
    retrieved = client.get_tensor("test_tensor")
    if retrieved is not None:
        logger.info(f"Retrieved tensor shape: {retrieved.shape}")
        logger.info("✓ Sequential example successful!")
    
    # Cleanup
    client.shutdown_server()
    client.close()
    server.cleanup()



def run_multiprocess_example():
    """Run example with multiple processes."""
    logger.info("=" * 60)
    logger.info("Multi-Process Example: Persistent Parameter Server")
    logger.info("=" * 60)
    
    port = 5555
    
    # Start server process (with pre-initialized tensors)
    server_proc = mp.Process(target=server_process, args=(port,))
    server_proc.start()
    
    # Run business process 1 (retrieves and modifies tensors)
    bp1 = mp.Process(target=business_process_1, args=(port,))
    bp1.start()
    bp1.join()
    
    logger.info("\n" + "=" * 60)
    logger.info("Business Process 1 has exited, but tensors remain in GPU memory!")
    logger.info("=" * 60 + "\n")
    
    # Run business process 2 (verifies modification)
    bp2 = mp.Process(target=business_process_2, args=(port,))
    bp2.start()
    bp2.join()
    
    logger.info("\n" + "=" * 60)
    logger.info("Business Process 2 has exited")
    logger.info("=" * 60 + "\n")
    
    # Run business process 3 (final verification and shutdown)
    bp3 = mp.Process(target=business_process_3, args=(port,))
    bp3.start()
    bp3.join()
    
    # Wait for server to shutdown
    server_proc.join(timeout=2)
    if server_proc.is_alive():
        server_proc.terminate()
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    import sys
    
    # Set multiprocessing start method
    mp.set_start_method("spawn", force=True)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--sequential":
        run_sequential_example()
    elif len(sys.argv) > 1 and sys.argv[1] == "--server":
        launch_server()
    elif len(sys.argv) > 1 and sys.argv[1] == "--client":
        launch_client()
    elif len(sys.argv) > 1 and sys.argv[1] == "--server-ckpts":
        checkpoint_path = sys.argv[2] if len(sys.argv) > 2 else None
        tp = int(sys.argv[3]) if len(sys.argv) > 3 else 2
        port = int(sys.argv[4]) if len(sys.argv) > 4 else 5556
        launch_server_checkpoints(checkpoint_path=checkpoint_path, tp=tp, port=port)
    elif len(sys.argv) > 1 and sys.argv[1] == "--client-ckpts":
        launch_client_checkpoints()
    else:
        RuntimeError("Please specify --sequential, --server, --client, --server-ckpts, or --client-ckpts")