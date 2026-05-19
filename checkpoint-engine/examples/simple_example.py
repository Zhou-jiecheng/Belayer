#!/usr/bin/env python
"""
简单示例：演示如何使用 Persistent Parameter Server

使用方法：
1. 准备checkpoint文件（包含tensor的字典）
2. 启动server
3. 客户端连接并获取tensor
"""

import torch
from checkpoint_engine.persistent_ps import (
    PersistentParameterServer,
    ParameterServerClient,
)
from loguru import logger


def create_sample_checkpoint():
    """创建一个示例checkpoint文件"""
    checkpoint = {
        "model.weight": torch.randn(1000, 1000, dtype=torch.float32),
        "model.bias": torch.randn(1000, dtype=torch.float32),
        "optimizer.state": torch.randn(500, dtype=torch.float32),
    }
    
    torch.save(checkpoint, "sample_checkpoint.pt")
    logger.info("Created sample_checkpoint.pt with 3 tensors")
    return checkpoint


def server_example():
    """服务器端示例"""
    logger.info("=== Server Example ===")
    
    # 方式1: 直接传入tensor字典
    tensors = {
        "weights": torch.randn(100, 100),
        "bias": torch.zeros(100),
    }
    
    server = PersistentParameterServer(
        tensors=tensors,
        device_id=0,
        zmq_port=5555,
    )
    
    # 启动服务器（在实际使用中，这会阻塞）
    server.start(daemon=True)
    
    logger.info("Server started with tensors: " + str(list(tensors.keys())))
    logger.info("Server listening on tcp://127.0.0.1:5555")
    logger.info("Press Ctrl+C to stop server")
    
    # 在实际使用中，服务器会持续运行
    # server.server_thread.join()
    
    return server


def client_example():
    """客户端示例"""
    logger.info("=== Client Example ===")
    
    # 连接到server
    client = ParameterServerClient(
        zmq_host="127.0.0.1",
        zmq_port=5555,
        device_id=0,
    )
    
    # 列出所有tensor
    tensors = client.list_tensors()
    logger.info(f"Available tensors: {tensors}")
    
    # 获取tensor信息
    for name in tensors:
        info = client.get_tensor_info(name)
        logger.info(f"{name}: shape={info['shape']}, dtype={info['dtype']}")
    
    # 获取tensor（零拷贝）
    weights = client.get_tensor("weights")
    bias = client.get_tensor("bias")
    
    logger.info(f"Retrieved weights: {weights.shape}")
    logger.info(f"Retrieved bias: {bias.shape}")
    
    # 使用tensor进行计算
    input_data = torch.randn(10, 100, device=weights.device)
    output = torch.matmul(input_data, weights.T) + bias
    logger.info(f"Computed output: {output.shape}")
    
    # 关闭连接
    client.close()
    
    return weights, bias


def main():
    """主函数：演示完整流程"""
    import time
    
    logger.info("Starting Persistent Parameter Server Example")
    logger.info("=" * 60)
    
    # 1. 启动服务器
    server = server_example()
    time.sleep(0.5)  # 等待服务器启动
    
    logger.info("\n" + "=" * 60)
    
    # 2. 客户端连接
    weights, bias = client_example()
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed successfully!")
    logger.info("=" * 60)
    
    # 清理
    client = ParameterServerClient(zmq_port=5555)
    client.shutdown_server()
    client.close()
    server.cleanup()


if __name__ == "__main__":
    # 示例1: 完整流程
    main()
    
    # 示例2: 从checkpoint文件启动
    # checkpoint = create_sample_checkpoint()
    # server = PersistentParameterServer(
    #     tensors=checkpoint,
    #     device_id=0,
    #     zmq_port=5555,
    # )
    # server.start(daemon=False)
