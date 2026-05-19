#!/usr/bin/env python
"""
Command-line tool for managing the Persistent Parameter Server.
"""

import argparse
import sys

import torch
from loguru import logger

from checkpoint_engine.persistent_ps import (
    ParameterServerClient,
    run_server_daemon,
)


def cmd_start_server(args):
    """Start the parameter server."""
    logger.info(f"Starting parameter server on device {args.device_id}, port {args.port}")
    run_server_daemon(
        device_id=args.device_id,
        zmq_port=args.port,
        zmq_host=args.host,
    )


def cmd_list_tensors(args):
    """List all tensors in the parameter server."""
    try:
        client = ParameterServerClient(zmq_host=args.host, zmq_port=args.port)
        tensors = client.list_tensors()
        
        if not tensors:
            print("No tensors registered.")
        else:
            print(f"Total tensors: {len(tensors)}")
            print("\nRegistered tensors:")
            for name in tensors:
                info = client.get_tensor_info(name)
                if info:
                    print(f"  - {name}")
                    print(f"    Shape: {info['shape']}")
                    print(f"    Dtype: {info['dtype']}")
                    print(f"    Device: {info['device']}")
                    print(f"    Size: {info['nbytes'] / (1024**2):.2f} MB")
        
        client.close()
    except Exception as e:
        logger.error(f"Failed to list tensors: {e}")
        sys.exit(1)


def cmd_info(args):
    """Get information about a specific tensor."""
    try:
        client = ParameterServerClient(zmq_host=args.host, zmq_port=args.port)
        info = client.get_tensor_info(args.name)
        
        if info is None:
            print(f"Tensor '{args.name}' not found.")
            sys.exit(1)
        
        print(f"Tensor: {args.name}")
        print(f"  Shape: {info['shape']}")
        print(f"  Dtype: {info['dtype']}")
        print(f"  Device: {info['device']}")
        print(f"  Size: {info['nbytes'] / (1024**2):.2f} MB")
        
        client.close()
    except Exception as e:
        logger.error(f"Failed to get tensor info: {e}")
        sys.exit(1)


def cmd_register(args):
    """Register a tensor from a file."""
    try:
        client = ParameterServerClient(
            zmq_host=args.host,
            zmq_port=args.port,
            device_id=args.device_id,
        )
        
        # Load tensor from file
        logger.info(f"Loading tensor from {args.file}")
        tensor = torch.load(args.file)
        
        if not isinstance(tensor, torch.Tensor):
            logger.error("File does not contain a valid tensor")
            sys.exit(1)
        
        # Register tensor
        logger.info(f"Registering tensor '{args.name}'")
        success = client.register_tensor(args.name, tensor)
        
        if success:
            print(f"✓ Successfully registered tensor '{args.name}'")
            info = client.get_tensor_info(args.name)
            if info:
                print(f"  Shape: {info['shape']}")
                print(f"  Dtype: {info['dtype']}")
                print(f"  Size: {info['nbytes'] / (1024**2):.2f} MB")
        else:
            print(f"✗ Failed to register tensor '{args.name}'")
            sys.exit(1)
        
        client.close()
    except Exception as e:
        logger.error(f"Failed to register tensor: {e}")
        sys.exit(1)


def cmd_unregister(args):
    """Unregister a tensor."""
    try:
        client = ParameterServerClient(zmq_host=args.host, zmq_port=args.port)
        
        success = client.unregister_tensor(args.name)
        
        if success:
            print(f"✓ Successfully unregistered tensor '{args.name}'")
        else:
            print(f"✗ Failed to unregister tensor '{args.name}'")
            sys.exit(1)
        
        client.close()
    except Exception as e:
        logger.error(f"Failed to unregister tensor: {e}")
        sys.exit(1)


def cmd_shutdown(args):
    """Shutdown the parameter server."""
    try:
        client = ParameterServerClient(zmq_host=args.host, zmq_port=args.port)
        
        logger.info("Sending shutdown command to server...")
        success = client.shutdown_server()
        
        if success:
            print("✓ Server shutdown command sent successfully")
        else:
            print("✗ Failed to shutdown server")
            sys.exit(1)
        
        client.close()
    except Exception as e:
        logger.error(f"Failed to shutdown server: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Persistent Parameter Server CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server
  psctl start --device-id 0 --port 5555
  
  # List all tensors
  psctl list --port 5555
  
  # Get tensor info
  psctl info model_weights --port 5555
  
  # Register a tensor from file
  psctl register my_tensor /path/to/tensor.pt --port 5555
  
  # Unregister a tensor
  psctl unregister my_tensor --port 5555
  
  # Shutdown server
  psctl shutdown --port 5555
        """,
    )
    
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Server host address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Server port (default: 5555)",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Start command
    start_parser = subparsers.add_parser("start", help="Start the parameter server")
    start_parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    start_parser.set_defaults(func=cmd_start_server)
    
    # List command
    list_parser = subparsers.add_parser("list", help="List all tensors")
    list_parser.set_defaults(func=cmd_list_tensors)
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Get tensor information")
    info_parser.add_argument("name", help="Tensor name")
    info_parser.set_defaults(func=cmd_info)
    
    # Register command
    register_parser = subparsers.add_parser("register", help="Register a tensor from file")
    register_parser.add_argument("name", help="Tensor name")
    register_parser.add_argument("file", help="Path to tensor file (.pt)")
    register_parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="GPU device ID (default: 0)",
    )
    register_parser.set_defaults(func=cmd_register)
    
    # Unregister command
    unregister_parser = subparsers.add_parser("unregister", help="Unregister a tensor")
    unregister_parser.add_argument("name", help="Tensor name")
    unregister_parser.set_defaults(func=cmd_unregister)
    
    # Shutdown command
    shutdown_parser = subparsers.add_parser("shutdown", help="Shutdown the server")
    shutdown_parser.set_defaults(func=cmd_shutdown)
    
    args = parser.parse_args()
    
    if not hasattr(args, "func"):
        parser.print_help()
        sys.exit(1)
    
    args.func(args)


if __name__ == "__main__":
    main()
