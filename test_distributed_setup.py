#!/usr/bin/env python3
"""
Test script to verify distributed training setup works
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def test_ddp_setup(rank, world_size):
    """Test DDP setup without actual training"""
    print(f"Testing worker {rank} of {world_size}")
    
    # Set the rank environment variable for this process
    os.environ['RANK'] = str(rank)
    
    try:
        # Initialize distributed training
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout,
        )
        torch.cuda.set_device(rank)
        
        print(f"Rank {rank}: DDP setup successful!")
        
        # Test a simple operation
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{rank}')
            test_tensor = torch.randn(2, 3).to(device)
            print(f"Rank {rank}: Created tensor on {device}: {test_tensor.shape}")
        
        # Cleanup
        dist.destroy_process_group()
        print(f"Rank {rank}: Cleanup successful!")
        
    except Exception as e:
        print(f"Rank {rank}: Error during DDP setup: {e}")
        return False
    
    return True

def main():
    world_size = 2  # Test with 2 GPUs
    
    # Set up environment variables for distributed training
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = '0'  # Will be overridden by mp.spawn
    
    print(f"Testing distributed setup with {world_size} GPUs")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    
    if torch.cuda.device_count() < world_size:
        print(f"Warning: Only {torch.cuda.device_count()} GPUs available, but testing with {world_size}")
    
    # Launch distributed training test
    try:
        mp.spawn(test_ddp_setup, args=(world_size,), nprocs=world_size, join=True)
        print("Distributed setup test completed successfully!")
    except Exception as e:
        print(f"Distributed setup test failed: {e}")

if __name__ == "__main__":
    main()
