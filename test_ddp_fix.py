#!/usr/bin/env python3
"""
Test script to verify DDP fixes work
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import sys

# Add fairseq to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel


def test_model_parameters():
    """Test that all model parameters are used"""
    print("Testing model parameter usage...")
    
    # Create a small test configuration
    config = Wav2Vec2_2DConfig(
        conv_2d_feature_layers="[(16, 3, 2), (32, 3, 2)]",
        input_width=10,
        input_height=100,
        encoder_layers=2,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=256,
        encoder_attention_heads=2,
        use_spatial_embedding=False,  # Disable to avoid unused parameters
        use_scaled_rope=True,
        temporal_conv1d_enabled=True,
        temporal_steps=10,
        flattened_pool_dim=32,
        target_glu=False,  # Disable to avoid unused parameters
        quantizer_depth=1,
        quantizer_factor=2,
        latent_vars=64,
        latent_groups=1,
        num_negatives=5,
        cross_sample_negatives=2,
    )
    
    # Create model
    model = Wav2Vec2_2DModel(config)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Test forward pass
    batch_size = 2
    source = torch.randn(batch_size, 1, 100, 10)
    
    try:
        outputs = model(source=source, padding_mask=None)
        print("✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        print(f"   x shape: {outputs['x'].shape}")
        
        # Test backward pass
        loss = torch.mean(outputs['x'])
        loss.backward()
        print("✅ Backward pass successful")
        
        # Check which parameters received gradients
        unused_params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                unused_params.append(name)
        
        if unused_params:
            print(f"❌ Unused parameters found: {unused_params}")
        else:
            print("✅ All parameters received gradients")
            
    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        return False
    
    return len(unused_params) == 0


def test_ddp_setup():
    """Test DDP setup with the model"""
    print("\nTesting DDP setup...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping DDP test")
        return False
    
    # Create model
    config = Wav2Vec2_2DConfig(
        conv_2d_feature_layers="[(16, 3, 2), (32, 3, 2)]",
        input_width=10,
        input_height=100,
        encoder_layers=2,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=256,
        encoder_attention_heads=2,
        use_spatial_embedding=False,
        use_scaled_rope=True,
        temporal_conv1d_enabled=True,
        temporal_steps=10,
        flattened_pool_dim=32,
        target_glu=False,
        quantizer_depth=1,
        quantizer_factor=2,
        latent_vars=64,
        latent_groups=1,
        num_negatives=5,
        cross_sample_negatives=2,
    )
    
    model = Wav2Vec2_2DModel(config)
    model = model.cuda()
    
    # Test DDP setup
    try:
        ddp_model = DDP(
            model, 
            device_ids=[0], 
            output_device=0, 
            find_unused_parameters=True,
            broadcast_buffers=False
        )
        print("✅ DDP setup successful")
        
        # Test forward pass with DDP
        batch_size = 2
        source = torch.randn(batch_size, 1, 100, 10).cuda()
        
        outputs = ddp_model(source=source, padding_mask=None)
        print("✅ DDP forward pass successful")
        
        # Test backward pass with DDP
        loss = torch.mean(outputs['x'])
        loss.backward()
        print("✅ DDP backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ DDP setup failed: {e}")
        return False


def main():
    """Main test function"""
    print("=== DDP Fix Test ===")
    
    # Test 1: Model parameter usage
    param_test_passed = test_model_parameters()
    
    # Test 2: DDP setup
    ddp_test_passed = test_ddp_setup()
    
    print(f"\n=== Test Results ===")
    print(f"Parameter usage test: {'✅ PASSED' if param_test_passed else '❌ FAILED'}")
    print(f"DDP setup test: {'✅ PASSED' if ddp_test_passed else '❌ FAILED'}")
    
    if param_test_passed and ddp_test_passed:
        print("\n🎉 All tests passed! The DDP fix should work.")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")


if __name__ == "__main__":
    main()
