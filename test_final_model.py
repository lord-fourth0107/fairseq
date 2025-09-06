#!/usr/bin/env python3
"""
Test script for the fixed wav2vec2_2d_final.py model
"""

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import sys

# Add fairseq to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel


def test_model_creation():
    """Test that the model can be created and run forward pass"""
    print("Testing model creation and forward pass...")
    
    # Create the same config as in your final model
    config = Wav2Vec2_2DConfig(
        # 2D CNN feature extraction layers
        conv_2d_feature_layers="[(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]",
        
        # Input dimensions
        input_channels=1,
        input_height=3750,  # Time points (height dimension)
        input_width=93,     # Channels (width dimension)
        
        # Transformer parameters
        encoder_layers=6,
        encoder_embed_dim=384,
        encoder_ffn_embed_dim=1536,
        encoder_attention_heads=6,
        activation_fn="gelu",
        
        # Scaled RoPE parameters
        use_scaled_rope=True,
        rope_max_seq_len=4096,
        rope_scale_factor=1.0,
        rope_theta=10000.0,
        
        # Legacy spatial embedding parameters (disabled)
        use_spatial_embedding=False,
        
        # Masking parameters
        mask_prob=0.15,
        mask_length=5,
        mask_selection="static",
        
        # Other parameters
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        layer_norm_first=False,
        feature_grad_mult=1.0,
        conv_bias=False,
        extractor_mode="default",
        
        # Adaptive pooling after flattening
        flattened_pool_dim=256,
        
        # 1D CNN for temporal expansion
        temporal_conv1d_enabled=True,
        temporal_steps=50,
        
        # Negative sampling
        num_negatives=20,
        cross_sample_negatives=5,
        codebook_negatives=10,
        
        # Quantization parameters
        quantizer_depth=2,
        quantizer_factor=3,
        latent_vars=320,
        latent_groups=2,
        same_quantizer=True,
    )
    
    # Create model
    model = Wav2Vec2_2DModel(config)
    
    print(f"✅ Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    batch_size = 2
    source = torch.randn(batch_size, 1, 3750, 93)
    
    try:
        outputs = model(source=source, features_only=False)
        print("✅ Forward pass successful")
        print(f"   Output keys: {list(outputs.keys())}")
        
        # Test backward pass
        if 'loss' in outputs and outputs['loss'] is not None:
            loss = outputs['loss']
        else:
            # Manual loss computation
            features = outputs.get('features', outputs.get('x'))
            loss = torch.mean(features)
        
        loss.backward()
        print("✅ Backward pass successful")
        
        # Check for unused parameters
        unused_params = []
        for name, param in model.named_parameters():
            if param.grad is None:
                unused_params.append(name)
        
        if unused_params:
            print(f"⚠️  Unused parameters found: {len(unused_params)}")
            print(f"   First few: {unused_params[:5]}")
        else:
            print("✅ All parameters received gradients")
            
    except Exception as e:
        print(f"❌ Forward/backward pass failed: {e}")
        return False
    
    return True


def test_ddp_setup():
    """Test DDP setup with the model"""
    print("\nTesting DDP setup...")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, skipping DDP test")
        return False
    
    # Create model
    config = Wav2Vec2_2DConfig(
        conv_2d_feature_layers="[(32, 3, 2), (64, 3, 2)]",  # Smaller for testing
        input_channels=1,
        input_height=100,  # Smaller for testing
        input_width=10,
        encoder_layers=2,
        encoder_embed_dim=64,
        encoder_ffn_embed_dim=256,
        encoder_attention_heads=2,
        use_scaled_rope=True,
        use_spatial_embedding=False,
        flattened_pool_dim=32,
        temporal_conv1d_enabled=True,
        temporal_steps=10,
        num_negatives=5,
        cross_sample_negatives=2,
        quantizer_depth=1,
        quantizer_factor=2,
        latent_vars=64,
        latent_groups=1,
    )
    
    model = Wav2Vec2_2DModel(config)
    model = model.cuda()
    
    try:
        # Test DDP setup with the fixed configuration
        ddp_model = DDP(
            model, 
            device_ids=[0], 
            find_unused_parameters=True,  # This is the key fix!
            broadcast_buffers=False
        )
        print("✅ DDP setup successful")
        
        # Test forward pass with DDP
        batch_size = 2
        source = torch.randn(batch_size, 1, 100, 10).cuda()
        
        outputs = ddp_model(source=source, features_only=False)
        print("✅ DDP forward pass successful")
        
        # Test backward pass with DDP
        if 'loss' in outputs and outputs['loss'] is not None:
            loss = outputs['loss']
        else:
            features = outputs.get('features', outputs.get('x'))
            loss = torch.mean(features)
        
        loss.backward()
        print("✅ DDP backward pass successful")
        
        return True
        
    except Exception as e:
        print(f"❌ DDP setup failed: {e}")
        return False


def main():
    """Main test function"""
    print("=== Testing Fixed wav2vec2_2d_final.py Model ===")
    
    # Test 1: Model creation and forward pass
    model_test_passed = test_model_creation()
    
    # Test 2: DDP setup
    ddp_test_passed = test_ddp_setup()
    
    print(f"\n=== Test Results ===")
    print(f"Model creation test: {'✅ PASSED' if model_test_passed else '❌ FAILED'}")
    print(f"DDP setup test: {'✅ PASSED' if ddp_test_passed else '❌ FAILED'}")
    
    if model_test_passed and ddp_test_passed:
        print("\n🎉 All tests passed! Your fixed model should work for distributed training.")
        print("\nTo run your fixed model:")
        print("python wav2vec2_2d_final.py --world_size 3 --data_path /path/to/data --num_epochs 10")
    else:
        print("\n⚠️  Some tests failed. Check the issues above.")


if __name__ == "__main__":
    main()
