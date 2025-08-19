#!/usr/bin/env python3
"""
Simple test script for wav2vec2_2d without full fairseq initialization
"""

import sys
import os
import torch
import torch.nn as nn

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the specific modules we need
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.configs import CommonConfig
from fairseq.modules import Fp32LayerNorm
from fairseq.models.wav2vec.wav2vec2_2d import (
    Wav2Vec2_2DConfig, 
    Conv2DFeatureExtractionModel,
    Wav2Vec2_2DModel
)

def test_conv2d_feature_extraction():
    """Test the Conv2DFeatureExtractionModel"""
    print("🧪 Testing Conv2DFeatureExtractionModel...")
    
    # Create a simple config
    config = Wav2Vec2_2DConfig()
    config.conv_2d_feature_layers = "[(32, 3, 2), (64, 3, 2), (128, 3, 2)]"
    config.input_channels = 1
    config.input_height = 128
    config.input_width = 128
    config.extractor_mode = "layer_norm"
    
    # Create the model
    model = Conv2DFeatureExtractionModel(config)
    
    # Test input
    x = torch.randn(2, 1, 128, 128)
    print(f"📥 Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    
    print(f"📤 Output shape: {output.shape}")
    print("✅ Conv2DFeatureExtractionModel test passed!")
    return model

def test_wav2vec2_2d_model():
    """Test the complete Wav2Vec2_2DModel"""
    print("\n🧪 Testing Wav2Vec2_2DModel...")
    
    # Create config
    config = Wav2Vec2_2DConfig()
    config.conv_2d_feature_layers = "[(32, 3, 2), (64, 3, 2)]"
    config.input_channels = 1
    config.input_height = 64
    config.input_width = 64
    config.extractor_mode = "layer_norm"
    config.encoder_layers = 2
    config.encoder_embed_dim = 256
    config.encoder_attention_heads = 4
    config.use_spatial_embedding = True
    config.num_recording_sites = 10
    config.spatial_embed_dim = 64
    
    # Create model
    model = Wav2Vec2_2DModel(config)
    print("🏗️ Model created successfully!")
    
    # Test input
    x = torch.randn(2, 1, 64, 64)
    recording_site_ids = torch.randint(1, 10, (2,))
    print(f"📥 Input shape: {x.shape}")
    print(f"📥 Recording site IDs: {recording_site_ids}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, recording_site_ids=recording_site_ids, features_only=True)
    
    print(f"📤 Output shape: {output['x'].shape}")
    print("✅ Wav2Vec2_2DModel test passed!")
    return model

def test_fp32_layer_norm_compatibility():
    """Test Fp32LayerNorm compatibility with 2D inputs"""
    print("\n🧪 Testing Fp32LayerNorm Compatibility...")
    
    # Create a 2D tensor
    x = torch.randn(2, 64, 32, 32)
    print(f"📥 Input shape: {x.shape}")
    
    # Create Fp32LayerNorm
    layer_norm = Fp32LayerNorm(64)
    
    # Reshape for layer norm (B, C, H, W) -> (B*H*W, C)
    B, C, H, W = x.shape
    x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
    print(f"📥 Reshaped for Fp32LayerNorm: {x_reshaped.shape}")
    
    # Apply layer norm
    x_normed = layer_norm(x_reshaped)
    print(f"📤 Fp32LayerNorm output shape: {x_normed.shape}")
    
    # Reshape back
    x_final = x_normed.reshape(B, H, W, C).permute(0, 3, 1, 2)
    print(f"📤 Final output shape: {x_final.shape}")
    
    print("✅ Shape compatibility verified!")
    print("✅ Fp32LayerNorm works with 2D CNN outputs!")

def main():
    """Run all tests"""
    print("🚀 Starting wav2vec2_2d tests...\n")
    
    try:
        # Test 1: Conv2D Feature Extraction
        test_conv2d_feature_extraction()
        
        # Test 2: Complete Model
        test_wav2vec2_2d_model()
        
        # Test 3: Fp32LayerNorm Compatibility
        test_fp32_layer_norm_compatibility()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 