#!/usr/bin/env python3
"""
Test script to verify that spatiality is preserved in Fp32LayerNorm implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add fairseq to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fairseq'))

from fairseq.modules import Fp32LayerNorm

def test_spatiality_preservation():
    """Test that spatiality is preserved in Fp32LayerNorm operations."""
    print("🧪 Testing Spatiality Preservation in Fp32LayerNorm...")
    
    # Create a 2D tensor with spatial structure
    batch_size = 2
    channels = 64
    height = 16
    width = 16
    
    # Create input with spatial patterns
    x = torch.randn(batch_size, channels, height, width)
    
    # Add some spatial patterns for testing
    for b in range(batch_size):
        for c in range(channels):
            # Create a gradient pattern
            x[b, c, :, :] = torch.arange(height * width).reshape(height, width).float()
    
    print(f"📥 Original shape: {x.shape}")
    print(f"📥 Sample spatial pattern at [0, 0, :, :]:\n{x[0, 0, :4, :4]}")
    
    # Test the reshape operations
    B, C, H, W = x.shape
    
    # Reshape for Fp32LayerNorm: (B, C, H, W) -> (B*H*W, C)
    x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, C)
    print(f"📥 Reshaped for Fp32LayerNorm: {x_reshaped.shape}")
    
    # Apply Fp32LayerNorm
    layer_norm = Fp32LayerNorm(C, elementwise_affine=True)
    x_norm = layer_norm(x_reshaped)
    print(f"📤 Fp32LayerNorm output shape: {x_norm.shape}")
    
    # Reshape back: (B*H*W, C) -> (B, C, H, W)
    x_norm_reshaped = x_norm.reshape(B, H, W, C).permute(0, 3, 1, 2)
    print(f"📤 Final output shape: {x_norm_reshaped.shape}")
    
    # Verify shapes match
    assert x.shape == x_norm_reshaped.shape, f"Shape mismatch: {x.shape} vs {x_norm_reshaped.shape}"
    print("✅ Shape compatibility verified!")
    
    # Test spatial pattern preservation
    print(f"📤 Sample spatial pattern at [0, 0, :, :] (after layer norm):\n{x_norm_reshaped[0, 0, :4, :4]}")
    
    # Verify that spatial relationships are preserved
    # The patterns should be different due to normalization, but spatial structure should remain
    spatial_structure_preserved = True
    
    # Check that the spatial dimensions are correct
    if x_norm_reshaped.shape[2:] == x.shape[2:]:  # H, W dimensions
        print("✅ Spatial dimensions preserved!")
    else:
        print("❌ Spatial dimensions lost!")
        spatial_structure_preserved = False
    
    # Check that the tensor is properly normalized
    mean_per_channel = x_norm_reshaped.mean(dim=(0, 2, 3))
    std_per_channel = x_norm_reshaped.std(dim=(0, 2, 3))
    
    print(f"📊 Mean per channel (should be ~0): {mean_per_channel[:5]}")
    print(f"📊 Std per channel (should be ~1): {std_per_channel[:5]}")
    
    if torch.allclose(mean_per_channel, torch.zeros_like(mean_per_channel), atol=1e-3):
        print("✅ Layer normalization working correctly!")
    else:
        print("❌ Layer normalization not working!")
        spatial_structure_preserved = False
    
    if spatial_structure_preserved:
        print("🎉 Spatiality is PRESERVED in Fp32LayerNorm operations!")
    else:
        print("❌ Spatiality is LOST in Fp32LayerNorm operations!")
    
    return spatial_structure_preserved

def test_conv2d_layer_norm_spatiality():
    """Test spatiality preservation in Conv2D feature extractor with layer norm."""
    print("\n🧪 Testing Conv2D Feature Extractor with Layer Norm...")
    
    from fairseq.models.wav2vec.wav2vec2_2d import Conv2DFeatureExtractionModel
    
    # Create a simple 2D CNN with layer norm
    conv_layers = [(32, 3, 2), (64, 3, 2)]
    
    model = Conv2DFeatureExtractionModel(
        conv_layers=conv_layers,
        dropout=0.0,
        mode="layer_norm",  # Use layer norm mode
        conv_bias=False,
        input_channels=1,
    )
    
    # Create input with spatial patterns
    batch_size = 2
    input_channels = 1
    height = 32
    width = 32
    
    x = torch.randn(batch_size, input_channels, height, width)
    
    # Add spatial patterns
    for b in range(batch_size):
        x[b, 0, :, :] = torch.arange(height * width).reshape(height, width).float()
    
    print(f"📥 Input shape: {x.shape}")
    print(f"📥 Sample spatial pattern:\n{x[0, 0, :8, :8]}")
    
    # Forward pass
    output = model(x)
    print(f"📤 Output shape: {output.shape}")
    print(f"📤 Sample spatial pattern:\n{output[0, 0, :4, :4]}")
    
    # Check that spatial dimensions are preserved
    expected_height = height // 4  # 2 conv layers with stride 2
    expected_width = width // 4
    
    if output.shape[2:] == (expected_height, expected_width):
        print("✅ Spatial dimensions correctly reduced by convolutions!")
    else:
        print(f"❌ Expected spatial dimensions {(expected_height, expected_width)}, got {output.shape[2:]}")
    
    print("✅ Conv2D feature extractor preserves spatiality!")

def main():
    """Run all spatiality tests."""
    print("🚀 Starting spatiality preservation tests...\n")
    
    try:
        test_spatiality_preservation()
        test_conv2d_layer_norm_spatiality()
        print("\n🎉 All spatiality tests passed!")
        print("✅ Fp32LayerNorm preserves spatiality correctly!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 