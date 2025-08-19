#!/usr/bin/env python3
"""
Test script to verify Fp32LayerNorm compatibility with 2D CNN implementation.
"""

import torch
import torch.nn as nn
import sys
import os

# Add fairseq to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fairseq'))

from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Conv2DFeatureExtractionModel
from fairseq.modules import Fp32LayerNorm

def test_fp32_layer_norm_compatibility():
    """Test that Fp32LayerNorm works correctly with 2D CNN outputs."""
    print("🧪 Testing Fp32LayerNorm Compatibility...")
    
    # Test 1: Basic Fp32LayerNorm with 2D CNN outputs
    print("\n📋 Test 1: Basic Fp32LayerNorm with 2D CNN")
    
    # Create a simple 2D CNN output
    batch_size = 2
    channels = 64
    height = 32
    width = 32
    
    x = torch.randn(batch_size, channels, height, width)
    print(f"📥 Input shape: {x.shape}")
    
    # Reshape for Fp32LayerNorm: (B, C, H, W) -> (B*H*W, C)
    x_reshaped = x.permute(0, 2, 3, 1).reshape(-1, channels)
    print(f"📥 Reshaped for Fp32LayerNorm: {x_reshaped.shape}")
    
    # Apply Fp32LayerNorm
    layer_norm = Fp32LayerNorm(channels, elementwise_affine=True)
    x_norm = layer_norm(x_reshaped)
    print(f"📤 Fp32LayerNorm output shape: {x_norm.shape}")
    
    # Reshape back: (B*H*W, C) -> (B, C, H, W)
    x_norm_reshaped = x_norm.reshape(batch_size, height, width, channels).permute(0, 3, 1, 2)
    print(f"📤 Final output shape: {x_norm_reshaped.shape}")
    
    # Verify shapes match
    assert x.shape == x_norm_reshaped.shape, f"Shape mismatch: {x.shape} vs {x_norm_reshaped.shape}"
    print("✅ Shape compatibility verified!")

def test_conv2d_feature_extractor_with_layer_norm():
    """Test Conv2DFeatureExtractionModel with layer norm mode."""
    print("\n📋 Test 2: Conv2DFeatureExtractionModel with Layer Norm")
    
    # Configuration with layer norm
    conv_layers = [(32, 3, 2), (64, 3, 2), (128, 3, 2)]
    
    # Create model with layer norm mode
    model = Conv2DFeatureExtractionModel(
        conv_layers=conv_layers,
        dropout=0.0,
        mode="layer_norm",  # Use layer norm mode
        conv_bias=False,
        input_channels=1,
    )
    
    # Test input
    batch_size = 2
    input_channels = 1
    height = 64
    width = 64
    
    x = torch.randn(batch_size, input_channels, height, width)
    print(f"📥 Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"📤 Output shape: {output.shape}")
    
    # Check that all layer norms are Fp32LayerNorm instances
    layer_norm_count = 0
    for layer_norm in model.layer_norms:
        if layer_norm is not None:
            layer_norm_count += 1
            assert isinstance(layer_norm, Fp32LayerNorm), f"Expected Fp32LayerNorm, got {type(layer_norm)}"
    
    print(f"✅ Found {layer_norm_count} Fp32LayerNorm instances")
    print("✅ Conv2DFeatureExtractionModel with layer norm works correctly!")

def test_wav2vec2_2d_with_layer_norm():
    """Test complete wav2vec2_2d model with layer norm."""
    print("\n📋 Test 3: Complete Wav2Vec2_2D Model with Layer Norm")
    
    # Create configuration with layer norm
    cfg = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters
        conv_2d_feature_layers="[(32, 3, 2), (64, 3, 2), (128, 3, 2)]",
        input_channels=1,
        input_height=64,
        input_width=64,
        
        # Use layer norm mode
        extractor_mode="layer_norm",
        
        # Transformer parameters
        encoder_layers=1,  # Small for testing
        encoder_embed_dim=128,  # Small for testing
        encoder_ffn_embed_dim=512,
        encoder_attention_heads=4,
        
        # Spatial embedding parameters
        use_spatial_embedding=True,
        num_recording_sites=10,
        spatial_embed_dim=64,
        
        # Other parameters
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        encoder_layerdrop=0.0,
        dropout_input=0.0,
        dropout_features=0.0,
        final_dim=0,
        layer_norm_first=False,
        conv_bias=False,
        logit_temp=0.1,
        quantize_targets=False,
        quantize_input=False,
        same_quantizer=False,
        target_glu=False,
        feature_grad_mult=1.0,
        quantizer_depth=1,
        quantizer_factor=3,
        latent_vars=320,
        latent_groups=2,
        latent_dim=0,
        mask_length=10,
        mask_prob=0.65,
        mask_selection="static",
        mask_other=0,
        no_mask_overlap=False,
        mask_min_space=1,
        require_same_masks=True,
        mask_dropout=0.0,
        mask_channel_length=10,
        mask_channel_prob=0.0,
        mask_channel_before=False,
        mask_channel_selection="static",
        mask_channel_other=0,
        no_mask_channel_overlap=False,
        mask_channel_min_space=1,
        num_negatives=100,
        negatives_from_everywhere=False,
        cross_sample_negatives=0,
        codebook_negatives=0,
        conv_pos=128,
        conv_pos_groups=16,
        pos_conv_depth=1,
        latent_temp=(2, 0.5, 0.999995),
        max_positions=100000,
        checkpoint_activations=False,
        required_seq_len_multiple=2,
        crop_seq_to_multiple=1,
        depthwise_conv_kernel_size=31,
        attn_type="",
        pos_enc_type="abs",
        fp16=False,
        adp_num=-1,
        adp_dim=64,
        adp_act_fn="relu",
        adp_trf_idx="all",
    )
    
    # Create model
    model = cfg.build_model(cfg)
    print(f"🏗️ Model created successfully!")
    
    # Test input
    batch_size = 2
    input_channels = 1
    height = 64
    width = 64
    
    x = torch.randn(batch_size, input_channels, height, width)
    recording_site_ids = torch.randint(1, cfg.num_recording_sites, (batch_size,))
    
    print(f"📥 Input shape: {x.shape}")
    print(f"📥 Recording site IDs: {recording_site_ids}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, recording_site_ids=recording_site_ids, features_only=True)
        print(f"📤 Output shape: {output['x'].shape}")
    
    print("✅ Complete Wav2Vec2_2D model with layer norm works correctly!")

def main():
    """Run all compatibility tests."""
    print("🚀 Starting Fp32LayerNorm compatibility tests...\n")
    
    try:
        test_fp32_layer_norm_compatibility()
        test_conv2d_feature_extractor_with_layer_norm()
        test_wav2vec2_2d_with_layer_norm()
        print("\n🎉 All Fp32LayerNorm compatibility tests passed!")
        print("✅ The 2D CNN implementation is fully compatible with Fp32LayerNorm!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 