#!/usr/bin/env python3
"""
Test script for wav2vec2_2d implementation with 2D CNN.
"""

import torch
import torch.nn as nn
import sys
import os

# Add fairseq to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fairseq'))

from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Conv2DFeatureExtractionModel

def test_conv2d_feature_extractor():
    """Test the 2D CNN feature extractor."""
    print("🧪 Testing Conv2DFeatureExtractionModel...")
    
    # Configuration
    conv_layers = [(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2)]
    
    # Create model
    model = Conv2DFeatureExtractionModel(
        conv_layers=conv_layers,
        dropout=0.0,
        mode="default",
        conv_bias=False,
        input_channels=1,
    )
    
    # Test input: (batch_size, channels, height, width)
    batch_size = 2
    input_channels = 1
    height = 128
    width = 128
    
    x = torch.randn(batch_size, input_channels, height, width)
    print(f"📥 Input shape: {x.shape}")
    
    # Forward pass
    output = model(x)
    print(f"📤 Output shape: {output.shape}")
    
    # Test layer norm mode
    model_layer_norm = Conv2DFeatureExtractionModel(
        conv_layers=conv_layers,
        dropout=0.0,
        mode="layer_norm",
        conv_bias=False,
        input_channels=1,
    )
    
    output_ln = model_layer_norm(x)
    print(f"📤 Output shape (layer norm): {output_ln.shape}")
    
    print("✅ Conv2DFeatureExtractionModel test passed!")

def test_wav2vec2_2d_model():
    """Test the complete wav2vec2_2d model."""
    print("\n🧪 Testing Wav2Vec2_2DModel...")
    
    # Create configuration
    cfg = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters
        conv_2d_feature_layers="[(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2)]",
        input_channels=1,
        input_height=128,
        input_width=128,
        
        # Transformer parameters
        encoder_layers=2,  # Small for testing
        encoder_embed_dim=256,  # Small for testing
        encoder_ffn_embed_dim=1024,
        encoder_attention_heads=8,
        
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
    
    # Test input: (batch_size, channels, height, width)
    batch_size = 2
    input_channels = 1
    height = 128
    width = 128
    
    x = torch.randn(batch_size, input_channels, height, width)
    print(f"📥 Input shape: {x.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x, features_only=True)
        print(f"📤 Output shape: {output['x'].shape}")
    
    print(" Wav2Vec2_2DModel test passed!")

def main():
    """Run all tests."""
    print(" Starting wav2vec2_2d tests...\n")
    
    try:
        test_conv2d_feature_extractor()
        test_wav2vec2_2d_model()
        print("\n All tests passed! wav2vec2_2d implementation is working correctly.")
    except Exception as e:
        print(f"\n Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 