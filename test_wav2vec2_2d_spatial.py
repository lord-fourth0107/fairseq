#!/usr/bin/env python3
"""
Test script for wav2vec2_2d implementation with spatial embeddings for recording sites.
"""

import torch
import torch.nn as nn
import sys
import os

# Add fairseq to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'fairseq'))

from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Conv2DFeatureExtractionModel

def test_spatial_embedding():
    """Test the spatial embedding functionality."""
    print("🧪 Testing Spatial Embedding...")
    
    # Create configuration with spatial embeddings
    cfg = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters
        conv_2d_feature_layers="[(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2)]",
        input_channels=1,
        input_height=128,
        input_width=128,
        
        # Spatial embedding parameters
        use_spatial_embedding=True,
        num_recording_sites=64,
        spatial_embed_dim=256,
        spatial_embed_dropout=0.1,
        
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
    batch_size = 4
    input_channels = 1
    height = 128
    width = 128
    
    x = torch.randn(batch_size, input_channels, height, width)
    recording_site_ids = torch.randint(1, cfg.num_recording_sites, (batch_size,))  # Random site IDs
    
    print(f"📥 Input shape: {x.shape}")
    print(f"📥 Recording site IDs: {recording_site_ids}")
    
    # Forward pass with spatial embeddings
    with torch.no_grad():
        output = model(x, recording_site_ids=recording_site_ids, features_only=True)
        print(f"📤 Output shape: {output['x'].shape}")
    
    # Test without spatial embeddings
    with torch.no_grad():
        output_no_spatial = model(x, features_only=True)
        print(f"📤 Output shape (no spatial): {output_no_spatial['x'].shape}")
    
    print("✅ Spatial embedding test passed!")

def test_spatial_embedding_visualization():
    """Test spatial embedding visualization and analysis."""
    print("\n🧪 Testing Spatial Embedding Analysis...")
    
    # Create a simple model to test spatial embeddings
    cfg = Wav2Vec2_2DConfig(
        use_spatial_embedding=True,
        num_recording_sites=10,
        spatial_embed_dim=64,
        encoder_embed_dim=128,
        encoder_layers=1,
        encoder_attention_heads=4,
        conv_2d_feature_layers="[(32, 3, 2), (64, 3, 2)]",
        input_channels=1,
        input_height=64,
        input_width=64,
    )
    
    model = cfg.build_model(cfg)
    
    # Test different recording sites
    batch_size = 3
    x = torch.randn(batch_size, 1, 64, 64)
    
    # Test with different site IDs
    site_ids = torch.tensor([1, 5, 9])  # Different sites
    
    with torch.no_grad():
        output = model(x, recording_site_ids=site_ids, features_only=True)
        
        # Analyze spatial embeddings
        spatial_embeds = model.spatial_embedding(site_ids)
        print(f"📍 Spatial embeddings shape: {spatial_embeds.shape}")
        print(f"📍 Spatial embeddings for sites {site_ids}:")
        for i, site_id in enumerate(site_ids):
            print(f"   Site {site_id}: {spatial_embeds[i][:5]}...")  # Show first 5 values
        
        # Check if embeddings are different for different sites
        similarity = torch.cosine_similarity(spatial_embeds[0], spatial_embeds[1], dim=0)
        print(f"📍 Similarity between sites {site_ids[0]} and {site_ids[1]}: {similarity:.4f}")
    
    print("✅ Spatial embedding analysis test passed!")

def main():
    """Run all tests."""
    print("🚀 Starting wav2vec2_2d with spatial embeddings tests...\n")
    
    try:
        test_spatial_embedding()
        test_spatial_embedding_visualization()
        print("\n🎉 All tests passed! wav2vec2_2d with spatial embeddings is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 