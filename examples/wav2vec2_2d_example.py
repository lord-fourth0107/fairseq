#!/usr/bin/env python3
"""
Example usage of wav2vec2_2d model with 2D CNN feature extraction.
"""

import torch
import torch.nn as nn
from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel


def create_wav2vec2_2d_model():
    """Create a wav2vec2_2d model with 2D CNN feature extraction."""
    
    # Configuration for 2D CNN wav2vec2
    cfg = Wav2Vec2_2DConfig(
        # 2D CNN specific parameters
        conv_2d_feature_layers="[(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2)]",
        input_channels=1,  # Mono audio
        input_height=128,  # Spectrogram height
        input_width=128,   # Spectrogram width
        
        # Transformer parameters
        encoder_layers=12,
        encoder_embed_dim=768,
        encoder_ffn_embed_dim=3072,
        encoder_attention_heads=12,
        
        # Masking parameters
        mask_prob=0.65,
        mask_length=10,
        
        # Other parameters
        dropout=0.1,
        attention_dropout=0.1,
        activation_dropout=0.0,
        layer_norm_first=False,
        feature_grad_mult=1.0,
    )
    
    # Create model
    model = Wav2Vec2_2DModel(cfg)
    return model


def example_usage():
    """Example of how to use the wav2vec2_2d model."""
    
    # Create model
    model = create_wav2vec2_2d_model()
    model.eval()
    
    # Example input: (batch_size, channels, height, width)
    # This could be a spectrogram or other 2D audio representation
    batch_size = 2
    channels = 1  # Mono audio
    height = 128  # Spectrogram height
    width = 128   # Spectrogram width
    
    # Create dummy input (replace with actual spectrogram)
    dummy_input = torch.randn(batch_size, channels, height, width)
    
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model.forward(
            source=dummy_input,
            padding_mask=None,
            mask=False,  # No masking for inference
            features_only=True
        )
    
    print(f"Output keys: {output.keys()}")
    print(f"Features shape: {output['x'].shape}")
    print(f"Layer results: {len(output['layer_results'])} layers")


def compare_with_original():
    """Compare 2D CNN with original 1D CNN architecture."""
    
    print("=== 2D CNN wav2vec2 ===")
    model_2d = create_wav2vec2_2d_model()
    
    # Count parameters
    total_params = sum(p.numel() for p in model_2d.parameters())
    trainable_params = sum(p.numel() for p in model_2d.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    dummy_input = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        output_2d = model_2d.forward(dummy_input, features_only=True)
    
    print(f"2D CNN output shape: {output_2d['x'].shape}")
    
    print("\n=== Architecture Summary ===")
    print("1. 2D CNN Feature Extractor:")
    print("   - Input: (B, 1, 128, 128) - spectrogram")
    print("   - Layers: 4 conv2d layers with increasing channels")
    print("   - Output: (B, 512, H', W') - 2D features")
    
    print("\n2. 2D to 1D Conversion:")
    print("   - Reshape: (B, H'*W', 512) - flatten spatial dimensions")
    print("   - Project: Linear layer to transformer dimension")
    
    print("\n3. Transformer Encoder:")
    print("   - Same as original wav2vec2")
    print("   - 12 transformer layers")
    print("   - 768 embedding dimension")


if __name__ == "__main__":
    print("Wav2Vec2 2D CNN Example")
    print("=" * 50)
    
    example_usage()
    print("\n" + "=" * 50)
    compare_with_original() 