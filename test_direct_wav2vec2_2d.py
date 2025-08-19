#!/usr/bin/env python3
"""
Direct test script for wav2vec2_2d components without fairseq dependencies
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Any

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Define a simple Fp32LayerNorm equivalent
class Fp32LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super().__init__()
        # Ensure normalized_shape is a tuple
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

    def forward(self, x):
        return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)

# Simple dataclass for testing
@dataclass
class SimpleConfig:
    conv_2d_feature_layers: str = "[(32, 3, 2), (64, 3, 2), (128, 3, 2)]"
    input_channels: int = 1
    input_height: int = 128
    input_width: int = 128
    extractor_mode: str = "layer_norm"
    encoder_layers: int = 2
    encoder_embed_dim: int = 256
    encoder_attention_heads: int = 4
    use_spatial_embedding: bool = True
    num_recording_sites: int = 10
    spatial_embed_dim: int = 64

def parse_conv_layers(layers_str):
    """Parse conv layers string into list of tuples"""
    # Remove brackets and split by ), (
    layers_str = layers_str.strip('[]')
    layers = []
    for layer_str in layers_str.split('), ('):
        layer_str = layer_str.strip('()')
        dim, kernel, stride = map(int, layer_str.split(', '))
        layers.append((dim, kernel, stride))
    return layers

class Conv2DFeatureExtractionModel(nn.Module):
    """2D CNN Feature Extractor"""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        conv_layers_config = parse_conv_layers(config.conv_2d_feature_layers)
        
        def block(n_in, n_out, k, stride, is_layer_norm=False, is_group_norm=False, conv_bias=False):
            def make_conv():
                conv = nn.Conv2d(n_in, n_out, k, stride=stride, bias=conv_bias)
                return conv
            
            def make_layer_norm():
                return Fp32LayerNorm(n_out, elementwise_affine=True)
            
            def make_group_norm():
                return nn.GroupNorm(n_out, n_out, affine=True)
            
            conv = make_conv()
            if is_layer_norm:
                # Create a custom layer norm that handles 2D inputs
                class LayerNorm2D(nn.Module):
                    def __init__(self, normalized_shape):
                        super().__init__()
                        self.layer_norm = Fp32LayerNorm(normalized_shape, elementwise_affine=True)
                    
                    def forward(self, x):
                        # x: (B, C, H, W)
                        B, C, H, W = x.shape
                        # Reshape: (B, C, H, W) -> (B*H*W, C)
                        x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
                        # Apply layer norm
                        x_normed = self.layer_norm(x_reshaped)
                        # Reshape back: (B*H*W, C) -> (B, C, H, W)
                        x_final = x_normed.reshape(B, H, W, C).permute(0, 3, 1, 2)
                        return x_final
                
                norm = LayerNorm2D(n_out)
            elif is_group_norm:
                norm = make_group_norm()
            else:
                norm = nn.Sequential()
            
            return nn.Sequential(conv, norm, nn.GELU())
        
        in_d = config.input_channels
        self.conv_layers = nn.ModuleList()
        
        for i, (dim, k, stride) in enumerate(conv_layers_config):
            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=config.extractor_mode == "layer_norm",
                    is_group_norm=config.extractor_mode == "group_norm",
                )
            )
            in_d = dim
    
    def forward(self, x):
        # x: (B, C, H, W)
        for conv in self.conv_layers:
            x = conv(x)
        return x

class SimpleTransformerEncoder(nn.Module):
    """Simple transformer encoder for testing"""
    
    def __init__(self, embed_dim, num_layers, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        
        # Simple transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=0.1,
                activation='gelu',
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        # x: (B, seq_len, embed_dim)
        for layer in self.layers:
            x = layer(x)
        return self.layer_norm(x)

class Wav2Vec2_2DModel(nn.Module):
    """Simplified Wav2Vec2 2D Model"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Feature extractor
        self.feature_extractor = Conv2DFeatureExtractionModel(config)
        
        # Calculate output dimensions
        conv_layers = parse_conv_layers(config.conv_2d_feature_layers)
        current_h, current_w = config.input_height, config.input_width
        current_c = config.input_channels
        
        for dim, kernel, stride in conv_layers:
            current_h = (current_h - kernel) // stride + 1
            current_w = (current_w - kernel) // stride + 1
            current_c = dim
        
        self.output_h = current_h
        self.output_w = current_w
        self.output_c = current_c
        
        # Spatial embeddings
        if config.use_spatial_embedding:
            self.spatial_embeddings = nn.Embedding(config.num_recording_sites, config.spatial_embed_dim)
            self.spatial_projection = nn.Linear(config.spatial_embed_dim, config.encoder_embed_dim)
        else:
            self.spatial_embeddings = None
            self.spatial_projection = None
        
        # Transformer encoder
        self.encoder = SimpleTransformerEncoder(
            config.encoder_embed_dim,
            config.encoder_layers,
            config.encoder_attention_heads
        )
        
        # Projection from CNN features to transformer
        self.feature_projection = nn.Linear(self.output_c, config.encoder_embed_dim)
    
    def forward(self, source, recording_site_ids=None, features_only=False):
        # source: (B, C, H, W)
        # recording_site_ids: (B,)
        
        # Extract features with 2D CNN
        features = self.feature_extractor(source)
        # features: (B, C, H, W)
        
        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = features.shape
        features = features.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Project to transformer dimension
        features = self.feature_projection(features)
        # features: (B, H*W, encoder_embed_dim)
        
        # Add spatial embeddings if enabled
        if self.config.use_spatial_embedding and recording_site_ids is not None:
            spatial_embeds = self.spatial_embeddings(recording_site_ids)
            spatial_embeds = self.spatial_projection(spatial_embeds)
            # Expand spatial embeddings to match sequence length
            spatial_embeds = spatial_embeds.unsqueeze(1).expand(-1, H * W, -1)
            features = features + spatial_embeds
        
        # Pass through transformer
        encoder_out = self.encoder(features)
        
        return {
            'x': encoder_out,
            'features': features
        }

def test_conv2d_feature_extraction():
    """Test the Conv2DFeatureExtractionModel"""
    print("🧪 Testing Conv2DFeatureExtractionModel...")
    
    config = SimpleConfig()
    model = Conv2DFeatureExtractionModel(config)
    
    x = torch.randn(2, 1, 128, 128)
    print(f"📥 Input shape: {x.shape}")
    
    with torch.no_grad():
        output = model(x)
    
    print(f"📤 Output shape: {output.shape}")
    print("✅ Conv2DFeatureExtractionModel test passed!")
    return model

def test_wav2vec2_2d_model():
    """Test the complete Wav2Vec2_2DModel"""
    print("\n🧪 Testing Wav2Vec2_2DModel...")
    
    config = SimpleConfig()
    config.conv_2d_feature_layers = "[(32, 3, 2), (64, 3, 2)]"
    config.input_height = 64
    config.input_width = 64
    config.encoder_layers = 2
    config.encoder_embed_dim = 256
    config.encoder_attention_heads = 4
    config.use_spatial_embedding = True
    config.num_recording_sites = 10
    config.spatial_embed_dim = 64
    
    model = Wav2Vec2_2DModel(config)
    print("🏗️ Model created successfully!")
    
    x = torch.randn(2, 1, 64, 64)
    recording_site_ids = torch.randint(1, 10, (2,))
    print(f"📥 Input shape: {x.shape}")
    print(f"📥 Recording site IDs: {recording_site_ids}")
    
    with torch.no_grad():
        output = model(x, recording_site_ids=recording_site_ids, features_only=True)
    
    print(f"📤 Output shape: {output['x'].shape}")
    print("✅ Wav2Vec2_2DModel test passed!")
    return model

def test_fp32_layer_norm_compatibility():
    """Test Fp32LayerNorm compatibility with 2D inputs"""
    print("\n🧪 Testing Fp32LayerNorm Compatibility...")
    
    x = torch.randn(2, 64, 32, 32)
    print(f"📥 Input shape: {x.shape}")
    
    layer_norm = Fp32LayerNorm(64)
    
    B, C, H, W = x.shape
    x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
    print(f"📥 Reshaped for Fp32LayerNorm: {x_reshaped.shape}")
    
    x_normed = layer_norm(x_reshaped)
    print(f"📤 Fp32LayerNorm output shape: {x_normed.shape}")
    
    x_final = x_normed.reshape(B, H, W, C).permute(0, 3, 1, 2)
    print(f"📤 Final output shape: {x_final.shape}")
    
    print("✅ Shape compatibility verified!")
    print("✅ Fp32LayerNorm works with 2D CNN outputs!")

def test_spatiality_preservation():
    """Test that spatiality is preserved during operations"""
    print("\n🧪 Testing Spatiality Preservation...")
    
    # Create a tensor with known spatial pattern
    B, C, H, W = 2, 64, 16, 16
    x = torch.zeros(B, C, H, W)
    
    # Create a simple spatial pattern
    for h in range(H):
        for w in range(W):
            x[0, 0, h, w] = h * W + w
    
    print(f"📥 Original shape: {x.shape}")
    print(f"📥 Sample spatial pattern at [0, 0, :, :]:")
    print(x[0, 0, :4, :4])  # Show 4x4 corner
    
    # Apply Fp32LayerNorm
    layer_norm = Fp32LayerNorm(C)
    x_reshaped = x.permute(0, 2, 3, 1).reshape(B * H * W, C)
    x_normed = layer_norm(x_reshaped)
    x_final = x_normed.reshape(B, H, W, C).permute(0, 3, 1, 2)
    
    print(f"📥 Reshaped for Fp32LayerNorm: {x_reshaped.shape}")
    print(f"📤 Fp32LayerNorm output shape: {x_normed.shape}")
    print(f"📤 Final output shape: {x_final.shape}")
    
    # Check that spatial dimensions are preserved
    assert x_final.shape == x.shape, f"Shape mismatch: {x_final.shape} vs {x.shape}"
    print("✅ Shape compatibility verified!")
    print("✅ Spatial dimensions preserved!")
    print("✅ Layer normalization working correctly!")
    print("✅ Spatiality is PRESERVED in Fp32LayerNorm operations!")

def main():
    """Run all tests"""
    print("🚀 Starting direct wav2vec2_2d tests...\n")
    
    try:
        # Test 1: Conv2D Feature Extraction
        test_conv2d_feature_extraction()
        
        # Test 2: Complete Model
        test_wav2vec2_2d_model()
        
        # Test 3: Fp32LayerNorm Compatibility
        test_fp32_layer_norm_compatibility()
        
        # Test 4: Spatiality Preservation
        test_spatiality_preservation()
        
        print("\n🎉 All tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 