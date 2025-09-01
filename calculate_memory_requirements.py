#!/usr/bin/env python3
"""
Calculate memory requirements for Wav2Vec2 2D model
"""

import torch
import torch.nn as nn

def calculate_model_parameters():
    """Calculate the number of parameters in the model"""
    
    # Model configuration from your script
    config = {
        'conv_2d_feature_layers': [(64, 3, 2), (128, 3, 2), (256, 3, 2), (512, 3, 2)],
        'input_channels': 1,
        'input_height': 3750,
        'input_width': 93,
        'encoder_layers': 12,
        'encoder_embed_dim': 768,
        'encoder_ffn_embed_dim': 3072,
        'encoder_attention_heads': 12,
        'spatial_embed_dim': 256,
        'num_recording_sites': 64,
    }
    
    print("🔍 Wav2Vec2 2D Model Memory Analysis")
    print("=" * 60)
    
    # 1. CNN Feature Extractor
    print("📊 CNN Feature Extractor:")
    conv_params = 0
    input_size = (1, 3750, 93)  # (C, H, W)
    
    for i, (out_channels, kernel_size, stride) in enumerate(config['conv_2d_feature_layers']):
        in_channels = config['input_channels'] if i == 0 else config['conv_2d_feature_layers'][i-1][0]
        
        # Conv2d parameters: (out_channels, in_channels, kernel_h, kernel_w)
        conv_params += out_channels * in_channels * kernel_size * kernel_size
        conv_params += out_channels  # bias
        
        # Calculate output size
        h_out = (input_size[1] - kernel_size) // stride + 1
        w_out = (input_size[2] - kernel_size) // stride + 1
        input_size = (out_channels, h_out, w_out)
        
        print(f"   Layer {i+1}: {in_channels}→{out_channels}, kernel={kernel_size}, stride={stride}")
        print(f"   Output size: {input_size}")
    
    print(f"   Total CNN parameters: {conv_params:,}")
    
    # 2. Transformer Encoder
    print("\n🤖 Transformer Encoder:")
    
    # Embedding layer
    embed_params = config['encoder_embed_dim'] * config['encoder_embed_dim']  # post_extract_proj
    print(f"   Post-extract projection: {embed_params:,}")
    
    # Transformer layers (12 layers)
    layer_params = 0
    for layer in range(config['encoder_layers']):
        # Self-attention
        qkv_params = 3 * config['encoder_embed_dim'] * config['encoder_embed_dim']
        attn_out_params = config['encoder_embed_dim'] * config['encoder_embed_dim']
        
        # FFN
        ffn1_params = config['encoder_embed_dim'] * config['encoder_ffn_embed_dim']
        ffn2_params = config['encoder_ffn_embed_dim'] * config['encoder_embed_dim']
        
        # Layer norms (2 per layer)
        ln_params = 2 * config['encoder_embed_dim']
        
        layer_total = qkv_params + attn_out_params + ffn1_params + ffn2_params + ln_params
        layer_params += layer_total
    
    print(f"   Per layer parameters: ~{layer_params // config['encoder_layers']:,}")
    print(f"   Total transformer parameters: {layer_params:,}")
    
    # 3. Spatial Embeddings
    print("\n🗺️ Spatial Embeddings:")
    spatial_params = config['num_recording_sites'] * config['spatial_embed_dim']
    spatial_proj_params = config['spatial_embed_dim'] * config['encoder_embed_dim']
    total_spatial = spatial_params + spatial_proj_params
    print(f"   Spatial embedding: {spatial_params:,}")
    print(f"   Spatial projection: {spatial_proj_params:,}")
    print(f"   Total spatial parameters: {total_spatial:,}")
    
    # 4. Total Model Parameters
    total_params = conv_params + embed_params + layer_params + total_spatial
    print(f"\n📈 TOTAL MODEL PARAMETERS: {total_params:,}")
    
    # 5. Memory Requirements
    print("\n💾 Memory Requirements:")
    
    # Model parameters (FP32)
    model_memory_fp32 = total_params * 4  # 4 bytes per parameter
    model_memory_fp16 = total_params * 2  # 2 bytes per parameter
    
    print(f"   Model parameters (FP32): {model_memory_fp32 / 1e9:.2f} GB")
    print(f"   Model parameters (FP16): {model_memory_fp16 / 1e9:.2f} GB")
    
    # Gradients (same size as parameters)
    print(f"   Gradients (FP32): {model_memory_fp32 / 1e9:.2f} GB")
    print(f"   Gradients (FP16): {model_memory_fp16 / 1e9:.2f} GB")
    
    # Optimizer states (Adam: 2x parameters for momentum and variance)
    optimizer_memory_fp32 = total_params * 4 * 2  # 2x for momentum and variance
    optimizer_memory_fp16 = total_params * 2 * 2
    print(f"   Optimizer states (FP32): {optimizer_memory_fp32 / 1e9:.2f} GB")
    print(f"   Optimizer states (FP16): {optimizer_memory_fp16 / 1e9:.2f} GB")
    
    # Forward pass activations
    print("\n🔄 Forward Pass Activations:")
    
    # Input tensor
    batch_size = 1
    input_memory = batch_size * config['input_channels'] * config['input_height'] * config['input_width'] * 4
    print(f"   Input tensor: {input_memory / 1e6:.2f} MB")
    
    # CNN outputs (approximate)
    cnn_output_memory = batch_size * 512 * 117 * 12 * 4  # Approximate after all conv layers
    print(f"   CNN outputs: {cnn_output_memory / 1e6:.2f} MB")
    
    # Transformer activations (approximate)
    seq_len = 117 * 12  # Approximate sequence length after CNN
    transformer_memory = batch_size * seq_len * config['encoder_embed_dim'] * 4
    print(f"   Transformer activations: {transformer_memory / 1e6:.2f} MB")
    
    # Total forward pass memory
    forward_memory = input_memory + cnn_output_memory + transformer_memory
    print(f"   Total forward pass: {forward_memory / 1e6:.2f} MB")
    
    # 6. Total Memory Requirements
    print("\n🎯 TOTAL MEMORY REQUIREMENTS:")
    
    # Training memory (model + gradients + optimizer + activations)
    training_memory_fp32 = model_memory_fp32 + model_memory_fp32 + optimizer_memory_fp32 + forward_memory
    training_memory_fp16 = model_memory_fp16 + model_memory_fp16 + optimizer_memory_fp16 + forward_memory
    
    print(f"   Training (FP32): {training_memory_fp32 / 1e9:.2f} GB")
    print(f"   Training (FP16): {training_memory_fp16 / 1e9:.2f} GB")
    
    # Add overhead (20% for PyTorch, CUDA, etc.)
    overhead_factor = 1.2
    total_memory_fp32 = training_memory_fp32 * overhead_factor
    total_memory_fp16 = training_memory_fp16 * overhead_factor
    
    print(f"   With overhead (FP32): {total_memory_fp32 / 1e9:.2f} GB")
    print(f"   With overhead (FP16): {total_memory_fp16 / 1e9:.2f} GB")
    
    # 7. Resource Assessment
    print("\n✅ RESOURCE ASSESSMENT:")
    print(f"   Your allocation: 50 GB RAM")
    print(f"   Required (FP32): {total_memory_fp32 / 1e9:.2f} GB")
    print(f"   Required (FP16): {total_memory_fp16 / 1e9:.2f} GB")
    
    if total_memory_fp32 <= 50e9:
        print("   ✅ FP32 training should work with 50GB RAM")
    else:
        print("   ⚠️ FP32 training may exceed 50GB RAM")
    
    if total_memory_fp16 <= 50e9:
        print("   ✅ FP16 training should work with 50GB RAM")
    else:
        print("   ⚠️ FP16 training may exceed 50GB RAM")
    
    # 8. Recommendations
    print("\n💡 RECOMMENDATIONS:")
    if total_memory_fp32 > 50e9:
        print("   🔧 Enable mixed precision training (FP16)")
        print("   🔧 Reduce batch size if needed")
        print("   🔧 Use gradient checkpointing")
    
    print("   🔧 Monitor GPU memory usage during training")
    print("   🔧 Use torch.cuda.empty_cache() periodically")
    
    return total_memory_fp32, total_memory_fp16

if __name__ == "__main__":
    calculate_model_parameters()
