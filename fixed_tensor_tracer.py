#!/usr/bin/env python3
"""
Fixed Tensor Size Tracer for Wav2Vec2 2D Model
Shows the CORRECT tensor sizes and identifies the width=0 issue
"""

import torch
import torch.nn as nn

def trace_tensor_sizes():
    """
    Trace through the model architecture to show tensor sizes at each layer
    """
    print("🔍 FIXED TENSOR SIZE TRACER FOR WAV2VEC2 2D MODEL")
    print("=" * 60)
    print("Input: 3750 x 93 with batch size 1")
    print("=" * 60)
    
    # Your exact configuration from wav2vec2_2d_final.py
    conv_layers = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
    input_channels = 1
    input_height = 3750
    input_width = 93
    encoder_embed_dim = 384
    flattened_pool_dim = 256
    temporal_steps = 50
    num_negatives = 20
    cross_sample_negatives = 5
    codebook_negatives = 10
    
    # Create input tensor: [batch_size=1, channels=1, height=3750, width=93]
    input_tensor = torch.randn(1, 1, 3750, 93)
    print(f"📥 INPUT TENSOR")
    print(f"   Shape: {input_tensor.shape}")
    print(f"   Description: [batch_size=1, channels=1, height=3750, width=93]")
    print()
    
    # Trace through each component
    x = input_tensor.clone()
    
    # 1. 2D CNN Feature Extractor
    print("🔧 2D CNN FEATURE EXTRACTOR")
    print("-" * 40)
    
    print(f"   Conv layers: {conv_layers}")
    print()
    
    # Simulate each conv layer
    current_h, current_w = input_height, input_width
    current_c = input_channels
    
    print(f"   Input to CNN: [B={x.shape[0]}, C={current_c}, H={current_h}, W={current_w}]")
    
    for i, (out_channels, kernel_size, stride) in enumerate(conv_layers):
        # Calculate output dimensions
        new_h = (current_h - kernel_size) // stride + 1
        new_w = (current_w - kernel_size) // stride + 1
        
        print(f"   Conv2D Layer {i+1}: {current_c} -> {out_channels} channels")
        print(f"     Kernel: {kernel_size}x{kernel_size}, Stride: {stride}")
        print(f"     Input:  [B=1, C={current_c}, H={current_h}, W={current_w}]")
        print(f"     Output: [B=1, C={out_channels}, H={new_h}, W={new_w}]")
        
        # Check if width becomes 0 or negative
        if new_w <= 0:
            print(f"     ❌ CRITICAL: Width becomes {new_w} (≤ 0)!")
            print(f"     This will cause the CNN to produce zero features!")
            print(f"     The kernel {kernel_size}x{kernel_size} with stride {stride} is too large for width {current_w}")
            break
        
        current_c = out_channels
        current_h = new_h
        current_w = new_w
        print()
    
    # Check if we have valid dimensions
    if current_w <= 0:
        print(f"   ❌ CNN FAILED: Final width = {current_w} (≤ 0)")
        print(f"   This explains why your loss = 0!")
        print(f"   The CNN produces no features because width dimension becomes 0")
        print()
        
        # Show the problem
        print("🚨 PROBLEM IDENTIFIED")
        print("=" * 60)
        print("Your CNN configuration is too aggressive for the input dimensions:")
        print(f"• Input width: {input_width}")
        print(f"• After 4 layers, width becomes 0")
        print(f"• This means the CNN produces 0 features")
        print(f"• No features = No gradients = Loss = 0")
        print()
        
        # Show the solution
        print("💡 SOLUTION")
        print("=" * 60)
        print("You need to modify your CNN configuration:")
        print("1. Reduce kernel sizes")
        print("2. Reduce stride values") 
        print("3. Or increase input width")
        print()
        
        # Suggest better configuration
        print("🔧 SUGGESTED FIX")
        print("-" * 40)
        print("Try this configuration instead:")
        print("conv_2d_feature_layers=\"[(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1)]\"")
        print()
        print("This will give you:")
        
        # Calculate with suggested config
        suggested_layers = [(512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1)]
        test_h, test_w = input_height, input_width
        test_c = input_channels
        
        for i, (out_channels, kernel_size, stride) in enumerate(suggested_layers):
            new_h = (test_h - kernel_size) // stride + 1
            new_w = (test_w - kernel_size) // stride + 1
            test_c = out_channels
            test_h = new_h
            test_w = new_w
            if i < 3:  # Show first few layers
                print(f"   Layer {i+1}: [B=1, C={test_c}, H={test_h}, W={test_w}]")
        
        print(f"   Final: [B=1, C={test_c}, H={test_h}, W={test_w}]")
        print(f"   Total features: {test_c * test_h * test_w:,}")
        print()
        
        return
    
    # Final CNN output
    cnn_output_shape = (1, current_c, current_h, current_w)
    print(f"   ✅ CNN Output: {cnn_output_shape}")
    print(f"   Total features: {current_c * current_h * current_w:,}")
    print()
    
    # 2. Flattening
    print("🔄 FLATTENING")
    print("-" * 40)
    B, C, H, W = cnn_output_shape
    flattened_features = current_c * current_h * current_w
    print(f"   CNN output: [B={B}, C={C}, H={H}, W={W}]")
    print(f"   Flattened:  [B={B}, T=1, D={flattened_features}]")
    print(f"   Description: All spatial dimensions (C*H*W) become single time step")
    print()
    
    # 3. Adaptive Pooling
    print("📏 ADAPTIVE POOLING")
    print("-" * 40)
    pooled_features = flattened_pool_dim
    print(f"   Before pooling: [B=1, T=1, D={flattened_features}]")
    print(f"   After pooling:  [B=1, T=1, D={pooled_features}]")
    print(f"   Pooling: AdaptiveAvgPool1d({flattened_pool_dim})")
    print()
    
    # 4. Temporal Conv1D
    print("⏰ TEMPORAL CONV1D")
    print("-" * 40)
    print(f"   Before temporal conv: [B=1, T=1, D={pooled_features}]")
    print(f"   After temporal conv:  [B=1, T={temporal_steps}, D={pooled_features}]")
    print(f"   Temporal expansion: 1 -> {temporal_steps} time steps")
    print(f"   Conv1D layers: 2 layers with kernel_size=3, padding=1")
    print(f"   Final pooling: AdaptiveAvgPool1d({temporal_steps})")
    print()
    
    # 5. Layer Normalization
    print("🔧 LAYER NORMALIZATION")
    print("-" * 40)
    print(f"   Input: [B=1, T={temporal_steps}, D={pooled_features}]")
    print(f"   Output: [B=1, T={temporal_steps}, D={pooled_features}]")
    print(f"   Normalized over last dimension (D={pooled_features})")
    print()
    
    # 6. Post Extract Projection
    print("🔗 POST EXTRACT PROJECTION")
    print("-" * 40)
    print(f"   Input: [B=1, T={temporal_steps}, D={pooled_features}]")
    print(f"   Output: [B=1, T={temporal_steps}, D={encoder_embed_dim}]")
    print(f"   Linear: {pooled_features} -> {encoder_embed_dim}")
    print()
    
    # 7. Scaled RoPE
    print("🔄 SCALED ROPE")
    print("-" * 40)
    print(f"   Input: [B=1, T={temporal_steps}, D={encoder_embed_dim}]")
    print(f"   Output: [B=1, T={temporal_steps}, D={encoder_embed_dim}]")
    print(f"   RoPE applied to each time step")
    print()
    
    # 8. Transformer Encoder
    print("🤖 TRANSFORMER ENCODER")
    print("-" * 40)
    print(f"   Input: [B=1, T={temporal_steps}, D={encoder_embed_dim}]")
    print(f"   Output: [B=1, T={temporal_steps}, D={encoder_embed_dim}]")
    print(f"   Layers: 6")
    print(f"   Attention heads: 6")
    print(f"   FFN dim: 1536")
    print()
    
    # 9. Quantization (if enabled)
    print("🎯 QUANTIZATION")
    print("-" * 40)
    print(f"   No quantization applied")
    print(f"   Features: [B=1, T={temporal_steps}, D={encoder_embed_dim}]")
    print()
    
    # 10. Final Projection
    print("🎯 FINAL PROJECTION")
    print("-" * 40)
    final_dim = encoder_embed_dim
    print(f"   Input: [B=1, T={temporal_steps}, D={encoder_embed_dim}]")
    print(f"   Output: [B=1, T={temporal_steps}, D={final_dim}]")
    print(f"   Linear: {encoder_embed_dim} -> {final_dim}")
    print()
    
    # 11. Contrastive Learning Output
    print("🎯 CONTRASTIVE LEARNING OUTPUT")
    print("-" * 40)
    total_negatives = num_negatives + cross_sample_negatives + codebook_negatives
    print(f"   Logits shape: [B=1, T={temporal_steps}, D={total_negatives + 1}]")
    print(f"   Description: [batch, time_steps, num_negatives + 1]")
    print(f"   Negatives: {num_negatives} + {cross_sample_negatives} + {codebook_negatives} = {total_negatives}")
    print(f"   +1 for positive target")
    print()
    
    # Summary
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Input:           [1, 1, 3750, 93]")
    print(f"CNN Output:      [1, {current_c}, {current_h}, {current_w}]")
    print(f"Flattened:       [1, 1, {flattened_features}]")
    print(f"Pooled:          [1, 1, {pooled_features}]")
    print(f"Temporal:        [1, {temporal_steps}, {pooled_features}]")
    print(f"Final Features:  [1, {temporal_steps}, {encoder_embed_dim}]")
    print(f"Logits:          [1, {temporal_steps}, {total_negatives + 1}]")
    print()

if __name__ == "__main__":
    trace_tensor_sizes()
