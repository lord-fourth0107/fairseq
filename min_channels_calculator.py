#!/usr/bin/env python3
"""
Minimum Channels Calculator for Wav2Vec2 2D CNN
Calculates the minimum number of channels needed to prevent zero output
"""

def calculate_min_channels():
    """
    Calculate minimum channels needed for each CNN layer configuration
    """
    print("🔍 MINIMUM CHANNELS CALCULATOR")
    print("=" * 60)
    print("Input: 3750 x 93 with batch size 1")
    print("=" * 60)
    
    # Your current problematic config
    current_config = [(512, 10, 5), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 3, 2), (512, 2, 2), (512, 2, 2)]
    
    print("📊 ANALYZING CURRENT CONFIGURATION")
    print("-" * 40)
    print("Current config:", current_config)
    print()
    
    # Test current config
    print("Testing current config:")
    current_h, current_w = 3750, 93
    current_c = 1
    
    for i, (out_channels, kernel_size, stride) in enumerate(current_config):
        new_h = (current_h - kernel_size) // stride + 1
        new_w = (current_w - kernel_size) // stride + 1
        
        print(f"   Layer {i+1}: {current_c} -> {out_channels} channels")
        print(f"     Kernel: {kernel_size}x{kernel_size}, Stride: {stride}")
        print(f"     Input:  [B=1, C={current_c}, H={current_h}, W={current_w}]")
        print(f"     Output: [B=1, C={out_channels}, H={new_h}, W={new_w}]")
        
        if new_w <= 0:
            print(f"     ❌ FAILS: Width becomes {new_w} (≤ 0)")
            print(f"     This is where the CNN produces zero output!")
            break
        elif new_h <= 0:
            print(f"     ❌ FAILS: Height becomes {new_h} (≤ 0)")
            print(f"     This is where the CNN produces zero output!")
            break
        else:
            print(f"     ✅ OK")
        
        current_c = out_channels
        current_h = new_h
        current_w = new_w
        print()
    
    print("=" * 60)
    print("🔧 FINDING MINIMUM CHANNELS FOR EACH LAYER")
    print("=" * 60)
    
    # Test different configurations
    configs_to_test = [
        # Conservative configs
        ([(64, 3, 1), (128, 3, 1), (256, 3, 1), (512, 3, 1)], "Conservative"),
        ([(128, 3, 1), (256, 3, 1), (512, 3, 1), (512, 3, 1)], "Moderate"),
        ([(256, 3, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1)], "Aggressive"),
        
        # With some stride
        ([(64, 3, 1), (128, 3, 2), (256, 3, 1), (512, 3, 1)], "Mixed 1"),
        ([(128, 3, 1), (256, 3, 2), (512, 3, 1), (512, 3, 1)], "Mixed 2"),
        ([(256, 3, 1), (512, 3, 2), (512, 3, 1), (512, 3, 1)], "Mixed 3"),
        
        # With larger kernels but smaller strides
        ([(64, 5, 1), (128, 3, 1), (256, 3, 1), (512, 3, 1)], "Large Kernel 1"),
        ([(128, 5, 1), (256, 3, 1), (512, 3, 1), (512, 3, 1)], "Large Kernel 2"),
        ([(256, 5, 1), (512, 3, 1), (512, 3, 1), (512, 3, 1)], "Large Kernel 3"),
        
        # Your original but with smaller strides
        ([(512, 10, 2), (512, 3, 1), (512, 3, 1), (512, 3, 1)], "Original Modified 1"),
        ([(512, 10, 3), (512, 3, 1), (512, 3, 1), (512, 3, 1)], "Original Modified 2"),
        ([(512, 10, 4), (512, 3, 1), (512, 3, 1), (512, 3, 1)], "Original Modified 3"),
    ]
    
    successful_configs = []
    
    for config, name in configs_to_test:
        print(f"Testing {name}: {config}")
        
        test_h, test_w = 3750, 93
        test_c = 1
        success = True
        
        for i, (out_channels, kernel_size, stride) in enumerate(config):
            new_h = (test_h - kernel_size) // stride + 1
            new_w = (test_w - kernel_size) // stride + 1
            
            if new_w <= 0 or new_h <= 0:
                print(f"   ❌ FAILS at layer {i+1}: H={new_h}, W={new_w}")
                success = False
                break
            
            test_c = out_channels
            test_h = new_h
            test_w = new_w
        
        if success:
            print(f"   ✅ SUCCESS: Final [B=1, C={test_c}, H={test_h}, W={test_w}]")
            print(f"   Total features: {test_c * test_h * test_w:,}")
            successful_configs.append((config, name, test_c, test_h, test_w))
        print()
    
    print("=" * 60)
    print("📊 SUCCESSFUL CONFIGURATIONS")
    print("=" * 60)
    
    if successful_configs:
        print("These configurations work without producing zero output:")
        print()
        
        for i, (config, name, final_c, final_h, final_w) in enumerate(successful_configs):
            total_features = final_c * final_h * final_w
            print(f"{i+1}. {name}")
            print(f"   Config: {config}")
            print(f"   Final: [B=1, C={final_c}, H={final_h}, W={final_w}]")
            print(f"   Features: {total_features:,}")
            print()
        
        # Find the minimum channels needed
        min_channels = min(config[0][0] for config, _, _, _, _ in successful_configs)
        print(f"🎯 MINIMUM CHANNELS NEEDED: {min_channels}")
        print()
        
        # Show the best configurations
        print("🏆 RECOMMENDED CONFIGURATIONS")
        print("-" * 40)
        
        # Sort by total features (more features = better representation)
        successful_configs.sort(key=lambda x: x[2] * x[3] * x[4], reverse=True)
        
        for i, (config, name, final_c, final_h, final_w) in enumerate(successful_configs[:3]):
            total_features = final_c * final_h * final_w
            print(f"Option {i+1}: {name}")
            print(f"   conv_2d_feature_layers=\"{config}\"")
            print(f"   Final features: {total_features:,}")
            print()
        
        # Show the most conservative option
        conservative_configs = [c for c in successful_configs if c[0][0][0] <= 128]
        if conservative_configs:
            best_conservative = min(conservative_configs, key=lambda x: x[0][0][0])
            print("💡 MOST CONSERVATIVE (safest):")
            print(f"   conv_2d_feature_layers=\"{best_conservative[0]}\"")
            print(f"   Starts with only {best_conservative[0][0][0]} channels")
            print()
        
    else:
        print("❌ NO CONFIGURATIONS FOUND!")
        print("All tested configurations produce zero output.")
        print("You may need to reduce the number of layers or use even smaller kernels/strides.")
    
    print("=" * 60)
    print("🔧 GENERAL RULES FOR PREVENTING ZERO OUTPUT")
    print("=" * 60)
    print("1. Start with smaller channel counts (64, 128, 256)")
    print("2. Use stride=1 for most layers")
    print("3. Use smaller kernels (3x3 instead of 10x10)")
    print("4. Test each layer incrementally")
    print("5. Ensure width and height never become ≤ 0")
    print()
    print("Formula: output_size = (input_size - kernel_size) // stride + 1")
    print("For width=93: 93 - kernel_size must be ≥ 0 for stride=1")
    print("So kernel_size ≤ 93 for stride=1")

if __name__ == "__main__":
    calculate_min_channels()
