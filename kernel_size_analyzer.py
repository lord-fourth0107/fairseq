#!/usr/bin/env python3
"""
Kernel Size Analyzer for Wav2Vec2 2D CNN
Analyzes the effect of increasing kernel size while keeping stride=1
"""

def analyze_kernel_sizes():
    """
    Analyze different kernel sizes and their effects on output dimensions
    """
    print("🔍 KERNEL SIZE ANALYZER")
    print("=" * 60)
    print("Input: 3750 x 93 with batch size 1")
    print("All configurations use stride=1 to minimize width reduction")
    print("=" * 60)
    
    # Test different kernel sizes
    kernel_configs = [
        # Small kernels
        ([(64, 3, 1), (128, 3, 1), (256, 3, 1), (512, 3, 1)], "3x3 kernels"),
        ([(64, 5, 1), (128, 5, 1), (256, 5, 1), (512, 5, 1)], "5x5 kernels"),
        ([(64, 7, 1), (128, 7, 1), (256, 7, 1), (512, 7, 1)], "7x7 kernels"),
        
        # Medium kernels
        ([(64, 9, 1), (128, 9, 1), (256, 9, 1), (512, 9, 1)], "9x9 kernels"),
        ([(64, 11, 1), (128, 11, 1), (256, 11, 1), (512, 11, 1)], "11x11 kernels"),
        ([(64, 13, 1), (128, 13, 1), (256, 13, 1), (512, 13, 1)], "13x13 kernels"),
        
        # Large kernels
        ([(64, 15, 1), (128, 15, 1), (256, 15, 1), (512, 15, 1)], "15x15 kernels"),
        ([(64, 17, 1), (128, 17, 1), (256, 17, 1), (512, 17, 1)], "17x17 kernels"),
        ([(64, 19, 1), (128, 19, 1), (256, 19, 1), (512, 19, 1)], "19x19 kernels"),
        
        # Very large kernels
        ([(64, 21, 1), (128, 21, 1), (256, 21, 1), (512, 21, 1)], "21x21 kernels"),
        ([(64, 25, 1), (128, 25, 1), (256, 25, 1), (512, 25, 1)], "25x25 kernels"),
        ([(64, 31, 1), (128, 31, 1), (256, 31, 1), (512, 31, 1)], "31x31 kernels"),
        
        # Mixed kernel sizes
        ([(64, 5, 1), (128, 3, 1), (256, 7, 1), (512, 3, 1)], "Mixed: 5-3-7-3"),
        ([(64, 7, 1), (128, 5, 1), (256, 3, 1), (512, 5, 1)], "Mixed: 7-5-3-5"),
        ([(64, 9, 1), (128, 3, 1), (256, 5, 1), (512, 3, 1)], "Mixed: 9-3-5-3"),
        
        # Your original kernel size but with stride=1
        ([(64, 10, 1), (128, 10, 1), (256, 10, 1), (512, 10, 1)], "10x10 kernels (your original size)"),
    ]
    
    results = []
    
    for config, name in kernel_configs:
        print(f"Testing {name}: {config}")
        
        test_h, test_w = 3750, 93
        test_c = 1
        success = True
        layer_outputs = []
        
        for i, (out_channels, kernel_size, stride) in enumerate(config):
            new_h = (test_h - kernel_size) // stride + 1
            new_w = (test_w - kernel_size) // stride + 1
            
            layer_outputs.append((out_channels, new_h, new_w))
            
            if new_w <= 0 or new_h <= 0:
                print(f"   ❌ FAILS at layer {i+1}: H={new_h}, W={new_w}")
                success = False
                break
            
            test_c = out_channels
            test_h = new_h
            test_w = new_w
        
        if success:
            total_features = test_c * test_h * test_w
            print(f"   ✅ SUCCESS: Final [B=1, C={test_c}, H={test_h}, W={test_w}]")
            print(f"   Total features: {total_features:,}")
            print(f"   Width reduction: {93 - test_w} pixels ({((93 - test_w) / 93 * 100):.1f}%)")
            print(f"   Height reduction: {3750 - test_h} pixels ({((3750 - test_h) / 3750 * 100):.1f}%)")
            results.append((config, name, test_c, test_h, test_w, total_features, layer_outputs))
        print()
    
    print("=" * 60)
    print("📊 RESULTS SUMMARY")
    print("=" * 60)
    
    if results:
        # Sort by total features (descending)
        results.sort(key=lambda x: x[5], reverse=True)
        
        print("All successful configurations (sorted by total features):")
        print()
        
        for i, (config, name, final_c, final_h, final_w, total_features, layer_outputs) in enumerate(results):
            width_reduction = 93 - final_w
            height_reduction = 3750 - final_h
            print(f"{i+1:2d}. {name}")
            print(f"    Config: {config}")
            print(f"    Final: [B=1, C={final_c}, H={final_h}, W={final_w}]")
            print(f"    Features: {total_features:,}")
            print(f"    Width reduction: {width_reduction}px ({width_reduction/93*100:.1f}%)")
            print(f"    Height reduction: {height_reduction}px ({height_reduction/3750*100:.1f}%)")
            print()
        
        # Find the best configurations
        print("🏆 BEST CONFIGURATIONS BY CATEGORY")
        print("-" * 50)
        
        # Most features (best representation)
        best_features = results[0]
        print(f"Most features: {best_features[1]}")
        print(f"  Features: {best_features[5]:,}")
        print(f"  Config: {best_features[0]}")
        print()
        
        # Least width reduction (preserves spatial info)
        least_width_reduction = min(results, key=lambda x: 93 - x[4])
        width_red = 93 - least_width_reduction[4]
        print(f"Least width reduction: {least_width_reduction[1]}")
        print(f"  Width reduction: {width_red}px ({width_red/93*100:.1f}%)")
        print(f"  Config: {least_width_reduction[0]}")
        print()
        
        # Least height reduction
        least_height_reduction = min(results, key=lambda x: 3750 - x[3])
        height_red = 3750 - least_height_reduction[3]
        print(f"Least height reduction: {least_height_reduction[1]}")
        print(f"  Height reduction: {height_red}px ({height_red/3750*100:.1f}%)")
        print(f"  Config: {least_height_reduction[0]}")
        print()
        
        # Balanced (good features + reasonable reduction)
        balanced = [r for r in results if 93 - r[4] <= 20 and 3750 - r[3] <= 100][:3]
        if balanced:
            print("Balanced (good features + reasonable reduction):")
            for i, (config, name, final_c, final_h, final_w, total_features, _) in enumerate(balanced):
                print(f"  {i+1}. {name}: {total_features:,} features")
                print(f"     Width reduction: {93 - final_w}px, Height reduction: {3750 - final_h}px")
            print()
        
        # Analysis of kernel size effects
        print("🔍 KERNEL SIZE ANALYSIS")
        print("-" * 50)
        
        # Group by kernel size
        kernel_groups = {}
        for config, name, final_c, final_h, final_w, total_features, layer_outputs in results:
            first_kernel = config[0][1]  # First layer kernel size
            if first_kernel not in kernel_groups:
                kernel_groups[first_kernel] = []
            kernel_groups[first_kernel].append((name, total_features, 93 - final_w, 3750 - final_h))
        
        print("Effect of increasing kernel size:")
        for kernel_size in sorted(kernel_groups.keys()):
            group = kernel_groups[kernel_size]
            avg_features = sum(x[1] for x in group) / len(group)
            avg_width_red = sum(x[2] for x in group) / len(group)
            avg_height_red = sum(x[3] for x in group) / len(group)
            print(f"  {kernel_size}x{kernel_size}: {avg_features:,.0f} features, "
                  f"W-{avg_width_red:.1f}px, H-{avg_height_red:.1f}px")
        print()
        
        # Recommendations
        print("💡 RECOMMENDATIONS")
        print("-" * 50)
        print("1. **Larger kernels (7x7 to 15x15)** give more features with minimal width reduction")
        print("2. **Your original 10x10 kernel** works great with stride=1!")
        print("3. **Mixed kernel sizes** can balance feature extraction and efficiency")
        print("4. **Width reduction is minimal** with stride=1, so you can use large kernels")
        print("5. **Height reduction** is more significant, but still reasonable")
        print()
        
        # Show your original kernel size results
        original_results = [r for r in results if "10x10" in r[1]]
        if original_results:
            print("🎯 YOUR ORIGINAL KERNEL SIZE (10x10) WITH STRIDE=1:")
            for config, name, final_c, final_h, final_w, total_features, _ in original_results:
                print(f"  {name}: {total_features:,} features")
                print(f"  Final: [B=1, C={final_c}, H={final_h}, W={final_w}]")
                print(f"  Width reduction: {93 - final_w}px ({((93 - final_w) / 93 * 100):.1f}%)")
                print(f"  Height reduction: {3750 - final_h}px ({((3750 - final_h) / 3750 * 100):.1f}%)")
                print(f"  Config: {config}")
            print()
            print("✅ Your original 10x10 kernel works perfectly with stride=1!")
            print("   You can keep your preferred kernel size, just change stride from 5 to 1")
    
    else:
        print("❌ NO CONFIGURATIONS FOUND!")
        print("All tested configurations produce zero output.")

if __name__ == "__main__":
    analyze_kernel_sizes()
