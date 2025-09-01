#!/usr/bin/env python3
"""
Analyze channel variation across different sessions and understand the dimension mismatch
"""

import os
import pickle
import glob
from collections import defaultdict

def analyze_channel_variation():
    """Analyze why we're getting different channel counts"""
    
    print("🔍 ANALYZING CHANNEL VARIATION ISSUE")
    print("=" * 50)
    
    # The error showed: [1, 699, 512] vs [357888, 768]
    # This suggests the model was configured for different dimensions
    
    print("📊 ERROR ANALYSIS:")
    print("   • Error: mat1 and mat2 shapes cannot be multiplied (699x512 and 357888x768)")
    print("   • This means:")
    print("     - Actual features: [699, 512] (sequence_length=699, feature_dim=512)")
    print("     - Model weights: [357888, 768] (wrong input_size=357888, output_size=768)")
    print("     - Expected: [512, 768] (correct input_size=512, output_size=768)")
    
    print("\n🔍 ROOT CAUSE ANALYSIS:")
    print("   • The model's post_extract_proj was initialized with wrong dimensions")
    print("   • 357888 = embed * output_height * output_width (from old config)")
    print("   • But actual data produces features with 512 dimensions")
    
    print("\n📈 DIMENSION FLOW:")
    print("   1. Input: [1, 1, 3750, 77] (batch, channels, height, width)")
    print("   2. 2D CNN: [1, 512, H', W'] (feature extraction)")
    print("   3. Reshape: [1, H'*W', 512] (for transformer)")
    print("   4. Layer Norm: [1, 699, 512] (normalize over 512)")
    print("   5. Post-Extract Proj: [1, 699, 768] (512 -> 768)")
    
    print("\n🎯 WHY DIFFERENT SESSIONS MIGHT HAVE DIFFERENT CHANNELS:")
    print("   • Different recording sessions may use different probe configurations")
    print("   • Some sessions might have 77 channels, others 93 channels")
    print("   • The model needs to adapt to each session's specific channel count")
    
    print("\n💡 SOLUTIONS:")
    print("   1. ✅ Dynamic model configuration (already implemented)")
    print("   2. ✅ Layer recreation with correct dimensions (already implemented)")
    print("   3. 🔄 Session-specific model adaptation")
    print("   4. 🔄 Padding/truncation to standardize channel counts")
    
    return True

def check_model_configuration_issue():
    """Check what's happening with model configuration"""
    
    print("\n🔧 MODEL CONFIGURATION ISSUE:")
    print("=" * 40)
    
    print("📊 The Problem:")
    print("   • Model created with hardcoded config: input_height=3750, input_width=93")
    print("   • But actual data might have: input_height=3750, input_width=77")
    print("   • This causes dimension mismatches in all layers")
    
    print("\n🔍 Layer-by-Layer Analysis:")
    print("   1. Feature Extractor:")
    print("      - Expected: [1, 1, 3750, 93] -> [1, 512, H1, W1]")
    print("      - Actual:   [1, 1, 3750, 77] -> [1, 512, H2, W2]")
    print("      - H1 ≠ H2, W1 ≠ W2")
    
    print("   2. Reshape Operation:")
    print("      - Expected: [1, H1*W1, 512]")
    print("      - Actual:   [1, H2*W2, 512]")
    print("      - H1*W1 ≠ H2*W2")
    
    print("   3. Layer Norm:")
    print("      - Expected: normalize over 512 (correct)")
    print("      - Actual:   normalize over 512 (correct)")
    print("      - ✅ This should work")
    
    print("   4. Post-Extract Proj:")
    print("      - Expected: [H1*W1*512, 768] (wrong!)")
    print("      - Actual:   [512, 768] (correct)")
    print("      - ❌ This is the source of the error")
    
    print("\n✅ SOLUTION IMPLEMENTED:")
    print("   • Recreate post_extract_proj with correct dimensions: [512, 768]")
    print("   • This should fix the matrix multiplication error")

def suggest_improvements():
    """Suggest improvements for handling variable channel counts"""
    
    print("\n🚀 IMPROVEMENTS FOR VARIABLE CHANNEL COUNTS:")
    print("=" * 50)
    
    print("1. 📊 Session-Aware Model Creation:")
    print("   • Create model after loading each session's data")
    print("   • Update all layer dimensions based on actual data")
    print("   • Save session-specific model configurations")
    
    print("\n2. 🔄 Dynamic Layer Recreation:")
    print("   • Recreate all affected layers (not just layer_norm and post_extract_proj)")
    print("   • Include: feature extractor, spatial embeddings, etc.")
    print("   • Ensure all dimensions are consistent")
    
    print("\n3. 📏 Standardization Options:")
    print("   • Option A: Pad shorter sessions to match longest session")
    print("   • Option B: Truncate longer sessions to match shortest session")
    print("   • Option C: Use session-specific models (current approach)")
    
    print("\n4. 🎯 Recommended Approach:")
    print("   • Continue with dynamic model recreation (current solution)")
    print("   • Add more comprehensive layer recreation")
    print("   • Test with different session channel counts")
    
    return True

if __name__ == "__main__":
    analyze_channel_variation()
    check_model_configuration_issue()
    suggest_improvements()
    
    print("\n" + "="*60)
    print("🎯 CONCLUSION:")
    print("   Yes, different sessions likely have different channel counts!")
    print("   This is why we're getting dimension mismatches.")
    print("   The solution is to recreate model layers with correct dimensions.")
    print("   Your current fix should work, but may need extension to other layers.")
    print("="*60)
