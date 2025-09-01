#!/usr/bin/env python3
"""
Simple test script to verify the environment and basic functionality
before running the full training.
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

def test_environment():
    """Test basic environment setup"""
    print("🔍 Testing Environment Setup")
    print("=" * 50)
    
    # Test Python version
    print(f"Python version: {sys.version}")
    
    # Test PyTorch
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Test imports
    try:
        from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel
        print("✅ Fairseq imports successful")
    except Exception as e:
        print(f"❌ Fairseq import failed: {e}")
        return False
    
    # Test data path
    data_path = "/scratch/us2193/neural_probe_data"
    if os.path.exists(data_path):
        print(f"✅ Data path exists: {data_path}")
        # Count pickle files
        pickle_files = list(Path(data_path).glob("*.pkl"))
        print(f"   Found {len(pickle_files)} pickle files")
    else:
        print(f"❌ Data path not found: {data_path}")
        return False
    
    return True

def test_model_creation():
    """Test basic model creation"""
    print("\n🏗️ Testing Model Creation")
    print("=" * 50)
    
    try:
        from fairseq.models.wav2vec.wav2vec2_2d import Wav2Vec2_2DConfig, Wav2Vec2_2DModel
        
        # Create a simple config
        config = Wav2Vec2_2DConfig(
            input_height=3750,
            input_width=93,
            encoder_embed_dim=768,
            encoder_layers=12,
            encoder_attention_heads=12,
            feature_grad_mult=0.0,
            encoder_layerdrop=0.0,
            dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            final_dim=256,
            layer_norm_first=False,
            conv_feature_layers="[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]",
            conv_bias=False,
            logit_temp=0.1,
            target_glu=False,
            feature_extractor_activation="gelu",
            model_parallel_size=1,
            quantize_targets=False,
            quantize_input=False,
            same_quantizer=False,
            target_quantizer_blocks=0,
            codebook_negatives=0,
            num_negatives=100,
            cross_sample_negatives=0,
            sample_distance=-1,
            logit_temp=0.1,
            target_glu=False,
            feature_extractor_activation="gelu",
            model_parallel_size=1,
            quantize_targets=False,
            quantize_input=False,
            same_quantizer=False,
            target_quantizer_blocks=0,
            codebook_negatives=0,
            num_negatives=100,
            cross_sample_negatives=0,
            sample_distance=-1,
        )
        
        print("✅ Config created successfully")
        
        # Create model
        model = Wav2Vec2_2DModel(config)
        print("✅ Model created successfully")
        
        # Test forward pass with dummy data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Create dummy input
        dummy_input = torch.randn(1, 1, 3750, 93).to(device)
        print(f"Dummy input shape: {dummy_input.shape}")
        
        # Test feature extractor
        with torch.no_grad():
            features = model.feature_extractor(dummy_input)
            print(f"✅ Feature extractor output: {features.shape}")
        
        print("✅ Model test successful")
        return True
        
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_data_loading():
    """Test basic data loading"""
    print("\n📊 Testing Data Loading")
    print("=" * 50)
    
    try:
        import pickle
        from pathlib import Path
        
        data_path = Path("/scratch/us2193/neural_probe_data")
        pickle_files = list(data_path.glob("*.pkl"))
        
        if not pickle_files:
            print("❌ No pickle files found")
            return False
        
        # Test loading first file
        test_file = pickle_files[0]
        print(f"Testing file: {test_file.name}")
        
        with open(test_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✅ File loaded successfully")
        print(f"   Data type: {type(data)}")
        
        if isinstance(data, dict):
            print(f"   Keys: {list(data.keys())}")
            if 'probe_dump' in data:
                print(f"   Probe dump length: {len(data['probe_dump'])}")
                if len(data['probe_dump']) > 0:
                    print(f"   First item type: {type(data['probe_dump'][0])}")
                    if isinstance(data['probe_dump'][0], tuple) and len(data['probe_dump'][0]) >= 2:
                        print(f"   First item shape: {data['probe_dump'][0][0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🧪 Wav2Vec2 2D Setup Test")
    print("=" * 60)
    
    tests = [
        ("Environment", test_environment),
        ("Model Creation", test_model_creation),
        ("Data Loading", test_data_loading),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    print("\n📋 Test Results Summary")
    print("=" * 60)
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
    
    all_passed = all(result for _, result in results)
    if all_passed:
        print("\n🎉 All tests passed! Ready to run training.")
    else:
        print("\n⚠️ Some tests failed. Please fix issues before training.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
