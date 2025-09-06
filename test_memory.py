#!/usr/bin/env python3
"""
Simple memory test script to diagnose the issue
"""

import os
import pickle
import torch
import numpy as np
import sys

def test_memory_usage():
    """Test memory usage step by step"""
    print("=== Memory Test Started ===")
    
    # Test 1: Basic imports
    print("1. Testing basic imports...")
    try:
        import torch
        import numpy as np
        print("   ✅ Basic imports successful")
    except Exception as e:
        print(f"   ❌ Import error: {e}")
        return
    
    # Test 2: CUDA availability
    print("2. Testing CUDA...")
    try:
        print(f"   CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"   GPU count: {torch.cuda.device_count()}")
            print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print("   ✅ CUDA test successful")
    except Exception as e:
        print(f"   ❌ CUDA error: {e}")
        return
    
    # Test 3: Load one pickle file
    print("3. Testing pickle file loading...")
    try:
        data_path = "/scratch/mkp6112/LFP/region_decoding/script/Allen_w2v2/Allen"
        files = [f for f in os.listdir(data_path) if f.endswith('.pickle')]
        if not files:
            print("   ❌ No pickle files found")
            return
        
        print(f"   Found {len(files)} pickle files")
        
        # Load first file
        filepath = os.path.join(data_path, files[0])
        print(f"   Loading: {files[0]}")
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        print(f"   Data type: {type(data)}")
        print(f"   Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        if isinstance(data, list) and len(data) > 0:
            print(f"   First element type: {type(data[0])}")
            if hasattr(data[0], 'shape'):
                print(f"   First element shape: {data[0].shape}")
        
        print("   ✅ Pickle loading successful")
        
    except Exception as e:
        print(f"   ❌ Pickle loading error: {e}")
        return
    
    # Test 4: Create small tensor
    print("4. Testing tensor creation...")
    try:
        # Create a very small tensor
        small_tensor = torch.randn(10, 10)
        print(f"   Small tensor shape: {small_tensor.shape}")
        print("   ✅ Small tensor creation successful")
        
        # Try to move to GPU
        if torch.cuda.is_available():
            gpu_tensor = small_tensor.cuda()
            print(f"   GPU tensor shape: {gpu_tensor.shape}")
            print("   ✅ GPU tensor creation successful")
        
    except Exception as e:
        print(f"   ❌ Tensor creation error: {e}")
        return
    
    # Test 5: Sample data processing
    print("5. Testing data sampling...")
    try:
        if isinstance(data, list) and len(data) > 1000:
            # Sample 100 elements
            step = len(data) // 100
            sampled_data = data[::step]
            print(f"   Sampled data length: {len(sampled_data)}")
            
            # Handle tuple format - extract first element from each tuple
            if isinstance(sampled_data[0], tuple):
                print(f"   First tuple length: {len(sampled_data[0])}")
                print(f"   First tuple element types: {[type(x) for x in sampled_data[0]]}")
                
                # Extract first element from each tuple
                first_elements = [item[0] for item in sampled_data if len(item) > 0]
                print(f"   Extracted first elements: {len(first_elements)}")
                
                if first_elements and hasattr(first_elements[0], 'shape'):
                    print(f"   First element shape: {first_elements[0].shape}")
                    
                    # Convert to numpy
                    np_data = np.array(first_elements)
                    print(f"   Numpy data shape: {np_data.shape}")
                    
                    # Convert to tensor
                    tensor_data = torch.FloatTensor(np_data)
                    print(f"   Tensor data shape: {tensor_data.shape}")
                    print("   ✅ Data sampling successful")
                else:
                    print("   ❌ No valid array data found in tuples")
            else:
                # Direct conversion if not tuples
                np_data = np.array(sampled_data)
                print(f"   Numpy data shape: {np_data.shape}")
                
                # Convert to tensor
                tensor_data = torch.FloatTensor(np_data)
                print(f"   Tensor data shape: {tensor_data.shape}")
                print("   ✅ Data sampling successful")
        
    except Exception as e:
        print(f"   ❌ Data sampling error: {e}")
        return
    
    # Test 6: Simple model creation
    print("6. Testing simple model creation...")
    try:
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(10, 5)
            
            def forward(self, x):
                return self.linear(x)
        
        model = SimpleModel()
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())}")
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("   ✅ Model moved to GPU")
        
        print("   ✅ Simple model creation successful")
        
    except Exception as e:
        print(f"   ❌ Model creation error: {e}")
        return
    
    print("=== All Tests Passed! ===")
    print("The issue might be with distributed training setup.")


if __name__ == "__main__":
    test_memory_usage()
