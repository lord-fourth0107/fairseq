#!/usr/bin/env python3
"""
Test script to check what shapes ModifiedSessionDataset returns
"""

import sys
import os
sys.path.append(os.path.abspath('.'))

from modified_session_dataset import ModifiedSessionDataset
from torch.utils.data import DataLoader

def test_dataset_shapes():
    """Test what shapes the ModifiedSessionDataset returns"""
    print("🧪 Testing ModifiedSessionDataset shapes...")
    
    # Test with a sample pickle file
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    
    try:
        # Create dataset
        dataset = ModifiedSessionDataset(pickle_path, subset_data=0.01)  # Use small subset
        
        print(f"✅ Dataset created successfully")
        print(f"📊 Dataset length: {len(dataset)}")
        
        # Test getting a single sample
        if len(dataset) > 0:
            sample_tensor, probe_id = dataset[0]
            print(f"🔬 Sample tensor shape: {sample_tensor.shape}")
            print(f"🔬 Sample tensor dtype: {sample_tensor.dtype}")
            print(f"🔬 Probe ID: {probe_id}")
            print(f"🔬 Tensor dimensions: {len(sample_tensor.shape)}")
            
            # Test DataLoader
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
            
            for i, (batch_tensor, probe_ids) in enumerate(dataloader):
                print(f"📦 Batch {i}:")
                print(f"   Tensor shape: {batch_tensor.shape}")
                print(f"   Probe IDs: {probe_ids}")
                print(f"   Tensor dtype: {batch_tensor.dtype}")
                print(f"   Tensor range: [{batch_tensor.min():.6f}, {batch_tensor.max():.6f}]")
                
                if i >= 2:  # Test first 3 batches
                    break
        
        print(f"✅ Test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error testing dataset: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_shapes()
