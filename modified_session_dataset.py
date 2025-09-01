#!/usr/bin/env python3
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import defaultdict

class ModifiedSessionDataset(Dataset):
    """
    Modified SessionDataset that creates 2D matrices: [timePoints(3750) × uniqueChannels(93)]
    """
    
    def __init__(self, data_path, subset_data=1.0):
        self.data_path = data_path
        self.subset_data = subset_data
        
        # Load and organize data
        self.probe_matrices = self._load_and_organize_data()
        
    def _load_and_organize_data(self):
        """Load pickle data and organize into 2D matrices per probe"""
        print(f"🔧 Loading and organizing data from: {self.data_path}")
        
        with open(self.data_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded {len(data):,} recordings")
        
        # Organize by probe and LFP channel
        probe_data = defaultdict(lambda: defaultdict(list))
        
        for signal, label in data:
            parts = label.split('_')
            if len(parts) == 5:
                session, count, probe, lfp_channel_index, brain_region = parts
                probe_data[probe][lfp_channel_index].append(signal)
        
        print(f"✓ Found {len(probe_data)} probes")
        
        # Create 2D matrices for each probe
        probe_matrices = {}
        
        for probe_id, lfp_channels in probe_data.items():
            print(f"  🔬 Processing Probe {probe_id}: {len(lfp_channels)} LFP channels")
            
            # Get unique LFP channel IDs and sort them
            unique_lfp_channels = sorted(lfp_channels.keys())
            
            # Create 2D matrix: [timePoints × uniqueChannels]
            time_points = 3750
            num_channels = len(unique_lfp_channels)
            
            matrix_2d = np.zeros((time_points, num_channels))
            
            # Fill matrix with signals from each LFP channel
            for ch_idx, lfp_channel_id in enumerate(unique_lfp_channels):
                channel_signals = lfp_channels[lfp_channel_id]
                # Use the first signal from each LFP channel
                if channel_signals:
                    matrix_2d[:, ch_idx] = channel_signals[0]
            
            probe_matrices[probe_id] = {
                'matrix': matrix_2d,
                'unique_lfp_channels': unique_lfp_channels,
                'shape': matrix_2d.shape
            }
            
            print(f"    ✅ Matrix created: {matrix_2d.shape}")
        
        return probe_matrices
    
    def __len__(self):
        """Return number of probe matrices"""
        return len(self.probe_matrices)
    
    def __getitem__(self, idx):
        """Get a 2D matrix for a specific probe"""
        probe_ids = list(self.probe_matrices.keys())
        probe_id = probe_ids[idx]
        
        matrix_data = self.probe_matrices[probe_id]
        matrix = matrix_data['matrix']
        
        # Convert to PyTorch tensor
        tensor = torch.tensor(matrix, dtype=torch.float32)
        
        # Reshape for 2D CNN: [1, 1, time_points, channels]
        # This treats time_points as height and channels as width
        tensor_2d = tensor.unsqueeze(0).unsqueeze(0)
        
        return tensor_2d, probe_id
    
    def get_probe_info(self, probe_id):
        """Get information about a specific probe"""
        if probe_id in self.probe_matrices:
            return self.probe_matrices[probe_id]
        return None
    
    def get_all_probe_ids(self):
        """Get all probe IDs"""
        return list(self.probe_matrices.keys())

def create_data_loader(data_path, batch_size=1, subset_data=1.0):
    """Create DataLoader for the modified dataset"""
    from torch.utils.data import DataLoader
    
    dataset = ModifiedSessionDataset(data_path, subset_data)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return dataloader, dataset

# Example usage and testing
if __name__ == "__main__":
    # Test the modified dataset
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    
    print("🧪 Testing ModifiedSessionDataset...")
    
    # Create dataset
    dataset = ModifiedSessionDataset(pickle_path)
    
    print(f"\n📊 Dataset Info:")
    print(f"  • Number of probes: {len(dataset)}")
    print(f"  • Probe IDs: {dataset.get_all_probe_ids()}")
    
    # Test getting a sample
    if len(dataset) > 0:
        sample_tensor, probe_id = dataset[0]
        print(f"\n🔬 Sample from Probe {probe_id}:")
        print(f"  • Tensor shape: {sample_tensor.shape}")
        print(f"  • Data type: {sample_tensor.dtype}")
        print(f"  • Value range: [{sample_tensor.min():.6f}, {sample_tensor.max():.6f}]")
        
        # Get probe info
        probe_info = dataset.get_probe_info(probe_id)
        print(f"  • Matrix shape: {probe_info['shape']}")
        print(f"  • Unique LFP channels: {len(probe_info['unique_lfp_channels'])}")
    
    # Test DataLoader
    print(f"\n🔄 Testing DataLoader...")
    dataloader, dataset = create_data_loader(pickle_path, batch_size=1)
    
    for i, (batch_tensor, probe_ids) in enumerate(dataloader):
        print(f"  Batch {i}: {batch_tensor.shape}, Probe: {probe_ids}")
        if i >= 2:  # Test first 3 batches
            break
    
    print(f"\n✅ ModifiedSessionDataset ready for 2D CNN training!")
    print(f"   • Input shape: [batch, 1, 3750, 93]")
    print(f"   • Height: 3750 time points")
    print(f"   • Width: 93 unique channels")
    print(f"   • Ready for spatial-temporal 2D CNN processing")
