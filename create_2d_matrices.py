#!/usr/bin/env python3
import pickle
import numpy as np
import torch
from collections import defaultdict
import matplotlib.pyplot as plt

def create_2d_matrices_from_pickle(file_path, save_matrices=True, visualize=True):
    """Convert pickle data to 2D matrices: [timePoints(3750) × uniqueChannels(93)]"""
    print(f"🔧 Creating 2D matrices from pickle: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded {len(data):,} recordings")
        
        # Parse labels and organize by probe
        probe_data = defaultdict(lambda: defaultdict(list))
        
        for i, (signal, label) in enumerate(data):
            parts = label.split('_')
            if len(parts) == 5:
                session, count, probe, lfp_channel_index, brain_region = parts
                
                probe_data[probe][lfp_channel_index].append({
                    'signal': signal,
                    'count': int(count),
                    'session': session,
                    'brain_region': brain_region
                })
        
        print(f"✓ Found {len(probe_data)} probes")
        
        # Process each probe
        probe_matrices = {}
        
        for probe_id, lfp_channels in probe_data.items():
            print(f"\n🔬 Processing Probe {probe_id}:")
            print(f"  📍 LFP channels: {len(lfp_channels)}")
            
            # Get unique LFP channel IDs and sort them
            unique_lfp_channels = sorted(lfp_channels.keys())
            print(f"  🎯 Unique LFP channels: {len(unique_lfp_channels)}")
            
            # Check how many recordings per LFP channel
            recordings_per_channel = [len(lfp_channels[ch]) for ch in unique_lfp_channels]
            print(f"  📊 Recordings per channel: {min(recordings_per_channel)} - {max(recordings_per_channel)}")
            
            # Create 2D matrix: [timePoints × uniqueChannels]
            time_points = 3750  # Signal length
            num_channels = len(unique_lfp_channels)
            
            # Initialize matrix
            matrix_2d = np.zeros((time_points, num_channels))
            channel_info = {}
            
            print(f"  🔧 Creating matrix: [{time_points} × {num_channels}]")
            
            # Fill matrix with signals from each LFP channel
            for ch_idx, lfp_channel_id in enumerate(unique_lfp_channels):
                channel_recordings = lfp_channels[lfp_channel_id]
                
                # Use the first recording from each LFP channel
                # (or you could average multiple recordings if available)
                if channel_recordings:
                    signal = channel_recordings[0]['signal']
                    matrix_2d[:, ch_idx] = signal
                    
                    # Store channel info
                    channel_info[lfp_channel_id] = {
                        'index': ch_idx,
                        'count': channel_recordings[0]['count'],
                        'session': channel_recordings[0]['session'],
                        'brain_region': channel_recordings[0]['brain_region'],
                        'num_recordings': len(channel_recordings)
                    }
            
            # Store the matrix and info
            probe_matrices[probe_id] = {
                'matrix': matrix_2d,
                'channel_info': channel_info,
                'unique_lfp_channels': unique_lfp_channels,
                'shape': matrix_2d.shape,
                'time_points': time_points,
                'num_channels': num_channels
            }
            
            print(f"  ✅ Matrix created: {matrix_2d.shape}")
            print(f"  📊 Signal stats - Mean: {np.mean(matrix_2d):.6f}, Std: {np.std(matrix_2d):.6f}")
            print(f"  📊 Signal range: [{np.min(matrix_2d):.6f}, {np.max(matrix_2d):.6f}]")
        
        # Save matrices if requested
        if save_matrices:
            print(f"\n💾 Saving matrices...")
            for probe_id, probe_data in probe_matrices.items():
                filename = f"probe_{probe_id}_2d_matrix.npy"
                np.save(filename, probe_data['matrix'])
                print(f"  ✅ Saved: {filename}")
                
                # Save channel info
                info_filename = f"probe_{probe_id}_channel_info.npy"
                np.save(info_filename, probe_data['channel_info'], allow_pickle=True)
                print(f"  ✅ Saved: {info_filename}")
        
        # Visualize if requested
        if visualize:
            visualize_2d_matrices(probe_matrices)
        
        return probe_matrices
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def visualize_2d_matrices(probe_matrices):
    """Visualize the 2D matrices"""
    print(f"\n📊 Creating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('2D Matrix Analysis: [TimePoints × UniqueChannels]', fontsize=16, fontweight='bold')
    
    for i, (probe_id, probe_data) in enumerate(probe_matrices.items()):
        matrix = probe_data['matrix']
        
        # 1. Full matrix heatmap
        ax1 = axes[0, 0] if i == 0 else axes[0, 1]
        im1 = ax1.imshow(matrix, cmap='viridis', aspect='auto')
        ax1.set_xlabel('Channel Index')
        ax1.set_ylabel('Time Points')
        ax1.set_title(f'Probe {probe_id} - Full Matrix\n{matrix.shape}')
        plt.colorbar(im1, ax=ax1, label='Amplitude')
        
        # 2. Sample time series from different channels
        ax2 = axes[1, 0] if i == 0 else axes[1, 1]
        num_channels = matrix.shape[1]
        sample_channels = [0, num_channels//4, num_channels//2, 3*num_channels//4, num_channels-1]
        colors = plt.cm.plasma(np.linspace(0, 1, len(sample_channels)))
        
        for j, ch_idx in enumerate(sample_channels):
            time_axis = np.arange(matrix.shape[0]) / 30000  # Assuming 30kHz
            ax2.plot(time_axis, matrix[:, ch_idx], color=colors[j], alpha=0.7, 
                    label=f'Channel {ch_idx}', linewidth=1)
        
        ax2.set_xlabel('Time (seconds)')
        ax2.set_ylabel('Amplitude')
        ax2.set_title(f'Probe {probe_id} - Sample Signals')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n📋 MATRIX SUMMARY:")
    print("=" * 60)
    for probe_id, probe_data in probe_matrices.items():
        matrix = probe_data['matrix']
        print(f"\n🔬 Probe {probe_id}:")
        print(f"  📐 Matrix shape: {matrix.shape}")
        print(f"  ⏱️  Time points: {probe_data['time_points']}")
        print(f"  📍 Channels: {probe_data['num_channels']}")
        print(f"  📊 Signal stats: Mean={np.mean(matrix):.6f}, Std={np.std(matrix):.6f}")
        print(f"  📊 Range: [{np.min(matrix):.6f}, {np.max(matrix):.6f}]")

def create_torch_tensors(probe_matrices):
    """Convert matrices to PyTorch tensors for 2D CNN"""
    print(f"\n🔥 Converting to PyTorch tensors...")
    
    torch_tensors = {}
    
    for probe_id, probe_data in probe_matrices.items():
        matrix = probe_data['matrix']
        
        # Convert to PyTorch tensor
        tensor = torch.tensor(matrix, dtype=torch.float32)
        
        # Reshape for 2D CNN: [batch, channels, height, width]
        # Height = time points, Width = channels
        tensor_2d_cnn = tensor.unsqueeze(0).unsqueeze(0)  # [1, 1, time_points, channels]
        
        torch_tensors[probe_id] = {
            'tensor': tensor,
            'tensor_2d_cnn': tensor_2d_cnn,
            'shape': tensor.shape,
            'shape_2d_cnn': tensor_2d_cnn.shape
        }
        
        print(f"  ✅ Probe {probe_id}:")
        print(f"    📐 Original: {tensor.shape}")
        print(f"    🔥 2D CNN: {tensor_2d_cnn.shape}")
    
    return torch_tensors

if __name__ == "__main__":
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    
    # Create 2D matrices
    probe_matrices = create_2d_matrices_from_pickle(pickle_path, save_matrices=True, visualize=True)
    
    if probe_matrices:
        # Convert to PyTorch tensors
        torch_tensors = create_torch_tensors(probe_matrices)
        
        print(f"\n🎯 READY FOR 2D CNN TRAINING!")
        print("=" * 50)
        print("✅ 2D matrices created: [3750 × 93]")
        print("✅ PyTorch tensors ready: [1, 1, 3750, 93]")
        print("✅ Data organized by probe")
        print("✅ Channel information preserved")
        print("\n🚀 Next steps:")
        print("  1. Use these matrices for SSL training")
        print("  2. Reshape as needed for your 2D CNN")
        print("  3. Train on spatial-temporal patterns")
