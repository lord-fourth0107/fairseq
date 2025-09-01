#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch

def visualize_cnn_reshaping(file_path, batch_size=16):
    """Visualize how to reshape pickle data for 2D CNN training"""
    print(f"🔍 Visualizing CNN reshaping strategy: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded {len(data):,} recordings")
        
        # Sample a batch of data
        sample_data = data[:batch_size]
        print(f"✓ Using batch size: {batch_size}")
        
        # Extract signals and labels
        signals = [item[0] for item in sample_data]
        labels = [item[1] for item in sample_data]
        
        # Parse labels to get channel indices
        channel_indices = []
        for label in labels:
            parts = label.split('_')
            if len(parts) == 5:
                channel_indices.append(int(parts[1]))  # count field
        
        print(f"✓ Channel indices in batch: {channel_indices}")
        
        # Create visualization
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Original Data Structure
        ax1 = plt.subplot(2, 4, 1)
        # Show original signals as individual time series
        for i, signal in enumerate(signals[:8]):  # Show first 8
            time_axis = np.arange(len(signal)) / 30000  # Assuming 30kHz
            ax1.plot(time_axis, signal + i * 0.1, alpha=0.7, linewidth=1, 
                    label=f'Ch {channel_indices[i]}' if i < 4 else '')
        
        ax1.set_xlabel('Time (seconds)')
        ax1.set_ylabel('Amplitude (offset)')
        ax1.set_title('Original Signals\n(Individual Time Series)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Current Reshaping (Wrong)
        ax2 = plt.subplot(2, 4, 2)
        # Current wrong reshaping: [batch, 1, batch_size, time]
        wrong_reshaped = torch.tensor(signals).unsqueeze(0).unsqueeze(0)
        print(f"❌ Wrong reshaping shape: {wrong_reshaped.shape}")
        
        # Show as heatmap
        im2 = ax2.imshow(wrong_reshaped[0, 0, :, :100].numpy(), cmap='viridis', aspect='auto')
        ax2.set_xlabel('Time Points')
        ax2.set_ylabel('Batch Index')
        ax2.set_title('Current (Wrong) Reshaping\n[batch, 1, batch_size, time]')
        plt.colorbar(im2, ax=ax2, label='Amplitude')
        
        # 3. Proposed Reshaping (Correct)
        ax3 = plt.subplot(2, 4, 3)
        # Proposed correct reshaping: [batch, 1, channels, time]
        # Group by channel indices and create spatial organization
        
        # Create a mapping from channel index to spatial position
        unique_channels = sorted(set(channel_indices))
        channel_to_pos = {ch: i for i, ch in enumerate(unique_channels)}
        
        # Create spatial matrix
        max_channels = len(unique_channels)
        spatial_matrix = np.zeros((max_channels, len(signals[0])))
        
        for i, (signal, ch_idx) in enumerate(zip(signals, channel_indices)):
            if ch_idx in channel_to_pos:
                pos = channel_to_pos[ch_idx]
                spatial_matrix[pos, :] = signal
        
        im3 = ax3.imshow(spatial_matrix[:, :100], cmap='viridis', aspect='auto')
        ax3.set_xlabel('Time Points')
        ax3.set_ylabel('Channel Index')
        ax3.set_title('Proposed (Correct) Reshaping\n[batch, 1, channels, time]')
        plt.colorbar(im3, ax=ax3, label='Amplitude')
        
        # 4. Channel Index Distribution
        ax4 = plt.subplot(2, 4, 4)
        ax4.hist(channel_indices, bins=20, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Channel Index')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Channel Index Distribution\nin Current Batch')
        ax4.grid(True, alpha=0.3)
        
        # 5. Spatial Organization Strategy
        ax5 = plt.subplot(2, 4, 5)
        # Show how to organize channels spatially
        channel_positions = np.array(channel_indices)
        
        # Try to find spatial pattern
        if len(unique_channels) <= 100:  # If reasonable number of channels
            # Create a grid layout
            grid_size = int(np.ceil(np.sqrt(len(unique_channels))))
            spatial_grid = np.zeros((grid_size, grid_size))
            
            for i, ch_idx in enumerate(unique_channels):
                row = i // grid_size
                col = i % grid_size
                spatial_grid[row, col] = ch_idx
            
            im5 = ax5.imshow(spatial_grid, cmap='plasma', aspect='auto')
            ax5.set_xlabel('Spatial X')
            ax5.set_ylabel('Spatial Y')
            ax5.set_title('Spatial Channel Organization\n(Grid Layout)')
            plt.colorbar(im5, ax=ax5, label='Channel Index')
        else:
            ax5.text(0.5, 0.5, f'Too many channels\n({len(unique_channels)})\nfor grid visualization', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=10)
            ax5.set_title('Spatial Channel Organization')
        
        # 6. Reshaping Comparison
        ax6 = plt.subplot(2, 4, 6)
        # Show the difference between approaches
        approaches = ['Current\n(Wrong)', 'Proposed\n(Correct)']
        shapes = [
            f'[1, 1, {batch_size}, 3750]',
            f'[1, 1, {len(unique_channels)}, 3750]'
        ]
        colors = ['#FF6B6B', '#4ECDC4']
        
        bars = ax6.bar(approaches, [batch_size, len(unique_channels)], color=colors, alpha=0.7)
        ax6.set_ylabel('Height Dimension (Channels)')
        ax6.set_title('Reshaping Comparison')
        
        # Add shape labels
        for bar, shape in zip(bars, shapes):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    shape, ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # 7. Training Strategy
        ax7 = plt.subplot(2, 4, 7)
        ax7.axis('off')
        
        strategy_text = f"""
TRAINING STRATEGY:

1. Data Organization:
   • Group recordings by channel index
   • Create spatial matrix: [channels × time]
   • Batch multiple spatial matrices

2. Reshaping for 2D CNN:
   • Input: [batch, 1, channels, time]
   • Height: Channel indices (spatial)
   • Width: Time points (temporal)

3. Model Architecture:
   • 2D CNN processes spatial-temporal patterns
   • Learns channel relationships
   • Captures spatial dependencies

4. Benefits:
   • Spatial organization preserved
   • Channel relationships learned
   • Suitable for brain region mapping
        """
        
        ax7.text(0.05, 0.95, strategy_text, transform=ax7.transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        
        # 8. Implementation Code
        ax8 = plt.subplot(2, 4, 8)
        ax8.axis('off')
        
        code_text = f"""
IMPLEMENTATION:

# Group by channel index
channel_groups = defaultdict(list)
for signal, label in data:
    ch_idx = int(label.split('_')[1])
    channel_groups[ch_idx].append(signal)

# Create spatial matrix
channels = sorted(channel_groups.keys())
spatial_matrix = np.zeros((len(channels), 3750))
for i, ch_idx in enumerate(channels):
    spatial_matrix[i] = channel_groups[ch_idx][0]

# Reshape for 2D CNN
input_tensor = torch.tensor(spatial_matrix).unsqueeze(0).unsqueeze(0)
# Shape: [1, 1, {len(unique_channels)}, 3750]
        """
        
        ax8.text(0.05, 0.95, code_text, transform=ax8.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        plt.suptitle('CNN Reshaping Strategy for Probe Data', fontsize=16, fontweight='bold', y=0.98)
        plt.show()
        
        # Print implementation recommendations
        print(f"\n🎯 IMPLEMENTATION RECOMMENDATIONS:")
        print("=" * 60)
        print(f"📊 Current batch analysis:")
        print(f"  • Batch size: {batch_size}")
        print(f"  • Unique channels: {len(unique_channels)}")
        print(f"  • Channel range: {min(unique_channels)} to {max(unique_channels)}")
        print(f"  • Signal length: {len(signals[0])} samples")
        
        print(f"\n🔧 Reshaping strategy:")
        print(f"  • Current (wrong): [1, 1, {batch_size}, 3750]")
        print(f"  • Proposed (correct): [1, 1, {len(unique_channels)}, 3750]")
        
        print(f"\n💡 Key insights:")
        print(f"  • Channel indices represent spatial positions")
        print(f"  • Each channel index is a different spatial location")
        print(f"  • Height dimension should be channel indices, not batch size")
        print(f"  • This preserves spatial relationships for 2D CNN")
        
        print(f"\n🚀 Next steps:")
        print(f"  1. Modify SessionDataset to group by channel index")
        print(f"  2. Create spatial matrices for each recording session")
        print(f"  3. Reshape as [batch, 1, channels, time]")
        print(f"  4. Train 2D CNN on spatial-temporal patterns")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plt.style.use('default')
    sns.set_palette("husl")
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    visualize_cnn_reshaping(pickle_path)
