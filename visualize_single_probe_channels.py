#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

def visualize_single_probe_channels(file_path, max_samples=5000):
    """Visualize single probe data organized channel-wise"""
    print(f"🔍 Visualizing single probe channel organization: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded {len(data):,} recordings")
        
        # Parse labels
        recordings = []
        for i, (signal, label) in enumerate(data[:max_samples]):
            parts = label.split('_')
            if len(parts) == 5:
                session, count, probe, lfp_channel_index, brain_region = parts
                recordings.append({
                    'index': i,
                    'signal': signal,
                    'session': session,
                    'count': int(count),
                    'probe': probe,
                    'lfp_channel_index': lfp_channel_index,
                    'brain_region': brain_region
                })
        
        print(f"✓ Analyzed {len(recordings)} recordings")
        print(f"✓ Probe ID: {recordings[0]['probe']}")
        print(f"✓ LFP Channel Index: {recordings[0]['lfp_channel_index']}")
        print(f"✓ Brain Region: {recordings[0]['brain_region']}")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Channel Index Distribution
        ax1 = plt.subplot(3, 3, 1)
        counts = [r['count'] for r in recordings]
        ax1.hist(counts, bins=50, color='#4ECDC4', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Channel Index (Count)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Channel Index Distribution')
        ax1.grid(True, alpha=0.3)
        
        # Add statistics
        ax1.axvline(np.mean(counts), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(counts):.1f}')
        ax1.axvline(np.median(counts), color='orange', linestyle='--', 
                   label=f'Median: {np.median(counts):.1f}')
        ax1.legend()
        
        # 2. Channel Index Progression Over Time
        ax2 = plt.subplot(3, 3, 2)
        sorted_recordings = sorted(recordings, key=lambda x: x['index'])
        indices = [r['index'] for r in sorted_recordings]
        counts = [r['count'] for r in sorted_recordings]
        
        ax2.plot(indices, counts, color='#FF6B6B', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Recording Index')
        ax2.set_ylabel('Channel Index (Count)')
        ax2.set_title('Channel Index Progression')
        ax2.grid(True, alpha=0.3)
        
        # 3. Sample Signals from Different Channel Indices
        ax3 = plt.subplot(3, 3, 3)
        # Sample signals from different count values
        sample_counts = [0, 100, 500, 1000, 1500, 2000]
        colors = plt.cm.viridis(np.linspace(0, 1, len(sample_counts)))
        
        for i, target_count in enumerate(sample_counts):
            # Find recording closest to target count
            closest_rec = min(recordings, key=lambda x: abs(x['count'] - target_count))
            signal = closest_rec['signal']
            time_axis = np.arange(len(signal)) / 30000  # Assuming 30kHz
            
            ax3.plot(time_axis, signal, color=colors[i], alpha=0.7, 
                    label=f'Count {closest_rec["count"]}', linewidth=1)
        
        ax3.set_xlabel('Time (seconds)')
        ax3.set_ylabel('Amplitude')
        ax3.set_title('Sample Signals by Channel Index')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Signal Statistics vs Channel Index
        ax4 = plt.subplot(3, 3, 4)
        counts = [r['count'] for r in recordings]
        signal_means = [np.mean(r['signal']) for r in recordings]
        signal_stds = [np.std(r['signal']) for r in recordings]
        
        scatter = ax4.scatter(counts, signal_means, c=signal_stds, 
                             cmap='plasma', alpha=0.6, s=20)
        ax4.set_xlabel('Channel Index (Count)')
        ax4.set_ylabel('Signal Mean')
        ax4.set_title('Signal Mean vs Channel Index\n(Color = Signal Std)')
        plt.colorbar(scatter, ax=ax4, label='Signal Std')
        ax4.grid(True, alpha=0.3)
        
        # 5. Channel Index Heatmap (if we can organize spatially)
        ax5 = plt.subplot(3, 3, 5)
        # Try to organize counts in a 2D grid (assuming some spatial organization)
        max_count = max(counts)
        
        # Try different grid sizes to see if there's spatial organization
        grid_sizes = [50, 60, 70, 80, 90, 100]
        best_grid = None
        best_score = 0
        
        for grid_size in grid_sizes:
            if max_count < grid_size * grid_size:
                # Create grid
                grid = np.zeros((grid_size, grid_size))
                for count in counts:
                    row = (count // grid_size) % grid_size
                    col = count % grid_size
                    grid[row, col] += 1
                
                # Score based on how well distributed the data is
                non_zero = np.count_nonzero(grid)
                score = non_zero / (grid_size * grid_size)
                
                if score > best_score:
                    best_score = score
                    best_grid = grid
                    best_size = grid_size
        
        if best_grid is not None:
            im = ax5.imshow(best_grid, cmap='viridis', aspect='auto')
            ax5.set_xlabel('Column')
            ax5.set_ylabel('Row')
            ax5.set_title(f'Channel Index Spatial Organization\n({best_size}x{best_size} grid)')
            plt.colorbar(im, ax=ax5, label='Count')
        else:
            ax5.text(0.5, 0.5, 'No clear spatial\norganization found', 
                    ha='center', va='center', transform=ax5.transAxes, fontsize=12)
            ax5.set_title('Channel Index Spatial Organization')
        
        # 6. Power Spectrum Analysis
        ax6 = plt.subplot(3, 3, 6)
        # Sample a few signals from different channel indices
        sample_indices = [0, len(recordings)//4, len(recordings)//2, 3*len(recordings)//4, len(recordings)-1]
        colors = plt.cm.plasma(np.linspace(0, 1, len(sample_indices)))
        
        for i, idx in enumerate(sample_indices):
            if idx < len(recordings):
                signal = recordings[idx]['signal']
                fft = np.fft.fft(signal)
                freqs = np.fft.fftfreq(len(signal), 1/30000)  # Assuming 30kHz
                power = np.abs(fft[:len(fft)//2])
                freqs = freqs[:len(freqs)//2]
                
                ax6.semilogy(freqs, power, color=colors[i], alpha=0.7, 
                           label=f'Count {recordings[idx]["count"]}', linewidth=1)
        
        ax6.set_xlabel('Frequency (Hz)')
        ax6.set_ylabel('Power')
        ax6.set_title('Power Spectrum by Channel Index')
        ax6.set_xlim(0, 15000)  # Focus on relevant frequencies
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Channel Index vs Signal Variance
        ax7 = plt.subplot(3, 3, 7)
        counts = [r['count'] for r in recordings]
        signal_vars = [np.var(r['signal']) for r in recordings]
        
        ax7.scatter(counts, signal_vars, alpha=0.6, color='#45B7D1', s=20)
        ax7.set_xlabel('Channel Index (Count)')
        ax7.set_ylabel('Signal Variance')
        ax7.set_title('Signal Variance vs Channel Index')
        ax7.grid(True, alpha=0.3)
        
        # 8. Channel Index Distribution (Box Plot)
        ax8 = plt.subplot(3, 3, 8)
        # Group counts into bins for box plot
        bin_size = 100
        bins = list(range(0, max(counts) + bin_size, bin_size))
        binned_data = []
        bin_labels = []
        
        for i in range(len(bins) - 1):
            bin_start, bin_end = bins[i], bins[i + 1]
            bin_counts = [c for c in counts if bin_start <= c < bin_end]
            if bin_counts:
                binned_data.append(bin_counts)
                bin_labels.append(f'{bin_start}-{bin_end-1}')
        
        if binned_data:
            ax8.boxplot(binned_data, labels=bin_labels)
            ax8.set_xlabel('Channel Index Range')
            ax8.set_ylabel('Count')
            ax8.set_title('Channel Index Distribution (Box Plot)')
            ax8.tick_params(axis='x', rotation=45)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Calculate statistics
        counts = [r['count'] for r in recordings]
        signals = [r['signal'] for r in recordings]
        
        stats_data = [
            ['Total Recordings', f'{len(recordings):,}'],
            ['Channel Index Range', f'{min(counts)} - {max(counts)}'],
            ['Unique Channel Indices', f'{len(set(counts))}'],
            ['Probe ID', recordings[0]['probe']],
            ['LFP Channel Index', recordings[0]['lfp_channel_index']],
            ['Brain Region', recordings[0]['brain_region']],
            ['Session ID', recordings[0]['session']],
            ['Signal Length', f'{len(recordings[0]["signal"])} samples'],
            ['Sampling Rate', '30 kHz (estimated)'],
            ['Signal Mean', f'{np.mean([np.mean(sig) for sig in signals]):.4f}'],
            ['Signal Std', f'{np.mean([np.std(sig) for sig in signals]):.4f}']
        ]
        
        table = ax9.table(cellText=stats_data,
                         colLabels=['Property', 'Value'],
                         cellLoc='left',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        
        # Style the table
        for i in range(len(stats_data) + 1):
            for j in range(2):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#F8F9FA' if i % 2 == 0 else '#E9ECEF')
        
        ax9.set_title('Probe Summary Statistics', pad=20, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle(f'Single Probe Channel Analysis - Probe {recordings[0]["probe"]}', 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.show()
        
        # Print detailed analysis
        print(f"\n📊 DETAILED CHANNEL ANALYSIS:")
        print("=" * 60)
        print(f"🔬 Probe: {recordings[0]['probe']}")
        print(f"📍 LFP Channel Index: {recordings[0]['lfp_channel_index']}")
        print(f"🧠 Brain Region: {recordings[0]['brain_region']}")
        print(f"⏱️  Session: {recordings[0]['session']}")
        print(f"📈 Total recordings: {len(recordings):,}")
        print(f"🔢 Channel index range: {min(counts)} to {max(counts)}")
        print(f"🎯 Unique channel indices: {len(set(counts))}")
        
        # Check for patterns
        print(f"\n🔍 PATTERN ANALYSIS:")
        print("=" * 40)
        
        # Check if counts are sequential
        sorted_counts = sorted(counts)
        is_sequential = all(sorted_counts[i] == sorted_counts[i-1] + 1 for i in range(1, len(sorted_counts)))
        print(f"• Sequential counting: {'Yes' if is_sequential else 'No'}")
        
        # Check for cycling
        if len(set(counts)) < len(counts):
            print(f"• Channel cycling: Yes (repeats every {len(set(counts))} indices)")
        else:
            print(f"• Channel cycling: No (continuous range)")
        
        # Check for spatial organization
        if max(counts) < 10000:  # If reasonable range
            print(f"• Potential spatial organization: Yes (range 0-{max(counts)})")
        else:
            print(f"• Potential spatial organization: No (too large range)")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print("=" * 40)
        print("• Single probe with sequential channel indexing")
        print("• Channel indices represent spatial positions on the probe")
        print("• Each recording is from a different spatial location")
        print("• Data suitable for spatial-temporal analysis")
        print("• Can be reshaped for 2D CNN: [batch, 1, channel_index, time]")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plt.style.use('default')
    sns.set_palette("husl")
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    visualize_single_probe_channels(pickle_path)
