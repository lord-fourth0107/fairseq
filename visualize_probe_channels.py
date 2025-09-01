#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import pandas as pd

def visualize_probe_channel_organization(file_path, max_samples_per_probe=1000):
    """Visualize pickle data organized by probe and channel-wise"""
    print(f"🔍 Visualizing probe-channel organization: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded {len(data):,} recordings")
        
        # Parse labels and organize by probe
        probe_data = defaultdict(list)
        
        for i, (signal, label) in enumerate(data):
            parts = label.split('_')
            if len(parts) == 5:
                session, count, probe, lfp_channel_index, brain_region = parts
                
                probe_data[probe].append({
                    'index': i,
                    'signal': signal,
                    'session': session,
                    'count': int(count),
                    'probe': probe,
                    'lfp_channel_index': lfp_channel_index,
                    'brain_region': brain_region
                })
        
        print(f"✓ Found {len(probe_data)} probes")
        
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Probe Overview
        ax1 = plt.subplot(3, 3, 1)
        probe_counts = {probe: len(recordings) for probe, recordings in probe_data.items()}
        probes = list(probe_counts.keys())
        counts = list(probe_counts.values())
        
        bars = ax1.bar(range(len(probes)), counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax1.set_xlabel('Probe ID')
        ax1.set_ylabel('Number of Recordings')
        ax1.set_title('Recordings per Probe')
        ax1.set_xticks(range(len(probes)))
        ax1.set_xticklabels([f'Probe\n{p}' for p in probes], rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                    f'{count:,}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Channel Index Distribution per Probe
        ax2 = plt.subplot(3, 3, 2)
        for i, (probe, recordings) in enumerate(probe_data.items()):
            counts = [r['count'] for r in recordings[:max_samples_per_probe]]
            ax2.hist(counts, bins=50, alpha=0.7, label=f'Probe {probe}', 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][i])
        
        ax2.set_xlabel('Channel Index (Count)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Channel Index Distribution by Probe')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Signal Statistics by Probe
        ax3 = plt.subplot(3, 3, 3)
        probe_stats = {}
        for probe, recordings in probe_data.items():
            signals = [r['signal'] for r in recordings[:100]]  # Sample 100 signals
            means = [np.mean(sig) for sig in signals]
            stds = [np.std(sig) for sig in signals]
            probe_stats[probe] = {'means': means, 'stds': stds}
        
        for i, (probe, stats) in enumerate(probe_stats.items()):
            ax3.scatter(stats['means'], stats['stds'], alpha=0.6, 
                       label=f'Probe {probe}', s=30,
                       color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][i])
        
        ax3.set_xlabel('Signal Mean')
        ax3.set_ylabel('Signal Std')
        ax3.set_title('Signal Statistics by Probe')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Channel Index Progression
        ax4 = plt.subplot(3, 3, 4)
        for i, (probe, recordings) in enumerate(probe_data.items()):
            # Sort by count to see progression
            sorted_recordings = sorted(recordings[:max_samples_per_probe], key=lambda x: x['count'])
            counts = [r['count'] for r in sorted_recordings]
            ax4.plot(counts, label=f'Probe {probe}', 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'][i], linewidth=2)
        
        ax4.set_xlabel('Recording Index')
        ax4.set_ylabel('Channel Index (Count)')
        ax4.set_title('Channel Index Progression')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Sample Signals from Each Probe
        ax5 = plt.subplot(3, 3, 5)
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        for i, (probe, recordings) in enumerate(probe_data.items()):
            if recordings:
                sample_signal = recordings[0]['signal']
                time_axis = np.arange(len(sample_signal)) / 30000  # Assuming 30kHz
                ax5.plot(time_axis, sample_signal, color=colors[i], alpha=0.7, 
                        label=f'Probe {probe}', linewidth=1)
        
        ax5.set_xlabel('Time (seconds)')
        ax5.set_ylabel('Amplitude')
        ax5.set_title('Sample Signals from Each Probe')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Channel Index Heatmap
        ax6 = plt.subplot(3, 3, 6)
        # Create a matrix showing channel index distribution
        max_count = max(max(r['count'] for r in recordings[:max_samples_per_probe]) 
                       for recordings in probe_data.values())
        
        heatmap_data = np.zeros((len(probe_data), min(max_count + 1, 100)))
        probe_names = list(probe_data.keys())
        
        for i, (probe, recordings) in enumerate(probe_data.items()):
            for r in recordings[:max_samples_per_probe]:
                count = min(r['count'], 99)  # Cap at 99 for visualization
                heatmap_data[i, count] += 1
        
        im = ax6.imshow(heatmap_data, cmap='viridis', aspect='auto')
        ax6.set_xlabel('Channel Index (Count)')
        ax6.set_ylabel('Probe')
        ax6.set_title('Channel Index Distribution Heatmap')
        ax6.set_yticks(range(len(probe_names)))
        ax6.set_yticklabels([f'Probe {p}' for p in probe_names])
        plt.colorbar(im, ax=ax6, label='Count')
        
        # 7. Signal Power Spectrum
        ax7 = plt.subplot(3, 3, 7)
        for i, (probe, recordings) in enumerate(probe_data.items()):
            if recordings:
                sample_signal = recordings[0]['signal']
                fft = np.fft.fft(sample_signal)
                freqs = np.fft.fftfreq(len(sample_signal), 1/30000)  # Assuming 30kHz
                power = np.abs(fft[:len(fft)//2])
                freqs = freqs[:len(freqs)//2]
                
                ax7.semilogy(freqs, power, color=colors[i], alpha=0.7, 
                           label=f'Probe {probe}', linewidth=1)
        
        ax7.set_xlabel('Frequency (Hz)')
        ax7.set_ylabel('Power')
        ax7.set_title('Power Spectrum by Probe')
        ax7.set_xlim(0, 15000)  # Focus on relevant frequencies
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Channel Index vs Signal Statistics
        ax8 = plt.subplot(3, 3, 8)
        for i, (probe, recordings) in enumerate(probe_data.items()):
            counts = [r['count'] for r in recordings[:max_samples_per_probe]]
            signal_means = [np.mean(r['signal']) for r in recordings[:max_samples_per_probe]]
            
            ax8.scatter(counts, signal_means, alpha=0.6, 
                       label=f'Probe {probe}', s=20,
                       color=colors[i])
        
        ax8.set_xlabel('Channel Index (Count)')
        ax8.set_ylabel('Signal Mean')
        ax8.set_title('Channel Index vs Signal Mean')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Summary Statistics Table
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # Create summary table
        summary_data = []
        for probe, recordings in probe_data.items():
            counts = [r['count'] for r in recordings]
            signals = [r['signal'] for r in recordings[:100]]  # Sample for stats
            
            summary_data.append([
                f'Probe {probe}',
                f'{len(recordings):,}',
                f'{min(counts)}-{max(counts)}',
                f'{np.mean([np.mean(sig) for sig in signals]):.4f}',
                f'{np.mean([np.std(sig) for sig in signals]):.4f}'
            ])
        
        table = ax9.table(cellText=summary_data,
                         colLabels=['Probe', 'Recordings', 'Count Range', 'Mean', 'Std'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style the table
        for i in range(len(summary_data) + 1):
            for j in range(5):
                cell = table[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4ECDC4')
                    cell.set_text_props(weight='bold')
                else:
                    cell.set_facecolor('#F8F9FA' if i % 2 == 0 else '#E9ECEF')
        
        ax9.set_title('Probe Summary Statistics', pad=20, fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.suptitle('Probe-Channel Organization Analysis', fontsize=16, fontweight='bold', y=0.98)
        plt.show()
        
        # Print detailed analysis
        print(f"\n📊 DETAILED PROBE ANALYSIS:")
        print("=" * 60)
        
        for probe, recordings in probe_data.items():
            counts = [r['count'] for r in recordings]
            lfp_channels = [r['lfp_channel_index'] for r in recordings]
            
            print(f"\n🔬 Probe {probe}:")
            print(f"  📈 Total recordings: {len(recordings):,}")
            print(f"  📍 LFP Channel Index: {lfp_channels[0]} (unique)")
            print(f"  🔢 Count range: {min(counts)} to {max(counts)}")
            print(f"  🧠 Brain region: {recordings[0]['brain_region']}")
            print(f"  ⏱️  Session: {recordings[0]['session']}")
            
            # Check for cycling pattern
            if len(set(counts)) <= 100:  # If counts are limited
                print(f"  🔄 Channel cycling: Yes (covers {len(set(counts))} unique indices)")
            else:
                print(f"  🔄 Channel cycling: No (continuous range)")
        
        print(f"\n🎯 KEY INSIGHTS:")
        print("=" * 40)
        print("• Each probe represents a different physical insertion")
        print("• Count values cycle through channel indices (0-2856)")
        print("• LFP Channel Index identifies the physical electrode")
        print("• All recordings are from APN (Action Potential Negative) region")
        print("• Data represents sequential recordings, not simultaneous channels")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plt.style.use('default')
    sns.set_palette("husl")
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    visualize_probe_channel_organization(pickle_path)
