#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import pandas as pd
from datetime import datetime
import os

class PickleAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.signals = None
        self.labels = None
        self.label_parts = None
        
    def load_data(self):
        """Load the pickle file and extract basic information"""
        print(f"🔍 Loading pickle file: {self.file_path}")
        
        try:
            with open(self.file_path, 'rb') as f:
                self.data = pickle.load(f)
            
            print(f"✓ Successfully loaded {len(self.data):,} recordings")
            
            # Extract signals and labels
            self.signals = np.array([item[0] for item in self.data])
            self.labels = [item[1] for item in self.data]
            
            # Parse label parts
            self.label_parts = []
            for label in self.labels:
                parts = label.split('_')
                if len(parts) == 5:
                    self.label_parts.append({
                        'session_id': parts[0],
                        'channel_index': int(parts[1]),
                        'recording_session': parts[2],
                        'channel_id': parts[3],
                        'recording_type': parts[4]
                    })
            
            print(f"✓ Extracted {len(self.signals)} signals with shape {self.signals.shape}")
            print(f"✓ Parsed {len(self.label_parts)} labels")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
        
        return True
    
    def basic_statistics(self):
        """Display basic statistics about the data"""
        print("\n📊 BASIC STATISTICS:")
        print("=" * 60)
        
        if self.signals is None:
            print("❌ No data loaded")
            return
        
        print(f"Total recordings: {len(self.signals):,}")
        print(f"Signal length: {self.signals.shape[1]} time points")
        print(f"Data type: {self.signals.dtype}")
        print(f"Memory usage: {self.signals.nbytes / (1024**3):.2f} GB")
        
        # Signal statistics
        print(f"\nSignal Statistics:")
        print(f"  Min value: {np.min(self.signals):.8f}")
        print(f"  Max value: {np.max(self.signals):.8f}")
        print(f"  Mean value: {np.mean(self.signals):.8f}")
        print(f"  Std value: {np.std(self.signals):.8f}")
        
        # Label analysis
        if self.label_parts:
            unique_sessions = set(p['session_id'] for p in self.label_parts)
            unique_channels = set(p['channel_id'] for p in self.label_parts)
            unique_types = set(p['recording_type'] for p in self.label_parts)
            
            print(f"\nLabel Analysis:")
            print(f"  Unique session IDs: {len(unique_sessions)}")
            print(f"  Unique channel IDs: {len(unique_channels)}")
            print(f"  Recording types: {list(unique_types)}")
    
    def visualize_signal_samples(self, num_samples=12):
        """Visualize sample signals from different channels"""
        if self.signals is None:
            print("❌ No data loaded")
            return
        
        print(f"\n📈 Visualizing {num_samples} sample signals...")
        
        # Select samples from different parts of the dataset
        indices = np.linspace(0, len(self.signals)-1, num_samples, dtype=int)
        
        fig, axes = plt.subplots(4, 3, figsize=(15, 12))
        fig.suptitle('Sample Neural Signals from Pickle File', fontsize=16, fontweight='bold')
        
        for i, idx in enumerate(indices):
            row, col = i // 3, i % 3
            signal = self.signals[idx]
            label = self.labels[idx]
            
            axes[row, col].plot(signal, linewidth=0.8, alpha=0.8)
            axes[row, col].set_title(f'Recording {idx}\n{label}', fontsize=10)
            axes[row, col].set_xlabel('Time Points')
            axes[row, col].set_ylabel('Amplitude')
            axes[row, col].grid(True, alpha=0.3)
            
            # Add statistics text
            mean_val = np.mean(signal)
            std_val = np.std(signal)
            axes[row, col].text(0.02, 0.98, f'μ={mean_val:.6f}\nσ={std_val:.6f}', 
                               transform=axes[row, col].transAxes, 
                               verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def visualize_signal_distribution(self):
        """Visualize the distribution of signal values"""
        if self.signals is None:
            print("❌ No data loaded")
            return
        
        print(f"\n📊 Visualizing signal value distribution...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Signal Value Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Flatten all signals for distribution analysis
        all_values = self.signals.flatten()
        
        # Histogram of all values
        axes[0, 0].hist(all_values, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of All Signal Values')
        axes[0, 0].set_xlabel('Signal Amplitude')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot of signal statistics per recording
        means = np.mean(self.signals, axis=1)
        stds = np.std(self.signals, axis=1)
        
        axes[0, 1].boxplot([means, stds], labels=['Mean', 'Std'])
        axes[0, 1].set_title('Distribution of Mean and Std per Recording')
        axes[0, 1].set_ylabel('Value')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Scatter plot: Mean vs Std
        axes[1, 0].scatter(means, stds, alpha=0.5, s=1)
        axes[1, 0].set_xlabel('Mean Value')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].set_title('Mean vs Standard Deviation per Recording')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Time series of mean values across recordings
        axes[1, 1].plot(means[:1000], alpha=0.7, linewidth=0.8)
        axes[1, 1].set_xlabel('Recording Index')
        axes[1, 1].set_ylabel('Mean Signal Value')
        axes[1, 1].set_title('Mean Signal Values Across First 1000 Recordings')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_label_patterns(self):
        """Visualize patterns in the labels"""
        if not self.label_parts:
            print("❌ No label data parsed")
            return
        
        print(f"\n🏷️ Visualizing label patterns...")
        
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(self.label_parts)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Label Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Channel index progression
        channel_indices = df['channel_index'].values
        axes[0, 0].plot(channel_indices[:1000], linewidth=0.8)
        axes[0, 0].set_title('Channel Index Progression (First 1000)')
        axes[0, 0].set_xlabel('Recording Index')
        axes[0, 0].set_ylabel('Channel Index')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Channel index distribution
        axes[0, 1].hist(channel_indices, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[0, 1].set_title('Distribution of Channel Indices')
        axes[0, 1].set_xlabel('Channel Index')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Channel ID frequency
        channel_id_counts = Counter(df['channel_id'])
        top_channels = dict(sorted(channel_id_counts.items(), key=lambda x: x[1], reverse=True)[:10])
        
        axes[1, 0].bar(range(len(top_channels)), list(top_channels.values()), 
                       color='lightcoral', alpha=0.7)
        axes[1, 0].set_title('Top 10 Most Frequent Channel IDs')
        axes[1, 0].set_xlabel('Channel ID Rank')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_xticks(range(len(top_channels)))
        axes[1, 0].set_xticklabels(list(top_channels.keys()), rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recording type distribution
        type_counts = Counter(df['recording_type'])
        axes[1, 1].pie(type_counts.values(), labels=type_counts.keys(), autopct='%1.1f%%')
        axes[1, 1].set_title('Distribution of Recording Types')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_signal_heatmap(self, num_recordings=100):
        """Create a heatmap visualization of multiple signals"""
        if self.signals is None:
            print("❌ No data loaded")
            return
        
        print(f"\n🔥 Creating signal heatmap for {num_recordings} recordings...")
        
        # Select subset of recordings
        subset_signals = self.signals[:num_recordings]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Signal Heatmap Analysis', fontsize=16, fontweight='bold')
        
        # Raw signal heatmap
        im1 = ax1.imshow(subset_signals, aspect='auto', cmap='RdBu_r', 
                         extent=[0, subset_signals.shape[1], 0, num_recordings])
        ax1.set_title('Raw Signal Heatmap')
        ax1.set_xlabel('Time Points')
        ax1.set_ylabel('Recording Index')
        plt.colorbar(im1, ax=ax1, label='Signal Amplitude')
        
        # Normalized signal heatmap
        normalized_signals = (subset_signals - np.mean(subset_signals, axis=1, keepdims=True)) / np.std(subset_signals, axis=1, keepdims=True)
        im2 = ax2.imshow(normalized_signals, aspect='auto', cmap='RdBu_r',
                         extent=[0, normalized_signals.shape[1], 0, num_recordings])
        ax2.set_title('Normalized Signal Heatmap (Z-score)')
        ax2.set_xlabel('Time Points')
        ax2.set_ylabel('Recording Index')
        plt.colorbar(im2, ax=ax2, label='Normalized Amplitude')
        
        plt.tight_layout()
        plt.show()
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Complete Pickle File Analysis")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        # Run all analyses
        self.basic_statistics()
        self.visualize_signal_samples()
        self.visualize_signal_distribution()
        self.visualize_label_patterns()
        self.visualize_signal_heatmap()
        
        print("\n✅ Complete analysis finished!")

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Initialize analyzer
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    analyzer = PickleAnalyzer(pickle_path)
    
    # Run complete analysis
    analyzer.run_complete_analysis()
