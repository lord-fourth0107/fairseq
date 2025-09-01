#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

class NWBAnalyzer:
    def __init__(self, file_path):
        self.file_path = file_path
        self.nwbfile = None
        self.lfp_data = None
        self.electrodes = None
        self.sampling_rate = None
        
    def load_data(self):
        """Load the NWB file and extract basic information"""
        print(f"🔍 Loading NWB file: {self.file_path}")
        
        try:
            # Try to import pynwb
            try:
                import pynwb
                from pynwb import NWBHDF5IO
                print(f"✓ Successfully imported pynwb version: {pynwb.__version__}")
            except ImportError:
                print("❌ pynwb not installed. Installing...")
                os.system("pip install pynwb")
                try:
                    import pynwb
                    from pynwb import NWBHDF5IO
                    print(f"✓ Successfully imported pynwb version: {pynwb.__version__}")
                except ImportError:
                    print("❌ Failed to install pynwb. Please install manually: pip install pynwb")
                    return False
            
            # Check if file exists
            if not os.path.exists(self.file_path):
                print(f"❌ File not found: {self.file_path}")
                return False
            
            print(f"✓ File exists, size: {os.path.getsize(self.file_path) / (1024*1024*1024):.2f} GB")
            
            # Open and read the NWB file
            print("🔍 Opening NWB file...")
            with NWBHDF5IO(self.file_path, 'r') as io:
                self.nwbfile = io.read()
                
                print(f"✓ Successfully loaded NWB file!")
                print(f"📋 Session: {self.nwbfile.session_description}")
                print(f"📋 Subject: {self.nwbfile.subject.species} - {self.nwbfile.subject.genotype}")
                print(f"📋 Session start: {self.nwbfile.session_start_time}")
                
                # Extract LFP data
                if hasattr(self.nwbfile, 'acquisition') and self.nwbfile.acquisition:
                    for key, value in self.nwbfile.acquisition.items():
                        if 'lfp' in key.lower() and hasattr(value, 'data'):
                            # Get the data shape and type without loading into memory
                            self.lfp_data = value.data
                            self.sampling_rate = value.rate if hasattr(value, 'rate') else None
                            print(f"✓ Found LFP data: {key}")
                            print(f"  - Shape: {self.lfp_data.shape}")
                            print(f"  - Data type: {self.lfp_data.dtype}")
                            break
                
                # Extract electrode information
                if hasattr(self.nwbfile, 'electrodes') and self.nwbfile.electrodes:
                    self.electrodes = self.nwbfile.electrodes
                    print(f"✓ Found electrode information: {len(self.electrodes)} channels")
                
                if self.lfp_data is None:
                    print("❌ No LFP data found in acquisition")
                    return False
                
        except Exception as e:
            print(f"❌ Error loading NWB file: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        return True
    
    def basic_statistics(self):
        """Display basic statistics about the data"""
        print("\n📊 BASIC STATISTICS:")
        print("=" * 60)
        
        if self.lfp_data is None:
            print("❌ No data loaded")
            return
        
        print(f"LFP Data Shape: {self.lfp_data.shape}")
        print(f"Total time points: {self.lfp_data.shape[0]:,}")
        print(f"Number of channels: {self.lfp_data.shape[1]}")
        print(f"Data type: {self.lfp_data.dtype}")
        
        # Calculate memory usage safely
        try:
            # Estimate memory usage based on shape and dtype
            element_size = np.dtype(self.lfp_data.dtype).itemsize
            estimated_memory = self.lfp_data.shape[0] * self.lfp_data.shape[1] * element_size
            print(f"Estimated memory usage: {estimated_memory / (1024**3):.2f} GB")
        except:
            print("Memory usage: Unable to calculate (HDF5 dataset)")
        
        if self.sampling_rate:
            duration = self.lfp_data.shape[0] / self.sampling_rate
            print(f"Sampling rate: {self.sampling_rate} Hz")
            print(f"Total duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        
        # Signal statistics - load small chunks to avoid memory issues
        print(f"\nSignal Statistics (sampling from data):")
        try:
            # Sample first 10000 time points from first 10 channels for statistics
            sample_data = self.lfp_data[:10000, :10]
            print(f"  Min value: {np.min(sample_data):.8f}")
            print(f"  Max value: {np.max(sample_data):.8f}")
            print(f"  Mean value: {np.mean(sample_data):.8f}")
            print(f"  Std value: {np.std(sample_data):.8f}")
        except Exception as e:
            print(f"  Unable to compute statistics: {e}")
        
        # Channel statistics
        try:
            # Sample from first 10000 time points for all channels
            sample_data = self.lfp_data[:10000, :]
            channel_means = np.mean(sample_data, axis=0)
            channel_stds = np.std(sample_data, axis=0)
            print(f"\nChannel Statistics (sampled):")
            print(f"  Mean across channels: {np.mean(channel_means):.8f}")
            print(f"  Std across channels: {np.std(channel_means):.8f}")
            print(f"  Channel with max mean: {np.argmax(channel_means)} ({np.max(channel_means):.8f})")
            print(f"  Channel with min mean: {np.argmin(channel_means)} ({np.min(channel_means):.8f})")
        except Exception as e:
            print(f"  Unable to compute channel statistics: {e}")
    
    def visualize_channel_samples(self, num_channels=12, time_window=10000):
        """Visualize sample channels over a time window"""
        if self.lfp_data is None:
            print("❌ No data loaded")
            return
        
        print(f"\n📈 Visualizing {num_channels} sample channels...")
        
        try:
            # Select channels and time window
            channel_indices = np.linspace(0, self.lfp_data.shape[1]-1, num_channels, dtype=int)
            time_data = self.lfp_data[:time_window, channel_indices]
            
            fig, axes = plt.subplots(4, 3, figsize=(15, 12))
            fig.suptitle(f'Sample LFP Channels (First {time_window:,} time points)', fontsize=16, fontweight='bold')
            
            for i, ch_idx in enumerate(channel_indices):
                row, col = i // 3, i % 3
                signal = time_data[:, i]
                
                axes[row, col].plot(signal, linewidth=0.8, alpha=0.8)
                axes[row, col].set_title(f'Channel {ch_idx}', fontsize=10)
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
        except Exception as e:
            print(f"❌ Error in channel visualization: {e}")
    
    def visualize_signal_distribution(self):
        """Visualize the distribution of signal values"""
        if self.lfp_data is None:
            print("❌ No data loaded")
            return
        
        print(f"\n📊 Visualizing signal value distribution...")
        
        try:
            # Sample data for visualization to avoid memory issues
            sample_size = min(100000, self.lfp_data.shape[0])  # Max 100k time points
            sample_data = self.lfp_data[:sample_size, :]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'LFP Signal Value Distribution Analysis (Sampled: {sample_size:,} time points)', fontsize=16, fontweight='bold')
            
            # Flatten sampled data for distribution analysis
            all_values = sample_data.flatten()
            
            # Histogram of all values
            axes[0, 0].hist(all_values, bins=100, alpha=0.7, color='skyblue', edgecolor='black')
            axes[0, 0].set_title('Distribution of Sampled LFP Values')
            axes[0, 0].set_xlabel('Signal Amplitude')
            axes[0, 0].set_ylabel('Frequency')
            axes[0, 0].grid(True, alpha=0.3)
            
            # Box plot of signal statistics per channel
            channel_means = np.mean(sample_data, axis=0)
            channel_stds = np.std(sample_data, axis=0)
            
            axes[0, 1].boxplot([channel_means, channel_stds], labels=['Mean', 'Std'])
            axes[0, 1].set_title('Distribution of Mean and Std per Channel')
            axes[0, 1].set_ylabel('Value')
            axes[0, 1].grid(True, alpha=0.3)
            
            # Scatter plot: Mean vs Std per channel
            axes[1, 0].scatter(channel_means, channel_stds, alpha=0.7, s=20)
            axes[1, 0].set_xlabel('Mean Value per Channel')
            axes[1, 0].set_ylabel('Standard Deviation per Channel')
            axes[1, 0].set_title('Mean vs Standard Deviation per Channel')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Channel mean values across all channels
            axes[1, 1].plot(channel_means, alpha=0.7, linewidth=0.8, marker='o', markersize=3)
            axes[1, 1].set_xlabel('Channel Index')
            axes[1, 1].set_ylabel('Mean Signal Value')
            axes[1, 1].set_title('Mean Signal Values Across All Channels')
            axes[1, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Error in signal distribution visualization: {e}")
    
    def visualize_electrode_metadata(self):
        """Visualize electrode metadata and spatial information"""
        if self.electrodes is None:
            print("❌ No electrode information available")
            return
        
        print(f"\n🔌 Visualizing electrode metadata...")
        
        try:
            # Extract electrode information
            electrode_data = {}
            for col in self.electrodes.colnames:
                electrode_data[col] = self.electrodes[col][:]
            
            # Create DataFrame
            df = pd.DataFrame(electrode_data)
            
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle('Electrode Metadata Analysis', fontsize=16, fontweight='bold')
            
            # 3D spatial plot
            if 'x' in df.columns and 'y' in df.columns and 'z' in df.columns:
                ax = fig.add_subplot(2, 3, 1, projection='3d')
                scatter = ax.scatter(df['x'], df['y'], df['z'], c=range(len(df)), cmap='viridis', s=50)
                ax.set_xlabel('X Position')
                ax.set_ylabel('Y Position')
                ax.set_zlabel('Z Position')
                ax.set_title('3D Electrode Positions')
                plt.colorbar(scatter, ax=ax, label='Channel Index')
            
            # 2D projection (X vs Y)
            if 'x' in df.columns and 'y' in df.columns:
                axes[0, 1].scatter(df['x'], df['y'], c=range(len(df)), cmap='viridis', s=50)
                axes[0, 1].set_xlabel('X Position')
                axes[0, 1].set_ylabel('Y Position')
                axes[0, 1].set_title('2D Electrode Positions (X vs Y)')
                axes[0, 1].grid(True, alpha=0.3)
            
            # 2D projection (X vs Z)
            if 'x' in df.columns and 'z' in df.columns:
                axes[0, 2].scatter(df['x'], df['z'], c=range(len(df)), cmap='viridis', s=50)
                axes[0, 2].set_xlabel('X Position')
                axes[0, 2].set_ylabel('Z Position')
                axes[0, 2].set_title('2D Electrode Positions (X vs Z)')
                axes[0, 2].grid(True, alpha=0.3)
            
            # Probe vertical position
            if 'probe_vertical_position' in df.columns:
                axes[1, 0].plot(df['probe_vertical_position'], 'o-', alpha=0.7)
                axes[1, 0].set_xlabel('Channel Index')
                axes[1, 0].set_ylabel('Vertical Position')
                axes[1, 0].set_title('Probe Vertical Positions')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Probe horizontal position
            if 'probe_vertical_position' in df.columns:
                axes[1, 1].plot(df['probe_horizontal_position'], 'o-', alpha=0.7)
                axes[1, 1].set_xlabel('Channel Index')
                axes[1, 1].set_ylabel('Horizontal Position')
                axes[1, 1].set_title('Probe Horizontal Positions')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Location distribution
            if 'location' in df.columns:
                location_counts = Counter(df['location'])
                axes[1, 2].pie(location_counts.values(), labels=location_counts.keys(), autopct='%1.1f%%')
                axes[1, 2].set_title('Distribution of Brain Locations')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Error in electrode metadata visualization: {e}")
    
    def visualize_signal_heatmap(self, time_window=50000):
        """Create a heatmap visualization of the LFP data"""
        if self.lfp_data is None:
            print("❌ No data loaded")
            return
        
        print(f"\n🔥 Creating LFP signal heatmap for {time_window:,} time points...")
        
        try:
            # Select time window
            subset_data = self.lfp_data[:time_window, :]
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
            fig.suptitle('LFP Signal Heatmap Analysis', fontsize=16, fontweight='bold')
            
            # Raw signal heatmap
            im1 = ax1.imshow(subset_data.T, aspect='auto', cmap='RdBu_r', 
                             extent=[0, time_window, 0, subset_data.shape[1]])
            ax1.set_title('Raw LFP Signal Heatmap')
            ax1.set_xlabel('Time Points')
            ax1.set_ylabel('Channel Index')
            plt.colorbar(im1, ax=ax1, label='Signal Amplitude')
            
            # Normalized signal heatmap
            normalized_data = (subset_data - np.mean(subset_data, axis=0, keepdims=True)) / np.std(subset_data, axis=0, keepdims=True)
            im2 = ax2.imshow(normalized_data.T, aspect='auto', cmap='RdBu_r',
                             extent=[0, time_window, 0, normalized_data.shape[1]])
            ax2.set_title('Normalized LFP Signal Heatmap (Z-score)')
            ax2.set_xlabel('Time Points')
            ax2.set_ylabel('Channel Index')
            plt.colorbar(im2, ax=ax2, label='Normalized Amplitude')
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Error in signal heatmap visualization: {e}")
    
    def visualize_power_spectrum(self, num_channels=6, time_window=100000):
        """Visualize power spectrum of selected channels"""
        if self.lfp_data is None:
            print("❌ No data loaded")
            return
        
        print(f"\n📊 Visualizing power spectrum for {num_channels} channels...")
        
        try:
            # Select channels and time window
            channel_indices = np.linspace(0, self.lfp_data.shape[1]-1, num_channels, dtype=int)
            time_data = self.lfp_data[:time_window, channel_indices]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            fig.suptitle(f'Power Spectrum Analysis (First {time_window:,} time points)', fontsize=16, fontweight='bold')
            
            for i, ch_idx in enumerate(channel_indices):
                row, col = i // 3, i % 3
                signal = time_data[:, i]
                
                # Compute power spectrum
                fft_vals = np.fft.fft(signal)
                freqs = np.fft.fftfreq(len(signal), d=1/self.sampling_rate) if self.sampling_rate else np.fft.fftfreq(len(signal))
                power = np.abs(fft_vals)**2
                
                # Plot only positive frequencies
                pos_mask = freqs > 0
                axes[row, col].semilogy(freqs[pos_mask], power[pos_mask], linewidth=0.8)
                axes[row, col].set_title(f'Channel {ch_idx}')
                axes[row, col].set_xlabel('Frequency (Hz)' if self.sampling_rate else 'Frequency')
                axes[row, col].set_ylabel('Power')
                axes[row, col].grid(True, alpha=0.3)
                axes[row, col].set_xlim(0, freqs[pos_mask].max())
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"❌ Error in power spectrum visualization: {e}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Complete NWB File Analysis")
        print("=" * 80)
        
        # Load data
        if not self.load_data():
            return
        
        # Run all analyses
        self.basic_statistics()
        self.visualize_channel_samples()
        self.visualize_signal_distribution()
        self.visualize_electrode_metadata()
        self.visualize_signal_heatmap()
        self.visualize_power_spectrum()
        
        print("\n✅ Complete analysis finished!")

if __name__ == "__main__":
    # Set style for better plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Initialize analyzer
    nwb_path = "/Users/uttamsingh/Downloads/probe_810755797_lfp.nwb"
    analyzer = NWBAnalyzer(nwb_path)
    
    # Run complete analysis
    analyzer.run_complete_analysis()
