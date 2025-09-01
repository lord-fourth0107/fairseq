#!/usr/bin/env python3
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

def analyze_nwb_simple(file_path):
    """Simple but robust NWB file analysis"""
    print(f"🔍 Simple NWB Analysis: {file_path}")
    print("=" * 80)
    
    try:
        # Import pynwb
        import pynwb
        from pynwb import NWBHDF5IO
        print(f"✓ pynwb version: {pynwb.__version__}")
        
        # Check file
        if not os.path.exists(file_path):
            print(f"❌ File not found")
            return
        
        print(f"✓ File size: {os.path.getsize(file_path) / (1024*1024*1024):.2f} GB")
        
        # Open file and keep it open
        io = NWBHDF5IO(file_path, 'r')
        nwbfile = io.read()
        
        print(f"✓ Loaded NWB file")
        print(f"📋 Session: {nwbfile.session_description}")
        print(f"📋 Subject: {nwbfile.subject.species} - {nwbfile.subject.genotype}")
        print(f"📋 Session start: {nwbfile.session_start_time}")
        
        # Extract basic info
        if hasattr(nwbfile, 'acquisition') and nwbfile.acquisition:
            print(f"\n📊 Acquisition data:")
            for key, value in nwbfile.acquisition.items():
                print(f"  - {key}: {type(value).__name__}")
                if hasattr(value, 'data'):
                    print(f"    Shape: {value.data.shape}")
                    print(f"    Type: {value.data.dtype}")
                    if hasattr(value, 'rate'):
                        print(f"    Rate: {value.rate} Hz")
        
        # Extract electrode info
        if hasattr(nwbfile, 'electrodes') and nwbfile.electrodes:
            print(f"\n🔌 Electrodes: {len(nwbfile.electrodes)} channels")
            if hasattr(nwbfile.electrodes, 'colnames'):
                print(f"  Columns: {nwbfile.electrodes.colnames}")
        
        # Try to load a small sample of data
        print(f"\n📈 Loading data samples...")
        
        # Find LFP data
        lfp_data = None
        for key, value in nwbfile.acquisition.items():
            if 'lfp' in key.lower() and hasattr(value, 'data'):
                lfp_data = value.data
                break
        
        if lfp_data is not None:
            print(f"✓ Found LFP data with shape: {lfp_data.shape}")
            
            # Load small samples for analysis
            try:
                # Sample 1000 time points from first 10 channels
                sample_data = lfp_data[:1000, :10]
                print(f"✓ Loaded sample: {sample_data.shape}")
                
                # Basic statistics
                print(f"\n📊 Sample Statistics:")
                print(f"  Min: {np.min(sample_data):.8f}")
                print(f"  Max: {np.max(sample_data):.8f}")
                print(f"  Mean: {np.mean(sample_data):.8f}")
                print(f"  Std: {np.std(sample_data):.8f}")
                
                # Visualize sample
                visualize_sample_data(sample_data, "LFP Sample Data")
                
            except Exception as e:
                print(f"❌ Error loading sample: {e}")
        
        # Extract electrode metadata
        if hasattr(nwbfile, 'electrodes') and nwbfile.electrodes:
            print(f"\n🔌 Extracting electrode metadata...")
            try:
                electrode_data = {}
                for col in nwbfile.electrodes.colnames:
                    electrode_data[col] = nwbfile.electrodes[col][:]
                
                df = pd.DataFrame(electrode_data)
                print(f"✓ Electrode DataFrame: {df.shape}")
                print(f"  Columns: {list(df.columns)}")
                
                # Show some metadata
                if 'location' in df.columns:
                    locations = df['location'].value_counts()
                    print(f"\n📍 Brain Locations:")
                    for loc, count in locations.items():
                        print(f"  {loc}: {count} channels")
                
                if 'x' in df.columns and 'y' in df.columns:
                    print(f"\n📍 Spatial Info:")
                    print(f"  X range: {df['x'].min():.2f} to {df['x'].max():.2f}")
                    print(f"  Y range: {df['y'].min():.2f} to {df['y'].max():.2f}")
                
                # Visualize electrode positions
                visualize_electrode_positions(df)
                
            except Exception as e:
                print(f"❌ Error extracting electrode data: {e}")
        
        # Close the file
        io.close()
        
        print(f"\n✅ Analysis complete!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

def visualize_sample_data(data, title):
    """Visualize sample LFP data"""
    print(f"\n📈 Creating sample visualization...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(title, fontsize=16, fontweight='bold')
    
    # Time series of first few channels
    for i in range(min(4, data.shape[1])):
        row, col = i // 2, i % 2
        signal = data[:, i]
        axes[row, col].plot(signal, linewidth=0.8, alpha=0.8)
        axes[row, col].set_title(f'Channel {i}')
        axes[row, col].set_xlabel('Time Points')
        axes[row, col].set_ylabel('Amplitude')
        axes[row, col].grid(True, alpha=0.3)
        
        # Add stats
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        axes[row, col].text(0.02, 0.98, f'μ={mean_val:.6f}\nσ={std_val:.6f}', 
                           transform=axes[row, col].transAxes, 
                           verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.show()

def visualize_electrode_positions(df):
    """Visualize electrode positions"""
    print(f"\n🔌 Creating electrode position visualization...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Electrode Positions', fontsize=16, fontweight='bold')
    
    # 2D positions
    if 'x' in df.columns and 'y' in df.columns:
        scatter = axes[0].scatter(df['x'], df['y'], c=range(len(df)), cmap='viridis', s=50)
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        axes[0].set_title('2D Electrode Positions (X vs Y)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[0], label='Channel Index')
    
    # Vertical positions
    if 'probe_vertical_position' in df.columns:
        axes[1].plot(df['probe_vertical_position'], 'o-', alpha=0.7)
        axes[1].set_xlabel('Channel Index')
        axes[1].set_ylabel('Vertical Position')
        axes[1].set_title('Probe Vertical Positions')
        axes[1].grid(True, alpha=0.3)
    
    # Horizontal positions
    if 'probe_horizontal_position' in df.columns:
        axes[2].plot(df['probe_horizontal_position'], 'o-', alpha=0.7)
        axes[2].set_xlabel('Channel Index')
        axes[2].set_ylabel('Horizontal Position')
        axes[2].set_title('Probe Horizontal Positions')
        axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Analyze NWB file
    nwb_path = "/Users/uttamsingh/Downloads/probe_810755797_lfp.nwb"
    analyze_nwb_simple(nwb_path)
