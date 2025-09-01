#!/usr/bin/env python3
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter

def compare_formats_detailed():
    """Detailed comparison of pickle vs NWB formats"""
    print("🔍 DETAILED FORMAT COMPARISON: Pickle vs NWB")
    print("=" * 80)
    
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    nwb_path = "/Users/uttamsingh/Downloads/probe_810755797_lfp.nwb"
    
    # ===== PICKLE FILE ANALYSIS =====
    print("\n📋 PICKLE FILE ANALYSIS:")
    print("-" * 60)
    
    try:
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        
        print(f"✓ File size: {os.path.getsize(pickle_path) / (1024**3):.2f} GB")
        print(f"✓ Data type: {type(pickle_data)}")
        print(f"✓ Total recordings: {len(pickle_data):,}")
        
        if len(pickle_data) > 0:
            first_item = pickle_data[0]
            if isinstance(first_item, tuple) and len(first_item) == 2:
                signal, label = first_item
                print(f"✓ Signal shape: {signal.shape}")
                print(f"✓ Signal data type: {signal.dtype}")
                print(f"✓ Label format: {label}")
                
                # Parse label structure
                parts = label.split('_')
                print(f"✓ Label components:")
                print(f"  - Session ID: {parts[0]}")
                print(f"  - Channel index: {parts[1]} (increments)")
                print(f"  - Recording session: {parts[2]}")
                print(f"  - Channel ID: {parts[3]}")
                print(f"  - Recording type: {parts[4]}")
                
                # Count unique values
                unique_labels = set()
                unique_channels = set()
                unique_types = set()
                
                for i in range(min(1000, len(pickle_data))):
                    _, label = pickle_data[i]
                    unique_labels.add(label)
                    parts = label.split('_')
                    unique_channels.add(parts[3])
                    unique_types.add(parts[4])
                
                print(f"\n📊 Statistics (first 1000 recordings):")
                print(f"  Unique labels: {len(unique_labels)}")
                print(f"  Unique channel IDs: {len(unique_channels)}")
                print(f"  Recording types: {list(unique_types)}")
                
    except Exception as e:
        print(f"❌ Error reading pickle: {e}")
    
    # ===== NWB FILE ANALYSIS =====
    print("\n🧠 NWB FILE ANALYSIS:")
    print("-" * 60)
    
    try:
        import pynwb
        from pynwb import NWBHDF5IO
        
        print(f"✓ File size: {os.path.getsize(nwb_path) / (1024**3):.2f} GB")
        
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwbfile = io.read()
            
            print(f"✓ Session: {nwbfile.session_description}")
            print(f"✓ Subject: {nwbfile.subject.species} - {nwbfile.subject.genotype}")
            print(f"✓ Session start: {nwbfile.session_start_time}")
            
            # LFP data info
            if hasattr(nwbfile, 'acquisition') and nwbfile.acquisition:
                for key, value in nwbfile.acquisition.items():
                    if 'lfp' in key.lower() and hasattr(value, 'data'):
                        print(f"✓ LFP data shape: {value.data.shape}")
                        print(f"✓ Data type: {value.data.dtype}")
                        if hasattr(value, 'rate'):
                            print(f"✓ Sampling rate: {value.rate} Hz")
            
            # Electrode info
            if hasattr(nwbfile, 'electrodes') and nwbfile.electrodes:
                print(f"✓ Electrodes: {len(nwbfile.electrodes)} channels")
                if hasattr(nwbfile.electrodes, 'colnames'):
                    print(f"✓ Metadata columns: {nwbfile.electrodes.colnames}")
                
                # Extract electrode metadata
                electrode_data = {}
                for col in nwbfile.electrodes.colnames:
                    electrode_data[col] = nwbfile.electrodes[col][:]
                
                df = pd.DataFrame(electrode_data)
                
                if 'location' in df.columns:
                    locations = df['location'].value_counts()
                    print(f"\n📍 Brain Locations:")
                    for loc, count in locations.items():
                        print(f"  {loc}: {count} channels")
                
                if 'x' in df.columns and 'y' in df.columns:
                    print(f"\n📍 Spatial Coverage:")
                    print(f"  X range: {df['x'].min():.0f} to {df['x'].max():.0f} μm")
                    print(f"  Y range: {df['y'].min():.0f} to {df['y'].max():.0f} μm")
                    print(f"  Total area: {(df['x'].max() - df['x'].min()) * (df['y'].max() - df['y'].min()) / 1e6:.2f} mm²")
        
    except Exception as e:
        print(f"❌ Error reading NWB: {e}")
    
    # ===== COMPARISON SUMMARY =====
    print("\n📊 FORMAT COMPARISON SUMMARY:")
    print("=" * 80)
    
    print("🔴 PICKLE FORMAT:")
    print("   + Simple Python structure")
    print("   + Easy to load and manipulate")
    print("   + Contains both data and labels")
    print("   + Good for quick analysis and prototyping")
    print("   - No standardized metadata")
    print("   - No built-in validation")
    print("   - Platform-dependent")
    print("   - No spatial coordinates")
    print("   - Limited brain region information")
    
    print("\n🟢 NWB FORMAT:")
    print("   + Standardized neuroscience format")
    print("   + Rich metadata and annotations")
    print("   + Built-in validation")
    print("   + Platform-independent")
    print("   + Hierarchical organization")
    print("   + 3D spatial coordinates for electrodes")
    print("   + Detailed brain region mapping")
    print("   + Sampling rate and timing information")
    print("   - More complex to work with")
    print("   - Requires specialized libraries")
    print("   - Larger file size")
    
    # ===== RECOMMENDATIONS FOR SSL TRAINING =====
    print("\n🎯 RECOMMENDATIONS FOR YOUR SSL TRAINING:")
    print("=" * 80)
    
    print("📈 DATA STRUCTURE FOR 2D CNN:")
    print("  • Pickle: 265,608 recordings × 3,750 time points")
    print("    → Reshape to: (batch_size, 1, 265608, 3750)")
    print("    → Each recording becomes a 'channel' in height dimension")
    print("    → Time points become width dimension")
    
    print("\n  • NWB: 93 channels × 10,715,666 time points")
    print("    → Reshape to: (batch_size, 1, 93, time_window)")
    print("    → 93 physical channels become height dimension")
    print("    → Time becomes width dimension")
    
    print("\n🧠 BRAIN REGION MAPPING:")
    print("  • Pickle: Limited brain region info (APN, CA1, DG, MB, VISam)")
    print("    → Good for basic SSL training")
    print("    → Brain mapping requires additional metadata")
    
    print("\n  • NWB: Rich spatial information")
    print("    → 3D coordinates for each electrode")
    print("    → Precise brain region mapping")
    print("    → Ideal for spatial-aware SSL training")
    
    print("\n💡 FINAL RECOMMENDATION:")
    print("  • Use PICKLE for: Quick SSL training, prototyping, basic analysis")
    print("  • Use NWB for: Production SSL training, brain region mapping, publication")
    print("  • For your goal of mapping probe data to brain coordinates:")
    print("    → Start with pickle for SSL training")
    print("    → Use NWB metadata for final brain mapping")
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete!")

if __name__ == "__main__":
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Run detailed comparison
    compare_formats_detailed()
