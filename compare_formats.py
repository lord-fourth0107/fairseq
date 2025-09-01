#!/usr/bin/env python3
import pickle
import os
import numpy as np

def compare_formats():
    """Compare the pickle and NWB file formats"""
    print("🔍 COMPARISON: Pickle vs NWB File Formats")
    print("=" * 80)
    
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    nwb_path = "/Users/uttamsingh/Downloads/probe_810755797_lfp.nwb"
    
    print("📊 FILE OVERVIEW:")
    print("-" * 60)
    print(f"Pickle file: {os.path.basename(pickle_path)}")
    print(f"  Size: {os.path.getsize(pickle_path) / (1024*1024*1024):.2f} GB")
    print(f"  Format: Python pickle (list of tuples)")
    
    print(f"\nNWB file: {os.path.basename(nwb_path)}")
    print(f"  Size: {os.path.getsize(nwb_path) / (1024*1024*1024):.2f} GB")
    print(f"  Format: Neurodata Without Borders (HDF5)")
    print()
    
    # Analyze pickle file
    print("📋 PICKLE FILE ANALYSIS:")
    print("-" * 60)
    try:
        with open(pickle_path, 'rb') as f:
            pickle_data = pickle.load(f)
        
        print(f"✓ Data type: {type(pickle_data)}")
        print(f"✓ Total items: {len(pickle_data):,}")
        
        if len(pickle_data) > 0:
            first_item = pickle_data[0]
            if isinstance(first_item, tuple) and len(first_item) == 2:
                signal, label = first_item
                print(f"✓ Item structure: (signal_data, label_string)")
                print(f"✓ Signal shape: {signal.shape}")
                print(f"✓ Signal data type: {signal.dtype}")
                print(f"✓ Label example: '{label}'")
                
                # Count unique labels
                unique_labels = set()
                for i in range(min(1000, len(pickle_data))):
                    _, label = pickle_data[i]
                    unique_labels.add(label)
                print(f"✓ Unique labels in first 1000: {len(unique_labels)}")
                
                # Parse label structure
                if unique_labels:
                    sample_label = list(unique_labels)[0]
                    parts = sample_label.split('_')
                    print(f"✓ Label format: {len(parts)} parts separated by '_'")
                    print(f"  - Session ID: {parts[0]}")
                    print(f"  - Channel index: {parts[1]} (increments)")
                    print(f"  - Recording session: {parts[2]}")
                    print(f"  - Channel ID: {parts[3]}")
                    print(f"  - Recording type: {parts[4]}")
        
    except Exception as e:
        print(f"❌ Error reading pickle: {e}")
    
    print()
    
    # Analyze NWB file
    print("🧠 NWB FILE ANALYSIS:")
    print("-" * 60)
    try:
        import pynwb
        from pynwb import NWBHDF5IO
        
        with NWBHDF5IO(nwb_path, 'r') as io:
            nwbfile = io.read()
            
            print(f"✓ Session: {nwbfile.session_description}")
            print(f"✓ Subject: {nwbfile.subject.species} - {nwbfile.subject.genotype}")
            print(f"✓ Session start: {nwbfile.session_start_time}")
            
            # Acquisition data
            if hasattr(nwbfile, 'acquisition') and nwbfile.acquisition:
                for key, value in nwbfile.acquisition.items():
                    if hasattr(value, 'data'):
                        print(f"✓ Data: {key}")
                        print(f"  - Shape: {value.data.shape}")
                        print(f"  - Data type: {value.data.dtype}")
                        print(f"  - Time unit: {value.time_unit}")
                        print(f"  - Total time points: {value.data.shape[0]:,}")
                        print(f"  - Number of channels: {value.data.shape[1]}")
            
            # Electrodes
            if hasattr(nwbfile, 'electrodes') and nwbfile.electrodes:
                print(f"✓ Electrodes: {len(nwbfile.electrodes)} channels")
                if hasattr(nwbfile.electrodes, 'colnames'):
                    print(f"  - Metadata columns: {nwbfile.electrodes.colnames}")
            
            # Processing
            if hasattr(nwbfile, 'processing') and nwbfile.processing:
                for key, value in nwbfile.processing.items():
                    print(f"✓ Processing: {key}")
                    if hasattr(value, 'data_interfaces'):
                        for data_key, data_value in value.data_interfaces.items():
                            print(f"  - Interface: {data_key}")
                            if hasattr(data_value, 'data'):
                                print(f"    Shape: {data_value.data.shape}")
        
    except Exception as e:
        print(f"❌ Error reading NWB: {e}")
    
    print()
    
    # Comparison summary
    print("📊 FORMAT COMPARISON SUMMARY:")
    print("-" * 60)
    print("🔴 PICKLE FORMAT:")
    print("   + Simple Python structure")
    print("   + Easy to load and manipulate")
    print("   + Contains both data and labels")
    print("   + Good for quick analysis")
    print("   - No standardized metadata")
    print("   - No built-in validation")
    print("   - Platform-dependent")
    print()
    
    print("🟢 NWB FORMAT:")
    print("   + Standardized neuroscience format")
    print("   + Rich metadata and annotations")
    print("   + Built-in validation")
    print("   + Platform-independent")
    print("   + Hierarchical organization")
    print("   - More complex to work with")
    print("   - Requires specialized libraries")
    print("   - Larger file size")
    print()
    
    print("🎯 RECOMMENDATIONS:")
    print("-" * 60)
    print("• Use PICKLE for: Quick data access, simple analysis, prototyping")
    print("• Use NWB for: Publication, sharing, long-term storage, metadata")
    print("• For your SSL training: Both formats work, but NWB provides better")
    print("  channel information and metadata for brain region mapping")
    
    print("\n" + "=" * 80)
    print("✅ Comparison complete!")

if __name__ == "__main__":
    compare_formats()
