#!/usr/bin/env python3
import os
import sys

def analyze_nwb_file(file_path):
    """Analyze the NWB file structure and show its contents"""
    print(f"Analyzing NWB file: {file_path}")
    print("=" * 80)
    
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
                return
        
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"❌ File not found: {file_path}")
            return
        
        print(f"✓ File exists, size: {os.path.getsize(file_path) / (1024*1024*1024):.2f} GB")
        print()
        
        # Open and read the NWB file
        print("🔍 Opening NWB file...")
        with NWBHDF5IO(file_path, 'r') as io:
            nwbfile = io.read()
            
            print(f"✓ Successfully loaded NWB file!")
            print(f"📋 File structure:")
            print(f"   - Session description: {nwbfile.session_description}")
            print(f"   - Session start time: {nwbfile.session_start_time}")
            print(f"   - Institution: {nwbfile.institution}")
            print(f"   - Lab: {nwbfile.lab}")
            print(f"   - Experimenter: {nwbfile.experimenter}")
            print(f"   - Subject: {nwbfile.subject}")
            print()
            
            # Show all available fields
            print("🔍 Available fields in NWB file:")
            print("-" * 60)
            for field in dir(nwbfile):
                if not field.startswith('_') and not callable(getattr(nwbfile, field)):
                    value = getattr(nwbfile, field)
                    if value is not None:
                        print(f"   📊 {field}: {type(value).__name__}")
                        if hasattr(value, '__len__'):
                            print(f"      Length: {len(value)}")
                        if hasattr(value, 'shape'):
                            print(f"      Shape: {value.shape}")
                        if hasattr(value, 'dtype'):
                            print(f"      Data type: {value.dtype}")
                        print()
            
            # Check for specific neural data structures
            print("🧠 Neural Data Analysis:")
            print("-" * 60)
            
            # Look for acquisition data
            if hasattr(nwbfile, 'acquisition') and nwbfile.acquisition:
                print("📈 Acquisition data found:")
                for key, value in nwbfile.acquisition.items():
                    print(f"   - {key}: {type(value).__name__}")
                    if hasattr(value, 'data'):
                        print(f"     Data shape: {value.data.shape}")
                        print(f"     Data type: {value.data.dtype}")
                        print(f"     Time unit: {value.time_unit}")
                        if hasattr(value, 'description'):
                            print(f"     Description: {value.description}")
                        print()
            
            # Look for processing modules
            if hasattr(nwbfile, 'processing') and nwbfile.processing:
                print("⚙️ Processing modules found:")
                for key, value in nwbfile.processing.items():
                    print(f"   - {key}: {type(value).__name__}")
                    if hasattr(value, 'data_interfaces'):
                        for data_key, data_value in value.data_interfaces.items():
                            print(f"     Data interface: {data_key}")
                            if hasattr(data_value, 'data'):
                                print(f"       Data shape: {data_value.data.shape}")
                            print()
            
            # Look for electrodes
            if hasattr(nwbfile, 'electrodes') and nwbfile.electrodes:
                print("🔌 Electrode information:")
                print(f"   - Number of electrodes: {len(nwbfile.electrodes)}")
                if hasattr(nwbfile.electrodes, 'colnames'):
                    print(f"   - Column names: {nwbfile.electrodes.colnames}")
                print()
            
            # Look for units (spike data)
            if hasattr(nwbfile, 'units') and nwbfile.units:
                print("⚡ Units (spike data):")
                print(f"   - Number of units: {len(nwbfile.units)}")
                if hasattr(nwbfile.units, 'colnames'):
                    print(f"   - Column names: {nwbfile.units.colnames}")
                print()
            
            # Look for trials
            if hasattr(nwbfile, 'trials') and nwbfile.trials:
                print("🎯 Trials information:")
                print(f"   - Number of trials: {len(nwbfile.trials)}")
                if hasattr(nwbfile.trials, 'colnames'):
                    print(f"   - Column names: {nwbfile.trials.colnames}")
                print()
            
            # Show file hierarchy
            print("📁 File Hierarchy:")
            print("-" * 60)
            def print_hierarchy(obj, prefix="", max_depth=3, current_depth=0):
                if current_depth >= max_depth:
                    print(f"{prefix}... (max depth reached)")
                    return
                
                if hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes)):
                    try:
                        for key, value in obj.items():
                            print(f"{prefix}📁 {key}: {type(value).__name__}")
                            if hasattr(value, '__len__'):
                                print(f"{prefix}   Length: {len(value)}")
                            if hasattr(value, 'shape'):
                                print(f"{prefix}   Shape: {value.shape}")
                            print_hierarchy(value, prefix + "   ", max_depth, current_depth + 1)
                    except (AttributeError, TypeError):
                        pass
                elif hasattr(obj, '__len__') and len(obj) < 10:
                    for i, item in enumerate(obj):
                        print(f"{prefix}📄 [{i}]: {type(item).__name__}")
                        if hasattr(item, 'shape'):
                            print(f"{prefix}   Shape: {item.shape}")
            
            print_hierarchy(nwbfile)
            
        print("\n" + "=" * 80)
        print("✅ NWB file analysis complete!")
        
    except Exception as e:
        print(f"❌ Error analyzing NWB file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the NWB file path
    nwb_path = "/Users/uttamsingh/Downloads/probe_810755797_lfp.nwb"
    analyze_nwb_file(nwb_path)
