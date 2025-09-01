#!/usr/bin/env python3
import pickle
import numpy as np

def unpickle_and_show(file_path, max_items=10):
    """Unpickle the file and show its structure clearly"""
    print(f"Unpickling file: {file_path}")
    print("=" * 80)
    
    try:
        # Load the pickle file
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Successfully loaded pickle file!")
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        print()
        
        if isinstance(data, list):
            print(f"📋 This is a LIST containing {len(data)} items")
            print(f"Each item is a tuple of (signal_data, label_string)")
            print()
            
            # Show first few items in detail
            print("🔍 DETAILED VIEW OF FIRST FEW ITEMS:")
            print("-" * 60)
            
            for i in range(min(max_items, len(data))):
                item = data[i]
                print(f"\n📊 Item {i+1}:")
                print(f"   Type: {type(item)}")
                print(f"   Length: {len(item)}")
                
                if isinstance(item, tuple) and len(item) == 2:
                    signal, label = item
                    
                    print(f"   📈 Signal data:")
                    print(f"      Type: {type(signal)}")
                    print(f"      Shape: {signal.shape if hasattr(signal, 'shape') else 'N/A'}")
                    print(f"      Data type: {signal.dtype if hasattr(signal, 'dtype') else 'N/A'}")
                    
                    if hasattr(signal, 'shape'):
                        print(f"      First 5 values: {signal[:5]}")
                        print(f"      Last 5 values: {signal[-5:]}")
                        print(f"      Min value: {np.min(signal):.8f}")
                        print(f"      Max value: {np.max(signal):.8f}")
                        print(f"      Mean value: {np.mean(signal):.8f}")
                        print(f"      Std value: {np.std(signal):.8f}")
                    
                    print(f"   🏷️  Label:")
                    print(f"      Type: {type(label)}")
                    print(f"      Value: '{label}'")
                    
                    # Parse the label
                    parts = label.split('_')
                    print(f"      Parsed parts:")
                    for j, part in enumerate(parts):
                        print(f"         [{j}] '{part}'")
                    
                else:
                    print(f"   ❌ Unexpected format: {item}")
                
                print("-" * 40)
            
            # Show summary statistics
            print(f"\n📊 SUMMARY STATISTICS:")
            print(f"   Total items: {len(data)}")
            
            if len(data) > 0 and isinstance(data[0], tuple) and len(data[0]) == 2:
                # Check signal shapes
                signal_shapes = set()
                label_patterns = set()
                
                for i in range(min(100, len(data))):  # Check first 100 items
                    signal, label = data[i]
                    if hasattr(signal, 'shape'):
                        signal_shapes.add(signal.shape)
                    label_patterns.add(label)
                
                print(f"   Signal shapes found: {list(signal_shapes)}")
                print(f"   Unique labels in first 100: {len(label_patterns)}")
                
                # Show a few sample labels
                print(f"   Sample labels:")
                for i, (signal, label) in enumerate(data[:5]):
                    print(f"      [{i+1}] {label}")
                
                if len(data) > 5:
                    print(f"      ... and {len(data) - 5} more")
            
        elif isinstance(data, dict):
            print(f"📚 This is a DICTIONARY with keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"   Key '{key}': {type(value)} - {value.shape if hasattr(value, 'shape') else len(value) if hasattr(value, '__len__') else 'N/A'}")
        
        elif hasattr(data, 'shape'):
            print(f"🔢 This is a NUMPY ARRAY")
            print(f"   Shape: {data.shape}")
            print(f"   Data type: {data.dtype}")
            print(f"   First few values: {data.flatten()[:10]}")
        
        else:
            print(f"❓ Unknown data type: {data}")
        
        print("\n" + "=" * 80)
        print("✅ Pickle file analysis complete!")
        
    except Exception as e:
        print(f"❌ Error loading pickle file: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Use the sample pickle file path
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    unpickle_and_show(pickle_path, max_items=5)
