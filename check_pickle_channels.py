#!/usr/bin/env python3
import pickle
import numpy as np
from collections import Counter, defaultdict

def check_pickle_channel_structure(file_path, max_samples=10000):
    """Check if pickle file contains data from 93 channels"""
    print(f"🔍 Checking pickle file for 93-channel structure: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded {len(data):,} recordings")
        
        # Parse all labels to understand the structure
        print(f"\n📋 Analyzing label structure...")
        
        label_analysis = []
        for i, (signal, label) in enumerate(data[:max_samples]):
            parts = label.split('_')
            if len(parts) == 5:
                label_analysis.append({
                    'index': i,
                    'session_id': parts[0],
                    'channel_index': int(parts[1]),
                    'recording_session': parts[2],
                    'channel_id': parts[3],
                    'recording_type': parts[4]
                })
        
        print(f"✓ Analyzed {len(label_analysis)} labels")
        
        # Check for channel cycling patterns
        print(f"\n🔄 Checking for channel cycling patterns...")
        
        # Group by channel_id to see if we have 93 unique channels
        channel_id_groups = defaultdict(list)
        for item in label_analysis:
            channel_id_groups[item['channel_id']].append(item)
        
        print(f"✓ Found {len(channel_id_groups)} unique channel IDs")
        
        # Show first few channel IDs
        print(f"\n📊 First 10 unique channel IDs:")
        for i, (ch_id, items) in enumerate(list(channel_id_groups.items())[:10]):
            print(f"  [{i+1}] {ch_id}: {len(items)} recordings")
        
        # Check if we have exactly 93 channels
        if len(channel_id_groups) == 93:
            print(f"\n🎯 EXACTLY 93 CHANNELS FOUND!")
        elif len(channel_id_groups) > 93:
            print(f"\n⚠️  More than 93 channels: {len(channel_id_groups)}")
        else:
            print(f"\n⚠️  Fewer than 93 channels: {len(channel_id_groups)}")
        
        # Analyze channel index progression
        print(f"\n📈 Analyzing channel index progression...")
        
        # Check if channel_index cycles through 0-92 (or 1-93)
        channel_indices = [item['channel_index'] for item in label_analysis]
        unique_indices = sorted(set(channel_indices))
        
        print(f"✓ Channel indices found: {unique_indices}")
        print(f"✓ Range: {min(unique_indices)} to {max(unique_indices)}")
        
        # Check if indices cycle
        if len(unique_indices) <= 93:
            print(f"✓ Channel indices cycle through {len(unique_indices)} values")
            
            # Look for cycling pattern
            cycling_patterns = []
            current_cycle = []
            last_index = -1
            
            for item in label_analysis[:1000]:  # Check first 1000
                if item['channel_index'] <= last_index:  # Reset detected
                    if current_cycle:
                        cycling_patterns.append(current_cycle)
                        current_cycle = []
                current_cycle.append(item['channel_index'])
                last_index = item['channel_index']
            
            if cycling_patterns:
                print(f"✓ Detected {len(cycling_patterns)} cycling patterns")
                print(f"  First cycle length: {len(cycling_patterns[0])}")
                print(f"  First cycle indices: {cycling_patterns[0]}")
        
        # Check for spatial organization
        print(f"\n📍 Checking for spatial organization...")
        
        # Look at recording types to see if they correspond to brain regions
        recording_types = Counter([item['recording_type'] for item in label_analysis])
        print(f"✓ Recording types: {dict(recording_types)}")
        
        # Check if recording types cycle with channel indices
        type_channel_mapping = defaultdict(set)
        for item in label_analysis:
            type_channel_mapping[item['recording_type']].add(item['channel_index'])
        
        print(f"\n📊 Recording type vs channel index mapping:")
        for rec_type, channels in type_channel_mapping.items():
            print(f"  {rec_type}: channels {sorted(channels)}")
        
        # Final assessment
        print(f"\n🎯 FINAL ASSESSMENT:")
        print("=" * 60)
        
        if len(channel_id_groups) == 93:
            print("✅ STRONG EVIDENCE: Pickle contains data from 93 channels")
            print("   - 93 unique channel IDs found")
            print("   - Data likely organized by physical electrode channels")
        elif len(unique_indices) <= 93:
            print("🟡 MODERATE EVIDENCE: Pickle may contain 93-channel data")
            print("   - Channel indices cycle through ≤93 values")
            print("   - Could represent recordings from 93 electrodes")
        else:
            print("🔴 WEAK EVIDENCE: Pickle unlikely to contain 93-channel data")
            print("   - More than 93 unique channel indices")
            print("   - Data likely represents sequential recordings, not channels")
        
        print(f"\n📊 Summary:")
        print(f"  - Unique channel IDs: {len(channel_id_groups)}")
        print(f"  - Channel index range: {min(unique_indices)} to {max(unique_indices)}")
        print(f"  - Total recordings: {len(data):,}")
        print(f"  - Recording types: {list(recording_types.keys())}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    check_pickle_channel_structure(pickle_path)
