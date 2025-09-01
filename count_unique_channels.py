#!/usr/bin/env python3
import pickle
from collections import defaultdict

def count_unique_channels_per_probe(file_path):
    """Count unique channels per probe in the pickle file"""
    print(f"🔍 Counting unique channels per probe: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded {len(data):,} recordings")
        
        # Parse labels and organize by probe
        probe_channels = defaultdict(set)
        probe_lfp_channels = defaultdict(set)
        
        for i, (signal, label) in enumerate(data):
            parts = label.split('_')
            if len(parts) == 5:
                session, count, probe, lfp_channel_index, brain_region = parts
                
                # Count field (channel index)
                channel_index = int(count)
                probe_channels[probe].add(channel_index)
                
                # LFP channel index (physical electrode)
                probe_lfp_channels[probe].add(lfp_channel_index)
        
        print(f"\n📊 UNIQUE CHANNELS PER PROBE:")
        print("=" * 60)
        
        total_unique_channels = 0
        for probe in sorted(probe_channels.keys()):
            channel_indices = probe_channels[probe]
            lfp_channels = probe_lfp_channels[probe]
            
            print(f"\n🔬 Probe {probe}:")
            print(f"  📈 Total recordings: {len([r for r in data if r[1].split('_')[2] == probe]):,}")
            print(f"  🔢 Unique channel indices (count field): {len(channel_indices)}")
            print(f"  📍 Unique LFP channel indices: {len(lfp_channels)}")
            print(f"  🎯 Channel index range: {min(channel_indices)} to {max(channel_indices)}")
            print(f"  📍 LFP channel indices: {sorted(lfp_channels)}")
            
            total_unique_channels += len(channel_indices)
        
        print(f"\n🎯 SUMMARY:")
        print("=" * 40)
        print(f"• Total probes: {len(probe_channels)}")
        print(f"• Total unique channel indices: {total_unique_channels}")
        
        # Check if all probes have the same number of channels
        channel_counts = [len(channels) for channels in probe_channels.values()]
        if len(set(channel_counts)) == 1:
            print(f"• All probes have same number of channels: {channel_counts[0]}")
        else:
            print(f"• Different probes have different channel counts: {channel_counts}")
        
        # Show the actual channel indices for each probe
        print(f"\n📋 DETAILED CHANNEL INDICES:")
        print("=" * 50)
        for probe in sorted(probe_channels.keys()):
            channel_indices = sorted(probe_channels[probe])
            print(f"\nProbe {probe} - Channel indices:")
            print(f"  Count: {len(channel_indices)}")
            print(f"  Range: {min(channel_indices)} to {max(channel_indices)}")
            print(f"  First 10: {channel_indices[:10]}")
            print(f"  Last 10: {channel_indices[-10:]}")
            
            # Check if sequential
            is_sequential = all(channel_indices[i] == channel_indices[i-1] + 1 
                             for i in range(1, len(channel_indices)))
            print(f"  Sequential: {'Yes' if is_sequential else 'No'}")
        
        return probe_channels, probe_lfp_channels
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    probe_channels, probe_lfp_channels = count_unique_channels_per_probe(pickle_path)
