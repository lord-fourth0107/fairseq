#!/usr/bin/env python3
import pickle
from collections import defaultdict, Counter

def analyze_channel_patterns(file_path):
    """Analyze patterns in LFP channel IDs"""
    print(f"🔍 Analyzing LFP channel ID patterns: {file_path}")
    print("=" * 80)
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"✓ Loaded {len(data):,} recordings")
        
        # Parse labels and organize by channel index (count field)
        channel_index_to_lfp = defaultdict(list)
        lfp_channel_sequence = []
        
        for i, (signal, label) in enumerate(data):
            parts = label.split('_')
            if len(parts) == 5:
                session, count, probe, lfp_channel_index, brain_region = parts
                channel_index = int(count)
                channel_index_to_lfp[channel_index].append(lfp_channel_index)
                lfp_channel_sequence.append(lfp_channel_index)
        
        print(f"✓ Analyzed {len(lfp_channel_sequence)} LFP channel IDs")
        
        # Check if LFP channel IDs repeat in cycles
        print(f"\n🔄 PATTERN ANALYSIS:")
        print("=" * 50)
        
        # Get unique LFP channel IDs
        unique_lfp_channels = list(set(lfp_channel_sequence))
        print(f"• Total unique LFP channel IDs: {len(unique_lfp_channels)}")
        print(f"• First 10 unique LFP channels: {unique_lfp_channels[:10]}")
        print(f"• Last 10 unique LFP channels: {unique_lfp_channels[-10:]}")
        
        # Check if the sequence repeats every 93 elements
        print(f"\n🔍 CHECKING FOR 93-CHANNEL CYCLES:")
        print("=" * 50)
        
        # Look for repeating patterns
        cycle_length = 93
        if len(lfp_channel_sequence) >= cycle_length * 2:
            # Check first cycle
            first_cycle = lfp_channel_sequence[:cycle_length]
            print(f"• First cycle (channels 0-92): {first_cycle}")
            
            # Check if second cycle matches first cycle
            if len(lfp_channel_sequence) >= cycle_length * 2:
                second_cycle = lfp_channel_sequence[cycle_length:cycle_length*2]
                print(f"• Second cycle (channels 93-185): {second_cycle}")
                
                if first_cycle == second_cycle:
                    print(f"✅ PATTERN FOUND: LFP channels repeat every 93 recordings!")
                else:
                    print(f"❌ No exact repetition found in first two cycles")
                    
                    # Check for partial matches
                    matches = sum(1 for a, b in zip(first_cycle, second_cycle) if a == b)
                    print(f"• Partial matches: {matches}/{cycle_length} ({matches/cycle_length*100:.1f}%)")
        
        # Analyze the full sequence for patterns
        print(f"\n📊 FULL SEQUENCE ANALYSIS:")
        print("=" * 50)
        
        # Count occurrences of each LFP channel ID
        lfp_counts = Counter(lfp_channel_sequence)
        print(f"• LFP channel ID frequency distribution:")
        
        # Show frequency of each LFP channel
        for lfp_id, count in sorted(lfp_counts.items()):
            print(f"  {lfp_id}: {count} occurrences")
        
        # Check if all LFP channels appear the same number of times
        count_values = list(lfp_counts.values())
        if len(set(count_values)) == 1:
            print(f"✅ All LFP channels appear exactly {count_values[0]} times")
        else:
            print(f"❌ LFP channels appear different numbers of times")
            print(f"  Min occurrences: {min(count_values)}")
            print(f"  Max occurrences: {max(count_values)}")
            print(f"  Unique occurrence counts: {sorted(set(count_values))}")
        
        # Check for cycling pattern in the sequence
        print(f"\n🔄 CYCLING PATTERN ANALYSIS:")
        print("=" * 50)
        
        # Look for the pattern: does the sequence cycle through the 93 LFP channels?
        expected_cycle_length = len(unique_lfp_channels)
        print(f"• Expected cycle length: {expected_cycle_length}")
        
        # Check if the sequence cycles through all unique LFP channels
        if len(lfp_channel_sequence) >= expected_cycle_length:
            first_cycle = lfp_channel_sequence[:expected_cycle_length]
            unique_in_first_cycle = set(first_cycle)
            all_unique = set(unique_lfp_channels)
            
            if unique_in_first_cycle == all_unique:
                print(f"✅ First {expected_cycle_length} recordings contain all unique LFP channels")
                
                # Check if this pattern repeats
                if len(lfp_channel_sequence) >= expected_cycle_length * 2:
                    second_cycle = lfp_channel_sequence[expected_cycle_length:expected_cycle_length*2]
                    if set(second_cycle) == all_unique:
                        print(f"✅ Second cycle also contains all unique LFP channels")
                        print(f"✅ CONFIRMED: LFP channels cycle every {expected_cycle_length} recordings")
                    else:
                        print(f"❌ Second cycle doesn't contain all unique LFP channels")
            else:
                print(f"❌ First {expected_cycle_length} recordings don't contain all unique LFP channels")
                print(f"  Missing: {all_unique - unique_in_first_cycle}")
        
        # Show the actual sequence pattern
        print(f"\n📋 SEQUENCE PATTERN:")
        print("=" * 50)
        print(f"• Total recordings: {len(lfp_channel_sequence)}")
        print(f"• Unique LFP channels: {len(unique_lfp_channels)}")
        print(f"• Recordings per LFP channel: {len(lfp_channel_sequence) // len(unique_lfp_channels)}")
        
        # Show first few and last few in sequence
        print(f"• First 20 LFP channels in sequence: {lfp_channel_sequence[:20]}")
        print(f"• Last 20 LFP channels in sequence: {lfp_channel_sequence[-20:]}")
        
        # Check if there's a regular pattern
        if len(lfp_channel_sequence) % len(unique_lfp_channels) == 0:
            cycles = len(lfp_channel_sequence) // len(unique_lfp_channels)
            print(f"✅ Perfect cycling: {cycles} complete cycles of {len(unique_lfp_channels)} LFP channels")
        else:
            remainder = len(lfp_channel_sequence) % len(unique_lfp_channels)
            cycles = len(lfp_channel_sequence) // len(unique_lfp_channels)
            print(f"⚠️  Partial cycling: {cycles} complete cycles + {remainder} extra recordings")
        
        return lfp_channel_sequence, unique_lfp_channels, lfp_counts
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    sequence, unique_channels, counts = analyze_channel_patterns(pickle_path)
