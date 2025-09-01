#!/usr/bin/env python3
"""
Check the number of channels across different sessions to see if they vary
"""

import os
import pickle
import glob
from collections import defaultdict

def check_channels_per_session(data_path, max_sessions=10):
    """Check the number of unique channels per session"""
    print(f"🔍 Checking channels per session in: {data_path}")
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(data_path, "*.pickle"))
    print(f"📁 Found {len(pickle_files)} pickle files")
    
    if not pickle_files:
        print("❌ No pickle files found!")
        return
    
    session_channels = {}
    
    for i, pickle_file in enumerate(pickle_files[:max_sessions]):
        try:
            print(f"\n📊 Analyzing session {i+1}/{min(len(pickle_files), max_sessions)}: {os.path.basename(pickle_file)}")
            
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            print(f"   📈 Total recordings: {len(data):,}")
            
            # Parse labels to get channel information
            unique_lfp_channels = set()
            unique_probes = set()
            channel_counts = defaultdict(int)
            
            for signal, label in data[:1000]:  # Sample first 1000 for speed
                parts = label.split('_')
                if len(parts) == 5:
                    session, count, probe, lfp_channel_index, brain_region = parts
                    unique_lfp_channels.add(lfp_channel_index)
                    unique_probes.add(probe)
                    channel_counts[lfp_channel_index] += 1
            
            session_channels[os.path.basename(pickle_file)] = {
                'unique_lfp_channels': len(unique_lfp_channels),
                'unique_probes': len(unique_probes),
                'sample_channels': sorted(list(unique_lfp_channels))[:10],  # First 10 channels
                'total_recordings': len(data)
            }
            
            print(f"   🔬 Unique LFP channels: {len(unique_lfp_channels)}")
            print(f"   🔬 Unique probes: {len(unique_probes)}")
            print(f"   📋 Sample channels: {sorted(list(unique_lfp_channels))[:10]}")
            
        except Exception as e:
            print(f"   ❌ Error processing {pickle_file}: {e}")
    
    # Summary
    print(f"\n📊 SUMMARY:")
    print(f"{'Session':<30} {'LFP Channels':<12} {'Probes':<8} {'Total Recordings':<15}")
    print("-" * 70)
    
    channel_counts = []
    for session, info in session_channels.items():
        print(f"{session:<30} {info['unique_lfp_channels']:<12} {info['unique_probes']:<8} {info['total_recordings']:<15}")
        channel_counts.append(info['unique_lfp_channels'])
    
    # Check if all sessions have the same number of channels
    unique_channel_counts = set(channel_counts)
    print(f"\n🔍 Analysis:")
    print(f"   • Unique channel counts found: {sorted(unique_channel_counts)}")
    print(f"   • Number of different channel counts: {len(unique_channel_counts)}")
    
    if len(unique_channel_counts) == 1:
        print(f"   ✅ All sessions have the same number of channels: {list(unique_channel_counts)[0]}")
    else:
        print(f"   ⚠️ Sessions have different numbers of channels!")
        print(f"   📊 Channel count distribution:")
        for count in sorted(unique_channel_counts):
            sessions_with_count = [s for s, info in session_channels.items() if info['unique_lfp_channels'] == count]
            print(f"      {count} channels: {len(sessions_with_count)} sessions")
    
    return session_channels

def check_specific_sessions(data_path, session_names):
    """Check specific sessions by name"""
    print(f"\n🎯 Checking specific sessions: {session_names}")
    
    session_channels = {}
    
    for session_name in session_names:
        # Look for pickle files containing this session name
        pattern = os.path.join(data_path, f"*{session_name}*.pickle")
        matching_files = glob.glob(pattern)
        
        if not matching_files:
            print(f"   ❌ No files found for session: {session_name}")
            continue
            
        for pickle_file in matching_files:
            try:
                print(f"\n📊 Analyzing: {os.path.basename(pickle_file)}")
                
                with open(pickle_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Parse all labels to get complete channel information
                unique_lfp_channels = set()
                unique_probes = set()
                
                for signal, label in data:
                    parts = label.split('_')
                    if len(parts) == 5:
                        session, count, probe, lfp_channel_index, brain_region = parts
                        unique_lfp_channels.add(lfp_channel_index)
                        unique_probes.add(probe)
                
                session_channels[os.path.basename(pickle_file)] = {
                    'unique_lfp_channels': len(unique_lfp_channels),
                    'unique_probes': len(unique_probes),
                    'all_channels': sorted(list(unique_lfp_channels)),
                    'total_recordings': len(data)
                }
                
                print(f"   🔬 Unique LFP channels: {len(unique_lfp_channels)}")
                print(f"   🔬 Unique probes: {len(unique_probes)}")
                print(f"   📋 All channels: {sorted(list(unique_lfp_channels))}")
                
            except Exception as e:
                print(f"   ❌ Error processing {pickle_file}: {e}")
    
    return session_channels

if __name__ == "__main__":
    # Check the specific pickle file we've been working with
    test_pickle = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    
    print("🔍 Checking the specific pickle file we've been working with...")
    
    if os.path.exists(test_pickle):
        print(f"📁 Found test pickle: {test_pickle}")
        
        try:
            with open(test_pickle, 'rb') as f:
                data = pickle.load(f)
            
            print(f"📈 Total recordings: {len(data):,}")
            
            # Parse labels to get channel information
            unique_lfp_channels = set()
            unique_probes = set()
            session_info = defaultdict(lambda: {'channels': set(), 'probes': set(), 'count': 0})
            
            for signal, label in data:
                parts = label.split('_')
                if len(parts) == 5:
                    session, count, probe, lfp_channel_index, brain_region = parts
                    unique_lfp_channels.add(lfp_channel_index)
                    unique_probes.add(probe)
                    session_info[session]['channels'].add(lfp_channel_index)
                    session_info[session]['probes'].add(probe)
                    session_info[session]['count'] += 1
            
            print(f"🔬 Unique LFP channels: {len(unique_lfp_channels)}")
            print(f"🔬 Unique probes: {len(unique_probes)}")
            print(f"📋 All channels: {sorted(list(unique_lfp_channels))}")
            
            # Check if this single file contains multiple sessions
            print(f"\n📊 Session breakdown in this file:")
            print(f"{'Session':<15} {'LFP Channels':<12} {'Probes':<8} {'Recordings':<10}")
            print("-" * 50)
            
            for session, info in session_info.items():
                print(f"{session:<15} {len(info['channels']):<12} {len(info['probes']):<8} {info['count']:<10}")
            
            # Check if all sessions in this file have the same number of channels
            channel_counts = [len(info['channels']) for info in session_info.values()]
            unique_channel_counts = set(channel_counts)
            
            print(f"\n🔍 Analysis:")
            print(f"   • Unique channel counts in this file: {sorted(unique_channel_counts)}")
            
            if len(unique_channel_counts) == 1:
                print(f"   ✅ All sessions in this file have the same number of channels: {list(unique_channel_counts)[0]}")
            else:
                print(f"   ⚠️ Sessions in this file have different numbers of channels!")
                print(f"   📊 Channel count distribution:")
                for count in sorted(unique_channel_counts):
                    sessions_with_count = [s for s, info in session_info.items() if len(info['channels']) == count]
                    print(f"      {count} channels: {sessions_with_count}")
            
        except Exception as e:
            print(f"❌ Error processing {test_pickle}: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"❌ Test pickle file not found: {test_pickle}")
        print("Please update the path to your actual pickle file.")
    
    # Also check if you want to analyze a different data path
    print(f"\n💡 To check a different data path, modify the 'data_path' variable in the script.")
    print(f"   Example: data_path = '/path/to/your/pickle/files'")
