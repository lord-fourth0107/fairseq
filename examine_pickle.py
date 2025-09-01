#!/usr/bin/env python3
import pickle
import numpy as np
import matplotlib.pyplot as plt

def examine_pickle(file_path):
    """Load and examine pickle file contents"""
    print(f"Loading pickle file: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"\nPickle file loaded successfully!")
        print(f"Data type: {type(data)}")
        print(f"Data length: {len(data) if hasattr(data, '__len__') else 'N/A'}")
        
        if isinstance(data, list):
            print(f"List length: {len(data)}")
            if len(data) > 0:
                print(f"First element type: {type(data[0])}")
                print(f"First element shape: {data[0].shape if hasattr(data[0], 'shape') else 'N/A'}")
                
                # Check if it's a list of tuples (signal, label)
                if isinstance(data[0], tuple) and len(data[0]) == 2:
                    signal, label = data[0]
                    print(f"Signal type: {type(signal)}")
                    print(f"Signal shape: {signal.shape if hasattr(signal, 'shape') else len(signal)}")
                    print(f"Label type: {type(label)}")
                    print(f"Label value: {label}")
                    
                    # If signal is large, plot it
                    if hasattr(signal, 'shape') and signal.shape[0] > 1000:
                        print(f"\nSignal is large ({signal.shape[0]} points), plotting...")
                        plt.figure(figsize=(12, 6))
                        plt.plot(signal[:10000])  # Plot first 10k points
                        plt.title(f"First 10,000 points of signal (total: {signal.shape[0]})")
                        plt.xlabel("Time points")
                        plt.ylabel("Amplitude")
                        plt.grid(True)
                        plt.tight_layout()
                        plt.savefig('signal_plot.png', dpi=150, bbox_inches='tight')
                        print("Plot saved as 'signal_plot.png'")
                        
                        # Also plot a sample of multiple recordings
                        if len(data) > 1:
                            plt.figure(figsize=(12, 8))
                            for i in range(min(5, len(data))):
                                signal_i = data[i][0]
                                plt.subplot(5, 1, i+1)
                                plt.plot(signal_i[:5000])  # First 5k points
                                plt.title(f"Recording {i+1} (first 5k points)")
                                plt.ylabel("Amplitude")
                            plt.xlabel("Time points")
                            plt.tight_layout()
                            plt.savefig('multiple_recordings.png', dpi=150, bbox_inches='tight')
                            print("Multiple recordings plot saved as 'multiple_recordings.png'")
                    
                else:
                    print(f"First element content: {data[0]}")
        
        elif isinstance(data, dict):
            print(f"Dictionary keys: {list(data.keys())}")
            for key, value in data.items():
                print(f"  {key}: {type(value)} - {value.shape if hasattr(value, 'shape') else len(value) if hasattr(value, '__len__') else 'N/A'}")
        
        elif hasattr(data, 'shape'):
            print(f"Array shape: {data.shape}")
            print(f"Array dtype: {data.dtype}")
            print(f"Array range: [{data.min():.6f}, {data.max():.6f}]")
            
            # If array is large, plot it
            if data.size > 10000:
                print(f"\nArray is large ({data.size} elements), plotting...")
                if len(data.shape) == 1:
                    plt.figure(figsize=(12, 6))
                    plt.plot(data[:10000])  # Plot first 10k points
                    plt.title(f"First 10,000 points (total: {data.shape[0]})")
                    plt.xlabel("Index")
                    plt.ylabel("Value")
                    plt.grid(True)
                    plt.tight_layout()
                    plt.savefig('array_plot.png', dpi=150, bbox_inches='tight')
                    print("Plot saved as 'array_plot.png'")
                elif len(data.shape) == 2:
                    plt.figure(figsize=(10, 8))
                    plt.imshow(data[:100, :100], aspect='auto', cmap='viridis')
                    plt.colorbar()
                    plt.title(f"Array heatmap (showing first 100x100 of {data.shape})")
                    plt.xlabel("Column")
                    plt.ylabel("Row")
                    plt.tight_layout()
                    plt.savefig('array_heatmap.png', dpi=150, bbox_inches='tight')
                    print("Heatmap saved as 'array_heatmap.png'")
        
        else:
            print(f"Data content: {data}")
            
    except Exception as e:
        print(f"Error loading pickle file: {e}")

def print_first_100_recordings(file_path):
    """Print detailed information for the first 100 recordings"""
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS OF FIRST 100 RECORDINGS")
    print(f"{'='*80}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            print(f"Total recordings available: {len(data)}")
            print(f"Showing details for first 100 recordings:\n")
            
            for i in range(min(100, len(data))):
                if isinstance(data[i], tuple) and len(data[i]) == 2:
                    signal, label = data[i]
                    
                    # Get signal statistics
                    if hasattr(signal, 'shape'):
                        signal_mean = np.mean(signal)
                        signal_std = np.std(signal)
                        signal_min = np.min(signal)
                        signal_max = np.max(signal)
                        
                        print(f"Recording {i+1:3d}:")
                        print(f"  Label: {label}")
                        print(f"  Signal shape: {signal.shape}")
                        print(f"  Signal stats: mean={signal_mean:8.6f}, std={signal_std:8.6f}, range=[{signal_min:8.6f}, {signal_max:8.6f}]")
                        
                        # Show first few values
                        print(f"  First 5 values: {signal[:5]}")
                        print(f"  Last 5 values: {signal[-5:]}")
                        print()
                    else:
                        print(f"Recording {i+1:3d}: {data[i]}")
                else:
                    print(f"Recording {i+1:3d}: {data[i]}")
                    
        else:
            print("Data is not in expected format (list of tuples)")
            
    except Exception as e:
        print(f"Error in detailed analysis: {e}")

def print_labels_only(file_path, num_recordings=150):
    """Print only the labels of the first N recordings"""
    print(f"\n{'='*80}")
    print(f"LABELS OF FIRST {num_recordings} RECORDINGS")
    print(f"{'='*80}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            print(f"Total recordings available: {len(data)}")
            print(f"Showing labels for first {num_recordings} recordings:\n")
            
            for i in range(min(num_recordings, len(data))):
                if isinstance(data[i], tuple) and len(data[i]) == 2:
                    signal, label = data[i]
                    print(f"Recording {i+1:3d}: {label}")
                else:
                    print(f"Recording {i+1:3d}: {data[i]}")
                    
        else:
            print("Data is not in expected format (list of tuples)")
            
    except Exception as e:
        print(f"Error in label printing: {e}")

def find_channel_index_resets(file_path, max_recordings=10000):
    """Find when the channel index resets, indicating different sessions"""
    print(f"\n{'='*80}")
    print("FINDING CHANNEL INDEX RESETS")
    print(f"{'='*80}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            print(f"Total recordings available: {len(data)}")
            print(f"Analyzing first {min(max_recordings, len(data))} recordings for channel index resets...\n")
            
            resets = []
            current_session = None
            current_channel = None
            session_start = 0
            
            for i in range(min(max_recordings, len(data))):
                if isinstance(data[i], tuple) and len(data[i]) == 2:
                    signal, label = data[i]
                    
                    # Parse the label to extract session and channel
                    try:
                        parts = label.split('_')
                        if len(parts) >= 4:
                            session_id = parts[0]
                            channel_idx = int(parts[1])
                            
                            # Check if this is a new session
                            if current_session is None:
                                current_session = session_id
                                current_channel = channel_idx
                                session_start = i
                                print(f"Session {session_id} starts at recording {i+1}")
                            
                            elif session_id != current_session:
                                # New session detected
                                print(f"Session {current_session} ends at recording {i} (channels 0-{current_channel})")
                                print(f"Session {session_id} starts at recording {i+1}")
                                resets.append({
                                    'recording_index': i,
                                    'old_session': current_session,
                                    'new_session': session_id,
                                    'old_session_channels': current_channel + 1,
                                    'old_session_start': session_start + 1,
                                    'old_session_end': i
                                })
                                current_session = session_id
                                current_channel = channel_idx
                                session_start = i
                            
                            # Check if channel index reset within same session
                            elif channel_idx < current_channel:
                                print(f"Channel index reset detected at recording {i+1}: {current_channel} -> {channel_idx}")
                                print(f"  Session: {session_id}")
                                print(f"  Previous range: 0-{current_channel}")
                                print(f"  New range starts at: {channel_idx}")
                                resets.append({
                                    'recording_index': i,
                                    'type': 'channel_reset',
                                    'session': session_id,
                                    'old_max_channel': current_channel,
                                    'old_session_end': i
                                })
                                current_channel = channel_idx
                            
                            # Update current channel if this one is higher
                            elif channel_idx > current_channel:
                                current_channel = channel_idx
                                
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing label '{label}' at recording {i+1}: {e}")
                        continue
            
            # Print summary of the current session
            if current_session is not None:
                print(f"\nCurrent session {current_session} has channels 0-{current_channel}")
                print(f"  Started at recording {session_start + 1}")
                print(f"  Current recording: {min(max_recordings, len(data))}")
            
            # Print summary of all resets
            if resets:
                print(f"\n{'='*60}")
                print("SUMMARY OF CHANNEL INDEX RESETS")
                print(f"{'='*60}")
                for i, reset in enumerate(resets):
                    if 'type' in reset and reset['type'] == 'channel_reset':
                        print(f"Reset {i+1}: Channel index reset at recording {reset['recording_index']+1}")
                        print(f"  Session: {reset['session']}")
                        print(f"  Channel range: {reset['old_max_channel']} -> {reset['new_channel']}")
                    else:
                        print(f"Reset {i+1}: Session change at recording {reset['recording_index']+1}")
                        print(f"  {reset['old_session']} -> {reset['new_session']}")
                        print(f"  {reset['old_session']}: recordings {reset['old_session_start']}-{reset['old_session_end']} (channels 0-{reset['old_session_channels']-1})")
                    print()
            else:
                print("\nNo channel index resets detected in the analyzed range.")
                
        else:
            print("Data is not in expected format (list of tuples)")
            
    except Exception as e:
        print(f"Error in channel reset analysis: {e}")

def count_unique_channel_ids(file_path, max_recordings=50000):
    """Count unique channel IDs when considering the third underscore value as channel ID"""
    print(f"\n{'='*80}")
    print("COUNTING UNIQUE CHANNEL IDs (850264150 format)")
    print(f"{'='*80}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            print(f"Total recordings available: {len(data)}")
            print(f"Analyzing first {min(max_recordings, len(data))} recordings for unique channel IDs...\n")
            
            unique_channel_ids = set()
            channel_id_counts = {}
            sample_labels = []
            
            for i in range(min(max_recordings, len(data))):
                if isinstance(data[i], tuple) and len(data[i]) == 2:
                    signal, label = data[i]
                    
                    try:
                        parts = label.split('_')
                        if len(parts) >= 4:
                            # Consider the third underscore value (index 3) as channel ID
                            channel_id = parts[3]  # This would be "850264150" in your example
                            
                            unique_channel_ids.add(channel_id)
                            
                            if channel_id in channel_id_counts:
                                channel_id_counts[channel_id] += 1
                            else:
                                channel_id_counts[channel_id] = 1
                                if len(sample_labels) < 10:  # Keep first 10 examples
                                    sample_labels.append(label)
                                    
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing label '{label}' at recording {i+1}: {e}")
                        continue
            
            print(f"Found {len(unique_channel_ids)} unique channel IDs")
            print(f"Sample labels with different channel IDs:")
            for label in sample_labels:
                print(f"  {label}")
            
            print(f"\nChannel ID distribution (showing first 20):")
            sorted_counts = sorted(channel_id_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (channel_id, count) in enumerate(sorted_counts[:20]):
                print(f"  Channel ID {channel_id}: {count} recordings")
            
            if len(sorted_counts) > 20:
                print(f"  ... and {len(sorted_counts) - 20} more channel IDs")
            
            # Check if this matches the previous analysis
            print(f"\nComparison with previous analysis:")
            print(f"  Previous: Found 2,856 channels based on index resets")
            print(f"  Current: Found {len(unique_channel_ids)} unique channel IDs")
            
            if len(unique_channel_ids) == 2856:
                print(f"  ✓ MATCH: Both analyses agree on 2,856 channels!")
            else:
                print(f"  ✗ MISMATCH: Different channel counts detected")
                
        else:
            print("Data is not in expected format (list of tuples)")
            
    except Exception as e:
        print(f"Error in channel ID counting: {e}")

def analyze_timestamps(file_path, max_recordings=10000):
    """Analyze if the second underscore value is a timestamp and check for 3-second differences"""
    print(f"\n{'='*80}")
    print("ANALYZING TIMESTAMPS (810755797 format)")
    print(f"{'='*80}")
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        
        if isinstance(data, list) and len(data) > 0:
            print(f"Total recordings available: {len(data)}")
            print(f"Analyzing first {min(max_recordings, len(data))} recordings for timestamp patterns...\n")
            
            timestamps = []
            timestamp_diffs = []
            sample_labels = []
            
            for i in range(min(max_recordings, len(data))):
                if isinstance(data[i], tuple) and len(data[i]) == 2:
                    signal, label = data[i]
                    
                    try:
                        parts = label.split('_')
                        if len(parts) >= 3:
                            # Consider the second underscore value (index 2) as timestamp
                            timestamp = int(parts[2])  # This would be "810755797" in your example
                            
                            timestamps.append(timestamp)
                            
                            if len(sample_labels) < 10:  # Keep first 10 examples
                                sample_labels.append(label)
                                
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing label '{label}' at recording {i+1}: {e}")
                        continue
            
            # Calculate timestamp differences
            if len(timestamps) > 1:
                for i in range(1, len(timestamps)):
                    diff = timestamps[i] - timestamps[i-1]
                    timestamp_diffs.append(diff)
                
                print(f"Found {len(timestamps)} timestamps")
                print(f"Sample labels with timestamps:")
                for label in sample_labels:
                    print(f"  {label}")
                
                print(f"\nTimestamp analysis:")
                print(f"  First timestamp: {timestamps[0]}")
                print(f"  Last timestamp: {timestamps[-1]}")
                print(f"  Total time span: {timestamps[-1] - timestamps[0]} units")
                
                print(f"\nTimestamp differences (consecutive recordings):")
                print(f"  Min difference: {min(timestamp_diffs)}")
                print(f"  Max difference: {max(timestamp_diffs)}")
                print(f"  Mean difference: {np.mean(timestamp_diffs):.2f}")
                print(f"  Median difference: {np.median(timestamp_diffs):.2f}")
                print(f"  Std difference: {np.std(timestamp_diffs):.2f}")
                
                # Check for 3-second pattern
                print(f"\nChecking for 3-second pattern:")
                expected_3sec = 3  # Assuming timestamp units are in seconds
                tolerance = 0.1  # 10% tolerance
                
                close_to_3sec = [diff for diff in timestamp_diffs if abs(diff - expected_3sec) <= tolerance]
                print(f"  Differences close to 3 seconds (±{tolerance}s): {len(close_to_3sec)} out of {len(timestamp_diffs)}")
                
                if len(close_to_3sec) > 0:
                    print(f"  Percentage close to 3s: {len(close_to_3sec)/len(timestamp_diffs)*100:.1f}%")
                
                # Show distribution of differences
                print(f"\nTimestamp difference distribution:")
                unique_diffs = sorted(set(timestamp_diffs))
                for diff in unique_diffs[:20]:  # Show first 20 unique differences
                    count = timestamp_diffs.count(diff)
                    print(f"  {diff}: {count} occurrences")
                
                if len(unique_diffs) > 20:
                    print(f"  ... and {len(unique_diffs) - 20} more unique differences")
                
                # Check if timestamps are monotonically increasing
                is_monotonic = all(timestamps[i] <= timestamps[i+1] for i in range(len(timestamps)-1))
                print(f"\nTimestamp monotonicity:")
                print(f"  Timestamps are monotonically increasing: {'✓ YES' if is_monotonic else '✗ NO'}")
                
                # If not monotonic, find where they decrease
                if not is_monotonic:
                    decreases = []
                    for i in range(1, len(timestamps)):
                        if timestamps[i] < timestamps[i-1]:
                            decreases.append((i, timestamps[i-1], timestamps[i]))
                    
                    print(f"  Found {len(decreases)} timestamp decreases:")
                    for i, old_ts, new_ts in decreases[:10]:  # Show first 10
                        print(f"    Recording {i+1}: {old_ts} -> {new_ts} (diff: {new_ts - old_ts})")
                    
                    if len(decreases) > 10:
                        print(f"    ... and {len(decreases) - 10} more decreases")
                
            else:
                print("Not enough timestamps to analyze differences")
                
        else:
            print("Data is not in expected format (list of tuples)")
            
    except Exception as e:
        print(f"Error in timestamp analysis: {e}")

if __name__ == "__main__":
    # Use the sample pickle file path you provided
    pickle_path = "/Users/uttamsingh/Downloads/715093703_810755797.pickle"
    examine_pickle(pickle_path)
    print_first_100_recordings(pickle_path)
    print_labels_only(pickle_path, 150)
    find_channel_index_resets(pickle_path, 10000)
    count_unique_channel_ids(pickle_path, 50000)
    analyze_timestamps(pickle_path, 10000)
