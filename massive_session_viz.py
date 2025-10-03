#!/usr/bin/env python3
"""
Massive Session Visualization Script
Processes ALL pickle files in a directory and creates comprehensive visualizations
with perfect cube rendering for CCF coordinates.

Features:
- Processes all pickle files in input directory
- Creates 2 massive visualizations: same color vs different colors per session
- Handles overlapping cubes (latest color wins)
- Generates comprehensive statistics
- Uses multiprocessing for large datasets
- Perfect cube visualization style
"""

import os
import sys
import pickle
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import multiprocessing as mp
from tqdm import tqdm
import glob
from collections import defaultdict
import time
import json

def load_coordinate_lookup(input_path):
    """Load and merge CSV files to create coordinate lookup."""
    print("Loading coordinate data...")
    
    # Find CSV files in input path
    joined_path = os.path.join(input_path, 'joined.csv')
    channels_path = os.path.join(input_path, 'channels.csv')
    
    if not os.path.exists(joined_path):
        raise FileNotFoundError(f"joined.csv not found at {joined_path}")
    if not os.path.exists(channels_path):
        raise FileNotFoundError(f"channels.csv not found at {channels_path}")
    
    # Load CSV files
    joined_df = pd.read_csv(joined_path)
    channels_df = pd.read_csv(channels_path)
    
    print(f"Loaded joined.csv: {joined_df.shape}")
    print(f"Loaded channels.csv: {channels_df.shape}")
    
    # Merge on probe_id
    merged_df = pd.merge(
        joined_df,
        channels_df,
        left_on='probe_id',
        right_on='ecephys_probe_id',
        how='inner'
    )
    
    print(f"Merged data shape: {merged_df.shape}")
    
    # Create coordinate lookup - use channel-specific CCF coordinates from channels.csv
    coord_lookup = {}
    for _, row in merged_df.iterrows():
        key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
        coord_lookup[key] = {
            'ap': row['anterior_posterior_ccf_coordinate_y'],  # From channels.csv
            'dv': row['dorsal_ventral_ccf_coordinate_y'],      # From channels.csv
            'lr': row['left_right_ccf_coordinate_y'],          # From channels.csv
            'probe_h': row['probe_horizontal_position'],
            'probe_v': row['probe_vertical_position'],
            'structure': row['ecephys_structure_acronym_y'] if 'ecephys_structure_acronym_y' in row else 'Unknown'
        }
    
    print(f"Created coordinate lookup with {len(coord_lookup)} entries")
    return coord_lookup

def extract_session_id_from_filename(filename):
    """Extract session ID from pickle filename."""
    basename = os.path.basename(filename)
    if '_' in basename:
        return basename.split('_')[0]
    return None

def process_single_pickle_file(args):
    """Process a single pickle file and extract channel coordinates."""
    pickle_path, coord_lookup, voxel_size = args
    
    try:
        # Extract session ID from filename
        session_id = extract_session_id_from_filename(pickle_path)
        if not session_id:
            print(f"Warning: Could not extract session ID from {pickle_path}")
            return None
        
        print(f"Processing file: {pickle_path}, Session ID: {session_id}")
        
        # Load pickle file
        with open(pickle_path, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Loaded pickle file with {len(data['labels'])} labels")
        
        # Extract coordinates
        coordinates = []
        probe_ids = set()
        
        for entry in data['labels']:
            try:
                # Parse the label string
                parts = entry.split('_')
                if len(parts) >= 3:
                    session_id_from_label = parts[0]
                    probe_id = parts[1]
                    channel_id = parts[2]
                    
                    # Verify session ID matches filename
                    if session_id_from_label != session_id:
                        continue
                    
                    # Look up coordinates
                    key = (session_id_from_label, probe_id, channel_id)
                    if key in coord_lookup:
                        coords = coord_lookup[key]
                        coordinates.append({
                            'ap': coords['ap'],
                            'dv': coords['dv'],
                            'lr': coords['lr'],
                            'probe_id': probe_id,
                            'channel_id': channel_id,
                            'session_id': session_id_from_label
                        })
                        probe_ids.add(probe_id)
            except Exception as e:
                continue
        
        print(f"Found {len(coordinates)} coordinates for {len(probe_ids)} probes")
        
        return {
            'session_id': session_id,
            'probe_ids': list(probe_ids),
            'coordinates': coordinates,
            'file_path': pickle_path,
            'total_channels': len(coordinates)
        }
        
    except Exception as e:
        print(f"Error processing {pickle_path}: {e}")
        return None

def process_all_pickle_files(pickle_dir, coord_lookup, voxel_size, num_workers=8):
    """Process all pickle files in parallel."""
    print(f"Scanning for pickle files in {pickle_dir}...")
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(pickle_dir, "*.pickle"))
    print(f"Found {len(pickle_files)} pickle files")
    
    if not pickle_files:
        print("No pickle files found!")
        return []
    
    # Prepare arguments for multiprocessing
    args_list = [(pf, coord_lookup, voxel_size) for pf in pickle_files]
    
    # Process files in parallel
    print(f"Processing {len(pickle_files)} files with {num_workers} workers...")
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(process_single_pickle_file, args_list), 
                          total=len(pickle_files), desc="Processing files"))
    
    # Filter out None results
    valid_results = [r for r in results if r is not None]
    print(f"Successfully processed {len(valid_results)} files")
    
    return valid_results

def create_voxel_grid(coordinates, voxel_size):
    """Create voxel grid and assign coordinates to voxels."""
    if not coordinates:
        return {}, {}
    
    # Convert to numpy arrays
    ap_coords = np.array([c['ap'] for c in coordinates])
    dv_coords = np.array([c['dv'] for c in coordinates])
    lr_coords = np.array([c['lr'] for c in coordinates])
    
    # Calculate voxel grid dimensions
    ap_min, ap_max = np.min(ap_coords), np.max(ap_coords)
    dv_min, dv_max = np.min(dv_coords), np.max(dv_coords)
    lr_min, lr_max = np.min(lr_coords), np.max(lr_coords)
    
    # Create voxel grid
    ap_voxels = np.arange(ap_min, ap_max + voxel_size, voxel_size)
    dv_voxels = np.arange(dv_min, dv_max + voxel_size, voxel_size)
    lr_voxels = np.arange(lr_min, lr_max + voxel_size, voxel_size)
    
    # Assign coordinates to voxels
    voxel_to_coords = defaultdict(list)
    voxel_to_session = {}
    
    for coord in coordinates:
        ap_idx = int((coord['ap'] - ap_min) / voxel_size)
        dv_idx = int((coord['dv'] - dv_min) / voxel_size)
        lr_idx = int((coord['lr'] - lr_min) / voxel_size)
        
        voxel_key = (ap_idx, dv_idx, lr_idx)
        voxel_to_coords[voxel_key].append(coord)
        voxel_to_session[voxel_key] = coord['session_id']  # Latest session wins
    
    return voxel_to_coords, voxel_to_session

def create_perfect_cube(center, size, color, alpha=0.7):
    """Create a perfect cube for visualization."""
    x, y, z = center
    half_size = size / 2
    
    # Define the 8 vertices of the cube
    vertices = np.array([
        [x - half_size, y - half_size, z - half_size],
        [x + half_size, y - half_size, z - half_size],
        [x + half_size, y + half_size, z - half_size],
        [x - half_size, y + half_size, z - half_size],
        [x - half_size, y - half_size, z + half_size],
        [x + half_size, y - half_size, z + half_size],
        [x + half_size, y + half_size, z + half_size],
        [x - half_size, y + half_size, z + half_size]
    ])
    
    # Define the 6 faces of the cube
    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],  # bottom
        [vertices[4], vertices[5], vertices[6], vertices[7]],  # top
        [vertices[0], vertices[1], vertices[5], vertices[4]],  # front
        [vertices[2], vertices[3], vertices[7], vertices[6]],  # back
        [vertices[0], vertices[3], vertices[7], vertices[4]],  # left
        [vertices[1], vertices[2], vertices[6], vertices[5]]   # right
    ]
    
    return Poly3DCollection(faces, facecolor=color, alpha=alpha, edgecolor='black', linewidth=0.1)

def create_visualization(voxel_to_coords, voxel_to_session, voxel_size, output_path, 
                        title, use_different_colors=True):
    """Create 3D visualization with perfect cubes."""
    print(f"Creating visualization: {title}")
    
    # Get unique sessions for color mapping
    unique_sessions = list(set(voxel_to_session.values()))
    n_sessions = len(unique_sessions)
    
    # Create color map
    if use_different_colors and n_sessions > 1:
        colors = plt.cm.tab20(np.linspace(0, 1, n_sessions))
        session_to_color = {session: colors[i] for i, session in enumerate(unique_sessions)}
    else:
        # Use single color
        session_to_color = {session: 'red' for session in unique_sessions}
    
    # Create figure
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create cubes
    cubes = []
    for voxel_key, coords in voxel_to_coords.items():
        if not coords:
            continue
        
        # Calculate voxel center
        ap_idx, dv_idx, lr_idx = voxel_key
        center = (
            ap_idx * voxel_size + voxel_size / 2,
            dv_idx * voxel_size + voxel_size / 2,
            lr_idx * voxel_size + voxel_size / 2
        )
        
        # Get color for this voxel
        session = voxel_to_session[voxel_key]
        color = session_to_color[session]
        
        # Create cube
        cube = create_perfect_cube(center, voxel_size, color)
        cubes.append(cube)
        ax.add_collection3d(cube)
    
    # Set axis labels and limits
    ax.set_xlabel('Anterior-Posterior (μm)')
    ax.set_ylabel('Dorsal-Ventral (μm)')
    ax.set_zlabel('Left-Right (μm)')
    ax.set_title(f'{title}\n{len(cubes)} voxels, {len(unique_sessions)} sessions')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization to {output_path}")

def generate_statistics(all_results, voxel_to_coords, voxel_to_session):
    """Generate comprehensive statistics."""
    stats = {
        'total_files_processed': len(all_results),
        'total_sessions': len(set(r['session_id'] for r in all_results)),
        'total_probes': len(set(probe for r in all_results for probe in r['probe_ids'])),
        'total_channels': sum(r['total_channels'] for r in all_results),
        'total_voxels': len(voxel_to_coords),
        'voxels_with_channels': len([v for v in voxel_to_coords.values() if v]),
        'session_breakdown': {},
        'probe_breakdown': {},
        'voxel_density_stats': {}
    }
    
    # Session breakdown
    for result in all_results:
        session_id = result['session_id']
        if session_id not in stats['session_breakdown']:
            stats['session_breakdown'][session_id] = {
                'files': 0,
                'probes': set(),
                'channels': 0
            }
        stats['session_breakdown'][session_id]['files'] += 1
        stats['session_breakdown'][session_id]['probes'].update(result['probe_ids'])
        stats['session_breakdown'][session_id]['channels'] += result['total_channels']
    
    # Convert sets to counts
    for session_data in stats['session_breakdown'].values():
        session_data['probes'] = len(session_data['probes'])
    
    # Probe breakdown
    for result in all_results:
        for probe_id in result['probe_ids']:
            if probe_id not in stats['probe_breakdown']:
                stats['probe_breakdown'][probe_id] = {
                    'sessions': set(),
                    'channels': 0
                }
            stats['probe_breakdown'][probe_id]['sessions'].add(result['session_id'])
            stats['probe_breakdown'][probe_id]['channels'] += result['total_channels']
    
    # Convert sets to counts
    for probe_data in stats['probe_breakdown'].values():
        probe_data['sessions'] = len(probe_data['sessions'])
    
    # Voxel density stats
    channel_counts = [len(coords) for coords in voxel_to_coords.values() if coords]
    if channel_counts:
        stats['voxel_density_stats'] = {
            'min_channels_per_voxel': min(channel_counts),
            'max_channels_per_voxel': max(channel_counts),
            'mean_channels_per_voxel': np.mean(channel_counts),
            'median_channels_per_voxel': np.median(channel_counts),
            'std_channels_per_voxel': np.std(channel_counts)
        }
    
    return stats

def save_statistics(stats, output_dir):
    """Save statistics to JSON file."""
    stats_path = os.path.join(output_dir, 'massive_viz_statistics.json')
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    # Recursively convert numpy types
    def recursive_convert(obj):
        if isinstance(obj, dict):
            return {k: recursive_convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [recursive_convert(item) for item in obj]
        else:
            return convert_numpy(obj)
    
    converted_stats = recursive_convert(stats)
    
    with open(stats_path, 'w') as f:
        json.dump(converted_stats, f, indent=2)
    
    print(f"Statistics saved to {stats_path}")

def print_statistics_summary(stats):
    """Print a summary of statistics."""
    print("\n" + "="*80)
    print("MASSIVE VISUALIZATION STATISTICS SUMMARY")
    print("="*80)
    print(f"Total files processed: {stats['total_files_processed']}")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Total probes: {stats['total_probes']}")
    print(f"Total channels: {stats['total_channels']}")
    print(f"Total voxels: {stats['total_voxels']}")
    print(f"Voxels with channels: {stats['voxels_with_channels']}")
    
    if stats['voxel_density_stats']:
        print(f"\nVoxel Density Statistics:")
        print(f"  Min channels per voxel: {stats['voxel_density_stats']['min_channels_per_voxel']}")
        print(f"  Max channels per voxel: {stats['voxel_density_stats']['max_channels_per_voxel']}")
        print(f"  Mean channels per voxel: {stats['voxel_density_stats']['mean_channels_per_voxel']:.2f}")
        print(f"  Median channels per voxel: {stats['voxel_density_stats']['median_channels_per_voxel']:.2f}")
        print(f"  Std channels per voxel: {stats['voxel_density_stats']['std_channels_per_voxel']:.2f}")
    
    print(f"\nTop 10 Sessions by Channel Count:")
    session_channels = [(sid, data['channels']) for sid, data in stats['session_breakdown'].items()]
    session_channels.sort(key=lambda x: x[1], reverse=True)
    for i, (session_id, channels) in enumerate(session_channels[:10]):
        print(f"  {i+1}. Session {session_id}: {channels} channels")
    
    print(f"\nTop 10 Probes by Channel Count:")
    probe_channels = [(pid, data['channels']) for pid, data in stats['probe_breakdown'].items()]
    probe_channels.sort(key=lambda x: x[1], reverse=True)
    for i, (probe_id, channels) in enumerate(probe_channels[:10]):
        print(f"  {i+1}. Probe {probe_id}: {channels} channels")
    
    print("="*80)

def main():
    parser = argparse.ArgumentParser(description='Massive Session Visualization')
    parser.add_argument('pickle_dir', help='Directory containing pickle files')
    parser.add_argument('csv_dir', help='Directory containing CSV files')
    parser.add_argument('--output-dir', default='massive_session_viz', 
                       help='Output directory for visualizations')
    parser.add_argument('--voxel-size', type=float, default=1.0, 
                       help='Voxel size in mm (default: 1.0)')
    parser.add_argument('--workers', type=int, default=8, 
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--joined-csv', default='joined.csv', 
                       help='Joined CSV filename (default: joined.csv)')
    parser.add_argument('--channels-csv', default='channels.csv', 
                       help='Channels CSV filename (default: channels.csv)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load coordinate data
    joined_csv_path = os.path.join(args.csv_dir, args.joined_csv)
    channels_csv_path = os.path.join(args.csv_dir, args.channels_csv)
    
    coord_lookup = load_coordinate_data(joined_csv_path, channels_csv_path)
    
    # Process all pickle files
    print(f"\nProcessing all pickle files in {args.pickle_dir}...")
    all_results = process_all_pickle_files(args.pickle_dir, coord_lookup, 
                                        args.voxel_size, args.workers)
    
    if not all_results:
        print("No valid results found!")
        return
    
    # Combine all coordinates
    print("\nCombining all coordinates...")
    all_coordinates = []
    for result in all_results:
        all_coordinates.extend(result['coordinates'])
    
    print(f"Total coordinates to process: {len(all_coordinates)}")
    
    # Create voxel grid
    print("Creating voxel grid...")
    voxel_to_coords, voxel_to_session = create_voxel_grid(all_coordinates, args.voxel_size)
    
    # Generate statistics
    print("Generating statistics...")
    stats = generate_statistics(all_results, voxel_to_coords, voxel_to_session)
    print_statistics_summary(stats)
    save_statistics(stats, args.output_dir)
    
    # Create visualizations
    print("\nCreating visualizations...")
    
    # Same color visualization
    same_color_path = os.path.join(args.output_dir, 'massive_viz_same_color.png')
    create_visualization(voxel_to_coords, voxel_to_session, args.voxel_size, 
                        same_color_path, 'Massive Visualization - Same Color', 
                        use_different_colors=False)
    
    # Different colors per session visualization
    diff_color_path = os.path.join(args.output_dir, 'massive_viz_different_colors.png')
    create_visualization(voxel_to_coords, voxel_to_session, args.voxel_size, 
                        diff_color_path, 'Massive Visualization - Different Colors Per Session', 
                        use_different_colors=True)
    
    print(f"\nVisualization complete! Output saved to {args.output_dir}")
    print(f"Files created:")
    print(f"  - {same_color_path}")
    print(f"  - {diff_color_path}")
    print(f"  - {os.path.join(args.output_dir, 'massive_viz_statistics.json')}")

if __name__ == '__main__':
    main()
