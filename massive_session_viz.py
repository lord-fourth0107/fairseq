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

def extract_coordinates_from_label(label):
    """Extract CCF coordinates from enriched label."""
    if not isinstance(label, str):
        return None
    
    parts = label.split('_')
    if len(parts) < 9:
        return None
    
    try:
        # Last 5 parts should be coordinates: ap, dv, lr, probe_h, probe_v
        ap = float(parts[-5])
        dv = float(parts[-4])
        lr = float(parts[-3])
        probe_h = float(parts[-2])
        probe_v = float(parts[-1])
        return {'ap': ap, 'dv': dv, 'lr': lr, 'probe_h': probe_h, 'probe_v': probe_v}
    except (ValueError, IndexError):
        return None

def extract_session_id_from_filename(filename):
    """Extract session ID from pickle filename."""
    basename = os.path.basename(filename)
    if '_' in basename:
        return basename.split('_')[0]
    return None

def process_single_pickle_file(args):
    """Process a single pickle file and extract channel coordinates."""
    pickle_path, voxel_size = args  # Removed coord_lookup parameter
    
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
        
        print(f"Loaded pickle file with {len(data)} entries")
        
        # Extract coordinates directly from enriched labels
        coordinates = []
        probe_ids = set()
        
        for entry in data:
            try:
                # Each entry is a tuple: (signal_array, enriched_label)
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    signal, label = entry[0], entry[1]
                    
                    # Extract coordinates directly from enriched label
                    coords = extract_coordinates_from_label(label)
                    if coords is not None:
                        # Parse the label to get session, probe, channel info
                        parts = label.split('_')
                        if len(parts) >= 3:
                            session_id_from_label = parts[0]
                            channel_id = parts[1]
                            probe_id = parts[2]
                            
                            # Verify session ID matches filename
                            if session_id_from_label == session_id:
                                coordinates.append({
                                    'ap': coords['ap'],
                                    'dv': coords['dv'],
                                    'lr': coords['lr'],
                                    'probe_id': probe_id,
                                    'channel_id': channel_id,
                                    'session_id': session_id_from_label,
                                    'probe_h': coords['probe_h'],
                                    'probe_v': coords['probe_v']
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

def process_all_pickle_files(pickle_dir, voxel_size, num_workers=8, batch_size=None):
    """Process all pickle files in parallel."""
    print(f"Scanning for pickle files in {pickle_dir}...")
    
    # Find all pickle files
    pickle_files = glob.glob(os.path.join(pickle_dir, "*.pickle"))
    print(f"Found {len(pickle_files)} pickle files")
    
    if not pickle_files:
        print("No pickle files found!")
        return []
    
    # If batch_size is specified, process files in batches
    if batch_size and batch_size < len(pickle_files):
        print(f"Processing files in batches of {batch_size}")
        all_results = []
        
        for i in range(0, len(pickle_files), batch_size):
            batch_files = pickle_files[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(pickle_files) + batch_size - 1)//batch_size} ({len(batch_files)} files)")
            
            # Prepare arguments for multiprocessing
            args_list = [(pf, voxel_size) for pf in batch_files]
            
            # Process batch in parallel
            with mp.Pool(num_workers) as pool:
                batch_results = list(tqdm(pool.imap(process_single_pickle_file, args_list), 
                                      total=len(batch_files), desc=f"Batch {i//batch_size + 1}"))
            
            # Filter out None results and add to all_results
            valid_batch_results = [r for r in batch_results if r is not None]
            all_results.extend(valid_batch_results)
            print(f"Batch {i//batch_size + 1} completed: {len(valid_batch_results)} valid results")
    else:
        # Process all files at once (original behavior)
        print(f"Processing all {len(pickle_files)} files at once")
        args_list = [(pf, voxel_size) for pf in pickle_files]
        
        # Process files in parallel
        print(f"Processing {len(pickle_files)} files with {num_workers} workers...")
        with mp.Pool(num_workers) as pool:
            results = list(tqdm(pool.imap(process_single_pickle_file, args_list), 
                              total=len(pickle_files), desc="Processing files"))
        
        # Filter out None results
        all_results = [r for r in results if r is not None]
    
    print(f"Successfully processed {len(all_results)} files total")
    return all_results

def create_voxel_grid(coordinates, voxel_size):
    """Create voxel grid and assign coordinates to voxels."""
    if not coordinates:
        return {}, {}
    
    # Convert to numpy arrays
    ap_coords = np.array([c['ap'] for c in coordinates])
    dv_coords = np.array([c['dv'] for c in coordinates])
    lr_coords = np.array([c['lr'] for c in coordinates])
    probe_h_coords = np.array([c['probe_h'] for c in coordinates])
    probe_v_coords = np.array([c['probe_v'] for c in coordinates])
    
    # Check if CCF coordinates have variation
    ap_range = np.max(ap_coords) - np.min(ap_coords)
    dv_range = np.max(dv_coords) - np.min(dv_coords)
    lr_range = np.max(lr_coords) - np.min(lr_coords)
    
    # If CCF coordinates have no variation, use probe coordinates for spatial variation
    if ap_range == 0 and dv_range == 0 and lr_range == 0:
        print("CCF coordinates identical - using probe coordinates for spatial variation")
        # Use probe coordinates: H for X, V for Y, LR for Z (even though LR is constant)
        x_coords = probe_h_coords
        y_coords = probe_v_coords
        z_coords = lr_coords  # Keep LR as Z-axis
    else:
        print("Using CCF coordinates for spatial variation")
        # Use CCF coordinates
        x_coords = ap_coords
        y_coords = dv_coords
        z_coords = lr_coords
    
    # Calculate voxel grid dimensions
    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)
    z_min, z_max = np.min(z_coords), np.max(z_coords)
    
    # Create voxel grid
    x_voxels = np.arange(x_min, x_max + voxel_size, voxel_size)
    y_voxels = np.arange(y_min, y_max + voxel_size, voxel_size)
    z_voxels = np.arange(z_min, z_max + voxel_size, voxel_size)
    
    # Assign coordinates to voxels
    voxel_to_coords = defaultdict(list)
    voxel_to_session = {}
    
    for i, coord in enumerate(coordinates):
        x_idx = int((x_coords[i] - x_min) / voxel_size)
        y_idx = int((y_coords[i] - y_min) / voxel_size)
        z_idx = int((z_coords[i] - z_min) / voxel_size)
        
        voxel_key = (x_idx, y_idx, z_idx)
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
                        title, use_different_colors=True, use_probe_coords=False):
    """Create 3D visualization with perfect cubes using the working method."""
    print(f"Creating visualization: {title}")
    
    # Get unique sessions for color mapping
    unique_sessions = list(set(voxel_to_session.values()))
    n_sessions = len(unique_sessions)
    
    # Create figure
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Define colors for sessions - grayscale gradient
    if n_sessions > 1:
        # Different grayscale colors for different sessions
        session_colors = plt.cm.Greys(np.linspace(0.3, 0.8, n_sessions))
        session_to_color = {session: session_colors[i] for i, session in enumerate(unique_sessions)}
    else:
        # Single session gets one grayscale color
        session_to_color = {unique_sessions[0]: plt.cm.Greys(0.6)}
    
    # Define the 8 vertices of a unit cube
    unit_cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    
    # Define the 6 faces of the cube
    cube_faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [0, 3, 7, 4],  # Left face
        [1, 2, 6, 5]   # Right face
    ]
    
    # Create all cube faces
    all_faces = []
    all_colors = []
    voxel_centers = []
    
    voxel_index = 0
    for voxel_key, coords in voxel_to_coords.items():
        if not coords:
            continue
        
        # Calculate voxel center (actual coordinates, not indices)
        x_idx, y_idx, z_idx = voxel_key
        
        # Get the actual coordinate ranges from the first coordinate
        first_coord = coords[0]
        if use_probe_coords:
            # Use probe coordinates
            x_min = min(c['probe_h'] for c in coords)
            y_min = min(c['probe_v'] for c in coords)
            z_min = min(c['lr'] for c in coords)
            center = (
                x_min + x_idx * voxel_size,
                y_min + y_idx * voxel_size,
                z_min + z_idx * voxel_size
            )
        else:
            # Use CCF coordinates
            x_min = min(c['ap'] for c in coords)
            y_min = min(c['dv'] for c in coords)
            z_min = min(c['lr'] for c in coords)
            center = (
                x_min + x_idx * voxel_size,
                y_min + y_idx * voxel_size,
                z_min + z_idx * voxel_size
            )
        
        voxel_centers.append(center)
        
        # Scale and translate the unit cube
        cube_vertices = unit_cube_vertices * voxel_size + np.array(center)
        
        # Add faces for this cube
        for face in cube_faces:
            face_vertices = cube_vertices[face]
            all_faces.append(face_vertices)
            # Get color based on session
            session = voxel_to_session[voxel_key]
            all_colors.append(session_to_color[session])
        
        voxel_index += 1
    
    # Create Poly3DCollection for all faces
    if all_faces:
        cube_collection = Poly3DCollection(all_faces, facecolor=all_colors, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_collection3d(cube_collection)
        
        # Add scatter points for voxel centers with session colors
        centers_array = np.array(voxel_centers)
        scatter_colors = []
        for voxel_key, coords in voxel_to_coords.items():
            if coords:
                session = voxel_to_session[voxel_key]
                scatter_colors.append(session_to_color[session])
        
        ax.scatter(centers_array[:, 0], centers_array[:, 1], centers_array[:, 2], 
                  c=scatter_colors, s=50, alpha=0.9, edgecolors='none')
    
    # Set axis labels based on coordinate system used
    if use_probe_coords:
        ax.set_xlabel('Probe Horizontal (μm)', fontsize=14)
        ax.set_ylabel('Probe Vertical (μm)', fontsize=14)
        ax.set_zlabel('Left-Right (μm)', fontsize=14)
    else:
        ax.set_xlabel('Anterior-Posterior (μm)', fontsize=14)
        ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=14)
        ax.set_zlabel('Left-Right (μm)', fontsize=14)
    
    ax.set_title(f'{title}\n{len(voxel_centers)} voxels, {len(unique_sessions)} sessions', fontsize=16)
    
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
    parser.add_argument('pickle_dir', help='Directory containing enriched pickle files')
    parser.add_argument('--output-dir', default='massive_session_viz', 
                       help='Output directory for visualizations')
    parser.add_argument('--voxel-size', type=float, default=1.0, 
                       help='Voxel size in mm (default: 1.0)')
    parser.add_argument('--workers', type=int, default=8, 
                       help='Number of parallel workers (default: 8)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Number of files to process in each batch (default: process all at once)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all pickle files (coordinates are already embedded in enriched labels)
    print(f"\nProcessing all enriched pickle files in {args.pickle_dir}...")
    all_results = process_all_pickle_files(args.pickle_dir, args.voxel_size, 
                                        args.workers, args.batch_size)
    
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
    
    # Check if probe coordinates were used
    use_probe_coords = len(set(c['ap'] for c in all_coordinates)) == 1 and len(set(c['dv'] for c in all_coordinates)) == 1 and len(set(c['lr'] for c in all_coordinates)) == 1
    
    # Same color visualization
    same_color_path = os.path.join(args.output_dir, 'massive_viz_same_color.png')
    create_visualization(voxel_to_coords, voxel_to_session, args.voxel_size, 
                        same_color_path, 'Massive Visualization - Same Color', 
                        use_different_colors=False, use_probe_coords=use_probe_coords)
    
    # Different colors per session visualization
    diff_color_path = os.path.join(args.output_dir, 'massive_viz_different_colors.png')
    create_visualization(voxel_to_coords, voxel_to_session, args.voxel_size, 
                        diff_color_path, 'Massive Visualization - Different Colors Per Session', 
                        use_different_colors=True, use_probe_coords=use_probe_coords)
    
    print(f"\nVisualization complete! Output saved to {args.output_dir}")
    print(f"Files created:")
    print(f"  - {same_color_path}")
    print(f"  - {diff_color_path}")
    print(f"  - {os.path.join(args.output_dir, 'massive_viz_statistics.json')}")

if __name__ == '__main__':
    main()
