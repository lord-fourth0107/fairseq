#!/usr/bin/env python3
"""
Session Probe Voxel Visualization
================================
Groups pickles by session ID, extracts CCF coordinates for each channel,
assigns them to 1mm³ voxels, and visualizes perfect cubes with different colors per probe.
HPC-friendly with multiprocessing support.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from collections import defaultdict
from tqdm import tqdm
import multiprocessing as mp
from functools import partial

def load_coordinate_lookup(input_path: str) -> dict:
    """Load and merge coordinate data from CSV files."""
    joined_path = os.path.join(input_path, 'joined.csv')
    channels_path = os.path.join(input_path, 'channels.csv')
    
    if not os.path.exists(joined_path):
        raise FileNotFoundError(f"joined.csv not found at {joined_path}")
    if not os.path.exists(channels_path):
        raise FileNotFoundError(f"channels.csv not found at {channels_path}")
    
    joined_df = pd.read_csv(joined_path)
    channels_df = pd.read_csv(channels_path)
    merged_df = pd.merge(
        joined_df,
        channels_df,
        left_on='probe_id',
        right_on='ecephys_probe_id',
        how='inner',
    )
    
    coord_lookup = {}
    for _, row in merged_df.iterrows():
        key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
        coord_lookup[key] = {
            'ap': row['anterior_posterior_ccf_coordinate_y'],
            'dv': row['dorsal_ventral_ccf_coordinate_y'],
            'lr': row['left_right_ccf_coordinate_y'],
        }
    return coord_lookup

def extract_channel_coords_from_pickle(pickle_path: str, coord_lookup: dict):
    """Extract unique channel coordinates from a single pickle file."""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    
    # Extract session_id and probe_id from filename
    filename = os.path.basename(pickle_path)
    if '_' not in filename:
        return None, None, []
    
    session_id, probe_id = filename.replace('.pickle', '').split('_', 1)
    
    # Collect unique channel coordinates
    unique_coords = set()
    for entry in data:
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            _, label = entry
            parts = label.split('_')
            if len(parts) >= 4:
                try:
                    label_session = parts[0]
                    label_probe = parts[2]
                    channel_id = parts[3]
                    
                    # Verify session and probe match filename
                    if label_session == session_id and label_probe == probe_id:
                        key = (session_id, probe_id, channel_id)
                        if key in coord_lookup:
                            coords = coord_lookup[key]
                            coord_tuple = (
                                float(coords['ap']),
                                float(coords['dv']),
                                float(coords['lr'])
                            )
                            unique_coords.add(coord_tuple)
                except (ValueError, IndexError):
                    continue
    
    coords_array = np.array(list(unique_coords), dtype=float) if unique_coords else np.array([])
    return session_id, probe_id, coords_array

def process_pickle_file(args):
    """Worker function to process a single pickle file."""
    pickle_path, coord_lookup = args
    try:
        session_id, probe_id, coords = extract_channel_coords_from_pickle(pickle_path, coord_lookup)
        return {
            'file': pickle_path,
            'session_id': session_id,
            'probe_id': probe_id,
            'coords': coords,
            'success': True,
            'error': None
        }
    except Exception as e:
        return {
            'file': pickle_path,
            'session_id': None,
            'probe_id': None,
            'coords': None,
            'success': False,
            'error': str(e)
        }

def group_pickles_by_session(pickle_files: list, coord_lookup: dict, num_workers: int = None):
    """Group pickle files by session ID and extract coordinates for each probe."""
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(pickle_files))
    
    print(f"Processing {len(pickle_files)} pickle files with {num_workers} workers...")
    
    # Process files in parallel
    worker_args = [(pickle_path, coord_lookup) for pickle_path in pickle_files]
    
    session_data = defaultdict(dict)  # session_id -> {probe_id: coords}
    
    if num_workers == 1:
        # Sequential processing
        for args in tqdm(worker_args, desc="Processing files"):
            result = process_pickle_file(args)
            if result['success'] and result['coords'] is not None and len(result['coords']) > 0:
                session_data[result['session_id']][result['probe_id']] = result['coords']
            elif not result['success']:
                print(f"Error processing {result['file']}: {result['error']}")
    else:
        # Parallel processing
        with mp.Pool(processes=num_workers) as pool:
            results = pool.imap(process_pickle_file, worker_args)
            for result in tqdm(results, total=len(worker_args), desc="Processing files"):
                if result['success'] and result['coords'] is not None and len(result['coords']) > 0:
                    session_data[result['session_id']][result['probe_id']] = result['coords']
                elif not result['success']:
                    print(f"Error processing {result['file']}: {result['error']}")
    
    return session_data

def create_voxel_grid(coords: np.ndarray, voxel_size: float = 1000.0):
    """Create a voxel grid from coordinates (voxel_size in micrometers)."""
    if len(coords) == 0:
        return np.array([]), np.array([]), np.array([])
    
    # Convert coordinates to voxel indices
    voxel_coords = np.floor(coords / voxel_size).astype(int)
    
    # Get unique voxels
    unique_voxels = np.unique(voxel_coords, axis=0)
    
    return unique_voxels

def create_cube_vertices(voxel_coords: np.ndarray, voxel_size: float = 1000.0):
    """Create cube vertices for each voxel coordinate."""
    if len(voxel_coords) == 0:
        return []
    
    cubes = []
    for voxel in voxel_coords:
        # Convert voxel coordinates back to real coordinates
        x, y, z = voxel * voxel_size
        
        # Define the 8 vertices of a cube
        vertices = np.array([
            [x, y, z], [x + voxel_size, y, z], [x + voxel_size, y + voxel_size, z], [x, y + voxel_size, z],
            [x, y, z + voxel_size], [x + voxel_size, y, z + voxel_size], 
            [x + voxel_size, y + voxel_size, z + voxel_size], [x, y + voxel_size, z + voxel_size]
        ])
        
        # Define the 6 faces of the cube
        faces = [
            [0, 1, 2, 3],  # bottom
            [4, 5, 6, 7],  # top
            [0, 1, 5, 4],  # front
            [2, 3, 7, 6],  # back
            [0, 3, 7, 4],  # left
            [1, 2, 6, 5]   # right
        ]
        
        cubes.append((vertices, faces))
    
    return cubes

def plot_perfect_cubes(session_data: dict, session_id: str, output_dir: str, voxel_size: float = 1000.0):
    """Create 3D visualization of perfect cubes for all probes in a session."""
    if session_id not in session_data:
        print(f"Session {session_id} not found in data")
        return
    
    probe_data = session_data[session_id]
    if not probe_data:
        print(f"No probe data found for session {session_id}")
        return
    
    # Define colors for probes
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
    
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    all_cubes = []
    all_colors = []
    
    for i, (probe_id, coords) in enumerate(probe_data.items()):
        if len(coords) == 0:
            continue
        
        print(f"Probe {probe_id}: {len(coords)} unique channels")
        print(f"  Coordinate ranges: AP [{coords[:, 0].min():.0f}, {coords[:, 0].max():.0f}], "
              f"DV [{coords[:, 1].min():.0f}, {coords[:, 1].max():.0f}], "
              f"LR [{coords[:, 2].min():.0f}, {coords[:, 2].max():.0f}]")
        
        # Create voxel grid
        voxel_coords = create_voxel_grid(coords, voxel_size)
        print(f"  Created {len(voxel_coords)} unique voxels")
        
        if len(voxel_coords) == 0:
            continue
        
        # Create cube vertices
        cubes = create_cube_vertices(voxel_coords, voxel_size)
        print(f"  Created {len(cubes)} cubes")
        
        # Add cubes to collection
        for vertices, faces in cubes:
            all_cubes.append((vertices, faces))
            all_colors.append(colors[i % len(colors)])
    
    print(f"Total cubes to plot: {len(all_cubes)}")
    
    # Plot all cubes
    if all_cubes:
        for (vertices, faces), color in zip(all_cubes, all_colors):
            cube_faces = []
            for face in faces:
                cube_faces.append([vertices[face[0]], vertices[face[1]], vertices[face[2]], vertices[face[3]]])
            
            poly3d = Poly3DCollection(cube_faces, alpha=0.7, facecolor=color, edgecolor='black', linewidth=0.5)
            ax.add_collection3d(poly3d)
        
        # Set axis limits based on actual data
        all_vertices = np.concatenate([vertices for vertices, _ in all_cubes])
        ax.set_xlim(all_vertices[:, 0].min(), all_vertices[:, 0].max())
        ax.set_ylim(all_vertices[:, 1].min(), all_vertices[:, 1].max())
        ax.set_zlim(all_vertices[:, 2].min(), all_vertices[:, 2].max())
    else:
        print("No cubes to plot!")
        return
    
    # Set labels and title
    ax.set_xlabel('AP (μm)')
    ax.set_ylabel('DV (μm)')
    ax.set_zlabel('LR (μm)')
    ax.set_title(f'Session {session_id} - All Probes Voxel Visualization\n(1mm³ voxels, different colors per probe)')
    
    # Create legend
    legend_elements = []
    for i, probe_id in enumerate(probe_data.keys()):
        if len(session_data[session_id][probe_id]) > 0:
            legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', 
                                            markerfacecolor=colors[i % len(colors)], 
                                            markersize=10, label=f'Probe {probe_id}'))
    
    if legend_elements:
        ax.legend(handles=legend_elements, loc='upper left')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save plot
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'session_{session_id}_all_probes_voxels.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def find_pickle_files(input_path: str):
    """Find all pickle files in the given path."""
    if os.path.isfile(input_path):
        if input_path.endswith('.pickle'):
            return [input_path]
        else:
            print(f"Error: {input_path} is not a pickle file")
            return []
    elif os.path.isdir(input_path):
        pickle_files = []
        for file in os.listdir(input_path):
            if file.endswith('.pickle'):
                pickle_files.append(os.path.join(input_path, file))
        if not pickle_files:
            print(f"No pickle files found in {input_path}")
            return []
        return sorted(pickle_files)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return []

def main():
    parser = argparse.ArgumentParser(description='Session Probe Voxel Visualization')
    parser.add_argument('input_path', help='Path to directory containing pickle files')
    parser.add_argument('--session-id', help='Specific session ID to process (if not provided, processes first session found)')
    parser.add_argument('--workers', type=int, default=None, help='Number of worker processes (default: auto-detect)')
    parser.add_argument('--voxel-size', type=float, default=1000.0, help='Voxel size in micrometers (default: 1000 = 1mm)')
    parser.add_argument('--output-dir', default='session_voxel_viz', help='Output directory for plots')
    
    args = parser.parse_args()
    
    # Find pickle files
    pickle_files = find_pickle_files(args.input_path)
    if not pickle_files:
        return
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Load coordinate lookup
    coord_lookup = load_coordinate_lookup(args.input_path)
    print(f"Loaded coordinate lookup with {len(coord_lookup)} entries")
    
    # Group pickles by session
    session_data = group_pickles_by_session(pickle_files, coord_lookup, args.workers)
    
    if not session_data:
        print("No valid session data found")
        return
    
    print(f"Found {len(session_data)} sessions: {list(session_data.keys())}")
    
    # Select session to visualize
    if args.session_id:
        if args.session_id not in session_data:
            print(f"Session {args.session_id} not found. Available sessions: {list(session_data.keys())}")
            return
        selected_session = args.session_id
    else:
        # Use first session
        selected_session = list(session_data.keys())[0]
        print(f"Using first session: {selected_session}")
    
    # Create visualization
    plot_perfect_cubes(session_data, selected_session, args.output_dir, args.voxel_size)
    
    # Print summary
    probe_data = session_data[selected_session]
    print(f"\nSession {selected_session} summary:")
    for probe_id, coords in probe_data.items():
        print(f"  Probe {probe_id}: {len(coords)} unique channels")

if __name__ == '__main__':
    mp.freeze_support()
    main()
