#!/usr/bin/env python3
"""
Perfect Cube Voxel Visualization
================================
Creates perfect 1mm³ cubes for each channel location.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
import argparse

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

def extract_ccf_coordinates_from_label(label, coord_lookup):
    """Extract CCF coordinates from enriched label."""
    try:
        # Parse the enriched label: session_count_probe_channel_brain_region_ap_dv_lr_probe_h_probe_v
        parts = label.split('_')
        if len(parts) >= 9:
            session_id = parts[0]
            probe_id = parts[2]
            channel_id = parts[3]
            
            # Look up coordinates
            lookup_key = (session_id, probe_id, channel_id)
            if lookup_key in coord_lookup:
                coords = coord_lookup[lookup_key]
                return {
                    'ap': float(coords['ap']),
                    'dv': float(coords['dv']),
                    'lr': float(coords['lr']),
                    'probe_h': float(coords['probe_h']),
                    'probe_v': float(coords['probe_v']),
                    'session_id': session_id,
                    'probe_id': probe_id,
                    'channel_id': channel_id,
                    'structure': coords.get('structure', 'Unknown')
                }
    except (ValueError, IndexError) as e:
        pass
    
    return None

def process_single_pickle(pickle_file, coord_lookup):
    """Process a single pickle file and extract unique channel coordinates."""
    print(f"Processing {os.path.basename(pickle_file)}...")
    
    try:
        with open(pickle_file, 'rb') as f:
            data = pickle.load(f)
        
        # Extract filename for probe identification
        filename = os.path.basename(pickle_file)
        name_no_ext = os.path.splitext(filename)[0]
        if '_' not in name_no_ext:
            print(f"Filename does not match pattern session_probe: {filename}")
            return None, None, None
        session_id, probe_id = name_no_ext.split('_', 1)
        probe_color = 'red'
        
        print(f"Probe ID: {probe_id}, Session: {session_id}, Color: {probe_color}")
        
        # Collect unique channel coordinates
        unique_channels = set()
        coordinates = []
        
        for entry in tqdm(data, desc="Processing channels"):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                signal, label = entry
                coords = extract_ccf_coordinates_from_label(label, coord_lookup)
                if coords:
                    # Create unique key for this channel
                    channel_key = (coords['session_id'], coords['probe_id'], coords['channel_id'])
                    if channel_key not in unique_channels:
                        unique_channels.add(channel_key)
                        coordinates.append(coords)
        
        print(f"Found {len(coordinates)} unique channels")
        return coordinates, probe_id, probe_color
        
    except Exception as e:
        print(f"Error processing {pickle_file}: {e}")
        return None, None, None

def create_voxel_grid(coordinates, voxel_size_mm=1.0):
    """Create 1mm³ voxel grid and mark occupied voxels."""
    print(f"Creating {voxel_size_mm}mm³ voxel grid...")
    
    # Convert coordinates to numpy array
    coords_array = np.array([(c['ap'], c['dv'], c['lr']) for c in coordinates])
    
    # Calculate grid bounds
    ap_min, ap_max = coords_array[:, 0].min(), coords_array[:, 0].max()
    dv_min, dv_max = coords_array[:, 1].min(), coords_array[:, 1].max()
    lr_min, lr_max = coords_array[:, 2].min(), coords_array[:, 2].max()
    
    print(f"Coordinate ranges:")
    print(f"  AP: {ap_min:.1f} to {ap_max:.1f} μm")
    print(f"  DV: {dv_min:.1f} to {dv_max:.1f} μm")
    print(f"  LR: {lr_min:.1f} to {lr_max:.1f} μm")
    
    # Convert to mm and create grid
    voxel_size_um = voxel_size_mm * 1000  # Convert mm to μm
    
    ap_min_mm = np.floor(ap_min / voxel_size_um) * voxel_size_um
    ap_max_mm = np.ceil(ap_max / voxel_size_um) * voxel_size_um
    dv_min_mm = np.floor(dv_min / voxel_size_um) * voxel_size_um
    dv_max_mm = np.ceil(dv_max / voxel_size_um) * voxel_size_um
    lr_min_mm = np.floor(lr_min / voxel_size_um) * voxel_size_um
    lr_max_mm = np.ceil(lr_max / voxel_size_um) * voxel_size_um
    
    # Calculate grid dimensions
    ap_size = int((ap_max_mm - ap_min_mm) / voxel_size_um) + 1
    dv_size = int((dv_max_mm - dv_min_mm) / voxel_size_um) + 1
    lr_size = int((lr_max_mm - lr_min_mm) / voxel_size_um) + 1
    
    print(f"Voxel grid dimensions: {ap_size} × {dv_size} × {lr_size}")
    print(f"Total voxels: {ap_size * dv_size * lr_size:,}")
    
    # Create 3D grid (1 = occupied, 0 = empty)
    voxel_grid = np.zeros((ap_size, dv_size, lr_size), dtype=bool)
    
    # Mark occupied voxels
    occupied_voxels = set()
    for coord in tqdm(coordinates, desc="Marking occupied voxels"):
        ap, dv, lr = coord['ap'], coord['dv'], coord['lr']
        
        # Convert to voxel indices
        ap_idx = int((ap - ap_min_mm) / voxel_size_um)
        dv_idx = int((dv - dv_min_mm) / voxel_size_um)
        lr_idx = int((lr - lr_min_mm) / voxel_size_um)
        
        # Check bounds and mark voxel
        if 0 <= ap_idx < ap_size and 0 <= dv_idx < dv_size and 0 <= lr_idx < lr_size:
            voxel_grid[ap_idx, dv_idx, lr_idx] = True
            occupied_voxels.add((ap_idx, dv_idx, lr_idx))
    
    print(f"Occupied voxels: {len(occupied_voxels)}")
    
    return voxel_grid, {
        'ap_min': ap_min_mm, 'ap_max': ap_max_mm,
        'dv_min': dv_min_mm, 'dv_max': dv_max_mm,
        'lr_min': lr_min_mm, 'lr_max': lr_max_mm,
        'voxel_size_um': voxel_size_um,
        'ap_size': ap_size, 'dv_size': dv_size, 'lr_size': lr_size
    }

def create_perfect_cube_plot(voxel_grid, grid_info, probe_id, probe_color, output_dir="perfect_cube_viz"):
    """Create 3D plot with perfect cubes for each occupied voxel."""
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating perfect cube visualization for probe {probe_id}...")
    
    # Find occupied voxels
    occupied_voxels = np.where(voxel_grid)
    
    if len(occupied_voxels[0]) == 0:
        print("No occupied voxels found!")
        return
    
    # Get coordinates of occupied voxels
    ap_coords = occupied_voxels[0] * grid_info['voxel_size_um'] + grid_info['ap_min']
    dv_coords = occupied_voxels[1] * grid_info['voxel_size_um'] + grid_info['dv_min']
    lr_coords = occupied_voxels[2] * grid_info['voxel_size_um'] + grid_info['lr_min']
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create perfect cubes using Poly3DCollection
    voxel_size = grid_info['voxel_size_um']
    
    # Define the 8 vertices of a unit cube
    unit_cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # Bottom face
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]   # Top face
    ])
    
    # Define the 6 faces of the cube (each face has 4 vertices)
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
    for i in range(len(ap_coords)):
        # Scale and translate the unit cube
        cube_vertices = unit_cube_vertices * voxel_size + np.array([ap_coords[i], dv_coords[i], lr_coords[i]])
        
        # Add faces for this cube
        for face in cube_faces:
            face_vertices = cube_vertices[face]
            all_faces.append(face_vertices)
    
    # Create Poly3DCollection for all faces
    cube_collection = Poly3DCollection(all_faces, facecolor=probe_color, alpha=0.7, 
                                     edgecolor='black', linewidth=0.5)
    ax.add_collection3d(cube_collection)
    
    # Set labels
    ax.set_xlabel('Anterior-Posterior (μm)', fontsize=14)
    ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=14)
    ax.set_zlabel('Left-Right (μm)', fontsize=14)
    ax.set_title(f'Probe {probe_id} - Perfect 1mm³ Cubes', fontsize=16)
    
    # Set equal aspect ratio
    max_range = np.array([ap_coords.max() - ap_coords.min(),
                         dv_coords.max() - dv_coords.min(),
                         lr_coords.max() - lr_coords.min()]).max() / 2.0
    
    mid_x = np.mean(ap_coords)
    mid_y = np.mean(dv_coords)
    mid_z = np.mean(lr_coords)
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'probe_{probe_id}_perfect_cubes.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, f'probe_{probe_id}_perfect_cubes.png')}")
    plt.close()

def create_combined_perfect_cube_plot(probe_to_voxels, grid_info, output_dir, session_id):
    """Create a single 3D plot with perfect cubes for each probe in different colors."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Creating combined perfect cube visualization for session {session_id}...")

    # Color cycle for probes
    color_cycle = [
        'red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray',
        'cyan', 'magenta', 'olive', 'teal'
    ]

    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')

    voxel_size = grid_info['voxel_size_um']

    unit_cube_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ])
    cube_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [2, 3, 7, 6],
        [0, 3, 7, 4],
        [1, 2, 6, 5]
    ]

    all_centers = []

    for idx, (probe_id, occupied_voxels) in enumerate(probe_to_voxels.items()):
        if not occupied_voxels:
            continue
        probe_color = color_cycle[idx % len(color_cycle)]

        ap_idx_arr = np.array([v[0] for v in occupied_voxels], dtype=int)
        dv_idx_arr = np.array([v[1] for v in occupied_voxels], dtype=int)
        lr_idx_arr = np.array([v[2] for v in occupied_voxels], dtype=int)

        ap_coords = ap_idx_arr * voxel_size + grid_info['ap_min']
        dv_coords = dv_idx_arr * voxel_size + grid_info['dv_min']
        lr_coords = lr_idx_arr * voxel_size + grid_info['lr_min']

        all_centers.append(np.vstack([ap_coords, dv_coords, lr_coords]).T)

        faces = []
        for i in range(len(ap_coords)):
            cube_vertices = unit_cube_vertices * voxel_size + np.array([ap_coords[i], dv_coords[i], lr_coords[i]])
            for face in cube_faces:
                faces.append(cube_vertices[face])

        collection = Poly3DCollection(faces, facecolor=probe_color, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.add_collection3d(collection)
        ax.scatter(lr_coords, dv_coords, ap_coords, c=probe_color, s=10, alpha=0.9, edgecolors='none')

    if all_centers:
        all_centers = np.concatenate(all_centers, axis=0)
        ap_vals = all_centers[:, 0]
        dv_vals = all_centers[:, 1]
        lr_vals = all_centers[:, 2]
        max_range = np.array([
            ap_vals.max() - ap_vals.min(),
            dv_vals.max() - dv_vals.min(),
            lr_vals.max() - lr_vals.min(),
        ]).max() / 2.0
        mid_x = np.mean(ap_vals)
        mid_y = np.mean(dv_vals)
        mid_z = np.mean(lr_vals)
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_xlabel('Anterior-Posterior (μm)', fontsize=14)
    ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=14)
    ax.set_zlabel('Left-Right (μm)', fontsize=14)
    ax.set_title(f'Session {session_id} - Perfect Cubes (Probes colored)', fontsize=16)
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(output_dir, f'session_{session_id}_all_probes_perfect_cubes.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {out_path}")
    plt.close()

def compute_occupied_voxels_for_coords(coordinates, grid_info):
    """Given raw coordinates and a fixed grid, compute occupied voxel indices."""
    occupied = set()
    voxel = grid_info['voxel_size_um']
    for coord in coordinates:
        ap, dv, lr = coord['ap'], coord['dv'], coord['lr']
        ap_idx = int((ap - grid_info['ap_min']) / voxel)
        dv_idx = int((dv - grid_info['dv_min']) / voxel)
        lr_idx = int((lr - grid_info['lr_min']) / voxel)
        if (0 <= ap_idx < grid_info['ap_size'] and
            0 <= dv_idx < grid_info['dv_size'] and
            0 <= lr_idx < grid_info['lr_size']):
            occupied.add((ap_idx, dv_idx, lr_idx))
    return occupied

def create_simple_cube_plot(voxel_grid, grid_info, probe_id, probe_color, output_dir="perfect_cube_viz"):
    """Create 3D plot with simple cube representation using scatter points."""
    
    print(f"Creating simple cube visualization for probe {probe_id}...")
    
    # Find occupied voxels
    occupied_voxels = np.where(voxel_grid)
    
    if len(occupied_voxels[0]) == 0:
        print("No occupied voxels found!")
        return
    
    # Get coordinates of occupied voxels (center of each voxel)
    ap_coords = occupied_voxels[0] * grid_info['voxel_size_um'] + grid_info['ap_min'] + grid_info['voxel_size_um']/2
    dv_coords = occupied_voxels[1] * grid_info['voxel_size_um'] + grid_info['dv_min'] + grid_info['voxel_size_um']/2
    lr_coords = occupied_voxels[2] * grid_info['voxel_size_um'] + grid_info['lr_min'] + grid_info['voxel_size_um']/2
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot occupied voxels as large cubes (using scatter with large size)
    voxel_size = grid_info['voxel_size_um']
    ax.scatter(ap_coords, dv_coords, lr_coords, 
              c=probe_color, s=voxel_size**2, alpha=0.8, edgecolors='black', linewidth=1)
    
    # Set labels
    ax.set_xlabel('Anterior-Posterior (μm)', fontsize=14)
    ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=14)
    ax.set_zlabel('Left-Right (μm)', fontsize=14)
    ax.set_title(f'Probe {probe_id} - Simple Cube Representation', fontsize=16)
    
    # Set equal aspect ratio
    max_range = np.array([ap_coords.max() - ap_coords.min(),
                         dv_coords.max() - dv_coords.min(),
                         lr_coords.max() - lr_coords.min()]).max() / 2.0
    
    mid_x = np.mean(ap_coords)
    mid_y = np.mean(dv_coords)
    mid_z = np.mean(lr_coords)
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'probe_{probe_id}_simple_cubes.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, f'probe_{probe_id}_simple_cubes.png')}")
    plt.close()

def print_voxel_statistics(voxel_grid, grid_info, coordinates, probe_id):
    """Print voxel statistics."""
    print("\n" + "="*60)
    print(f"VOXEL STATISTICS - PROBE {probe_id}")
    print("="*60)
    
    total_voxels = voxel_grid.size
    occupied_voxels = np.sum(voxel_grid)
    
    print(f"Total voxels: {total_voxels:,}")
    print(f"Occupied voxels: {occupied_voxels:,} ({occupied_voxels/total_voxels*100:.1f}%)")
    print(f"Empty voxels: {total_voxels - occupied_voxels:,} ({(total_voxels - occupied_voxels)/total_voxels*100:.1f}%)")
    print(f"Unique channels: {len(coordinates)}")
    print(f"Channels per occupied voxel: {len(coordinates)/occupied_voxels:.1f}")
    
    print(f"\nGrid dimensions:")
    print(f"  AP: {grid_info['ap_size']} voxels ({grid_info['ap_size']} mm)")
    print(f"  DV: {grid_info['dv_size']} voxels ({grid_info['dv_size']} mm)")
    print(f"  LR: {grid_info['lr_size']} voxels ({grid_info['lr_size']} mm)")
    print(f"  Voxel size: {grid_info['voxel_size_um']/1000:.1f} mm³")

def main():
    parser = argparse.ArgumentParser(description='Perfect Cube Voxel Visualization')
    parser.add_argument('pickle_or_dir', help='Path to a pickle file or directory containing pickles')
    parser.add_argument('input_path', help='Path to directory containing CSV files (joined.csv, channels.csv)')
    parser.add_argument('--output-dir', default='perfect_cube_viz', 
                       help='Output directory for visualizations (default: perfect_cube_viz)')
    parser.add_argument('--voxel-size', type=float, default=1.0,
                       help='Voxel size in mm (default: 1.0)')
    parser.add_argument('--session-id', default=None,
                       help='If provided and input is a directory, combine all pickles matching this session id into one plot')
    
    args = parser.parse_args()
    
    print("Perfect Cube Voxel Visualization")
    print("="*40)
    print(f"Pickle or dir: {args.pickle_or_dir}")
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Voxel size: {args.voxel_size}mm³")
    
    try:
        # Load coordinate lookup
        coord_lookup = load_coordinate_lookup(args.input_path)
        
        # If a directory is provided and session-id is set, combine all matching pickles
        if os.path.isdir(args.pickle_or_dir) and args.session_id:
            print(f"Combining all pickles for session {args.session_id}...")
            # Discover pickles matching the session id pattern
            all_files = [f for f in os.listdir(args.pickle_or_dir) if f.endswith('.pickle')]
            matching = [os.path.join(args.pickle_or_dir, f) for f in all_files if f.startswith(f"{args.session_id}_")]
            if not matching:
                print(f"No pickles found for session {args.session_id} in {args.pickle_or_dir}")
                return
            print(f"Found {len(matching)} pickles for session {args.session_id}")

            # First pass: collect all coordinates to establish a common grid
            all_coords = []
            per_probe_coords = {}
            for pkl in matching:
                coords, probe_id, _ = process_single_pickle(pkl, coord_lookup)
                if coords is None or not coords:
                    continue
                per_probe_coords[probe_id] = coords
                all_coords.extend(coords)
            if not all_coords:
                print("No coordinates found across pickles.")
                return

            # Build a common grid using all coordinates
            common_grid, grid_info = create_voxel_grid(all_coords, args.voxel_size)

            # For each probe, compute occupied voxels on the common grid
            probe_to_voxels = {}
            for probe_id, coords in per_probe_coords.items():
                occupied = compute_occupied_voxels_for_coords(coords, grid_info)
                probe_to_voxels[probe_id] = occupied

            # Render combined perfect cubes
            create_combined_perfect_cube_plot(probe_to_voxels, grid_info, args.output_dir, args.session_id)
            print(f"\nCombined perfect cube visualization completed! Check the '{args.output_dir}' directory for images.")
        else:
            # Process a single pickle file path
            if not os.path.isfile(args.pickle_or_dir):
                print("Input is not a file. Provide a pickle file or a directory with --session-id.")
                return
            coordinates, probe_id, probe_color = process_single_pickle(args.pickle_or_dir, coord_lookup)
            if coordinates is None:
                print("Failed to process pickle file!")
                return
            if not coordinates:
                print("No channel coordinates found!")
                return
            voxel_grid, grid_info = create_voxel_grid(coordinates, args.voxel_size)
            print_voxel_statistics(voxel_grid, grid_info, coordinates, probe_id)
            create_perfect_cube_plot(voxel_grid, grid_info, probe_id, probe_color, args.output_dir)
            create_simple_cube_plot(voxel_grid, grid_info, probe_id, probe_color, args.output_dir)
            print(f"\nPerfect cube visualization completed! Check the '{args.output_dir}' directory for images.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
