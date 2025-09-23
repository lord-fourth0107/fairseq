#!/usr/bin/env python3
"""
Simple 3D Voxel Visualization
============================
Creates 1mm³ voxels colored red for each channel location.
Shows a clear 3D representation of channel distribution.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def load_pickle_files(input_path):
    """Load all pickle files from input path."""
    pickle_files = []
    for file in os.listdir(input_path):
        if file.endswith('.pickle'):
            pickle_files.append(os.path.join(input_path, file))
    
    print(f"Found {len(pickle_files)} pickle files")
    return pickle_files

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

def collect_unique_channels(pickle_files, coord_lookup):
    """Collect unique channel CCF coordinates from all pickle files."""
    print("Collecting unique channel coordinates...")
    
    unique_channels = set()
    all_coordinates = []
    
    for pickle_file in tqdm(pickle_files, desc="Processing pickle files"):
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract filename for probe identification
            filename = os.path.basename(pickle_file)
            if '715093703_810755797' in filename:
                probe_id = '810755797'
                session_id = '715093703'
            elif '847657808_848037578' in filename:
                probe_id = '848037578'
                session_id = '847657808'
            else:
                continue
            
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    signal, label = entry
                    coords = extract_ccf_coordinates_from_label(label, coord_lookup)
                    if coords:
                        # Create unique key for this channel
                        channel_key = (coords['session_id'], coords['probe_id'], coords['channel_id'])
                        if channel_key not in unique_channels:
                            unique_channels.add(channel_key)
                            all_coordinates.append(coords)
            
        except Exception as e:
            print(f"Error processing {pickle_file}: {e}")
            continue
    
    print(f"Found {len(all_coordinates)} unique channels")
    return all_coordinates

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

def create_3d_voxel_plot(voxel_grid, grid_info, output_dir="simple_voxel_viz"):
    """Create 3D plot with red voxels for occupied channels."""
    
    os.makedirs(output_dir, exist_ok=True)
    print("Creating 3D voxel visualization...")
    
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
    
    # Plot occupied voxels as red cubes
    ax.scatter(ap_coords, dv_coords, lr_coords, 
              c='red', s=1000, alpha=0.8, edgecolors='darkred', linewidth=1)
    
    # Set labels
    ax.set_xlabel('Anterior-Posterior (μm)', fontsize=14)
    ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=14)
    ax.set_zlabel('Left-Right (μm)', fontsize=14)
    ax.set_title('Channel Locations - 1mm³ Voxels (Red = Occupied)', fontsize=16)
    
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
    plt.savefig(os.path.join(output_dir, '3d_voxel_channels.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, '3d_voxel_channels.png')}")
    plt.close()

def create_voxel_cube_plot(voxel_grid, grid_info, output_dir="simple_voxel_viz"):
    """Create 3D plot with actual cube representation of voxels."""
    
    print("Creating 3D cube visualization...")
    
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
    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each occupied voxel as a cube
    voxel_size = grid_info['voxel_size_um']
    
    for i in range(len(ap_coords)):
        # Create cube coordinates
        x = ap_coords[i]
        y = dv_coords[i]
        z = lr_coords[i]
        
        # Define cube vertices
        cube_vertices = np.array([
            [x, y, z], [x + voxel_size, y, z], [x + voxel_size, y + voxel_size, z], [x, y + voxel_size, z],
            [x, y, z + voxel_size], [x + voxel_size, y, z + voxel_size], [x + voxel_size, y + voxel_size, z + voxel_size], [x, y + voxel_size, z + voxel_size]
        ])
        
        # Define cube faces
        faces = [
            [0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4],
            [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]
        ]
        
        # Plot cube faces
        for face in faces:
            vertices = cube_vertices[face]
            ax.plot_surface(
                vertices[:, 0].reshape(2, 2), 
                vertices[:, 1].reshape(2, 2), 
                vertices[:, 2].reshape(2, 2),
                color='red', alpha=0.7, edgecolor='darkred', linewidth=0.5
            )
    
    # Set labels
    ax.set_xlabel('Anterior-Posterior (μm)', fontsize=14)
    ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=14)
    ax.set_zlabel('Left-Right (μm)', fontsize=14)
    ax.set_title('Channel Locations - 1mm³ Voxel Cubes (Red = Occupied)', fontsize=16)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_voxel_cubes.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, '3d_voxel_cubes.png')}")
    plt.close()

def print_voxel_statistics(voxel_grid, grid_info, coordinates):
    """Print voxel statistics."""
    print("\n" + "="*60)
    print("VOXEL STATISTICS")
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
    parser = argparse.ArgumentParser(description='Simple 3D Voxel Visualization')
    parser.add_argument('input_path', help='Path to directory containing pickle files and CSV files')
    parser.add_argument('--output-dir', default='simple_voxel_viz', 
                       help='Output directory for visualizations (default: simple_voxel_viz)')
    parser.add_argument('--voxel-size', type=float, default=1.0,
                       help='Voxel size in mm (default: 1.0)')
    
    args = parser.parse_args()
    
    print("Simple 3D Voxel Visualization")
    print("="*40)
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Voxel size: {args.voxel_size}mm³")
    
    try:
        # Load coordinate lookup
        coord_lookup = load_coordinate_lookup(args.input_path)
        
        # Load pickle files
        pickle_files = load_pickle_files(args.input_path)
        
        if not pickle_files:
            print("No pickle files found!")
            return
        
        # Collect unique channel coordinates
        coordinates = collect_unique_channels(pickle_files, coord_lookup)
        
        if not coordinates:
            print("No channel coordinates found!")
            return
        
        # Create voxel grid
        voxel_grid, grid_info = create_voxel_grid(coordinates, args.voxel_size)
        
        # Print statistics
        print_voxel_statistics(voxel_grid, grid_info, coordinates)
        
        # Create visualizations
        create_3d_voxel_plot(voxel_grid, grid_info, args.output_dir)
        create_voxel_cube_plot(voxel_grid, grid_info, args.output_dir)
        
        print(f"\nVoxel visualization completed! Check the '{args.output_dir}' directory for images.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
