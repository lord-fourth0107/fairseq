#!/usr/bin/env python3
"""
3D Channel Voxel Density Visualization
=====================================
Creates 1mm³ voxel histograms of channel density across the Allen mouse brain atlas.
Shows slice-by-slice and 3D volume visualizations.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse
from collections import defaultdict

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

def collect_all_ccf_coordinates(pickle_files, coord_lookup, exclude_hippocampus=True):
    """Collect CCF coordinates for all unique channels from all probes."""
    print("Collecting CCF coordinates from all pickle files...")
    
    all_coordinates = []
    probe_stats = {}
    hippocampus_structures = {'CA1', 'CA2', 'CA3', 'DG', 'SUB', 'POST', 'PRE', 'PAR'}
    
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
            
            probe_coords = []
            unique_channels = set()
            hippocampus_count = 0
            
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    signal, label = entry
                    coords = extract_ccf_coordinates_from_label(label, coord_lookup)
                    if coords:
                        # Filter out hippocampus if requested
                        if exclude_hippocampus and coords['structure'] in hippocampus_structures:
                            hippocampus_count += 1
                            continue
                            
                        probe_coords.append(coords)
                        unique_channels.add(coords['channel_id'])
            
            probe_stats[probe_id] = {
                'session_id': session_id,
                'total_entries': len(data),
                'unique_channels': len(unique_channels),
                'hippocampus_excluded': hippocampus_count,
                'coordinates': probe_coords
            }
            
            all_coordinates.extend(probe_coords)
            
        except Exception as e:
            print(f"Error processing {pickle_file}: {e}")
            continue
    
    print(f"Collected {len(all_coordinates)} total coordinate entries")
    if exclude_hippocampus:
        print(f"Excluded hippocampus channels: {sum(stats['hippocampus_excluded'] for stats in probe_stats.values())}")
    
    return all_coordinates, probe_stats

def create_voxel_grid(coordinates, voxel_size_mm=1.0):
    """Create 1mm³ voxel grid and count channels per voxel."""
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
    
    # Create 3D histogram
    voxel_counts = np.zeros((ap_size, dv_size, lr_size), dtype=int)
    
    # Count channels per voxel
    for coord in tqdm(coordinates, desc="Counting channels per voxel"):
        ap, dv, lr = coord['ap'], coord['dv'], coord['lr']
        
        # Convert to voxel indices
        ap_idx = int((ap - ap_min_mm) / voxel_size_um)
        dv_idx = int((dv - dv_min_mm) / voxel_size_um)
        lr_idx = int((lr - lr_min_mm) / voxel_size_um)
        
        # Check bounds
        if 0 <= ap_idx < ap_size and 0 <= dv_idx < dv_size and 0 <= lr_idx < lr_size:
            voxel_counts[ap_idx, dv_idx, lr_idx] += 1
    
    # Calculate statistics
    occupied_voxels = np.sum(voxel_counts > 0)
    max_density = np.max(voxel_counts)
    mean_density = np.mean(voxel_counts[voxel_counts > 0]) if occupied_voxels > 0 else 0
    
    print(f"Voxel statistics:")
    print(f"  Occupied voxels: {occupied_voxels:,}")
    print(f"  Max channels per voxel: {max_density}")
    print(f"  Mean channels per occupied voxel: {mean_density:.1f}")
    
    return voxel_counts, {
        'ap_min': ap_min_mm, 'ap_max': ap_max_mm,
        'dv_min': dv_min_mm, 'dv_max': dv_max_mm,
        'lr_min': lr_min_mm, 'lr_max': lr_max_mm,
        'voxel_size_um': voxel_size_um,
        'ap_size': ap_size, 'dv_size': dv_size, 'lr_size': lr_size
    }

def create_slice_visualizations(voxel_counts, grid_info, output_dir="voxel_visualizations"):
    """Create 2D slice visualizations of voxel density."""
    
    os.makedirs(output_dir, exist_ok=True)
    print("Creating slice-by-slice visualizations...")
    
    ap_size, dv_size, lr_size = grid_info['ap_size'], grid_info['dv_size'], grid_info['lr_size']
    
    # Create AP slices (coronal sections)
    fig, axes = plt.subplots(2, 5, figsize=(25, 10))
    axes = axes.flatten()
    
    slice_indices = np.linspace(0, ap_size-1, 10, dtype=int)
    
    for i, ap_idx in enumerate(slice_indices):
        if i >= len(axes):
            break
            
        # Get slice data
        slice_data = voxel_counts[ap_idx, :, :]
        
        # Create heatmap
        im = axes[i].imshow(slice_data, cmap='hot', aspect='equal', origin='lower')
        axes[i].set_title(f'AP Slice {ap_idx} (AP={grid_info["ap_min"] + ap_idx * grid_info["voxel_size_um"]:.0f}μm)')
        axes[i].set_xlabel('LR (Left-Right)')
        axes[i].set_ylabel('DV (Dorsal-Ventral)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Channel Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'ap_slices_coronal.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'ap_slices_coronal.png')}")
    plt.close()
    
    # Create DV slices (horizontal sections)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    slice_indices = np.linspace(0, dv_size-1, 6, dtype=int)
    
    for i, dv_idx in enumerate(slice_indices):
        if i >= len(axes):
            break
            
        # Get slice data
        slice_data = voxel_counts[:, dv_idx, :]
        
        # Create heatmap
        im = axes[i].imshow(slice_data, cmap='hot', aspect='equal', origin='lower')
        axes[i].set_title(f'DV Slice {dv_idx} (DV={grid_info["dv_min"] + dv_idx * grid_info["voxel_size_um"]:.0f}μm)')
        axes[i].set_xlabel('LR (Left-Right)')
        axes[i].set_ylabel('AP (Anterior-Posterior)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Channel Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'dv_slices_horizontal.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'dv_slices_horizontal.png')}")
    plt.close()
    
    # Create LR slices (sagittal sections)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    slice_indices = np.linspace(0, lr_size-1, 6, dtype=int)
    
    for i, lr_idx in enumerate(slice_indices):
        if i >= len(axes):
            break
            
        # Get slice data
        slice_data = voxel_counts[:, :, lr_idx]
        
        # Create heatmap
        im = axes[i].imshow(slice_data, cmap='hot', aspect='equal', origin='lower')
        axes[i].set_title(f'LR Slice {lr_idx} (LR={grid_info["lr_min"] + lr_idx * grid_info["voxel_size_um"]:.0f}μm)')
        axes[i].set_xlabel('DV (Dorsal-Ventral)')
        axes[i].set_ylabel('AP (Anterior-Posterior)')
        
        # Add colorbar
        plt.colorbar(im, ax=axes[i], label='Channel Count')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'lr_slices_sagittal.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'lr_slices_sagittal.png')}")
    plt.close()

def create_3d_voxel_visualization(voxel_counts, grid_info, output_dir="voxel_visualizations"):
    """Create 3D visualization of voxel density."""
    
    print("Creating 3D voxel visualization...")
    
    # Find voxels with channels
    occupied_voxels = np.where(voxel_counts > 0)
    
    if len(occupied_voxels[0]) == 0:
        print("No occupied voxels found!")
        return
    
    # Get coordinates of occupied voxels
    ap_coords = occupied_voxels[0] * grid_info['voxel_size_um'] + grid_info['ap_min']
    dv_coords = occupied_voxels[1] * grid_info['voxel_size_um'] + grid_info['dv_min']
    lr_coords = occupied_voxels[2] * grid_info['voxel_size_um'] + grid_info['lr_min']
    
    # Get channel counts
    channel_counts = voxel_counts[occupied_voxels]
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot voxels with size and color based on channel count
    scatter = ax.scatter(ap_coords, dv_coords, lr_coords, 
                        c=channel_counts, s=channel_counts*2, 
                        cmap='hot', alpha=0.7, edgecolors='black', linewidth=0.1)
    
    # Set labels
    ax.set_xlabel('Anterior-Posterior (μm)', fontsize=12)
    ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=12)
    ax.set_zlabel('Left-Right (μm)', fontsize=12)
    ax.set_title('3D Channel Density - 1mm³ Voxels', fontsize=14)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Channels per Voxel', rotation=270, labelpad=15)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '3d_voxel_density.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, '3d_voxel_density.png')}")
    plt.close()

def create_density_histogram(voxel_counts, output_dir="voxel_visualizations"):
    """Create histogram of voxel density distribution."""
    
    print("Creating density distribution histogram...")
    
    # Get non-zero voxel counts
    occupied_counts = voxel_counts[voxel_counts > 0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Linear scale histogram
    ax1.hist(occupied_counts, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('Channels per Voxel')
    ax1.set_ylabel('Number of Voxels')
    ax1.set_title('Distribution of Channel Density (Linear Scale)')
    ax1.grid(True, alpha=0.3)
    
    # Log scale histogram
    ax2.hist(occupied_counts, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax2.set_xlabel('Channels per Voxel')
    ax2.set_ylabel('Number of Voxels')
    ax2.set_title('Distribution of Channel Density (Log Scale)')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'density_distribution.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(output_dir, 'density_distribution.png')}")
    plt.close()

def print_voxel_statistics(voxel_counts, grid_info, probe_stats):
    """Print comprehensive voxel statistics."""
    print("\n" + "="*80)
    print("3D VOXEL DENSITY STATISTICS")
    print("="*80)
    
    # Overall statistics
    total_voxels = voxel_counts.size
    occupied_voxels = np.sum(voxel_counts > 0)
    max_density = np.max(voxel_counts)
    mean_density = np.mean(voxel_counts[voxel_counts > 0]) if occupied_voxels > 0 else 0
    
    print(f"\nVoxel Grid Statistics:")
    print(f"  Total voxels: {total_voxels:,}")
    print(f"  Occupied voxels: {occupied_voxels:,} ({occupied_voxels/total_voxels*100:.1f}%)")
    print(f"  Empty voxels: {total_voxels - occupied_voxels:,} ({(total_voxels - occupied_voxels)/total_voxels*100:.1f}%)")
    print(f"  Max channels per voxel: {max_density}")
    print(f"  Mean channels per occupied voxel: {mean_density:.1f}")
    
    # Density distribution
    occupied_counts = voxel_counts[voxel_counts > 0]
    if len(occupied_counts) > 0:
        print(f"  Median channels per voxel: {np.median(occupied_counts):.1f}")
        print(f"  Std channels per voxel: {np.std(occupied_counts):.1f}")
        
        # Percentiles
        percentiles = [25, 50, 75, 90, 95, 99]
        print(f"  Density percentiles:")
        for p in percentiles:
            val = np.percentile(occupied_counts, p)
            print(f"    {p}th percentile: {val:.1f} channels/voxel")
    
    # Per-probe statistics
    print(f"\nPer-Probe Statistics:")
    for probe_id, stats in probe_stats.items():
        print(f"  Probe {probe_id}: {stats['unique_channels']} channels")
    
    # Grid dimensions
    print(f"\nGrid Dimensions:")
    print(f"  AP: {grid_info['ap_size']} voxels ({grid_info['ap_size']} mm)")
    print(f"  DV: {grid_info['dv_size']} voxels ({grid_info['dv_size']} mm)")
    print(f"  LR: {grid_info['lr_size']} voxels ({grid_info['lr_size']} mm)")
    print(f"  Voxel size: {grid_info['voxel_size_um']/1000:.1f} mm³")

def main():
    parser = argparse.ArgumentParser(description='3D Channel Voxel Density Visualization')
    parser.add_argument('input_path', help='Path to directory containing pickle files and CSV files')
    parser.add_argument('--output-dir', default='voxel_visualizations', 
                       help='Output directory for visualizations (default: voxel_visualizations)')
    parser.add_argument('--voxel-size', type=float, default=1.0,
                       help='Voxel size in mm (default: 1.0)')
    parser.add_argument('--include-hippocampus', action='store_true',
                       help='Include hippocampus channels (default: exclude)')
    
    args = parser.parse_args()
    
    print("3D Channel Voxel Density Visualization")
    print("="*50)
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Voxel size: {args.voxel_size}mm³")
    print(f"Include hippocampus: {args.include_hippocampus}")
    
    try:
        # Load coordinate lookup
        coord_lookup = load_coordinate_lookup(args.input_path)
        
        # Load pickle files
        pickle_files = load_pickle_files(args.input_path)
        
        if not pickle_files:
            print("No pickle files found!")
            return
        
        # Collect all CCF coordinates
        all_coordinates, probe_stats = collect_all_ccf_coordinates(
            pickle_files, coord_lookup, exclude_hippocampus=not args.include_hippocampus)
        
        if not all_coordinates:
            print("No CCF coordinates found!")
            return
        
        # Create voxel grid
        voxel_counts, grid_info = create_voxel_grid(all_coordinates, args.voxel_size)
        
        # Print statistics
        print_voxel_statistics(voxel_counts, grid_info, probe_stats)
        
        # Create visualizations
        create_slice_visualizations(voxel_counts, grid_info, args.output_dir)
        create_3d_voxel_visualization(voxel_counts, grid_info, args.output_dir)
        create_density_histogram(voxel_counts, args.output_dir)
        
        print(f"\nVoxel visualization completed! Check the '{args.output_dir}' directory for images.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
