#!/usr/bin/env python3
"""
Brain Cube Histogram Visualization
==================================
Divides the mouse brain into 1mm³ cubes and creates a 3D histogram
showing how many channels fall into each cube.
Shows slice-by-slice views and 3D visualization.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm

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

def create_cube_histogram(coords_dict, cube_size=1000.0):
    """Create a 3D histogram of channel density in 1mm³ cubes."""
    print(f"Creating cube histogram with {cube_size}μm cubes...")
    
    # Get all coordinates
    all_coords = np.concatenate([coords for coords in coords_dict.values()])
    
    if len(all_coords) == 0:
        print("No coordinates found")
        return None, None, None
    
    # Calculate coordinate ranges
    ap_min, ap_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    dv_min, dv_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    lr_min, lr_max = all_coords[:, 2].min(), all_coords[:, 2].max()
    
    print(f"Coordinate ranges:")
    print(f"  AP: [{ap_min:.0f}, {ap_max:.0f}] μm (range: {ap_max-ap_min:.0f} μm)")
    print(f"  DV: [{dv_min:.0f}, {dv_max:.0f}] μm (range: {dv_max-dv_min:.0f} μm)")
    print(f"  LR: [{lr_min:.0f}, {lr_max:.0f}] μm (range: {lr_max-lr_min:.0f} μm)")
    
    # Calculate number of cubes needed
    ap_cubes = int(np.ceil((ap_max - ap_min) / cube_size))
    dv_cubes = int(np.ceil((dv_max - dv_min) / cube_size))
    lr_cubes = int(np.ceil((lr_max - lr_min) / cube_size))
    
    print(f"Cube grid dimensions: {ap_cubes} x {dv_cubes} x {lr_cubes} = {ap_cubes*dv_cubes*lr_cubes} total cubes")
    
    # Initialize histogram
    histogram = np.zeros((ap_cubes, dv_cubes, lr_cubes), dtype=int)
    
    # Count channels in each cube
    for coords in coords_dict.values():
        for coord in coords:
            ap, dv, lr = coord
            
            # Calculate cube indices
            ap_idx = int((ap - ap_min) // cube_size)
            dv_idx = int((dv - dv_min) // cube_size)
            lr_idx = int((lr - lr_min) // cube_size)
            
            # Ensure indices are within bounds
            ap_idx = max(0, min(ap_idx, ap_cubes - 1))
            dv_idx = max(0, min(dv_idx, dv_cubes - 1))
            lr_idx = max(0, min(lr_idx, lr_cubes - 1))
            
            histogram[ap_idx, dv_idx, lr_idx] += 1
    
    # Calculate statistics
    occupied_cubes = np.sum(histogram > 0)
    max_channels_per_cube = np.max(histogram)
    total_channels = np.sum(histogram)
    
    print(f"Histogram statistics:")
    print(f"  Total channels: {total_channels}")
    print(f"  Occupied cubes: {occupied_cubes}")
    print(f"  Max channels per cube: {max_channels_per_cube}")
    print(f"  Average channels per occupied cube: {total_channels/occupied_cubes:.1f}")
    
    return histogram, (ap_min, ap_max, dv_min, dv_max, lr_min, lr_max), (ap_cubes, dv_cubes, lr_cubes)

def plot_slice_views(histogram, coord_ranges, cube_dims, output_dir, cube_size=1000.0):
    """Create slice-by-slice views of the histogram."""
    print("Creating slice-by-slice views...")
    
    ap_min, ap_max, dv_min, dv_max, lr_min, lr_max = coord_ranges
    ap_cubes, dv_cubes, lr_cubes = cube_dims
    
    # Create AP slices (coronal view)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Select 6 representative AP slices
    ap_slice_indices = np.linspace(0, ap_cubes-1, 6).astype(int)
    
    for i, ap_idx in enumerate(ap_slice_indices):
        ax = axes[i]
        
        # Get slice data
        slice_data = histogram[ap_idx, :, :]
        
        # Create heatmap
        im = ax.imshow(slice_data.T, cmap='hot', aspect='equal', origin='lower')
        
        # Set labels
        ax.set_title(f'AP Slice {i+1} (AP = {ap_min + ap_idx*cube_size:.0f} μm)')
        ax.set_xlabel('DV (μm)')
        ax.set_ylabel('LR (μm)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Channels per cube')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'ap_slices_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Create DV slices (horizontal view)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Select 6 representative DV slices
    dv_slice_indices = np.linspace(0, dv_cubes-1, 6).astype(int)
    
    for i, dv_idx in enumerate(dv_slice_indices):
        ax = axes[i]
        
        # Get slice data
        slice_data = histogram[:, dv_idx, :]
        
        # Create heatmap
        im = ax.imshow(slice_data.T, cmap='hot', aspect='equal', origin='lower')
        
        # Set labels
        ax.set_title(f'DV Slice {i+1} (DV = {dv_min + dv_idx*cube_size:.0f} μm)')
        ax.set_xlabel('AP (μm)')
        ax.set_ylabel('LR (μm)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Channels per cube')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'dv_slices_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()
    
    # Create LR slices (sagittal view)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Select 6 representative LR slices
    lr_slice_indices = np.linspace(0, lr_cubes-1, 6).astype(int)
    
    for i, lr_idx in enumerate(lr_slice_indices):
        ax = axes[i]
        
        # Get slice data
        slice_data = histogram[:, :, lr_idx]
        
        # Create heatmap
        im = ax.imshow(slice_data.T, cmap='hot', aspect='equal', origin='lower')
        
        # Set labels
        ax.set_title(f'LR Slice {i+1} (LR = {lr_min + lr_idx*cube_size:.0f} μm)')
        ax.set_xlabel('AP (μm)')
        ax.set_ylabel('DV (μm)')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Channels per cube')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'lr_slices_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_3d_histogram(histogram, coord_ranges, cube_dims, output_dir, cube_size=1000.0):
    """Create 3D visualization of the histogram."""
    print("Creating 3D histogram visualization...")
    
    ap_min, ap_max, dv_min, dv_max, lr_min, lr_max = coord_ranges
    ap_cubes, dv_cubes, lr_cubes = cube_dims
    
    # Find occupied cubes
    occupied_indices = np.where(histogram > 0)
    
    if len(occupied_indices[0]) == 0:
        print("No occupied cubes found")
        return
    
    # Get coordinates of occupied cubes
    ap_coords = ap_min + occupied_indices[0] * cube_size
    dv_coords = dv_min + occupied_indices[1] * cube_size
    lr_coords = lr_min + occupied_indices[2] * cube_size
    
    # Get channel counts
    channel_counts = histogram[occupied_indices]
    
    # Create 3D plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create scatter plot with size proportional to channel count
    scatter = ax.scatter(ap_coords, dv_coords, lr_coords, 
                        c=channel_counts, s=channel_counts*50, 
                        cmap='hot', alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Set labels
    ax.set_xlabel('AP (μm)')
    ax.set_ylabel('DV (μm)')
    ax.set_zlabel('LR (μm)')
    ax.set_title('3D Channel Density Histogram\n(Size ∝ Channel Count)')
    
    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Channels per cube')
    
    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, '3d_histogram.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_histogram_statistics(histogram, output_dir):
    """Create histogram statistics plots."""
    print("Creating histogram statistics...")
    
    # Flatten histogram for statistics
    flat_hist = histogram.flatten()
    occupied_counts = flat_hist[flat_hist > 0]
    
    # Create statistics plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Histogram of channel counts per cube
    ax1 = axes[0, 0]
    ax1.hist(occupied_counts, bins=20, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Channels per cube')
    ax1.set_ylabel('Number of cubes')
    ax1.set_title('Distribution of Channel Counts per Cube')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax2 = axes[0, 1]
    sorted_counts = np.sort(occupied_counts)[::-1]
    cumulative = np.cumsum(sorted_counts)
    ax2.plot(range(len(cumulative)), cumulative)
    ax2.set_xlabel('Cube rank (by channel count)')
    ax2.set_ylabel('Cumulative channels')
    ax2.set_title('Cumulative Channel Distribution')
    ax2.grid(True, alpha=0.3)
    
    # Box plot
    ax3 = axes[1, 0]
    ax3.boxplot(occupied_counts)
    ax3.set_ylabel('Channels per cube')
    ax3.set_title('Channel Count Distribution')
    ax3.grid(True, alpha=0.3)
    
    # Statistics text
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    stats_text = f"""
    Histogram Statistics:
    
    Total cubes: {histogram.size:,}
    Occupied cubes: {len(occupied_counts):,}
    Empty cubes: {histogram.size - len(occupied_counts):,}
    
    Min channels/cube: {np.min(occupied_counts)}
    Max channels/cube: {np.max(occupied_counts)}
    Mean channels/cube: {np.mean(occupied_counts):.1f}
    Median channels/cube: {np.median(occupied_counts):.1f}
    Std channels/cube: {np.std(occupied_counts):.1f}
    
    Occupancy rate: {len(occupied_counts)/histogram.size*100:.1f}%
    """
    
    ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=12, fontfamily='monospace')
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'histogram_statistics.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Brain Cube Histogram Visualization')
    parser.add_argument('input_path', help='Directory containing pickle files and CSV files')
    parser.add_argument('--session-id', help='Specific session ID to process (if not provided, processes first session found)')
    parser.add_argument('--output-dir', default='brain_cube_histogram', help='Output directory for plots')
    parser.add_argument('--cube-size', type=float, default=1000.0, help='Cube size in micrometers (default: 1000 = 1mm)')
    
    args = parser.parse_args()
    
    # Find pickle files
    pickle_files = []
    for file in os.listdir(args.input_path):
        if file.endswith('.pickle'):
            pickle_files.append(os.path.join(args.input_path, file))
    
    if not pickle_files:
        print(f"No pickle files found in {args.input_path}")
        return
    
    print(f"Found {len(pickle_files)} pickle files")
    
    # Load coordinate lookup
    coord_lookup = load_coordinate_lookup(args.input_path)
    print(f"Loaded coordinate lookup with {len(coord_lookup)} entries")
    
    # Extract coordinates for each pickle
    coords_dict = {}
    for pickle_file in pickle_files:
        session_id, probe_id, coords = extract_channel_coords_from_pickle(pickle_file, coord_lookup)
        if coords is not None and len(coords) > 0:
            coords_dict[probe_id] = coords
            print(f"Probe {probe_id}: {len(coords)} unique channels")
    
    if not coords_dict:
        print("No valid coordinate data found")
        return
    
    # Group by session if specified
    if args.session_id:
        # Filter to only probes from the specified session
        filtered_coords = {}
        for pickle_file in pickle_files:
            session_id, probe_id, coords = extract_channel_coords_from_pickle(pickle_file, coord_lookup)
            if session_id == args.session_id and coords is not None and len(coords) > 0:
                filtered_coords[probe_id] = coords
        coords_dict = filtered_coords
    
    if not coords_dict:
        print(f"No data found for session {args.session_id}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create cube histogram
    histogram, coord_ranges, cube_dims = create_cube_histogram(coords_dict, args.cube_size)
    
    if histogram is None:
        print("Failed to create histogram")
        return
    
    # Generate visualizations
    plot_slice_views(histogram, coord_ranges, cube_dims, args.output_dir, args.cube_size)
    plot_3d_histogram(histogram, coord_ranges, cube_dims, args.output_dir, args.cube_size)
    plot_histogram_statistics(histogram, args.output_dir)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
