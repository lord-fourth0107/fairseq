#!/usr/bin/env python3
"""
Brain Grid Overlay Visualization
================================
Uses AllenSDK to load mouse brain template, overlays 250μm grid,
and colors grid cells containing channels from pickle files.
Shows horizontal and vertical views with magnified versions.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
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

def load_allen_atlas():
    """Load Allen mouse brain atlas using MouseConnectivityCache."""
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
        
        mcc = MouseConnectivityCache(manifest_file='allen_atlas_manifest.json')
        
        print("Loading Allen atlas...")
        annot, annot_info = mcc.get_annotation_volume()
        template, template_info = mcc.get_template_volume()
        
        print(f"Annotation volume shape: {annot.shape}")
        print(f"Template volume shape: {template.shape}")
        
        return annot, template, annot_info, template_info
        
    except Exception as e:
        print(f"Failed to load Allen atlas: {e}")
        return None, None, None, None

def create_grid_overlay(template, coords_dict, output_dir, grid_size=250.0, voxel_size=25.0):
    """Create grid overlay visualizations for horizontal and vertical views."""
    print(f"Creating grid overlay with {grid_size}μm grid cells...")
    
    # Calculate grid dimensions
    grid_voxels = int(grid_size / voxel_size)  # Number of voxels per grid cell
    
    # Get coordinate ranges
    all_coords = np.concatenate([coords for coords in coords_dict.values()])
    ap_min, ap_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    dv_min, dv_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    lr_min, lr_max = all_coords[:, 2].min(), all_coords[:, 2].max()
    
    print(f"Coordinate ranges: AP [{ap_min:.0f}, {ap_max:.0f}], DV [{dv_min:.0f}, {dv_max:.0f}], LR [{lr_min:.0f}, {lr_max:.0f}]")
    
    # Create horizontal view (AP-LR plane)
    create_horizontal_grid_view(template, coords_dict, output_dir, grid_size, voxel_size, 
                              ap_min, ap_max, lr_min, lr_max, grid_voxels)
    
    # Create vertical view (DV-LR plane)
    create_vertical_grid_view(template, coords_dict, output_dir, grid_size, voxel_size,
                             dv_min, dv_max, lr_min, lr_max, grid_voxels)

def create_horizontal_grid_view(template, coords_dict, output_dir, grid_size, voxel_size,
                               ap_min, ap_max, lr_min, lr_max, grid_voxels):
    """Create horizontal grid view (AP-LR plane)."""
    print("Creating horizontal grid view...")
    
    # Calculate slice position (middle of DV range)
    all_coords = np.concatenate([coords for coords in coords_dict.values()])
    dv_mid = (all_coords[:, 1].min() + all_coords[:, 1].max()) / 2
    slice_idx = int(dv_mid / voxel_size)
    slice_idx = max(0, min(slice_idx, template.shape[1] - 1))
    
    print(f"Using DV slice at index {slice_idx} (DV = {slice_idx * voxel_size:.0f}μm)")
    
    # Get template slice
    template_slice = template[:, slice_idx, :]
    
    # Create figure with single row layout
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Normal view
    ax1 = axes[0]
    ax1.imshow(template_slice, cmap='gray', aspect='equal')
    ax1.set_title('Horizontal View - Normal Scale')
    ax1.set_xlabel('LR (μm)')
    ax1.set_ylabel('AP (μm)')
    
    # Magnified view
    ax2 = axes[1]
    ax2.imshow(template_slice, cmap='gray', aspect='equal')
    ax2.set_title('Horizontal View - Magnified')
    ax2.set_xlabel('LR (μm)')
    ax2.set_ylabel('AP (μm)')
    
    # Add grid overlay to both views
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for j, (probe_id, coords) in enumerate(coords_dict.items()):
        if len(coords) == 0:
            continue
        
        color = colors[j % len(colors)]
        print(f"Processing probe {probe_id} with {len(coords)} channels")
        
        # Convert coordinates to voxel space
        voxel_coords = np.round(coords / voxel_size).astype(int)
        
        # Filter coordinates within slice tolerance
        mask = np.abs(voxel_coords[:, 1] - slice_idx) <= 2
        slice_coords = voxel_coords[mask]
        
        print(f"  Probe {probe_id}: {len(slice_coords)} channels within slice tolerance")
        
        if len(slice_coords) == 0:
            continue
        
        # Create grid cells for each coordinate
        for coord in slice_coords:
            ap_vox, dv_vox, lr_vox = coord
            
            # Calculate grid cell boundaries
            grid_ap = (ap_vox // grid_voxels) * grid_voxels
            grid_lr = (lr_vox // grid_voxels) * grid_voxels
            
            # Convert back to physical coordinates
            grid_ap_phys = grid_ap * voxel_size
            grid_lr_phys = grid_lr * voxel_size
            
            # Create rectangle for grid cell
            rect = Rectangle((grid_lr_phys, grid_ap_phys), grid_size, grid_size,
                           facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)
            
            # Add to magnified view with smaller tolerance
            if abs(dv_vox - slice_idx) <= 1:  # Tighter tolerance for magnified view
                rect2 = Rectangle((grid_lr_phys, grid_ap_phys), grid_size, grid_size,
                                facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                ax2.add_patch(rect2)
    
    # Set axis limits
    ax1.set_xlim(lr_min, lr_max)
    ax1.set_ylim(ap_min, ap_max)
    ax2.set_xlim(lr_min, lr_max)
    ax2.set_ylim(ap_min, ap_max)
    
    # Create legend
    legend_elements = []
    for j, probe_id in enumerate(coords_dict.keys()):
        if len(coords_dict[probe_id]) > 0:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[j % len(colors)], 
                                               alpha=0.6, label=f'Probe {probe_id}'))
    
    if legend_elements:
        ax1.legend(handles=legend_elements, loc='upper right')
        ax2.legend(handles=legend_elements, loc='upper right')
    
    # Add grid lines
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add grid size annotation
    ax1.text(0.02, 0.98, f'Grid size: {grid_size}μm', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.98, f'Grid size: {grid_size}μm', transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'horizontal_grid_overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_vertical_grid_view(template, coords_dict, output_dir, grid_size, voxel_size,
                            dv_min, dv_max, lr_min, lr_max, grid_voxels):
    """Create vertical grid view (DV-LR plane)."""
    print("Creating vertical grid view...")
    
    # Calculate slice position (middle of AP range)
    all_coords = np.concatenate([coords for coords in coords_dict.values()])
    ap_mid = (all_coords[:, 0].min() + all_coords[:, 0].max()) / 2
    slice_idx = int(ap_mid / voxel_size)
    slice_idx = max(0, min(slice_idx, template.shape[0] - 1))
    
    # Get template slice
    template_slice = template[slice_idx, :, :]
    
    # Create figure with single row layout
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Normal view
    ax1 = axes[0]
    ax1.imshow(template_slice, cmap='gray', aspect='equal')
    ax1.set_title('Vertical View - Normal Scale')
    ax1.set_xlabel('LR (μm)')
    ax1.set_ylabel('DV (μm)')
    
    # Magnified view
    ax2 = axes[1]
    ax2.imshow(template_slice, cmap='gray', aspect='equal')
    ax2.set_title('Vertical View - Magnified')
    ax2.set_xlabel('LR (μm)')
    ax2.set_ylabel('DV (μm)')
    
    # Add grid overlay to both views
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for j, (probe_id, coords) in enumerate(coords_dict.items()):
        if len(coords) == 0:
            continue
        
        color = colors[j % len(colors)]
        print(f"Processing probe {probe_id} with {len(coords)} channels")
        
        # Convert coordinates to voxel space
        voxel_coords = np.round(coords / voxel_size).astype(int)
        
        # Filter coordinates within slice tolerance
        mask = np.abs(voxel_coords[:, 0] - slice_idx) <= 2
        slice_coords = voxel_coords[mask]
        
        print(f"  Probe {probe_id}: {len(slice_coords)} channels within slice tolerance")
        
        if len(slice_coords) == 0:
            continue
        
        # Create grid cells for each coordinate
        for coord in slice_coords:
            ap_vox, dv_vox, lr_vox = coord
            
            # Calculate grid cell boundaries
            grid_dv = (dv_vox // grid_voxels) * grid_voxels
            grid_lr = (lr_vox // grid_voxels) * grid_voxels
            
            # Convert back to physical coordinates
            grid_dv_phys = grid_dv * voxel_size
            grid_lr_phys = grid_lr * voxel_size
            
            # Create rectangle for grid cell
            rect = Rectangle((grid_lr_phys, grid_dv_phys), grid_size, grid_size,
                           facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
            ax1.add_patch(rect)
            
            # Add to magnified view with smaller tolerance
            if abs(ap_vox - slice_idx) <= 1:  # Tighter tolerance for magnified view
                rect2 = Rectangle((grid_lr_phys, grid_dv_phys), grid_size, grid_size,
                                facecolor=color, alpha=0.6, edgecolor='black', linewidth=0.5)
                ax2.add_patch(rect2)
    
    # Set axis limits
    ax1.set_xlim(lr_min, lr_max)
    ax1.set_ylim(dv_min, dv_max)
    ax2.set_xlim(lr_min, lr_max)
    ax2.set_ylim(dv_min, dv_max)
    
    # Create legend
    legend_elements = []
    for j, probe_id in enumerate(coords_dict.keys()):
        if len(coords_dict[probe_id]) > 0:
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[j % len(colors)], 
                                               alpha=0.6, label=f'Probe {probe_id}'))
    
    if legend_elements:
        ax1.legend(handles=legend_elements, loc='upper right')
        ax2.legend(handles=legend_elements, loc='upper right')
    
    # Add grid lines
    ax1.grid(True, alpha=0.3, linewidth=0.5)
    ax2.grid(True, alpha=0.3, linewidth=0.5)
    
    # Add grid size annotation
    ax1.text(0.02, 0.98, f'Grid size: {grid_size}μm', transform=ax1.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(0.02, 0.98, f'Grid size: {grid_size}μm', transform=ax2.transAxes, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, 'vertical_grid_overlay.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Brain Grid Overlay Visualization')
    parser.add_argument('input_path', help='Directory containing pickle files and CSV files')
    parser.add_argument('--session-id', help='Specific session ID to process (if not provided, processes first session found)')
    parser.add_argument('--output-dir', default='brain_grid_overlay', help='Output directory for plots')
    parser.add_argument('--grid-size', type=float, default=250.0, help='Grid cell size in micrometers (default: 250)')
    
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
    
    # Load Allen atlas
    annot, template, annot_info, template_info = load_allen_atlas()
    
    if template is None:
        print("Failed to load Allen atlas")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate grid overlay visualizations
    create_grid_overlay(template, coords_dict, args.output_dir, args.grid_size)
    
    print(f"\nGrid overlay visualizations saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
