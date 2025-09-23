#!/usr/bin/env python3
"""
Probe CCF Grid Visualization
============================
Creates a grid of CCF coordinate visualizations for each probe with different colors.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_coordinate_lookup(input_path: str) -> dict:
    joined_path = os.path.join(input_path, 'joined.csv')
    channels_path = os.path.join(input_path, 'channels.csv')
    
    joined_df = pd.read_csv(joined_path)
    channels_df = pd.read_csv(channels_path)
    merged_df = pd.merge(
        joined_df,
        channels_df,
        left_on='probe_id',
        right_on='ecephys_probe_id',
        how='inner',
    )

    coord_lookup: dict = {}
    for _, row in merged_df.iterrows():
        key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
        coord_lookup[key] = {
            'ap': row['anterior_posterior_ccf_coordinate_y'],
            'dv': row['dorsal_ventral_ccf_coordinate_y'],
            'lr': row['left_right_ccf_coordinate_y'],
        }
    return coord_lookup

def extract_probe_channel_coords(pickle_file: str, coord_lookup: dict):
    """Extract coordinates grouped by probe."""
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    probe_coords = {}
    for entry in tqdm(data, desc=f"Loading {os.path.basename(pickle_file)}"):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            _, label = entry
            parts = label.split('_')
            if len(parts) < 4:
                continue
            
            session_id, count, probe_id, channel_id = parts[0], parts[1], parts[2], parts[3]
            key = (session_id, probe_id, channel_id)
            
            if key in coord_lookup:
                c = coord_lookup[key]
                coord = (float(c['ap']), float(c['dv']), float(c['lr']))
                
                if probe_id not in probe_coords:
                    probe_coords[probe_id] = []
                probe_coords[probe_id].append(coord)
    
    # Convert to numpy arrays
    for probe_id in probe_coords:
        probe_coords[probe_id] = np.array(probe_coords[probe_id], dtype=float)
    
    return probe_coords

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

def ccf_to_voxel_coords(ap, dv, lr, annot_info):
    """Convert CCF coordinates to voxel coordinates."""
    voxel_size = 25.0  # 25 micrometers per voxel
    ap_voxel = int(ap / voxel_size)
    dv_voxel = int(dv / voxel_size)
    lr_voxel = int(lr / voxel_size)
    return ap_voxel, dv_voxel, lr_voxel

def create_probe_grid_visualization(probe_coords, annot, template, annot_info, output_path):
    """Create a grid visualization showing each probe's channels overlaid on Allen atlas."""
    
    probe_ids = list(probe_coords.keys())
    n_probes = len(probe_ids)
    
    if n_probes == 0:
        print("No probe data found.")
        return
    
    # Define colors for each probe
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # Create subplots - 2 rows, 3 columns for up to 6 probes
    n_cols = min(3, n_probes)
    n_rows = (n_probes + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    
    # Handle single subplot case
    if n_probes == 1:
        axes_flat = [axes]
    else:
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
    
    for i, probe_id in enumerate(probe_ids):
        coords = probe_coords[probe_id]
        color = colors[i % len(colors)]
        
        ax = axes_flat[i]
        
        # Calculate slice index (middle of AP range)
        ap_min, ap_max = coords[:, 0].min(), coords[:, 0].max()
        slice_ap = (ap_min + ap_max) / 2.0
        slice_idx = int(slice_ap / 25.0)
        
        # Ensure slice index is within bounds
        slice_idx = max(0, min(slice_idx, annot.shape[0] - 1))
        
        # Show annotation slice
        ax.imshow(annot[slice_idx, :, :], cmap='gray', aspect='equal', alpha=0.7)
        
        # Convert coordinates to voxel space
        voxel_coords = []
        for coord in coords:
            ap_vox, dv_vox, lr_vox = ccf_to_voxel_coords(coord[0], coord[1], coord[2], annot_info)
            voxel_coords.append((ap_vox, dv_vox, lr_vox))
        
        voxel_coords = np.array(voxel_coords)
        
        # Plot channels
        for coord in voxel_coords:
            if (0 <= coord[0] < annot.shape[0] and 
                0 <= coord[1] < annot.shape[1] and 
                0 <= coord[2] < annot.shape[2]):
                ax.scatter(coord[2], coord[1], c=color, s=30, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        ax.set_title(f'Probe {probe_id}\n(N={len(coords)} channels)', fontsize=12)
        ax.set_xlabel('LR (voxels)')
        ax.set_ylabel('DV (voxels)')
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_probes, len(axes_flat)):
        axes_flat[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def create_combined_probe_visualization(probe_coords, annot, template, annot_info, output_path):
    """Create a single plot with all probes overlaid in different colors."""
    
    probe_ids = list(probe_coords.keys())
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Calculate overall slice index
    all_coords = np.concatenate(list(probe_coords.values()))
    ap_min, ap_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    slice_ap = (ap_min + ap_max) / 2.0
    slice_idx = int(slice_ap / 25.0)
    slice_idx = max(0, min(slice_idx, annot.shape[0] - 1))
    
    # Show annotation slice
    ax.imshow(annot[slice_idx, :, :], cmap='gray', aspect='equal', alpha=0.7)
    
    # Plot each probe's channels
    for i, probe_id in enumerate(probe_ids):
        coords = probe_coords[probe_id]
        color = colors[i % len(colors)]
        
        # Convert coordinates to voxel space
        voxel_coords = []
        for coord in coords:
            ap_vox, dv_vox, lr_vox = ccf_to_voxel_coords(coord[0], coord[1], coord[2], annot_info)
            voxel_coords.append((ap_vox, dv_vox, lr_vox))
        
        voxel_coords = np.array(voxel_coords)
        
        # Plot channels
        for coord in voxel_coords:
            if (0 <= coord[0] < annot.shape[0] and 
                0 <= coord[1] < annot.shape[1] and 
                0 <= coord[2] < annot.shape[2]):
                ax.scatter(coord[2], coord[1], c=color, s=30, alpha=0.8, 
                          edgecolors='black', linewidth=0.5, label=f'Probe {probe_id}')
    
    ax.set_title(f'All Probes Combined\n(AP slice at {slice_ap:.0f} μm)', fontsize=14)
    ax.set_xlabel('LR (voxels)')
    ax.set_ylabel('DV (voxels)')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Probe CCF grid visualization')
    parser.add_argument('pickle_file', help='Path to pickle file')
    parser.add_argument('input_path', help='Directory containing joined.csv and channels.csv')
    parser.add_argument('--output', default=None, help='Output image path')
    args = parser.parse_args()

    # Load coordinate data
    coord_lookup = load_coordinate_lookup(args.input_path)
    probe_coords = extract_probe_channel_coords(args.pickle_file, coord_lookup)
    
    if not probe_coords:
        print('No probe coordinates found.')
        return

    print(f"Found {len(probe_coords)} probes:")
    for probe_id, coords in probe_coords.items():
        print(f"  Probe {probe_id}: {len(coords)} channels")

    # Load Allen atlas
    annot, template, annot_info, template_info = load_allen_atlas()
    
    if annot is None:
        print("Failed to load Allen atlas.")
        return

    base = os.path.splitext(os.path.basename(args.pickle_file))[0]
    
    # Create grid visualization
    grid_output = args.output or f"{base}_probe_grid.png"
    create_probe_grid_visualization(probe_coords, annot, template, annot_info, grid_output)
    
    # Create combined visualization
    combined_output = f"{base}_probes_combined.png"
    create_combined_probe_visualization(probe_coords, annot, template, annot_info, combined_output)

if __name__ == '__main__':
    main()
