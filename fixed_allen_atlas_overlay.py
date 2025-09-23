#!/usr/bin/env python3
"""
Fixed Allen Atlas Overlay
========================
Uses the correct AllenSDK MouseConnectivityCache to load the annotation volume.
Based on: https://allensdk.readthedocs.io/en/stable/_static/examples/nb/mouse_connectivity.html
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

def extract_channel_coords(pickle_file: str, coord_lookup: dict):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    
    coords = []
    for entry in tqdm(data, desc=f"Loading {os.path.basename(pickle_file)}"):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            _, label = entry
            parts = label.split('_')
            if len(parts) < 4:
                continue
            key = (parts[0], parts[2], parts[3])
            if key in coord_lookup:
                c = coord_lookup[key]
                coords.append((float(c['ap']), float(c['dv']), float(c['lr'])))
    
    return np.array(coords, dtype=float)

def load_allen_atlas():
    """Load Allen mouse brain atlas using MouseConnectivityCache."""
    try:
        from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
        
        # Create cache with manifest file
        mcc = MouseConnectivityCache(manifest_file='allen_atlas_manifest.json')
        
        # Get annotation volume and template
        print("Loading Allen atlas...")
        annot, annot_info = mcc.get_annotation_volume()
        template, template_info = mcc.get_template_volume()
        
        print(f"Annotation volume shape: {annot.shape}")
        print(f"Template volume shape: {template.shape}")
        print(f"Annotation info: {annot_info}")
        
        return annot, template, annot_info, template_info
        
    except Exception as e:
        print(f"Failed to load Allen atlas: {e}")
        return None, None, None, None

def ccf_to_voxel_coords(ap, dv, lr, annot_info):
    """Convert CCF coordinates to voxel coordinates."""
    # CCF coordinates are in micrometers
    # Voxel size is 25 micrometers according to the documentation
    voxel_size = 25.0
    
    # Convert to voxel coordinates
    ap_voxel = int(ap / voxel_size)
    dv_voxel = int(dv / voxel_size)
    lr_voxel = int(lr / voxel_size)
    
    return ap_voxel, dv_voxel, lr_voxel

def create_coronal_slice_with_atlas(coords_um, slice_ap_um, tol_um, annot, template, annot_info, output_path):
    """Create coronal slice visualization with Allen atlas background."""
    
    # Filter channels within slice tolerance
    mask = np.abs(coords_um[:, 0] - slice_ap_um) <= tol_um
    if not np.any(mask):
        print("No channel voxels within slice tolerance.")
        return
    
    slice_coords = coords_um[mask]
    print(f"Channels in slice: {len(slice_coords)}")
    
    # Convert CCF coordinates to voxel coordinates
    voxel_coords = []
    for coord in slice_coords:
        ap_vox, dv_vox, lr_vox = ccf_to_voxel_coords(coord[0], coord[1], coord[2], annot_info)
        voxel_coords.append((ap_vox, dv_vox, lr_vox))
    
    voxel_coords = np.array(voxel_coords)
    
    # Find the slice index closest to our AP coordinate
    slice_idx = int(slice_ap_um / 25.0)  # 25um voxel size
    
    # Ensure slice index is within bounds
    if slice_idx >= annot.shape[0]:
        slice_idx = annot.shape[0] - 1
    elif slice_idx < 0:
        slice_idx = 0
    
    # Create the visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Template slice
    axes[0].imshow(template[slice_idx, :, :], cmap='gray', aspect='equal')
    axes[0].set_title('Template Slice')
    axes[0].set_xlabel('LR (voxels)')
    axes[0].set_ylabel('DV (voxels)')
    
    # Annotation slice
    axes[1].imshow(annot[slice_idx, :, :], cmap='tab20', aspect='equal')
    axes[1].set_title('Annotation Slice')
    axes[1].set_xlabel('LR (voxels)')
    axes[1].set_ylabel('DV (voxels)')
    
    # Overlay channels on annotation
    axes[2].imshow(annot[slice_idx, :, :], cmap='gray', aspect='equal', alpha=0.7)
    
    # Plot channel locations
    for coord in voxel_coords:
        if 0 <= coord[0] < annot.shape[0] and 0 <= coord[1] < annot.shape[1] and 0 <= coord[2] < annot.shape[2]:
            axes[2].scatter(coord[2], coord[1], c='red', s=100, alpha=0.9, edgecolors='darkred')
    
    axes[2].set_title(f'Channels Overlay (N={len(slice_coords)})')
    axes[2].set_xlabel('LR (voxels)')
    axes[2].set_ylabel('DV (voxels)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Allen atlas overlay with channel voxels')
    parser.add_argument('pickle_file', help='Path to pickle file')
    parser.add_argument('input_path', help='Directory containing joined.csv and channels.csv')
    parser.add_argument('--slice-ap', type=float, default=None, help='AP slice (μm). Default: center of channel AP range')
    parser.add_argument('--tol-um', type=float, default=50.0, help='Slice tolerance (μm) to include nearby channels (default 50)')
    parser.add_argument('--output', default=None, help='Output image path')
    args = parser.parse_args()

    # Load coordinate data
    coord_lookup = load_coordinate_lookup(args.input_path)
    coords = extract_channel_coords(args.pickle_file, coord_lookup)
    
    if coords.size == 0:
        print('No channel coordinates found.')
        return

    print(f"Found {len(coords)} channel coordinates")
    print(f"AP range: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f}")
    print(f"DV range: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f}")
    print(f"LR range: {coords[:, 2].min():.1f} to {coords[:, 2].max():.1f}")

    # Load Allen atlas
    annot, template, annot_info, template_info = load_allen_atlas()
    
    if annot is None:
        print("Failed to load Allen atlas. Creating simple visualization instead.")
        # Fallback to simple visualization
        from simple_data_plot import plot_2d_projections
        base = os.path.splitext(os.path.basename(args.pickle_file))[0]
        plot_2d_projections(coords, f"{base}_fallback_2d.png")
        return

    # Choose slice
    ap_min, ap_max = coords[:, 0].min(), coords[:, 0].max()
    slice_ap = args.slice_ap if args.slice_ap is not None else (ap_min + ap_max) / 2.0

    base = os.path.splitext(os.path.basename(args.pickle_file))[0]
    out = args.output or f"{base}_allen_atlas_overlay.png"
    
    create_coronal_slice_with_atlas(coords, slice_ap, args.tol_um, annot, template, annot_info, out)

if __name__ == '__main__':
    main()
