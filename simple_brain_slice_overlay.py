#!/usr/bin/env python3
"""
Simple Brain Slice Overlay (No AllenSDK Required)
================================================
Creates a coronal slice visualization with channel voxels overlaid on a simple brain outline.
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

    coord_lookup: dict = {}
    for _, row in merged_df.iterrows():
        key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
        coord_lookup[key] = {
            'ap': row['anterior_posterior_ccf_coordinate_y'],
            'dv': row['dorsal_ventral_ccf_coordinate_y'],
            'lr': row['left_right_ccf_coordinate_y'],
        }
    return coord_lookup

def extract_unique_channel_coords(pickle_file: str, coord_lookup: dict):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)
    unique = {}
    for entry in tqdm(data, desc=f"Loading {os.path.basename(pickle_file)}"):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            _, label = entry
            parts = label.split('_')
            if len(parts) < 4:
                continue
            key = (parts[0], parts[2], parts[3])
            if key in unique:
                continue
            lk = (parts[0], parts[2], parts[3])
            if lk in coord_lookup:
                c = coord_lookup[lk]
                unique[key] = (float(c['ap']), float(c['dv']), float(c['lr']))
    return np.array(list(unique.values()), dtype=float)

def create_simple_brain_outline(coords_um):
    """Create a simple brain outline based on actual coordinate ranges."""
    # Get actual coordinate ranges
    dv_min, dv_max = coords_um[:, 1].min(), coords_um[:, 1].max()
    lr_min, lr_max = coords_um[:, 2].min(), coords_um[:, 2].max()
    
    # Create a brain outline that encompasses the actual data
    # Add some padding around the data
    dv_padding = (dv_max - dv_min) * 0.2
    lr_padding = (lr_max - lr_min) * 0.2
    
    dv_center = (dv_min + dv_max) / 2
    lr_center = (lr_min + lr_max) / 2
    
    # Create an ellipse that covers the data range
    theta = np.linspace(0, 2*np.pi, 100)
    a = (dv_max - dv_min) / 2 + dv_padding
    b = (lr_max - lr_min) / 2 + lr_padding
    
    x = dv_center + a * np.cos(theta)
    y = lr_center + b * np.sin(theta)
    
    return x, y

def overlay_slice_simple(coords_um: np.ndarray, slice_ap_um: float, tol_um: float, output_path: str, title_note: str = ""):
    """Create a simple coronal slice visualization without Allen atlas."""
    
    # Print coordinate ranges for debugging
    print(f"AP range: {coords_um[:, 0].min():.1f} to {coords_um[:, 0].max():.1f}")
    print(f"DV range: {coords_um[:, 1].min():.1f} to {coords_um[:, 1].max():.1f}")
    print(f"LR range: {coords_um[:, 2].min():.1f} to {coords_um[:, 2].max():.1f}")
    print(f"Slice AP: {slice_ap_um:.1f}, tolerance: ±{tol_um:.1f}")
    
    # Filter channels within slice tolerance
    mask = np.abs(coords_um[:, 0] - slice_ap_um) <= tol_um
    if not np.any(mask):
        print("No channel voxels within slice tolerance.")
        return
    
    slice_coords = coords_um[mask]
    print(f"Channels in slice: {len(slice_coords)}")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Create simple brain outline based on actual data ranges
    brain_x, brain_y = create_simple_brain_outline(coords_um)
    ax.plot(brain_x, brain_y, 'k-', linewidth=2, label='Brain Outline')
    ax.fill(brain_x, brain_y, color='lightgray', alpha=0.3)
    
    # Plot channel locations
    ax.scatter(slice_coords[:, 1], slice_coords[:, 2], 
              c='red', s=50, alpha=0.8, edgecolors='black', linewidth=0.5,
              label=f'Channels in ±{tol_um:.0f}µm slice (N={len(slice_coords)})')
    
    # Set labels and title
    ax.set_xlabel('Dorsal-Ventral (μm)', fontsize=12)
    ax.set_ylabel('Left-Right (μm)', fontsize=12)
    ax.set_title(f'Coronal Slice at AP ≈ {slice_ap_um:.0f} μm {title_note}', fontsize=14)
    
    # Set equal aspect ratio
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Simple brain slice overlay with channel voxels')
    parser.add_argument('pickle_file', help='Path to pickle file')
    parser.add_argument('input_path', help='Directory containing joined.csv and channels.csv')
    parser.add_argument('--slice-ap', type=float, default=None, help='AP slice (μm). Default: center of channel AP range')
    parser.add_argument('--tol-um', type=float, default=50.0, help='Slice tolerance (μm) to include nearby channels (default 50)')
    parser.add_argument('--output', default=None, help='Output image path')
    args = parser.parse_args()

    coord_lookup = load_coordinate_lookup(args.input_path)
    coords = extract_unique_channel_coords(args.pickle_file, coord_lookup)
    if coords.size == 0:
        print('No channel coordinates found.')
        return

    # Choose slice
    ap_min, ap_max = coords[:, 0].min(), coords[:, 0].max()
    slice_ap = args.slice_ap if args.slice_ap is not None else (ap_min + ap_max) / 2.0

    base = os.path.splitext(os.path.basename(args.pickle_file))[0]
    out = args.output or f"{base}_simple_coronal_slice.png"
    overlay_slice_simple(coords, slice_ap, args.tol_um, out, title_note=f"(Total N={coords.shape[0]} ch)")

if __name__ == '__main__':
    main()
