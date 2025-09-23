#!/usr/bin/env python3
"""
Realistic Brain Slice Overlay
============================
Creates a coronal slice visualization with channel voxels overlaid on a realistic mouse brain outline.
Uses actual mouse brain dimensions and CCF coordinate system.
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

def create_realistic_brain_outline():
    """Create a realistic mouse brain outline based on CCF dimensions."""
    # Mouse brain dimensions in CCF space (approximate)
    # AP: ~10mm, DV: ~6mm, LR: ~6mm
    # CCF coordinates are in micrometers
    
    # Create a more realistic brain shape
    theta = np.linspace(0, 2*np.pi, 200)
    
    # Main brain outline (ellipse)
    a = 3000  # DV radius (micrometers)
    b = 3000  # LR radius (micrometers)
    
    # Create the main brain outline
    x_main = a * np.cos(theta)
    y_main = b * np.sin(theta)
    
    # Add some brain structure details (simplified)
    # Cerebellum (lower part)
    theta_cb = np.linspace(np.pi, 2*np.pi, 50)
    x_cb = 0.7 * a * np.cos(theta_cb) - 500
    y_cb = 0.5 * b * np.sin(theta_cb) - 1000
    
    # Combine outlines
    x = np.concatenate([x_main, x_cb])
    y = np.concatenate([y_main, y_cb])
    
    return x, y

def create_brain_regions_outline():
    """Create a more detailed brain outline with major regions."""
    # This creates a more anatomically accurate outline
    theta = np.linspace(0, 2*np.pi, 300)
    
    # Main brain outline
    a = 3000
    b = 3000
    
    # Create multiple ellipses for different brain regions
    regions = []
    
    # Main cortex
    x_cortex = a * np.cos(theta)
    y_cortex = b * np.sin(theta)
    regions.append((x_cortex, y_cortex))
    
    # Hippocampus (simplified)
    theta_hc = np.linspace(0, 2*np.pi, 100)
    x_hc = 0.3 * a * np.cos(theta_hc) + 1000
    y_hc = 0.4 * b * np.sin(theta_hc) + 500
    regions.append((x_hc, y_hc))
    
    # Thalamus
    theta_th = np.linspace(0, 2*np.pi, 100)
    x_th = 0.2 * a * np.cos(theta_th) - 500
    y_th = 0.3 * b * np.sin(theta_th) + 800
    regions.append((x_th, y_th))
    
    return regions

def overlay_slice_realistic(coords_um: np.ndarray, slice_ap_um: float, tol_um: float, output_path: str, title_note: str = ""):
    """Create a realistic coronal slice visualization."""
    
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
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create realistic brain outline
    brain_x, brain_y = create_realistic_brain_outline()
    ax.plot(brain_x, brain_y, 'k-', linewidth=3, label='Brain Outline')
    ax.fill(brain_x, brain_y, color='lightblue', alpha=0.2)
    
    # Add brain regions
    regions = create_brain_regions_outline()
    colors = ['lightgray', 'lightgreen', 'lightyellow']
    labels = ['Cortex', 'Hippocampus', 'Thalamus']
    
    for i, (x_reg, y_reg) in enumerate(regions):
        ax.plot(x_reg, y_reg, 'k-', linewidth=1, alpha=0.7)
        ax.fill(x_reg, y_reg, color=colors[i], alpha=0.1)
    
    # Plot channel locations
    ax.scatter(slice_coords[:, 1], slice_coords[:, 2], 
              c='red', s=100, alpha=0.9, edgecolors='darkred', linewidth=1,
              label=f'Channels in ±{tol_um:.0f}µm slice (N={len(slice_coords)})')
    
    # Set labels and title
    ax.set_xlabel('Dorsal-Ventral (μm)', fontsize=14)
    ax.set_ylabel('Left-Right (μm)', fontsize=14)
    ax.set_title(f'Coronal Slice at AP ≈ {slice_ap_um:.0f} μm {title_note}', fontsize=16)
    
    # Set equal aspect ratio and limits
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=12)
    
    # Set reasonable limits based on brain outline
    ax.set_xlim(-4000, 4000)
    ax.set_ylim(-4000, 4000)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Realistic brain slice overlay with channel voxels')
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
    out = args.output or f"{base}_realistic_coronal_slice.png"
    overlay_slice_realistic(coords, slice_ap, args.tol_um, out, title_note=f"(Total N={coords.shape[0]} ch)")

if __name__ == '__main__':
    main()
