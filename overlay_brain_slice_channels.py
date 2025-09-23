#!/usr/bin/env python3
"""
Overlay Mouse Brain Coronal Slice with Channel Voxels
=====================================================
- Loads Allen CCF annotation volume via AllenSDK (25 µm resolution)
- Loads channel CCF coordinates from a pickle (using joined.csv + channels.csv)
- Selects an AP slice and overlays channel voxels that fall within a tolerance
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from allensdk.core.reference_space_cache import ReferenceSpaceCache
    ALLEN_OK = True
except Exception:
    ALLEN_OK = False

RES_UM = 25.0  # Allen atlas voxel size in microns at resolution=25


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


def overlay_slice(atlas, coords_um: np.ndarray, slice_ap_um: float, tol_um: float, output_path: str, title_note: str = ""):
    # Convert microns to voxel indices
    coords_vox = np.clip(np.round(coords_um / RES_UM).astype(int), 0, np.array(atlas.shape) - 1)
    slice_x = int(round(slice_ap_um / RES_UM))
    tol = max(1, int(round(tol_um / RES_UM)))

    mask = np.abs(coords_vox[:, 0] - slice_x) <= tol
    if not np.any(mask):
        print("No channel voxels within slice tolerance; still saving atlas slice.")

    y = coords_vox[mask, 1]
    z = coords_vox[mask, 2]

    fig, ax = plt.subplots(figsize=(9, 9))
    atlas_slice = atlas[slice_x, :, :]
    ax.imshow(atlas_slice.T, cmap='gray', origin='lower')

    if y.size > 0:
        ax.scatter(y, z, c='red', s=30, alpha=0.85, edgecolors='black', linewidth=0.3, label=f'Channels in ±{tol_um:.0f}µm')
        ax.legend(loc='upper right')

    ax.set_title(f'Coronal Slice at AP ≈ {slice_ap_um:.0f} µm {title_note}')
    ax.set_xlabel('Anterior-Posterior (vox idx along Y)')
    ax.set_ylabel('Dorsal-Ventral (vox idx along Z)')
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Overlay Allen CCF coronal slice with channel voxels from a pickle file')
    parser.add_argument('pickle_file', help='Path to pickle file')
    parser.add_argument('input_path', help='Directory containing joined.csv and channels.csv')
    parser.add_argument('--slice-ap', type=float, default=None, help='AP slice (µm). Default: center of channel AP range')
    parser.add_argument('--tol-um', type=float, default=50.0, help='Slice tolerance (µm) to include nearby channels (default 50)')
    parser.add_argument('--manifest', default='mouse_ccf.json', help='AllenSDK manifest file path')
    parser.add_argument('--output', default=None, help='Output image path')
    args = parser.parse_args()

    if not ALLEN_OK:
        print('AllenSDK not installed. Install with: pip install allensdk')
        sys.exit(1)

    coord_lookup = load_coordinate_lookup(args.input_path)
    coords = extract_unique_channel_coords(args.pickle_file, coord_lookup)
    if coords.size == 0:
        print('No channel coordinates found.')
        sys.exit(0)

    # Choose slice
    ap_min, ap_max = coords[:, 0].min(), coords[:, 0].max()
    slice_ap = args.slice_ap if args.slice_ap is not None else (ap_min + ap_max) / 2.0

    # Load atlas
    print('Loading Allen atlas...')
    rsp_cache = ReferenceSpaceCache(
        resolution=int(RES_UM), 
        reference_space_key='ccf_2017',
        manifest=args.manifest
    )
    atlas, _ = rsp_cache.get_annotation_volume()

    base = os.path.splitext(os.path.basename(args.pickle_file))[0]
    out = args.output or f"{base}_coronal_slice_overlay.png"
    overlay_slice(atlas, coords, slice_ap, args.tol_um, out, title_note=f"(N={coords.shape[0]} ch)")


if __name__ == '__main__':
    main()
