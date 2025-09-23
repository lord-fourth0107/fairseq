#!/usr/bin/env python3
"""
Voxel Channel Histogram
=======================
For a given pickle file, voxelize channel CCF coordinates into 1mm³ voxels and
plot a histogram of number of channels per occupied voxel.
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


def extract_coords_from_label(label: str, coord_lookup: dict):
    parts = label.split('_')
    if len(parts) < 4:
        return None
    key = (parts[0], parts[2], parts[3])
    if key in coord_lookup:
        c = coord_lookup[key]
        return float(c['ap']), float(c['dv']), float(c['lr'])
    return None


def count_channels_per_voxel(pickle_file: str, coord_lookup: dict, voxel_size_um: float = 1000.0):
    with open(pickle_file, 'rb') as f:
        data = pickle.load(f)

    voxel_counts = {}
    seen_channels = set()

    for entry in tqdm(data, desc=f"Voxelizing {os.path.basename(pickle_file)}"):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            _, label = entry
            parts = label.split('_')
            if len(parts) < 4:
                continue
            # ensure unique per channel
            ch_key = (parts[0], parts[2], parts[3])
            if ch_key in seen_channels:
                continue
            seen_channels.add(ch_key)

            coords = extract_coords_from_label(label, coord_lookup)
            if coords is None:
                continue
            ap, dv, lr = coords
            ap0 = int(np.floor(ap / voxel_size_um))
            dv0 = int(np.floor(dv / voxel_size_um))
            lr0 = int(np.floor(lr / voxel_size_um))
            vox_key = (ap0, dv0, lr0)
            voxel_counts[vox_key] = voxel_counts.get(vox_key, 0) + 1

    return voxel_counts


def plot_histogram(voxel_counts: dict, output_path: str, title: str):
    counts = np.array(list(voxel_counts.values()), dtype=int)
    if counts.size == 0:
        print('No occupied voxels found. Skipping plot.')
        return

    plt.figure(figsize=(10, 6))
    plt.hist(counts, bins=min(50, counts.max()), color='steelblue', edgecolor='black', alpha=0.85)
    plt.xlabel('Channels per Voxel (1mm³)')
    plt.ylabel('Number of Voxels')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot histogram of number of channels per 1mm³ voxel for a pickle file')
    parser.add_argument('pickle_file', help='Path to pickle file')
    parser.add_argument('input_path', help='Directory containing joined.csv and channels.csv')
    parser.add_argument('--output', default=None, help='Output image path (default: derived from pickle name)')
    parser.add_argument('--voxel-size', type=float, default=1.0, help='Voxel size in mm (default 1.0)')
    args = parser.parse_args()

    coord_lookup = load_coordinate_lookup(args.input_path)
    voxel_counts = count_channels_per_voxel(args.pickle_file, coord_lookup, voxel_size_um=args.voxel_size * 1000.0)

    base = os.path.splitext(os.path.basename(args.pickle_file))[0]
    out = args.output or f"{base}_voxel_channel_hist.png"
    title = f"{base} - Channels per 1mm³ Voxel (Occupied Voxels: {len(voxel_counts)})"
    plot_histogram(voxel_counts, out, title)


if __name__ == '__main__':
    main()
