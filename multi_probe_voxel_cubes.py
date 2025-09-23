#!/usr/bin/env python3
"""
Multi-Probe Cube Visualization (Single Plot)
===========================================
Given one or more pickle files, draw 1mm³ voxel cubes for each probe in the
same 3D plot, using a distinct color per probe.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - needed for 3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm


def load_coordinate_lookup(input_path: str) -> dict:
    print("Loading coordinate data...")
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
    print(f"Created coordinate lookup with {len(coord_lookup)} entries")
    return coord_lookup


def extract_coords_from_label(label: str, coord_lookup: dict):
    parts = label.split('_')
    if len(parts) < 4:
        return None
    session_id = parts[0]
    probe_id = parts[2]
    channel_id = parts[3]
    key = (session_id, probe_id, channel_id)
    if key in coord_lookup:
        c = coord_lookup[key]
        return float(c['ap']), float(c['dv']), float(c['lr']), probe_id
    return None


essential_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 'brown']


def collect_unique_channel_coords(pickle_path: str, coord_lookup: dict):
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    unique = {}
    for entry in tqdm(data, desc=f"Scanning {os.path.basename(pickle_path)}"):
        if isinstance(entry, (list, tuple)) and len(entry) >= 2:
            _, label = entry
            res = extract_coords_from_label(label, coord_lookup)
            if res is None:
                continue
            ap, dv, lr, probe_id = res
            ch_key = (label.split('_')[0], probe_id, label.split('_')[3])
            if ch_key not in unique:
                unique[ch_key] = (ap, dv, lr, probe_id)
    return list(unique.values())


def plot_cubes_for_probes(probe_to_coords: dict, output_path: str, voxel_size_um: float = 1000.0):
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')

    # Determine colors per probe
    probe_ids = list(probe_to_coords.keys())
    color_map = {pid: essential_colors[i % len(essential_colors)] for i, pid in enumerate(probe_ids)}

    # For aspect limits
    all_min = np.array([np.inf, np.inf, np.inf], dtype=float)
    all_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)

    # Define unit cube vertices and faces
    unit_vertices = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
    ], dtype=float)
    faces_idx = [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 5, 4], [2, 3, 7, 6], [0, 3, 7, 4], [1, 2, 6, 5]]

    for pid, coords in probe_to_coords.items():
        if not coords:
            continue
        color = color_map[pid]
        # Voxelize: map coordinates to lower-corner of 1mm cube; deduplicate
        voxel_origins = set()
        for ap, dv, lr in coords:
            ap0 = np.floor(ap / voxel_size_um) * voxel_size_um
            dv0 = np.floor(dv / voxel_size_um) * voxel_size_um
            lr0 = np.floor(lr / voxel_size_um) * voxel_size_um
            voxel_origins.add((ap0, dv0, lr0))
        # Build faces
        faces = []
        for ap0, dv0, lr0 in voxel_origins:
            base = np.array([ap0, dv0, lr0], dtype=float)
            verts = unit_vertices * voxel_size_um + base
            for f in faces_idx:
                faces.append(verts[f])
            # expand bounds
            all_min = np.minimum(all_min, base)
            all_max = np.maximum(all_max, base + voxel_size_um)
        collection = Poly3DCollection(faces, facecolor=color, alpha=0.75, edgecolor='black', linewidth=0.4)
        ax.add_collection3d(collection)

    # Axes labels/title
    ax.set_xlabel('Anterior-Posterior (μm)', fontsize=13)
    ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=13)
    ax.set_zlabel('Left-Right (μm)', fontsize=13)
    ax.set_title('Multi-Probe Channel Cubes (1mm³)', fontsize=15)

    # Legend
    handles = [plt.Line2D([0], [0], marker='s', color='w', label=f'Probe {pid}', markerfacecolor=color_map[pid], markersize=10) for pid in probe_ids]
    ax.legend(handles=handles, labels=[f'Probe {pid} ({len(probe_to_coords[pid])} ch)' for pid in probe_ids])

    # Set limits and equal-ish aspect
    ranges = all_max - all_min
    max_range = np.max(ranges)
    centers = (all_max + all_min) / 2.0
    ax.set_xlim(centers[0] - max_range/2, centers[0] + max_range/2)
    ax.set_ylim(centers[1] - max_range/2, centers[1] + max_range/2)
    ax.set_zlim(centers[2] - max_range/2, centers[2] + max_range/2)

    # Nice view
    ax.view_init(elev=20, azim=45)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Draw 1mm³ cubes for channels from one or more pickles in a single plot, colored per probe')
    parser.add_argument('pickle_files', nargs='+', help='One or more pickle files')
    parser.add_argument('input_path', help='Directory containing joined.csv and channels.csv')
    parser.add_argument('--output', default='multi_probe_voxel_cubes.png', help='Output image path')
    args = parser.parse_args()

    coord_lookup = load_coordinate_lookup(args.input_path)

    probe_to_coords = {}
    for p in args.pickle_files:
        with open(p, 'rb') as f:
            data = pickle.load(f)
        unique = {}
        for entry in tqdm(data, desc=f"Scanning {os.path.basename(p)}"):
            if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                _, label = entry
                res = extract_coords_from_label(label, coord_lookup)
                if res is None:
                    continue
                ap, dv, lr, pid = res
                ch_key = (label.split('_')[0], pid, label.split('_')[3])
                if ch_key not in unique:
                    unique[ch_key] = (ap, dv, lr)
        coords = list(unique.values())
        if coords:
            probe_to_coords.setdefault(pid, []).extend(coords)

    if not probe_to_coords:
        print('No coordinates found to plot.')
        return

    plot_cubes_for_probes(probe_to_coords, args.output)


if __name__ == '__main__':
    main()
