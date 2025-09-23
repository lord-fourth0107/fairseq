#!/usr/bin/env python3
"""
Simple Data Plot
================
Just plot the channel data clearly without trying to overlay on brain atlases.
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

def plot_3d_coordinates(coords: np.ndarray, output_path: str):
    """Simple 3D plot of coordinates."""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot all coordinates
    ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], 
               c='red', s=50, alpha=0.7)
    
    ax.set_xlabel('AP (μm)')
    ax.set_ylabel('DV (μm)')
    ax.set_zlabel('LR (μm)')
    ax.set_title(f'Channel Coordinates (N={len(coords)})')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def plot_2d_projections(coords: np.ndarray, output_path: str):
    """Plot 2D projections."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # AP vs DV
    axes[0, 0].scatter(coords[:, 0], coords[:, 1], c='red', s=30, alpha=0.7)
    axes[0, 0].set_xlabel('AP (μm)')
    axes[0, 0].set_ylabel('DV (μm)')
    axes[0, 0].set_title('AP vs DV')
    axes[0, 0].grid(True, alpha=0.3)
    
    # AP vs LR
    axes[0, 1].scatter(coords[:, 0], coords[:, 2], c='blue', s=30, alpha=0.7)
    axes[0, 1].set_xlabel('AP (μm)')
    axes[0, 1].set_ylabel('LR (μm)')
    axes[0, 1].set_title('AP vs LR')
    axes[0, 1].grid(True, alpha=0.3)
    
    # DV vs LR
    axes[1, 0].scatter(coords[:, 1], coords[:, 2], c='green', s=30, alpha=0.7)
    axes[1, 0].set_xlabel('DV (μm)')
    axes[1, 0].set_ylabel('LR (μm)')
    axes[1, 0].set_title('DV vs LR')
    axes[1, 0].grid(True, alpha=0.3)
    
    # All coordinates histogram
    axes[1, 1].hist(coords.flatten(), bins=50, alpha=0.7, color='purple')
    axes[1, 1].set_xlabel('Coordinate Value (μm)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title('Coordinate Distribution')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Simple coordinate visualization')
    parser.add_argument('pickle_file', help='Path to pickle file')
    parser.add_argument('input_path', help='Directory containing joined.csv and channels.csv')
    args = parser.parse_args()

    coord_lookup = load_coordinate_lookup(args.input_path)
    coords = extract_channel_coords(args.pickle_file, coord_lookup)
    
    if coords.size == 0:
        print('No channel coordinates found.')
        return

    print(f"Found {len(coords)} channel coordinates")
    print(f"AP range: {coords[:, 0].min():.1f} to {coords[:, 0].max():.1f}")
    print(f"DV range: {coords[:, 1].min():.1f} to {coords[:, 1].max():.1f}")
    print(f"LR range: {coords[:, 2].min():.1f} to {coords[:, 2].max():.1f}")

    base = os.path.splitext(os.path.basename(args.pickle_file))[0]
    
    # 3D plot
    plot_3d_coordinates(coords, f"{base}_3d_coords.png")
    
    # 2D projections
    plot_2d_projections(coords, f"{base}_2d_projections.png")

if __name__ == '__main__':
    main()
