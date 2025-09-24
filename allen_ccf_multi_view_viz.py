#!/usr/bin/env python3
"""
Allen CCF Multi-View Visualization
==================================
Creates comprehensive Allen CCF visualizations with channel overlays in multiple views:
- Coronal slices (AP view)
- Sagittal slices (LR view) 
- Horizontal slices (DV view)
Uses the working AllenSDK MouseConnectivityCache approach.
"""

import os
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def ccf_to_voxel_coords(ap, dv, lr, annot_info):
    """Convert CCF coordinates to voxel coordinates."""
    voxel_size = 25.0  # 25 micrometers per voxel
    ap_voxel = int(ap / voxel_size)
    dv_voxel = int(dv / voxel_size)
    lr_voxel = int(lr / voxel_size)
    return ap_voxel, dv_voxel, lr_voxel

def create_coronal_slices(annot, template, coords_dict, output_dir, voxel_size=25.0):
    """Create coronal slice visualizations."""
    print("Creating coronal slices...")
    
    # Calculate slice positions for each probe
    all_coords = np.concatenate([coords for coords in coords_dict.values()])
    ap_min, ap_max = all_coords[:, 0].min(), all_coords[:, 0].max()
    
    # Create multiple slices
    slice_positions = np.linspace(ap_min, ap_max, 5)
    
    for i, slice_ap in enumerate(slice_positions):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convert to voxel coordinates
        slice_idx = int(slice_ap / voxel_size)
        slice_idx = max(0, min(slice_idx, annot.shape[0] - 1))
        
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
        
        # Overlay channels
        axes[2].imshow(annot[slice_idx, :, :], cmap='gray', aspect='equal', alpha=0.7)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for j, (probe_id, coords) in enumerate(coords_dict.items()):
            if len(coords) == 0:
                continue
            
            # Convert coordinates to voxel space
            voxel_coords = []
            for coord in coords:
                ap_vox, dv_vox, lr_vox = ccf_to_voxel_coords(coord[0], coord[1], coord[2], None)
                voxel_coords.append((ap_vox, dv_vox, lr_vox))
            
            voxel_coords = np.array(voxel_coords)
            
            # Plot channels within slice tolerance
            mask = np.abs(voxel_coords[:, 0] - slice_idx) <= 2  # ±2 voxels tolerance
            slice_coords = voxel_coords[mask]
            
            if len(slice_coords) > 0:
                axes[2].scatter(slice_coords[:, 2], slice_coords[:, 1], 
                              c=colors[j % len(colors)], s=30, alpha=0.8, 
                              edgecolors='black', linewidth=0.5, label=f'Probe {probe_id}')
        
        axes[2].set_title(f'Channels Overlay (AP slice {i+1}/5)')
        axes[2].set_xlabel('LR (voxels)')
        axes[2].set_ylabel('DV (voxels)')
        axes[2].legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'coronal_slice_{i+1}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

def create_sagittal_slices(annot, template, coords_dict, output_dir, voxel_size=25.0):
    """Create sagittal slice visualizations."""
    print("Creating sagittal slices...")
    
    # Calculate slice positions
    all_coords = np.concatenate([coords for coords in coords_dict.values()])
    lr_min, lr_max = all_coords[:, 2].min(), all_coords[:, 2].max()
    
    slice_positions = np.linspace(lr_min, lr_max, 5)
    
    for i, slice_lr in enumerate(slice_positions):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convert to voxel coordinates
        slice_idx = int(slice_lr / voxel_size)
        slice_idx = max(0, min(slice_idx, annot.shape[2] - 1))
        
        # Template slice
        axes[0].imshow(template[:, :, slice_idx], cmap='gray', aspect='equal')
        axes[0].set_title('Template Slice')
        axes[0].set_xlabel('AP (voxels)')
        axes[0].set_ylabel('DV (voxels)')
        
        # Annotation slice
        axes[1].imshow(annot[:, :, slice_idx], cmap='tab20', aspect='equal')
        axes[1].set_title('Annotation Slice')
        axes[1].set_xlabel('AP (voxels)')
        axes[1].set_ylabel('DV (voxels)')
        
        # Overlay channels
        axes[2].imshow(annot[:, :, slice_idx], cmap='gray', aspect='equal', alpha=0.7)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for j, (probe_id, coords) in enumerate(coords_dict.items()):
            if len(coords) == 0:
                continue
            
            # Convert coordinates to voxel space
            voxel_coords = []
            for coord in coords:
                ap_vox, dv_vox, lr_vox = ccf_to_voxel_coords(coord[0], coord[1], coord[2], None)
                voxel_coords.append((ap_vox, dv_vox, lr_vox))
            
            voxel_coords = np.array(voxel_coords)
            
            # Plot channels within slice tolerance
            mask = np.abs(voxel_coords[:, 2] - slice_idx) <= 2
            slice_coords = voxel_coords[mask]
            
            if len(slice_coords) > 0:
                axes[2].scatter(slice_coords[:, 0], slice_coords[:, 1], 
                              c=colors[j % len(colors)], s=30, alpha=0.8, 
                              edgecolors='black', linewidth=0.5, label=f'Probe {probe_id}')
        
        axes[2].set_title(f'Channels Overlay (LR slice {i+1}/5)')
        axes[2].set_xlabel('AP (voxels)')
        axes[2].set_ylabel('DV (voxels)')
        axes[2].legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'sagittal_slice_{i+1}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

def create_horizontal_slices(annot, template, coords_dict, output_dir, voxel_size=25.0):
    """Create horizontal slice visualizations."""
    print("Creating horizontal slices...")
    
    # Calculate slice positions
    all_coords = np.concatenate([coords for coords in coords_dict.values()])
    dv_min, dv_max = all_coords[:, 1].min(), all_coords[:, 1].max()
    
    slice_positions = np.linspace(dv_min, dv_max, 5)
    
    for i, slice_dv in enumerate(slice_positions):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Convert to voxel coordinates
        slice_idx = int(slice_dv / voxel_size)
        slice_idx = max(0, min(slice_idx, annot.shape[1] - 1))
        
        # Template slice
        axes[0].imshow(template[:, slice_idx, :], cmap='gray', aspect='equal')
        axes[0].set_title('Template Slice')
        axes[0].set_xlabel('AP (voxels)')
        axes[0].set_ylabel('LR (voxels)')
        
        # Annotation slice
        axes[1].imshow(annot[:, slice_idx, :], cmap='tab20', aspect='equal')
        axes[1].set_title('Annotation Slice')
        axes[1].set_xlabel('AP (voxels)')
        axes[1].set_ylabel('LR (voxels)')
        
        # Overlay channels
        axes[2].imshow(annot[:, slice_idx, :], cmap='gray', aspect='equal', alpha=0.7)
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        for j, (probe_id, coords) in enumerate(coords_dict.items()):
            if len(coords) == 0:
                continue
            
            # Convert coordinates to voxel space
            voxel_coords = []
            for coord in coords:
                ap_vox, dv_vox, lr_vox = ccf_to_voxel_coords(coord[0], coord[1], coord[2], None)
                voxel_coords.append((ap_vox, dv_vox, lr_vox))
            
            voxel_coords = np.array(voxel_coords)
            
            # Plot channels within slice tolerance
            mask = np.abs(voxel_coords[:, 1] - slice_idx) <= 2
            slice_coords = voxel_coords[mask]
            
            if len(slice_coords) > 0:
                axes[2].scatter(slice_coords[:, 0], slice_coords[:, 2], 
                              c=colors[j % len(colors)], s=30, alpha=0.8, 
                              edgecolors='black', linewidth=0.5, label=f'Probe {probe_id}')
        
        axes[2].set_title(f'Channels Overlay (DV slice {i+1}/5)')
        axes[2].set_xlabel('AP (voxels)')
        axes[2].set_ylabel('LR (voxels)')
        axes[2].legend()
        
        plt.tight_layout()
        output_path = os.path.join(output_dir, f'horizontal_slice_{i+1}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()

def main():
    parser = argparse.ArgumentParser(description='Allen CCF Multi-View Visualization')
    parser.add_argument('input_path', help='Directory containing pickle files and CSV files')
    parser.add_argument('--session-id', help='Specific session ID to process (if not provided, processes first session found)')
    parser.add_argument('--output-dir', default='allen_ccf_multi_view', help='Output directory for plots')
    parser.add_argument('--views', nargs='+', choices=['coronal', 'sagittal', 'horizontal'], 
                       default=['coronal', 'sagittal', 'horizontal'], help='Which views to generate')
    
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
    
    if annot is None:
        print("Failed to load Allen atlas")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate visualizations
    if 'coronal' in args.views:
        create_coronal_slices(annot, template, coords_dict, args.output_dir)
    
    if 'sagittal' in args.views:
        create_sagittal_slices(annot, template, coords_dict, args.output_dir)
    
    if 'horizontal' in args.views:
        create_horizontal_slices(annot, template, coords_dict, args.output_dir)
    
    print(f"\nAll visualizations saved to: {args.output_dir}")

if __name__ == '__main__':
    main()
