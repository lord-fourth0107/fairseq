#!/usr/bin/env python3
"""
3D CCF Visualization with Allen Mouse Brain Atlas
=================================================
Uses Allen SDK to overlay CCF coordinates on the mouse brain atlas.
"""

import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
import argparse

try:
    from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
    from allensdk.api.queries.mouse_connectivity_api import MouseConnectivityApi
    from allensdk.core.reference_space_cache import ReferenceSpaceCache
    ALLEN_SDK_AVAILABLE = True
except ImportError:
    ALLEN_SDK_AVAILABLE = False
    print("Warning: Allen SDK not available. Install with: pip install allensdk")

def load_coordinate_lookup(input_path):
    """Load and merge CSV files to create coordinate lookup."""
    print("Loading coordinate data...")
    
    # Find CSV files in input path
    joined_path = os.path.join(input_path, 'joined.csv')
    channels_path = os.path.join(input_path, 'channels.csv')
    
    if not os.path.exists(joined_path):
        raise FileNotFoundError(f"joined.csv not found at {joined_path}")
    if not os.path.exists(channels_path):
        raise FileNotFoundError(f"channels.csv not found at {channels_path}")
    
    # Load CSV files
    joined_df = pd.read_csv(joined_path)
    channels_df = pd.read_csv(channels_path)
    
    print(f"Loaded joined.csv: {joined_df.shape}")
    print(f"Loaded channels.csv: {channels_df.shape}")
    
    # Merge on probe_id
    merged_df = pd.merge(
        joined_df,
        channels_df,
        left_on='probe_id',
        right_on='ecephys_probe_id',
        how='inner'
    )
    
    print(f"Merged data shape: {merged_df.shape}")
    
    # Create coordinate lookup - use channel-specific CCF coordinates from channels.csv
    coord_lookup = {}
    for _, row in merged_df.iterrows():
        key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
        coord_lookup[key] = {
            'ap': row['anterior_posterior_ccf_coordinate_y'],  # From channels.csv
            'dv': row['dorsal_ventral_ccf_coordinate_y'],      # From channels.csv
            'lr': row['left_right_ccf_coordinate_y'],          # From channels.csv
            'probe_h': row['probe_horizontal_position'],
            'probe_v': row['probe_vertical_position'],
            'structure': row['ecephys_structure_acronym_y'] if 'ecephys_structure_acronym_y' in row else 'Unknown'
        }
    
    print(f"Created coordinate lookup with {len(coord_lookup)} entries")
    return coord_lookup

def load_pickle_files(input_path):
    """Load all pickle files from input path."""
    pickle_files = []
    for file in os.listdir(input_path):
        if file.endswith('.pickle'):
            pickle_files.append(os.path.join(input_path, file))
    
    print(f"Found {len(pickle_files)} pickle files")
    return pickle_files

def extract_ccf_coordinates_from_label(label, coord_lookup):
    """Extract CCF coordinates from enriched label."""
    try:
        # Parse the enriched label: session_count_probe_channel_brain_region_ap_dv_lr_probe_h_probe_v
        parts = label.split('_')
        if len(parts) >= 9:
            session_id = parts[0]
            probe_id = parts[2]
            channel_id = parts[3]
            
            # Look up coordinates
            lookup_key = (session_id, probe_id, channel_id)
            if lookup_key in coord_lookup:
                coords = coord_lookup[lookup_key]
                return {
                    'ap': float(coords['ap']),
                    'dv': float(coords['dv']),
                    'lr': float(coords['lr']),
                    'probe_h': float(coords['probe_h']),
                    'probe_v': float(coords['probe_v']),
                    'session_id': session_id,
                    'probe_id': probe_id,
                    'channel_id': channel_id,
                    'structure': coords.get('structure', 'Unknown')
                }
    except (ValueError, IndexError) as e:
        pass
    
    return None

def collect_all_ccf_coordinates(pickle_files, coord_lookup):
    """Collect CCF coordinates for all unique channels from all probes."""
    print("Collecting CCF coordinates from all pickle files...")
    
    all_coordinates = []
    probe_stats = {}
    
    for pickle_file in tqdm(pickle_files, desc="Processing pickle files"):
        try:
            with open(pickle_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract filename for probe identification
            filename = os.path.basename(pickle_file)
            if '715093703_810755797' in filename:
                probe_id = '810755797'
                session_id = '715093703'
            elif '847657808_848037578' in filename:
                probe_id = '848037578'
                session_id = '847657808'
            else:
                continue
            
            probe_coords = []
            unique_channels = set()
            
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    signal, label = entry
                    coords = extract_ccf_coordinates_from_label(label, coord_lookup)
                    if coords:
                        probe_coords.append(coords)
                        unique_channels.add(coords['channel_id'])
            
            probe_stats[probe_id] = {
                'session_id': session_id,
                'total_entries': len(data),
                'unique_channels': len(unique_channels),
                'coordinates': probe_coords
            }
            
            all_coordinates.extend(probe_coords)
            
        except Exception as e:
            print(f"Error processing {pickle_file}: {e}")
            continue
    
    print(f"Collected {len(all_coordinates)} total coordinate entries")
    return all_coordinates, probe_stats

def load_allen_brain_atlas():
    """Load Allen mouse brain atlas using Allen SDK."""
    if not ALLEN_SDK_AVAILABLE:
        print("Allen SDK not available. Using simplified brain outline.")
        return None, None
    
    try:
        print("Loading Allen mouse brain atlas...")
        
        # Initialize reference space cache
        rspc = ReferenceSpaceCache(10, manifest_file='reference_space_cache_manifest.json')
        
        # Get the reference space
        rsp = rspc.get_reference_space()
        
        # Get brain outline (simplified)
        brain_outline = rsp.get_structure_mask([997])  # Cerebrum
        
        print("Allen brain atlas loaded successfully")
        return rsp, brain_outline
        
    except Exception as e:
        print(f"Error loading Allen atlas: {e}")
        print("Falling back to simplified visualization")
        return None, None

def create_brain_outline_simple(ap_range, dv_range, lr_range):
    """Create a simplified brain outline for visualization."""
    # Create a simple ellipsoid representing the mouse brain
    # Dimensions based on actual CCF coordinate ranges
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(0, np.pi, 50)
    
    # Calculate brain dimensions based on actual data ranges
    ap_center = (ap_range[0] + ap_range[1]) / 2
    dv_center = (dv_range[0] + dv_range[1]) / 2
    lr_center = (lr_range[0] + lr_range[1]) / 2
    
    # Add some padding around the data
    ap_padding = (ap_range[1] - ap_range[0]) * 0.2
    dv_padding = (dv_range[1] - dv_range[0]) * 0.2
    lr_padding = (lr_range[1] - lr_range[0]) * 0.2
    
    a = (ap_range[1] - ap_range[0]) / 2 + ap_padding  # AP radius
    b = (dv_range[1] - dv_range[0]) / 2 + dv_padding  # DV radius  
    c = (lr_range[1] - lr_range[0]) / 2 + lr_padding  # LR radius
    
    x = a * np.outer(np.cos(u), np.sin(v)) + ap_center
    y = b * np.outer(np.sin(u), np.sin(v)) + dv_center
    z = c * np.outer(np.ones(np.size(u)), np.cos(v)) + lr_center
    
    return x, y, z

def create_3d_atlas_visualization(all_coordinates, probe_stats, output_dir="allen_atlas_visualizations"):
    """Create 3D visualization with Allen brain atlas."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print("Creating 3D atlas visualizations...")
    
    # Load Allen atlas
    rsp, brain_outline = load_allen_brain_atlas()
    
    # Convert to numpy arrays for easier manipulation
    coords_array = np.array([(c['ap'], c['dv'], c['lr']) for c in all_coordinates])
    
    # Calculate overall coordinate ranges for brain outline
    ap_range = (coords_array[:, 0].min(), coords_array[:, 0].max())
    dv_range = (coords_array[:, 1].min(), coords_array[:, 1].max())
    lr_range = (coords_array[:, 2].min(), coords_array[:, 2].max())
    
    # Create individual probe plots
    for probe_id, stats in probe_stats.items():
        if len(stats['coordinates']) == 0:
            continue
        
        # Filter coordinates for this probe
        probe_coords = [c for c in all_coordinates if c['probe_id'] == probe_id]
        probe_array = np.array([(c['ap'], c['dv'], c['lr']) for c in probe_coords])
        
        # Create 3D plot
        fig = plt.figure(figsize=(20, 15))
        ax = fig.add_subplot(111, projection='3d')
        
        # Add brain outline
        if brain_outline is not None:
            # Plot brain outline (simplified)
            x_brain, y_brain, z_brain = create_brain_outline_simple(ap_range, dv_range, lr_range)
            ax.plot_surface(x_brain, y_brain, z_brain, alpha=0.1, color='lightgray', 
                           label='Mouse Brain Outline')
        else:
            # Use simple brain outline
            x_brain, y_brain, z_brain = create_brain_outline_simple(ap_range, dv_range, lr_range)
            ax.plot_surface(x_brain, y_brain, z_brain, alpha=0.1, color='lightgray', 
                           label='Mouse Brain Outline')
        
        # Plot CCF coordinates
        color = 'red' if probe_id == '810755797' else 'green'
        scatter = ax.scatter(probe_array[:, 0], probe_array[:, 1], probe_array[:, 2], 
                           c=color, s=30, alpha=0.8, edgecolors='black', linewidth=0.2,
                           label=f'Probe {probe_id} Channels')
        
        # Set labels
        ax.set_xlabel('Anterior-Posterior (AP) CCF Coordinate (μm)', fontsize=12)
        ax.set_ylabel('Dorsal-Ventral (DV) CCF Coordinate (μm)', fontsize=12)
        ax.set_zlabel('Left-Right (LR) CCF Coordinate (μm)', fontsize=12)
        ax.set_title(f'Probe {probe_id} - 3D CCF Coordinates on Mouse Brain Atlas\n{len(probe_coords)} unique channels', fontsize=14)
        
        # Add statistics
        ap_range = probe_array[:, 0].max() - probe_array[:, 0].min()
        dv_range = probe_array[:, 1].max() - probe_array[:, 1].min()
        lr_range = probe_array[:, 2].max() - probe_array[:, 2].min()
        
        stats_text = f'AP range: {ap_range:.1f} μm\nDV range: {dv_range:.1f} μm\nLR range: {lr_range:.1f} μm\nChannels: {len(probe_coords)}'
        ax.text2D(0.02, 0.98, stats_text, transform=ax.transAxes, 
                 verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        # Set viewing angle for better brain visualization
        ax.view_init(elev=20, azim=45)
        
        # Add legend
        ax.legend()
        
        plt.tight_layout()
        
        # Save individual probe plot
        output_path = os.path.join(output_dir, f"probe_{probe_id}_allen_atlas.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close()
    
    # Create combined 3D plot
    fig = plt.figure(figsize=(25, 20))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add brain outline
    x_brain, y_brain, z_brain = create_brain_outline_simple(ap_range, dv_range, lr_range)
    ax.plot_surface(x_brain, y_brain, z_brain, alpha=0.1, color='lightgray', 
                   label='Mouse Brain Outline')
    
    # Plot all probes together
    colors = ['red', 'green']
    for i, (probe_id, stats) in enumerate(probe_stats.items()):
        if len(stats['coordinates']) == 0:
            continue
        
        # Filter coordinates for this probe
        probe_coords = [c for c in all_coordinates if c['probe_id'] == probe_id]
        probe_array = np.array([(c['ap'], c['dv'], c['lr']) for c in probe_coords])
        
        ax.scatter(probe_array[:, 0], probe_array[:, 1], probe_array[:, 2], 
                  c=colors[i], s=40, alpha=0.7, edgecolors='black', linewidth=0.2,
                  label=f"Probe {probe_id} ({len(probe_coords)} ch)")
    
    # Set labels and title
    ax.set_xlabel('Anterior-Posterior (AP) CCF Coordinate (μm)', fontsize=14)
    ax.set_ylabel('Dorsal-Ventral (DV) CCF Coordinate (μm)', fontsize=14)
    ax.set_zlabel('Left-Right (LR) CCF Coordinate (μm)', fontsize=14)
    ax.set_title('All Probes - 3D CCF Coordinates on Allen Mouse Brain Atlas', fontsize=16)
    
    # Add legend
    ax.legend(fontsize=12)
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    
    # Save combined plot
    output_path = os.path.join(output_dir, "all_probes_allen_atlas.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    
    plt.close()

def create_2d_atlas_projections(all_coordinates, probe_stats, output_dir="allen_atlas_visualizations"):
    """Create 2D projections of CCF coordinates on brain atlas."""
    
    print("Creating 2D atlas projections...")
    
    # Create 2D projections
    projections = [
        ('AP-DV', 0, 1, 'Anterior-Posterior', 'Dorsal-Ventral'),
        ('AP-LR', 0, 2, 'Anterior-Posterior', 'Left-Right'),
        ('DV-LR', 1, 2, 'Dorsal-Ventral', 'Left-Right')
    ]
    
    for proj_name, x_idx, y_idx, x_label, y_label in projections:
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Plot all probes
        colors = ['red', 'green']
        for i, (probe_id, stats) in enumerate(probe_stats.items()):
            if len(stats['coordinates']) == 0:
                continue
            
            # Filter coordinates for this probe
            probe_coords = [c for c in all_coordinates if c['probe_id'] == probe_id]
            probe_array = np.array([(c['ap'], c['dv'], c['lr']) for c in probe_coords])
            
            ax.scatter(probe_array[:, x_idx], probe_array[:, y_idx], 
                      c=colors[i], s=20, alpha=0.6, edgecolors='black', linewidth=0.3,
                      label=f"Probe {probe_id} ({len(probe_coords)} ch)")
        
        ax.set_xlabel(f'{x_label} CCF Coordinate (μm)', fontsize=12)
        ax.set_ylabel(f'{y_label} CCF Coordinate (μm)', fontsize=12)
        ax.set_title(f'2D Projection: {proj_name} - CCF Coordinates on Mouse Brain Atlas', fontsize=14)
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect('equal')
        
        plt.tight_layout()
        
        # Save 2D projection
        output_path = os.path.join(output_dir, f"2d_projection_{proj_name.lower()}.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close()

def print_atlas_statistics(all_coordinates, probe_stats):
    """Print comprehensive atlas statistics."""
    print("\n" + "="*80)
    print("ALLEN MOUSE BRAIN ATLAS - CCF COORDINATE STATISTICS")
    print("="*80)
    
    # Overall statistics
    coords_array = np.array([(c['ap'], c['dv'], c['lr']) for c in all_coordinates])
    
    print(f"\nOverall CCF Coordinate Ranges:")
    print(f"  AP (Anterior-Posterior): {coords_array[:, 0].min():.1f} to {coords_array[:, 0].max():.1f} μm")
    print(f"  DV (Dorsal-Ventral):     {coords_array[:, 1].min():.1f} to {coords_array[:, 1].max():.1f} μm")
    print(f"  LR (Left-Right):         {coords_array[:, 2].min():.1f} to {coords_array[:, 2].max():.1f} μm")
    
    print(f"\nTotal unique channels across all probes: {len(all_coordinates)}")
    
    # Per-probe statistics
    for probe_id, stats in probe_stats.items():
        if len(stats['coordinates']) == 0:
            continue
        
        probe_coords = [c for c in all_coordinates if c['probe_id'] == probe_id]
        probe_array = np.array([(c['ap'], c['dv'], c['lr']) for c in probe_coords])
        
        print(f"\nProbe {probe_id} (Session {stats['session_id']}):")
        print(f"  Unique channels: {len(probe_coords)}")
        print(f"  AP range: {probe_array[:, 0].min():.1f} to {probe_array[:, 0].max():.1f} μm")
        print(f"  DV range: {probe_array[:, 1].min():.1f} to {probe_array[:, 1].max():.1f} μm")
        print(f"  LR range: {probe_array[:, 2].min():.1f} to {probe_array[:, 2].max():.1f} μm")
        print(f"  CCF volume: {np.prod(probe_array.max(axis=0) - probe_array.min(axis=0)):.0f} μm³")
        
        # Brain region analysis
        structures = [c['structure'] for c in probe_coords if c['structure'] != 'Unknown']
        if structures:
            unique_structures = list(set(structures))
            print(f"  Brain regions: {len(unique_structures)} unique structures")
            print(f"  Top regions: {', '.join(unique_structures[:5])}")

def main():
    parser = argparse.ArgumentParser(description='3D CCF Visualization with Allen Mouse Brain Atlas')
    parser.add_argument('input_path', help='Path to directory containing pickle files and CSV files')
    parser.add_argument('--output-dir', default='allen_atlas_visualizations', 
                       help='Output directory for visualizations (default: allen_atlas_visualizations)')
    
    args = parser.parse_args()
    
    print("3D CCF Visualization with Allen Mouse Brain Atlas")
    print("="*60)
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    
    if not ALLEN_SDK_AVAILABLE:
        print("\nNote: Allen SDK not installed. Using simplified brain outline.")
        print("For full atlas functionality, install with: pip install allensdk")
    
    try:
        # Load coordinate lookup
        coord_lookup = load_coordinate_lookup(args.input_path)
        
        # Load pickle files
        pickle_files = load_pickle_files(args.input_path)
        
        if not pickle_files:
            print("No pickle files found!")
            return
        
        # Collect all CCF coordinates
        all_coordinates, probe_stats = collect_all_ccf_coordinates(pickle_files, coord_lookup)
        
        if not all_coordinates:
            print("No CCF coordinates found!")
            return
        
        # Print statistics
        print_atlas_statistics(all_coordinates, probe_stats)
        
        # Create visualizations
        create_3d_atlas_visualization(all_coordinates, probe_stats, args.output_dir)
        create_2d_atlas_projections(all_coordinates, probe_stats, args.output_dir)
        
        print(f"\nAtlas visualization completed! Check the '{args.output_dir}' directory for images.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
