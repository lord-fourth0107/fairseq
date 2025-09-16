#!/usr/bin/env python3
"""
Voxelize Pickle Files and Create 3D Brain Visualization
Creates a 3D voxel grid from enriched pickle files and overlays it on a mouse brain.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import cKDTree
from collections import defaultdict
import argparse
import glob
from tqdm import tqdm

def load_pickle_files(input_path):
    """Load all pickle files from the input path."""
    if os.path.isfile(input_path):
        if input_path.endswith('.pickle'):
            return [input_path]
        else:
            print(f"Error: {input_path} is not a pickle file")
            return []
    elif os.path.isdir(input_path):
        pattern = os.path.join(input_path, "*.pickle")
        pickle_files = glob.glob(pattern)
        if not pickle_files:
            print(f"No pickle files found in {input_path}")
            return []
        return sorted(pickle_files)
    else:
        print(f"Error: {input_path} is not a valid file or directory")
        return []

def extract_coordinates_from_label(label):
    """Extract CCF coordinates from enriched label."""
    if not isinstance(label, str):
        return None
    
    parts = label.split('_')
    if len(parts) < 9:
        return None
    
    try:
        # Last 5 parts should be coordinates: ap, dv, lr, probe_h, probe_v
        ap = float(parts[-5])
        dv = float(parts[-4])
        lr = float(parts[-3])
        probe_h = float(parts[-2])
        probe_v = float(parts[-1])
        return {'ap': ap, 'dv': dv, 'lr': lr, 'probe_h': probe_h, 'probe_v': probe_v}
    except (ValueError, IndexError):
        return None

def debug_coordinate_extraction(pickle_files, sample_size=10):
    """Debug coordinate extraction to see what's happening."""
    print("\n" + "=" * 60)
    print("DEBUGGING COORDINATE EXTRACTION")
    print("=" * 60)
    
    for pickle_path in pickle_files[:2]:  # Check first 2 files
        print(f"\nFile: {os.path.basename(pickle_path)}")
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, (list, tuple)):
                print("  Data is not a list/tuple")
                continue
                
            print(f"  Total entries: {len(data)}")
            
            # Sample some entries
            sample_entries = data[:sample_size] if len(data) >= sample_size else data
            
            for i, entry in enumerate(sample_entries):
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    label = entry[1]
                    print(f"  Entry {i}: {label}")
                    coords = extract_coordinates_from_label(label)
                    if coords:
                        print(f"    -> AP: {coords['ap']}, DV: {coords['dv']}, LR: {coords['lr']}")
                    else:
                        print(f"    -> No coordinates extracted")
                else:
                    print(f"  Entry {i}: Invalid format")
                    
        except Exception as e:
            print(f"  Error: {e}")

def collect_coordinates_from_pickles(pickle_files, sample_size=None):
    """Collect all coordinates from pickle files."""
    all_coords = []
    file_stats = {}
    
    print(f"Processing {len(pickle_files)} pickle files...")
    
    for pickle_path in tqdm(pickle_files, desc="Loading coordinates"):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, (list, tuple)):
                continue
                
            coords_in_file = 0
            total_entries = len(data)
            
            # Sample data if sample_size is specified
            if sample_size and total_entries > sample_size:
                indices = np.random.choice(total_entries, sample_size, replace=False)
                data = [data[i] for i in indices]
            
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    label = entry[1]
                    coords = extract_coordinates_from_label(label)
                    if coords is not None:
                        all_coords.append([coords['ap'], coords['dv'], coords['lr']])
                        coords_in_file += 1
            
            file_stats[pickle_path] = {
                'total_entries': total_entries,
                'coords_found': coords_in_file,
                'enrichment_rate': coords_in_file / total_entries if total_entries > 0 else 0
            }
            
        except Exception as e:
            print(f"Error processing {pickle_path}: {e}")
            file_stats[pickle_path] = {'error': str(e)}
    
    return np.array(all_coords), file_stats

def create_voxel_grid(coordinates, voxel_size=100):
    """Create a 3D voxel grid from coordinates."""
    if len(coordinates) == 0:
        print("No coordinates found!")
        return None, None, None
    
    print(f"Creating voxel grid from {len(coordinates)} coordinates...")
    
    # Get coordinate ranges
    min_coords = np.min(coordinates, axis=0)
    max_coords = np.max(coordinates, axis=0)
    
    print(f"Coordinate ranges:")
    print(f"  AP: {min_coords[0]:.2f} to {max_coords[0]:.2f}")
    print(f"  DV: {min_coords[1]:.2f} to {max_coords[1]:.2f}")
    print(f"  LR: {min_coords[2]:.2f} to {max_coords[2]:.2f}")
    
    # Check if all coordinates are the same (no spatial variation)
    if np.allclose(min_coords, max_coords):
        print("WARNING: All coordinates are identical! This suggests the data may not be properly enriched.")
        print("Creating a single voxel at the coordinate location...")
        
        # Create a single voxel grid
        voxel_grid = np.array([[[len(coordinates)]]])  # Single voxel with count
        return voxel_grid, min_coords, voxel_size
    
    # Create voxel grid
    grid_shape = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
    
    # Ensure minimum grid size of 1x1x1
    grid_shape = np.maximum(grid_shape, [1, 1, 1])
    
    print(f"Voxel grid shape: {grid_shape}")
    voxel_grid = np.zeros(grid_shape)
    
    # Voxelize coordinates
    for coord in coordinates:
        voxel_idx = ((coord - min_coords) / voxel_size).astype(int)
        voxel_idx = np.clip(voxel_idx, 0, np.array(grid_shape) - 1)
        voxel_grid[tuple(voxel_idx)] += 1
    
    return voxel_grid, min_coords, voxel_size

def create_brain_outline():
    """Create a simple brain outline for visualization."""
    # This is a simplified brain outline - in practice, you'd load a real brain atlas
    # For now, we'll create a basic ellipsoid shape
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    u, v = np.meshgrid(u, v)
    
    # Brain-like ellipsoid (approximate mouse brain dimensions)
    x = 6000 * np.cos(u) * np.sin(v)
    y = 4000 * np.sin(u) * np.sin(v)
    z = 3000 * np.cos(v)
    
    return x, y, z

def plot_3d_voxel_density(voxel_grid, min_coords, voxel_size, coordinates, output_path=None):
    """Create 3D visualization of voxel density overlaid on brain outline."""
    print("Creating 3D visualization...")
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get voxel positions
    voxel_positions = np.where(voxel_grid > 0)
    voxel_coords = np.array(voxel_positions).T * voxel_size + min_coords
    voxel_densities = voxel_grid[voxel_positions]
    
    # Create brain outline
    brain_x, brain_y, brain_z = create_brain_outline()
    ax.plot_surface(brain_x, brain_y, brain_z, alpha=0.1, color='lightgray', label='Brain outline')
    
    # Plot voxel density
    if len(voxel_coords) > 0:
        scatter = ax.scatter(voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2], 
                           c=voxel_densities, cmap='hot', s=50, alpha=0.7, 
                           label=f'Voxel density (n={len(voxel_coords)})')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Voxel Density', rotation=270, labelpad=15)
    
    # Plot original coordinates as small points
    if len(coordinates) > 0:
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 
                  c='blue', s=1, alpha=0.3, label=f'Original coordinates (n={len(coordinates)})')
    
    # Set labels and title
    ax.set_xlabel('Anterior-Posterior (μm)')
    ax.set_ylabel('Dorsal-Ventral (μm)')
    ax.set_zlabel('Left-Right (μm)')
    ax.set_title('Neural Activity Voxelization on Mouse Brain')
    
    # Add legend
    ax.legend()
    
    # Set equal aspect ratio
    max_range = np.array([coordinates[:, 0].max() - coordinates[:, 0].min(),
                         coordinates[:, 1].max() - coordinates[:, 1].min(),
                         coordinates[:, 2].max() - coordinates[:, 2].min()]).max() / 2.0
    mid_x = (coordinates[:, 0].max() + coordinates[:, 0].min()) * 0.5
    mid_y = (coordinates[:, 1].max() + coordinates[:, 1].min()) * 0.5
    mid_z = (coordinates[:, 2].max() + coordinates[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to: {output_path}")
    
    plt.show()

def print_statistics(file_stats, coordinates):
    """Print processing statistics."""
    print("\n" + "=" * 60)
    print("VOXELIZATION STATISTICS")
    print("=" * 60)
    
    total_files = len(file_stats)
    successful_files = sum(1 for stats in file_stats.values() if 'error' not in stats)
    total_entries = sum(stats.get('total_entries', 0) for stats in file_stats.values() if 'error' not in stats)
    total_coords = sum(stats.get('coords_found', 0) for stats in file_stats.values() if 'error' not in stats)
    
    print(f"Files processed: {successful_files}/{total_files}")
    print(f"Total entries: {total_entries}")
    print(f"Coordinates extracted: {total_coords}")
    print(f"Overall enrichment rate: {total_coords/total_entries*100:.1f}%" if total_entries > 0 else "N/A")
    
    print(f"\nPer-file statistics:")
    for pickle_path, stats in file_stats.items():
        if 'error' in stats:
            print(f"  {os.path.basename(pickle_path)}: ERROR - {stats['error']}")
        else:
            print(f"  {os.path.basename(pickle_path)}: {stats['coords_found']}/{stats['total_entries']} "
                  f"({stats['enrichment_rate']*100:.1f}% enriched)")
    
    if len(coordinates) > 0:
        print(f"\nCoordinate statistics:")
        print(f"  Total coordinates: {len(coordinates)}")
        print(f"  AP range: {coordinates[:, 0].min():.2f} to {coordinates[:, 0].max():.2f}")
        print(f"  DV range: {coordinates[:, 1].min():.2f} to {coordinates[:, 1].max():.2f}")
        print(f"  LR range: {coordinates[:, 2].min():.2f} to {coordinates[:, 2].max():.2f}")

def main():
    """Main function to voxelize and visualize pickle files."""
    parser = argparse.ArgumentParser(
        description="Voxelize Pickle Files and Create 3D Brain Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Voxelize all pickle files in a directory
  python voxelize_and_visualize.py /path/to/pickle/files/
  
  # Voxelize with custom voxel size
  python voxelize_and_visualize.py /path/to/pickle/files/ --voxel-size 200
  
  # Sample only 1000 entries per file for faster processing
  python voxelize_and_visualize.py /path/to/pickle/files/ --sample-size 1000
  
  # Save visualization to file
  python voxelize_and_visualize.py /path/to/pickle/files/ --output brain_visualization.png
        """
    )
    
    parser.add_argument(
        'input_path',
        nargs='?',
        default=os.path.expanduser("~/Downloads"),
        help='Path to pickle file or directory containing pickle files (default: ~/Downloads)'
    )
    
    parser.add_argument(
        '--voxel-size',
        type=int,
        default=100,
        help='Voxel size in micrometers (default: 100)'
    )
    
    parser.add_argument(
        '--sample-size',
        type=int,
        default=None,
        help='Sample size per file for faster processing (default: use all data)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for visualization image (default: display only)'
    )
    
    args = parser.parse_args()
    
    print("Pickle File Voxelization and 3D Visualization Tool")
    print("=" * 60)
    print(f"Input path: {args.input_path}")
    print(f"Voxel size: {args.voxel_size} μm")
    if args.sample_size:
        print(f"Sample size per file: {args.sample_size}")
    print()
    
    # Load pickle files
    pickle_files = load_pickle_files(args.input_path)
    if not pickle_files:
        return
    
    # Debug coordinate extraction
    debug_coordinate_extraction(pickle_files)
    
    # Collect coordinates
    coordinates, file_stats = collect_coordinates_from_pickles(pickle_files, args.sample_size)
    
    if len(coordinates) == 0:
        print("No coordinates found in any pickle files!")
        return
    
    # Create voxel grid
    voxel_grid, min_coords, voxel_size = create_voxel_grid(coordinates, args.voxel_size)
    
    if voxel_grid is None:
        return
    
    # Print statistics
    print_statistics(file_stats, coordinates)
    
    # Create visualization
    plot_3d_voxel_density(voxel_grid, min_coords, voxel_size, coordinates, args.output)
    
    print("\nVoxelization and visualization completed!")

if __name__ == "__main__":
    main()
