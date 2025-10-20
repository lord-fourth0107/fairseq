#!/usr/bin/env python3
"""
Multi-Channel Voxel Cube Visualization

This script creates 3D visualizations where:
1. The brain is divided into 1mm³ voxels
2. Each coordinate paints an entire cube
3. Multiple channels selecting the same voxel repaint it with different hues
4. All files from the previous example are included

Based on the perfect cube visualization approach but with hue variation for overlapping voxels.
"""

import os
import sys
import pickle
import logging
import argparse
from datetime import datetime
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from tqdm import tqdm
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'multi_channel_voxel_viz_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def _looks_like_number(s: str) -> bool:
    """Check if string looks like a number (int or float)."""
    try:
        float(s)
        return True
    except ValueError:
        return False

def is_label_enriched(label):
    """Check if label contains enriched coordinate information.
    Expected enriched label suffix: _{ap}_{dv}_{lr}_{probe_h}_{probe_v}
    We check that the label has at least 9 underscore-separated parts and the
    last 5 parts are numeric-like.
    """
    if not isinstance(label, str):
        return False
    parts = label.split('_')
    if len(parts) < 9:
        return False
    tail = parts[-5:]
    return all(_looks_like_number(x) for x in tail)

def extract_ccf_coordinates(label):
    """Extract CCF coordinates from enriched label string."""
    if not is_label_enriched(label):
        return None
    
    parts = label.split('_')
    try:
        # Last 5 parts should be: ap, dv, lr, probe_h, probe_v
        ap = float(parts[-5])
        dv = float(parts[-4])
        lr = float(parts[-3])
        return (ap, dv, lr)
    except (ValueError, IndexError):
        return None

def extract_session_probe_info(label):
    """Extract session and probe information from label string."""
    if not isinstance(label, str):
        return None
    
    parts = label.split('_')
    if len(parts) >= 3:
        session_id = parts[0]  # First part is session ID
        probe_id = parts[2]   # Third part is probe ID
        return session_id, probe_id
    return None

def collect_all_coordinates(pickle_dir, voxel_size=1.0):
    """Collect all coordinates from pickle files and organize by voxel."""
    logger.info(f"Scanning directory: {pickle_dir}")
    
    # Find all pickle files
    all_pickle_files = []
    for root, dirs, files in os.walk(pickle_dir):
        for file in files:
            if file.endswith('.pickle'):
                all_pickle_files.append(os.path.join(root, file))
    
    logger.info(f"Found {len(all_pickle_files)} pickle files")
    
    if not all_pickle_files:
        logger.error("No pickle files found!")
        return None, None, None
    
    # Data structures for voxelization
    voxel_to_channels = defaultdict(set)  # (voxel_x, voxel_y, voxel_z) -> set of unique channel keys
    voxel_to_channel_info = defaultdict(list)  # (voxel_x, voxel_y, voxel_z) -> list of channel info
    session_probe_coords = defaultdict(list)  # (session_id, probe_id) -> list of coords
    all_coords = []  # All coordinates for bounds calculation
    
    # Process each pickle file
    successful_files = 0
    for pickle_path in tqdm(all_pickle_files, desc="Processing files", unit="file"):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            filename = os.path.basename(pickle_path)
            logger.info(f"Processing {filename}: {len(data)} entries")
            
            file_channels = 0
            seen_channels_in_file = set()  # Track unique channels per file
            
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    signal, label = entry[0], entry[1]
                    
                    if is_label_enriched(label):
                        # Extract CCF coordinates
                        ccf_coords = extract_ccf_coordinates(label)
                        if ccf_coords:
                            ap, dv, lr = ccf_coords
                            
                            # Convert to voxel coordinates (1mm³ voxels)
                            # Coordinates are in micrometers, voxel_size is in mm
                            voxel_size_um = voxel_size * 1000  # Convert mm to μm
                            voxel_x = int(np.round(ap / voxel_size_um))
                            voxel_y = int(np.round(dv / voxel_size_um))
                            voxel_z = int(np.round(lr / voxel_size_um))
                            
                            voxel_key = (voxel_x, voxel_y, voxel_z)
                            
                            # Create unique channel key for deduplication
                            session_probe_info = extract_session_probe_info(label)
                            if session_probe_info:
                                session_id, probe_id = session_probe_info
                                # Use session_id, probe_id, and channel_id for uniqueness
                                parts = label.split('_')
                                channel_id = parts[3] if len(parts) > 3 else 'unknown'
                                channel_key = (session_id, probe_id, channel_id)
                                
                                # Only add if we haven't seen this channel before
                                if channel_key not in seen_channels_in_file:
                                    seen_channels_in_file.add(channel_key)
                                    
                                    # Store channel information
                                    channel_info = {
                                        'coordinates': ccf_coords,
                                        'session_id': session_id,
                                        'probe_id': probe_id,
                                        'channel_id': channel_id,
                                        'filename': filename
                                    }
                                    
                                    # Add to voxel (using set for deduplication)
                                    voxel_to_channels[voxel_key].add(channel_key)
                                    voxel_to_channel_info[voxel_key].append(channel_info)
                                    all_coords.append(ccf_coords)
                                    file_channels += 1
                                    
                                    # Store coordinates for session/probe grouping
                                    session_probe_coords[(session_id, probe_id)].append(ccf_coords)
            
            if file_channels > 0:
                successful_files += 1
                logger.info(f"  Added {file_channels} unique channels from {filename}")
            
        except Exception as e:
            logger.error(f"Error processing {pickle_path}: {e}")
            continue
    
    logger.info(f"Successfully processed {successful_files} files")
    logger.info(f"Total coordinates collected: {len(all_coords)}")
    logger.info(f"Unique voxels: {len(voxel_to_channels)}")
    
    return voxel_to_channels, session_probe_coords, all_coords

def create_voxel_grid_info(all_coords, voxel_size=1.0):
    """Create grid information for voxelization."""
    if not all_coords:
        return None
    
    coords_array = np.array(all_coords)
    
    # Calculate bounds in micrometers
    ap_min, ap_max = coords_array[:, 0].min(), coords_array[:, 0].max()
    dv_min, dv_max = coords_array[:, 1].min(), coords_array[:, 1].max()
    lr_min, lr_max = coords_array[:, 2].min(), coords_array[:, 2].max()
    
    logger.info(f"Coordinate ranges:")
    logger.info(f"  AP: {ap_min:.1f} to {ap_max:.1f} μm")
    logger.info(f"  DV: {dv_min:.1f} to {dv_max:.1f} μm")
    logger.info(f"  LR: {lr_min:.1f} to {lr_max:.1f} μm")
    
    # Convert to mm and create grid bounds
    voxel_size_um = voxel_size * 1000  # Convert mm to μm
    
    ap_min_mm = np.floor(ap_min / voxel_size_um) * voxel_size_um
    ap_max_mm = np.ceil(ap_max / voxel_size_um) * voxel_size_um
    dv_min_mm = np.floor(dv_min / voxel_size_um) * voxel_size_um
    dv_max_mm = np.ceil(dv_max / voxel_size_um) * voxel_size_um
    lr_min_mm = np.floor(lr_min / voxel_size_um) * voxel_size_um
    lr_max_mm = np.ceil(lr_max / voxel_size_um) * voxel_size_um
    
    # Calculate grid dimensions
    ap_size = int((ap_max_mm - ap_min_mm) / voxel_size_um) + 1
    dv_size = int((dv_max_mm - dv_min_mm) / voxel_size_um) + 1
    lr_size = int((lr_max_mm - lr_min_mm) / voxel_size_um) + 1
    
    logger.info(f"Voxel grid dimensions: {ap_size} × {dv_size} × {lr_size}")
    logger.info(f"Total voxels: {ap_size * dv_size * lr_size:,}")
    
    return {
        'ap_min': ap_min_mm,
        'ap_max': ap_max_mm,
        'dv_min': dv_min_mm,
        'dv_max': dv_max_mm,
        'lr_min': lr_min_mm,
        'lr_max': lr_max_mm,
        'voxel_size_um': voxel_size_um,
        'ap_size': ap_size,
        'dv_size': dv_size,
        'lr_size': lr_size
    }

def create_perfect_cube_vertices(center, size):
    """Create vertices for a perfect cube centered at the given point."""
    half_size = size / 2
    x, y, z = center
    
    vertices = np.array([
        [x - half_size, y - half_size, z - half_size],  # 0
        [x + half_size, y - half_size, z - half_size],  # 1
        [x + half_size, y + half_size, z - half_size],  # 2
        [x - half_size, y + half_size, z - half_size],  # 3
        [x - half_size, y - half_size, z + half_size],  # 4
        [x + half_size, y - half_size, z + half_size],  # 5
        [x + half_size, y + half_size, z + half_size],  # 6
        [x - half_size, y + half_size, z + half_size]   # 7
    ])
    
    # Define the 6 faces of the cube
    faces = [
        [0, 1, 2, 3],  # Bottom face
        [4, 5, 6, 7],  # Top face
        [0, 1, 5, 4],  # Front face
        [2, 3, 7, 6],  # Back face
        [0, 3, 7, 4],  # Left face
        [1, 2, 6, 5]   # Right face
    ]
    
    return vertices, faces

def get_hue_for_channel_count(channel_count, max_channels=10):
    """Get hue value based on number of channels in voxel."""
    # Use HSV color space: hue varies from 0 to 1 based on channel count
    hue = (channel_count - 1) / max(max_channels - 1, 1)
    hue = min(hue, 1.0)  # Cap at 1.0
    return hue

def create_multi_channel_cube_visualization(voxel_to_channels, grid_info, output_dir, voxel_size=1.0):
    """Create 3D visualization with perfect cubes and hue variation."""
    logger.info("Creating multi-channel cube visualization...")
    
    if not voxel_to_channels:
        logger.error("No voxel data to visualize!")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find maximum channel count for hue scaling
    max_channels = max(len(channels) for channels in voxel_to_channels.values())
    logger.info(f"Maximum channels per voxel: {max_channels}")
    
    # Create 3D plot
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Process each voxel and create individual cube collections
    voxel_size_um = grid_info['voxel_size_um']
    
    # Process each voxel
    for voxel_key, unique_channels in tqdm(voxel_to_channels.items(), desc="Creating cubes"):
        voxel_x, voxel_y, voxel_z = voxel_key
        
        # Convert voxel coordinates back to physical coordinates
        ap_coord = voxel_x * voxel_size_um + grid_info['ap_min']
        dv_coord = voxel_y * voxel_size_um + grid_info['dv_min']
        lr_coord = voxel_z * voxel_size_um + grid_info['lr_min']
        
        # Create cube vertices
        cube_center = (ap_coord + voxel_size_um/2, dv_coord + voxel_size_um/2, lr_coord + voxel_size_um/2)
        vertices, faces = create_perfect_cube_vertices(cube_center, voxel_size_um)
        
        # Determine color based on number of unique channels
        channel_count = len(unique_channels)
        hue = get_hue_for_channel_count(channel_count, max_channels)
        
        # Convert HSV to RGB
        import matplotlib.colors as mcolors
        rgb_color = mcolors.hsv_to_rgb([hue, 0.8, 0.9])  # High saturation and value
        
        # Determine alpha based on channel count (more channels = more opaque)
        alpha = min(0.3 + (channel_count - 1) * 0.1, 0.9)
        
        # Create cube faces
        cube_faces = []
        for face in faces:
            face_vertices = vertices[face]
            cube_faces.append(face_vertices)
        
        # Create individual Poly3DCollection for this cube
        cube_collection = Poly3DCollection(cube_faces, facecolor=rgb_color, alpha=alpha, 
                                         edgecolor='black', linewidth=0.1)
        ax.add_collection3d(cube_collection)
    
    logger.info(f"Created {len(voxel_to_channels)} cubes")
    
    # Set labels and title
    ax.set_xlabel('Anterior-Posterior (μm)', fontsize=14)
    ax.set_ylabel('Dorsal-Ventral (μm)', fontsize=14)
    ax.set_zlabel('Left-Right (μm)', fontsize=14)
    ax.set_title(f'Multi-Channel Voxel Cubes (1mm³)\nMax Channels per Voxel: {max_channels}', fontsize=16)
    
    # Calculate actual coordinate ranges from the cubes
    all_cube_coords = []
    for voxel_key, unique_channels in voxel_to_channels.items():
        voxel_x, voxel_y, voxel_z = voxel_key
        ap_coord = voxel_x * voxel_size_um + grid_info['ap_min']
        dv_coord = voxel_y * voxel_size_um + grid_info['dv_min']
        lr_coord = voxel_z * voxel_size_um + grid_info['lr_min']
        all_cube_coords.append([ap_coord, dv_coord, lr_coord])
    
    if all_cube_coords:
        all_cube_coords = np.array(all_cube_coords)
        
        # Set equal aspect ratio based on actual data
        max_range = np.array([
            all_cube_coords[:, 0].max() - all_cube_coords[:, 0].min(),
            all_cube_coords[:, 1].max() - all_cube_coords[:, 1].min(),
            all_cube_coords[:, 2].max() - all_cube_coords[:, 2].min()
        ]).max() / 2.0
        
        mid_x = np.mean(all_cube_coords[:, 0])
        mid_y = np.mean(all_cube_coords[:, 1])
        mid_z = np.mean(all_cube_coords[:, 2])
        
        # Add some padding around the data
        padding = max_range * 0.1
        
        ax.set_xlim(mid_x - max_range - padding, mid_x + max_range + padding)
        ax.set_ylim(mid_y - max_range - padding, mid_y + max_range + padding)
        ax.set_zlim(mid_z - max_range - padding, mid_z + max_range + padding)
        
        logger.info(f"Data center: AP={mid_x:.1f}, DV={mid_y:.1f}, LR={mid_z:.1f}")
        logger.info(f"Data range: {max_range:.1f} μm")
    else:
        # Fallback to grid bounds if no cubes
        ax.set_xlim(grid_info['ap_min'], grid_info['ap_max'])
        ax.set_ylim(grid_info['dv_min'], grid_info['dv_max'])
        ax.set_zlim(grid_info['lr_min'], grid_info['lr_max'])
    
    # Set viewing angle
    ax.view_init(elev=20, azim=45)
    
    # Add colorbar with reasonable scaling
    # Cap the colorbar at a reasonable maximum for better visualization
    colorbar_max = min(max_channels, 20)  # Cap at 20 for better color distribution
    sm = plt.cm.ScalarMappable(cmap=plt.cm.hsv, norm=plt.Normalize(vmin=1, vmax=colorbar_max))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Number of Channels per Voxel', fontsize=12)
    
    # Add note about actual max if it's higher than colorbar max
    if max_channels > colorbar_max:
        ax.text2D(0.02, 0.98, f'Note: Actual max = {max_channels}', 
                 transform=ax.transAxes, fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Save the plot
    output_file = os.path.join(output_dir, f"multi_channel_voxel_cubes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    logger.info(f"Multi-channel cube visualization saved to: {output_file}")
    plt.close()
    
    return output_file

def save_voxel_statistics(voxel_to_channels, voxel_to_channel_info, session_probe_coords, grid_info, output_dir):
    """Save detailed statistics about voxel usage."""
    logger.info("Saving voxel statistics...")
    
    # Calculate statistics
    voxel_stats = {
        'grid_info': grid_info,
        'total_voxels': len(voxel_to_channels),
        'total_unique_channels': sum(len(unique_channels) for unique_channels in voxel_to_channels.values()),
        'max_channels_per_voxel': max(len(unique_channels) for unique_channels in voxel_to_channels.values()),
        'min_channels_per_voxel': min(len(unique_channels) for unique_channels in voxel_to_channels.values()),
        'avg_channels_per_voxel': sum(len(unique_channels) for unique_channels in voxel_to_channels.values()) / len(voxel_to_channels),
        'unique_sessions': len(set(channel['session_id'] for channels in voxel_to_channel_info.values() for channel in channels)),
        'unique_probes': len(set(channel['probe_id'] for channels in voxel_to_channel_info.values() for channel in channels)),
        'unique_session_probe_combinations': len(session_probe_coords)
    }
    
    # Channel count distribution
    channel_counts = [len(unique_channels) for unique_channels in voxel_to_channels.values()]
    voxel_stats['channel_count_distribution'] = {
        str(i): channel_counts.count(i) for i in range(1, max(channel_counts) + 1)
    }
    
    # Save to JSON
    output_file = os.path.join(output_dir, f"voxel_statistics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(voxel_stats, f, indent=2)
    
    logger.info(f"Voxel statistics saved to: {output_file}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("VOXEL STATISTICS SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total voxels: {voxel_stats['total_voxels']:,}")
    logger.info(f"Total unique channels: {voxel_stats['total_unique_channels']:,}")
    logger.info(f"Max channels per voxel: {voxel_stats['max_channels_per_voxel']}")
    logger.info(f"Min channels per voxel: {voxel_stats['min_channels_per_voxel']}")
    logger.info(f"Avg channels per voxel: {voxel_stats['avg_channels_per_voxel']:.2f}")
    logger.info(f"Unique sessions: {voxel_stats['unique_sessions']}")
    logger.info(f"Unique probes: {voxel_stats['unique_probes']}")
    logger.info(f"Session-probe combinations: {voxel_stats['unique_session_probe_combinations']}")
    
    return output_file

def main():
    parser = argparse.ArgumentParser(description='Multi-Channel Voxel Cube Visualization')
    parser.add_argument('--pickle_dir', type=str, required=True,
                       help='Directory containing pickle files')
    parser.add_argument('--output_dir', type=str, default='multi_channel_voxel_viz',
                       help='Output directory for visualizations')
    parser.add_argument('--voxel_size', type=float, default=1.0,
                       help='Voxel size in mm (default: 1.0)')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("MULTI-CHANNEL VOXEL CUBE VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"Pickle directory: {args.pickle_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Voxel size: {args.voxel_size} mm")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Collect all coordinates
        voxel_to_channels, voxel_to_channel_info, session_probe_coords, all_coords = collect_all_coordinates(
            args.pickle_dir, args.voxel_size
        )
        
        if not voxel_to_channels:
            logger.error("No voxel data collected!")
            return
        
        # Create grid information
        grid_info = create_voxel_grid_info(all_coords, args.voxel_size)
        if not grid_info:
            logger.error("Failed to create grid information!")
            return
        
        # Create visualization
        viz_file = create_multi_channel_cube_visualization(
            voxel_to_channels, grid_info, args.output_dir, args.voxel_size
        )
        
        # Save statistics
        stats_file = save_voxel_statistics(
            voxel_to_channels, voxel_to_channel_info, session_probe_coords, grid_info, args.output_dir
        )
        
        logger.info("=" * 60)
        logger.info("VISUALIZATION COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"Visualization saved to: {viz_file}")
        logger.info(f"Statistics saved to: {stats_file}")
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        raise

if __name__ == "__main__":
    main()
