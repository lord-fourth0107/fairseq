#!/usr/bin/env python3
"""
Visualize Individual Channel Coordinates for Each Probe
Shows 3D scatter plot of all channels for each probe with different colors.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import glob
from collections import defaultdict
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

def collect_probe_channel_data(pickle_files):
    """Collect channel coordinates for each probe."""
    probe_data = defaultdict(lambda: {'channels': [], 'coordinates': [], 'session_id': None, 'probe_id': None})
    
    print(f"Processing {len(pickle_files)} pickle files...")
    
    for pickle_path in tqdm(pickle_files, desc="Loading probe data"):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, (list, tuple)):
                continue
            
            # Sample data for efficiency (first 1000 entries per file)
            sample_data = data[:1000] if len(data) > 1000 else data
            
            for entry in sample_data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    label = entry[1]
                    coords = extract_coordinates_from_label(label)
                    
                    if coords is not None:
                        # Extract probe info from label
                        parts = label.split('_')
                        if len(parts) >= 4:
                            session_id = parts[0]
                            channel_id = parts[3]
                            probe_id = parts[2]
                            
                            probe_key = f"{session_id}_{probe_id}"
                            probe_data[probe_key]['session_id'] = session_id
                            probe_data[probe_key]['probe_id'] = probe_id
                            probe_data[probe_key]['channels'].append(channel_id)
                            probe_data[probe_key]['coordinates'].append([
                                coords['ap'], coords['dv'], coords['lr'], 
                                coords['probe_h'], coords['probe_v']
                            ])
                            
        except Exception as e:
            print(f"Error processing {pickle_path}: {e}")
    
    return probe_data

def create_brain_outline():
    """Create a simple brain outline for visualization."""
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    u, v = np.meshgrid(u, v)
    
    # Brain-like ellipsoid (approximate mouse brain dimensions)
    x = 6000 * np.cos(u) * np.sin(v)
    y = 4000 * np.sin(u) * np.sin(v)
    z = 3000 * np.cos(v)
    
    return x, y, z

def plot_probe_channels_3d(probe_data, output_dir="probe_visualizations"):
    """Create 3D visualizations for each probe showing channel coordinates."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating 3D visualizations for {len(probe_data)} probes...")
    
    for probe_key, data in tqdm(probe_data.items(), desc="Creating visualizations"):
        if len(data['coordinates']) == 0:
            continue
            
        coordinates = np.array(data['coordinates'])
        channels = data['channels']
        session_id = data['session_id']
        probe_id = data['probe_id']
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # 3D CCF coordinates plot
        ax1 = fig.add_subplot(221, projection='3d')
        
        # Create brain outline
        brain_x, brain_y, brain_z = create_brain_outline()
        ax1.plot_surface(brain_x, brain_y, brain_z, alpha=0.1, color='lightgray', label='Brain outline')
        
        # Plot channel coordinates
        scatter1 = ax1.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 
                              c=range(len(coordinates)), cmap='viridis', s=50, alpha=0.8)
        
        # Add colorbar
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.5, aspect=20)
        cbar1.set_label('Channel Index', rotation=270, labelpad=15)
        
        # Set labels and title
        ax1.set_xlabel('Anterior-Posterior (μm)')
        ax1.set_ylabel('Dorsal-Ventral (μm)')
        ax1.set_zlabel('Left-Right (μm)')
        ax1.set_title(f'Probe {probe_id} - CCF Coordinates\n{len(coordinates)} channels')
        
        # Set equal aspect ratio
        max_range = np.array([coordinates[:, 0].max() - coordinates[:, 0].min(),
                             coordinates[:, 1].max() - coordinates[:, 1].min(),
                             coordinates[:, 2].max() - coordinates[:, 2].min()]).max() / 2.0
        mid_x = (coordinates[:, 0].max() + coordinates[:, 0].min()) * 0.5
        mid_y = (coordinates[:, 1].max() + coordinates[:, 1].min()) * 0.5
        mid_z = (coordinates[:, 2].max() + coordinates[:, 2].min()) * 0.5
        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)
        
        # Probe horizontal vs vertical positions (2D)
        ax2 = fig.add_subplot(222)
        probe_h = [coord[3] for coord in data['coordinates'] if len(coord) > 3]
        probe_v = [coord[4] for coord in data['coordinates'] if len(coord) > 4]
        
        if probe_h and probe_v:
            scatter2 = ax2.scatter(probe_h, probe_v, c=range(len(probe_h)), cmap='plasma', s=50, alpha=0.8)
            ax2.set_xlabel('Probe Horizontal Position (μm)')
            ax2.set_ylabel('Probe Vertical Position (μm)')
            ax2.set_title(f'Probe {probe_id} - Horizontal vs Vertical\n{len(probe_h)} channels')
            ax2.grid(True, alpha=0.3)
            
            # Add colorbar
            cbar2 = plt.colorbar(scatter2, ax=ax2)
            cbar2.set_label('Channel Index', rotation=270, labelpad=15)
        else:
            ax2.text(0.5, 0.5, 'No probe position data available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f'Probe {probe_id} - Horizontal vs Vertical\nNo data')
        
        # AP vs DV projection
        ax3 = fig.add_subplot(223)
        scatter3 = ax3.scatter(coordinates[:, 0], coordinates[:, 1], 
                              c=range(len(coordinates)), cmap='viridis', s=50, alpha=0.8)
        ax3.set_xlabel('Anterior-Posterior (μm)')
        ax3.set_ylabel('Dorsal-Ventral (μm)')
        ax3.set_title(f'Probe {probe_id} - AP vs DV Projection\n{len(coordinates)} channels')
        ax3.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar3 = plt.colorbar(scatter3, ax=ax3)
        cbar3.set_label('Channel Index', rotation=270, labelpad=15)
        
        # LR vs DV projection
        ax4 = fig.add_subplot(224)
        scatter4 = ax4.scatter(coordinates[:, 2], coordinates[:, 1], 
                              c=range(len(coordinates)), cmap='viridis', s=50, alpha=0.8)
        ax4.set_xlabel('Left-Right (μm)')
        ax4.set_ylabel('Dorsal-Ventral (μm)')
        ax4.set_title(f'Probe {probe_id} - LR vs DV Projection\n{len(coordinates)} channels')
        ax4.grid(True, alpha=0.3)
        
        # Add colorbar
        cbar4 = plt.colorbar(scatter4, ax=ax4)
        cbar4.set_label('Channel Index', rotation=270, labelpad=15)
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f"probe_{probe_id}_session_{session_id}_channels.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close()  # Close to free memory

def create_summary_plot(probe_data, output_dir="probe_visualizations"):
    """Create a summary plot showing all probes together."""
    
    print("Creating summary plot with all probes...")
    
    fig = plt.figure(figsize=(20, 15))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create brain outline
    brain_x, brain_y, brain_z = create_brain_outline()
    ax.plot_surface(brain_x, brain_y, brain_z, alpha=0.1, color='lightgray', label='Brain outline')
    
    # Color map for different probes
    colors = plt.cm.tab20(np.linspace(0, 1, len(probe_data)))
    
    all_coordinates = []
    probe_labels = []
    
    for i, (probe_key, data) in enumerate(probe_data.items()):
        if len(data['coordinates']) == 0:
            continue
            
        coordinates = np.array(data['coordinates'])
        session_id = data['session_id']
        probe_id = data['probe_id']
        
        # Sample coordinates for visualization (max 100 per probe)
        if len(coordinates) > 100:
            indices = np.random.choice(len(coordinates), 100, replace=False)
            coordinates = coordinates[indices]
        
        all_coordinates.append(coordinates)
        probe_labels.append(f"Probe {probe_id}")
        
        # Plot this probe's channels
        ax.scatter(coordinates[:, 0], coordinates[:, 1], coordinates[:, 2], 
                  c=[colors[i]] * len(coordinates), s=30, alpha=0.7, 
                  label=f"Probe {probe_id} ({len(data['coordinates'])} ch)")
    
    # Set labels and title
    ax.set_xlabel('Anterior-Posterior (μm)')
    ax.set_ylabel('Dorsal-Ventral (μm)')
    ax.set_zlabel('Left-Right (μm)')
    ax.set_title(f'All Probes Channel Distribution\n{len(probe_data)} probes total')
    
    # Set equal aspect ratio
    if all_coordinates:
        all_coords = np.vstack(all_coordinates)
        max_range = np.array([all_coords[:, 0].max() - all_coords[:, 0].min(),
                             all_coords[:, 1].max() - all_coords[:, 1].min(),
                             all_coords[:, 2].max() - all_coords[:, 2].min()]).max() / 2.0
        mid_x = (all_coords[:, 0].max() + all_coords[:, 0].min()) * 0.5
        mid_y = (all_coords[:, 1].max() + all_coords[:, 1].min()) * 0.5
        mid_z = (all_coords[:, 2].max() + all_coords[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the summary plot
    output_path = os.path.join(output_dir, "all_probes_summary.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved summary: {output_path}")
    
    plt.close()

def print_probe_statistics(probe_data):
    """Print statistics for each probe."""
    print("\n" + "=" * 80)
    print("PROBE CHANNEL STATISTICS")
    print("=" * 80)
    
    for probe_key, data in probe_data.items():
        if len(data['coordinates']) == 0:
            continue
            
        coordinates = np.array(data['coordinates'])
        session_id = data['session_id']
        probe_id = data['probe_id']
        
        print(f"\nProbe {probe_id} (Session {session_id}):")
        print(f"  Channels: {len(coordinates)}")
        print(f"  AP range: {coordinates[:, 0].min():.1f} to {coordinates[:, 0].max():.1f} μm")
        print(f"  DV range: {coordinates[:, 1].min():.1f} to {coordinates[:, 1].max():.1f} μm")
        print(f"  LR range: {coordinates[:, 2].min():.1f} to {coordinates[:, 2].max():.1f} μm")
        print(f"  Spatial extent: {coordinates[:, 0].max() - coordinates[:, 0].min():.1f} × {coordinates[:, 1].max() - coordinates[:, 1].min():.1f} × {coordinates[:, 2].max() - coordinates[:, 2].min():.1f} μm")
        
        # Add probe position statistics if available
        if len(coordinates[0]) > 3:
            probe_h = coordinates[:, 3]
            probe_v = coordinates[:, 4]
            print(f"  Probe H range: {probe_h.min():.1f} to {probe_h.max():.1f} μm")
            print(f"  Probe V range: {probe_v.min():.1f} to {probe_v.max():.1f} μm")
            print(f"  Probe extent: {probe_h.max() - probe_h.min():.1f} × {probe_v.max() - probe_v.min():.1f} μm")

def main():
    """Main function to visualize probe channels."""
    parser = argparse.ArgumentParser(
        description="Visualize Individual Channel Coordinates for Each Probe",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize all probes in a directory
  python visualize_probe_channels.py /path/to/pickle/files/
  
  # Visualize specific probe file
  python visualize_probe_channels.py /path/to/probe.pickle
  
  # Save to custom output directory
  python visualize_probe_channels.py /path/to/pickle/files/ --output-dir my_visualizations
        """
    )
    
    parser.add_argument(
        'input_path',
        nargs='?',
        default=os.path.expanduser("~/Downloads"),
        help='Path to pickle file or directory containing pickle files (default: ~/Downloads)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default="probe_visualizations",
        help='Output directory for visualization images (default: probe_visualizations)'
    )
    
    args = parser.parse_args()
    
    print("Probe Channel Coordinate Visualization Tool")
    print("=" * 60)
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load pickle files
    pickle_files = load_pickle_files(args.input_path)
    if not pickle_files:
        return
    
    # Collect probe data
    probe_data = collect_probe_channel_data(pickle_files)
    
    if not probe_data:
        print("No probe data found!")
        return
    
    # Print statistics
    print_probe_statistics(probe_data)
    
    # Create individual probe visualizations
    plot_probe_channels_3d(probe_data, args.output_dir)
    
    # Create summary plot
    create_summary_plot(probe_data, args.output_dir)
    
    print(f"\nVisualization completed! Check the '{args.output_dir}' directory for images.")

if __name__ == "__main__":
    main()
