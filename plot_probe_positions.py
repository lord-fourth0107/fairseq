#!/usr/bin/env python3
"""
Plot Horizontal and Vertical Probe Positions for Unique Channels
Shows the physical layout of channels on each probe.
"""

import pickle
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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

def extract_coordinates_from_label(label, coord_lookup):
    """Extract CCF coordinates using channel ID lookup."""
    if not isinstance(label, str):
        return None
    
    parts = label.split('_')
    if len(parts) < 5:
        return None
    
    try:
        # Label format: {session}_{count}_{probe}_{lfp_channel_index}_{brain_region}
        session_id = parts[0]
        probe_id = parts[2]
        channel_id = parts[3]
        
        # Look up coordinates using the channel ID
        lookup_key = (session_id, probe_id, channel_id)
        if lookup_key in coord_lookup:
            return coord_lookup[lookup_key]
        else:
            return None
    except (ValueError, IndexError):
        return None

def load_coordinate_lookup(input_path):
    """Load coordinate lookup from CSV files."""
    import pandas as pd
    
    # Determine the directory to look for CSV files
    if os.path.isfile(input_path):
        csv_dir = os.path.dirname(input_path)
    else:
        csv_dir = input_path
    
    # Look for CSV files
    joined_path = os.path.join(csv_dir, 'joined.csv')
    channels_path = os.path.join(csv_dir, 'channels.csv')
    
    if not os.path.exists(joined_path) or not os.path.exists(channels_path):
        print("CSV files not found, cannot load coordinates")
        return {}
    
    # Load and merge CSV files
    joined_df = pd.read_csv(joined_path)
    channels_df = pd.read_csv(channels_path)
    
    merged_df = pd.merge(
        joined_df,
        channels_df,
        left_on='probe_id',
        right_on='ecephys_probe_id',
        how='inner'
    )
    
    # Create lookup dictionary
    coord_lookup = {}
    for _, row in merged_df.iterrows():
        key = (str(row['session_id']), str(row['probe_id']), str(row['id']))
        coord_lookup[key] = {
            'ap': row['anterior_posterior_ccf_coordinate_x'],
            'dv': row['dorsal_ventral_ccf_coordinate_x'],
            'lr': row['left_right_ccf_coordinate_x'],
            'probe_h': row['probe_horizontal_position'],
            'probe_v': row['probe_vertical_position']
        }
    
    print(f"Loaded coordinate lookup with {len(coord_lookup)} entries")
    return coord_lookup

def collect_unique_probe_channels(pickle_files, coord_lookup):
    """Collect unique channel positions for each probe."""
    probe_data = defaultdict(lambda: {'channels': set(), 'positions': [], 'session_id': None, 'probe_id': None})
    
    print(f"Processing {len(pickle_files)} pickle files...")
    
    for pickle_path in tqdm(pickle_files, desc="Loading probe data"):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            if not isinstance(data, (list, tuple)):
                continue
            
            # Process all data to get all unique channels
            sample_data = data
            
            for entry in sample_data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    label = entry[1]
                    coords = extract_coordinates_from_label(label, coord_lookup)
                    
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
                            
                            # Only add if we haven't seen this channel before
                            if channel_id not in probe_data[probe_key]['channels']:
                                probe_data[probe_key]['channels'].add(channel_id)
                                probe_data[probe_key]['positions'].append([
                                    float(coords['probe_h']), float(coords['probe_v']), channel_id
                                ])
                            
        except Exception as e:
            print(f"Error processing {pickle_path}: {e}")
    
    return probe_data

def plot_probe_positions(probe_data, output_dir="probe_visualizations"):
    """Create plots showing horizontal vs vertical probe positions for unique channels."""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Creating probe position plots for {len(probe_data)} probes...")
    
    for probe_key, data in tqdm(probe_data.items(), desc="Creating plots"):
        if len(data['positions']) == 0:
            continue
            
        positions = np.array(data['positions'])
        channels = data['channels']
        session_id = data['session_id']
        probe_id = data['probe_id']
        
        # Create figure with extended X-axis canvas
        fig, ax = plt.subplots(figsize=(20, 8))
        
        # Plot unique channel positions
        h_positions = positions[:, 0].astype(float)
        v_positions = positions[:, 1].astype(float)
        
        # Use red for first probe, green for second probe
        color = 'red' if probe_id == 810755797 else 'green'
        ax.scatter(h_positions, v_positions, 
                  c=color, s=100, alpha=0.8, edgecolors='black', linewidth=0.5)
        
        # Set labels and title
        ax.set_xlabel('Probe Horizontal Position (μm)', fontsize=12)
        ax.set_ylabel('Probe Vertical Position (μm)', fontsize=12)
        ax.set_title(f'Probe {probe_id} - Unique Channel Positions (Session {session_id})\n{len(positions)} unique channels', fontsize=14)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Set aspect ratio to spread out X-axis more
        ax.set_aspect('auto')
        
        # Add statistics text
        h_range = h_positions.max() - h_positions.min()
        v_range = v_positions.max() - v_positions.min()
        stats_text = f'H range: {h_range:.1f} μm\nV range: {v_range:.1f} μm\nChannels: {len(positions)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the plot
        output_path = os.path.join(output_dir, f"probe_{probe_id}_positions.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        
        plt.close()  # Close to free memory

def create_combined_plot(probe_data, output_dir="probe_visualizations"):
    """Create a combined plot showing all probes together."""
    
    print("Creating combined probe positions plot...")
    
    fig, ax = plt.subplots(figsize=(25, 10))
    
    # Use red and green colors for the two probes
    colors = ['red', 'green']
    
    for i, (probe_key, data) in enumerate(probe_data.items()):
        if len(data['positions']) == 0:
            continue
            
        positions = np.array(data['positions'])
        session_id = data['session_id']
        probe_id = data['probe_id']
        
        # Plot this probe's unique channel positions
        h_positions = positions[:, 0].astype(float)
        v_positions = positions[:, 1].astype(float)
        ax.scatter(h_positions, v_positions, 
                  c=[colors[i]] * len(positions), s=80, alpha=0.7, 
                  label=f"Probe {probe_id} ({len(positions)} ch)", edgecolors='black', linewidth=0.3)
    
    # Set labels and title
    ax.set_xlabel('Probe Horizontal Position (μm)', fontsize=12)
    ax.set_ylabel('Probe Vertical Position (μm)', fontsize=12)
    ax.set_title('All Probes - Unique Channel Positions', fontsize=16)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set aspect ratio to spread out X-axis more
    ax.set_aspect('auto')
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    
    # Save the combined plot
    output_path = os.path.join(output_dir, "all_probes_positions.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved combined plot: {output_path}")
    
    plt.close()

def print_probe_statistics(probe_data):
    """Print statistics for each probe."""
    print("\n" + "=" * 80)
    print("PROBE POSITION STATISTICS")
    print("=" * 80)
    
    for probe_key, data in probe_data.items():
        if len(data['positions']) == 0:
            continue
            
        positions = np.array(data['positions'])
        session_id = data['session_id']
        probe_id = data['probe_id']
        
        # Extract only the numeric columns (first 2) for statistics
        h_positions = positions[:, 0].astype(float)
        v_positions = positions[:, 1].astype(float)
        
        print(f"\nProbe {probe_id} (Session {session_id}):")
        print(f"  Unique channels: {len(positions)}")
        print(f"  H range: {h_positions.min():.1f} to {h_positions.max():.1f} μm")
        print(f"  V range: {v_positions.min():.1f} to {v_positions.max():.1f} μm")
        print(f"  Probe extent: {h_positions.max() - h_positions.min():.1f} × {v_positions.max() - v_positions.min():.1f} μm")

def main():
    """Main function to plot probe positions."""
    parser = argparse.ArgumentParser(
        description="Plot Horizontal and Vertical Probe Positions for Unique Channels",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Plot all probes in a directory
  python plot_probe_positions.py /path/to/pickle/files/
  
  # Plot specific probe file
  python plot_probe_positions.py /path/to/probe.pickle
  
  # Save to custom output directory
  python plot_probe_positions.py /path/to/pickle/files/ --output-dir my_plots
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
    
    print("Probe Position Visualization Tool")
    print("=" * 50)
    print(f"Input path: {args.input_path}")
    print(f"Output directory: {args.output_dir}")
    print()
    
    # Load pickle files
    pickle_files = load_pickle_files(args.input_path)
    if not pickle_files:
        return
    
    # Load coordinate lookup
    coord_lookup = load_coordinate_lookup(args.input_path)
    if not coord_lookup:
        print("No coordinate data available, cannot create visualizations")
        return
    
    # Collect unique probe channel data
    probe_data = collect_unique_probe_channels(pickle_files, coord_lookup)
    
    if not probe_data:
        print("No probe data found!")
        return
    
    # Print statistics
    print_probe_statistics(probe_data)
    
    # Create individual probe position plots
    plot_probe_positions(probe_data, args.output_dir)
    
    # Create combined plot
    create_combined_plot(probe_data, args.output_dir)
    
    print(f"\nVisualization completed! Check the '{args.output_dir}' directory for images.")

if __name__ == "__main__":
    main()
