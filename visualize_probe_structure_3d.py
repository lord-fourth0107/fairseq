#!/usr/bin/env python3
"""
3D Probe Structure Visualization
Creates a 3D voxel visualization of probe structures from successfully enriched pickle files.
Only processes files marked as SUCCESS in validation results.
"""

import pickle
import os
import sys
import argparse
import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import json
import logging
from datetime import datetime
from tqdm import tqdm

def setup_logging():
    """Setup logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )

def _looks_like_number(s: str) -> bool:
    """Check if a string looks like a number."""
    try:
        float(s)
        return True
    except Exception:
        return False

def is_label_enriched(label: str) -> bool:
    """Check if a label has the enriched format with CCF coordinates."""
    if not isinstance(label, str):
        return False
    parts = label.split('_')
    if len(parts) < 9:
        return False
    tail = parts[-5:]
    return all(_looks_like_number(x) for x in tail)

def extract_ccf_coordinates(label: str):
    """Extract CCF coordinates (ap, dv, lr) from an enriched label."""
    if not is_label_enriched(label):
        return None
    parts = label.split('_')
    try:
        ap, dv, lr = map(float, parts[-5:-2])  # Last 5 parts: ap, dv, lr, probe_h, probe_v
        return (ap, dv, lr)
    except Exception:
        return None

def extract_session_probe_info(label: str):
    """Extract session and probe information from label."""
    if not isinstance(label, str) or '_' not in label:
        return None
    parts = label.split('_')
    if len(parts) < 4:
        return None
    try:
        session_id = parts[0]
        probe_id = parts[2]
        return session_id, probe_id
    except Exception:
        return None

def get_valid_sessions():
    """Get list of valid sessions from validation results."""
    # Based on your validation output, these are the SUCCESS files
    valid_sessions = [
        "715093703_810755797.pickle",
        "715093703_810755801.pickle", 
        "715093703_810755805.pickle",
        "715093703_810755807.pickle",
        "719161530_729445648.pickle",
        "719161530_729445650.pickle",
        "719161530_729445652.pickle",
        "719161530_729445656.pickle",
        "719161530_729445658.pickle",
        "721123822_760213137.pickle",
        "721123822_760213142.pickle",
        "721123822_760213145.pickle",
        "721123822_760213147.pickle",
        "721123822_760213150.pickle",
        "721123822_760213153.pickle",
        "743475441_769325777.pickle",
        "743475441_769325781.pickle",
        "743475441_769326325.pickle",
        "743475441_769326329.pickle",
        "743475441_769326332.pickle",
        "744228101_757904508.pickle",
        "744228101_757904510.pickle",
        "744228101_757904516.pickle",
        "744228101_757904520.pickle",
        "744228101_757904522.pickle",
        "746083955_760647913.pickle",
        "750332458_757904554.pickle",
        "750749662_769322802.pickle",
        "750749662_769322804.pickle",
        "750749662_769322806.pickle",
        "750749662_769322808.pickle",
        "750749662_769322810.pickle",
        "750749662_769322812.pickle",
        "751348571_757984813.pickle",
        "751348571_757984818.pickle",
        "751348571_757984820.pickle",
        "751348571_757984822.pickle",
        "751348571_757984826.pickle",
        "751348571_757984830.pickle",
        "754312389_756781553.pickle",
        "754312389_756781555.pickle",
        "754312389_756781557.pickle",
        "754312389_756781559.pickle",
        "754312389_756781561.pickle",
        "754829445_760640033.pickle",
        "754829445_760640037.pickle",
        "754829445_760640039.pickle",
        "754829445_760640043.pickle",
        "754829445_760640049.pickle",
        "755434585_760642621.pickle",
        "755434585_760642624.pickle",
        "755434585_760642628.pickle",
        "755434585_760642631.pickle",
        "755434585_760642634.pickle",
        "756029989_760640083.pickle",
        "756029989_760640087.pickle",
        "756029989_760640090.pickle",
        "756029989_760640094.pickle",
        "756029989_760640097.pickle",
        "756029989_760640104.pickle",
        "757216464_769322745.pickle",
        "757216464_769322747.pickle",
        "757216464_769322749.pickle",
        "757216464_769322751.pickle",
        "757216464_769322753.pickle",
        "757216464_769322755.pickle",
        "757970808_769322852.pickle",
        "757970808_769322856.pickle",
        "757970808_769322858.pickle",
        "757970808_769322860.pickle",
        "757970808_769322862.pickle",
        "758798717_770930067.pickle",
        "758798717_770930071.pickle",
        "758798717_770930073.pickle",
        "758798717_770930077.pickle",
        "759883607_769322785.pickle",
        "759883607_769322793.pickle",
        "759883607_769322797.pickle",
        "760345702_810753195.pickle",
        "760345702_810753197.pickle",
        "760345702_810753199.pickle",
        "760345702_810753201.pickle",
        "760345702_810753203.pickle",
        "761418226_768908579.pickle",
        "761418226_768908582.pickle",
        "761418226_768908585.pickle",
        "761418226_768908587.pickle",
        "761418226_768908589.pickle",
        "761418226_768908591.pickle",
        "762602078_768909178.pickle",
        "762602078_768909180.pickle",
        "762602078_768909182.pickle",
        "762602078_768909184.pickle",
        "762602078_768909186.pickle",
        "763673393_773463013.pickle",
        "763673393_773463015.pickle",
        "763673393_773463017.pickle",
        "763673393_773463019.pickle",
        "763673393_773463026.pickle",
        "766640955_773592318.pickle",
        "766640955_773592320.pickle",
        "766640955_773592324.pickle",
        "766640955_773592328.pickle",
        "766640955_773592330.pickle",
        "767871931_773462985.pickle",
        "767871931_773462990.pickle",
        "767871931_773462993.pickle",
        "767871931_773462995.pickle",
        "767871931_773462997.pickle",
        "767871931_773462999.pickle",
        "768515987_773549842.pickle",
        "768515987_773549846.pickle",
        "768515987_773549848.pickle",
        "768515987_773549850.pickle",
        "768515987_773549852.pickle",
        "768515987_773549856.pickle",
        "771160300_773621937.pickle",
        "771160300_773621939.pickle",
        "771160300_773621942.pickle",
        "771160300_773621945.pickle",
        "771160300_773621948.pickle",
        "771160300_773621950.pickle",
        "771990200_773654723.pickle",
        "771990200_773654726.pickle",
        "771990200_773654728.pickle",
        "771990200_773654730.pickle",
        "771990200_773654732.pickle",
        "771990200_773654734.pickle",
        "773418906_792676154.pickle",
        "773418906_792676156.pickle",
        "773418906_792676158.pickle",
        "773418906_792676160.pickle",
        "773418906_792676162.pickle",
        "773418906_792676164.pickle",
        "774875821_792602650.pickle",
        "774875821_792602652.pickle",
        "774875821_792602654.pickle",
        "774875821_792602656.pickle",
        "774875821_792602658.pickle",
        "774875821_792602660.pickle",
        "778240327_792607545.pickle",
        "778240327_792607547.pickle",
        "778240327_792607549.pickle",
        "778240327_792607553.pickle",
        "778240327_792607557.pickle",
        "778240327_792607559.pickle",
        "778998620_792626841.pickle",
        "778998620_792626844.pickle",
        "778998620_792626847.pickle",
        "778998620_792626851.pickle",
        "778998620_792626853.pickle",
        "778998620_792626855.pickle",
        "779839471_792645490.pickle",
        "779839471_792645493.pickle",
        "779839471_792645497.pickle",
        "779839471_792645499.pickle",
        "779839471_792645501.pickle",
        "779839471_792645504.pickle",
        "781842082_792586879.pickle",
        "781842082_792586883.pickle",
        "781842082_792586887.pickle",
        "786091066_792623916.pickle",
        "786091066_792623919.pickle",
        "786091066_792623921.pickle",
        "786091066_792623925.pickle",
        "786091066_792623928.pickle",
        "786091066_792623931.pickle",
        "787025148_792586836.pickle",
        "787025148_792586840.pickle",
        "787025148_792586842.pickle",
        "787025148_792586845.pickle",
        "787025148_792586848.pickle",
        "787025148_792586852.pickle",
        "789848216_805002027.pickle",
        "789848216_805002029.pickle",
        "789848216_805002031.pickle",
        "789848216_805002033.pickle",
        "789848216_805002035.pickle",
        "791319847_805008600.pickle",
        "791319847_805008602.pickle",
        "791319847_805008604.pickle",
        "791319847_805008606.pickle",
        "791319847_805008608.pickle",
        "791319847_805008610.pickle",
        "793224716_805124802.pickle",
        "793224716_805124804.pickle",
        "793224716_805124806.pickle",
        "793224716_805124809.pickle",
        "793224716_805124812.pickle",
        "793224716_805124815.pickle",
        "794812542_810758777.pickle",
        "794812542_810758779.pickle",
        "794812542_810758781.pickle",
        "794812542_810758783.pickle",
        "794812542_810758785.pickle",
        "794812542_810758787.pickle",
        "797828357_805579734.pickle",
        "797828357_805579738.pickle",
        "797828357_805579741.pickle",
        "797828357_805579745.pickle",
        "797828357_805579749.pickle",
        "797828357_805579753.pickle",
        "798911424_800036196.pickle",
        "798911424_800036198.pickle",
        "798911424_800036200.pickle",
        "798911424_800036202.pickle",
        "798911424_800036204.pickle",
        "799864342_805579700.pickle",
        "799864342_805579703.pickle",
        "799864342_805579706.pickle",
        "799864342_805579710.pickle",
        "799864342_805579713.pickle",
        "816200189_836943713.pickle",
        "816200189_836943715.pickle",
        "816200189_836943717.pickle",
        "816200189_836943719.pickle",
        "816200189_836943721.pickle",
        "819186360_820311754.pickle",
        "819186360_820311760.pickle",
        "819186360_820311762.pickle",
        "819186360_820311764.pickle",
        "819701982_836962814.pickle",
        "819701982_836962816.pickle",
        "819701982_836962820.pickle",
        "819701982_836962822.pickle",
        "819701982_836962824.pickle",
        "821695405_822645895.pickle",
        "821695405_822645899.pickle",
        "821695405_822645901.pickle",
        "829720705_832129154.pickle",
        "829720705_832129157.pickle",
        "829720705_832129159.pickle",
        "829720705_832129161.pickle",
        "831882777_832810573.pickle",
        "831882777_832810576.pickle",
        "831882777_832810578.pickle",
        "835479236_837761708.pickle",
        "835479236_837761714.pickle",
        "835479236_837761716.pickle",
        "839068429_841435557.pickle",
        "839068429_841435559.pickle",
        "839068429_868929135.pickle",
        "839068429_868929138.pickle",
        "839068429_868929140.pickle",
        "839068429_868929142.pickle",
        "840012044_841431754.pickle",
        "840012044_841431758.pickle",
        "840012044_868297127.pickle",
        "840012044_868297129.pickle",
        "840012044_868297131.pickle",
        "840012044_868297133.pickle",
        "847657808_848037568.pickle",
        "847657808_848037570.pickle",
        "847657808_848037572.pickle",
        "847657808_848037574.pickle",
        "847657808_848037576.pickle",
        "847657808_848037578.pickle"
    ]
    
    return valid_sessions

def collect_probe_coordinates(pickle_dir, voxel_size=1.0):
    """Collect CCF coordinates from successful pickle files."""
    logger = logging.getLogger(__name__)
    
    # Get list of valid sessions
    valid_sessions = get_valid_sessions()
    logger.info(f"Found {len(valid_sessions)} valid sessions")
    
    # Create mapping from filename to full path
    pickle_pattern = os.path.join(pickle_dir, "*.pickle")
    all_pickle_files = glob.glob(pickle_pattern)
    all_pickle_files = [f for f in all_pickle_files if not f.endswith('.backup')]
    
    file_mapping = {os.path.basename(f): f for f in all_pickle_files}
    
    # Filter to only successful files
    successful_paths = []
    for filename in valid_sessions:
        if filename in file_mapping:
            successful_paths.append(file_mapping[filename])
        else:
            logger.warning(f"Could not find path for valid file: {filename}")
    
    logger.info(f"Processing {len(successful_paths)} successful files")
    
    # Data structures for voxelization
    voxel_counts = defaultdict(int)  # (voxel_x, voxel_y, voxel_z) -> count
    session_probe_coords = defaultdict(list)  # (session_id, probe_id) -> list of coords
    all_coords = []  # All coordinates for bounds calculation
    
    # Process each successful file
    for pickle_path in tqdm(successful_paths, desc="Processing files", unit="file"):
        try:
            with open(pickle_path, 'rb') as f:
                data = pickle.load(f)
            
            filename = os.path.basename(pickle_path)
            logger.info(f"Processing {filename}: {len(data)} entries")
            
            for entry in data:
                if isinstance(entry, (list, tuple)) and len(entry) >= 2:
                    signal, label = entry[0], entry[1]
                    
                    if is_label_enriched(label):
                        # Extract CCF coordinates
                        ccf_coords = extract_ccf_coordinates(label)
                        if ccf_coords:
                            ap, dv, lr = ccf_coords
                            
                            # Convert to voxel coordinates (1mm³ voxels)
                            voxel_x = int(np.round(ap / voxel_size))
                            voxel_y = int(np.round(dv / voxel_size))
                            voxel_z = int(np.round(lr / voxel_size))
                            
                            voxel_key = (voxel_x, voxel_y, voxel_z)
                            voxel_counts[voxel_key] += 1
                            
                            # Store coordinates for session/probe grouping
                            session_probe_info = extract_session_probe_info(label)
                            if session_probe_info:
                                session_id, probe_id = session_probe_info
                                session_probe_coords[(session_id, probe_id)].append(ccf_coords)
                            
                            all_coords.append(ccf_coords)
        
        except Exception as e:
            logger.error(f"Error processing {pickle_path}: {e}")
    
    logger.info(f"Collected {len(all_coords)} total coordinates")
    logger.info(f"Found {len(voxel_counts)} unique voxels")
    logger.info(f"Found {len(session_probe_coords)} unique session-probe combinations")
    
    return voxel_counts, session_probe_coords, all_coords

def create_3d_voxel_visualization(voxel_counts, output_dir, voxel_size=1.0):
    """Create 3D voxel visualization of probe structure."""
    logger = logging.getLogger(__name__)
    
    if not voxel_counts:
        logger.error("No voxel data to visualize")
        return
    
    # Convert voxel counts to arrays for visualization
    voxel_coords = list(voxel_counts.keys())
    voxel_values = list(voxel_counts.values())
    
    # Separate coordinates
    x_coords = [coord[0] for coord in voxel_coords]
    y_coords = [coord[1] for coord in voxel_coords]
    z_coords = [coord[2] for coord in voxel_coords]
    
    logger.info(f"Visualizing {len(voxel_coords)} voxels")
    logger.info(f"X range: {min(x_coords)} to {max(x_coords)}")
    logger.info(f"Y range: {min(y_coords)} to {max(y_coords)}")
    logger.info(f"Z range: {min(z_coords)} to {max(z_coords)}")
    
    # Create 3D scatter plot
    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map based on voxel density (number of channels)
    max_count = max(voxel_values)
    min_count = min(voxel_values)
    
    # Normalize counts for color mapping
    normalized_counts = [(count - min_count) / (max_count - min_count) for count in voxel_values]
    
    # Create scatter plot with color-coded voxels
    scatter = ax.scatter(x_coords, y_coords, z_coords, 
                        c=normalized_counts, 
                        cmap='viridis',
                        s=20,  # Size of voxels
                        alpha=0.7)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Channel Density (Normalized)', rotation=270, labelpad=20)
    
    # Set labels and title
    ax.set_xlabel('Anterior-Posterior (CCF coordinates)')
    ax.set_ylabel('Dorsal-Ventral (CCF coordinates)')
    ax.set_zlabel('Left-Right (CCF coordinates)')
    ax.set_title('3D Probe Structure Visualization\n(1mm³ Voxels, Color = Channel Density)')
    
    # Set equal aspect ratio
    max_range = max(max(x_coords) - min(x_coords), 
                   max(y_coords) - min(y_coords), 
                   max(z_coords) - min(z_coords)) / 2
    
    mid_x = (max(x_coords) + min(x_coords)) / 2
    mid_y = (max(y_coords) + min(y_coords)) / 2
    mid_z = (max(z_coords) + min(z_coords)) / 2
    
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)
    
    # Save the plot
    output_file = os.path.join(output_dir, f"probe_structure_3d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"3D visualization saved to: {output_file}")
    
    # Also create a side view (2D projection)
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    
    # XY projection (coronal view)
    scatter1 = ax1.scatter(x_coords, y_coords, c=normalized_counts, cmap='viridis', s=10, alpha=0.7)
    ax1.set_xlabel('Anterior-Posterior')
    ax1.set_ylabel('Dorsal-Ventral')
    ax1.set_title('Coronal View (XY projection)')
    ax1.set_aspect('equal')
    
    # XZ projection (horizontal view)
    scatter2 = ax2.scatter(x_coords, z_coords, c=normalized_counts, cmap='viridis', s=10, alpha=0.7)
    ax2.set_xlabel('Anterior-Posterior')
    ax2.set_ylabel('Left-Right')
    ax2.set_title('Horizontal View (XZ projection)')
    ax2.set_aspect('equal')
    
    # YZ projection (sagittal view)
    scatter3 = ax3.scatter(y_coords, z_coords, c=normalized_counts, cmap='viridis', s=10, alpha=0.7)
    ax3.set_xlabel('Dorsal-Ventral')
    ax3.set_ylabel('Left-Right')
    ax3.set_title('Sagittal View (YZ projection)')
    ax3.set_aspect('equal')
    
    # Add colorbar
    plt.colorbar(scatter1, ax=ax1, shrink=0.8)
    plt.colorbar(scatter2, ax=ax2, shrink=0.8)
    plt.colorbar(scatter3, ax=ax3, shrink=0.8)
    
    # Save 2D projections
    output_file_2d = os.path.join(output_dir, f"probe_structure_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.tight_layout()
    plt.savefig(output_file_2d, dpi=300, bbox_inches='tight')
    logger.info(f"2D projections saved to: {output_file_2d}")
    
    plt.close('all')

def create_session_probe_visualization(session_probe_coords, output_dir):
    """Create separate visualizations for each session-probe combination."""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating individual visualizations for {len(session_probe_coords)} session-probe combinations")
    
    # Create a grid of subplots
    n_combinations = len(session_probe_coords)
    n_cols = min(4, n_combinations)
    n_rows = (n_combinations + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_rows == 1:
        axes = [axes] if n_cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for idx, ((session_id, probe_id), coords) in enumerate(session_probe_coords.items()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Extract coordinates
        x_coords = [coord[0] for coord in coords]
        y_coords = [coord[1] for coord in coords]
        z_coords = [coord[2] for coord in coords]
        
        # Create 3D scatter plot
        ax = fig.add_subplot(n_rows, n_cols, idx+1, projection='3d')
        ax.scatter(x_coords, y_coords, z_coords, s=20, alpha=0.7)
        
        ax.set_xlabel('AP')
        ax.set_ylabel('DV')
        ax.set_zlabel('LR')
        ax.set_title(f'Session {session_id}\nProbe {probe_id}\n({len(coords)} channels)')
    
    # Hide unused subplots
    for idx in range(len(session_probe_coords), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    output_file = os.path.join(output_dir, f"individual_probes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    logger.info(f"Individual probe visualizations saved to: {output_file}")
    plt.close()

def save_voxel_data(voxel_counts, session_probe_coords, output_dir):
    """Save voxel data for further analysis."""
    logger = logging.getLogger(__name__)
    
    # Save voxel counts
    voxel_data = {
        'voxel_size_mm': 1.0,
        'total_voxels': len(voxel_counts),
        'voxel_counts': dict(voxel_counts),
        'session_probe_coords': {f"{session}_{probe}": coords for (session, probe), coords in session_probe_coords.items()},
        'summary': {
            'total_coordinates': sum(voxel_counts.values()),
            'unique_sessions': len(set(session for session, _ in session_probe_coords.keys())),
            'unique_probes': len(set(probe for _, probe in session_probe_coords.keys())),
            'unique_session_probe_combinations': len(session_probe_coords)
        }
    }
    
    output_file = os.path.join(output_dir, f"voxel_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_file, 'w') as f:
        json.dump(voxel_data, f, indent=2)
    
    logger.info(f"Voxel data saved to: {output_file}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("VOXELIZATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total voxels: {voxel_data['summary']['total_coordinates']:,}")
    logger.info(f"Unique voxels: {voxel_data['summary']['total_coordinates']:,}")
    logger.info(f"Unique sessions: {voxel_data['summary']['unique_sessions']}")
    logger.info(f"Unique probes: {voxel_data['summary']['unique_probes']}")
    logger.info(f"Session-probe combinations: {voxel_data['summary']['unique_session_probe_combinations']}")

def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="3D Probe Structure Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Usage Examples:
  # Create visualization with default settings
  python visualize_probe_structure_3d.py /scratch/us2193/LFP/enriched_pickles_Allen/ /scratch/us2193/LFP/probe_visualizations/
  
  # Create visualization with custom voxel size
  python visualize_probe_structure_3d.py /scratch/us2193/LFP/enriched_pickles_Allen/ /scratch/us2193/LFP/probe_visualizations/ --voxel-size 0.5
  
  # Create visualization with custom input path
  python visualize_probe_structure_3d.py /path/to/your/pickle/files/ /path/to/output/ --voxel-size 1.0
        """
    )
    
    parser.add_argument(
        'pickle_dir',
        help='Directory containing pickle files'
    )
    
    parser.add_argument(
        'output_dir',
        help='Directory to save visualizations'
    )
    
    parser.add_argument(
        '--voxel-size',
        type=float,
        default=1.0,
        help='Voxel size in mm (default: 1.0)'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging()
    
    logger = logging.getLogger(__name__)
    logger.info("3D Probe Structure Visualization Tool")
    logger.info("=" * 50)
    logger.info(f"Pickle directory: {args.pickle_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Voxel size: {args.voxel_size}mm")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Check if pickle directory exists
        if not os.path.exists(args.pickle_dir):
            logger.error(f"Pickle directory does not exist: {args.pickle_dir}")
            return
        
        # Collect probe coordinates
        voxel_counts, session_probe_coords, all_coords = collect_probe_coordinates(
            args.pickle_dir, args.voxel_size
        )
        
        if not voxel_counts:
            logger.error("No coordinates collected")
            return
        
        # Create visualizations
        logger.info("Creating 3D voxel visualization...")
        create_3d_voxel_visualization(voxel_counts, args.output_dir, args.voxel_size)
        
        logger.info("Creating individual probe visualizations...")
        create_session_probe_visualization(session_probe_coords, args.output_dir)
        
        # Save data
        logger.info("Saving voxel data...")
        save_voxel_data(voxel_counts, session_probe_coords, args.output_dir)
        
        logger.info("Visualization completed successfully!")
        
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
