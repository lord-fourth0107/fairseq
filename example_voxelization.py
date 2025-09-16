#!/usr/bin/env python3
"""
Example script showing how to use the voxelization and visualization tools.
"""

import os
import subprocess
import sys

def run_voxelization_example():
    """Run the voxelization example with Downloads folder."""
    
    print("Neural Activity Voxelization Example")
    print("=" * 50)
    
    # Check if Downloads folder exists and has pickle files
    downloads_path = os.path.expanduser("~/Downloads")
    if not os.path.exists(downloads_path):
        print(f"Downloads folder not found: {downloads_path}")
        return
    
    # Look for pickle files
    import glob
    pickle_files = glob.glob(os.path.join(downloads_path, "*.pickle"))
    
    if not pickle_files:
        print(f"No pickle files found in {downloads_path}")
        print("Please ensure you have enriched pickle files in the Downloads folder")
        return
    
    print(f"Found {len(pickle_files)} pickle files in Downloads folder")
    
    # Run voxelization with different parameters
    print("\n1. Basic voxelization (100μm voxels):")
    cmd1 = [sys.executable, "voxelize_and_visualize.py", downloads_path]
    subprocess.run(cmd1)
    
    print("\n2. High-resolution voxelization (50μm voxels):")
    cmd2 = [sys.executable, "voxelize_and_visualize.py", downloads_path, "--voxel-size", "50"]
    subprocess.run(cmd2)
    
    print("\n3. Fast processing (sample 1000 entries per file):")
    cmd3 = [sys.executable, "voxelize_and_visualize.py", downloads_path, "--sample-size", "1000"]
    subprocess.run(cmd3)
    
    print("\n4. Save visualization to file:")
    cmd4 = [sys.executable, "voxelize_and_visualize.py", downloads_path, "--output", "brain_visualization.png"]
    subprocess.run(cmd4)

if __name__ == "__main__":
    run_voxelization_example()
