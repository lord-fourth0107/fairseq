#!/usr/bin/env python3
"""
Example usage script for Multi-Channel Voxel Cube Visualization

This script demonstrates how to use the multi_channel_voxel_cube_viz.py script
to create 3D visualizations of brain voxels with hue variation based on
the number of channels per voxel.
"""

import os
import subprocess
import sys
from pathlib import Path

def run_multi_channel_viz(pickle_dir, output_dir="multi_channel_voxel_viz", voxel_size=1.0):
    """Run the multi-channel voxel cube visualization."""
    
    # Check if the visualization script exists
    viz_script = Path(__file__).parent / "multi_channel_voxel_cube_viz.py"
    if not viz_script.exists():
        print(f"Error: Visualization script not found at {viz_script}")
        return False
    
    # Check if pickle directory exists
    if not os.path.exists(pickle_dir):
        print(f"Error: Pickle directory not found: {pickle_dir}")
        return False
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable,
        str(viz_script),
        "--pickle_dir", pickle_dir,
        "--output_dir", output_dir,
        "--voxel_size", str(voxel_size)
    ]
    
    print("=" * 60)
    print("RUNNING MULTI-CHANNEL VOXEL CUBE VISUALIZATION")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    try:
        # Run the visualization
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        print("=" * 60)
        print("VISUALIZATION COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        
        # List output files
        output_files = list(Path(output_dir).glob("*"))
        if output_files:
            print("Generated files:")
            for file in sorted(output_files):
                print(f"  - {file.name}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Error running visualization: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    """Main function with example usage."""
    
    # Example usage - you can modify these paths as needed
    examples = [
        {
            "name": "Example 1: Default 1mm³ voxels",
            "pickle_dir": "/path/to/your/pickle/files",  # Replace with actual path
            "output_dir": "multi_channel_viz_1mm",
            "voxel_size": 1.0
        },
        {
            "name": "Example 2: High resolution 0.5mm³ voxels",
            "pickle_dir": "/path/to/your/pickle/files",  # Replace with actual path
            "output_dir": "multi_channel_viz_0.5mm",
            "voxel_size": 0.5
        },
        {
            "name": "Example 3: Low resolution 2mm³ voxels",
            "pickle_dir": "/path/to/your/pickle/files",  # Replace with actual path
            "output_dir": "multi_channel_viz_2mm",
            "voxel_size": 2.0
        }
    ]
    
    print("Multi-Channel Voxel Cube Visualization Examples")
    print("=" * 60)
    print()
    print("This script demonstrates how to create 3D visualizations where:")
    print("1. The brain is divided into voxels (default: 1mm³)")
    print("2. Each coordinate paints an entire cube")
    print("3. Multiple channels selecting the same voxel repaint it with different hues")
    print("4. Hue varies from red (1 channel) to violet (many channels)")
    print()
    
    # Check if we have a pickle directory argument
    if len(sys.argv) > 1:
        pickle_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else "multi_channel_voxel_viz"
        voxel_size = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        
        print(f"Running with provided arguments:")
        print(f"  Pickle directory: {pickle_dir}")
        print(f"  Output directory: {output_dir}")
        print(f"  Voxel size: {voxel_size} mm")
        print()
        
        success = run_multi_channel_viz(pickle_dir, output_dir, voxel_size)
        if success:
            print("Visualization completed successfully!")
        else:
            print("Visualization failed!")
            sys.exit(1)
    else:
        print("Usage examples:")
        print()
        for i, example in enumerate(examples, 1):
            print(f"{i}. {example['name']}")
            print(f"   Command: python {Path(__file__).name} {example['pickle_dir']} {example['output_dir']} {example['voxel_size']}")
            print()
        
        print("To run with your own data:")
        print(f"  python {Path(__file__).name} <pickle_directory> [output_directory] [voxel_size_mm]")
        print()
        print("Examples:")
        print(f"  # Default 1mm³ voxels")
        print(f"  python {Path(__file__).name} /path/to/pickle/files")
        print(f"  # High resolution 0.5mm³ voxels")
        print(f"  python {Path(__file__).name} /path/to/pickle/files my_viz_output 0.5")
        print(f"  # Low resolution 2mm³ voxels")
        print(f"  python {Path(__file__).name} /path/to/pickle/files my_viz_output 2.0")

if __name__ == "__main__":
    main()
