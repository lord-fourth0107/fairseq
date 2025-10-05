#!/bin/bash
# HPC Environment Setup Script for Pickle Enrichment
# This script helps configure the environment for running enrichment on HPC

echo "=========================================="
echo "HPC Pickle Enrichment Setup"
echo "=========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check Python installation
echo "Checking Python installation..."
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1)
    echo "✓ Python found: $PYTHON_VERSION"
else
    echo "✗ Python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check required Python packages
echo "Checking Python packages..."
REQUIRED_PACKAGES=("pandas" "numpy" "tqdm" "matplotlib")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python3 -c "import $package" 2>/dev/null; then
        echo "✓ $package is installed"
    else
        echo "✗ $package is not installed"
        echo "  Install with: pip install $package"
    fi
done

# Check SLURM availability
echo "Checking SLURM availability..."
if command_exists sbatch; then
    echo "✓ SLURM is available"
    echo "  SLURM version: $(sbatch --version 2>&1 | head -n1)"
else
    echo "✗ SLURM not found. This script is designed for SLURM-based HPC systems"
    echo "  You may need to adapt the job script for your cluster's job scheduler"
fi

# Check available resources
echo "Checking available resources..."
if command_exists sinfo; then
    echo "Available partitions:"
    sinfo -o "%P %A %N %T" | head -10
else
    echo "sinfo command not available"
fi

# Create example configuration
echo "Creating example configuration..."
cat > hpc_config_example.sh << 'EOF'
#!/bin/bash
# Example HPC Configuration
# Copy this file to hpc_config.sh and modify the values

# Input directory containing pickle files and CSV files
INPUT_DIR="/path/to/your/pickle/files"

# SLURM job parameters
JOB_NAME="pickle_enrichment"
TIME_LIMIT="24:00:00"
PARTITION="general"  # Change to your cluster's partition
NODES=1
NTASKS=1
CPUS_PER_TASK=32  # Adjust based on your needs
MEMORY="64G"      # Adjust based on your needs

# Processing parameters
WORKERS=32        # Should match CPUS_PER_TASK
BATCH_SIZE=5000   # Adjust based on memory
MAX_MEMORY=16     # GB, should be less than SLURM memory

# Email notifications
EMAIL="your_email@domain.com"

# Output directory (will be created automatically)
OUTPUT_DIR="enrichment_results_$(date +%Y%m%d_%H%M%S)"
EOF

echo "✓ Created example configuration: hpc_config_example.sh"

# Create a simple test script
echo "Creating test script..."
cat > test_enrichment.py << 'EOF'
#!/usr/bin/env python3
"""
Simple test script to verify the enrichment setup
"""

import os
import sys
import pickle
import pandas as pd
import numpy as np

def create_test_data():
    """Create test pickle and CSV files for validation."""
    print("Creating test data...")
    
    # Create test CSV files
    test_joined_data = {
        'session_id': ['715093703'] * 10,
        'probe_id': ['810755797'] * 10,
        'anterior_posterior_ccf_coordinate_x': np.random.uniform(5000, 6000, 10),
        'dorsal_ventral_ccf_coordinate_x': np.random.uniform(3000, 4000, 10),
        'left_right_ccf_coordinate_x': np.random.uniform(2000, 3000, 10)
    }
    
    test_channels_data = {
        'id': [f'channel_{i}' for i in range(10)],
        'ecephys_probe_id': ['810755797'] * 10,
        'probe_horizontal_position': np.random.uniform(0, 100, 10),
        'probe_vertical_position': np.random.uniform(0, 100, 10)
    }
    
    joined_df = pd.DataFrame(test_joined_data)
    channels_df = pd.DataFrame(test_channels_data)
    
    joined_df.to_csv('test_joined.csv', index=False)
    channels_df.to_csv('test_channels.csv', index=False)
    
    # Create test pickle file
    test_pickle_data = []
    for i in range(10):
        signal = np.random.randn(1000)  # Random signal
        label = f"715093703_{i}_810755797_channel_{i}_CA1"
        test_pickle_data.append((signal, label))
    
    with open('test_data.pickle', 'wb') as f:
        pickle.dump(test_pickle_data, f)
    
    print("✓ Test data created:")
    print("  - test_joined.csv")
    print("  - test_channels.csv") 
    print("  - test_data.pickle")

def test_enrichment():
    """Test the enrichment process."""
    print("Testing enrichment process...")
    
    try:
        # Import the enrichment script
        from enrich_pickle_hpc import load_coordinate_data, enrich_pickle_file
        
        # Load coordinate data
        coord_lookup = load_coordinate_data('.')
        if coord_lookup is None:
            print("✗ Failed to load coordinate data")
            return False
        
        print(f"✓ Loaded coordinate lookup with {len(coord_lookup)} entries")
        
        # Test enrichment
        stats = enrich_pickle_file('test_data.pickle', coord_lookup, batch_size=5, num_workers=1)
        if stats is None:
            print("✗ Enrichment failed")
            return False
        
        print(f"✓ Enrichment successful:")
        print(f"  Total entries: {stats['total']}")
        print(f"  Enriched: {stats['enriched']}")
        print(f"  Success rate: {stats['enriched']/stats['total']*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        return False

def cleanup():
    """Clean up test files."""
    test_files = ['test_joined.csv', 'test_channels.csv', 'test_data.pickle']
    for file in test_files:
        if os.path.exists(file):
            os.remove(file)
    print("✓ Cleaned up test files")

if __name__ == "__main__":
    print("Testing HPC Enrichment Setup")
    print("=" * 30)
    
    create_test_data()
    
    if test_enrichment():
        print("\n✓ All tests passed! Setup is ready for HPC.")
    else:
        print("\n✗ Tests failed. Please check your setup.")
        sys.exit(1)
    
    cleanup()
EOF

echo "✓ Created test script: test_enrichment.py"

# Create a quick start guide
echo "Creating quick start guide..."
cat > HPC_QUICK_START.md << 'EOF'
# HPC Pickle Enrichment Quick Start Guide

## Prerequisites
1. Python 3.7+ with required packages (pandas, numpy, tqdm, matplotlib)
2. SLURM-based HPC cluster access
3. Pickle files and CSV files (joined.csv, channels.csv) in the same directory

## Setup Steps

### 1. Test Your Environment
```bash
python3 test_enrichment.py
```

### 2. Configure HPC Job
```bash
cp hpc_config_example.sh hpc_config.sh
# Edit hpc_config.sh with your specific settings
```

### 3. Update SLURM Script
Edit `run_enrichment_hpc.slurm`:
- Change `INPUT_DIR` to your pickle files directory
- Adjust SLURM parameters (partition, memory, CPUs) for your cluster
- Update email address for notifications

### 4. Submit Job
```bash
sbatch run_enrichment_hpc.slurm
```

### 5. Monitor Job
```bash
squeue -u $USER
```

## File Structure
```
your_pickle_directory/
├── *.pickle                    # Your pickle files
├── joined.csv                  # Required coordinate data
├── channels.csv                # Required channel data
└── enrichment_results_*/       # Output directory (created automatically)
    ├── enrichment.log          # Main process log
    ├── enrichment_summary_*.json # Detailed statistics
    └── job_summary.txt         # Job summary
```

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce `BATCH_SIZE` or increase `MEMORY` in SLURM script
2. **Timeout errors**: Increase `TIME_LIMIT` in SLURM script
3. **Missing CSV files**: Ensure joined.csv and channels.csv are in the input directory
4. **Permission errors**: Check file permissions and directory access

### Performance Tuning
- **Workers**: Set to match `CPUS_PER_TASK` in SLURM script
- **Batch size**: Larger for more memory, smaller for less memory
- **Memory limit**: Should be less than SLURM memory allocation

## Support
Check the log files in the output directory for detailed error information.
EOF

echo "✓ Created quick start guide: HPC_QUICK_START.md"

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Run: python3 test_enrichment.py"
echo "2. Edit: hpc_config_example.sh → hpc_config.sh"
echo "3. Update: run_enrichment_hpc.slurm with your paths"
echo "4. Submit: sbatch run_enrichment_hpc.slurm"
echo ""
echo "See HPC_QUICK_START.md for detailed instructions"
echo "=========================================="
