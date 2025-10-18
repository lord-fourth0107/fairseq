#!/bin/bash
#SBATCH --job-name=probe_viz
#SBATCH --account=pr_126_tandon_priority
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --gres=none
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=us2193@nyu.edu
#SBATCH --output=probe_viz_%j.out
#SBATCH --error=probe_viz_%j.err

# Paths - UPDATE THESE PATHS
REPO_DIR=/vast/us2193/fairseq
# Directory containing pickle files
INPUT_DIR=/scratch/us2193/LFP/enriched_pickles_Allen
# Directory for visualization outputs
OUTPUT_DIR=/scratch/us2193/LFP/probe_visualizations
VENV_DIR=/vast/us2193/fairseq/wav2vec2_env_py39

# Configuration
VOXEL_SIZE=1.0  # mm

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE}"
echo "Repo:  ${REPO_DIR}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Voxel size: ${VOXEL_SIZE}mm"

# Activate venv
source "${VENV_DIR}/bin/activate"
python -V

# Create output directory
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory $INPUT_DIR does not exist!"
    exit 1
fi

# Count pickle files
PICKLE_COUNT=$(find "$INPUT_DIR" -name "*.pickle" -not -name "*.backup" | wc -l)
echo "Found $PICKLE_COUNT pickle files to process"

if [ $PICKLE_COUNT -eq 0 ]; then
    echo "ERROR: No pickle files found in $INPUT_DIR"
    exit 1
fi

# Run 3D probe visualization
echo "Starting 3D probe structure visualization..."
python -u visualize_probe_structure_3d.py "${INPUT_DIR}" "${OUTPUT_DIR}" --voxel-size ${VOXEL_SIZE}

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "3D probe visualization completed successfully!"
    echo "End time: $(date)"
    echo "=========================================="
    
    # Show output files
    echo "Generated files:"
    ls -la "${OUTPUT_DIR}/"
    
else
    echo "=========================================="
    echo "3D probe visualization failed with errors!"
    echo "End time: $(date)"
    echo "Check the log files for details"
    echo "=========================================="
    exit 1
fi

echo "Job Over"
