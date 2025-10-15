#!/bin/bash
#SBATCH --job-name=pickle_enrichment_serial
#SBATCH --account=pr_126_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=none
#SBATCH --mail-type=END
#SBATCH --mail-user=us2193@nyu.edu
#SBATCH --output=enrichment_serial_%j.out
#SBATCH --error=enrichment_serial_%j.err

# Paths - UPDATE THESE PATHS
REPO_DIR=/vast/us2193/fairseq
INPUT_DIR=/scratch/us2193/LFP/Allen  # Directory containing pickle files and CSV files
OUTPUT_DIR=/scratch/us2193/LFP/enriched_pickles  # Directory for enriched pickle files
VENV_DIR=/vast/us2193/fairseq/wav2vec2_env_py39

# Configuration
BATCH_SIZE=10000
MAX_MEMORY=16  # GB

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE}"
echo "Repo:  ${REPO_DIR}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Batch size: ${BATCH_SIZE}"
echo "Max memory: ${MAX_MEMORY}GB"

# Activate venv
source "${VENV_DIR}/bin/activate"
python -V

# Create output directory
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

# Validate input directory and required files
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory $INPUT_DIR does not exist!"
    exit 1
fi

if [ ! -f "$INPUT_DIR/joined.csv" ]; then
    echo "ERROR: joined.csv not found in $INPUT_DIR"
    exit 1
fi

if [ ! -f "$INPUT_DIR/channels.csv" ]; then
    echo "ERROR: channels.csv not found in $INPUT_DIR"
    exit 1
fi

# Count pickle files
PICKLE_COUNT=$(find "$INPUT_DIR" -name "*.pickle" -not -name "*.backup" | wc -l)
echo "Found $PICKLE_COUNT pickle files to process"

if [ $PICKLE_COUNT -eq 0 ]; then
    echo "ERROR: No pickle files found in $INPUT_DIR"
    exit 1
fi

# Run serial enrichment
echo "Starting serial enrichment process..."
python -u enrich_pickle_serial.py "${INPUT_DIR}" "${OUTPUT_DIR}" \
    --batch-size $BATCH_SIZE \
    --max-memory $MAX_MEMORY \
    --log-level INFO

echo "Job Over"