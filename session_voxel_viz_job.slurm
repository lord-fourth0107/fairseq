#!/bin/bash
#SBATCH --job-name=session_voxel_viz
#SBATCH --account=pr_126_tandon_advanced
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=4:00:00
#SBATCH --gres=none
#SBATCH --mail-type=END
#SBATCH --mail-user=mkp6112@nyu.edu
#SBATCH --output=session_voxel_viz_%j.out
#SBATCH --error=session_voxel_viz_%j.err

# Paths
REPO_DIR=/vast/us2193/fairseq
INPUT_DIR=/scratch/us2193/LFP/Allen
VENV_DIR=/vast/us2193/fairseq/wav2vec2_env_py39
OUTPUT_DIR=/scratch/us2193/LFP/session_voxel_viz

# Optional: specific session ID (leave empty to use first session found)
SESSION_ID=""  # e.g., "715093703"

# Avoid thread oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "SLURM_CPUS_ON_NODE=${SLURM_CPUS_ON_NODE}"
echo "Repo:  ${REPO_DIR}"
echo "Input: ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"
echo "Session ID: ${SESSION_ID:-'auto-select first'}"

# Activate venv
source "${VENV_DIR}/bin/activate"
python -V

# Create output directory
mkdir -p "${OUTPUT_DIR}"

cd "${REPO_DIR}"

# Run visualization
if [ -n "${SESSION_ID}" ]; then
    python -u session_probe_voxel_viz.py "${INPUT_DIR}" \
        --session-id "${SESSION_ID}" \
        --workers "${SLURM_CPUS_ON_NODE}" \
        --voxel-size 1000.0 \
        --output-dir "${OUTPUT_DIR}"
else
    python -u session_probe_voxel_viz.py "${INPUT_DIR}" \
        --workers "${SLURM_CPUS_ON_NODE}" \
        --voxel-size 1000.0 \
        --output-dir "${OUTPUT_DIR}"
fi

echo "Job Over"
