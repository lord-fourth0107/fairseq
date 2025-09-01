#!/bin/bash
#SBATCH --job-name=test_setup
#SBATCH --output=test_setup_%j.out
#SBATCH --error=test_setup_%j.err
#SBATCH --account=pr_126_tandon_priority
#SBATCH --time=00:15:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --nodes=1
#SBATCH --ntasks=1

echo "🧪 Testing Wav2Vec2 2D Setup"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"

# Activate virtual environment
source /vast/us2193/wav2vec2_env_py39/bin/activate

# Navigate to directory
cd /vast/us2193/fairseq

# Run test
python test_setup.py

echo "End time: $(date)"
