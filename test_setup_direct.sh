#!/bin/bash
# Direct srun command for testing
# Usage: ./test_setup_direct.sh

echo "🧪 Testing Wav2Vec2 2D Setup with Direct srun"
echo "Start time: $(date)"

# Activate virtual environment
source /vast/us2193/wav2vec2_env_py39/bin/activate

# Navigate to directory
cd /vast/us2193/fairseq

# Run test with srun
echo "🏃 Running test with srun..."
srun --account=pr_126_tandon_priority \
     --gres=gpu:1 \
     --cpus-per-task=8 \
     --mem=16GB \
     --time=00:10:00 \
     --pty \
     python test_setup.py

echo "End time: $(date)"
