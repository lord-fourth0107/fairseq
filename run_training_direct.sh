#!/bin/bash
# Direct srun command matching your working configuration
# Usage: ./run_training_direct.sh

echo "🚀 Starting Wav2Vec2 2D Training with Direct srun"
echo "Start time: $(date)"

# Activate virtual environment
echo "📦 Activating virtual environment..."
source /vast/us2193/wav2vec2_env_py39/bin/activate

# Check if activation was successful
if [ $? -eq 0 ]; then
    echo "✅ Virtual environment activated successfully"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Navigate to the correct directory
cd /vast/us2193/fairseq

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=32
export MKL_NUM_THREADS=32

# Run the training with your exact srun configuration
echo "🏃 Starting training with srun..."
srun --account=pr_126_tandon_priority \
     --gres=gpu:1 \
     --cpus-per-task=32 \
     --mem=50GB \
     --time=03:30:00 \
     --pty \
     python wav2vec2_2d_single_gpu.py

# Check exit status
EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
elif [ $EXIT_CODE -eq 130 ]; then
    echo "🛑 Training was interrupted"
else
    echo "❌ Training failed with error code: $EXIT_CODE"
fi

echo "End time: $(date)"
echo "Job completed"
