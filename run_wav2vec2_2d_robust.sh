#!/bin/bash
#SBATCH --job-name=wav2vec2_2d_training
#SBATCH --output=wav2vec2_2d_%j.out
#SBATCH --error=wav2vec2_2d_%j.err
#SBATCH --account=pr_126_tandon_priority
#SBATCH --time=03:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB
#SBATCH --nodes=1
#SBATCH --ntasks=1

# Set up environment
echo "🚀 Starting Wav2Vec2 2D Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
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

# Check GPU availability
echo "🔍 Checking GPU availability..."
nvidia-smi
if [ $? -eq 0 ]; then
    echo "✅ GPU is available"
else
    echo "❌ GPU not available"
    exit 1
fi

# Check Python and PyTorch
echo "🐍 Checking Python environment..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Set environment variables for better performance
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Navigate to the correct directory
cd /vast/us2193/fairseq

# Check if the training script exists
if [ -f "wav2vec2_2d_single_gpu.py" ]; then
    echo "✅ Training script found"
else
    echo "❌ Training script not found"
    exit 1
fi

# Run the training with timeout and error handling
echo "🏃 Starting training..."
timeout 23h python wav2vec2_2d_single_gpu.py

# Check exit status
EXIT_CODE=$?
echo "Training finished with exit code: $EXIT_CODE"

if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
elif [ $EXIT_CODE -eq 124 ]; then
    echo "⏰ Training timed out after 23 hours"
elif [ $EXIT_CODE -eq 130 ]; then
    echo "🛑 Training was interrupted"
else
    echo "❌ Training failed with error code: $EXIT_CODE"
fi

echo "End time: $(date)"
echo "Job completed"
