#!/bin/bash

# Multi-GPU training launcher for wav2vec2_2d with Scaled RoPE
# Usage: ./launch_multi_gpu.sh [num_gpus] [data_path] [output_path] [epochs]

# Default values
NUM_GPUS=${1:-2}  # Default to 2 GPUs
DATA_PATH=${2:-"/Users/uttamsingh/Downloads"}
OUTPUT_PATH=${3:-"./outputs"}
EPOCHS=${4:-10}

echo "🚀 Launching Multi-GPU wav2vec2_2d training with Scaled RoPE"
echo "📊 GPUs: $NUM_GPUS"
echo "📁 Data path: $DATA_PATH"
echo "📁 Output path: $OUTPUT_PATH"
echo "📅 Epochs: $EPOCHS"
echo ""

# Check if CUDA is available
if ! python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')"; then
    echo "❌ CUDA not available. Exiting."
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_PATH"

# Run multi-GPU training
python wav2vec2_2d_multi_gpu.py \
    --data_path "$DATA_PATH" \
    --output_path "$OUTPUT_PATH" \
    --num_epochs "$EPOCHS" \
    --world_size "$NUM_GPUS"

echo "✅ Multi-GPU training completed!"
