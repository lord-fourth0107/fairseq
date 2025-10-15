#!/bin/bash
#SBATCH --job-name=pickle_enrichment
#SBATCH --output=enrichment_%j.out
#SBATCH --error=enrichment_%j.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=uttamsingh@example.com

# HPC Pickle File Enrichment Job Script
# This script runs the HPC-optimized pickle enrichment process
# Adjust the SLURM parameters above based on your HPC cluster's configuration

echo "=========================================="
echo "HPC Pickle Enrichment Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Start time: $(date)"
echo "=========================================="

# Load required modules (adjust based on your HPC cluster)
# Uncomment and modify these lines based on your cluster's module system
module load python/3.9
module load cuda/11.8
module load gcc/9.3.0
module load openmpi/4.1.0

# Set up environment
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export OMP_NUM_THREADS=1  # Prevent OpenMP conflicts with multiprocessing

# Configuration variables
INPUT_DIR="/path/to/your/pickle/files"  # CHANGE THIS PATH TO YOUR ACTUAL DATA DIRECTORY
WORKERS=64  # Adjust based on SLURM_CPUS_PER_TASK
BATCH_SIZE=10000  # Adjust based on memory requirements
MAX_MEMORY=32  # GB, adjust based on SLURM_MEM_PER_NODE

# Validate input directory
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: Input directory $INPUT_DIR does not exist!"
    echo "Please update the INPUT_DIR variable in this script"
    exit 1
fi

# Check for required CSV files
if [ ! -f "$INPUT_DIR/joined.csv" ]; then
    echo "ERROR: joined.csv not found in $INPUT_DIR"
    exit 1
fi

if [ ! -f "$INPUT_DIR/channels.csv" ]; then
    echo "ERROR: channels.csv not found in $INPUT_DIR"
    exit 1
fi

echo "Input directory: $INPUT_DIR"
echo "Workers: $WORKERS"
echo "Batch size: $BATCH_SIZE"
echo "Max memory: ${MAX_MEMORY}GB"

# Count pickle files
PICKLE_COUNT=$(find "$INPUT_DIR" -name "*.pickle" | wc -l)
echo "Found $PICKLE_COUNT pickle files to process"

if [ $PICKLE_COUNT -eq 0 ]; then
    echo "ERROR: No pickle files found in $INPUT_DIR"
    exit 1
fi

# Create output directory for logs and results
OUTPUT_DIR="enrichment_results_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUTPUT_DIR"
echo "Output directory: $OUTPUT_DIR"

# Run the enrichment script
echo "Starting enrichment process..."
python enrich_pickle_hpc.py "$INPUT_DIR" \
    --workers $WORKERS \
    --batch-size $BATCH_SIZE \
    --max-memory $MAX_MEMORY \
    --log-file "$OUTPUT_DIR/enrichment.log" \
    --log-level INFO

# Check exit status
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "Enrichment completed successfully!"
    echo "End time: $(date)"
    echo "=========================================="
    
    # Move log files to output directory
    mv enrichment_${SLURM_JOB_ID}.out "$OUTPUT_DIR/"
    mv enrichment_${SLURM_JOB_ID}.err "$OUTPUT_DIR/"
    
    # Create summary
    echo "Creating job summary..."
    cat > "$OUTPUT_DIR/job_summary.txt" << EOF
HPC Pickle Enrichment Job Summary
================================
Job ID: $SLURM_JOB_ID
Node: $SLURM_NODELIST
CPUs Used: $SLURM_CPUS_PER_TASK
Memory Allocated: $SLURM_MEM_PER_NODE MB
Start Time: $(date -d @$SLURM_JOB_START_TIME 2>/dev/null || echo "Unknown")
End Time: $(date)
Input Directory: $INPUT_DIR
Pickle Files Processed: $PICKLE_COUNT
Workers Used: $WORKERS
Batch Size: $BATCH_SIZE
Max Memory: ${MAX_MEMORY}GB

Log Files:
- enrichment.log: Main process log
- enrichment_${SLURM_JOB_ID}.out: SLURM stdout
- enrichment_${SLURM_JOB_ID}.err: SLURM stderr
- enrichment_summary_*.json: Detailed statistics

Output Directory: $OUTPUT_DIR
EOF
    
    echo "Job summary saved to: $OUTPUT_DIR/job_summary.txt"
    
else
    echo "=========================================="
    echo "Enrichment failed with errors!"
    echo "End time: $(date)"
    echo "Check the log files for details"
    echo "=========================================="
    exit 1
fi

# Optional: Send notification email (uncomment if needed)
# echo "Enrichment job $SLURM_JOB_ID completed successfully" | mail -s "HPC Job Complete" your_email@domain.com

echo "Job completed. Check $OUTPUT_DIR for results and logs."
