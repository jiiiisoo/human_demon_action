#!/bin/bash
#SBATCH --job-name=droid_convert
#SBATCH --output=slurm_logs/convert_%j.out
#SBATCH --error=slurm_logs/convert_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --mem=100G
#SBATCH -q main

# Print job info
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "======================================================================"

# Create log directory
mkdir -p slurm_logs

# Activate conda environment
source /home/jisookim/miniconda3/etc/profile.d/conda.sh
conda activate human_demon

# Change to working directory
cd /home/jisookim/human_demon_action/vae_latent

# Run conversion (all episodes, or pass number as argument)
MAX_EPISODES=${1:-""}

echo "======================================================================"
echo "Starting DROID dataset conversion..."
echo "Max episodes: ${MAX_EPISODES:-All}"
echo "======================================================================"

if [ -z "$MAX_EPISODES" ]; then
    bash data/run_convert_parallel.sh
else
    bash data/run_convert_parallel.sh $MAX_EPISODES
fi

# Print completion info
echo ""
echo "======================================================================"
echo "Conversion completed at: $(date)"
echo "======================================================================"

