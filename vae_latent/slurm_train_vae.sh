#!/bin/bash
#SBATCH --job-name=vae_droid
#SBATCH --output=slurm_logs/vae_train_%j.out
#SBATCH --error=slurm_logs/vae_train_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=500:00:00
#SBATCH --mem=1500G
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

# Print environment info
echo ""
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "Number of GPUs: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Change to working directory
cd /home/jisookim/human_demon_action/vae_latent

# Run training
echo "======================================================================"
echo "Starting VAE training..."
echo "======================================================================"

bash run_train.sh

# Print completion info
echo ""
echo "======================================================================"
echo "Job completed at: $(date)"
echo "======================================================================"

