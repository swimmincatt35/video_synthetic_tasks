#!/bin/bash
#SBATCH --job-name=debug_dist_seq_data
#SBATCH --nodes=1
#SBATCH --ntasks=2          
#SBATCH --gres=gpu:2       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=plai
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   

PORT=$((10000 + RANDOM % 50000))

# Activate your virtual environment
# source /ubc/cs/research/fwood/chsu35/video_synthetic_tasks/test/bin/activate

# torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT seq_dataset.py

singularity exec --nv \
  --bind /ubc/cs/research/fwood/chsu35/video_synthetic_tasks \
  --bind /ubc/cs/research/plai-scratch \
  /ubc/cs/research/plai-scratch/chsu35/singularity_setup/video_synth_tasks/ubc_env.sif \
  torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT seq_dataset.py

