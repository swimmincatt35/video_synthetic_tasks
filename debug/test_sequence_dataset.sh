#!/bin/bash
#SBATCH --job-name=debug_dist_seq_data
#SBATCH --nodes=1
#SBATCH --ntasks=2          
#SBATCH --gres=gpu:2       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=ubcml
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   
#SBATCH --exclude=ubc-ml03,ubc-ml04   

# Activate your virtual environment
source /ubc/cs/research/plai-scratch/chsu35/virtual_envs/vid_synth_tasks/bin/activate

PORT=$((10000 + RANDOM % 50000))

torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT test_sequence_dataset.py


