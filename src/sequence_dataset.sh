#!/bin/bash
#SBATCH --job-name=debug_dist
#SBATCH --nodes=1
#SBATCH --ntasks=2          
#SBATCH --gres=gpu:2       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=ubcml-rti 
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   
#SBATCH --exclude=ubc-ml03   

# Activate your virtual environment
source /ubc/cs/research/plai-scratch/chsu35/virtual_envs/synth_tasks/bin/activate

PORT=$((10000 + RANDOM % 50000))

torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT sequence_dataset.py


