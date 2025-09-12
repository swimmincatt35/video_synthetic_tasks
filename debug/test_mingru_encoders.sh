#!/bin/bash
#SBATCH --job-name=debug_mingru
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --gres=gpu:1       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=ubcml-rti 
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   
#SBATCH --exclude=ubc-ml03   

# Activate your virtual environment
source /ubc/cs/research/plai-scratch/chsu35/virtual_envs/vid_synth_tasks/bin/activate

PORT=$((10000 + RANDOM % 50000))

# torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT test_xlstm_encoders.py

python3 test_mingru_encoders.py

