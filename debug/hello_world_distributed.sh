#!/bin/bash
#SBATCH --job-name=hw_dist
#SBATCH --nodes=1
#SBATCH --ntasks=2          
#SBATCH --gres=gpu:2       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=ubcml-rti 
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err      

source /ubc/cs/research/plai-scratch/chsu35/virtual_envs/synth_tasks/bin/activate
torchrun --nproc_per_node=2 hello_world_distibuted.py

