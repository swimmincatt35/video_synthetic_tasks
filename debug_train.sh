#!/bin/bash
#SBATCH --job-name=debug_train
#SBATCH --nodes=1
#SBATCH --ntasks=2          
#SBATCH --gres=gpu:2       
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G  
#SBATCH --time=01:00:00
#SBATCH --partition=plai
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   

# Activate your virtual environment
source /ubc/cs/research/plai-scratch/chsu35/virtual_envs/vid_synth_tasks/bin/activate

PORT=$((10000 + RANDOM % 50000))

nvidia-smi

torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT train.py \
    --output_dim 32 \
    --num_layers 1 \
    --num_heads 2 \
    --rnn_type mingru \
    --dataset_name MNIST \
    --seq_length 256 \
    --synth_task ind_head \
    --batch_size 2 \
    --train_iters 20 \
    --eval_samples 8 \
    --eval_every 5 \
    --save_every 10 \
    --project debug-project \
    --run_name debug-run
