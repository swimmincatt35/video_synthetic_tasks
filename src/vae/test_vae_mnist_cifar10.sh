#!/bin/bash
#SBATCH --job-name=test_vae
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --gres=gpu:1       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=plai
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   

# module load apptainer/1.2.4 

# SIF_PATH="/project/def-fwood/chsuae/singularity/ubc_env.sif"
SIF_PATH="/ubc/cs/research/plai-scratch/chsu35/singularity_setup/video_synth_tasks/ubc_env.sif"

# Define paths to bind
# PROJECT="/project/def-fwood/chsuae/video_synthetic_tasks"
# DATASET_ROOT="/scratch/chsuae/datasets"
# OUTPUT_DIR=...
PROJECT="/ubc/cs/research/fwood/chsu35/video_synthetic_tasks"
DATASET_ROOT="/ubc/cs/research/plai-scratch/chsu35/datasets"
OUTPUT_DIR="/ubc/cs/research/plai-scratch/chsu35/vae-test-runs"
CKPT_DIR="/ubc/cs/research/plai-scratch/chsu35/vae-runs/"

# Checkpoints
# DATASET=cifar10
# CKPT_PATH="/ubc/cs/research/plai-scratch/chsu35/vae-runs/vae-cifar10-lr0.001-b128-kld0.0001/checkpoints/vae_epoch_300.pt"
DATASET=mnist
CKPT_PATH="/ubc/cs/research/plai-scratch/chsu35/vae-runs/vae-mnist-lr0.001-b128-kld0.0001/checkpoints/vae_epoch_300.pt"

singularity exec --nv --bind ${PROJECT} --bind ${DATASET_ROOT} --bind ${OUTPUT_DIR} --bind ${CKPT_DIR} \
  ${SIF_PATH} python3 test_vae_mnist_cifar10.py \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --batch_size 64 \
    --latent_ch 4 \
    --base_ch 32 \
    --resume ${CKPT_PATH} \
    --output_dir ${OUTPUT_DIR}

echo "[INFO] Evaluation finished."
