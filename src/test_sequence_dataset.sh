#!/bin/bash
#SBATCH --job-name=narval_dist_seq_data
#SBATCH --account=rrg-fwood
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100_3g.20gb:2 
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Available gpu slices
# --gpus=a100_1g.5gb:2
# --gpus=a100_2g.10gb:2
# --gpus=a100_3g.20gb:2
# --gpus=a100_4g.20gb:2
# --gpus=a100_40gb

module load apptainer/1.2.4 

SIF_PATH="/project/def-fwood/chsuae/singularity/ubc_env.sif"
# SIF_PATH="/scratch/chsuae/singularity_setup/video_synth_tasks/ubc_env.sif"

# Random port for distributed training
PORT=$((10000 + RANDOM % 50000))

# Define paths to bind
PROJECT="/project/def-fwood/chsuae/video_synthetic_tasks"
DATASET_ROOT="/scratch/chsuae/datasets"

# PROJECT="/ubc/cs/research/fwood/chsu35/video_synthetic_tasks"
# DATASET_ROOT="/ubc/cs/research/plai-scratch/chsu35/datasets"

# Set other parameters
SEED=42
DATASET="MNIST"
DATASET_TYPE="selective"
BATCH_SIZE=4
NUM_STEPS=100

# Run the training script with arguments
apptainer exec --nv --bind ${PROJECT} --bind ${DATASET_ROOT} \
  ${SIF_PATH} \
  torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT seq_dataset.py \
    --seed ${SEED} \
    --dataset ${DATASET} \
    --dataset_root ${DATASET_ROOT} \
    --dataset_type ${DATASET_TYPE} \
    --batch_size ${BATCH_SIZE} \
    --num_steps ${NUM_STEPS}


# UBC / Compute Canada Slurm

# #!/bin/bash
#SBATCH --job-name=debug_dist_seq_data
#SBATCH --nodes=1
#SBATCH --ntasks=2          
#SBATCH --gres=gpu:2       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=plai
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   