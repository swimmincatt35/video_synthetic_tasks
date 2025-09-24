#!/bin/bash
#SBATCH --job-name=debug_vae
#SBATCH --nodes=1
#SBATCH --ntasks=1          
#SBATCH --gres=gpu:1       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=plai
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   

# Available gpu slices
# --gpus=a100_1g.5gb:2
# --gpus=a100_2g.10gb:2
# --gpus=a100_3g.20gb:2
# --gpus=a100_4g.20gb:2
# --gpus=a100_40gb
# --partition=gpubase_bygpu_b1

# module load apptainer/1.2.4 

# SIF_PATH="/project/def-fwood/chsuae/singularity/ubc_env.sif"
SIF_PATH="/ubc/cs/research/plai-scratch/chsu35/singularity_setup/video_synth_tasks/ubc_env.sif"

# Define paths to bind
# PROJECT="/project/def-fwood/chsuae/video_synthetic_tasks"
# DATASET_ROOT="/scratch/chsuae/datasets"
# OUTPUT_DIR=...
PROJECT="/ubc/cs/research/fwood/chsu35/video_synthetic_tasks"
DATASET_ROOT="/ubc/cs/research/plai-scratch/chsu35/datasets"
OUTPUT_DIR="/ubc/cs/research/plai-scratch/chsu35/vae-runs"

# --------- Hyperparameters ----------
DATASET="mnist"
BATCH_SIZE=128
EPOCHS=500
LATENT_CH=4
BASE_CH=32
LR=7e-4 # 1e-3 / 7e-4
KLD_COEF=5e-4 # 1e-4 / 5e-4
SAVE_INTV=50
DUMP_INT=25
WANDB_CONF="/ubc/cs/research/fwood/chsu35/video_synthetic_tasks/src/vae/configs/wandb_config.json"

# --------- Run Training ----------
echo "[INFO] Starting training job $SLURM_JOB_ID ..."
singularity exec --nv --bind ${PROJECT} --bind ${DATASET_ROOT} --bind ${OUTPUT_DIR} \
  ${SIF_PATH} python3 train_vae_mnist_cifar10.py \
    --dataset $DATASET \
    --dataset_root $DATASET_ROOT \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --latent_ch $LATENT_CH \
    --base_ch $BASE_CH \
    --lr $LR \
    --kld_coef $KLD_COEF \
    --output_dir $OUTPUT_DIR \
    --save_interval $SAVE_INTV \
    --dump_interval $DUMP_INT \
    --wandb_config $WANDB_CONF


# UBC / Compute Canada Slurm

# #!/bin/bash
#SBATCH --job-name=debug_vae
#SBATCH --nodes=1
#SBATCH --ntasks=2          
#SBATCH --gres=gpu:2       
#SBATCH --cpus-per-task=4
#SBATCH --time=01:00:00
#SBATCH --partition=plai
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   


#!/bin/bash
#SBATCH --job-name=narval_vae
#SBATCH --account=rrg-fwood
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=a100_40gb
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=40:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err