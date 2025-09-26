#!/bin/bash
#SBATCH --job-name=train_rnn-test
#SBATCH --nodes=1
#SBATCH --ntasks=8          
#SBATCH --gres=gpu:8       
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=ubcml
#SBATCH --output=logs/%x_%j.out    
#SBATCH --error=logs/%x_%j.err   

NUM_GPUS=8

# --------- Multi gpu testing ---------
PORT=$((10000 + RANDOM % 50000))

# --------- Image ---------
SIF_PATH="/ubc/cs/research/plai-scratch/chsu35/singularity_setup/video_synth_tasks/ubc_env.sif"

PROJECT="/ubc/cs/research/fwood/chsu35/video_synthetic_tasks"
SCRATCH="/ubc/cs/research/plai-scratch/chsu35"
DATASET_ROOT="${SCRATCH}/datasets"
VAE_CKPT_DIR="${SCRATCH}/vae-runs"
RNN_CKPT_DIR="${SCRATCH}/rnn-runs" 

# --------- Issue selective_scan_interface.py line#20 ---------
HOST_PATH="${PROJECT}/src/selective_scan_interface.py"
CONTAINER_PATH="/usr/local/lib/python3.10/dist-packages/mamba_ssm/ops/selective_scan_interface.py"

DATASET="mnist" # "mnist"
VAE_PATH="${VAE_CKPT_DIR}/vae-mnist-lr0.001-b128-kld0.0001/checkpoints/vae_epoch_300.pt"
TRAIN_SEQ_LEN=32 # 32 / 64
RNN_PATH="${RNN_CKPT_DIR}/seq${TRAIN_SEQ_LEN}-ind_head-mnist-mingru-ly4-b64-lr0.0001-1000k/ckpt_500000.pt"
RNN_TYPE="mingru"
NUM_LAYERS=4 # 2 / 4
NUM_HEADS=4
SYNTH_TASK="ind_head" # "ind_head" / "sel_copy"
BATCH_SIZE=64 # 128 / 8 

nvidia-smi

# --------- Run Testing ----------
echo "[INFO] Starting testing job $SLURM_JOB_ID ..."
singularity exec --nv \
    --bind ${HOST_PATH}:${CONTAINER_PATH} --bind ${PROJECT} --bind ${DATASET_ROOT} --bind ${VAE_CKPT_DIR} --bind ${RNN_CKPT_DIR} \
    ${SIF_PATH} torchrun --nproc_per_node ${NUM_GPUS} test.py \
    --train_seq_len $TRAIN_SEQ_LEN \
    --rnn_ckpt_path $RNN_PATH \
    --rnn_type $RNN_TYPE \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --vae_path $VAE_PATH \
    --dataset_name $DATASET \
    --synth_task $SYNTH_TASK \
    --batch_size $BATCH_SIZE \
    --eval_samples $((10 * BATCH_SIZE))