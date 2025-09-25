#!/bin/bash
#SBATCH --job-name=train_run
#SBATCH --account=rrg-fwood
#SBATCH --nodes=2
#SBATCH --ntasks=2
#SBATCH --gpus=a100:2
#SBATCH --partition=gpubase_bygpu_b1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

module load apptainer/1.2.4 

# --------- Multi gpu training ---------
PORT=$((10000 + RANDOM % 50000))

# --------- Image ---------
SIF_PATH="/project/def-fwood/chsuae/singularity/ubc_env.sif"
# SIF_PATH="/ubc/cs/research/plai-scratch/chsu35/singularity_setup/video_synth_tasks/ubc_env.sif"

# --------- Define paths to bind ---------
PROJECT="/project/def-fwood/chsuae/video_synthetic_tasks"
DATASET_ROOT="/scratch/chsuae/datasets"
OUTPUT_DIR="/scratch/chsuae/rnn-runs"
CKPT_DIR="/scratch/chsuae/vae-runs"
# PROJECT="/ubc/cs/research/fwood/chsu35/video_synthetic_tasks"
# DATASET_ROOT="/ubc/cs/research/plai-scratch/chsu35/datasets"
# OUTPUT_DIR="/ubc/cs/research/plai-scratch/chsu35/rnn-runs"
# CKPT_DIR="/ubc/cs/research/plai-scratch/chsu35/vae-runs"

# --------- Issue selective_scan_interface.py line#20 ---------
HOST_PATH="${PROJECT}/src/selective_scan_interface.py"
CONTAINER_PATH="/usr/local/lib/python3.10/dist-packages/mamba_ssm/ops/selective_scan_interface.py"

# --------- Hyperparameters ---------
DATASET="cifar10" # "cifar10" / "mnist"
# VAE_PATH="${CKPT_DIR}/vae-mnist-lr0.001-b128-kld0.0001/checkpoints/vae_epoch_300.pt"
VAE_PATH="${CKPT_DIR}/vae-cifar10-lr0.001-b128-kld0.0001/checkpoints/vae_epoch_300.pt"
SYNTH_TASK="ind_head" # "ind_head" / "sel_copy"
BATCH_SIZE=64 # 128 / 8 
TRAIN_ITERS=1000000 # 1000 / 2
# The mamba paper trained mamba for 25 epochs (8192 steps/epoch), batch size 8,     => 200000 iters
# And other baselines were trained for 50 epochs (8192 steps/epoch), batch size 8,  => 400000 iters
LR=1e-4 # 1e-4 / 1e-5 / 5e-4
LOG_EVERY=1000 # 10 / 1
EVAL_EVERY=5000 # 50 / 1
SAVE_EVERY=$((TRAIN_ITERS / 4)) # 250 / 1
NUM_LAYERS=4 # 2 / 4
NUM_HEADS=4
RNN_TYPE="mingru"
WANDB_CONF="${PROJECT}/configs/wandb_config.json"

# --------- Debugging ---------
FIXED_HEAD=-1 # -1 / 252(2K,B8) / 200 / 100
SEQ_LEN=-1 # -1 / 128 / 64 / 32(30K,B8) / 16(8K,B8) / 8(2K,B8)

nvidia-smi

# --------- Run Training ----------
echo "[INFO] Starting training job $SLURM_JOB_ID ..."
apptainer exec --nv \
    --bind ${HOST_PATH}:${CONTAINER_PATH} --bind ${PROJECT} --bind ${DATASET_ROOT} --bind ${OUTPUT_DIR} --bind ${CKPT_DIR} \
    ${SIF_PATH} torchrun --nproc_per_node=2 --rdzv_endpoint=localhost:$PORT train.py \
    --num_layers $NUM_LAYERS \
    --num_heads $NUM_HEADS \
    --rnn_type $RNN_TYPE \
    --dataset_name $DATASET \
    --dataset_root $DATASET_ROOT \
    --vae_path $VAE_PATH \
    --synth_task $SYNTH_TASK \
    --batch_size $BATCH_SIZE \
    --train_iters $TRAIN_ITERS \
    --lr $LR \
    --eval_samples $((10 * BATCH_SIZE)) \
    --log_every $LOG_EVERY \
    --eval_every $EVAL_EVERY \
    --save_every $SAVE_EVERY \
    --save_dir $OUTPUT_DIR \
    --wandb_config $WANDB_CONF \
    --fixed_head $FIXED_HEAD \
    --seq_len $SEQ_LEN


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
#SBATCH --gpus=a100
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
# --gpus=a100
# --partition=gpubase_bygpu_b1
# sinfo -o "%P %G %D %N" # to check for more