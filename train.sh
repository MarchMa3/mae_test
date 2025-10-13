#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate mae

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2
export OMP_NUM_THREADS=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export WANDB_API_KEY="6cf65e599a667502406f2f26fd010cf12ac94b99"
export WANDB_MODE=online
export WANDB_DIR=./wandb_logs

mkdir -p log/train
mkdir -p wandb_logs
mkdir -p checkpoints_mimic

LOG_FILE="log/train/train_mimic_$(date +%Y%m%d_%H%M%S).log"

torchrun --nproc_per_node=3 --master_port=29501 train_v2.py \
    --data data/labevents_mimic.pkl \
    --model_size small \
    --batch_size 128 \
    --epochs 100 \
    --lr 5e-4 \
    --min_lr 5e-6 \
    --warmup_epochs 15 \
    --num_workers 8 \
    --weight_decay 0.05 \
    --patience 20 \
    --mask_ratio 0.5 \
    --drop_ratio 0.1 \
    --attn_drop_ratio 0.1 \
    2>&1 | tee "$LOG_FILE"

exit ${PIPESTATUS[0]}