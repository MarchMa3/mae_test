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

mkdir -p log
mkdir -p wandb_logs
mkdir -p checkpoints_mimic

LOG_FILE="log/train/train_mimic_$(date +%Y%m%d_%H%M%S).log"

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "Log file: $LOG_FILE"
echo "WandB project: mae-mimic"
echo ""

echo "Starting training..."
echo "Log will be saved to: $LOG_FILE"

torchrun --nproc_per_node=3 train_v2.py \
    --data data/labevents_mimic.pkl \
    --model_size small \
    --batch_size 128 \
    --epochs 100 \
    --lr 1e-4 \
    --num_workers 8 \
    --weight_decay 0.05 \
    --patience 10 \
    2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -ne 0 ]; then
    echo "Training failed with exit code: $EXIT_CODE"
    echo "Check log: $LOG_FILE"
else
    echo "Training completed successfully!"
    echo "Log saved: $LOG_FILE"
fi
echo "=========================================="

exit $EXIT_CODE