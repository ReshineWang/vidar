#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=5
source activate vidar
# DATASET_PATH=/data/dex/vidar/data/cube-human-test
DATASET_PATH=/data/dex/RoboTwin/data
SAVE_DIR=./output/$1
accelerate launch --main_process_port=29348 train_idm.py \
    --learning_rate 1e-5 \
    --use_normalization \
    --use_transform \
    --batch_size 32 \
    --num_iterations 40000 \
    --eval_interval 5000 \
    --num_workers 16 \
    --prefetch_factor 16 \
    --dataset_path $DATASET_PATH \
    --save_dir $SAVE_DIR \
    --wandb_mode offline \
    --model_name dino-resnet \
    --mask_weight 3e-3 \
    --run_name "VIDAR" \
    --domain RoboTwin \
    --task_config demo_clean_vidar

    # --use_gt_mask \


     