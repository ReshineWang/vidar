#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_VISIBLE_DEVICES=4
# source activate vidar
# DATASET_PATH=/data/dex/vidar/data/cube-human-test
DATASET_PATH=/data/dex/RoboTwin/data
DINO_FEATURE_PATH=/data/dex/vidar/vidar_ckpts/dino_feature

SAVE_DIR=./output/$1
accelerate launch --main_process_port=29352 train_idm.py \
    --learning_rate 3e-4 \
    --use_normalization \
    --use_transform \
    --batch_size 32 \
    --num_iterations 200000 \
    --eval_interval 5000 \
    --num_workers 8 \
    --prefetch_factor 4 \
    --dataset_path $DATASET_PATH \
    --save_dir $SAVE_DIR \
    --wandb_mode offline \
    --model_name resnet_plus \
    --mask_weight 3e-3 \
    --run_name "VIDAR" \
    --domain RoboTwin \
    --task_config demo_clean_vidar \
    --num_frames 3 \
    --mixed_precision bf16 \
    --do_flip \
    --crop_and_resize  # Enable crop and resize to 832x480 (matches wan2.1 output)

    # --do-flip

    # --use_gt_mask \


     