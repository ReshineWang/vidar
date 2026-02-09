#!/bin/bash
# source activate vidar
export CUDA_VISIBLE_DEVICES=1
# TEST_DATASET_PATH=/data/dex/vidar/data/cube-human-test
SAVE_DIR=./output/$1
# HDF5 Input logic
HDF5_PATH=/data/dex/RoboTwin/data/beat_block_hammer/demo_clean_vidar/data/episode1.hdf5
DATASET_PATH=/data/dex/RoboTwin/data_test

accelerate launch --main_process_port=29348 eval_idm.py \
    --load_from /data/dex/vidar/vidar_old/Vidar/vidar/output/VIDAR_20260129_134534/200000.pt \
    --dataset_path $DATASET_PATH \
    --model_name resnet_plus \
    --mask_weight 3e-3 \
    --use_transform \
    --domain RoboTwin \
    --save_dir $SAVE_DIR \
    --num_frames 3 \
    --task_config demo_clean \
    --crop_and_resize \
    --compute_smoothness \
    # --use_gt_mask
        # --hdf5_path $HDF5_PATH \
            # --val_indices ./output/VIDAR_20260119_170225/val_indices.pt \

