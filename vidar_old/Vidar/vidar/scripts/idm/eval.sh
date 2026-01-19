#!/bin/bash
source activate vidar
export CUDA_VISIBLE_DEVICES=7
TEST_DATASET_PATH=/data/dex/vidar/data/cube-human-test
SAVE_DIR=./output/$1
# HDF5 Input logic
HDF5_PATH=/data/dex/RoboTwin/data/beat_block_hammer/demo_clean_vidar/data/episode1.hdf5

accelerate launch --main_process_port=29348 eval_idm.py \
    --load_from /data/dex/vidar/vidar_old/Vidar/vidar/output/VIDAR_20260116_031740/55000.pt \
    --hdf5_path $HDF5_PATH \
    --model_name dino-resnet \
    --mask_weight 3e-3 \
    --use_normalization \
    --use_transform \
    --save_dir $SAVE_DIR \
    # --use_gt_mask