#!/bin/bash
# source activate vidar
export CUDA_VISIBLE_DEVICES=1
# TEST_DATASET_PATH=/data/dex/vidar/data/cube-human-test
SAVE_DIR=./output
# HDF5 Input logic
TEST_VIDEO_PATH=/data/dex/large-video-planner/outputs/2026-01-21/18-27-14/wandb/latest-run/files/media/videos/validation_vis/video_pred_0_0_884122b4b123eacfb8e2.mp4
# TEST_HDF5_PATH=/data/dex/RoboTwin/data/click_bell/demo_clean_vidar/data/episode0.hdf5
# TEST_VIDEO_PATH=/data/dex/RoboTwin/data/beat_block_hammer/demo_clean_vidar/video/episode0.mp4
HDF5_PATH="/data/dex/RoboTwin/data/beat_block_hammer/demo_clean_vidar/data/episode0.hdf5"


accelerate launch --main_process_port=29348 eval_idm.py \
    --load_from /data/dex/vidar/vidar_ckpts/resnet_plus_robotwin/big_view.pt \
    --hdf5_path $HDF5_PATH \
    --model_name resnet_plus \
    --mask_weight 3e-3 \
    --use_transform \
    --domain RoboTwin \
    --save_dir $SAVE_DIR \
    --num_frames 3 \
    --task_config demo_clean_vidar \
    --compute_smoothness \
    --crop_and_resize           # 如果是用生成的视频，就不加这个参数。如果是用robotwin 采样的视频，就加这个参数（因为训练idm的时候加了这个参数）
    # --use_gt_mask
    # --hdf5_path $HDF5_PATH \
    # --video_path $TEST_VIDEO_PATH \
    # --val_indices ./output/VIDAR_20260119_170225/val_indices.pt \


