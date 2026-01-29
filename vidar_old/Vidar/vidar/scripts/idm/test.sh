#!/bin/bash
source activate vidar
export CUDA_VISIBLE_DEVICES=1
# TEST_DATASET_PATH=/data/dex/vidar/data/cube-human-test
SAVE_DIR=./output/vis2
# HDF5 Input logic
TEST_VIDEO_PATH=/data/dex/vidar/vidar_old/Vidar/vidar/output/vis2/video_pred_0_0_88c276429048f3fa0de3.mp4
# TEST_HDF5_PATH=/data/dex/RoboTwin/data/click_bell/demo_clean_vidar/data/episode0.hdf5

accelerate launch --main_process_port=29348 eval_idm.py \
    --load_from /data/dex/vidar/vidar_old/Vidar/vidar/output/VIDAR_20260125_015042/200000.pt \
    --video_path $TEST_VIDEO_PATH \
    --model_name resnet_plus \
    --mask_weight 3e-3 \
    --use_transform \
    --domain RoboTwin \
    --save_dir $SAVE_DIR \
    --num_frames 3 \
    # --crop_and_resize
    # --use_gt_mask
        # --hdf5_path $HDF5_PATH \
            # --val_indices ./output/VIDAR_20260119_170225/val_indices.pt \

