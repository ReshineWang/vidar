#!/bin/bash
source activate vidar
# your cuda device id
export CUDA_VISIBLE_DEVICES=

# metadata file to be used for inference
# prompt and image will be read from the file
DATA_JSONS=""

#  Directory to save the generated videos
SAVE_DIR=""

# load the checkpoint parameters from the path
DIT_WEIGHT=""

NPROC=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
ALLOW_RESIZE_FOR_SP=1 torchrun --nproc_per_node=$NPROC \
    sample_vm.py \
    --model HYVideo-T/2 \
    --i2v-mode \
    --i2v-resolution 540 \
    --i2v-dit-weight ${DIT_WEIGHT} \
    --infer-steps 50 \
    --video-length 61 \
    --flow-reverse \
    --flow-shift 17.0 \
    --embedded-cfg-scale 6.0 \
    --seed 0 \
    --save-path ${SAVE_DIR} \
    --ulysses-degree $NPROC \
    --ring-degree 1 \
    --metadata-paths ${DATA_JSONS}

## simply providing image paths and string type prompts is available if metadata path is not provided
# --image-paths xxx \
# --prompt "xxx"
