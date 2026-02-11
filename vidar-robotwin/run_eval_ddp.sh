#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin
export CUDA_VISIBLE_DEVICES=0,1,2,3

# --- 配置区域 ---
# 任务配置
TASK_CONFIG=${1:-"hd_clean"}
MODEL=${2:-"../vidar_ckpts/vidar.pt"}
IDM=${3:-"../vidar_ckpts/idm.pt"}
PREFIX=${4:-"debug_ddp"}

# 采样参数
NUM_NEW_FRAMES=${5:-60}
NUM_SAMPLING_STEP=${6:-20}
CFG=${7:-5.0}

# 模式: vidar (默认) | gt_action (使用真实 Action) | idm_action (使用 IDM 推理 HDF5 观测)
MODE=${8:-"vidar"}

# Server 脚本位置 (根据需要修改，支持 T2V 或 I2V)
SERVER_SCRIPT="../server/stand_worker.sh"

# --- 启动 ---
echo "Starting Unified DDP Evaluation..."
echo "Model: $MODEL"
echo "Prefix: $PREFIX"
echo "Mode: $MODE"
echo "Server: $SERVER_SCRIPT"

# 设置 Master Port 防止冲突
export MASTER_PORT=11452
# 使用 torchrun 启动
# nproc_per_node 自动设为 GPU 数量 (或者手动指定 8)
# GPU_COUNT=$(nvidia-smi -L | wc -l)
# 强制使用单卡
GPU_COUNT=4

torchrun --nproc_per_node=$GPU_COUNT --master_port=$MASTER_PORT \
    policy/AR/run_eval_ddp.py \
    --server_script "$SERVER_SCRIPT" \
    --model "$MODEL" \
    --idm "$IDM" \
    --prefix "$PREFIX" \
    --task_config "$TASK_CONFIG" \
    --rollout_prefill_num 1 \
    --num_new_frames "$NUM_NEW_FRAMES" \
    --num_sampling_step "$NUM_SAMPLING_STEP" \
    --cfg "$CFG" \
    --mode "$MODE"

echo "Evaluation finished."

