#!/bin/bash

# Source conda to ensure activate works
source /root/miniconda3/etc/profile.d/conda.sh

# Ensure we are in the correct directory (parent of 'server' package)
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_DIR=$(dirname "$SCRIPT_DIR")
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"

# 接收参数
export MODEL=$1
export IDM=$2
export WORKER_PORT=$3
export CUDA_VISIBLE_DEVICES=$4

# 设置单卡运行必要的环境变量
export WORLD_SIZE=1
export RANK=0
export LOCAL_RANK=0

conda activate vidar

echo "Starting Standalone Worker on Port $WORKER_PORT Device $CUDA_VISIBLE_DEVICES"
exec uvicorn server.causal_worker:api --host localhost --port $WORKER_PORT --workers 1

