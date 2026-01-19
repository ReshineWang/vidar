# Robotwin Eval code for Vidar/Vidarc

## Overview
We utilize a client-server architecture for evaluation. This repository serves as the **client**, responsible for managing the server, sending requests, and executing evaluations upon receiving actions.
Before proceeding, please ensure that the server-side environment and code are properly set up.

## Env Setup
Please refer to [READMEEnv.md](READMEEnv.md).

## Evaluation
This module provides a unified evaluation script based on `torch.distributed` (DDP), designed to simplify the multi-GPU/multi-task evaluation workflow in `Client-Server` mode.

## Features

1.  **Unified Architecture**: Uses `torchrun` to launch a single Python script.
2.  **Automatic Task Distribution**: Automatically splits task lists leveraging DDP's `rank` and `world_size`, removing the need for manual assignment.
3.  **Robust Process Management**: Utilizes Python Context Manager to manage the Server lifecycle, ensuring that the Server and its subprocesses are cleanly terminated regardless of normal completion or abnormal exit.
4.  **Decoupled Design**: All paths (Server script, model, task descriptions) are passed via arguments rather than being hardcoded.
5.  **Resumable Execution**: Automatically skips tasks that already have existing logs.

## Dependencies

- PyTorch (for `torch.distributed`)
- Existing `vidar` Server script
- Existing `script/eval_policy.py` Client script

## Usage

```bash
conda activate RoboTwin-hb

# eval with vidarc
bash run_eval_ddp_causal.sh

# eval with vidar
bash run_eval_ddp.sh 
```

### Parameters

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--server_script` | Path to the Server startup script (Required) | - |
| `--model` | Path to the model (Required) | - |
| `--idm` | Path to the Inverse Dynamics Model | - |
| `--prefix` | Prefix for the output directory (Required) | "debug" |
| `--task_dir` | Directory containing task description files | "./description/task_instruction" |
| `--server_cwd` | Working directory for the Server script | "../cosmos-predict2" |
| `--base_port` | Starting port number (Rank 0 uses base, Rank 1 uses base+1...) | 25400 |
